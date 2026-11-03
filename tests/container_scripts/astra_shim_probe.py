"""Probe for T-0015, run INSIDE the pinned vLLM container (never on the host).

The host has no ``vllm`` and no ``torch``; everything here needs both. The script imports (it
never boots an engine), collects *facts* about the upstream DiffusionGemma borrow and about the
plugin's shim, and prints them as one JSON object on a line prefixed with ``ASTRA_PROBE_JSON``.
All assertions live in ``tests/contract/test_vllm_shim_gpu.py`` on the host, so a failure is
readable without digging through container logs.

Sections mirror the acceptance criteria of ``docs/tasks/T-0015.md``:

* ``shim``    -> criterion 1 (the shim is shaped the way the task says)
* ``upstream``-> criterion 2 (the upstream defect is still present; the deletion trigger)
* ``audit``   -> criterion 3 (no borrowed method reads an attribute that does not resolve)

Nothing here decides pass/fail. Keep it that way: the moment this script starts asserting, a
failure becomes a container traceback instead of a pytest message.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
import traceback
from typing import Any

MARKER = "ASTRA_PROBE_JSON"

DONOR_MODULE = "vllm.model_executor.models.gemma4_mm"
DONOR_CLASS = "Gemma4ForConditionalGeneration"
STOCK_MODULE = "vllm.model_executor.models.diffusion_gemma"
STOCK_CLASS = "DiffusionGemmaForConditionalGeneration"
INFO_CLASS = "DiffusionGemmaProcessingInfo"

# The attribute whose absence kills the EngineCore (upstream commit 95c914442). Only used by the
# *upstream* section, which is a targeted regression check on a named defect; the audit section
# below never mentions it and would find it on its own.
DEFECT_ATTR = "_enable_mm_lora"


# --------------------------------------------------------------------------- AST helpers


def class_def(module_src: str, name: str) -> ast.ClassDef:
    for node in ast.parse(module_src).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise LookupError(f"no class {name} in the given module source")


def func_def(node: ast.ClassDef, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for child in node.body:
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) and child.name == name:
            return child
    raise LookupError(f"no method {name} on class {node.name}")


def borrowed_methods(stock_src: str, donor: str) -> dict[str, str]:
    """Class-level assignments in ``DiffusionGemma`` whose value is ``<donor>.<attr>``.

    Returns ``{attribute name on DiffusionGemma: attribute name on the donor}``. Both the plain
    ``X = Donor.x`` form and the ``X = staticmethod(Donor.x)`` form are recognised. Deliberately
    *not* a hardcoded list: if upstream borrows a seventh method, it appears here by itself.
    """
    out: dict[str, str] = {}
    for stmt in class_def(stock_src, STOCK_CLASS).body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target, value = stmt.target, stmt.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in ("staticmethod", "classmethod")
            and len(value.args) == 1
        ):
            value = value.args[0]
        if (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == donor
        ):
            out[target.id] = value.attr
    return out


def _str_consts(node: ast.AST) -> set[str]:
    return {
        n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def dispatch_literals(fn: ast.AST) -> dict[str, set[str]]:
    """Group the string literals this function dispatches on, keyed by the variable they key.

    Two syntactic forms count, both of which upstream uses:
    ``mm_input_by_modality["audio"] = ...`` (subscript with a literal key) and
    ``modality == "audio"`` (equality against a literal). The caller decides which groups are
    modality dispatches by intersecting with the modalities the model declares support for.
    """
    groups: dict[str, set[str]] = {}
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            groups.setdefault(node.value.id, set()).add(node.slice.value)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                if (
                    isinstance(op, ast.Eq | ast.NotEq)
                    and isinstance(comparator, ast.Constant)
                    and isinstance(comparator.value, str)
                ):
                    groups.setdefault(node.left.id, set()).add(comparator.value)
    return groups


def unreachable_literals(fn: ast.AST, supported: set[str]) -> set[str]:
    """Modality literals this deployment can never produce, derived, not hand-written.

    A variable is a *modality dispatch* iff at least one of the literals it is keyed/compared
    against is a modality the model declares support for. The remaining literals of that same
    group are modalities it declares no support for, so branches guarded by them are dead. If
    the model ever declares audio support, this set empties and every audio branch is audited
    again -- which is the point.
    """
    dead: set[str] = set()
    for literals in dispatch_literals(fn).values():
        if literals & supported:
            dead |= literals - supported
    return dead


def self_loads(fn: ast.AST) -> dict[str, list[set[str]]]:
    """``{attribute: [guard-literal set per load site]}`` for every ``self.<attr>`` *read*.

    A guard-literal set is the string constants of the conditions that must hold for the load
    to run: the tests of the enclosing ``if``/``while`` bodies (``else`` branches add nothing --
    the negation of a test says nothing about which literal is live), plus the literal key of an
    enclosing subscript assignment target, which is how upstream writes its modality dispatch.
    """
    sites: dict[str, list[set[str]]] = {}

    def loads_in(node: ast.AST, guards: set[str]) -> None:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.ctx, ast.Load)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "self"
            ):
                sites.setdefault(sub.attr, []).append(set(guards))

    def visit(stmts: list[ast.stmt], guards: set[str]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.If | ast.While):
                loads_in(stmt.test, guards)
                visit(stmt.body, guards | _str_consts(stmt.test))
                visit(stmt.orelse, guards)
            elif isinstance(stmt, ast.For | ast.AsyncFor):
                loads_in(stmt.iter, guards)
                loads_in(stmt.target, guards)
                visit(stmt.body, guards)
                visit(stmt.orelse, guards)
            elif isinstance(stmt, ast.With | ast.AsyncWith | ast.Try):
                for field in ("items", "handlers", "finalbody"):
                    for part in getattr(stmt, field, []) or []:
                        loads_in(part, guards)
                visit(getattr(stmt, "body", []), guards)
                visit(getattr(stmt, "orelse", []), guards)
            elif isinstance(stmt, ast.Assign | ast.AnnAssign | ast.AugAssign):
                extra: set[str] = set()
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for target in targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)
                    ):
                        extra.add(target.slice.value)
                    loads_in(target, guards)
                if stmt.value is not None:
                    loads_in(stmt.value, guards | extra)
            elif isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                visit(stmt.body, guards)
            else:
                loads_in(stmt, guards)

    body = fn.body if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef) else [fn]
    visit(body, set())
    return sites


def self_stores(cls: type) -> set[str]:
    """Every ``self.<attr> = ...`` in the source of ``cls`` and its bases.

    An import-only probe cannot instantiate the model (no weights, no config), so this is the
    static stand-in for "the instance will have this attribute". It is deliberately generous:
    anything any class in the MRO ever assigns to ``self`` counts as resolvable. The defect the
    audit is looking for survives that generosity, because the attribute is assigned only in a
    class that is *not* in the MRO.
    """
    names: set[str] = set()
    for base in cls.__mro__:
        names |= _self_stores_of(base)
    return names


def _self_stores_of(cls: type) -> set[str]:
    """``self.<attr> = ...`` targets in the source of ``cls`` alone (not its bases)."""
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return set()
    try:
        tree = ast.parse(_dedent(src))
    except SyntaxError:
        return set()
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }


def _dedent(src: str) -> str:
    import textwrap

    return textwrap.dedent(src)


# --------------------------------------------------------------------------- sections


def describe(obj: Any) -> dict[str, Any]:
    return {
        "module": getattr(obj, "__module__", None),
        "qualname": getattr(obj, "__qualname__", None),
        "id": id(obj),
    }


def shim_section(astra_cls: type, donor_cls: type, stock_cls: type) -> dict[str, Any]:
    bases = [describe(b) for b in astra_cls.__bases__]
    own = sorted(vars(astra_cls))
    facts: dict[str, Any] = {
        "class": describe(astra_cls),
        "bases": bases,
        "base0_is_stock": bool(astra_cls.__bases__ and astra_cls.__bases__[0] is stock_cls),
        "mro": [describe(c) for c in astra_cls.__mro__],
        "own_attributes": own,
    }
    facts["enable_mm_lora"] = {
        "present": hasattr(astra_cls, DEFECT_ATTR),
        "defined_in_own_class_body": DEFECT_ATTR in vars(astra_cls),
        "is_false": getattr(astra_cls, DEFECT_ATTR, None) is False,
        "repr": repr(getattr(astra_cls, DEFECT_ATTR, "<absent>")),
        "type": type(getattr(astra_cls, DEFECT_ATTR, None)).__name__,
    }
    borrowed_image = getattr(donor_cls, "_process_image_input", None)
    astra_image = getattr(astra_cls, "_process_image_input", None)
    facts["process_image_input"] = {
        "astra": describe(astra_image),
        "donor": describe(borrowed_image),
        "is_donor_function": astra_image is borrowed_image,
        "shadowed_in_own_class_body": "_process_image_input" in vars(astra_cls),
    }
    return facts


def upstream_section(stock_cls: type, stock_src: str, donor_src: str) -> dict[str, Any]:
    """Criterion 2: is the upstream defect still there? Three independent ways it could be fixed."""
    defining = [
        f"{c.__module__}.{c.__qualname__}" for c in stock_cls.__mro__ if DEFECT_ATTR in vars(c)
    ]
    stock_body = class_def(stock_src, STOCK_CLASS)
    assigns_in_source = DEFECT_ATTR in {
        node.attr
        for node in ast.walk(stock_body)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }
    donor_body = class_def(donor_src, DONOR_CLASS)
    bare_reads = sorted(
        {
            fn.name
            for fn in donor_body.body
            if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef)
            and DEFECT_ATTR in self_loads(fn)
        }
    )
    getattr_guarded = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == DEFECT_ATTR
        for node in ast.walk(donor_body)
    )
    return {
        "attribute": DEFECT_ATTR,
        "stock_mro_classes_defining_it": defining,
        "stock_assigns_it_in_source": assigns_in_source,
        "stock_subclasses_donor": any(c.__qualname__ == DONOR_CLASS for c in stock_cls.__mro__),
        "donor_methods_reading_it_bare": bare_reads,
        "donor_reads_it_via_getattr": getattr_guarded,
    }


def modality_literals(fn: ast.AST, supported: set[str]) -> set[str]:
    """Every literal of every dispatch group in ``fn`` that is a modality dispatch.

    Same rule as ``unreachable_literals`` (a group is a modality dispatch iff it names at least
    one declared modality), but keeps the live literals too, so a failure can say *which*
    modality a reachable site is guarded by.
    """
    out: set[str] = set()
    for literals in dispatch_literals(fn).values():
        if literals & supported:
            out |= literals
    return out


def shim_members(astra_cls: type, stock_cls: type) -> dict[str, list[str]]:
    """What the shim itself adds on top of the stock class, split into state and behaviour.

    Only classes in the shim's MRO that are *not* in the stock MRO count. ``state`` is plain,
    non-callable, non-descriptor data in their class bodies (``_enable_mm_lora = False``) plus
    anything their source assigns to ``self``. ``behaviour`` is everything else they define:
    functions, static/class methods, properties, other descriptors. The audit accepts shim
    *state* as a fix for an attribute the stock class lacks; shim *behaviour* that a live
    branch reaches is a stub standing in for missing upstream code, and the audit reports it.
    """
    stock_mro = set(stock_cls.__mro__)
    state: set[str] = set()
    behaviour: set[str] = set()
    for cls in astra_cls.__mro__:
        if cls in stock_mro:
            continue
        for name, value in vars(cls).items():
            if name.startswith("__") and name.endswith("__"):
                continue
            is_descriptor = any(
                hasattr(type(value), hook) for hook in ("__get__", "__set__", "__delete__")
            )
            if callable(value) or is_descriptor:
                behaviour.add(name)
            else:
                state.add(name)
        state |= _self_stores_of(cls)
    return {"state": sorted(state - behaviour), "behaviour": sorted(behaviour)}


def audit_borrowed(
    stock_src: str,
    donor_src: str,
    supported: list[str],
    stock_present: set[str],
    shim_state: set[str],
    shim_behaviour: set[str],
) -> dict[str, Any]:
    """The criterion-3 audit on plain data, so the host can exercise it without vLLM.

    Unresolved means unresolved on the **stock** class: what the borrowed code would hit if the
    shim did not exist. Each such attribute is then classified -- excluded as dead (every load
    site is guarded by a modality the model does not declare), supplied by the shim as state,
    or supplied by the shim as behaviour (a stub). The shim's members never decide deadness;
    only upstream's declared modalities do, so a stub cannot hide a branch that becomes live.
    """
    borrowed = borrowed_methods(stock_src, DONOR_CLASS)
    donor_body = class_def(donor_src, DONOR_CLASS)
    supported_set = set(supported)

    methods: dict[str, Any] = {}
    for own_name, donor_name in sorted(borrowed.items()):
        fn = func_def(donor_body, donor_name)
        dead = unreachable_literals(fn, supported_set)
        modal = modality_literals(fn, supported_set)
        loads = self_loads(fn)
        unresolved: dict[str, Any] = {}
        for attr, guard_sets in sorted(loads.items()):
            if attr in stock_present:
                continue
            reachable = [sorted(g) for g in guard_sets if not (g & dead)]
            if attr in shim_behaviour:
                supplied: str | None = "behaviour"
            elif attr in shim_state:
                supplied = "state"
            else:
                supplied = None
            unresolved[attr] = {
                "sites": len(guard_sets),
                "guards": [sorted(g) for g in guard_sets],
                "reachable_sites": reachable,
                "reachable_modalities": sorted({m for g in reachable for m in g if m in modal}),
                "excluded_as_dead": not reachable,
                "supplied_by_shim": supplied,
            }
        methods[own_name] = {
            "donor_method": donor_name,
            "self_loads": sorted(loads),
            "dead_modality_literals": sorted(dead),
            "unresolved": unresolved,
        }
    return {
        "supported_modalities": sorted(supported_set),
        "borrowed": borrowed,
        "methods": methods,
        "shim_state": sorted(shim_state),
        "shim_behaviour": sorted(shim_behaviour),
    }


def audit_section(
    astra_cls: type, stock_cls: type, stock_src: str, donor_src: str, supported: list[str]
) -> dict[str, Any]:
    """Criterion 3: every ``self.<attr>`` a borrowed method reads must resolve on the shim.

    Resolution is computed against the stock class, then the shim's contribution is classified
    (see ``audit_borrowed``). Computing it against the shim instead -- ``dir(astra_cls)`` --
    would count the shim's own audio stubs as resolved before the modality exclusion ran, and
    the audit could never notice the day upstream declares audio (T-0015 closing round).
    """
    members = shim_members(astra_cls, stock_cls)
    out = audit_borrowed(
        stock_src,
        donor_src,
        supported,
        stock_present=set(dir(stock_cls)) | self_stores(stock_cls),
        shim_state=set(members["state"]),
        shim_behaviour=set(members["behaviour"]),
    )
    resolvable = set(dir(astra_cls)) | self_stores(astra_cls)
    out["resolvable_sample"] = sorted(a for a in resolvable if a.startswith("_enable"))
    return out


def supported_modalities(info_cls: type) -> list[str]:
    """Modalities DiffusionGemma declares, read off upstream's own processing info.

    ``get_supported_mm_limits`` is a plain literal return in v0.29.0, so it can be called
    unbound. If upstream makes it depend on the config, this raises and the host test says so
    rather than silently falling back to a hand-written list.
    """

    class _NoCtx:
        def __getattr__(self, name: str) -> Any:
            raise AttributeError(
                f"{INFO_CLASS}.get_supported_mm_limits now needs self.{name}; the audit's "
                "'which modalities can this model receive' oracle must be re-derived"
            )

    return sorted(info_cls.get_supported_mm_limits(_NoCtx()))


def main() -> int:
    out: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import vllm

        donor_mod = __import__(DONOR_MODULE, fromlist=["*"])
        stock_mod = __import__(STOCK_MODULE, fromlist=["*"])
        donor_cls = getattr(donor_mod, DONOR_CLASS)
        stock_cls = getattr(stock_mod, STOCK_CLASS)
        donor_src = inspect.getsource(donor_mod)
        stock_src = inspect.getsource(stock_mod)
        out["vllm_version"] = getattr(vllm, "__version__", None)

        import astra_vllm_plugin

        architecture = astra_vllm_plugin.ARCHITECTURE
        out["architecture"] = architecture
        astra_cls = getattr(astra_vllm_plugin, architecture, None)
        if astra_cls is None:
            model_mod = __import__("astra_vllm_plugin.model", fromlist=["*"])
            astra_cls = getattr(model_mod, architecture, None)
        if astra_cls is None:
            out["error"] = (
                f"the plugin exposes no class named {architecture!r} on astra_vllm_plugin or "
                "astra_vllm_plugin.model; --hf-overrides names that string, so a class must "
                "answer to it"
            )
            print(MARKER, json.dumps(out))
            return 0

        out["shim"] = shim_section(astra_cls, donor_cls, stock_cls)
        out["upstream"] = upstream_section(stock_cls, stock_src, donor_src)
        out["audit"] = audit_section(
            astra_cls,
            stock_cls,
            stock_src,
            donor_src,
            supported_modalities(getattr(stock_mod, INFO_CLASS)),
        )
    except Exception:  # the host test turns this into a readable failure
        out["error"] = traceback.format_exc()
    print(MARKER, json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
