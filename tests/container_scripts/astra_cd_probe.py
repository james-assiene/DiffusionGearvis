"""Probe for T-0067 criterion 11, run INSIDE the pinned vLLM container (never on the host; no engine boot).

Reports facts only (one JSON line prefixed ``ASTRA_PROBE_JSON``); every assertion lives in
``tests/contract/test_vllm_cd_coupling_gpu.py``:

* ``step``: ``_compiled_sample_step``'s parameter names and the sha256 of its source text, from the line
  ``def _compiled_sample_step(`` through the first ``    return scaled`` line (the vendored fixture's rule);
* ``attrs``: the host sends ``{"<module>.<Class>": [attribute, ...]}`` in ``argv[1]`` (JSON). Each class is
  imported by its dotted path; for each attribute, whether it exists on the class (``dir``, dataclass fields,
  annotations up the MRO, or a ``self.<name> = ...`` store anywhere in the class source);
* ``missing_classes``: ``{"<module>.<Class>": reason}`` for every requested class that cannot be imported. A missing
  class is reported, never raised, so the step and ``custom_sampler`` facts still arrive and the host row fails
  with "class X not found in vLLM <version>" (the first tier-2 run died on a ``KeyError`` here instead);
* ``custom_sampler``: whether ``DiffusionGemmaModelState.custom_sampler``'s source returns a 2-tuple
  (``return <call>, None``) and its annotation;
* ``versions``: vllm and torch versions.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import inspect
import json
import re
import sys
import traceback
from typing import Any

MARKER = "ASTRA_PROBE_JSON"


def _segment(text: str, start: str, end: str) -> str:
    lines = text.splitlines(keepends=True)
    i = next(k for k, line in enumerate(lines) if re.match(start, line))
    j = next(k for k in range(i, len(lines)) if re.match(end, lines[k]))
    return "".join(lines[i : j + 1])


def _init_attrs(cls: type) -> set[str]:
    names = set(dir(cls))
    if dataclasses.is_dataclass(cls):
        names |= {f.name for f in dataclasses.fields(cls)}
    for klass in inspect.getmro(cls):
        names |= set(getattr(klass, "__annotations__", {}))
    try:
        tree = ast.parse(inspect.getsource(cls))
    except (OSError, TypeError):
        return names
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and isinstance(node.ctx, ast.Store)
        ):
            names.add(node.attr)
    return names


def resolve_class(dotted: str, importer: Any = importlib.import_module) -> type:
    """``"pkg.mod.Class"`` -> the class. Raises ``LookupError`` naming what is missing."""
    module, _, name = dotted.rpartition(".")
    if not module:
        raise LookupError(f"{dotted!r} is not a dotted <module>.<Class> path")
    try:
        mod = importer(module)
    except Exception as exc:  # ImportError, or whatever the module raises at import
        raise LookupError(f"module {module} does not import ({type(exc).__name__}: {exc})") from exc
    cls = getattr(mod, name, None)
    if not isinstance(cls, type):
        raise LookupError(f"module {module} has no class {name}")
    return cls


def check_reads(
    wanted: dict[str, list[str]], importer: Any = importlib.import_module
) -> tuple[dict[str, dict[str, bool]], dict[str, str]]:
    """(attrs, missing_classes): per found class, attribute -> exists; per missing class, the reason."""
    attrs: dict[str, dict[str, bool]] = {}
    missing: dict[str, str] = {}
    for dotted, names in wanted.items():
        try:
            have = _init_attrs(resolve_class(dotted, importer))
        except LookupError as exc:
            missing[dotted] = str(exc)
            continue
        attrs[dotted] = {n: n in have for n in names}
    return attrs, missing


def main() -> dict[str, Any]:
    wanted = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    import torch
    import vllm
    from vllm.model_executor.models import diffusion_gemma as dg

    text = inspect.getsource(dg)
    seg = _segment(text, r"def _compiled_sample_step\(", r"    return scaled\n")
    tree = ast.parse(seg)
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    params = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    attrs, missing_classes = check_reads(wanted)
    cs_src = inspect.getsource(dg.DiffusionGemmaModelState.custom_sampler)
    returns = [
        n
        for n in ast.walk(ast.parse(cs_src.lstrip() if cs_src.startswith(" ") else cs_src))
        if isinstance(n, ast.Return)
    ]
    arities = [len(r.value.elts) if isinstance(r.value, ast.Tuple) else None for r in returns]
    return {
        "step": {"params": params, "sha256": hashlib.sha256(seg.encode()).hexdigest()},
        "attrs": attrs,
        "missing_classes": missing_classes,
        "custom_sampler": {"return_tuple_arities": arities},
        "versions": {"vllm": vllm.__version__, "torch": torch.__version__},
    }


if __name__ == "__main__":
    try:
        out = main()
    except Exception:
        out = {"error": traceback.format_exc()}
    print(MARKER + json.dumps(out), flush=True)
