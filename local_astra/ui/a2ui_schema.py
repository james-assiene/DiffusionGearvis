"""A2UI v0.9.1 schema registry, message validation and surface topology checks.

Pipeline position (docs/12 §1.1): ``json_repair`` fixer → **this module** (JSON Schema, then
topology) → per-component resample → render. Nothing reaches the HUD's renderer unvalidated.

Vendored schemas (``schemas/a2ui_v0_9_1/``) come from the ``a2ui-agent-sdk`` 0.6.0 wheel,
``a2ui/assets/0.9.1/``. Their ``$id``s live under ``https://a2ui.org/specification/v0_9/``;
``server_to_client.json`` references the catalog by the *relative* ``catalog.json``, which
resolves to ``…/v0_9/catalog.json`` — a URI no vendored file declares. The registry therefore
also binds the active catalog to that URI. That is exactly the hook a custom catalog uses: pass
``catalog=`` and ``anyComponent`` resolves into it.
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from importlib import resources
from types import MappingProxyType
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

A2UI_SPEC_VERSION = "v0.9.1"
_BASE = "https://a2ui.org/specification/v0_9/"
SERVER_TO_CLIENT_ID = _BASE + "server_to_client.json"
COMMON_TYPES_ID = _BASE + "common_types.json"
BASIC_CATALOG_ID = _BASE + "catalogs/basic/catalog.json"
#: URI the message schema's relative ``catalog.json`` reference resolves to.
ACTIVE_CATALOG_ALIAS = _BASE + "catalog.json"

#: catalogId strings the HUD renderer knows (basic catalog under both spec path spellings).
KNOWN_CATALOG_IDS: frozenset[str] = frozenset(
    {BASIC_CATALOG_ID, "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"}
)

#: catalogId of the v2 semantic-card catalog (T-0184): the basic catalog copied verbatim plus
#: seven leaf kinds. Deliberately *not* in ``KNOWN_CATALOG_IDS``: only a validator built with
#: ``catalog=load_cards_v2_catalog()`` knows it.
CARDS_V2_CATALOG_ID = "gearvis:cards/v2"

#: ``date`` / ``date-time`` / ``time`` must all be checked (rfc3339-validator is a dependency):
#: DateTimeInput.min/max is a oneOf over the three formats, so an unchecked format makes every
#: string match twice and fail.
_FORMATS = Draft202012Validator.FORMAT_CHECKER

MESSAGE_KINDS: tuple[str, ...] = (
    "createSurface",
    "updateComponents",
    "updateDataModel",
    "deleteSurface",
)


def _load_vendored(name: str) -> dict[str, Any]:
    with resources.files("local_astra.ui.schemas.a2ui_v0_9_1").joinpath(name).open("rb") as f:
        data: dict[str, Any] = json.load(f)
    return data


def load_cards_v2_catalog() -> dict[str, Any]:
    """A fresh copy of the ``gearvis:cards/v2`` catalog (``schemas/cards_v2/catalog.json``).

    Reads package data only; registers nothing, so ``A2UIValidator()`` stays the basic catalog.
    """
    ref = resources.files("local_astra.ui.schemas.cards_v2").joinpath("catalog.json")
    with ref.open("rb") as f:
        data: dict[str, Any] = json.load(f)
    return data


class OverLimitInt(int):
    """An integer literal longer than ``sys.get_int_max_str_digits()`` digits, never converted.

    CPython refuses ``int()`` of such a literal (T-0185). It stands in for the literal with a value
    beyond the largest double (``±10**309``), so every numeric check treats it like any integer
    ``JSON.parse`` reads as ``Infinity`` (T-0184 C10). ``text`` is the literal as written; it is
    printed nowhere: ``repr``, ``str`` and ``format`` give its digit count only.
    """

    text: str

    def __new__(cls, text: str) -> OverLimitInt:
        self = super().__new__(cls, -(10**309) if text.startswith("-") else 10**309)
        self.text = text
        return self

    def __getnewargs__(self) -> tuple[str]:  # type: ignore[override]
        return (self.text,)

    def __repr__(self) -> str:
        digits = len(self.text.lstrip("-")) - self.text.count("_")
        return f"<integer literal of {digits} digits>"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return repr(self)


def parse_concatenated_json(text: str) -> list[Any]:
    """Parse JSONL *or* pretty-printed concatenated JSON values (what models actually emit).

    An integer literal past the interpreter's int-string limit becomes an ``OverLimitInt``
    instead of raising ``ValueError`` (T-0185); the limit is read, never changed.
    """
    limit = sys.get_int_max_str_digits()

    def parse_int(literal: str) -> int:
        if limit and len(literal) - literal.startswith("-") > limit:
            return OverLimitInt(literal)
        return int(literal)

    dec = json.JSONDecoder(parse_int=parse_int)
    out: list[Any] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, end = dec.raw_decode(text, i)
        out.append(obj)
        i = end
    return out


@dataclass(frozen=True)
class SchemaError:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path or '$'}: {self.message}"


@dataclass(frozen=True)
class TopologyReport:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    #: component ids reachable from root, in BFS order (what a progressive renderer can paint)
    reachable: tuple[str, ...] = ()
    #: ids of components that failed schema validation (candidates for per-component resample)
    invalid_components: tuple[str, ...] = ()


@dataclass(frozen=True)
class SurfaceReport:
    ok: bool
    schema_errors: tuple[SchemaError, ...] = ()
    topology: TopologyReport | None = None
    surface_ids: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()
    problems: tuple[str, ...] = field(default_factory=tuple)


# Reference-carrying properties of the basic catalog (docs/12 §1.1 topology step).
_CHILD_LIST_PROPS: dict[str, tuple[str, ...]] = {
    "Row": ("children",),
    "Column": ("children",),
    "List": ("children",),
}
_SINGLE_CHILD_PROPS: dict[str, tuple[str, ...]] = {
    "Card": ("child",),
    "Button": ("child",),
    "Modal": ("trigger", "content"),
}


def _refs(
    component: Mapping[str, Any],
    list_props: Mapping[str, Sequence[str]],
    single_props: Mapping[str, Sequence[str]],
) -> list[str]:
    kind = component.get("component")
    refs: list[str] = []
    if not isinstance(kind, str):
        return refs
    for prop in list_props.get(kind, ()):
        v = component.get(prop)
        if isinstance(v, list):
            refs.extend(x for x in v if isinstance(x, str))
        elif isinstance(v, dict) and isinstance(v.get("componentId"), str):
            refs.append(v["componentId"])
    for prop in single_props.get(kind, ()):
        v = component.get(prop)
        if isinstance(v, str):
            refs.append(v)
    if kind == "Tabs":
        tabs = component.get("tabs")
        if isinstance(tabs, list):
            refs.extend(
                t["child"] for t in tabs if isinstance(t, dict) and isinstance(t.get("child"), str)
            )
    return refs


def component_children(component: Mapping[str, Any]) -> list[str]:
    """Ids referenced by one basic-catalog component (static lists, single children, tabs, templates)."""
    return _refs(component, _CHILD_LIST_PROPS, _SINGLE_CHILD_PROPS)


class A2UIValidator:
    """Validate A2UI server→client messages against the vendored v0.9.1 schemas.

    ``catalog`` replaces the basic catalog for ``anyComponent`` resolution (custom ``edu``
    catalog, T-0010). ``extra_child_props`` extends the topology walker for custom components
    that reference children, e.g. ``{"Steps": ("children",), "Flashcard": ("front", "back")}``.
    """

    def __init__(
        self,
        catalog: Mapping[str, Any] | None = None,
        *,
        extra_child_list_props: Mapping[str, Sequence[str]] | None = None,
        extra_single_child_props: Mapping[str, Sequence[str]] | None = None,
        known_catalog_ids: Iterable[str] = KNOWN_CATALOG_IDS,
    ) -> None:
        self.server_to_client = _load_vendored("server_to_client.json")
        self.common_types = _load_vendored("common_types.json")
        self.catalog: dict[str, Any] = (
            dict(catalog) if catalog is not None else _load_vendored("catalog.json")
        )
        self._child_list_props = dict(_CHILD_LIST_PROPS)
        self._single_child_props = dict(_SINGLE_CHILD_PROPS)
        for k, v in (extra_child_list_props or {}).items():
            self._child_list_props[k] = tuple(v)
        for k, v in (extra_single_child_props or {}).items():
            self._single_child_props[k] = tuple(v)
        self.known_catalog_ids = frozenset(known_catalog_ids) | {
            str(self.catalog.get("catalogId", ""))
        } - {""}
        self._kind_validators: dict[str, Draft202012Validator] = {}

    @cached_property
    def registry(self) -> Registry[Any]:
        resources_ = [
            (
                SERVER_TO_CLIENT_ID,
                Resource.from_contents(self.server_to_client, default_specification=DRAFT202012),
            ),
            (
                COMMON_TYPES_ID,
                Resource.from_contents(self.common_types, default_specification=DRAFT202012),
            ),
            (
                BASIC_CATALOG_ID,
                Resource.from_contents(
                    _load_vendored("catalog.json"), default_specification=DRAFT202012
                ),
            ),
            (
                ACTIVE_CATALOG_ALIAS,
                Resource.from_contents(self.catalog, default_specification=DRAFT202012),
            ),
        ]
        cat_id = self.catalog.get("$id")
        if isinstance(cat_id, str) and cat_id not in {BASIC_CATALOG_ID, ACTIVE_CATALOG_ALIAS}:
            resources_.append(
                (cat_id, Resource.from_contents(self.catalog, default_specification=DRAFT202012))
            )
        reg: Registry[Any] = Registry().with_resources(resources_)
        return reg

    @cached_property
    def envelope_validator(self) -> Draft202012Validator:
        """Message schema with component items lightened; components are validated per kind.

        Validating 38 components through the catalog's 18-way ``oneOf`` costs ~300 ms; dispatching
        on the ``component`` constant costs ~30 ms for the same surface (measured 2026-09-12,
        jsonschema 4.26). The lightened envelope keeps every other constraint of the message
        schema (required keys, additionalProperties, version enum).
        """
        light = copy.deepcopy(self.server_to_client)
        items = light["$defs"]["UpdateComponentsMessage"]["properties"]["updateComponents"][
            "properties"
        ]["components"]
        items["items"] = {"type": "object", "required": ["id", "component"]}
        return Draft202012Validator(light, registry=self.registry, format_checker=_FORMATS)

    @cached_property
    def message_validator(self) -> Draft202012Validator:
        """Full (slow) message validator; used by tests as the oracle for the fast path."""
        return Draft202012Validator(
            self.server_to_client, registry=self.registry, format_checker=_FORMATS
        )

    @cached_property
    def component_validator(self) -> Draft202012Validator:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": ACTIVE_CATALOG_ALIAS + "#/$defs/anyComponent",
        }
        return Draft202012Validator(schema, registry=self.registry, format_checker=_FORMATS)

    @property
    def child_list_props(self) -> Mapping[str, tuple[str, ...]]:
        """Per kind, the props holding a *list* of child ids (ui_gate prunes entries from these)."""
        return MappingProxyType(self._child_list_props)

    @property
    def single_child_props(self) -> Mapping[str, tuple[str, ...]]:
        """Per kind, the props holding *one* child id — all required, so a dangling one is fatal."""
        return MappingProxyType(self._single_child_props)

    @property
    def component_kinds(self) -> frozenset[str]:
        comps = self.catalog.get("components")
        return frozenset(comps) if isinstance(comps, Mapping) else frozenset()

    def _kind_validator(self, kind: str) -> Draft202012Validator:
        v = self._kind_validators.get(kind)
        if v is None:
            schema = {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$ref": ACTIVE_CATALOG_ALIAS + "#/components/" + kind,
            }
            v = self._kind_validators[kind] = Draft202012Validator(
                schema, registry=self.registry, format_checker=_FORMATS
            )
        return v

    # --- schema -------------------------------------------------------------------------------

    @staticmethod
    def _deepest(e: Any) -> Any:
        best = e
        while best.context:
            best = max(best.context, key=lambda c: len(list(c.absolute_path)))
        return best

    def envelope_errors(self, message: Any) -> list[SchemaError]:
        errs = []
        for e in sorted(
            self.envelope_validator.iter_errors(message), key=lambda e: list(e.absolute_path)
        ):
            best = self._deepest(e)
            errs.append(
                SchemaError(path="/".join(str(p) for p in best.absolute_path), message=best.message)
            )
        return errs

    def component_errors(self, component: Any) -> list[SchemaError]:
        """Errors for one component, dispatching on its ``component`` kind when known."""
        kind = component.get("component") if isinstance(component, Mapping) else None
        if isinstance(kind, str) and kind in self.component_kinds:
            validator = self._kind_validator(kind)
        elif isinstance(kind, str):
            return [SchemaError(path="component", message=f"unknown component kind {kind!r}")]
        else:
            validator = self.component_validator
        found = [self._deepest(e) for e in validator.iter_errors(component)]
        if len(found) > 1:
            # a failing allOf branch also trips unevaluatedProperties; that second error is noise
            found = [e for e in found if e.validator != "unevaluatedProperties"] or found
        return [
            SchemaError(path="/".join(str(p) for p in e.absolute_path), message=e.message)
            for e in found
        ]

    def message_errors(self, message: Any) -> list[SchemaError]:
        """Fast path: envelope, then each component by kind. Same verdict as the full schema."""
        errs = self.envelope_errors(message)
        if errs or not isinstance(message, Mapping):
            return errs
        body = message.get("updateComponents")
        if isinstance(body, Mapping) and isinstance(body.get("components"), list):
            for i, c in enumerate(body["components"]):
                for ce in self.component_errors(c):
                    errs.append(
                        SchemaError(
                            path=f"updateComponents/components/{i}/{ce.path}".rstrip("/"),
                            message=ce.message,
                        )
                    )
        return errs

    def invalid_component_ids(self, components: Sequence[Mapping[str, Any]]) -> list[str]:
        """Ids (or ``#<index>`` when the id is missing) of components that fail the catalog schema."""
        bad = []
        for i, c in enumerate(components):
            if self.component_errors(c):
                cid = c.get("id") if isinstance(c, Mapping) else None
                bad.append(cid if isinstance(cid, str) and cid else f"#{i}")
        return bad

    # --- topology -----------------------------------------------------------------------------

    def _children(self, component: Mapping[str, Any]) -> list[str]:
        return _refs(component, self._child_list_props, self._single_child_props)

    def check_topology(
        self,
        components: Sequence[Any],
        *,
        invalid: Sequence[str] | None = None,
        partial: bool = False,
    ) -> TopologyReport:
        """Ids unique; exactly one ``root``; every reference resolves; no cycles; report orphans.

        ``invalid`` lets the caller pass component ids that already failed schema validation
        (avoids validating twice); when omitted they are computed here. ``partial`` is the
        progressive-render mode (docs/12 §1.1): a reference to a component that has not arrived
        yet, or a missing ``root``, is a warning, not an error — the HUD paints what is reachable.
        """
        errors: list[str] = []
        warnings: list[str] = []
        soft = warnings if partial else errors
        by_id: dict[str, Mapping[str, Any]] = {}
        for i, c in enumerate(components):
            if not isinstance(c, Mapping):
                errors.append(f"component #{i} is not an object")
                continue
            cid = c.get("id")
            if not isinstance(cid, str) or not cid:
                errors.append(f"component #{i} has no id")
                continue
            if cid in by_id:
                errors.append(f"duplicate id {cid!r}")
                continue
            by_id[cid] = c
        if "root" not in by_id:
            soft.append("no component with id 'root'")
        for cid, c in by_id.items():
            for ref in self._children(c):
                if ref not in by_id:
                    soft.append(f"{cid!r} references missing component {ref!r}")
                if ref == cid:
                    errors.append(f"{cid!r} references itself")
        reachable: list[str] = []
        if "root" in by_id:
            seen: set[str] = set()
            stack_path: list[str] = []

            def visit(cid: str) -> None:
                if cid in stack_path:
                    cycle = " -> ".join([*stack_path[stack_path.index(cid) :], cid])
                    if f"cycle: {cycle}" not in errors:
                        errors.append(f"cycle: {cycle}")
                    return
                if cid in seen or cid not in by_id:
                    return
                seen.add(cid)
                reachable.append(cid)
                stack_path.append(cid)
                for ref in self._children(by_id[cid]):
                    visit(ref)
                stack_path.pop()

            visit("root")
            orphans = [cid for cid in by_id if cid not in seen]
            if orphans:
                warnings.append("unreachable from root: " + ", ".join(sorted(orphans)))
        if invalid is None:
            invalid = self.invalid_component_ids(list(components))
        return TopologyReport(
            ok=not errors and not invalid,
            errors=tuple(errors),
            warnings=tuple(warnings),
            reachable=tuple(reachable),
            invalid_components=tuple(invalid),
        )

    # --- whole surface ------------------------------------------------------------------------

    def validate_messages(
        self,
        messages: Sequence[Any],
        *,
        final: bool = True,
        prior: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> SurfaceReport:
        """Validate a sequence of A2UI messages as one surface update (order matters).

        Components accumulate per surface across ``updateComponents`` messages (the spec allows
        many; a later definition of an id replaces the earlier one) and on top of ``prior`` — the
        components already accepted on that surface by earlier canvases. Topology is checked on
        the accumulated tree; with ``final=False`` (progressive render) forward references are
        warnings.
        """
        schema_errors: list[SchemaError] = []
        problems: list[str] = []
        kinds: list[str] = []
        surface_ids: list[str] = []
        created: set[str] = set(prior or ())
        trees: dict[str, dict[str, Mapping[str, Any]]] = {
            sid: {str(c.get("id")): c for c in comps if isinstance(c, Mapping)}
            for sid, comps in (prior or {}).items()
        }
        invalid_by_surface: dict[str, list[str]] = {}
        last_updated: str | None = None
        for i, m in enumerate(messages):
            errs = self.envelope_errors(m)
            if errs:
                schema_errors.extend(
                    SchemaError(path=f"[{i}]/{e.path}", message=e.message) for e in errs
                )
                continue
            kind = next(k for k in MESSAGE_KINDS if k in m)
            kinds.append(kind)
            body = m[kind]
            sid = body["surfaceId"]
            if sid not in surface_ids:
                surface_ids.append(sid)
            if kind == "createSurface":
                if sid in created:
                    problems.append(f"[{i}] createSurface for existing surface {sid!r}")
                created.add(sid)
                trees.setdefault(sid, {})
                if body["catalogId"] not in self.known_catalog_ids:
                    problems.append(f"[{i}] unknown catalogId {body['catalogId']!r}")
            elif sid not in created:
                problems.append(f"[{i}] {kind} for surface {sid!r} before createSurface")
            if kind == "updateComponents":
                comps = body["components"]
                tree = trees.setdefault(sid, {})
                invalid = invalid_by_surface.setdefault(sid, [])
                seen_here: set[str] = set()
                for j, c in enumerate(comps):
                    cid = c.get("id") if isinstance(c, Mapping) else None
                    label = cid if isinstance(cid, str) and cid else f"#{j}"
                    cerrs = self.component_errors(c)
                    if cerrs:
                        invalid.append(label)
                        schema_errors.extend(
                            SchemaError(
                                path=f"[{i}]/updateComponents/components/{j}/{e.path}".rstrip("/"),
                                message=e.message,
                            )
                            for e in cerrs
                        )
                    if isinstance(cid, str) and cid:
                        if cid in seen_here:
                            problems.append(f"[{i}] duplicate id {cid!r}")
                        seen_here.add(cid)
                        tree[cid] = c
                    else:
                        problems.append(f"[{i}] component #{j} has no id")
                last_updated = sid
            if kind == "deleteSurface":
                created.discard(sid)
                trees.pop(sid, None)
        topology: TopologyReport | None = None
        for sid in surface_ids:
            if sid not in invalid_by_surface:
                continue
            report = self.check_topology(
                list(trees.get(sid, {}).values()),
                invalid=sorted(set(invalid_by_surface[sid])),
                partial=not final,
            )
            if sid == last_updated:
                topology = report
            problems.extend(f"surface {sid!r}: {e}" for e in report.errors)
            problems.extend(
                f"surface {sid!r}: invalid component {c}" for c in report.invalid_components
            )
        return SurfaceReport(
            ok=not schema_errors and not problems,
            schema_errors=tuple(schema_errors),
            topology=topology,
            surface_ids=tuple(surface_ids),
            kinds=tuple(kinds),
            problems=tuple(problems),
        )

    def validate_text(self, text: str, *, final: bool = True) -> SurfaceReport:
        try:
            msgs = parse_concatenated_json(text)
        except json.JSONDecodeError as e:
            return SurfaceReport(ok=False, problems=(f"invalid json: {e.msg} at {e.pos}",))
        return self.validate_messages(msgs, final=final)

    def iter_components(self, messages: Sequence[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
        for m in messages:
            body = m.get("updateComponents")
            if isinstance(body, Mapping) and isinstance(body.get("components"), list):
                yield from body["components"]
