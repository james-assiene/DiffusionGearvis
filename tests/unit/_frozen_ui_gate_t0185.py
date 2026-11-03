# Frozen copy of local_astra/conductor/ui_gate.py at main 7646b8b6 (the branch base of T-0185, branch
# T-0185-int-digit-limit), blob faff5376f5bf9440274af4bd54da388ccad72458, sha256 129b0b5e435d82084629b75c3854c480964a29e9afc874c4589a7542d66ca0c0,
# followed by BASE's ``parse_concatenated_json`` (local_astra/ui/a2ui_schema.py at 7646b8b6, blob ab7db0de), verbatim.
# The second part rebinds this module's global ``parse_concatenated_json`` (the first part imports the live one from
# local_astra.ui) to BASE's, so the oracle stays BASE if T-0185 changes that function (T-0185 implementation note 6).
# Never edit. tests/unit/test_t0185_int_digit_limit.py runs it beside the live gate (criteria 9 and 10) and pins both
# parts against a sha256 literal and `git show 7646b8b6:<path>`.
# ruff: noqa
# --- verbatim copy of 7646b8b6:local_astra/conductor/ui_gate.py below this line ---
"""UI gate: ``ui`` segments → validated A2UI surfaces (docs/12 §1.1). Task T-0010.

Pipeline, per canvas commit of the ``ui`` channel:

1. **Fixer** — strict ``parse_concatenated_json`` first; only when that fails, the structural
   (bracket) pass ``local_astra.ui.bracket_repair.repair_structure`` (T-0108: it inserts or
   deletes only ``{ } [ ] , "`` outside strings), then
   ``json_repair.loads(text, stream_stable=True, logging=True)`` on its output (pinned in
   ``services/versions.lock``; skipped, with an empty log, when the structural pass already made
   the text parse). On **both** paths (after the strict parse, and before salvage on the
   unguided one) every ``updateComponents.components`` list then goes through the component
   stage ``local_astra.ui.component_repair.repair_components`` (T-0108b: rules BC, LI, TF, DT,
   CPO, CPV, RB, and T-0127's CC, all-or-nothing per schema-invalid component; a valid component is the identical
   object). ``repaired`` = structural edits + json_repair log entries + component repairs +
   salvage edits, 0 for valid input; ``structural_repairs`` and ``component_repairs`` are their
   terms. Grounding is always against the original canvas, never the repaired text. A canvas nested deeper than the parsers can follow
   yields no surface and a problem naming nesting; the gate never raises on it (criterion 9).
   Schema-guided salvage is applied with a derived schema (local ``$ref`` only,
   ``unevaluatedProperties`` stripped), component by component and only to the components the
   catalog rejects (T-0010 A11). Because every salvage call is handed a one-element *array*,
   the top-level schema violation that makes json_repair raise ``ValueError`` cannot occur:
   that arm is kept as a guard but is unreachable by construction, and the failures that do
   happen are ``Unresolvable`` (a ``$ref`` the fixer cannot follow) and ``KeyError`` (a catalog
   without ``anyComponent``) — each falls back to unguided repair. Behaviours pinned in
   ``tests/unit/test_json_repair_contract.py``.
2. **Schema** — ``A2UIValidator`` envelope + per-component (fast path, ~30 ms per surface).
3. **Topology** — root exists, references resolve, no cycles; orphans are warnings.
4. **Resample** — components in ``TopologyReport.invalid_components`` are re-requested
   individually (single-canvas, ``length_bounded``), at most ``max_resamples`` per turn; the
   surface renders without them meanwhile (their parents drop the reference).
5. **Render** — ``AstraUiSurface`` with ``canvas_idx`` and ``final``. Every component that
   validates is emitted, not only the subset reachable from ``root`` (T-0010 A10): an
   unreachable component is inert in the renderer, and it is usually the node the next canvas
   hangs under ``root``.

Constrained decoding is used only for the envelope / closing tag (two-state automaton, docs/12
§1.4); the body relies on repair + validation.

Two properties hold by construction and are the reason for the shape of the code:

*A repair never invents content* (re-scoped by T-0108b, H-T108b (b) and (d)). Whatever the
fixer returns, every string value in the output of :meth:`UiGate.repair` is a substring of the
input (or of its JSON-escaped rendering). The component stage may rename a key to the catalog
property of that kind (TF: ``placeholder`` → ``label``) or move it (BC: ``children`` → ``child``),
which keeps the model's string; it adds only catalog keys, only ``""`` and ``[]`` as new
non-strings, and creates no id. The one exception is the reserved namespace: RB's Text id
``__gate_<button id>_label`` and event name ``__gate.clicked`` are the only strings the gate
creates, and they alone (with the ``"component": "Text"`` of that one gate-made Text) are exempt
when a repaired component is grounded against the canvas; a repair that grounding would change
is dropped whole and the original passes through. T-0127's CC adds exactly one more: for a
repaired Card ``<id>``, the id ``__gate_<id>_body`` and the ``"component": "Column"`` of that one
gate-made Column. No reserved id is ever resampled.
``json_repair``'s salvage mode fills in schema *defaults* (``variant: "body"``,
``justify: "start"``, ``filterable: false``) and rewrites a value whose shape it does not
understand (``ChoicePicker.value: {"path": "/answers/q1"}`` → ``[]``) — so a salvaged component
is also compared, leaf by leaf, against what the unguided parse held at the same path: a
non-string leaf survives only if it is unchanged, and a component the catalog already accepts
is never handed to salvage at all (T-0010 A11).

*Nothing that fails validation reaches the renderer.* A surface is emitted only when
``A2UIValidator`` reports ``ok`` on the exact messages that are emitted, after pruning. Pruning
is per component: a bad component is dropped and its parents lose the reference, so one bad node
never discards a surface. A component with no ``id`` cannot be re-requested, so it is dropped by
its index and named in ``report.problems`` rather than in ``resample`` (T-0010 A8). A topology
failure that pruning cannot fix (a cycle, T-0010 A2) yields no surface at all.

*An incomplete card is never reported clean* (T-0127, a delta to T-0010's "pruning is reported
only as resample"). When a surface is emitted, ``report.problems`` gains one line per lost id —
a top-level component written on this canvas (after repair) or dropped by salvage that the
accumulated tree does not hold — naming its kind and the reason, and one "reference pruned" line
per reference to an id defined nowhere (not a loss). ``GateResult.lost`` holds those ids, recomputed
per call; ``GateResult.complete`` is true iff a surface was emitted, the accumulated tree holds
``root``, ``root`` reaches every component in it, and ``lost`` is empty. ``ok`` keeps its meaning
(safe to render) and still alone decides emission. With no surface, ``lost`` is empty and no line
is added.

*One surface per turn.* The gate rewrites ``surfaceId`` in every emitted message to
``f"{surface_id_prefix}-{turn_id}"`` (T-0010 A3/A7). The renderer keys on the id *inside* the
messages, so leaving the model's own ``lesson_1`` in place would make turn 10 morph turn 9's
surface instead of mounting a new one. A canvas that creates more than one surface keeps the
first ``createSurface``'s surface and drops the other surfaces' messages, naming them in
``report.problems`` — v1 renders one surface per turn, and a second one is a spec question.

Because a reference to a component that has not been generated yet is pruned (otherwise the
partial canvas would not validate), the gate keeps every accepted component twice — as emitted
and as written — and re-sends a parent with the reference restored on the canvas where the child
finally arrives. That is the morph docs/12 §1.1 asks for; without it a late child stays an orphan.
The as-written copy of an id is updated only on a canvas that emits that id, and a copy the
catalog rejects is never restored (T-0108b criterion 8): a re-started fence cut mid-component
must not overwrite an accepted component with its stub.
A container whose *required* single child is missing (``Card``/``Button``/``Modal``) cannot be
emitted at all, so it is remembered as deferred and re-emitted, with its parent, on the canvas
that brings the child (T-0010 A9).

Q11 metrics (criterion 6) come from ``GateResult``: ``ui_repairs`` = ``repaired``,
``ui_resamples`` = ``len(resample)``, ``ui_valid`` = ``report.ok``. Writing them into
``TurnRecord.extras`` belongs to the caller (``session.py``), not here.

*One catalog per surface* (T-0184 C1/C2). The gate holds one validator per catalog — the basic
one (``validator``) and the ``gearvis:cards/v2`` one — and never merges them: a v2 kind on a basic
surface stays invalid. The ``catalogId`` of the first ``createSurface`` the canvas carries picks
the validator, and the choice is remembered for the turn, so a later canvas with no
``createSurface`` is still checked against it; the component stage, salvage (with that catalog's
fixer schema), pruning and validation all use it. Any id other than the v2 one gets the basic
validator, which is BASE's behaviour byte for byte.

*No non-finite number* (T-0184, hand-off from the group A review). ``parse_concatenated_json``
accepts ``NaN``/``Infinity`` and JSON Schema's bounds let them through, so both validators report
a non-finite number anywhere in a component (or in a message outside its components) as a schema
error. The component is then invalid like any other: pruned, lost and resampled.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

import json_repair
from referencing.exceptions import Unresolvable

from local_astra.contracts.protocol import AstraUiSurface
from local_astra.ui import (
    A2UI_SPEC_VERSION,
    MESSAGE_KINDS,
    A2UIValidator,
    SurfaceReport,
    parse_concatenated_json,
)
from local_astra.ui.a2ui_schema import CARDS_V2_CATALOG_ID, SchemaError, load_cards_v2_catalog
from local_astra.ui.bracket_repair import repair_structure
from local_astra.ui.component_repair import (
    RESERVED_EVENT,
    RESERVED_PREFIX,
    repair_components,
    reserved_body_id,
    reserved_label_id,
)
from local_astra.ui.fixer_schema import components_array_schema

#: ``A2UIValidator`` reports a re-``createSurface`` as a problem; T-0010 A4 says a re-send within
#: the same turn is idempotent, so exactly that problem is forgiven (nothing else is).
_DUPLICATE_CREATE = re.compile(
    r"^\[(?P<idx>\d+)\] createSurface for existing surface '(?P<sid>.*)'$"
)
#: T-0184 C9: the catalogId of such a re-send is ignored too (the HUD answers "Surface already
#: exists" and keeps the first catalog), so its "unknown catalogId" at the same index is forgiven.
_UNKNOWN_CATALOG = re.compile(r"^\[(?P<idx>\d+)\] unknown catalogId ")

#: sentinel for "this value was not in the model's output" — see ``_grounded``
_UNGROUNDED = object()

#: ``_grounded``'s ``original``: "there is nothing to compare against" (the unguided fixer path,
#: where only the string rule applies) and "the original had no value at this path" (A11).
_ANY = object()
_ABSENT = object()


class _FiniteOnly(A2UIValidator):
    """``inner`` with every non-finite number reported as a schema error (T-0184 hand-off).

    A view of ``inner``: it shares its schemas and caches and changes nothing for finite input.
    ``validate_messages`` and ``message_errors`` reach the two overrides through ``self``, and the
    component stage calls ``component_errors``, so every verdict of the gate sees the refusal.
    """

    def __init__(self, inner: A2UIValidator) -> None:
        vars(self).update(vars(inner))

    def component_errors(self, component: Any) -> list[SchemaError]:
        return [*super().component_errors(component), *_non_finite_errors(component, "")]

    def envelope_errors(self, message: Any) -> list[SchemaError]:
        errors = super().envelope_errors(message)
        if not isinstance(message, Mapping):
            return errors
        body = message.get("updateComponents")
        if isinstance(body, Mapping):
            # components are checked one by one by ``component_errors``
            rest = {k: v for k, v in body.items() if k != "components"}
            message = {**message, "updateComponents": rest}
        return [*errors, *_non_finite_errors(message, "")]


def _non_finite_errors(node: Any, path: str) -> Iterator[SchemaError]:
    """A schema error per ``NaN``/``Infinity``/``-Infinity`` leaf of ``node``, at its path.

    An integer beyond ``sys.float_info.max`` counts too (T-0184 C10): ``JSON.parse`` makes it
    ``Infinity``. The int/float comparison is exact, and the message never prints the integer.
    """
    if isinstance(node, float) and not math.isfinite(node):
        yield SchemaError(path=path, message=f"{node!r} is not a finite number")
    elif isinstance(node, int) and not isinstance(node, bool) and abs(node) > sys.float_info.max:
        yield SchemaError(
            path=path, message="integer beyond the float range is not a finite number"
        )
    elif isinstance(node, Mapping):
        for key, value in node.items():
            yield from _non_finite_errors(value, f"{path}/{key}" if path else str(key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _non_finite_errors(value, f"{path}/{i}" if path else str(i))


def _finite_only(validator: A2UIValidator) -> A2UIValidator:
    return validator if isinstance(validator, _FiniteOnly) else _FiniteOnly(validator)


@lru_cache(maxsize=1)
def _cards_v2_validator() -> A2UIValidator:
    """The ``gearvis:cards/v2`` validator, shared by every gate (it holds only schemas and caches)."""
    return _FiniteOnly(A2UIValidator(catalog=load_cards_v2_catalog()))


@dataclass(frozen=True)
class GateResult:
    surface: AstraUiSurface | None
    report: SurfaceReport
    #: component ids to re-request individually
    resample: tuple[str, ...] = ()
    repaired: int = 0
    #: T-0108: the part of ``repaired`` made by the structural (bracket) pass
    structural_repairs: int = 0
    #: T-0108b: the part of ``repaired`` made by the component stage (``repair_components``)
    component_repairs: int = 0
    #: T-0127 criterion 4: a surface was emitted, the accumulated tree holds ``root``, every
    #: component in it is reachable from ``root``, and ``lost`` is empty. ``ok`` is unchanged.
    complete: bool = False
    #: T-0127 criterion 3: ids written on this canvas (after repair) or dropped by salvage that the
    #: accumulated tree does not hold; never a pruned reference; empty when nothing was emitted
    lost: tuple[str, ...] = ()
    #: whether the accumulated tree holds ``root``, and its ids ``root`` does not reach (for the
    #: incomplete-final note, T-0127 criterion 6)
    root: bool = False
    unreachable: tuple[str, ...] = ()


class _TooDeep(Exception):
    """The canvas nests deeper than a parser can follow (T-0108 criterion 9); never escapes."""


@dataclass(frozen=True)
class _Pruned:
    """What pruning kept, and the two kinds of drop the caller has to react to."""

    messages: list[Any]
    #: per surface, components dropped because a *required* child had not arrived yet (T-0010 A9);
    #: they are re-checked on every later canvas of the turn
    deferred: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: ``#<index>`` labels of id-less components dropped — reported, never resampled (T-0010 A8)
    idless: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Repair:
    """What the fixer produced, plus what it had to throw away."""

    messages: list[Any]
    edits: int
    #: ids the schema-guided salvage dropped; they still deserve a resample request
    dropped: tuple[str, ...] = ()
    #: T-0127: the kind each of ``dropped`` was written as (for its loss line)
    dropped_kinds: Mapping[str, str] = field(default_factory=dict)
    #: the structural edits included in ``edits`` (T-0108)
    structural: int = 0
    #: the component-stage edits included in ``edits`` (T-0108b)
    component_repairs: int = 0


@dataclass
class UiGate:
    validator: A2UIValidator = field(default_factory=A2UIValidator)
    max_resamples: int = 3
    surface_id_prefix: str = "turn"

    #: turn the accumulated state below belongs to; a new turn_id resets everything
    _turn_id: str | None = field(default=None, init=False, repr=False)
    #: per surface, the components already accepted this turn, *as emitted* (references pruned)
    _emitted: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    #: per surface, the same components *as the model wrote them* (used to restore references
    #: once a forward-referenced component finally arrives — docs/12 §1.1 progressive render)
    _as_written: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    #: per surface, components dropped on an earlier canvas because a required single child
    #: was missing; ``_restore_references`` re-checks them on every later canvas (T-0010 A9)
    _deferred: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _resamples_left: int = field(default=0, init=False, repr=False)
    #: per catalog (``_FiniteOnly`` view of ``validator``, or the v2 one), the derived fixer schema
    _fixer_schemas: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    #: T-0184 C1: the catalogId the turn's surface was created with (``None`` until one is seen)
    _catalog_id: str | None = field(default=None, init=False, repr=False)
    #: T-0184 C9a: set once a surface of the turn is emitted under a known ``_catalog_id``; from
    #: then on a later ``createSurface`` never changes the catalog
    _catalog_locked: bool = field(default=False, init=False, repr=False)
    _basic: A2UIValidator = field(init=False, repr=False)
    _cards_v2: A2UIValidator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._basic = _finite_only(self.validator)
        self._cards_v2 = _cards_v2_validator()

    @property
    def _validator(self) -> A2UIValidator:
        """The validator of this turn's catalog (T-0184 C1): v2 for its id, else the basic one."""
        return self._cards_v2 if self._catalog_id == CARDS_V2_CATALOG_ID else self._basic

    def _note_catalog(self, messages: Sequence[Any]) -> None:
        """Remember the catalogId of the canvas's first ``createSurface`` for the rest of the turn.

        A canvas without one (a later canvas of the turn, a cut before it) keeps the choice. Once
        the turn has emitted a surface the choice is locked (T-0184 C9/C9a: the first catalog the
        HUD saw wins), so a later ``createSurface`` naming another catalog does not change it.
        """
        if self._catalog_locked:
            return
        for message in messages:
            body = message.get("createSurface") if isinstance(message, Mapping) else None
            if isinstance(body, Mapping):
                catalog_id = body.get("catalogId")
                if isinstance(catalog_id, str):
                    self._catalog_id = catalog_id
                return

    # --- 1. fixer -----------------------------------------------------------------------------

    def repair(self, raw: str) -> tuple[list[Any], int]:
        """Fixer step: returns (messages, number_of_repairs)."""
        result = self._repair(raw)
        return result.messages, result.edits

    def _repair(self, raw: str) -> _Repair:
        try:
            strict = parse_concatenated_json(raw)
        except json.JSONDecodeError:
            pass
        except RecursionError as err:
            raise _TooDeep(f"the canvas nests deeper than the parser can follow ({err})") from err
        else:
            # T-0010 A5 as amended by T-0108b: the fixer never sees well-formed JSON; only the
            # component stage edits a parseable canvas, and only its schema-invalid components.
            self._note_catalog(strict)
            kept: list[Any] = []
            component_edits = 0
            for item in strict:
                if isinstance(item, dict):
                    repaired, n = self._repair_components(item, raw)
                    kept.append(repaired)
                    component_edits += n
            return _Repair(messages=kept, edits=component_edits, component_repairs=component_edits)

        # T-0108: the structural pass runs only on text that failed to parse, and only inserts
        # or deletes ``{ } [ ] , "`` outside strings; grounding below stays against ``raw``.
        text, structural = repair_structure(raw)
        fixed, log = _fix_unguided(text, repaired=structural > 0)
        # the unguided fixer only drops and closes, so grounding it needs the string rule only;
        # it is also the reference the salvaged components are compared against below.
        parsed = [_grounded(m, raw) for m in _as_list(fixed) if isinstance(m, dict)]
        self._note_catalog(parsed)  # T-0184 C1: before the component stage and salvage
        edits = structural + len(log)
        component_edits = 0
        dropped: list[str] = []
        dropped_kinds: dict[str, str] = {}
        messages: list[Any] = []
        for message in parsed:
            # T-0108b: the component stage runs before salvage, so json_repair's salvage never
            # fills in defaults on a component the stage can repair from the model's own data.
            message, n = self._repair_components(message, raw)
            component_edits += n
            body = message.get("updateComponents")
            if not isinstance(body, dict) or not isinstance(body.get("components"), list):
                messages.append(message)
                continue
            salvaged, salvage_edits = self._salvage(body["components"], raw)
            missing = _missing_ids(body["components"], salvaged)
            dropped.extend(missing)
            dropped_kinds.update(
                (_id_of(c), _kind_of(c)) for c in body["components"] if _id_of(c) in missing
            )
            edits += salvage_edits
            messages.append({**message, "updateComponents": {**body, "components": salvaged}})
        return _Repair(
            messages=messages,
            edits=edits + component_edits,
            dropped=tuple(dropped),
            dropped_kinds=dropped_kinds,
            structural=structural,
            component_repairs=component_edits,
        )

    def _repair_components(self, message: dict[str, Any], raw: str) -> tuple[dict[str, Any], int]:
        """T-0108b: ``message`` with its components through ``repair_components``.

        Every component the stage rewrote or created is grounded against the original canvas
        (C7): its string values must be the model's, except exactly the reserved namespace of
        that repair (``__gate.clicked``, ``__gate_<id>_label`` for the repaired Button ``<id>``,
        and the ``"component": "Text"`` of that one gate-made Text). All-or-nothing holds after
        grounding (2026-10-17 clarification): if grounding would change any component of a
        repair, the original component passes through unchanged instead. Components that come
        back as the identical input object are passed through.
        """
        body = message.get("updateComponents")
        if not isinstance(body, dict) or not isinstance(body.get("components"), list):
            return message, 0
        before: list[Any] = body["components"]
        after, edits = repair_components(before, self._validator)
        if after is before:
            return message, 0
        groups = _repair_groups(before, after)
        if groups is None:
            return message, 0  # not in the stage's documented order: keep the canvas as written
        out: list[Any] = []
        for original, items in groups:
            if items[0] is original or all(_repair_grounded(c, original, raw) for c in items):
                out.extend(items)
                continue
            out.append(original)
            edits -= repair_components([original], self._validator)[1]
        if len(out) == len(before) and all(a is b for a, b in zip(out, before, strict=True)):
            return message, 0
        return {**message, "updateComponents": {**body, "components": out}}, max(edits, 0)

    def _salvage(self, components: list[Any], raw: str) -> tuple[list[Any], int]:
        """Schema-guided salvage of the components the catalog rejects, in the original order.

        T-0010 A11(a): a component that validates on its own is passed through byte-identical.
        Salvage mode fills in defaults and rewrites what it does not understand, so running it
        over a *valid* neighbour of a broken component destroys data bindings the model got right
        (``ChoicePicker.value: {"path": "/answers/q1"}`` came back as ``[]``). Each failing
        component is salvaged alone, so neither the drop nor the rewrite of one component can
        reach another, and what comes back is re-grounded against the component as parsed.
        """
        schema = self._schema()
        if schema is None:
            return components, 0
        out: list[Any] = []
        edits = 0
        for component in components:
            if not self._validator.component_errors(component):
                out.append(component)
                continue
            salvaged, salvage_edits = _salvage_one(component, schema)
            out.extend(
                g for c in salvaged if (g := _grounded(c, raw, component)) is not _UNGROUNDED
            )
            edits += salvage_edits
        return out, edits

    def _schema(self) -> dict[str, Any] | None:
        """The derived fixer schema of this turn's catalog (T-0184 C1), or ``None`` when that
        catalog cannot produce one."""
        validator = self._validator
        key = "cards_v2" if validator is self._cards_v2 else "basic"
        schema = self._fixer_schemas.get(key)
        if schema is None:
            try:
                schema = components_array_schema(validator.catalog, validator.common_types)
            except KeyError:
                return None  # no ``anyComponent``: repair unguided rather than guess
            self._fixer_schemas[key] = schema
        return schema

    # --- 2-5. gate ----------------------------------------------------------------------------

    def gate(self, raw: str, *, turn_id: str, canvas_idx: int, final: bool) -> GateResult:
        self._begin_turn(turn_id)
        # T-0108 criterion 9: the gate returns a result for every canvas. A canvas nested deeper
        # than the parsers (or the recursive grounding and validation after them) can follow is
        # one from which no A2UI message can be recovered.
        try:
            return self._gate(raw, turn_id=turn_id, canvas_idx=canvas_idx, final=final)
        except _TooDeep as err:
            why = str(err)
        except RecursionError as err:
            why = f"the canvas nests deeper than the gate can follow ({err})"
        problem = f"no A2UI message could be recovered from the canvas: {why}"
        return GateResult(None, SurfaceReport(ok=False, problems=(problem,)))

    def _gate(self, raw: str, *, turn_id: str, canvas_idx: int, final: bool) -> GateResult:
        # T-0010 A3/A7: one stable surface id per turn, rewritten *inside* the messages, so that
        # canvas k morphs canvas k-1 and turn n+1 mounts instead of morphing turn n's surface.
        surface_id = f"{self.surface_id_prefix}-{turn_id}"
        repair = self._repair(raw)
        structural = repair.structural
        if not repair.messages:
            problem = "no A2UI message could be recovered from the canvas"
            has_root, unreachable = self._shape(surface_id)
            return GateResult(
                None,
                SurfaceReport(ok=False, problems=(problem,)),
                (),
                repair.edits,
                structural,
                repair.component_repairs,
                root=has_root,
                unreachable=unreachable,
            )

        scoped, notes = self._single_surface(repair.messages, surface_id)
        components = list(self._validator.iter_components(scoped))
        labels = self._validator.invalid_component_ids(components)
        invalid = [i for i in labels if not _is_index(i)]
        known = {cid for tree in self._emitted.values() for cid in tree}
        pruned = self._prune(scoped, labels, known=known)
        messages = pruned.messages + self._restore_references(pruned.messages, _by_surface(scoped))
        notes.extend(
            f"component {label} has no id and was dropped (it cannot be resampled)"
            for label in pruned.idless
        )

        report = self._noted(
            self._forgive_idempotent_create(
                self._validator.validate_messages(messages, final=final, prior=self._prior()),
                messages,
            ),
            notes,
        )
        resample = self._budgeted(invalid, repair.dropped)
        if not report.ok:
            # T-0127 ruling: with no surface, ``lost`` is empty and no loss line is added
            has_root, unreachable = self._shape(surface_id)
            return GateResult(
                None,
                report,
                resample,
                repair.edits,
                structural,
                repair.component_repairs,
                root=has_root,
                unreachable=unreachable,
            )

        self._remember(messages, scoped, pruned.deferred)
        # T-0184 C9a: the catalog locks at the first surface emitted, never to an unknown id
        if self._catalog_id in self._known_catalog_ids():
            self._catalog_locked = True
        tree = self._emitted.get(surface_id, {})
        lost, lines = self._losses(components, repair, tree, set(invalid), pruned, surface_id)
        report = self._noted(report, lines)
        has_root, unreachable = self._shape(surface_id)
        surface = AstraUiSurface(
            surface_id=surface_id,
            canvas_idx=canvas_idx,
            final=final,
            messages=tuple(messages),
            repaired=repair.edits,
        )
        return GateResult(
            surface,
            report,
            resample,
            repair.edits,
            structural,
            repair.component_repairs,
            complete=has_root and not unreachable and not lost,
            lost=lost,
            root=has_root,
            unreachable=unreachable,
        )

    # --- T-0127: losses and completeness --------------------------------------------------------

    def _losses(
        self,
        components: Sequence[Any],
        repair: _Repair,
        tree: Mapping[str, Any],
        invalid: set[str],
        pruned: _Pruned,
        surface_id: str,
    ) -> tuple[tuple[str, ...], list[str]]:
        """``(lost, problem lines)`` of this call (T-0127 criterion 3), recomputed per call.

        A component written on this canvas (after repair, top level only) or dropped by salvage
        is lost when the accumulated tree does not hold its id; reserved ids are the gate's own and
        count as emitted. A reference to an id that is defined nowhere is a pruned reference: it
        gets a line, and it is not a loss.
        """
        deferred = pruned.deferred.get(surface_id, {})
        written: dict[str, str] = {}
        for component in components:
            cid = _id_of(component)
            if cid:
                written.setdefault(cid, _kind_of(component))
        for cid in repair.dropped:
            written.setdefault(cid, repair.dropped_kinds.get(cid, "unknown kind"))
        lost: list[str] = []
        lines: list[str] = []
        for cid, kind in written.items():
            if cid in tree or cid.startswith(RESERVED_PREFIX):
                continue
            if cid in repair.dropped:
                why = "dropped by salvage"
            elif cid in invalid:
                why = "the catalog rejects it and no repair applies"
            elif cid in deferred:
                why = "its required child is missing"
            else:
                why = "pruned after a component it references was dropped"
            lost.append(cid)
            lines.append(f"component {cid} ({kind}) was written but not emitted: {why}")
        pruned_refs: list[str] = []
        for component in components:
            if not isinstance(component, Mapping):
                continue
            for ref in self._refs(component):
                if ref in written or ref in tree or ref in pruned_refs:
                    continue
                pruned_refs.append(ref)
                lines.append(
                    f"reference pruned: {_id_of(component) or '(no id)'} references {ref}, "
                    "which was never defined"
                )
        return tuple(lost), lines

    def _shape(self, surface_id: str) -> tuple[bool, tuple[str, ...]]:
        """Whether the accumulated tree holds ``root``, and the ids ``root`` does not reach."""
        tree = self._emitted.get(surface_id, {})
        reached: set[str] = set()
        stack = ["root"] if "root" in tree else []
        while stack:
            cid = stack.pop()
            if cid in reached or cid not in tree:
                continue
            reached.add(cid)
            stack.extend(self._refs(tree[cid]))
        return "root" in tree, tuple(cid for cid in tree if cid not in reached)

    def _refs(self, component: Mapping[str, Any]) -> list[str]:
        """The ids ``component`` references (static lists, templates, single children, tabs)."""
        kind = component.get("component")
        if not isinstance(kind, str):
            return []
        refs: list[str] = []
        for prop in self._validator.child_list_props.get(kind, ()):
            value = component.get(prop)
            if isinstance(value, list):
                refs.extend(v for v in value if isinstance(v, str))
            elif isinstance(value, Mapping) and isinstance(value.get("componentId"), str):
                refs.append(value["componentId"])
        for prop in self._validator.single_child_props.get(kind, ()):
            value = component.get(prop)
            if isinstance(value, str):
                refs.append(value)
        tabs = component.get("tabs")
        if kind == "Tabs" and isinstance(tabs, list):
            refs.extend(c for t in tabs if (c := _tab_child(t)) is not None)
        return refs

    # --- A7: one surface per turn ---------------------------------------------------------------

    def _single_surface(
        self, messages: Sequence[Any], surface_id: str
    ) -> tuple[list[Any], list[str]]:
        """Messages of the one surface this turn owns, with ``surfaceId`` rewritten to it.

        The renderer keys on the id inside the messages, so every emitted message carries
        ``surface_id``. A canvas that creates more than one surface keeps the first
        ``createSurface``'s surface and drops the others, naming them (T-0010 A7); v1 renders one
        surface per turn and merging two of them would silently collide ids.
        """
        created: list[str] = []
        for message in messages:
            kind, sid = _message_surface(message)
            if kind == "createSurface" and sid is not None and sid not in created:
                created.append(sid)
        extra = set(created[1:])
        notes = [
            f"canvas creates more than one surface: messages for {sid!r} dropped, "
            f"{created[0]!r} kept (v1 renders one surface per turn)"
            for sid in created[1:]
        ]
        kept: list[Any] = []
        for message in messages:
            _, sid = _message_surface(message)
            if sid in extra:
                continue
            kept.append(_rewritten(message, surface_id))
        return kept, notes

    def _noted(self, report: SurfaceReport, notes: Sequence[str]) -> SurfaceReport:
        """Report gate-level drops without changing the verdict.

        ``ok`` is the validator's verdict on the messages that are actually emitted; a dropped
        second surface (A7) and a dropped id-less component (A8) are reported, not fatal — what
        is emitted is still exactly what the validator approved.
        """
        if not notes:
            return report
        return replace(report, problems=(*report.problems, *notes))

    # --- pruning ------------------------------------------------------------------------------

    def prune_invalid(
        self, messages: Sequence[Any], invalid: Sequence[str], *, known: Iterable[str] = ()
    ) -> list[Any]:
        """Drop invalid components and the references to them so the rest can render.

        ``invalid`` holds the labels ``A2UIValidator.invalid_component_ids`` returns: component
        ids, or ``#<index>`` for a component with no id — those are dropped by their index in the
        canvas (T-0010 A8), because an id-less component cannot be re-requested and would
        otherwise take the whole surface down with it.

        ``known`` are ids that exist outside ``messages`` (components accepted by earlier canvases
        of this turn): a reference to one of those resolves and must survive. Dropping a component
        can invalidate its parent — a ``Card`` without its required ``child``, a ``Column`` whose
        ``children`` list empties — so the pass repeats until it reaches a fixed point.
        """
        return self._prune(messages, invalid, known=known).messages

    def _prune(
        self, messages: Sequence[Any], invalid: Sequence[str], *, known: Iterable[str] = ()
    ) -> _Pruned:
        """:meth:`prune_invalid` plus what the caller has to remember: the components dropped for
        a child that has not arrived yet (A9) and the id-less ones dropped by index (A8)."""
        original: list[list[Any] | None] = [_components_of(m)[1] for m in messages]
        by_index = {label[1:] for label in invalid if _is_index(label)}
        idless: list[str] = []
        working: list[list[Any]] = []
        flat = 0
        for comps in original:
            kept_here: list[Any] = []
            for component in comps or ():
                if str(flat) in by_index:
                    idless.append(f"#{flat}")
                else:
                    kept_here.append(component)
                flat += 1
            working.append(kept_here)
        # pre-pruning shape of every component, so a deferred one is remembered as written
        as_written = {_id_of(c): c for comps in original for c in comps or () if _id_of(c)}
        dropped = {label for label in invalid if not _is_index(label)}
        deferred: dict[str, dict[str, Any]] = {}

        while True:
            changed = False
            present = {_id_of(c) for comps in working for c in comps} | set(known)
            present -= dropped
            present.discard("")
            for i, comps in enumerate(working):
                surface_id = _components_of(messages[i])[0]
                for j, component in enumerate(comps):
                    cid = _id_of(component)
                    if cid in dropped:
                        continue
                    pruned = self._prune_refs(component, present)
                    if pruned is None:
                        dropped.add(cid)
                        if cid and surface_id is not None:
                            # A9: a required child that has not been generated yet — the canvas
                            # that brings it re-emits this component instead of losing it.
                            held = as_written.get(cid, component)
                            deferred.setdefault(surface_id, {})[cid] = held
                        changed = True
                        continue
                    if pruned is component:
                        continue
                    comps[j] = pruned
                    changed = True
                    if self._validator.component_errors(pruned):
                        dropped.add(cid)
            if not changed:
                break

        out: list[Any] = []
        for message, before, after in zip(messages, original, working, strict=True):
            if before is None:
                out.append(message)
                continue
            kept = [c for c in after if _id_of(c) not in dropped]
            if not kept:
                continue  # an updateComponents with no components left is not worth emitting
            if kept == before:
                out.append(message)
                continue
            body = dict(message["updateComponents"])
            body["components"] = kept
            out.append({**message, "updateComponents": body})
        return _Pruned(messages=out, deferred=deferred, idless=tuple(idless))

    def _prune_refs(self, component: Any, present: set[str]) -> Any:
        """Component with references to absent ids removed; ``None`` when it cannot survive."""
        if not isinstance(component, Mapping):
            return component
        kind = component.get("component")
        if not isinstance(kind, str):
            return component
        pruned: dict[str, Any] | None = None
        for prop in self._validator.child_list_props.get(kind, ()):
            value = component.get(prop)
            if isinstance(value, list):
                kept = [v for v in value if not isinstance(v, str) or v in present]
                if kept != value:
                    pruned = dict(component) if pruned is None else pruned
                    pruned[prop] = kept
            elif isinstance(value, Mapping) and isinstance(value.get("componentId"), str):
                # a data-bound template: without its template component there is nothing to repeat
                if value["componentId"] not in present:
                    return None
        for prop in self._validator.single_child_props.get(kind, ()):
            value = component.get(prop)
            if isinstance(value, str) and value not in present:
                return None  # Card.child / Button.child / Modal.trigger are required
        tabs = component.get("tabs")
        if kind == "Tabs" and isinstance(tabs, list):
            kept_tabs = [t for t in tabs if _tab_child(t) is None or _tab_child(t) in present]
            if kept_tabs != tabs:
                pruned = dict(component) if pruned is None else pruned
                pruned["tabs"] = kept_tabs
        return component if pruned is None else pruned

    def _restore_references(
        self, messages: Sequence[Any], fresh: Mapping[str, Mapping[str, Any]]
    ) -> list[Any]:
        """Re-emit earlier parents whose pruned references resolve now (progressive render).

        Canvas *k-1* referenced a component that had not been generated yet; the reference was
        pruned so that what was rendered validated. When canvas *k* brings that component, the
        parent is re-sent with the reference restored — A2UI replaces a component by id, so this
        is exactly the morph docs/12 §1.1 describes. Without it a late child would stay an orphan.
        For an id this canvas emits, the fresh definition wins (T-0127 criterion 7 as amended,
        ec7966f): its references are restored from *this* canvas's as-written copy, ``fresh``, and
        compared with what this canvas emits for it, never with or from an older copy (restoring
        the older one re-sent a stale definition after the model's own).
        """
        arriving: dict[str, dict[str, Any]] = {}
        for message in messages:
            sid, comps = _components_of(message)
            if sid is None:
                continue
            here = arriving.setdefault(sid, {})
            for c in comps or ():
                here[_id_of(c)] = c
        restored: list[Any] = []
        for surface_id in dict.fromkeys([*self._as_written, *self._deferred]):
            written = self._as_written.get(surface_id, {})
            emitted_now = arriving.get(surface_id, {})
            fresh_here = fresh.get(surface_id, {})
            present = set(self._emitted.get(surface_id, {})) | set(emitted_now)
            present.discard("")
            components: list[Any] = []
            components.extend(self._restore_deferred(surface_id, present, set(emitted_now)))
            candidates: dict[str, tuple[Any, Any]] = {}
            for cid, component in written.items():
                if cid in emitted_now:
                    if cid in fresh_here:
                        candidates[cid] = (fresh_here[cid], emitted_now[cid])
                elif cid in self._emitted.get(surface_id, {}):
                    candidates[cid] = (component, self._emitted[surface_id][cid])
                # else: the component itself was dropped; nothing to restore it to
            for cid, emitted in emitted_now.items():
                if cid and cid not in candidates and cid in fresh_here:
                    candidates[cid] = (fresh_here[cid], emitted)
            for component, emitted in candidates.values():
                pruned = self._prune_refs(component, present)
                if pruned is None or pruned == emitted:
                    continue
                if self._validator.component_errors(pruned):
                    continue  # T-0108b criterion 8: never re-emit a copy the catalog rejects
                components.append(pruned)
            if components:
                restored.append(
                    {
                        "version": A2UI_SPEC_VERSION,
                        "updateComponents": {"surfaceId": surface_id, "components": components},
                    }
                )
        return restored

    def _restore_deferred(
        self, surface_id: str, present: set[str], arriving: set[str]
    ) -> list[Any]:
        """Components held back on an earlier canvas whose required child has arrived (A9).

        ``Card``/``Button``/``Modal`` cannot be emitted without their single child, so pruning
        dropped them outright; a dropped component is not in ``_as_written``, so the reference
        restoration above would never look at it again. They are re-checked here, and ``present``
        grows as they come back (a chain of containers restores bottom-up in one call). The caller
        adds the restored ids to ``_emitted``; ``_remember`` then clears them from ``_deferred``.
        An id the model re-sends on this canvas is left alone: the fresh definition wins.
        """
        pending = {
            cid: c for cid, c in self._deferred.get(surface_id, {}).items() if cid not in arriving
        }
        restored: list[Any] = []
        while pending:
            back = []
            for cid, component in pending.items():
                pruned = self._prune_refs(component, present)
                if pruned is None or self._validator.component_errors(pruned):
                    continue
                back.append((cid, pruned))
            if not back:
                break
            for cid, pruned in back:
                del pending[cid]
                present.add(cid)
                restored.append(pruned)
        return restored

    # --- per-turn state -----------------------------------------------------------------------

    def _begin_turn(self, turn_id: str) -> None:
        if turn_id == self._turn_id:
            return
        self._turn_id = turn_id
        self._emitted = {}
        self._as_written = {}
        self._deferred = {}
        self._resamples_left = self.max_resamples
        self._catalog_id = None  # T-0184 C1: a turn never inherits the previous turn's catalog
        self._catalog_locked = False

    def _prior(self) -> dict[str, list[Any]]:
        return {sid: list(tree.values()) for sid, tree in self._emitted.items()}

    def _remember(
        self,
        emitted: Sequence[Any],
        as_written: Sequence[Any],
        deferred: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        #: per surface, the ids emitted on *this* canvas (T-0108b criterion 8)
        now: dict[str, set[str]] = {}
        for message in emitted:
            surface_id, comps = _components_of(message)
            if surface_id is None:
                continue
            self._emitted.setdefault(surface_id, {})
            self._as_written.setdefault(surface_id, {})
            for component in comps or ():
                cid = _id_of(component)
                if cid:
                    self._emitted[surface_id][cid] = component
                    now.setdefault(surface_id, set()).add(cid)
        for message in as_written:
            surface_id, comps = _components_of(message)
            if surface_id is None or surface_id not in self._as_written:
                continue
            for component in comps or ():
                # T-0108b criterion 8: a component pruned on this canvas (a stub cut mid-fence)
                # never overwrites the copy an earlier canvas emitted
                cid = _id_of(component)
                if cid in now.get(surface_id, ()):
                    self._as_written[surface_id][cid] = component
        for surface_id, tree in (deferred or {}).items():
            self._deferred.setdefault(surface_id, {}).update(tree)
        for surface_id, pending in self._deferred.items():
            # a deferred component that made it out on this canvas is an ordinary accepted one now
            for cid in [c for c in pending if c in self._emitted.get(surface_id, {})]:
                self._as_written.setdefault(surface_id, {})[cid] = pending.pop(cid)

    def _budgeted(self, invalid: Sequence[str], dropped: Sequence[str]) -> tuple[str, ...]:
        """Ids to re-request, in document order, within this turn's remaining budget."""
        ordered: list[str] = []
        for cid in [*invalid, *dropped]:
            # T-0108b: a gate-reserved id is the gate's own; the model is never asked for one
            if cid and cid not in ordered and not cid.startswith(RESERVED_PREFIX):
                ordered.append(cid)
        ordered = ordered[: max(self._resamples_left, 0)]
        self._resamples_left -= len(ordered)
        return tuple(ordered)

    def _known_catalog_ids(self) -> frozenset[str]:
        """Every catalogId a validator of this gate knows (basic or v2)."""
        return self._basic.known_catalog_ids | self._cards_v2.known_catalog_ids

    def _forgive_idempotent_create(
        self, report: SurfaceReport, messages: Sequence[Any]
    ) -> SurfaceReport:
        """T-0010 A4: re-sending ``createSurface`` for a surface this turn already created is a
        no-op, not an error. So is the catalogId it names, when that id is a known catalog of
        another validator (T-0184 C9/C9a: the first catalog wins); a truncated or unknown id
        stands. Every other problem stands."""
        resent = {
            m.group("idx")
            for p in report.problems
            if self._is_idempotent_create(p) and (m := _DUPLICATE_CREATE.match(p))
        }
        known = self._known_catalog_ids()
        kept = tuple(
            p
            for p in report.problems
            if not self._is_idempotent_create(p)
            and not (
                (m := _UNKNOWN_CATALOG.match(p))
                and m.group("idx") in resent
                and _create_catalog_id(messages, int(m.group("idx"))) in known
            )
        )
        if len(kept) == len(report.problems):
            return report
        return replace(report, problems=kept, ok=not report.schema_errors and not kept)

    def _is_idempotent_create(self, problem: str) -> bool:
        match = _DUPLICATE_CREATE.match(problem)
        return match is not None and match.group("sid") in self._emitted


# --- helpers ----------------------------------------------------------------------------------


def _create_catalog_id(messages: Sequence[Any], index: int) -> Any:
    """The catalogId of ``messages[index]``'s ``createSurface`` (``None`` when there is none)."""
    if not 0 <= index < len(messages) or not isinstance(messages[index], Mapping):
        return None
    body = messages[index].get("createSurface")
    return body.get("catalogId") if isinstance(body, Mapping) else None


def _fix(text: str, **options: Any) -> tuple[Any, list[Any]]:
    """``json_repair.loads`` with ``logging=True``, typed: (parsed value, repair log)."""
    # with logging=True the library returns (value, log); its own annotation is the union of
    # every value it can return, so the pair is unpacked through Any.
    result: Any = json_repair.loads(text, logging=True, **options)
    parsed, log = result
    return parsed, list(log)


def _fix_unguided(text: str, *, repaired: bool) -> tuple[Any, list[Any]]:
    """The unguided fixer on (structurally repaired) text: ``(parsed value, repair log)``.

    Text the structural pass made valid is parsed strictly, with an empty log: json_repair would
    log its "more json elements" note on a valid concatenation, which is not a repair (T-0108
    criterion 1). json_repair's only ``ValueError`` here is its nesting limit (criterion 9).
    """
    if repaired:
        try:
            return parse_concatenated_json(text), []
        except json.JSONDecodeError:
            pass
    try:
        return _fix(text, stream_stable=True)
    except ValueError as err:
        raise _TooDeep(f"the canvas nests deeper than the fixer can follow ({err})") from err


def _as_list(fixed: Any) -> list[Any]:
    if isinstance(fixed, list):
        return fixed
    if isinstance(fixed, dict):
        return [fixed]
    return []  # json_repair returns "" when there is nothing to recover


def _is_index(label: str) -> bool:
    """``invalid_component_ids`` labels an id-less component ``#<index>``; it cannot be resampled."""
    return label.startswith("#")


def _id_of(component: Any) -> str:
    cid = component.get("id") if isinstance(component, Mapping) else None
    return cid if isinstance(cid, str) else ""


def _by_surface(messages: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """Per surface, the components of ``messages`` by id, as written (the last definition wins)."""
    out: dict[str, dict[str, Any]] = {}
    for message in messages:
        sid, comps = _components_of(message)
        if sid is None:
            continue
        here = out.setdefault(sid, {})
        for component in comps or ():
            if cid := _id_of(component):
                here[cid] = component
    return out


def _kind_of(component: Any) -> str:
    kind = component.get("component") if isinstance(component, Mapping) else None
    return kind if isinstance(kind, str) and kind else "unknown kind"


def _tab_child(tab: Any) -> str | None:
    child = tab.get("child") if isinstance(tab, Mapping) else None
    return child if isinstance(child, str) else None


def _message_surface(message: Any) -> tuple[str | None, str | None]:
    """``(kind, surfaceId)`` of an A2UI server→client message, ``(None, None)`` if it is neither."""
    if not isinstance(message, Mapping):
        return None, None
    for kind in MESSAGE_KINDS:
        body = message.get(kind)
        if isinstance(body, Mapping):
            sid = body.get("surfaceId")
            return kind, (sid if isinstance(sid, str) else None)
    return None, None


def _rewritten(message: Any, surface_id: str) -> Any:
    """``message`` addressed to ``surface_id`` (T-0010 A7); unchanged when it already is."""
    kind, sid = _message_surface(message)
    if kind is None or sid is None or sid == surface_id:
        return message
    body = dict(message[kind])
    body["surfaceId"] = surface_id
    return {**message, kind: body}


def _components_of(message: Any) -> tuple[str | None, list[Any] | None]:
    """``(surfaceId, components)`` of an ``updateComponents`` message, ``(None, None)`` otherwise."""
    if not isinstance(message, Mapping):
        return None, None
    body = message.get("updateComponents")
    if not isinstance(body, Mapping):
        surface = message.get("createSurface")
        sid = surface.get("surfaceId") if isinstance(surface, Mapping) else None
        return (sid if isinstance(sid, str) else None), None
    sid = body.get("surfaceId")
    comps = body.get("components")
    return (sid if isinstance(sid, str) else None), (comps if isinstance(comps, list) else None)


def _missing_ids(before: Sequence[Any], after: Sequence[Any]) -> list[str]:
    kept = {_id_of(c) for c in after}
    return [cid for c in before if (cid := _id_of(c)) and cid not in kept]


def _salvage_one(component: Any, schema: Mapping[str, Any]) -> tuple[list[Any], int]:
    """Salvage a single component (as a one-element array); the input back on any fixer failure.

    ``Unresolvable`` is raised by a derived schema whose ``$ref``s json_repair cannot follow.
    ``ValueError`` is what json_repair raises for a violation of the *top-level* schema (pinned
    in ``test_json_repair_contract.py``); the top level here is always a one-element array
    matching ``{"type": "array"}``, so that arm is unreachable by construction and is kept only
    as a guard — a fixer failure must never become an exception out of the gate (T-0010 A11).
    """
    try:
        salvaged, log = _fix(json.dumps([component]), schema=schema, schema_repair_mode="salvage")
    except (ValueError, Unresolvable):
        return [component], 0
    if not isinstance(salvaged, list):
        return [component], 0
    return salvaged, len(log)


def _grounded(
    value: Any, raw: str, original: Any = _ANY, *, exempt: Callable[[str, str], bool] | None = None
) -> Any:
    """``value`` with everything the model did not write removed.

    The fixer is allowed to *drop* and to *close* things; it is not allowed to add text. Salvage
    mode does add: it fills in schema defaults, and it rewrites a value whose shape it does not
    understand. So a string leaf survives only if the input contains it, and — when ``original``
    is given, i.e. for a salvaged component — every other leaf survives only if it is exactly
    what the unguided parse held at the same path (T-0010 A11(b)). Without that second rule a
    destroyed data binding (``{"path": "/answers/q1"}`` → ``[]``) and an invented boolean
    (``filterable: false``) reach the renderer: neither is a string, so neither was checked.
    """
    if isinstance(value, str):
        if _in_raw(value, raw) or (exempt is not None and exempt(value, raw)):
            return value
        return _UNGROUNDED
    if isinstance(value, Mapping):
        if original is not _ANY and not isinstance(original, Mapping):
            return _UNGROUNDED  # salvage put an object where the model had something else
        out: dict[str, Any] = {}
        for key, item in value.items():
            kept = _grounded(item, raw, _at(original, key), exempt=exempt)
            if kept is not _UNGROUNDED:
                out[key] = kept
        return out
    if isinstance(value, list):
        if original is not _ANY and not isinstance(original, list):
            return _UNGROUNDED
        kept_items = [
            _grounded(item, raw, _at(original, i), exempt=exempt) for i, item in enumerate(value)
        ]
        return [item for item in kept_items if item is not _UNGROUNDED]
    if original is _ANY:
        return value
    # numbers, booleans, null: the model wrote it or it goes (``True == 1`` is not "the same")
    return value if type(value) is type(original) and value == original else _UNGROUNDED


def _at(original: Any, key: Any) -> Any:
    """``original`` one step down; ``_ANY`` propagates, a path it does not have is ``_ABSENT``."""
    if original is _ANY:
        return _ANY
    if isinstance(original, Mapping):
        return original.get(key, _ABSENT)
    if isinstance(original, list) and isinstance(key, int):
        return original[key] if 0 <= key < len(original) else _ABSENT
    return _ABSENT


def _repair_groups(
    before: Sequence[Any], after: Sequence[Any]
) -> list[tuple[Any, list[Any]]] | None:
    """``after`` split per component of ``before``: ``(original, what the stage emitted for it)``.

    ``repair_components`` keeps the order and puts lifted children right after their parent, so
    each original is followed by either itself (identical object) or its repaired version (same
    id) plus its lifted children. ``None`` when ``after`` is not in that shape.
    """
    groups: list[tuple[Any, list[Any]]] = []
    j = 0
    for item in after:
        if j < len(before) and item is before[j]:
            groups.append((item, [item]))
            j += 1
        elif j < len(before) and _id_of(item) and _id_of(item) == _id_of(before[j]):
            groups.append((before[j], [item]))
            j += 1
        elif groups and groups[-1][1][0] is not groups[-1][0]:
            groups[-1][1].append(item)  # a child lifted by the repair before it
        else:
            return None
    return groups if j == len(before) else None


def _repair_grounded(component: Any, original: Any, raw: str) -> bool:
    """T-0108b C7 (H-T108b (d)): whether grounding against ``raw`` leaves ``component`` unchanged.

    Exempt are exactly the reserved strings of the repair of ``original``: the event name, the id
    ``__gate_<id>_label`` for that Button's ``<id>``, and — on that one gate-made Text only — its
    ``"component": "Text"`` (2026-10-17 clarification); its ``text`` is still grounded. T-0127
    adds, for a repaired Card ``<id>`` only, the id ``__gate_<id>_body`` and — on that one
    gate-made Column only — its ``"component": "Column"``; its children are still grounded.
    """
    label_id = reserved_label_id(_id_of(original))
    is_card = isinstance(original, Mapping) and original.get("component") == "Card"
    body_id = reserved_body_id(_id_of(original)) if is_card else None
    reserved = (
        (RESERVED_EVENT, label_id) if body_id is None else (RESERVED_EVENT, label_id, body_id)
    )

    def exempt(text: str, _raw: str) -> bool:
        return text in reserved

    if isinstance(component, Mapping) and (
        (_id_of(component) == label_id and component.get("component") == "Text")
        or (
            (body_id is not None and _id_of(component) == body_id)
            and component.get("component") == "Column"
        )
    ):
        rest = {k: v for k, v in component.items() if k != "component"}
        return bool(_grounded(rest, raw, exempt=exempt) == rest)
    return bool(_grounded(component, raw, exempt=exempt) == component)


def _in_raw(text: str, raw: str) -> bool:
    """A string is the model's if it appears in the output, literally or JSON-escaped."""
    return text in raw or json.dumps(text)[1:-1] in raw


# --- verbatim copy of 7646b8b6:local_astra/ui/a2ui_schema.py parse_concatenated_json below this line ---
def parse_concatenated_json(text: str) -> list[Any]:
    """Parse JSONL *or* pretty-printed concatenated JSON values (what models actually emit)."""
    dec = json.JSONDecoder()
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
