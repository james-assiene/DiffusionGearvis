"""T-0184 group C, the gate: C1-C3 and the hand-off criterion on non-finite numbers (docs/tasks/T-0184.md).

**C1. Per-surface catalog; the basic one is unchanged.** ``UiGate()`` validates each surface against the catalog its
``createSurface`` names, remembers it for the turn, and uses that catalog's fixer schema. The differential half runs the
gate beside ``tests/unit/_frozen_ui_gate_t0184.py`` (BASE 3cb090f1's ``ui_gate.py``, verbatim; a row pins the copy against
``git show``) over the T-0164 census (``genui_census_20261027.jsonl``, both basic id spellings) and the stored Q12 trial
sample (``q12_trials_sample.jsonl``, MANIFEST row), and requires every ``GateResult`` to be field-equal. Those rows are
green at BASE by construction (the live gate *is* the frozen one there) and are not pending: they are the guard that C
changes nothing on a basic surface.

**C2. v2 kinds only under the v2 id.** A v2 kind on a basic surface is invalid and pruned (green at BASE, unmarked: the
guard against merging the catalogs, implementation note 1); basic kinds on a v2 surface repair exactly as on a basic one
(the oracle is the frozen gate on the same canvas under the basic id, per rule BC, LI, TF, DT, CPO, CPV, RB, CC, and the
55 stored T-0108b component lists); no component rule fires on a v2 kind.

**C3. Progressive v2 canvases.** Every exemplar of ``cards_v2_exemplars.jsonl``, written as compact JSONL (one message per
line, ``separators=(",", ":")``, ``ensure_ascii=False``: the shape the prompt's example teaches), is cut at every 256-token
boundary and at ``RANDOM_CUTS`` random character offsets (``random.Random(0)`` per exemplar; 10, not the criterion's 50,
by the coordinator's budget ruling of 2026-10-28, see the constant), each cut through a fresh ``UiGate()`` with
``final=False``. The token boundaries are character offsets computed once with the pinned tokenizer (``tokenizer.json`` of
``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` rev 2ea8372, ``tokenizers`` 0.23.2 through ``uv run --with``,
``add_special_tokens=False``) and pinned below; an opt-in row re-derives them when ``tokenizers`` is importable.

**Hand-off (group A review), the C criterion added by the orchestrator:** the gate refuses non-finite numbers. A surface
carrying ``NaN``/``Infinity`` in any number role is invalid there, and the rest of the surface is handled exactly as for
any other invalid component. The oracle for "exactly as" is the same canvas with the non-finite value replaced by a finite
value the catalog already rejects (``-1`` for ``Bars.value``, ``140`` for ``percent``, a string for ``Slider.value``…):
the two ``GateResult``\\s must agree on everything but the wording of the schema errors. Rows on both catalogs (basic
``Slider.value``/``max``, ``Text.weight``; v2 ``Bars.value``, ``StatTiles.percent``, ``Comparison.highlight``), on the
unguided path, and a too-wide guard (the words "NaN"/"Infinity" in a text role and large finite numbers still render).

**The interface these rows assume** (the task names ``UiGate()`` and nothing else): ``UiGate()`` with no argument is the
dual-catalog gate; ``GateResult`` keeps BASE's field names; ``local_astra.ui.a2ui_schema.CARDS_V2_CATALOG_ID`` and
``load_cards_v2_catalog`` (group A, merged) give the v2 id and catalog for this module's *own* validation oracle.

Pending rows fail at BASE because ``UiGate()`` there knows only the basic catalog: a v2 surface yields no surface at all
("unknown catalogId 'gearvis:cards/v2'"), and a basic ``Slider.value: NaN`` validates.

Regime: CPU only, no service, no socket; deterministic (seed 0 where a seed applies), n = 1 per row.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import re
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.ui_gate import UiGate
from local_astra.ui import A2UIValidator
from local_astra.ui.a2ui_schema import CARDS_V2_CATALOG_ID, load_cards_v2_catalog
from tests import _git_range as gr
from tests.unit import _frozen_ui_gate_t0184 as frozen

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "a2ui"
CENSUS = FIXTURES / "genui_census_20261027.jsonl"
Q12 = FIXTURES / "q12_trials_sample.jsonl"
EXEMPLARS = FIXTURES / "cards_v2_exemplars.jsonl"
COMPONENT_ROWS = FIXTURES / "t0108b_components.jsonl"
FROZEN_FILE = Path(__file__).with_name("_frozen_ui_gate_t0184.py")

#: T-0184 group C's BASE: main at 3cb090f1, the branch point of ``T-0184-c-gate-prompt`` (group A merged). A literal.
BASE = "3cb090f14e49eb584e9b582e75625a6ce4ebc4fe"
#: sha256 of ``git show 3cb090f1:local_astra/conductor/ui_gate.py`` (blob 8ca03b3e), measured 2026-10-28.
BASE_GATE_SHA256 = "7f453218fb406dac3dbfc1a5983755aa94753eb0992b6c321b5c1f5c5d613904"
FROZEN_MARKER = (
    "# --- verbatim copy of 3cb090f1:local_astra/conductor/ui_gate.py below this line ---\n"
)

V2_ID = "gearvis:cards/v2"
BASIC_V091 = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
BASIC_V09 = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"
BASIC_IDS = (BASIC_V091, BASIC_V09)
V2_KINDS = ("Stat", "StatTiles", "Comparison", "Bars", "Steps", "Source", "Actions")

#: C3's 256-token boundaries, as character offsets into each exemplar's compact JSONL text (the offset of the first
#: character of token 256k). Computed by the test-author on 2026-10-28 with the pinned tokenizer (module docstring):
#: only ``stat-icons`` (378 tokens, 1 166 characters) reaches a second canvas; every other exemplar is under 256 tokens.
#: ``test_c3_token_boundaries_are_the_tokenizers`` re-derives them when ``tokenizers`` is importable.
TOKEN_BOUNDARIES: dict[str, tuple[int, ...]] = {"stat-icons": (777,)}
#: C3 names 50 random offsets per exemplar; the coordinator (2026-10-28) cut it to 10 (``random.Random(0).sample``,
#: per exemplar) to meet the task's "fast serial suite grows by <= 3 s". Re-run against both C reviews'
#: mutants: every kill of the 50-cut row (M8/M9, M10a/b, M16 grounding dropped on salvage, M17) stays a kill at 10.
RANDOM_CUTS = 10

GATE_FIELDS = tuple(f.name for f in dataclasses.fields(frozen.GateResult))


# --- helpers ----------------------------------------------------------------------------------------------------


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _compact(messages: list[Any]) -> str:
    return "\n".join(
        json.dumps(m, ensure_ascii=False, separators=(",", ":"), allow_nan=True) for m in messages
    )


def _canvas(catalog_id: str, components: list[Any], sid: str = "s") -> str:
    return _compact(
        [
            {"version": "v0.9.1", "createSurface": {"surfaceId": sid, "catalogId": catalog_id}},
            {"version": "v0.9.1", "updateComponents": {"surfaceId": sid, "components": components}},
        ]
    )


def _snapshot(result: Any) -> dict[str, Any]:
    """Every BASE ``GateResult`` field of ``result``, as plain data (the surface dumped, the report as a dict)."""
    out: dict[str, Any] = {}
    for name in GATE_FIELDS:
        assert hasattr(result, name), f"C1: GateResult lost BASE's field {name!r}"
        value = getattr(result, name)
        if name == "surface":
            value = None if value is None else value.model_dump(mode="python")
        elif name == "report":
            value = dataclasses.asdict(value)
        out[name] = value
    return out


def _first_difference(a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    for name in GATE_FIELDS:
        if a.get(name) != b.get(name):
            return f"field {name!r}:\n  tip:  {a.get(name)!r}\n  base: {b.get(name)!r}"
    return "(no field differs)"


def _emitted(result: Any) -> dict[str, dict[str, Any]]:
    """Components the result's surface emits, by id (the last definition wins)."""
    out: dict[str, dict[str, Any]] = {}
    if result.surface is None:
        return out
    for message in result.surface.messages:
        body = message.get("updateComponents") if isinstance(message, Mapping) else None
        if isinstance(body, Mapping):
            for c in body.get("components", []):
                out[str(c.get("id"))] = c
    return out


def _catalog_of(result: Any) -> str | None:
    for message in result.surface.messages:
        body = message.get("createSurface") if isinstance(message, Mapping) else None
        if isinstance(body, Mapping):
            return str(body.get("catalogId"))
    return None


def _leaves(node: Any, key: str | None = None) -> Iterator[tuple[str | None, Any]]:
    if isinstance(node, Mapping):
        for k, v in node.items():
            yield from _leaves(v, str(k))
    elif isinstance(node, list | tuple):
        for v in node:
            yield from _leaves(v, key)
    else:
        yield key, node


def _non_finite(messages: Any) -> list[Any]:
    return [v for _, v in _leaves(list(messages)) if isinstance(v, float) and not math.isfinite(v)]


def _v2_validator() -> A2UIValidator:
    return A2UIValidator(catalog=load_cards_v2_catalog())


def _exemplars() -> list[dict[str, Any]]:
    return _jsonl(EXEMPLARS)


def _exemplar_component(kind: str) -> dict[str, Any]:
    """The first ``kind`` component written in the exemplar fixture (hand-written, catalog-valid: group A's A3)."""
    for row in _exemplars():
        for m in row["messages"]:
            for c in m.get("updateComponents", {}).get("components", []):
                if c.get("component") == kind:
                    return dict(c)
    raise AssertionError(f"the exemplar fixture holds no {kind}")


def _require_surface(result: Any, what: str) -> None:
    assert result.surface is not None, (
        f"{what}: the gate emitted no surface. Problems: {list(result.report.problems)[:4]}"
    )


# --- C1: the frozen oracle is BASE's gate, verbatim -------------------------------------------------------------


def test_frozen_gate_is_base_verbatim() -> None:
    """The oracle is what it claims: every byte below the marker equals BASE's ``ui_gate.py`` (sha256 literal), and,
    when 3cb090f1 is in the object store, ``git show`` of it."""
    text = FROZEN_FILE.read_text(encoding="utf-8")
    assert text.count(FROZEN_MARKER) == 1, "the frozen copy lost its marker line"
    body = text.split(FROZEN_MARKER, 1)[1]
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == BASE_GATE_SHA256, (
        "tests/unit/_frozen_ui_gate_t0184.py was edited below its marker: it must stay BASE's ui_gate.py verbatim"
    )
    gr.require_commits(BASE)
    shown = gr._git("show", f"{BASE}:local_astra/conductor/ui_gate.py")
    assert shown.returncode == 0, shown.stderr
    assert shown.stdout == body


# --- C1: the differential (green at BASE, unmarked) --------------------------------------------------------------


def _census_rows() -> list[dict[str, Any]]:
    return [row for row in _jsonl(CENSUS) if "header" not in row]


def _run_both(
    calls: list[tuple[str, str, int, bool]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """``(tip, base)`` snapshots of one fresh live gate and one fresh frozen gate over the same calls."""
    tip, base = UiGate(), frozen.UiGate()
    out = []
    for raw, turn, idx, final in calls:
        out.append(
            (
                _snapshot(tip.gate(raw, turn_id=turn, canvas_idx=idx, final=final)),
                _snapshot(base.gate(raw, turn_id=turn, canvas_idx=idx, final=final)),
            )
        )
    return out


@pytest.mark.parametrize("catalog_id", BASIC_IDS, ids=["v0_9_1", "v0_9"])
def test_c1_census_gives_base_gate_results(catalog_id: str) -> None:
    """C1 differential on the T-0164 census (13 surfaces): each canvas, as the JSONL of its messages, through a fresh
    live gate and a fresh frozen gate per surface, under both basic id spellings. Every ``GateResult`` is field-equal.

    Budget (T-0184 Regime, "the fast serial suite grows by <= 3 s"): the ``v0_9_1`` spelling (the gate's own) replays
    every canvas with ``final=False`` (the session's progressive calls); the ``v0_9`` spelling replays each surface's
    last canvas with ``final=True`` (13 calls per gate), which is what the spelling can change: the catalog the turn is
    gated against. Neither this row nor the Q12 row (``final=False`` only, since 97c08598) replays ``final=True`` after
    progressive canvases of the same turn against BASE; that sequence is checked only on v2 surfaces, without the frozen
    oracle (the C1 remembered-for-the-turn row and C9a's cut-then-whole row)."""
    rows = _census_rows()
    assert len(rows) == 13, "premise: the census holds 13 surfaces"
    surfaces = 0
    for row in rows:
        texts = [
            _compact(canvas).replace(BASIC_V091, catalog_id).replace(BASIC_V09, catalog_id)
            for canvas in row["canvases"]
        ]
        assert all(catalog_id in t for t in texts if "createSurface" in t), (
            "premise: the spelling is in the canvas"
        )
        turn = row["surface_id"].removeprefix("turn-")
        if catalog_id == BASIC_V091:
            calls = [(t, turn, i, False) for i, t in enumerate(texts)]
        else:
            calls = [(texts[-1], turn, len(texts) - 1, True)]
        for k, (tip, base) in enumerate(_run_both(calls)):
            assert tip == base, (
                f"{row['key']} call {k} ({catalog_id}): {_first_difference(tip, base)}"
            )
            surfaces += base["surface"] is not None
    assert surfaces > 0, "premise: the census yields surfaces at BASE"


def test_c1_q12_sample_gives_base_gate_results() -> None:
    """C1 differential on the stored Q12 sample (17 trials, 30 ui commits; MANIFEST row): every ui commit in order
    through a fresh live gate and a fresh frozen gate per trial (``final=False``, turn id = request id, the commit's
    canvas_idx: the T-0098c ``ui_reading``, which is how the session calls it). Field-equal throughout. (The trailing
    ``final=True`` repeat was dropped for the budget, 2026-10-28; the census row's ``v0_9`` half gates ``final=True``.)"""
    rows = _jsonl(Q12)
    header, trials = rows[0]["header"], rows[1:]
    assert len(trials) == header["trials"] == 17 and header["ui_commits"] == 30
    shown = complete = 0
    for trial in trials:
        commits = trial["ui_commits"]
        rid = trial["request_id"]
        calls = [(c["text"], rid, int(c["canvas_idx"] or 0), False) for c in commits]
        for k, (tip, base) in enumerate(_run_both(calls)):
            assert tip == base, f"{rid} call {k}: {_first_difference(tip, base)}"
            shown += base["surface"] is not None
            complete += bool(base["complete"])
    assert (shown, complete) == (20, 14), (
        f"premise (MANIFEST): 20 surfaces, 14 complete over the 30 commits at BASE; got {shown}, {complete}"
    )


# --- C1: the catalog named by createSurface, remembered for the turn, with its fixer schema -------------------------


def _v2_tree(*extra: dict[str, Any], children: tuple[str, ...] = ("s",)) -> list[dict[str, Any]]:
    return [
        {"id": "root", "component": "Column", "children": list(children)},
        {
            "id": "s",
            "component": "Stat",
            "label": "PARIS · NOW",
            "value": "17°",
            "detail": "Partly cloudy",
        },
        *extra,
    ]


def test_c1_v2_surface_validates_against_the_v2_catalog() -> None:
    """``UiGate()`` (no argument) emits a v2 surface: the v2 id kept, the Stat emitted, complete, nothing repaired."""
    result = UiGate().gate(_canvas(V2_ID, _v2_tree()), turn_id="a", canvas_idx=0, final=True)
    _require_surface(result, "C1, a v2 surface through UiGate()")
    assert result.report.ok
    assert _catalog_of(result) == V2_ID
    assert set(_emitted(result)) == {"root", "s"}
    assert result.complete and result.repaired == 0 and result.lost == () and result.resample == ()


def test_c1_catalog_is_remembered_for_the_turn_and_reset_by_the_next() -> None:
    """Canvas 1 of a v2 turn carries no ``createSurface`` (the surface is already created): its Bars still validates
    against v2. A new turn whose ``createSurface`` names the basic catalog does not inherit v2: its Stat is pruned."""
    gate = UiGate()
    first = gate.gate(_canvas(V2_ID, _v2_tree()), turn_id="a", canvas_idx=0, final=False)
    _require_surface(first, "C1, canvas 0 of a v2 turn")
    bars = {"id": "b", "component": "Bars", "bars": [{"label": "K2", "value": 8611}], "unit": "m"}
    update = _compact(
        [
            {
                "version": "v0.9.1",
                "updateComponents": {
                    "surfaceId": "s",
                    "components": _v2_tree(bars, children=("s", "b")),
                },
            }
        ]
    )
    second = gate.gate(update, turn_id="a", canvas_idx=1, final=True)
    _require_surface(second, "C1, canvas 1 of the same v2 turn (no createSurface)")
    assert _emitted(second)["b"] == bars, _emitted(second)
    assert second.complete and second.lost == ()

    later = gate.gate(_canvas(BASIC_V091, _v2_tree()), turn_id="b", canvas_idx=0, final=True)
    _require_surface(later, "C1, the next turn under the basic id")
    assert "s" not in _emitted(later) and later.lost == ("s",), (
        "the basic turn must not inherit the v2 catalog"
    )


def test_c1_v2_surface_salvages_with_the_v2_fixer_schema() -> None:
    """The unguided path on a v2 surface salvages a v2 kind with the v2 fixer schema.

    The canvas lacks its last ``}`` (so the strict parse fails) and its Bars carries a bar with no ``value``. The v2
    salvage schema drops that one bar and keeps the Bars (json_repair's salvage drops the array item that fails its
    schema; ``tests/unit/test_json_repair_contract.py``); the basic one has no Bars at all, so salvage would drop the
    whole component and the card would lose its chart. Expected by hand: the Bars is emitted with the K2 bar and the
    model's unit, every string grounded, the card complete."""
    bars = {
        "id": "b",
        "component": "Bars",
        "bars": [{"label": "K2", "value": 8611}, {"label": "Everest"}],
        "unit": "m",
    }
    text = _canvas(V2_ID, [{"id": "root", "component": "Column", "children": ["t", "b"]},
                           {"id": "t", "component": "Text", "text": "Heights"}, bars])  # fmt: skip
    result = UiGate().gate(text[:-1], turn_id="a", canvas_idx=0, final=False)
    _require_surface(result, "C1, an unguided v2 canvas")
    emitted = _emitted(result)
    assert "b" in emitted, (
        f"the Bars was dropped (the basic fixer schema has no Bars); lost={result.lost}"
    )
    assert emitted["b"] == {
        "id": "b",
        "component": "Bars",
        "bars": [{"label": "K2", "value": 8611}],
        "unit": "m",
    }
    assert result.complete and result.lost == () and result.repaired > 0


# --- C2: v2 kinds only under the v2 id ------------------------------------------------------------------------------


@pytest.mark.parametrize("catalog_id", BASIC_IDS, ids=["v0_9_1", "v0_9"])
@pytest.mark.parametrize("kind", V2_KINDS)
def test_c2_v2_kind_on_a_basic_surface_is_pruned(kind: str, catalog_id: str) -> None:
    """C2, green at BASE and unmarked: the guard against merged catalogs (implementation note 1). A catalog-valid v2
    component (the exemplar's) on a basic surface is invalid there: dropped, its reference pruned, reported lost and
    resampled; the rest of the card is emitted. Expected by hand."""
    x = {**_exemplar_component(kind), "id": "x"}
    tree = [
        {"id": "root", "component": "Column", "children": ["h", "x"]},
        {"id": "h", "component": "Text", "text": "Kept"},
        x,
    ]
    assert _v2_validator().component_errors(x) == [], f"premise: the {kind} is valid under v2"
    result = UiGate().gate(_canvas(catalog_id, tree), turn_id="a", canvas_idx=0, final=True)
    _require_surface(result, f"C2, {kind} on a basic surface")
    emitted = _emitted(result)
    assert set(emitted) == {"root", "h"}, sorted(emitted)
    assert emitted["root"]["children"] == ["h"]
    assert result.lost == ("x",) and result.resample == ("x",) and not result.complete
    assert result.component_repairs == 0


#: One canvas per component-repair rule (component_repair.py's docstring), each on a basic surface. The premise row
#: proves each fires at BASE (component_repairs >= 1) and leaves the card complete.
RULE_CANVASES: dict[str, list[dict[str, Any]]] = {
    "BC": [
        {"id": "root", "component": "Column", "children": ["b"]},
        {
            "id": "b",
            "component": "Button",
            "children": ["lbl"],
            "action": {"event": {"name": "go"}},
        },
        {"id": "lbl", "component": "Text", "text": "Go"},
    ],
    "LI": [
        {
            "id": "root",
            "component": "Column",
            "children": [{"id": "t", "component": "Text", "text": "Hi"}],
        }
    ],
    "TF": [
        {"id": "root", "component": "Column", "children": ["f"]},
        {"id": "f", "component": "TextField", "placeholder": "Name"},
    ],
    "DT": [
        {"id": "root", "component": "Column", "children": ["d"]},
        {"id": "d", "component": "DateTimeInput", "label": "When"},
    ],
    "CPO": [
        {"id": "root", "component": "Column", "children": ["c"]},
        {
            "id": "c",
            "component": "ChoicePicker",
            "label": "Size",
            "options": ["S", "M"],
            "value": [],
        },
    ],
    "CPV": [
        {"id": "root", "component": "Column", "children": ["c"]},
        {
            "id": "c",
            "component": "ChoicePicker",
            "label": "Size",
            "options": [{"label": "S", "value": "S"}, {"label": "M", "value": "M"}],
            "value": "S",
        },
    ],
    "RB": [
        {"id": "root", "component": "Column", "children": ["b"]},
        {"id": "b", "component": "Button", "text": "Search"},
    ],
    "CC": [
        {"id": "root", "component": "Column", "children": ["k"]},
        {"id": "k", "component": "Card", "children": ["t1", "t2"]},
        {"id": "t1", "component": "Text", "text": "One"},
        {"id": "t2", "component": "Text", "text": "Two"},
    ],
}


def _plain(snapshot: dict[str, Any]) -> dict[str, Any]:
    """``snapshot`` through one JSON round trip. Applied to *both* sides of a C2 comparison: it maps tuples to lists
    and nothing else (every snapshot key is a string and every leaf a str, int, float, bool or None), so two plain
    snapshots are equal exactly when the originals are equal up to tuple/list."""
    out: dict[str, Any] = json.loads(json.dumps(snapshot))
    return out


def _as_basic(snapshot: dict[str, Any]) -> dict[str, Any]:
    """A v2-surface snapshot, plain, with the v2 id written back as the basic one (the only byte the catalog may
    change). Compare it with ``_plain`` of the BASE snapshot."""
    return _plain(
        json.loads(json.dumps(snapshot).replace(json.dumps(V2_ID), json.dumps(BASIC_V091)))
    )


@pytest.mark.parametrize("rule", sorted(RULE_CANVASES))
def test_c2_rule_canvas_premise_fires_at_base(rule: str) -> None:
    """Premise (unmarked): on a basic surface the frozen BASE gate applies ``rule`` and the card is complete."""
    result = frozen.UiGate().gate(
        _canvas(BASIC_V091, RULE_CANVASES[rule]), turn_id="r", canvas_idx=0, final=True
    )
    assert result.surface is not None and result.complete, (rule, result.report.problems)
    assert result.component_repairs >= 1, (
        f"{rule} did not fire at BASE: the row would prove nothing"
    )


@pytest.mark.parametrize("rule", sorted(RULE_CANVASES))
def test_c2_basic_kinds_repair_on_a_v2_surface_as_on_basic(rule: str) -> None:
    """C2: the same canvas under the v2 id gives the BASE gate's basic result, field for field (repairs, reserved
    ids, completeness, losses), the catalog id aside."""
    raw = _canvas(BASIC_V091, RULE_CANVASES[rule])
    base = _snapshot(frozen.UiGate().gate(raw, turn_id="r", canvas_idx=0, final=True))
    tip_result = UiGate().gate(
        raw.replace(BASIC_V091, V2_ID), turn_id="r", canvas_idx=0, final=True
    )
    _require_surface(tip_result, f"C2, rule {rule} on a v2 surface")
    assert _catalog_of(tip_result) == V2_ID
    tip, base = _as_basic(_snapshot(tip_result)), _plain(base)
    assert tip == base, f"rule {rule}: {_first_difference(tip, base)}"


def test_c2_stored_component_lists_repair_on_a_v2_surface_as_on_basic() -> None:
    """C2 over the stored T-0108b component lists (real model output): the first list of each of its 19 classes (every
    rule class and one all-valid list), each as a strict canvas under the v2 id, equals the BASE gate under the basic id,
    the catalog id aside. One per class, not all 55: the budget (T-0184 Regime), and the other lists of a class repeat
    its shape (every mutant of both C reviews that this row killed at 55 lists it still kills at 19)."""
    rows = _jsonl(COMPONENT_ROWS)
    assert len(rows) == 55
    firsts: dict[str, dict[str, Any]] = {}
    for row in rows:
        firsts.setdefault(row["class"], row)
    assert len(firsts) == 19, sorted(firsts)
    for row in firsts.values():
        raw = _canvas(BASIC_V091, row["components"])
        base = _snapshot(frozen.UiGate().gate(raw, turn_id="r", canvas_idx=0, final=False))
        tip_result = UiGate().gate(
            raw.replace(BASIC_V091, V2_ID), turn_id="r", canvas_idx=0, final=False
        )
        if base["surface"] is not None:
            _require_surface(tip_result, f"C2, {row['row_id']} on a v2 surface")
        tip, base = _as_basic(_snapshot(tip_result)), _plain(base)
        assert tip == base, f"{row['row_id']}: {_first_difference(tip, base)}"


#: A v2 kind written in the wrong shape a component rule repairs on a basic kind (the rule named in the key). None of
#: the rules may fire on a v2 kind: the component is invalid and pruned, never rewritten.
RULE_SHAPED_V2: dict[str, dict[str, Any]] = {
    "LI-Steps-inline-children": {
        "id": "x",
        "component": "Steps",
        "steps": [{"title": "Whisk"}],
        "children": [{"id": "y", "component": "Text", "text": "Fold"}],
    },
    "BC-Actions-children": {"id": "x", "component": "Actions", "labels": ["Go"], "children": ["t"]},
    "TF-Stat-placeholder": {"id": "x", "component": "Stat", "placeholder": "NOW", "value": "17°"},
    "RB-Actions-text": {"id": "x", "component": "Actions", "text": "Search"},
    "CC-Comparison-children": {
        "id": "x",
        "component": "Comparison",
        "columns": [{"title": "A"}, {"title": "B"}],
        "rows": [{"label": "Price", "cells": ["1", "2"]}],
        "children": ["t"],
    },
    "CPV-Bars-string": {"id": "x", "component": "Bars", "bars": "K2"},
    "CPO-StatTiles-strings": {"id": "x", "component": "StatTiles", "tiles": ["Photos", "Apps"]},
}


@pytest.mark.parametrize("case", sorted(RULE_SHAPED_V2))
def test_c2_no_component_rule_fires_on_a_v2_kind(case: str) -> None:
    """C2: on a v2 surface, a rule-shaped invalid v2 component is pruned as written (lost, resampled), the stage
    reports zero component repairs, nothing in the reserved ``__gate_`` namespace appears, and the rest is emitted."""
    x = RULE_SHAPED_V2[case]
    assert _v2_validator().component_errors(x), f"premise: {case} is invalid under v2"
    tree = [
        {"id": "root", "component": "Column", "children": ["t", "x"]},
        {"id": "t", "component": "Text", "text": "Kept"},
        x,
    ]
    result = UiGate().gate(_canvas(V2_ID, tree), turn_id="a", canvas_idx=0, final=True)
    _require_surface(result, f"C2, {case}")
    emitted = _emitted(result)
    assert set(emitted) == {"root", "t"}, sorted(emitted)
    assert result.component_repairs == 0, f"{case}: a component rule fired on a v2 kind"
    assert "__gate_" not in json.dumps(result.surface.model_dump(mode="python"))
    assert result.lost == ("x",) and "x" in result.resample and not result.complete


# --- C3: progressive v2 canvases ------------------------------------------------------------------------------------


def _exemplar_text(row: dict[str, Any]) -> str:
    return _compact(row["messages"])


def _cuts(key: str, text: str) -> list[int]:
    rng = random.Random(0)
    offsets = rng.sample(range(1, len(text)), RANDOM_CUTS)
    return sorted({*TOKEN_BOUNDARIES.get(key, ()), *offsets})


_RESERVED = re.compile(r"^__gate[._]")


def _ungrounded(messages: Any, raw: str) -> list[str]:
    """String leaves of ``messages`` that are not the model's (T-0108b): not in ``raw`` literally or JSON-escaped.
    Exempt: ``surfaceId`` (the gate rewrites it, T-0010 A7) and the reserved ``__gate`` namespace."""
    bad = []
    for key, value in _leaves(list(messages)):
        if not isinstance(value, str) or key == "surfaceId" or _RESERVED.match(value):
            continue
        if value in raw or json.dumps(value)[1:-1] in raw:
            continue
        bad.append(value)
    return bad


def test_c3_exemplar_fixture_is_group_as() -> None:
    """Premise (unmarked): the C3 corpus is the 19 hand-written v2 exemplars, every one v2 and catalog-valid."""
    rows = _exemplars()
    assert len(rows) == 19
    v = _v2_validator()
    for row in rows:
        assert row["messages"][0]["createSurface"]["catalogId"] == V2_ID, row["key"]
        assert v.validate_messages(row["messages"], final=True).ok, row["key"]


@pytest.mark.parametrize("key", [row["key"] for row in _exemplars()])
def test_c3_progressive_cuts_never_raise_validate_and_stay_grounded(key: str) -> None:
    """C3, per exemplar: every cut (the pinned 256-token boundaries plus ``RANDOM_CUTS`` random offsets, seed 0) through a fresh
    ``UiGate()`` with ``final=False`` never raises; every emitted surface validates under the v2 catalog (this
    module's own validator, ``final=False``) and every string in it is grounded in the cut. The full text gives
    ``complete`` with ``repaired == 0`` (``final=False``, and ``final=True``)."""
    row = next(r for r in _exemplars() if r["key"] == key)
    text = _exemplar_text(row)
    v = _v2_validator()
    shown = 0
    for k in _cuts(key, text):
        cut = text[:k]
        try:
            result = UiGate().gate(cut, turn_id="p", canvas_idx=0, final=False)
        except (
            Exception
        ) as err:  # the criterion is "never raises": report the cut, not the traceback
            raise AssertionError(f"C3 {key}: the gate raised at cut {k}: {err!r}") from err
        if result.surface is None:
            continue
        shown += 1
        report = v.validate_messages(list(result.surface.messages), final=False)
        assert report.ok, (
            f"C3 {key} cut {k}: an emitted surface does not validate: {report.problems[:3]}"
        )
        assert _catalog_of(result) == V2_ID
        assert not (bad := _ungrounded(result.surface.messages, cut)), (
            f"C3 {key} cut {k}: ungrounded {bad}"
        )
    for final in (False,):  # C3 names final=False; final=True dropped for the budget (2026-10-28)
        whole = UiGate().gate(text, turn_id="p", canvas_idx=0, final=final)
        _require_surface(whole, f"C3 {key}, the full text (final={final})")
        assert whole.complete, (
            f"C3 {key} final={final}: not complete; lost={whole.lost} {whole.report.problems[:3]}"
        )
        assert whole.repaired == 0, f"C3 {key} final={final}: repaired={whole.repaired}"
        assert v.validate_messages(list(whole.surface.messages), final=True).ok
    assert shown > 0, f"C3 {key}: no cut produced a surface"


def test_c3_token_boundaries_are_the_tokenizers() -> None:
    """Opt-in (needs ``tokenizers``: ``uv run --with tokenizers==0.23.2 pytest …``): re-derive ``TOKEN_BOUNDARIES`` from
    the pinned tokenizer. Skips naming the file when the snapshot is not in the local HF cache."""
    path = _tokenizer_path()
    tokenizers = pytest.importorskip(
        "tokenizers", reason="opt-in: run with `uv run --with tokenizers==0.23.2`"
    )
    tok = tokenizers.Tokenizer.from_file(str(path))
    derived: dict[str, tuple[int, ...]] = {}
    for row in _exemplars():
        enc = tok.encode(_exemplar_text(row), add_special_tokens=False)
        cut = tuple(enc.offsets[i][0] for i in range(256, len(enc.ids), 256))
        if cut:
            derived[row["key"]] = cut
    assert derived == TOKEN_BOUNDARIES


#: The pinned tokenizer (services/versions.lock: model rev 2ea8372; tests/cd/_compile_support.py's sha256).
MODEL_ID = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
MODEL_REVISION = "2ea837236295d617ac27f8c17d61228081932c40"
TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"


def _tokenizer_path() -> Path:
    import os

    if os.environ.get("HF_HUB_CACHE"):
        hub = Path(os.environ["HF_HUB_CACHE"])
    elif os.environ.get("HF_HOME"):
        hub = Path(os.environ["HF_HOME"]) / "hub"
    else:
        hub = Path.home() / ".cache" / "huggingface" / "hub"
    path = (
        hub
        / f"models--{MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / MODEL_REVISION
        / "tokenizer.json"
    )
    if not path.is_file():
        pytest.skip(f"the pinned tokenizer is not in the local HF cache: {path}")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == TOKENIZER_SHA256, (
        f"{path} is not the pinned file"
    )
    return path


# --- hand-off: non-finite numbers are refused ------------------------------------------------------------------------

NAN, INF = float("nan"), float("inf")

#: (catalog, the component with a non-finite number, the same component with a finite value the catalog rejects)
NON_FINITE: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {
    "v2-Bars.value-NaN": (
        V2_ID,
        {"id": "x", "component": "Bars", "bars": [{"label": "K2", "value": NAN}]},
        {"id": "x", "component": "Bars", "bars": [{"label": "K2", "value": -1}]},
    ),
    "v2-Bars.value-Infinity": (
        V2_ID,
        {"id": "x", "component": "Bars", "bars": [{"label": "K2", "value": INF}]},
        {"id": "x", "component": "Bars", "bars": [{"label": "K2", "value": -1}]},
    ),
    "v2-Bars.value--Infinity": (
        V2_ID,
        {"id": "x", "component": "Bars", "bars": [{"label": "K2", "value": -INF}]},
        {"id": "x", "component": "Bars", "bars": [{"label": "K2", "value": -1}]},
    ),
    "v2-StatTiles.percent-NaN": (
        V2_ID,
        {
            "id": "x",
            "component": "StatTiles",
            "tiles": [{"title": "Photos", "value": "212 GB", "percent": NAN}],
        },
        {
            "id": "x",
            "component": "StatTiles",
            "tiles": [{"title": "Photos", "value": "212 GB", "percent": 140}],
        },
    ),
    "v2-StatTiles.percent-Infinity": (
        V2_ID,
        {
            "id": "x",
            "component": "StatTiles",
            "tiles": [{"title": "Photos", "value": "212 GB", "percent": INF}],
        },
        {
            "id": "x",
            "component": "StatTiles",
            "tiles": [{"title": "Photos", "value": "212 GB", "percent": 140}],
        },
    ),
    "v2-Comparison.highlight-NaN": (
        V2_ID,
        {
            "id": "x",
            "component": "Comparison",
            "columns": [{"title": "A"}, {"title": "B"}],
            "rows": [{"label": "Price", "cells": ["1", "2"]}],
            "highlight": NAN,
        },
        {
            "id": "x",
            "component": "Comparison",
            "columns": [{"title": "A"}, {"title": "B"}],
            "rows": [{"label": "Price", "cells": ["1", "2"]}],
            "highlight": -1,
        },
    ),
    "basic-Slider.value-NaN": (
        BASIC_V091,
        {"id": "x", "component": "Slider", "label": "Volume", "value": NAN, "min": 0, "max": 10},
        {"id": "x", "component": "Slider", "label": "Volume", "value": "high", "min": 0, "max": 10},
    ),
    "basic-Slider.max-Infinity": (
        BASIC_V091,
        {"id": "x", "component": "Slider", "label": "Volume", "value": 5, "min": 0, "max": INF},
        {"id": "x", "component": "Slider", "label": "Volume", "value": 5, "min": 0, "max": "ten"},
    ),
    "basic-Text.weight-NaN": (
        BASIC_V091,
        {"id": "x", "component": "Text", "text": "Weighted", "weight": NAN},
        {"id": "x", "component": "Text", "text": "Weighted", "weight": "heavy"},
    ),
}


def _with(x: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": "root", "component": "Column", "children": ["t", "x"]},
        {"id": "t", "component": "Text", "text": "Kept"},
        x,
    ]


def _verdict(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Everything but the wording of the schema errors and problems (the two invalid values are named differently)."""
    out = {k: v for k, v in snapshot.items() if k != "report"}
    out["ok"] = snapshot["report"]["ok"]
    return out


@pytest.mark.parametrize("case", sorted(NON_FINITE))
def test_non_finite_number_is_invalid_and_handled_like_any_invalid_component(case: str) -> None:
    """Hand-off criterion: ``NaN``/``Infinity`` in a number role makes the component invalid. The surface is emitted
    without it, it is lost and resampled, nothing non-finite reaches the renderer, and the whole ``GateResult`` (but
    the error wording) equals that of the same canvas carrying a finite value the catalog rejects."""
    catalog_id, bad, control = NON_FINITE[case]
    raw = _canvas(catalog_id, _with(bad))
    assert re.search(r"\b-?(NaN|Infinity)\b", raw), (
        "premise: the canvas carries a bare non-finite literal"
    )
    result = UiGate().gate(raw, turn_id="n", canvas_idx=0, final=True)
    _require_surface(result, f"{case}: the rest of the card")
    emitted = _emitted(result)
    assert "x" not in emitted, (
        f"{case}: the component with a non-finite number was emitted: {emitted.get('x')}"
    )
    assert set(emitted) == {"root", "t"} and emitted["root"]["children"] == ["t"]
    assert result.lost == ("x",) and "x" in result.resample and not result.complete
    assert _non_finite(result.surface.messages) == []
    reference = UiGate().gate(
        _canvas(catalog_id, _with(control)), turn_id="n", canvas_idx=0, final=True
    )
    assert reference.lost == ("x",), f"premise: the control value is invalid under {catalog_id}"
    assert _verdict(_snapshot(result)) == _verdict(_snapshot(reference)), (
        f"{case}: not handled like any other invalid component: "
        f"{_first_difference(_verdict(_snapshot(result)), _verdict(_snapshot(reference)))}"
    )


@pytest.mark.parametrize(
    "case",
    [
        "v2-Bars.value-NaN",
        "v2-StatTiles.percent-Infinity",
        # green at BASE (json_repair reads a bare NaN as the string "NaN", which the Slider rejects): a guard
        "basic-Slider.value-NaN",
    ],
)
def test_non_finite_number_never_reaches_the_renderer_on_the_unguided_path(case: str) -> None:
    """Hand-off, the fixer path: the same canvas missing its last ``}`` (strict parse fails; structural pass, fixer,
    salvage). Whatever survives, no emitted number is non-finite, and the sibling Text is emitted. The basic case is
    green at BASE (json_repair 0.63.4 reads a bare ``NaN`` as the string ``"NaN"``) and is kept as the guard that a
    change of parser on this path cannot let one through."""
    catalog_id, bad, _ = NON_FINITE[case]
    raw = _canvas(catalog_id, _with(bad))[:-1]
    result = UiGate().gate(raw, turn_id="n", canvas_idx=0, final=False)
    _require_surface(result, f"{case} (unguided): the rest of the card")
    assert "t" in _emitted(result)
    assert _non_finite(result.surface.messages) == [], (
        f"{case}: a non-finite number reached the surface"
    )


def test_finite_numbers_and_the_words_nan_and_infinity_render_on_basic() -> None:
    """Too-wide guard (green at BASE, unmarked): the refusal is of non-finite *numbers*. Text that says "NaN" or
    "Infinity", and the largest finite double in a number role, still render, complete and unrepaired."""
    tree = [
        {"id": "root", "component": "Column", "children": ["a", "b", "c"]},
        {"id": "a", "component": "Text", "text": "NaN"},
        {"id": "b", "component": "Text", "text": "-Infinity and Infinity"},
        {"id": "c", "component": "Slider", "label": "Infinity", "value": 1e308, "min": -1e308, "max": 1.7976931348623157e308},
    ]  # fmt: skip
    result = UiGate().gate(_canvas(BASIC_V091, tree), turn_id="f", canvas_idx=0, final=True)
    _require_surface(result, "finite numbers on basic")
    assert set(_emitted(result)) == {"root", "a", "b", "c"}
    assert result.complete and result.repaired == 0 and result.lost == ()


def test_finite_numbers_and_the_words_nan_and_infinity_render_on_v2() -> None:
    """Too-wide guard on v2: ``Bars`` values 0, 8611.5 and 1e308, ``percent`` 0 and 100, ``highlight`` 0, and the
    words "NaN"/"Infinity" as a Stat's value and detail all render, complete and unrepaired."""
    tree = [
        {"id": "root", "component": "Column", "children": ["s", "b", "p", "c"]},
        {"id": "s", "component": "Stat", "label": "RESULT", "value": "NaN", "detail": "Infinity"},
        {"id": "b", "component": "Bars", "bars": [{"label": "zero", "value": 0}, {"label": "half", "value": 8611.5},
                                                  {"label": "huge", "value": 1e308}]},
        {"id": "p", "component": "StatTiles", "tiles": [{"title": "Empty", "value": "0", "percent": 0},
                                                        {"title": "Full", "value": "1", "percent": 100}]},
        {"id": "c", "component": "Comparison", "columns": [{"title": "A"}, {"title": "B"}],
         "rows": [{"label": "Price", "cells": ["1", "2"]}], "highlight": 0},
    ]  # fmt: skip
    result = UiGate().gate(_canvas(V2_ID, tree), turn_id="f", canvas_idx=0, final=True)
    _require_surface(result, "finite numbers on v2")
    assert set(_emitted(result)) == {"root", "s", "b", "p", "c"}
    assert result.complete and result.repaired == 0 and result.lost == ()


def _with_data(catalog_id: str, value: Any, path: Any = "/") -> str:
    return _compact(
        [
            {"version": "v0.9.1", "createSurface": {"surfaceId": "s", "catalogId": catalog_id}},
            {
                "version": "v0.9.1",
                "updateComponents": {
                    "surfaceId": "s",
                    "components": [
                        {"id": "root", "component": "Column", "children": ["t"]},
                        {"id": "t", "component": "Text", "text": "Kept"},
                    ],
                },
            },
            {
                "version": "v0.9.1",
                "updateDataModel": {"surfaceId": "s", "path": path, "value": value},
            },
        ]
    )


@pytest.mark.parametrize("catalog_id", [BASIC_V091, V2_ID], ids=["basic", "v2"])
@pytest.mark.parametrize(
    "value", [{"level": NAN}, {"rows": [1, INF]}, -INF], ids=["NaN", "Infinity-in-list", "-Inf"]
)
def test_non_finite_number_outside_the_components_fails_the_message(
    catalog_id: str, value: Any
) -> None:
    """Adversarial, the envelope case (coordinator, 2026-10-28; the implementation's reading of "a surface carrying
    NaN/Infinity … is invalid" for a number outside any component): a non-finite number in ``updateDataModel.value``
    makes that message an envelope error, so nothing is emitted and nothing non-finite reaches the renderer. The
    oracle for the rest of the verdict is BASE's handling of any other envelope error: the same canvas with
    ``updateDataModel.path`` a number (which the message schema rejects) gives the same ``GateResult`` but the wording."""
    raw = _with_data(catalog_id, value)
    assert re.search(r"\b-?(NaN|Infinity)\b", raw), (
        "premise: a bare non-finite literal in the data model"
    )
    result = UiGate().gate(raw, turn_id="d", canvas_idx=0, final=True)
    assert result.surface is None, "a message carrying a non-finite number reached the renderer"
    assert not result.report.ok and result.report.schema_errors, result.report
    assert any(
        "updateDataModel" in str(e) or e.path.startswith("[2]") for e in result.report.schema_errors
    ), [str(e) for e in result.report.schema_errors]
    control = UiGate().gate(
        _with_data(catalog_id, {"level": 3}, path=5), turn_id="d", canvas_idx=0, final=True
    )
    assert control.surface is None and not control.report.ok, (
        "premise: a numeric path is an envelope error"
    )
    assert _verdict(_snapshot(result)) == _verdict(_snapshot(control)), _first_difference(
        _verdict(_snapshot(result)), _verdict(_snapshot(control))
    )


@pytest.mark.parametrize("catalog_id", [BASIC_V091, V2_ID], ids=["basic", "v2"])
def test_finite_data_model_values_and_the_word_nan_still_pass_the_envelope(catalog_id: str) -> None:
    """Too-wide guard for the envelope case: finite extremes and the *string* "NaN" in the data model are not
    refused; the surface is emitted, complete and unrepaired."""
    raw = _with_data(
        catalog_id, {"big": 1.7976931348623157e308, "zero": 0, "neg": -1e308, "word": "NaN"}
    )
    result = UiGate().gate(raw, turn_id="d", canvas_idx=0, final=True)
    _require_surface(result, f"finite data model values ({catalog_id})")
    assert result.complete and result.repaired == 0 and result.lost == ()


# --- C9 and the reviewer-A rows (amendment 2026-10-28) -------------------------------------------------------------


def test_c9_reset_a_turn_without_create_surface_does_not_inherit_v2() -> None:
    """C1/C9 reset (reviewer A, mutant M2: the remembered id not cleared at a new turn). Turn ``a`` creates a v2
    surface. Turn ``b``'s only canvas has no ``createSurface`` and its ``root`` is a Stat: with nothing remembered the
    turn is on the basic catalog, which rejects the Stat, so ``root`` is the one id to resample and is not emitted.
    Under M2 the Stat validates against v2 and nothing is resampled."""
    gate = UiGate()
    first = gate.gate(_canvas(V2_ID, _v2_tree()), turn_id="a", canvas_idx=0, final=True)
    _require_surface(first, "C9 reset, turn a (v2)")
    stat_root = {"id": "root", "component": "Stat", "label": "NOW", "value": "17°"}
    update = _compact(
        [{"version": "v0.9.1", "updateComponents": {"surfaceId": "s", "components": [stat_root]}}]
    )
    second = gate.gate(update, turn_id="b", canvas_idx=0, final=True)
    assert second.resample == ("root",), (
        f"turn b inherited turn a's catalog: resample={second.resample}"
    )
    assert "root" not in _emitted(second)


def test_c9_two_turns_in_one_gate_use_each_turns_fixer_schema() -> None:
    """C1/C9 (reviewer A, mutant M4b: one fixer schema cached for the gate's life). In one gate, turn ``a`` is an
    unguided basic canvas that needs salvage (a Stat the basic catalog rejects), so the basic salvage schema is built
    first; turn ``b`` is C1's unguided v2 canvas whose Bars has one bar with no value. The v2 schema must be used in
    turn ``b``: the Bars survives with the K2 bar, and the card is complete."""
    gate = UiGate()
    basic = _canvas(
        BASIC_V091,
        [
            {"id": "root", "component": "Column", "children": ["t", "x"]},
            {"id": "t", "component": "Text", "text": "Kept"},
            {"id": "x", "component": "Stat", "label": "NOW", "value": "17°"},
        ],
    )
    first = gate.gate(basic[:-1], turn_id="a", canvas_idx=0, final=False)
    _require_surface(first, "C9 two-turn, turn a (basic salvage)")
    assert "x" in first.lost, "premise: turn a's Stat went through basic salvage and was dropped"
    bars = {
        "id": "b",
        "component": "Bars",
        "bars": [{"label": "K2", "value": 8611}, {"label": "Everest"}],
        "unit": "m",
    }
    v2 = _canvas(
        V2_ID,
        [
            {"id": "root", "component": "Column", "children": ["t", "b"]},
            {"id": "t", "component": "Text", "text": "Heights"},
            bars,
        ],
    )
    second = gate.gate(v2[:-1], turn_id="b", canvas_idx=0, final=False)
    _require_surface(second, "C9 two-turn, turn b (v2 salvage)")
    assert _emitted(second).get("b") == {
        "id": "b",
        "component": "Bars",
        "bars": [{"label": "K2", "value": 8611}],
        "unit": "m",
    }, f"turn b salvaged with turn a's (basic) schema: lost={second.lost}"
    assert second.complete and second.lost == ()


def test_c9_first_catalog_wins_for_the_turn() -> None:
    """C9: commit 0 creates a basic surface with a Text; commit 1 of the same turn re-sends ``createSurface`` naming
    the v2 id, with a Bars. The turn's catalog is the first one (the HUD answers the second create with "Surface
    already exists"), so the Bars is invalid there: pruned, lost, and the card is not complete."""
    gate = UiGate()
    first = gate.gate(
        _canvas(
            BASIC_V091,
            [
                {"id": "root", "component": "Column", "children": ["t"]},
                {"id": "t", "component": "Text", "text": "Heights"},
            ],
        ),
        turn_id="a",
        canvas_idx=0,
        final=False,
    )
    _require_surface(first, "C9, commit 0 (basic)")
    second = gate.gate(
        _canvas(
            V2_ID,
            [
                {"id": "root", "component": "Column", "children": ["t", "b"]},
                {"id": "t", "component": "Text", "text": "Heights"},
                {"id": "b", "component": "Bars", "bars": [{"label": "K2", "value": 8611}]},
            ],
        ),
        turn_id="a",
        canvas_idx=1,
        final=True,
    )
    assert "b" not in _emitted(second), (
        "C9: a later createSurface switched the turn to the v2 catalog"
    )
    assert "b" in second.lost and not second.complete, (second.lost, second.complete)


#: C9a: where a first ui commit is cut inside the catalogId string (55 of the 827 stored trials cut there).
V2_ID_CUTS = ("gearvis:ca", "gearvis:cards/", "gearvis:cards/v")


@pytest.mark.parametrize("prefix", V2_ID_CUTS)
def test_c9a_a_first_commit_cut_inside_the_catalog_id_does_not_lock_the_turn(prefix: str) -> None:
    """C9a (reviewer A, round 2): the catalog locks at the first ``createSurface`` the gate *emits*. Commit 0 is a v2
    canvas cut inside its catalogId string (``final=False``): nothing can be emitted from it, so nothing locks. Commit 1
    of the same turn, in the same gate, is the whole v2 canvas with a Bars: it is gated as v2, the Bars is emitted and
    the card is complete. A gate that locked at the first id it parsed (``"gearvis:ca"``) loses every later canvas."""
    tree = [
        {"id": "root", "component": "Column", "children": ["t", "b"]},
        {"id": "t", "component": "Text", "text": "Heights"},
        {"id": "b", "component": "Bars", "bars": [{"label": "K2", "value": 8611}], "unit": "m"},
    ]
    full = _canvas(V2_ID, tree)
    cut = full[: full.index(json.dumps(V2_ID)) + 1 + len(prefix)]
    assert cut.endswith(prefix) and V2_ID not in cut, (
        "premise: commit 0 ends inside the catalogId string"
    )
    gate = UiGate()
    first = gate.gate(cut, turn_id="a", canvas_idx=0, final=False)
    assert first.surface is None, f"premise: nothing is emitted from a cut create ({prefix!r})"
    second = gate.gate(full, turn_id="a", canvas_idx=1, final=True)
    _require_surface(second, f"C9a, commit 1 after a cut at {prefix!r}")
    assert _catalog_of(second) == V2_ID
    assert _emitted(second).get("b") == tree[2], (
        f"C9a {prefix!r}: the Bars was lost: lost={second.lost}"
    )
    assert second.complete and second.lost == ()


#: t0108a-r13-primes_table-K1 (run e6c3a81, stored trials; MANIFEST of q12_trials_sample.jsonl names the run), its
#: ``ui`` commit at canvas 6, verbatim: a later canvas of a basic turn cut inside the catalogId string.
STORED_CUT_CANVAS = '{"version":"v0.9.1","createSurface":{"surfaceId":"s","catalogId":"https://'


def test_c9a_a_later_canvas_cut_inside_the_catalog_id_gives_no_surface_as_at_base() -> None:
    """C9a, the other side: in a basic turn that has already emitted a surface, a later canvas cut inside the
    catalogId (the stored shape ``'catalogId': 'https://'``) names an id no validator knows and is refused, as at
    BASE: no surface. The oracle is the frozen BASE gate over the same two calls, field for field."""
    tree = [
        {"id": "root", "component": "Column", "children": ["h"]},
        {"id": "h", "component": "Text", "text": "Primes and squares", "variant": "h3"},
    ]
    calls = [(_canvas(BASIC_V091, tree), "r13", 5, False), (STORED_CUT_CANVAS, "r13", 6, False)]
    results = _run_both(calls)
    (tip0, base0), (tip1, base1) = results
    assert base0["surface"] is not None, "premise: the turn has emitted a basic surface"
    assert base1["surface"] is None, "premise: BASE refuses the cut canvas"
    assert tip1["surface"] is None, "C9a: a truncated catalogId on a later canvas was forgiven"
    assert (tip0, tip1) == (base0, base1), _first_difference(tip1, base1)


def test_c9a_forgiveness_reads_the_resent_create_not_the_first() -> None:
    """C9a (reviewer B, L6): only a re-sent ``createSurface`` naming a known catalog is forgiven, and the id read is the
    re-sent message's own. Canvas 1 is canvas 0 (a basic surface) plus a trailing ``createSurface`` whose catalogId is
    ``"https://"``. Message 0 is a known (basic) create, message 2 is not: no surface, and the problem names index 2,
    exactly as the frozen BASE gate reports it. Reading ``messages[0]`` instead would forgive the unknown id."""
    tree = [
        {"id": "root", "component": "Column", "children": ["t"]},
        {"id": "t", "component": "Text", "text": "Hi"},
    ]
    first = _canvas(BASIC_V091, tree)
    tail = '\n{"version":"v0.9.1","createSurface":{"surfaceId":"s","catalogId":"https://"}}'
    (tip0, base0), (tip1, base1) = _run_both(
        [(first, "a", 0, False), (first + tail, "a", 1, False)]
    )
    assert base0["surface"] is not None and base1["surface"] is None, (
        "premise: BASE refuses canvas 1"
    )
    assert "[2] unknown catalogId 'https://'" in base1["report"]["problems"], base1["report"][
        "problems"
    ]
    assert tip1["surface"] is None, (
        f"L6: the unknown re-sent id was forgiven: {tip1['report']['problems']}"
    )
    assert (tip0, tip1) == (base0, base1), _first_difference(tip1, base1)


# --- C10: integers beyond the largest double are non-finite --------------------------------------------------------

HUGE = 10**309  # > sys.float_info.max (1.797…e308); JSON.parse reads it as Infinity
assert sys.float_info.max < HUGE

#: (catalog, the component with a huge integer, the same component with a finite value the catalog rejects)
HUGE_INTS: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {
    "v2-Bars.value-10**309": (
        V2_ID,
        {"id": "x", "component": "Bars", "bars": [{"label": "K2", "value": HUGE}]},
        {"id": "x", "component": "Bars", "bars": [{"label": "K2", "value": -1}]},
    ),
    "v2-Bars.value--10**309": (
        V2_ID,
        {"id": "x", "component": "Bars", "bars": [{"label": "K2", "value": -HUGE}]},
        {"id": "x", "component": "Bars", "bars": [{"label": "K2", "value": -1}]},
    ),
    "basic-Slider.max-10**309": (
        BASIC_V091,
        {"id": "x", "component": "Slider", "label": "Volume", "value": 5, "min": 0, "max": HUGE},
        {"id": "x", "component": "Slider", "label": "Volume", "value": 5, "min": 0, "max": "ten"},
    ),
    "basic-Text.weight-10**309": (
        BASIC_V091,
        {"id": "x", "component": "Text", "text": "Weighted", "weight": HUGE},
        {"id": "x", "component": "Text", "text": "Weighted", "weight": "heavy"},
    ),
}


def _beyond_double(messages: Any) -> list[Any]:
    """Numbers no double can hold: NaN/±Infinity floats, and integers whose magnitude exceeds the largest double."""
    out = []
    for _, v in _leaves(list(messages)):
        if isinstance(v, bool):
            continue
        if (isinstance(v, float) and not math.isfinite(v)) or (
            isinstance(v, int) and abs(v) > sys.float_info.max
        ):
            out.append(v)
    return out


@pytest.mark.parametrize("case", sorted(HUGE_INTS))
def test_c10_huge_integer_in_a_component_is_refused_like_nan(case: str) -> None:
    """C10: an integer past the largest double makes its component invalid exactly as NaN does: the rest emitted,
    the component lost and resampled, no such number emitted, and the ``GateResult`` (but the error wording) equal to
    the same canvas with a finite value the catalog rejects."""
    catalog_id, bad, control = HUGE_INTS[case]
    raw = _canvas(catalog_id, _with(bad))
    assert str(HUGE) in raw, "premise: the canvas carries the integer literal"
    result = UiGate().gate(raw, turn_id="h", canvas_idx=0, final=True)
    _require_surface(result, f"{case}: the rest of the card")
    emitted = _emitted(result)
    assert "x" not in emitted, f"C10 {case}: the component with a huge integer was emitted"
    assert result.lost == ("x",) and "x" in result.resample and not result.complete
    assert _beyond_double(result.surface.messages) == []
    reference = UiGate().gate(
        _canvas(catalog_id, _with(control)), turn_id="h", canvas_idx=0, final=True
    )
    assert reference.lost == ("x",), "premise: the control value is invalid"
    assert _verdict(_snapshot(result)) == _verdict(_snapshot(reference)), _first_difference(
        _verdict(_snapshot(result)), _verdict(_snapshot(reference))
    )


@pytest.mark.parametrize("catalog_id", [BASIC_V091, V2_ID], ids=["basic", "v2"])
@pytest.mark.parametrize("value", [{"level": HUGE}, -HUGE], ids=["nested", "bare-negative"])
def test_c10_huge_integer_outside_the_components_fails_the_message(
    catalog_id: str, value: Any
) -> None:
    """C10, the envelope: an integer past the largest double in ``updateDataModel.value`` fails that message exactly
    as NaN does (no surface; the verdict equal to that of any other envelope error, the numeric-path control)."""
    raw = _with_data(catalog_id, value)
    result = UiGate().gate(raw, turn_id="h", canvas_idx=0, final=True)
    assert result.surface is None, (
        "C10: a message carrying an integer past the largest double reached the renderer"
    )
    assert not result.report.ok and result.report.schema_errors
    control = UiGate().gate(
        _with_data(catalog_id, {"level": 3}, path=5), turn_id="h", canvas_idx=0, final=True
    )
    assert _verdict(_snapshot(result)) == _verdict(_snapshot(control)), _first_difference(
        _verdict(_snapshot(result)), _verdict(_snapshot(control))
    )


@pytest.mark.parametrize("catalog_id", [BASIC_V091, V2_ID], ids=["basic", "v2"])
def test_c10_too_wide_guard_largest_finite_values_still_pass(catalog_id: str) -> None:
    """C10 too-wide guard: 1e308, the largest double, and the integer 10**308 (below it) are finite and pass, in a
    component and in the data model."""
    big_int = 10**308
    assert big_int < sys.float_info.max
    if catalog_id == V2_ID:
        x = {
            "id": "x",
            "component": "Bars",
            "bars": [
                {"label": "a", "value": 1e308},
                {"label": "b", "value": big_int},
                {"label": "c", "value": sys.float_info.max},
            ],
        }
    else:
        x = {
            "id": "x",
            "component": "Slider",
            "label": "Volume",
            "value": 1e308,
            "min": 0,
            "max": big_int,
        }
    result = UiGate().gate(_canvas(catalog_id, _with(x)), turn_id="w", canvas_idx=0, final=True)
    _require_surface(result, f"C10 guard, component ({catalog_id})")
    assert "x" in _emitted(result) and result.complete and result.lost == ()
    data = UiGate().gate(
        _with_data(catalog_id, {"a": 1e308, "b": big_int}), turn_id="w", canvas_idx=0, final=True
    )
    _require_surface(data, f"C10 guard, data model ({catalog_id})")
    assert data.complete


# --- C8 (gate half): no network call is added ------------------------------------------------------------------------

_NETWORK_MODULES = ("socket", "http", "urllib", "httpx", "requests", "aiohttp", "websockets", "ssl")


def test_the_gate_imports_no_network_module() -> None:
    """C8 (green at BASE, unmarked): ``ui_gate.py`` imports nothing that can open a connection. Read with ``ast``
    (imports only, so prose cannot trip it)."""
    import ast

    tree = ast.parse(
        (REPO / "local_astra" / "conductor" / "ui_gate.py").read_text(encoding="utf-8")
    )
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    hits = sorted(n for n in names if n.split(".")[0] in _NETWORK_MODULES)
    assert hits == [], f"C8: ui_gate.py imports {hits}"


def test_v2_catalog_id_constant_is_the_task_id() -> None:
    """The id this module relies on is H-T184-4's (group A, merged; unmarked)."""
    assert CARDS_V2_CATALOG_ID == V2_ID
