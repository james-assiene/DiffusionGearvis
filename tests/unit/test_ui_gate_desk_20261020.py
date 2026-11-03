"""T-0127 criterion 8 (and 3, 4, 7 on real data): the six card turns of the 2026-10-20 desk session, exactly.

The fixture ``tests/fixtures/a2ui/desk_20261020_card_children.jsonl`` holds, verbatim from the session log, every
``ui`` commit, every ``ui.gate`` record and the final ``astra.ui.surface`` of turns t3, t5, t6, t8, t9 and t10 (the
sixth case, "10 largest countries", located by the test-author: criterion 8(d); provenance in the MANIFEST). Each
turn's ``ui`` commits are gated in log order by a fresh ``UiGate`` with ``final=False`` and the logged turn id and
canvas_idx, which is what ``Session._gate_surface`` did at the desk.

"CC disabled" (criteria 8(a) and 8(c)) is a validator handed to the component stage that reads a Card carrying
``children`` as catalog-valid, so no component rule fires on it, and delegates everything else to the real validator.
The gate's own validation still uses the real validator, so the Card is pruned (strict path) or dropped by salvage
(unguided path) as at the base. Reading: CC fires only on a Card the validator *passed to* ``repair_components``
rejects (criterion 1: "a Card that the catalog rejects"); an implementation that judges the Card by another route
would make the disabled arm fire anyway, and these rows would say so.

Expected trees are hand-written from the fixture's last canvases and the criteria: CC with one id sets ``child``; CC
with several ids wraps them in ``__gate_<card id>_body`` (H-T127-1); a reference to an id the model never defined is
pruned (t8's ``r6``, t10's even ``t`` ids); t3's Lists with ``items`` and Texts with ``variant: "body2"`` stay invalid
(H-T127-2/3 declined) and are pruned.

Reading where the task is open: criterion 8(a) "byte for byte with the rules disabled" cannot hold for t8 (seq 9867)
after criterion 7, which removes the third message (the ``r3`` re-send) that the log holds. The row compares the
first two messages exactly and allows the third only as the logged one; ``test_t8_emits_two_messages`` pins its
removal. t10 (seq 11870) has the same third message (``l``, ``r1``, ``r9``) and is treated the same way.

Regime: CPU, pure; the session's regime is in the MANIFEST (lean, canvas 256, 48 steps, thinking on, v0.29.0,
DiffusionGemma NVFP4); the gate is deterministic, n = 1 per turn.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import ui_gate as ui_gate_module
from local_astra.conductor.ui_gate import GateResult, UiGate
from local_astra.ui import component_repair as component_repair_module

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "a2ui" / "desk_20261020_card_children.jsonl"
CATALOG = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"

#: the six turns, t9 first (the otherwise-valid card), then in log order
TURNS = ("t9", "t3", "t5", "t6", "t8", "t10")
#: the five the task expects complete after CC (criterion 8(b))
COMPLETE_AFTER = ("t9", "t5", "t6", "t8", "t10")
#: the Card each turn wrote with ``children`` (criterion 8(c): ``lost`` names it when CC is disabled)
CARD_OF = {"t3": "root", "t5": "card", "t6": "card", "t8": "root", "t9": "c", "t10": "card"}
#: the turns whose logged final surface carries a third, restore-made message (criterion 7)
RESTORE_RESEND = {"t8", "t10"}


# --- the fixture ------------------------------------------------------------------------------------------------


@cache
def records() -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip())


def ui_commits(turn: str) -> list[dict[str, Any]]:
    out = [
        r["data"]
        for r in records()
        if r["kind"] == "llm.commit" and r["turn_id"] == turn and r["data"]["segment"] == "ui"
    ]
    assert out, f"fixture: no ui commit for {turn}"
    return out


def logged_gates(turn: str) -> list[dict[str, Any]]:
    return [r["data"] for r in records() if r["kind"] == "ui.gate" and r["turn_id"] == turn]


def logged_final(turn: str) -> dict[str, Any]:
    finals = [
        r["data"]["msg"]
        for r in records()
        if r["kind"] == "wire.out"
        and r["data"]["msg"].get("type") == "astra.ui.surface"
        and r["data"]["msg"].get("surface_id") == f"turn-{turn}"
        and r["data"]["msg"].get("final") is True
    ]
    assert len(finals) == 1, f"fixture: {len(finals)} final surfaces for {turn}"
    return finals[0]


# --- disabling CC (criteria 8(a), 8(c)) -------------------------------------------------------------------------


class NoCardChildren:
    """A validator for the component stage only: a Card carrying ``children`` reads as valid, so no rule fires on
    it. Every other name is the real validator's (delegation, not enumeration)."""

    def __init__(self, real: Any) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def component_errors(self, component: Any) -> Any:
        if (
            isinstance(component, Mapping)
            and component.get("component") == "Card"
            and "children" in component
        ):
            return []
        return self._real.component_errors(component)


def disable_cc(monkeypatch: pytest.MonkeyPatch) -> None:
    real = component_repair_module.repair_components

    def without_cc(components: Any, validator: Any, *args: Any, **kwargs: Any) -> Any:
        return real(components, NoCardChildren(validator), *args, **kwargs)

    monkeypatch.setattr(component_repair_module, "repair_components", without_cc)
    if hasattr(ui_gate_module, "repair_components"):
        monkeypatch.setattr(ui_gate_module, "repair_components", without_cc)


@contextmanager
def cc_disabled() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        disable_cc(mp)
        yield


# --- gating a turn --------------------------------------------------------------------------------------------


def gate_turn(turn: str) -> list[GateResult]:
    gate = UiGate()
    return [
        gate.gate(c["text"], turn_id=turn, canvas_idx=c["canvas_idx"], final=False)
        for c in ui_commits(turn)
    ]


def tree(results: Sequence[GateResult]) -> dict[str, Any]:
    """The surface as the renderer holds it after these results: by id, the last definition wins."""
    out: dict[str, Any] = {}
    for result in results:
        for m in result.surface.messages if result.surface is not None else ():
            body = m.get("updateComponents") if isinstance(m, Mapping) else None
            for c in (body or {}).get("components", ()):
                out[c["id"]] = c
    return out


def verdict(result: GateResult) -> tuple[bool, tuple[str, ...]]:
    """``(complete, lost)`` of one call (criterion 4); ``NotImplementedError`` while the fields do not exist."""
    if not hasattr(result, "complete") or not hasattr(result, "lost"):
        raise NotImplementedError("T-0127 criterion 4: GateResult has no `complete`/`lost` yet")
    complete, lost = result.complete, result.lost
    assert isinstance(complete, bool), f"criterion 4: complete is {complete!r}"
    assert isinstance(lost, tuple) and all(isinstance(x, str) for x in lost), (
        f"criterion 4: lost is {lost!r}"
    )
    return complete, lost


def names(line: str, token: str) -> bool:
    """Whether ``line`` names ``token`` as a whole id (``l`` does not match inside ``List`` or ``lost``)."""
    return re.search(rf"(?<![\w\-]){re.escape(token)}(?![\w\-])", line) is not None


def problem_naming(result: GateResult, *tokens: str) -> list[str]:
    return [p for p in result.report.problems if all(names(p, t) for t in tokens)]


# --- hand-written trees ---------------------------------------------------------------------------------------


def text(cid: str, s: str, variant: str | None = None) -> dict[str, Any]:
    c: dict[str, Any] = {"id": cid, "component": "Text", "text": s}
    if variant is not None:
        c["variant"] = variant
    return c


def kids(cid: str, kind: str, *children: str) -> dict[str, Any]:
    return {"id": cid, "component": kind, "children": list(children)}


def card(cid: str, child: str) -> dict[str, Any]:
    return {"id": cid, "component": "Card", "child": child}


def by_id(*components: dict[str, Any]) -> dict[str, Any]:
    out = {c["id"]: c for c in components}
    assert len(out) == len(components), "hand tree: duplicate id"
    return out


def rows(pairs: Sequence[tuple[str, str, str, str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rid, aid, a, bid, b in pairs:
        out += [kids(rid, "Row", aid, bid), text(aid, a), text(bid, b)]
    return out


T9_ITEMS = [
    "225g unsalted butter",
    "225g caster sugar",
    "4 large eggs",
    "275g self-raising flour",
    "2 tsp baking powder",
    "4 tbsp milk",
    "Finely grated zest of 2 lemons",
    "Juice of 1.5 lemons",
    "175g granulated sugar",
]
T6_STEPS = [
    "1. Remove the wheel from the bike",
    "2. Deflate and use tire levers to pry the edge off",
    "3. Remove the old tube and check for debris",
    "4. Insert a new partially inflated tube",
    "5. Push the tire bead back onto the rim",
    "6. Pump up the tire and reinstall the wheel",
]
T10_COUNTRIES = [
    "Russia: 146m",
    "Canada: 38m",
    "China: 1.4b",
    "USA: 333m",
    "Brazil: 214m",
    "Australia: 26m",
    "India: 1.4b",
    "Kazakhstan: 19m",
    "Algeria: 45m",
    "DRC: 100m",
]

#: criterion 8(b): the accumulated tree after the last call, per complete turn
EXPECTED_TREE: dict[str, dict[str, Any]] = {
    "t9": by_id(
        kids("root", "Column", "c"),
        card("c", "__gate_c_body"),
        kids("__gate_c_body", "Column", "h", "l"),
        text("h", "Shopping List", "h3"),
        kids("l", "List", *[f"i{k}" for k in range(1, 10)]),
        *[text(f"i{k}", s) for k, s in enumerate(T9_ITEMS, start=1)],
    ),
    "t5": by_id(
        kids("root", "Column", "card"),
        card("card", "__gate_card_body"),
        kids("__gate_card_body", "Column", "h", "r1", "r2", "r3", "r4"),
        text("h", "iPhone 16 vs Pixel 9", "h3"),
        *rows(
            [
                ("r1", "a1", "Chip:", "b1", "A18"),
                ("r2", "a2", "Display:", "b2", "Super Retina X"),
                ("r3", "a3", "Camera:", "b3", "Dual 48MP"),
                ("r4", "a4", "Port:", "b4", "USB-C"),
            ]
        ),
    ),
    "t6": by_id(
        kids("root", "Column", "h", "card"),
        text("h", "How to change a bike tyre", "h3"),
        card("card", "steps"),
        kids("steps", "Column", *[f"s{k}" for k in range(1, 7)]),
        *[text(f"s{k}", s) for k, s in enumerate(T6_STEPS, start=1)],
    ),
    "t8": by_id(
        card("root", "__gate_root_body"),
        # r6 was never defined: the reference is pruned (criterion 3), not a loss
        kids("__gate_root_body", "Column", "h", "r1", "r2", "r3", "r4", "r5", "d", "r7"),
        text("h", "Expense Summary", "h3"),
        *rows(
            [
                ("r1", "a1", "Rent:", "b1", "1200"),
                ("r2", "a2", "Food:", "b2", "420"),
                ("r3", "a3", "Travel:", "b3", "180"),
                ("r4", "a4", "Energy:", "b4", "95"),
                ("r5", "a5", "Phone:", "b5", "30"),
                ("r7", "a7", "Total:", "b7", "1925"),
            ]
        ),
        {"id": "d", "component": "Divider"},
    ),
    "t10": by_id(
        kids("root", "Column", "h", "card"),
        text("h", "10 Largest Countries by Area", "h3"),
        card("card", "l"),
        kids("l", "List", *[f"r{k}" for k in range(1, 11)]),
        # each Row's second Text (t2, t4, ... t20) was never defined: pruned references
        *[kids(f"r{k}", "Row", f"t{2 * k - 1}") for k in range(1, 11)],
        *[text(f"t{2 * k - 1}", s) for k, s in enumerate(T10_COUNTRIES, start=1)],
    ),
}

#: criterion 8(b): t3 after the task -- the root Card repaired, the out-of-scope Lists and Texts pruned
T3_TREE = by_id(
    card("root", "__gate_root_body"),
    kids("__gate_root_body", "Column", "title", "ingredients", "steps"),
    text("title", "Simple Omelette", "h3"),
    kids("ingredients", "Column"),
    kids("steps", "Column"),
)
T3_LOST = {"h2": "Text", "inglist": "List", "h2b": "Text", "steplist": "List"}


# --- premises ---------------------------------------------------------------------------------------------------


def test_premise_the_fixture_is_the_six_turns() -> None:
    """Guard: 34 records; per turn, the logged ``ui.gate`` records pair one-to-one with the ``ui`` commits; every
    logged gate said ``problems: []`` (the evidence table) and every final surface is ``final: true``."""
    assert len(records()) == 34
    for turn in TURNS:
        commits, gates = ui_commits(turn), logged_gates(turn)
        assert [c["canvas_idx"] for c in commits] == [g["canvas_idx"] for g in gates], turn
        assert all(g["problems"] == [] and g["gated"] for g in gates), turn
        assert logged_final(turn)["canvas_idx"] == commits[-1]["canvas_idx"], turn
    assert ui_commits("t10")[-1]["text"].count("10 Largest Countries by Area") == 1


# --- criterion 8(a): today, reproduced ------------------------------------------------------------------------


@pytest.mark.parametrize("turn", TURNS)
def test_today_reproduces_logged_surfaces(turn: str) -> None:
    """Criterion 8(a), green at base and after: with CC disabled, every call reproduces the logged ``ui.gate``
    verdict (``ok``, ``repairs``, ``structural_repairs``, surface emitted or not) and the last call's messages are
    the logged final surface's, byte for byte as JSON. t8/t10: the first two messages exactly; a third only as the
    logged one (criterion 7 removes it; see the module docstring)."""
    with cc_disabled():
        results = gate_turn(turn)
    for result, logged in zip(results, logged_gates(turn), strict=True):
        got = {
            "ok": bool(result.report.ok),
            "repairs": result.repaired,
            "structural_repairs": result.structural_repairs,
            "surface_emitted": result.surface is not None,
        }
        want = {k: logged[k] for k in got}
        assert got == want, f"{turn} canvas {logged['canvas_idx']}: {got} != logged {want}"
    last = results[-1]
    final = logged_final(turn)
    assert last.surface is not None and last.surface.canvas_idx == final["canvas_idx"]
    got_msgs = json.loads(json.dumps(list(last.surface.messages)))
    want_msgs = final["messages"]
    if turn in RESTORE_RESEND:
        assert len(want_msgs) == 3, "premise: the log holds the restore re-send"
        assert got_msgs[:2] == want_msgs[:2], f"{turn}: the first two messages changed"
        assert len(got_msgs) in (2, 3) and got_msgs[2:] in ([], want_msgs[2:]), got_msgs[2:]
    else:
        assert got_msgs == want_msgs, (
            f"{turn}: emitted messages differ from the logged final surface"
        )


# --- criterion 8(b): after the task ---------------------------------------------------------------------------


@pytest.mark.parametrize("turn", COMPLETE_AFTER)
def test_desk_cards_complete_after_repair(turn: str) -> None:
    """Criterion 8(b): t5, t6, t8, t9 and t10 end ``complete`` with nothing lost, and the renderer's tree is exactly
    the hand-written one: every model-written component present (the Card in its CC shape) and every model string
    present. Mutations 1 (always wrap) and 2 (reversed wrap) redden t6/t10 and t5/t8/t9 respectively."""
    results = gate_turn(turn)
    complete, lost = verdict(results[-1])
    got = tree(results)
    assert got == EXPECTED_TREE[turn], (
        f"{turn}:\n got  {json.dumps(got, ensure_ascii=False)}\n want "
        f"{json.dumps(EXPECTED_TREE[turn], ensure_ascii=False)}"
    )
    assert lost == (), f"{turn}: lost {lost}"
    assert complete is True, f"{turn}: not complete; problems {results[-1].report.problems}"
    assert results[-1].report.ok


def test_t3_is_reported_incomplete() -> None:
    """Criterion 8(b), t3: the root Card is repaired (wrapped, three ids), but List ``items`` and Text ``variant:
    "body2"`` are out of scope (H-T127-2/3 declined), so the two Lists and the two Texts are lost, each named in
    ``problems`` with its kind, and the turn is not complete."""
    results = gate_turn("t3")
    complete, lost = verdict(results[-1])
    assert tree(results) == T3_TREE, json.dumps(tree(results))
    assert set(lost) == set(T3_LOST) and len(lost) == len(T3_LOST), f"lost {lost}"
    for cid, kind in T3_LOST.items():
        assert problem_naming(results[-1], cid, kind), (
            f"criterion 3: no problems line names {cid} ({kind}): {results[-1].report.problems}"
        )
    assert complete is False


# --- criterion 8(c): CC disabled ------------------------------------------------------------------------------


@pytest.mark.parametrize("turn", TURNS)
def test_without_cc_cards_are_reported_lost(turn: str) -> None:
    """Criterion 8(c), adversarial (the evidence's failure mode: ``ok, problems []`` on a card the user never saw):
    with CC disabled every case is incomplete, ``lost`` names its Card, and ``problems`` names it with its kind.
    The verdict ``ok`` is unchanged (mutation 9): the surface is still emitted."""
    with cc_disabled():
        results = gate_turn(turn)
    last = results[-1]
    complete, lost = verdict(last)
    cid = CARD_OF[turn]
    assert last.surface is not None and last.report.ok, "ok keeps its meaning (criterion 3)"
    assert complete is False, f"{turn}: a surface without its Card read complete"
    assert cid in lost, f"{turn}: lost {lost} does not name the Card {cid!r}"
    assert last.report.problems, f"{turn}: problems is empty"
    assert problem_naming(last, cid, "Card"), (
        f"criterion 3: no problems line names {cid} (Card): {last.report.problems}"
    )


# --- criterion 3 on t8: a dangling reference is not a loss ----------------------------------------------------


def test_t8_dangling_r6_is_pruned_reference_not_loss() -> None:
    """Criterion 3: ``r6`` is referenced by the root Card and never defined. It is listed as "reference pruned"
    (the criterion's words, read case-insensitively), it is not in ``lost``, and t8 is still complete."""
    results = gate_turn("t8")
    complete, lost = verdict(results[-1])
    assert "r6" not in lost, lost
    lines = problem_naming(results[-1], "r6")
    assert any("reference pruned" in p.lower() for p in lines), (
        f"criterion 3: r6 is not listed as a pruned reference: {results[-1].report.problems}"
    )
    assert complete is True


# --- criterion 7 on real data -------------------------------------------------------------------------------


def updated_ids(result: GateResult) -> list[list[str]]:
    assert result.surface is not None
    return [
        [c["id"] for c in m["updateComponents"]["components"]]
        for m in result.surface.messages
        if "updateComponents" in m
    ]


@pytest.mark.parametrize("cc", ["on", "off"])
def test_t8_emits_two_messages(cc: str) -> None:
    """Criterion 7: t8's canvas 2 re-sends ``r3`` itself, so the restore does not re-send it after it: 2 messages
    (createSurface + one updateComponents), not 3, and ``r3`` appears once. Independent of CC (mutation 8)."""
    if cc == "off":
        with cc_disabled():
            results = gate_turn("t8")
    else:
        results = gate_turn("t8")
    last = results[-1]
    assert last.surface is not None
    assert len(last.surface.messages) == 2, updated_ids(last)
    assert sum(ids.count("r3") for ids in updated_ids(last)) == 1
    assert tree(results)["r3"] == kids("r3", "Row", "a3", "b3")


def test_t10_restore_does_not_resend_arriving_ids() -> None:
    """Criterion 7 on the sixth case: canvas 3 re-sends ``l``, ``r1`` and ``r9`` itself, so no restore message
    re-sends them (the log's third message disappears) and each id is defined once in the last call."""
    results = gate_turn("t10")
    last = results[-1]
    assert last.surface is not None
    flat = [cid for ids in updated_ids(last) for cid in ids]
    assert len(last.surface.messages) == 2, updated_ids(last)
    assert len(flat) == len(set(flat)), f"an id is sent twice in one call: {flat}"
