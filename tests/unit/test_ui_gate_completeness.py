"""T-0127 criteria 3, 4, 7 and 11 on hand-written canvases: losses named in ``problems``, ``GateResult.complete`` and
``GateResult.lost``, the restore that overwrote a fresh definition, and the negatives.

``complete`` is read from criterion 4 exactly: true iff a surface was emitted, the accumulated tree holds ``root``,
every component in it is reachable from ``root``, and ``lost`` is empty. ``lost`` (criterion 3) is the ids written on
*this* canvas, after repair, that are absent from the emitted messages, plus ids dropped by salvage; a reference to an
id the model never defined is a pruned reference, not a loss. Every expected value below is written by hand.

Readings where the task is open (named in the test-author report):

- A Card held back because its required ``child`` has not arrived yet (T-0010 A9) is written on the canvas and absent
  from what it emits, so it is in that call's ``lost`` (the literal criterion 3); the canvas that brings the child
  completes it, and that call's ``lost`` is empty (losses recomputed per call).
- A loss line names the id and its kind as whole words; the reason's wording is not pinned.

Rows that hold at the base e78e10b and must hold after (criterion 11, the restore premise) are guards; every other
row is red at the base and needs the task. With no surface emitted, ``complete`` is false, ``lost`` is empty and no
loss line is added (orchestrator ruling, 2026-10-20).

Regime: CPU, pure, the vendored A2UI v0.9.1 schema; deterministic, n = 1 per row.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.ui_gate import GateResult, UiGate
from local_astra.conductor.ui_instructions import UI_EXAMPLE
from local_astra.ui import MESSAGE_KINDS, A2UIValidator, parse_concatenated_json
from local_astra.ui import component_repair as component_repair_module

REPO = Path(__file__).resolve().parents[2]
QUIZ = REPO / "tests" / "fixtures" / "a2ui" / "quiz_example.jsonl"
DESK = REPO / "tests" / "fixtures" / "a2ui" / "desk_20261020_card_children.jsonl"
CATALOG = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
CREATE = '{"version":"v0.9.1","createSurface":{"surfaceId":"s","catalogId":"' + CATALOG + '"}}'
TURN = "t"
SID = f"turn-{TURN}"


def update(*components: str) -> str:
    return (
        '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":['
        + ",".join(components)
        + "]}}"
    )


def canvas(*components: str) -> str:
    return CREATE + "\n" + update(*components)


ROOT_A = '{"id":"root","component":"Column","children":["a"]}'
A = '{"id":"a","component":"Text","text":"one half"}'
Z = '{"id":"z","component":"Text","text":"stray words"}'

#: all valid, root reaches everything
VALID = canvas(ROOT_A, A)
#: cut inside the header: no message recoverable, no surface
CUT = '{"version":"v0.9.1","createSur'
#: a valid component and no root yet (progressive-render warning, a surface is emitted)
NO_ROOT = canvas(A)
#: a valid orphan nobody references
ORPHAN = canvas(ROOT_A, A, Z)
#: two components the catalog rejects and no rule repairs (H-T127-2/3 declined): a Text variant and a List of strings
LOSSES = canvas(
    '{"id":"root","component":"Column","children":["a","b","l"]}',
    A,
    '{"id":"b","component":"Text","text":"Instructions","variant":"body2"}',
    '{"id":"l","component":"List","items":["2 eggs","Cheese"]}',
)
#: a Card whose required child arrives on the next canvas (T-0010 A9)
DEFERRED_1 = canvas(
    '{"id":"root","component":"Column","children":["c"]}',
    '{"id":"c","component":"Card","child":"x"}',
)
DEFERRED_2 = canvas('{"id":"x","component":"Text","text":"inside the card"}')


def gate_seq(*raws: str) -> list[GateResult]:
    gate = UiGate()
    return [gate.gate(r, turn_id=TURN, canvas_idx=i, final=False) for i, r in enumerate(raws)]


def verdict(result: GateResult) -> tuple[bool, tuple[str, ...]]:
    """``(complete, lost)`` (criterion 4); ``NotImplementedError`` while the fields do not exist."""
    if not hasattr(result, "complete") or not hasattr(result, "lost"):
        raise NotImplementedError("T-0127 criterion 4: GateResult has no `complete`/`lost` yet")
    complete, lost = result.complete, result.lost
    assert isinstance(complete, bool), f"criterion 4: complete is {complete!r}"
    assert isinstance(lost, tuple) and all(isinstance(x, str) for x in lost), lost
    return complete, lost


def names(line: str, token: str) -> bool:
    return re.search(rf"(?<![\w\-]){re.escape(token)}(?![\w\-])", line) is not None


def tree(results: Sequence[GateResult]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for result in results:
        for m in result.surface.messages if result.surface is not None else ():
            for c in (m.get("updateComponents") or {}).get("components", ()):
                out[c["id"]] = c
    return out


# --- criterion 4: the truth table -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "raws", "want_surface", "want_complete", "want_lost"),
    [
        ("all valid", [VALID], True, True, ()),
        ("no surface", [CUT], False, False, ()),
        ("no root", [NO_ROOT], True, False, ()),
        ("unreachable component", [ORPHAN], True, False, ()),
        ("lost ids", [LOSSES], True, False, ("b", "l")),
    ],
)
def test_complete_truth_table(
    name: str,
    raws: list[str],
    want_surface: bool,
    want_complete: bool,
    want_lost: tuple[str, ...],
) -> None:
    """Criterion 4: each condition alone makes ``complete`` false; all of them together make it true. "lost ids" is
    the row mutation 5 (``complete`` ignores ``lost``) fails; "unreachable component" is mutation 4's."""
    last = gate_seq(*raws)[-1]
    complete, lost = verdict(last)
    assert (last.surface is not None) == want_surface, f"setup ({name}): {last.report}"
    assert sorted(lost) == sorted(want_lost), f"{name}: lost {lost}"
    assert complete is want_complete, (
        f"{name}: complete {complete}; problems {last.report.problems}"
    )


def test_a_deferred_container_completed_later() -> None:
    """Criterion 4 with T-0010 A9: the Card waiting for its child is lost on canvas 1 (incomplete); canvas 2 brings
    only the child, the gate restores the Card and the reference to it, and that call is complete with nothing lost."""
    r1, r2 = gate_seq(DEFERRED_1, DEFERRED_2)
    c1, lost1 = verdict(r1)
    c2, lost2 = verdict(r2)
    assert r1.surface is not None and r2.surface is not None
    assert "c" in lost1 and c1 is False, (c1, lost1)
    assert tree([r1, r2])["root"]["children"] == ["c"], "setup: A9 restored the Card"
    assert lost2 == () and c2 is True, (c2, lost2, r2.report.problems)


def test_a_progressive_three_canvas_fence_recomputes_losses_per_call() -> None:
    """Adversarial (mutation 6: losses accumulate across canvases). One fence, three canvases: canvas 1 is cut inside
    ``b`` (a stub, lost, incomplete); canvas 2 is cut inside ``b``'s text; canvas 3 is whole. Canvas 3 is complete
    and its ``lost`` is empty although canvas 1 lost ``b``."""
    full = canvas(
        '{"id":"root","component":"Column","children":["h","b"]}',
        '{"id":"h","component":"Text","text":"Your trip"}',
        '{"id":"b","component":"Text","text":"Tuesday"}',
    )
    cut1 = full[: full.index('{"id":"b"') + len('{"id":"b","component":"')]
    cut2 = full[: full.index('"Tuesday"') + len('"Tue')]
    r1, _, r3 = gate_seq(cut1, cut2, full)
    c1, lost1 = verdict(r1)
    c3, lost3 = verdict(r3)
    assert r1.surface is not None, f"setup: canvas 1 still renders root and h: {r1.report}"
    assert "b" in lost1 and c1 is False, (c1, lost1)
    assert lost3 == () and c3 is True, (c3, lost3, r3.report.problems)


def test_the_complete_verdict_reads_the_accumulated_tree() -> None:
    """Criterion 4 "accumulated tree": canvas 1 has the orphan ``z``; canvas 2 omits it but the renderer still holds
    it (unreachable), so canvas 2 is not complete although its own components all hang under ``root``."""
    r1, r2 = gate_seq(ORPHAN, VALID)
    assert "z" in tree([r1, r2]), "setup: z was emitted on canvas 1"
    assert verdict(r2)[0] is False


def test_a_canvas_with_only_create_surface_is_not_complete() -> None:
    """Criterion 4 (reviewer B, ``has_root``): a canvas holding only ``createSurface`` emits a surface with no
    components; the accumulated tree has no ``root``, so the call is not complete, and nothing is lost."""
    (result,) = gate_seq(CREATE)
    assert result.surface is not None, f"setup: {result.report}"
    assert verdict(result) == (False, ())


def test_reserved_ids_count_as_emitted() -> None:
    """Criterion 2 / implementation note "reserved ids count as emitted" (mutant x6): a component in the reserved
    namespace that the gate does not emit (here a model-written ``__gate_zz`` Text the catalog rejects) is never named
    in ``lost``, so it cannot make an otherwise complete call incomplete."""
    raw = canvas(
        ROOT_A, A, '{"id":"__gate_zz","component":"Text","text":"one half","variant":"body2"}'
    )
    (result,) = gate_seq(raw)
    assert result.surface is not None
    assert "__gate_zz" not in tree([result]), "setup: the reserved component is not emitted"
    assert verdict(result) == (True, ()), result.report.problems


# --- criterion 7 as amended (ec7966f): the fresh definition wins, its references are restored ------------------

PROBE_C0 = canvas(
    '{"id":"root","component":"Column","children":["card"]}',
    '{"id":"card","component":"Card","child":"t"}',
)


@pytest.mark.parametrize("card_on_c1", [True, False], ids=["card-resent", "card-omitted"])
def test_reviewer_a_probe_a_resent_parent_keeps_a_restored_child(card_on_c1: bool) -> None:
    """Criterion 7 as amended (reviewer A's probe, red at beda641): c0 is ``root[card]`` with ``card{child: t}`` and
    no ``t``, so ``card`` is deferred and ``root`` emitted as ``[]``. c1 re-sends ``root[card]`` and brings ``t``
    (and, in one arm, a ``card`` with ``weight: "heavy"``, which the catalog rejects). After c1 the renderer's
    ``root`` reaches ``card``, ``card`` reaches ``t``, and the call is complete."""
    comps = ['{"id":"root","component":"Column","children":["card"]}']
    if card_on_c1:
        comps.append('{"id":"card","component":"Card","child":"t","weight":"heavy"}')
    comps.append('{"id":"t","component":"Text","text":"inside the card"}')
    r0, r1 = gate_seq(PROBE_C0, canvas(*comps))
    got = tree([r0, r1])
    assert got["root"]["children"] == ["card"], got
    assert got.get("card", {}).get("child") == "t" and "t" in got, got
    assert verdict(r1)[0] is True, (verdict(r1), r1.report.problems)


# --- criterion 3: losses are named ------------------------------------------------------------------------------


def test_problems_names_each_lost_id_with_its_kind() -> None:
    """Criterion 3: one line per lost id naming the id and its kind; ``ok`` keeps its meaning (the surface without
    them validates and is emitted)."""
    last = gate_seq(LOSSES)[-1]
    assert last.surface is not None and last.report.ok
    _, lost = verdict(last)
    assert sorted(lost) == ["b", "l"]
    for cid, kind in (("b", "Text"), ("l", "List")):
        lines = [p for p in last.report.problems if names(p, cid) and names(p, kind)]
        assert lines, f"criterion 3: no problems line names {cid} ({kind}): {last.report.problems}"


def test_a_dangling_reference_is_named_but_not_lost() -> None:
    """Criterion 3: ``root`` references ``ghost``, which the model never defined: "reference pruned", not a loss;
    the call is complete."""
    last = gate_seq(canvas('{"id":"root","component":"Column","children":["a","ghost"]}', A))[-1]
    complete, lost = verdict(last)
    assert "ghost" not in lost
    assert any(
        names(p, "ghost") and "reference pruned" in p.lower() for p in last.report.problems
    ), last.report.problems
    assert complete is True


# --- criterion 7: restore never overwrites a fresh definition ----------------------------------------------------

#: canvas 1: ``r3`` references ``b3``, cut as a stub, so ``r3`` is emitted as ``[a3]`` and remembered as ``[a3, b3]``
RESTORE_1 = canvas(
    '{"id":"root","component":"Column","children":["r3"]}',
    '{"id":"r3","component":"Row","children":["a3","b3"]}',
    '{"id":"a3","component":"Text","text":"Travel:"}',
    '{"id":"b3","component":"',
)
RESTORE_1 = RESTORE_1[: RESTORE_1.index('{"id":"b3"') + len('{"id":"b3","component":"')]
#: canvas 2: the model re-sends ``r3`` as ``[a3]`` and moves ``b3`` under ``root``
RESTORE_2 = canvas(
    '{"id":"root","component":"Column","children":["r3","b3"]}',
    '{"id":"r3","component":"Row","children":["a3"]}',
    '{"id":"a3","component":"Text","text":"Travel:"}',
    '{"id":"b3","component":"Text","text":"180"}',
)


def test_premise_restore_repro_first_canvas() -> None:
    """Guard (holds at the base and after): canvas 1 emits ``r3`` pruned to ``[a3]``."""
    (r1,) = gate_seq(RESTORE_1)
    assert tree([r1])["r3"] == {"id": "r3", "component": "Row", "children": ["a3"]}


def test_restore_does_not_overwrite_the_fresh_definition() -> None:
    """Criterion 7 (mutation 8): reproduced at e78e10b, canvas 2 ends with a third message re-sending the stale
    ``r3: [a3, b3]`` after the model's fresh ``r3: [a3]``. After the fix the renderer ends with the fresh ``r3`` and
    canvas 2 emits 2 messages."""
    r1, r2 = gate_seq(RESTORE_1, RESTORE_2)
    assert r2.surface is not None
    assert tree([r1, r2])["r3"] == {"id": "r3", "component": "Row", "children": ["a3"]}
    assert len(r2.surface.messages) == 2, r2.surface.messages


# --- criterion 11: negatives (green at the base and after) -------------------------------------------------------


def rewritten(raw: str) -> list[Any]:
    """Oracle: the parsed canvas with every ``surfaceId`` set to the turn's (T-0010 A7), nothing else changed."""
    out = []
    for m in parse_concatenated_json(raw):
        m = dict(m)
        for kind in MESSAGE_KINDS:
            if isinstance(m.get(kind), Mapping):
                m[kind] = {**m[kind], "surfaceId": SID}
        out.append(m)
    return out


@pytest.mark.parametrize("name", ["VALID", "UI_EXAMPLE", "quiz"])
def test_an_all_valid_canvas_is_emitted_as_today_with_no_problems(name: str) -> None:
    """Criterion 11: an all-valid canvas whose root reaches everything gives the base's messages (the canvas with its
    surface id rewritten), ``repaired == 0`` and ``problems == ()``."""
    raw = {"VALID": VALID, "UI_EXAMPLE": UI_EXAMPLE, "quiz": QUIZ.read_text()}[name]
    (result,) = gate_seq(raw)
    assert result.surface is not None and result.report.ok
    assert result.repaired == 0
    assert result.report.problems == (), result.report.problems
    assert json.loads(json.dumps(list(result.surface.messages))) == rewritten(raw)


@pytest.mark.parametrize("name", ["VALID", "UI_EXAMPLE", "quiz"])
def test_an_all_valid_canvas_is_complete(name: str) -> None:
    """Criterion 4 on the same canvases: complete, nothing lost (the too-wide direction of mutations 4-5)."""
    raw = {"VALID": VALID, "UI_EXAMPLE": UI_EXAMPLE, "quiz": QUIZ.read_text()}[name]
    (result,) = gate_seq(raw)
    assert verdict(result) == (True, ())


def test_valid_components_of_the_desk_canvases_come_back_identical() -> None:
    """Criterion 11: on every strict-parseable desk canvas (the T-0127 fixture), every component the catalog accepts
    comes out of the component stage as the identical object."""
    validator = A2UIValidator()
    rows = [json.loads(line) for line in DESK.read_text().splitlines() if line.strip()]
    checked = 0
    for r in rows:
        if r["kind"] != "llm.commit" or r["data"]["segment"] != "ui":
            continue
        try:
            messages = parse_concatenated_json(r["data"]["text"])
        except json.JSONDecodeError:
            continue
        for m in messages:
            comps = (m.get("updateComponents") or {}).get("components")
            if not isinstance(comps, list):
                continue
            out, _ = component_repair_module.repair_components(comps, validator)
            for c in comps:
                if not validator.component_errors(c):
                    checked += 1
                    assert any(o is c for o in out), f"valid {c.get('id')} was not passed through"
    assert checked >= 50, f"premise: only {checked} valid components checked"
