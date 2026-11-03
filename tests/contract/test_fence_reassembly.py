"""A surface that needed two canvases is reassembled by the real gate. T-0052 group A, criterion 2.

Three real components, no fake in between: ``VllmClient`` reading a recorded SSE stream
(``tests/fixtures/llm/flight_canvases.json``, the human's turn `t6` of 2026-09-16), and ``UiGate``,
which has known how to morph a surface across canvases since T-0010 (`_restore_deferred`, `_prior`,
per-component pruning). What was missing is the client handing it the second canvas at all.

"Fake backend" here means no engine produced the text — the recorded canvases stand in for the
model, exactly as ``test_prompt_example_reaches_a_surface.py`` uses the prompt's own literal. The
``real`` parametrisation ``.claude/rules/tests.md`` asks for is criterion 16's live eval
(``tests/eval/test_a2ui_prompt_live.py``, ``gpu``), which asserts reassembly on what the model
actually emits; the task names that split.

Regime: CPU, n = 1 per row, no seed, no timing claimed.
"""

from __future__ import annotations

import pytest

from local_astra.conductor.llm_client import UI_OPEN, VllmClient
from local_astra.conductor.ui_gate import GateResult, UiGate
from local_astra.contracts.llm import Commit, Message, Prompt, TextPart
from tests.fakes.llm import fence_lifecycle, flight_canvases, sse_transport

pytestmark = pytest.mark.contract

FIX = flight_canvases()
REC = FIX["recorded"]
PROMPT = Prompt(
    messages=(Message(role="user", content=(TextPart(text="build me a flight booking ui"),)),),
    prefix_len=1,
)
FLIGHT = [(c["reasoning"], c["content"]) for c in FIX["canvases"]]


async def commits_of(canvases: list[tuple[str | None, str | None]], *, rid: str) -> list[Commit]:
    client = VllmClient(transport=sse_transport(canvases))
    return [c async for c in client.generate(PROMPT, rid, None)]


def ids_of(surface) -> list[str]:
    return [
        str(c.get("id"))
        for m in surface.messages
        if isinstance(m, dict) and isinstance(m.get("updateComponents"), dict)
        for c in m["updateComponents"]["components"]
    ]


def gate_all(commits: list[Commit], *, gate: UiGate, turn_id: str) -> list[GateResult]:
    """Every ``ui`` commit through the real gate, in ascending canvas order (criterion 2)."""
    results = []
    for commit in sorted(
        [c for c in commits if c.segment == "ui"], key=lambda c: (c.canvas_idx, commits.index(c))
    ):
        results.append(
            gate.gate(commit.text, turn_id=turn_id, canvas_idx=commit.canvas_idx, final=True)
        )
    return results


async def test_two_canvases_become_one_surface_with_components_from_both() -> None:
    """Criterion 2, by component id, never by count. **Two components to seven.**

    Measured on the real ``UiGate`` over this fixture (coordinator's re-run, 2026-09-16): canvas 1
    alone renders ``root`` and ``header`` and nothing else — that is the half interface the human
    saw — while the reassembled 1159-character body renders ``root, header, row1, row2, row3,
    divider, results``. So the five ids this row asserts are exactly the ones reassembly buys, and
    the baseline they are measured against is pinned by
    ``test_the_first_canvas_still_publishes_before_the_second_arrives`` below.

    ``date1``/``date2``/``passengers``/``search`` are deliberately **not** asserted: they are leaf
    inputs written without the properties the A2UI v0.9.1 catalog *requires* (``date1`` is
    ``{"id":"date1","component":"DateTimeInput"}`` with no properties at all), so they leave
    through the resample path rather than the render path in **both** runs. Reassembly does not
    change them, this task may not touch ``ui_gate.py``, and pinning them here would make a
    correct implementation red. See the fixture's ``gate_note`` and T-0052 ruling A7.
    """
    commits = await commits_of(FLIGHT, rid="t6-gen")
    gate = UiGate()
    results = gate_all(commits, gate=gate, turn_id="t6")

    assert results, "no ui commit reached the gate: the second canvas never became a surface"
    surfaces = [r.surface for r in results if r.surface is not None]
    assert surfaces, "the gate produced no surface at all from the carried fence. reports: " + repr(
        [(r.report.ok, list(r.report.problems)) for r in results]
    )
    final = ids_of(surfaces[-1])

    missing = [i for i in REC["rendered_only_after_reassembly"] if i not in final]
    assert not missing, (
        f"components {missing} render only once the second canvas is reassembled and are absent "
        f"from the final surface: the panel is still showing half an interface. final surface "
        f"ids: {final}"
    )
    assert len(final) > len(REC["rendered_from_canvas_1_alone"]), (
        "the reassembled surface must be strictly richer than the half one the human saw "
        f"({REC['rendered_from_canvas_1_alone']}); got {final}"
    )
    for first_canvas_id in ("root", "header"):
        assert first_canvas_id in final, (
            f"{first_canvas_id!r} came from canvas 1 and must survive the morph: reassembly must "
            f"extend the surface, not replace it. final surface ids: {final}"
        )


async def test_the_first_canvas_still_publishes_before_the_second_arrives() -> None:
    """The progressive property must survive the fix (T-0010 criteria 1–3, docs/12 §1.1).

    Not pending: it is green today and its job is to *stay* green. A client that made the panel
    correct by holding the whole surface until the fence closed would pass every criterion-2 row
    and make the first paint arrive a canvas later — on this turn, seconds later. So: the first
    ``ui`` commit is the canvas the fence opened in, it carries only that canvas's text, and the
    gate turns it into a surface on its own.

    The three numbers on the right are the session log's own ``ui.gate`` record (seq 2181): 409
    raw characters, 4 repairs, a surface emitted. If they drift, either the fixture is no longer
    that turn or the gate changed underneath it — both worth a failure. The component list is the
    other half of criterion 2's arithmetic: exactly ``root`` and ``header`` here, seven ids once
    the second canvas is reassembled.
    """
    commits = await commits_of(FLIGHT, rid="t6-gen")
    ui = [c for c in commits if c.segment == "ui"]
    assert ui, "the recorded turn opens a fence in canvas 1; no ui commit is a broken replay"

    first = ui[0]
    assert first.canvas_idx == 1, (
        f"the first ui commit must be published for the canvas the fence opened in, not held "
        f"until the fence closes; got canvas {first.canvas_idx}"
    )
    assert REC["json_tail_canvas2"] not in first.text, (
        "the first ui commit already contains canvas 2's JSON: the client is buffering the whole "
        "surface and the panel paints a canvas late"
    )
    assert len(first.text) == REC["ui_gate_canvas1"]["raw_len"] == 409, (
        f"the log recorded 409 raw characters for canvas 1; this commit carries {len(first.text)}"
    )

    result = UiGate().gate(first.text, turn_id="t6", canvas_idx=1, final=True)
    assert result.report.ok and result.surface is not None, (
        f"half a surface must still render: {list(result.report.problems)}"
    )
    assert result.repaired == REC["ui_gate_canvas1"]["repairs"], (
        f"the log recorded {REC['ui_gate_canvas1']['repairs']} repairs for this canvas; the gate "
        f"made {result.repaired}"
    )
    assert ids_of(result.surface) == REC["rendered_from_canvas_1_alone"] == ["root", "header"], (
        "the baseline the fix is measured against: canvas 1 on its own renders exactly two "
        f"components, root and header. got {ids_of(result.surface)}"
    )


async def test_a_surface_abandoned_at_the_bound_leaves_the_gate_consistent() -> None:
    """Adversarial: an abandoned carry must not poison the next turn (T-0010 A3/A7).

    Ten canvases that never balance, so the carry is abandoned at the bound; then a new turn with
    an ordinary little surface. The next turn must mount its own surface id and must not inherit a
    single component of the one that was abandoned — the gate keeps per-turn state, and a client
    that abandons a carry without letting the turn end is how that state leaks.
    """
    junk = '{"id":"ghost","component":"Column","children":["x' + "y" * 40 + '","kids":['
    endless: list[tuple[str | None, str | None]] = [(None, UI_OPEN + junk)]
    endless += [(None, junk) for _ in range(9)]
    endless += [(None, "And that is the table.")]

    abandoned = await commits_of(endless, rid="t1-gen")
    carried = sorted({c.canvas_idx for c in abandoned if c.segment == "ui"})
    assert len(carried) > 1, (
        "setup: the carry must actually span canvases before it can be abandoned; with no carry "
        f"at all this row proves nothing. ui canvases: {carried}"
    )
    lifecycles = [fence_lifecycle(c) for c in abandoned if c.segment == "ui"]
    assert len(set(lifecycles)) >= 2, (
        f"setup: the abandonment must be announced on the commit, not silent: {lifecycles}"
    )

    gate = UiGate()
    gate_all(abandoned, gate=gate, turn_id="t1")

    good = f'{UI_OPEN}{{"version":"v0.9.1","createSurface":{{"surfaceId":"s","catalogId":"https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"}}}}\n{{"version":"v0.9.1","updateComponents":{{"surfaceId":"s","components":[{{"id":"root","component":"Text","text":"hello"}}]}}}}'
    next_turn = await commits_of([(None, good)], rid="t2-gen")
    results = gate_all(next_turn, gate=gate, turn_id="t2")
    surfaces = [r.surface for r in results if r.surface is not None]
    assert surfaces, "the turn after an abandoned carry must still produce a surface: " + repr(
        [(r.report.ok, list(r.report.problems)) for r in results]
    )
    assert surfaces[-1].surface_id == "turn-t2", (
        f"T-0010 A3/A7: turn n+1 mounts its own surface id; got {surfaces[-1].surface_id!r}"
    )
    assert "ghost" not in ids_of(surfaces[-1]), (
        f"a component of the abandoned surface reappeared in the next turn: {ids_of(surfaces[-1])}"
    )
