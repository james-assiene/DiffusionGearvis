"""T-0044 criterion 2, end to end over the *text in the prompt* (contract, fake backend).

The block's example is the one piece of the prompt whose meaning is not a matter of wording: it
is a payload that the running system either turns into a surface or does not. So it is pushed
through the two real components it will meet on a live turn —

    llm_client.split_segments(None, UI_OPEN + UI_EXAMPLE + UI_CLOSE, False)   ->  one ``ui`` piece
    UiGate().gate(that piece, turn_id=…, canvas_idx=0, final=True)            ->  a surface

— with no fake in between: ``split_segments`` is pure and ``UiGate`` needs nothing but the
vendored schemas. "Fake backend" here means no engine produced the text: the prompt's own literal
stands in for the model's output. The ``real`` parametrisation required by
``.claude/rules/tests.md`` is criterion 9's live eval (``tests/eval/test_a2ui_prompt_live.py``,
marked ``gpu``), where the same assertions are made about what the model actually emits — the
task names that split explicitly.

Why this is not the same test as ``tests/unit/test_ui_instructions.py``: there the example is
judged by the validator alone. Here it also has to survive the *fence*, which is where a
plausible example ("here is a surface: {…}", a payload wrapped in a markdown code block, a
trailing comma after the closing fence) quietly turns into ``answer`` text and is spoken aloud
instead of shown.
"""

from __future__ import annotations

import pytest

from local_astra.conductor.llm_client import UI_CLOSE, UI_OPEN, VllmClient, split_segments
from local_astra.conductor.ui_gate import UiGate
from local_astra.contracts.llm import Message, Prompt, TextPart
from local_astra.ui import parse_concatenated_json
from tests.fakes.llm import carried_ui_chars, sse_transport

try:  # T-0044 creates it; a missing import must not error at collection
    from local_astra.conductor import ui_instructions as _ui_instructions
except ImportError:  # pragma: no cover - until T-0044 lands
    _ui_instructions = None

pytestmark = pytest.mark.contract


def example() -> str:
    if _ui_instructions is None or getattr(_ui_instructions, "UI_EXAMPLE", None) is None:
        raise NotImplementedError(
            "local_astra.conductor.ui_instructions.UI_EXAMPLE does not exist: the prompt still "
            "shows the model no A2UI example (docs/tasks/T-0044.md criterion 2)"
        )
    return str(_ui_instructions.UI_EXAMPLE)


def _payload_components(payload: str) -> list[dict]:
    """The components the example *writes*, straight from the text (an independent reading)."""
    return [
        c
        for message in parse_concatenated_json(payload)
        if isinstance(message, dict) and isinstance(message.get("updateComponents"), dict)
        for c in message["updateComponents"]["components"]
    ]


def components(surface) -> list[dict]:
    return [
        c
        for m in surface.messages
        if isinstance(m, dict) and isinstance(m.get("updateComponents"), dict)
        for c in m["updateComponents"]["components"]
    ]


def test_the_fenced_example_is_exactly_one_ui_segment():
    """Criterion 2/4: the example, fenced, is one ``ui`` piece and produces no answer text.

    Every piece of answer text here would be spoken: ``split_segments`` sends the content channel
    to ``answer`` outside the fence, and only ``Commit(segment="answer")`` reaches the speech gate
    (``.claude/rules/conductor.md``). An example that leaks one brace into ``answer`` is an
    example that makes the assistant read JSON out loud.
    """
    pieces = split_segments(None, UI_OPEN + example() + UI_CLOSE, False)

    assert [segment for segment, _, _ in pieces] == ["ui"], (
        f"expected exactly one ui piece; got {[(s, t[:40]) for s, t, _ in pieces]}"
    )
    _, text, in_ui_after = pieces[0]
    assert text == example(), (
        "the ui piece must carry the example verbatim (the fence markers belong to the piece "
        f"they delimit); got {text[:80]!r}"
    )
    assert in_ui_after is False, (
        "the fence is closed by the end of the canvas: in_ui_after must be False, otherwise the "
        "client is left believing a surface is still open (llm_client M7)"
    )
    assert not any(s == "answer" for s, _, _ in pieces), "no part of the example may be voiced"


def test_the_example_becomes_a_surface_through_the_real_gate():
    """Criterion 2: the ``ui`` piece is turned into a surface by the *real* ``UiGate``.

    ``ui_repairs == 0`` is the strong half: the fixer only logs edits for input it had to repair
    (T-0010 A5), so a nonzero count means the example is JSON the model would have to be lucky to
    reproduce. ``resample == ()`` means no component was rejected by the catalog.
    """
    gate = UiGate()
    (piece,) = split_segments(None, UI_OPEN + example() + UI_CLOSE, False)
    result = gate.gate(piece[1], turn_id="t0044", canvas_idx=0, final=True)

    assert result.report.ok, (
        "criterion 2: the gate rejected the prompt's own example — the block would be teaching "
        f"the shape that yields ui_valid: false.\nproblems: {list(result.report.problems)}\n"
        f"schema errors: {[str(e) for e in result.report.schema_errors]}"
    )
    assert result.surface is not None, "report.ok with no surface is a gate bug, not an example"
    assert result.repaired == 0, (
        f"the example needed {result.repaired} fixer edit(s): it is not JSON the model can copy "
        "verbatim and have accepted (T-0010 A5 counts edits only for input it had to repair)"
    )
    assert result.resample == (), (
        f"the example contains components the catalog rejects: {list(result.resample)}"
    )
    emitted = components(result.surface)
    assert len(emitted) >= 1, (
        "a surface with no component renders an empty box; the example must show at least one"
    )
    written = [c.get("id") for c in _payload_components(example())]
    assert [c.get("id") for c in emitted] == written, (
        "the gate emitted a different set of components than the example wrote: a component the "
        "gate prunes (a dangling child, a container whose required child never arrived — T-0010 "
        f"A9) is a shape the block must not teach. wrote {written}, emitted "
        f"{[c.get('id') for c in emitted]}"
    )
    assert result.surface.surface_id == "turn-t0044", (
        f"T-0010 A3/A7: one surface id per turn, rewritten by the gate; got "
        f"{result.surface.surface_id!r}"
    )


async def test_a_canvas_that_ends_inside_the_fence_carries_into_the_next_one():
    """The straddle, after T-0052 (criteria 13/14, T-0044 ambiguity Q2 closed in that direction).

    **Withdrawn row, named reason:** this file used to hold
    ``test_a_canvas_that_ends_inside_the_fence_carries_nothing_into_the_next_one``, which pinned
    T-0005 amendments **B7/M7** — one chunk is one canvas, the fence state is thrown away between
    chunks, a lone ``</a2ui>`` in the next chunk is stripped as markup. Those amendments are
    superseded by T-0052 **D1**: on 2026-09-16 the model wrote a surface that did not fit in one
    canvas, the tail was classified ``answer``, and Pocket TTS read 807 characters of A2UI JSON
    out loud. The row is not "adjusted to pass" — it is replaced by the opposite assertion, which
    is what the evidence decided.

    The prompt's own example is cut in half *inside* its component array (bracket depth is above
    zero at the cut, so no structural close can fire there) and delivered as two canvases of one
    request, with the closing fence in the second. What must come out: two ``ui`` commits, not one
    byte of the payload in the answer channel, the example reassembled verbatim, and a surface
    with exactly the components the example writes.
    """
    payload = example()
    anchor = payload.find('"components"')
    assert anchor > 0, (
        "this row needs a cut point inside the example's component array; the example no longer "
        "contains a 'components' key, so the cut has to be chosen again"
    )
    cut = anchor + len('"components":[') + 4
    head, tail = payload[:cut], payload[cut:]

    prompt = Prompt(
        messages=(Message(role="user", content=(TextPart(text="show me"),)),), prefix_len=1
    )
    client = VllmClient(
        transport=sse_transport([(None, UI_OPEN + head), (None, tail + UI_CLOSE + " There it is.")])
    )
    commits = [c async for c in client.generate(prompt, "t0052-straddle", None)]

    ui = [c for c in commits if c.segment == "ui"]
    answers = [c.text for c in commits if c.segment == "answer"]
    assert len(ui) == 2, (
        "one ui commit per canvas of the straddling fence (progressive render): got "
        + repr([(c.canvas_idx, c.segment, c.text[:40]) for c in commits])
    )
    assert answers == [" There it is."], (
        f"only the prose after the closing fence may be spoken; got {answers}"
    )
    assert carried_ui_chars([c.text for c in ui]) == len(payload), (
        f"the example is {len(payload)} characters; the carry reassembled "
        f"{carried_ui_chars([c.text for c in ui])}"
    )

    gate = UiGate()
    results = [gate.gate(c.text, turn_id="t0052", canvas_idx=c.canvas_idx, final=True) for c in ui]
    surfaces = [r.surface for r in results if r.surface is not None]
    assert surfaces, "a fence that straddled two canvases produced no surface: " + repr(
        [(r.report.ok, list(r.report.problems)) for r in results]
    )
    written = [c.get("id") for c in _payload_components(payload)]
    assert [c.get("id") for c in components(surfaces[-1])] == written, (
        f"the reassembled surface must be the example: wrote {written}, emitted "
        f"{[c.get('id') for c in components(surfaces[-1])]}"
    )
