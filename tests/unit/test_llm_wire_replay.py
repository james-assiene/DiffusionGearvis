"""T-0088 criterion 8: the 24 recorded arm-(b)/(e) streams replayed through ``VllmClient`` in id mode.

The streams are T-0087's own wire, verbatim (``tests/fixtures/llm/t0087_streams.jsonl``), served over
``httpx.MockTransport``; the detokeniser is T-0088a's subset of the pinned file. The oracles are the fields the
engine itself served on the same chunks (``reasoning``, ``content``, ``stop_reason``) and ``native.parse_call``.
The served fields are oracles only: the client must reproduce them from the ids (a client that copies ``content``
passes these rows, which is why ``test_llm_id_mode.py::test_served_fields_are_never_read`` exists).

The last row is the tools-off half (criterion 10, green by design): the same bytes read with tools off give main's
reading of ``content``/``reasoning``.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_astra.conductor.llm_client import VllmClient
from local_astra.toolcall import native
from local_astra.toolcall.score import ParseFailure
from tests.fakes import vllm_sse as V
from tests.unit import _llm_id_support as S

ROWS = S.recorded()
IDS = [f"{r['arm']}/{r['case']}" for r in ROWS]
TOOL_CASES = {r["case"] for r in ROWS if not r["case"].startswith("chat-")}


def test_the_fixture_is_the_24_recorded_streams() -> None:
    assert len(ROWS) == 24
    assert sorted({r["arm"] for r in ROWS}) == ["b", "e"]
    assert sum(1 for r in ROWS if r["case"] in TOOL_CASES) == 20


async def _replay(row: dict[str, Any]) -> list[Any]:
    wire = V.Wire(streams=[row["lines"]])
    cl = S.client(wire.transport())
    got = await S.collect(cl, S.tools_prompt())
    await cl.aclose()
    return got


@pytest.mark.parametrize("row", ROWS, ids=IDS)
async def test_replay_matches_what_the_engine_served(row: dict[str, Any]) -> None:
    reasoning, content, stop_reason = S.served(row)
    got = await _replay(row)
    assert S.texts(got, "thought") == reasoning, "criterion 8: thought == served reasoning"
    assert S.texts(got, "answer") == content, "criterion 8: answer == served content"
    tools_ = S.tool_commits(got)
    if row["case"] in TOOL_CASES:
        assert len(tools_) == 1, "criterion 8: a tool commit on every tool stream"
        assert tools_[0].text.startswith("<|tool_call>")
        parsed = native.parse_call(tools_[0].text.removeprefix("<|tool_call>"))
        assert not isinstance(parsed, ParseFailure), f"criterion 8: parse_call refused: {parsed}"
        assert S.stop_id(tools_[0]) == stop_reason == S.TOOL_RESPONSE
    else:
        assert tools_ == [], "criterion 8: no tool commit on a chat stream"
        assert stop_reason == S.TURN_CLOSE


async def test_replay_counts() -> None:
    """The aggregate the task states: 24/24 thought and answer, 20/20 tool commits that parse, 24/24 stop ids.

    The stop of a chat stream is not carried by any commit; the per-row test checks the served value."""
    agree = parsed = 0
    for row in ROWS:
        reasoning, content, stop = S.served(row)
        got = await _replay(row)
        agree += S.texts(got, "thought") == reasoning and S.texts(got, "answer") == content
        for t in S.tool_commits(got):
            ok = not isinstance(
                native.parse_call(t.text.removeprefix("<|tool_call>")), ParseFailure
            )
            parsed += ok and S.stop_id(t) == stop
    assert (agree, parsed) == (24, 20)


@pytest.mark.parametrize("row", ROWS, ids=IDS)
async def test_tools_off_reads_the_same_bytes_from_content_and_reasoning(
    row: dict[str, Any],
) -> None:
    """Criterion 10 (green by design): without tools the client reads the served fields exactly as main does."""
    reasoning, content, _ = S.served(row)
    wire = V.Wire(streams=[row["lines"]])
    cl = VllmClient(transport=wire.transport())
    got = [c async for c in cl.generate(S.make_prompt(), "r1")]
    await cl.aclose()
    assert [(c.segment, c.text) for c in got] == [
        *([("thought", reasoning)] if reasoning else []),
        *([("answer", content)] if content else []),
    ]
