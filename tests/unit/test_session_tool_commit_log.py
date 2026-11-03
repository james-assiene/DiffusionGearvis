"""T-0092b criterion 4: the request path with ``tools = dry_run`` — the ``tool`` commit is logged, kept, never submitted.

Over the fake LLM scripting a ``tool`` commit (``FakeLlm.tool_commit``, T-0088's scriptable shape), with the kit of
``tests/unit/_tools_switch_support.py`` injected:

* the request is tools-on (criterion 4's first line);
* the ``tool`` commit's ``llm.commit`` record carries ``stop_id``; the records of every other segment carry no
  ``stop_id`` key and exactly today's eight keys (byte-identical to the base);
* it is never submitted as ``LlmCommit``: the machine never sees it, the speech gate never voices it, and
  ``canvases`` counts only the submitted commits (2 here, counted by hand: one thought, one answer);
* ``LlmDone`` carries the judged call (moving pin: T-0092c criterion 1 superseded "no tool in this task").

"Kept for the turn" has no observable surface until T-0092c reads it; it is not pinned here (see the report).
Regime: CPU over fakes, ``ScriptedClock``, n = 1.
"""

from __future__ import annotations

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.turn import TurnState
from tests.fakes.llm import FakeLlm
from tests.unit import _tools_switch_support as T
from tests.unit._session_log_rig import StepClock, of_kind, read_records
from tests.unit.test_session_loop import answer, thought

#: The keys of an ``llm.commit`` record at the base a502871 (session.py ``_run_generate``), written out.
BASE_COMMIT_KEYS = {
    "request_id",
    "attempt_id",
    "canvas_idx",
    "segment",
    "text",
    "thought_open",
    "fence_close",
    "fence_continued",
}
CALL = "<|tool_call>call:click_at{x:500,y:250}<tool_call|>"


async def run_tool_turn(tmp_path, *, stop_id: int | None = 50):
    from local_astra.telemetry.session_log import SessionLog

    script = [
        thought("the user wants the button"),
        answer("I will click it. "),
        FakeLlm.tool_commit(CALL, stop_id=stop_id, canvas_idx=1),
    ]
    log = SessionLog(tmp_path / "logs", clock=StepClock())
    rig = T.session_rig(tmp_path, kit=T.make_kit(), script=script, session_log=log)
    await rig.session.start()
    try:
        await rig.ask("click the button")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and rig.lines(), "the turn never ended"
        )
    finally:
        await rig.session.aclose()
    return rig, read_records(log.path)


@pytest.mark.parametrize("stop_id", [50, 106, None])
async def test_the_tool_commit_is_logged_with_its_stop_id(tmp_path, stop_id):
    """Criterion 4: one ``llm.commit`` record for the ``tool`` commit, its text verbatim, ``stop_id`` as the
    ``Commit`` carried it (None for a length cut: the key is present, the value null)."""
    _rig, records = await run_tool_turn(tmp_path, stop_id=stop_id)
    tools = [
        r["data"] for r in of_kind(records, "llm.commit") if r["data"].get("segment") == "tool"
    ]
    assert len(tools) == 1, f"T-0092b criterion 4: the tool commit is logged once; got {tools}"
    assert tools[0]["text"] == CALL
    assert "stop_id" in tools[0] and tools[0]["stop_id"] == stop_id, (
        f"T-0092b criterion 4: the tool commit's record carries stop_id={stop_id!r}; it has "
        f"{tools[0].get('stop_id', '<absent>')!r}"
    )
    assert set(tools[0]) == BASE_COMMIT_KEYS | {"stop_id"}, sorted(tools[0])


async def test_other_segments_records_carry_no_stop_id_and_todays_keys(tmp_path):
    """Criterion 4: ``llm.commit`` records of every other segment are byte-identical in shape to today's."""
    _rig, records = await run_tool_turn(tmp_path)
    others = [
        r["data"] for r in of_kind(records, "llm.commit") if r["data"].get("segment") != "tool"
    ]
    assert {o["segment"] for o in others} == {"thought", "answer"}, (
        "premise: both other segments were logged"
    )
    for data in others:
        assert set(data) == BASE_COMMIT_KEYS, (
            f"T-0092b criterion 4: a {data['segment']} record has exactly today's keys; it has {sorted(data)}"
        )


async def test_the_tool_commit_never_reaches_the_machine_the_gate_or_canvases(tmp_path):
    """Criterion 4: not submitted as ``LlmCommit``; never voiced; ``canvases`` == 2 (thought + answer)."""
    rig, _ = await run_tool_turn(tmp_path)
    commits = rig.machine.events(E.LlmCommit)
    assert [c.segment for c in commits] == ["thought", "answer"], (
        f"T-0092b criterion 4: only the thought and answer commits reach the machine; it saw "
        f"{[(c.segment, c.text) for c in commits]}"
    )
    spoken = " ".join(rig.tts.texts())
    assert "call:" not in spoken and "<|tool_call>" not in spoken and "I will click it" in spoken, (
        spoken
    )
    record = rig.records()[-1]
    assert record.canvases == 2, (
        f"T-0092b criterion 4: canvases counts the two submitted commits; {record.canvases}"
    )
    dones = rig.machine.events(E.LlmDone)
    # Moving pin (T-0092c criterion 1 supersedes T-0092b's "LlmDone carries no tool in this task"): the kept tool
    # commit is now judged, and pass 1's LlmDone carries it.
    tools_ = [
        (d.tool.call_id, d.tool.status, d.tool.name, d.tool.arguments)
        for d in dones
        if d.tool is not None
    ]
    assert tools_ == [("t1-call", "valid", "click_at", {"x": 500, "y": 250})], (
        f"T-0092c criterion 1: the kept tool commit is judged and carried by LlmDone; got {tools_}"
    )
    assert record.outcome == "spoken"


async def test_the_dry_run_request_is_tools_on(tmp_path):
    """Criterion 4: ``_run_generate``'s request carries the tools (the fake logs ``Prompt.tools``)."""
    rig, _records = await run_tool_turn(tmp_path)
    assert rig.llm.tools_log and all(t is not None for t in rig.llm.tools_log), (
        f"T-0092b criterion 4: with dry_run the turn's generate is tools-on; tools_log={rig.llm.tools_log}"
    )
    assert rig.llm.bodies and rig.llm.bodies[0].get("tool_choice") == "none", (
        "the real body is arm (b)"
    )
