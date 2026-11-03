"""T-0171 criteria 5-11 and 13: the session resends once when the engine answers with its empty "retryable" response.

The failure is ``tests/fakes/llm.engine_empty_error``: the exception the **real** ``VllmClient`` raises, in id mode,
for ``tests/fakes/vllm_sse.engine_empty`` (HTTP 200, the role chunk, ``[DONE]``; the T-0125 smoke's lost generate).
Passed itself as an entry of ``fail_generates`` / ``fail_steps``, it is called with the engine id of the request it
ends, so its message names that attempt. Before T-0171 it is a plain ``LlmWireError``; after, an ``LlmEngineEmpty``.

The rig is ``_tool_turn_support.tool_rig``: tools ``dry_run`` (so id mode), the screen frame attached (an image
prompt), the HUD publisher a ``Recorder``, the session log a real ``SessionLog`` read back after ``aclose``.

**What the rows read** (all from the task file, verbatim where it is verbatim):

- the retry note: ``kind == "note"``, ``what == RETRY_NOTE``, keys ``request_id``, ``failed_attempt_id``,
  ``attempt_id``, ``phase`` (``"generate"`` or ``"tool_step"``; ``step`` too for a step), no prompt text;
- one ``llm.request`` record per attempt (``attempt_id`` = the engine id the fake was handed);
- the resend: the body the real ``VllmClient.request_body`` builds (``FakeLlm.bodies``) equals the first one once the
  engine request id is taken out, and the ``Prompt`` handed to the backend is equal;
- the turn line's ``extras["llm_empty_retries"] == 1`` on a retried turn, the key absent otherwise;
- the error path (criteria 7-8): an ``error`` session-log record for the machine id, one ``LlmError`` to the machine,
  a ``HudError``.

**Which rows were pending** (activated at the start of ``/implement T-0171``). The rows that need the retry (criteria 5, 6, 7, 9 "during the retry", 10, 11, and the
over-long independence of 13). The negative rows (criterion 8, criterion 9 "between the failure and the resend", the
out-of-scope paths of criterion 13, and criterion 12) hold on the base already and are active, so they guard the
implementation from its first commit; each says which too-wide retry it would catch.

**Choices the task leaves open, fixed here** (listed in the test-author's report):

- "abort between the failure and the resend" is staged by the failing request itself: its error factory submits a
  ``UserTakeover`` and awaits ``Session.abort_request(machine id)`` before the error is raised, so the abort has fully
  run (engine id released, ``LlmAbort`` reached the engine) when the session sees the failure. "No resend" means the
  backend's ``generate`` is not called again (the fake records the call where the real client builds the body), and
  no ``llm.request`` record for a second attempt is written;
- "the denoising view stays real" is observed through a real ``CanvasRelay`` fed by the fake LLM: each commit the fake
  yields is preceded on the tap by one denoise frame and one commit frame of that request (what the plugin does for a
  canvas it committed). An attempt that yields nothing puts nothing on the tap, as the engine's empty response does.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, n = 1 per row, deterministic.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_astra.conductor import llm_client
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit, ImagePart, LlmOptions, Prompt
from local_astra.contracts.memory import Episode
from local_astra.contracts.perception import Source
from local_astra.contracts.turn import TurnState
from local_astra.memory.captioner import IdleCaptioner
from tests import _git_range as gr
from tests.fakes.canvas_tap import (
    StubPieces,
    denoise,
    optional_piece_kwargs,
    pack,
)
from tests.fakes.canvas_tap import (
    commit as tap_commit,
)
from tests.fakes.llm import engine_empty_error, http_error, is_caption_prompt
from tests.fakes.memory import FakeMemory
from tests.unit._tool_turn_support import (
    REPAIR_ID,
    REPAIRABLE_CLICK,
    REPAIRED_CLICK,
    VALID_CLICK,
    ToolLlm,
    answer,
    assert_concurrency,
    thought,
    tool,
    tool_rig,
)
from tests.unit.test_session_over_long_prompt import history_messages, two_turns_of_history
from tests.unit.test_session_tool_loop import LOOP, PREFIX, QUESTION, REPORT, generates_for

REPO = Path(__file__).resolve().parents[2]
#: Criterion 12's base: main at 3e4d45b7, merged into this branch at b99d4bad (T-0170 legitimately added
#: ``Commit.implicit_thought`` to ``contracts/llm.py`` after T-0171 branched from b372edaa). A literal, never a moving
#: ref (tests.md).
BASE_SHA = "3e4d45b7"

#: criterion 5, verbatim
RETRY_NOTE = "the engine returned an empty retryable response; retrying once"
NOTE_KEYS = frozenset({"what", "request_id", "failed_attempt_id", "attempt_id", "phase"})
EXTRA = "llm_empty_retries"
#: what ``generate`` raises today for an id-mode stream that ends without a finish_reason (T-0088 criterion 5)
TRUNCATED = "{rid}: the stream ended without a finish_reason"

ANSWER = "The page shows a login form. "
STEP1 = "t1-gen"
STEP2 = "t1-gen-s2"


# --- reading the rig ------------------------------------------------------------------------------------------


def engine_ids(rig: Any, machine_id: str) -> list[str]:
    return [rid for rid, _ in generates_for(rig, machine_id)]


def prompts_of(rig: Any, machine_id: str) -> list[Prompt]:
    return [p for _, p in generates_for(rig, machine_id)]


def options_of(rig: Any, engine_id: str) -> LlmOptions:
    found = [o for rid, _, o in rig.llm.generates if rid == engine_id]
    assert len(found) == 1, (engine_id, found)
    return found[0]


def _strip_request_id(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _strip_request_id(v) for k, v in value.items() if k != "astra_request_id"}
    if isinstance(value, list):
        return [_strip_request_id(v) for v in value]
    return value


def body_of(rig: Any, engine_id: str) -> dict[str, Any]:
    """The body the real client would send for ``engine_id``, with the engine request id taken out (and checked to
    appear nowhere else in it)."""
    found = [b for b in rig.llm.bodies if engine_id in json.dumps(b)]
    assert len(found) == 1, f"one body names {engine_id}; got {len(found)}"
    stripped = _strip_request_id(copy.deepcopy(found[0]))
    assert engine_id not in json.dumps(stripped), "the engine id is only the request id"
    return stripped


def images(prompt: Prompt) -> list[tuple[str, str | None]]:
    return [
        (p.url, p.phash) for m in prompt.messages for p in m.content if isinstance(p, ImagePart)
    ]


def records(rig: Any) -> list[dict[str, Any]]:
    return rig.log_records


def index_of(rig: Any, kind: str, pred: Any) -> list[int]:
    return [i for i, r in enumerate(records(rig)) if r["kind"] == kind and pred(r["data"])]


def retry_notes(rig: Any) -> list[dict[str, Any]]:
    return [d for d in rig.kind("note") if d.get("what") == RETRY_NOTE]


def requests_for(rig: Any, machine_id: str) -> list[dict[str, Any]]:
    return [d for d in rig.kind("llm.request") if d.get("request_id") == machine_id]


def error_records(rig: Any, machine_id: str) -> list[dict[str, Any]]:
    return [d for d in rig.kind("error") if d.get("request_id") == machine_id]


def llm_errors(rig: Any, machine_id: str) -> list[Any]:
    return [e for e in rig.machine.events(E.LlmError) if e.request_id == machine_id]


def plain_wire_error(rid: str) -> BaseException:
    """The plain ``LlmWireError`` the client raises today for a stream truncated after ids (criterion 2)."""
    return llm_client.LlmWireError(TRUNCATED.format(rid=rid))


def transport_error(rid: str) -> BaseException:
    return httpx.RemoteProtocolError(
        f"{rid}: peer closed connection without sending complete message body"
    )


def assert_one_retry_note(
    rig: Any, machine_id: str, failed: str, retry: str, phase: str
) -> dict[str, Any]:
    found = retry_notes(rig)
    assert len(found) == 1, f"T-0171 criterion 5: one retry note; got {found}"
    note = found[0]
    missing = NOTE_KEYS - set(note)
    assert not missing, f"T-0171 criterion 5: the note lacks {sorted(missing)}: {note}"
    assert (note["request_id"], note["failed_attempt_id"], note["attempt_id"], note["phase"]) == (
        machine_id,
        failed,
        retry,
        phase,
    ), f"T-0171 criterion 5: the note names the machine id, both attempts and the phase; got {note}"
    assert QUESTION not in json.dumps(note) and "screen" not in json.dumps(note), (
        f"T-0171 implementation notes: ids and phase only in the note, no prompt text: {note}"
    )
    return note


def assert_requests_logged(rig: Any, machine_id: str, failed: str, retry: str) -> None:
    """One ``llm.request`` per attempt, the retry's written after the failed one and before any commit of the retry."""
    reqs = requests_for(rig, machine_id)
    assert [r.get("attempt_id") for r in reqs] == [failed, retry], (
        f"T-0171 criterion 5: a new llm.request record for the resend; got {[r.get('attempt_id') for r in reqs]}"
    )
    first_req, retry_req = index_of(rig, "llm.request", lambda d: d.get("request_id") == machine_id)
    note = index_of(rig, "note", lambda d: d.get("what") == RETRY_NOTE)
    commits = index_of(rig, "llm.commit", lambda d: d.get("attempt_id") == retry)
    assert commits, "premise: the retry's commits are logged"
    assert first_req < note[0] < commits[0] and retry_req < commits[0], (
        "T-0171 criterion 5: the note and the retry's llm.request are written before the retry's first commit"
    )
    failed_commits = index_of(rig, "llm.commit", lambda d: d.get("attempt_id") == failed)
    assert failed_commits == [], "premise: the failed attempt committed nothing"


# =========================================================================== criterion 5


async def test_step1_engine_empty_is_retried_once_with_same_prompt(tmp_path: Path) -> None:
    """Criterion 5 (and 10): the step-1 generate ends empty-retryable with nothing delivered -> the session releases
    the attempt, logs the note and a new ``llm.request``, resends the identical request under a fresh engine id once
    the first is gone, and the turn is answered from the retry's commits with no error anywhere. A second, ordinary
    turn carries no ``llm_empty_retries`` key."""
    script = [thought("Looking at the screen.", 0), answer(ANSWER, 1)]
    async with tool_rig(tmp_path, script=script) as rig:
        rig.llm.fail_generates(engine_empty_error, None)
        await rig.ask(QUESTION)
        await rig.turn_done()
        ids = engine_ids(rig, STEP1)
        spoken = rig.spoken_sources()
        hud_errors = rig.commands(C.HudError)
        dones = rig.dones(STEP1)
        rig.llm.script = [answer("Nothing else. ")]
        await rig.ask("anything else")
        await rig.turn_done("the second turn never ended")
    assert len(ids) == 2, (
        f"T-0171 criterion 5: the empty retryable response is resent once (2 requests for {STEP1}); got {ids}"
    )
    failed, retry = ids
    assert failed != retry, "T-0171 criterion 5: the resend carries a fresh attempt id"
    first, second = prompts_of(rig, STEP1)
    assert images(first) and images(first) == images(second), (
        "T-0171 criterion 5: the same ImagePart data and phash are resent"
    )
    assert first == second, (
        "T-0171 criterion 5: the same Prompt is resent (not rebuilt with a new frame)"
    )
    assert options_of(rig, failed) == options_of(rig, retry), "criterion 5: the same options"
    assert body_of(rig, failed) == body_of(rig, retry), (
        "T-0171 criterion 5: the body equals the first except for the attempt id"
    )
    assert_one_retry_note(rig, STEP1, failed, retry, "generate")
    assert_requests_logged(rig, STEP1, failed, retry)
    assert ANSWER.strip() in spoken, (
        f"T-0171 criterion 5: the turn is answered from the retry; spoke {spoken!r}"
    )
    assert len(dones) == 1 and dones[0].request_id == STEP1, dones
    assert llm_errors(rig, STEP1) == [], "criterion 5: no LlmError from the first attempt"
    assert error_records(rig, STEP1) == [], "criterion 5: no error record from the first attempt"
    assert hud_errors == [], "criterion 5: nothing failed on the page"
    record = rig.record("t1")
    assert record.outcome == "spoken", record.outcome
    assert record.extras.get(EXTRA) == 1, (
        f"T-0171 criterion 5: extras[{EXTRA!r}] == 1; got {record.extras}"
    )
    assert EXTRA not in rig.record("t2").extras, (
        "criterion 5: the key is absent when there is no retry"
    )
    # criterion 10: one live generate at a time; the failed attempt was over before the resend started
    assert_concurrency(rig)
    starts = [rid for what, rid, _ in rig.llm.calls if what == "generate" and rid in ids]
    assert starts == [failed, retry], starts


# =========================================================================== criterion 6


async def test_tool_step_engine_empty_is_retried_once(tmp_path: Path) -> None:
    """Criterion 6: the same for a tool-loop step (``_run_step``): the step's request is resent once, identical but
    for its attempt id, the note says ``phase: "tool_step"`` and the step, and the loop ends ``final``."""
    async with tool_rig(tmp_path, policy=LOOP, script=[answer(PREFIX), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(REPORT)])
        rig.llm.fail_steps(engine_empty_error, None)
        await rig.ask(QUESTION)
        await rig.until(
            lambda: bool(generates_for(rig, STEP2)), "the step-2 request was never sent"
        )
        await rig.turn_done()
        ids = engine_ids(rig, STEP2)
        spoken = rig.spoken_sources()
        hud_errors = rig.commands(C.HudError)
    assert len(ids) == 2, f"T-0171 criterion 6: the step request is resent once; got {ids}"
    failed, retry = ids
    assert failed != retry
    first, second = prompts_of(rig, STEP2)
    assert first == second and images(first) == images(second), "criterion 6: the same step Prompt"
    assert body_of(rig, failed) == body_of(rig, retry), (
        "criterion 6: the same body but for the attempt id"
    )
    note = assert_one_retry_note(rig, STEP2, failed, retry, "tool_step")
    assert note.get("step") == 2, f"T-0171 criterion 6: the note carries the step; got {note}"
    assert_requests_logged(rig, STEP2, failed, retry)
    assert REPORT in spoken, (
        f"criterion 6: the step's answer comes from the retry; spoke {spoken!r}"
    )
    assert llm_errors(rig, STEP2) == [] and error_records(rig, STEP2) == [] and hud_errors == []
    extras = rig.record("t1").extras
    assert extras.get("tool_loop_end") == "final", extras
    assert extras.get(EXTRA) == 1, extras
    assert_concurrency(rig)


# =========================================================================== criterion 7


@pytest.mark.parametrize("phase", ["generate", "tool_step"])
async def test_second_engine_empty_ends_turn_with_error(tmp_path: Path, phase: str) -> None:
    """Criterion 7: the retry ends empty too -> today's error path (an ``error`` record, then ``LlmError``), and
    exactly two requests were sent."""
    if phase == "generate":
        mid, what = STEP1, "llm.generate"
        cm = tool_rig(tmp_path, script=[answer(ANSWER)])
    else:
        mid, what = STEP2, "llm.tool_step"
        cm = tool_rig(tmp_path, policy=LOOP, script=[answer(PREFIX), tool(VALID_CLICK)])
    async with cm as rig:
        if phase == "generate":
            rig.llm.fail_generates(engine_empty_error, engine_empty_error)
        else:
            rig.llm.script_steps([answer(REPORT)])
            rig.llm.fail_steps(engine_empty_error, engine_empty_error)
        await rig.ask(QUESTION)
        await rig.turn_done()
        await rig.advance(2_000)
        ids = engine_ids(rig, mid)
        errs = llm_errors(rig, mid)
        hud_errors = rig.commands(C.HudError)
    assert len(ids) == 2, f"T-0171 criterion 7: exactly 2 requests for {mid}; got {ids}"
    assert len(errs) == 1, f"T-0171 criterion 7: one LlmError to the machine; got {errs}"
    assert ids[1] in errs[0].message, (
        f"criterion 7: the error is the retry's (it names {ids[1]}); got {errs[0].message!r}"
    )
    recs = error_records(rig, mid)
    assert [r.get("what") for r in recs] == [what], (
        f"criterion 7: one {what} error record; got {recs}"
    )
    assert hud_errors, "criterion 7: the page is told"
    assert len(retry_notes(rig)) == 1, retry_notes(rig)
    assert_concurrency(rig)


# =========================================================================== criterion 8 (negative)


@pytest.mark.parametrize("phase", ["generate", "tool_step"])
@pytest.mark.parametrize("segment", ["thought", "answer"])
@pytest.mark.parametrize("error", ["engine_empty", "plain_wire"])
async def test_no_retry_after_a_commit(
    tmp_path: Path, phase: str, segment: str, error: str
) -> None:
    """Criterion 8 (adversarial: the real client cannot end empty after a commit, the fake can): a commit of the
    attempt was delivered, then the error -> no resend, the error path runs. A thought commit counts (nothing was
    voiced, but the guard is on what was delivered). Catches a retry keyed on the exception type alone, on step 1
    and (review A-F1) at the tool-step site."""
    first = thought("Let me look.", 0) if segment == "thought" else answer("Partly. ", 0)
    factory = engine_empty_error if error == "engine_empty" else plain_wire_error
    if phase == "generate":
        mid, what = STEP1, "llm.generate"
        cm = tool_rig(tmp_path, script=[first])
    else:
        mid, what = STEP2, "llm.tool_step"
        cm = tool_rig(tmp_path, policy=LOOP, script=[answer(PREFIX), tool(VALID_CLICK)])
    async with cm as rig:
        if phase == "generate":
            rig.llm.fail_generates_after_script(factory)
        else:
            rig.llm.script_steps([first], [answer(REPORT)])
            rig.llm.fail_steps_after_script(factory)
        await rig.ask(QUESTION)
        await rig.turn_done()
        await rig.advance(2_000)
        ids = engine_ids(rig, mid)
        errs = llm_errors(rig, mid)
    assert len(ids) == 1, f"T-0171 criterion 8: no retry after a delivered commit; got {ids}"
    assert [
        d.get("attempt_id") for d in rig.kind("llm.commit") if d.get("request_id") == mid
    ] == ids, "premise: the commit was delivered"
    assert len(errs) == 1, f"criterion 8: the error path runs; got {errs}"
    assert [r.get("what") for r in error_records(rig, mid)] == [what]
    assert retry_notes(rig) == []
    assert EXTRA not in rig.record("t1").extras


@pytest.mark.parametrize("phase", ["generate", "tool_step"])
@pytest.mark.parametrize("error", ["plain_wire", "transport"])
async def test_plain_wire_error_not_retried(tmp_path: Path, phase: str, error: str) -> None:
    """Criterion 8, second half, and "any other truncated stream stays the error it is today": a plain
    ``LlmWireError`` (``type(exc) is LlmWireError``) or a transport error with nothing delivered is not retried, on
    step 1 or on a tool step. Catches a retry keyed on ``isinstance(exc, LlmWireError)`` (the base class)."""
    factory = plain_wire_error if error == "plain_wire" else transport_error
    if phase == "generate":
        mid, what = STEP1, "llm.generate"
        cm = tool_rig(tmp_path, script=[answer(ANSWER)])
    else:
        mid, what = STEP2, "llm.tool_step"
        cm = tool_rig(tmp_path, policy=LOOP, script=[answer(PREFIX), tool(VALID_CLICK)])
    async with cm as rig:
        if phase == "generate":
            rig.llm.fail_generates(factory, None)
        else:
            rig.llm.script_steps([answer(REPORT)])
            rig.llm.fail_steps(factory, None)
        await rig.ask(QUESTION)
        await rig.turn_done()
        await rig.advance(2_000)
        ids = engine_ids(rig, mid)
        errs = llm_errors(rig, mid)
    assert len(ids) == 1, f"T-0171 criterion 8: a {error} error is not retried; got {ids}"
    assert len(errs) == 1 and ids[0] in errs[0].message, errs
    assert [r.get("what") for r in error_records(rig, mid)] == [what]
    assert retry_notes(rig) == []
    assert EXTRA not in rig.record("t1").extras


# =========================================================================== criterion 9


@pytest.mark.parametrize("takeover", [True, False], ids=["barge-in", "abort"])
async def test_abort_between_failure_and_resend_sends_nothing(
    tmp_path: Path, takeover: bool
) -> None:
    """Criterion 9: the request is aborted (``Session.abort_request``, with or without the barge-in that causes it)
    after it failed and before the session sees the failure -> no resend, no ``llm.request`` for a second attempt.
    Catches a resend that does not check the abort it raced."""
    calls: list[str] = []
    async with tool_rig(tmp_path, script=[answer(ANSWER)]) as rig:

        async def abort_then_fail(rid: str) -> BaseException:
            exc = engine_empty_error(rid)
            if takeover:
                now = rig.clock.now_ms
                rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
            await rig.session.abort_request(STEP1)
            calls.append(rid)
            return exc

        rig.llm.fail_generates(abort_then_fail, None)
        await rig.ask(QUESTION)
        await rig.until(lambda: bool(calls), "the failing request was never sent")
        await rig.advance(3_000)
        ids = engine_ids(rig, STEP1)
        aborted = list(rig.llm.aborted)
        spoken = rig.spoken_sources()
    assert len(calls) == 1 and ids[:1] == calls, (
        f"premise: the failing request was the first; {ids} vs {calls}"
    )
    assert calls[0] in aborted, f"premise: the abort reached the engine; aborted {aborted}"
    assert len(ids) == 1, f"T-0171 criterion 9: no resend after the abort; got {ids}"
    assert len(requests_for(rig, STEP1)) == 1, (
        f"criterion 9: no llm.request record for a resend that was not sent; got {requests_for(rig, STEP1)}"
    )
    assert ANSWER.strip() not in spoken
    assert_concurrency(rig)


async def test_abort_during_retry_no_third_request(tmp_path: Path) -> None:
    """Criterion 9: a barge-in while the retry is live aborts the retry (its own engine id reaches ``abort``), the
    turn does not answer, and no third request follows."""
    async with tool_rig(tmp_path, script=[answer(ANSWER)]) as rig:
        rig.llm.fail_generates(engine_empty_error, None)
        rig.llm.hold()
        await rig.ask(QUESTION)
        await rig.until(
            lambda: len(engine_ids(rig, STEP1)) >= 2,
            f"T-0171 criterion 5: no resend after the empty retryable response; sent {engine_ids(rig, STEP1)}",
            limit_ms=5_000,
        )
        failed, retry = engine_ids(rig, STEP1)
        assert rig.state.state is TurnState.THINKING, "premise: the retry is live, nothing spoken"
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.until(
            lambda: retry in rig.llm.aborted,
            f"T-0171 criterion 9: the abort targets the retry {retry}; aborted {rig.llm.aborted}",
            limit_ms=2_000,
        )
        rig.llm.release()
        await rig.advance(3_000)
        ids = engine_ids(rig, STEP1)
        spoken = rig.spoken_sources()
        dones = rig.dones(STEP1)
    assert ids == [failed, retry], f"T-0171 criterion 9: no third request; got {ids}"
    assert dones == [], f"the aborted retry never reaches llm.done; got {dones}"
    assert ANSWER.strip() not in spoken
    assert_concurrency(rig)


# =========================================================================== criterion 11


class TapLlm(ToolLlm):
    """``ToolLlm`` that also plays the plugin's tap into a real ``CanvasRelay`` (``relay``): one denoise frame and one
    commit frame of the request, then a flush, before each turn commit it yields. An attempt that yields nothing puts
    nothing on the tap, as the engine's empty response does."""

    relay: Any = None

    async def generate(  # type: ignore[override]
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None = None,
        *,
        xargs: Mapping[str, str] | None = None,
        cd: Any = None,
    ) -> AsyncIterator[Commit]:
        caption = is_caption_prompt(prompt)
        async for c in super().generate(prompt, request_id, options, xargs=xargs, cd=cd):
            if not caption and self.relay is not None:
                self.relay.receive(pack(denoise(request_id, c.canvas_idx, 1)))
                self.relay.flush()  # a commit discards a pending denoise frame (relay A3): publish it first
                self.relay.receive(pack(tap_commit(request_id, c.canvas_idx, 2)))
                self.relay.flush()
            yield c


async def test_no_canvas_messages_for_failed_attempt(tmp_path: Path) -> None:
    """Criterion 11: the failed attempt produces no ``astra.canvas.*`` message, the retry introduces no
    ``astra.canvas.synthetic``, and every canvas message of the turn carries the retry's attempt id."""
    from local_astra.conductor.canvas_relay import CanvasRelay

    llm = TapLlm([thought("Reading the form.", 0), answer(ANSWER, 1)])
    async with tool_rig(tmp_path, llm=llm) as rig:
        llm.relay = CanvasRelay(
            publish=rig.published,
            clock=rig.clock,
            state=rig.session.turn_state_message,
            **optional_piece_kwargs(CanvasRelay, StubPieces()),
        )
        llm.fail_generates(engine_empty_error, None)
        await rig.ask(QUESTION)
        await rig.turn_done()
        ids = engine_ids(rig, STEP1)
        canvas = [
            m
            for m in rig.published.messages
            if str(getattr(m, "type", "")).startswith("astra.canvas.")
        ]
    assert len(ids) == 2, f"T-0171 criterion 5: the empty response is resent once; got {ids}"
    failed, retry = ids
    assert canvas, "T-0171 criterion 11: the retry's real canvases reach the denoising view"
    assert [m for m in canvas if m.type == "astra.canvas.synthetic"] == [], (
        "T-0171 criterion 11: no synthetic canvas for the failed attempt or the retry"
    )
    assert [m for m in canvas if getattr(m, "request_id", None) == failed] == [], (
        f"T-0171 criterion 11: the failed attempt {failed} has no canvas message"
    )
    assert {getattr(m, "request_id", None) for m in canvas} == {retry}, (
        f"T-0171 criterion 11: every canvas message of the turn carries the retry's id {retry}; got "
        f"{sorted({str(getattr(m, 'request_id', None)) for m in canvas})}"
    )
    assert {m.phase for m in canvas} == {"denoise", "commit"}, "premise: both phases were relayed"


# =========================================================================== criterion 13: out of scope stays as-is

ERRORS = {
    "plain_wire": plain_wire_error,
    "engine_empty": engine_empty_error,
}


async def _repair_obs(tmp_path: Path, error: str) -> dict[str, Any]:
    async with tool_rig(
        tmp_path,
        script=[answer(ANSWER), tool(REPAIRABLE_CLICK)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        rig.llm.continuation_errors = [ERRORS[error]]
        await rig.ask(QUESTION)
        await rig.turn_done()
        await rig.advance(2_000)
        obs = {
            "continuations": len(rig.llm.continuations()),
            "statuses": rig.statuses(),
            "repair_errors": len(llm_errors(rig, REPAIR_ID)),
            "hud_errors": len(rig.commands(C.HudError)),
            "executed": len(rig.executor.invocations),
        }
    record = rig.record("t1")
    obs["tool_status"] = (record.tool or {}).get("status")
    obs["outcome"] = record.outcome
    obs["notes"] = len(retry_notes(rig))
    assert obs["statuses"][-1] == "refused" and obs["repair_errors"] == 1, (
        f"premise: the repair failed; {obs}"
    )
    return obs


async def _pass2_obs(tmp_path: Path, error: str) -> dict[str, Any]:
    from tests.unit import test_session_reground as R

    exc = ERRORS[error]("sess-t1-reground-9")
    rig, browser, extras = await R.one_click_turn(tmp_path, [exc])
    (rec,) = R.regrounds(rig)
    obs = {
        "pass2_requests": len(rig.llm.pass2_ids),
        "outcome": rec["outcome"],
        "executed": dict(R.inv_of(browser, "t1-call").arguments),
        "reground": {k: v for k, v in extras.get("reground", {}).items() if k != "ms"},
        "notes": len(retry_notes(rig)),
    }
    assert obs["outcome"] == "error", f"premise: pass 2 failed; {obs}"
    return obs


class _PrefillFails(ToolLlm):
    prefill_error: BaseException | None = None

    async def prefill(self, prefix: Prompt, request_id: str) -> Any:
        self.prefills.append((request_id, prefix))
        if self.prefill_error is not None:
            raise self.prefill_error
        return await super().prefill(prefix, request_id)


async def _prefill_obs(tmp_path: Path, error: str) -> dict[str, Any]:
    llm = _PrefillFails([answer(ANSWER)])
    llm.prefill_error = ERRORS[error]("sess-t1-prefill-1")
    async with tool_rig(tmp_path, llm=llm) as rig:
        await rig.ask(QUESTION)
        await rig.turn_done()
        await rig.advance(2_000)
        obs = {
            "prefills": len(llm.prefills),
            "generates": len(engine_ids(rig, STEP1)),
            "spoken": ANSWER.strip() in rig.spoken_sources(),
            "hud_errors": len(rig.commands(C.HudError)),
        }
    obs["errors"] = sorted(str(d.get("what")) for d in rig.kind("error"))
    obs["outcome"] = rig.record("t1").outcome
    obs["notes"] = len(retry_notes(rig))
    assert obs["prefills"] >= 1, f"premise: a prefill was attempted; {obs}"
    return obs


class _CaptionBackend:
    """A caption backend whose generate raises ``error`` at the first ``__anext__``, counting requests."""

    def __init__(self, error: Any) -> None:
        self.error = error
        self.requests: list[str] = []
        self.aborted: list[str] = []

    def generate(
        self, prompt: Prompt, request_id: str, options: LlmOptions | None = None, **_: Any
    ) -> Any:
        async def stream() -> AsyncIterator[Commit]:
            self.requests.append(request_id)
            raise self.error(request_id)
            yield  # pragma: no cover

        return stream()

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


async def _captioner_obs(tmp_path: Path, error: str) -> dict[str, Any]:
    thumb = tmp_path / "thumb.jpg"
    thumb.write_bytes(b"\xff\xd8" + b"x" * 64 + b"\xff\xd9")
    memory = FakeMemory()
    eid = await memory.ingest(
        Episode(ts_ms=1, source=Source.SCREEN, thumbnail_path=str(thumb), phash="0" * 16)
    )
    backend = _CaptionBackend(ERRORS[error])
    raised: BaseException | None = None
    try:
        await IdleCaptioner(backend, memory).run_once()  # type: ignore[arg-type]
    except Exception as exc:  # the row compares what escapes
        raised = exc
    return {
        "requests": len(backend.requests),
        "raised_wire_error": isinstance(raised, llm_client.LlmWireError),
        "caption": memory.episodes[eid].caption,
    }


class _CaptionFails(ToolLlm):
    """``ToolLlm`` whose caption requests raise ``caption_error(engine id)`` at the first ``__anext__`` (where the real
    client raises); every other request is the ``ToolLlm`` one. ``caption_ids`` records each caption request."""

    caption_error: Any = None

    async def generate(  # type: ignore[override]
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None = None,
        *,
        xargs: Mapping[str, str] | None = None,
        cd: Any = None,
    ) -> AsyncIterator[Commit]:
        if is_caption_prompt(prompt) and self.caption_error is not None:
            self.caption_ids.append(request_id)
            raise self.caption_error(request_id)
        async for c in super().generate(prompt, request_id, options, xargs=xargs, cd=cd):
            yield c


async def _live_caption_obs(tmp_path: Path, error: str) -> dict[str, Any]:
    """Review B-T2: the live caption path -- the session's own ``captioner``, which reaches the backend through
    ``session._CaptionLlm`` -- not a stub ``IdleCaptioner``. A retry added in that wrapper sends a second caption."""
    llm = _CaptionFails([answer(ANSWER)])
    llm.caption_ids = []  # type: ignore[attr-defined]
    llm.caption_error = ERRORS[error]
    thumb = tmp_path / "thumb.jpg"
    thumb.write_bytes(b"\xff\xd8" + b"x" * 64 + b"\xff\xd9")
    async with tool_rig(tmp_path, llm=llm) as rig:
        memory = rig.session.memory
        eid = await memory.ingest(
            Episode(ts_ms=1, source=Source.SCREEN, thumbnail_path=str(thumb), phash="1" * 16)
        )
        raised: BaseException | None = None
        try:
            await rig.session.captioner.run_once()
        except Exception as exc:  # the row compares what escapes
            raised = exc
        await rig.advance(1_000)
        obs = {
            "requests": len(llm.caption_ids),  # type: ignore[attr-defined]
            "raised_wire_error": isinstance(raised, llm_client.LlmWireError),
            "caption": memory.episodes[eid].caption,
        }
    obs["notes"] = len(retry_notes(rig))
    assert obs["requests"] >= 1, f"premise: the live caption path reached the backend; {obs}"
    return obs


OBSERVERS = {
    "tool_repair": _repair_obs,
    "reground_pass2": _pass2_obs,
    "prefill": _prefill_obs,
    "captioner": _captioner_obs,
    "captioner_live": _live_caption_obs,
}


@pytest.mark.parametrize("path", sorted(OBSERVERS))
async def test_repair_and_pass2_unchanged(tmp_path: Path, path: str) -> None:
    """Criterion 13: the tool repair, ReGround pass 2, the prefill and the captioner treat the engine's empty response
    exactly as a plain ``LlmWireError``: the same run, observed the same way, with each error, gives the same result
    (no retry, no retry note, the same ending). Catches the retry leaking into a path the task keeps out of scope."""
    observe = OBSERVERS[path]
    (tmp_path / "plain").mkdir()
    (tmp_path / "empty").mkdir()
    plain = await observe(tmp_path / "plain", "plain_wire")
    empty = await observe(tmp_path / "empty", "engine_empty")
    assert empty == plain, (
        f"T-0171 criterion 13: {path} treats LlmEngineEmpty as LlmWireError today;\n plain: {plain}\n empty: {empty}"
    )


def test_warmup_unchanged() -> None:
    """Criterion 13, the repair warm-up (T-0101): an empty retryable response to one warm-up request fails that tool
    exactly as a plain ``LlmWireError`` does, and the warm-up sends one request per tool either way."""
    from tests.unit import test_repair_warmup as W

    def run(error: str) -> dict[str, Any]:
        llm = W._llm()
        llm.continuation_errors = [None, ERRORS[error], None, None, None]
        report = W._run(llm, W._kit())
        statuses = {t: W._entry(report, t).status for t in sorted(report.per_tool)}
        return {
            "requests": len(llm.generates),
            "statuses": {t: s.split(":", 1)[0] for t, s in statuses.items()},
        }

    plain, empty = run("plain_wire"), run("engine_empty")
    assert "failed" in plain["statuses"].values(), f"premise: one warm-up request failed; {plain}"
    assert empty == plain, (
        f"T-0171 criterion 13: the warm-up is unchanged; plain {plain}, empty {empty}"
    )


# =========================================================================== criterion 13: the over-long retry


@pytest.mark.parametrize(
    ("order", "answered"),
    [
        (("413", "empty", None), True),
        (("empty", "413", None), True),
        (("413", "empty", "413"), False),
        (("empty", "413", "empty"), False),
    ],
    ids=["413-empty-ok", "empty-413-ok", "413-empty-413", "empty-413-empty"],
)
async def test_over_long_retry_still_independent(
    tmp_path: Path, order: tuple[str | None, ...], answered: bool
) -> None:
    """Criterion 13: the over-long (413) retry and the empty-response retry are independent, at most one of each per
    request: 413 then empty then success is three requests and an answer (and the reverse); a second failure of
    either kind after both retries were spent ends the turn with the error. The empty retry resends the request it
    retries unchanged; the over-long retry drops one more history turn."""
    entries = {"413": http_error(413), "empty": engine_empty_error, None: None}
    async with tool_rig(tmp_path, script=[answer("A number. ")]) as rig:
        await two_turns_of_history(rig)
        rig.llm.script = [answer(ANSWER)]
        rig.llm.fail_generates(*(entries[o] for o in order))
        await rig.ask("and what is the total")
        await rig.turn_done("the third turn never ended")
        await rig.advance(2_000)
        mid = "t3-gen"
        ids = engine_ids(rig, mid)
        prompts = prompts_of(rig, mid)
        spoken = rig.spoken_sources()
        errs = llm_errors(rig, mid)
    assert len(ids) == 3, f"T-0171 criterion 13: one retry of each kind, so 3 requests; got {ids}"
    a, b, c = prompts
    if order[0] == "413":
        assert history_messages(b) == history_messages(a) - 2, (
            "the over-long retry drops one history turn"
        )
        assert body_of(rig, ids[1]) == body_of(rig, ids[2]), (
            "the empty retry resends the trimmed request"
        )
    else:
        assert body_of(rig, ids[0]) == body_of(rig, ids[1]), (
            "the empty retry resends the request unchanged"
        )
        assert history_messages(c) == history_messages(b) - 2, (
            "the over-long retry drops one history turn"
        )
    record = rig.record("t3")
    assert record.extras.get("prompt_error_status") == 413, record.extras
    assert record.extras.get(EXTRA) == 1, record.extras
    if answered:
        assert ANSWER.strip() in spoken and errs == [], (spoken, errs)
    else:
        assert ANSWER.strip() not in spoken and len(errs) == 1, (spoken, errs)


# =========================================================================== criterion 12

#: sha256 of each file criterion 12 keeps byte-identical, computed from ``git show 3e4d45b7:<path>`` (an invariant
#: pins content; ``test_the_content_pins_are_the_base_blobs`` re-derives them). When T-0171 merges, a later task that
#: legitimately edits one of these files converts this row to T-0171's closed range (tests.md, "Pin both ends").
PINNED = {
    "local_astra/conductor/turn_machine.py": "33776b8bd41d0b52df3556c3253d2e1ca2efea24a7d4aad4133c7dbc7a92d2fd",
    "local_astra/contracts/__init__.py": "56f54b85df9d9c51f44f88c29ed383fb783c2c552f80d20c18463686c4ad8c34",
    "local_astra/contracts/base.py": "8a71b74d453c882d56e68d2118212778a0647725b0f97f0f3c1ab41d21cc7b0f",
    "local_astra/contracts/cd.py": "79e59ab4670011dea0d42724971074bfa2d0f00a29755d3c616e3e1a78a22b91",
    "local_astra/contracts/codec.py": "ec10fa7c84ec4845197321c37111b91a65ec263cc2fa2228baf9ff6fb3f92338",
    "local_astra/contracts/commands.py": "687448bc557c85c2e91bff7b983a6293cd7cbf2533ca2a598a3a66702f8d9fd6",
    "local_astra/contracts/events.py": "71f430714720e5bb098c3627a4cc6b324ac053348f96b814751e2f5d8164c086",
    "local_astra/contracts/llm.py": "80855e4a51637c0525bb4f8c39149ff80f46c5af8ddf5fe7929865d4c7d4fd5a",
    "local_astra/contracts/memory.py": "25e07d187b1d1b7ec44ebafc02734d7e6fd6dc1cc0b5b2904f9e0e519a9141ff",
    "local_astra/contracts/perception.py": "f053a559001c9cf265dc13fdc6883a8c15f8f2d5fe6cc44ff020dc8cb7913cab",
    "local_astra/contracts/protocol.py": "3ffa1360b805dbc15e69fb869da2d2f7f607060c5ef1cceb73fd265d363fc675",
    "local_astra/contracts/protocols.py": "e43e12b6188f50f65cd95bdf2fa9f3eb1b0fcbef0fb8cd8aa747fc4a8a111343",
    "local_astra/contracts/speech.py": "da61f85786a30bf1733c8fe3015684daf0961cc40e53f69bf2abb75681565591",
    "local_astra/contracts/toolcall.py": "b9546244cc41bb4c731264090839d6ca2a0aef3fb473cf723e7c82f8266b3f47",
    "local_astra/contracts/turn.py": "4b450c209e925267f7e811ced05715988191620d1e8d198baa31de1e6a927d8d",
}


#: T-0178 B6a re-pin (closed range, the T-0162 way in test_session_tool_loop): T-0178 B edits these two files. T-0171's
#: claim about them is that they kept 3e4d45b7's content until T-0178 branched, so they are checked at T-0178's literal
#: base; their T-0178 content is pinned as literals (sha256 of each file at the implementation commit f8e26760).
T0178_BASE = "0fd25414"
T0178_MOVED = {
    # T-0182 B re-pin: the afc57335 content (B1 ack_clip, B4 still_working); T-0179's ebaf4248 content (f462bff5...)
    # is kept as history in T0179_CONTENT, checked at T-0182's base
    "local_astra/conductor/turn_machine.py": "507c51ebc9ffc7c225e6a4b0c4607e25f9e4d52c524476feacbe8f0e37af5ec6",
    "local_astra/contracts/commands.py": "776033c56d60eb06a2cab6560d2a10c7ed5d8943835c23e92907cb5026220057",
}
#: T-0179 B re-pin: T-0178's content of the same two files (amendment B3a, 52be36b0), checked at T-0179's literal base
#: aeac453f, so the T-0178 pin is kept as history rather than dropped
T0179_BASE = "aeac453f"
T0178_CONTENT = {
    "local_astra/conductor/turn_machine.py": "9b01cdfe986686374ecd9459388cb86caedd7a17e1a65f4d583feeeb31606d96",
    "local_astra/contracts/commands.py": "d525335b372a5e3d55d4b89cd97dd3a3965bf0db20c57ef7180122aeb634a6df",
}


#: T-0182 B re-pin (closed range, the T-0179 way above): B edits ``turn_machine.py`` (moved in T0178_MOVED) and
#: ``contracts/turn.py`` (B4's one ``MachineState`` field). At T-0182's literal base BASE_B the machine was T-0179's
#: content and ``turn.py`` still 3e4d45b7's; their T-0182 content is a literal (sha256 at b8d26ae7).
T0182_BASE = "7646b8b6409a30fcc3794c2453f4b77e080cbdb6"
T0179_CONTENT = {
    "local_astra/conductor/turn_machine.py": "f462bff53ff188a4ccb6b4f500ca820ff111ca86f71efa50417e492b16e87468",
    "local_astra/contracts/turn.py": "4b450c209e925267f7e811ced05715988191620d1e8d198baa31de1e6a927d8d",
}
T0182_MOVED = {
    "local_astra/contracts/turn.py": "9a8f98131bd905f856a36ab8a972ad1f89b6d4f4a7da3ef15558a291da2e3d58",
}

#: T-0181 A0-7 re-pin (base-then-literal, the T-0182 way above): A0 edits ``turn_machine.py`` (the ``MicMuted`` row)
#: and ``contracts/events.py`` (``MicMuted``). At T-0181's literal base 021c2ef6 they still hold their earlier pins
#: (the machine T0178_MOVED's afc57335 content, events.py PINNED's 3e4d45b7 content); in the working tree the only
#: allowed change is A0's exact blobs at the implementation commit 9dbddf96 (sha256 of each, computed 2026-10-28).
T0181_BASE = "021c2ef6d33a916727fbf83ed554a2c4a5fbeef2"
T0181_MOVED = {
    "local_astra/conductor/turn_machine.py": "634c9260161c14935f70965166ebf9e876a8974543b543df7dfc51735635dcc7",
    "local_astra/contracts/events.py": "76aabfefc8faad5b7ee772f8a3ca6e5024a735aaa4a1d45fb42386acb4996934",
}


#: T-0177 A-P2 re-pin (closed range, the T-0178 way above): T-0177 A1 edits ``contracts/protocol.py``. T-0171's claim
#: about it is that it kept 3e4d45b7's content until T-0177 branched, so it is checked at T-0177's literal base. Its
#: T-0177 content cannot be a literal before the implementation exists, so the working tree is held to "BASE plus
#: exactly A1's additions" by :func:`_t0177_protocol_drift` (an AST comparison). Whoever merges T-0177 may convert this
#: to a sha256 literal of the merged content (tests.md, "Pin both ends").
T0177_BASE = "7646b8b6409a30fcc3794c2453f4b77e080cbdb6"
T0177_MOVED = "local_astra/contracts/protocol.py"
#: A1's three message models, added to the union and the tuple, and their wire tags per direction
T0177_MODELS = ("AstraVisionSelect", "AstraVisionState", "AstraVisionFrame")
T0177_TAGS = {
    "CLIENT_TO_SERVER": ("astra.vision.select",),
    "SERVER_TO_CLIENT": ("astra.vision.state", "astra.vision.frame"),
}
#: top-level names A1 must add; any other new top-level name must mention Vision (an alias such as a source literal)
T0177_NEW_NAMES = (*T0177_MODELS, "VisionImage")
#: T-0180 A re-pin (closed range): T-0177 merged into main; its ``protocol.py`` content is the blob at main c0cb001e
#: ("T-0177: group A merged", the commit T-0180's branch merged), so the T-0177 range closes at that literal and the
#: working tree is held to "c0cb001e plus exactly T-0180 A1's additions" by the same checker.
T0177_TIP = "c0cb001ee5b638d88903ca6129f9baf0a87c3000"
T0177_PROTOCOL_SHA256 = "8a7e548d74107a5c3a3b7361df3c39479637833027adc846deaad5d409d7ee55"
T0180_MODELS = ("AstraThought",)
T0180_TAGS = {"SERVER_TO_CLIENT": ("astra.thought",)}
#: top-level names T-0180 A1 must add; any other new top-level name must mention Thought
T0180_NEW_NAMES = ("AstraThought", "THOUGHT_MAX_CHARS")
#: T-0181 A re-pin (closed range, the same way): T-0180 merged into main; its ``protocol.py`` content is the blob at
#: T-0181 A's base 74ae8b25 ("T-0180: group B merged", the merge base of T-0181 A's branch), sha256 computed
#: 2026-10-28. The T-0180 range closes at that literal; the working tree is either that literal (group A not yet
#: implemented) or that literal plus exactly T-0181 A1's mic pair (``AstraMicMute`` client->server only,
#: ``AstraMicState`` server->client only), and any other new top-level name must mention Mic.
T0181_A_BASE = "74ae8b25229c65494530c18a26e00e503a932c19"
T0180_PROTOCOL_SHA256 = "c8653d383df21117e132b4750ecbf55e8370d3aba4fec5b5f1c17f6b7afa68b2"
T0181_MODELS = ("AstraMicMute", "AstraMicState")
T0181_TAGS = {
    "CLIENT_TO_SERVER": ("astra.mic.mute",),
    "SERVER_TO_CLIENT": ("astra.mic.state",),
}
T0181_NEW_NAMES = T0181_MODELS
#: T-0193 A re-pin (closed range, the same way): T-0181 merged into main; its ``protocol.py`` content is the blob at
#: T-0193 A's base 1076bc57 ("T-0193: group A0 merged; notes for group A"), sha256 computed 2026-10-29. The working
#: tree is that literal plus exactly T-0193 A-1's ``AstraTurnStop`` (client->server only).
T0193_A_BASE = "1076bc570efa46e0c6843724a8364926ddbfa33b"
T0181_PROTOCOL_SHA256 = "23620730a7702a286d41e622543f23b18d88fe231ad5817bf58c8a0fbaa66173"
T0193_MODELS = ("AstraTurnStop",)
T0193_TAGS = {"CLIENT_TO_SERVER": ("astra.turn.stop",)}
#: T-0213 A2-read re-pin (R-5 named re-pin, closed range the same way): T-0193 merged into main; its ``protocol.py``
#: content is the blob at T-0213 A2-read's literal base 92584dea ("T-0213: R-9 an item for every catalog row"),
#: sha256 computed 2026-10-31 (equal to the blob at T-0193 A's 2b49ef4f). The working tree is that literal plus
#: exactly A2r-1's ``AstraSettingsState`` (server->client only) and its wire helpers.
T0213_A2R_BASE = "92584deae8b30e5b5fc5d032d632932b758afd7f"
T0193_PROTOCOL_SHA256 = "e51352a1bbf71210ce9e43f441a1416dc40b2a48ae1744902542fc3a65027724"
#: T-0213 A2a-1 (named re-pin): + ``AstraSettingsSet`` (``astra.settings.set``, client->server only), same range.
T0213_MODELS = ("AstraSettingsState", "AstraSettingsSet")
T0213_TAGS = {
    "SERVER_TO_CLIENT": ("astra.settings.state",),
    "CLIENT_TO_SERVER": ("astra.settings.set",),
}
T0213_NEW_NAMES = (
    "AstraSettingsState",
    "AstraSettingsSet",
    "SettingItem",
    "CatalogEntry",
    "SettingRefusal",
    "SettingValue",
    "SettingSource",
    "SettingKindName",
)


def _t0177_protocol_drift(
    base_src: str,
    now_src: str,
    *,
    models: tuple[str, ...] = T0177_MODELS,
    tags: dict[str, tuple[str, ...]] = T0177_TAGS,
    new_names: tuple[str, ...] = T0177_NEW_NAMES,
    mention: str = "vision",
) -> list[str]:
    """Every way ``now_src`` differs from ``base_src`` beyond T-0177 A1's additions (empty: only those).

    The keywords default to T-0177 A1; T-0180 A1 reuses the checker with its own models, tags, names and the word
    ``thought`` (the description below reads "A1" as whichever task's additions are passed).

    - every top-level statement of BASE is present with an identical AST, except ``AstraMessage``,
      ``ASTRA_MESSAGE_TYPES``, ``CLIENT_TO_SERVER`` and ``SERVER_TO_CLIENT``, which must equal BASE's once A1's members
      are removed, and must contain each of those members;
    - an ``import`` may only gain names;
    - the new top-level names include ``VisionImage`` and the three models, and every other new name mentions Vision.
    """
    import ast

    def named(tree: ast.Module) -> tuple[dict[str, ast.stmt], list[ast.stmt]]:
        by_name: dict[str, ast.stmt] = {}
        other: list[ast.stmt] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                by_name[node.name] = node
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    by_name[target.id] = node
                else:
                    other.append(node)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                by_name[node.target.id] = node
            else:
                other.append(node)
        return by_name, other

    def union(node: ast.expr) -> list[str]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return union(node.left) + union(node.right)
        return [ast.dump(node)]

    def members(name: str, node: ast.stmt) -> tuple[str, list[str]]:
        """(the statement with its member list blanked, its members as dumps)"""
        value = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None
        assert value is not None
        if name == "AstraMessage":
            assert isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Tuple)
            first, *rest = value.slice.elts
            return ast.dump(value.value) + "|".join(ast.dump(r) for r in rest), union(first)
        if name == "ASTRA_MESSAGE_TYPES":
            assert isinstance(value, ast.Tuple)
            return ast.dump(node.annotation) if isinstance(node, ast.AnnAssign) else "", [
                ast.dump(e) for e in value.elts
            ]
        assert isinstance(value, ast.Call) and len(value.args) == 1
        arg = value.args[0]
        assert isinstance(arg, ast.Set)
        shell = ast.dump(value.func) + (
            ast.dump(node.annotation) if isinstance(node, ast.AnnAssign) else ""
        )
        return shell, [ast.dump(e) for e in arg.elts]

    added = {
        "AstraMessage": [ast.dump(ast.Name(id=m, ctx=ast.Load())) for m in models],
        "ASTRA_MESSAGE_TYPES": [ast.dump(ast.Name(id=m, ctx=ast.Load())) for m in models],
        **{k: [ast.dump(ast.Constant(value=t)) for t in v] for k, v in tags.items()},
    }
    base_named, base_other = named(ast.parse(base_src))
    now_named, now_other = named(ast.parse(now_src))
    drift: list[str] = []
    for name, node in base_named.items():
        if name not in now_named:
            drift.append(f"{name} was removed")
        elif name in added:
            base_shell, base_members = members(name, node)
            now_shell, now_members = members(name, now_named[name])
            if now_shell != base_shell:
                drift.append(f"{name}: changed beyond its member list")
            missing = [m for m in added[name] if m not in now_members]
            if missing:
                drift.append(f"{name}: lacks A1's {missing}")
            if [m for m in now_members if m not in added[name]] != base_members:
                drift.append(f"{name}: its other members changed")
        elif ast.dump(now_named[name]) != ast.dump(node):
            drift.append(f"{name}: changed")
    new = set(now_named) - set(base_named)
    for name in new_names:
        if name not in new:
            drift.append(f"{name} was not added")
    for name in sorted(new - set(new_names)):
        if mention not in name.lower():
            drift.append(f"{name}: a new top-level name that is not A1's")
    base_imports = [n for n in base_other if isinstance(n, ast.ImportFrom | ast.Import)]
    now_imports = [n for n in now_other if isinstance(n, ast.ImportFrom | ast.Import)]
    if len(now_imports) != len(base_imports):
        drift.append("the import statements changed in number")
    for b, n in zip(base_imports, now_imports, strict=False):
        same_module = getattr(b, "module", None) == getattr(n, "module", None)
        b_names = {a.name for a in b.names}
        n_names = {a.name for a in n.names}
        if not same_module or not b_names <= n_names:
            drift.append(f"an import lost names or moved: {ast.dump(b)}")
    rest_base = [ast.dump(n) for n in base_other if not isinstance(n, ast.ImportFrom | ast.Import)]
    rest_now = [ast.dump(n) for n in now_other if not isinstance(n, ast.ImportFrom | ast.Import)]
    if rest_base != rest_now:
        drift.append("an unnamed top-level statement (the module docstring, ...) changed")
    return drift


_DRIFT_BASE = '''"""doc"""
from x import A
class M(C):
    f: int
AstraMessage = Annotated[A1 | A2, Field(discriminator="type")]
ASTRA_MESSAGE_TYPES: tuple[type[C], ...] = (A1, A2)
CLIENT_TO_SERVER: frozenset[str] = frozenset({"a"})
SERVER_TO_CLIENT: frozenset[str] = frozenset({"b"})
'''
_DRIFT_GOOD = '''"""doc"""
from x import A, B
class M(C):
    f: int
VisionSource = Literal["browser", "shared"]
class VisionImage(C):
    identity: str
class AstraVisionSelect(C):
    source: VisionSource
class AstraVisionState(C):
    source: VisionSource
class AstraVisionFrame(C):
    images: list[VisionImage]
AstraMessage = Annotated[A1 | A2 | AstraVisionSelect | AstraVisionState | AstraVisionFrame, Field(discriminator="type")]
ASTRA_MESSAGE_TYPES: tuple[type[C], ...] = (A1, A2, AstraVisionSelect, AstraVisionState, AstraVisionFrame)
CLIENT_TO_SERVER: frozenset[str] = frozenset({"a", "astra.vision.select"})
SERVER_TO_CLIENT: frozenset[str] = frozenset({"b", "astra.vision.state", "astra.vision.frame"})
'''


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        (None, None, None),
        ("    f: int\nVision", "    f: str\nVision", "M: changed"),
        (
            '{"b", "astra',
            '{"b", "astra.vision.select", "astra',
            "SERVER_TO_CLIENT: its other members",
        ),
        ("A1 | A2 | Astra", "A1 | Astra", "AstraMessage: its other members"),
        ("(A1, A2, Astra", "(A2, A1, Astra", "ASTRA_MESSAGE_TYPES: its other members"),
        (', "astra.vision.select"})', "})", "CLIENT_TO_SERVER: lacks"),
        ("class VisionImage(C):", "class Image(C):", "VisionImage was not added"),
        ("VisionSource = Literal", "Extra = 1\nVisionSource = Literal", "Extra: a new top-level"),
        (
            'Field(discriminator="type")]',
            'Field(discriminator="kind")]',
            "AstraMessage: changed beyond",
        ),
        ("from x import A, B", "from x import B", "an import lost names"),
        ('"""doc"""', '"""changed"""', "unnamed top-level statement"),
    ],
)
def test_the_t0177_protocol_drift_checker(
    old: str | None, new: str | None, fragment: str | None
) -> None:
    """T-0177 A-P2's checker on hand-written sources: BASE plus exactly A1's additions (an alias that mentions Vision
    and a gained import name allowed) is no drift; each mutation of a different kind is named; BASE itself (A1 absent)
    is drift, which is why the re-pinned row is red until A1 lands."""
    if old is None:
        assert _t0177_protocol_drift(_DRIFT_BASE, _DRIFT_GOOD) == []
        before = _t0177_protocol_drift(_DRIFT_BASE, _DRIFT_BASE)
        assert any("lacks A1's" in d for d in before) and any("not added" in d for d in before), (
            before
        )
        return
    assert new is not None and fragment is not None
    assert old in _DRIFT_GOOD, f"premise: the mutation applies ({old!r})"
    drift = _t0177_protocol_drift(_DRIFT_BASE, _DRIFT_GOOD.replace(old, new, 1))
    assert any(fragment in d for d in drift), drift


_DRIFT_T0180 = '''"""doc"""
from x import A
class M(C):
    f: int
THOUGHT_MAX_CHARS = 16000
ThoughtEnd = Literal["open", "done"]
class AstraThought(C):
    end: ThoughtEnd
AstraMessage = Annotated[A1 | A2 | AstraThought, Field(discriminator="type")]
ASTRA_MESSAGE_TYPES: tuple[type[C], ...] = (A1, A2, AstraThought)
CLIENT_TO_SERVER: frozenset[str] = frozenset({"a"})
SERVER_TO_CLIENT: frozenset[str] = frozenset({"b", "astra.thought"})
'''


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        (None, None, None),
        ("THOUGHT_MAX_CHARS = 16000\n", "", "THOUGHT_MAX_CHARS was not added"),
        ('{"a"})', '{"a", "astra.thought"})', "CLIENT_TO_SERVER: changed"),
        ('"b", "astra.thought"', '"b"', "SERVER_TO_CLIENT: lacks"),
        ("ThoughtEnd = Literal", "VisionX = 1\nThoughtEnd = Literal", "VisionX: a new top-level"),
    ],
)
def test_the_drift_checker_with_t0180s_additions(
    old: str | None, new: str | None, fragment: str | None
) -> None:
    """The T-0180 re-pin's use of the checker: c0cb001e's content plus exactly AstraThought, its cap and an alias
    that mentions Thought is no drift; the client direction, a missing cap, a missing tag or a non-thought new name
    is named."""
    kwargs: dict[str, Any] = {
        "models": T0180_MODELS,
        "tags": T0180_TAGS,
        "new_names": T0180_NEW_NAMES,
        "mention": "thought",
    }
    if old is None:
        assert _t0177_protocol_drift(_DRIFT_BASE, _DRIFT_T0180, **kwargs) == []
        assert any(
            "AstraThought was not added" in d
            for d in _t0177_protocol_drift(_DRIFT_BASE, _DRIFT_BASE, **kwargs)
        )
        return
    assert new is not None and fragment is not None
    assert old in _DRIFT_T0180, f"premise: the mutation applies ({old!r})"
    drift = _t0177_protocol_drift(_DRIFT_BASE, _DRIFT_T0180.replace(old, new, 1), **kwargs)
    assert any(fragment in d for d in drift), drift


_DRIFT_T0181 = '''"""doc"""
from x import A
class M(C):
    f: int
class AstraMicMute(C):
    muted: StrictBool
class AstraMicState(C):
    muted: StrictBool
    available: StrictBool
    reason: str | None = None
AstraMessage = Annotated[A1 | A2 | AstraMicMute | AstraMicState, Field(discriminator="type")]
ASTRA_MESSAGE_TYPES: tuple[type[C], ...] = (A1, A2, AstraMicMute, AstraMicState)
CLIENT_TO_SERVER: frozenset[str] = frozenset({"a", "astra.mic.mute"})
SERVER_TO_CLIENT: frozenset[str] = frozenset({"b", "astra.mic.state"})
'''


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        (None, None, None),
        ('{"a", "astra.mic.mute"})', '{"a"})', "CLIENT_TO_SERVER: lacks"),
        (
            '{"b", "astra.mic.state"})',
            '{"b", "astra.mic.state", "astra.mic.mute"})',
            "SERVER_TO_CLIENT: its other",
        ),
        (
            '{"a", "astra.mic.mute"})',
            '{"a", "astra.mic.mute", "astra.mic.state"})',
            "CLIENT_TO_SERVER: its other",
        ),
        ("A1 | A2 | AstraMicMute | AstraMicState", "A1 | A2 | AstraMicMute", "AstraMessage: lacks"),
        (
            "class AstraMicMute(C):",
            "VisionX = 1\nclass AstraMicMute(C):",
            "VisionX: a new top-level",
        ),
        ("    f: int\nclass AstraMicMute", "    f: str\nclass AstraMicMute", "M: changed"),
    ],
)
def test_the_drift_checker_with_t0181s_additions(
    old: str | None, new: str | None, fragment: str | None
) -> None:
    """The T-0181 A re-pin's use of the checker: T-0180's merged content plus exactly the mic pair, one tag in each
    direction, is no drift; a tag missing or in the wrong direction, a model missing from the union, a non-mic new name
    or an edit to an existing class is named."""
    kwargs: dict[str, Any] = {
        "models": T0181_MODELS,
        "tags": T0181_TAGS,
        "new_names": T0181_NEW_NAMES,
        "mention": "mic",
    }
    if old is None:
        assert _t0177_protocol_drift(_DRIFT_BASE, _DRIFT_T0181, **kwargs) == []
        assert any(
            "AstraMicState was not added" in d
            for d in _t0177_protocol_drift(_DRIFT_BASE, _DRIFT_BASE, **kwargs)
        )
        return
    assert new is not None and fragment is not None
    assert old in _DRIFT_T0181, f"premise: the mutation applies ({old!r})"
    drift = _t0177_protocol_drift(_DRIFT_BASE, _DRIFT_T0181.replace(old, new, 1), **kwargs)
    assert any(fragment in d for d in drift), drift


def test_turn_machine_untouched() -> None:
    """Criterion 12: ``turn_machine.py`` and every file of ``local_astra/contracts/`` hash to their 3e4d45b7 content,
    and ``contracts/`` holds no new file (no new event type, no new command).

    T-0178 B6a re-pin: the two files T-0178 B edits are checked at T-0178's base (still 3e4d45b7's content there) and
    against T-0178's literals in the working tree; every other file is unchanged from 3e4d45b7. T-0181 A0-7: the
    machine and ``events.py`` keep those pins at T-0181's base 021c2ef6 and are A0's 9dbddf96 blobs in the tree.
    T-0181 A (the ``protocol.py`` re-scope): T-0180's range closes at the literal blob at 74ae8b25, and the tree is
    that blob or that blob plus exactly ``AstraMicMute``/``AstraMicState``."""
    import subprocess

    contracts = sorted(
        str(p.relative_to(REPO)) for p in (REPO / "local_astra" / "contracts").rglob("*.py")
    )
    expected = sorted(p for p in PINNED if p.startswith("local_astra/contracts/"))
    assert contracts == expected, (
        f"T-0171 criterion 12: contracts/ gained or lost files: {contracts}"
    )
    gr.require_commits(T0178_BASE)
    for path in T0178_MOVED:
        blob = subprocess.run(
            ["git", "show", f"{T0178_BASE}:{path}"], cwd=REPO, capture_output=True, check=True
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == PINNED[path], (
            f"T-0171 criterion 12: {path} differs from {BASE_SHA}'s content at T-0178's base {T0178_BASE}"
        )
    gr.require_commits(T0179_BASE)
    for path, sha in T0178_CONTENT.items():
        blob = subprocess.run(
            ["git", "show", f"{T0179_BASE}:{path}"], cwd=REPO, capture_output=True, check=True
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == sha, (
            f"T-0179 re-pin: {path} at T-0179's base {T0179_BASE} is T-0178's content"
        )
    gr.require_commits(T0182_BASE)
    for path, sha in T0179_CONTENT.items():
        blob = subprocess.run(
            ["git", "show", f"{T0182_BASE}:{path}"], cwd=REPO, capture_output=True, check=True
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == sha, (
            f"T-0182 re-pin: {path} at T-0182's base {T0182_BASE[:8]} is T-0179's content"
        )
    gr.require_commits(T0177_BASE)
    base_protocol = subprocess.run(
        ["git", "show", f"{T0177_BASE}:{T0177_MOVED}"], cwd=REPO, capture_output=True, check=True
    ).stdout
    assert hashlib.sha256(base_protocol).hexdigest() == PINNED[T0177_MOVED], (
        f"T-0171 criterion 12: {T0177_MOVED} differs from {BASE_SHA}'s content at T-0177's base {T0177_BASE}"
    )
    gr.require_commits(T0177_TIP)
    tip_protocol = subprocess.run(
        ["git", "show", f"{T0177_TIP}:{T0177_MOVED}"], cwd=REPO, capture_output=True, check=True
    ).stdout
    assert hashlib.sha256(tip_protocol).hexdigest() == T0177_PROTOCOL_SHA256, (
        f"T-0180 re-pin: {T0177_MOVED} at main {T0177_TIP[:8]} is T-0177's merged content"
    )
    drift = _t0177_protocol_drift(base_protocol.decode("utf-8"), tip_protocol.decode("utf-8"))
    assert drift == [], f"T-0177 A-P2: {T0177_MOVED} is BASE plus exactly A1's additions; {drift}"
    gr.require_commits(T0181_A_BASE)
    t0180_protocol = subprocess.run(
        ["git", "show", f"{T0181_A_BASE}:{T0177_MOVED}"], cwd=REPO, capture_output=True, check=True
    ).stdout
    assert hashlib.sha256(t0180_protocol).hexdigest() == T0180_PROTOCOL_SHA256, (
        f"T-0181 A re-pin: {T0177_MOVED} at T-0181 A's base {T0181_A_BASE[:8]} is T-0180's merged content"
    )
    drift_t0180 = _t0177_protocol_drift(
        tip_protocol.decode("utf-8"),
        t0180_protocol.decode("utf-8"),
        models=T0180_MODELS,
        tags=T0180_TAGS,
        new_names=T0180_NEW_NAMES,
        mention="thought",
    )
    assert drift_t0180 == [], (
        f"T-0180 A1: {T0177_MOVED} is T-0177's merged content plus exactly AstraThought's additions; {drift_t0180}"
    )
    # T-0193 A re-pin (closed range): T-0181 merged into main; its ``protocol.py`` content is the blob at T-0193 A's
    # base 1076bc57, which is T-0180's merged content plus exactly the mic pair
    gr.require_commits(T0193_A_BASE)
    t0181_protocol = subprocess.run(
        ["git", "show", f"{T0193_A_BASE}:{T0177_MOVED}"], cwd=REPO, capture_output=True, check=True
    ).stdout
    assert hashlib.sha256(t0181_protocol).hexdigest() == T0181_PROTOCOL_SHA256, (
        f"T-0193 A re-pin: {T0177_MOVED} at T-0193 A's base {T0193_A_BASE[:8]} is T-0181's merged content"
    )
    drift_t0181 = _t0177_protocol_drift(
        t0180_protocol.decode("utf-8"),
        t0181_protocol.decode("utf-8"),
        models=T0181_MODELS,
        tags=T0181_TAGS,
        new_names=T0181_NEW_NAMES,
        mention="mic",
    )
    assert drift_t0181 == [], (
        f"T-0181 A1: {T0177_MOVED} is T-0180's merged content plus exactly the mic pair's additions; {drift_t0181}"
    )
    # T-0193 A-1: the working tree is that blob plus exactly AstraTurnStop's four additions (the class, its
    # AstraMessage member, its ASTRA_MESSAGE_TYPES entry, its CLIENT_TO_SERVER tag); SERVER_TO_CLIENT is compared
    # whole, and no other new top-level name is admitted (only ``AstraTurnStop`` contains the mention).
    # T-0213 A2-read re-pin (R-5 named re-pin, closed range): T-0193 merged into main; its ``protocol.py`` content is
    # the blob at T-0213 A2-read's literal base 92584dea, still T-0181's merged content plus exactly AstraTurnStop's
    # additions, and the working tree is that blob plus exactly A2r-1's ``astra.settings.state`` (server->client only)
    gr.require_commits(T0213_A2R_BASE)
    t0193_protocol = subprocess.run(
        ["git", "show", f"{T0213_A2R_BASE}:{T0177_MOVED}"],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    assert hashlib.sha256(t0193_protocol).hexdigest() == T0193_PROTOCOL_SHA256, (
        f"T-0213 A2-read re-pin: {T0177_MOVED} at T-0213 A2-read's base {T0213_A2R_BASE[:8]} is T-0193's merged content"
    )
    drift_t0193 = _t0177_protocol_drift(
        t0181_protocol.decode("utf-8"),
        t0193_protocol.decode("utf-8"),
        models=T0193_MODELS,
        tags=T0193_TAGS,
        new_names=T0193_MODELS,
        mention="astraturnstop",
    )
    assert drift_t0193 == [], (
        f"T-0193 A-1: {T0177_MOVED} is T-0181's merged content plus exactly AstraTurnStop's additions; {drift_t0193}"
    )
    now_protocol = (REPO / T0177_MOVED).read_bytes()
    drift_t0213 = _t0177_protocol_drift(
        t0193_protocol.decode("utf-8"),
        now_protocol.decode("utf-8"),
        models=T0213_MODELS,
        tags=T0213_TAGS,
        new_names=T0213_NEW_NAMES,
        mention="setting",
    )
    assert drift_t0213 == [], (
        f"T-0213 A2r-1: {T0177_MOVED} is T-0193's merged content plus exactly the settings state's additions; "
        f"{drift_t0213}"
    )
    gr.require_commits(T0181_BASE)
    before_a0 = {**PINNED, **T0178_MOVED}
    for path in T0181_MOVED:
        blob = subprocess.run(
            ["git", "show", f"{T0181_BASE}:{path}"], cwd=REPO, capture_output=True, check=True
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == before_a0[path], (
            f"T-0181 A0-7: {path} at T-0181's base {T0181_BASE[:8]} is not its earlier pinned content"
        )
    want = {**PINNED, **T0178_MOVED, **T0182_MOVED, **T0181_MOVED}
    del want[T0177_MOVED]
    # T-0192 B re-pin (B-RV6, both ends pinned): the machine, events.py and commands.py hold the pins above at
    # T-0192 B's literal base da29ddbe (the base end), and in the working tree each is exactly its merged blob at the
    # implementation commit 97409910 (the tree end)
    from tests.unit.test_t0192_hand_back_machine import (
        T0192_BASE,
        T0192_BASE_SHA256,
        T0192_MOVED_SHA256,
        t0192_end_sha256,
    )

    for path, sha in T0192_BASE_SHA256.items():
        assert want.pop(path) == sha, (
            f"T-0192 B: {path} at T-0192 B's base is its earlier pinned content"
        )
        assert t0192_end_sha256(path, T0192_BASE) == sha, (
            f"T-0192 B: {path} at T-0192 B's base {T0192_BASE[:8]} is not its earlier pinned content"
        )
        # T-0193 A0 re-pin (A0-9; A0-R5, both ends pinned, the T-0192 B-RV6 way): the T-0192 tree end is the file
        # at T-0193 A0's literal base fdb9fc01 (the base end), and the working tree is exactly the merged 373df531
        # blob (``test_t0193_stop_row.T0193_MOVED_SHA256``, the tree end)
        from tests.unit.test_t0193_stop_row import T0193_BASE, T0193_BASE_SHA256, T0193_MOVED_SHA256

        assert T0193_BASE_SHA256[path] == T0192_MOVED_SHA256[path], (
            f"T-0193 A0: {path}'s base end is T-0192 B's tree end"
        )
        assert t0192_end_sha256(path, T0193_BASE) == T0192_MOVED_SHA256[path], (
            f"T-0192 B / T-0193 A0: {path} at T-0193 A0's base {T0193_BASE[:8]} is not the merged 97409910 blob"
        )
        assert t0192_end_sha256(path) == T0193_MOVED_SHA256[path], (
            f"T-0193 A0-R5: {path} is not the merged 373df531 blob"
        )
    # T-0193 A0 re-pin: contracts/turn.py (T0182_MOVED's b8d26ae7 content) is that content at fdb9fc01 and, in the
    # working tree, exactly the merged 373df531 blob (A0-R5)
    from tests.unit.test_t0193_stop_row import T0193_TURN, t0193_end_sha256

    assert want.pop(T0193_TURN) == T0193_BASE_SHA256[T0193_TURN]
    assert t0193_end_sha256(T0193_TURN, T0193_BASE) == T0193_BASE_SHA256[T0193_TURN]
    assert t0193_end_sha256(T0193_TURN) == T0193_MOVED_SHA256[T0193_TURN], (
        f"T-0193 A0-R5: {T0193_TURN} is not the merged 373df531 blob"
    )
    # T-0216 re-pin: R-1 (both ends pinned, the T-0193 A0 way): codec.py is PINNED's 3e4d45b7 content at T-0216's
    # literal base 423362b7 (the base end) and, in the working tree, exactly the e705e79b blob (the tree end)
    from tests.unit.test_t0216_hud_receive import (
        T0216_BASE,
        T0216_BASE_SHA256,
        T0216_CODEC,
        T0216_MOVED_SHA256,
        T0216C_BASE,
        T0216C_TREE_ENDS,
    )

    assert want.pop(T0216_CODEC) == T0216_BASE_SHA256, (
        f"T-0216 re-pin: R-1: {T0216_CODEC}'s base end is PINNED's 3e4d45b7 content"
    )
    assert t0192_end_sha256(T0216_CODEC, T0216_BASE) == T0216_BASE_SHA256, (
        f"T-0216 re-pin: R-1: {T0216_CODEC} at T-0216's base {T0216_BASE[:8]} is not its earlier pinned content"
    )
    # T-0216c re-pin: R-1 (both ends pinned, the T-0216 R-1 way): the e705e79b blob is codec.py at T-0216c's literal
    # base 7489b9ed (the base end); the working tree is one of ``T0216C_TREE_ENDS`` (the tree end: the e705e79b blob
    # or the T-0216c blob while T-0216c is pending, the T-0216c blob alone after activation)
    assert t0192_end_sha256(T0216_CODEC, T0216C_BASE) == T0216_MOVED_SHA256, (
        f"T-0216c re-pin: R-1: {T0216_CODEC} at T-0216c's base {T0216C_BASE[:8]} is not the e705e79b blob"
    )
    assert t0192_end_sha256(T0216_CODEC) in T0216C_TREE_ENDS, (
        f"T-0216c re-pin: R-1: {T0216_CODEC} is none of the pinned tree ends {T0216C_TREE_ENDS}"
    )
    changed = [
        path
        for path, sha in want.items()
        if hashlib.sha256((REPO / path).read_bytes()).hexdigest() != sha
    ]
    assert changed == [], f"T-0171 criterion 12 / T-0178 B6a: changed across the branch: {changed}"


def test_the_content_pins_are_the_base_blobs() -> None:
    """Provenance of ``PINNED``: each literal is the sha256 of the 3e4d45b7 blob (a fixed blob against a fixed
    literal)."""
    import subprocess

    gr.require_commits(BASE_SHA)
    for path, sha in PINNED.items():
        blob = subprocess.run(
            ["git", "show", f"{BASE_SHA}:{path}"], cwd=REPO, capture_output=True, check=True
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == sha, path
