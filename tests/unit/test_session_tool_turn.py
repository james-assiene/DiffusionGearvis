"""T-0092c: the tool turn end to end over the fakes -- detection, one seeded repair, dry-run execution, HUD, telemetry.

docs/tasks/T-0092c.md criteria 1-8, the carried items (T-0092a G8 re-check; T-0092b's "kept for the turn" with the
stale-generate case) and the adversarial rows the task names. The rig is ``tests/unit/_tool_turn_support.py``: the
T-0007 session with the dry-run kit of T-0092b, a recording executor, a ``Recorder`` publisher and a real session
log; the fake LLM scripts pass 1 (``script``) and the repair (``continuation``) separately.

Oracles, none of them the session's own bookkeeping:

* *what was judged* -> the verdicts are written out by hand from T-0091's rules (``_tool_turn_support`` names each
  call's status); ``live.classify`` is the oracle only where the criterion says "the parse reason";
* *what the repair sent* -> the fake's request log (``bodies`` is built by the real ``VllmClient.request_body``,
  ``cd_log`` holds the ``CdRequest``); the seed ids come from the fake's documented deterministic ``tokenize`` map;
* *what was published* -> the ``Recorder`` (wire order) and the real ``TurnJournal`` (what a reconnecting page is
  replayed);
* *what was voiced and stored* -> the machine's ``TtsSpeak`` commands and ``HistoryTurn``.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, n = 1 per row, deterministic. Every row is
red at the base ``4f8d4e0`` on a named assertion, except the guards marked "green at the base" in their docstring.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import ToolDraft, TurnState
from local_astra.toolcall import live
from tests import _git_range as gr
from tests.fakes.llm import http_error
from tests.unit import _toolcall_live_support as LS
from tests.unit._tool_turn_support import (
    BASE_SHA,
    CALL_ID,
    CC_CALL,
    NAMELESS,
    NARRATED_QUERY,
    PASS1_ID,
    REPAIR_ID,
    REPAIRABLE_CLICK,
    REPAIRABLE_DRAFT,
    REPAIRED_CLICK,
    SEARCH_NARRATED,
    SPOKEN,
    STILL_INVALID,
    TOOL_CALL_KEYS,
    VALID_CLICK,
    Q,
    ToolLlm,
    WatchingExecutor,
    answer,
    assert_concurrency,
    assert_history_clean,
    assert_never_voiced,
    final_message,
    frame,
    literal_in,
    patch_bound,
    seed_ids,
    thought,
    tool,
    tool_rig,
)
from tests.unit._tools_switch_support import make_kit

REPO = Path(__file__).resolve().parents[2]
QUESTION = "click the sign in button"


def draft_of(done: Any) -> Any:
    assert done.tool is not None, (
        f"T-0092c criterion 1: the session hands the machine LlmDone(tool=ToolDraft(...)) for a request with a "
        f"tool commit; LlmDone({done.request_id!r}) carries no tool"
    )
    return done.tool


def one(seq: Any, what: str) -> Any:
    items = list(seq)
    assert len(items) == 1, (
        f"T-0092c: expected exactly one {what}; got {len(items)}: {items!r:.300}"
    )
    return items[0]


def only_done(rig: Any, request_id: str) -> Any:
    dones = rig.dones(request_id)
    assert len(dones) == 1, f"one LlmDone({request_id!r}) reached the machine; got {dones}"
    return dones[0]


async def voice_turn(rig: Any, question: str = QUESTION) -> None:
    await rig.ask(question)
    await rig.turn_done()


# =====================================================================================================
# Criterion 1: detection
# =====================================================================================================


async def test_c1_a_valid_click_in_one_canvas_is_judged_published_and_dry_run(tmp_path):
    """Flow "a valid click in one canvas": ``LlmDone(tool=valid draft)``, ``proposed`` then ``dry_run``, one
    invocation with the call's own fields, the record's final state; the call is never voiced."""
    script = [thought("the user wants the button"), answer(SPOKEN), tool(VALID_CLICK, canvas_idx=0)]
    async with tool_rig(tmp_path, script=script) as rig:
        await voice_turn(rig)

    d = draft_of(only_done(rig, PASS1_ID))
    assert (d.call_id, d.status, d.name, d.arguments, d.reason, d.source) == (
        CALL_ID,
        "valid",
        "click_at",
        {"x": 500, "y": 250},
        None,
        "draft",
    ), f"T-0092c criterion 1: the draft the machine is handed; got {d!r}"

    assert rig.statuses() == ["proposed", "dry_run"], (
        f"T-0092c criterion 1/3: proposed, then dry_run; published {rig.statuses()}"
    )
    proposed, done = rig.tool_calls(CALL_ID)
    assert (proposed.turn_id, proposed.name, proposed.arguments, proposed.source) == (
        "t1",
        "click_at",
        {"x": 500, "y": 250},
        "draft",
    ), proposed
    assert done.action == "recorded click_at" and done.reason is None, (
        f"T-0092c criterion 3: dry_run carries the outcome's action and reason; got {done!r}"
    )

    invs = rig.executor.invocations
    assert len(invs) == 1, f"T-0092c criterion 3: the executor runs once; it ran {len(invs)} times"
    inv = invs[0]
    assert (inv.call_id, inv.turn_id, inv.name, inv.arguments, inv.source) == (
        CALL_ID,
        "t1",
        "click_at",
        {"x": 500, "y": 250},
        "draft",
    ), inv
    a = frame(0, "a")
    assert inv.frame is not None and (
        inv.frame.phash,
        inv.frame.width,
        inv.frame.height,
    ) == (a.phash, 1280, 720), (
        f"T-0092c criterion 3: the frame of the sent prompt's image; got {inv.frame!r}"
    )

    record = rig.record("t1")
    assert record.tool is not None and record.tool.get("status") == "dry_run", (
        f"T-0092c criterion 7: TurnRecord.tool holds the final state of the call; got {record.tool!r}"
    )
    assert record.tool.get("call_id") == CALL_ID, record.tool
    assert "I will click the Sign in button." in rig.tts.texts()
    assert_never_voiced(rig, arguments={"x": 500, "y": 250})
    assert_history_clean(rig, arguments={"x": 500, "y": 250})


async def test_c1_the_call_id_is_the_turn_id_plus_call_never_a_clock(tmp_path):
    """Criterion 1: ``f"{turn_id}-call"``, deterministic per turn -- turn 2's call is ``t2-call`` whatever the time."""
    async with tool_rig(tmp_path, script=[answer(SPOKEN), tool(VALID_CLICK)]) as rig:
        await voice_turn(rig)
        await rig.advance(1234)
        await voice_turn(rig, "and again")
    ids = [d.tool.call_id for d in rig.dones() if d.tool is not None]
    assert ids == ["t1-call", "t2-call"], f"T-0092c criterion 1: one call id per turn; got {ids}"
    assert [i.call_id for i in rig.executor.invocations] == ["t1-call", "t2-call"]


async def test_c1_a_call_split_across_canvases_is_one_call(tmp_path):
    """Flow "a valid call split across canvases": the answer spans canvases 0-1 and the request's single ``tool``
    commit comes last (canvas 2, T-0088 E6); one draft, one execution, the whole answer voiced."""
    script = [
        answer("I will click ", canvas_idx=0),
        answer("the Sign in button. ", canvas_idx=1),
        tool(VALID_CLICK, canvas_idx=2),
    ]
    async with tool_rig(tmp_path, script=script) as rig:
        await voice_turn(rig)
    d = draft_of(only_done(rig, PASS1_ID))
    assert (d.status, d.arguments) == ("valid", {"x": 500, "y": 250}), d
    assert rig.statuses() == ["proposed", "dry_run"], rig.statuses()
    assert rig.spoken_sources() == "I will click the Sign in button. "
    assert len(rig.executor.invocations) == 1


async def test_c1_no_spoken_sentence_google_search_with_query_problem(tmp_path):
    """Flow "no spoken sentence" (H4, H7, H2): a google_search with nothing before the tag.

    ``query_problem`` ("narration", ``score.query_problem``) is published and logged and never blocks; nothing is
    voiced; the dry run only *describes* the search (``would search for: <query>``, no URL)."""
    executor = WatchingExecutor(inner=LS.dry_run())
    script = [thought("they want the weather"), tool(SEARCH_NARRATED, canvas_idx=0)]
    async with tool_rig(tmp_path, script=script, executor=executor) as rig:
        await voice_turn(rig, "what is the weather in lisbon")

    assert rig.statuses() == ["proposed", "dry_run"], rig.statuses()
    proposed, done = rig.tool_calls(CALL_ID)
    assert proposed.query_problem == "narration", (
        f"T-0092c criterion 1 (H7): query_problem is published when set; got {proposed.query_problem!r}"
    )
    assert proposed.arguments == {"query": NARRATED_QUERY}
    assert done.action == f"would search for: {NARRATED_QUERY}", done.action
    assert "http" not in (done.action or "") and "://" not in (done.action or ""), (
        "H2: google_search is never executed beyond the dry-run description"
    )
    assert [c for c in rig.speaks() if c.text] == [], (
        f"T-0092c criterion 6: no spoken sentence -> nothing extra is voiced; spoke {rig.speaks()}"
    )
    assert rig.state.history[-1].assistant_spoken == "", rig.state.history[-1]
    assert_never_voiced(rig, arguments={"query": NARRATED_QUERY})
    rec = [r for r in rig.tool_records() if r.get("status") == "proposed"]
    assert rec and rec[0].get("query_problem") == "narration", (
        f"T-0092c criterion 7: the tool.call record carries query_problem; got {rig.tool_records()}"
    )


async def test_c1_a_nameless_draft_is_refused_without_tokenize_or_repair(tmp_path):
    """Flow "a nameless draft refused" (H6): one ``refused`` with ``name=""`` and ``arguments={}``, no tokenize,
    no repair request, no execution."""
    async with tool_rig(tmp_path, script=[answer(SPOKEN), tool(NAMELESS)]) as rig:
        await voice_turn(rig)
    d = draft_of(only_done(rig, PASS1_ID))
    assert (d.status, d.source) == ("refused", "draft"), d
    assert rig.statuses() == ["refused"], (
        f"T-0092c criteria 1/4: a refused verdict publishes one refused (its final message); {rig.statuses()}"
    )
    msg = rig.tool_calls(CALL_ID)[0]
    assert (msg.name, msg.arguments, msg.reason) == ("", {}, "no tool name"), msg
    assert rig.llm.tokenized == [] and rig.llm.continuations() == [], (
        "T-0092c criterion 1 (H6): no tokenize and no repair request for a nameless draft"
    )
    assert rig.commands(C.ToolRepairStart) == [] and rig.executor.invocations == []
    assert [c.reason for c in rig.commands(C.ToolCancel)] == ["refused"]


async def test_c1_a_cc_tool_commit_is_refused_with_its_parse_reason(tmp_path):
    """Flow "a ``<cc>`` tool commit refused" (H6, T-0091 amendment): refused with the parse reason, no repair."""
    expected = live.classify(CC_CALL)
    assert expected.status == "refused", "premise: T-0091 refuses a <cc> commit"
    async with tool_rig(tmp_path, script=[answer(SPOKEN), tool(CC_CALL)]) as rig:
        await voice_turn(rig)
    assert rig.statuses() == ["refused"], rig.statuses()
    msg = rig.tool_calls(CALL_ID)[0]
    assert msg.reason == expected.reason and msg.name == "" and msg.arguments == {}, msg
    assert rig.llm.tokenized == [] and rig.llm.continuations() == []
    assert_never_voiced(rig)


async def test_c1_repair_off_turns_a_repairable_draft_into_refused(tmp_path):
    """Flow "repair off" (T-0092b criterion 2, Q-b): ``repairable`` becomes ``refused("repair off: <reason>")``,
    shown on the card with its reason; no tokenize, no repair request."""
    reason = "payload manifest canvas_length 128 is not the profile's 256"
    executor = WatchingExecutor()
    kit = make_kit(repair_off=reason, executor=executor)
    async with tool_rig(
        tmp_path, script=[answer(SPOKEN), tool(REPAIRABLE_CLICK)], kit=kit, executor=executor
    ) as rig:
        await voice_turn(rig)
    d = draft_of(only_done(rig, PASS1_ID))
    assert (d.status, d.reason) == ("refused", f"repair off: {reason}"), d
    assert rig.statuses() == ["refused"], rig.statuses()
    assert rig.tool_calls(CALL_ID)[0].reason == f"repair off: {reason}"
    assert rig.llm.tokenized == [] and rig.llm.continuations() == []
    assert rig.commands(C.ToolRepairStart) == []


async def test_c1_without_a_tool_commit_llm_done_carries_no_tool(tmp_path):
    """Criterion 1: no ``tool`` commit -> ``LlmDone`` without a tool, nothing published, ``TurnRecord.tool`` null.
    Green at the base (a guard for the no-tool direction)."""
    async with tool_rig(tmp_path, script=[answer("It is a login page. ")]) as rig:
        await voice_turn(rig)
    assert all(d.tool is None for d in rig.dones())
    assert rig.tool_calls() == [] and rig.record("t1").tool is None


async def test_c1_tools_off_publishes_no_tool_call_and_llm_done_carries_no_tool(tmp_path):
    """Criteria 1 and 8: with tools off a ``tool`` commit is not judged (the executor is never reached).
    Green at the base."""
    async with tool_rig(tmp_path, script=[answer(SPOKEN), tool(VALID_CLICK)], kit=None) as rig:
        await voice_turn(rig)
    assert all(d.tool is None for d in rig.dones())
    assert rig.tool_calls() == [] and rig.executor.invocations == []
    assert rig.tool_records() == []


# =====================================================================================================
# Criterion 2: the repair
# =====================================================================================================


async def test_c2_repairable_then_repair_valid_executes_the_repaired_call(tmp_path):
    """Flow "repairable -> repair -> valid": ``ToolRepairStart``, one continuation, the repair's own call executed
    with ``source="repair"`` and the pass-1 call id (T7); wire order proposed -> repairing -> dry_run."""
    async with tool_rig(
        tmp_path,
        script=[answer(SPOKEN), tool(REPAIRABLE_CLICK)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        await voice_turn(rig)

    d = draft_of(only_done(rig, PASS1_ID))
    assert (d.status, d.name, d.call_id) == ("repairable", "click_at", CALL_ID), d
    starts = rig.commands(C.ToolRepairStart)
    assert [(s.request_id, s.draft_request_id, s.call_id) for s in starts] == [
        (REPAIR_ID, PASS1_ID, CALL_ID)
    ]
    assert len(rig.llm.continuations()) == 1, (
        f"T-0092c criterion 2: one repair generate; sent {len(rig.llm.continuations())}"
    )
    r = draft_of(only_done(rig, REPAIR_ID))
    assert (r.call_id, r.status, r.name, r.arguments, r.source) == (
        CALL_ID,
        "valid",
        "click_at",
        {"x": 12, "y": 5},
        "repair",
    ), r
    assert rig.statuses() == ["proposed", "repairing", "dry_run"], rig.statuses()
    final = final_message(rig)
    assert (final.source, final.arguments) == ("repair", {"x": 12, "y": 5}), final
    inv = one(rig.executor.invocations, "executor invocation")
    assert (inv.call_id, inv.source, inv.arguments) == (CALL_ID, "repair", {"x": 12, "y": 5}), inv
    assert rig.llm.tokenized == [REPAIRABLE_DRAFT.rstrip()], (
        f"T-0092c criterion 2 step 1 (F1): tokenize(draft.rstrip()); tokenized {rig.llm.tokenized}"
    )
    assert_concurrency(rig)
    assert rig.state.history[-1].assistant_spoken == SPOKEN


@pytest.mark.parametrize("stop_id", [50, 106, None])
async def test_c2_the_repair_body_is_the_seeded_continuation(tmp_path, stop_id):
    """The repair body, from the fake's request log (built by the real ``VllmClient.request_body``):

    ``continue_final_message`` with the last assistant text ``spoken_prefix + "<|tool_call>"`` after exactly pass 1's
    messages; the ``cd`` keyword with ``draft = (ids + [stop_id])[:256]`` (``stop_id`` from ``Commit.stop_id``, never
    re-derived), the tool's payload and ``record_tag`` = the repair's engine id; ``max_tokens`` 256; the same tools.
    """
    async with tool_rig(
        tmp_path,
        script=[answer(SPOKEN), tool(REPAIRABLE_CLICK, stop_id=stop_id)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        await voice_turn(rig)

    conts = rig.llm.continuations()
    assert len(conts) == 1, f"T-0092c criterion 2: one repair request; got {len(conts)}"
    engine_id, _prompt, options, cd = conts[0]
    body = one(rig.llm.continuation_bodies(), "repair body")
    pass1 = one(rig.llm.pass1_bodies(), "pass-1 body")
    assert body["continue_final_message"] is True and body.get("add_generation_prompt") is False
    assert body["messages"][-1] == {"role": "assistant", "content": SPOKEN + "<|tool_call>"}, (
        f"T-0092c criterion 2 step 2: the continuation is spoken_prefix + '<|tool_call>'; got {body['messages'][-1]}"
    )
    assert body["messages"][:-1] == pass1["messages"], (
        "T-0092c criterion 2: repair_prompt(pass1_prompt_as_sent, ...) continues exactly pass 1's messages"
    )
    assert body["tools"] == pass1["tools"] and body.get("tool_choice") == "none", "the same tools"
    assert options.max_tokens == 256 and body["max_tokens"] == 256, (options, body["max_tokens"])
    assert cd is not None, "T-0092c criterion 2 step 4: generate(..., cd=req)"
    assert LS.draft_ids_of(cd) == seed_ids(REPAIRABLE_DRAFT, stop_id), (
        f"T-0092c criterion 2 step 3: the seed is (tokenize(draft.rstrip()) + [stop_id={stop_id}])[:256]"
    )
    payload = live.load_payloads().tools["click_at"]
    assert (cd.mode, cd.automaton_sha256, cd.record_tag) == (
        "perstep_mar",
        payload.automaton_sha256,
        engine_id,
    ), (
        f"T-0092c criterion 2 step 3: the click_at payload, record_tag = the engine id; got "
        f"{(cd.mode, cd.automaton_sha256, cd.record_tag)} for {engine_id}"
    )
    assert body["vllm_xargs"]["astra_request_id"] == engine_id


async def test_c2_the_seed_is_cut_at_the_canvas_length(tmp_path):
    """T-0091 criterion 5 through the session: a draft of more than 256 ids is cut, and the stop falls off."""
    long_text = "z" * 300
    call = f"<|tool_call>call:click_at{{x:12,y:5,note:{Q}{long_text}{Q}}}<tool_call|>"
    assert live.classify(call).status == "repairable", "premise: a v1 name with a bad schema"
    draft = call[len("<|tool_call>") :]
    async with tool_rig(
        tmp_path,
        script=[answer(SPOKEN), tool(call)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        await voice_turn(rig)
    (_, _, _, cd) = one(rig.llm.continuations(), "repair request")
    ids = LS.draft_ids_of(cd)
    assert len(ids) == 256 and ids == seed_ids(draft, 50) and 50 not in ids[-1:], ids[-3:]


async def test_c2_a_repair_that_is_still_invalid_is_refused_with_its_reason(tmp_path):
    """Flow "repair -> still invalid" (R4): the repair's verdict is ``refused`` (never a second repair), the card
    ends ``refused`` with the verdict's reason, nothing is executed, the spoken prefix played whole."""
    async with tool_rig(
        tmp_path,
        script=[answer(SPOKEN), tool(REPAIRABLE_CLICK)],
        continuation=[tool(STILL_INVALID, canvas_idx=0)],
    ) as rig:
        await voice_turn(rig)
    r = draft_of(only_done(rig, REPAIR_ID))
    assert (r.status, r.source, r.reason) == ("refused", "repair", "schema invalid"), r
    assert rig.statuses() == ["proposed", "repairing", "refused"], rig.statuses()
    assert final_message(rig).reason == "schema invalid"
    assert rig.executor.invocations == [] and len(rig.llm.continuations()) == 1
    assert rig.state.history[-1].assistant_spoken == SPOKEN


async def test_c2_a_repair_with_no_tool_commit_is_refused(tmp_path):
    """Criterion 2, the verdict: no ``tool`` commit from the repair -> ``refused("no tool commit")``."""
    async with tool_rig(
        tmp_path, script=[answer(SPOKEN), tool(REPAIRABLE_CLICK)], continuation=[]
    ) as rig:
        await voice_turn(rig)
    r = draft_of(only_done(rig, REPAIR_ID))
    assert (r.status, r.reason, r.source) == ("refused", "no tool commit", "repair"), r
    assert rig.statuses() == ["proposed", "repairing", "refused"]
    assert final_message(rig).reason == "no tool commit"


async def test_c2_a_repair_timeout_ends_cancelled_and_aborts_the_repair(tmp_path):
    """Flow "repair timeout" (R6, H12 3000 ms, Q-c): ``cancelled`` with reason ``repair_timeout``, the engine abort
    for the repair's id, no execution; the turn still ends on its drain.

    Guard width: 100 ms before the deadline the card is still ``repairing`` (a timer that fires early would cut
    every repair); after it, ``cancelled``."""
    async with tool_rig(
        tmp_path,
        script=[answer(SPOKEN), tool(REPAIRABLE_CLICK)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        rig.llm.hold_continuation()
        await rig.ask(QUESTION)
        await rig.until_repair_live()
        done_t = only_done(rig, PASS1_ID).t_ms
        await rig.advance(max(0, done_t + 2900 - rig.clock.now_ms))
        assert rig.statuses()[-1] == "repairing", (
            f"guard width: 2900 ms after pass 1's llm.done the repair is still live; {rig.statuses()}"
        )
        await rig.until_status("cancelled", limit_ms=2_000)
        engine_id = one(rig.llm.continuations(), "repair request")[0]
        assert engine_id in rig.llm.aborted, (
            f"T-0092c criterion 2: LlmAbort(repair id) reaches the engine as {engine_id!r}; aborted {rig.llm.aborted}"
        )
        await rig.turn_done()
    final = final_message(rig)
    assert (final.status, final.reason) == ("cancelled", "repair_timeout"), final
    assert rig.executor.invocations == []
    assert rig.state.history[-1].assistant_spoken == SPOKEN
    assert_concurrency(rig)


async def test_c2_a_repair_http_error_cancels_the_call_and_speech_continues(tmp_path):
    """Flow "repair HTTP error" (R7): ``LlmError(repair id)``, the card ends ``refused`` with the repair error, the
    spoken prefix keeps playing to the end, and the turn is not failed (no error shown on the page)."""
    async with tool_rig(
        tmp_path,
        script=[answer(SPOKEN), tool(REPAIRABLE_CLICK)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        rig.llm.continuation_errors = [http_error(500, detail="engine fell over")]
        await voice_turn(rig)
    # Ruling I5: a repair failure does not fail the turn; the failure lives in TurnRecord.tool.
    record = rig.record("t1")
    assert record.tool is not None and record.tool.get("status") == "refused", (
        f"T-0092c I5: TurnRecord.tool carries the final status (refused); got {record.tool!r}"
    )
    assert "500" in str(record.tool.get("reason") or ""), (
        f"T-0092c I5: TurnRecord.tool carries the repair error as its reason; got {record.tool.get('reason')!r}"
    )
    assert record.outcome == "spoken", (
        f"T-0092c I5 (R7): the turn's normal outcome -- the prefix was played -- not 'error'; got {record.outcome!r}"
    )
    errors = [e for e in rig.machine.events(E.LlmError) if e.request_id == REPAIR_ID]
    assert len(errors) == 1, (
        f"T-0092c criterion 2: an exception from generate becomes LlmError({REPAIR_ID!r}); got "
        f"{rig.machine.events(E.LlmError)}"
    )
    assert rig.statuses() == ["proposed", "repairing", "refused"], rig.statuses()
    assert "500" in (final_message(rig).reason or ""), final_message(rig)
    assert rig.state.history[-1].assistant_spoken == SPOKEN, (
        "R7: the spoken prefix keeps playing; history holds all of it"
    )
    assert rig.commands(C.HudError) == [], "R7: the turn is not failed"
    assert rig.executor.invocations == []


async def test_c2_a_tokenize_error_becomes_llm_error_and_sends_no_repair(tmp_path):
    """Flow "a tokenize error": ``LlmError(repair id)``, no continuation, ``refused`` with the error."""
    async with tool_rig(
        tmp_path,
        script=[answer(SPOKEN), tool(REPAIRABLE_CLICK)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        rig.llm.tokenize_error = RuntimeError("tokenize exploded")
        await voice_turn(rig)
    assert [e.request_id for e in rig.machine.events(E.LlmError)] == [REPAIR_ID]
    assert rig.llm.continuations() == [], "no repair generate after a failed tokenize"
    assert rig.statuses() == ["proposed", "repairing", "refused"], rig.statuses()
    assert "tokenize exploded" in (final_message(rig).reason or "")
    assert rig.state.history[-1].assistant_spoken == SPOKEN


@pytest.mark.parametrize("guard", ["template", "marker"])
async def test_c2_a_continuation_guard_refusal_sends_no_generate(tmp_path, monkeypatch, guard):
    """Flow "a continuation_guard refusal": ``LlmDone(repair id, tool=refused("continuation_guard: <reason>"))``
    and no generate.

    Forced: after ``drop_special_literals`` the spoken prefix holds no channel marker and no literal, so neither
    guard can fire through the session (see the report); ``repair_prompt`` is replaced wherever it is bound."""
    patch_bound(monkeypatch, live, "repair_prompt", lambda *a, **k: live.Refusal(guard))
    async with tool_rig(
        tmp_path,
        script=[answer(SPOKEN), tool(REPAIRABLE_CLICK)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        await voice_turn(rig)
    r = draft_of(only_done(rig, REPAIR_ID))
    assert (r.status, r.reason, r.source) == (
        "refused",
        f"continuation_guard: {guard}",
        "repair",
    ), r
    assert rig.llm.continuations() == [], "a guard refusal sends no generate"
    assert rig.statuses() == ["proposed", "repairing", "refused"]
    assert final_message(rig).reason == f"continuation_guard: {guard}"


async def test_c2_repair_commits_are_logged_never_submitted_never_voiced(tmp_path):
    """Criterion 2 "commits are never voiced", adversarial "a repair whose output contains prose is never spoken":
    no repair commit becomes ``LlmCommit`` (so none reaches the gate or ``canvases``); every one is logged as
    ``llm.commit`` with the repair id; the thought is logged and never stored."""
    cont = [
        thought("fix the x coordinate", 0),
        answer("Sure, clicking it right now. ", 0),
        tool(REPAIRED_CLICK, canvas_idx=0),
    ]
    async with tool_rig(
        tmp_path, script=[answer(SPOKEN), tool(REPAIRABLE_CLICK)], continuation=cont
    ) as rig:
        await voice_turn(rig)
    late = [e for e in rig.machine.events(E.LlmCommit) if e.request_id == REPAIR_ID]
    assert late == [], (
        f"T-0092c criterion 2: no repair commit is submitted as LlmCommit; got {late}"
    )
    assert [e.text for e in rig.machine.events(E.LlmCommit)] == [SPOKEN]
    assert not any("clicking it right now" in t for t in rig.tts.texts()), rig.tts.texts()
    assert rig.record("t1").canvases == 1, "canvases counts only pass 1's submitted commit"
    logged = [r for r in rig.kind("llm.commit") if r.get("request_id") == REPAIR_ID]
    assert sorted(r["segment"] for r in logged) == ["answer", "thought", "tool"], (
        f"T-0092c criterion 2: every repair commit is logged with the repair id; got {logged}"
    )
    assert all("fix the x" not in h.assistant_spoken for h in rig.state.history)
    assert_never_voiced(rig, arguments={"x": 12, "y": 5})


async def test_c2_the_repair_request_is_logged_once(tmp_path):
    """Criterion 2 "request log": one ``llm.request {phase: "tool_repair", ...}`` with the ``CdRequest`` dump
    without ``automaton`` and with ``automaton_sha256``."""
    async with tool_rig(
        tmp_path,
        script=[answer(SPOKEN), tool(REPAIRABLE_CLICK)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        await voice_turn(rig)
    reqs = [r for r in rig.kind("llm.request") if r.get("phase") == "tool_repair"]
    assert len(reqs) == 1, f"T-0092c criterion 2: one tool_repair request record; got {len(reqs)}"
    req = reqs[0]
    engine_id = one(rig.llm.continuations(), "repair request")[0]
    assert (req.get("request_id"), req.get("attempt_id")) == (REPAIR_ID, engine_id), req
    assert set(req) >= {"phase", "request_id", "attempt_id", "options", "cd", "prompt"}, sorted(req)
    assert "automaton" not in req["cd"] and req["cd"].get("automaton_sha256") == (
        live.load_payloads().tools["click_at"].automaton_sha256
    ), req["cd"]
    assert req["options"].get("max_tokens") == 256


async def test_c2_pass1_prompt_as_sent_is_the_over_long_retry(tmp_path):
    """Criterion 2: ``pass1_prompt_as_sent`` is the exact prompt pass 1 sent, including after an over-long retry
    (T-0020: the 413 retry drops one history turn, so its messages differ from the first attempt's)."""
    async with tool_rig(tmp_path, script=[answer("A login page. ")]) as rig:
        await voice_turn(rig, "what is this")
        rig.llm.script = [answer(SPOKEN), tool(REPAIRABLE_CLICK)]
        rig.llm.script_continuation(tool(REPAIRED_CLICK, canvas_idx=0))
        rig.llm.fail_generates(http_error(413), None)
        await voice_turn(rig)
    bodies = rig.llm.pass1_bodies()
    assert len(bodies) == 3, (
        f"premise: turn 1, turn 2's rejected attempt and its retry; got {len(bodies)}"
    )
    assert bodies[1]["messages"] != bodies[2]["messages"], "premise: the retry dropped history"
    cont = one(rig.llm.continuation_bodies(), "repair body")
    assert cont["messages"][:-1] == bodies[2]["messages"], (
        "T-0092c criterion 2: the repair continues the retry's prompt, the one pass 1 actually sent"
    )


# =====================================================================================================
# Criterion 3: execution
# =====================================================================================================


async def test_c3_the_frame_is_the_sent_prompts_newest_image_not_a_later_snapshot(tmp_path):
    """Criterion 3: ``frame`` is the geometry of the newest image of the prompt pass 1 sent, looked up by phash --
    never ``_turn_frames()`` at execute time. A later snapshot (1920x1080) lands before the drain."""
    a = frame(0, "a")
    async with tool_rig(tmp_path, script=[answer(SPOKEN), tool(VALID_CLICK)], frames=[a]) as rig:
        await rig.ask(QUESTION)
        await rig.until(lambda: rig.state.llm_done, "pass 1 never finished")
        assert rig.state.state is TurnState.SPEAKING, "premise: the prefix is still playing"
        later = frame(rig.clock.now_ms, "b", width=1920, height=1080)
        rig.perception.add(later)
        await rig.session.executor.execute(C.SnapshotFrame(at_ms=rig.clock.now_ms))
        await rig.turn_done()
    inv = one(rig.executor.invocations, "executor invocation")
    assert inv.frame is not None, "T-0092c criterion 3: the sent prompt carried an image"
    assert (inv.frame.phash, inv.frame.width, inv.frame.height, inv.frame.source.value) == (
        a.phash,
        1280,
        720,
        "screen",
    ), f"T-0092c criterion 3: the frame pass 1 sent, not the later snapshot; got {inv.frame!r}"


async def test_c3_with_no_image_a_click_is_refused_no_frame(tmp_path):
    """Criterion 3: no image in the sent prompt -> ``frame`` None -> the dry run refuses a click with ``no frame``.
    A snapshot after pass 1 does not supply one."""
    executor = WatchingExecutor(inner=LS.dry_run())
    async with tool_rig(
        tmp_path, script=[answer(SPOKEN), tool(VALID_CLICK)], frames=[], executor=executor
    ) as rig:
        await rig.ask(QUESTION)
        await rig.until(lambda: rig.state.llm_done, "pass 1 never finished")
        rig.perception.add(frame(rig.clock.now_ms, "b"))
        await rig.session.executor.execute(C.SnapshotFrame(at_ms=rig.clock.now_ms))
        await rig.turn_done()
    inv = one(executor.invocations, "executor invocation")
    assert inv.frame is None, (
        f"T-0092c criterion 3: no image was sent, so no frame; got {inv.frame!r}"
    )
    final = final_message(rig)
    assert (final.status, final.reason) == ("refused", "no frame"), final


async def test_c3_an_executor_exception_publishes_refused_executor_error(tmp_path):
    """Criterion 3: an executor that raises -> ``refused(reason="executor error")`` and an ``error`` record."""
    executor = WatchingExecutor(raise_=RuntimeError("the executor blew up"))
    async with tool_rig(
        tmp_path, script=[answer(SPOKEN), tool(VALID_CLICK)], executor=executor
    ) as rig:
        await voice_turn(rig)
    final = final_message(rig)
    assert (final.status, final.reason) == ("refused", "executor error"), final
    assert any("blew up" in str(r) for r in rig.kind("error")), (
        f"T-0092c criterion 3: the exception is logged as an error; errors {rig.kind('error')}"
    )


async def test_g8_a_valid_call_executes_and_no_play_follows_in_the_same_turn(tmp_path):
    """Carried from T-0092a (ruling G8): the dry run speaks no tool result, so the "same-turn drift" case (a turn that
    speaks again after its drain is marked) stays unreachable: no ``play()`` and no ``tts.speak`` after the
    execution, in that turn."""
    async with tool_rig(tmp_path, script=[answer(SPOKEN), tool(VALID_CLICK)]) as rig:
        await voice_turn(rig)
        plays_at_end = len(rig.audio.plays)
    assert rig.executor.plays_at_call, "T-0092c criterion 3: the call executed"
    assert rig.executor.plays_at_call == [plays_at_end], (
        f"G8: no play() after the execution in the same turn; plays at the call {rig.executor.plays_at_call}, "
        f"at the end of the turn {plays_at_end}"
    )
    tuples = [cmds for _, _, cmds in rig.machine.log]
    at = next(i for i, cmds in enumerate(tuples) if any(isinstance(c, C.ToolExecute) for c in cmds))
    after = [c for cmds in tuples[at:] for c in cmds if isinstance(c, C.TtsSpeak)]
    assert after == [], f"G8: nothing is spoken from the execution on; got {after}"


# =====================================================================================================
# Criterion 4: cancel (Q-c)
# =====================================================================================================


async def test_c4_a_barge_in_during_the_spoken_prefix_cancels_the_valid_call(tmp_path):
    """Adversarial: a barge-in (onset -> PAUSED -> takeover) during the spoken prefix of a valid call ->
    ``cancelled(interrupted)``, and the executor is never called."""
    long_answer = "I will click the Sign in button at the top of the page for you right now. "
    async with tool_rig(tmp_path, script=[answer(long_answer), tool(VALID_CLICK)]) as rig:
        await rig.ask(QUESTION)
        await rig.until(lambda: rig.state.llm_done, "pass 1 never finished")
        assert rig.state.state is TurnState.SPEAKING, "premise: the prefix is playing"
        await rig.until_status("proposed")
        now = rig.clock.now_ms
        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
        await rig.advance(800)
        await rig.until(lambda: "cancelled" in rig.statuses(), "the call was never cancelled")
    final = final_message(rig)
    assert (final.status, final.reason) == ("cancelled", "interrupted"), final
    assert rig.executor.invocations == [], (
        "T-0092c criterion 8: never executed in an interrupted turn"
    )
    assert rig.commands(C.ToolExecute) == []


@pytest.mark.parametrize("how", ["tts_error", "disconnected"])
async def test_c4_a_closing_turn_cancel_is_written_to_the_closing_turns_record(tmp_path, how):
    """Criterion 4: ``error``/``disconnected`` publish ``cancelled`` with the cancel reason, and a cancel that
    arrives in the closing tuple is written to the closing turn's record (``self._closing``)."""
    long_answer = "I will click the Sign in button at the top of the page for you right now. "
    async with tool_rig(tmp_path, script=[answer(long_answer), tool(VALID_CLICK)]) as rig:
        await rig.ask(QUESTION)
        await rig.until(lambda: rig.state.llm_done, "pass 1 never finished")
        await rig.until_status("proposed")
        now = rig.clock.now_ms
        if how == "tts_error":
            rig.submit(E.TtsError(t_ms=now, utterance_id="t1-u9", message="tts down"))
            reason = "error"
        else:
            rig.submit(E.ClientDisconnected(t_ms=now))
            reason = "disconnected"
        await rig.turn_done()
    final = final_message(rig)
    assert (final.status, final.reason) == ("cancelled", reason), final
    rec = rig.record("t1")
    assert rec.tool is not None and rec.tool.get("status") == "cancelled", (
        f"T-0092c criteria 4/7: the closing turn's record holds the cancelled call; got {rec.tool!r}"
    )
    assert rig.executor.invocations == []


async def test_c4_a_thinking_takeover_during_the_repair_cancels_with_takeover(tmp_path):
    """Criterion 4 / R8: a repair started in THINKING (nothing spoken) and a takeover -> the repair is aborted and
    the call ends ``cancelled(takeover)``."""
    async with tool_rig(
        tmp_path,
        script=[tool(REPAIRABLE_CLICK, canvas_idx=0)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        rig.llm.hold_continuation()
        await rig.ask(QUESTION)
        await rig.until_repair_live()
        assert rig.state.state is TurnState.THINKING, "premise: nothing was spoken"
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.until(lambda: "cancelled" in rig.statuses(), "the call was never cancelled")
        engine_id = one(rig.llm.continuations(), "repair request")[0]
        assert engine_id in rig.llm.aborted, rig.llm.aborted
    final = final_message(rig)
    assert (final.status, final.reason) == ("cancelled", "takeover"), final
    assert_concurrency(rig)


# =====================================================================================================
# Criterion 6: speech and history
# =====================================================================================================


async def test_c6_a_spliced_literal_is_never_voiced_never_stored_and_the_repair_is_sent(tmp_path):
    """Adversarial (T-0088 E11 note): ``...<|t`` / ``ool>...`` across two answer commits splice ``<|tool>``.

    The session runs ``drop_special_literals`` on the concatenated text: the repair prompt is sent (the guard does
    not refuse it), its continuation holds no literal, nothing voiced holds one and history holds none."""
    script = [answer("Sure<|t", 0), answer("ool> thing. ", 1), tool(REPAIRABLE_CLICK, canvas_idx=1)]
    async with tool_rig(
        tmp_path, script=script, continuation=[tool(REPAIRED_CLICK, canvas_idx=0)]
    ) as rig:
        await voice_turn(rig)
    # Ruling I4: tools-on answer commits are cleaned before they are submitted -- one by one and concatenated.
    submitted = [e.text for e in rig.machine.events(E.LlmCommit) if e.segment == "answer"]
    assert submitted, "premise: the answer reached the machine"
    for text in submitted:
        assert literal_in(text) == [], (
            f"T-0092c I4: an answer commit submitted to the machine holds {literal_in(text)}"
        )
    assert literal_in("".join(submitted)) == [], (
        f"T-0092c I4: the submitted answer commits, concatenated, hold {literal_in(''.join(submitted))}: {submitted}"
    )
    assert "".join(submitted) == "Sure thing. ", submitted
    bodies = rig.llm.continuation_bodies()
    assert len(bodies) == 1, (
        "T-0092c criterion 1 (E11): spoken_prefix = drop_special_literals(concatenated answers), so the "
        f"continuation guard does not refuse; repair requests sent {len(bodies)}, statuses {rig.statuses()}"
    )
    content = bodies[0]["messages"][-1]["content"]
    assert content == "Sure thing. <|tool_call>", content
    assert literal_in(content.removesuffix("<|tool_call>")) == []
    assert_never_voiced(rig, what=" (spliced literal)")
    assert_history_clean(rig)


@pytest.mark.parametrize("stop_id", [50, 106])
async def test_c6_the_spoken_prefix_and_history_hold_only_the_answer(tmp_path, stop_id):
    """Criterion 6: the concatenated ``TtsSpeak.source`` is the spoken prefix; history is the played text only."""
    async with tool_rig(
        tmp_path, script=[answer(SPOKEN), tool(VALID_CLICK, stop_id=stop_id)]
    ) as rig:
        await voice_turn(rig)
    assert rig.spoken_sources() == SPOKEN
    assert rig.state.history[-1].assistant_spoken == SPOKEN
    assert rig.statuses() == ["proposed", "dry_run"], rig.statuses()


# =====================================================================================================
# Criterion 7: telemetry
# =====================================================================================================


async def test_c7_one_tool_call_record_per_published_message_with_its_keys(tmp_path):
    """Criterion 7: ``tool.call`` -- one record per published ``astra.tool.call``, the same status, in order, with
    ``{call_id, source, name, arguments, parse, schema_valid, query_problem, repair_ms, action, reason, resolved,
    spoken_prefix}``."""
    async with tool_rig(
        tmp_path,
        script=[answer(SPOKEN), tool(REPAIRABLE_CLICK)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        await voice_turn(rig)
    records = rig.tool_records()
    assert (
        [r.get("status") for r in records] == rig.statuses() == ["proposed", "repairing", "dry_run"]
    ), (
        f"T-0092c criterion 7: one tool.call record per publish; records {[r.get('status') for r in records]}, "
        f"published {rig.statuses()}"
    )
    for r in records:
        # I13 (reviewer 2): the keys are exact -- criterion 7 names them all, so an extra key is a contract change
        assert set(r) == TOOL_CALL_KEYS, (
            f"T-0092c criterion 7 / I13: the tool.call keys are exact; missing {sorted(TOOL_CALL_KEYS - set(r))}, "
            f"extra {sorted(set(r) - TOOL_CALL_KEYS)}"
        )
        assert r["call_id"] == CALL_ID and r["spoken_prefix"] == SPOKEN, r
    assert records[-1]["source"] == "repair" and records[-1]["action"] == "recorded click_at"
    assert rig.record("t1").tool.get("status") == "dry_run"


async def test_c7_repair_ms_and_the_spans_are_on_the_session_clock(tmp_path):
    """Criterion 7: ``repair_ms`` from session clock to session clock; ``tool.repair`` from ``ToolRepairStart``
    handled to the repair's ``LlmDone`` submitted; ``tool.execute``; and the ``llm.done`` span stays pass 1's.

    The repair is held 400 ms of virtual time, so a duration taken from any other clock reads about 0."""
    async with tool_rig(
        tmp_path,
        script=[answer(SPOKEN), tool(REPAIRABLE_CLICK)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        rig.llm.hold_continuation()
        await rig.ask(QUESTION)
        await rig.until_repair_live()
        t_gen = next(
            t
            for kind, rid, t in rig.llm.calls
            if kind == "generate" and rid == one(rig.llm.continuations(), "repair request")[0]
        )
        done_t = only_done(rig, PASS1_ID).t_ms
        await rig.advance(400)
        t_release = rig.clock.now_ms
        rig.llm.release_continuation()
        await rig.turn_done()
    spans = rig.record("t1").spans
    assert "tool.repair" in spans and "tool.execute" in spans, (
        f"T-0092c criterion 7: spans {sorted(spans)}"
    )
    start, end = spans["tool.repair"]
    assert done_t <= start <= t_gen and end == t_release, (
        f"tool.repair runs from the handled ToolRepairStart ({done_t}..{t_gen}) to the repair's LlmDone "
        f"({t_release}); got {(start, end)}"
    )
    final = rig.tool_records()[-1]
    assert final.get("repair_ms") == end - start and final["repair_ms"] >= 400, (
        f"T-0092c criterion 7: repair_ms is session clock to session clock; got {final.get('repair_ms')}"
    )
    assert spans["llm.done"][1] <= t_gen < t_release, (
        f"T-0092c criterion 7: the llm.done span stays pass 1's; got {spans['llm.done']}"
    )
    ex0, ex1 = spans["tool.execute"]
    assert t_release <= ex0 <= ex1, spans["tool.execute"]
    assert rig.record("t1").unknown_spans() == []


async def test_c7_the_wire_order_and_the_journal_replay(tmp_path):
    """Wire publish order ``proposed -> repairing -> dry_run``, read through the real ``TurnJournal``: a page that
    reconnects after the turn is replayed exactly the final state of the call (T-0090 criterion 3, U10)."""
    from local_astra.conductor.hud_server import TurnJournal

    async with tool_rig(
        tmp_path,
        script=[answer(SPOKEN), tool(REPAIRABLE_CLICK)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        await voice_turn(rig)
    assert rig.statuses() == ["proposed", "repairing", "dry_run"], rig.statuses()
    journal = TurnJournal()
    for m in rig.published.messages:
        journal.record(m)
    replayed = [m for m in journal.messages() if getattr(m, "type", None) == "astra.tool.call"]
    assert [(m.call_id, m.status, m.source) for m in replayed] == [
        (CALL_ID, "dry_run", "repair")
    ], replayed
    assert all(m.replay is False for m in rig.tool_calls()), "live messages carry replay=False"


# =====================================================================================================
# Criterion 8 and the adversarial rows
# =====================================================================================================


async def test_c8_a_barge_in_during_the_repair_cancels_its_task_and_aborts_the_engine(tmp_path):
    """Adversarial: a barge-in during the repair -> the repair task is cancelled, the engine abort is sent for the
    repair's id, ``cancelled(interrupted)``, and at most one generate is live at every step."""
    long_answer = "I will click the Sign in button at the top of the page for you right now. "
    async with tool_rig(
        tmp_path,
        script=[answer(long_answer), tool(REPAIRABLE_CLICK)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        rig.llm.hold_continuation()
        await rig.ask(QUESTION)
        await rig.until_repair_live()
        assert rig.state.state is TurnState.SPEAKING, (
            "premise: the prefix is playing during the repair"
        )
        now = rig.clock.now_ms
        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
        await rig.advance(800)
        await rig.until(lambda: "cancelled" in rig.statuses(), "the call was never cancelled")
        engine_id = one(rig.llm.continuations(), "repair request")[0]
        assert engine_id in rig.llm.aborted, (
            f"the engine abort for {engine_id}; aborted {rig.llm.aborted}"
        )
        assert engine_id not in rig.llm.live_generates
        rig.llm.release_continuation()
        await rig.advance(200)
    final = final_message(rig)
    assert (final.status, final.reason) == ("cancelled", "interrupted"), final
    assert [d for d in rig.dones(REPAIR_ID)] == [], "the cancelled repair submits no LlmDone"
    assert rig.executor.invocations == []
    assert_concurrency(rig)


async def test_c8_a_stale_pass1_llm_done_during_the_repair_is_ignored(tmp_path):
    """Adversarial: a second pass-1 ``llm.done`` (with its own valid call) arriving while the repair runs is stale
    (R10): the repair's verdict alone decides, and the stale call is never executed or published."""
    async with tool_rig(
        tmp_path,
        script=[answer(SPOKEN), tool(REPAIRABLE_CLICK)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        rig.llm.hold_continuation()
        await rig.ask(QUESTION)
        await rig.until_repair_live()
        stale = ToolDraft(
            call_id=CALL_ID, status="valid", name="click_at", arguments={"x": 999, "y": 999}
        )
        rig.submit(E.LlmDone(t_ms=rig.clock.now_ms, request_id=PASS1_ID, tool=stale))
        await rig.pump()
        rig.llm.release_continuation()
        await rig.turn_done()
    assert [i.arguments for i in rig.executor.invocations] == [{"x": 12, "y": 5}], (
        rig.executor.invocations
    )
    assert rig.statuses() == ["proposed", "repairing", "dry_run"], rig.statuses()
    assert all(m.arguments != {"x": 999, "y": 999} for m in rig.tool_calls())


async def test_c8_a_stale_drain_from_the_previous_turn_does_not_execute_this_turns_call(tmp_path):
    """Adversarial, T-0092a x tools: turn 2 commits, gets its valid ``llm.done``, then turn 1's held drain arrives
    -> no ``ToolExecute`` until turn 2's own drain. The session's drain gate stays."""
    from tests.unit.test_session_stale_playback import StaleAudio

    s1, s2 = "Turn one says this. ", "Second answer here. "
    async with tool_rig(tmp_path, script=[answer(s1)], audio_cls=StaleAudio) as rig:
        await rig.ask("first question")
        await rig.until(
            lambda: rig.state.llm_done and rig.state.state is TurnState.SPEAKING, "t1 never spoke"
        )
        await rig.until(lambda: "mark_drain" in rig.audio.calls, "turn 1's drain was never marked")
        rig.audio.hold_events()
        await rig.until(
            lambda: any(isinstance(e, E.PlaybackDrained) for e in rig.audio.pending_events()),
            "turn 1's run never drained",
        )
        rig.submit(E.TtsError(t_ms=rig.clock.now_ms, utterance_id="t1-late", message="tts down"))
        await rig.pump()
        assert rig.state.state is TurnState.IDLE, "premise: tts.error ends turn 1"
        rig.llm.script = [answer(s2), tool(VALID_CLICK)]
        rig.submit(E.UserText(t_ms=rig.clock.now_ms, text="second question"))
        await rig.until(
            lambda: (
                rig.state.turn_id == "t2"
                and rig.state.state is TurnState.SPEAKING
                and rig.state.llm_done
            ),
            "t2 never reached SPEAKING with llm.done",
        )
        await rig.until_status("proposed", "t2-call")
        rig.audio.release_events()
        await rig.pump()
        assert rig.executor.invocations == [] and rig.commands(C.ToolExecute) == [], (
            "T-0092c criterion 8: turn 1's stale drain must not execute turn 2's call"
        )
        assert rig.state.state is TurnState.SPEAKING and rig.state.turn_id == "t2"
        await rig.turn_done()
    assert [i.call_id for i in rig.executor.invocations] == ["t2-call"], rig.executor.invocations
    assert rig.statuses("t2-call") == ["proposed", "dry_run"]


#: T-0092c's merge commit b48a707; BASE = ``git merge-base b48a707^1 b48a707^2`` = 4f8d4e0 (= ``BASE_SHA``), TIP =
#: ``b48a707^2`` (T-0094b criterion 2). The uncommitted-changes half is retired with the working-tree comparison.
T0092C_MERGE = "b48a707"
T0092C_BASE = "4f8d4e0"
T0092C_TIP = "f827f33"
T0092C_UNCHANGED = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/contracts",
    "local_astra/conductor/llm_client.py",
    "local_astra/toolcall",
    "local_astra/conductor/tool_executor.py",
    "local_astra/conductor/hud_server.py",
)
#: Power pair (T-0094b ruling P2): T-0091's d55cfce (parent a7c00a8) edited ``toolcall/live.py``.
T0092C_PRIOR_PARENT = "a7c00a8"
T0092C_PRIOR = "d55cfce"


async def test_c8_the_owned_scope_is_unchanged_from_the_base():
    """Criterion 8, negative: ``turn_machine.py``, ``contracts/**``, ``llm_client.py``, ``toolcall/**``,
    ``tool_executor.py`` and ``hud_server.py`` do not differ between T-0092c's BASE (the literal base ``4f8d4e0``) and
    its TIP (T-0094b: a closed range, never the working tree). Green at the base."""
    assert T0092C_BASE == BASE_SHA
    gr.prove_range(T0092C_BASE, T0092C_TIP, T0092C_MERGE)
    changed = gr.changed(T0092C_BASE, T0092C_TIP, *T0092C_UNCHANGED)
    assert changed == [], f"T-0092c criterion 8: changed outside the owned files:\n{changed}"


def test_the_c8_owned_scope_guard_has_power():
    """T-0094b criterion 5 (adversarial): over d55cfce's parent and d55cfce the same predicate reports ``live.py`` under
    the toolcall entry of the tuple; a predicate that reports nothing fails here."""
    gr.require_commits(T0092C_PRIOR_PARENT, T0092C_PRIOR)
    assert gr.is_ancestor(T0092C_PRIOR, T0092C_BASE)
    toolcall = next(p for p in T0092C_UNCHANGED if p.endswith("/toolcall"))
    assert f"{toolcall}/live.py" in gr.changed(T0092C_PRIOR_PARENT, T0092C_PRIOR, *T0092C_UNCHANGED)


# =====================================================================================================
# Carried from the T-0092b review: "kept for the turn", including the stale-generate case
# =====================================================================================================


class StubbornLlm(ToolLlm):
    """A backend whose first turn stream delivers one more chunk after its consumer is cancelled.

    Deliberately adversarial: the stream yields a thought, then waits; when the session cancels it (the abort of a
    THINKING takeover), it yields a buffered ``tool`` commit before ending. That commit belongs to the aborted
    request, whose turn is over -- the session must log it and never make it the next turn's call."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.late = 0
        self._first = True

    async def generate(
        self,
        prompt: Any,
        request_id: str,
        options: Any = None,
        *,
        xargs: Any = None,
        cd: Any = None,
    ):  # type: ignore[override]
        from tests.fakes.llm import is_caption_prompt

        if self._first and not is_caption_prompt(prompt):
            self._first = False
            self.gen_prompts.append((request_id, prompt))
            self.live_generates.add(request_id)
            self._note("generate", request_id)
            try:
                yield thought("thinking about the button").model_copy(
                    update={"request_id": request_id}
                )
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.late += 1
                    yield tool(VALID_CLICK).model_copy(update={"request_id": request_id})
                    raise
            finally:
                self.live_generates.discard(request_id)
            return
        async for c in super().generate(prompt, request_id, options, xargs=xargs, cd=cd):
            yield c


async def test_carried_a_stale_generates_tool_commit_is_logged_and_never_the_next_turns_call(
    tmp_path,
):
    """T-0092b criterion 4, the stale-generate case: a ``tool`` commit from a request that is no longer the live
    generate (aborted by a THINKING takeover, delivered late) is logged, and is never kept for the turn now open --
    turn 2's ``llm.done`` carries no tool, nothing is published or executed."""
    llm = StubbornLlm([answer("Second answer here. ")])
    async with tool_rig(tmp_path, llm=llm) as rig:
        await rig.ask(QUESTION)
        await rig.until(
            lambda: rig.state.state is TurnState.THINKING and rig.machine.events(E.LlmCommit),
            "t1 stuck",
        )
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.until(
            lambda: llm.late == 1, "the stubborn stream never delivered its late commit"
        )
        assert rig.state.turn_id == "t2", "premise: the takeover opened turn 2"
        await rig.ask("and the second question", onset=False)
        await rig.turn_done()
    t2_dones = rig.dones("t2-gen")
    assert t2_dones and all(d.tool is None for d in t2_dones), (
        f"T-0092b criterion 4: the stale request's tool commit is not turn 2's call; LlmDone(t2-gen) {t2_dones}"
    )
    assert rig.tool_calls() == [] and rig.executor.invocations == []
    assert rig.record("t2").tool is None
    stale = [r for r in rig.kind("llm.commit") if r.get("segment") == "tool"]
    assert len(stale) == 1 and stale[0]["request_id"] == PASS1_ID, (
        f"T-0092b criterion 4: the late tool commit is still logged; got {stale}"
    )


# =====================================================================================================
# Rulings after review (I10-I13): the rows of I13
#
# Each row names its review item. Rows that pin a defect at 1783b8b are red there on a named assertion (P4, P1,
# P5); the others pin behaviour that is correct at 1783b8b and each kills the reviewer's named mutant (M8, M6a,
# M7, M21, M26, M27, the tools-off inert-handler mutant). Oracles: hand-computed text, the tools-off session over
# the same script (no cleaner exists there), and drop_special_literals over the whole concatenation (the
# definition in criterion 1 and ruling I4).
# =====================================================================================================

from local_astra.contracts.llm import Commit  # noqa: E402
from local_astra.toolcall.special_tokens import drop_special_literals  # noqa: E402

#: the thought words of the P4 rows: none may be voiced or stored
SECRET = "secret"


def answer_events(rig: Any, request_id: str = PASS1_ID) -> list[Any]:
    return [
        e
        for e in rig.machine.events(E.LlmCommit)
        if e.request_id == request_id and e.segment == "answer"
    ]


def open_thought(text: str, canvas_idx: int = 0) -> Commit:
    """An ``answer`` commit whose canvas ends inside an unclosed thought (``thought_open=True``)."""
    return Commit(
        request_id="?", canvas_idx=canvas_idx, segment="answer", text=text, thought_open=True
    )


# --- I10 / P4: the cleaner never crosses a thought boundary ----------------------------------------------------


@pytest.mark.parametrize("tools", ["on", "off"])
@pytest.mark.parametrize(
    "inside",
    [
        # the reviewer's probe: a '<' then a space (I12 would release it; the I4 rule at 1783b8b holds it)
        f"Note that 3 < 5 {SECRET}",
        # a held tail made only of literal characters, so an I12-only fix still holds it and the splice stays
        f"the {SECRET} plan is <hidden",
    ],
    ids=["lt-space", "lt-literal-chars"],
)
async def test_i10_p4_text_inside_an_unclosed_thought_is_never_voiced_or_stored(
    tmp_path, tools, inside
):
    """I13/P4 (reviewer 1, finding 1, HIGH; CLAUDE.md: never voice a canvas that ends inside an unclosed thought).

    Canvas 0 ends inside an unclosed thought (``thought_open=True``); canvas 1 is a closed answer, ``so yes. ``.
    Ruling I10: a ``thought_open=True`` commit is submitted unchanged and never goes through the cleaner; held text
    keeps the flag of its commit. So with tools on, as with tools off, the voiced text and history are ``so yes. ``
    and hold no word of the open thought; no ``thought_open=False`` commit carries text from canvas 0.

    Red at 1783b8b with tools on: the cleaner held ``< 5 secret`` (``<hidden``) from canvas 0 and released it inside
    canvas 1's ``thought_open=False`` commit, which was voiced and stored."""
    kit = "default" if tools == "on" else None
    script = [open_thought(inside, 0), answer("so yes. ", 1)]
    async with tool_rig(tmp_path, script=script, kit=kit) as rig:
        await voice_turn(rig)
    voiced = rig.spoken_sources()
    stored = rig.state.history[-1].assistant_spoken
    assert SECRET not in voiced and "hidden" not in voiced and "<" not in voiced, (
        f"I10 (tools {tools}): text inside an unclosed thought was voiced: {voiced!r}"
    )
    assert SECRET not in stored and "hidden" not in stored, (
        f"I10 (tools {tools}): text inside an unclosed thought was stored in history: {stored!r}"
    )
    assert (voiced, stored) == ("so yes. ", "so yes. "), (voiced, stored)
    events = answer_events(rig)
    opened = [e for e in events if e.thought_open]
    assert [e.text for e in opened] == [inside], (
        f"I10: a thought_open=True commit is submitted unchanged; got {[e.text for e in opened]}"
    )
    leaked = [e.text for e in events if not e.thought_open and SECRET in e.text]
    assert leaked == [], (
        f"I10: held thought text crossed into a thought_open=False commit: {leaked}"
    )


# --- I11 / P1: a takeover before the machine accepts the draft ----------------------------------------------


class RacingLlm(ToolLlm):
    """Runs ``on_exhausted(request_id)`` once, right after pass 1's stream yields its last commit and before the
    generator returns -- so an event submitted there is queued *before* the session's ``LlmDone`` for that request
    (the STT event that lands while the pump is busy). Nothing awaits in between, so the order is deterministic."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.on_exhausted: Any = None

    async def generate(  # type: ignore[override]
        self,
        prompt: Any,
        request_id: str,
        options: Any = None,
        *,
        xargs: Any = None,
        cd: Any = None,
    ):
        from tests.fakes.llm import is_caption_prompt

        async for c in super().generate(prompt, request_id, options, xargs=xargs, cd=cd):
            yield c
        hook, continuation = self.on_exhausted, getattr(prompt, "continue_final_message", False)
        if hook is not None and not continuation and not is_caption_prompt(prompt):
            # ``request_id`` is the engine id here; the first turn generate is pass 1 of t1
            self.on_exhausted = None
            hook(request_id)


async def test_i11_p1_a_takeover_queued_before_pass1s_done_never_leaves_the_card_at_proposed(
    tmp_path,
):
    """I13/P1 (reviewer 1, finding 2). A ``UserTakeover`` is queued after pass 1's last commit and before its
    ``LlmDone(tool=valid)``: the machine closes t1 and the late ``LlmDone`` is stale, so it never accepts the draft
    and never emits ``ToolCancel``. Ruling I11: ``proposed`` is published only when the machine accepts, and a turn
    that closes cancels its calls with no final (``turn closed``). So the card is either never shown or ends with one
    final, ``cancelled`` -- never stuck at ``proposed``; the call is never executed.

    Red at 1783b8b: ``proposed`` is published from the generate task, and nothing ever ends the card."""
    long_answer = "I will click the Sign in button at the top of the page for you right now. "
    llm = RacingLlm([answer(long_answer), tool(VALID_CLICK)])
    async with tool_rig(tmp_path, llm=llm) as rig:

        def takeover(_: str) -> None:
            now = rig.clock.now_ms
            rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))

        llm.on_exhausted = takeover
        await rig.ask(QUESTION)
        await rig.until(lambda: llm.on_exhausted is None, "pass 1 never reached its end")
        await rig.advance(3000)
        await rig.pump()
        order = [
            type(e).__name__
            for e in rig.machine.events()
            if isinstance(e, E.UserTakeover | E.LlmDone)
        ]
        assert order[:2] == ["UserTakeover", "LlmDone"], (
            f"premise: the takeover reached the machine before pass 1's LlmDone; got {order}"
        )
        assert rig.state.turn_id != "t1", "premise: the takeover closed t1"
    statuses = rig.statuses()
    if statuses:
        final = final_message(rig)
        assert (final.status, final.reason) == ("cancelled", "turn closed"), (
            f"I11: a call of a closed turn without a final ends cancelled('turn closed'); published {statuses}, "
            f"final {final!r}"
        )
    assert rig.executor.invocations == [] and rig.commands(C.ToolExecute) == []
    t1 = rig.record("t1").tool
    assert t1 is None or t1.get("status") == "cancelled", (
        f"I11: TurnRecord.tool is never left at an intermediate state; got {t1!r}"
    )


# --- I12 / P5: a maths answer is voiced before pass 1 ends -------------------------------------------------------

MATHS = ["Since 3 < 5, ", "the first step is to subtract. ", "Then divide by two. ", "Done."]
FIRST_SENTENCE = "Since 3 < 5, the first step is to subtract."


async def _voiced_while_pass1_is_held(tmp_path: Path, kit: Any) -> tuple[str, Any]:
    async with tool_rig(
        tmp_path, script=[answer(p, i) for i, p in enumerate(MATHS)], kit=kit
    ) as rig:
        rig.llm.hold_after(3)
        await rig.ask("what is the first step")
        await rig.advance(3000)
        assert rig.dones(PASS1_ID) == [], "premise: pass 1 is still streaming (held before 'Done.')"
        before = rig.spoken_sources()
        rig.llm.release()
        await rig.turn_done()
    return before, rig


async def test_i12_p5_a_maths_answer_is_voiced_before_pass1s_llm_done(tmp_path):
    """I13/P5 (reviewer 1, finding 3; ruling I12). Text with a ``<`` that a non-literal character follows can never
    become a literal, so it is released at once: "Since 3 < 5, the first step is to subtract." reaches ``TtsSpeak``
    while pass 1 is still streaming, before its ``llm.done`` -- and tools on voices, while held, exactly what tools off
    voices (the oracle: no cleaner exists with tools off).

    Red at 1783b8b: the I4 rule holds everything from the ``<`` (no ``>`` follows), so nothing is voiced until
    pass 1 ends."""
    off_before, _ = await _voiced_while_pass1_is_held(tmp_path / "off", None)
    assert off_before.startswith(FIRST_SENTENCE), (
        f"premise (tools off): the first sentence is voiced while pass 1 is held; got {off_before!r}"
    )
    on_before, rig = await _voiced_while_pass1_is_held(tmp_path / "on", "default")
    assert on_before.startswith(FIRST_SENTENCE), (
        f"I12: with tools on the first sentence reaches TtsSpeak before pass 1's llm.done; voiced while held "
        f"{on_before!r}"
    )
    assert on_before == off_before, (
        f"I12: tools on voices what tools off voices while pass 1 streams; on {on_before!r}, off {off_before!r}"
    )
    log = rig.machine.log
    done_at = next(
        i
        for i, (ev, _, _) in enumerate(log)
        if isinstance(ev, E.LlmDone) and ev.request_id == PASS1_ID
    )
    spoke_at = next(
        i
        for i, (_, _, cmds) in enumerate(log)
        if any(isinstance(c, C.TtsSpeak) and "subtract" in c.source for c in cmds)
    )
    assert spoke_at < done_at, (
        f"I12: the first sentence's TtsSpeak (step {spoke_at}) precedes pass 1's llm.done (step {done_at})"
    )
    assert rig.spoken_sources() == "".join(MATHS)


# --- I13 / P3, M8: the end-of-stream flush --------------------------------------------------------------------


async def test_i13_m8_a_tail_held_to_the_end_of_the_stream_is_released_there(tmp_path):
    """I13/P3 (mutant M8: drop the end-of-stream flush). The answer ends in ``a<b``: every character after the last
    space is a literal character, so under both I4 and I12 ``<b`` may still start a literal and is held until the
    end of the response, where it is released (I4) as a ``thought_open=False`` commit (I10). Expected by hand: the
    whole answer, voiced and stored (it holds no literal, so ``drop_special_literals`` leaves it as it is)."""
    parts = ["Compare the two values. ", "The answer is a<b"]
    whole = "".join(parts)
    assert drop_special_literals(whole) == whole, "premise: no literal in the answer"
    async with tool_rig(tmp_path, script=[answer(p, i) for i, p in enumerate(parts)]) as rig:
        await voice_turn(rig)
    events = answer_events(rig)
    assert "".join(e.text for e in events) == whole, (
        f"I4: the held tail is released at the end of the response; submitted {[e.text for e in events]}"
    )
    assert events[-1].text.endswith("<b") and events[-1].thought_open is False, (
        f"I10: the end-of-stream tail commit is thought_open=False; got {events[-1]!r}"
    )
    assert rig.spoken_sources() == whole, rig.spoken_sources()
    assert rig.state.history[-1].assistant_spoken == whole


# --- I13 / P2, M6a: the frame at judge time --------------------------------------------------------------------


async def test_i13_m6a_a_snapshot_taken_during_pass1_is_not_the_calls_frame(tmp_path):
    """I13/P2 (mutant M6a: the frame from ``_turn_frames()`` at judge time). Pass 1 has sent its prompt with frame
    ``a``; while it streams, a snapshot adds ``b`` (1920x1080) to the turn's frames. The call's frame is the newest
    image of the prompt pass 1 *sent* (criterion 3), so it is ``a``, not ``b``."""
    a = frame(0, "a")
    async with tool_rig(tmp_path, script=[answer(SPOKEN), tool(VALID_CLICK)], frames=[a]) as rig:
        rig.llm.hold_after(1)
        await rig.ask(QUESTION)
        await rig.until(lambda: rig.state.state is TurnState.SPEAKING, "the prefix never played")
        assert not rig.state.llm_done, "premise: pass 1 is still streaming"
        later = frame(rig.clock.now_ms, "b", width=1920, height=1080)
        rig.perception.add(later)
        await rig.session.executor.execute(C.SnapshotFrame(at_ms=rig.clock.now_ms))
        rig.llm.release()
        await rig.turn_done()
    sent = [
        p.phash
        for _, prompt in rig.llm.gen_prompts
        for m in prompt.messages
        for p in getattr(m, "content", ())
        if getattr(p, "phash", None)
    ]
    assert later.phash not in sent, "premise: the later frame was never sent to the model"
    inv = one(rig.executor.invocations, "executor invocation")
    assert inv.frame is not None and (inv.frame.phash, inv.frame.width, inv.frame.height) == (
        a.phash,
        1280,
        720,
    ), (
        f"T-0092c criterion 3: the frame pass 1 sent, not a snapshot taken during pass 1; got {inv.frame!r}"
    )


# --- I13 / M7: a nested splice ------------------------------------------------------------------------------


async def test_i13_m7_a_nested_splice_is_cleaned_as_one_text(tmp_path):
    """I13/M7 (mutant: hold back only the longest suffix that starts a literal). ``<tool|<|t`` / ``ool>`` / ``> for
    you. ``: deleting ``<|tool>`` splices ``<tool|>``, which is deleted too. By hand: "I pressed it <tool|<|tool>>
    for you. " -> "I pressed it <tool|> for you. " -> "I pressed it  for you. ". A cleaner that released
    ``<tool|`` after the first commit could never take it back."""
    parts = ["I pressed it <tool|<|t", "ool>", "> for you. "]
    expected = "I pressed it  for you. "
    assert drop_special_literals("".join(parts)) == expected, "premise: the hand computation"
    async with tool_rig(tmp_path, script=[answer(p, i) for i, p in enumerate(parts)]) as rig:
        await voice_turn(rig)
    submitted = "".join(e.text for e in answer_events(rig))
    assert submitted == expected, (
        f"I4: drop_special_literals runs over the concatenation; submitted {submitted!r}, expected {expected!r}"
    )
    assert "<tool|" not in rig.spoken_sources() and rig.spoken_sources() == expected, (
        rig.spoken_sources()
    )
    assert_never_voiced(rig, what=" (nested splice)")
    assert_history_clean(rig)
    assert "<tool|" not in rig.state.history[-1].assistant_spoken


# --- I13 / M26: an empty cleaned commit is not submitted ------------------------------------------------------


async def test_i13_m26_a_commit_that_cleans_to_nothing_is_not_submitted(tmp_path):
    """I13/M26 (mutant: submit the cleaned commit even when it is empty). ``Sure<|t`` / ``ool>`` / `` thing. ``:
    canvas 1 completes ``<|tool>``, which is deleted, so it releases nothing -- and nothing empty reaches the
    machine. Expected by hand: "Sure thing. "."""
    parts = ["Sure<|t", "ool>", " thing. "]
    async with tool_rig(tmp_path, script=[answer(p, i) for i, p in enumerate(parts)]) as rig:
        await voice_turn(rig)
    texts = [e.text for e in answer_events(rig)]
    assert "" not in texts, f"I13/M26: an empty answer commit was submitted to the machine: {texts}"
    assert "".join(texts) == "Sure thing. ", texts


# --- I13 / M21: a stale generate's end of stream judges nothing --------------------------------------------------


class StaleEndLlm(ToolLlm):
    """Adversarial: the first turn stream swallows its cancellation and ends *normally* later -- after the next
    turn's ``tool`` commit has been kept. Its end of stream is therefore stale and runs while t2 is open with a kept
    call. The stale request itself yields only a thought: it has no ``tool`` commit of its own."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.stale_release = asyncio.Event()
        self.swallowed = 0
        self.stale_ended = False
        self._first = True

    async def generate(  # type: ignore[override]
        self,
        prompt: Any,
        request_id: str,
        options: Any = None,
        *,
        xargs: Any = None,
        cd: Any = None,
    ):
        from tests.fakes.llm import is_caption_prompt

        if self._first and not is_caption_prompt(prompt):
            self._first = False
            self.gen_prompts.append((request_id, prompt))
            self._note("generate", request_id)
            yield thought("thinking about the button").model_copy(update={"request_id": request_id})
            while not self.stale_release.is_set():
                try:
                    await self.stale_release.wait()
                except asyncio.CancelledError:
                    self.swallowed += 1
            self.stale_ended = True
            return
        async for c in super().generate(prompt, request_id, options, xargs=xargs, cd=cd):
            yield c
            if c.segment == "tool" and not self.stale_release.is_set():
                self.stale_release.set()
                for _ in range(20):
                    await asyncio.sleep(0)


async def test_i13_m21_a_stale_generates_end_of_stream_does_not_judge_the_open_turns_call(tmp_path):
    """I13/M21 (mutant: drop the "still the live generate" guard from the judge). t1's stream is aborted by a
    THINKING takeover, swallows the cancel and ends normally after t2's ``tool`` commit was kept. Criterion 1:
    t1's request had no ``tool`` commit, so its ``LlmDone`` carries no tool; t2's call is judged once, by t2's own
    end of stream (one ``proposed``, one ``dry_run``), and executed once with t2's arguments."""
    llm = StaleEndLlm([answer("Second answer here. "), tool(VALID_CLICK)])
    async with tool_rig(tmp_path, llm=llm) as rig:
        await rig.ask(QUESTION)
        await rig.until(
            lambda: rig.state.state is TurnState.THINKING and rig.machine.events(E.LlmCommit),
            "t1 never thought",
        )
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.until(lambda: llm.swallowed >= 1, "premise: t1's stream was never cancelled")
        assert rig.state.turn_id == "t2", "premise: the takeover opened t2"
        await rig.ask("and the second question", onset=False)
        await rig.turn_done()
    assert llm.stale_ended, "premise: the stale stream ended normally after t2's tool commit"
    stale = rig.dones(PASS1_ID)
    assert all(d.tool is None for d in stale), (
        f"T-0092c criterion 1: the stale t1 request kept no tool commit, so its LlmDone carries no tool; got {stale}"
    )
    assert rig.statuses("t2-call") == ["proposed", "dry_run"], (
        f"T-0092c criterion 1: t2's call is judged once, by its own request; published {rig.statuses('t2-call')}"
    )
    assert rig.tool_calls("t1-call") == []
    assert [(i.call_id, i.arguments) for i in rig.executor.invocations] == [
        ("t2-call", {"x": 500, "y": 250})
    ]


# --- I13 / M27: repair_ms and the tool.repair span on the error path -------------------------------------------


class SlowFailingTokenizeLlm(ToolLlm):
    """``tokenize`` waits for ``gate``, then raises: a repair that fails after virtual time has passed."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.gate = asyncio.Event()

    async def tokenize(self, text: str) -> list[int]:
        self.tokenized.append(text)
        await self.gate.wait()
        raise RuntimeError("tokenize failed late")


async def test_i13_m27_a_failed_repair_still_records_repair_ms_and_the_span(tmp_path):
    """I13/M27 (mutant: the error path skips the repair span). The repair's tokenize fails 300 ms (virtual) after
    ``ToolRepairStart``. Criterion 7: ``tool.repair`` runs from ``ToolRepairStart`` handled to the repair's
    ``LlmError`` submitted, and ``repair_ms`` is that duration on the session clock -- on the error path as on the
    success path."""
    llm = SlowFailingTokenizeLlm([answer(SPOKEN), tool(REPAIRABLE_CLICK)])
    async with tool_rig(tmp_path, llm=llm) as rig:
        await rig.ask(QUESTION)
        await rig.until(lambda: bool(llm.tokenized), "the repair never tokenized")
        t_tokenize = rig.clock.now_ms
        await rig.advance(300)
        llm.gate.set()
        await rig.turn_done()
    errors = [e for e in rig.machine.events(E.LlmError) if e.request_id == REPAIR_ID]
    assert len(errors) == 1, f"premise: the failure reached the machine as LlmError({REPAIR_ID})"
    spans = rig.record("t1").spans
    assert "tool.repair" in spans, (
        f"T-0092c criterion 7: the tool.repair span is recorded on the error path; spans {sorted(spans)}"
    )
    start, end = spans["tool.repair"]
    assert start <= t_tokenize and end == errors[0].t_ms and end - start >= 300, (
        f"tool.repair from ToolRepairStart handled (<= {t_tokenize}) to LlmError submitted ({errors[0].t_ms}); "
        f"got {(start, end)}"
    )
    final = rig.tool_records()[-1]
    assert final["status"] == "refused" and final.get("repair_ms") == end - start, (
        f"T-0092c criterion 7: repair_ms on the error path is the span's duration ({end - start}); got {final!r}"
    )


# --- I13 (reviewer 2): with tools off the three handlers are inert -------------------------------------------


async def test_i13_tools_off_the_tool_handlers_are_inert_and_note_once_each(tmp_path):
    """I13 (reviewer 2; criterion 5): with ``kit=None`` and the session log on, executing the three tool commands
    publishes no ``astra.tool.call``, submits no event and writes exactly one ``note`` each. The mutant this kills
    deletes the ``tool_kit is None`` guards (the session's tool methods then run with tools off)."""
    from tests.unit.test_session_loop import TOOL_COMMANDS, command_union_members, sample_command

    kinds_ = {k.__name__: k for k in command_union_members()}
    commands = [sample_command(kinds_[n]) for n in TOOL_COMMANDS]
    async with tool_rig(tmp_path, kit=None) as rig:
        submitted: list[Any] = []
        real_submit = rig.session.submit

        def spy(event: Any) -> None:
            submitted.append(event)
            real_submit(event)

        rig.session.submit = spy  # type: ignore[method-assign]
        for command in commands:
            await rig.session.executor.execute(command)
        await drain(rig)
        rig.session.submit = real_submit  # type: ignore[method-assign]
    assert rig.tool_calls() == [], f"tools off: no astra.tool.call; got {rig.tool_calls()}"
    assert submitted == [], f"tools off: the tool handlers submit no event; got {submitted}"
    assert rig.tool_records() == []
    notes = [n.get("what") for n in rig.kind("note")]
    assert sorted(w for w in notes if w in {c.type for c in commands}) == sorted(
        c.type for c in commands
    ), f"tools off: exactly one note per tool command; notes {notes}"


async def drain(rig: Any) -> None:
    for _ in range(4):
        await rig.pump()
        await asyncio.sleep(0)


# =====================================================================================================
# Ruling I15 (re-review at fc4fe5f): a thought commit is not a boundary for the cleaner
#
# The machine builds the committed answer and history from *all* ``thought_open=False`` commits concatenated, so a
# ``thought_open=True`` commit in between bypasses the cleaner and neither flushes nor resets its hold. Oracles: the
# hand-computed concatenation of the voiced commits, cleaned by ``drop_special_literals`` (criterion 1, I15).
# =====================================================================================================


async def test_i15_a_held_tail_survives_a_thought_commit_whole(tmp_path):
    """I15 row 1 (reviewer 1, P10 "C"). ``Press <b`` / thought / ``ack> now. ``: ``<b`` is held, the thought passes
    unchanged and does not flush the hold, and ``<back>`` -- not a literal -- is voiced and stored whole: by hand,
    ``Press <back> now. ``. The tools-off session over the same script is the second oracle.

    At fc4fe5f the spoken text and history are already right (the machine re-joins the flushed ``<b``); the row is
    red there on I15's "neither flushed nor reset": I14 submits the held ``<b`` ahead of the thought commit."""
    script = [answer("Press <b", 0), open_thought(f" {SECRET}", 1), answer("ack> now. ", 2)]
    expected = "Press <back> now. "
    assert drop_special_literals("Press <b" + "ack> now. ") == expected, (
        "premise: the hand computation"
    )
    async with tool_rig(tmp_path / "off", script=script, kit=None) as off:
        await voice_turn(off)
    assert (off.spoken_sources(), off.state.history[-1].assistant_spoken) == (expected, expected), (
        "premise (tools off): the machine joins the voiced commits across the thought"
    )
    async with tool_rig(tmp_path / "on", script=script) as rig:
        await voice_turn(rig)
    assert rig.spoken_sources() == expected, (
        f"I15: the spoken source is the voiced commits joined across the thought; got {rig.spoken_sources()!r}"
    )
    assert rig.state.history[-1].assistant_spoken == expected, rig.state.history[-1]
    assert SECRET not in rig.spoken_sources()
    # I15 "the hold is neither flushed nor reset": ``<b`` may still start a literal, so nothing of it reaches the
    # machine before the thought commit does -- the voiced text submitted ahead of the thought is ``Press `` only.
    events = answer_events(rig)
    at = next(i for i, e in enumerate(events) if e.thought_open)
    before = "".join(e.text for e in events[:at] if not e.thought_open)
    assert before == "Press ", (
        f"I15: a thought commit does not flush the hold; voiced text submitted before the thought {before!r}, "
        f"commits {[(e.text, e.thought_open) for e in events]}"
    )


@pytest.mark.parametrize(
    "first, inside, last",
    [("Press <|tool", "S", "_call> now. "), ("Look <|t", SECRET, "ool> here. ")],
    ids=["tool_call", "tool"],
)
async def test_i15_a_literal_spliced_across_a_thought_is_never_voiced_or_stored(
    tmp_path, first, inside, last
):
    """I15 row 2 (reviewer 1, P9 "B" and P10 "D"). ``Press <|tool`` / thought ``S`` / ``_call> now. ``: the voiced
    commits joined are ``Press <|tool_call> now. ``, a literal the cleaner must delete; ``Look <|t`` / thought /
    ``ool> here. `` splices ``<|tool>`` the same way. No ``SPECIAL_TOKENS`` literal is in any ``TtsSpeak.source``
    (concatenated, too) or in ``HistoryTurn.assistant_spoken``; the voiced text is ``drop_special_literals`` of the
    joined voiced commits.

    Red at fc4fe5f: I14 flushed the held ``<|tool`` at the thought and reset, so the splice reached speech and
    history."""
    expected = drop_special_literals(first + last)
    assert literal_in(first + last) and not literal_in(expected), (
        "premise: the join splices a literal"
    )
    async with tool_rig(
        tmp_path, script=[answer(first, 0), open_thought(inside, 1), answer(last, 2)]
    ) as rig:
        await voice_turn(rig)
    sources = [c.source for c in rig.speaks()]
    bad = [s for s in sources if literal_in(s)] + (
        [rig.spoken_sources()] if literal_in(rig.spoken_sources()) else []
    )
    assert bad == [], f"I15: a literal spliced across a thought reached a TtsSpeak.source: {bad}"
    stored = rig.state.history[-1].assistant_spoken
    assert literal_in(stored) == [], (
        f"I15: a literal spliced across a thought was stored: {stored!r}"
    )
    assert rig.spoken_sources() == expected and stored == expected, (rig.spoken_sources(), stored)
    assert_never_voiced(rig, what=" (splice across a thought)")


async def _shown_card(rig: Any) -> None:
    await rig.ask(QUESTION)
    await rig.until_status("proposed")
    assert rig.statuses() == ["proposed"], (
        f"premise: the card is shown and has no final; {rig.statuses()}"
    )


async def test_i15_close_calls_on_a_shown_card_publishes_one_turn_closed_and_records_it(tmp_path):
    """I15 row 3, first half (reviewer 2's probe): ``_close_calls`` called directly on the open turn with a shown
    card (``proposed``) publishes exactly one ``cancelled("turn closed")``, writes it as a ``tool.call`` record, and
    ``TurnRecord.tool`` shows it; a second call publishes and records nothing (one final per call, I2)."""
    long_answer = "I will click the Sign in button at the top of the page for you right now. "
    async with tool_rig(tmp_path, script=[answer(long_answer), tool(VALID_CLICK)]) as rig:
        await _shown_card(rig)
        turn = rig.session._turn
        assert turn is not None and turn.turn_id == "t1", "premise: t1 is open"
        rig.session._close_calls(turn)
        after_first = list(rig.statuses())
        rig.session._close_calls(turn)
        assert rig.statuses() == after_first, (
            f"I2: nothing is published after a final; {after_first} -> {rig.statuses()}"
        )
        await rig.turn_done()
    assert after_first == ["proposed", "cancelled"], after_first
    final = final_message(rig)
    assert (final.status, final.reason) == ("cancelled", "turn closed"), final
    records = [(r["status"], r["reason"]) for r in rig.tool_records()]
    assert records == [("proposed", None), ("cancelled", "turn closed")], (
        f"I11: the cancel is a tool.call record too (published through the one seam); records {records}"
    )
    tool_state = rig.record("t1").tool
    assert tool_state is not None and (tool_state["status"], tool_state["reason"]) == (
        "cancelled",
        "turn closed",
    ), f"I11/I15: TurnRecord.tool shows the turn-closed cancel; got {tool_state!r}"


from tests.unit.test_session_loop import RecordingMachine  # noqa: E402


class ForgetfulMachine(RecordingMachine):  # type: ignore[misc]
    """The production machine with every ``ToolCancel`` dropped from its output: a turn that closes leaves its
    shown call without a final, which is exactly the case I11's turn-close net exists for."""

    def step(self, state: Any, event: Any):  # type: ignore[no-untyped-def]
        new, cmds = super().step(state, event)
        return new, tuple(c for c in cmds if not isinstance(c, C.ToolCancel))


async def test_i15_a_closing_turn_closes_its_shown_call_before_the_record_is_written(tmp_path):
    """I15 row 3, second half: the same net through the pump. The machine double never cancels (``ToolCancel``
    dropped); a barge-in closes t1 with its card at ``proposed``. The session ends the card with one
    ``cancelled("turn closed")``, logged as a ``tool.call`` record, *before* the turn record is written -- so
    ``TurnRecord.tool`` shows it. The call is never executed."""
    long_answer = "I will click the Sign in button at the top of the page for you right now. "
    async with tool_rig(
        tmp_path, script=[answer(long_answer), tool(VALID_CLICK)], machine_cls=ForgetfulMachine
    ) as rig:
        await _shown_card(rig)
        now = rig.clock.now_ms
        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
        await rig.advance(800)
        await rig.until(lambda: rig.state.turn_id != "t1" or bool(rig.records()), "t1 never closed")
        await rig.advance(200)
    final = final_message(rig)
    assert (final.status, final.reason) == ("cancelled", "turn closed"), (
        f"I11: a closing turn's shown call without a final ends cancelled('turn closed'); {rig.statuses()}"
    )
    assert [(r["status"], r["reason"]) for r in rig.tool_records()][-1] == (
        "cancelled",
        "turn closed",
    )
    tool_state = rig.record("t1").tool
    assert tool_state is not None and (tool_state["status"], tool_state["reason"]) == (
        "cancelled",
        "turn closed",
    ), f"I15: the cancel lands before the turn record is written; TurnRecord.tool {tool_state!r}"
    assert rig.executor.invocations == []


# =====================================================================================================
# T-0100: refuse instead of repair (H-T99d, H-T99e, Q1-Q4), through the session (criteria 9-11)
# =====================================================================================================
#
# The verdicts are written out by hand from docs/tasks/T-0100.md (the three reason strings verbatim). The session is
# not owned by T-0100: these rows hold through ``_judge_tool`` as written once ``live`` refuses. The aliased/in-range
# repair rows are guards that held before T-0100 too.

from tests.unit._tool_turn_support import (  # noqa: E402
    ALIASED_KEY,
    OUT_OF_LIST_KEY,
    OUT_OF_RANGE_CLICK,
    REPAIRED_KEY,
)

T0100_REFUSALS = [
    pytest.param(
        OUT_OF_LIST_KEY, "press_key", {"combo": "Ctrl+S"}, "combo not in enum", id="ctrl_s"
    ),
    pytest.param(
        OUT_OF_RANGE_CLICK,
        "click_at",
        {"x": 1088, "y": 748},
        "coordinate out of range",
        id="click_1088_748",
    ),
]


def _assert_refused_at_detection(
    rig: Any, name: str, arguments: dict[str, Any], reason: str
) -> None:
    msgs = rig.tool_calls()
    assert [(m.status, m.source, m.name, m.arguments, m.reason) for m in msgs] == [
        ("refused", "draft", name, arguments, reason)
    ], (
        f"T-0100 criterion 9: exactly one astra.tool.call, refused at detection with its name, arguments and "
        f"reason; published {[(m.status, m.name, m.arguments, m.reason) for m in msgs]}"
    )
    d = draft_of(only_done(rig, PASS1_ID))
    assert (d.status, d.name, d.arguments, d.reason, d.source) == (
        "refused",
        name,
        arguments,
        reason,
        "draft",
    ), d
    assert rig.commands(C.ToolRepairStart) == [], "T-0100 criterion 9: no ToolRepairStart"
    assert rig.commands(C.ToolExecute) == [], "T-0100 criterion 9: no ToolExecute"
    assert rig.executor.invocations == [], "T-0100 criterion 9: the executor is never called"
    assert rig.llm.continuations() == [] and rig.llm.tokenized == [], (
        "T-0100 criterion 9: no tokenize and no repair generate"
    )


@pytest.mark.parametrize(("text", "name", "arguments", "reason"), T0100_REFUSALS[:1])
async def test_out_of_enum_key_refused_no_repair_no_execute(
    tmp_path, text, name, arguments, reason
):
    """T-0100 criterion 9 (G7; architecture §5), tools ``dry_run`` and repair on: pass 1's ``press_key{combo:Ctrl+S}``
    is one ``refused`` card (source ``draft``, its name and arguments, "combo not in enum"), no ``tool_repair``
    request in the log, no ``ToolRepairStart``, no ``ToolExecute``, the executor never called, and
    ``TurnRecord.tool.status == "refused"``."""
    async with tool_rig(tmp_path, script=[answer(SPOKEN), tool(text)]) as rig:
        await voice_turn(rig)
    _assert_refused_at_detection(rig, name, arguments, reason)
    repairs = [r for r in rig.kind("llm.request") if r.get("phase") == "tool_repair"]
    assert repairs == [], (
        f"T-0100 criterion 9: no llm.request with phase tool_repair; got {repairs}"
    )
    assert [(r.get("status"), r.get("reason")) for r in rig.tool_records()] == [("refused", reason)]
    record = rig.record("t1").tool
    assert record is not None and record.get("status") == "refused", record


@pytest.mark.parametrize(("text", "name", "arguments", "reason"), T0100_REFUSALS[1:])
async def test_out_of_range_click_refused_no_repair_no_execute(
    tmp_path, text, name, arguments, reason
):
    """T-0100 criterion 9 for H-T99e: the a794a98 cpix-03 draft ``click_at{x:1088,y:748}`` is refused "coordinate out
    of range" at detection -- the (108, 174) coercion of the T-0099 run can no longer happen."""
    async with tool_rig(tmp_path, script=[answer(SPOKEN), tool(text)]) as rig:
        await voice_turn(rig)
    _assert_refused_at_detection(rig, name, arguments, reason)
    repairs = [r for r in rig.kind("llm.request") if r.get("phase") == "tool_repair"]
    assert repairs == [], (
        f"T-0100 criterion 9: no llm.request with phase tool_repair; got {repairs}"
    )
    record = rig.record("t1").tool
    assert record is not None and record.get("status") == "refused", record


@pytest.mark.parametrize(("text", "name", "arguments", "reason"), T0100_REFUSALS)
async def test_refusal_voices_only_the_committed_prefix(tmp_path, text, name, arguments, reason):
    """T-0100 criterion 10 (Q4, H4 unchanged): on a refused turn the device hears only the answer committed before
    the call -- no ack clip and nothing new after the refusal, no call text -- and history stores only that."""
    async with tool_rig(tmp_path, script=[answer(SPOKEN), tool(text)]) as rig:
        await voice_turn(rig)
    assert [(m.status, m.reason) for m in rig.tool_calls()] == [("refused", reason)], (
        f"premise (T-0100): refused at detection; published {[(m.status, m.reason) for m in rig.tool_calls()]}"
    )
    assert rig.spoken_sources() == SPOKEN, rig.spoken_sources()
    assert "".join(rig.tts.texts()).split() == SPOKEN.split(), rig.tts.texts()
    log = rig.machine.log
    done_at = next(
        i
        for i, (ev, _s, _c) in enumerate(log)
        if isinstance(ev, E.LlmDone) and ev.request_id == PASS1_ID
    )
    after = [c for _ev, _s, cmds in log[done_at:] for c in cmds]
    late = [c for c in after if isinstance(c, C.PlayClip) or (isinstance(c, C.TtsSpeak) and c.text)]
    assert late == [], (
        f"T-0100 criterion 10: nothing voiced or played after the refusal; got {late}"
    )
    assert rig.state.history[-1].assistant_spoken == SPOKEN, rig.state.history[-1]
    assert_never_voiced(rig, arguments=arguments)
    assert_history_clean(rig, arguments=arguments)


REPAIR_OFF = "payload manifest canvas_length 128 is not the profile's 256"


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        pytest.param(OUT_OF_LIST_KEY, "combo not in enum", id="ctrl_s"),
        pytest.param(OUT_OF_RANGE_CLICK, "coordinate out of range", id="x_1088"),
        pytest.param(ALIASED_KEY, f"repair off: {REPAIR_OFF}", id="aliased_still_repair_off"),
        pytest.param(REPAIRABLE_CLICK, f"repair off: {REPAIR_OFF}", id="in_range_still_repair_off"),
    ],
)
async def test_repair_off_keeps_the_h99_reasons(tmp_path, text, reason):
    """T-0100 criterion 10: with ``repair_off`` set, a draft H-T99d/H-T99e refuse keeps its own reason (it was never
    repairable), and a draft that would otherwise be repairable keeps "repair off: ..." (T-0092b, Q-b)."""
    executor = WatchingExecutor()
    kit = make_kit(repair_off=REPAIR_OFF, executor=executor)
    async with tool_rig(
        tmp_path, script=[answer(SPOKEN), tool(text)], kit=kit, executor=executor
    ) as rig:
        await voice_turn(rig)
    d = draft_of(only_done(rig, PASS1_ID))
    assert (d.status, d.reason) == ("refused", reason), d
    assert [(m.status, m.reason) for m in rig.tool_calls()] == [("refused", reason)]
    assert rig.llm.tokenized == [] and rig.llm.continuations() == []
    assert rig.executor.invocations == []


@pytest.mark.parametrize(
    ("text", "repaired", "name", "arguments"),
    [
        pytest.param(ALIASED_KEY, REPAIRED_KEY, "press_key", {"combo": "Ctrl+C"}, id="control_c"),
        pytest.param(
            REPAIRABLE_CLICK, REPAIRED_CLICK, "click_at", {"x": 12, "y": 5}, id="click_extra_key"
        ),
    ],
)
async def test_aliased_combo_is_repaired_once(tmp_path, text, repaired, name, arguments):
    """T-0100 criterion 9, adversarial (Q1): ``Control+C`` maps into the list, so it is not refused -- exactly one
    repair starts (one ``ToolRepairStart``, one continuation, one ``tool_repair`` request with that tool's payload)
    and the repaired call is dry-run; the same for an in-range click with an extra key. Guard, green today: the
    "refuse every invalid press_key" mutant fails it."""
    async with tool_rig(
        tmp_path, script=[answer(SPOKEN), tool(text)], continuation=[tool(repaired, canvas_idx=0)]
    ) as rig:
        await voice_turn(rig)
    d = draft_of(only_done(rig, PASS1_ID))
    assert (d.status, d.name, d.reason) == ("repairable", name, "schema invalid"), d
    assert len(rig.commands(C.ToolRepairStart)) == 1
    (_eid, _prompt, _options, cd) = one(rig.llm.continuations(), "repair request")
    assert cd.automaton_sha256 == live.load_payloads().tools[name].automaton_sha256
    assert rig.statuses() == ["proposed", "repairing", "dry_run"], rig.statuses()
    inv = one(rig.executor.invocations, "executor invocation")
    assert (inv.name, inv.arguments, inv.source) == (name, arguments, "repair"), inv
    repairs = [r for r in rig.kind("llm.request") if r.get("phase") == "tool_repair"]
    assert len(repairs) == 1, repairs


def test_t0100_a_refused_draft_with_name_and_arguments_validates() -> None:
    """T-0100 criterion 11: the existing contract already admits ``ToolDraft(status="refused", name=..., arguments=...,
    source="draft")`` -- no contract change is needed. Guard."""
    d = ToolDraft(
        call_id=CALL_ID,
        status="refused",
        name="press_key",
        arguments={"combo": "Ctrl+S"},
        reason="combo not in enum",
        source="draft",
    )
    assert (d.status, d.name, d.arguments, d.reason) == (
        "refused",
        "press_key",
        {"combo": "Ctrl+S"},
        "combo not in enum",
    )


def test_t0100_the_session_fixtures_mean_what_they_say() -> None:
    """The fixtures' premises, against the hand-written verdicts: the moved repairable fixture is in range and
    repairable today and after T-0100; the aliased key too. Guard."""
    assert (
        "1200" not in REPAIRABLE_CLICK
        and "1200" not in STILL_INVALID
        and "1200" not in REPAIRABLE_DRAFT
    )
    for text in (REPAIRABLE_CLICK, ALIASED_KEY):
        v = live.classify(text)
        assert (v.status, v.reason) == ("repairable", "schema invalid"), (text, v)
    rv = live.repair_verdict(STILL_INVALID)
    assert (rv.status, rv.reason) == ("refused", "schema invalid"), rv
    assert REPAIRABLE_CLICK.removeprefix("<|tool_call>") == REPAIRABLE_DRAFT


# =====================================================================================================
# T-0102: refuse a retracted call (H-T101c), through the session (criteria 9-10, rulings Q3 and Q5)
# =====================================================================================================
#
# The verdict is written out by hand from docs/tasks/T-0102.md (reason "model text after call", pinned by value). The
# session is not owned by T-0102 (criterion 9: if this fails through ``_judge_tool`` as written, the task is
# replanned): these rows hold once ``live.classify`` refuses. Regime: CPU over the fakes, ``ScriptedClock``, thinking
# on, tools ``dry_run``, repair on unless stated, n = 1 per row, deterministic.

H101C = "model text after call"
#: df379e4 kout-03 rep 2, pass 1's ``tool`` commit (stop id 1), verbatim from
#: ``t0101-launch/logs/t0101-warm-rerun-df379e4/sessions/20261015-201518-858015.jsonl`` (seq 37). The same literal as
#: ``tests/unit/test_toolcall_live.py::DF379E4_KOUT03_PASS1``.
DF379E4_KOUT03_PASS1 = (
    '<|tool_call>call:press_key{combo:<|"|>Ctrl+T<|"|>} Wait, the tool `press_key` only takes one key or specific '
    "combo from a list. It doesn't support Ctrl+Shift+T directly. Let me re-check the list: `Enter`, `Escape`, `Tab`, "
    "`Backspace`, `Delete`, `Space`, `ArrowUp`, `ArrowDown`, `ArrowLeft`, `ArrowRight`, `Home`, `End`, `PageUp`, "
    "`PageDown`, `Ctrl+A`, `Ctrl+C`, `Ctrl+V`, `Ctrl+X`, `Ctrl+Z`, `Ctrl+L`, `Ctrl+T`, `Ctrl+W`, `Alt+Tab`, "
    "`Ctrl+Enter`.\nI cannot perform Ctrl+Shift+T with the available tools. I should inform the user.\n\nActually, "
    'looking at the prompt, "Reopen the tab I just closed with Ctrl+Shift+T."\nIf I can\'t do it, I should say so.\n'
    "Wait, I should check if I can simulate it. No.\nI'll just state I can't perform that action.<channel|>I cannot "
    "perform that specific shortcut directly."
)
#: the answer committed before it in the same session (seq 28), voiced as "I will press Ctrl Shift T ..."
DF379E4_KOUT03_SPOKEN = "I will press Ctrl Shift T to reopen the last closed tab.\n"
#: what the df379e4 repair wrote (seq 48): offered as the continuation, so a repair that runs would execute Ctrl+T
DF379E4_KOUT03_REPAIR = f"<|tool_call>call:press_key{{combo:{Q}Ctrl+T{Q}}}<tool_call|>"
#: words that occur only after the call in the retraction (never in the spoken prefix)
POST_CALL_WORDS = ("Wait", "cannot", "shortcut", "Actually", "inform")
#: criterion 9 (adversarial): a clean cut, nothing after the call -> still exactly one repair
CLEAN_CUT_CLICK = "<|tool_call>call:click_at{x:12,y:5"


def _voiced(rig: Any) -> list[str]:
    return [t for c in rig.speaks() for t in (c.text, c.source if c.text else "") if t]


@pytest.mark.parametrize(
    "stop_id", [1, 50, None], ids=["stop_1_as_logged", "stop_50", "length_cut"]
)
async def test_retracted_call_refused_no_repair_no_execute(tmp_path, stop_id):
    """T-0102 criterion 9 (G7; architecture §5), tools ``dry_run`` and repair on: pass 1's tool commit is the df379e4
    retraction. Exactly one ``astra.tool.call`` -- ``refused``, source ``draft``, name ``press_key``, reason "model
    text after call" -- the machine is handed that refused draft (arguments None: it does not parse), no
    ``llm.request`` with phase ``tool_repair``, no ``ToolRepairStart``, no ``ToolExecute``, the executor never called
    (although a repair continuation that would dry-run Ctrl+T is scripted), ``TurnRecord.tool.status == "refused"``.
    Q3: the stop id is not an input -- the logged stop id 1, stop id 50 and a length cut give the same verdict."""
    async with tool_rig(
        tmp_path,
        script=[answer(DF379E4_KOUT03_SPOKEN), tool(DF379E4_KOUT03_PASS1, stop_id=stop_id)],
        continuation=[tool(DF379E4_KOUT03_REPAIR, canvas_idx=0)],
    ) as rig:
        await voice_turn(rig)
    msgs = rig.tool_calls()
    assert [(m.status, m.source, m.name, m.reason) for m in msgs] == [
        ("refused", "draft", "press_key", H101C)
    ], (
        f"T-0102 criterion 9: exactly one astra.tool.call, refused at detection; published "
        f"{[(m.status, m.source, m.name, m.reason) for m in msgs]}"
    )
    d = draft_of(only_done(rig, PASS1_ID))
    assert (d.status, d.name, d.arguments, d.reason, d.source) == (
        "refused",
        "press_key",
        None,
        H101C,
        "draft",
    ), d
    repairs = [r for r in rig.kind("llm.request") if r.get("phase") == "tool_repair"]
    assert repairs == [], (
        f"T-0102 criterion 9: no llm.request with phase tool_repair; got {repairs}"
    )
    assert rig.commands(C.ToolRepairStart) == [], "T-0102 criterion 9: no ToolRepairStart"
    assert rig.commands(C.ToolExecute) == [], "T-0102 criterion 9: no ToolExecute"
    assert rig.executor.invocations == [], "T-0102 criterion 9: the executor is never called"
    assert rig.llm.continuations() == [] and rig.llm.tokenized == [], (
        "T-0102 criterion 9: no tokenize and no repair generate"
    )
    assert [(r.get("status"), r.get("reason")) for r in rig.tool_records()] == [("refused", H101C)]
    record = rig.record("t1").tool
    assert record is not None and record.get("status") == "refused", record


async def test_clean_cut_still_repaired_once(tmp_path):
    """T-0102 criterion 9, adversarial (the guard-too-wide direction): a clean cut ``click_at{x:12,y:5`` (nothing after
    the call) is still ``repairable`` "no close" -- exactly one repair starts (one ``ToolRepairStart``, one
    continuation with click_at's payload, one ``tool_repair`` request) and the repaired call is dry-run once. Guard,
    green before and after T-0102: "refuse on the 'no close' reason" fails it."""
    async with tool_rig(
        tmp_path,
        script=[answer(SPOKEN), tool(CLEAN_CUT_CLICK, stop_id=None)],
        continuation=[tool(REPAIRED_CLICK, canvas_idx=0)],
    ) as rig:
        await voice_turn(rig)
    d = draft_of(only_done(rig, PASS1_ID))
    assert (d.status, d.name, d.reason, d.source) == (
        "repairable",
        "click_at",
        "no close",
        "draft",
    ), d
    assert len(rig.commands(C.ToolRepairStart)) == 1
    (_eid, _prompt, _options, cd) = one(rig.llm.continuations(), "repair request")
    assert cd.automaton_sha256 == live.load_payloads().tools["click_at"].automaton_sha256
    repairs = [r for r in rig.kind("llm.request") if r.get("phase") == "tool_repair"]
    assert len(repairs) == 1, repairs
    assert rig.statuses() == ["proposed", "repairing", "dry_run"], rig.statuses()
    inv = one(rig.executor.invocations, "executor invocation")
    assert (inv.name, inv.arguments, inv.source) == ("click_at", {"x": 12, "y": 5}, "repair"), inv


async def test_retraction_voices_only_the_committed_prefix(tmp_path):
    """T-0102 criterion 9 / H4, ruling Q5 (card only): on the refused retraction the device hears only the answer
    committed before the call -- no ack clip, nothing voiced or played after the refusal, none of the model's
    post-call words ("Wait, ...", "I cannot perform that specific shortcut directly.") -- and history stores only that
    answer."""
    async with tool_rig(
        tmp_path,
        script=[answer(DF379E4_KOUT03_SPOKEN), tool(DF379E4_KOUT03_PASS1, stop_id=1)],
        continuation=[tool(DF379E4_KOUT03_REPAIR, canvas_idx=0)],
    ) as rig:
        await voice_turn(rig)
    assert [(m.status, m.reason) for m in rig.tool_calls()] == [("refused", H101C)], (
        f"premise (T-0102): refused at detection; published {[(m.status, m.reason) for m in rig.tool_calls()]}"
    )
    assert rig.spoken_sources() == DF379E4_KOUT03_SPOKEN, rig.spoken_sources()
    assert "".join(rig.tts.texts()).split() == DF379E4_KOUT03_SPOKEN.split(), rig.tts.texts()
    log = rig.machine.log
    done_at = next(
        i
        for i, (ev, _s, _c) in enumerate(log)
        if isinstance(ev, E.LlmDone) and ev.request_id == PASS1_ID
    )
    after = [c for _ev, _s, cmds in log[done_at:] for c in cmds]
    late = [c for c in after if isinstance(c, C.PlayClip) or (isinstance(c, C.TtsSpeak) and c.text)]
    assert late == [], (
        f"T-0102 criterion 9 (Q5): nothing voiced or played after the refusal; got {late}"
    )
    leaked = [(w, t) for t in _voiced(rig) for w in POST_CALL_WORDS if w in t]
    assert leaked == [], f"T-0102 (H4, Q5): the model's post-call words were voiced: {leaked}"
    spoken = rig.state.history[-1].assistant_spoken
    assert spoken.split() == DF379E4_KOUT03_SPOKEN.split(), rig.state.history[-1]
    assert not [w for w in POST_CALL_WORDS if w in spoken], spoken
    assert_never_voiced(rig)
    assert_history_clean(rig)


@pytest.mark.parametrize(
    ("text", "stop_id", "reason"),
    [
        pytest.param(
            DF379E4_KOUT03_PASS1,
            1,
            H101C,
            id="retraction_keeps_h101c",
        ),
        pytest.param(
            CLEAN_CUT_CLICK, None, f"repair off: {REPAIR_OFF}", id="clean_cut_still_repair_off"
        ),
    ],
)
async def test_repair_off_keeps_the_h101c_reason(tmp_path, text, stop_id, reason):
    """T-0102 criterion 10: with ``repair_off`` set, the refused retraction keeps "model text after call" (it was
    never repairable), not ``"repair off: ..."``; a clean cut that would otherwise be repairable keeps "repair off:
    ..." (T-0092b, Q-b; that row is a guard, green before and after)."""
    executor = WatchingExecutor()
    kit = make_kit(repair_off=REPAIR_OFF, executor=executor)
    async with tool_rig(
        tmp_path,
        script=[answer(DF379E4_KOUT03_SPOKEN), tool(text, stop_id=stop_id)],
        kit=kit,
        executor=executor,
    ) as rig:
        await voice_turn(rig)
    d = draft_of(only_done(rig, PASS1_ID))
    assert (d.status, d.reason) == ("refused", reason), d
    assert [(m.status, m.reason) for m in rig.tool_calls()] == [("refused", reason)]
    assert rig.llm.tokenized == [] and rig.llm.continuations() == []
    assert rig.executor.invocations == []
