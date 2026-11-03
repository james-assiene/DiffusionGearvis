"""T-0107e criteria 2-7: a turn whose user transcript is a clear no executes no call (over the session fakes).

Every row runs the T-0007 session over ``tests/unit/_tool_turn_support.tool_rig`` (``ToolLlm``, the recording executor,
the HUD ``Recorder``, the real session log, ``ScriptedClock``); voice turns through ``rig.ask`` (one STT word per
whitespace-separated token), typed turns through ``E.UserText``. Expected values are written by hand from the task:
the refusal reason is the literal of the Definition (``USER_DECLINED_REASON = "the user said no"``), the call ids are
T-0092c's/T-0114's (``t1-call``, ``t1-call-s2``), what is voiced and stored is the model's committed answer text as at
BASE (H-T107e-3: the "I will ..." sentence is still voiced), and every verdict is premised on ``live.classify``.

**A row that must stay green when the rule is too wide is still red at the base.** At 9275442 nothing is ever refused
for a decline, so a row that only checks "this executes" would pass at the base and XPASS its strict pending marker.
Each such row therefore runs, in the same session and after its own turns, the control turn ``_control`` (``No, leave
it.`` + a valid click), which must be refused with the decline reason: red at the base on the control and, once
implemented, discriminating in both directions (too wide: its own turn; absent: the control).

Readings where the task is silent (listed in the test-author's report):

- the declined card is published once (``["refused"]``, no ``proposed`` first): a declined draft reaches the machine
  ``refused`` at detection, so T-0092c's refused-at-detection publication applies (as T-0107a's rows read it);
- ``user_declined: true`` and ``verdict_status`` are on **every** ``tool.call`` record of a declined call and on no
  record of any other call; ``verdict_status`` is the ``live.classify`` status before the rule; ``asked_first`` is
  absent from a declined record unless ``asks_user`` was also true (then ``True``, criterion 5);
- ``test_step2_judge_declines`` patches the predicate where the session reaches it (``patch_bound``: the module
  attribute and every same-object binding in ``local_astra.conductor``), so it assumes the session calls
  ``user_declines`` **once per judge** (criterion 3: "the same check runs at every judge"); the spy also records the
  argument, which is how criterion 4's transcript is pinned (the live turn's words joined with single spaces, equal to
  the ``user_text`` stored in history);
- a ui-origin turn is opened by ``E.UiAction(name="__gate.clicked", context={"label": "No"})``; its transcript is the
  session's ``[ui] __gate.clicked {"label":"No"}`` (H-T107e-2: never declines).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools ``dry_run``, loop off
(``POLICY``) or on (``TurnPolicy(thinking=True, tool_loop=True)``) as each row says, n = 1, deterministic.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.contracts.turn import TurnState
from local_astra.toolcall import live
from tests.unit import test_session_tool_loop as tsl
from tests.unit._tool_turn_support import (
    ALIASED_KEY,
    CALL_ID,
    NAMELESS,
    OUT_OF_RANGE_CLICK,
    PASS1_ID,
    REPAIRABLE_CLICK,
    REPAIRED_CLICK,
    SPOKEN,
    VALID_CLICK,
    ToolRig,
    answer,
    assert_history_clean,
    assert_never_voiced,
    assert_one_final,
    final_message,
    patch_bound,
    tool,
    tool_rig,
)
from tests.unit.test_session_loop import POLICY

#: the task's Definition, verbatim (never imported from the module under test)
REASON = "the user said no"
#: T-0107a's reason, verbatim (docs/02 §5 T-0107a delta)
ASKED_REASON = "asked the user first"
MODULE = "local_astra.toolcall.user_declines"
LOOP = tsl.LOOP
CLICK_ARGS = {"x": 500, "y": 250}
NO = "No, leave it."
ASK = "Shall I place the order for 42.99 euros? "
NEW_KEYS = ("user_declined", "verdict_status")


# --- reading the rig ----------------------------------------------------------------------------------------


def done_tool(rig: ToolRig, request_id: str) -> Any:
    found = rig.dones(request_id)
    assert len(found) == 1, f"one LlmDone({request_id!r}) reached the machine; got {found}"
    assert found[0].tool is not None, f"LlmDone({request_id!r}) carries no ToolDraft"
    return found[0].tool


def records_of(rig: ToolRig, call_id: str) -> list[dict[str, Any]]:
    return [r for r in rig.tool_records() if r.get("call_id") == call_id]


def assert_declined_live(rig: ToolRig, request_id: str, call_id: str) -> None:
    """Criteria 2/3 (through R5): the machine is handed ``ToolDraft(status="refused", reason=REASON)``; one refused
    card with the reason; ``tool.cancel(refused)``; no ``tool.execute`` and no executor call for the call."""
    d = done_tool(rig, request_id)
    assert (d.call_id, d.status, d.reason) == (call_id, "refused", REASON), (
        f"T-0107e criteria 2/3: the turn's transcript is a clear no, so {call_id} reaches the machine "
        f"ToolDraft(status='refused', reason={REASON!r}); got {d!r}"
    )
    assert rig.statuses(call_id) == ["refused"], (
        f"T-0107e criterion 2 (R5): one refused card for {call_id}; published {rig.statuses(call_id)}"
    )
    assert final_message(rig, call_id).reason == REASON, final_message(rig, call_id)
    cancels = [c.reason for c in rig.commands(C.ToolCancel) if c.call_id == call_id]
    assert cancels == ["refused"], (
        f"T-0107e criterion 2: tool.cancel(refused) for {call_id}; got {cancels}"
    )
    assert_not_executed(rig, call_id)


def assert_not_executed(rig: ToolRig, call_id: str) -> None:
    ran = [i.call_id for i in rig.executor.invocations if i.call_id == call_id]
    executes = [c for c in rig.commands(C.ToolExecute) if c.call_id == call_id]
    assert not ran and not executes, (
        f"T-0107e criterion 2: no tool.execute for a declined call; executor ran {ran}, ToolExecute {executes}"
    )


def assert_declined_records(
    rig: ToolRig, call_id: str, verdict: str, *, asked: bool = False
) -> None:
    """Criterion 5: every ``tool.call`` record of a declined call carries ``user_declined: true`` and
    ``verdict_status``; ``asked_first: true`` only when ``asks_user`` was also true. Read after ``aclose``."""
    recs = records_of(rig, call_id)
    assert recs, f"T-0107e criterion 5: no tool.call record for {call_id}"
    got = [(r.get("user_declined"), r.get("verdict_status")) for r in recs]
    assert got == [(True, verdict)] * len(recs), (
        f"T-0107e criterion 5: every tool.call record of {call_id} carries user_declined: true and "
        f"verdict_status: {verdict!r}; got {got}"
    )
    flags = [r.get("asked_first", "<absent>") for r in recs]
    want = [True if asked else "<absent>"] * len(recs)
    assert flags == want, (
        f"T-0107e criterion 5: asked_first on {call_id}'s records is {flags}, expected {want}"
    )
    reasons = [r.get("reason") for r in recs]
    assert all(x == REASON for x in reasons), (
        f"T-0107e criterion 5: the record's reason is still the decline; got {reasons}"
    )


def assert_plain_records(rig: ToolRig, call_id: str) -> None:
    recs = records_of(rig, call_id)
    assert recs, f"premise: {call_id} wrote tool.call records"
    carrying = [r["user_declined"] for r in recs if "user_declined" in r]
    assert not carrying, (
        f"T-0107e criterion 5: user_declined is written only on declined calls; {call_id} has {carrying}"
    )


def assert_no_repair(rig: ToolRig) -> None:
    assert rig.llm.tokenized == [] and rig.llm.continuations() == [], (
        "T-0107e criterion 2 (before any repair): no tokenize and no repair request; "
        f"tokenized {len(rig.llm.tokenized)}, continuations {len(rig.llm.continuations())}"
    )
    assert rig.commands(C.ToolRepairStart) == [], rig.commands(C.ToolRepairStart)


def assert_voiced_and_stored(rig: ToolRig, text: str, transcript: str, turn: int = 1) -> None:
    """Criterion 2 (as at BASE): the device hears the model's committed answer text, history stores exactly it,
    and the stored ``user_text`` is the turn's transcript; no conductor speech is added."""
    voiced = " ".join(
        c.text for c in rig.speaks() if c.utterance_id.split("-")[0] == f"t{turn}"
    ).split()
    assert voiced == text.split(), (
        f"T-0107e criterion 2: the committed answer is voiced as at BASE (nothing added); voiced {voiced}, "
        f"expected {text.split()}"
    )
    h = rig.state.history[turn - 1]
    assert h.assistant_spoken == text, (
        f"T-0107e criterion 2: history stores the answer; got {h.assistant_spoken!r}"
    )
    assert h.user_text == transcript, (
        f"T-0107e criterion 4: history's user_text is the transcript; got {h.user_text!r}"
    )


async def until_turns(rig: ToolRig, n: int, what: str, *, limit_ms: int = 30_000) -> None:
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= n,
        f"T-0107e: {what} never ended",
        limit_ms=limit_ms,
    )
    await rig.advance(300)


async def run_turn(
    rig: ToolRig,
    n: int,
    script: Sequence[Commit],
    *,
    question: str,
    steps: Sequence[Sequence[Commit]] = (),
    continuation: Sequence[Commit] | None = None,
    typed: bool = False,
) -> None:
    rig.llm.script = list(script)
    if steps:
        rig.llm.script_steps(*steps)
    if continuation is not None:
        rig.llm.script_continuation(*continuation)
    if typed:
        rig.submit(E.UserText(t_ms=rig.clock.now_ms, text=question))
    else:
        await rig.ask(question)
    await until_turns(rig, n, f"turn {n} ({question!r})")


async def _control(rig: ToolRig, n: int) -> None:
    """The control turn: ``No, leave it.`` + a valid click, which must be declined (red at the base)."""
    await run_turn(rig, n, [answer(SPOKEN), tool(VALID_CLICK)], question=NO)
    assert_declined_live(rig, f"t{n}-gen", f"t{n}-call")


def _patch_predicate(monkeypatch: pytest.MonkeyPatch, replacement: Any) -> None:
    try:
        mod = importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"T-0107e owned files: {MODULE} (user_declines) does not exist: {exc}"
        ) from None
    assert callable(getattr(mod, "user_declines", None)), f"T-0107e: {MODULE} has no user_declines"
    patch_bound(monkeypatch, mod, "user_declines", replacement)


# =====================================================================================================
# Criterion 2: step 1, voice and typed, loop off and on
# =====================================================================================================


@pytest.mark.parametrize("policy", [POLICY, LOOP], ids=["loop_off", "loop_on"])
@pytest.mark.parametrize("typed", [False, True], ids=["voice", "typed"])
async def test_declined_valid_call_is_refused(tmp_path: Path, typed: bool, policy: Any) -> None:
    """Criterion 2 (mutations 1-3, 6): ``No, leave it.``, spoken or typed, then a valid click: refused with the decline
    reason before it can execute -- no ``tool.execute``, no repair, ``tool.cancel(refused)``, the card ``refused`` with
    the reason, the model's committed answer voiced and stored as at BASE, the two log fields; with the loop on, no step
    request and ``tool_loop_end == "refused"``."""
    assert live.classify(VALID_CLICK).status == "valid", "premise: the click is valid"
    async with tool_rig(tmp_path, policy=policy) as rig:
        await run_turn(
            rig,
            1,
            [answer(SPOKEN), tool(VALID_CLICK)],
            question=NO,
            steps=[[answer(tsl.S3)]] if policy is LOOP else (),
            typed=typed,
        )
        assert_declined_live(rig, PASS1_ID, CALL_ID)
    assert rig.executor.invocations == [], rig.executor.invocations
    assert_no_repair(rig)
    assert_voiced_and_stored(rig, SPOKEN, NO)
    assert_never_voiced(rig, arguments=CLICK_ARGS)
    assert_history_clean(rig, arguments=CLICK_ARGS)
    assert_declined_records(rig, CALL_ID, "valid")
    record = rig.record("t1")
    assert record.tool is not None and record.tool.get("status") == "refused", record.tool
    if policy is LOOP:
        assert rig.llm.step_prompts == [], (
            f"T-0107e criterion 2: no step request after a declined step 1; "
            f"{[r for r, _ in rig.llm.step_prompts]}"
        )
        assert record.extras.get("tool_loop_end") == "refused", record.extras


@pytest.mark.parametrize(
    "draft",
    [
        pytest.param(REPAIRABLE_CLICK, id="repairable_click"),
        pytest.param(ALIASED_KEY, id="repairable_alias"),
    ],
)
async def test_declined_repairable_starts_no_repair(tmp_path: Path, draft: str) -> None:
    """Criterion 2, adversarial (mutation 5, refuse only valid drafts; "before any repair"): a repairable draft in a
    declined turn is refused with the decline reason -- no tokenize, no repair request, no ``ToolRepairStart``, no
    execution, though a valid repair is scripted -- and its records keep ``verdict_status: "repairable"``."""
    assert live.classify(draft).status == "repairable", f"premise: {draft!r} classifies repairable"
    async with tool_rig(
        tmp_path, script=[answer(SPOKEN), tool(draft)], continuation=[tool(REPAIRED_CLICK)]
    ) as rig:
        await rig.ask("Stop!")
        await until_turns(rig, 1, "the declined repairable turn")
        assert_declined_live(rig, PASS1_ID, CALL_ID)
    assert_no_repair(rig)
    assert rig.executor.invocations == [] and rig.commands(C.ToolExecute) == []
    assert_voiced_and_stored(rig, SPOKEN, "Stop!")
    assert_declined_records(rig, CALL_ID, "repairable")


@pytest.mark.parametrize(
    "draft",
    [
        pytest.param(NAMELESS, id="refused_nameless"),
        pytest.param(OUT_OF_RANGE_CLICK, id="refused_out_of_range"),
    ],
)
async def test_declined_refused_verdict_keeps_decline_reason(tmp_path: Path, draft: str) -> None:
    """Criterion 2, "whatever the verdict": an already-refused draft (no v1 tool name; a click out of 0-999) in a
    declined turn reaches the machine with the decline reason, not its verdict's reason; its records keep
    ``verdict_status: "refused"``."""
    verdict = live.classify(draft)
    assert verdict.status == "refused", f"premise: {draft!r} classifies refused"
    assert verdict.reason != REASON, "premise: the verdict's own reason differs from the decline"
    async with tool_rig(tmp_path, script=[answer(SPOKEN), tool(draft)]) as rig:
        await rig.ask("Don't send it")
        await until_turns(rig, 1, "the declined refused-verdict turn")
        assert_declined_live(rig, PASS1_ID, CALL_ID)
    assert_no_repair(rig)
    assert_declined_records(rig, CALL_ID, "refused")


@pytest.mark.parametrize("typed", [False, True], ids=["voice", "typed"])
async def test_decline_takes_precedence_over_asked(tmp_path: Path, typed: bool) -> None:
    """Criteria 2 and 5, adversarial for mutation 10 (swap the precedence with ``asked the user first``): the
    transcript is a no **and** the model's voiced text asks before a valid click. The machine gets the decline reason;
    the records carry ``user_declined: true``, ``asked_first: true`` and ``verdict_status: "valid"``; the reason is
    still the decline."""
    async with tool_rig(tmp_path) as rig:
        await run_turn(rig, 1, [answer(ASK), tool(VALID_CLICK)], question=NO, typed=typed)
        assert_declined_live(rig, PASS1_ID, CALL_ID)
        assert final_message(rig, CALL_ID).reason != ASKED_REASON
    assert_no_repair(rig)
    assert_voiced_and_stored(rig, ASK, NO)
    assert_declined_records(rig, CALL_ID, "valid", asked=True)


# =====================================================================================================
# Criterion 3: loop step k >= 2 (and criterion 4's transcript, read through the spy)
# =====================================================================================================


async def test_step2_judge_declines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 3 (mutation 6, step 1 only) and criterion 4: the predicate is patched to decline on its second call
    only. Loop on; step 1 voices ``S1`` and clicks (executed, plain records); step 2's valid click is judged with the
    predicate true: ``t1-call-s2`` is refused with the decline reason and never executed, the loop ends ``refused``,
    no step-3 request is sent (one is scripted). The predicate was called twice, each time with the live turn's
    transcript (``open the settings``, the words joined with single spaces), which equals history's ``user_text``."""
    calls: list[str] = []

    def spy(transcript: str) -> bool:
        calls.append(transcript)
        return len(calls) == 2

    _patch_predicate(monkeypatch, spy)
    question = "open the  settings"  # two spaces: ``rig.ask`` splits on whitespace, the machine joins with one
    joined = "open the settings"
    async with tool_rig(tmp_path, policy=LOOP, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(tsl.S2), tool(tsl.CLICK2)], [answer(tsl.S3)])
        await rig.ask(question)
        await tsl.until_steps(rig, 1)
        await until_turns(rig, 1, "the loop turn")
        assert_declined_live(rig, "t1-gen-s2", "t1-call-s2")
    assert calls == [joined, joined], (
        f"T-0107e criteria 3/4: the predicate runs at each of the two judges on the live transcript; calls {calls}"
    )
    assert rig.state.history[-1].user_text == joined, rig.state.history[-1]
    assert [i.call_id for i in rig.executor.invocations] == [CALL_ID], (
        f"T-0107e criterion 3: only step 1's call executes; executor ran "
        f"{[i.call_id for i in rig.executor.invocations]}"
    )
    assert len(rig.llm.step_prompts) == 1, (
        f"T-0107e criterion 3: a declined step ends the loop; no step-3 request; step requests "
        f"{[r for r, _ in rig.llm.step_prompts]}"
    )
    assert rig.record("t1").extras.get("tool_loop_end") == "refused", rig.record("t1").extras
    assert rig.spoken_sources() == tsl.S1 + tsl.S2, (
        f"T-0107e criterion 3: step 1's then step 2's committed text are voiced; got {rig.spoken_sources()!r}"
    )
    assert_plain_records(rig, CALL_ID)
    assert_declined_records(rig, "t1-call-s2", "valid")


# =====================================================================================================
# Too wide: commands that are not a no execute (with the control)
# =====================================================================================================

#: criterion 1's false rows that are commands a user gives before an action, and the model's own "no"
ALMOST_NO = {
    "cancel_meeting": ("Cancel my 3pm meeting", SPOKEN),
    "stop_timer": ("Stop the timer", SPOKEN),
    "annule_commande": ("Annule la commande", SPOKEN),
    "dont_forget": ("Don't forget to save", SPOKEN),
    "dont_close_click_save": ("Don't close the tab, click Save", SPOKEN),
    "dont_alice_bob": ("Don't send it to Alice, send it to Bob", SPOKEN),
    "dont_worry": ("Don't worry, send it", SPOKEN),
    "no_worries": ("No worries, send it", SPOKEN),
    "no_problem": ("No problem, go ahead", SPOKEN),
    "no_need": ("No need to ask, just send it", SPOKEN),
    "no_wait_instead": ("no wait, send it to Bob instead", SPOKEN),
    "no_but": ("No, but send it tomorrow", SPOKEN),
    "no_dot_instead": ("No. Open the settings instead.", SPOKEN),
    "do_not_disturb": ("Do not disturb mode on", SPOKEN),
    "oh_no_reload": ("Oh no, the page crashed, reload it", SPOKEN),
    "notify": ("Notify Bob", SPOKEN),
    "number": ("No. 10 Downing Street opening hours", SPOKEN),
    "trailing_no": ("Which mark do you have ever seen? No", SPOKEN),
    # criterion 4: the model's text is never read
    "model_says_no": ("click the sign in button", "No, stop. I will click it. "),
}


@pytest.mark.parametrize("case", list(ALMOST_NO), ids=list(ALMOST_NO))
async def test_almost_no_commands_execute(tmp_path: Path, case: str) -> None:
    """Too wide (mutations 1-4; criterion 4's "never reads the model's text"): a transcript that is not a clear no by
    criterion 1 -- a command opened by a head (``Cancel my 3pm meeting``), a head reversed (``No, but ...``), a number
    (``No. 10``), a trailing ``No``, or a no in the model's own voiced text -- runs its valid click in dry run as at
    BASE, with no ``user_declined``/``verdict_status`` field; then the control turn is declined."""
    transcript, spoken = ALMOST_NO[case]
    async with tool_rig(tmp_path, script=[answer(spoken), tool(VALID_CLICK)]) as rig:
        await rig.ask(transcript)
        await until_turns(rig, 1, f"the {case} turn")
        assert rig.statuses(CALL_ID) == ["proposed", "dry_run"], (
            f"T-0107e criterion 1: {transcript!r} is not a no; {CALL_ID} published {rig.statuses(CALL_ID)}"
        )
        assert [(i.call_id, i.arguments) for i in rig.executor.invocations] == [
            (CALL_ID, CLICK_ARGS)
        ]
        assert rig.state.history[0].user_text == transcript, rig.state.history[0]
        await _control(rig, 2)
    assert_plain_records(rig, CALL_ID)
    recs = records_of(rig, CALL_ID)
    assert not [r for r in recs if "verdict_status" in r], (
        f"T-0107e criterion 5: verdict_status on an ordinary call: {recs}"
    )
    assert_declined_records(rig, "t2-call", "valid")


# =====================================================================================================
# Criterion 6: nothing held
# =====================================================================================================


async def test_previous_turn_no_does_not_carry(tmp_path: Path) -> None:
    """Criterion 6 (mutation 7, read history or the previous turn): turn 1 ``No, leave it.`` is declined; turn 2,
    an ordinary request whose model re-issues the same call, executes in dry run as at BASE although history holds
    ``No, leave it.`` (its prompt carries it); turn 3, typed ordinary, executes too; turn 4 ``Stop.`` is declined
    again: nothing was held either way."""
    async with tool_rig(tmp_path) as rig:
        await run_turn(rig, 1, [answer(SPOKEN), tool(VALID_CLICK)], question=NO)
        assert_declined_live(rig, PASS1_ID, CALL_ID)
        assert rig.executor.invocations == []

        await run_turn(
            rig, 2, [answer("Clicking it now. "), tool(VALID_CLICK)], question="click it please"
        )
        prompts = [p for rid, p in rig.llm.gen_prompts if "-t2-gen-" in rid]
        assert prompts, f"premise: turn 2 sent a generate; {[r for r, _ in rig.llm.gen_prompts]}"
        users = [m.text() for m in prompts[0].messages if m.role == "user"]
        assert any(NO in t for t in users), (
            f"premise: turn 2's prompt holds turn 1's 'No, leave it.' in history; user messages {users}"
        )
        assert rig.statuses("t2-call") == ["proposed", "dry_run"], (
            f"T-0107e criterion 6: a previous turn's no declines nothing; t2-call published "
            f"{rig.statuses('t2-call')}"
        )

        await run_turn(
            rig, 3, [answer("Done. "), tool(VALID_CLICK)], question="and again", typed=True
        )
        assert rig.statuses("t3-call") == ["proposed", "dry_run"], rig.statuses("t3-call")
        assert [i.call_id for i in rig.executor.invocations] == ["t2-call", "t3-call"], (
            rig.executor.invocations
        )

        await run_turn(rig, 4, [answer(SPOKEN), tool(VALID_CLICK)], question="Stop.")
        assert_declined_live(rig, "t4-gen", "t4-call")
    assert_declined_records(rig, CALL_ID, "valid")
    assert_plain_records(rig, "t2-call")
    assert_plain_records(rig, "t3-call")
    assert_declined_records(rig, "t4-call", "valid")


async def test_stop_while_thinking_is_next_words(tmp_path: Path) -> None:
    """Criterion 4 (the reviewers' "also read ``next_words``" mutant): the turn ``click the sign in button`` reaches
    THINKING with its generate held; ``Stop.`` is then heard (an STT word after the turn's end of turn, so the machine
    files it in ``next_words``, criterion 4 of T-0007). The live turn's words are unchanged, so its valid click
    executes in dry run as at BASE, with plain records; the stored ``user_text`` is the live turn's transcript.

    The live transcript is ``Uh, wait`` -- hesitation only, so not a no (criterion 1: nothing follows the skipped
    hesitation) -- chosen so that both shapes of the mutant decline: ``next_words`` alone (``Stop.``) and the live
    words followed by ``next_words`` (``Uh, wait Stop.``: the hesitation is skipped, then the head)."""
    question = "Uh, wait"
    async with tool_rig(tmp_path, script=[answer(SPOKEN), tool(VALID_CLICK)]) as rig:
        rig.llm.hold()
        await rig.ask(question)
        await rig.until(
            lambda: rig.state.state is TurnState.THINKING,
            "premise: the turn never reached THINKING",
        )
        assert rig.state.t_eot_ms is not None, "premise: the turn's end of turn is set"
        now = rig.clock.now_ms
        rig.submit(E.SttWord(t_ms=now, text="Stop.", start_ms=now - 150, end_ms=now))
        await rig.pump()
        assert rig.state.state is TurnState.THINKING, rig.state.state
        assert [w.text for w in rig.state.next_words] == ["Stop."], (
            f"premise: 'Stop.' lands in next_words; next_words {rig.state.next_words}, words {rig.state.words}"
        )
        assert " ".join(w.text for w in rig.state.words) == question, rig.state.words
        rig.llm.release()
        await until_turns(rig, 1, "the thinking turn")
        assert rig.statuses(CALL_ID) == ["proposed", "dry_run"], (
            f"T-0107e criterion 4: a 'Stop.' in next_words belongs to the next turn and declines nothing in the "
            f"live one; {CALL_ID} published {rig.statuses(CALL_ID)}"
        )
        assert [i.call_id for i in rig.executor.invocations] == [CALL_ID], rig.executor.invocations
        assert rig.state.history[0].user_text == question, rig.state.history[0]
    assert_plain_records(rig, CALL_ID)


# =====================================================================================================
# Criterion 4: a ui-origin turn never declines (H-T107e-2)
# =====================================================================================================


@pytest.mark.parametrize(
    ("name", "context"),
    [
        pytest.param("__gate.clicked", {"label": "No"}, id="gate_no"),
        pytest.param("__gate.clicked", {"label": "Cancel"}, id="gate_cancel"),
        pytest.param("no", {}, id="action_named_no"),
    ],
)
async def test_ui_click_never_declines(tmp_path: Path, name: str, context: dict[str, Any]) -> None:
    """Criterion 4 (H-T107e-2: cards are illustrations): a click on a gate-made ``No``/``Cancel`` button, or an action
    named ``no``, opens a ui-origin turn whose transcript is ``[ui] <name> <context>``; it never declines, so the
    model's valid click executes in dry run as at BASE; then the control turn is declined."""
    async with tool_rig(tmp_path, script=[answer(SPOKEN), tool(VALID_CLICK)]) as rig:
        rig.submit(
            E.UiAction(t_ms=rig.clock.now_ms, surface_id="turn-t0", name=name, context=context)
        )
        await until_turns(rig, 1, "the ui turn")
        stored = rig.state.history[0].user_text
        assert stored.startswith("[ui] "), f"premise: a ui-origin transcript; got {stored!r}"
        assert rig.statuses(CALL_ID) == ["proposed", "dry_run"], (
            f"T-0107e criterion 4: a ui click never declines; {CALL_ID} published {rig.statuses(CALL_ID)}"
        )
        await _control(rig, 2)
    assert_plain_records(rig, CALL_ID)
    assert_declined_records(rig, "t2-call", "valid")


# =====================================================================================================
# Criterion 7: barge-in, no double handling (I2)
# =====================================================================================================


async def test_barge_in_stop_single_final(tmp_path: Path) -> None:
    """Criterion 7 (I2): turn 1's valid call is proposed while its long prefix plays; the user barges in and says
    ``Stop.``. Turn 1's call gets BASE's ``tool.cancel(interrupted)`` (card ``cancelled``, reason ``interrupted``);
    the turn the interruption opens refuses its own valid draft with the decline reason. Each call id has exactly one
    final ``astra.tool.call``, and nothing executes."""
    long_answer = "I will click the Sign in button at the top of the page for you right now. "
    async with tool_rig(tmp_path, script=[answer(long_answer), tool(VALID_CLICK)]) as rig:
        await rig.ask("sign me in")
        await rig.until(lambda: rig.state.llm_done, "pass 1 never finished")
        assert rig.state.state is TurnState.SPEAKING, "premise: the prefix is playing"
        await rig.until_status("proposed")
        rig.llm.script = [answer(SPOKEN), tool(VALID_CLICK)]
        now = rig.clock.now_ms
        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
        await rig.advance(800)
        await rig.until(lambda: "cancelled" in rig.statuses(), "the call was never cancelled")
        await rig.until(lambda: rig.state.turn_id == "t2", "the interruption opened no turn")
        await rig.words("Stop.")
        await rig.end_of_turn()
        await until_turns(rig, 2, "the Stop turn")
        first = final_message(rig, CALL_ID)
        assert (first.status, first.reason) == ("cancelled", "interrupted"), first
        assert assert_one_final(rig, CALL_ID) == "cancelled"
        assert assert_one_final(rig, "t2-call") == "refused", (
            f"T-0107e criterion 7: the Stop turn refuses its own draft; t2-call published {rig.statuses('t2-call')}"
        )
        assert_declined_live(rig, "t2-gen", "t2-call")
        cancels = [(c.call_id, c.reason) for c in rig.commands(C.ToolCancel)]
        assert cancels == [(CALL_ID, "interrupted"), ("t2-call", "refused")], (
            f"T-0107e criterion 7: one tool.cancel per call, BASE's for the interrupted one; got {cancels}"
        )
    assert rig.executor.invocations == [] and rig.commands(C.ToolExecute) == []
    assert rig.state.history[-1].user_text == "Stop.", rig.state.history[-1]
    assert_declined_records(rig, "t2-call", "valid")
    assert_plain_records(rig, CALL_ID)


# =====================================================================================================
# Criterion 2/3: no invocation at all in a declined turn
# =====================================================================================================

#: (policy, transcript, typed, step-1 script, step scripts, continuation)
DECLINED: dict[
    str, tuple[Any, str, bool, list[Commit], list[list[Commit]], list[Commit] | None]
] = {
    "stop_valid": (POLICY, "Stop!", False, [answer(SPOKEN), tool(VALID_CLICK)], [], None),
    "curly_dont_typed": (
        POLICY,
        "Don’t send it",
        True,
        [answer(SPOKEN), tool(VALID_CLICK)],
        [],
        None,
    ),
    "cancel_repairable": (
        POLICY,
        "Cancel.",
        False,
        [answer(SPOKEN), tool(REPAIRABLE_CLICK)],
        [],
        [tool(REPAIRED_CLICK)],
    ),
    "non_merci_loop": (
        LOOP,
        "Non merci",
        False,
        [answer(tsl.S1), tool(VALID_CLICK)],
        [[answer(tsl.S2), tool(tsl.CLICK2)], [answer(tsl.S3)]],
        None,
    ),
    "wait_no_dont_loop_typed": (
        LOOP,
        "Wait, no, don't.",
        True,
        [answer(tsl.S1), tool(VALID_CLICK)],
        [[answer(tsl.S2), tool(tsl.CLICK2)], [answer(tsl.S3)]],
        None,
    ),
    "no_other_one": (
        POLICY,
        "No, the other one",
        False,
        [answer(SPOKEN), tool(VALID_CLICK)],
        [],
        None,
    ),
}


@pytest.mark.parametrize("name", list(DECLINED), ids=list(DECLINED))
async def test_no_invocation_in_declined_turn(tmp_path: Path, name: str) -> None:
    """Criteria 2/3 over the fakes: in each declined turn (voice or typed; loop off or on with steps scripted; valid
    or repairable with a repair scripted; criterion 1 true forms incl. U+2019 and the accepted over-refusal) no
    ``ToolInvocation`` reaches the executor, no ``tool.execute`` is issued, no step request and no repair is sent."""
    policy, transcript, typed, script, steps, continuation = DECLINED[name]
    async with tool_rig(tmp_path, policy=policy) as rig:
        await run_turn(
            rig,
            1,
            script,
            question=transcript,
            steps=steps,
            continuation=continuation,
            typed=typed,
        )
        assert_declined_live(rig, PASS1_ID, CALL_ID)
    assert rig.executor.invocations == [], (
        f"T-0107e: a declined turn executes nothing; the executor ran "
        f"{[i.call_id for i in rig.executor.invocations]}"
    )
    assert rig.commands(C.ToolExecute) == [], rig.commands(C.ToolExecute)
    assert rig.llm.step_prompts == [], [r for r, _ in rig.llm.step_prompts]
    assert_no_repair(rig)
    assert rig.state.history[0].user_text == transcript, rig.state.history[0]


# =====================================================================================================
# Criterion 5: the log fields
# =====================================================================================================


async def test_log_fields_absent_on_ordinary_calls(tmp_path: Path) -> None:
    """Criterion 5, last item (mutation 8, the fields on every call): one session, loop off -- a valid call, a
    repairable call and its repair, a refused (nameless) call, and an asked-first call, each in a turn that is not a
    no: no record holds ``user_declined`` (not even ``false``/``null``); the ordinary ones hold no ``verdict_status``
    and the asked-first one keeps T-0107a's fields only. Then a declined turn's call holds both new fields on every
    record."""
    async with tool_rig(tmp_path) as rig:
        await run_turn(rig, 1, [answer(SPOKEN), tool(VALID_CLICK)], question="go ahead")
        await run_turn(
            rig,
            2,
            [answer(SPOKEN), tool(REPAIRABLE_CLICK)],
            question="click the small one",
            continuation=[tool(REPAIRED_CLICK)],
        )
        await run_turn(rig, 3, [answer(SPOKEN), tool(NAMELESS)], question="open example dot org")
        await run_turn(rig, 4, [answer(ASK), tool(VALID_CLICK)], question="buy it")
        await run_turn(rig, 5, [answer(SPOKEN), tool(VALID_CLICK)], question="Nope")
    assert rig.statuses("t1-call")[-1] == "dry_run" and "repairing" in rig.statuses("t2-call"), (
        f"premise: the ordinary calls ran their paths; {rig.statuses('t1-call')}, {rig.statuses('t2-call')}"
    )
    assert rig.statuses("t3-call") == ["refused"], rig.statuses("t3-call")
    assert final_message(rig, "t4-call").reason == ASKED_REASON, final_message(rig, "t4-call")
    for cid in ("t1-call", "t2-call", "t3-call", "t4-call"):
        assert_plain_records(rig, cid)
    for cid in ("t1-call", "t2-call", "t3-call"):
        extra = [r for r in records_of(rig, cid) if "verdict_status" in r or "asked_first" in r]
        assert not extra, f"T-0107e criterion 5: ordinary records are as at BASE; {cid}: {extra}"
    asked = [(r.get("asked_first"), r.get("verdict_status")) for r in records_of(rig, "t4-call")]
    assert asked and all(a == (True, "valid") for a in asked), asked
    assert_declined_records(rig, "t5-call", "valid")
