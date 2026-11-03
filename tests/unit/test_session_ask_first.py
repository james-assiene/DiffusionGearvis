"""T-0107a criteria 2-5: a step whose voiced text asks the user a question executes no call (over the session fakes).

Every row runs the T-0007 session over ``tests/unit/_tool_turn_support.tool_rig`` (``ToolLlm``, the recording
executor, the HUD ``Recorder``, the real session log, ``ScriptedClock``). Expected values are written by hand from
the task: the refusal reason is the literal of the task's Definition (``ASKED_FIRST_REASON = "asked the user
first"``), the call ids are T-0092c's/T-0114's (``t1-call``, ``t1-call-s2``), what is voiced and stored is the
step's own committed answer text (H4, H8), and every verdict is premised on ``live.classify`` (T-0091, not under
test here).

**A row that must stay green when the rule is too wide is still red at the base.** At d3765de no call is ever
refused for asking, so a row that only checks "this does not refuse" would pass at the base and XPASS its strict
pending marker. Every such row therefore runs, in the same session and after its own turn, the control turn
``_control`` (``Shall I place the order for 42.99 euros?`` + a valid click), which must be refused: the row is red at
the base on the control and, once implemented, discriminates both directions (too wide: its own turn; absent: the
control).

Readings where the task is silent (listed in the test-author's report):

- the refused card is published once (``["refused"]``, no ``proposed`` first): an asked-first draft reaches the
  machine ``refused`` at detection, so T-0092c's refused-at-detection publication (criteria 1/4) applies;
- the two log fields are on **every** ``tool.call`` record of an asked-first call (one per published message) and on
  no record of any other call; ``asked_first`` is the literal ``True`` and ``verdict_status`` the ``live.classify``
  status before the rule (``valid``, ``repairable`` or ``refused``);
- a loop step k >= 2's "voiced text" is the answer text of that step's stream (what R15 voices when its call is
  refused), never its thought and never text after a cut ``<end_of_turn>`` (T-0126);
- for a step-2 question the turn's voice is step 1's text then step 2's (``spoken_sources``), as T-0114's
  step-2-refusal row has it.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools ``dry_run``, loop off
(``POLICY``) or on (``TurnPolicy(thinking=True, tool_loop=True)``) as each row says, n = 1, deterministic.
"""

from __future__ import annotations

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
    final_message,
    thought,
    tool,
    tool_rig,
)
from tests.unit.test_session_loop import POLICY

#: the task's Definition, verbatim (never imported from the module under test)
REASON = "asked the user first"
LOOP = tsl.LOOP
CLICK_ARGS = {"x": 500, "y": 250}

#: criterion 1's true forms, each followed by the space a committed sentence carries
ASK = "Shall I place the order for 42.99 euros? "
ASKS = {
    "order": ASK,
    "send_mid_text": "Send it to Omar? He is in your chats. ",
    "french_spacing": "Je passe la commande ? ",
    "closing_quote": 'Delete the standup event?" ',
    "fullwidth": "Shall I delete the standup event？ ",
    "bracket": "Should I go ahead? (It can't be undone.) ",
    # the 2026-10-19 widening (docs/tasks/T-0107a.md, Rulings)
    "interrobang": "Shall I pay now?! ",
    "guillemets": "« Je passe la commande ? » ",
    # the definition replaced after reviewers A and B (2026-10-19): NBSP French typography (the call executed in
    # B's probe under the widened definition) and Markdown bold
    "guillemets_nbsp": "« Je passe la commande ? » ",
    "markdown_bold": "**Shall I pay?** ",
}
#: criterion 1's false forms that a voiced prefix can carry (the click-02 labels and the near misses)
NOT_ASKS = {
    "label_bare": "I will click the Forgot password? link. ",
    "label_quoted": "I will click the 'Forgot password?' link. ",
    "digits": "Is it 3?4 ",
    "lowercase_after": "what? no idea. I will click it. ",
}

KEYS = ("asked_first", "verdict_status")


# --- reading the rig ----------------------------------------------------------------------------------------


def done_tool(rig: ToolRig, request_id: str) -> Any:
    found = rig.dones(request_id)
    assert len(found) == 1, f"one LlmDone({request_id!r}) reached the machine; got {found}"
    assert found[0].tool is not None, f"LlmDone({request_id!r}) carries no ToolDraft"
    return found[0].tool


def records_of(rig: ToolRig, call_id: str) -> list[dict[str, Any]]:
    return [r for r in rig.tool_records() if r.get("call_id") == call_id]


def assert_refused_asked(rig: ToolRig, request_id: str, call_id: str, verdict: str) -> None:
    """Criterion 2/3: the machine is handed ``ToolDraft(status="refused", reason=REASON)``; the card ends refused with
    that reason; ``tool.cancel(refused)``; every ``tool.call`` record of the call carries the two fields (the log is
    read back after ``aclose``, so call this after the rig's block)."""
    assert_refused_live(rig, request_id, call_id)
    assert_asked_records(rig, call_id, verdict)


def assert_refused_live(rig: ToolRig, request_id: str, call_id: str) -> None:
    """The machine, HUD and command half of ``assert_refused_asked`` (callable inside the rig's block)."""
    d = done_tool(rig, request_id)
    assert (d.call_id, d.status, d.reason) == (call_id, "refused", REASON), (
        f"T-0107a criteria 2/3: the step's voiced text asks, so {call_id} reaches the machine "
        f"ToolDraft(status='refused', reason={REASON!r}); got {d!r}"
    )
    assert rig.statuses(call_id) == ["refused"], (
        f"T-0107a criterion 2 (R5, T-0092c 1/4): one refused card for {call_id}; published {rig.statuses(call_id)}"
    )
    assert final_message(rig, call_id).reason == REASON, final_message(rig, call_id)
    cancels = [c.reason for c in rig.commands(C.ToolCancel) if c.call_id == call_id]
    assert cancels == ["refused"], (
        f"T-0107a criterion 2: tool.cancel(refused) for {call_id}; got {cancels}"
    )


def assert_asked_records(rig: ToolRig, call_id: str, verdict: str) -> None:
    recs = records_of(rig, call_id)
    assert recs, f"T-0107a criterion 2: no tool.call record for {call_id}"
    got = [(r.get("asked_first"), r.get("verdict_status")) for r in recs]
    assert got == [(True, verdict)] * len(recs), (
        f"T-0107a criterion 2: every tool.call record of {call_id} carries asked_first: true and "
        f"verdict_status: {verdict!r}; got {got}"
    )


def assert_plain_records(rig: ToolRig, call_id: str) -> None:
    recs = records_of(rig, call_id)
    assert recs, f"premise: {call_id} wrote tool.call records"
    carrying = [{k: r[k] for k in KEYS if k in r} for r in recs if any(k in r for k in KEYS)]
    assert not carrying, (
        f"T-0107a criterion 2: asked_first/verdict_status are written only on asked-first calls; {call_id} has "
        f"{carrying}"
    )


def assert_nothing_executed(rig: ToolRig, call_id: str) -> None:
    ran = [i.call_id for i in rig.executor.invocations if i.call_id == call_id]
    executes = [c for c in rig.commands(C.ToolExecute) if c.call_id == call_id]
    assert not ran and not executes, (
        f"T-0107a criterion 2: no tool.execute for an asked-first call; executor ran {ran}, ToolExecute {executes}"
    )


def assert_no_repair(rig: ToolRig) -> None:
    assert rig.llm.tokenized == [] and rig.llm.continuations() == [], (
        "T-0107a criterion 2 (decide before any repair): no tokenize and no repair request; "
        f"tokenized {len(rig.llm.tokenized)}, continuations {len(rig.llm.continuations())}"
    )
    assert rig.commands(C.ToolRepairStart) == [], rig.commands(C.ToolRepairStart)


def assert_voiced_and_stored(rig: ToolRig, text: str, turn: int = 1) -> None:
    """H4/H8: the device hears the step's committed answer text and history stores exactly it."""
    voiced = " ".join(
        c.text for c in rig.speaks() if c.utterance_id.split("-")[0] == f"t{turn}"
    ).split()
    assert voiced == text.split(), (
        f"T-0107a criterion 2: the question is voiced as committed; voiced {voiced}, expected {text.split()}"
    )
    stored = rig.state.history[turn - 1].assistant_spoken
    assert stored == text, f"T-0107a criterion 2: history stores the question; got {stored!r}"


async def until_turns(rig: ToolRig, n: int, what: str, *, limit_ms: int = 30_000) -> None:
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= n,
        f"T-0107a: {what} never ended",
        limit_ms=limit_ms,
    )
    await rig.advance(300)


async def run_turn(
    rig: ToolRig,
    n: int,
    script: Sequence[Commit],
    *,
    steps: Sequence[Sequence[Commit]] = (),
    continuation: Sequence[Commit] | None = None,
    question: str = "go ahead",
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
    """The control turn: ``ASK`` + a valid click, which must be refused (red at the base: nothing asks there)."""
    await run_turn(rig, n, [answer(ASK), tool(VALID_CLICK)], question="buy it")
    assert_refused_live(rig, f"t{n}-gen", f"t{n}-call")
    assert_nothing_executed(rig, f"t{n}-call")


# =====================================================================================================
# Criterion 2: step 1
# =====================================================================================================


@pytest.mark.parametrize("policy", [POLICY, LOOP], ids=["loop_off", "loop_on"])
@pytest.mark.parametrize("prefix", list(ASKS.values()), ids=list(ASKS))
async def test_valid_call_after_question_is_refused(
    tmp_path: Path, prefix: str, policy: Any
) -> None:
    """Criterion 2 (mutations 1, 2, 7): a valid click after a spoken prefix that asks -- each true form of criterion
    1, U+FF1F and the bracket settlement included -- is refused with the reason before it can execute: no
    ``tool.execute``, no repair, ``tool.cancel(refused)``, the card ``refused`` with the reason, the question voiced
    and stored as committed, the two log fields, and with the loop on no step request and
    ``tool_loop_end == "refused"``."""
    assert live.classify(VALID_CLICK).status == "valid", "premise: the click is valid"
    async with tool_rig(tmp_path, policy=policy, script=[answer(prefix), tool(VALID_CLICK)]) as rig:
        if policy is LOOP:
            rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask("buy the blue one")
        await until_turns(rig, 1, "the asking turn")
    assert_refused_asked(rig, PASS1_ID, CALL_ID, "valid")
    assert_nothing_executed(rig, CALL_ID)
    assert rig.executor.invocations == [], rig.executor.invocations
    assert_no_repair(rig)
    assert_voiced_and_stored(rig, prefix)
    assert_never_voiced(rig, arguments=CLICK_ARGS)
    assert_history_clean(rig, arguments=CLICK_ARGS)
    record = rig.record("t1")
    assert record.tool is not None and record.tool.get("status") == "refused", record.tool
    if policy is LOOP:
        assert rig.llm.step_prompts == [], (
            f"T-0107a criterion 2: no step request after an asked-first step 1; "
            f"{[r for r, _ in rig.llm.step_prompts]}"
        )
        assert record.extras.get("tool_loop_end") == "refused", record.extras


#: T-0107a2 (owned files, the session row): the glued question, verbatim from the task
GLUED = "Shall I pay?Then "


@pytest.mark.parametrize("policy", [POLICY, LOOP], ids=["loop_off", "loop_on"])
async def test_glued_question_refuses(tmp_path: Path, policy: Any) -> None:
    """T-0107a2 (criterion 2 over the session, named re-scope R1): ``Shall I pay?Then `` -- a ``?`` glued to an
    uppercase word, the model's real shape -- before a valid click asks, so the click is refused with the reason
    before it can execute, loop off and on: the same pins as ``test_valid_call_after_question_is_refused`` (no
    ``tool.execute``, no repair, ``tool.cancel(refused)``, the refused card, the two log fields, the text voiced and
    stored as committed; with the loop on, no step request and ``tool_loop_end == "refused"``). Red at the base
    (87cf028), where clause (a) makes the glued mark a non-question and the click executes in dry run. Unmarked:
    ``tests/conftest.py`` accepts no ``T-0107a2`` pending id (``^T-\\d{4}[a-z]?$``)."""
    assert live.classify(VALID_CLICK).status == "valid", "premise: the click is valid"
    async with tool_rig(tmp_path, policy=policy, script=[answer(GLUED), tool(VALID_CLICK)]) as rig:
        if policy is LOOP:
            rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask("buy the blue one")
        await until_turns(rig, 1, "the glued-question turn")
    assert_refused_asked(rig, PASS1_ID, CALL_ID, "valid")
    assert_nothing_executed(rig, CALL_ID)
    assert rig.executor.invocations == [], rig.executor.invocations
    assert_no_repair(rig)
    assert_voiced_and_stored(rig, GLUED)
    assert_never_voiced(rig, arguments=CLICK_ARGS)
    assert_history_clean(rig, arguments=CLICK_ARGS)
    if policy is LOOP:
        assert rig.llm.step_prompts == [], (
            f"T-0107a2: no step request after an asked-first step 1; {[r for r, _ in rig.llm.step_prompts]}"
        )
        assert rig.record("t1").extras.get("tool_loop_end") == "refused", rig.record("t1").extras


@pytest.mark.parametrize(
    ("draft", "verdict"),
    [
        pytest.param(REPAIRABLE_CLICK, "repairable", id="repairable_click"),
        pytest.param(ALIASED_KEY, "repairable", id="repairable_alias"),
        pytest.param(NAMELESS, "refused", id="refused_nameless"),
        pytest.param(OUT_OF_RANGE_CLICK, "refused", id="refused_out_of_range"),
    ],
)
async def test_repairable_after_question_starts_no_repair(
    tmp_path: Path, draft: str, verdict: str
) -> None:
    """Criterion 2, adversarial (mutation 4, refuse only ``valid`` drafts; "decide before any repair"): a repairable
    draft after a question is refused with the reason -- no tokenize, no repair request, no ``ToolRepairStart``, no
    execution, though a valid repair is scripted -- and an already-refused draft takes the reason too ("whatever the
    verdict"), its log record keeping the original verdict in ``verdict_status``."""
    assert live.classify(draft).status == verdict, f"premise: {draft!r} classifies {verdict}"
    async with tool_rig(
        tmp_path, script=[answer(ASK), tool(draft)], continuation=[tool(REPAIRED_CLICK)]
    ) as rig:
        await rig.ask("buy the blue one")
        await until_turns(rig, 1, "the asking turn")
    assert_refused_asked(rig, PASS1_ID, CALL_ID, verdict)
    assert_no_repair(rig)
    assert rig.executor.invocations == [] and rig.commands(C.ToolExecute) == []
    assert_voiced_and_stored(rig, ASK)
    assert_never_voiced(rig)


# =====================================================================================================
# Criterion 3: loop step k >= 2
# =====================================================================================================


@pytest.mark.parametrize("asked", [ASKS["order"], ASKS["send_mid_text"]], ids=["end", "mid_text"])
async def test_step2_question_ends_loop(tmp_path: Path, asked: str) -> None:
    """Criterion 3 (mutation 5, step 1 only; mutation 2 on the mid-text form): loop on; step 1 voices ``S1`` and
    clicks (executed); step 2's voiced text asks before a valid click: ``t1-call-s2`` is refused with the reason and
    never executed, the loop ends ``refused`` and no step-3 request is sent (a step-3 script is ready, so a missing
    rule would send it); the question is voiced after step 1's text; step 1's records carry neither field."""
    async with tool_rig(tmp_path, policy=LOOP, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(asked), tool(tsl.CLICK2)], [answer(tsl.S3)])
        await rig.ask("buy the blue one")
        await tsl.until_steps(rig, 1)
        await until_turns(rig, 1, "the loop turn")
    assert [i.call_id for i in rig.executor.invocations] == [CALL_ID], (
        f"T-0107a criterion 3: only step 1's call executes; executor ran "
        f"{[i.call_id for i in rig.executor.invocations]}"
    )
    assert_refused_asked(rig, "t1-gen-s2", "t1-call-s2", "valid")
    assert_nothing_executed(rig, "t1-call-s2")
    assert len(rig.llm.step_prompts) == 1, (
        f"T-0107a criterion 3: the loop ends at the asking step; no step-3 request; step requests "
        f"{[r for r, _ in rig.llm.step_prompts]}"
    )
    assert rig.record("t1").extras.get("tool_loop_end") == "refused", rig.record("t1").extras
    assert rig.spoken_sources() == tsl.S1 + asked, (
        f"T-0107a criterion 3: step 1's text then the question are voiced; got {rig.spoken_sources()!r}"
    )
    assert asked.strip() in rig.state.history[-1].assistant_spoken, rig.state.history[-1]
    assert_plain_records(rig, CALL_ID)
    assert_never_voiced(rig, arguments={"x": 120, "y": 640})
    assert_history_clean(rig, arguments={"x": 120, "y": 640})


# =====================================================================================================
# Adversarial: the rule reads the voiced text only
# =====================================================================================================

#: (script, voiced): a question that is never voiced -- in a thought segment, in an inline thought block of the
#: answer channel (the raw stream holds ``... the order? I think so. ...``, a sentence-final question), or after a cut ``<end_of_turn>`` (T-0126: the
#: routed raw text, ``step_text``, still holds it)
UNVOICED = {
    "thought_segment": (
        [thought("Shall I place the order? I think so. "), answer(SPOKEN), tool(VALID_CLICK)],
        SPOKEN,
    ),
    "inline_thought_block": (
        [
            answer(f"<|channel>thought Shall I place the order? I think so. <channel|>{SPOKEN}"),
            tool(VALID_CLICK),
        ],
        SPOKEN,
    ),
    "after_end_of_turn": (
        [answer(f"{SPOKEN}<end_of_turn>user: Shall I place the order? Yes. "), tool(VALID_CLICK)],
        SPOKEN,
    ),
}


@pytest.mark.parametrize("case", list(UNVOICED), ids=list(UNVOICED))
async def test_question_inside_thought_does_not_refuse(tmp_path: Path, case: str) -> None:
    """Adversarial (mutation 3, read the raw stream): a question the session never voices -- inside a thought, or
    cut after ``<end_of_turn>`` -- does not refuse: the click executes in dry run, the voice and history hold only
    the answer, the records carry neither field. Then the control turn is refused (red at the base)."""
    script, voiced = UNVOICED[case]
    async with tool_rig(tmp_path, script=script) as rig:
        await rig.ask("click the sign in button")
        await until_turns(rig, 1, f"the {case} turn")
        assert rig.statuses(CALL_ID) == ["proposed", "dry_run"], (
            f"T-0107a: an unvoiced question asks nothing; {CALL_ID} published {rig.statuses(CALL_ID)}"
        )
        assert [i.call_id for i in rig.executor.invocations] == [CALL_ID], rig.executor.invocations
        assert_voiced_and_stored(rig, voiced)
        await _control(rig, 2)
    assert_plain_records(rig, CALL_ID)
    assert_asked_records(rig, "t2-call", "valid")


#: step 2's raw answer: a voiced sentence, then a cut ``<end_of_turn>`` and an asking question the session never voices
STEP2_EOT = f"{tsl.S2} <end_of_turn>user: Shall I place the order? Yes. "


async def test_step2_question_after_end_of_turn_does_not_refuse(tmp_path: Path) -> None:
    """Adversarial (reviewer A's M9 gap: the step-2 version of ``after_end_of_turn``; mutation 3 at step k >= 2, read
    the step's raw ``step_text`` instead of its voiced text): loop on; step 1 voices ``S1`` and clicks; step 2's raw
    answer holds ``S2``, then ``<end_of_turn>`` (cut, T-0126) and ``Shall I place the order? Yes.``, then a valid
    click. The question is never voiced, so ``t1-call-s2`` is not refused: it executes in dry run (both calls ran),
    the loop goes on to step 3, the voice and history hold no part of the question, and neither call's records carry
    the two fields. Then the control turn is refused."""
    async with tool_rig(tmp_path, policy=LOOP, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(STEP2_EOT), tool(tsl.CLICK2)], [answer(tsl.S3)])
        await rig.ask("open the settings")
        await tsl.until_steps(rig, 1)
        await until_turns(rig, 1, "the step-2 end_of_turn turn")
        d = done_tool(rig, "t1-gen-s2")
        assert (d.call_id, d.status) == ("t1-call-s2", "valid"), (
            f"T-0107a: a question after a cut <end_of_turn> in step 2 asks nothing; got {d!r}"
        )
        assert rig.statuses("t1-call-s2") == ["proposed", "dry_run"], (
            f"T-0107a: t1-call-s2 published {rig.statuses('t1-call-s2')}"
        )
        assert [i.call_id for i in rig.executor.invocations] == [CALL_ID, "t1-call-s2"], (
            f"T-0107a: both steps' calls execute; executor ran {[i.call_id for i in rig.executor.invocations]}"
        )
        assert len(rig.llm.step_prompts) == 2, (
            f"premise: the loop went on to step 3; step requests {[r for r, _ in rig.llm.step_prompts]}"
        )
        voiced = rig.spoken_sources()
        stored = rig.state.history[-1].assistant_spoken
        for where, text in (("voiced", voiced), ("stored", stored)):
            bad = [w for w in ("Shall", "order", "user:", "<end_of_turn>") if w in text]
            assert not bad, (
                f"T-0107a (T-0126): the cut question reached the {where} text: {bad} in {text!r}"
            )
        await _control(rig, 2)
    assert_plain_records(rig, CALL_ID)
    assert_plain_records(rig, "t1-call-s2")
    assert_asked_records(rig, "t2-call", "valid")


@pytest.mark.parametrize("prefix", list(NOT_ASKS.values()), ids=list(NOT_ASKS))
async def test_quoted_label_question_executes(tmp_path: Path, prefix: str) -> None:
    """Too wide (mutation 1, "contains ?"; the ten click-02 trials of the Goal): a prefix whose ``?`` ends no
    sentence -- the bare and the quoted "Forgot password?" label, ``3?4``, ``what? no idea.`` -- executes its valid
    click in dry run as at the base, with neither log field; then the control turn is refused."""
    async with tool_rig(tmp_path, script=[answer(prefix), tool(VALID_CLICK)]) as rig:
        await rig.ask("click forgot password")
        await until_turns(rig, 1, "the label turn")
        assert rig.statuses(CALL_ID) == ["proposed", "dry_run"], (
            f"T-0107a criterion 1: {prefix!r} does not ask; {CALL_ID} published {rig.statuses(CALL_ID)}"
        )
        assert [(i.call_id, i.arguments) for i in rig.executor.invocations] == [
            (CALL_ID, CLICK_ARGS)
        ]
        assert_voiced_and_stored(rig, prefix)
        await _control(rig, 2)
    assert_plain_records(rig, CALL_ID)
    assert_asked_records(rig, "t2-call", "valid")


# =====================================================================================================
# Criterion 5: nothing held
# =====================================================================================================


async def test_next_turn_yes_is_ordinary(tmp_path: Path) -> None:
    """Criterion 5: after an asked-first refusal, a typed ``yes`` starts an ordinary ``llm.start`` whose prompt's
    history holds the question; the model re-issues the call with a voiced text that does not ask, and it executes
    in dry run (``t2-call``: proposed, dry_run; one invocation with the call's arguments; neither log field). A third
    turn that asks again is refused again: nothing was held."""
    async with tool_rig(tmp_path) as rig:
        await run_turn(rig, 1, [answer(ASK), tool(VALID_CLICK)], question="buy the blue one")
        assert_refused_live(rig, PASS1_ID, CALL_ID)
        assert rig.executor.invocations == []

        await run_turn(
            rig,
            2,
            [answer("Placing the order now. "), tool(VALID_CLICK)],
            question="yes",
            typed=True,
        )
        starts = [c for c in rig.commands(C.LlmStart) if c.request_id == "t2-gen"]
        assert [c.transcript for c in starts] == ["yes"], (
            f"T-0107a criterion 5: the yes is an ordinary llm.start; got {starts}"
        )
        prompts = [p for rid, p in rig.llm.gen_prompts if "-t2-gen-" in rid]
        assert prompts, f"premise: turn 2 sent a generate; {[r for r, _ in rig.llm.gen_prompts]}"
        assistant = [m.text() for m in prompts[0].messages if m.role == "assistant"]
        assert any(ASK.strip() in t for t in assistant), (
            f"T-0107a criterion 5 (H8): turn 2's history holds the question; assistant messages {assistant}"
        )
        assert rig.statuses("t2-call") == ["proposed", "dry_run"], rig.statuses("t2-call")
        assert [(i.call_id, i.arguments) for i in rig.executor.invocations] == [
            ("t2-call", CLICK_ARGS)
        ], rig.executor.invocations

        await run_turn(
            rig, 3, [answer(ASKS["send_mid_text"]), tool(VALID_CLICK)], question="and send it"
        )
        assert_refused_live(rig, "t3-gen", "t3-call")
        assert len(rig.executor.invocations) == 1, rig.executor.invocations
    assert_asked_records(rig, CALL_ID, "valid")
    assert_plain_records(rig, "t2-call")
    assert_asked_records(rig, "t3-call", "valid")


# =====================================================================================================
# Criterion 4: never in the same turn
# =====================================================================================================

#: (policy, step-1 script, step scripts, continuation, the asking step k, the call ids that may execute before it)
SCENARIOS: dict[
    str, tuple[Any, list[Commit], list[list[Commit]], list[Commit] | None, int, list[str]]
] = {
    "s1_valid": (POLICY, [answer(ASK), tool(VALID_CLICK)], [], None, 1, []),
    "s1_repairable": (
        POLICY,
        [answer(ASK), tool(REPAIRABLE_CLICK)],
        [],
        [tool(REPAIRED_CLICK)],
        1,
        [],
    ),
    "s1_fullwidth": (POLICY, [answer(ASKS["fullwidth"]), tool(VALID_CLICK)], [], None, 1, []),
    "s1_loop": (
        LOOP,
        [answer(ASKS["french_spacing"]), tool(VALID_CLICK)],
        [[answer(tsl.S2), tool(tsl.CLICK2)], [answer(tsl.S3)]],
        None,
        1,
        [],
    ),
    "s2_loop": (
        LOOP,
        [answer(tsl.S1), tool(VALID_CLICK)],
        [
            [answer(ASKS["closing_quote"]), tool(tsl.CLICK2)],
            [answer(tsl.S3), tool(tsl.CLICK3)],
            [answer(tsl.S3)],
        ],
        None,
        2,
        ["t1-call"],
    ),
    "s3_loop": (
        LOOP,
        [answer(tsl.S1), tool(VALID_CLICK)],
        [
            [answer(tsl.S2), tool(tsl.CLICK2)],
            [answer(ASKS["bracket"]), tool(tsl.CLICK3)],
            [answer(tsl.S3)],
        ],
        None,
        3,
        ["t1-call", "t1-call-s2"],
    ),
}


@pytest.mark.parametrize("name", list(SCENARIOS), ids=list(SCENARIOS))
async def test_no_invocation_after_asking_step(tmp_path: Path, name: str) -> None:
    """Criterion 4 over the fakes: in each scripted turn where a step's voiced text asks (step 1, 2 or 3; loop off
    or on; valid or repairable; U+FF1F, French spacing, a closing quote, a bracket), no ``ToolInvocation`` reaches
    the executor after that step's answer text is committed: the executor ran exactly the earlier steps' calls, and
    no ``tool.execute`` is issued after the asking step's first answer commit reaches the machine."""
    policy, script, steps, continuation, k, before = SCENARIOS[name]
    asking = tsl.step_id(k)
    async with tool_rig(tmp_path, policy=policy, script=script, continuation=continuation) as rig:
        if steps:
            rig.llm.script_steps(*steps)
        await rig.ask("do it")
        if k > 1:
            await tsl.until_steps(rig, k - 1)
        await until_turns(rig, 1, f"the {name} turn")
    ran = [i.call_id for i in rig.executor.invocations]
    assert ran == before, (
        f"T-0107a criterion 4: step {k} asks, so only {before} may execute; the executor ran {ran}"
    )
    log = rig.machine.log
    first = next(
        (
            i
            for i, (ev, _s, _c) in enumerate(log)
            if isinstance(ev, E.LlmCommit) and ev.request_id == asking and ev.segment == "answer"
        ),
        None,
    )
    assert first is not None, f"premise: step {k}'s answer reached the machine ({asking})"
    late = [c for _ev, _s, cmds in log[first:] for c in cmds if isinstance(c, C.ToolExecute)]
    assert late == [], (
        f"T-0107a criterion 4: a tool.execute after the asking step's answer was committed: {late}"
    )
    d = done_tool(rig, asking)
    assert (d.status, d.reason) == ("refused", REASON), d


# =====================================================================================================
# The log fields
# =====================================================================================================


async def test_log_fields_absent_on_ordinary_calls(tmp_path: Path) -> None:
    """Criterion 2, last sentence (mutation 6, the fields on every call): one session, loop off -- a valid call, a
    repairable call and its repair, a refused (nameless) call, each after a prefix that does not ask: none of their
    ``tool.call`` records holds ``asked_first`` or ``verdict_status`` (not even ``false``/``null``); then an asking
    turn's call holds both on every record."""
    async with tool_rig(tmp_path) as rig:
        await run_turn(rig, 1, [answer(SPOKEN), tool(VALID_CLICK)])
        await run_turn(
            rig, 2, [answer(SPOKEN), tool(REPAIRABLE_CLICK)], continuation=[tool(REPAIRED_CLICK)]
        )
        await run_turn(rig, 3, [answer(SPOKEN), tool(NAMELESS)])
        await run_turn(rig, 4, [answer(ASK), tool(VALID_CLICK)], question="buy it")
    assert rig.statuses("t1-call")[-1] == "dry_run" and "repairing" in rig.statuses("t2-call"), (
        f"premise: the ordinary calls ran their paths; {rig.statuses('t1-call')}, {rig.statuses('t2-call')}"
    )
    assert rig.statuses("t3-call") == ["refused"], rig.statuses("t3-call")
    for cid in ("t1-call", "t2-call", "t3-call"):
        assert_plain_records(rig, cid)
    assert_refused_asked(rig, "t4-gen", "t4-call", "valid")
