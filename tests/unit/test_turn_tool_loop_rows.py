"""T-0112 (T-0111a a1) criteria 3-9 and 11: the tool-result loop's rows R11-R18, on the virtual clock.

Every expectation is written by hand from ``docs/tasks/T-0111a.md`` §2 (the rows table, Ids,
History) and §3 (D16 speech), with the rulings H-T111a-2 (budget, then one report step), -3 (a
step >= 2 is voiced only if it ends without an executed call), -4 (no new ``TurnState``) and -8 (a
refused or repairable call at step >= 2 ends the loop and voices the step). Nothing is captured
from an implementation. Time stamps are read off the ``clock`` fixture; deadlines are the
triggering event's ``t_ms`` plus the policy value, as for ``tool_repair`` in T-0089.

Where the spec fixes a whole tuple (R11, R12, R18) the row compares the multiset of commands, with
``hud.state`` last; where it fixes only some commands, the row asserts those and the forbidden ones.
"""

from __future__ import annotations

import types
import typing

import pydantic
import pytest

from local_astra.conductor.turn_machine import TurnMachine
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import HistoryTurn, MachineState, TurnState
from tests.unit.test_turn_invariants import LOOP_POLICY_DEFAULTS, loop_policy, tool_result_event
from tests.unit.test_turn_tool_rows import ARGS, done, draft, step, to_thinking

RESULT_MS = 15_000  # T-0111a §2: TurnPolicy.tool_result_timeout_ms
LLM_TIMEOUT_MS = 20_000  # TurnPolicy.llm_timeout_ms, armed again for every step's generate (R12)
USER = "open the menu"  # to_thinking's transcript
_MISSING = object()


# --- helpers -------------------------------------------------------------------------------------


def need(name: str):
    cls = getattr(C, name, _MISSING)
    assert cls is not _MISSING, f"T-0112: contracts/commands.py gains {name}"
    return cls


def loop(**overrides) -> TurnMachine:
    return TurnMachine(loop_policy(tool_loop=True, **overrides))


def say(m, s, clock, text, rid, *, open_=False, segment="answer"):
    ev = E.LlmCommit(
        t_ms=clock.advance(300),
        request_id=rid,
        canvas_idx=0,
        segment=segment,
        text=text,
        thought_open=open_,
    )
    return step(m, s, ev)


def drain(m, s, clock, text):
    return step(m, s, E.PlaybackDrained(t_ms=clock.advance(50), played_ms=400, played_text=text))


def result(m, s, clock, call_id, status="dry_run"):
    return step(m, s, tool_result_event(clock.advance(40), call_id, status))


def execute(call_id, *, name="click_at", arguments=ARGS, turn_id="t1"):
    return C.ToolExecute(
        call_id=call_id, turn_id=turn_id, name=name, arguments=arguments, source="draft"
    )


def llm_continue(k: int, call_id: str, turn: str = "t1"):
    return need("LlmContinue")(request_id=f"{turn}-gen-s{k}", step=k, call_id=call_id)


def bag(cmds) -> list[str]:
    return sorted(repr(c) for c in cmds)


def of(cmds, cls) -> list:
    return [c for c in cmds if isinstance(c, cls)]


def sources(cmds) -> str:
    return "".join(c.source for c in of(cmds, C.TtsSpeak))


def cancels(cmds) -> list[tuple[str, str]]:
    return [(c.call_id, c.reason) for c in of(cmds, C.ToolCancel)]


def executed_step1(m, clock, prefix="A."):
    """Step 1 voices ``prefix`` and drafts the valid call c1; the drain completes it (R11)."""
    s = to_thinking(m, clock)
    s, _ = say(m, s, clock, prefix, "t1-gen")
    s, _ = step(m, s, done(clock.advance(10), "t1-gen", draft("valid")))
    assert s.state is TurnState.SPEAKING
    return drain(m, s, clock, prefix)


def at_step2(m, clock, prefix="A."):
    s, _ = executed_step1(m, clock, prefix)
    s, cmds = result(m, s, clock, "c1")
    assert s.active_generate == "t1-gen-s2", f"R12 did not start step 2: {cmds}"
    return s, cmds


# --- contracts (T-0111a §2 "Policy", "MachineState", "New event", "Other additions"; criterion 11)


def test_policy_defaults_are_loop_off_eight_steps_fifteen_seconds():
    p = loop_policy()
    got = {name: getattr(p, name) for name in LOOP_POLICY_DEFAULTS}
    assert got == LOOP_POLICY_DEFAULTS


def test_machine_state_gains_the_per_turn_loop_fields_with_their_defaults():
    fields = MachineState.model_fields
    expected = {
        "tool_step": 1,
        "tool_steps_executed": 0,
        "awaiting_result": None,
        "step_text": "",
        "played_prior": "",
    }
    missing = [name for name in expected if name not in fields]
    assert not missing, f"T-0111a §2: MachineState gains {missing}"
    s = MachineState()
    assert {name: getattr(s, name) for name in expected} == expected


def test_criterion_11_tool_result_carries_no_text():
    """The machine never receives tool-result text: the event has exactly these fields.

    T-0192 B1 re-pin (before implementation): the fields are T-0112's four plus exactly ``hand_back``, typed
    ``Literal["checkout"] | None`` with default ``None`` ("still no text"): it takes ``"checkout"`` or ``None`` and
    refuses any other string, so no text can ride on it."""
    cls = getattr(E, "ToolResult", None)
    assert cls is not None, "T-0112: contracts/events.py gains ToolResult"
    assert set(cls.model_fields) == {"type", "t_ms", "call_id", "status", "hand_back"}, (
        f"T-0112 criterion 11 / T-0192 B1: ToolResult's fields are {sorted(cls.model_fields)}"
    )
    hand_back = cls.model_fields["hand_back"]
    assert typing.get_origin(hand_back.annotation) in (typing.Union, types.UnionType)
    assert set(typing.get_args(hand_back.annotation)) == {typing.Literal["checkout"], type(None)}, (
        f'T-0192 B1: hand_back is Literal["checkout"] | None; it is {hand_back.annotation}'
    )
    assert hand_back.default is None and not hand_back.is_required()
    for status in ("dry_run", "executed", "refused", "failed", "timeout"):
        assert cls(t_ms=1, call_id="c1", status=status).status == status
        assert cls(t_ms=1, call_id="c1", status=status).hand_back is None
        assert (
            cls(t_ms=1, call_id="c1", status=status, hand_back="checkout").hand_back == "checkout"
        )
    with pytest.raises(pydantic.ValidationError):
        cls(t_ms=1, call_id="c1", status="dry_run", text="1. Paris 14C")
    with pytest.raises(pydantic.ValidationError):
        cls(t_ms=1, call_id="c1", status="ok")
    for text in ("I will complete the purchase", "Checkout", "checkout ", "", "true"):
        with pytest.raises(pydantic.ValidationError):
            cls(t_ms=1, call_id="c1", status="executed", hand_back=text)
    assert cls in E.EVENT_TYPES


def test_llm_continue_timer_and_cancel_reason_contracts():
    cls = need("LlmContinue")
    assert set(cls.model_fields) == {"type", "request_id", "step", "call_id"}
    assert "tool_result" in typing.get_args(C.TimerName)
    assert "step_budget" in typing.get_args(C.ToolCancelReason)
    # T-0111a §2 "Other additions": the post-T-0089 reasons plus exactly one new member
    assert set(typing.get_args(C.ToolCancelReason)) == {
        "interrupted",
        "takeover",
        "error",
        "disconnected",
        "refused",
        "repair_failed",
        "repair_timeout",
        "step_budget",
        "reset",  # T-0162 A-2 (pre-authorised pin move): _tool_cancel(state, "reset")
        "stopped",  # T-0193 A0-1 (census move): the HUD Stop button's _tool_cancel(state, "stopped")
    }


# --- criterion 1 (negative), by hand: loop off is today's R1 -------------------------------------


def test_loop_off_executes_and_idles_as_today_and_ignores_tool_results(clock):
    m = TurnMachine(loop_policy(tool_loop=False))
    s, cmds = executed_step1(m, clock)
    assert s.state is TurnState.IDLE
    assert of(cmds, C.ToolExecute) == [execute("c1")]
    assert of(cmds, C.CommitHistory) == [C.CommitHistory(user_only=False)]
    assert of(cmds, C.ScheduleSummary)
    after, cmds = result(m, s, clock, "c1")
    assert (after, cmds) == (s, [])


# --- criterion 3: R11 ----------------------------------------------------------------------------


def test_r11_drained_prefix_executes_and_arms_tool_result_in_one_tuple(clock):
    m = loop()
    s = to_thinking(m, clock)
    llm_start = s.llm_start_ms
    s, _ = say(m, s, clock, "A.", "t1-gen")
    s, _ = step(m, s, done(clock.advance(10), "t1-gen", draft("valid")))
    s, cmds = drain(m, s, clock, "A.")
    t = clock.now_ms
    assert bag(cmds[:-1]) == bag(
        [execute("c1"), C.ArmTimer(name="tool_result", at_ms=t + RESULT_MS)]
    )
    assert cmds[-1] == C.HudState(state="THINKING", turn_id="t1")
    assert s.state is TurnState.THINKING and s.awaiting_result == "c1"
    assert (s.tool_steps_executed, s.tool_step, s.active_generate) == (1, 1, None)
    assert s.played_prior == "A."
    assert s.history == (), "R11: no commit_history, so nothing is stored yet"
    assert s.llm_start_ms == llm_start and s.first_canvas_measured is True, "not reset per step"
    assert s.committed_answer == "A." and s.spoken_text == "A.", "cumulative"


def test_r11_with_nothing_spoken_executes_from_thinking_without_closing_the_turn(clock):
    m = loop()
    s = to_thinking(m, clock)
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen", draft("valid")))
    t = clock.now_ms
    assert bag(cmds) == bag(
        [
            C.CancelTimer(name="llm_timeout"),
            execute("c1"),
            C.ArmTimer(name="tool_result", at_ms=t + RESULT_MS),
        ]
    ), cmds
    assert s.state is TurnState.THINKING and s.awaiting_result == "c1"


def test_r11_adversarial_a_speak_after_the_last_drain_withholds_the_execute(clock):
    m = loop()
    s = to_thinking(m, clock)
    s, _ = say(m, s, clock, "A.", "t1-gen")
    s, _ = drain(m, s, clock, "A.")
    s, cmds = say(m, s, clock, "B.", "t1-gen")
    assert sources(cmds) == "B."
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen", draft("valid")))
    assert not of(cmds, C.ToolExecute), f"the drain did not cover B.: {cmds}"
    assert not [c for c in cmds if isinstance(c, C.ArmTimer) and c.name == "tool_result"]
    assert s.state is TurnState.SPEAKING
    s, cmds = drain(m, s, clock, "B.")
    assert of(cmds, C.ToolExecute) == [execute("c1")]
    assert s.state is TurnState.THINKING and s.awaiting_result == "c1"


def test_r11_resets_the_drain_so_a_resume_of_the_end_report_does_not_complete(clock):
    """R11 "per-step fields reset": step 1's drain does not count for step 2's speech. The end
    report pauses on an onset; a backchannel resumes it (criterion 17(a)) and must not end the
    turn, because nothing of "Found it." has drained yet. The next drain completes (R15)."""
    m = loop()
    s, _ = at_step2(m, clock)
    s, _ = say(m, s, clock, "Found it.", "t1-gen-s2")
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen-s2"))
    assert sources(cmds) == "Found it." and s.state is TurnState.SPEAKING
    onset = clock.advance(100)
    s, cmds = step(m, s, E.UserSpeechOnset(t_ms=onset, onset_ms=onset))
    assert s.state is TurnState.PAUSED and C.PlaybackPause() in cmds
    s, cmds = step(m, s, E.UserBackchannel(t_ms=clock.advance(200), duration_ms=200))
    assert s.state is TurnState.SPEAKING, f"the resume completed the turn: {cmds}"
    assert bag(cmds[:-1]) == bag([C.CancelTimer(name="takeover"), C.PlaybackResume()]), cmds
    assert cmds[-1] == C.HudState(state="SPEAKING", turn_id="t1")
    assert not of(cmds, C.CommitHistory) and not of(cmds, C.ScheduleSummary)
    assert s.history == ()
    s, cmds = drain(m, s, clock, "Found it.")
    assert s.state is TurnState.IDLE
    assert of(cmds, C.CommitHistory) == [C.CommitHistory(user_only=False)]
    assert s.history[-1] == HistoryTurn(user_text=USER, assistant_spoken="A. Found it.")


# --- criterion 4: R12 and R13 --------------------------------------------------------------------


@pytest.mark.parametrize("status", ["dry_run", "executed", "refused", "failed", "timeout"])
def test_r12_a_matching_result_emits_exactly_one_continue(clock, status):
    m = loop()
    s, _ = executed_step1(m, clock)
    s, cmds = result(m, s, clock, "c1", status)
    t = clock.now_ms
    assert bag(cmds) == bag(
        [
            C.CancelTimer(name="tool_result"),
            llm_continue(2, "c1"),
            C.ArmTimer(name="llm_timeout", at_ms=t + LLM_TIMEOUT_MS),
        ]
    ), cmds
    assert s.state is TurnState.THINKING
    assert (s.active_generate, s.tool_step, s.awaiting_result) == ("t1-gen-s2", 2, None)


def _duplicate(m, clock):
    s, _ = at_step2(m, clock)
    return s, "c1"


def _stale_id_while_awaiting(m, clock):
    s, _ = executed_step1(m, clock)
    return s, "c0"


def _before_execute(m, clock):
    s = to_thinking(m, clock)
    s, _ = say(m, s, clock, "A.", "t1-gen")
    s, _ = step(m, s, done(clock.advance(10), "t1-gen", draft("valid")))
    return s, "c1"


def _after_an_interrupt(m, clock):
    s, _ = _before_execute(m, clock)
    onset = clock.advance(100)
    s, _ = step(m, s, E.UserSpeechOnset(t_ms=onset, onset_ms=onset))
    s, cmds = step(m, s, E.UserTakeover(t_ms=clock.advance(700), onset_ms=onset))
    assert s.state is TurnState.INTERRUPTED and cancels(cmds) == [("c1", "interrupted")]
    return s, "c1"


def _after_a_takeover_after_execute(m, clock):
    s, _ = executed_step1(m, clock)
    s, _ = step(m, s, E.UserTakeover(t_ms=clock.advance(700), onset_ms=clock.now_ms - 600))
    assert s.turn_id == "t2"
    return s, "c1"


@pytest.mark.parametrize(
    "setup",
    [
        _duplicate,
        _stale_id_while_awaiting,
        _before_execute,
        _after_an_interrupt,
        _after_a_takeover_after_execute,
    ],
    ids=["duplicate", "stale", "before_execute", "after_interrupt", "after_takeover"],
)
def test_r13_adversarial_an_unmatched_result_emits_nothing(clock, setup):
    m = loop()
    s, call_id = setup(m, clock)
    after, cmds = result(m, s, clock, call_id)
    assert (after, cmds) == (s, []), f"R13: {cmds}"


def test_r13_adversarial_the_cut_turns_result_is_inert_once_the_next_turn_is_generating(clock):
    """R13 (any non-matching result: nothing) and R17 (the new turn opens with every per-turn
    field reset, §2 MachineState): c1 was awaited by t1, which the takeover closed. Once t2 is
    THINKING with its own generate live, c1's late result must not start a step of t2."""
    m = loop()
    s, _ = executed_step1(m, clock)
    s, _ = _new_words_then_takeover(m, s, clock)
    assert (s.state, s.turn_id) == (TurnState.LISTENING, "t2")
    s, cmds = _to_next_thinking(m, s, clock)
    assert of(cmds, C.LlmStart) == [C.LlmStart(request_id="t2-gen", transcript="no stop")]
    assert s.state is TurnState.THINKING and s.active_generate == "t2-gen"
    after, cmds = result(m, s, clock, "c1")
    assert not of(cmds, need("LlmContinue")), f"R13: t1's result continued t2: {cmds}"
    assert not of(cmds, C.ArmTimer), f"R13: {cmds}"
    assert (after, cmds) == (s, []), f"R13: {cmds}"


def test_r13_guard_width_a_stale_result_does_not_consume_the_wait(clock):
    m = loop()
    s, _ = executed_step1(m, clock)
    s, _ = result(m, s, clock, "c0")
    s, cmds = result(m, s, clock, "c1")
    assert of(cmds, need("LlmContinue")) == [llm_continue(2, "c1")]


def test_r13_guard_width_a_result_before_execute_does_not_block_the_execute(clock):
    m = loop()
    s, _ = _before_execute(m, clock)
    s, _ = result(m, s, clock, "c1")
    s, cmds = drain(m, s, clock, "A.")
    assert of(cmds, C.ToolExecute) == [execute("c1")]


# --- criterion 5: R14 and R15 --------------------------------------------------------------------


def test_r14_r15_nothing_is_spoken_before_the_step_2_done_then_exactly_step_text(clock):
    m = loop()
    s, _ = at_step2(m, clock)
    s, cmds = say(m, s, clock, "Found ", "t1-gen-s2")
    assert not of(cmds, C.TtsSpeak) and s.state is TurnState.THINKING, cmds
    s, cmds = say(m, s, clock, "it.", "t1-gen-s2")
    assert not of(cmds, C.TtsSpeak) and s.state is TurnState.THINKING, cmds
    assert s.step_text == "Found it."
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen-s2"))
    assert sources(cmds) == "Found it."
    assert not of(cmds, C.ToolExecute) and not of(cmds, C.ToolCancel)
    assert s.state is TurnState.SPEAKING
    s, cmds = drain(m, s, clock, "Found it.")
    assert s.state is TurnState.IDLE
    assert of(cmds, C.CommitHistory) == [C.CommitHistory(user_only=False)]
    assert of(cmds, C.ScheduleSummary)
    assert s.history[-1] == HistoryTurn(user_text=USER, assistant_spoken="A. Found it.")


def test_r15_a_valid_step_2_call_withholds_its_text_and_executes(clock):
    m = loop()
    s, _ = at_step2(m, clock)
    s, _ = say(m, s, clock, "W.", "t1-gen-s2")
    s, cmds = step(
        m,
        s,
        done(
            clock.advance(10),
            "t1-gen-s2",
            draft("valid", call_id="c2", name="google_search", arguments={"query": "x"}),
        ),
    )
    t = clock.now_ms
    assert bag(cmds) == bag(
        [
            C.CancelTimer(name="llm_timeout"),
            execute("c2", name="google_search", arguments={"query": "x"}),
            C.ArmTimer(name="tool_result", at_ms=t + RESULT_MS),
        ]
    ), cmds
    assert s.state is TurnState.THINKING and s.awaiting_result == "c2"
    assert s.tool_steps_executed == 2


def test_r14_adversarial_an_open_thought_at_step_2_contributes_nothing(clock):
    m = loop()
    s, _ = at_step2(m, clock)
    s, cmds = say(m, s, clock, "SECRET plan", "t1-gen-s2", open_=True)
    assert not of(cmds, C.TtsSpeak)
    s, _ = say(m, s, clock, "B.", "t1-gen-s2")
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen-s2"))
    assert sources(cmds) == "B."


def test_r15_an_empty_toolless_step_ends_idle_with_the_played_history(clock):
    m = loop()
    s, _ = at_step2(m, clock)
    s, _ = say(m, s, clock, "SECRET plan", "t1-gen-s2", open_=True)
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen-s2"))
    assert not of(cmds, C.TtsSpeak)
    assert s.state is TurnState.IDLE
    assert of(cmds, C.CommitHistory), cmds
    assert s.history[-1] == HistoryTurn(user_text=USER, assistant_spoken="A.")


# T-0178 B3 re-scope: a non-policy refusal's text is withheld (was: voiced)
def test_r15_a_refused_step_2_call_is_cancelled_and_its_text_withheld(clock):
    """T-0178 B3/B4a (re-scopes T-0112's "…its_text_voiced"): the draft's reason ("nameless") is neither ask-first nor
    user-declined, so the step's text is withheld, the ``not_finished`` clip plays and the turn completes with the
    earlier played text (the policy refusals that still voice are ``test_t0178_machine.py``'s BASE differential)."""
    m = loop()
    s, _ = at_step2(m, clock)
    s, _ = say(m, s, clock, "I cannot.", "t1-gen-s2")
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen-s2", draft("refused", call_id="c2")))
    assert cancels(cmds) == [("c2", "refused")]
    assert sources(cmds) == "" and not of(cmds, C.TtsSpeak), cmds
    assert not of(cmds, C.ToolExecute)
    assert [c.kind for c in of(cmds, C.PlayClip)] == ["not_finished"], cmds
    assert s.state is TurnState.IDLE
    assert s.history[-1] == HistoryTurn(user_text=USER, assistant_spoken="A.")


# --- criterion 6: R16 ----------------------------------------------------------------------------


# T-0178 B2 re-scope of R16: the report step's text is withheld (was: voiced)
def test_r16_the_step_after_the_budget_cancels_its_call_and_withholds_its_text(clock):
    """T-0178 B2/B4 (re-scopes T-0112's "…voices_its_text"): the call past the budget is cancelled ``step_budget``
    as before, "Done." is never voiced, the ``out_of_steps`` clip plays and the turn completes with "A."."""
    m = loop(max_tool_steps=2)
    s, _ = at_step2(m, clock)
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen-s2", draft("valid", call_id="c2")))
    assert of(cmds, C.ToolExecute) == [execute("c2")]
    s, cmds = result(m, s, clock, "c2")
    assert of(cmds, need("LlmContinue")) == [llm_continue(3, "c2")]
    s, _ = say(m, s, clock, "Done.", "t1-gen-s3")
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen-s3", draft("valid", call_id="c3")))
    assert cancels(cmds) == [("c3", "step_budget")]
    assert not of(cmds, C.ToolExecute)
    assert sources(cmds) == "" and not of(cmds, C.TtsSpeak), cmds
    assert [c.kind for c in of(cmds, C.PlayClip)] == ["out_of_steps"], cmds
    assert s.state is TurnState.IDLE
    assert s.history[-1] == HistoryTurn(user_text=USER, assistant_spoken="A.")


def _call_on_every_step(m, clock):
    """A model that answers every step with a valid call and no text. Returns the generates and
    executes it caused and the final state."""
    s = to_thinking(m, clock)
    generates, executed = ["t1-gen"], []
    for _ in range(40):
        if s.state is TurnState.IDLE:
            break
        if s.awaiting_result is not None:
            s, cmds = result(m, s, clock, s.awaiting_result)
        else:
            assert s.active_generate is not None, f"stuck in {s.state.value}: {s}"
            call = f"c{len(generates)}"
            s, cmds = step(
                m, s, done(clock.advance(10), s.active_generate, draft("valid", call_id=call))
            )
        generates += [c.request_id for c in of(cmds, need("LlmContinue"))]
        executed += [c.call_id for c in of(cmds, C.ToolExecute)]
    return generates, executed, s


def test_r16_adversarial_a_model_calling_every_step_stops_after_three_generates(clock):
    generates, executed, s = _call_on_every_step(loop(max_tool_steps=2), clock)
    assert generates == ["t1-gen", "t1-gen-s2", "t1-gen-s3"]
    assert executed == ["c1", "c2"]
    assert s.state is TurnState.IDLE


def test_r16_at_the_default_budget_eight_executes_then_one_report_step(clock):
    m = loop()
    generates, executed, s = _call_on_every_step(m, clock)
    assert generates == ["t1-gen", *(f"t1-gen-s{k}" for k in range(2, 10))]
    assert executed == [f"c{k}" for k in range(1, 9)]
    assert s.state is TurnState.IDLE


# --- criterion 7: R17 ----------------------------------------------------------------------------


def _new_words_then_takeover(m, s, clock):
    """The user says "no stop" while the loop thinks, then takes over."""
    first = clock.advance(200)
    for i, w in enumerate(["no", "stop"]):
        start = first + i * 250
        s, _ = step(m, s, E.SttWord(t_ms=start + 400, text=w, start_ms=start, end_ms=start + 200))
    clock.advance(700)
    return step(m, s, E.UserTakeover(t_ms=clock.now_ms, onset_ms=first))


def _to_next_thinking(m, s, clock):
    clock.advance(600)
    t_eot = clock.now_ms - 50
    s, _ = step(m, s, E.SttEndOfTurn(t_ms=clock.now_ms, t_eot_ms=t_eot, p_end=1.0))
    s, _ = step(m, s, E.SttFlushed(t_ms=clock.advance(100), t_eot_ms=t_eot))
    return step(m, s, E.Tick(t_ms=clock.advance(200)))


def test_r17_a_takeover_after_an_execute_opens_a_turn_with_only_the_new_words(clock):
    m = loop()
    s, _ = executed_step1(m, clock)
    s, cmds = _new_words_then_takeover(m, s, clock)
    assert (s.state, s.turn_id) == (TurnState.LISTENING, "t2")
    assert not of(cmds, C.LlmAbort) and not of(cmds, C.ToolCancel) and not of(cmds, C.ToolExecute)
    assert C.PrefillOpen(request_id="t2-prefill") in cmds
    assert s.history[-1] == HistoryTurn(
        user_text=USER, assistant_spoken="A. [interrupted]", interrupted=True
    )
    s, cmds = _to_next_thinking(m, s, clock)
    starts = of(cmds, C.LlmStart)
    assert starts == [C.LlmStart(request_id="t2-gen", transcript="no stop")], starts


def test_r17_adversarial_during_step_2_aborts_and_drops_the_first_request(clock):
    m = loop()
    s, _ = at_step2(m, clock)
    s, cmds = _new_words_then_takeover(m, s, clock)
    assert C.LlmAbort(request_id="t1-gen-s2") in cmds
    assert s.history[-1] == HistoryTurn(
        user_text=USER, assistant_spoken="A. [interrupted]", interrupted=True
    )
    s, cmds = _to_next_thinking(m, s, clock)
    (start,) = of(cmds, C.LlmStart)
    assert "open" not in start.transcript.split() and start.transcript == "no stop"


def test_r17_guard_width_a_takeover_before_any_execute_keeps_the_words_as_today(clock):
    m = loop()
    s = to_thinking(m, clock)
    s, _ = _new_words_then_takeover(m, s, clock)
    s, cmds = _to_next_thinking(m, s, clock)
    assert of(cmds, C.LlmStart) == [C.LlmStart(request_id="t2-gen", transcript=f"{USER} no stop")]


# --- criterion 8: history across playback runs ---------------------------------------------------


def test_history_joins_two_playback_runs(clock):
    m = loop()
    s, _ = at_step2(m, clock, prefix="A.")
    s, _ = say(m, s, clock, "B.", "t1-gen-s2")
    s, _ = step(m, s, done(clock.advance(10), "t1-gen-s2"))
    s, _ = drain(m, s, clock, "B.")
    assert s.history[-1] == HistoryTurn(user_text=USER, assistant_spoken="A. B.")


def test_history_adversarial_the_second_run_does_not_erase_the_first_when_cut(clock):
    m = loop()
    s, _ = at_step2(m, clock, prefix="A.")
    s, _ = say(m, s, clock, "B.", "t1-gen-s2")
    s, _ = step(m, s, done(clock.advance(10), "t1-gen-s2"))
    onset = clock.advance(100)
    s, _ = step(m, s, E.UserSpeechOnset(t_ms=onset, onset_ms=onset))
    s, _ = step(m, s, E.UserTakeover(t_ms=clock.advance(700), onset_ms=onset))
    assert s.state is TurnState.INTERRUPTED
    s, _ = step(m, s, E.PlaybackStopped(t_ms=clock.advance(20), played_ms=0, played_text=""))
    assert s.history[-1] == HistoryTurn(
        user_text=USER, assistant_spoken="A. [interrupted]", interrupted=True
    )


def test_history_withheld_step_text_is_never_stored(clock):
    m = loop()
    s, _ = at_step2(m, clock, prefix="A.")
    s, _ = say(m, s, clock, "W.", "t1-gen-s2")
    s, _ = step(m, s, done(clock.advance(10), "t1-gen-s2", draft("valid", call_id="c2")))
    s, cmds = result(m, s, clock, "c2")
    assert of(cmds, need("LlmContinue")) == [llm_continue(3, "c2")]
    s, _ = say(m, s, clock, "B.", "t1-gen-s3")
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen-s3"))
    assert sources(cmds) == "B.", "the withheld W. is never voiced"
    s, _ = drain(m, s, clock, "B.")
    assert s.history[-1] == HistoryTurn(user_text=USER, assistant_spoken="A. B.")


#: T-0111a Goal and §2 History: a turn's stored text is every played run joined. In each failure
#: or degrade row below step 1 played "A." (drained) and nothing after it was reported played, so
#: the stored entry is the user's words with "A.", never user-only (which would drop what the
#: user heard).
PLAYED_STEP_1 = HistoryTurn(user_text=USER, assistant_spoken="A.")


def test_history_tts_error_during_the_end_report_keeps_the_first_run(clock):
    m = loop()
    s, _ = at_step2(m, clock, prefix="A.")
    s, _ = say(m, s, clock, "B.", "t1-gen-s2")
    s, _ = step(m, s, done(clock.advance(10), "t1-gen-s2"))
    assert s.state is TurnState.SPEAKING
    s, cmds = step(m, s, E.TtsError(t_ms=clock.advance(30), message="tts died"))
    assert s.state is TurnState.IDLE and C.Degrade(mode="text_only") in cmds
    assert of(cmds, C.CommitHistory) == [C.CommitHistory(user_only=False)], cmds
    assert s.history[-1] == PLAYED_STEP_1


def test_history_disconnect_while_awaiting_the_result_keeps_the_first_run(clock):
    m = loop()
    s, _ = executed_step1(m, clock, prefix="A.")
    assert s.awaiting_result == "c1"
    s, cmds = step(m, s, E.ClientDisconnected(t_ms=clock.advance(30)))
    assert s.state is TurnState.IDLE and C.Degrade(mode="text_only") in cmds
    assert of(cmds, C.CommitHistory) == [C.CommitHistory(user_only=False)], cmds
    assert s.history[-1] == PLAYED_STEP_1


def test_history_an_accepted_step_2_llm_error_keeps_the_first_run(clock):
    m = loop()
    s, _ = at_step2(m, clock, prefix="A.")
    s, cmds = step(m, s, E.LlmError(t_ms=clock.advance(30), request_id="t1-gen-s2", message="x"))
    assert s.state is TurnState.IDLE and C.HudError(message="x") in cmds
    assert C.LlmAbort(request_id="t1-gen-s2") in cmds
    assert of(cmds, C.CommitHistory) == [C.CommitHistory(user_only=False)], cmds
    assert s.history[-1] == PLAYED_STEP_1


def test_history_the_step_2_llm_timeout_keeps_the_first_run(clock):
    m = loop()
    s, _ = at_step2(m, clock, prefix="A.")
    deadline = clock.now_ms + LLM_TIMEOUT_MS  # R12 armed llm_timeout at the result's t_ms
    early, cmds = step(m, s, E.Tick(t_ms=deadline - 1))
    assert (early, cmds) == (s, [])
    s, cmds = step(m, s, E.Tick(t_ms=deadline))
    assert s.state is TurnState.IDLE and C.HudError(message="llm timeout") in cmds
    assert of(cmds, C.CommitHistory) == [C.CommitHistory(user_only=False)], cmds
    assert s.history[-1] == PLAYED_STEP_1


# --- criterion 9: R18 and P5' --------------------------------------------------------------------


def test_r18_the_tool_result_deadline_fails_the_turn_with_played_history(clock):
    m = loop()
    s, _ = executed_step1(m, clock)
    deadline = clock.now_ms + RESULT_MS
    early, cmds = step(m, s, E.Tick(t_ms=deadline - 1))
    assert (early, cmds) == (s, []), "guard width: one ms early does nothing"
    s, cmds = step(m, s, E.Tick(t_ms=deadline))
    assert [c for c in cmds if not isinstance(c, C.CancelTimer)] == [
        C.HudError(message="tool timeout"),
        C.CommitHistory(user_only=False),
        C.HudState(state="IDLE", turn_id="t1"),
    ], cmds
    assert s.history[-1] == HistoryTurn(user_text=USER, assistant_spoken="A.")
    loop_fields = (s.tool_step, s.tool_steps_executed, s.awaiting_result, s.step_text)
    assert loop_fields == (1, 0, None, "") and s.played_prior == "", "per-turn reset"


# T-0178 B3 re-scope: the repairable step's text is withheld (was: voiced)
def test_p5_prime_a_repairable_step_2_draft_never_starts_a_repair(clock):
    """P5' as before (no repair at step >= 2, cancelled ``refused``); T-0178 B3/B4a re-scope: a repairable draft is a
    malformed one, so its text is withheld and the ``not_finished`` clip plays."""
    m = loop()
    s, _ = at_step2(m, clock)
    s, _ = say(m, s, clock, "Which one?", "t1-gen-s2")
    s, cmds = step(m, s, done(clock.advance(10), "t1-gen-s2", draft("repairable", call_id="c2")))
    assert not of(cmds, C.ToolRepairStart), cmds
    assert s.active_generate is None and s.tool_repairs == 0
    assert cancels(cmds) == [("c2", "refused")]
    assert sources(cmds) == "" and not of(cmds, C.TtsSpeak), cmds
    assert [c.kind for c in of(cmds, C.PlayClip)] == ["not_finished"], cmds
