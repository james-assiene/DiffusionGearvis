"""T-0089 criterion 2: the tool-turn rows R1-R10 of the pure turn machine, on the virtual clock.

Expectations come from ``docs/tasks/T-0089.md`` (rows R1-R10, implementation notes) and the
umbrella's H5 (``docs/tasks/T-0080d.md``: execute only after the spoken prefix has fully played and
only if the turn was not interrupted). Nothing is captured from an implementation.

Conventions, as in ``test_turn_invariants.py``: rows assert the commands the criterion requires
(and their relative order where the criterion gives one), never an exact command list. Every time
stamp is read off the ``clock`` fixture (``tests/conftest.py::VirtualClock``), so the ``tool_repair``
deadline is computed here by hand as ``LlmDone.t_ms + 3000`` (H12; the implementation note says the
deadline comes from that event's ``t_ms`` only).

Guard width (``.claude/rules/tests.md``): the execute deferral has rows on both sides — rows where it
must *not* fire (at ``llm.done`` while audio is queued, on ``playback.stopped``, on a barge-in, on an
error, a ``tts.error``, a disconnect, a repair timeout) and rows where a deferral that is too wide
would wrongly withhold it (a drain without ``tts.done``, a pause that resumed, drain-before-done).
"""

from __future__ import annotations

import pytest

from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts import turn as T
from local_astra.contracts.turn import TurnState
from tests.unit.test_turn_invariants import state_without_t0089

POLICY = TurnPolicy(thinking=True)
REPAIR_MS = 3000  # H12 / criterion 1: TurnPolicy.tool_repair_timeout_ms
ARGS = {"x": 412, "y": 87}
_MISSING = object()


# --- names the task introduces, read so that their absence is a named failure -------------------


def need_cmd(name: str):
    cls = getattr(C, name, _MISSING)
    assert cls is not _MISSING, f"T-0089 criterion 1: contracts/commands.py gains {name}"
    return cls


def draft(
    status: str, *, call_id="c1", name="click_at", arguments=ARGS, reason=None, source="draft"
):
    cls = getattr(T, "ToolDraft", _MISSING)
    assert cls is not _MISSING, "T-0089 criterion 1: contracts/turn.py gains ToolDraft"
    if status == "refused":
        name, arguments, reason = None, None, reason or "nameless"
    if status == "repairable":
        arguments, reason = None, reason or "schema"
    return cls(
        call_id=call_id,
        status=status,
        name=name,
        arguments=arguments,
        reason=reason,
        source=source,
    )


def done(t_ms: int, request_id: str, tool=None):
    assert "tool" in E.LlmDone.model_fields, "T-0089 criterion 1: events.LlmDone gains tool"
    return E.LlmDone(t_ms=t_ms, request_id=request_id, tool=tool)


def of(cmds, name: str) -> list:
    cls = need_cmd(name)
    return [c for c in cmds if isinstance(c, cls)]


def kinds(cmds) -> list[str]:
    return [type(c).__name__ for c in cmds]


def pending_tool(state):
    assert "tool" in T.MachineState.model_fields, "T-0089 criterion 1: MachineState.tool"
    return state.tool


def assert_no_execute(cmds, why: str):
    assert not of(cmds, "ToolExecute"), f"{why}: got {kinds(cmds)}"


def assert_cancel(cmds, reason: str, call_id: str = "c1"):
    cancels = of(cmds, "ToolCancel")
    assert [(c.call_id, c.reason) for c in cancels] == [(call_id, reason)], (
        f"expected exactly ToolCancel({call_id}, {reason}); got {kinds(cmds)} / "
        f"{[(c.call_id, c.reason) for c in cancels]}"
    )


def assert_execute_then_history(cmds, *, turn_id="t1", source="draft", user_only: bool):
    """R1: ToolExecute, then the history command (and schedule_summary), hud.state last."""
    execs = of(cmds, "ToolExecute")
    assert len(execs) == 1, f"exactly one ToolExecute: {kinds(cmds)}"
    ex = execs[0]
    assert (ex.call_id, ex.turn_id, ex.name, ex.arguments, ex.source) == (
        "c1",
        turn_id,
        "click_at",
        ARGS,
        source,
    )
    i = cmds.index(ex)
    hist = [j for j, c in enumerate(cmds) if isinstance(c, C.CommitHistory)]
    assert len(hist) == 1 and cmds[hist[0]].user_only is user_only, kinds(cmds)
    assert i < hist[0], f"ToolExecute goes before CommitHistory: {kinds(cmds)}"
    for j, c in enumerate(cmds):
        if isinstance(c, C.ScheduleSummary):
            assert i < j, f"ToolExecute goes before ScheduleSummary: {kinds(cmds)}"
    assert isinstance(cmds[-1], C.HudState), f"hud.state stays last: {kinds(cmds)}"


# --- drivers -------------------------------------------------------------------------------------


def step(m, s, ev):
    s2, cmds = m.step(s, ev)
    return s2, list(cmds)


def to_thinking(m, clock):
    """IDLE -> THINKING for turn t1 with the transcript "open the menu"."""
    s = m.initial()
    onset = clock.advance(1000)
    s, _ = step(m, s, E.UserSpeechOnset(t_ms=onset, onset_ms=onset))
    for i, w in enumerate(["open", "the", "menu"]):
        start = onset + i * 250
        s, _ = step(m, s, E.SttWord(t_ms=start + 400, text=w, start_ms=start, end_ms=start + 200))
    clock.advance(900)
    t_eot = clock.now_ms - 50
    s, _ = step(m, s, E.SttEndOfTurn(t_ms=clock.now_ms, t_eot_ms=t_eot, p_end=1.0))
    s, _ = step(m, s, E.SttFlushed(t_ms=clock.advance(100), t_eot_ms=t_eot))
    s, _ = step(m, s, E.Tick(t_ms=clock.advance(POLICY.eot_settle_ms)))
    assert s.state is TurnState.THINKING and s.active_generate == "t1-gen"
    return s


def answer(s, clock, text, *, request_id=None, segment="answer"):
    return E.LlmCommit(
        t_ms=clock.advance(200),
        request_id=request_id or s.active_generate,
        canvas_idx=0,
        segment=segment,
        text=text,
    )


def to_speaking(m, clock, text="Opening the menu now."):
    s = to_thinking(m, clock)
    s, _ = step(m, s, answer(s, clock, text))
    assert s.state is TurnState.SPEAKING
    return s


def drained(clock, text="Opening the menu now."):
    return E.PlaybackDrained(t_ms=clock.advance(400), played_ms=1500, played_text=text)


def start_repair(m, s, clock):
    """R2 from whatever state ``s`` is in; returns (state, cmds, done_t_ms)."""
    t = clock.advance(50)
    s, cmds = step(m, s, done(t, "t1-gen", draft("repairable")))
    return s, cmds, t


@pytest.fixture
def m():
    return TurnMachine(POLICY)


def sentence_segmenter(text: str, *, final: bool):
    """Conserving segmenter: whole sentences end at ``.``, the tail is held until ``final``."""
    out, rest = [], text
    while (i := rest.find(".")) != -1:
        out.append((rest[: i + 1], rest[: i + 1].strip()))
        rest = rest[i + 1 :]
    if final and rest:
        out.append((rest, rest.strip()))
        rest = ""
    return out, rest


# --- R1: a valid pass-1 call ---------------------------------------------------------------------


def test_r1_thinking_with_nothing_committed_executes_then_commits_user_only(m, clock):
    s = to_thinking(m, clock)
    s, cmds = step(m, s, done(clock.advance(300), "t1-gen", draft("valid")))
    assert s.state is TurnState.IDLE
    assert_execute_then_history(cmds, user_only=True)
    assert pending_tool(s) is None and s.tool_repairs == 0
    assert s.history[-1].assistant_spoken == "", "H8: nothing extra is spoken or stored"


def test_r1_speaking_defers_until_drained(m, clock):
    s = to_speaking(m, clock)
    s, cmds = step(m, s, done(clock.advance(100), "t1-gen", draft("valid")))
    assert s.state is TurnState.SPEAKING and s.llm_done is True
    assert_no_execute(cmds, "R1/H5: SPEAKING defers execution to the drain")
    assert not of(cmds, "ToolCancel")
    assert pending_tool(s) is not None, "R1: the call becomes pending"
    s, cmds = step(m, s, E.TtsDone(t_ms=clock.advance(50), utterance_id=s.utterances[-1]))
    assert_no_execute(cmds, "tts.done is synthesis, not playback")
    s, cmds = step(m, s, drained(clock))
    assert s.state is TurnState.IDLE
    assert_execute_then_history(cmds, user_only=False)
    assert pending_tool(s) is None


def test_r1_guard_width_drain_without_tts_done_still_executes(m, clock):
    """Too wide: a deferral that also waited for tts.done (or anything but the drain) fails here."""
    s = to_speaking(m, clock)
    s, _ = step(m, s, done(clock.advance(100), "t1-gen", draft("valid")))
    s, cmds = step(m, s, drained(clock))
    assert s.state is TurnState.IDLE
    assert_execute_then_history(cmds, user_only=False)


def test_r1_guard_width_drain_before_done_executes_on_done(clock):
    """Amended by ruling T11: the reviewer's trace, with the sentence segmenter.

    SPEAKING with " Opening it" held, the drain lands, then the final ``llm.done`` carries a valid
    call. A5 makes that ``llm.done`` speak the held clause, so the drain the machine remembered no
    longer covers what was spoken. The tuple must not execute; ``playback_drained`` becomes false,
    and the turn completes, and executes, on the next ``playback.drained``. The original
    nothing-held version of this row (the too-wide side of the deferral) is kept verbatim as
    ``test_t11_guard_width_drain_before_done_with_nothing_held_executes_on_done``.
    """
    m = TurnMachine(POLICY, segment=sentence_segmenter)
    s = to_thinking(m, clock)
    s, cmds = step(m, s, answer(s, clock, "Sure. Opening it"))
    assert [c.source for c in cmds if isinstance(c, C.TtsSpeak)] == ["Sure."]
    s, cmds = step(m, s, drained(clock, "Sure."))
    assert_no_execute(cmds, "no call is known yet")
    assert s.state is TurnState.SPEAKING, "llm.done has not arrived"
    s, cmds = step(m, s, done(clock.advance(100), "t1-gen", draft("valid")))
    assert [c.source for c in cmds if isinstance(c, C.TtsSpeak)] == [" Opening it"], (
        "A5: the final llm.done still flushes the held clause"
    )
    assert_no_execute(cmds, "T11: never execute in a tuple that speaks")
    assert s.state is TurnState.SPEAKING, f"T11: the turn does not complete: {kinds(cmds)}"
    assert s.playback_drained is False, "T11: the new utterance has not drained"
    assert s.llm_done is True
    assert pending_tool(s) is not None and pending_tool(s).call_id == "c1", "still pending"
    assert not [c for c in cmds if isinstance(c, C.CommitHistory | C.ScheduleSummary)]
    assert not of(cmds, "ToolCancel"), "T11 defers the call, it does not cancel it"
    s, cmds = step(m, s, drained(clock, "Sure. Opening it"))
    assert s.state is TurnState.IDLE, "T11: the next drain completes the turn"
    assert_execute_then_history(cmds, user_only=False)
    assert s.history[-1].assistant_spoken == "Sure. Opening it"


def test_t11_guard_width_drain_before_done_with_nothing_held_executes_on_done(m, clock):
    """Too wide (ruling T11's other side): with nothing held, the llm.done after the drain speaks
    nothing, so it completes and executes at once -- this is the pre-T11 row, unchanged."""
    s = to_speaking(m, clock)
    s, cmds = step(m, s, drained(clock))
    assert_no_execute(cmds, "no call is known yet")
    s, cmds = step(m, s, done(clock.advance(100), "t1-gen", draft("valid")))
    assert not [c for c in cmds if isinstance(c, C.TtsSpeak)], "nothing was held"
    assert s.state is TurnState.IDLE
    assert_execute_then_history(cmds, user_only=False)


def test_t11_toolless_twin_completes_in_the_flushing_tuple_as_on_main(clock):
    """T11 changes nothing without a tool: the reviewer's trace with a toolless ``llm.done``
    speaks the held clause and completes in the same tuple, exactly as main's machine does."""
    from tests.unit import _frozen_turn_machine_main as frozen_main

    m = TurnMachine(POLICY, segment=sentence_segmenter)
    main = frozen_main.TurnMachine(
        frozen_main.TurnPolicy(thinking=True), segment=sentence_segmenter
    )
    s = to_thinking(m, clock)
    s, _ = step(m, s, answer(s, clock, "Sure. Opening it"))
    s, _ = step(m, s, drained(clock, "Sure."))
    ev = done(clock.advance(100), "t1-gen")
    after, cmds = step(m, s, ev)
    assert [c.source for c in cmds if isinstance(c, C.TtsSpeak)] == [" Opening it"]
    assert after.state is TurnState.IDLE, "main's order: the toolless turn completes here"
    assert C.CommitHistory(user_only=False) in cmds
    assert [c for c in cmds if isinstance(c, C.ScheduleSummary)]
    assert after.history[-1].assistant_spoken == "Sure.", "only what was reported played"
    main_after, main_cmds = main.step(s, ev)
    assert cmds == list(main_cmds), "P9: toolless rows are main's"
    assert state_without_t0089(after) == state_without_t0089(main_after), "P9 (T15)"


def test_p3_a_drain_before_any_speech_does_not_cover_the_speech_after_it(m, clock):
    """P3/H5, found by ``ToolTurnModel`` at ``--hypothesis-seed=3`` (full gate) once T10/T11 moved
    its draws: a ``playback.drained`` in THINKING (nothing queued yet) leaves the machine's
    drained flag set, the answer is then spoken, and the valid ``llm.done`` executes while that
    answer is unplayed. The executing tuple holds no ``TtsSpeak``, so T11's tuple rule alone does
    not catch it; P3 ("never while that turn has queued or unplayed utterances") does."""
    s = to_thinking(m, clock)
    s, _ = step(m, s, E.PlaybackDrained(t_ms=clock.advance(100), played_ms=0, played_text=""))
    assert s.state is TurnState.THINKING
    s, cmds = step(m, s, answer(s, clock, "Opening the menu now."))
    assert [c.source for c in cmds if isinstance(c, C.TtsSpeak)] == ["Opening the menu now."]
    s, cmds = step(m, s, done(clock.advance(100), "t1-gen", draft("valid")))
    assert_no_execute(cmds, "P3/H5: the answer queued after that drain has not played")
    assert not of(cmds, "ToolCancel"), "the call is deferred, not cancelled"
    s, cmds = step(m, s, drained(clock))
    assert s.state is TurnState.IDLE
    assert_execute_then_history(cmds, user_only=False)


def _turn_2_thinking_after_a_stale_drain(m, clock, text):
    """Turn 1 speaks ``text`` and ends on ``tts.error`` (no flush); turn 2 reaches THINKING;
    turn 1's drain then arrives carrying ``text`` (see the stale-drain row for the session path)."""
    s = to_speaking(m, clock, text)
    s, _ = step(m, s, done(clock.advance(100), "t1-gen"))
    s, cmds = step(m, s, E.TtsError(t_ms=clock.advance(50), message="tts died"))
    assert s.state is TurnState.IDLE
    assert not [c for c in cmds if isinstance(c, C.TtsCancel | C.PlaybackFlush)], (
        "premise: tts.error leaves the audio already queued playing"
    )
    onset = clock.advance(200)
    s, _ = step(m, s, E.UserSpeechOnset(t_ms=onset, onset_ms=onset))
    s, _ = step(m, s, E.SttWord(t_ms=onset + 400, text="again", start_ms=onset, end_ms=onset + 200))
    clock.advance(900)
    t_eot = clock.now_ms - 50
    s, _ = step(m, s, E.SttEndOfTurn(t_ms=clock.now_ms, t_eot_ms=t_eot, p_end=1.0))
    s, _ = step(m, s, E.SttFlushed(t_ms=clock.advance(100), t_eot_ms=t_eot))
    s, _ = step(m, s, E.Tick(t_ms=clock.advance(POLICY.eot_settle_ms)))
    assert s.state is TurnState.THINKING and s.active_generate == "t2-gen"
    s, _ = step(m, s, E.PlaybackDrained(t_ms=clock.advance(100), played_ms=1500, played_text=text))
    return s


def test_p3_a_stale_drain_whose_text_matches_the_next_answer_does_not_cover_it(m, clock):
    """P3/H5 against T14's text comparison: a drain from an *earlier* turn's audio lands in the
    next turn's THINKING, and its ``played_text`` equals what that turn then speaks.

    Session path (``session.py`` / ``audio_io.py`` at c236d19), traced by reading, not run:
    turn 1 is SPEAKING with ``llm.done`` seen; the last utterance's synthesis fails, so
    ``_speak_one`` submits ``tts.error`` and its ``finally`` calls ``_maybe_mark_drain`` while
    the session's state is still SPEAKING, which marks the run's drain. The machine's
    ``tts.error`` row ends turn 1 without ``tts.cancel``/``playback.flush`` (asserted below), so
    the earlier utterance keeps rendering, and its ``playback.drained`` (``played_text`` = that
    run's sources) is reported whenever the last sample renders -- after the user, speaking
    over audio in IDLE, has opened turn 2. If turn 2's answer is the same sentence (a repeated
    command), T14's ``played_text == spoken_text`` holds by coincidence.
    """
    text = "Opening the menu now."
    s = _turn_2_thinking_after_a_stale_drain(m, clock, text)
    s, cmds = step(m, s, answer(s, clock, text))
    assert [c.source for c in cmds if isinstance(c, C.TtsSpeak)] == [text]
    s, cmds = step(m, s, done(clock.advance(100), "t2-gen", draft("valid")))
    assert_no_execute(cmds, "P3/H5: turn 2's answer was queued after that drain and has not played")
    assert not of(cmds, "ToolCancel"), "the call is deferred, not cancelled"
    s, cmds = step(m, s, drained(clock, text))
    assert s.state is TurnState.IDLE
    assert_execute_then_history(cmds, turn_id="t2", user_only=False)


def test_t15_guard_width_a_genuine_drain_after_a_stale_one_still_executes_on_done(m, clock):
    """Too wide (T15): the flag a stale drain leaves must not poison the turn's genuine drain.
    After the stale drain, turn 2 speaks, its own drain reports all of it, and the valid
    ``llm.done`` (nothing held) executes at once."""
    text = "Opening the menu now."
    s = _turn_2_thinking_after_a_stale_drain(m, clock, text)
    s, cmds = step(m, s, answer(s, clock, text))
    assert [c.source for c in cmds if isinstance(c, C.TtsSpeak)] == [text]
    s, _ = step(m, s, drained(clock, text))
    assert s.state is TurnState.SPEAKING
    s, cmds = step(m, s, done(clock.advance(100), "t2-gen", draft("valid")))
    assert not [c for c in cmds if isinstance(c, C.TtsSpeak)], "nothing was held"
    assert s.state is TurnState.IDLE
    assert_execute_then_history(cmds, turn_id="t2", user_only=False)


def test_t15_spoken_after_drain_is_a_per_turn_field_defaulting_to_false():
    """T15 mechanism: ``MachineState.spoken_after_drain: bool = False`` in ``_PER_TURN_RESET``,
    so a turn never inherits the previous turn's "speech followed the drain"."""
    from local_astra.conductor import turn_machine as tm

    field = T.MachineState.model_fields.get("spoken_after_drain")
    assert field is not None, "T15: MachineState gains spoken_after_drain"
    assert field.annotation is bool and field.default is False
    assert tm._PER_TURN_RESET.get("spoken_after_drain", _MISSING) is False, (
        "T15: spoken_after_drain is reset to False with every per-turn field"
    )


def _drained_then_paused_with_a_held_commit(m, clock):
    """SPEAKING, drained, then paused, and a commit arrives while paused (criterion 16 holds it)."""
    s = to_speaking(m, clock)
    s, _ = step(m, s, drained(clock))
    t = clock.advance(100)
    s, _ = step(m, s, E.UserSpeechOnset(t_ms=t, onset_ms=t))
    assert s.state is TurnState.PAUSED
    s, cmds = step(m, s, answer(s, clock, " Then the file menu."))
    assert not [c for c in cmds if isinstance(c, C.TtsSpeak)], "criterion 16"
    return s


def test_t11_resume_that_speaks_does_not_execute_until_the_next_drain(m, clock):
    """T11's rule is about the tuple, not the event: a resume whose flush speaks is a completion
    with a TtsSpeak in it, so it does not complete, and the call runs on the next drain."""
    s = _drained_then_paused_with_a_held_commit(m, clock)
    s, cmds = step(m, s, done(clock.advance(50), "t1-gen", draft("valid")))
    assert s.state is TurnState.PAUSED and s.llm_done is True
    assert_no_execute(cmds, "T4: still paused")
    s, cmds = step(m, s, E.UserBackchannel(t_ms=clock.advance(100), duration_ms=200))
    assert [c.source for c in cmds if isinstance(c, C.TtsSpeak)] == [" Then the file menu."]
    assert_no_execute(cmds, "T11: the resume spoke, so it cannot execute")
    assert s.state is TurnState.SPEAKING and s.playback_drained is False
    assert pending_tool(s) is not None
    s, cmds = step(m, s, drained(clock, "Opening the menu now. Then the file menu."))
    assert s.state is TurnState.IDLE
    assert_execute_then_history(cmds, user_only=False)


def test_t11_toolless_twin_of_the_resume_completes_as_on_main(m, clock):
    from tests.unit import _frozen_turn_machine_main as frozen_main

    main = frozen_main.TurnMachine(frozen_main.TurnPolicy(thinking=True))
    s = _drained_then_paused_with_a_held_commit(m, clock)
    s, _ = step(m, s, done(clock.advance(50), "t1-gen"))
    ev = E.UserBackchannel(t_ms=clock.advance(100), duration_ms=200)
    after, cmds = step(m, s, ev)
    assert [c.source for c in cmds if isinstance(c, C.TtsSpeak)] == [" Then the file menu."]
    assert after.state is TurnState.IDLE, "main's order: the toolless resume completes here"
    main_after, main_cmds = main.step(s, ev)
    assert cmds == list(main_cmds), "P9: toolless rows are main's"
    assert state_without_t0089(after) == state_without_t0089(main_after), "P9 (T15)"


def test_r1_guard_width_a_pause_that_resumed_is_not_an_interruption(m, clock):
    """H5 says "not interrupted"; a backchannel pause that resumed is not one (too-wide guard)."""
    s = to_speaking(m, clock)
    s, _ = step(m, s, done(clock.advance(100), "t1-gen", draft("valid")))
    t = clock.advance(100)
    s, cmds = step(m, s, E.UserSpeechOnset(t_ms=t, onset_ms=t))
    assert s.state is TurnState.PAUSED
    assert not of(cmds, "ToolCancel"), "a pause is not yet an interruption"
    s, cmds = step(m, s, E.UserBackchannel(t_ms=clock.advance(200), duration_ms=200))
    assert s.state is TurnState.SPEAKING
    assert not of(cmds, "ToolCancel")
    s, cmds = step(m, s, drained(clock))
    assert s.state is TurnState.IDLE
    assert_execute_then_history(cmds, user_only=False)


def test_r1_guard_absent_playback_stopped_in_speaking_never_executes(m, clock):
    """Mutant named by the task: executing on PlaybackStopped instead of the drain."""
    s = to_speaking(m, clock)
    s, _ = step(m, s, done(clock.advance(100), "t1-gen", draft("valid")))
    s, cmds = step(
        m,
        s,
        E.PlaybackStopped(t_ms=clock.advance(100), played_ms=300, played_text="Opening"),
    )
    assert_no_execute(cmds, "H5: playback.stopped is not a drain")


def test_r1_tool_commits_and_the_call_are_never_voiced(m, clock):
    s = to_speaking(m, clock)
    tool_commit = E.LlmCommit.model_construct(
        t_ms=clock.advance(100),
        request_id="t1-gen",
        canvas_idx=1,
        segment="tool",
        text="call:click_at{x:412,y:87}",
        thought_open=False,
        type="llm.commit",
    )
    s, cmds = step(m, s, tool_commit)
    assert not of(cmds, "ToolExecute")
    assert not [c for c in cmds if isinstance(c, C.TtsSpeak)], "a tool commit is never voiced"
    s, cmds = step(m, s, done(clock.advance(100), "t1-gen", draft("valid")))
    assert all("click_at" not in c.source for c in cmds if isinstance(c, C.TtsSpeak))


# --- R2 / R3: one repair ---------------------------------------------------------------------------


def test_r2_thinking_repairable_starts_exactly_one_repair(m, clock):
    s = to_thinking(m, clock)
    s, cmds, t = start_repair(m, s, clock)
    assert s.state is TurnState.THINKING
    assert s.llm_done is False, "R2: llm_done stays false"
    assert s.active_generate == "t1-repair"
    starts = of(cmds, "ToolRepairStart")
    assert [(r.request_id, r.draft_request_id, r.call_id) for r in starts] == [
        ("t1-repair", "t1-gen", "c1")
    ]
    arms = [c for c in cmds if isinstance(c, C.ArmTimer) and c.name == "tool_repair"]
    assert [a.at_ms for a in arms] == [t + REPAIR_MS], "deadline = LlmDone.t_ms + 3000"
    assert C.CancelTimer(name="llm_timeout") in cmds, "R2 cancels llm_timeout"
    assert not of(cmds, "ToolExecute") and not of(cmds, "ToolCancel")
    assert not [c for c in cmds if isinstance(c, C.LlmStart | C.CommitHistory)]
    assert s.tool_repairs == 1 and pending_tool(s) is not None


def test_r2_speaking_repairable_flushes_the_held_prefix_and_keeps_speaking(clock):
    m = TurnMachine(POLICY, segment=sentence_segmenter)
    s = to_thinking(m, clock)
    s, cmds = step(m, s, answer(s, clock, "Sure. Opening it"))
    assert [c.source for c in cmds if isinstance(c, C.TtsSpeak)] == ["Sure."]
    s, cmds, _ = start_repair(m, s, clock)
    assert s.state is TurnState.SPEAKING and s.active_generate == "t1-repair"
    assert [c.source for c in cmds if isinstance(c, C.TtsSpeak)] == [" Opening it"], (
        "pass 1 is done, so its held clause is spoken now (A5: llm.done flushes the gate)"
    )
    assert len(of(cmds, "ToolRepairStart")) == 1


def test_r3_valid_repair_in_thinking_executes_with_source_repair(m, clock):
    s = to_thinking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, cmds = step(m, s, done(clock.advance(200), "t1-repair", draft("valid", source="repair")))
    assert s.state is TurnState.IDLE
    assert_execute_then_history(cmds, source="repair", user_only=True)
    assert not [t for t in s.timers if t.name == "tool_repair"]


def test_r3_valid_repair_in_speaking_waits_for_the_drain(m, clock):
    s = to_speaking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, cmds = step(m, s, done(clock.advance(200), "t1-repair", draft("valid", source="repair")))
    assert s.state is TurnState.SPEAKING and s.llm_done is True
    assert_no_execute(cmds, "H5: deferred to the drain")
    s, cmds = step(m, s, drained(clock))
    assert_execute_then_history(cmds, source="repair", user_only=False)


# --- R4: a repair that did not produce a valid call -------------------------------------------------


@pytest.mark.parametrize(
    "tool",
    [
        lambda: draft("refused", source="repair"),
        lambda: draft("repairable"),
        lambda: None,
    ],
    ids=["refused", "repairable-again", "no-tool"],
)
def test_r4_failed_repair_cancels_and_never_repairs_twice(m, clock, tool):
    s = to_speaking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, cmds = step(m, s, done(clock.advance(200), "t1-repair", tool()))
    assert_cancel(cmds, "repair_failed")
    assert not of(cmds, "ToolRepairStart"), "R4: never a second repair"
    assert s.llm_done is True and s.active_generate is None
    s, cmds = step(m, s, drained(clock))
    assert s.state is TurnState.IDLE
    assert_no_execute(cmds, "R4: the call was cancelled")
    assert not of(cmds, "ToolCancel"), "P8: exactly one terminal per call"


def test_r4_failed_repair_in_thinking_completes_as_today(m, clock):
    s = to_thinking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, cmds = step(m, s, done(clock.advance(200), "t1-repair", None))
    assert_cancel(cmds, "repair_failed")
    assert s.state is TurnState.IDLE
    assert C.CommitHistory(user_only=True) in cmds


# --- R5: refused pass-1 ---------------------------------------------------------------------------


def test_r5_refused_in_thinking_cancels_and_idles(m, clock):
    s = to_thinking(m, clock)
    s, cmds = step(m, s, done(clock.advance(200), "t1-gen", draft("refused")))
    assert_cancel(cmds, "refused")
    assert_no_execute(cmds, "refused")
    assert not of(cmds, "ToolRepairStart")
    assert s.state is TurnState.IDLE and C.CommitHistory(user_only=True) in cmds


def test_r5_refused_in_speaking_cancels_once_and_a_later_barge_in_owes_nothing(m, clock):
    s = to_speaking(m, clock)
    s, cmds = step(m, s, done(clock.advance(100), "t1-gen", draft("refused")))
    assert_cancel(cmds, "refused")
    assert s.state is TurnState.SPEAKING and s.llm_done is True
    t = clock.advance(100)
    s, _ = step(m, s, E.UserSpeechOnset(t_ms=t, onset_ms=t))
    s, cmds = step(m, s, E.UserTakeover(t_ms=clock.advance(700), onset_ms=t))
    assert s.state is TurnState.INTERRUPTED
    assert not of(cmds, "ToolCancel"), "nothing is pending any more (cancel only when pending)"


# --- R6: the repair deadline --------------------------------------------------------------------


def test_r6_repair_deadline_guard_width_one_ms_early_does_nothing(m, clock):
    s = to_speaking(m, clock)
    s, _, t = start_repair(m, s, clock)
    s2, cmds = step(m, s, E.Tick(t_ms=t + REPAIR_MS - 1))
    assert not of(cmds, "ToolCancel") and s2.active_generate == "t1-repair"


def test_r6_repair_deadline_in_speaking_aborts_cancels_and_completes_on_drain(m, clock):
    s = to_speaking(m, clock)
    s, _, t = start_repair(m, s, clock)
    s, cmds = step(m, s, E.Tick(t_ms=t + REPAIR_MS))
    assert C.LlmAbort(request_id="t1-repair") in cmds
    assert_cancel(cmds, "repair_timeout")
    assert s.llm_done is True and s.active_generate is None
    s, cmds = step(m, s, drained(clock))
    assert s.state is TurnState.IDLE
    assert_no_execute(cmds, "P2: the turn hit tool_repair")


def test_r6_repair_deadline_in_thinking_idles(m, clock):
    s = to_thinking(m, clock)
    s, _, t = start_repair(m, s, clock)
    s, cmds = step(m, s, E.Tick(t_ms=t + REPAIR_MS + 5))
    assert C.LlmAbort(request_id="t1-repair") in cmds
    assert_cancel(cmds, "repair_timeout")
    assert s.state is TurnState.IDLE and C.CommitHistory(user_only=True) in cmds


# --- R7: the repair request fails -------------------------------------------------------------------


def test_r7_repair_error_in_speaking_leaves_speech_running(m, clock):
    s = to_speaking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, cmds = step(m, s, E.LlmError(t_ms=clock.advance(100), request_id="t1-repair", message="x"))
    assert_cancel(cmds, "repair_failed")
    assert C.LlmAbort(request_id="t1-repair") in cmds, "T8: the repair is aborted"
    assert s.state is TurnState.SPEAKING, "R7: the turn is not failed"
    assert not [c for c in cmds if isinstance(c, C.TtsCancel | C.PlaybackFlush)]
    assert not [c for c in cmds if isinstance(c, C.CommitHistory)]
    assert s.llm_done is True
    s, cmds = step(m, s, drained(clock))
    assert s.state is TurnState.IDLE
    assert_no_execute(cmds, "R7")
    assert C.CommitHistory(user_only=False) in cmds
    assert s.history[-1].assistant_spoken == "Opening the menu now."


def test_t9_r7_after_the_drain_completes_the_turn_at_once(m, clock):
    """Ruling T9: the spoken prefix had drained before the repair failed, so R7's "complete as
    today" ends the turn in the same tuple, with no ToolExecute."""
    s = to_speaking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, _ = step(m, s, drained(clock))
    assert s.state is TurnState.SPEAKING, "llm_done is still false while the repair runs"
    s, cmds = step(m, s, E.LlmError(t_ms=clock.advance(100), request_id="t1-repair", message="x"))
    assert s.state is TurnState.IDLE, f"T9: completes at once: {kinds(cmds)}"
    assert_cancel(cmds, "repair_failed")
    assert C.LlmAbort(request_id="t1-repair") in cmds
    assert_no_execute(cmds, "R7")
    assert C.CommitHistory(user_only=False) in cmds
    assert [c for c in cmds if isinstance(c, C.ScheduleSummary)]
    assert not [c for c in cmds if isinstance(c, C.HudError)], "R7: the turn is not failed"
    assert s.history[-1].assistant_spoken == "Opening the menu now."


def test_r7_repair_error_in_thinking_completes_as_today(m, clock):
    s = to_thinking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, cmds = step(m, s, E.LlmError(t_ms=clock.advance(100), request_id="t1-repair", message="x"))
    assert_cancel(cmds, "repair_failed")
    assert s.state is TurnState.IDLE and C.CommitHistory(user_only=True) in cmds


# --- R8: every interruption, error and disconnect cancels ------------------------------------------


def barge_in(m, s, clock):
    t = clock.advance(100)
    s, c1 = step(m, s, E.UserSpeechOnset(t_ms=t, onset_ms=t))
    assert s.state is TurnState.PAUSED
    s, c2 = step(m, s, E.UserTakeover(t_ms=clock.advance(700), onset_ms=t))
    assert s.state is TurnState.INTERRUPTED
    return s, c1, c2


def test_r8_adversarial_barge_in_after_valid_done_before_drain_cancels(m, clock):
    s = to_speaking(m, clock)
    s, _ = step(m, s, done(clock.advance(100), "t1-gen", draft("valid")))
    s, c1, c2 = barge_in(m, s, clock)
    assert_no_execute(c1 + c2, "H5: interrupted")
    assert_cancel(c2, "interrupted")
    assert pending_tool(s) is None
    s, c3 = step(
        m, s, E.PlaybackStopped(t_ms=clock.advance(50), played_ms=400, played_text="Opening")
    )
    assert s.state is TurnState.LISTENING
    s, c4 = step(m, s, drained(clock, "Opening"))
    assert_no_execute(c3 + c4, "an interrupted turn never executes later")
    assert not of(c3 + c4, "ToolCancel"), "P8: exactly one terminal"


def test_r8_long_offset_while_paused_cancels_as_interrupted(m, clock):
    s = to_speaking(m, clock)
    s, _ = step(m, s, done(clock.advance(100), "t1-gen", draft("valid")))
    t = clock.advance(100)
    s, _ = step(m, s, E.UserSpeechOnset(t_ms=t, onset_ms=t))
    t2 = clock.advance(650)
    s, cmds = step(m, s, E.UserSpeechOffset(t_ms=t2, offset_ms=t2, speech_ms=650))
    assert s.state is TurnState.INTERRUPTED
    assert_cancel(cmds, "interrupted")


def test_r8_barge_in_during_repair_aborts_the_repair_and_cancels(m, clock):
    s = to_speaking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, _, c2 = barge_in(m, s, clock)
    assert C.LlmAbort(request_id="t1-repair") in c2
    assert_cancel(c2, "interrupted")


def test_r8_thinking_takeover_during_repair_cancels_as_takeover(m, clock):
    s = to_thinking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    t = clock.advance(100)
    s, _ = step(m, s, E.UserSpeechOnset(t_ms=t, onset_ms=t))
    s, cmds = step(m, s, E.UserTakeover(t_ms=clock.advance(700), onset_ms=t))
    assert s.state is TurnState.LISTENING and s.turn_id == "t2"
    assert C.LlmAbort(request_id="t1-repair") in cmds
    assert_cancel(cmds, "takeover")
    assert_no_execute(cmds, "P2: takeover")
    assert pending_tool(s) is None and s.tool_repairs == 0


@pytest.mark.parametrize(
    "event, reason",
    [
        (lambda t: E.LlmError(t_ms=t, message="engine died"), "error"),
        (lambda t: E.TtsError(t_ms=t, message="tts died"), "error"),
        (lambda t: E.ClientDisconnected(t_ms=t), "disconnected"),
    ],
    ids=["llm-error", "tts-error", "disconnect"],
)
def test_r8_error_rows_cancel_a_pending_valid_call(m, clock, event, reason):
    s = to_speaking(m, clock)
    s, _ = step(m, s, done(clock.advance(100), "t1-gen", draft("valid")))
    s, cmds = step(m, s, event(clock.advance(50)))
    assert s.state is TurnState.IDLE
    assert_cancel(cmds, reason)
    assert_no_execute(cmds, f"P2: {reason}")


@pytest.mark.parametrize(
    "event, reason",
    [
        (lambda t: E.TtsError(t_ms=t, message="tts died"), "error"),
        (lambda t: E.ClientDisconnected(t_ms=t), "disconnected"),
    ],
    ids=["tts-error", "disconnect"],
)
def test_r8_error_rows_during_repair_abort_it_and_cancel(m, clock, event, reason):
    s = to_speaking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, cmds = step(m, s, event(clock.advance(50)))
    assert s.state is TurnState.IDLE
    assert C.LlmAbort(request_id="t1-repair") in cmds
    assert_cancel(cmds, reason)


def test_r8_no_cancel_without_a_pending_call(m, clock):
    """Cancel only when ``state.tool`` is set: a toolless turn's error rows emit none."""
    s = to_speaking(m, clock)
    s, _ = step(m, s, done(clock.advance(100), "t1-gen"))
    _, cmds = step(m, s, E.ClientDisconnected(t_ms=clock.advance(50)))
    assert not of(cmds, "ToolCancel")
    _, _, c2 = barge_in(m, s, clock)
    assert not of(c2, "ToolCancel")


# --- R9 / R10: repair commits and stale ids --------------------------------------------------------


def test_r9_repair_commit_in_thinking_is_never_voiced_and_does_not_start_speaking(m, clock):
    s = to_thinking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, cmds = step(m, s, answer(s, clock, "Clicking the menu.", request_id="t1-repair"))
    assert s.state is TurnState.THINKING, "R9: a repair commit is not an answer"
    assert not [c for c in cmds if isinstance(c, C.TtsSpeak)]
    s, cmds = step(m, s, done(clock.advance(100), "t1-repair", draft("valid", source="repair")))
    assert not [c for c in cmds if isinstance(c, C.TtsSpeak)], "nor flushed on its llm.done"


def test_r9_repair_commit_in_speaking_is_never_voiced(m, clock):
    s = to_speaking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, cmds = step(m, s, answer(s, clock, "Clicking the menu.", request_id="t1-repair"))
    assert not [c for c in cmds if isinstance(c, C.TtsSpeak)]
    s, cmds = step(m, s, done(clock.advance(100), "t1-repair", draft("valid", source="repair")))
    assert not [c for c in cmds if isinstance(c, C.TtsSpeak)]
    s, cmds = step(m, s, drained(clock))
    assert s.history[-1].assistant_spoken == "Opening the menu now."


def test_r10_adversarial_stale_pass1_valid_done_during_repair_is_ignored(m, clock):
    s = to_speaking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    before = s
    s, cmds = step(m, s, done(clock.advance(100), "t1-gen", draft("valid", call_id="c9")))
    assert (s, cmds) == (before, []), f"stale pass-1 llm.done moved the machine: {kinds(cmds)}"
    s, cmds = step(m, s, E.LlmError(t_ms=clock.advance(10), request_id="t1-gen", message="x"))
    assert (s, cmds) == (before, []), "a stale pass-1 llm.error is ignored too"


def test_r10_a_repair_id_of_another_turn_is_stale(m, clock):
    s = to_speaking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    before = s
    s, cmds = step(m, s, done(clock.advance(100), "t0-repair", draft("valid", source="repair")))
    assert (s, cmds) == (before, [])


def test_adversarial_second_repairable_from_the_repair_is_cancelled_not_repaired(m, clock):
    s = to_thinking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, cmds = step(m, s, done(clock.advance(100), "t1-repair", draft("repairable")))
    assert not of(cmds, "ToolRepairStart")
    assert_cancel(cmds, "repair_failed")
    assert s.state is TurnState.IDLE and s.active_generate is None


# --- the toolless row is unchanged (criterion 2 last line) -----------------------------------------


def test_toolless_done_in_speaking_has_no_tool_command(m, clock):
    s = to_speaking(m, clock)
    s, cmds = step(m, s, done(clock.advance(100), "t1-gen"))
    s, c2 = step(m, s, drained(clock))
    for name in ("ToolExecute", "ToolCancel", "ToolRepairStart"):
        assert not of(cmds + c2, name)
    assert s.state is TurnState.IDLE


# --- ruling T4: PAUSED is a speaking state (R1, R2, R5, R6) ----------------------------------------


def to_paused(m, clock):
    s = to_speaking(m, clock)
    t = clock.advance(100)
    s, _ = step(m, s, E.UserSpeechOnset(t_ms=t, onset_ms=t))
    assert s.state is TurnState.PAUSED
    return s, t


def test_t4_r1_valid_in_paused_waits_for_resume_and_drain(m, clock):
    s, _ = to_paused(m, clock)
    s, cmds = step(m, s, done(clock.advance(50), "t1-gen", draft("valid")))
    assert s.state is TurnState.PAUSED and s.llm_done is True
    assert_no_execute(cmds, "T4: still paused")
    assert not of(cmds, "ToolCancel") and pending_tool(s) is not None
    s, cmds = step(m, s, E.UserBackchannel(t_ms=clock.advance(100), duration_ms=200))
    assert s.state is TurnState.SPEAKING
    assert_no_execute(cmds, "T4: resumed, not yet drained")
    s, cmds = step(m, s, drained(clock))
    assert s.state is TurnState.IDLE
    assert_execute_then_history(cmds, user_only=False)


def test_t4_r1_valid_in_paused_then_takeover_cancels_interrupted(m, clock):
    s, onset = to_paused(m, clock)
    s, _ = step(m, s, done(clock.advance(50), "t1-gen", draft("valid")))
    s, cmds = step(m, s, E.UserTakeover(t_ms=clock.advance(600), onset_ms=onset))
    assert s.state is TurnState.INTERRUPTED
    assert_cancel(cmds, "interrupted")
    assert_no_execute(cmds, "H5")


def test_t4_r2_repairable_in_paused_starts_the_repair_without_speaking(clock):
    m = TurnMachine(POLICY, segment=sentence_segmenter)
    s = to_thinking(m, clock)
    s, _ = step(m, s, answer(s, clock, "Sure. Opening it"))
    t = clock.advance(100)
    s, _ = step(m, s, E.UserSpeechOnset(t_ms=t, onset_ms=t))
    assert s.state is TurnState.PAUSED
    t_done = clock.advance(50)
    s, cmds = step(m, s, done(t_done, "t1-gen", draft("repairable")))
    assert s.state is TurnState.PAUSED and s.active_generate == "t1-repair"
    assert [(r.request_id, r.draft_request_id, r.call_id) for r in of(cmds, "ToolRepairStart")] == [
        ("t1-repair", "t1-gen", "c1")
    ]
    arms = [c.at_ms for c in cmds if isinstance(c, C.ArmTimer) and c.name == "tool_repair"]
    assert arms == [t_done + REPAIR_MS]
    assert not [c for c in cmds if isinstance(c, C.TtsSpeak)], "criterion 16: no synthesis paused"


def test_t4_r5_refused_in_paused_cancels_at_once(m, clock):
    s, _ = to_paused(m, clock)
    s, cmds = step(m, s, done(clock.advance(50), "t1-gen", draft("refused")))
    assert_cancel(cmds, "refused")
    assert s.state is TurnState.PAUSED


def _paused_during_repair(m, clock, *, before_deadline_ms: int):
    """Start a repair in SPEAKING, then pause ``before_deadline_ms`` before its deadline."""
    s = to_speaking(m, clock)
    s, _, t = start_repair(m, s, clock)
    onset = t + REPAIR_MS - before_deadline_ms
    clock.advance(onset - clock.now_ms)
    s, _ = step(m, s, E.UserSpeechOnset(t_ms=onset, onset_ms=onset))
    assert s.state is TurnState.PAUSED and s.active_generate == "t1-repair"
    return s, t + REPAIR_MS, onset


def test_t4_r6_repair_deadline_in_paused_fires_when_no_pause_timer_is_due(m, clock):
    s, deadline, _ = _paused_during_repair(m, clock, before_deadline_ms=100)  # takeover at +500
    s, cmds = step(m, s, E.Tick(t_ms=deadline))
    assert C.LlmAbort(request_id="t1-repair") in cmds
    assert_cancel(cmds, "repair_timeout")
    assert s.state is TurnState.PAUSED and s.llm_done is True
    s, _ = step(m, s, E.UserBackchannel(t_ms=deadline + 50, duration_ms=200))
    s, cmds = step(m, s, drained(clock))
    assert s.state is TurnState.IDLE
    assert_no_execute(cmds, "P2: the turn hit tool_repair")


def test_t4_r6_in_paused_takeover_is_checked_before_tool_repair(m, clock):
    s, deadline, onset = _paused_during_repair(m, clock, before_deadline_ms=700)  # takeover first
    s, cmds = step(m, s, E.Tick(t_ms=deadline))
    assert onset + POLICY.takeover_ms <= deadline
    assert s.state is TurnState.INTERRUPTED, "T4: takeover wins over tool_repair"
    assert_cancel(cmds, "interrupted")


def test_t4_r6_in_paused_resume_is_checked_before_tool_repair(m, clock):
    s, deadline, onset = _paused_during_repair(m, clock, before_deadline_ms=800)
    t_off = onset + 100
    s, _ = step(m, s, E.UserSpeechOffset(t_ms=t_off, offset_ms=t_off, speech_ms=100))
    assert t_off + POLICY.resume_max_ms <= deadline  # resume is due at the deadline tick too
    s, cmds = step(m, s, E.Tick(t_ms=deadline))
    assert s.state is TurnState.SPEAKING, "T4: resume is checked first"
    assert not of(cmds, "ToolCancel"), "the resume row returns before tool_repair is looked at"
    s, cmds = step(m, s, E.Tick(t_ms=deadline + 1))
    assert_cancel(cmds, "repair_timeout")


# --- ruling T5: LlmError(request_id=None) during a repair is R7 ---------------------------------------


def test_t5_none_id_llm_error_during_repair_is_repair_failed_and_speech_goes_on(m, clock):
    s = to_speaking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, cmds = step(m, s, E.LlmError(t_ms=clock.advance(100), message="engine hiccup"))
    assert_cancel(cmds, "repair_failed")
    assert C.LlmAbort(request_id="t1-repair") in cmds, "T8: the repair is aborted"
    assert s.state is TurnState.SPEAKING, "T5: the turn is not failed"
    assert not [c for c in cmds if isinstance(c, C.TtsCancel | C.PlaybackFlush | C.CommitHistory)]
    s, cmds = step(m, s, drained(clock))
    assert s.state is TurnState.IDLE
    assert_no_execute(cmds, "R7")


# --- ruling T13: rows for the mutants that survived review ------------------------------------------


def test_t13_r3_stamps_source_repair_even_when_the_repair_draft_says_draft(m, clock):
    s = to_thinking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, cmds = step(m, s, done(clock.advance(200), "t1-repair", draft("valid", source="draft")))
    assert s.state is TurnState.IDLE
    assert_execute_then_history(cmds, source="repair", user_only=True)


def test_t13_t7_the_pending_call_id_wins_over_a_valid_repair_draft(m, clock):
    s = to_thinking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    repaired = draft("valid", call_id="c7", source="repair")
    s, cmds = step(m, s, done(clock.advance(200), "t1-repair", repaired))
    assert_execute_then_history(cmds, source="repair", user_only=True)  # call_id "c1"


def test_t13_t7_the_pending_call_id_wins_over_a_failed_repair_draft(m, clock):
    s = to_speaking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    failed = draft("refused", call_id="c7", source="repair")
    s, cmds = step(m, s, done(clock.advance(200), "t1-repair", failed))
    assert_cancel(cmds, "repair_failed", call_id="c1")


def assert_tool_repair_cancelled(s, cmds, why: str):
    assert C.CancelTimer(name="tool_repair") in cmds, f"{why}: no CancelTimer(tool_repair)"
    assert not [t for t in s.timers if t.name == "tool_repair"], f"{why}: still armed"


def test_t13_barge_in_during_a_repair_cancels_the_tool_repair_timer(m, clock):
    s = to_speaking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, _, c2 = barge_in(m, s, clock)
    assert_tool_repair_cancelled(s, c2, "interrupt during the repair")


@pytest.mark.parametrize(
    "event",
    [
        lambda t: done(t, "t1-repair", draft("valid", source="repair")),
        lambda t: done(t, "t1-repair", None),
        lambda t: E.LlmError(t_ms=t, request_id="t1-repair", message="x"),
    ],
    ids=["R3", "R4", "R7"],
)
def test_t13_the_end_of_a_repair_in_speaking_cancels_the_tool_repair_timer(m, clock, event):
    s = to_speaking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    s, cmds = step(m, s, event(clock.advance(200)))
    assert s.state is TurnState.SPEAKING and s.active_generate is None
    assert_tool_repair_cancelled(s, cmds, "the repair is over")


def test_t13_t6_a_clause_held_while_paused_during_a_repair_is_spoken_at_resume(clock):
    """T6/T4: pass 1 is done once the repair runs, so its held clause is final at the resume."""
    m = TurnMachine(POLICY, segment=sentence_segmenter)
    s = to_thinking(m, clock)
    s, cmds = step(m, s, answer(s, clock, "Sure. Opening it"))
    assert [c.source for c in cmds if isinstance(c, C.TtsSpeak)] == ["Sure."]
    t = clock.advance(100)
    s, _ = step(m, s, E.UserSpeechOnset(t_ms=t, onset_ms=t))
    s, cmds, _ = start_repair(m, s, clock)
    assert s.state is TurnState.PAUSED and s.active_generate == "t1-repair"
    assert not [c for c in cmds if isinstance(c, C.TtsSpeak)], "criterion 16"
    s, cmds = step(m, s, E.UserBackchannel(t_ms=clock.advance(100), duration_ms=200))
    assert s.state is TurnState.SPEAKING
    assert [c.source for c in cmds if isinstance(c, C.TtsSpeak)] == [" Opening it"], (
        "T6: the clause held while paused is spoken at the resume, not left to the repair"
    )
    assert s.active_generate == "t1-repair" and s.llm_done is False


def test_t13_an_executed_call_with_an_empty_transcript_still_commits_history(m, clock):
    """R1 in THINKING with no word heard: the tuple still carries ``CommitHistory`` after the
    ``ToolExecute`` (A10's forced command), and no history entry is invented."""
    s = m.initial()
    onset = clock.advance(1000)
    s, _ = step(m, s, E.UserSpeechOnset(t_ms=onset, onset_ms=onset))
    t_eot = clock.advance(600)
    s, _ = step(m, s, E.SttEndOfTurn(t_ms=t_eot, t_eot_ms=t_eot, p_end=1.0))
    s, _ = step(m, s, E.SttFlushed(t_ms=clock.advance(100), t_eot_ms=t_eot))
    s, _ = step(m, s, E.Tick(t_ms=clock.advance(POLICY.eot_settle_ms)))
    assert s.state is TurnState.THINKING and s.words == ()
    s, cmds = step(m, s, done(clock.advance(300), "t1-gen", draft("valid")))
    assert s.state is TurnState.IDLE
    assert_execute_then_history(cmds, user_only=True)
    assert s.history == (), "nothing was said, so no history entry is invented"
