"""T-0162 group A, criterion A-2: the pure machine's ``session.reset`` row, one table row per state.

Every expectation is read off ``docs/tasks/T-0162.md`` A-2 and computed from the state the row is
delivered in -- never from an implementation (there is none yet). From **any** state the row:

- aborts whatever is live (``llm.abort`` for ``active_generate``, ``prefill_abort`` for
  ``active_prefill``), exactly once each, and nothing that is not live;
- cancels a pending tool call with ``ToolCancel(call_id, "reset")`` (``_tool_cancel(state, "reset")``);
- emits ``tts.cancel`` and ``playback.flush`` in SPEAKING, PAUSED and INTERRUPTED, and ``stop_clip``
  in THINKING when an acknowledgement clip may be playing (``clip_playing``);
- emits **no** ``commit_history``, no ``schedule_summary`` and no ``tts.speak``;
- ends the turn (``_end_turn``) with ``history=()`` and ``next_words=()``, then emits
  ``ResetSession(ended_turn_id)`` -- the turn the reset cut, ``None`` from IDLE;
- keeps ``turn_seq`` counting (turn ids are never reused) and keeps ``first_canvas_ms``.

Adversarial (the task's R1, "a stale aborted-request canvas or tool message repaints the page"): a
late ``playback.stopped`` / ``playback.drained`` / ``llm.commit`` / ``llm.done`` (with a valid tool
draft) / ``tool.result`` naming the pre-reset request or call emits nothing and stores nothing.

Names this module assumes, all from the task text: ``events.SessionReset`` (tag ``session.reset``),
``commands.ResetSession`` (tag ``reset_session``, field ``ended_turn_id``), and ``"reset"`` as a
``ToolCancelReason``. They are read through ``reset_event`` / ``reset_cls`` so that, until group A
lands, every row fails on a named assertion rather than an ``AttributeError``.

Regime: pure machine, ``VirtualClock`` fixture, ``TurnPolicy(thinking=True)``; n = 1 per row; no
measurement.
"""

from __future__ import annotations

import typing

import pytest

from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import TurnState
from tests.unit.test_turn_tool_rows import (
    answer,
    done,
    draft,
    drained,
    start_repair,
    step,
    to_speaking,
    to_thinking,
)

POLICY = TurnPolicy(thinking=True)

VOICING = (TurnState.SPEAKING, TurnState.PAUSED, TurnState.INTERRUPTED)

#: Command tags the reset row may emit (A-2), plus the bookkeeping a turn end may carry: the
#: machine's own ``hud.state`` on a state change, a ``cancel_timer`` for a deadline it drops, an
#: ``unfreeze`` of a frozen endpointing frame and a ``captioner.abort``. Anything else -- above all
#: ``commit_history``, ``schedule_summary``, ``tts.speak``, a new ``llm.start``/``prefill_open`` or a
#: ``tool.execute`` -- is a row that does more than reset.
ALLOWED = frozenset(
    {
        "llm.abort",
        "prefill_abort",
        "tool.cancel",
        "tts.cancel",
        "playback.flush",
        "stop_clip",
        "reset_session",
        "hud.state",
        "cancel_timer",
        "unfreeze",
        "captioner.abort",
    }
)
FORBIDDEN = ("commit_history", "schedule_summary", "tts.speak")


# --- the names the task adds, read so their absence is a named failure ----------------------------


def reset_event(t_ms: int):
    cls = getattr(E, "SessionReset", None)
    assert cls is not None, (
        "T-0162 A-2: contracts/events.py gains SessionReset (type session.reset)"
    )
    assert cls.model_fields["type"].default == "session.reset", cls.model_fields["type"].default
    return cls(t_ms=t_ms)


def reset_cls():
    cls = getattr(C, "ResetSession", None)
    assert cls is not None, (
        "T-0162 A-2: contracts/commands.py gains ResetSession(ended_turn_id: str | None)"
    )
    return cls


def tags(cmds) -> list[str]:
    return [c.type for c in cmds]


def of_type(cmds, tag: str) -> list:
    return [c for c in cmds if c.type == tag]


# --- the states, each reached through real events (the clip flag is the one exception) ------------


def idle_fresh(m, clock):
    return m.initial()


def idle_after_a_turn(m, clock):
    """A whole answered turn: history holds one entry and the first-canvas EMA is measured."""
    s = to_speaking(m, clock)
    s, _ = step(m, s, done(clock.advance(50), "t1-gen"))
    for u in s.utterances:
        s, _ = step(m, s, E.TtsDone(t_ms=clock.advance(10), utterance_id=u))
    s, _ = step(m, s, drained(clock))
    assert s.state is TurnState.IDLE and len(s.history) == 1, "premise: turn 1 completed"
    assert s.first_canvas_ms is not None, "premise: the first commit measured the EMA"
    return s


def listening(m, clock):
    t = clock.advance(1000)
    s, _ = step(m, m.initial(), E.UserSpeechOnset(t_ms=t, onset_ms=t))
    s, _ = step(m, s, E.SttWord(t_ms=t + 400, text="open", start_ms=t, end_ms=t + 200))
    assert s.state is TurnState.LISTENING and s.active_prefill is not None
    return s


def endpointing(m, clock):
    s = listening(m, clock)
    t = clock.advance(900)
    s, _ = step(m, s, E.SttEndOfTurn(t_ms=t, t_eot_ms=t - 50, p_end=1.0))
    assert s.state is TurnState.ENDPOINTING, s.state
    return s


def thinking(m, clock):
    return to_thinking(m, clock)


def thinking_with_a_clip(m, clock):
    """THINKING while an acknowledgement clip plays (the flag is set directly: the clip is timer-
    driven and the flag is exactly the per-turn fact A-2 keys ``stop_clip`` on)."""
    return to_thinking(m, clock).model_copy(update={"clip_playing": True})


def speaking(m, clock):
    return to_speaking(m, clock)


def paused(m, clock):
    s = to_speaking(m, clock)
    t = clock.advance(200)
    s, _ = step(m, s, E.UserSpeechOnset(t_ms=t, onset_ms=t))
    assert s.state is TurnState.PAUSED, s.state
    return s


def interrupted(m, clock):
    s = paused(m, clock)
    t = clock.advance(500)
    s, _ = step(m, s, E.UserTakeover(t_ms=t, onset_ms=t - 500))
    assert s.state is TurnState.INTERRUPTED, s.state
    return s


def repair_live(m, clock):
    s = to_thinking(m, clock)
    s, _, _ = start_repair(m, s, clock)
    assert s.active_generate == "t1-repair" and s.tool is not None, "premise: R2 started a repair"
    return s


def tool_pending(m, clock):
    """SPEAKING with a valid call held until the spoken prefix drains (R1's deferral)."""
    s = to_speaking(m, clock)
    s, _ = step(m, s, done(clock.advance(50), "t1-gen", draft("valid")))
    assert s.state is TurnState.SPEAKING and s.tool is not None, "premise: the call is pending"
    return s


STATES = {
    "IDLE": idle_fresh,
    "IDLE-after-a-turn": idle_after_a_turn,
    "LISTENING": listening,
    "ENDPOINTING": endpointing,
    "THINKING": thinking,
    "THINKING-clip": thinking_with_a_clip,
    "SPEAKING": speaking,
    "PAUSED": paused,
    "INTERRUPTED": interrupted,
    "repair-live": repair_live,
    "tool-pending": tool_pending,
}


def check_reset_row(before, after, cmds) -> None:
    """A-2's command set and end state, computed from ``before`` alone."""
    k = tags(cmds)
    for tag in FORBIDDEN:
        assert tag not in k, f"A-2: the reset row emits no {tag}; got {k}"
    stray = [t for t in k if t not in ALLOWED]
    assert not stray, (
        f"A-2: the reset row emits only aborts, cancels and ResetSession; extra {stray}"
    )

    aborts = [c.request_id for c in of_type(cmds, "llm.abort")]
    want = [before.active_generate] if before.active_generate is not None else []
    assert aborts == want, f"A-2: one llm.abort per live generate {want}; got {aborts}"
    prefill_aborts = [c.request_id for c in of_type(cmds, "prefill_abort")]
    want = [before.active_prefill] if before.active_prefill is not None else []
    assert prefill_aborts == want, (
        f"A-2: one prefill_abort per live prefill {want}; got {prefill_aborts}"
    )

    cancels = [(c.call_id, c.reason) for c in of_type(cmds, "tool.cancel")]
    want_cancel = [(before.tool.call_id, "reset")] if before.tool is not None else []
    assert cancels == want_cancel, (
        f"A-2: _tool_cancel(state, 'reset') gives {want_cancel}; got {cancels}"
    )

    voicing = before.state in VOICING
    for tag in ("tts.cancel", "playback.flush"):
        n = len(of_type(cmds, tag))
        assert n == (1 if voicing else 0), (
            f"A-2: {tag} exactly once in SPEAKING/PAUSED/INTERRUPTED and never elsewhere; "
            f"{n} in {before.state.value}"
        )
    if before.state is TurnState.THINKING and before.clip_playing:
        assert len(of_type(cmds, "stop_clip")) == 1, (
            f"A-2: stop_clip in THINKING with a clip; got {k}"
        )

    resets = [c for c in cmds if isinstance(c, reset_cls())]
    ended = None if before.state is TurnState.IDLE else before.turn_id
    assert [c.ended_turn_id for c in resets] == [ended], (
        f"A-2: exactly one ResetSession(ended_turn_id={ended!r}) from {before.state.value}; "
        f"got {[c.ended_turn_id for c in resets]} in {k}"
    )
    at = cmds.index(resets[0])
    for i, c in enumerate(cmds):
        if c.type in ("llm.abort", "prefill_abort", "tool.cancel", "tts.cancel", "playback.flush"):
            assert i < at, f"A-2: the aborts and cancels come before ResetSession; got {k}"
    if after.state is not before.state:
        assert k[-1] == "hud.state", f"criterion 21: hud.state stays last; got {k}"

    assert after.state is TurnState.IDLE, after.state
    assert after.history == (), f"A-2: history=() after a reset; got {after.history}"
    assert after.next_words == () and after.words == (), (after.words, after.next_words)
    assert after.active_generate is None and after.active_prefill is None
    assert after.tool is None and after.turn_phashes == ()
    assert after.committed_answer == "" and after.spoken_text == "" and after.held_text == ""
    assert after.turn_seq == before.turn_seq, "A-2: turn_seq keeps counting (no rewind)"
    assert after.first_canvas_ms == before.first_canvas_ms, "A-2: the first-canvas EMA survives"


@pytest.mark.parametrize("name", list(STATES))
def test_the_reset_row_from_every_state(name, clock):
    m = TurnMachine(POLICY)
    before = STATES[name](m, clock)
    untouched = before.model_copy(deep=True)
    after, cmds = m.step(before, reset_event(clock.advance(5)))
    assert before == untouched, "the machine is pure: step mutated the state it was given"
    check_reset_row(before, after, list(cmds))


def test_the_contract_names_the_task_adds():
    """A-2: the event joins EVENT_TYPES, the command joins the union, "reset" is a cancel reason."""
    ev = reset_event(0)
    assert type(ev) in E.EVENT_TYPES, "A-2: SessionReset is in EVENT_TYPES"
    cls = reset_cls()
    assert cls.model_fields["type"].default == "reset_session", cls.model_fields["type"].default
    assert set(cls.model_fields) == {"type", "ended_turn_id"}, sorted(cls.model_fields)
    members = typing.get_args(typing.get_args(C.Command)[0])
    assert cls in members, "A-2: ResetSession is a member of the Command union"
    assert "reset" in typing.get_args(C.ToolCancelReason), (
        "A-2 names _tool_cancel(state, 'reset'): 'reset' is a ToolCancelReason"
    )


def test_turn_ids_are_never_reused_across_a_reset(clock):
    """A-2: ``turn_seq`` keeps counting, so the turn after a reset is t2, not a second t1."""
    m = TurnMachine(POLICY)
    s = idle_after_a_turn(m, clock)
    seen = {"t1"}
    s, _ = m.step(s, reset_event(clock.advance(5)))
    t = clock.advance(500)
    s, _ = m.step(s, E.UserSpeechOnset(t_ms=t, onset_ms=t))
    assert s.state is TurnState.LISTENING
    assert s.turn_seq == 2 and s.turn_id == "t2" and s.turn_id not in seen, (s.turn_seq, s.turn_id)
    # and a reset mid-turn, then another turn: t3
    s, _ = m.step(s, reset_event(clock.advance(5)))
    t = clock.advance(500)
    s, _ = m.step(s, E.UserSpeechOnset(t_ms=t, onset_ms=t))
    assert s.turn_id == "t3", s.turn_id


def test_the_first_canvas_ema_survives_and_still_updates(clock):
    """A-2: ``first_canvas_ms`` survives the reset (the ack-clip decision keeps its estimate)."""
    m = TurnMachine(POLICY)
    s = idle_after_a_turn(m, clock)
    ema = s.first_canvas_ms
    s, _ = m.step(s, reset_event(clock.advance(5)))
    assert s.first_canvas_ms == ema


def test_the_next_turn_after_a_reset_has_no_history_to_commit_beside(clock):
    """A-2 / A-7: after the reset the next answered turn stores itself only (history of one)."""
    m = TurnMachine(POLICY)
    s = idle_after_a_turn(m, clock)
    s, _ = m.step(s, reset_event(clock.advance(5)))
    # turn t2: the same canonical path, with its own words
    onset = clock.advance(1000)
    s, _ = step(m, s, E.UserSpeechOnset(t_ms=onset, onset_ms=onset))
    s, _ = step(m, s, E.SttWord(t_ms=onset + 400, text="again", start_ms=onset, end_ms=onset + 200))
    clock.advance(900)
    t_eot = clock.now_ms - 50
    s, _ = step(m, s, E.SttEndOfTurn(t_ms=clock.now_ms, t_eot_ms=t_eot, p_end=1.0))
    s, _ = step(m, s, E.SttFlushed(t_ms=clock.advance(100), t_eot_ms=t_eot))
    s, _ = step(m, s, E.Tick(t_ms=clock.advance(POLICY.eot_settle_ms)))
    assert s.state is TurnState.THINKING and s.active_generate == "t2-gen"
    s, _ = step(m, s, answer(s, clock, "Second answer."))
    s, _ = step(m, s, done(clock.advance(50), "t2-gen"))
    s, _ = step(m, s, drained(clock, "Second answer."))
    assert s.state is TurnState.IDLE
    assert [h.user_text for h in s.history] == ["again"], (
        f"the pre-reset turn is gone from history for good; got {[h.user_text for h in s.history]}"
    )


# --- adversarial: late events for the pre-reset request are inert --------------------------------


@pytest.mark.parametrize("name", ["SPEAKING", "PAUSED", "THINKING", "tool-pending", "repair-live"])
def test_late_events_for_the_pre_reset_request_are_inert(name, clock):
    """R1 / A-2: what the engine and the device still say about the cut turn moves nothing.

    The session aborts and flushes, but an aborted request can still have a commit or a done in
    flight, and the device reports what it had played when it was flushed. Each of those, keyed on
    the pre-reset request, call or turn, must emit no command and change no state -- above all no
    ``commit_history`` (the cut turn's played text must not come back into the new session's
    history) and no ``tool.execute`` (a valid draft on a stale ``llm.done``).
    """
    m = TurnMachine(POLICY)
    before = STATES[name](m, clock)
    live = before.active_generate or f"{before.turn_id}-gen"
    call = before.tool.call_id if before.tool is not None else "c1"
    s, _ = m.step(before, reset_event(clock.advance(5)))
    reset_state = s
    late = [
        E.PlaybackStopped(t_ms=clock.advance(5), played_ms=900, played_text="Opening the menu"),
        E.PlaybackDrained(
            t_ms=clock.advance(5), played_ms=1500, played_text="Opening the menu now."
        ),
        E.LlmCommit(
            t_ms=clock.advance(5),
            request_id=live,
            canvas_idx=7,
            segment="answer",
            text="A stale sentence. ",
            thought_open=False,
        ),
        done(clock.advance(5), live, draft("valid", call_id=call)),
        E.ToolResult(t_ms=clock.advance(5), call_id=call, status="executed"),
    ]
    for ev in late:
        s2, cmds = m.step(s, ev)
        assert list(cmds) == [], f"a late {ev.type} after the reset emitted {tags(cmds)}"
        assert s2.history == () and s2.state is TurnState.IDLE, (
            f"a late {ev.type} after the reset moved the machine: {s2.state.value}, {s2.history}"
        )
        assert s2.model_copy(update={"tts_done": s.tts_done}) == s, (
            f"a late {ev.type} changed the post-reset state"
        )
        s = s2
    assert s.history == reset_state.history == ()


def test_a_reset_while_idle_cuts_no_turn(clock):
    """A-2: from IDLE there is nothing to end -- ``ended_turn_id`` is None and nothing is aborted."""
    m = TurnMachine(POLICY)
    s = idle_after_a_turn(m, clock)
    s2, cmds = m.step(s, reset_event(clock.advance(5)))
    resets = [c for c in cmds if isinstance(c, reset_cls())]
    assert [c.ended_turn_id for c in resets] == [None], tags(cmds)
    assert not of_type(cmds, "llm.abort") and not of_type(cmds, "prefill_abort")
    assert s2.history == (), "the history of the finished turn is cleared too"


@pytest.mark.parametrize("name", ["IDLE-after-a-turn", "SPEAKING", "PAUSED"])
def test_the_next_turn_after_a_reset_opens_with_no_barge_in_onset(name, clock):
    """Reviewer A (M16): whatever the reset leaves behind for its own IDLE ends when the next turn opens.

    ``pause_onset_ms`` is the onset of a barge-in that paused playback (contracts/turn.py); a turn opened by a fresh
    onset has had no barge-in, so in LISTENING it is ``None`` -- including from PAUSED, where the cut turn had one.
    """
    m = TurnMachine(POLICY)
    s = STATES[name](m, clock)
    s, _ = m.step(s, reset_event(clock.advance(5)))
    t = clock.advance(500)
    s, _ = m.step(s, E.UserSpeechOnset(t_ms=t, onset_ms=t))
    assert s.state is TurnState.LISTENING, s.state
    assert s.pause_onset_ms is None, (
        f"a turn opened after a reset carries a barge-in onset {s.pause_onset_ms} it never had"
    )
    # and the new turn behaves as one: a word lands in it
    s, _ = m.step(s, E.SttWord(t_ms=t + 300, text="hello", start_ms=t, end_ms=t + 200))
    assert [w.text for w in s.words] == ["hello"], s.words
