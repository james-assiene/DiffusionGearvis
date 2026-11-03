"""T-0193 group A0, the pure machine: ``UserStop`` (the HUD Stop button) ends a speaking or thinking turn in IDLE
(docs/tasks/T-0193.md A0-1..A0-7, A0-9's pin moves; docs/02 §4).

Every expected tuple is written by hand from the criteria:

- A0-3 (SPEAKING/PAUSED, the turn's own id): ``llm.abort`` if a generate (or the repair) is live; ``tts.cancel``,
  ``playback.flush``; ``tool.cancel(stopped)`` if a call is pending; ``cancel_timer`` for each of ``takeover``,
  ``resume``, ``tool_repair`` that is armed (in that order); ``arm_timer(interrupt_timeout, t + 300)``;
  ``hud.state(INTERRUPTED)`` (last, criterion 21). The turn is marked as stopped.
- A0-4 (the exits of a stopped INTERRUPTED): ``truncate_history_to_played(played_ms, played_text)`` (row 19's
  arguments: the event's on ``playback.stopped``, ``(None, "")`` on the timeout), then ``hud.state(IDLE)``; history
  gains ``HistoryTurn(user_text, joined played + " [interrupted]" or "[interrupted]", interrupted=True)``; the state
  is IDLE with every per-turn field at its ``MachineState`` default (the expected state is built from the contract's
  defaults plus the fields that survive a turn, never from ``_PER_TURN_RESET``).
- A0-5 (THINKING): ``llm.abort`` if a generate is live; ``tool.cancel(stopped)`` if a call is pending; ``stop_clip``
  iff a clip plays; ``hud.state(IDLE)``; history as A0-4 with ``played_prior`` (H-T193-1's default: kept).
- A0-2: IDLE, LISTENING, ENDPOINTING (before and after the flush), INTERRUPTED and any ``turn_id`` other than the
  current one: ``(s, ())``.

Readings where the criteria are silent (stated in the test-author's report):

- A0-5's tuple lists no ``cancel_timer``: the THINKING stop ends the turn the way the disconnect and reset rows do, its
  timers dropped with the per-turn reset, so the rows pin the tuple exactly as listed (no ``cancel_timer``);
- A0-5's order is the order the criterion lists (abort, cancel, stop_clip, hud.state);
- A0-6's late ``playback.stopped`` / ``playback.drained`` are held to "no command, no history change, no state
  change of ``state``/``turn_id``/``turn_seq``" (BASE's IDLE rows record ``played_text`` without a command; A0-6 asks
  only that no second history entry is added), while a late ``llm.commit``/``llm.done``/``tool.result``/``tick`` must
  leave the whole state unchanged.

The pin moves (A0-9: ``TURN_MACHINE_SHA256`` "re-pinned wherever it is pinned") cannot know the implementation's bytes
before it exists. They move the T-0192 B way: each file is checked at T-0193 A0's literal base fdb9fc01 against its
earlier sha256 pin (the base end), and the working tree is held to "that blob plus exactly A0's additions" by an AST
comparison (``t0193_drift``). A0-R5, after the implementation was final: the census pins now hold the working tree
to sha256 literals of the merged blobs at 373df531 (``T0193_MOVED_SHA256``, as T-0192 B-RV6 did at 97409910); the
drift rows stay as the check of the A0 delta's shape.

Regime: CPU, pure ``step``, virtual time carried by the events; n = 1 per row, deterministic.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import pathlib
import subprocess
import typing
from typing import Any

import pytest

from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import HistoryTurn, MachineState, Timer, TurnState
from tests import _git_range as gr
from tests.unit import test_turn_invariants as ti

USER = "what is this"
ANSWER = "Yes. "
PREFIX = "Opening it."
STEP2 = "I will check the price. "
REPORT = "Done."
INTERRUPT_MS = 300  # TurnPolicy.interrupt_timeout_ms (docs/02 §4 row 22)
#: ``MachineState``'s fields at BASE fdb9fc01, copied by hand from ``git show fdb9fc01:local_astra/contracts/turn.py``
#: (``MachineState``, in declaration order). A0-1 adds exactly one: the per-turn stop mark.
BASE_STATE_FIELDS = (
    "state",
    "turn_id",
    "turn_seq",
    "onset_ms",
    "t_eot_ms",
    "words",
    "next_words",
    "turn_phashes",
    "active_generate",
    "active_prefill",
    "committed_answer",
    "spoken_text",
    "held_text",
    "utterances",
    "utterance_seq",
    "timers",
    "history",
    "fillers_this_turn",
    "first_canvas_ms",
    "first_canvas_measured",
    "llm_start_ms",
    "flush_t_ms",
    "pause_onset_ms",
    "played_text",
    "llm_done",
    "tts_done",
    "playback_drained",
    "clip_playing",
    "degraded",
    "tool",
    "tool_repairs",
    "spoken_after_drain",
    "tool_step",
    "tool_steps_executed",
    "awaiting_result",
    "step_text",
    "played_prior",
    "still_working_played",
)
#: ``ToolCancelReason`` at BASE (fdb9fc01 ``contracts/commands.py`` lines 219-229, by hand)
BASE_CANCEL_REASONS = (
    "interrupted",
    "takeover",
    "error",
    "disconnected",
    "refused",
    "repair_failed",
    "repair_timeout",
    "step_budget",
    "reset",
)
#: ``EVENT_TYPES`` class names at BASE (fdb9fc01 ``contracts/events.py`` lines 296-324, by hand), in order
BASE_EVENT_CLASSES = (
    "UserSpeechOnset",
    "UserBackchannel",
    "UserTakeover",
    "UserPause",
    "UserSpeechOffset",
    "UserNotOwner",
    "UserText",
    "SttWord",
    "SttStep",
    "SttEndOfTurn",
    "SttFlushed",
    "SttError",
    "SceneChanged",
    "LlmCommit",
    "LlmDone",
    "LlmError",
    "ToolResult",
    "PrefillReady",
    "TtsDone",
    "TtsError",
    "PlaybackStopped",
    "PlaybackDrained",
    "UiAction",
    "ClientDisconnected",
    "SessionReset",
    "MicMuted",
    "Tick",
)
#: the command tags at BASE (fdb9fc01 ``contracts/commands.py``'s ``Command`` union, by hand)
BASE_COMMAND_TAGS = (
    "snapshot_frame",
    "prefill_open",
    "prefill_reissue",
    "prefill_abort",
    "freeze_after",
    "unfreeze",
    "check_owner",
    "llm.start",
    "llm.abort",
    "llm.continue",
    "captioner.abort",
    "play_clip",
    "stop_clip",
    "arm_timer",
    "cancel_timer",
    "tts.speak",
    "tts.cancel",
    "playback.pause",
    "playback.resume",
    "playback.flush",
    "truncate_history_to_played",
    "commit_history",
    "schedule_summary",
    "hud.state",
    "hud.error",
    "degrade",
    "tool.repair_start",
    "tool.execute",
    "tool.cancel",
    "reset_session",
)
BASE_TIMER_NAMES = (
    "eot_settle",
    "takeover",
    "resume",
    "llm_timeout",
    "interrupt_timeout",
    "interim_update",
    "tool_repair",
    "tool_result",
)
BASE_CLIP_KINDS = (
    "ack",
    "hmm",
    "still_working",
    "out_of_steps",
    "not_finished",
    "lost_thread",
    "at_checkout",
)
#: A0-4 / A0-5: what a stop and the stopped turn's exit never emit
NEVER_AFTER_STOP = (
    "snapshot_frame",
    "prefill_open",
    "commit_history",
    "schedule_summary",
    "play_clip",
    "tts.speak",
    "hud.error",
    "llm.start",
    "llm.continue",
    "tool.execute",
    "tool.repair_start",
)


# =====================================================================================================================
# helpers
# =====================================================================================================================


def stop(t_ms: int, turn_id: str) -> Any:
    """``UserStop(t_ms, turn_id)``, or a named failure while A0-1's event does not exist."""
    cls = getattr(E, "UserStop", None)
    assert cls is not None, (
        "T-0193 A0-1: contracts/events.py gains UserStop (type 'user.stop', t_ms, turn_id: str)"
    )
    return cls(t_ms=t_ms, turn_id=turn_id)


def stopped_cancel(call_id: str) -> C.ToolCancel:
    """``ToolCancel(call_id, "stopped")``, or a named failure while A0-1's reason does not exist."""
    assert "stopped" in typing.get_args(C.ToolCancelReason), (
        f'T-0193 A0-1: ToolCancelReason gains "stopped"; it is {typing.get_args(C.ToolCancelReason)}'
    )
    return C.ToolCancel(call_id=call_id, reason="stopped")  # type: ignore[arg-type]


def machine(**kw: Any) -> TurnMachine:
    """``TurnPolicy(thinking=True)`` with the ack clip off (the cli default, T-0182 B-R4) unless overridden."""
    return TurnMachine(TurnPolicy(**{"thinking": True, "ack_clip": False, **kw}))


def thinking(
    m: Any, *, at: int = 1000, state: MachineState | None = None
) -> tuple[MachineState, int]:
    """IDLE -> THINKING with the three words of ``USER``; returns the state and the llm.start time."""
    words = [("what", at + 100), ("is", at + 400), ("this", at + 700)]
    s, cmds, t0 = ti.drive_to_thinking(m, onset_ms=at, t_eot_ms=at + 1600, words=words, state=state)
    assert s.state is TurnState.THINKING and any(isinstance(c, C.LlmStart) for c in cmds), "premise"
    return s, t0


def speaking(
    m: Any, *, at: int = 1000, state: MachineState | None = None
) -> tuple[MachineState, int]:
    """THINKING -> SPEAKING on one answer commit (``ANSWER``), the generate still live."""
    s, t0 = thinking(m, at=at, state=state)
    s, cmds = m.step(s, ti.commit(s, ANSWER, t=t0 + 300))
    assert s.state is TurnState.SPEAKING and [
        c.source for c in cmds if isinstance(c, C.TtsSpeak)
    ] == [ANSWER]
    assert s.active_generate == f"{s.turn_id}-gen", "premise: the generate is live"
    return s, t0 + 300


def click(call_id: str, status: str = "valid") -> Any:
    if status == "repairable":
        return ti.tool_draft(call_id=call_id, status=status, name="click_at", reason="schema")
    return ti.tool_draft(
        call_id=call_id, status=status, name="click_at", arguments={"x": 500, "y": 250}
    )


def loop_awaiting(m: Any, *, prefix: bool) -> tuple[MachineState, int, int]:
    """Step 1 of a loop turn executes ``c1`` (R11), optionally after speaking ``PREFIX`` (drained first). Returns the
    awaiting state, llm.start time and the time of the last event."""
    s, t0 = thinking(m)
    t = t0 + 200
    if prefix:
        s, _ = m.step(s, ti.commit(s, PREFIX, t=t))
        assert s.state is TurnState.SPEAKING, "premise: the prefix is voiced"
        t += 100
        s, _ = m.step(s, ti.llm_done(t, s.active_generate, click("c1")))
        t += 600
        s, cmds = m.step(s, E.PlaybackDrained(t_ms=t, played_ms=600, played_text=PREFIX))
    else:
        s, cmds = m.step(s, ti.llm_done(t, s.active_generate, click("c1")))
    assert [c.call_id for c in cmds if isinstance(c, C.ToolExecute)] == ["c1"], ti.kinds(cmds)
    assert s.state is TurnState.THINKING and s.awaiting_result == "c1", "premise: c1 is awaited"
    return s, t0, t


def loop_step2(m: Any) -> tuple[MachineState, int]:
    """A loop turn at step 2: ``PREFIX`` played, c1 executed and answered, step 2's generate live with ``STEP2``
    committed (R14: withheld so far)."""
    s, _, t = loop_awaiting(m, prefix=True)
    t += 50
    s, cmds = m.step(s, E.ToolResult(t_ms=t, call_id="c1", status="executed"))
    assert any(isinstance(c, C.LlmContinue) for c in cmds), ti.kinds(cmds)
    assert s.active_generate == "t1-gen-s2" and s.tool_step == 2, "premise: step 2 is live"
    t += 100
    s, cmds = m.step(s, ti.commit(s, STEP2, t=t))
    assert cmds == () and s.state is TurnState.THINKING, "premise: R14 holds step 2's text"
    return s, t


def idle_after(pre: MachineState, history: tuple[HistoryTurn, ...]) -> MachineState:
    """The expected IDLE state after ``pre``'s turn ends: ``MachineState``'s defaults plus what survives a turn
    (``turn_id``, ``turn_seq``, ``first_canvas_ms``, ``degraded``, ``next_words``) and ``history``."""
    return MachineState(
        turn_id=pre.turn_id,
        turn_seq=pre.turn_seq,
        history=history,
        first_canvas_ms=pre.first_canvas_ms,
        degraded=pre.degraded,
        next_words=pre.next_words,
    )


def entry(assistant: str) -> HistoryTurn:
    return HistoryTurn(user_text=USER, assistant_spoken=assistant, interrupted=True)


def hud(state: TurnState, turn_id: str = "t1") -> C.HudState:
    return C.HudState(state=state.value, turn_id=turn_id)


def arm_interrupt(t_ms: int) -> C.ArmTimer:
    return C.ArmTimer(name="interrupt_timeout", at_ms=t_ms + INTERRUPT_MS)


def base_fields(s: MachineState) -> dict[str, Any]:
    return {name: getattr(s, name) for name in BASE_STATE_FIELDS}


def stop_field() -> str:
    """The one ``MachineState`` field A0-1 adds (its name is the implementation's; it is read, never called)."""
    added = sorted(set(MachineState.model_fields) - set(BASE_STATE_FIELDS))
    assert len(added) == 1, (
        f"T-0193 A0-1: MachineState gains exactly one per-turn stop field; added {added}"
    )
    return added[0]


def inert(m: Any, s: MachineState, ev: Any, why: str) -> None:
    new, cmds = m.step(s, ev)
    assert (new, tuple(cmds)) == (s, ()), (
        f"{why}: {ev.type} moved the machine / emitted {ti.kinds(cmds)}"
    )


def frozen_machine(**kw: Any) -> Any:
    """BASE's machine (fdb9fc01), imported on use."""
    from tests.unit import _frozen_turn_machine_t0193 as frozen

    return frozen.TurnMachine(frozen.TurnPolicy(**{"thinking": True, "ack_clip": False, **kw}))


# =====================================================================================================================
# A0-1: the contract
# =====================================================================================================================


def test_a0_1_user_stop_event() -> None:
    """A0-1: ``UserStop`` has type ``"user.stop"`` and exactly ``t_ms`` and ``turn_id: str`` (required); it validates
    through the ``Event`` union and joins ``EVENT_TYPES`` once, BASE's order otherwise unchanged."""
    ev = stop(5, "t1")
    assert type(ev).model_fields["type"].default == "user.stop"
    assert set(type(ev).model_fields) == {"t_ms", "type", "turn_id"}, sorted(type(ev).model_fields)
    assert ev.turn_id == "t1" and ev.t_ms == 5
    for bad in ({"t_ms": 5}, {"t_ms": 5, "turn_id": 3}, {"t_ms": 5, "turn_id": "t1", "x": 1}):
        with pytest.raises(ValueError):
            type(ev)(**bad)
    parsed = E.EventAdapter.validate_python({"type": "user.stop", "t_ms": 7, "turn_id": "t2"})
    assert type(parsed) is type(ev) and parsed.turn_id == "t2"
    names = [cls.__name__ for cls in E.EVENT_TYPES]
    assert names.count("UserStop") == 1, names
    assert [n for n in names if n != "UserStop"] == list(BASE_EVENT_CLASSES), names
    union = [cls.__name__ for cls in typing.get_args(typing.get_args(E.Event)[0])]
    assert union.count("UserStop") == 1 and sorted(union) == sorted(
        [*BASE_EVENT_CLASSES, "UserStop"]
    ), union


def test_a0_1_cancel_reason_gains_stopped_and_nothing_else_is_added() -> None:
    """A0-1: ``ToolCancelReason`` is BASE's plus exactly ``"stopped"``; no command kind, timer or clip kind is added;
    ``MachineState`` gains exactly one field, excluded from dumps (the ``still_working_played`` precedent)."""
    reasons = typing.get_args(C.ToolCancelReason)
    assert sorted(reasons) == sorted([*BASE_CANCEL_REASONS, "stopped"]), reasons
    assert stopped_cancel("c1").reason == "stopped"
    tags = [
        cls.model_fields["type"].default for cls in typing.get_args(typing.get_args(C.Command)[0])
    ]
    assert sorted(tags) == sorted(BASE_COMMAND_TAGS), tags
    assert typing.get_args(C.TimerName) == BASE_TIMER_NAMES
    assert typing.get_args(C.ClipKind) == BASE_CLIP_KINDS
    name = stop_field()
    assert MachineState.model_fields[name].exclude is True, (
        f"A0-1: MachineState.{name} is excluded from dumps"
    )
    assert set(MachineState().model_dump()) == set(BASE_STATE_FIELDS) - {"still_working_played"}


# =====================================================================================================================
# A0-2: no-op states and the wrong turn id
# =====================================================================================================================


def _listening(m: Any) -> MachineState:
    s, _ = m.step(MachineState(), E.UserSpeechOnset(t_ms=1000, onset_ms=1000))
    s, _ = m.step(s, E.SttWord(t_ms=1500, text="what", start_ms=1100, end_ms=1300))
    assert s.state is TurnState.LISTENING and s.active_prefill == "t1-prefill", "premise"
    return s


def _noop_states(m: Any) -> dict[str, MachineState]:
    listening = _listening(m)
    before_flush, _ = m.step(listening, E.SttEndOfTurn(t_ms=1800, t_eot_ms=1750))
    after_flush, _ = m.step(before_flush, E.SttFlushed(t_ms=1900, t_eot_ms=1750))
    assert before_flush.state is TurnState.ENDPOINTING and before_flush.flush_t_ms is None
    assert after_flush.state is TurnState.ENDPOINTING and after_flush.flush_t_ms == 1900
    s, t = speaking(m)
    interrupted, _ = m.step(s, E.UserTakeover(t_ms=t + 100, onset_ms=t + 50))
    assert interrupted.state is TurnState.INTERRUPTED, "premise: a voice interrupt"
    done, t = speaking(m)
    done, _ = m.step(done, E.LlmDone(t_ms=t + 10, request_id="t1-gen"))
    done, _ = m.step(done, E.PlaybackDrained(t_ms=t + 600, played_ms=500, played_text=ANSWER))
    assert done.state is TurnState.IDLE and done.turn_id == "t1", (
        "premise: t1 completed, IDLE keeps its id"
    )
    return {
        "idle-initial": MachineState(),
        "idle-after-turn": done,
        "listening": listening,
        "endpointing-before-flush": before_flush,
        "endpointing-after-flush": after_flush,
        "interrupted-voice": interrupted,
    }


@pytest.mark.parametrize(
    "which",
    [
        "idle-initial",
        "idle-after-turn",
        "listening",
        "endpointing-before-flush",
        "endpointing-after-flush",
        "interrupted-voice",
    ],
)
def test_noop_states(which: str) -> None:
    """A0-2: a stop naming the current turn is ``(s, ())`` in IDLE, LISTENING, ENDPOINTING (before and after the
    flush) and INTERRUPTED: LISTENING/ENDPOINTING belong to the user (the live prefill is not aborted), INTERRUPTED
    is already stopping. The premise row below shows the same id acts in SPEAKING."""
    m = machine()
    s = _noop_states(m)[which]
    stop(1, s.turn_id)
    inert(m, s, stop(9000, s.turn_id), f"A0-2 ({which})")


def _second_turn(m: Any, where: str) -> MachineState:
    """t1 answered and completed; t2 in ``where`` (thinking, speaking or paused)."""
    s, t = speaking(m)
    s, _ = m.step(s, E.LlmDone(t_ms=t + 10, request_id="t1-gen"))
    s, _ = m.step(s, E.PlaybackDrained(t_ms=t + 600, played_ms=500, played_text=ANSWER))
    assert s.state is TurnState.IDLE, "premise"
    if where == "thinking":
        s, _ = thinking(m, at=10_000, state=s)
    else:
        s, t = speaking(m, at=10_000, state=s)
        if where == "paused":
            s, _ = m.step(s, E.UserSpeechOnset(t_ms=t + 100, onset_ms=t + 100))
    assert s.turn_id == "t2" and s.state is TurnState[where.upper()], "premise"
    return s


@pytest.mark.parametrize("where", ["thinking", "speaking", "paused"])
@pytest.mark.parametrize("turn_id", ["t1", "t3", "t0", "", "T2", "t2 ", "2"])
def test_wrong_turn_id_noop(where: str, turn_id: str) -> None:
    """A0-2, adversarial (a stale click must not cut the next turn): in t2's THINKING, SPEAKING and PAUSED, a stop
    naming the previous turn t1, a later id, or a near miss of "t2" is ``(s, ())``."""
    m = machine()
    s = _second_turn(m, where)
    inert(m, s, stop(20_000, turn_id), f"A0-2 wrong id {turn_id!r} in {where}")


@pytest.mark.parametrize("where", ["thinking", "speaking", "paused"])
def test_the_current_id_acts_in_the_second_turn(where: str) -> None:
    """A0-3/A0-5, the other direction of the turn-id guard: the same states, stopped with t2's own id, act (a guard
    that compares with the wrong field, or refuses every stop after the first turn, fails here)."""
    m = machine()
    s = _second_turn(m, where)
    new, cmds = m.step(s, stop(20_000, "t2"))
    want = TurnState.IDLE if where == "thinking" else TurnState.INTERRUPTED
    assert new.state is want and C.LlmAbort(request_id="t2-gen") in cmds, ti.kinds(cmds)


# =====================================================================================================================
# A0-3: SPEAKING / PAUSED
# =====================================================================================================================


def test_speaking_stop_tuple_generate_live() -> None:
    """A0-3, generate live: exactly abort, cancel, flush, arm the interrupt timeout, INTERRUPTED; nothing of
    ``NEVER_AFTER_STOP``; the generate is no longer live."""
    m = machine()
    s, t = speaking(m)
    new, cmds = m.step(s, stop(t + 100, "t1"))
    assert tuple(cmds) == (
        C.LlmAbort(request_id="t1-gen"),
        C.TtsCancel(),
        C.PlaybackFlush(),
        arm_interrupt(t + 100),
        hud(TurnState.INTERRUPTED),
    ), ti.kinds(cmds)
    assert new.state is TurnState.INTERRUPTED and new.active_generate is None and new.tool is None
    assert new.turn_id == "t1" and new.turn_seq == 1 and new.history == ()
    assert getattr(new, stop_field()), "A0-3: the turn is marked as stopped"


def test_speaking_stop_tuple_repair_live() -> None:
    """A0-3, the repair live (SPEAKING, pass 1 drafted a repairable ``c1``): the repair is aborted, ``c1`` is cancelled
    as ``stopped`` (never executed), the ``tool_repair`` timer is cancelled."""
    m = machine()
    s, t = speaking(m)
    s, cmds = m.step(s, ti.llm_done(t + 10, "t1-gen", click("c1", "repairable")))
    assert s.active_generate == "t1-repair" and s.tool is not None, f"premise: R2 {ti.kinds(cmds)}"
    new, cmds = m.step(s, stop(t + 100, "t1"))
    assert tuple(cmds) == (
        C.LlmAbort(request_id="t1-repair"),
        C.TtsCancel(),
        C.PlaybackFlush(),
        stopped_cancel("c1"),
        C.CancelTimer(name="tool_repair"),
        arm_interrupt(t + 100),
        hud(TurnState.INTERRUPTED),
    ), ti.kinds(cmds)
    assert new.state is TurnState.INTERRUPTED and new.active_generate is None and new.tool is None


def test_speaking_stop_with_a_valid_call_pending() -> None:
    """A0-3, loop off, generation done with a valid ``c1`` pending until the drain (T-0089 R1): no abort (nothing is
    live), ``c1`` cancelled as ``stopped``; then the exit never executes it."""
    m = machine()
    s, t = speaking(m)
    s, _ = m.step(s, ti.llm_done(t + 10, "t1-gen", click("c1")))
    assert s.active_generate is None and s.tool is not None and s.state is TurnState.SPEAKING, (
        "premise"
    )
    new, cmds = m.step(s, stop(t + 100, "t1"))
    assert tuple(cmds) == (
        C.TtsCancel(),
        C.PlaybackFlush(),
        stopped_cancel("c1"),
        arm_interrupt(t + 100),
        hud(TurnState.INTERRUPTED),
    ), ti.kinds(cmds)
    end, cmds = m.step(new, E.PlaybackDrained(t_ms=t + 200, played_ms=300, played_text=ANSWER))
    assert not [c for c in cmds if isinstance(c, C.ToolExecute)], ti.kinds(cmds)
    end, cmds = m.step(end, E.Tick(t_ms=t + 100 + INTERRUPT_MS))
    assert end.state is TurnState.IDLE and not [c for c in cmds if isinstance(c, C.ToolExecute)]


def test_paused_stop_cancels_takeover_and_resume() -> None:
    """A0-3 in PAUSED, every timer of the row armed (hand-built: the repair live, the onset's ``takeover`` and a
    ``resume``): abort the repair, cancel speech, cancel ``c1``, then cancel ``takeover``, ``resume``, ``tool_repair``
    in that order, arm the interrupt timeout. ``llm_timeout`` is not one of the row's timers."""
    m = machine()
    s, t = speaking(m)
    s, _ = m.step(s, ti.llm_done(t + 10, "t1-gen", click("c1", "repairable")))
    s, cmds = m.step(s, E.UserSpeechOnset(t_ms=t + 50, onset_ms=t + 50))
    assert s.state is TurnState.PAUSED and C.ArmTimer(name="takeover", at_ms=t + 650) in cmds, (
        "premise"
    )
    s = s.model_copy(update={"timers": (*s.timers, Timer(name="resume", at_ms=t + 700))})
    new, cmds = m.step(s, stop(t + 100, "t1"))
    assert tuple(cmds) == (
        C.LlmAbort(request_id="t1-repair"),
        C.TtsCancel(),
        C.PlaybackFlush(),
        stopped_cancel("c1"),
        C.CancelTimer(name="takeover"),
        C.CancelTimer(name="resume"),
        C.CancelTimer(name="tool_repair"),
        arm_interrupt(t + 100),
        hud(TurnState.INTERRUPTED),
    ), ti.kinds(cmds)
    assert {x.name for x in new.timers} == {"interrupt_timeout"}, new.timers


def test_paused_stop_after_a_plain_onset() -> None:
    """A0-3 in PAUSED reached the ordinary way (an onset over speech, generate live): only ``takeover`` is armed."""
    m = machine()
    s, t = speaking(m)
    s, _ = m.step(s, E.UserSpeechOnset(t_ms=t + 50, onset_ms=t + 50))
    new, cmds = m.step(s, stop(t + 100, "t1"))
    assert tuple(cmds) == (
        C.LlmAbort(request_id="t1-gen"),
        C.TtsCancel(),
        C.PlaybackFlush(),
        C.CancelTimer(name="takeover"),
        arm_interrupt(t + 100),
        hud(TurnState.INTERRUPTED),
    ), ti.kinds(cmds)
    assert new.state is TurnState.INTERRUPTED


# =====================================================================================================================
# A0-4: the exits of a stopped INTERRUPTED
# =====================================================================================================================


def test_stopped_interrupted_exit_playback_stopped() -> None:
    """A0-4 on ``playback.stopped(400, "Yes.")``: truncate with the event's arguments, IDLE; history keeps the played
    prefix and the marker; every per-turn field is back at its default (the stop mark included); no new turn."""
    m = machine()
    s, t = speaking(m)
    s, _ = m.step(s, stop(t + 100, "t1"))
    new, cmds = m.step(s, E.PlaybackStopped(t_ms=t + 150, played_ms=400, played_text="Yes."))
    assert tuple(cmds) == (
        C.TruncateHistoryToPlayed(played_ms=400, played_text="Yes."),
        hud(TurnState.IDLE),
    ), ti.kinds(cmds)
    assert new == idle_after(s, (entry("Yes. [interrupted]"),)), new


def test_stopped_interrupted_exit_timeout() -> None:
    """A0-4 on the ``interrupt_timeout`` tick (``played_ms = None``, nothing reported played): the history entry is
    the bare marker. Guard width: a tick 1 ms before the deadline does nothing."""
    m = machine()
    s, t = speaking(m)
    s, _ = m.step(s, stop(t + 100, "t1"))
    inert(m, s, E.Tick(t_ms=t + 100 + INTERRUPT_MS - 1), "A0-4 before the deadline")
    new, cmds = m.step(s, E.Tick(t_ms=t + 100 + INTERRUPT_MS))
    assert tuple(cmds) == (
        C.TruncateHistoryToPlayed(played_ms=None, played_text=""),
        hud(TurnState.IDLE),
    ), ti.kinds(cmds)
    assert new == idle_after(s, (entry("[interrupted]"),)), new


def test_stopped_exit_joins_the_earlier_runs() -> None:
    """A0-4 with the loop on: step 1 played ``PREFIX``; step 2's report (no call) is voiced; the stop cuts it and the
    stopped run reports "Done." played: history holds both runs joined with the marker (T-0112 History)."""
    m = machine(tool_loop=True)
    s, _, t = loop_awaiting(m, prefix=True)
    s, _ = m.step(s, E.ToolResult(t_ms=t + 50, call_id="c1", status="executed"))
    s, _ = m.step(s, ti.commit(s, REPORT, t=t + 100))
    s, cmds = m.step(s, E.LlmDone(t_ms=t + 110, request_id="t1-gen-s2"))
    assert s.state is TurnState.SPEAKING and s.active_generate is None, (
        f"premise: R15 voiced {ti.kinds(cmds)}"
    )
    s, cmds = m.step(s, stop(t + 200, "t1"))
    assert tuple(cmds) == (
        C.TtsCancel(),
        C.PlaybackFlush(),
        arm_interrupt(t + 200),
        hud(TurnState.INTERRUPTED),
    ), ti.kinds(cmds)
    new, cmds = m.step(s, E.PlaybackStopped(t_ms=t + 250, played_ms=200, played_text=REPORT))
    assert tuple(cmds) == (
        C.TruncateHistoryToPlayed(played_ms=200, played_text=REPORT),
        hud(TurnState.IDLE),
    ), ti.kinds(cmds)
    assert new == idle_after(s, (entry(f"{PREFIX} {REPORT} [interrupted]"),)), new


def _voice_interrupts(m: Any, s: MachineState, t: int) -> dict[str, list[Any]]:
    """Four ways a voice barge-in reaches INTERRUPTED from SPEAKING (each list ends in INTERRUPTED)."""
    onset = E.UserSpeechOnset(t_ms=t + 50, onset_ms=t + 50)
    return {
        "takeover": [E.UserTakeover(t_ms=t + 100, onset_ms=t + 50)],
        "long-offset": [onset, E.UserSpeechOffset(t_ms=t + 800, offset_ms=t + 800, speech_ms=750)],
        "takeover-tick": [onset, E.Tick(t_ms=t + 650)],
        "word-in-pause": [
            onset,
            E.SttWord(t_ms=t + 500, text="wait", start_ms=t + 60, end_ms=t + 300),
        ],
    }


@pytest.mark.parametrize("path", ["takeover", "long-offset", "takeover-tick", "word-in-pause"])
@pytest.mark.parametrize("exit_", ["stopped", "timeout"])
@pytest.mark.parametrize(
    "after_a_stop",
    [pytest.param(False, id="fresh"), pytest.param(True, id="after-a-stopped-turn")],
)
def test_voice_interrupt_still_relistens(path: str, exit_: str, after_a_stop: bool) -> None:
    """A0-4 (negative): an INTERRUPTED without the stop mark still relistens, byte-equal to BASE's machine (commands
    exactly, every BASE field): LISTENING, a new turn, ``prefill_open``. ``after_a_stopped_turn``: t1 was stopped and
    exited first, so a stop mark that survives its turn (into t2) would send t2's voice barge-in to IDLE. The
    ``fresh`` cells are a guard (green at BASE, where the live machine is the frozen one, and after A0)."""
    m, old = machine(), frozen_machine()
    state: MachineState | None = None
    turn = "t1"
    if after_a_stop:
        s, t = speaking(m)
        s, _ = m.step(s, stop(t + 100, "t1"))
        state, _ = m.step(s, E.Tick(t_ms=t + 100 + INTERRUPT_MS))
        assert state.state is TurnState.IDLE and len(state.history) == 1, "premise: t1 was stopped"
        turn = "t2"
    s, t = speaking(m, at=10_000 if after_a_stop else 1000, state=state)
    assert s.turn_id == turn
    events = _voice_interrupts(m, s, t)[path]
    deadline = None
    for ev in events:
        new, cmds = m.step(s, ev)
        old_new, old_cmds = old.step(s, ev)
        assert tuple(cmds) == tuple(old_cmds) and base_fields(new) == base_fields(old_new), (
            path,
            ti.kinds(cmds),
        )
        s = new
        if any(isinstance(c, C.ArmTimer) and c.name == "interrupt_timeout" for c in cmds):
            deadline = next(
                c.at_ms for c in cmds if isinstance(c, C.ArmTimer) and c.name == "interrupt_timeout"
            )
    assert s.state is TurnState.INTERRUPTED and deadline is not None, f"premise: {path} interrupted"
    if exit_ == "stopped":
        ev: Any = E.PlaybackStopped(t_ms=deadline - 100, played_ms=300, played_text="Yes.")
    else:
        ev = E.Tick(t_ms=deadline)
    new, cmds = m.step(s, ev)
    old_new, old_cmds = old.step(s, ev)
    assert tuple(cmds) == tuple(old_cmds), f"A0-4: {ti.kinds(cmds)} != BASE {ti.kinds(old_cmds)}"
    assert base_fields(new) == base_fields(old_new), (
        "A0-4: the voice exit's state differs from BASE"
    )
    assert new.state is TurnState.LISTENING and new.turn_id != turn, new.state
    assert any(isinstance(c, C.PrefillOpen) for c in cmds), ti.kinds(cmds)


# =====================================================================================================================
# A0-5: THINKING
# =====================================================================================================================


def test_thinking_step1_stop() -> None:
    """A0-5, step 1, the generate live, no clip: abort, IDLE; history ``(USER, "[interrupted]")`` (H-T193-1)."""
    m = machine()
    s, t0 = thinking(m)
    new, cmds = m.step(s, stop(t0 + 100, "t1"))
    assert tuple(cmds) == (C.LlmAbort(request_id="t1-gen"), hud(TurnState.IDLE)), ti.kinds(cmds)
    assert new == idle_after(s, (entry("[interrupted]"),)), new


def test_thinking_stop_stores_played_prior_only() -> None:
    """A0-5's history is ``played_prior`` + the marker. A playback terminal delivered in THINKING (BASE records its
    ``played_text`` there; here t1's run reporting late into t2) is not text t2 played: the stop stores
    ``(USER, "[interrupted]")``, not that text. (Reading: A0-5 names ``played_prior``, not R17's ``_played``.)"""
    m = machine()
    s, t = speaking(m)
    s, _ = m.step(s, E.LlmDone(t_ms=t + 10, request_id="t1-gen"))
    s, _ = m.step(s, E.PlaybackDrained(t_ms=t + 600, played_ms=500, played_text=ANSWER))
    s, t0 = thinking(m, at=10_000, state=s)
    s, cmds = m.step(s, E.PlaybackStopped(t_ms=t0 + 50, played_ms=500, played_text="Yes."))
    assert cmds == () and s.state is TurnState.THINKING and s.played_text == "Yes.", (
        "premise: BASE records it"
    )
    new, cmds = m.step(s, stop(t0 + 100, "t2"))
    assert tuple(cmds) == (C.LlmAbort(request_id="t2-gen"), hud(TurnState.IDLE, "t2")), ti.kinds(
        cmds
    )
    assert new.history[-1] == entry("[interrupted]"), new.history


def test_thinking_ack_clip_stop() -> None:
    """A0-5 with the ack clip playing (``ack_clip`` on, the T-0182 B1 machine): ``stop_clip`` after the abort."""
    m = machine(ack_clip=True)
    s, t0 = thinking(m)
    assert s.clip_playing, "premise: the ack plays"
    new, cmds = m.step(s, stop(t0 + 100, "t1"))
    assert tuple(cmds) == (C.LlmAbort(request_id="t1-gen"), C.StopClip(), hud(TurnState.IDLE)), (
        ti.kinds(cmds)
    )
    assert new == idle_after(s, (entry("[interrupted]"),)), new


def test_thinking_awaiting_result_stop() -> None:
    """A0-5 while a result is awaited (loop on, ``PREFIX`` played, c1 executed): nothing is live and no call is
    pending, so the tuple is ``hud.state(IDLE)`` alone; history keeps the played run with the marker."""
    m = machine(tool_loop=True)
    s, _, t = loop_awaiting(m, prefix=True)
    new, cmds = m.step(s, stop(t + 100, "t1"))
    assert tuple(cmds) == (hud(TurnState.IDLE),), ti.kinds(cmds)
    assert new == idle_after(s, (entry(f"{PREFIX} [interrupted]"),)), new


def test_thinking_step2_stop() -> None:
    """A0-5 at step 2 (``STEP2`` committed, withheld by R14): abort ``t1-gen-s2``; the withheld text is never stored."""
    m = machine(tool_loop=True)
    s, t = loop_step2(m)
    new, cmds = m.step(s, stop(t + 100, "t1"))
    assert tuple(cmds) == (C.LlmAbort(request_id="t1-gen-s2"), hud(TurnState.IDLE)), ti.kinds(cmds)
    assert new == idle_after(s, (entry(f"{PREFIX} [interrupted]"),)), new


def test_thinking_repair_stop() -> None:
    """A0-5 with the repair live (loop off, ack on so a clip plays too): abort the repair, cancel ``c1`` as
    ``stopped``, stop the clip, in that order."""
    m = machine(ack_clip=True)
    s, t0 = thinking(m)
    s, cmds = m.step(s, ti.llm_done(t0 + 100, "t1-gen", click("c1", "repairable")))
    assert s.active_generate == "t1-repair" and s.clip_playing, f"premise: R2 {ti.kinds(cmds)}"
    new, cmds = m.step(s, stop(t0 + 200, "t1"))
    assert tuple(cmds) == (
        C.LlmAbort(request_id="t1-repair"),
        stopped_cancel("c1"),
        C.StopClip(),
        hud(TurnState.IDLE),
    ), ti.kinds(cmds)
    assert new == idle_after(s, (entry("[interrupted]"),)), new


def test_thinking_still_working_clip_stop() -> None:
    """A0-5 with T-0182's ``still_working`` playing (loop on, 5 s): ``stop_clip``; a later tick plays nothing."""
    m = machine(tool_loop=True, still_working_after_ms=5_000)
    s, t0, _t = loop_awaiting(m, prefix=False)
    s, cmds = m.step(s, E.Tick(t_ms=t0 + 5_000))
    assert cmds == (C.PlayClip(kind="still_working"),) and s.clip_playing, (
        f"premise: {ti.kinds(cmds)}"
    )
    new, cmds = m.step(s, stop(t0 + 5_100, "t1"))
    assert tuple(cmds) == (C.StopClip(), hud(TurnState.IDLE)), ti.kinds(cmds)
    assert new == idle_after(s, (entry("[interrupted]"),)), new
    for later in (t0 + 9_000, t0 + 30_000):
        inert(m, new, E.Tick(t_ms=later), "A0-6: no still_working after the stop")


# =====================================================================================================================
# A0-6 / A0-7: after a stop (adversarial)
# =====================================================================================================================


def _no_effect(m: Any, s: MachineState, ev: Any, why: str) -> MachineState:
    """A late terminal in INTERRUPTED/IDLE: no command, no history change, no state or turn change."""
    new, cmds = m.step(s, ev)
    assert tuple(cmds) == (), f"{why}: {ev.type} emitted {ti.kinds(cmds)}"
    assert (new.state, new.turn_id, new.turn_seq, new.history) == (
        s.state,
        s.turn_id,
        s.turn_seq,
        s.history,
    ), why
    return new


def test_late_events_after_a_speaking_stop_are_inert() -> None:
    """A0-6 on the SPEAKING path: after the stop, the aborted request's ``llm.commit`` (answer text) and ``llm.done``
    (with a valid call) emit nothing; a ``playback.drained`` in INTERRUPTED neither completes nor speaks; the timeout
    exit stores one entry; a late ``playback.stopped`` after it adds no second entry; ticks do nothing; ``turn_seq``
    moves only when the next onset opens t2. A0-7: one ``llm.abort`` for ``t1-gen`` over the whole trace."""
    m = machine()
    s, t = speaking(m)
    trace: list[Any] = []
    s, cmds = m.step(s, stop(t + 100, "t1"))
    trace += cmds
    inert(
        m,
        s,
        E.LlmCommit(
            t_ms=t + 110, request_id="t1-gen", canvas_idx=1, segment="answer", text="More. "
        ),
        "A0-6",
    )
    inert(m, s, ti.llm_done(t + 120, "t1-gen", click("c9")), "A0-6")
    s = _no_effect(
        m, s, E.PlaybackDrained(t_ms=t + 130, played_ms=200, played_text="Yes."), "A0-6 drained"
    )
    s, cmds = m.step(s, E.Tick(t_ms=t + 100 + INTERRUPT_MS))
    trace += cmds
    assert s.state is TurnState.IDLE and s.history == (entry("[interrupted]"),), s.history
    s = _no_effect(
        m,
        s,
        E.PlaybackStopped(t_ms=t + 500, played_ms=400, played_text="Yes."),
        "A0-6 late stopped",
    )
    s = _no_effect(
        m,
        s,
        E.PlaybackDrained(t_ms=t + 520, played_ms=400, played_text="Yes."),
        "A0-6 late drained",
    )
    for later in (t + 1_000, t + 25_000):
        new, cmds = m.step(s, E.Tick(t_ms=later))
        assert cmds == () and new.state is TurnState.IDLE and new.turn_seq == 1
        s = new
    s, cmds = m.step(s, E.UserSpeechOnset(t_ms=t + 30_000, onset_ms=t + 30_000))
    assert s.turn_id == "t2" and s.turn_seq == 2 and s.state is TurnState.LISTENING
    aborts = [c.request_id for c in trace if isinstance(c, C.LlmAbort)]
    assert aborts == ["t1-gen"], f"A0-7: one abort per live request; {aborts}"
    for c in trace:
        assert c.type not in NEVER_AFTER_STOP, f"A0-4: {c.type} after the stop"


def test_late_events_after_a_thinking_stop_are_inert() -> None:
    """A0-6 on the loop path: stopped at step 2; then the aborted step's ``llm.commit`` and ``llm.done(valid c2)``, a
    ``tool.result`` for c1 (already answered) and a hand-back for it, a tick past every deadline: each ``(s, ())``.
    And stopped while c1 is awaited: its ``tool.result`` (R13) and the R18 deadline tick move nothing (no
    ``llm.continue``, no ``hud.error``)."""
    m = machine(tool_loop=True, still_working_after_ms=5_000)
    s, t = loop_step2(m)
    s, _ = m.step(s, stop(t + 100, "t1"))
    late = [
        E.LlmCommit(
            t_ms=t + 110, request_id="t1-gen-s2", canvas_idx=2, segment="answer", text="Buying. "
        ),
        ti.llm_done(t + 120, "t1-gen-s2", click("c2")),
        E.ToolResult(t_ms=t + 130, call_id="c1", status="executed"),
        E.ToolResult(t_ms=t + 140, call_id="c1", status="executed", hand_back="checkout"),
        E.Tick(t_ms=t + 60_000),
    ]
    for ev in late:
        inert(m, s, ev, "A0-6 (step 2)")
    m2 = machine(tool_loop=True)
    s, _t0, t = loop_awaiting(m2, prefix=True)
    s, _ = m2.step(s, stop(t + 100, "t1"))
    for ev in (
        E.ToolResult(t_ms=t + 200, call_id="c1", status="executed"),
        E.Tick(t_ms=t + 20_000),
        E.LlmDone(t_ms=t + 20_100, request_id="t1-gen"),
    ):
        inert(m2, s, ev, "A0-6 (awaiting)")


@pytest.mark.parametrize("where", ["thinking", "speaking"])
def test_double_stop(where: str) -> None:
    """A0-2/A0-7: a second stop for the same turn (INTERRUPTED, then IDLE) does nothing; over the trace there is one
    abort, one ``tts.cancel`` and one ``playback.flush`` at most, and one history entry."""
    m = machine()
    if where == "thinking":
        s, t = thinking(m)
    else:
        s, t = speaking(m)
    trace: list[Any] = []
    s, cmds = m.step(s, stop(t + 100, "t1"))
    trace += cmds
    if where == "speaking":
        inert(m, s, stop(t + 150, "t1"), "A0-2: a second stop in INTERRUPTED")
        s, cmds = m.step(s, E.Tick(t_ms=t + 100 + INTERRUPT_MS))
        trace += cmds
    assert s.state is TurnState.IDLE
    inert(m, s, stop(t + 1_000, "t1"), "A0-2: a stop after the turn ended")
    k = ti.kinds(trace)
    assert k.count("llm.abort") == 1, k
    assert k.count("tts.cancel") == (1 if where == "speaking" else 0), k
    assert k.count("playback.flush") == (1 if where == "speaking" else 0), k
    assert len(s.history) == 1, s.history


def test_a_stop_in_idle_changes_nothing_downstream() -> None:
    """A0-2 (negative): a stop delivered in IDLE leaves the next turn exactly as on a machine that never saw it."""
    m = machine()
    plain, _t = speaking(m)
    s, _ = m.step(MachineState(), stop(500, "t0"))
    assert s == MachineState()
    stopped_first, _ = speaking(m, state=s)
    assert stopped_first == plain


# =====================================================================================================================
# The pin moves (A0-9): the machine and the contract files change only by A0's named additions
# =====================================================================================================================

#: T-0193 A0's literal base: main fdb9fc01 ("T-0193 planned"), where group A0 branched. Never a moving ref.
T0193_BASE = "fdb9fc01a829708aefab1ad81be894cd65cbeedd"
T0193_MACHINE = "local_astra/conductor/turn_machine.py"
T0193_EVENTS = "local_astra/contracts/events.py"
T0193_COMMANDS = "local_astra/contracts/commands.py"
T0193_TURN = "local_astra/contracts/turn.py"
#: sha256 of each file at T0193_BASE (``git show fdb9fc01:<path> | sha256sum``, 2026-10-29). Each is the file's earlier
#: pin: the machine, events.py and commands.py are T-0192 B's tree ends at 97409910
#: (``test_t0192_hand_back_machine.T0192_MOVED_SHA256``), turn.py T-0182 B's b8d26ae7 content.
T0193_BASE_SHA256 = {
    T0193_MACHINE: "305060235e23ff46223688ddbf5e331c1c108e0390c984ec9d8f47c0dcd58b40",
    T0193_EVENTS: "02016d69192d601fa37b18c4f60559b85ab910a77caf21c061eaee672b1e2ddf",
    T0193_COMMANDS: "658cd09344999ac97d726719a87077e6930757a967bc2763feb27aba917e679a",
    T0193_TURN: "9a8f98131bd905f856a36ab8a972ad1f89b6d4f4a7da3ef15558a291da2e3d58",
}
#: A0-R5, the tree end: T-0193 group A0's final implementation commit (373df531, "UserStop event and rows"), a literal
#: sha, never a moving ref.
T0193_TIP = "373df531eacaea792ff20c785f276d093df8fd75"
#: sha256 of each file's merged blob at T0193_TIP (``git show 373df531:<path> | sha256sum``, 2026-10-29). The census
#: pins (test_session_loop's ``TURN_MACHINE_SHA256``, test_session_tool_loop, test_session_engine_empty_retry,
#: test_t0181_mic_states) hold the working tree to these bytes, and the base ends to ``T0193_BASE_SHA256`` (both ends
#: pinned). ``t0193_drift`` stays as the check of the A0 delta's shape.
T0193_MOVED_SHA256 = {
    T0193_MACHINE: "9944ee489b89bcb15720c5d130c735bfb1168145f5b730dc72b2a9433b41a266",
    T0193_EVENTS: "1812fb3f191bff27d1461f92f7ecf294a1d7718a41003ccc9b333d128d789c34",
    T0193_COMMANDS: "307a016fa2e3b10fe47fa0d41e6e3b24cf93e1d0ca34f1da47a5a86e2dd2dfaf",
    T0193_TURN: "6b51c0869077cc79edc7e8471b20702373b786f20cecf1124fe22a76b8e7ecd2",
}
_REPO = pathlib.Path(__file__).resolve().parents[2]
#: the frozen copy's body (``git show fdb9fc01:local_astra/conductor/turn_machine.py | sed '1,17d' | sha256sum``) and
#: the machine's lines 1-17 (``| head -17 | sha256sum``), so the machine's base end needs no git
T0193_FROZEN_BODY_SHA256 = "9625f9545629f035e2f04bd961a953925a3a5f313f4714cbf7e3e892fdcd09d3"
T0193_MACHINE_HEAD_SHA256 = "3627faf45fe36a8cc943b4b8729a64230d28fac3b52ecd5ad6933e6956dd9e82"
T0193_MACHINE_HEAD_LINES = 17


def _frozen_body() -> str:
    from tests.unit import _frozen_turn_machine_t0193 as frozen

    text = pathlib.Path(frozen.__file__).read_text(encoding="utf-8")
    start, end = "dropped) ---\n", "# --- end of verbatim copy ---\n"
    body = text[text.index(start) + len(start) : text.index(end)].rstrip("\n") + "\n"
    assert hashlib.sha256(body.encode()).hexdigest() == T0193_FROZEN_BODY_SHA256, (
        "premise: _frozen_turn_machine_t0193 is fdb9fc01's machine verbatim"
    )
    return body


def t0193_end_sha256(path: str, ref: str | None = None) -> str:
    """sha256 of ``path`` in the working tree (``ref`` None) or of its blob at the literal commit ``ref`` (read with
    git; skips, never passes, when the commit is absent)."""
    if ref is None:
        return hashlib.sha256((_REPO / path).read_bytes()).hexdigest()
    gr.require_commits(ref)
    blob = subprocess.run(
        ["git", "-C", str(_REPO), "show", f"{ref}:{path}"], capture_output=True, check=True
    ).stdout
    return hashlib.sha256(blob).hexdigest()


def t0193_base_blob(path: str) -> str:
    """``path`` at T0193_BASE, checked against its earlier pin; the machine's body (line 18 on) comes from the frozen
    copy (no git); the contract files are read with git (skips, never passes, when the commit is absent)."""
    if path == T0193_MACHINE:
        return _frozen_body()
    gr.require_commits(T0193_BASE)
    blob = subprocess.run(
        ["git", "-C", str(_REPO), "show", f"{T0193_BASE}:{path}"], capture_output=True, check=True
    ).stdout
    assert hashlib.sha256(blob).hexdigest() == T0193_BASE_SHA256[path], (
        f"T-0193 A0 pin move: {path} at A0's base {T0193_BASE[:8]} is not its earlier pinned content"
    )
    return blob.decode("utf-8")


def _named(node: ast.stmt) -> str | None:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    first = body[0] if body else None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return body[1:]
    return body


def _shell(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """The statement without its body (name, bases, decorators, arguments, return annotation)."""
    bare = copy.copy(node)
    bare.body = []
    return ast.dump(bare)


def _drift(
    base: list[ast.stmt],
    now: list[ast.stmt],
    where: str,
    *,
    moved: dict[str, Any],
    may_add: dict[type, set[str] | None],
    docstring_free: bool = False,
) -> list[str]:
    """Every way ``now`` differs from ``base`` beyond the names in ``moved`` (each checked by its rule) and new named
    statements allowed by ``may_add`` (statement type -> allowed names, ``None`` for any name): BASE's named statements
    all present, in BASE's order, each with an identical AST; imports unchanged but for added names; every unnamed
    statement identical (the leading docstring ignored when ``docstring_free``)."""
    if docstring_free:
        base, now = _without_docstring(base), _without_docstring(now)
    drift: list[str] = []
    base_order = [k for k in map(_named, base) if k is not None]
    now_order = [k for k in map(_named, now) if k is not None]
    assert len(set(base_order)) == len(base_order), f"premise: {where} binds each BASE name once"
    if len(set(now_order)) != len(now_order):
        drift.append(f"{where}: a name is bound twice")
    base_named = {_named(n): n for n in base if _named(n) is not None}
    now_named = {_named(n): n for n in now if _named(n) is not None}
    missing = [k for k in base_order if k not in now_named]
    if missing:
        drift.append(f"{where}: BASE's {missing} removed")
    if [k for k in now_order if k in base_named] != [k for k in base_order if k in now_named]:
        drift.append(f"{where}: BASE's named statements reordered")
    for name, node in base_named.items():
        if name not in now_named:
            continue
        if name in moved:
            drift.extend(moved[name](node, now_named[name]))
        elif ast.dump(now_named[name]) != ast.dump(node):
            drift.append(f"{where}.{name}: changed")
    for name in now_order:
        if name in base_named:
            continue
        kind = type(now_named[name])
        allowed = next((names for t, names in may_add.items() if issubclass(kind, t)), set())
        if allowed is not None and name not in allowed:
            drift.append(f"{where}.{name}: a new {kind.__name__} is not one of A0's additions")
    imports = (ast.Import, ast.ImportFrom)
    base_imports = [n for n in base if isinstance(n, imports)]
    now_imports = [n for n in now if isinstance(n, imports)]
    if len(now_imports) != len(base_imports):
        drift.append(f"{where}: the import statements changed in number")
    for b, n in zip(base_imports, now_imports, strict=False):
        if getattr(b, "module", None) != getattr(n, "module", None) or not {
            a.name for a in b.names
        } <= {a.name for a in n.names}:
            drift.append(f"{where}: an import lost names or moved: {ast.dump(b)}")
    base_rest = [ast.dump(n) for n in base if _named(n) is None and not isinstance(n, imports)]
    now_rest = [ast.dump(n) for n in now if _named(n) is None and not isinstance(n, imports)]
    if base_rest != now_rest:
        drift.append(f"{where}: an unnamed statement (docstring, expression, ...) changed")
    return drift


# --- the machine: the UserStop rows, the stopped-INTERRUPTED exit, the per-turn stop mark ------------------------

#: methods whose body A0 may change: ``_dispatch`` must (the UserStop branch); the exit lives in one or more of
#: ``_relisten``, ``_on_playback_stopped``, ``_on_tick``; ``_interrupt`` may take the cancel reason in (implementation
#: note "reuse ``_interrupt``, passing the cancel reason in")
_MAY_CHANGE = {"_dispatch", "_interrupt", "_relisten", "_on_playback_stopped", "_on_tick"}
_EXIT_METHODS = ("_relisten", "_on_playback_stopped", "_on_tick")
#: methods whose signature may gain keyword parameters (with defaults)
_MAY_WIDEN = {"_interrupt", "_relisten"}


def _widened_only(base: ast.FunctionDef, now: ast.FunctionDef) -> bool:
    b, n = base.args, now.args
    same_positional = [a.arg for a in b.posonlyargs + b.args] == [
        a.arg for a in n.posonlyargs + n.args
    ]
    bkw = {a.arg: ast.dump(a) for a in b.kwonlyargs}
    nkw = {a.arg: ast.dump(a) for a in n.kwonlyargs}
    kept = all(nkw.get(k) == v for k, v in bkw.items())
    new_have_defaults = all(
        d is not None for a, d in zip(n.kwonlyargs, n.kw_defaults, strict=True) if a.arg not in bkw
    )
    rest = [ast.dump(x) for x in (*base.decorator_list, base.returns)] == [
        ast.dump(x) for x in (*now.decorator_list, now.returns)
    ]
    return (
        same_positional
        and kept
        and new_have_defaults
        and rest
        and b.vararg is None
        and n.vararg is None
    )


def _method_rule(name: str) -> Any:
    def rule(base: ast.stmt, now: ast.stmt) -> list[str]:
        assert isinstance(base, ast.FunctionDef)
        if not isinstance(now, ast.FunctionDef):
            return [f"TurnMachine.{name}: no longer a method"]
        if _shell(now) != _shell(base) and not (name in _MAY_WIDEN and _widened_only(base, now)):
            return [f"TurnMachine.{name}: its signature changed"]
        if name == "_dispatch" and ast.dump(
            ast.Module(_without_docstring(now.body), [])
        ) == ast.dump(ast.Module(_without_docstring(base.body), [])):
            return ["TurnMachine._dispatch: the UserStop branch is missing (its code is BASE's)"]
        return []

    return rule


def _turn_machine_class(base: ast.stmt, now: ast.stmt) -> list[str]:
    assert isinstance(base, ast.ClassDef)
    if not isinstance(now, ast.ClassDef) or _shell(now) != _shell(base):
        return ["TurnMachine: its class statement (bases, decorators) changed"]
    drift = _drift(
        base.body,
        now.body,
        "TurnMachine",
        moved={name: _method_rule(name) for name in _MAY_CHANGE},
        may_add={ast.FunctionDef: None},
    )
    base_methods = {n.name: ast.dump(n) for n in base.body if isinstance(n, ast.FunctionDef)}
    now_methods = {n.name: ast.dump(n) for n in now.body if isinstance(n, ast.FunctionDef)}
    if all(now_methods.get(name) == base_methods[name] for name in _EXIT_METHODS):
        drift.append(
            "TurnMachine: the stopped-INTERRUPTED exit is missing (_relisten, _on_playback_stopped, _on_tick)"
        )
    return drift


def _per_turn_reset_gains_one_key(base: ast.stmt, now: ast.stmt) -> list[str]:
    """A0: ``_PER_TURN_RESET`` is BASE's dict with exactly one key added (the stop mark), every BASE entry kept."""
    assert isinstance(base, ast.AnnAssign) and isinstance(base.value, ast.Dict)
    if not isinstance(now, ast.AnnAssign) or not isinstance(now.value, ast.Dict):
        return ["_PER_TURN_RESET: no longer an annotated dict literal"]
    if ast.dump(now.annotation) != ast.dump(base.annotation):
        return ["_PER_TURN_RESET: its annotation changed"]
    was = [
        (ast.dump(k), ast.dump(v)) for k, v in zip(base.value.keys, base.value.values, strict=True)
    ]  # type: ignore[arg-type]
    got = [
        (ast.dump(k), ast.dump(v)) for k, v in zip(now.value.keys, now.value.values, strict=True)
    ]  # type: ignore[arg-type]
    extra = [pair for pair in got if pair not in was]
    if [pair for pair in got if pair in was] != was or len(extra) != 1 or len(got) != len(was) + 1:
        return [
            f"_PER_TURN_RESET: not BASE's {len(was)} entries plus exactly one ({len(got)} entries)"
        ]
    return []


# --- the contract files ---------------------------------------------------------------------------------------------


def _union_names(node: ast.expr) -> list[str] | None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = _union_names(node.left), _union_names(node.right)
        return None if left is None or right is None else [*left, *right]
    return [node.id] if isinstance(node, ast.Name) else None


def _event_union_gains_user_stop(base: ast.stmt, now: ast.stmt) -> list[str]:
    """A0-1: ``Event = Annotated[<union>, Field(...)]`` gains exactly ``UserStop`` in the union."""

    def parts(node: ast.stmt) -> tuple[list[str] | None, list[str]] | None:
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript)):
            return None
        sl = node.value.slice
        if not isinstance(sl, ast.Tuple) or len(sl.elts) != 2:
            return None
        return _union_names(sl.elts[0]), [ast.dump(node.value.value), ast.dump(sl.elts[1])]

    b, n = parts(base), parts(now)
    assert b is not None and b[0] is not None, (
        "premise: BASE's Event is Annotated[<union>, Field(...)]"
    )
    if n is None or n[0] is None or n[1] != b[1]:
        return ["Event: no longer BASE's Annotated[<union>, Field(...)]"]
    names = n[0]
    if names.count("UserStop") != 1 or [x for x in names if x != "UserStop"] != b[0]:
        return [f"Event: the union is not BASE's plus exactly UserStop ({names})"]
    return []


def _event_types_gains_user_stop(base: ast.stmt, now: ast.stmt) -> list[str]:
    assert isinstance(base, ast.AnnAssign) and isinstance(base.value, ast.Tuple)
    if not isinstance(now, ast.AnnAssign) or not isinstance(now.value, ast.Tuple):
        return ["EVENT_TYPES: no longer an annotated tuple"]
    if ast.dump(now.annotation) != ast.dump(base.annotation):
        return ["EVENT_TYPES: its annotation changed"]
    was = [ast.dump(e) for e in base.value.elts]
    got = [ast.dump(e) for e in now.value.elts]
    stop_ = ast.dump(ast.Name(id="UserStop", ctx=ast.Load()))
    if got.count(stop_) != 1 or [x for x in got if x != stop_] != was:
        return ["EVENT_TYPES: not BASE's plus exactly UserStop"]
    return []


def _literal_members(node: ast.stmt) -> list[Any] | None:
    if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript)):
        return None
    if not (isinstance(node.value.value, ast.Name) and node.value.value.id == "Literal"):
        return None
    try:
        members = ast.literal_eval(node.value.slice)
    except ValueError:
        return None
    return list(members) if isinstance(members, tuple) else [members]


def _cancel_reason_gains_stopped(base: ast.stmt, now: ast.stmt) -> list[str]:
    was, got = _literal_members(base), _literal_members(now)
    assert was is not None, "premise: BASE's ToolCancelReason is a Literal"
    if got is None or got.count("stopped") != 1 or [x for x in got if x != "stopped"] != was:
        return [f"ToolCancelReason: {got!r} is not BASE's plus exactly 'stopped'"]
    return []


def _machine_state_gains_one_field(base: ast.stmt, now: ast.stmt) -> list[str]:
    """A0-1: ``MachineState`` is BASE's class plus exactly one annotated field (the stop mark), anywhere in its body;
    its docstring and every BASE statement unchanged."""
    assert isinstance(base, ast.ClassDef)
    if not isinstance(now, ast.ClassDef) or _shell(now) != _shell(base):
        return ["MachineState: its class statement changed"]
    was = [ast.dump(n) for n in base.body]
    got = [ast.dump(n) for n in now.body]
    added = [n for n in now.body if ast.dump(n) not in was]
    if (
        [x for x in got if x in was] != was
        or len(added) != 1
        or not isinstance(added[0], ast.AnnAssign)
    ):
        return [
            f"MachineState: not BASE's body plus exactly one annotated field ({len(was)} -> {len(got)})"
        ]
    return []


#: per file: (the top-level names that move, with their rule; the new top-level statements allowed; whether the module
#: docstring may change)
_T0193_RULES: dict[str, tuple[dict[str, Any], dict[type, set[str] | None], bool]] = {
    T0193_MACHINE: (
        {"TurnMachine": _turn_machine_class, "_PER_TURN_RESET": _per_turn_reset_gains_one_key},
        {ast.FunctionDef: None, ast.Assign: None, ast.AnnAssign: None},
        False,
    ),
    T0193_EVENTS: (
        {"Event": _event_union_gains_user_stop, "EVENT_TYPES": _event_types_gains_user_stop},
        {ast.ClassDef: {"UserStop"}},
        True,
    ),
    T0193_COMMANDS: ({"ToolCancelReason": _cancel_reason_gains_stopped}, {}, False),
    T0193_TURN: ({"MachineState": _machine_state_gains_one_field}, {}, False),
}


def _split_machine(src: str) -> tuple[str, str]:
    lines = src.splitlines(keepends=True)
    return "".join(lines[:T0193_MACHINE_HEAD_LINES]), "".join(lines[T0193_MACHINE_HEAD_LINES:])


def t0193_drift(path: str, now_src: str | None = None, base_src: str | None = None) -> list[str]:
    """Every way ``path`` (the working tree's, or ``now_src``) differs from its T0193_BASE blob (or ``base_src``)
    beyond A0's named additions; empty when only those changed. At BASE itself it names the missing additions."""
    if base_src is None:
        base_src = t0193_base_blob(path)
    if now_src is None:
        now_src = (_REPO / path).read_text(encoding="utf-8")
        if path == T0193_MACHINE:
            head, now_src = _split_machine(now_src)
            if hashlib.sha256(head.encode()).hexdigest() != T0193_MACHINE_HEAD_SHA256:
                return [
                    f"{path}: lines 1-{T0193_MACHINE_HEAD_LINES} (the module docstring) are not BASE's"
                ]
    moved, may_add, docstring_free = _T0193_RULES[path]
    return _drift(
        ast.parse(base_src).body,
        ast.parse(now_src).body,
        path,
        moved=moved,
        may_add=may_add,
        docstring_free=docstring_free,
    )


def _edit(src: str, old: str, new: str) -> str:
    assert src.count(old) == 1, f"premise: {old!r} occurs once in the BASE blob"
    return src.replace(old, new)


#: A0 as minimal edits of the BASE blobs (what an implementation could look like), for the checker's positive row only
_A0_MACHINE_EDITS = (
    (
        '    "still_working_played": False,\n}',
        '    "still_working_played": False,\n    "stopped": False,\n}',
    ),
    (
        "        if isinstance(event, E.ToolResult):\n",
        "        if isinstance(event, E.UserStop):\n            return state, []\n        if isinstance(event, E.ToolResult):\n",
    ),
    (
        "        if state.state is TurnState.INTERRUPTED:\n            return self._relisten(",
        "        if state.state is TurnState.INTERRUPTED and state.stopped:\n            return state, []\n"
        "        if state.state is TurnState.INTERRUPTED:\n            return self._relisten(",
    ),
)


def _a0_blob(path: str) -> str:
    src = t0193_base_blob(path)
    if path == T0193_MACHINE:
        for old, new in _A0_MACHINE_EDITS:
            src = _edit(src, old, new)
        return src.replace(
            "    def _interrupt(self, state: MachineState, *, t_ms: int) -> tuple[MachineState, list[Command]]:",
            '    def _interrupt(\n        self, state: MachineState, *, t_ms: int, reason: str = "interrupted"\n'
            "    ) -> tuple[MachineState, list[Command]]:",
        )
    if path == T0193_EVENTS:
        src = _edit(
            src,
            "class Tick(_Event):",
            'class UserStop(_Event):\n    type: Literal["user.stop"] = "user.stop"\n'
            "    turn_id: str\n\n\nclass Tick(_Event):",
        )
        src = _edit(
            src, "    | MicMuted\n    | Tick,", "    | MicMuted\n    | UserStop\n    | Tick,"
        )
        return _edit(
            src, "    MicMuted,\n    Tick,\n)", "    MicMuted,\n    UserStop,\n    Tick,\n)"
        )
    if path == T0193_COMMANDS:
        return _edit(src, '    "reset",\n]', '    "reset",\n    "stopped",\n]')
    return _edit(
        src,
        "    still_working_played: bool = Field(default=False, exclude=True)\n",
        "    still_working_played: bool = Field(default=False, exclude=True)\n"
        "    stopped: bool = Field(default=False, exclude=True)\n",
    )


@pytest.mark.parametrize("path", [T0193_MACHINE, T0193_EVENTS, T0193_COMMANDS, T0193_TURN])
def test_pin_move_tree_is_the_base_plus_exactly_a0(path: str) -> None:
    """A0-9: the working tree's file is its fdb9fc01 blob plus exactly A0's additions (red until A0 is implemented;
    the census pins in test_session_loop, test_session_tool_loop, test_session_engine_empty_retry and
    test_t0181_mic_states hold the same)."""
    drift = t0193_drift(path)
    assert drift == [], (
        f"T-0193 A0-9: {path} is not fdb9fc01's blob plus exactly A0's additions: {drift}"
    )


def test_pin_move_base_ends_are_the_base_blobs() -> None:
    """Provenance of the base ends: at fdb9fc01 each file hashes to its earlier pin, and the machine's blob is the head
    literal's 17 lines followed by the frozen body."""
    gr.require_commits(T0193_BASE)
    for path in (T0193_EVENTS, T0193_COMMANDS, T0193_TURN):
        t0193_base_blob(path)  # asserts the sha256
    blob = subprocess.run(
        ["git", "-C", str(_REPO), "show", f"{T0193_BASE}:{T0193_MACHINE}"],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    assert hashlib.sha256(blob.encode()).hexdigest() == T0193_BASE_SHA256[T0193_MACHINE]
    head, body = _split_machine(blob)
    assert hashlib.sha256(head.encode()).hexdigest() == T0193_MACHINE_HEAD_SHA256
    assert body.rstrip("\n") + "\n" == _frozen_body()


def test_pin_move_tree_ends_are_the_merged_blobs() -> None:
    """A0-R5 provenance of the tree ends: at 373df531 each file hashes to its ``T0193_MOVED_SHA256`` literal (a fixed
    blob against a fixed literal), each differs from its base end, and that merged blob is exactly the base blob plus
    A0's additions (``t0193_drift`` of the two fixed blobs is empty; the machine's lines 1-17 are BASE's)."""
    gr.require_commits(T0193_TIP, T0193_BASE)
    for path, sha in T0193_MOVED_SHA256.items():
        merged = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{T0193_TIP}:{path}"],
            capture_output=True,
            check=True,
        ).stdout
        assert hashlib.sha256(merged).hexdigest() == sha, (
            f"A0-R5: {path} at {T0193_TIP[:8]} is not its tree-end literal"
        )
        assert sha != T0193_BASE_SHA256[path], f"A0-R5: {path}'s tree end equals its base end"
        base = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{T0193_BASE}:{path}"],
            capture_output=True,
            check=True,
        ).stdout
        assert hashlib.sha256(base).hexdigest() == T0193_BASE_SHA256[path]
        merged_src, base_src = merged.decode("utf-8"), base.decode("utf-8")
        if path == T0193_MACHINE:
            head, merged_src = _split_machine(merged_src)
            assert hashlib.sha256(head.encode()).hexdigest() == T0193_MACHINE_HEAD_SHA256, (
                "A0-R5: the merged machine's lines 1-17 are not BASE's"
            )
            base_src = _split_machine(base_src)[1]
        drift = t0193_drift(path, merged_src, base_src)
        assert drift == [], f"A0-R5: {path} at {T0193_TIP[:8]} is not BASE plus exactly A0: {drift}"


def test_pin_move_checker_names_the_missing_additions_at_base() -> None:
    """At BASE each file's drift names the A0 addition it lacks (why the moved pins are red until A0 lands)."""
    for path, needles in (
        (
            T0193_MACHINE,
            (
                "_dispatch: the UserStop branch is missing",
                "stopped-INTERRUPTED exit is missing",
                "_PER_TURN_RESET",
            ),
        ),
        (T0193_EVENTS, ("Event: the union", "EVENT_TYPES: not BASE's plus exactly UserStop")),
        (T0193_COMMANDS, ("ToolCancelReason",)),
        (T0193_TURN, ("MachineState: not BASE's body plus exactly one",)),
    ):
        base = t0193_base_blob(path)
        drift = t0193_drift(path, base, base)
        assert len(drift) == len(needles), (path, drift)
        for needle in needles:
            assert any(needle in d for d in drift), (path, needle, drift)


def test_pin_move_checker_accepts_exactly_the_named_additions() -> None:
    """Positive: each BASE blob plus only A0's addition (and, in the machine, a new private helper) has no drift."""
    for path in (T0193_MACHINE, T0193_EVENTS, T0193_COMMANDS, T0193_TURN):
        assert t0193_drift(path, _a0_blob(path), t0193_base_blob(path)) == [], path
    machine_src = t0193_base_blob(T0193_MACHINE)
    helper = (
        _a0_blob(T0193_MACHINE) + "\n\ndef _stop_marker(played: str) -> str:\n    return played\n"
    )
    assert t0193_drift(T0193_MACHINE, helper, machine_src) == []
    method = _edit(
        _a0_blob(T0193_MACHINE),
        "    def _on_client_disconnected(",
        "    def _on_user_stop(self, state, ev):\n        return state, []\n\n    def _on_client_disconnected(",
    )
    assert t0193_drift(T0193_MACHINE, method, machine_src) == []
    events = t0193_base_blob(T0193_EVENTS)
    doc = _a0_blob(T0193_EVENTS).replace(
        "``client.disconnected``)", "``client.disconnected``, ``user.stop``)", 1
    )
    assert doc != _a0_blob(T0193_EVENTS) and t0193_drift(T0193_EVENTS, doc, events) == []


def test_pin_move_checker_refuses_every_other_edit() -> None:
    """Negative: with A0's additions in place, each further edit is named (another method's body, a new import, a new
    class, a removed name, a second reset key, a second event or field or reason, a changed signature)."""
    machine_src = t0193_base_blob(T0193_MACHINE)
    good = _a0_blob(T0193_MACHINE)
    for mutant, needle in (
        (
            _edit(good, 'new = new.model_copy(update={"step_text": ""})\n', "new = new\n"),
            "_finish_step: changed",
        ),
        (good + "\nimport time\n", "import statements changed in number"),
        (good + "\n\nclass Extra:\n    pass\n", "Extra: a new ClassDef"),
        (
            _edit(
                good, '    "stopped": False,\n}', '    "stopped": False,\n    "stop_t_ms": None,\n}'
            ),
            "_PER_TURN_RESET",
        ),
        (_edit(good, "_POLICY_REASONS = (", "_OTHER_REASONS = ("), "['_POLICY_REASONS'] removed"),
        (
            _edit(
                good,
                "self, state: MachineState, ev: E.PlaybackStopped\n",
                "self, state: MachineState, ev: E.Event\n",
            ),
            "_on_playback_stopped: its signature changed",
        ),
        (
            _edit(
                good,
                "    def _abort_live(self, state: MachineState)",
                "    def _abort_live(self, state)",
            ),
            "_abort_live",
        ),
    ):
        drift = t0193_drift(T0193_MACHINE, mutant, machine_src)
        assert any(needle in d for d in drift), (needle, drift)
    no_exit = _edit(
        good,
        "        if state.state is TurnState.INTERRUPTED and state.stopped:\n            return state, []\n",
        "",
    )
    assert any("exit is missing" in d for d in t0193_drift(T0193_MACHINE, no_exit, machine_src))
    events = t0193_base_blob(T0193_EVENTS)
    good_events = _a0_blob(T0193_EVENTS)
    for mutant, needle in (
        (
            good_events + "\n\nclass UserCancel(_Event):\n    type: Literal['x'] = 'x'\n",
            "UserCancel: a new ClassDef",
        ),
        (
            _edit(
                good_events,
                "    UserStop,\n    Tick,\n)",
                "    UserStop,\n    UserStop,\n    Tick,\n)",
            ),
            "EVENT_TYPES",
        ),
        (_edit(good_events, "    | UserStop\n    | Tick,", "    | Tick,"), "Event: the union"),
    ):
        drift = t0193_drift(T0193_EVENTS, mutant, events)
        assert any(needle in d for d in drift), (needle, drift)
    commands = t0193_base_blob(T0193_COMMANDS)
    for members in ('    "stopped",\n    "user",\n]', '    "halted",\n]'):
        mutant = _edit(_a0_blob(T0193_COMMANDS), '    "stopped",\n]', members)
        assert any(
            "ToolCancelReason" in d for d in t0193_drift(T0193_COMMANDS, mutant, commands)
        ), members
    turn = t0193_base_blob(T0193_TURN)
    two = _edit(
        _a0_blob(T0193_TURN),
        "    stopped: bool = Field(default=False, exclude=True)\n",
        "    stopped: bool = Field(default=False, exclude=True)\n    stop_t_ms: int | None = None\n",
    )
    assert any("MachineState" in d for d in t0193_drift(T0193_TURN, two, turn))
