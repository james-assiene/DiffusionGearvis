"""T-0181 group A0: the ``MicMuted`` row (docs/tasks/T-0181.md, "## Group A0 replan (2026-10-28)", A0-1..A0-7).

James ruled H-T181-2 (2026-10-28): muting mid-question drops the question ("just drop it"). The machine gains one event,
``MicMuted(t_ms)``, and one row: in LISTENING, and in ENDPOINTING while ``flush_t_ms is None`` (H-T181-8 default), the
open spoken question is abandoned (IDLE; only the live prefill is aborted; nothing is started, stored or summarised;
``words`` and ``next_words`` cleared). Everywhere else (IDLE, THINKING, SPEAKING, PAUSED, INTERRUPTED, and ENDPOINTING
after the flush, which is where a typed or ui turn lives) ``step(s, MicMuted)`` is ``(s, ())`` (A0-3; H-T181-9 default:
no row in PAUSED/INTERRUPTED).

Oracles, never the implementation:

- the expected abandon state is built from ``MachineState``'s own defaults plus the four fields that survive a turn
  (``turn_id``, ``turn_seq``, ``history``, ``first_canvas_ms``; ``degraded`` too), so "every per-turn field reset"
  is checked against the contract's defaults, not against ``_PER_TURN_RESET``;
- the expected commands are written by hand from A0-2 (``prefill_abort`` of the live prefill, then ``hud.state``);
- A0-6's differential steps ``tests/unit/_frozen_turn_machine_t0181.py`` (a verbatim copy of ``turn_machine.py`` at
  A0's base, main 021c2ef6) beside the live machine on traces with no ``MicMuted``.

Activated at the start of A0's implementation (no ``pending`` marker): the rows go red with an ``AssertionError``
naming A0-1 until ``contracts/events.py`` declares ``MicMuted`` (``mic_muted`` below). The differential and the frozen-copy rows are guards: green before the implementation on
purpose (the machine is BASE), and they must stay green after it.

Regime: CPU, pure, ``TurnPolicy(thinking=True)`` unless stated; Hypothesis rows ``derandomize=True``.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
from typing import Any

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import HistoryTurn, MachineState, TurnState, Word
from tests import _git_range as gr
from tests.unit import _frozen_turn_machine_t0181 as frozen
from tests.unit import test_turn_invariants as ti

#: A0's base, a literal (``.claude/rules/tests.md``): main 021c2ef6 ("T-0180: group A merged") when the branch merged
#: main for A0. ``turn_machine.py`` and ``contracts/events.py`` there are the post-T-0182-B files.
BASE_A0 = "021c2ef6d33a916727fbf83ed554a2c4a5fbeef2"
#: The A0 implementation tip. Started at BASE_A0 (no implementation yet); bumped by the test-author (A0-7) to the
#: implementation commit 9dbddf96 ("T-0181 A0: MicMuted event and the turn-machine abandon row"), closed at the merge.
TIP_A0 = "9dbddf9655631bc5e7d7ec1d97762377d0f237b2"
#: sha256 of the frozen copy's body (``git show 021c2ef6:local_astra/conductor/turn_machine.py | sed '1,17d' |
#: sha256sum``, computed by hand 2026-10-28)
FROZEN_BODY_SHA256 = "e4fc38eeb229fd436f31ec86d00d13a24fcda883d130264f9abf48b0031bde34"
#: A0-1 / A11: these contract files are byte-identical over BASE_A0..TIP_A0
UNCHANGED_A0 = (
    "local_astra/contracts/commands.py",
    "local_astra/contracts/turn.py",
    "local_astra/contracts/codec.py",
)
#: ``EVENT_TYPES`` class names at BASE_A0, copied by hand from ``git show 021c2ef6:local_astra/contracts/events.py``
#: (lines 278-305), in order
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
    "Tick",
)

POLICY = TurnPolicy(thinking=True)
SETTLE = POLICY.eot_settle_ms

#: A0-2: what the abandon row may never emit
FORBIDDEN = (
    "llm.start",
    "llm.continue",
    "commit_history",
    "schedule_summary",
    "tts.speak",
    "play_clip",
    "hud.error",
    "degrade",
    "reset_session",
)


def mic_muted(t_ms: int) -> Any:
    """``E.MicMuted(t_ms=...)``; its absence is A0-1's failed criterion, never an ``AttributeError``."""
    cls = getattr(E, "MicMuted", None)
    assert cls is not None, (
        'T-0181 A0-1: contracts/events.py gains MicMuted(_Event), type "mic.muted", no field beyond t_ms'
    )
    return cls(t_ms=t_ms)


def run(m: TurnMachine, events: list[Any], s: MachineState) -> tuple[MachineState, list[Any]]:
    out: list[Any] = []
    for ev in events:
        s, cmds = m.step(s, ev)
        out.extend(cmds)
    return s, out


def word(text: str, start_ms: int, *, delay: int = 400) -> E.SttWord:
    """A word on the audio clock, delivered ``delay`` ms after its start (STT lag)."""
    return E.SttWord(t_ms=start_ms + delay, text=text, start_ms=start_ms, end_ms=start_ms + 150)


def answered_turn(m: TurnMachine) -> MachineState:
    """t1 asked "what is this", answered "Ok.", drained: IDLE with one history entry and a first-canvas EMA."""
    s, _ = run(
        m,
        [
            E.UserSpeechOnset(t_ms=1000, onset_ms=1000),
            word("what", 1000),
            word("is", 1200),
            word("this", 1400),
            E.SttEndOfTurn(t_ms=1900, t_eot_ms=1800),
            E.SttFlushed(t_ms=2000, t_eot_ms=1800),
            E.Tick(t_ms=2000 + SETTLE),
        ],
        m.initial(),
    )
    assert s.state is TurnState.THINKING and s.active_generate == "t1-gen", (
        "premise: t1 is thinking"
    )
    s, _ = m.step(
        s,
        E.LlmCommit(t_ms=2500, request_id="t1-gen", canvas_idx=0, segment="answer", text="Ok."),
    )
    s, _ = run(
        m,
        [
            E.LlmDone(t_ms=2600, request_id="t1-gen"),
            E.PlaybackDrained(t_ms=3000, played_ms=400, played_text="Ok."),
        ],
        s,
    )
    assert s.state is TurnState.IDLE, "premise: t1 completed"
    assert s.history == (HistoryTurn(user_text="what is this", assistant_spoken="Ok."),)
    assert s.first_canvas_ms is not None and s.turn_seq == 1
    return s


def listening(m: TurnMachine) -> MachineState:
    """t2 open in LISTENING with two words heard ("drop me"), after the answered t1."""
    s, _ = run(
        m,
        [E.UserSpeechOnset(t_ms=5000, onset_ms=5000), word("drop", 5000), word("me", 5200)],
        answered_turn(m),
    )
    assert s.state is TurnState.LISTENING and s.turn_id == "t2", "premise: t2 is listening"
    assert s.active_prefill == "t2-prefill" and s.active_generate is None
    assert [w.text for w in s.words] == ["drop", "me"]
    return s


def endpointing_before_flush(m: TurnMachine) -> MachineState:
    """t2 in ENDPOINTING: the end-of-turn (t_eot 5400) arrived, its Marker echo has not."""
    s, _ = m.step(listening(m), E.SttEndOfTurn(t_ms=5700, t_eot_ms=5400))
    assert s.state is TurnState.ENDPOINTING and s.t_eot_ms == 5400 and s.flush_t_ms is None
    return s


def endpointing_after_flush(m: TurnMachine) -> MachineState:
    """t2 in ENDPOINTING after the flush (5800), inside the settle window."""
    s, _ = m.step(endpointing_before_flush(m), E.SttFlushed(t_ms=5800, t_eot_ms=5400))
    assert s.state is TurnState.ENDPOINTING and s.flush_t_ms == 5800
    return s


def abandoned(before: MachineState) -> MachineState:
    """A0-2's state after the abandon, from the contract's defaults: IDLE, every per-turn field and both word buffers
    at ``MachineState()``'s values; only what outlives a turn is carried over."""
    return MachineState(
        turn_id=before.turn_id,
        turn_seq=before.turn_seq,
        history=before.history,
        first_canvas_ms=before.first_canvas_ms,
        degraded=before.degraded,
    )


def abandon_commands(before: MachineState) -> tuple[Any, ...]:
    """A0-2's exact commands: the live prefill's abort (no generate is live in LISTENING/ENDPOINTING), then the
    automatic ``hud.state(IDLE)`` naming the abandoned turn."""
    assert before.active_generate is None and before.active_prefill is not None
    return (
        C.PrefillAbort(request_id=before.active_prefill),
        C.HudState(state=TurnState.IDLE.value, turn_id=before.turn_id),
    )


# =====================================================================================================================
# A0-1: the contract
# =====================================================================================================================


def test_mic_muted_contract() -> None:
    """A0-1: exactly one new event class ``MicMuted(_Event)``, ``type == "mic.muted"``, no field beyond ``t_ms``, in
    ``Event`` and ``EVENT_TYPES``; its docstring names the session clock and says there is no unmute event."""
    ev = mic_muted(42)
    cls = type(ev)
    assert issubclass(cls, E._Event)
    assert set(cls.model_fields) == {"t_ms", "type"}, (
        f"A0-1: no field beyond t_ms; got {set(cls.model_fields)}"
    )
    assert cls.model_fields["type"].default == "mic.muted"
    assert ev.t_ms == 42
    names = [c.__name__ for c in E.EVENT_TYPES]
    assert names.count("MicMuted") == 1, f"A0-1: MicMuted once in EVENT_TYPES; got {names}"
    # T-0193 A0-1 (census move): UserStop is the one later event class; it is set aside here and pinned by
    # test_t0193_stop_row.test_a0_1_user_stop_event
    assert names.count("UserStop") == 1, f"T-0193 A0-1: UserStop once in EVENT_TYPES; got {names}"
    assert [n for n in names if n not in ("MicMuted", "UserStop")] == list(BASE_EVENT_CLASSES), (
        "A0-1: EVENT_TYPES is BASE's plus MicMuted (and T-0193's UserStop), nothing else added, removed or reordered"
    )
    declared = sorted(
        name
        for name, obj in vars(E).items()
        if isinstance(obj, type) and issubclass(obj, E._Event) and obj is not E._Event
    )
    assert declared == sorted([*BASE_EVENT_CLASSES, "MicMuted", "UserStop"]), (
        f"A0-1: contracts/events.py gains exactly one class; it declares {declared}"
    )
    parsed = E.EventAdapter.validate_python({"type": "mic.muted", "t_ms": 7})
    assert isinstance(parsed, cls) and parsed.t_ms == 7, "A0-1: MicMuted is in the Event union"
    with pytest.raises(ValueError):
        E.EventAdapter.validate_python({"type": "mic.muted", "t_ms": 7, "muted": True})
    with pytest.raises(ValueError):
        cls(t_ms=-1)
    doc = (cls.__doc__ or "").lower()
    assert "session" in doc and "clock" in doc, (
        "A0-1: the docstring says t_ms is the session clock (P10)"
    )
    assert "unmute" in doc, "A0-1: the docstring says there is no unmute event"


def test_commands_turn_and_codec_unchanged_over_base_tip() -> None:
    """A0-1 / A11 (first bullet as replanned): ``contracts/commands.py``, ``contracts/turn.py`` and
    ``contracts/codec.py`` are byte-identical over BASE_A0..TIP_A0 (two literals; TIP_A0 is bumped by whoever commits
    the implementation). Power (A0-7 bump): over the same range the predicate sees A0's two edited files, so the
    range is not empty and an edit to any of the three would be seen."""
    gr.require_commits(BASE_A0, TIP_A0)
    assert gr.is_ancestor(BASE_A0, TIP_A0), f"{BASE_A0} is not an ancestor of {TIP_A0}"
    assert gr.changed(BASE_A0, TIP_A0, *UNCHANGED_A0) == [], (
        "A0-1/A11: A0 changes no command, state or codec"
    )
    edited = ("local_astra/conductor/turn_machine.py", "local_astra/contracts/events.py")
    assert gr.changed(BASE_A0, TIP_A0, *edited) == sorted(edited), (
        "power: the range predicate sees A0's turn_machine.py and events.py edits"
    )


# =====================================================================================================================
# A0-2: the abandon row
# =====================================================================================================================


def test_abandons_listening() -> None:
    """A0-2 in LISTENING: IDLE; commands exactly ``prefill_abort(t2-prefill)``, ``hud.state(IDLE, t2)``; the state is
    the contract's IDLE with t1's history, ``turn_seq`` and EMA kept; the next turn is t3 and starts empty."""
    m = TurnMachine(POLICY)
    s = listening(m)
    new, cmds = m.step(s, mic_muted(5300))
    assert tuple(cmds) == abandon_commands(s), f"A0-2: got {ti.kinds(cmds)}"
    assert new.state is TurnState.IDLE
    assert new.words == () and new.next_words == ()
    assert new.history == s.history and new.turn_seq == s.turn_seq == 2
    assert new.first_canvas_ms == s.first_canvas_ms
    assert new == abandoned(s), "A0-2: every per-turn field reset, nothing else moved"
    nxt, cmds = m.step(new, E.UserSpeechOnset(t_ms=6000, onset_ms=6000))
    assert nxt.turn_id == "t3" and nxt.state is TurnState.LISTENING, (
        "A0-2: the next turn gets a new id"
    )
    assert nxt.words == (), f"A0-2: the next turn starts with none of t2's words; got {nxt.words}"
    assert C.PrefillOpen(request_id="t3-prefill") in cmds


def test_abandons_endpointing_before_flush() -> None:
    """A0-2 in ENDPOINTING with ``flush_t_ms is None`` (H-T181-8): the same abandon as LISTENING; no ``unfreeze`` or
    anything else beside the prefill abort and the ``hud.state``."""
    m = TurnMachine(POLICY)
    s = endpointing_before_flush(m)
    new, cmds = m.step(s, mic_muted(5750))
    assert tuple(cmds) == abandon_commands(s), f"A0-2: got {ti.kinds(cmds)}"
    assert new == abandoned(s)
    assert new.t_eot_ms is None and new.flush_t_ms is None and new.onset_ms is None


def test_endpointing_with_words_after_t_eot_drops_next_words() -> None:
    """A0-2: a word heard after ``t_eot`` (audio 5600 > 5400) is buffered for the *next* turn (criterion 4); the
    abandon drops it too (``next_words == ()``), because it is the dropped question's tail."""
    m = TurnMachine(POLICY)
    s, _ = m.step(endpointing_before_flush(m), word("please", 5600))
    assert [w.text for w in s.next_words] == ["please"], "premise: the late word is next_words"
    assert [w.text for w in s.words] == ["drop", "me"]
    new, cmds = m.step(s, mic_muted(6100))
    assert tuple(cmds) == abandon_commands(s)
    assert new.words == () and new.next_words == (), "A0-2: next_words cleared by the abandon"
    assert new == abandoned(s)


@pytest.mark.parametrize("via", ["relisten", "takeover"])
def test_abandons_a_listening_turn_however_it_opened(via: str) -> None:
    """A0-2 is keyed on the state, not on how LISTENING was reached: a relisten after a barge-in (its history entry,
    ``[interrupted]``, was already stored) and a THINKING takeover (``keep_words``: the concatenated question) are
    abandoned like an onset's turn, with history as it stood."""
    m = TurnMachine(POLICY)
    s, _ = run(
        m,
        [
            E.UserSpeechOnset(t_ms=5000, onset_ms=5000),
            word("tell", 5000),
            E.SttEndOfTurn(t_ms=5500, t_eot_ms=5300),
            E.SttFlushed(t_ms=5600, t_eot_ms=5300),
            E.Tick(t_ms=5600 + SETTLE),
        ],
        answered_turn(m),
    )
    assert s.state is TurnState.THINKING, "premise"
    if via == "relisten":
        s, _ = m.step(
            s,
            E.LlmCommit(
                t_ms=6000, request_id="t2-gen", canvas_idx=0, segment="answer", text="Well. "
            ),
        )
        s, _ = run(
            m,
            [
                E.UserTakeover(t_ms=6300, onset_ms=6200),
                E.PlaybackStopped(t_ms=6400, played_ms=200, played_text="Well. "),
            ],
            s,
        )
    else:
        s, _ = m.step(s, E.UserTakeover(t_ms=6300, onset_ms=6200))
    assert s.state is TurnState.LISTENING and s.turn_id == "t3", f"premise ({via})"
    new, cmds = m.step(s, mic_muted(6500))
    assert tuple(cmds) == abandon_commands(s), f"A0-2 ({via}): got {ti.kinds(cmds)}"
    assert new == abandoned(s)


# =====================================================================================================================
# A0-3: no row elsewhere
# =====================================================================================================================


def _idle_fresh(m: TurnMachine) -> MachineState:
    return m.initial()


def _idle_after_answer(m: TurnMachine) -> MachineState:
    return answered_turn(m)


def _idle_with_next_words(m: TurnMachine) -> MachineState:
    """IDLE holding a word for the next turn (a word that arrived while nothing was open)."""
    s, _ = m.step(answered_turn(m), word("hey", 4000))
    assert s.state is TurnState.IDLE and s.next_words, "premise"
    return s


def _idle_after_reset(m: TurnMachine) -> MachineState:
    s, _ = m.step(listening(m), E.SessionReset(t_ms=5400))
    assert s.state is TurnState.IDLE and s.pause_onset_ms is not None, (
        "premise: the T-0162 reset marker"
    )
    return s


def _endpointing_flushed(m: TurnMachine) -> MachineState:
    return endpointing_after_flush(m)


def _endpointing_flushed_late_word(m: TurnMachine) -> MachineState:
    s, _ = m.step(endpointing_after_flush(m), word("please", 5600))
    assert s.next_words, "premise"
    return s


def _thinking(m: TurnMachine) -> MachineState:
    s, _ = m.step(endpointing_after_flush(m), E.Tick(t_ms=5800 + SETTLE))
    assert s.state is TurnState.THINKING and s.clip_playing, "premise: thinking, ack playing"
    return s


def _thinking_with_onset(m: TurnMachine) -> MachineState:
    s, _ = m.step(_thinking(m), E.UserSpeechOnset(t_ms=6000, onset_ms=6000))
    assert s.state is TurnState.THINKING and s.timers, "premise: takeover armed"
    return s


def _speaking(m: TurnMachine) -> MachineState:
    s, _ = m.step(
        _thinking(m),
        E.LlmCommit(t_ms=6200, request_id="t2-gen", canvas_idx=0, segment="answer", text="Sure. "),
    )
    assert s.state is TurnState.SPEAKING, "premise"
    return s


def _paused(m: TurnMachine) -> MachineState:
    s, _ = m.step(_speaking(m), E.UserSpeechOnset(t_ms=6400, onset_ms=6400))
    assert s.state is TurnState.PAUSED, "premise"
    return s


def _interrupted(m: TurnMachine) -> MachineState:
    s, _ = m.step(_paused(m), E.UserTakeover(t_ms=6500, onset_ms=6400))
    assert s.state is TurnState.INTERRUPTED, "premise"
    return s


def _typed_seeded(m: TurnMachine) -> MachineState:
    return typed_turn(m)


def _loop_awaiting(m: TurnMachine) -> MachineState:
    """A live tool loop waiting for its call's result, in THINKING."""
    s = _thinking(m)
    draft = ti.tool_draft(call_id="c1", status="valid", name="search", arguments={"q": "x"})
    s, _ = m.step(s, ti.llm_done(6300, "t2-gen", draft))
    assert s.state is TurnState.THINKING and s.awaiting_result == "c1", "premise: awaiting c1"
    return s


ELSEWHERE = {
    "idle-fresh": (_idle_fresh, POLICY),
    "idle-after-answer": (_idle_after_answer, POLICY),
    "idle-next-words": (_idle_with_next_words, POLICY),
    "idle-after-reset": (_idle_after_reset, POLICY),
    "endpointing-flushed": (_endpointing_flushed, POLICY),
    "endpointing-flushed-late-word": (_endpointing_flushed_late_word, POLICY),
    "typed-seeded": (_typed_seeded, POLICY),
    "thinking": (_thinking, POLICY),
    "thinking-takeover-armed": (_thinking_with_onset, POLICY),
    "speaking": (_speaking, POLICY),
    "paused": (_paused, POLICY),
    "interrupted": (_interrupted, POLICY),
    "loop-awaiting-result": (
        _loop_awaiting,
        TurnPolicy(thinking=True, tool_loop=True, max_tool_steps=3, still_working_after_ms=1000),
    ),
}


@pytest.mark.parametrize("where", sorted(ELSEWHERE))
def test_no_row_elsewhere(where: str) -> None:
    """A0-3: in IDLE, THINKING, SPEAKING, PAUSED, INTERRUPTED and ENDPOINTING after the flush, ``step(s, MicMuted)``
    returns ``(s, ())``: the state unchanged (``next_words`` included) and no command."""
    build, policy = ELSEWHERE[where]
    m = TurnMachine(policy)
    s = build(m)
    new, cmds = m.step(s, mic_muted(9000))
    assert tuple(cmds) == (), f"A0-3 ({where}, {s.state.value}): emitted {ti.kinds(cmds)}"
    assert new == s, (
        f"A0-3 ({where}): MicMuted moved the machine from {s.state.value} to {new.state.value}"
    )


# =====================================================================================================================
# A0-4: a stale tail moves nothing
# =====================================================================================================================


@pytest.mark.parametrize("at", ["listening", "endpointing"])
def test_stale_tail_after_abandon(at: str) -> None:
    """A0-4: after the abandon, the run's ``stt.end_of_turn``, an ``stt.flushed`` naming the abandoned ``t_eot_ms``
    and a ``Tick`` past ``flush + eot_settle_ms`` each step to ``(same state, ())``: no LISTENING/ENDPOINTING comes
    back, no ``llm.start``."""
    m = TurnMachine(POLICY)
    s = listening(m) if at == "listening" else endpointing_before_flush(m)
    after, _ = m.step(s, mic_muted(5750))
    assert after.state is TurnState.IDLE, "A0-2 premise"
    tail = [
        E.SttEndOfTurn(t_ms=5760, t_eot_ms=5400),
        E.SttFlushed(t_ms=5800, t_eot_ms=5400),
        E.Tick(t_ms=5800 + SETTLE),
        E.Tick(t_ms=5800 + SETTLE + 1000),
    ]
    for ev in tail:
        new, cmds = m.step(after, ev)
        assert tuple(cmds) == (), (
            f"A0-4: a late {ev.type} after the abandon emitted {ti.kinds(cmds)}"
        )
        assert new == after, (
            f"A0-4: a late {ev.type} after the abandon moved the machine to {new.state.value}"
        )


# =====================================================================================================================
# A0-5 (adversarial): typed and ui turns are never abandoned
# =====================================================================================================================


def typed_turn(m: TurnMachine) -> MachineState:
    """A typed question seeded as ``session._open_pending_transcript`` seeds it: the session's own onset opens t2
    (IDLE -> LISTENING), and in the same pump step the state is written to ENDPOINTING with the typed transcript as
    the word buffer, ``t_eot_ms`` at the request's audio time and ``flush_t_ms`` at the session's ``now``."""
    s, _ = m.step(answered_turn(m), E.UserSpeechOnset(t_ms=7000, onset_ms=7000))
    assert s.state is TurnState.LISTENING and s.turn_id == "t2", (
        "premise: the typed request's onset opened t2"
    )
    return s.model_copy(
        update={
            "words": (Word(text="how many legs has a spider", start_ms=7000, end_ms=7000),),
            "state": TurnState.ENDPOINTING,
            "t_eot_ms": 7000,
            "flush_t_ms": 7000,
        }
    )


@pytest.mark.parametrize(
    "mutes",
    [(7000,), (7050,), (7000, 7060), (7000 + SETTLE - 1,)],
    ids=["at-seed", "mid-window", "twice", "last-ms"],
)
def test_typed_turn_survives_mute(mutes: tuple[int, ...]) -> None:
    """A0-5, adversarial (kills "abandon every ENDPOINTING"): a typed turn muted at any point before THINKING still
    reaches ``llm.start`` with its typed transcript at the settle tick; no ``prefill_abort`` before it."""
    m = TurnMachine(POLICY)
    s = typed_turn(m)
    out: list[Any] = []
    t_prev = 7000
    for t in mutes:
        if t > t_prev:
            s, cmds = m.step(s, E.Tick(t_ms=t))
            out.extend(cmds)
        s, cmds = m.step(s, mic_muted(t))
        out.extend(cmds)
        assert s.state is TurnState.ENDPOINTING, (
            f"A0-5: muting at {t} moved the typed turn to {s.state.value}"
        )
        t_prev = t
    s, cmds = m.step(s, E.Tick(t_ms=7000 + SETTLE))
    out.extend(cmds)
    starts = [c for c in out if isinstance(c, C.LlmStart)]
    assert starts == [C.LlmStart(request_id="t2-gen", transcript="how many legs has a spider")], (
        f"A0-5: the typed question is answered; got {ti.kinds(out)}"
    )
    assert s.state is TurnState.THINKING


# =====================================================================================================================
# The next turn and history
# =====================================================================================================================


def test_next_turn_after_abandon_has_no_old_words() -> None:
    """Adversarial (kills a mutant that keeps ``next_words``): t2 heard "drop me", then "please" after its
    end-of-turn; muted before the flush. t3 asks "fresh": its ``llm.start`` transcript is exactly "fresh", not
    "please fresh" or "drop me ... fresh"."""
    m = TurnMachine(POLICY)
    s, _ = m.step(endpointing_before_flush(m), word("please", 5600))
    assert s.next_words, "premise"
    s, _ = m.step(s, mic_muted(6100))
    s, cmds = run(
        m,
        [
            E.UserSpeechOnset(t_ms=8000, onset_ms=8000),
            word("fresh", 8000),
            E.SttEndOfTurn(t_ms=8500, t_eot_ms=8300),
            E.SttFlushed(t_ms=8600, t_eot_ms=8300),
            E.Tick(t_ms=8600 + SETTLE),
        ],
        s,
    )
    starts = [c for c in cmds if isinstance(c, C.LlmStart)]
    assert starts == [C.LlmStart(request_id="t3-gen", transcript="fresh")], f"got {starts}"


def test_after_abandon_relisten_history_kept() -> None:
    """After an abandoned t2, t3 ("fresh", answered "Sure.") completes as usual: history is t1's entry then t3's,
    hand-written; nothing of t2 ("drop me") is ever stored and no ``commit_history`` is emitted for it."""
    m = TurnMachine(POLICY)
    s = listening(m)
    s, cmds = m.step(s, mic_muted(5300))
    assert "commit_history" not in ti.kinds(cmds)
    s, _ = run(
        m,
        [
            E.UserSpeechOnset(t_ms=8000, onset_ms=8000),
            word("fresh", 8000),
            E.SttEndOfTurn(t_ms=8500, t_eot_ms=8300),
            E.SttFlushed(t_ms=8600, t_eot_ms=8300),
            E.Tick(t_ms=8600 + SETTLE),
        ],
        s,
    )
    assert s.state is TurnState.THINKING and s.active_generate == "t3-gen", "premise"
    s, _ = m.step(
        s,
        E.LlmCommit(t_ms=9000, request_id="t3-gen", canvas_idx=0, segment="answer", text="Sure."),
    )
    s, cmds = run(
        m,
        [
            E.LlmDone(t_ms=9100, request_id="t3-gen"),
            E.PlaybackDrained(t_ms=9500, played_ms=300, played_text="Sure."),
        ],
        s,
    )
    assert s.state is TurnState.IDLE
    assert s.history == (
        HistoryTurn(user_text="what is this", assistant_spoken="Ok."),
        HistoryTurn(user_text="fresh", assistant_spoken="Sure."),
    )


# =====================================================================================================================
# A0-6: the differential (no MicMuted -> BASE exactly)
# =====================================================================================================================

#: L8's recipes (tool drafts, repairs, results), plus long ticks and session resets: no recipe makes a MicMuted
_LONG = (st.tuples(st.just("tick"), st.integers(1000, 7000)),)
#: Reviewer B's addition: drive a fresh turn to ENDPOINTING before the flush (with or without a late word in
#: ``next_words``), then deliver one of the events whose rows touch the word buffers there. Uniform draws reach
#: "ENDPOINTING, next_words set" only by luck, so a change to those rows (say ``_on_stt_error`` clearing
#: ``next_words``) would pass the differential unseen.
_ENDPOINT = st.tuples(
    st.just("endpoint"),
    st.booleans(),
    st.sampled_from(["word", "stt_error", "reset", "disconnect", "onset"]),
)
_DIFF_EV = st.one_of(ti._L8_EV_RESULTS, *(_LONG * 2), st.just(("reset",)), *((_ENDPOINT,) * 2))


def _endpoint_events(recipe: Any, t: int) -> list[Any]:
    """From IDLE: onset, one word inside the turn, the end-of-turn (t_eot = t + 500), no Marker echo; with ``late``
    a word heard after t_eot (``next_words``); then the closing event."""
    _, late, then = recipe
    events: list[Any] = [
        E.UserSpeechOnset(t_ms=t + 50, onset_ms=t + 50),
        E.SttWord(t_ms=t + 450, text="what", start_ms=t + 50, end_ms=t + 250),
        E.SttEndOfTurn(t_ms=t + 550, t_eot_ms=t + 500),
    ]
    if late:
        events.append(E.SttWord(t_ms=t + 900, text="please", start_ms=t + 600, end_ms=t + 750))
    events.append(
        {
            "word": E.SttWord(t_ms=t + 1000, text="more", start_ms=t + 800, end_ms=t + 950),
            "stt_error": E.SttError(t_ms=t + 1000, message="x"),
            "reset": E.SessionReset(t_ms=t + 1000),
            "disconnect": E.ClientDisconnected(t_ms=t + 1000),
            "onset": E.UserSpeechOnset(t_ms=t + 1000, onset_ms=t + 1000),
        }[then]
    )
    return events


DIFF_POLICIES = {
    "default": {},
    "no-thinking": {"thinking": False},
    "no-ack": {"ack_clip": False},
    "loop-2": {"tool_loop": True, "max_tool_steps": 2},
    "loop-3-still-working": {
        "tool_loop": True,
        "max_tool_steps": 3,
        "still_working_after_ms": 1500,
    },
}


def _diff_trace(live: TurnMachine, old: Any, recipes: list[Any], *, max_events: int = 60) -> None:
    s = live.initial()
    assert s == old.initial()
    t, spoken, calls, n = 0, "", [], 0
    queue = list(recipes)
    while queue and n < max_events:
        recipe = queue.pop(0)
        if recipe[0] == "reset":
            events: list[Any] = [E.SessionReset(t_ms=t + 5)]
        elif recipe[0] == "endpoint":
            # end whatever is open first (a session reset ends every state), then walk the fresh turn
            events = [] if s.state is TurnState.IDLE else [E.SessionReset(t_ms=t + 5)]
            events += _endpoint_events(recipe, t + 10)
        else:
            parts = ti._l8_expand(recipe, s)
            if parts is not None:
                queue[:0] = parts
                continue
            events = ti._l8_events(recipe, s, t, spoken, calls)
        for ev in events:
            if n >= max_events:
                break
            assert ev.type != "mic.muted", "A0-6 premise: these traces carry no MicMuted"
            n += 1
            t = max(t, ev.t_ms)
            new, cmds = live.step(s, ev)
            base_new, base_cmds = old.step(s, ev)
            assert tuple(cmds) == tuple(base_cmds), (
                f"A0-6: {ev.type} in {s.state.value}: {ti.kinds(cmds)} != BASE {ti.kinds(base_cmds)}"
            )
            assert new == base_new, (
                f"A0-6: {ev.type} in {s.state.value}: the state diverged from BASE"
            )
            spoken += "".join(c.source for c in cmds if isinstance(c, C.TtsSpeak))
            s = new
            if s.state is TurnState.IDLE:
                spoken = ""


@pytest.mark.parametrize("cell", sorted(DIFF_POLICIES))
def test_differential_without_mic_muted(cell: str) -> None:
    """A0-6: on every event sequence with no ``MicMuted``, ``step`` returns exactly BASE's commands and state
    (``_frozen_turn_machine_t0181``, main 021c2ef6), under five policies. A guard: green before the implementation
    (the machine is BASE) and after it. Regime: 250 derandomized examples of 4-40 recipes, at most 60 events, per
    policy, plus the explicit examples; the ``endpoint`` recipe drives to ENDPOINTING before the flush, with or
    without ``next_words``, and then delivers a word, ``stt.error``, a reset, a disconnect or an onset."""
    kw = DIFF_POLICIES[cell]
    live = TurnMachine(TurnPolicy(**{"thinking": True, **kw}))
    old = frozen.TurnMachine(frozen.TurnPolicy(**{"thinking": True, **kw}))

    @settings(max_examples=250, deadline=None, derandomize=True, database=None)
    @given(recipes=st.lists(_DIFF_EV, min_size=4, max_size=40))
    @example(recipes=[("turn",), ("onset",), ("eot",), ("tick", 2000)])
    @example(recipes=[("turn",), ("reset",), ("turn",), ("tick", 3000)])
    @example(recipes=[("endpoint", True, "stt_error"), ("turn",), ("tick", 3000), ("drained",)])
    @example(recipes=[("endpoint", True, "onset"), ("eot",), ("flushed",), ("tick", 3000)])
    @example(recipes=[("endpoint", True, "word"), ("flushed",), ("tick", 3000), ("drained",)])
    @example(
        recipes=[
            ("endpoint", False, "disconnect"),
            ("endpoint", True, "reset"),
            ("turn",),
            ("tick", 99),
        ]
    )
    # with next_words: a disconnect keeps them (they seed the next turn), a reset empties them
    @example(recipes=[("endpoint", True, "disconnect"), ("turn",), ("tick", 3000), ("drained",)])
    @example(recipes=[("endpoint", True, "reset"), ("turn",), ("tick", 3000), ("drained",)])
    def run_trace(recipes: list[Any]) -> None:
        _diff_trace(live, old, recipes)

    run_trace()


def _frozen_body() -> str:
    text = pathlib.Path(frozen.__file__).read_text()
    start, end = "dropped) ---\n", "# --- end of verbatim copy ---\n"
    return text[text.index(start) + len(start) : text.index(end)].rstrip("\n") + "\n"


def test_frozen_copy_is_base_verbatim() -> None:
    """A0-6's oracle is only an oracle while nobody edits it: the body hashes to the hand-computed sha and, when the
    base commit is in the clone, equals ``turn_machine.py`` there from its line 18 on."""
    body = _frozen_body()
    assert hashlib.sha256(body.encode()).hexdigest() == FROZEN_BODY_SHA256
    assert frozen.BASE_SHA == BASE_A0
    shown = subprocess.run(
        ["git", "show", f"{BASE_A0}:local_astra/conductor/turn_machine.py"],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).parent,
    )
    if shown.returncode != 0:
        pytest.skip(f"base commit {BASE_A0} is not in this clone")
    assert body == "".join(shown.stdout.splitlines(keepends=True)[17:]).rstrip("\n") + "\n"


# =====================================================================================================================
# Review rulings A0-R1..A0-R3 (2026-10-28)
# =====================================================================================================================


def test_late_word_then_onset_then_mute_drops_next_words() -> None:
    """A0-R2: ENDPOINTING (t_eot 5400), a late word ("please", audio 5600 > 5400, so ``next_words``), then an onset
    that unfreezes the turn back to LISTENING with ``next_words`` still set; a mute there clears ``next_words`` too.
    The next turn's ``llm.start`` transcript is exactly its own word "fresh".

    Kills both shapes that clear ``next_words`` only from the ENDPOINTING arm: ``extra={"next_words": ()} if
    state.state is TurnState.ENDPOINTING else None`` and the one keyed on ``state.t_eot_ms is not None`` (the onset
    reset ``t_eot_ms`` to None)."""
    m = TurnMachine(POLICY)
    s, _ = m.step(endpointing_before_flush(m), word("please", 5600))
    assert [w.text for w in s.next_words] == ["please"], "premise: the late word is next_words"
    s, cmds = m.step(s, E.UserSpeechOnset(t_ms=6100, onset_ms=6100))
    assert s.state is TurnState.LISTENING and s.t_eot_ms is None, (
        "premise: the onset unfroze the turn"
    )
    assert C.Unfreeze() in cmds
    assert [w.text for w in s.next_words] == ["please"], "premise: next_words survives the unfreeze"
    new, cmds = m.step(s, mic_muted(6200))
    assert tuple(cmds) == abandon_commands(s), f"A0-2: got {ti.kinds(cmds)}"
    assert new.next_words == (), (
        f"A0-R2: the mute clears next_words in LISTENING too; got {new.next_words}"
    )
    assert new == abandoned(s)
    s, cmds = run(
        m,
        [
            E.UserSpeechOnset(t_ms=8000, onset_ms=8000),
            word("fresh", 8000),
            E.SttEndOfTurn(t_ms=8500, t_eot_ms=8300),
            E.SttFlushed(t_ms=8600, t_eot_ms=8300),
            E.Tick(t_ms=8600 + SETTLE),
        ],
        new,
    )
    starts = [c for c in cmds if isinstance(c, C.LlmStart)]
    assert starts == [C.LlmStart(request_id="t3-gen", transcript="fresh")], f"A0-R2: got {starts}"


def test_typed_turn_reopened_by_speech_is_dropped_by_mute() -> None:
    """A0-R3: a typed turn (ENDPOINTING, ``flush_t_ms`` set) reopened by a mic onset inside ``eot_settle_ms`` is back
    in LISTENING, a voice turn; a mute then drops it like any spoken question (A0-2's exact commands and state), and
    the old settle deadline starts nothing."""
    m = TurnMachine(POLICY)
    s, cmds = m.step(typed_turn(m), E.UserSpeechOnset(t_ms=7100, onset_ms=7100))
    assert SETTLE > 100, "premise: the onset is inside the settle window"
    assert s.state is TurnState.LISTENING and s.flush_t_ms is None, (
        "premise: the typed turn reopened"
    )
    new, cmds = m.step(s, mic_muted(7150))
    assert tuple(cmds) == abandon_commands(s), f"A0-R3: got {ti.kinds(cmds)}"
    assert new == abandoned(s)
    after, cmds = m.step(new, E.Tick(t_ms=7000 + SETTLE + 1))
    assert (after, tuple(cmds)) == (new, ()), (
        f"A0-R3: the settle tick moved the machine: {ti.kinds(cmds)}"
    )


#: The invariants of ``TurnMachineModel`` (``@invariant`` methods), checked after every replayed rule, as Hypothesis
#: does
_MODEL_INVARIANTS = (
    "at_most_one_generate_and_one_prefill",
    "frames_are_unique_and_bounded",
    "history_never_contains_unplayed_text",
    "idle_means_no_frames",
    "idle_means_no_live_requests",
    "never_speak_thoughts",
    "no_answer_or_history_for_an_abandoned_turn",
    "state_stays_bounded",
    "step_is_pure",
)


def _replay(model: Any, program: list[tuple[str, dict[str, Any]]]) -> list[Any]:
    """Run ``program`` (rule name, kwargs) on ``model`` with every invariant after each rule; return the commands
    emitted after the first ``mic_muted`` rule."""
    for name in _MODEL_INVARIANTS:
        assert callable(getattr(model, name)), f"premise: {name} is an invariant of the model"
    after_mute: list[Any] = []
    muted = False
    for rule, kwargs in program:
        n = len(model.commands)
        getattr(model, rule)(**kwargs)
        if muted:
            after_mute.extend(model.commands[n:])
        muted = muted or rule == "mic_muted"
        for name in _MODEL_INVARIANTS:
            getattr(model, name)()
    return after_mute


@pytest.mark.parametrize("stale", ["playback_drained", "playback_stopped"])
def test_a0_r1_stale_playback_then_disconnect_is_not_a_leak(stale: str) -> None:
    """A0-R1 regression (reviewer A's shrunk trace, ``--hypothesis-seed=3`` at 9dbddf96; reviewer B's stopped
    variant): a whole answered turn, ``llm.error``, an onset opening t2, ``mic.muted`` abandoning t2, then a stale
    playback terminal in IDLE and ``client.disconnected``: BASE's IDLE ``_commit_played`` emits a
    ``commit_history`` after the abandon. It carries none of t2's words, so the content property holds.

    Power: the premise asserts t2 held a word and a ``commit_history`` followed the abandon, so the old turn-id
    reading of the property (``commit_history`` while ``turn_id`` names t2) would have failed on this exact trace."""
    model = ti.TurnMachineLongRun()
    stale_kwargs: dict[str, Any] = {"played": 1263}
    if stale == "playback_stopped":
        stale_kwargs["frac"] = 0.5
    program: list[tuple[str, dict[str, Any]]] = [
        ("late_events_naming_a_request_this_session_never_started", {}),
        ("whole_turn_up_to_thinking_or_speaking", {"answers": True, "text": "Yes. "}),
        ("error", {"kind": "llm"}),
        ("onset", {}),
        ("word", {"word": "what"}),
        ("mic_muted", {}),
        (stale, stale_kwargs),
        ("error", {"kind": "disconnect"}),
    ]
    after_mute = _replay(model, program)
    assert model.abandoned_words, "premise: the abandoned turn held a word"
    assert "commit_history" in ti.kinds(after_mute), (
        f"premise: a commit_history followed the abandon (the false positive's shape); got {ti.kinds(after_mute)}"
    )
    assert model.s.turn_id == "t2", (
        "premise: the commit_history was emitted while turn_id named the abandoned t2"
    )
    assert model.abandoned_leaks == []


#: Reviewer B's replayed failing program (``TurnMachineLongRun``, fresh Hypothesis database), its rule calls in order
_REVIEWER_B_PROGRAM: list[tuple[str, dict[str, Any]]] = [
    ("late_events_naming_a_request_this_session_never_started", {}),
    ("session_reset", {}),
    ("takeover", {}),
    ("playback_stopped", {"frac": 0.9864136198832083, "played": 1299}),
    ("not_owner", {"score": 0.37824848359046437}),
    ("takeover", {}),
    ("end_of_turn", {}),
    ("backchannel", {}),
    ("scene_changed", {"phash": "b2b2b2b2b2b2b2b2"}),
    ("flushed", {}),
    ("playback_drained", {"played": 1263}),
    ("not_owner", {"score": 1.0}),
    ("whole_turn_up_to_thinking_or_speaking", {"answers": True, "text": "Look at line two. "}),
    ("telemetry", {"kind": "pause", "value": 0.8681059224354108}),
    ("word", {"word": "mm-hm"}),
    ("onset", {}),
    ("onset", {}),
    ("end_of_turn", {}),
    ("user_text", {"text": "à\x1f\x8bÎ"}),
    ("backchannel", {}),
    ("end_of_turn", {}),
    ("tick", {"ms": 26}),
    ("commit", {"open_": False, "seg": "thought", "text": "Consider (a. b"}),
    ("tool_result_nobody_awaits", {"call_id": "zz-9", "status": "executed"}),
    ("scene_changed", {"phash": "c3c3c3c3c3c3c3c3"}),
    ("session_reset", {}),
    ("tick", {"ms": 170}),
    ("flushed", {}),
    ("error", {"kind": "llm"}),
    ("error", {"kind": "llm"}),
    ("tool_result_nobody_awaits", {"call_id": "t0-call", "status": "executed"}),
    ("stt_error", {}),
    ("end_of_turn", {}),
    ("playback_drained", {"played": 1711}),
    ("onset", {}),
    ("takeover", {}),
    ("playback_stopped", {"frac": 0.5084163701902563, "played": 587}),
    ("telemetry", {"kind": "pause", "value": 0.005}),
    ("playback_stopped", {"frac": 0.05657601683166004, "played": 349}),
    ("tts_done", {}),
    ("tool_result_nobody_awaits", {"call_id": "t0-call", "status": "executed"}),
    ("mic_muted", {}),
    ("playback_stopped", {"frac": 0.5888486464891344, "played": 580}),
    ("offset", {"speech_ms": 353}),
    ("error", {"kind": "disconnect"}),
]


def test_a0_r1_reviewer_b_program_replays_green() -> None:
    """A0-R1 regression: reviewer B's whole failing program (a stray ``playback.stopped`` in IDLE after the abandon,
    then ``client.disconnected``) replays with every invariant green; it does reach a ``commit_history`` after the
    abandon, so the old turn-id property failed here."""
    model = ti.TurnMachineLongRun()
    after_mute = _replay(model, _REVIEWER_B_PROGRAM)
    assert "commit_history" in ti.kinds(after_mute), f"premise; got {ti.kinds(after_mute)}"
    assert model.abandoned_leaks == []


class _KeepsNextWords(TurnMachine):
    """A mutant for the property's power: the abandon keeps ``next_words`` (the dropped question's late tail)."""

    def step(self, state: MachineState, event: Any) -> tuple[MachineState, list[Any]]:
        new, cmds = super().step(state, event)
        if event.type == "mic.muted" and new.state is TurnState.IDLE:
            new = new.model_copy(update={"next_words": state.next_words})
        return new, cmds


@pytest.mark.parametrize("machine", ["live", "keeps-next-words"])
def test_content_property_has_teeth(machine: str) -> None:
    """A0-R1 power: the content property is not vacuous. A model trace heard a word, endpointed, heard a late word
    (``next_words``), was muted, then answered a whole new turn. With the live machine the property holds and checked
    the later ``llm.start`` and history entry; with a machine that keeps ``next_words`` on the abandon, the late word
    reaches t2's transcript and the property fails (the mutant is stepped through ``_step`` so only the property, not
    the ``mic_muted`` rule's own postcondition, can catch it)."""
    model = ti.TurnMachineModel()
    if machine != "live":
        model.m = _KeepsNextWords(ti.POLICY, segment=model._segment)
    model.onset()
    model.word("what")
    model.end_of_turn()
    model.tick(800)
    model.word("please")
    assert [w.text for w in model.s.next_words] == ["please~2"], (
        "premise: the late word is next_words"
    )
    before = model.s
    model._step(mic_muted(model._advance(5)))
    model._mark_abandoned(before)
    assert model.abandoned_words == {"what~1", "please~2"}
    model.whole_turn_up_to_thinking_or_speaking("Yes. ", True)
    model.llm_done()
    model.playback_drained(400)
    starts = [c for c in model.commands if isinstance(c, C.LlmStart)]
    assert starts, "premise: the next turn started a request"
    if machine == "live":
        model.no_answer_or_history_for_an_abandoned_turn()
        assert starts[-1].transcript == "what~3", f"got {starts[-1].transcript!r}"
        assert model.s.history[-1].user_text == "what~3", "premise: t2 was stored"
        assert model.content_checks == 2, (
            "power: the llm.start and the history entry were both checked"
        )
        return
    with pytest.raises(AssertionError, match="abandoned turn's words"):
        model.no_answer_or_history_for_an_abandoned_turn()
