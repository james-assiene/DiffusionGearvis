"""Turn machine: table transitions and §4 invariants (docs/02 §4). Task T-0003.

Activated at the start of /implement T-0003 (the ``pending`` marker is gone): these tests fail
until the machine exists and are never weakened to get green (CLAUDE.md). Each test names the
acceptance criterion of ``docs/tasks/T-0003.md`` it comes from; nothing here is derived from an
implementation, because there is none.

Conventions: assertions name the *required* commands (order where it matters), never an exact
command list, so an implementation may add ``hud.state`` or timers. Word attribution is by audio
time. "All words ≤ t_eot received" is observable as ``stt.flushed`` (moshi-server Marker).

T-0059 group A criterion 6 adds to the stateful model, and to nothing else here: the
``scene_changed`` rule draws its perceptual hash from a small alphabet instead of repeating one
constant, and the per-turn frame list ``MachineState.turn_phashes`` is checked on every
transition (ordered by arrival, no duplicate, bounded by the cap, lost only at a turn boundary).
Every assertion T-0003 wrote is untouched; the field is read through ``turn_phashes`` below so
that its absence is a named criterion rather than an ``AttributeError``.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import ClassVar

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

from local_astra.conductor import turn_machine as turn_machine_module
from local_astra.conductor.speech_gate import SpeechGate
from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.contracts.turn import MachineState, TurnState, Word

POLICY = TurnPolicy(thinking=True)

#: What the STT client reports when the transcription socket dies (criterion 20(c)).
STT_DEAD = "transcription socket closed"

#: T-0059 criterion 6: the alphabet the ``scene_changed`` rule draws from. One constant (today's
#: ``"ab" * 8``, kept in the pool) means a trace exercises the append path or the dedup path but
#: never both inside one turn, depending on where the turn boundary falls — so with a single hash
#: "the same screen twice costs nothing" and "a second screen is appended" cannot both be
#: observed. Four is enough to overflow the default cap of three frames per turn, which is where
#: the last-slot replacement of D1 lives.
SCENE_PHASHES = ("ab" * 8, "a1a1a1a1a1a1a1a1", "b2b2b2b2b2b2b2b2", "c3c3c3c3c3c3c3c3")


def repair_id(turn_id: str) -> str:
    """T-0089 R2: the repair request of a turn is ``"{turn_id}-repair"``."""
    return f"{turn_id}-repair"


def kinds(cmds) -> list[str]:
    return [c.type for c in cmds]


def run(machine: TurnMachine, events, state: MachineState | None = None):
    """Apply events in order; return (state, all commands)."""
    state = state or machine.initial()
    out = []
    for ev in events:
        state, cmds = machine.step(state, ev)
        out.extend(cmds)
    return state, out


def to_thinking(machine: TurnMachine):
    """IDLE → LISTENING → ENDPOINTING → THINKING, the common prefix of many tests."""
    words = [
        E.SttWord(t_ms=1500 + i * 250, text=w, start_ms=1000 + i * 250, end_ms=1250 + i * 250)
        for i, w in enumerate(["what", "is", "this"])
    ]
    events = [
        E.UserSpeechOnset(t_ms=1000, onset_ms=1000),
        *words[:2],
        E.SttEndOfTurn(t_ms=1800, t_eot_ms=1750),
        words[2],  # arrives after end-of-turn, audio time 1500 ≤ 1750 → belongs to this turn
        E.SttFlushed(t_ms=2050, t_eot_ms=1750),  # moshi Marker: all words ≤ t_eot delivered
        E.Tick(t_ms=2050 + POLICY.eot_settle_ms),
    ]
    return run(machine, events)


def commit(s, text, *, idx=0, segment="answer", open_=False, t=3000):
    return E.LlmCommit(
        t_ms=t,
        request_id=s.active_generate,
        canvas_idx=idx,
        segment=segment,
        text=text,
        thought_open=open_,
    )


_MISSING = object()


def policy_field(name: str, expected):
    """Read a ``TurnPolicy`` field T-0003 introduces, with a readable message when it is absent.

    The criteria give both the field and its default; a missing field must read as a failed
    criterion, not as an ``AttributeError`` half-way through a test.
    """
    value = getattr(POLICY, name, _MISSING)
    assert value is not _MISSING, f"T-0003 adds TurnPolicy.{name} (default {expected!r})"
    assert value == expected, f"TurnPolicy.{name} default should be {expected!r}, got {value!r}"
    return value


def turn_phashes(state) -> tuple[str, ...]:
    """``MachineState.turn_phashes`` (T-0059 criterion 1; the name is contract)."""
    value = getattr(state, "turn_phashes", _MISSING)
    assert value is not _MISSING, (
        "T-0059 criterion 1 adds MachineState.turn_phashes: the distinct scene-change hashes of "
        "the current turn, in arrival order"
    )
    return value


def frame_slots(policy) -> int:
    """How many entries ``turn_phashes`` may hold under ``policy`` (T-0059 criteria 3 and 5).

    The turn's first frame is the onset snapshot, whose hash no event carries (D3), so it fills a
    frame slot the tuple cannot show: criterion 3 bounds the tuple by ``max_frames_per_turn - 1``.
    Criterion 5 pins the floor at one, because at ``max_frames_per_turn == 1`` the scene change
    *replaces* the onset frame and the tuple holds exactly the newest hash (docs/01 D9, "the last
    snapshot wins").
    """
    cap = getattr(policy, "max_frames_per_turn", _MISSING)
    assert cap is not _MISSING, (
        "T-0059 criteria 3 and 5 add TurnPolicy.max_frames_per_turn (default 3, D2)"
    )
    return cap - 1 if cap > 1 else 1


def segment_parameter():
    """The keyword-only ``segment`` parameter of ``TurnMachine.__init__`` (criterion 10)."""
    param = inspect.signature(TurnMachine.__init__).parameters.get("segment")
    assert param is not None, "criterion 10: TurnMachine(policy, *, segment=...) injection point"
    return param


def drive_to_thinking(machine, *, onset_ms, t_eot_ms, words=(), state=None, p_end=1.0):
    """IDLE -> THINKING with explicit timings.

    Returns ``(state, commands, t_ms)`` where ``t_ms`` is the delivery time of the tick that
    started the generation — i.e. the time ``llm.start`` was emitted, which criterion 8 measures
    the first canvas from.
    """
    eot_t_ms = t_eot_ms + 50
    flush_t_ms = eot_t_ms + 250
    start_t_ms = flush_t_ms + machine.policy.eot_settle_ms
    events = [
        E.UserSpeechOnset(t_ms=onset_ms, onset_ms=onset_ms),
        *[
            E.SttWord(t_ms=start + 400, text=text, start_ms=start, end_ms=start + 200)
            for text, start in words
        ],
        E.SttEndOfTurn(t_ms=eot_t_ms, t_eot_ms=t_eot_ms, p_end=p_end),
        E.SttFlushed(t_ms=flush_t_ms, t_eot_ms=t_eot_ms),
        E.Tick(t_ms=start_t_ms),
    ]
    state, cmds = run(machine, events, state)
    return state, cmds, start_t_ms


def sentence_segmenter(text: str, *, final: bool):
    """Reference segmenter that *conserves* text: complete sentences end at ``.``, tail held.

    ``"".join(source) + remainder == text`` by construction, so criterion 10's check passes and
    the test can watch where the held tail surfaces (it must, on ``llm.done``).
    """
    out: list[tuple[str, str]] = []
    rest = text
    while (i := rest.find(".")) != -1:
        cut = i + 1
        while cut < len(rest) and rest[cut] == " ":
            cut += 1
        out.append((rest[:cut], rest[:cut].strip()))
        rest = rest[cut:]
    if final and rest:
        out.append((rest, rest.strip()))
        rest = ""
    return out, rest


def inventing_segmenter(text: str, *, final: bool):
    """Returns text nobody committed (the headline failure mode of the speech path)."""
    return [("Buy now at example.com", "Buy now at example.com")], ""


def dropping_segmenter(text: str, *, final: bool):
    """Silently drops the second half of what the LLM committed."""
    half = text[: len(text) // 2]
    return [(half, half)], ""


def reordering_segmenter(text: str, *, final: bool):
    """Keeps every character but swaps the halves: order is part of "what was committed"."""
    half = len(text) // 2
    return [(text[half:], text[half:]), (text[:half], text[:half])], ""


class TestTable:
    """One test per selected row of docs/02 §4 (the test-author completes the table)."""

    def test_idle_onset_opens_listening_with_snapshot_and_prefill(self):
        m = TurnMachine(POLICY)
        s, cmds = m.step(m.initial(), E.UserSpeechOnset(t_ms=1000, onset_ms=1000))
        assert s.state is TurnState.LISTENING
        k = kinds(cmds)
        assert "snapshot_frame" in k and "prefill_open" in k
        assert k.index("snapshot_frame") < k.index("prefill_open"), "frame before prefill"
        snap = next(c for c in cmds if isinstance(c, C.SnapshotFrame))
        assert snap.at_ms == 1000, "frame at speech onset, not latest"
        assert s.active_prefill is not None and s.active_generate is None

    def test_scene_change_while_listening_reissues_prefill(self):
        m = TurnMachine(POLICY)
        s, _ = m.step(m.initial(), E.UserSpeechOnset(t_ms=1000, onset_ms=1000))
        s, cmds = m.step(s, E.SceneChanged(t_ms=1200, source="screen", phash="ab" * 8))
        assert s.state is TurnState.LISTENING
        assert "snapshot_frame" in kinds(cmds) and "prefill_reissue" in kinds(cmds)

    def test_end_of_turn_freezes_then_late_word_is_still_appended(self):
        m = TurnMachine(POLICY)
        s, _ = m.step(m.initial(), E.UserSpeechOnset(t_ms=1000, onset_ms=1000))
        s, cmds = m.step(s, E.SttEndOfTurn(t_ms=1800, t_eot_ms=1750))
        assert s.state is TurnState.ENDPOINTING and "freeze_after" in kinds(cmds)
        assert "llm.start" not in kinds(cmds)
        s, _ = m.step(s, E.SttWord(t_ms=1900, text="late", start_ms=1500, end_ms=1700))
        assert [w.text for w in s.words] == ["late"]
        s, _ = m.step(s, E.SttWord(t_ms=1950, text="next-turn", start_ms=1800, end_ms=2000))
        assert [w.text for w in s.words] == ["late"], "audio time > t_eot is not this turn"

    def test_onset_during_endpointing_unfreezes_back_to_listening(self):
        m = TurnMachine(POLICY)
        s, _ = run(
            m,
            [
                E.UserSpeechOnset(t_ms=1000, onset_ms=1000),
                E.SttEndOfTurn(t_ms=1800, t_eot_ms=1750),
            ],
        )
        s, cmds = m.step(s, E.UserSpeechOnset(t_ms=1850, onset_ms=1840))
        assert s.state is TurnState.LISTENING and "unfreeze" in kinds(cmds)

    def test_llm_does_not_start_before_the_flush_and_the_settle_window(self):
        m = TurnMachine(POLICY)
        s, _ = run(
            m,
            [
                E.UserSpeechOnset(t_ms=1000, onset_ms=1000),
                E.SttWord(t_ms=1500, text="what", start_ms=1000, end_ms=1250),
                E.SttEndOfTurn(t_ms=1800, t_eot_ms=1750),
            ],
        )
        s, cmds = m.step(s, E.Tick(t_ms=1800 + POLICY.eot_settle_ms + 500))
        assert s.state is TurnState.ENDPOINTING and "llm.start" not in kinds(cmds), "no flush yet"
        s, cmds = m.step(s, E.SttFlushed(t_ms=2400, t_eot_ms=1750))
        s, cmds2 = m.step(s, E.Tick(t_ms=2400 + POLICY.eot_settle_ms - 20))
        assert "llm.start" not in kinds(cmds) + kinds(cmds2), "settle window not elapsed"
        s, cmds3 = m.step(s, E.Tick(t_ms=2400 + POLICY.eot_settle_ms))
        assert s.state is TurnState.THINKING and "llm.start" in kinds(cmds3)

    def test_settle_tick_starts_llm_aborts_prefill_and_plays_ack_when_thinking(self):
        m = TurnMachine(POLICY)
        s, cmds = to_thinking(m)
        assert s.state is TurnState.THINKING
        k = kinds(cmds)
        assert k.index("prefill_abort") < k.index("llm.start")
        start = next(c for c in cmds if isinstance(c, C.LlmStart))
        assert start.transcript == "what is this"
        assert any(isinstance(c, C.PlayClip) and c.kind == "ack" for c in cmds), "thinking → ack"
        assert s.active_generate == start.request_id and s.active_prefill is None
        assert [w.text for w in s.words] == ["what", "is", "this"]

    def test_no_ack_clip_without_thinking_when_first_canvas_is_fast(self):
        m = TurnMachine(TurnPolicy(thinking=False))
        fast = m.initial().model_copy(update={"first_canvas_ms": 200})  # EMA below 300 ms
        s, cmds = run(
            m,
            [
                E.UserSpeechOnset(t_ms=1000, onset_ms=1000),
                E.SttWord(t_ms=1500, text="hi", start_ms=1000, end_ms=1250),
                E.SttEndOfTurn(t_ms=1800, t_eot_ms=1750),
                E.SttFlushed(t_ms=2050, t_eot_ms=1750),
                E.Tick(t_ms=2050 + POLICY.eot_settle_ms),
            ],
            state=fast,
        )
        assert s.state is TurnState.THINKING
        assert not any(isinstance(c, C.PlayClip) for c in cmds)

    def test_answer_commit_moves_to_speaking_and_speaks_only_committed_text(self):
        m = TurnMachine(POLICY)
        s, _ = to_thinking(m)
        s, cmds = m.step(s, commit(s, "It is a stack trace. The error is on line four."))
        assert s.state is TurnState.SPEAKING
        speaks = [c for c in cmds if isinstance(c, C.TtsSpeak)]
        assert speaks and all(c.source in s.committed_answer and c.text for c in speaks)
        assert speaks[0].source.startswith("It is a stack trace.")
        assert "stop_clip" in kinds(cmds) or not any(isinstance(c, C.PlayClip) for c in cmds)

    def test_thought_commit_and_unclosed_thought_never_speak(self):
        m = TurnMachine(POLICY)
        s, _ = to_thinking(m)
        s, cmds = m.step(s, commit(s, "The user asks about the trace", segment="thought", t=2500))
        assert s.state is TurnState.THINKING and not any(isinstance(c, C.TtsSpeak) for c in cmds)
        s, cmds = m.step(s, commit(s, "Line four", idx=1, open_=True, t=2800))
        assert not any(isinstance(c, C.TtsSpeak) for c in cmds)

    def test_onset_while_speaking_pauses_and_short_backchannel_resumes(self):
        m = TurnMachine(POLICY)
        s, _ = to_thinking(m)
        s, _ = m.step(s, commit(s, "First sentence. Second sentence."))
        s, cmds = m.step(s, E.UserSpeechOnset(t_ms=3400, onset_ms=3400))
        assert s.state is TurnState.PAUSED and "playback.pause" in kinds(cmds)
        s, cmds1 = m.step(s, E.UserSpeechOffset(t_ms=3650, offset_ms=3650, speech_ms=250))
        s, cmds2 = m.step(s, E.UserBackchannel(t_ms=3700, duration_ms=250))
        s, cmds3 = m.step(s, E.Tick(t_ms=3700 + POLICY.resume_max_ms))
        assert s.state is TurnState.SPEAKING
        assert "playback.resume" in kinds(cmds1) + kinds(cmds2) + kinds(cmds3)

    def test_takeover_while_paused_interrupts_then_history_keeps_only_played_text(self):
        m = TurnMachine(POLICY)
        s, _ = to_thinking(m)
        s, _ = m.step(s, commit(s, "First sentence. Second sentence."))
        s, _ = m.step(s, E.UserSpeechOnset(t_ms=3400, onset_ms=3400))
        s, cmds = m.step(s, E.UserTakeover(t_ms=3900, onset_ms=3400))
        assert s.state is TurnState.INTERRUPTED
        assert {"llm.abort", "tts.cancel", "playback.flush"} <= set(kinds(cmds))
        s, cmds = m.step(
            s, E.PlaybackStopped(t_ms=3950, played_ms=1200, played_text="First sentence.")
        )
        assert s.state is TurnState.LISTENING
        trunc = next(c for c in cmds if isinstance(c, C.TruncateHistoryToPlayed))
        assert trunc.played_ms == 1200 and trunc.played_text == "First sentence."
        assert "snapshot_frame" in kinds(cmds) and "prefill_open" in kinds(cmds)
        assert s.active_generate is None
        assert s.history[-1].assistant_spoken.startswith("First sentence.")
        assert "Second sentence" not in s.history[-1].assistant_spoken
        assert s.history[-1].interrupted is True

    def test_interrupted_times_out_into_listening_with_estimated_played_ms(self):
        m = TurnMachine(POLICY)
        s, _ = to_thinking(m)
        s, _ = m.step(s, commit(s, "First sentence. Second sentence."))
        s, _ = m.step(s, E.UserSpeechOnset(t_ms=3400, onset_ms=3400))
        s, _ = m.step(s, E.UserTakeover(t_ms=3900, onset_ms=3400))
        s, cmds = m.step(s, E.Tick(t_ms=3900 + POLICY.interrupt_timeout_ms))
        assert s.state is TurnState.LISTENING
        trunc = next(c for c in cmds if isinstance(c, C.TruncateHistoryToPlayed))
        assert trunc.played_ms is None, "executor estimates when playback.stopped never came"
        assert trunc.played_text == "", "the machine must not invent what was played"
        assert s.history[-1].assistant_spoken == "[interrupted]"
        assert s.history[-1].interrupted is True

    def test_done_and_drained_returns_to_idle_and_commits_history(self):
        m = TurnMachine(POLICY)
        s, _ = to_thinking(m)
        rid = s.active_generate
        s, _ = m.step(s, commit(s, "Done."))
        s, _ = m.step(s, E.LlmDone(t_ms=3100, request_id=rid))
        s, _ = m.step(s, E.TtsDone(t_ms=3200, utterance_id=s.utterances[-1]))
        s, cmds = m.step(s, E.PlaybackDrained(t_ms=3800, played_ms=600, played_text="Done."))
        assert s.state is TurnState.IDLE and "commit_history" in kinds(cmds)
        assert s.history[-1].assistant_spoken == "Done."
        assert s.history[-1].user_text == "what is this"
        assert s.active_generate is None and s.active_prefill is None

    def test_llm_error_goes_idle_with_hud_error_and_user_only_history(self):
        m = TurnMachine(POLICY)
        s, _ = to_thinking(m)
        s, cmds = m.step(s, E.LlmError(t_ms=2600, request_id=s.active_generate, message="boom"))
        assert s.state is TurnState.IDLE and "hud.error" in kinds(cmds)
        commit_cmd = next(c for c in cmds if isinstance(c, C.CommitHistory))
        assert commit_cmd.user_only is True

    @pytest.mark.parametrize("stage", ["listening", "thinking", "speaking"])
    def test_client_disconnect_from_any_state_degrades_and_idles(self, stage):
        m = TurnMachine(POLICY)
        if stage == "listening":
            s, _ = m.step(m.initial(), E.UserSpeechOnset(t_ms=1000, onset_ms=1000))
        else:
            s, _ = to_thinking(m)
            if stage == "speaking":
                s, _ = m.step(s, commit(s, "Hello there."))
        s, cmds = m.step(s, E.ClientDisconnected(t_ms=5000))
        assert s.state is TurnState.IDLE and "degrade" in kinds(cmds)
        assert next(c for c in cmds if isinstance(c, C.Degrade)).mode == "text_only"
        assert s.degraded is True
        assert s.active_generate is None and s.active_prefill is None

    # --- rows with no test before T-0003 was activated -----------------------------------------

    def test_idle_onset_aborts_the_captioner_and_opens_a_new_turn(self):
        """Criterion 1: captioner.abort, snapshot_frame, prefill_open — in that order."""
        m = TurnMachine(POLICY)
        initial = m.initial()
        s, cmds = m.step(initial, E.UserSpeechOnset(t_ms=1000, onset_ms=1000))
        k = kinds(cmds)
        assert "captioner.abort" in k, "the idle captioner yields to a live turn (test-plan I8)"
        assert k.index("captioner.abort") < k.index("snapshot_frame") < k.index("prefill_open")
        opened = next(c for c in cmds if isinstance(c, C.PrefillOpen))
        assert s.active_prefill == opened.request_id
        assert s.turn_seq == initial.turn_seq + 1
        assert s.turn_id != initial.turn_id, "a new turn gets a new id"
        twin, _ = TurnMachine(POLICY).step(initial, E.UserSpeechOnset(t_ms=1000, onset_ms=1000))
        assert twin.turn_id == s.turn_id, "turn_id is derived from turn_seq, not from a clock"

    def test_scene_change_outside_listening_is_ignored(self):
        """Criterion 2 (negative): scene.changed only re-warms the prefill while LISTENING."""
        m = TurnMachine(POLICY)
        scene = E.SceneChanged(t_ms=2900, source="screen", phash="cd" * 8)
        idle = m.initial()
        s, cmds = m.step(idle, scene)
        assert s == idle and list(cmds) == []
        thinking, _ = to_thinking(m)
        s, cmds = m.step(thinking, scene)
        assert s == thinking and list(cmds) == []

    def test_end_of_turn_below_p_threshold_stays_listening(self):
        """Criterion 3: the endpoint is gated on p_end >= TurnPolicy.eot_p_threshold."""
        threshold = policy_field("eot_p_threshold", 0.6)
        m = TurnMachine(POLICY)
        listening, _ = m.step(m.initial(), E.UserSpeechOnset(t_ms=1000, onset_ms=1000))
        low = E.SttEndOfTurn(t_ms=1800, t_eot_ms=1750, p_end=threshold - 0.1)
        s, cmds = m.step(listening, low)
        assert s == listening, "below the threshold the state is unchanged"
        assert list(cmds) == [], "below the threshold nothing is emitted"
        at = E.SttEndOfTurn(t_ms=1800, t_eot_ms=1750, p_end=threshold)
        s, cmds = m.step(listening, at)
        assert s.state is TurnState.ENDPOINTING, "the guard is >=, not >"
        freeze = next(c for c in cmds if isinstance(c, C.FreezeAfter))
        assert freeze.t_eot_ms == 1750 and s.t_eot_ms == 1750
        assert "llm.start" not in kinds(cmds)

    def test_check_owner_emitted_for_long_enough_speech(self):
        """Criterion 6: the owner check runs on the turn's audio window, before llm.start."""
        min_ms = policy_field("owner_check_min_speech_ms", 1500)
        m = TurnMachine(POLICY)
        s, cmds, start_t = drive_to_thinking(
            m, onset_ms=1000, t_eot_ms=1000 + min_ms, words=[("hello", 1100)]
        )
        assert s.state is TurnState.THINKING
        owner = [c for c in cmds if isinstance(c, C.CheckOwner)]
        assert len(owner) == 1, "exactly one owner check per turn"
        assert owner[0].from_ms == 1000 and owner[0].to_ms == 1000 + min_ms
        k = kinds(cmds)
        assert k.index("check_owner") < k.index("prefill_abort") < k.index("llm.start")
        armed = [c for c in cmds if isinstance(c, C.ArmTimer) and c.name == "llm_timeout"]
        assert len(armed) == 1 and armed[0].at_ms == start_t + POLICY.llm_timeout_ms
        assert ("llm_timeout", start_t + POLICY.llm_timeout_ms) in [
            (t.name, t.at_ms) for t in s.timers
        ]

    def test_no_check_owner_for_speech_shorter_than_the_embedding_minimum(self):
        """Criterion 6: speaker embeddings need >= 1.5 s (docs/01 D7); shorter turns skip it."""
        min_ms = policy_field("owner_check_min_speech_ms", 1500)
        m = TurnMachine(POLICY)
        s, cmds, _ = drive_to_thinking(
            m, onset_ms=1000, t_eot_ms=1000 + min_ms - 1, words=[("hello", 1100)]
        )
        assert s.state is TurnState.THINKING and "llm.start" in kinds(cmds)
        assert "check_owner" not in kinds(cmds)

    def test_first_canvas_ema_updates_on_first_commit_only(self):
        """Criterion 8: first_canvas_ms is the cross-turn EMA of the first-commit latency.

        Hand-computed: alpha = 0.3, previous EMA 1000 ms, measured 400 ms ->
        round(0.3 * 400 + 0.7 * 1000) = round(120 + 700) = 820.
        """
        policy_field("first_canvas_ema_alpha", 0.3)
        cold = TurnMachine(POLICY)
        s, _, start_t = drive_to_thinking(cold, onset_ms=1000, t_eot_ms=2600, words=[("hi", 1100)])
        s, _ = cold.step(s, commit(s, "An answer. ", t=start_t + 400))
        assert s.first_canvas_ms == 400, "no previous EMA: the measurement itself"

        warm = TurnMachine(POLICY)
        seeded = warm.initial().model_copy(update={"first_canvas_ms": 1000})
        s2, _, start_t2 = drive_to_thinking(
            warm, onset_ms=1000, t_eot_ms=2600, words=[("hi", 1100)], state=seeded
        )
        s2, _ = warm.step(s2, commit(s2, "An answer. ", t=start_t2 + 400))
        assert s2.first_canvas_ms == 820
        s2, _ = warm.step(s2, commit(s2, "And more.", idx=1, t=start_t2 + 1500))
        assert s2.first_canvas_ms == 820, "later commits of the same turn do not move the EMA"

    def test_no_filler_clips_or_interim_timer_come_from_this_machine(self):
        """Criterion 7 (negative): hmm / still_working / interim_update belong to T-0013."""
        m = TurnMachine(POLICY)
        s, cmds, _ = drive_to_thinking(m, onset_ms=1000, t_eot_ms=2600, words=[("hi", 1100)])
        seen = list(cmds)
        for i in range(10):  # stay inside llm_timeout_ms so the turn is still THINKING
            s, more = m.step(s, E.Tick(t_ms=3000 + i * 2000))
            seen.extend(more)
        assert s.state is TurnState.THINKING
        assert [c.kind for c in seen if isinstance(c, C.PlayClip)] == ["ack"]
        assert not any(isinstance(c, C.ArmTimer) and c.name == "interim_update" for c in seen)

    def test_default_segmenter_speaks_the_whole_commit_unsplit(self):
        """Criterion 10: the default segmenter returns ([(text, text)], "") — no splitting."""
        m = TurnMachine(POLICY)
        s, _ = to_thinking(m)
        text = "One thing. Then another."
        s, cmds = m.step(s, commit(s, text))
        speaks = [c for c in cmds if isinstance(c, C.TtsSpeak)]
        assert [c.source for c in speaks] == [text]
        assert s.spoken_text == text and s.committed_answer == text

    def test_injected_segmenter_drives_one_tts_speak_per_sentence(self):
        """Criteria 9 and 10: one tts.speak per returned sentence, in order, sources conserved."""
        segment_parameter()
        m = TurnMachine(POLICY, segment=sentence_segmenter)
        s, _ = to_thinking(m)
        s, cmds = m.step(s, commit(s, "One thing. Then another. "))
        speaks = [c for c in cmds if isinstance(c, C.TtsSpeak)]
        assert [c.source for c in speaks] == ["One thing. ", "Then another. "]
        assert [c.text for c in speaks] == ["One thing.", "Then another."]
        assert "".join(c.source for c in speaks) == s.committed_answer
        assert s.spoken_text == s.committed_answer
        assert len({c.utterance_id for c in speaks}) == 2
        assert [c.utterance_id for c in speaks] == list(s.utterances[-2:])

    def test_llm_done_flushes_the_segmenter_remainder(self):
        """Criterion 12: llm.done flushes with final=True and cancels the llm_timeout timer."""
        segment_parameter()
        m = TurnMachine(POLICY, segment=sentence_segmenter)
        s, _, _ = drive_to_thinking(m, onset_ms=1000, t_eot_ms=2600, words=[("hi", 1100)])
        rid = s.active_generate
        s, cmds = m.step(s, commit(s, "A full sentence. A dangling tail", t=3000))
        assert [c.source for c in cmds if isinstance(c, C.TtsSpeak)] == ["A full sentence. "]
        assert s.state is TurnState.SPEAKING
        s, cmds = m.step(s, E.LlmDone(t_ms=3100, request_id=rid))
        assert s.llm_done is True
        flushed = [c for c in cmds if isinstance(c, C.TtsSpeak)]
        assert [c.source for c in flushed] == ["A dangling tail"]
        assert all(c.source in s.committed_answer for c in flushed)
        assert s.spoken_text == s.committed_answer
        assert C.CancelTimer(name="llm_timeout") in list(cmds)
        assert "llm_timeout" not in [t.name for t in s.timers]
        k = kinds(cmds)
        bookkeeping = {"tts.speak", "arm_timer", "cancel_timer", "hud.state"}
        others = [i for i, name in enumerate(k) if name not in bookkeeping]
        assert max(i for i, name in enumerate(k) if name == "tts.speak") < min(
            others, default=len(k)
        ), "the flushed sentences come before anything else in the step"

    def test_llm_done_without_any_answer_idles_with_user_only_history(self):
        """Criterion 12: an empty answer ends the turn quietly — no hud.error, nothing spoken."""
        m = TurnMachine(POLICY)
        s, _, _ = drive_to_thinking(m, onset_ms=1000, t_eot_ms=2600, words=[("hi", 1100)])
        s, cmds = m.step(s, E.LlmDone(t_ms=3000, request_id=s.active_generate))
        assert s.state is TurnState.IDLE
        assert "hud.error" not in kinds(cmds)
        assert not any(isinstance(c, C.TtsSpeak) for c in cmds)
        assert next(c for c in cmds if isinstance(c, C.CommitHistory)).user_only is True

    def test_a_commit_from_a_stale_request_is_ignored(self):
        """Criterion 11: only the active generation may speak or extend the answer."""
        m = TurnMachine(POLICY)
        s, _ = to_thinking(m)
        stale = E.LlmCommit(
            t_ms=3000,
            request_id=s.active_generate + "-abandoned",
            canvas_idx=0,
            segment="answer",
            text="From a turn that was abandoned.",
            thought_open=False,
        )
        after, cmds = m.step(s, stale)
        assert after.state is TurnState.THINKING
        assert not any(isinstance(c, C.TtsSpeak) for c in cmds)
        assert after.committed_answer == "" and after.spoken_text == ""

    def test_llm_timeout_tick_ends_the_turn_with_an_error(self):
        """Criterion 14: the armed llm_timeout deadline fires through a plain tick."""
        m = TurnMachine(POLICY)
        s, _, start_t = drive_to_thinking(m, onset_ms=1000, t_eot_ms=2600, words=[("hi", 1100)])
        s, cmds = m.step(s, E.Tick(t_ms=start_t + POLICY.llm_timeout_ms - 1))
        assert s.state is TurnState.THINKING, "before the deadline nothing happens"
        assert "hud.error" not in kinds(cmds)
        s, cmds = m.step(s, E.Tick(t_ms=start_t + POLICY.llm_timeout_ms))
        assert s.state is TurnState.IDLE
        assert "hud.error" in kinds(cmds)
        assert next(c for c in cmds if isinstance(c, C.CommitHistory)).user_only is True
        assert not any(isinstance(c, C.TtsSpeak) for c in cmds)
        assert s.timers == () and s.active_generate is None

    def test_playback_pause_is_the_first_command_on_barge_in(self):
        """Criterion 16: the B4 budget (barge-in pause p95 <= 300 ms) needs pause issued first."""
        m = TurnMachine(POLICY)
        s, _ = to_thinking(m)
        s, _ = m.step(s, commit(s, "First sentence. Second sentence."))
        s, cmds = m.step(s, E.UserSpeechOnset(t_ms=3400, onset_ms=3350))
        assert s.state is TurnState.PAUSED
        assert kinds(cmds)[0] == "playback.pause", "nothing may precede the pause"
        armed = [c for c in cmds if isinstance(c, C.ArmTimer) and c.name == "takeover"]
        assert len(armed) == 1 and armed[0].at_ms == 3350 + POLICY.takeover_ms

    def test_thinking_onset_stops_clip_and_arms_takeover(self):
        """Criterion 15: a barge-in while thinking is not yet a takeover."""
        m = TurnMachine(POLICY)
        s, cmds, _ = drive_to_thinking(m, onset_ms=1000, t_eot_ms=2600, words=[("hi", 1100)])
        assert any(isinstance(c, C.PlayClip) for c in cmds), "thinking=True: the ack clip played"
        s, cmds = m.step(s, E.UserSpeechOnset(t_ms=3000, onset_ms=2990))
        assert s.state is TurnState.THINKING
        assert "stop_clip" in kinds(cmds)
        armed = [c for c in cmds if isinstance(c, C.ArmTimer) and c.name == "takeover"]
        assert len(armed) == 1 and armed[0].at_ms == 3000 + POLICY.takeover_ms
        assert not any(isinstance(c, C.LlmAbort) for c in cmds), "the generation keeps running"

    def test_thinking_takeover_aborts_and_keeps_transcript_for_next_turn(self):
        """Criterion 15: §4's concatenate_transcript cell, expressed as retained state."""
        m = TurnMachine(POLICY)
        s, _, _ = drive_to_thinking(
            m,
            onset_ms=1000,
            t_eot_ms=2600,
            words=[("what", 1100), ("is", 1400), ("this", 1700)],
        )
        rid = s.active_generate
        s, _ = m.step(s, E.UserSpeechOnset(t_ms=3000, onset_ms=2990))
        s, cmds = m.step(s, E.UserTakeover(t_ms=3100, onset_ms=2990))
        assert s.state is TurnState.LISTENING
        k = kinds(cmds)
        assert k.index("llm.abort") < k.index("snapshot_frame") < k.index("prefill_open")
        assert next(c for c in cmds if isinstance(c, C.LlmAbort)).request_id == rid
        assert next(c for c in cmds if isinstance(c, C.SnapshotFrame)).at_ms == 2990
        assert "commit_history" not in k, "an aborted turn is not committed to history"
        assert s.active_generate is None
        s, cmds = run(
            m,
            [
                E.SttWord(t_ms=3600, text="really", start_ms=3100, end_ms=3300),
                E.SttEndOfTurn(t_ms=3800, t_eot_ms=3700),
                E.SttFlushed(t_ms=3900, t_eot_ms=3700),
                E.Tick(t_ms=3900 + POLICY.eot_settle_ms),
            ],
            s,
        )
        assert s.state is TurnState.THINKING
        start = next(c for c in cmds if isinstance(c, C.LlmStart))
        assert start.transcript == "what is this really"


# --- stateful invariants -----------------------------------------------------------------------------


class TurnMachineModel(RuleBasedStateMachine):
    #: T-0004 A5/A9: the model drives the machine through the *real* speech gate, one gate per
    #: turn, because the default segmenter never holds anything back and so can never show the
    #: postcondition that matters — that a gate holding a half sentence still gives it up on
    #: ``llm.done``, and that a barged turn's held text is dropped instead of being voiced late.
    def __init__(self):
        super().__init__()
        self.gate = SpeechGate()
        self.m = TurnMachine(POLICY, segment=self._segment)
        self.s = self.m.initial()
        self.t = 0
        self.committed_answer = ""
        self.spoken: list[str] = []
        self.played_text = ""
        self.commands: list = []
        self.canvas = 0
        self.live_generates: set[str] = set()
        self.live_prefills: set[str] = set()
        self.aborted: set[str] = set()  # criterion 21: one llm.abort per request id
        self.cancelled_this_cause: set[str] = set()  # tts.cancel / playback.flush per cause
        # T-0004 A5: per-turn accounting. ``turn_committed`` is the model's own record of the
        # answer text the LLM committed in the turn now open (criteria 9 and 11 decide what
        # counts); ``turn_spoken`` is every ``TtsSpeak.source`` emitted since the turn opened.
        self.turn_committed = ""
        self.turn_spoken: list[str] = []
        #: turns whose answer was abandoned by a barge-in: nothing of theirs may be voiced again.
        self.barged: set[str] = set()
        #: last utterance id the machine asked TTS for, ever (the tts.done rule replays it)
        self.last_utterance_id: str | None = None
        #: T-0181 A0 (ruling A0-R1: "names an abandoned turn" means content). Every word the model hears carries a
        #: unique tag (``_heard``), so a word's text identifies the one utterance it came from. ``abandoned_words``
        #: is every word text of every turn a ``mic.muted`` abandoned (``words`` and ``next_words`` at the abandon);
        #: ``_step`` checks each ``llm.start`` transcript and each history entry added after an abandon against it
        #: and records every hit in ``abandoned_leaks``, which ``no_answer_or_history_for_an_abandoned_turn``
        #: asserts empty. ``content_checks`` counts the checks made while some abandoned word existed (power).
        self.word_seq = 0
        self.abandoned_words: set[str] = set()
        self.abandoned_leaks: list[tuple[str, str, list[str]]] = []
        self.content_checks = 0

    def _segment(self, text: str, *, final: bool):
        """The T-0004 gate behind T-0003's injected segmenter seam (criterion 10).

        Same adapter shape as ``tests/unit/test_speech_gate.py::gate_segmenter``: the machine
        hands back its whole unspoken tail — the gate's ``pending`` plus the new commit — so only
        the suffix is fed, and whatever the gate returns goes straight to the machine's
        conservation check. The adapter never repairs the gate's output.
        """
        gate = self.gate
        held = gate.pending
        assert text.startswith(held), (
            f"gate.pending {held!r} is not a prefix of the machine's unspoken tail {text!r}: "
            "the turn's gate and the machine's held text have gone out of step"
        )
        suffix = text[len(held) :]
        commit = Commit(request_id="r", canvas_idx=0, segment="answer", text=suffix)
        out = list(gate.feed(commit)) if suffix else []
        if final:
            out.extend(gate.close())
        return [(s.source, s.text) for s in out], gate.pending

    def _step(self, ev):
        before = self.s
        untouched = before.model_copy(deep=True)
        if (
            isinstance(ev, E.LlmCommit)
            and ev.segment == "answer"
            and not ev.thought_open
            and ev.request_id == before.active_generate
            and ev.request_id != repair_id(before.turn_id)
        ):
            # criteria 9 and 11: exactly the commits that extend the answer of the open turn.
            # T-0089 R9: a commit of the repair request is never answer text, so it never joins
            # the ledger the spoken text is checked against (no repair id exists before T-0089).
            self.turn_committed += ev.text
        self.s, cmds = self.m.step(before, ev)
        assert before == untouched, "criterion 22: step mutated the state it was given"
        self.commands.extend(cmds)
        # T-0004 A9 / T-0003 criteria 15 and 18: a barge-in abandons the answer of the turn it
        # cuts. Marked from the event and the resulting state, never from a command the row is
        # supposed to emit, so a row that forgets to interrupt does not silently exempt itself.
        barging = self.s.state is TurnState.INTERRUPTED or (
            isinstance(ev, E.UserTakeover)
            and before.state in (TurnState.THINKING, TurnState.SPEAKING, TurnState.PAUSED)
        )
        for c in cmds:
            if isinstance(c, C.TtsSpeak):
                assert c.source in self.committed_answer, "spoke text the LLM never committed"
                assert not barging and before.turn_id not in self.barged, (
                    f"tts.speak {c.source!r} for turn {before.turn_id}, which the user cut off"
                )
                self.spoken.append(c.source)
                self.turn_spoken.append(c.source)
                self.last_utterance_id = c.utterance_id
            elif isinstance(c, C.LlmStart):
                self.live_generates.add(c.request_id)
            elif type(c).__name__ == "ToolRepairStart":
                # T-0089 P4: the repair is a generate request like any other; the §4 bound of one
                # live generate counts it (so a repair beside a live pass 1 fails the invariant).
                self.live_generates.add(c.request_id)
            elif isinstance(c, C.LlmAbort):
                self.live_generates.discard(c.request_id)
            elif isinstance(c, C.PrefillOpen):
                self.live_prefills.add(c.request_id)
            elif isinstance(c, C.PrefillAbort):
                self.live_prefills.discard(c.request_id)
            if isinstance(c, C.LlmStart):
                self._check_no_abandoned_words("llm.start transcript", c.transcript)
        if self.s.history != before.history and self.s.history:
            # T-0181 A0-R1: ``_append_history`` appends (and trims from the front), so a changed, non-empty
            # history's last entry is the one this step added
            added = self.s.history[-1]
            self._check_no_abandoned_words(
                "history entry", f"{added.user_text} {added.assistant_spoken}"
            )
        if isinstance(ev, E.LlmDone):
            # The only way a request leaves vLLM without an abort: the generation finished by
            # itself (criterion 12). Every other exit has to be *commanded*, and is handled in the
            # command loop above.
            self.live_generates.discard(ev.request_id)
        # A17: the live sets are deliberately *not* cleared by event type. Clearing them on
        # llm.error / client.disconnected / a tts.error that idled asserted the outcome the row is
        # supposed to produce instead of observing it, so the invariant below could only fail if
        # the machine started two requests, never if it forgot to abort one — which is exactly the
        # defect A16 records (deleting llm.abort from the tts.error and client.disconnected rows
        # left all 73 tests green).
        if isinstance(ev, E.PlaybackStopped | E.PlaybackDrained):
            self.played_text += ev.played_text
        if barging:
            self.barged.add(before.turn_id)
        self._command_discipline(before, cmds)
        self._frame_accumulation(before, ev, cmds)
        self._postconditions(before, ev, cmds)
        self._spoken_is_a_prefix_of_what_was_committed(before, ev)
        self._start_of_turn_bookkeeping(before)
        return cmds

    def _spoken_is_a_prefix_of_what_was_committed(self, before, ev):
        """T-0004 A5, the postcondition the "or held by the gate" escape was rejected for.

        Two halves, and the pair is the whole of "nothing committed is silently dropped, nothing
        uncommitted is ever voiced" at the one seam where it is decidable:

        - at *every* step, the concatenated ``tts.speak`` sources of the open turn are a **prefix**
          of the answer text committed in that turn — so the gate may hold a clause back, but it
          can neither reorder, rewrite nor skip over one;
        - once ``llm.done`` has been processed for the live generate from THINKING or SPEAKING,
          that prefix is the **whole** committed answer (criterion 12: the flush is ``final=True``),
          so a gate that holds a sentence forever fails here.

        The PAUSED/INTERRUPTED rows are deliberately outside the equality: criterion 16 forbids
        synthesis while paused and criterion 17 speaks the held text on resume, so equality there
        is owed at the resume, not at ``llm.done``.

        The third clause is what is left of the "a complete committed sentence must be spoken"
        postcondition this replaced: in the states where the machine *does* feed the gate, the
        spoken sources plus whatever the gate is still holding account for the whole committed
        answer, with the gate — a fixture this model owns, not a field of the machine — as the
        oracle for the "still holding" part. So a machine that quietly deferred every sentence to
        the ``llm.done`` flush (first-speech latency, docs/01 §8) fails here and not only on a
        timing bench.
        """
        spoken = "".join(self.turn_spoken)
        assert self.turn_committed.startswith(spoken), (
            f"spoken {spoken!r} is not a prefix of the committed answer "
            f"{self.turn_committed!r} of turn {before.turn_id}"
        )
        if self.s.state in (TurnState.THINKING, TurnState.SPEAKING):
            assert spoken + self.gate.pending == self.turn_committed, (
                f"spoken {spoken!r} + held {self.gate.pending!r} != committed "
                f"{self.turn_committed!r}: the machine is sitting on text the gate gave it"
            )
        if (
            isinstance(ev, E.LlmDone)
            and ev.request_id == before.active_generate
            and before.state in (TurnState.THINKING, TurnState.SPEAKING)
        ):
            assert spoken == self.turn_committed, (
                "llm.done must flush the gate: "
                f"{spoken!r} != committed {self.turn_committed!r} (turn {before.turn_id})"
            )

    def _start_of_turn_bookkeeping(self, before):
        """One gate per turn (T-0004 A9) and a fresh A5 ledger with it.

        A9: the gate is constructed fresh per turn and ``close()`` *speaks*, so a turn that is
        cut off discards its gate instead of closing it. The model therefore drops the object —
        the held clause of the abandoned turn must never surface in the next one.
        """
        if self.s.turn_id != before.turn_id or (
            self.s.state is TurnState.IDLE and before.state is not TurnState.IDLE
        ):
            self.gate = SpeechGate()
            self.turn_committed = ""
            self.turn_spoken = []

    def _frame_accumulation(self, before, ev, cmds):
        """T-0059 criterion 6: what a transition may do to ``turn_phashes``, on every row.

        Stated as properties of the transition, never as an expected tuple, so the model is not
        the accumulation rule written twice:

        - **ordered and append-only** (D1): the entries before the last never change inside a
          turn, so every prefill the engine has already warmed stays a byte-prefix of the next
          request. A FIFO ("drop the oldest") and a "rebuild newest-first" both fail here.
        - **once each, bounded** (criteria 2 and 3): no duplicate, never more than
          ``max_frames_per_turn - 1`` entries.
        - **lost only at a turn boundary** (criterion 4): inside a turn the tuple never shrinks;
          at a boundary it is empty.
        - **nothing else writes it**: only a ``scene.changed`` delivered in LISTENING with a live
          prefill may change it, and a repeat of a screen already recorded changes neither the
          state nor the commands (criterion 2, the cost criterion).
        """
        old = turn_phashes(before)
        new = turn_phashes(self.s)
        slots = frame_slots(self.m.policy)
        assert len(set(new)) == len(new), f"criterion 2: a screen recorded twice: {new}"
        assert len(new) <= slots, (
            f"criterion 3: {new} exceeds the {slots} slot(s) the cap leaves beside the onset frame"
        )
        boundary = self.s.turn_id != before.turn_id or (
            self.s.state is TurnState.IDLE and before.state is not TurnState.IDLE
        )
        if boundary:
            assert new == (), (
                f"criterion 4: a turn boundary clears turn_phashes, got {new} "
                f"({before.state.value}/{before.turn_id} -> {self.s.state.value}/{self.s.turn_id})"
            )
            return
        assert len(new) >= len(old), (
            f"criterion 6: a frame is lost only at a turn boundary; {old} -> {new} on "
            f"{ev.type} in {before.state.value}"
        )
        keep = max(len(new) - 1, 0)
        assert new[:keep] == old[:keep], (
            f"D1: only the last slot may change inside a turn; {old} -> {new} on {ev.type}"
        )
        recording = (
            isinstance(ev, E.SceneChanged)
            and before.state is TurnState.LISTENING
            and before.active_prefill is not None
        )
        if not recording or ev.phash in old:
            assert new == old, f"{ev.type} in {before.state.value} changed {old} -> {new}"
            if recording:
                assert (self.s, list(cmds)) == (before, []), (
                    f"criterion 2: {ev.phash} is already recorded, so the row costs no snapshot "
                    f"and no re-issue; got {kinds(cmds)}"
                )
        else:
            assert new and new[-1] == ev.phash, (
                f"criteria 1 and 3: the newest screen is the live one, in the last slot; "
                f"{old} -> {new} for {ev.phash}"
            )
            assert kinds(cmds) == ["snapshot_frame", "prefill_reissue"], (
                f"criterion 1: a distinct screen is snapshotted then re-issued; got {kinds(cmds)}"
            )

    def _command_discipline(self, before, cmds):
        """Criterion 21: hud.state exactly on a state change, sane timers, one abort per cause."""
        changed = self.s.state is not before.state
        huds = [c for c in cmds if isinstance(c, C.HudState)]
        assert len(huds) == (1 if changed else 0), (before.state, self.s.state, kinds(cmds))
        if changed:
            assert huds[0].state == self.s.state.value
            assert huds[0].turn_id == self.s.turn_id

        names = [t.name for t in self.s.timers]
        assert len(names) == len(set(names)), f"two timers with the same name: {names}"
        expected: dict[str, int | None] = {}
        for c in cmds:
            if isinstance(c, C.ArmTimer):
                expected[c.name] = c.at_ms
            elif isinstance(c, C.CancelTimer):
                expected[c.name] = None
        live = {t.name: t.at_ms for t in self.s.timers}
        for name, at_ms in expected.items():
            if at_ms is None:
                assert name not in live, f"cancel_timer({name}) left the timer armed"
            else:
                assert live.get(name) == at_ms, f"arm_timer({name}) is not in state.timers"

        if changed:  # a state change is a new cause
            self.cancelled_this_cause.clear()
        for c in cmds:
            if isinstance(c, C.LlmAbort):
                assert c.request_id not in self.aborted, f"llm.abort twice for {c.request_id}"
                self.aborted.add(c.request_id)
            elif isinstance(c, C.TtsCancel | C.PlaybackFlush):
                assert c.type not in self.cancelled_this_cause, f"{c.type} twice for one cause"
                self.cancelled_this_cause.add(c.type)

    def _postconditions(self, before, ev, cmds):
        """Progress: the machine must *do* things, not merely stay consistent."""
        k = [c.type for c in cmds]
        if isinstance(ev, E.UserSpeechOnset) and before.state is TurnState.IDLE:
            assert self.s.state is TurnState.LISTENING
            assert "snapshot_frame" in k and "prefill_open" in k
        if (
            isinstance(ev, E.LlmCommit)
            and before.state is TurnState.THINKING
            and ev.segment == "answer"
            and not ev.thought_open
            and ev.request_id == before.active_generate
            and ev.request_id != repair_id(before.turn_id)  # T-0089 R9: not an answer
        ):
            # A4: the row is keyed on the commit, not on speech being produced — with the real
            # gate behind the segmenter seam a commit that ends mid-clause speaks nothing yet.
            # "nothing committed is silently dropped" is asserted where it is decidable instead,
            # in ``_spoken_is_a_prefix_of_what_was_committed`` (T-0004 A5, which rejected the
            # weaker "or held by the gate" escape this line would otherwise have needed).
            assert self.s.state is TurnState.SPEAKING
        if isinstance(ev, E.UserSpeechOnset) and before.state is TurnState.SPEAKING:
            assert self.s.state is TurnState.PAUSED and "playback.pause" in k
        if isinstance(ev, E.LlmError | E.ClientDisconnected):
            assert self.s.state is TurnState.IDLE
        if isinstance(ev, E.TtsError):
            # criterion 20(b): the degrade is unconditional, ending the turn is not. Asserted here
            # as a property so the long traces exercise it from every reachable state, not only
            # the seven hand-built ones in tests/unit/test_turn_edges.py.
            assert [c.mode for c in cmds if isinstance(c, C.Degrade)] == ["text_only"]
            assert self.s.degraded is True
            if before.state in (TurnState.SPEAKING, TurnState.PAUSED, TurnState.INTERRUPTED):
                assert self.s.state is TurnState.IDLE, "tts.error ends a turn being voiced"
            else:
                assert self.s.state is before.state, "tts.error alone never ends a silent turn"
        if self.s.turn_id != before.turn_id:
            # criterion 23 (and contracts/turn.py: these fields are "for the current turn")
            assert self.s.utterances == (), "a new turn starts with no utterances"
            assert self.s.committed_answer == "" and self.s.spoken_text == ""

    def _advance(self, ms: int) -> int:
        self.t += ms
        return self.t

    def _heard(self, word: str) -> str:
        """T-0181 A0-R1: the text of a heard word, tagged ``word~n`` with a per-model counter, so no two
        utterances in a trace share a word text and a word found in a transcript names the utterance it came
        from. The machine never reads a word's text (it only joins them), so the tag changes no row."""
        self.word_seq += 1
        return f"{word}~{self.word_seq}"

    def _check_no_abandoned_words(self, where: str, text: str) -> None:
        """T-0181 A0-R1: ``text`` (an ``llm.start`` transcript or a history entry produced after an abandon) holds
        none of the abandoned turns' words; a hit is recorded for the invariant to report."""
        if not self.abandoned_words:
            return
        self.content_checks += 1
        hits = sorted(set(text.split()) & self.abandoned_words)
        if hits:
            self.abandoned_leaks.append((where, text, hits))

    def _played_prefix(self, frac: float) -> str:
        spoken = "".join(self.spoken)
        return spoken[: int(len(spoken) * frac)]

    @initialize()
    def late_events_naming_a_request_this_session_never_started(self):
        """Three §4 rows delivered on every trace, before anything else (T-0058 A10).

        The rows for ``llm.prefill_ready``, ``llm.done`` and ``tts.done`` are reachable in a trace
        only through a ``@precondition`` — a live prefill, a live generate, an utterance the
        machine has already asked for — and those conditions hold at roughly 18 %, 19 % and 13 % of
        rule boundaries. That was enough while it was measured, but the 100 examples of one
        ``run_state_machine_as_test`` call are *not* independent draws: Hypothesis mutates a shared
        corpus, so a whole run can miss a region of the state space. Measured on this model
        (2026-09-17, hypothesis 6.168, this settings block): over ten seeds the number of
        ``llm.prefill_ready`` deliveries per run swung between 21 and 234 out of ~6 000 events, and
        over thirty seeds of the coverage run below **one** delivered ``llm.done`` and ``tts.done``
        zero times — a red row for a reason that has nothing to do with the rule set. On 2026-09-17 the
        same dice-roll came up against ``llm.prefill_ready`` in a serial whole-suite run, because
        Hypothesis 6.168 harvests integer and float literals from local modules into a
        process-global pool (``hypothesis.internal.constants_ast`` via
        ``providers._get_local_constants``) that feeds every draw, and a module elsewhere in the
        suite had grown some. Reachability of a §4 row must not depend on which literals another
        task happened to write, so this rule delivers those three **by construction**, on every
        trace: the coverage row now states "some rule delivers every §4 event type" and can no
        longer be decided by luck for these three. The conditional rules stay exactly as they were
        — they are what drives the invariants through a *live* prefill, generate and utterance.

        It is not filler. It is the wire case the executor's one pump makes possible and no other
        rule produces: an event naming a request **this session never started** (a turn that died,
        a reconnect, an engine still draining an aborted id). Ids ``t0-*`` are outside the
        machine's own numbering, which starts at ``t1``. What §4 and ``turn_machine`` owe here:
        ``llm.prefill_ready`` carries no row at all, ``llm.done`` is keyed on the live
        ``active_generate`` so a foreign id is inert, and ``tts.done`` sets a per-turn flag that
        ``_open_turn`` clears — so none of the three may start a turn, end one, or emit a command.
        A row that "helpfully" opened a turn on a late prefill would fail here, and so would one
        that ended a turn on a stale ``llm.done``.
        """
        before = self.s
        cmds = self._step(E.PrefillReady(t_ms=self.t, request_id="t0-prefill", prefix_tokens=7))
        assert (self.s, list(cmds)) == (before, []), (
            f"a late llm.prefill_ready naming an unknown request moved the machine to "
            f"{self.s.state.value} / emitted {kinds(cmds)}: §4 gives it no row"
        )

        before = self.s
        cmds = self._step(E.LlmDone(t_ms=self.t, request_id="t0-gen"))
        assert (self.s, list(cmds)) == (before, []), (
            f"a late llm.done naming an unknown request moved the machine to "
            f"{self.s.state.value} / emitted {kinds(cmds)}: it is keyed on active_generate"
        )

        before = self.s
        cmds = self._step(E.TtsDone(t_ms=self.t, utterance_id="t0-u0"))
        assert list(cmds) == [], f"a late tts.done emitted {kinds(cmds)}"
        assert self.s.model_copy(update={"tts_done": before.tts_done}) == before, (
            "a late tts.done changed more than the per-turn tts_done flag"
        )

    @rule(ms=st.integers(1, 800))
    def tick(self, ms):
        self._step(E.Tick(t_ms=self._advance(ms)))

    @rule()
    def onset(self):
        t = self._advance(50)
        self._step(E.UserSpeechOnset(t_ms=t, onset_ms=t))

    @rule(speech_ms=st.integers(50, 1500))
    def offset(self, speech_ms):
        t = self._advance(20)
        self._step(E.UserSpeechOffset(t_ms=t, offset_ms=t, speech_ms=speech_ms))

    @rule(word=st.sampled_from(["what", "is", "this", "ok", "mm-hm"]))
    def word(self, word):
        t = self._advance(200)
        text = self._heard(word)
        self._step(E.SttWord(t_ms=t, text=text, start_ms=max(0, t - 500), end_ms=max(0, t - 300)))

    @rule()
    def end_of_turn(self):
        t = self._advance(100)
        self._step(E.SttEndOfTurn(t_ms=t, t_eot_ms=max(0, t - 100)))

    @rule()
    def flushed(self):
        t = self._advance(5)
        self._step(E.SttFlushed(t_ms=t, t_eot_ms=max(0, t - 300)))

    #: Commit texts. Half of them end mid-clause on purpose: with the real speech gate behind the
    #: segmenter seam (T-0004), only an unfinished clause makes the gate *hold* text, and the A5
    #: postcondition — prefix always, equality on ``llm.done`` — is vacuous without one. The last
    #: pair also opens and closes a parenthesis, the gate's other hold rule (T-0004 A7).
    COMMIT_TEXTS = [
        "Yes. ",
        "The answer is four. ",
        "Look at line two. ",
        "The error is on",
        " line four.",
        "Consider (a. b",
        ") is the one. ",
    ]

    @rule(
        text=st.sampled_from(COMMIT_TEXTS),
        seg=st.sampled_from(["answer", "thought"]),
        open_=st.booleans(),
    )
    @precondition(lambda self: self.s.active_generate is not None)
    def commit(self, text, seg, open_):
        rid = self.s.active_generate
        if seg == "answer" and not open_ and rid != repair_id(self.s.turn_id):
            self.committed_answer += text
        self._step(
            E.LlmCommit(
                t_ms=self._advance(300),
                request_id=rid,
                canvas_idx=self.canvas,
                segment=seg,
                text=text,
                thought_open=open_,
            )
        )
        self.canvas += 1

    @rule()
    @precondition(lambda self: self.s.active_generate is not None)
    def llm_done(self):
        self._step(E.LlmDone(t_ms=self._advance(10), request_id=self.s.active_generate))

    @rule()
    @precondition(lambda self: self.last_utterance_id is not None)
    def tts_done(self):
        """``tts.done`` for the last utterance the machine asked for (criterion 13).

        Keyed on what was *emitted*, not on ``state.utterances`` still holding it: synthesis lags
        playback, so a real ``tts.done`` can arrive after the turn that ordered it has ended, and
        the machine must shrug at a stale one. The narrower precondition also made this rule all
        but dead — the A18 guard measured one delivery in a 200-example run, i.e. criterion 13's
        row was decided by luck.
        """
        self._step(E.TtsDone(t_ms=self._advance(10), utterance_id=self.last_utterance_id))

    @rule(played=st.integers(0, 5000), frac=st.floats(0.0, 1.0))
    def playback_stopped(self, played, frac):
        text = self._played_prefix(frac)
        self._step(E.PlaybackStopped(t_ms=self._advance(20), played_ms=played, played_text=text))

    @rule(played=st.integers(0, 5000))
    def playback_drained(self, played):
        self._step(
            E.PlaybackDrained(
                t_ms=self._advance(20), played_ms=played, played_text="".join(self.spoken)
            )
        )

    @rule()
    def takeover(self):
        t = self._advance(30)
        self._step(E.UserTakeover(t_ms=t, onset_ms=max(0, t - 700)))

    @rule()
    def backchannel(self):
        self._step(E.UserBackchannel(t_ms=self._advance(30), duration_ms=200))

    @rule(phash=st.sampled_from(SCENE_PHASHES))
    def scene_changed(self, phash):
        """T-0059 criterion 6: the hash is drawn, so one turn can see several screens."""
        self._step(E.SceneChanged(t_ms=self._advance(40), source="screen", phash=phash))

    @rule(phashes=st.lists(st.sampled_from(SCENE_PHASHES), min_size=2, max_size=4))
    @precondition(lambda self: self.s.state is TurnState.LISTENING and self.s.active_prefill)
    def scene_changes_in_one_utterance(self, phashes):
        """T-0059 criteria 3 and 6: several screens *inside one utterance*, which is the case.

        Steering the event stream, not the policy — the same device as
        ``tick_to_next_deadline`` below and for the same reason. Measured on the naive
        "append while under the cap, ignore the rest" stub (2026-09-17, hypothesis 6.168,
        ``--hypothesis-seed=0``): with only the single ``scene_changed`` rule, the 40-step run
        never gathered three distinct screens in one LISTENING turn and stayed green, while the
        200-step run caught it — i.e. the cap and the last-slot replacement of D1 were decided by
        how the turn boundaries happened to fall. One to four scene changes inside one utterance
        is exactly what the wire can produce (D2: the re-issue rate limit is 1 000 ms and an
        utterance is 2-5 s), so this rule makes the row reachable rather than lucky.
        """
        for phash in phashes:
            self._step(E.SceneChanged(t_ms=self._advance(40), source="screen", phash=phash))

    @rule(score=st.floats(0.0, 1.0))
    def not_owner(self, score):
        self._step(E.UserNotOwner(t_ms=self._advance(5), score=score))

    @rule()
    @precondition(lambda self: self.s.t_eot_ms is not None)
    def flushed_matching(self):
        """The real moshi Marker: the flush echoes the t_eot the machine is settling."""
        self._step(E.SttFlushed(t_ms=self._advance(5), t_eot_ms=self.s.t_eot_ms))

    @rule(tokens=st.integers(0, 4096))
    @precondition(lambda self: self.s.active_prefill is not None)
    def prefill_ready(self, tokens):
        rid = self.s.active_prefill
        self._step(E.PrefillReady(t_ms=self._advance(5), request_id=rid, prefix_tokens=tokens))

    @rule()
    def stt_error(self):
        """Criterion 20(c). No rule emitted ``stt.error`` at all, so the row was dead in traces."""
        self._step(E.SttError(t_ms=self._advance(5), message=STT_DEAD))

    @rule()
    @precondition(lambda self: bool(self.s.timers))
    def tick_to_next_deadline(self):
        """Tick straight to the earliest armed deadline, whichever timer it belongs to.

        Uniform ticks of 1..800 ms never reach the 20 s ``llm_timeout`` row or the PAUSED
        resume-by-tick row inside a 40-step trace (the reviewer instrumented it: zero visits to
        either, max simulated t ~ 31.9 s). This steers the *event stream*, not the policy — every
        deadline keeps its shipped value, so what those rows prove is unchanged; only whether a
        trace ever gets there. Shrinking ``llm_timeout_ms`` would have been the other option and
        would have quietly changed the regime the row is tested under.
        """
        at = min(t.at_ms for t in self.s.timers)
        self._step(E.Tick(t_ms=self._advance(max(1, at - self.t))))

    @rule(kind=st.sampled_from(["pause", "step", "ui"]), value=st.floats(0.0, 1.0))
    def telemetry(self, kind, value):
        """The three events that carry no §4 row: ``user.pause``, ``stt.step``, ``ui.action``.

        The machine is documented to ignore them (the owner veto and the HUD actions belong to the
        executor, the pause heads are telemetry), and "ignore" is an assertion like any other:
        routed through ``_step`` they are checked for command discipline, purity and
        non-mutation on every trace. They share one rule for the same reason the error rows do —
        so that rows the machine has nothing to say about do not take over the rule space — and
        they exist at all because of the A18 guard below, which is what found them missing.
        """
        t = self._advance(5)
        ev = {
            "pause": E.UserPause(t_ms=t, p_end=value),
            "step": E.SttStep(t_ms=t, pause_heads=(value,)),
            "ui": E.UiAction(t_ms=t, surface_id="s1", name="dismiss"),
        }[kind]
        self._step(ev)

    @rule(text=st.text(min_size=1, max_size=40).map(str.strip).filter(bool))
    @precondition(lambda self: hasattr(E, "UserText"))
    def user_text(self, text):
        """T-0039 criterion 4 / A8: the typed question reaches ``step`` from a rule too.

        ``user.text`` joins ``user.pause``, ``stt.step`` and ``ui.action`` as an event with no §4
        row, and the A18 coverage test below requires every ``EVENT_TYPES`` tag to be delivered by
        some rule — otherwise the invariants (purity, command discipline, bounded state, the
        never-speak-thoughts rule) would never see a trace containing one, and criterion 20's
        "the machine is byte-identical" claim would be untested on mixed traces. It is *not*
        folded into ``telemetry``'s ``sampled_from`` because it carries a payload the others do
        not, and the payload is what an implementer tempted to add a row would read.

        The precondition is the pre-implementation guard: ``contracts/events.py`` has no
        ``UserText`` until group A lands, and while it has none the tag is absent from
        ``EVENT_TYPES`` too, so the coverage test asks nothing of this rule. The moment the
        contract declares it, the rule starts firing and the bar rises by itself.

        No state change is expected: every invariant must still hold across the step, and
        ``_command_discipline`` asserts the empty command list (no ``hud.state`` without a state
        change).
        """
        before = self.s
        self._step(E.UserText(t_ms=self._advance(5), text=text))
        assert self.s == before, (
            f"criterion 20: user.text moved the machine from {before.state.value} to "
            f"{self.s.state.value}"
        )

    @rule(
        call_id=st.sampled_from(["t0-call", "zz-9"]),
        status=st.sampled_from(["dry_run", "executed", "refused", "failed", "timeout"]),
    )
    @precondition(lambda self: hasattr(E, "ToolResult"))
    def tool_result_nobody_awaits(self, call_id, status):
        """T-0112: ``tool.result`` reaches ``step`` from a rule too (the A18 coverage row).

        The ids are outside every model's numbering (calls are ``c<n>``), so no model awaits them:
        with ``tool_loop`` off the event has no row at all (L8), and with it on an unmatched result
        emits nothing (R13). Either way the step is inert, and that is asserted, not assumed.
        """
        before = self.s
        cmds = self._step(E.ToolResult(t_ms=self._advance(5), call_id=call_id, status=status))
        assert (self.s, list(cmds)) == (before, []), (
            f"T-0112 L8/R13: a ToolResult({call_id}) nobody awaits moved the machine from "
            f"{before.state.value} to {self.s.state.value} / emitted {kinds(cmds)}"
        )

    @rule(kind=st.sampled_from(["llm", "tts", "disconnect"]))
    def error(self, kind):
        """The three rows of criterion 20 / A9 that can end a turn from outside it.

        ``client.disconnected`` had no rule at all, so criterion 20(a) was unreachable in traces
        and the ``llm.abort`` it owes (A16) went unchecked; it shares this rule rather than getting
        its own so that turn-ending events do not become a third of the rule space.
        """
        t = self._advance(5)
        ev = {
            "llm": E.LlmError(t_ms=t, message="x"),
            "tts": E.TtsError(t_ms=t, message="x"),
            "disconnect": E.ClientDisconnected(t_ms=t),
        }[kind]
        self._step(ev)

    @rule()
    @precondition(lambda self: hasattr(E, "SessionReset"))
    def session_reset(self):
        """T-0162 A-2: the user's Restart, injected anywhere -- the user always wins.

        From whatever state the trace is in: the machine is IDLE afterwards with ``history == ()``
        and nothing live; every request that was live (by the commands the machine actually
        emitted, the same live sets ``idle_means_no_live_requests`` reads) gets exactly one abort and
        nothing else does; the row speaks nothing and stores nothing (no ``tts.speak``, no
        ``commit_history``, no ``schedule_summary``); ``turn_seq`` and ``first_canvas_ms`` survive.
        Then the wire's stale tail for the cut turn is delivered -- a ``playback.stopped``, an
        ``llm.commit`` and an ``llm.done`` naming the pre-reset request -- and each must emit
        nothing and change nothing: nothing is spoken or stored across a reset.

        The precondition is the pre-implementation guard (the ``user_text`` device): until
        ``contracts/events.py`` declares ``SessionReset`` this rule never fires; the moment it does,
        ``test_every_event_type_is_delivered_by_some_rule_of_the_stateful_model`` requires it.
        """
        before = self.s
        live_generates = set(self.live_generates)
        live_prefills = set(self.live_prefills)
        stale = before.active_generate or f"{before.turn_id}-gen"
        # nothing the cut turn committed may ever be voiced, whatever arrives late
        self.barged.add(before.turn_id)
        cmds = self._step(E.SessionReset(t_ms=self._advance(5)))
        k = kinds(cmds)
        assert self.s.state is TurnState.IDLE, (
            f"A-2: a reset ends in IDLE, not {self.s.state.value}"
        )
        assert self.s.history == () and self.s.next_words == (), (
            f"A-2: a reset clears history and next_words; got {self.s.history}, {self.s.next_words}"
        )
        assert self.s.active_generate is None and self.s.active_prefill is None
        for tag in ("tts.speak", "commit_history", "schedule_summary", "llm.start", "prefill_open"):
            assert tag not in k, f"A-2: the reset row emits no {tag}; got {k}"
        aborted = [c.request_id for c in cmds if isinstance(c, C.LlmAbort)]
        assert sorted(aborted) == sorted(live_generates), (
            f"A-2: exactly one llm.abort per live generate {sorted(live_generates)}; got {aborted}"
        )
        prefill_aborted = [c.request_id for c in cmds if isinstance(c, C.PrefillAbort)]
        assert sorted(prefill_aborted) == sorted(live_prefills), (
            f"A-2: exactly one prefill_abort per live prefill {sorted(live_prefills)}; got "
            f"{prefill_aborted}"
        )
        assert self.s.turn_seq == before.turn_seq, "A-2: turn ids are never reused"
        assert self.s.first_canvas_ms == before.first_canvas_ms, "A-2: the EMA survives"
        resets = [c for c in cmds if c.type == "reset_session"]
        assert len(resets) == 1, f"A-2: one ResetSession per reset; got {k}"

        after = self.s
        late = (
            E.PlaybackStopped(
                t_ms=self._advance(5), played_ms=900, played_text=self._played_prefix(1.0)
            ),
            E.LlmCommit(
                t_ms=self._advance(5),
                request_id=stale,
                canvas_idx=self.canvas,
                segment="answer",
                text="Yes. ",
                thought_open=False,
            ),
            E.LlmDone(t_ms=self._advance(5), request_id=stale),
        )
        self.canvas += 1
        for ev in late:
            cmds = self._step(ev)
            assert list(cmds) == [], f"A-2: a late {ev.type} after a reset emitted {kinds(cmds)}"
            assert self.s == after, f"A-2: a late {ev.type} after a reset moved the machine"

    #: T-0181 A0-2: what the abandon row may never emit
    MIC_MUTED_FORBIDDEN = (
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

    @rule()
    @precondition(lambda self: hasattr(E, "MicMuted"))
    def mic_muted(self):
        """T-0181 A0 (H-T181-2, James 2026-10-28: "just drop it"), injected anywhere.

        In LISTENING, and in ENDPOINTING before the flush (``flush_t_ms is None``, H-T181-8), the open spoken
        question is abandoned (A0-2): IDLE; every live request (by the commands the machine actually emitted, the
        live sets ``idle_means_no_live_requests`` reads) aborted exactly once and nothing else aborted; none of
        ``MIC_MUTED_FORBIDDEN``; ``history``, ``turn_seq`` and ``first_canvas_ms`` unchanged; ``words`` and
        ``next_words`` empty. The turn's words join ``self.abandoned_words`` (``_mark_abandoned``), so
        ``no_answer_or_history_for_an_abandoned_turn`` watches them for the rest of the trace; then the run's stale tail (A0-4) -- its end-of-turn, a Marker echo
        naming its ``t_eot_ms`` and a tick past the settle window -- each move nothing.

        Everywhere else (IDLE, THINKING, SPEAKING, PAUSED, INTERRUPTED, ENDPOINTING after the flush) the step is
        ``(state, ())`` (A0-3; H-T181-9: no row in PAUSED/INTERRUPTED).

        The precondition is the pre-implementation guard (the ``user_text``/``session_reset`` device): until
        ``contracts/events.py`` declares ``MicMuted`` this rule never fires; once it does, the A18 coverage row
        requires it. Inherited by ``ToolTurnModel`` and ``ToolLoopModel``, so the tool invariants see it too.
        """
        before = self.s
        live_generates = set(self.live_generates)
        live_prefills = set(self.live_prefills)
        abandoning = before.state is TurnState.LISTENING or (
            before.state is TurnState.ENDPOINTING and before.flush_t_ms is None
        )
        cmds = self._step(E.MicMuted(t_ms=self._advance(5)))
        k = kinds(cmds)
        if not abandoning:
            assert (self.s, list(cmds)) == (before, []), (
                f"T-0181 A0-3: mic.muted in {before.state.value} (flush_t_ms={before.flush_t_ms}) moved the "
                f"machine to {self.s.state.value} / emitted {k}"
            )
            return
        assert self.s.state is TurnState.IDLE, (
            f"T-0181 A0-2: mic.muted in {before.state.value} abandons the turn; reached {self.s.state.value}"
        )
        for tag in self.MIC_MUTED_FORBIDDEN:
            assert tag not in k, f"T-0181 A0-2: the abandon emits no {tag}; got {k}"
        aborted = [c.request_id for c in cmds if isinstance(c, C.LlmAbort)]
        assert sorted(aborted) == sorted(live_generates), (
            f"T-0181 A0-2: one llm.abort per live generate {sorted(live_generates)}; got {aborted}"
        )
        prefill_aborted = [c.request_id for c in cmds if isinstance(c, C.PrefillAbort)]
        assert sorted(prefill_aborted) == sorted(live_prefills), (
            f"T-0181 A0-2: one prefill_abort per live prefill {sorted(live_prefills)}; got {prefill_aborted}"
        )
        assert self.s.history == before.history, "T-0181 A0-2: the abandon stores nothing"
        assert self.s.turn_seq == before.turn_seq, "T-0181 A0-2: turn ids are never reused"
        assert self.s.first_canvas_ms == before.first_canvas_ms, "T-0181 A0-2: the EMA survives"
        assert self.s.words == () and self.s.next_words == (), (
            f"T-0181 A0-2: words and next_words cleared; got {self.s.words}, {self.s.next_words}"
        )
        self._mark_abandoned(before)

        after = self.s
        t_eot = before.t_eot_ms if before.t_eot_ms is not None else max(0, self.t - 100)
        tail = (
            E.SttEndOfTurn(t_ms=self._advance(5), t_eot_ms=t_eot),
            E.SttFlushed(t_ms=self._advance(5), t_eot_ms=t_eot),
            E.Tick(t_ms=self._advance(POLICY.eot_settle_ms + 1)),
        )
        for ev in tail:
            cmds = self._step(ev)
            assert list(cmds) == [], (
                f"T-0181 A0-4: a late {ev.type} after the abandon emitted {kinds(cmds)}"
            )
            assert self.s == after, (
                f"T-0181 A0-4: a late {ev.type} after the abandon moved the machine"
            )

    @rule(late=st.booleans(), reopen=st.booleans(), endpoint=st.booleans())
    @precondition(lambda self: hasattr(E, "MicMuted") and self.s.state is TurnState.IDLE)
    def question_then_mute(self, late, reopen, endpoint):
        """T-0181 A0-R1 power: drive a spoken question to the states the abandon row owns, holding words, then mute.

        Uniform rule choice seldom mutes a turn that holds words (and more seldom one holding ``next_words``), so
        without this rule ``no_answer_or_history_for_an_abandoned_turn`` would rarely have an abandoned word to look
        for. From IDLE: an onset and a word; with ``endpoint`` an end-of-turn (ENDPOINTING, before the flush); with
        ``late`` a word heard after ``t_eot`` (``next_words``); with ``reopen`` an onset that unfreezes the turn back
        to LISTENING (A0-R2). Then the ``mic_muted`` rule itself, so its postconditions and the stale tail run too.
        Each event goes through ``_step``."""
        onset = self._advance(50)
        self._step(E.UserSpeechOnset(t_ms=onset, onset_ms=onset))
        t = self._advance(400)
        self._step(E.SttWord(t_ms=t, text=self._heard("what"), start_ms=onset, end_ms=onset + 200))
        if endpoint:
            t = self._advance(100)
            self._step(E.SttEndOfTurn(t_ms=t, t_eot_ms=t - 50))
            if late:
                t = self._advance(300)
                self._step(
                    E.SttWord(t_ms=t, text=self._heard("please"), start_ms=t - 200, end_ms=t - 100)
                )
            if reopen:
                t = self._advance(50)
                self._step(E.UserSpeechOnset(t_ms=t, onset_ms=t))
        self.mic_muted()

    #: T-0193 A0: what a stop and the stopped turn's exit never emit (A0-4, A0-5, A0-7)
    USER_STOP_FORBIDDEN = (
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
    #: T-0193 A0: the user_stop rule's power counters (process-wide; reset and read by
    #: ``test_t0193_user_stop_invariants``)
    stop_counts: ClassVar[dict[str, int]] = {}

    def _stop_count(self, key: str) -> None:
        TurnMachineModel.stop_counts[key] = TurnMachineModel.stop_counts.get(key, 0) + 1

    def _stop_marks(self) -> list[tuple[str, int]]:
        """T-0193: (stopped turn, index into ``self.commands`` where its stop's tuple starts), per acting stop."""
        if not hasattr(self, "_t0193_marks"):
            self._t0193_marks: list[tuple[str, int]] = []
        return self._t0193_marks

    def _check_stop_entry(self, pre: MachineState, played: str) -> None:
        """A0-4/A0-5: the entry the stop's turn stored is ``(transcript, played + " [interrupted]" or "[interrupted]",
        interrupted)``; ``played`` is the joined played runs the caller read off the events."""
        from local_astra.contracts.turn import HistoryTurn

        want = HistoryTurn(
            user_text=" ".join(w.text for w in pre.words),
            assistant_spoken=f"{played} [interrupted]" if played else "[interrupted]",
            interrupted=True,
        )
        assert self.s.history and self.s.history[-1] == want, (
            f"T-0193 A0-4/A0-5: the stopped turn stores {want}; history {self.s.history}"
        )

    @rule(
        which=st.sampled_from(["current", "current", "current", "previous", "next", "foreign"]),
        exit_=st.sampled_from(["stopped", "timeout", "none"]),
        frac=st.floats(0.0, 1.0),
    )
    @precondition(lambda self: hasattr(E, "UserStop"))
    def user_stop(self, which, exit_, frac):
        """T-0193 A0 (the HUD Stop button), injected anywhere, naming the current turn or another one.

        A stop for another turn, or in IDLE, LISTENING, ENDPOINTING or INTERRUPTED, is ``(state, ())`` (A0-2). An acting
        stop: every live request (the live sets ``idle_means_no_live_requests`` reads) aborted exactly once, nothing
        else; a pending call cancelled as ``stopped`` and nothing else cancelled; none of ``USER_STOP_FORBIDDEN``; the
        turn id and ``turn_seq`` unchanged. THINKING -> IDLE in the stop's own tuple (``stop_clip`` iff a clip played;
        history ``played_prior`` + the marker, A0-5); SPEAKING/PAUSED -> INTERRUPTED (``tts.cancel``, ``playback.flush``,
        A0-3), then the exit drawn: ``playback.stopped`` or the ``interrupt_timeout`` tick -> IDLE with
        ``truncate_history_to_played`` first and the joined played text + marker stored (A0-4), or none (later rules
        exit it; ``a_stopped_turn_never_speaks_or_relistens`` watches). Then the aborted requests' late ``llm.commit``
        and ``llm.done`` move nothing (A0-6). Inherited by ``ToolTurnModel``/``ToolLoopModel``/``HandBackLoopModel``.
        """
        before = self.s
        seq = before.turn_seq
        turn_id = {
            "current": before.turn_id,
            "previous": f"t{seq - 1}",
            "next": f"t{seq + 1}",
            "foreign": "zz",
        }[which]
        acting = turn_id == before.turn_id and before.state in (
            TurnState.THINKING,
            TurnState.SPEAKING,
            TurnState.PAUSED,
        )
        live_generates = set(self.live_generates)
        live_prefills = set(self.live_prefills)
        pending = before.tool.call_id if getattr(before, "tool", None) is not None else None
        if acting:
            # nothing the stopped turn committed may be voiced again; for the tool models it is cut (L4, P2)
            self.barged.add(before.turn_id)
            for name in ("cut", "tainted"):
                if hasattr(self, name):
                    getattr(self, name).add(before.turn_id)
            self._stop_marks().append((before.turn_id, len(self.commands)))
        cmds = self._step(E.UserStop(t_ms=self._advance(5), turn_id=turn_id))
        k = kinds(cmds)
        if not acting:
            assert (self.s, list(cmds)) == (before, []), (
                f"T-0193 A0-2: a stop for {turn_id!r} in {before.state.value} (turn {before.turn_id}) moved the "
                f"machine to {self.s.state.value} / emitted {k}"
            )
            self._stop_count("noop_wrong_id" if turn_id != before.turn_id else "noop_state")
            return
        for tag in self.USER_STOP_FORBIDDEN:
            assert tag not in k, f"T-0193 A0-3/A0-5: a stop emits no {tag}; got {k}"
        aborted = [c.request_id for c in cmds if isinstance(c, C.LlmAbort)]
        assert sorted(aborted) == sorted(live_generates), (
            f"T-0193 A0-7: one llm.abort per live generate {sorted(live_generates)}; got {aborted}"
        )
        prefill_aborted = [c.request_id for c in cmds if isinstance(c, C.PrefillAbort)]
        assert sorted(prefill_aborted) == sorted(live_prefills), (
            f"T-0193 A0-7: one prefill_abort per live prefill {sorted(live_prefills)}; got {prefill_aborted}"
        )
        cancels = [(c.call_id, c.reason) for c in cmds if type(c).__name__ == "ToolCancel"]
        assert cancels == ([(pending, "stopped")] if pending else []), (
            f"T-0193 A0-3/A0-5: tool.cancel(stopped) iff a call is pending ({pending}); got {cancels}"
        )
        assert (self.s.turn_id, self.s.turn_seq) == (before.turn_id, seq), (
            "T-0193 A0-6: no new turn"
        )
        if before.state is TurnState.THINKING:
            assert self.s.state is TurnState.IDLE, (
                f"T-0193 A0-5: THINKING -> IDLE; got {self.s.state.value}"
            )
            assert ("stop_clip" in k) == before.clip_playing, (
                f"T-0193 A0-5: stop_clip iff a clip plays; {k}"
            )
            self._check_stop_entry(before, before.played_prior)
            self._stop_count("thinking")
            if before.awaiting_result is not None:
                self._stop_count("awaiting")
            if before.active_generate is not None and before.active_generate == repair_id(
                before.turn_id
            ):
                self._stop_count("repair")
            if before.clip_playing:
                self._stop_count("clip")
        else:
            assert self.s.state is TurnState.INTERRUPTED, (
                f"T-0193 A0-3: -> INTERRUPTED; got {self.s.state.value}"
            )
            assert "tts.cancel" in k and "playback.flush" in k, f"T-0193 A0-3: {k}"
            self._stop_count(before.state.value.lower())
            if exit_ == "none":
                return
            pre = self.s
            if exit_ == "stopped":
                text = self._played_prefix(frac)
                ev = E.PlaybackStopped(t_ms=self._advance(20), played_ms=250, played_text=text)
                want = C.TruncateHistoryToPlayed(played_ms=250, played_text=text)
            else:
                deadline = next(t.at_ms for t in pre.timers if t.name == "interrupt_timeout")
                ev = E.Tick(t_ms=self._advance(max(1, deadline - self.t)))
                text = ""
                want = C.TruncateHistoryToPlayed(played_ms=None, played_text="")
            cmds = self._step(ev)
            k = kinds(cmds)
            assert self.s.state is TurnState.IDLE, (
                f"T-0193 A0-4: the stopped exit -> IDLE; got {self.s.state.value}"
            )
            assert (self.s.turn_id, self.s.turn_seq) == (before.turn_id, seq), (
                "T-0193 A0-4: no new turn"
            )
            assert cmds and cmds[0] == want, f"T-0193 A0-4: {want} first; got {k}"
            for tag in self.USER_STOP_FORBIDDEN:
                assert tag not in k, f"T-0193 A0-4: the stopped exit emits no {tag}; got {k}"
            self._check_stop_entry(pre, " ".join(r for r in (pre.played_prior, text) if r))
            self._stop_count(f"exit_{exit_}")
        after = self.s
        for stale in sorted(live_generates):
            for ev in (
                E.LlmCommit(
                    t_ms=self._advance(5),
                    request_id=stale,
                    canvas_idx=self.canvas,
                    segment="answer",
                    text="Yes. ",
                    thought_open=False,
                ),
                E.LlmDone(t_ms=self._advance(5), request_id=stale),
            ):
                cmds = self._step(ev)
                assert list(cmds) == [] and self.s == after, (
                    f"T-0193 A0-6: a late {ev.type} of the aborted {stale} emitted {kinds(cmds)} / moved the machine"
                )
            self.canvas += 1

    @rule(
        text=st.sampled_from(COMMIT_TEXTS),
        answers=st.booleans(),
        pause=st.booleans(),
        exit_=st.sampled_from(["stopped", "timeout", "none"]),
    )
    @precondition(lambda self: hasattr(E, "UserStop") and self.s.state is TurnState.IDLE)
    def question_then_stop(self, text, answers, pause, exit_):
        """T-0193 power: drive a turn to THINKING or SPEAKING (PAUSED with ``pause``), then stop it."""
        self.whole_turn_up_to_thinking_or_speaking(text=text, answers=answers)
        if pause and self.s.state is TurnState.SPEAKING:
            t = self._advance(50)
            self._step(E.UserSpeechOnset(t_ms=t, onset_ms=t))
        self.user_stop(which="current", exit_=exit_, frac=0.5)

    @invariant()
    def a_stopped_turn_never_speaks_or_relistens(self):
        """T-0193 A0-4/A0-6, from the command log: after an acting stop, no command ever names the stopped turn in
        ``tts.speak``, ``llm.start``, ``llm.continue``, ``prefill_open`` or ``tool.execute``; while it is the current
        turn (until a ``hud.state`` names another turn) nothing plays a clip; and the next turn opens only after the
        stopped turn reached IDLE (a relisten would go INTERRUPTED -> LISTENING of a new turn directly)."""
        for turn, start in self._stop_marks():
            current = True
            idle = False
            for c in self.commands[start:]:
                if isinstance(c, C.TtsSpeak):
                    assert not c.utterance_id.startswith(f"{turn}-"), (
                        f"T-0193: {turn} spoke {c} after its stop"
                    )
                elif isinstance(c, C.LlmStart) or is_continue(c):
                    assert not c.request_id.startswith(f"{turn}-"), (
                        f"T-0193: {c} after {turn}'s stop"
                    )
                elif isinstance(c, C.PrefillOpen):
                    assert c.request_id != f"{turn}-prefill", f"T-0193: {c} after {turn}'s stop"
                elif type(c).__name__ == "ToolExecute":
                    assert c.turn_id != turn, f"T-0193: {c} after {turn}'s stop"
                elif isinstance(c, C.PlayClip):
                    assert not current, f"T-0193 A0-6: {c} while the stopped {turn} is current"
                elif isinstance(c, C.HudState) and current:
                    if c.turn_id != turn:
                        assert idle, (
                            f"T-0193 A0-4: {c.turn_id} opened before the stopped {turn} reached IDLE"
                        )
                        current = False
                    elif c.state == TurnState.IDLE.value:
                        idle = True

    @rule(text=st.sampled_from(COMMIT_TEXTS), answers=st.booleans())
    @precondition(lambda self: self.s.state is TurnState.IDLE)
    def whole_turn_up_to_thinking_or_speaking(self, text, answers):
        """Drive the canonical IDLE -> ... -> THINKING (-> SPEAKING) prefix in one rule.

        Uniform rule choice almost never walks it by accident (measured on this model: 4000 steps
        produced zero visits to SPEAKING), so without this rule every invariant and postcondition
        below would only ever see IDLE/LISTENING/ENDPOINTING. Each event still goes through
        ``_step``, so it is checked exactly like any other.

        ``answers=False`` stops at THINKING and leaves the machine there at a rule boundary. Before
        that, the rule always committed in the same step it started the generation, so no trace
        ever *sat* in THINKING and the ``llm_timeout`` row was unreachable however the ticks fell.
        """
        onset = self._advance(50)
        self._step(E.UserSpeechOnset(t_ms=onset, onset_ms=onset))
        t = self._advance(400)
        self._step(E.SttWord(t_ms=t, text=self._heard("what"), start_ms=onset, end_ms=onset + 200))
        t = self._advance(100)
        self._step(E.SttEndOfTurn(t_ms=t, t_eot_ms=t - 50))
        if self.s.t_eot_ms is not None:
            self._step(E.SttFlushed(t_ms=self._advance(5), t_eot_ms=self.s.t_eot_ms))
        self._step(E.Tick(t_ms=self._advance(POLICY.eot_settle_ms + 1)))
        if answers and self.s.active_generate is not None:
            self.committed_answer += text
            self._step(
                E.LlmCommit(
                    t_ms=self._advance(300),
                    request_id=self.s.active_generate,
                    canvas_idx=self.canvas,
                    segment="answer",
                    text=text,
                    thought_open=False,
                )
            )
            self.canvas += 1

    @invariant()
    def at_most_one_generate_and_one_prefill(self):
        assert len(self.live_generates) <= 1, self.live_generates
        assert len(self.live_prefills) <= 1, self.live_prefills
        assert self.s.active_generate in (None, *self.live_generates)
        assert self.s.active_prefill in (None, *self.live_prefills)

    @invariant()
    def never_speak_thoughts(self):
        for c in self.commands:
            if isinstance(c, C.TtsSpeak):
                assert c.source in self.committed_answer

    @invariant()
    def history_never_contains_unplayed_text(self):
        """History holds what the user *heard* (played_text from playback events), never more."""
        for h in self.s.history:
            stored = h.assistant_spoken.replace("[interrupted]", "").strip()
            assert stored == "" or stored in self.played_text, (stored, self.played_text)

    @invariant()
    def idle_means_no_live_requests(self):
        """IDLE means the session owns no vLLM request — not merely that it stopped tracking one.

        ``active_generate`` / ``active_prefill`` are what the machine *believes*; the live sets are
        built from the commands it actually emitted (``llm.start`` / ``prefill_open`` in,
        ``llm.abort`` / ``prefill_abort`` / an ``llm.done`` out), so a row that ends a turn without
        aborting what it started shows up here as an orphan. That is budget-relevant, not cosmetic:
        ``.claude/rules/conductor.md`` caps the session at one generate + one prefill + captioner
        against ``--max-num-seqs 3``, and the next turn's ``llm.start`` would run beside the ghost
        (amendments A16 and A17).
        """
        if self.s.state is TurnState.IDLE:
            assert self.s.active_generate is None and self.s.active_prefill is None
            assert not self.live_generates, (
                f"IDLE with an unaborted generate: {self.live_generates}"
            )
            assert not self.live_prefills, f"IDLE with an unaborted prefill: {self.live_prefills}"

    @invariant()
    def state_stays_bounded(self):
        """Criterion 23: no collection in the machine grows without bound."""
        cap = policy_field("max_buffered_words", 256)
        assert hasattr(self.s, "next_words"), "criterion 4 adds MachineState.next_words"
        assert len(self.s.history) <= POLICY.max_history_turns
        assert len(self.s.words) <= cap
        assert len(self.s.next_words) <= cap
        assert len(self.s.timers) <= 6

    @invariant()
    def frames_are_unique_and_bounded(self):
        """T-0059 criterion 6 beside criterion 23: the new collection is bounded too.

        ``turn_phashes`` is why the machine records hashes and not a counter (T-0059
        implementation notes): a "scene changes seen" integer grows without bound over a long
        LISTENING, and the bound below is what a counter could not satisfy.
        """
        frames = turn_phashes(self.s)
        assert len(frames) <= frame_slots(self.m.policy), frames
        assert len(set(frames)) == len(frames), frames

    @invariant()
    def idle_means_no_frames(self):
        """T-0059 criterion 4: "in IDLE, ``turn_phashes == ()``"."""
        if self.s.state is TurnState.IDLE:
            assert turn_phashes(self.s) == (), (
                f"IDLE carrying the screens of a finished turn: {turn_phashes(self.s)}"
            )

    @invariant()
    def step_is_pure(self):
        """§4 "the machine is pure": two identical calls on one state give one result.

        Purity is relative to the injected segmenter, and this model's segmenter is the stateful
        ``self.gate`` fixture. T-0148 (H-T148-1 option A): each probe call meets its own copy of
        the gate as it was at the probe, the original is restored afterwards, and it is asserted
        unchanged. Without the copies the first call fed the shared gate and the second met a gate
        ahead of the machine's held text (``_segment``'s prefix assertion fired inside the probe),
        or the gate drifted silently. The equality below is unchanged.
        """
        import copy
        import types

        ev = E.Tick(t_ms=self.t + 1)
        gate = self.gate
        snapshot = copy.deepcopy(gate)
        machine = self.m

        def step_with_own_gate(state, event):
            self.gate = copy.deepcopy(snapshot)
            return machine.step(state, event)

        # No try/finally: a failed probe ends the example and the model instance is discarded.
        self.m = types.SimpleNamespace(step=step_with_own_gate)
        assert self.m.step(self.s, ev) == self.m.step(self.s, ev)
        self.m = machine
        self.gate = gate
        assert self.gate == snapshot, (
            f"T-0148: the purity probe moved the model's speech gate: {snapshot!r} -> {self.gate!r}"
        )

    def _mark_abandoned(self, before: MachineState) -> None:
        """T-0181 A0-R1: every word the abandoned turn held (``words`` and ``next_words``, the dropped question's
        tail) joins ``abandoned_words`` for the rest of the trace."""
        self.abandoned_words |= {w.text for w in (*before.words, *before.next_words)}

    @invariant()
    def no_answer_or_history_for_an_abandoned_turn(self):
        """T-0181 A0 (Hypothesis property; ruling A0-R1, by content): no ``llm.start`` transcript and no history
        entry added after a ``mic.muted`` abandon contains any of the abandoned turn's words, at any point of the
        trace. A ``commit_history`` that carries none of them (BASE's IDLE ``_commit_played`` on a stale
        ``playback.drained``/``playback.stopped`` then ``client.disconnected``) is not a violation."""
        assert not self.abandoned_leaks, (
            f"T-0181 A0: an abandoned turn's words reached a later request or history: {self.abandoned_leaks}"
        )


TestTurnMachineInvariants = TurnMachineModel.TestCase
TestTurnMachineInvariants.settings = settings(
    max_examples=200, stateful_step_count=40, deadline=None
)


def test_returns_to_idle_after_done_tts_done_and_drained_within_one_timeout():
    m = TurnMachine(POLICY)
    s, _ = to_thinking(m)
    rid = s.active_generate
    s, _ = m.step(s, commit(s, "Ok."))
    s, _ = m.step(s, E.LlmDone(t_ms=3100, request_id=rid))
    s, _ = m.step(s, E.TtsDone(t_ms=3200, utterance_id=s.utterances[-1]))
    s, _ = m.step(s, E.PlaybackDrained(t_ms=3500, played_ms=300, played_text="Ok."))
    t = 3500
    for _ in range(10):
        if s.state is TurnState.IDLE:
            break
        t += POLICY.llm_timeout_ms // 10
        s, _ = m.step(s, E.Tick(t_ms=t))
    assert s.state is TurnState.IDLE


class TurnMachineLongRun(TurnMachineModel):
    """Same rules, many more steps: criterion 23 (bounded state) needs a long trace."""


TestTurnMachineBoundedState = TurnMachineLongRun.TestCase
TestTurnMachineBoundedState.settings = settings(
    max_examples=20, stateful_step_count=200, deadline=None
)


class EventCoverageModel(TurnMachineModel):
    """Same rules; records which event types a trace actually delivers to ``step`` (A18)."""

    delivered: ClassVar[set[str]] = set()

    def _step(self, ev):
        EventCoverageModel.delivered.add(ev.type)
        return super()._step(ev)


def test_every_event_type_is_delivered_by_some_rule_of_the_stateful_model():
    """T-0003 A18: the invariants' reach is bounded by the rule set, so pin the rule set.

    A17 made ``idle_means_no_live_requests`` able to catch a forgotten ``llm.abort`` on any row —
    but only on rows a trace reaches. ``client.disconnected`` had no rule at all and the
    ``llm_timeout`` row was unreachable however the ticks fell, so two §4 rows were unchecked
    while looking checked. A row added to §4 later with no rule would be invisible again.

    Mechanism: **instrument a run**. ``EventCoverageModel`` records the ``type`` of every event a
    rule hands to ``TurnMachine.step``, and the required set is read from
    ``contracts.events.EVENT_TYPES`` at call time. Neither side is written down here, so the test
    cannot pass vacuously: an event type counts only once a rule has actually constructed it and
    driven the machine with it (a constructor in dead code, or in a rule whose ``@precondition``
    is never true, contributes nothing), and a type added to the contract raises the bar by
    itself. The static alternative — parsing the rule bodies for ``E.X(...)`` — was rejected for
    exactly that: it counts a rule that can never fire.

    ``derandomize=True`` makes the run reproducible independently of ``--hypothesis-seed``. It is
    not enough on its own: the seed it fixes is only one input to the draws, and Hypothesis 6.168
    also feeds them a process-global pool of integer and float literals harvested from every
    *local* module in ``sys.modules``, so the traces this run produces depend on which other
    modules the suite has imported. T-0058 A10: that is how a serial whole-suite run on
    2026-09-17 made ``llm.prefill_ready`` go missing while the module passed alone. Three event
    types were reachable only through a ``@precondition`` and so only by luck; the
    ``@initialize`` rule on ``TurnMachineModel`` now delivers them on every trace, and what this
    row still catches is the case it was written for — a §4 event type that **no** rule of the
    model constructs at all.
    """
    EventCoverageModel.delivered.clear()
    run_state_machine_as_test(
        EventCoverageModel,
        settings=settings(
            max_examples=100, stateful_step_count=60, deadline=None, derandomize=True
        ),
    )
    required = {model.model_fields["type"].default for model in E.EVENT_TYPES}
    missing = required - EventCoverageModel.delivered
    assert not missing, (
        f"no rule of TurnMachineModel ever delivers {sorted(missing)}: those §4 rows are "
        "unreachable in every trace, so no stateful invariant can catch a defect on them"
    )
    assert EventCoverageModel.delivered <= required, (
        f"delivered an event type that is not in EVENT_TYPES: "
        f"{sorted(EventCoverageModel.delivered - required)}"
    )


def test_returns_to_idle_when_drained_arrives_before_done():
    """Criterion 13: the mirrored arrival order ends the turn the same way."""
    m = TurnMachine(POLICY)
    s, _ = to_thinking(m)
    rid = s.active_generate
    s, _ = m.step(s, commit(s, "Ok."))
    s, cmds = m.step(s, E.PlaybackDrained(t_ms=3400, played_ms=300, played_text="Ok."))
    assert s.state is TurnState.SPEAKING, "llm.done has not arrived yet"
    assert "commit_history" not in kinds(cmds)
    s, cmds = m.step(s, E.LlmDone(t_ms=3500, request_id=rid))
    assert s.state is TurnState.IDLE
    k = kinds(cmds)
    assert k.index("commit_history") < k.index("schedule_summary")
    assert next(c for c in cmds if isinstance(c, C.CommitHistory)).user_only is False
    assert s.history[-1].assistant_spoken == "Ok."
    assert s.history[-1].user_text == "what is this"
    assert s.history[-1].interrupted is False
    assert s.history[-1].frame_phash is None, "the executor fills the phash (T-0005/T-0007)"
    assert s.active_generate is None and s.active_prefill is None


def test_init_keeps_policy_positional_and_adds_a_keyword_only_segmenter():
    """Criteria 10 and 24: the injection point exists, session.py keeps constructing TurnMachine(policy)."""
    params = list(inspect.signature(TurnMachine.__init__).parameters)
    assert params[:2] == ["self", "policy"]
    assert segment_parameter().kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    "liar",
    [inventing_segmenter, dropping_segmenter, reordering_segmenter],
    ids=["invents", "drops", "reorders"],
)
def test_a_lying_segmenter_can_never_voice_uncommitted_text(liar):
    """Criterion 10, adversarial: the machine polices its segmenter instead of trusting it.

    The failure mode this targets is the speech path voicing whatever an upstream helper hands
    back (CLAUDE.md: "never voice text the LLM has not committed").
    """
    segment_parameter()
    m = TurnMachine(POLICY, segment=liar)
    s, _ = to_thinking(m)
    before = s.model_copy(deep=True)
    with pytest.raises(ValueError):
        m.step(s, commit(s, "The stack trace points at line four."))
    assert s == before, "a rejected commit leaves the state untouched"


def test_no_new_event_or_command_types():
    """Criterion 24 (negative, regression guard): the wire contracts do not grow for T-0003.

    Frozen by hand from contracts/events.py and contracts/commands.py on main; it passes today
    on purpose and fails the moment somebody adds, renames or removes a type (e.g. a
    ``concatenate_transcript`` command, which criterion 15 says must stay internal state).
    """
    expected_events = sorted(
        [
            "client.disconnected",
            "llm.commit",
            "llm.done",
            "llm.error",
            "llm.prefill_ready",
            "playback.drained",
            "playback.stopped",
            "scene.changed",
            "stt.end_of_turn",
            "stt.error",
            "stt.flushed",
            "stt.step",
            "stt.word",
            "tick",
            "tts.done",
            "tts.error",
            "ui.action",
            "user.backchannel",
            "user.not_owner",
            "user.pause",
            "user.speech_offset",
            "user.speech_onset",
            "user.takeover",
            # T-0039 criterion 4: the typed question arrives as an event so that it is ordered
            # against the frames, ticks and microphone onsets already on the one pump. It carries
            # **no row** — the row below proves that — so the machine is still byte-identical
            # (criterion 20). This line goes red until contracts/events.py declares UserText.
            "user.text",
        ]
    )
    expected_commands = sorted(
        [
            "arm_timer",
            "cancel_timer",
            "captioner.abort",
            "check_owner",
            "commit_history",
            "degrade",
            "freeze_after",
            "hud.error",
            "hud.state",
            "llm.abort",
            "llm.start",
            "play_clip",
            "playback.flush",
            "playback.pause",
            "playback.resume",
            "prefill_abort",
            "prefill_open",
            "prefill_reissue",
            "schedule_summary",
            "snapshot_frame",
            "stop_clip",
            "truncate_history_to_played",
            "tts.cancel",
            "tts.speak",
            "unfreeze",
        ]
    )
    # T-0089 criterion 1 grows the command union by exactly three members; ruling T3 names their
    # wire tags in the dotted style of llm.start and tts.speak.
    expected_commands = sorted(
        [*expected_commands, "tool.repair_start", "tool.execute", "tool.cancel"]
    )
    # T-0112 (T-0111a §2) grows each union by exactly one member: the event ``ToolResult`` and the
    # command ``LlmContinue``. The spec names the classes, not their wire tags; these tags follow
    # ruling T3's dotted style and the session log's own ``tool.result`` record (T-0111a §5).
    expected_events = sorted([*expected_events, "tool.result"])
    expected_commands = sorted([*expected_commands, "llm.continue"])
    # T-0162 A-2, the named spec change: the session reset adds exactly one event (SessionReset)
    # and one command (ResetSession).
    expected_events = sorted([*expected_events, "session.reset"])
    expected_commands = sorted([*expected_commands, "reset_session"])
    # T-0181 A0-1 (H-T181-2, the named spec change): the mic mute adds exactly one event (MicMuted) and no command.
    expected_events = sorted([*expected_events, "mic.muted"])
    # T-0193 A0-1 (the HUD Stop button, the named spec change): exactly one event (UserStop) and no command.
    expected_events = sorted([*expected_events, "user.stop"])
    assert sorted(m.model_fields["type"].default for m in E.EVENT_TYPES) == expected_events
    command_types = C.Command.__args__[0].__args__  # type: ignore[attr-defined]
    assert sorted(m.model_fields["type"].default for m in command_types) == expected_commands


def test_turn_machine_module_has_no_impure_or_gate_imports():
    """Criteria 22 and 24 (regression guard): pure module, no speech gate, no filler policy.

    Passes today (the stub imports nothing banned); it is here so the implementation cannot
    reach for a clock, a logger, a uuid or T-0004's gate on its way to green.
    """
    source = pathlib.Path(turn_machine_module.__file__).read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    banned = {"asyncio", "time", "random", "uuid", "logging", "os", "pathlib", "httpx"}
    assert not banned & {name.split(".")[0] for name in imported}, sorted(imported)
    forbidden_modules = ("local_astra.conductor.speech_gate", "local_astra.reflex.filler_policy")
    assert not [n for n in imported if n.startswith(forbidden_modules)], sorted(imported)


# --- T-0039 group A: the typed question carries no row (criteria 4 and 20) ----------------------
#
# ``turn_machine.py`` is sha-pinned and must have **zero diff** (criterion 20). ``user.text``
# therefore has to fall through the existing no-row branch to ``(state, [])`` in *every* state —
# including the ones a typed question can actually arrive in (THINKING, SPEAKING, PAUSED), where
# a row that "helpfully" opened a turn would abort a generation mid-answer.
#
# The state space is the pure machine's whole input domain, so the row below is a Hypothesis
# property over it rather than four hand-picked states (.claude/rules/tests.md).

TEXT_STATES = list(TurnState)


def user_text_event(t_ms: int, text: str):
    cls = getattr(E, "UserText", None)
    assert cls is not None, (
        'T-0039 criterion 4: contracts/events.py gains UserText(type="user.text", t_ms, text)'
    )
    return cls(t_ms=t_ms, text=text)


@pytest.mark.parametrize("state", TEXT_STATES, ids=[s.value for s in TEXT_STATES])
def test_user_text_has_no_row_in_any_state(state):
    """Criteria 4 and 20: identical state, no command, in all seven states.

    Hand-computed expectation: the machine's transition table (docs/02 §4) has no ``user.text``
    row, so the output is exactly the input and an empty command list — the same contract
    ``ui.action``, ``user.pause`` and ``stt.step`` already have.
    """
    machine = TurnMachine(POLICY)
    before = MachineState(
        state=state,
        turn_id="t7",
        turn_seq=7,
        onset_ms=1000,
        words=(Word(text="what", start_ms=1000, end_ms=1200),),
        committed_answer="It is a stack trace. ",
        spoken_text="It is a stack trace. ",
        active_generate="r1",
        llm_start_ms=2000,
    )
    frozen = before.model_copy(deep=True)
    after, cmds = machine.step(before, user_text_event(3000, "what is on my screen"))
    assert cmds == [] or tuple(cmds) == (), (
        f"criterion 4: user.text carries no row, so it emits no command; it emitted {kinds(cmds)}"
    )
    assert after == frozen, (
        "criterion 4: …and returns the state it was given, unchanged. The machine went from "
        f"{frozen} to {after}"
    )
    assert before == frozen, "the machine is pure: the input state is not mutated in place"


@settings(max_examples=50, deadline=None)
@given(
    state=st.sampled_from(TEXT_STATES),
    t_ms=st.integers(min_value=0, max_value=10**7),
    text=st.text(min_size=1, max_size=40),
)
def test_user_text_never_moves_the_machine_whatever_it_says(state, t_ms, text):
    """Criterion 20 as a property: no text and no arrival time buys a transition.

    The failure this guards is an implementer reaching into ``turn_machine.py`` "just to open the
    turn from the event" — a diff there escalates the task to two reviewers (§7) and would put a
    clock-shaped decision inside the pure machine.
    """
    machine = TurnMachine(POLICY)
    before = MachineState(state=state, turn_id="t1", turn_seq=1)
    after, cmds = machine.step(before, user_text_event(t_ms, text))
    assert (after, list(cmds)) == (before, []), (
        f"criterion 20: user.text in {state.value} produced {kinds(cmds)} and state "
        f"{after.state.value}"
    )


# --- T-0089: the tool turn (criterion 3, P1-P9) ------------------------------------------------------
#
# Every property below is read off docs/tasks/T-0089.md criterion 3 and the umbrella's H5
# (docs/tasks/T-0080d.md: execute only after the spoken prefix has fully played and only if the turn
# was not interrupted). The model keeps its own ledger of what the *events* said — which drafts were
# delivered, which turn was cut, errored or timed out — and checks the *commands* against it, so a
# row that forgets to cancel is caught by the ledger rather than exempted by it.
#
# Both directions of each guard: P2/P3 say when ``ToolExecute`` must not appear; ``_turn_closes``
# says that a valid, untainted call of a turn that ended *must* have been executed (a machine that
# cancels everything satisfies P2 and fails there).
#
# Regime: CPU, pure; Hypothesis settings as for ``TestTurnMachineInvariants`` (200 examples x 40
# steps for ``TestToolTurnInvariants``; 200 examples of up to 60 events for P9).

TOOL_NAMES = ("click_at", "google_search")
TOOL_ARGS = ({"x": 1, "y": 2}, {"query": "weather"})
REPAIR_TIMEOUT_MS = 3000  # H12; TurnPolicy.tool_repair_timeout_ms (criterion 1)
TERMINAL = ("ToolExecute", "ToolCancel")


def tool_draft(**fields):
    from local_astra.contracts import turn as turn_contracts

    cls = getattr(turn_contracts, "ToolDraft", None)
    assert cls is not None, "T-0089 criterion 1: contracts/turn.py gains ToolDraft"
    return cls(**fields)


def llm_done(t_ms: int, request_id: str, tool=None):
    assert "tool" in E.LlmDone.model_fields, "T-0089 criterion 1: events.LlmDone gains tool"
    return E.LlmDone(t_ms=t_ms, request_id=request_id, tool=tool)


def named(cmds, name: str) -> list:
    return [c for c in cmds if type(c).__name__ == name]


class ToolTurnModel(TurnMachineModel):
    """``TurnMachineModel`` plus tool drafts on ``llm.done``, ``tool`` commits and repairs.

    Every T-0003/T-0004/T-0059 invariant of the base model runs on these traces unchanged.
    """

    def __init__(self):
        super().__init__()
        self.call_seq = 0
        self.marker_seq = 0
        #: P6: text of every ``tool`` commit and every repair commit; none may be voiced
        self.unspeakable: list[str] = []
        self.valid_drafts: dict[str, list[tuple[str, str, dict, str]]] = {}
        self.opened: dict[str, list[str]] = {}  # turn -> call ids that became pending
        self.status: dict[str, str] = {}  # call id -> valid | repairable | refused | failed
        self.terminals: dict[str, list] = {}  # call id -> ToolExecute/ToolCancel commands
        self.tainted: set[str] = set()  # turns that may never execute (P2)
        self.executes: dict[str, int] = {}
        self.repairs: dict[str, int] = {}
        self.repair_deadline: dict[str, int] = {}
        self.drained_in_turn: set[str] = set()
        self.closed: set[str] = set()  # turns whose boundary has been checked
        #: ruling T8: a turn whose repair failed by llm.error while speaking; the next
        #: playback.drained in SPEAKING must complete it
        self.r7_awaiting_drain: str | None = None

    # --- ruling T8: R7 replaces "any llm.error ends in IDLE" while a repair is live ---------------

    def _r7_applies(self, before, ev) -> bool:
        return (
            isinstance(ev, E.LlmError)
            and self._repair_live(before)
            and ev.request_id in (None, before.active_generate)
        )

    def _postconditions(self, before, ev, cmds):
        """T8: exactly an accepted ``llm.error`` taken while a repair was live is exempt from the
        base model's "any llm.error ends in IDLE"; R7 is required in its place. Every other event,
        and every ``llm.error`` with no repair live, goes through the base postconditions.

        T9 corrects T8 for SPEAKING after the drain: there "complete as today" ends the turn at
        once. In PAUSED a drained turn completes at the resume, as a toolless ``llm.done`` in
        PAUSED does on main, so PAUSED keeps T8's "state unchanged"."""
        if not self._r7_applies(before, ev):
            return super()._postconditions(before, ev, cmds)
        repair = before.active_generate
        cancels = [(c.call_id, c.reason) for c in named(cmds, "ToolCancel")]
        assert cancels and all(reason == "repair_failed" for _, reason in cancels), (
            f"T8/R7: llm.error during the repair owes ToolCancel(repair_failed); got {cancels}"
        )
        assert C.LlmAbort(request_id=repair) in cmds, (
            f"T8: LlmAbort({repair}) missing: {kinds(cmds)}"
        )
        assert not named(cmds, "ToolExecute"), "T8: no ToolExecute in that turn"
        assert not [c for c in cmds if isinstance(c, C.HudError)], "R7: the turn is not failed"
        if before.state is TurnState.THINKING:
            assert self.s.state is TurnState.IDLE, "R7 in THINKING completes as today"
        elif before.state is TurnState.SPEAKING and before.playback_drained:
            # Ruling T9 (correction to T8): the spoken prefix had already drained, so R7's
            # "complete as today" ends the turn at once, exactly as a toolless llm.done in
            # SPEAKING after the drain does (criterion 13). Counterexample that T8 failed:
            # whole_tool_turn(slot=0, status='repairable'), playback_drained, error(kind='llm').
            assert self.s.state is TurnState.IDLE, (
                f"T9: R7 after the drain completes the turn; stayed {self.s.state.value}"
            )
            assert C.CommitHistory(user_only=False) in cmds, (
                f"T9: the completion commits what was played: {kinds(cmds)}"
            )
            assert [c for c in cmds if isinstance(c, C.ScheduleSummary)], (
                f"T9: the completion schedules the summary: {kinds(cmds)}"
            )
        else:
            assert self.s.state is before.state, (
                f"R7: speech keeps playing; {before.state.value} -> {self.s.state.value}"
            )
            self.r7_awaiting_drain = before.turn_id
        if self.s.turn_id != before.turn_id:
            assert self.s.utterances == () and self.s.committed_answer == ""

    # --- ledger helpers ------------------------------------------------------------------------

    def _pending_calls(self, turn: str) -> list[str]:
        return [c for c in self.opened.get(turn, []) if not self.terminals.get(c)]

    def _repair_live(self, state) -> bool:
        return state.active_generate is not None and state.active_generate == repair_id(
            state.turn_id
        )

    @staticmethod
    def _paused_timer_due(before, ev) -> bool:
        """Ruling T4: a PAUSED tick checks resume, then takeover, and only then tool_repair."""
        return any(t.name in ("resume", "takeover") and ev.t_ms >= t.at_ms for t in before.timers)

    def _expected_cancel(self, before, ev) -> set[str] | None:
        """The ToolCancel reason(s) a row owes when a call is pending (R6, R7, R8)."""
        if type(ev).__name__ == "UserStop":
            # T-0193 A0-3/A0-5: an acting stop cancels a pending call as "stopped" (and the turn never executes)
            acting = ev.turn_id == before.turn_id and before.state in (
                TurnState.THINKING,
                TurnState.SPEAKING,
                TurnState.PAUSED,
            )
            return {"stopped"} if acting else None
        speaking = (TurnState.SPEAKING, TurnState.PAUSED, TurnState.INTERRUPTED)
        if self.s.state is TurnState.INTERRUPTED and before.state is not TurnState.INTERRUPTED:
            return {"interrupted"}
        if isinstance(ev, E.UserTakeover) and before.state is TurnState.THINKING:
            return {"takeover"}
        if isinstance(ev, E.ClientDisconnected):
            return {"disconnected"}
        if type(ev).__name__ == "SessionReset":
            return {
                "reset"
            }  # T-0162 A-2: _tool_cancel(state, "reset"), and the turn never executes
        if isinstance(ev, E.TtsError) and before.state in speaking:
            return {"error"}
        if isinstance(ev, E.LlmError):
            live = self._repair_live(before)
            if ev.request_id is not None and ev.request_id != before.active_generate:
                return None  # stale (R10)
            if live:  # R7; ruling T5: a None id resolves to active_generate, the repair
                return {"repair_failed"}
            return {"error"}
        if (
            isinstance(ev, E.Tick)
            and before.state in (TurnState.THINKING, TurnState.SPEAKING, TurnState.PAUSED)
            and self._repair_live(before)
            and ev.t_ms >= self.repair_deadline.get(before.turn_id, 10**12)
            and not (before.state is TurnState.PAUSED and self._paused_timer_due(before, ev))
        ):
            return {"repair_timeout"}  # R6; ruling T4: in PAUSED after resume and takeover
        if (
            isinstance(ev, E.Tick)
            and before.state is TurnState.THINKING
            and not self._repair_live(before)
        ):
            deadline = next((t.at_ms for t in before.timers if t.name == "llm_timeout"), None)
            if deadline is not None and ev.t_ms >= deadline:
                return {"error"}
        return None

    # --- the checks ----------------------------------------------------------------------------

    def _step(self, ev):
        before = self.s
        cmds = list(super()._step(ev))
        turn = before.turn_id
        accepted = isinstance(ev, E.LlmDone) and ev.request_id == before.active_generate
        from_repair = accepted and ev.request_id == repair_id(turn)
        tool = getattr(ev, "tool", None) if accepted else None

        # P6 (extends "never tts.speak uncommitted")
        for c in cmds:
            if isinstance(c, C.TtsSpeak):
                for marker in self.unspeakable:
                    assert marker not in c.source, (
                        f"P6: voiced {c.source!r}, which holds tool/repair commit {marker!r}"
                    )

        if self.r7_awaiting_drain is not None and self.r7_awaiting_drain != before.turn_id:
            self.r7_awaiting_drain = None
        if (
            isinstance(ev, E.PlaybackDrained)
            and self.r7_awaiting_drain == turn
            and before.state is TurnState.SPEAKING
        ):
            assert self.s.state is TurnState.IDLE, (
                f"T8: the turn whose repair failed must complete on the next drain; "
                f"stayed {self.s.state.value}"
            )
            self.r7_awaiting_drain = None
        if isinstance(ev, E.PlaybackDrained) and before.state in (
            TurnState.SPEAKING,
            TurnState.PAUSED,
        ):
            self.drained_in_turn.add(turn)

        # what the events say about this turn (P2's list), before looking at the commands
        expected_cancel = self._expected_cancel(before, ev)
        if expected_cancel is not None and before.state is not TurnState.IDLE:
            self.tainted.add(turn)
        if self._repair_live(before) and (
            (isinstance(ev, E.LlmError) and ev.request_id in (None, before.active_generate))
            or expected_cancel == {"repair_timeout"}
        ):
            for call in self._pending_calls(turn):
                self.status[call] = "failed"
        if from_repair and (tool is None or tool.status != "valid"):
            for call in self._pending_calls(turn):
                self.status[call] = "failed"  # R4

        if tool is not None:
            self.status[tool.call_id] = tool.status
            if tool.status == "valid":
                via = "repair" if from_repair else tool.source
                self.valid_drafts.setdefault(turn, []).append(
                    (tool.call_id, tool.name, tool.arguments, via)
                )

        # P4 / P5 / R2
        starts = named(cmds, "ToolRepairStart")
        for r in starts:
            assert accepted and tool is not None and tool.status == "repairable", (
                f"P4: ToolRepairStart only answers a repairable llm.done; got {ev.type}"
            )
            assert not from_repair, "R4: a repair is never repaired"
            assert r.request_id == repair_id(turn) and r.draft_request_id == ev.request_id
            assert r.call_id == tool.call_id
            assert self.live_generates == {r.request_id}, (
                f"P4: repair started while {self.live_generates - {r.request_id}} is live"
            )
            self.repairs[turn] = self.repairs.get(turn, 0) + 1
            assert self.repairs[turn] <= 1, f"P5: two ToolRepairStart in turn {turn}"
            self.repair_deadline[turn] = ev.t_ms + REPAIR_TIMEOUT_MS
        if (
            accepted
            and not from_repair
            and tool is not None
            and tool.status == "repairable"
            and before.state in (TurnState.THINKING, TurnState.SPEAKING, TurnState.PAUSED)
            and self.repairs.get(turn, 0) == (1 if starts else 0)
        ):
            assert len(starts) == 1, f"R2: a first repairable draft starts a repair: {kinds(cmds)}"

        # ruling T4: PAUSED is a speaking state, so a draft delivered there is pending too
        opens = (
            accepted
            and tool is not None
            and before.state in (TurnState.THINKING, TurnState.SPEAKING, TurnState.PAUSED)
        )
        if opens and tool.call_id not in self.opened.setdefault(turn, []):
            self.opened[turn].append(tool.call_id)

        # R8 / R6 / R7, the must-happen direction: a pending call is cancelled with the reason
        pending = self._pending_calls(turn)
        if expected_cancel is not None and pending:
            got = {(c.call_id, c.reason) for c in named(cmds, "ToolCancel")}
            for call in pending:
                assert any(cid == call and reason in expected_cancel for cid, reason in got), (
                    f"R8: {ev.type} in {before.state.value} owes ToolCancel({call}, "
                    f"{sorted(expected_cancel)}); got {sorted(got)}"
                )

        # P8, at most one terminal per call; P1/P2/P3 on every ToolExecute
        for c in cmds:
            if type(c).__name__ in TERMINAL:
                assert c.call_id in self.opened.get(turn, []), (
                    f"T7: {type(c).__name__}({c.call_id}) names no pending call of turn {turn} "
                    f"({self.opened.get(turn, [])})"
                )
                self.terminals.setdefault(c.call_id, []).append(c)
                assert len(self.terminals[c.call_id]) == 1, (
                    f"P8: call {c.call_id} ended twice: {kinds(self.terminals[c.call_id])}"
                )
        for ex in named(cmds, "ToolExecute"):
            assert ex.turn_id == turn, f"P1: executed {ex.turn_id}'s call during {turn}"
            self.executes[turn] = self.executes.get(turn, 0) + 1
            assert self.executes[turn] == 1, f"P1: two ToolExecute in turn {turn}"
            drafts = self.valid_drafts.get(turn, [])
            match = [d for d in drafts if (d[0], d[1], d[2]) == (ex.call_id, ex.name, ex.arguments)]
            assert match, f"P1: {ex} is not a valid draft of {turn}: {drafts}"
            if match[-1][3] == "repair":
                assert ex.source == "repair", "R3: a repaired call executes with source repair"
            elif from_repair is False and match[-1][3] == "draft":
                assert ex.source == "draft"
            assert turn not in self.tainted, (
                f"P2: ToolExecute in turn {turn}, which was interrupted, taken over, errored, "
                f"disconnected or hit tool_repair (event {ev.type})"
            )
            assert self.s.state is TurnState.IDLE, f"P3: executed into {self.s.state.value}"
            # Ruling T11: P3 is enforced -- never execute in a tuple that speaks. A TtsSpeak in
            # the executing tuple is speech that has not been played, let alone drained (H5).
            speaks = [c.source for c in cmds if isinstance(c, C.TtsSpeak)]
            assert not speaks, (
                f"T11/P3: ToolExecute({ex.call_id}) in the same tuple as TtsSpeak {speaks} "
                f"(event {ev.type} in {before.state.value}): the tool ran before that speech played"
            )
            i = cmds.index(ex)
            hist = [j for j, c in enumerate(cmds) if isinstance(c, C.CommitHistory)]
            assert hist and i < hist[0], (
                f"P3: ToolExecute before this turn's CommitHistory: {kinds(cmds)}"
            )
            for j, c in enumerate(cmds):
                if isinstance(c, C.ScheduleSummary):
                    assert i < j, kinds(cmds)
            if before.state in (TurnState.SPEAKING, TurnState.PAUSED):
                assert turn in self.drained_in_turn, (
                    "P3/H5: executed while that turn's spoken prefix had not drained"
                )

        # turn boundary: P8 exactly one terminal, and which one
        boundary = self.s.turn_id != turn or (
            self.s.state is TurnState.IDLE and before.state is not TurnState.IDLE
        )
        if boundary and turn not in self.closed:
            self.closed.add(turn)
            for call in self.opened.get(turn, []):
                ends = self.terminals.get(call, [])
                assert len(ends) == 1, f"P8: call {call} of turn {turn} ended {len(ends)} times"
                should_run = self.status.get(call) == "valid" and turn not in self.tainted
                kind = type(ends[0]).__name__
                assert kind == ("ToolExecute" if should_run else "ToolCancel"), (
                    f"H5: call {call} (status {self.status.get(call)}, tainted "
                    f"{turn in self.tainted}) ended in {kind}"
                )
        return cmds

    # --- rules ---------------------------------------------------------------------------------

    def _draft(self, status, source, slot):
        if self._repair_live(self.s):
            call_id = (self._pending_calls(self.s.turn_id) or [f"c{self.call_seq}"])[0]
        else:
            self.call_seq += 1
            call_id = f"c{self.call_seq}"
        name, args = TOOL_NAMES[slot], TOOL_ARGS[slot]
        if status == "refused":
            return tool_draft(
                call_id=call_id,
                status=status,
                name=None,
                arguments=None,
                reason="r",
                source=source,
            )
        if status == "repairable":
            return tool_draft(
                call_id=call_id,
                status=status,
                name=name,
                arguments=None,
                reason="schema",
                source="draft",
            )
        return tool_draft(
            call_id=call_id, status=status, name=name, arguments=args, reason=None, source=source
        )

    @rule(
        status=st.sampled_from(["valid", "repairable", "refused"]),
        source=st.sampled_from(["draft", "repair"]),
        slot=st.integers(0, 1),
    )
    @precondition(lambda self: self.s.active_generate is not None)
    def llm_done_with_tool(self, status, source, slot):
        tool = self._draft(status, source, slot)
        self._step(llm_done(self._advance(10), self.s.active_generate, tool))

    @rule(status=st.sampled_from(["valid", "repairable", "refused", None]), slot=st.integers(0, 1))
    @precondition(lambda self: self.s.state is TurnState.IDLE)
    def whole_tool_turn(self, status, slot):
        self.whole_turn_up_to_thinking_or_speaking(text="Opening it. ", answers=slot == 0)
        if self.s.active_generate is None:
            return
        tool = None if status is None else self._draft(status, "draft", slot)
        self._step(llm_done(self._advance(10), self.s.active_generate, tool))

    @rule(status=st.sampled_from(["valid", "repairable", "refused", None]))
    @precondition(lambda self: self._repair_live(self.s))
    def repair_done(self, status):
        tool = None if status is None else self._draft(status, "repair", 0)
        if status == "repairable":
            tool = self._draft(status, "draft", 0)
        self._step(llm_done(self._advance(10), self.s.active_generate, tool))

    @rule()
    @precondition(lambda self: self.s.active_generate is not None)
    def tool_segment_commit(self):
        """``segment="tool"`` (T-0088 adds the literal; built unvalidated until then)."""
        self.marker_seq += 1
        text = f"call:CALLMARK{self.marker_seq}{{x:1}}"
        self.unspeakable.append(f"CALLMARK{self.marker_seq}")
        fields = {
            "t_ms": self._advance(50),
            "request_id": self.s.active_generate,
            "canvas_idx": self.canvas,
            "segment": "tool",
            "text": text,
            "thought_open": False,
        }
        try:
            ev = E.LlmCommit(**fields)
        except ValueError:
            ev = E.LlmCommit.model_construct(**fields, type="llm.commit")
        self.canvas += 1
        self._step(ev)

    @rule(seg=st.sampled_from(["answer", "thought"]))
    @precondition(lambda self: self._repair_live(self.s))
    def repair_commit(self, seg):
        self.marker_seq += 1
        marker = f"REPAIRMARK{self.marker_seq}"
        self.unspeakable.append(marker)
        self._step(
            E.LlmCommit(
                t_ms=self._advance(50),
                request_id=self.s.active_generate,
                canvas_idx=self.canvas,
                segment=seg,
                text=f"{marker} done. ",
            )
        )
        self.canvas += 1

    @rule()
    @precondition(lambda self: self._repair_live(self.s))
    def stale_pass1_done_during_repair(self):
        """R10: a pass-1 llm.done arriving after the repair started is stale."""
        before = self.s
        tool = tool_draft(
            call_id="stale",
            status="valid",
            name="click_at",
            arguments={"x": 9},
            reason=None,
            source="draft",
        )
        cmds = self._step(llm_done(self._advance(5), f"{before.turn_id}-gen", tool))
        assert (self.s, list(cmds)) == (before, []), f"R10: stale pass-1 llm.done: {kinds(cmds)}"

    @rule(played=st.integers(0, 5000))
    def playback_drained(self, played):
        """Overrides the base rule: the drain reports *this turn's* spoken text, as the session's
        does (``audio_io._played_text`` joins the sources of the current run's utterances, and
        ``session.py`` starts a fresh run ledger per turn), never the whole session's. A
        cross-turn ``played_text`` could equal this turn's ``spoken_text`` by coincidence and
        hide or fake the T14 check, which compares the two."""
        self._step(
            E.PlaybackDrained(
                t_ms=self._advance(20), played_ms=played, played_text="".join(self.turn_spoken)
            )
        )

    @rule(slot=st.integers(0, 1))
    @precondition(
        lambda self: (
            self.s.state is TurnState.SPEAKING
            and self.s.active_generate is not None
            and not self._repair_live(self.s)
        )
    )
    def drained_then_valid_done(self, slot):
        """Ruling T11: the reviewer's order -- the drain lands while pass 1 still holds a clause,
        then the final ``llm.done`` carries a valid call and flushes it. Walked on purpose, because
        uniform rules reach "SPEAKING, held clause, drained, valid llm.done" only by luck."""
        self._step(
            E.PlaybackDrained(
                t_ms=self._advance(20), played_ms=100, played_text="".join(self.turn_spoken)
            )
        )
        if self.s.state is TurnState.SPEAKING and self.s.active_generate is not None:
            tool = self._draft("valid", "draft", slot)
            self._step(llm_done(self._advance(10), self.s.active_generate, tool))

    @rule()
    @precondition(lambda self: self._repair_live(self.s))
    def tick_to_repair_deadline(self):
        at = self.repair_deadline.get(self.s.turn_id, self.t + 1)
        self._step(E.Tick(t_ms=self._advance(max(1, at - self.t))))

    @invariant()
    def repair_is_live_until_its_deadline(self):
        """P7: a Tick at the repair deadline, then playback.drained, reaches IDLE."""
        s = self.s
        if s.state not in (TurnState.THINKING, TurnState.SPEAKING) or not self._repair_live(s):
            return
        at = max(self.repair_deadline.get(s.turn_id, self.t), self.t)
        s, _ = self.m.step(s, E.Tick(t_ms=at))
        s, _ = self.m.step(
            s, E.PlaybackDrained(t_ms=at + 1, played_ms=0, played_text="".join(self.turn_spoken))
        )
        assert s.state is TurnState.IDLE, f"P7: stuck in {s.state.value} after the repair deadline"


TestToolTurnInvariants = ToolTurnModel.TestCase
TestToolTurnInvariants.settings = settings(max_examples=200, stateful_step_count=40, deadline=None)


def test_t9_the_seed_1_counterexample_replays_through_the_model():
    """Ruling T9: the trace T8's postcondition failed on a correct machine, replayed by hand so
    the T9 branch runs on every invocation, not only when ``--hypothesis-seed=1`` finds it."""
    model = ToolTurnModel()
    model.late_events_naming_a_request_this_session_never_started()
    model.whole_tool_turn(status="repairable", slot=0)
    assert model.s.state is TurnState.SPEAKING and model._repair_live(model.s)
    model.playback_drained(played=0)
    assert model.s.playback_drained is True
    model.error(kind="llm")
    assert model.s.state is TurnState.IDLE
    model.teardown()


# --- P9: no tool => identical to main ------------------------------------------------------------

from tests.unit import _frozen_turn_machine_main as frozen_main  # noqa: E402

#: Ruling T15: the ``MachineState`` fields T-0089 itself adds, which main's machine never writes.
#: P9 compares every other field exactly, and every command exactly. The one place this set is
#: named; ``test_turn_tool_rows.py`` imports it.
T0089_STATE_FIELDS = frozenset({"tool", "tool_repairs", "spoken_after_drain"})


#: T-0112 (T-0111a §2): the per-turn loop fields of ``MachineState``, inert with ``tool_loop`` off.
#: The base-SHA goldens compare every other state field and every command; named only here.
T0112_STATE_FIELDS = frozenset(
    {"tool_step", "tool_steps_executed", "awaiting_result", "step_text", "played_prior"}
)


def state_without_t0112(state) -> dict:
    """``state`` as JSON data with exactly T-0112's own fields left out."""
    return state.model_dump(mode="json", exclude=set(T0112_STATE_FIELDS))


def state_without_t0089(state) -> dict:
    """``state`` as data with exactly T-0089's own fields left out (ruling T15)."""
    return state.model_dump(exclude=set(T0089_STATE_FIELDS))


_ATOMIC = (
    st.tuples(st.just("tick"), st.integers(1, 800)),
    st.just(("deadline",)),
    st.just(("onset",)),
    st.tuples(st.just("offset"), st.integers(50, 1500)),
    st.tuples(st.just("word"), st.sampled_from(["what", "is", "ok", "mm-hm"])),
    st.just(("eot",)),
    st.just(("flushed",)),
    st.tuples(
        st.just("commit"),
        st.sampled_from(TurnMachineModel.COMMIT_TEXTS),
        st.sampled_from(["answer", "thought", "ui", "tool"]),
        st.booleans(),
        st.booleans(),
    ),
    st.tuples(st.just("done"), st.booleans()),
    st.just(("tts_done",)),
    st.tuples(st.just("stopped"), st.floats(0.0, 1.0)),
    st.just(("drained",)),
    st.just(("takeover",)),
    st.just(("backchannel",)),
    st.tuples(st.just("scene"), st.sampled_from(SCENE_PHASHES)),
    st.tuples(
        st.just("error"),
        st.sampled_from(["llm", "llm_active", "llm_stale", "tts", "stt", "disconnect"]),
    ),
    st.just(("turn",)),
    # ruling T10: a drain reporting only part of what was spoken (0.0 = nothing played), and a
    # turn with no word in it, so a toolless THINKING llm.done can meet an empty transcript
    st.tuples(st.just("drained_frac"), st.sampled_from([0.0, 0.5, 1.0])),
    st.just(("turn_silent",)),
)

#: Ruling T10: recipes that *drive* to the rows T-0089 rewrote (``_finish_generate`` and
#: ``_maybe_complete``), expanded against the live state by ``_expand``. Uniform atomic recipes
#: reach "answer commit on the live id, then llm.done in SPEAKING" only by luck.
_COMPOSITE = (
    st.tuples(st.just("speak"), st.sampled_from(TurnMachineModel.COMMIT_TEXTS)),
    st.tuples(st.just("pause"), st.sampled_from(TurnMachineModel.COMMIT_TEXTS)),
    st.tuples(
        st.just("complete"),
        st.sampled_from(TurnMachineModel.COMMIT_TEXTS),
        st.booleans(),  # True: llm.done then playback.drained; False: the other order
        st.sampled_from([0.0, 0.5, 1.0]),
    ),
    st.tuples(st.just("think_done"), st.booleans()),
)

#: Each composite is listed three times: about 40 % of the draws drive somewhere on purpose.
_EV = st.one_of(*_ATOMIC, *(_COMPOSITE * 3))


def _expand(recipe, s):
    """A composite recipe as atomic ones, decided against the state it starts from."""
    kind = recipe[0]
    if kind == "speak":
        lead: list[tuple] = []
        live = s.active_generate is not None
        if not live or s.state not in (TurnState.THINKING, TurnState.SPEAKING):
            # end whatever is open (llm.error ends a turn from every state, criterion 14),
            # then walk a fresh turn to THINKING, so the commit lands on a live id
            lead = [("error", "llm")] if s.state is not TurnState.IDLE else []
            lead.append(("turn",))
        return [*lead, ("commit", recipe[1], "answer", False, False)]
    if kind == "pause":
        return [("speak", recipe[1]), ("onset",)]
    if kind == "complete":
        _, text, done_first, frac = recipe
        ends = [("done", False), ("drained_frac", frac)]
        return [("speak", text), *(ends if done_first else ends[::-1])]
    if kind == "think_done":
        lead = [] if s.state is TurnState.IDLE else [("error", "llm")]
        if s.state is not TurnState.THINKING or s.active_generate is None:
            lead.append(("turn_silent",) if recipe[1] else ("turn",))
        else:
            lead = []
        return [*lead, ("done", False)]
    return None


def _events_for(recipe, s, t, spoken):
    kind = recipe[0]
    rid = s.active_generate
    if kind == "tick":
        return [E.Tick(t_ms=t + recipe[1])]
    if kind == "deadline":
        at = min((x.at_ms for x in s.timers), default=t + 1)
        return [E.Tick(t_ms=max(t + 1, at))]
    if kind == "onset":
        return [E.UserSpeechOnset(t_ms=t + 50, onset_ms=t + 50)]
    if kind == "offset":
        return [E.UserSpeechOffset(t_ms=t + 20, offset_ms=t + 20, speech_ms=recipe[1])]
    if kind == "word":
        return [
            E.SttWord(
                t_ms=t + 200, text=recipe[1], start_ms=max(0, t - 300), end_ms=max(0, t - 100)
            )
        ]
    if kind == "eot":
        return [E.SttEndOfTurn(t_ms=t + 100, t_eot_ms=t)]
    if kind == "flushed":
        return [E.SttFlushed(t_ms=t + 5, t_eot_ms=s.t_eot_ms if s.t_eot_ms is not None else t)]
    if kind == "commit":
        _, text, seg, open_, stale = recipe
        fields = {
            "t_ms": t + 300,
            "request_id": "t0-gen" if stale or rid is None else rid,
            "canvas_idx": 0,
            "segment": seg,
            "text": text,
            "thought_open": open_,
        }
        if seg == "tool":
            return [E.LlmCommit.model_construct(**fields, type="llm.commit")]
        return [E.LlmCommit(**fields)]
    if kind == "done":
        return [E.LlmDone(t_ms=t + 10, request_id="t0-gen" if recipe[1] or rid is None else rid)]
    if kind == "tts_done":
        return [E.TtsDone(t_ms=t + 10, utterance_id=(s.utterances or ("t0-u0",))[-1])]
    if kind == "stopped":
        text = spoken[: int(len(spoken) * recipe[1])]
        return [E.PlaybackStopped(t_ms=t + 20, played_ms=100, played_text=text)]
    if kind == "drained":
        return [E.PlaybackDrained(t_ms=t + 20, played_ms=100, played_text=spoken)]
    if kind == "drained_frac":
        text = spoken[: int(len(spoken) * recipe[1])]
        return [E.PlaybackDrained(t_ms=t + 20, played_ms=100, played_text=text)]
    if kind == "turn_silent":
        return [
            E.UserSpeechOnset(t_ms=t + 50, onset_ms=t + 50),
            E.SttEndOfTurn(t_ms=t + 550, t_eot_ms=t + 500),
            E.SttFlushed(t_ms=t + 555, t_eot_ms=t + 500),
            E.Tick(t_ms=t + 556 + POLICY.eot_settle_ms),
        ]
    if kind == "takeover":
        return [E.UserTakeover(t_ms=t + 30, onset_ms=max(0, t - 700))]
    if kind == "backchannel":
        return [E.UserBackchannel(t_ms=t + 30, duration_ms=200)]
    if kind == "scene":
        return [E.SceneChanged(t_ms=t + 40, source="screen", phash=recipe[1])]
    if kind == "error":
        return [
            {
                "llm": E.LlmError(t_ms=t + 5, message="x"),
                "llm_active": E.LlmError(t_ms=t + 5, request_id=rid or "t0-gen", message="x"),
                "llm_stale": E.LlmError(t_ms=t + 5, request_id="t0-gen", message="x"),
                "tts": E.TtsError(t_ms=t + 5, message="x"),
                "stt": E.SttError(t_ms=t + 5, message="x"),
                "disconnect": E.ClientDisconnected(t_ms=t + 5),
            }[recipe[1]]
        ]
    # "turn": IDLE -> THINKING -> SPEAKING in one recipe, so traces reach the speaking rows
    return [
        E.UserSpeechOnset(t_ms=t + 50, onset_ms=t + 50),
        E.SttWord(t_ms=t + 450, text="what", start_ms=t + 50, end_ms=t + 250),
        E.SttEndOfTurn(t_ms=t + 550, t_eot_ms=t + 500),
        E.SttFlushed(t_ms=t + 555, t_eot_ms=t + 500),
        E.Tick(t_ms=t + 556 + POLICY.eot_settle_ms),
    ]


#: Ruling T10 (amended): floors for ``test_p9_reaches_the_rows_t0089_rewrote_on_a_fixed_run``
#: only -- 400 derandomized examples, no database, no examples. Derandomized is not fixed by itself:
#: Hypothesis 6.168 also draws from a pool of literals harvested from every imported local module.
#: Measured 2026-10-12 with three import sets (this module alone; the seven gate modules; every
#: module under tests/unit and tests/contract), minima: speaking 652, paused 203, toolless
#: llm.done in SPEAKING 147, completed on llm.done 77, on the drain 56, with nothing played 40,
#: silent THINKING llm.done 46, a held clause flushed by llm.done 40 (sentences). The floors are a
#: third of those minima or less. Main's strategy at 2d4a462 reached every one of them zero times.
P9_MIN_VISITS = {
    "speaking": 150,
    "paused": 60,
    "toolless_done_in_speaking": 18,
    "completed_on_done": 3,
    "completed_on_drained": 12,
    "completed_nothing_played": 4,
    "silent_thinking_done": 6,
}
P9_MIN_VISITS_SENTENCES = {"done_flushes_a_held_clause": 4}

#: Ruling T10: traces that reach each of the four named mutants by construction, so the kill
#: does not depend on the seed (the floors above say the random traces get there too).
P9_EXAMPLES = (
    [("complete", "Yes. ", True, 1.0)],  # llm.done in SPEAKING, then the drain completes
    [("complete", "The error is on", False, 0.0)],  # drain (nothing played), then llm.done
    [("think_done", True)],  # toolless THINKING llm.done, empty transcript
    [("pause", "Look at line two. "), ("done", False), ("backchannel",), ("drained",)],
)


def _p9_trace(live, old, recipes, visits):
    s = live.initial()
    assert s == old.initial()
    t, spoken = 0, ""
    queue = list(recipes)
    while queue:
        recipe = queue.pop(0)
        parts = _expand(recipe, s)
        if parts is not None:
            queue[:0] = parts
            continue
        for ev in _events_for(recipe, s, t, spoken):
            t = max(t, ev.t_ms)
            assert getattr(ev, "tool", None) is None
            new_state, new_cmds = live.step(s, ev)
            old_state, old_cmds = old.step(s, ev)
            assert tuple(new_cmds) == tuple(old_cmds), (
                f"P9: {ev.type} in {s.state.value}: {kinds(new_cmds)} != main {kinds(old_cmds)}"
            )
            assert state_without_t0089(new_state) == state_without_t0089(old_state), (
                f"P9: {ev.type} in {s.state.value} diverged from main"
            )
            _p9_count(visits, s, ev, new_state, new_cmds)
            spoken += "".join(c.source for c in new_cmds if isinstance(c, C.TtsSpeak))
            s = new_state
            if s.state is TurnState.IDLE:
                spoken = ""


def _p9_count(visits, s, ev, new_state, cmds):
    """What the trace reached, read off the events and the oracle's agreed output."""
    if s.state is TurnState.SPEAKING:
        visits["speaking"] += 1
    if s.state is TurnState.PAUSED:
        visits["paused"] += 1
    accepted = isinstance(ev, E.LlmDone) and ev.request_id == s.active_generate
    if accepted and s.state is TurnState.SPEAKING:
        visits["toolless_done_in_speaking"] += 1
        if any(isinstance(c, C.TtsSpeak) for c in cmds):
            visits["done_flushes_a_held_clause"] += 1
    if accepted and s.state is TurnState.THINKING and not s.words:
        visits["silent_thinking_done"] += 1
    if s.state is TurnState.SPEAKING and new_state.state is TurnState.IDLE:
        if isinstance(ev, E.LlmDone):
            visits["completed_on_done"] += 1
            played = s.played_text
        elif isinstance(ev, E.PlaybackDrained):
            visits["completed_on_drained"] += 1
            played = ev.played_text
        else:
            return
        if not played:
            visits["completed_nothing_played"] += 1


def _p9_run(segmenter: str, *, derandomize: bool, examples=()) -> dict[str, int]:
    """Run the P9 differential over the shared strategy and return what the traces reached."""
    seg = sentence_segmenter if segmenter == "sentences" else None
    visits: dict[str, int] = dict.fromkeys([*P9_MIN_VISITS, *P9_MIN_VISITS_SENTENCES], 0)
    extra = {"derandomize": True, "database": None} if derandomize else {}

    @settings(max_examples=400 if derandomize else 200, deadline=None, **extra)
    @given(recipes=st.lists(_EV, min_size=4, max_size=40))
    def trace(recipes):
        live = TurnMachine(TurnPolicy(thinking=True), segment=seg)
        old = frozen_main.TurnMachine(frozen_main.TurnPolicy(thinking=True), segment=seg)
        _p9_trace(live, old, recipes, visits)

    for recipes in examples:
        trace = example(recipes=recipes)(trace)
    trace()
    return visits


@pytest.mark.parametrize("segmenter", ["default", "sentences"])
def test_p9_without_a_tool_the_machine_is_main(segmenter):
    """P9 (negative, differential): with no ``LlmDone.tool``, ``step`` returns exactly what main's
    frozen ``turn_machine.py`` returns -- every command exactly, and every state field except
    T-0089's own (ruling T15) -- at every step of the trace.

    Passes before T-0089 is implemented, on purpose: it is the guard that "no row changes for an
    ``llm.done`` without a tool" stays true while the rows are added.

    Ruling T10 (amended): this run stays random, for exploration, and asserts no visit floors;
    ``P9_EXAMPLES`` guarantee that the four T10 mutants are reached on every run. The floors live
    in ``test_p9_reaches_the_rows_t0089_rewrote_on_a_fixed_run``.
    Regime: 200 random examples of 4-40 recipes plus the examples, per segmenter, CPU, pure.
    """
    _p9_run(segmenter, derandomize=False, examples=P9_EXAMPLES)


@pytest.mark.parametrize("segmenter", ["default", "sentences"])
def test_p9_reaches_the_rows_t0089_rewrote_on_a_fixed_run(segmenter):
    """Ruling T10 (amended): the same strategy and the same full P9 comparison, derandomized and
    without the database or the examples, so the visit floors are decided by the strategy and not
    by the seed. Floors at random seeds failed correct code in 8 of 100 seeds (reviewer B).
    Regime: 400 derandomized examples of 4-40 recipes per segmenter, CPU, pure.
    """
    visits = _p9_run(segmenter, derandomize=True)
    floors = dict(P9_MIN_VISITS)
    if segmenter == "sentences":
        floors.update(P9_MIN_VISITS_SENTENCES)
    short = {k: (visits[k], n) for k, n in floors.items() if visits[k] < n}
    assert not short, f"T10: P9 did not reach these rows often enough (got, floor): {short}"


def test_p9_frozen_copy_is_mains_turn_machine_verbatim():
    """The P9 oracle is only an oracle while nobody edits it: sha256 of the copied body, computed
    by hand from ``git show e0174f9:local_astra/conductor/turn_machine.py | sed '1,17d'``."""
    import hashlib

    text = pathlib.Path(frozen_main.__file__).read_text()
    marker = "dropped) ---\n"
    body = text[text.index(marker) + len(marker) :]
    assert hashlib.sha256(body.encode()).hexdigest() == (
        "1e168ac0c7c560c541b371f99d58cee293d35af0bdb23d653588480ef012735e"
    )


# --- T-0112 (T-0111a a1): the tool-result loop, P1', P3', L1-L8, P5', P6', P8' ------------------
#
# Every property is read off docs/tasks/T-0111a.md §2 (rows R11-R18, the invariants) and §3 (D16
# speech), with the rulings H-T111a-2/3/4/8. The model keeps its own ledger of what the *events*
# said -- which drafts were delivered at which step, which call was executed and is awaited, which
# turn was cut -- and checks the *commands* against it. Nothing is read off an implementation.
#
# Regime: CPU, pure. ``TestToolLoopInvariants``: 300 examples x 80 rules, tool_loop on,
# ``max_tool_steps = 2`` (so the report step R16 is reachable inside a trace; R16 at the default
# 8 is a row in test_turn_tool_loop_rows.py). L8: up to 60 events per sequence.

from tests.unit import _frozen_turn_machine_t0089 as frozen_t0089  # noqa: E402

#: T-0111a §2 Policy, with the H-T111a-1/-2 defaults.
LOOP_POLICY_DEFAULTS = {"tool_loop": False, "max_tool_steps": 8, "tool_result_timeout_ms": 15_000}
TOOL_RESULT_MS = 15_000
LOOP_MAX_STEPS = 2

#: T-0178 B6: the stateful model mixes budgets 1-3 (``ToolLoopModel.choose_budget``); 2 stays the default
LOOP_BUDGETS = (1, 2, 3)
#: T-0178 B3: the policy refusals, which still voice (typed from ``toolcall.ask_first`` / ``toolcall.user_declines``)
POLICY_REASONS = ("asked the user first", "the user said no")
#: T-0178 amendment B3a: T-0176's forced-search refusals, the card reasons the session hands the machine for a search
#: drafted at the forced answer step (typed from T-0176 criterion 7). Voiced when the step's text is a plain answer
#: (``announces`` is None), withheld with ``not_finished`` when it only announces.
SEARCH_CAP_REASONS = (
    "Gearvis limit: 5 searches per question",
    "Gearvis limit: 3 failed searches per question",
)
#: the model's draft statuses for them (refused drafts carrying the reason)
SEARCH_CAP_STATUSES = {"capped": SEARCH_CAP_REASONS[0], "failcapped": SEARCH_CAP_REASONS[1]}
#: T-0178 amendment B4a: the two system clips
SYSTEM_CLIPS = ("out_of_steps", "not_finished")
TOOL_RESULT_STATUSES = ("dry_run", "executed", "refused", "failed", "timeout")
#: L4's list: after these the turn emits no further tool.execute or LlmContinue
#: T-0162 A-2 adds "reset": a Restart cuts the turn like a disconnect (its awaited result is stale).
CUT_REASONS = frozenset({"interrupted", "takeover", "error", "disconnected", "reset"})


def loop_policy(**overrides) -> TurnPolicy:
    """A ``TurnPolicy`` with the loop fields, or a named failure while they do not exist."""
    fields = getattr(TurnPolicy, "__dataclass_fields__", {})
    missing = [name for name in LOOP_POLICY_DEFAULTS if name not in fields]
    assert not missing, f"T-0112: TurnPolicy gains {missing} (T-0111a §2 Policy)"
    return TurnPolicy(thinking=True, **overrides)


def tool_result_event(t_ms: int, call_id: str, status: str = "dry_run"):
    cls = getattr(E, "ToolResult", None)
    assert cls is not None, "T-0112: contracts/events.py gains ToolResult(t_ms, call_id, status)"
    return cls(t_ms=t_ms, call_id=call_id, status=status)


def is_continue(c) -> bool:
    return type(c).__name__ == "LlmContinue"


def is_tool_result(ev) -> bool:
    return type(ev).__name__ == "ToolResult"


def joins_played_runs(stored: str, runs) -> bool:
    """L6 oracle, by brute force: ``stored`` is ``" ".join`` of a subsequence, in order, of the
    non-empty played texts ``runs`` of the turn (T-0111a §2 History: ``" ".join(nonempty(...))``)."""
    pieces = [r for r in runs if r]

    def fits(pos: int, start: int) -> bool:
        if pos == len(stored):
            return True
        for j in range(start, len(pieces)):
            chunk = (" " if pos else "") + pieces[j]
            if stored.startswith(chunk, pos) and fits(pos + len(chunk), j + 1):
                return True
        return False

    return fits(0, 0)


def stored_assistant_text(entry) -> str:
    """The played part of a history entry: the ``[interrupted]`` marker of A6 taken off."""
    text = entry.assistant_spoken
    if entry.interrupted:
        text = text.removesuffix("[interrupted]").removesuffix(" ")
    return text


class ToolLoopModel(TurnMachineModel):
    """``TurnMachineModel`` with ``tool_loop`` on: drafts at every step, ``ToolResult`` events
    (matching, duplicate, stale, early), step texts, the report step and takeovers after actions.

    The base invariants run on these traces. Three base checks are *replaced in this model only*,
    because the loop changes the premise they state, and each replacement is stricter on its
    subject: the speech ledger (step 1 keeps the base check; a step >= 2 must voice exactly its
    own committed text or nothing, R14/R15/L5), the THINKING-commit postcondition (a step >= 2
    commit stays in THINKING and speaks nothing, R14), and the history check (L6: each stored text
    is a join of this turn's played texts in run order, checked at the moment it is stored).

    T-0178 (B2-B6) re-scope: the speech ledger's step >= 2 premise is the honest ending. A step >= 2 voices its text
    only when it ends with no call (and is not a report-step announcement, B5) or with a call refused for a policy
    reason (ask-first, user-declined, B3); a valid call (executed with budget left, R15, or cancelled ``step_budget``,
    B2), a non-policy refusal or repairable draft (B3), and a report-step announcement (B5) withhold it. Budgets mix
    1-3 (``choose_budget``), drafts carry policy reasons, and commits may announce an action. L9 and L10 are added.

    Amendment B3a: drafts also carry T-0176's two search-cap card reasons (``capped`` / ``failcapped``). Such a
    refusal voices the step's text when it is a plain answer (``announces`` is None) and withholds it with
    ``not_finished`` when it only announces; L9 asserts, from the commands, that a non-policy refusal voices nothing
    else.
    """

    _pending_calls = ToolTurnModel._pending_calls
    _repair_live = ToolTurnModel._repair_live
    _paused_timer_due = ToolTurnModel.__dict__["_paused_timer_due"]
    _expected_cancel = ToolTurnModel._expected_cancel
    _r7_applies = ToolTurnModel._r7_applies

    def _draft(self, status, source, slot):
        """``ToolTurnModel._draft``, plus T-0178 B3's policy refusals: ``asked`` / ``declined`` are ``refused``
        drafts whose reason is ask-first / user-declined (the session's ``_judge_tool`` makes these)."""
        if status in ("asked", "declined"):
            d = ToolTurnModel._draft(self, "refused", source, slot)
            reason = POLICY_REASONS[0] if status == "asked" else POLICY_REASONS[1]
            return d.model_copy(update={"reason": reason})
        if status in SEARCH_CAP_STATUSES:
            # T-0178 B3a: a search refused at the forced answer step (``session._search_forced``)
            d = ToolTurnModel._draft(self, "refused", source, slot)
            return d.model_copy(update={"reason": SEARCH_CAP_STATUSES[status]})
        return ToolTurnModel._draft(self, status, source, slot)

    def _ending(self, turn, tool) -> str:
        """T-0178: how an accepted step >= 2 ``llm.done`` ends, from the events alone: ``execute`` (R15 -> R11),
        ``budget`` (B2), ``refused`` (B3: non-policy refusal or repairable), ``announced`` (B5: the report step only
        announces) or ``voiced`` (no call, or a policy refusal). Read before this event's executes are recorded.

        ``announces`` (T-0178 B5's classifier, the oracle) is imported here, not at module level: BASE-tree runs
        (``test_session_tool_loop``'s d3bc6ce capture) import this module against a tree that predates it."""
        from local_astra.toolcall.announced_call import announces

        executed = len(self.executed.get(turn, []))
        if tool is not None and tool.status == "valid":
            return "execute" if executed < self.max_steps else "budget"
        if tool is not None and tool.reason in SEARCH_CAP_REASONS:
            # T-0178 B3a: a plain answer is voiced; an announcement only is withheld as in B3
            return "voiced" if announces(self.step_text.get(turn, "")) is None else "refused"
        if tool is not None:
            return "voiced" if tool.reason in POLICY_REASONS else "refused"
        if executed >= self.max_steps and announces(self.step_text.get(turn, "")) is not None:
            return "announced"
        return "voiced"

    @initialize(budget=st.sampled_from(LOOP_BUDGETS))
    def choose_budget(self, budget):
        """T-0178 B6: the trace's ``max_tool_steps``, 1-3 (before any turn: ``initialize`` rules run first)."""
        self.max_steps = budget
        self.m = TurnMachine(
            loop_policy(tool_loop=True, max_tool_steps=budget), segment=self._segment
        )

    tool_segment_commit = ToolTurnModel.tool_segment_commit
    repair_commit = ToolTurnModel.repair_commit

    def __init__(self):
        super().__init__()
        self.m = TurnMachine(
            loop_policy(tool_loop=True, max_tool_steps=LOOP_MAX_STEPS), segment=self._segment
        )
        self.s = self.m.initial()
        self.call_seq = 0
        self.marker_seq = 0
        #: P6': tool commits, repair commits, unclosed/thought canvases and withheld step text
        self.unspeakable: list[str] = []
        self.repairs: dict[str, int] = {}
        self.repair_deadline: dict[str, int] = {}
        self.step_of: dict[str, int] = {}  # turn -> current step (1 + accepted results)
        self.step_text: dict[str, str] = {}  # turn -> answer text committed in this step (k >= 2)
        self.step_markers: dict[str, list[str]] = {}
        self.step_spoken_at: dict[str, int] = {}  # len of the turn's spoken text at step start
        self.step_voiced: dict[str, bool] = {}
        self.executed: dict[str, list[tuple[int, str]]] = {}  # turn -> [(step, call id)]
        self.generates: dict[str, int] = {}
        self.awaiting: dict[str, str] = {}  # turn -> executed call whose result is awaited
        self.result_deadline: dict[str, int] = {}
        self.answered: list[str] = []
        self.valid_drafts: dict[tuple[str, int], list[tuple]] = {}
        self.opened: dict[str, list[str]] = {}
        self.status: dict[str, str] = {}
        self.budget_left: dict[str, bool] = {}
        self.terminals: dict[str, list] = {}
        self.tainted: set[str] = set()
        self.cut: set[str] = set()
        self.spoke_since_drain = False
        self.runs: dict[str, list[str]] = {}
        self.run_spoken = ""
        self.closed: set[str] = set()
        #: R13/R17 adversarial: the last awaited call of each turn that was closed cut or awaiting
        self.cut_calls: list[str] = []
        self.l6_checked: list = []
        self.check_step = 1
        #: T-0178: the trace's budget (``choose_budget``), system clips per turn, turns with a step_budget cancel
        self.max_steps = LOOP_MAX_STEPS
        self.system_clips: dict[str, int] = {}
        self.budget_cut: set[str] = set()

    # --- replaced base checks (see the class docstring) ----------------------------------------

    def _spoken_is_a_prefix_of_what_was_committed(self, before, ev):
        if self.check_step == 1:
            return super()._spoken_is_a_prefix_of_what_was_committed(before, ev)
        turn = before.turn_id
        spoken = "".join(self.turn_spoken)
        tail = spoken[self.step_spoken_at.get(turn, 0) :]
        live = before.active_generate
        closing = isinstance(ev, E.LlmDone) and live is not None and ev.request_id == live
        if closing and not self.step_voiced.get(turn):
            # T-0178 re-scope (B2/B3/B5): voiced only for no call (not a report-step announcement) or a policy
            # refusal; was "every step that does not execute" (R15/R16 at BASE)
            self.step_voiced[turn] = self._ending(turn, getattr(ev, "tool", None)) == "voiced"
        expected = self.step_text.get(turn, "") if self.step_voiced.get(turn) else ""
        assert tail == expected, (
            f"R14/R15/L5/L9: step {self.check_step} of {turn} voiced {tail!r}; its committed text is "
            f"{self.step_text.get(turn, '')!r} and it is "
            f"{'voiced (no call, or a policy refusal)' if self.step_voiced.get(turn) else 'not voiced yet / withheld'}"
        )

    def _postconditions(self, before, ev, cmds):
        if self._r7_applies(before, ev):
            reasons = {c.reason for c in named(cmds, "ToolCancel")}
            assert reasons == {"repair_failed"}, f"R7: {reasons}"
            assert not named(cmds, "ToolExecute") and not [c for c in cmds if is_continue(c)]
            return
        if (
            self.check_step >= 2
            and isinstance(ev, E.LlmCommit)
            and before.active_generate is not None
            and ev.request_id == before.active_generate
        ):
            assert not [c for c in cmds if isinstance(c, C.TtsSpeak)], (
                f"R14: a step-{self.check_step} commit spoke: {kinds(cmds)}"
            )
            assert self.s.state is TurnState.THINKING, (
                f"R14/R17: a step >= 2 generation runs in THINKING; went {self.s.state.value}"
            )
            return
        super()._postconditions(before, ev, cmds)

    @invariant()
    def history_never_contains_unplayed_text(self):
        """Every stored entry passed L6 when it was stored (``_step``); nothing else writes it."""
        for h in self.s.history:
            assert h in self.l6_checked, f"L6: history entry {h} was never seen being stored"

    # --- the checks ----------------------------------------------------------------------------

    def _step(self, ev):
        before = self.s
        turn = before.turn_id
        k = self.step_of.get(turn, 1)
        self.check_step = k
        live = before.active_generate
        if (
            k >= 2
            and isinstance(ev, E.LlmCommit)
            and ev.segment == "answer"
            and not ev.thought_open
            and live is not None
            and ev.request_id == live
        ):
            self.step_text[turn] = self.step_text.get(turn, "") + ev.text
        if isinstance(ev, E.PlaybackDrained):
            self.spoke_since_drain = False
        cmds = list(super()._step(ev))

        speaks = [c for c in cmds if isinstance(c, C.TtsSpeak)]
        executes = named(cmds, "ToolExecute")
        continues = [c for c in cmds if is_continue(c)]
        accepted = isinstance(ev, E.LlmDone) and live is not None and ev.request_id == live
        from_repair = accepted and live == repair_id(turn)
        tool = getattr(ev, "tool", None) if accepted else None

        # P6' (and L5's "withheld text is never voiced")
        for c in speaks:
            for marker in self.unspeakable:
                assert marker not in c.source, f"P6': voiced {c.source!r}, holding {marker!r}"
        if k >= 2:
            assert not (speaks and executes), f"L5: step {k} spoke and executed: {kinds(cmds)}"

        # what the events say, before looking at the commands
        expected_cancel = self._expected_cancel(before, ev)
        if expected_cancel is not None and before.state is not TurnState.IDLE:
            self.tainted.add(turn)
            if expected_cancel & CUT_REASONS:
                self.cut.add(turn)
        if self._repair_live(before) and (
            (isinstance(ev, E.LlmError) and ev.request_id in (None, live))
            or expected_cancel == {"repair_timeout"}
        ):
            for call in self._pending_calls(turn):
                self.status[call] = "failed"
        if from_repair and (tool is None or tool.status != "valid"):
            for call in self._pending_calls(turn):
                self.status[call] = "failed"
        if tool is not None and before.state in (
            TurnState.THINKING,
            TurnState.SPEAKING,
            TurnState.PAUSED,
        ):
            call = tool.call_id
            self.status[call] = tool.status
            if call not in self.opened.setdefault(turn, []):
                self.opened[turn].append(call)
            self.budget_left[call] = len(self.executed.get(turn, [])) < self.max_steps
            if tool.status == "valid":
                self.valid_drafts.setdefault((turn, k), []).append(
                    (call, tool.name, tool.arguments)
                )

        # R15 / R16 / P5', the must-happen direction at a step >= 2 llm.done
        if accepted and not from_repair and k >= 2 and tool is not None:
            got = sorted((c.call_id, c.reason) for c in named(cmds, "ToolCancel"))
            ran = [c.call_id for c in executes]
            if tool.status == "valid" and self.budget_left[tool.call_id]:
                assert ran == [tool.call_id] and not got, f"R15->R11: {kinds(cmds)} {got}"
            elif tool.status == "valid":
                assert got == [(tool.call_id, "step_budget")] and not ran, f"R16: {got} {ran}"
            else:
                assert got == [(tool.call_id, "refused")] and not ran, f"R15/H-T111a-8: {got}"
        # T-0178 L9 / L10 / B4: the honest ending, decided from the events (``_ending``), before this tuple's
        # executes are recorded
        clips_now = [c for c in cmds if isinstance(c, C.PlayClip) and c.kind in SYSTEM_CLIPS]
        if clips_now:
            self.system_clips[turn] = self.system_clips.get(turn, 0) + len(clips_now)
            assert self.system_clips[turn] <= 1, (
                f"L10: {self.system_clips[turn]} system clips in {turn}: {kinds(cmds)}"
            )
        ending = self._ending(turn, tool) if accepted and not from_repair and k >= 2 else None
        if (
            accepted
            and not from_repair
            and k >= 2
            and tool is not None
            and tool.status != "valid"
            and tool.reason not in POLICY_REASONS
            and "".join(c.source for c in speaks)
        ):
            # L9 with the B3a exemption, read off the commands: a step >= 2 refused for a non-policy reason voices
            # only a search-cap refusal's plain answer, and what it voices is that answer
            from local_astra.toolcall.announced_call import announces

            said = self.step_text.get(turn, "")
            assert tool.reason in SEARCH_CAP_REASONS, (
                f"L9: step {k} of {turn} refused ({tool.reason!r}) voiced {kinds(cmds)}"
            )
            assert announces(said) is None, (
                f"L9/B3a: a forced-search refusal voiced an announcement {said!r}: {kinds(cmds)}"
            )
            assert "".join(c.source for c in speaks).strip() == said.strip(), (
                f"L9/B3a: the voiced text is the step's plain answer {said!r}; got "
                f"{[c.source for c in speaks]}"
            )
        if ending in ("budget", "refused", "announced"):
            markers = self.step_markers.get(turn, [])
            for c in speaks:
                for marker in markers:
                    assert marker not in c.source, (
                        f"L9: step {k} of {turn} ended {ending} and voiced {c.source!r} ({marker})"
                    )
            self.unspeakable.extend(markers)  # L9: and never later (P6')
            want = "not_finished" if ending == "refused" else "out_of_steps"
            assert [c.kind for c in clips_now] == [want], (
                f"B4/B4a: step {k} of {turn} ended {ending}, so one {want} clip; got {kinds(cmds)}"
            )
            assert self.s.state is TurnState.IDLE, (
                f"B4/L10: the withholding step completes the turn; went {self.s.state.value}"
            )
            if ending == "budget":
                self.budget_cut.add(turn)
        else:
            assert not clips_now, (
                f"B4 (guard width): a system clip on {ev.type} (ending {ending}) in {turn}: {kinds(cmds)}"
            )
        for r in named(cmds, "ToolRepairStart"):
            assert k == 1, f"P5': a repair started at step {k} of {turn}"
            self.repairs[turn] = self.repairs.get(turn, 0) + 1
            assert self.repairs[turn] <= 1, f"L1/P5: two repairs in {turn}"
            self.repair_deadline[turn] = ev.t_ms + REPAIR_TIMEOUT_MS
            assert r.call_id == tool.call_id

        # R8, the must-happen cancels of a pending call
        pending = self._pending_calls(turn)
        if expected_cancel is not None and pending:
            got = {(c.call_id, c.reason) for c in named(cmds, "ToolCancel")}
            for call in pending:
                assert any(cid == call and why in expected_cancel for cid, why in got), (
                    f"R8: {ev.type} in {before.state.value} owes ToolCancel({call}, "
                    f"{sorted(expected_cancel)}); got {sorted(got)}"
                )

        # P8': at most one terminal per call
        for c in cmds:
            if type(c).__name__ in TERMINAL:
                assert c.call_id in self.opened.get(turn, []), (
                    f"P8': {type(c).__name__}({c.call_id}) names no pending call of {turn}"
                )
                self.terminals.setdefault(c.call_id, []).append(c)
                assert len(self.terminals[c.call_id]) == 1, f"P8': {c.call_id} ended twice"

        # P1', P3', L2, L4 on every tool.execute
        assert len(executes) <= 1, f"P1': two tool.execute in one tuple: {kinds(cmds)}"
        for ex in executes:
            done = self.executed.setdefault(turn, [])
            assert ex.turn_id == turn, f"P1': executed {ex.turn_id}'s call during {turn}"
            assert turn not in self.tainted and turn not in self.cut, (
                f"L4/P2: tool.execute in {turn} after it was cut, errored or timed out"
            )
            assert len(done) < self.max_steps, f"P1': more than {self.max_steps} executes"
            assert all(step != k for step, _ in done), f"P1': two executes at step {k}"
            assert all(cid != ex.call_id for _, cid in done), f"P1': {ex.call_id} ran twice"
            assert (ex.call_id, ex.name, ex.arguments) in self.valid_drafts.get((turn, k), []), (
                f"P1': {ex} is not a valid draft of step {k}: {self.valid_drafts.get((turn, k))}"
            )
            assert not speaks, f"P3'(a): tool.execute beside tts.speak: {kinds(cmds)}"
            assert not self.spoke_since_drain, "P3'(b): executed after a speak the drain missed"
            for kind in (C.CommitHistory, C.LlmStart, C.ScheduleSummary):
                assert not [c for c in cmds if isinstance(c, kind)], f"P3'(d): {kinds(cmds)}"
            assert not continues, f"P3'(d)/L2: LlmContinue beside tool.execute: {kinds(cmds)}"
            assert self.s.state is TurnState.THINKING, f"P3'(d): went {self.s.state.value}"
            assert getattr(self.s, "awaiting_result", None) == ex.call_id, "R11: awaiting_result"
            arms = [c for c in cmds if isinstance(c, C.ArmTimer) and c.name == "tool_result"]
            assert arms, f"R11: tool.execute arms tool_result: {kinds(cmds)}"
            assert not self.live_generates, f"L2: executed while {self.live_generates} is live"
            assert turn not in self.awaiting, "L2: executed while a result is awaited"
            done.append((k, ex.call_id))
            self.awaiting[turn] = ex.call_id
            self.result_deadline[turn] = arms[-1].at_ms
            if k >= 2:
                self.unspeakable.extend(self.step_markers.get(turn, []))  # R15: withheld

        # L1: generates per turn
        for c in cmds:
            if isinstance(c, C.LlmStart):
                self.generates[turn] = self.generates.get(turn, 0) + 1

        # L3 / R12 / R13
        if is_tool_result(ev):
            matched = self.awaiting.get(turn) == ev.call_id and before.state is TurnState.THINKING
            if matched:
                assert len(continues) == 1, f"R12: one LlmContinue per accepted result: {cmds}"
                c = continues[0]
                assert (c.request_id, c.step, c.call_id) == (
                    f"{turn}-gen-s{k + 1}",
                    k + 1,
                    ev.call_id,
                ), f"R12/ids: {c}"
                assert C.CancelTimer(name="tool_result") in cmds, f"R12: {kinds(cmds)}"
                assert self.s.state is TurnState.THINKING and self.s.active_generate == c.request_id
                assert getattr(self.s, "awaiting_result", "unset") is None
            else:
                assert (self.s, cmds) == (before, []), (
                    f"R13/L3: an unmatched ToolResult({ev.call_id}) in {before.state.value} "
                    f"(awaiting {self.awaiting.get(turn)}) emitted {kinds(cmds)}"
                )
        else:
            assert not continues, f"L3: LlmContinue without an accepted ToolResult ({ev.type})"
        for c in continues:
            assert turn not in self.cut, f"L4: LlmContinue in {turn} after it was cut"
            assert not self.live_generates, f"L2: LlmContinue while {self.live_generates} is live"
            self.live_generates.add(c.request_id)
            self.generates[turn] = self.generates.get(turn, 0) + 1
            self.awaiting.pop(turn, None)
            self.answered.append(ev.call_id)
            self.step_of[turn] = k + 1
            self.step_text[turn] = ""
            self.step_markers[turn] = []
            self.step_spoken_at[turn] = len("".join(self.turn_spoken))
            self.step_voiced[turn] = False
        assert self.generates.get(turn, 0) <= self.max_steps + 1, (
            f"L1: {self.generates[turn]} generates in {turn} (max_tool_steps {self.max_steps})"
        )

        # R18: the result deadline fails the turn
        deadline = self.result_deadline.get(turn)
        if (
            isinstance(ev, E.Tick)
            and turn in self.awaiting
            and before.state is TurnState.THINKING
            and deadline is not None
            and ev.t_ms >= deadline
        ):
            assert self.s.state is TurnState.IDLE, f"R18: stayed {self.s.state.value}"
            assert C.HudError(message="tool timeout") in cmds, f"R18: {kinds(cmds)}"

        # L6: what playback reported, run by run; each stored text is a join of it
        if isinstance(ev, E.PlaybackStopped | E.PlaybackDrained):
            self.runs.setdefault(turn, []).append(ev.played_text)
            self.run_spoken = ""
        self.run_spoken += "".join(c.source for c in speaks)
        if speaks:
            self.spoke_since_drain = True
        # T-0162 A-2: a reset empties history; that stores nothing, so L6 has no entry to check
        # (the reset rule itself asserts history == ()). Only that exact shape is exempt.
        emptied_by_reset = type(ev).__name__ == "SessionReset" and self.s.history == ()
        if self.s.history != before.history and not emptied_by_reset:
            entry = self.s.history[-1]
            stored = stored_assistant_text(entry)
            assert joins_played_runs(stored, self.runs.get(turn, [])), (
                f"L6: stored {entry.assistant_spoken!r}; the turn's played texts were "
                f"{self.runs.get(turn, [])}"
            )
            self.l6_checked.append(entry)

        # R17: a takeover after an action opens a turn without the old words
        if (
            isinstance(ev, E.UserTakeover)
            and before.state is TurnState.THINKING
            and self.executed.get(turn)
        ):
            assert self.s.turn_id != turn and self.s.state is TurnState.LISTENING
            assert not executes and not continues
            kept = set(before.words) - set(before.next_words)
            assert not kept & set(self.s.words), (
                f"R17: the new turn kept the first request's words {kept & set(self.s.words)}"
            )
            entry = self.s.history[-1] if self.s.history else None
            assert entry is not None and entry.interrupted, f"R17: history {self.s.history}"
            assert entry.assistant_spoken.endswith("[interrupted]")

        # turn boundary: P8' exactly one terminal, of the right kind
        boundary = self.s.turn_id != turn or (
            self.s.state is TurnState.IDLE and before.state is not TurnState.IDLE
        )
        if boundary and turn not in self.closed:
            self.closed.add(turn)
            cut_short = turn in self.awaiting or turn in self.cut or turn in self.tainted
            if cut_short and self.executed.get(turn):
                self.cut_calls.append(self.executed[turn][-1][1])
            for call in self.opened.get(turn, []):
                ends = self.terminals.get(call, [])
                assert len(ends) == 1, f"P8': call {call} of {turn} ended {len(ends)} times"
                end = ends[0]
                must_run = (
                    self.status.get(call) == "valid"
                    and self.budget_left.get(call, True)
                    and turn not in self.tainted
                )
                if must_run:
                    assert type(end).__name__ == "ToolExecute", (
                        f"R11: valid call {call} with budget left, never cut, ended in {end}"
                    )
                if type(end).__name__ == "ToolCancel" and end.reason == "step_budget":
                    assert self.status.get(call) == "valid" and not self.budget_left.get(call)
            self.awaiting.pop(turn, None)
            self.cut.add(turn)
            # IDLE keeps the closed turn's id: later events there are outside any step
            self.step_of.pop(turn, None)
            self.spoke_since_drain = False
            self.run_spoken = ""
        return cmds

    # --- rules ---------------------------------------------------------------------------------

    @rule(played=st.integers(0, 5000), frac=st.floats(0.0, 1.0))
    def playback_stopped(self, played, frac):
        """Overrides the base rule: a terminal reports the current *run's* text (T-0092a run ids;
        the session's ``played_text`` is per run), so two runs of one turn report separately."""
        text = self.run_spoken[: int(len(self.run_spoken) * frac)]
        self._step(E.PlaybackStopped(t_ms=self._advance(20), played_ms=played, played_text=text))

    @rule(played=st.integers(0, 5000))
    def playback_drained(self, played):
        self._step(
            E.PlaybackDrained(t_ms=self._advance(20), played_ms=played, played_text=self.run_spoken)
        )

    @rule(
        # T-0178 B3: "asked" / "declined" are policy refusals
        status=st.sampled_from(["valid", "valid", "repairable", "refused", "asked", "declined"]),
        slot=st.integers(0, 1),
    )
    @precondition(lambda self: self.s.active_generate is not None)
    def llm_done_with_tool(self, status, slot):
        tool = self._draft(status, "draft", slot)
        self._step(llm_done(self._advance(10), self.s.active_generate, tool))

    @rule(status=st.sampled_from(["valid", "repairable", "refused", None]), slot=st.integers(0, 1))
    @precondition(lambda self: self.s.state is TurnState.IDLE)
    def whole_tool_turn(self, status, slot):
        self.whole_turn_up_to_thinking_or_speaking(text="Opening it. ", answers=slot == 0)
        if self.s.active_generate is None:
            return
        tool = None if status is None else self._draft(status, "draft", slot)
        self._step(llm_done(self._advance(10), self.s.active_generate, tool))

    @rule(status=st.sampled_from(["valid", "refused", None]))
    @precondition(lambda self: self._repair_live(self.s))
    def repair_done(self, status):
        tool = None if status is None else self._draft(status, "repair", 0)
        self._step(llm_done(self._advance(10), self.s.active_generate, tool))

    @rule(kind=st.sampled_from(["answer", "answer", "announce", "open", "thought"]))
    @precondition(lambda self: self.s.active_generate is not None and not self._repair_live(self.s))
    def marked_commit(self, kind):
        """A commit carrying a unique marker, so P6' can tell withheld or unclosed text apart. T-0178 B5: an
        ``announce`` commit is answer text whose last sentence announces an action (``announces`` reads it as one)."""
        self.marker_seq += 1
        marker = f"STEPMARK{self.marker_seq:05d}"  # fixed width: no marker contains another
        text = (
            f"{marker} noted. I will click the button. "
            if kind == "announce"
            else f"{marker} noted. "
        )
        turn = self.s.turn_id
        if kind in ("answer", "announce"):
            self.committed_answer += text
            if self.step_of.get(turn, 1) >= 2:
                self.step_markers.setdefault(turn, []).append(marker)
        else:
            self.unspeakable.append(marker)  # criterion 5 adversarial: an open thought
        self._step(
            E.LlmCommit(
                t_ms=self._advance(300),
                request_id=self.s.active_generate,
                canvas_idx=self.canvas,
                segment="thought" if kind == "thought" else "answer",
                text=text,
                thought_open=kind == "open",
            )
        )
        self.canvas += 1

    @rule(status=st.sampled_from(TOOL_RESULT_STATUSES))
    @precondition(lambda self: self.s.turn_id in self.awaiting)
    def result_for_the_awaited_call(self, status):
        self._step(tool_result_event(self._advance(30), self.awaiting[self.s.turn_id], status))

    @rule(
        which=st.sampled_from(["answered", "unknown", "not_yet", "pending", "cut"]),
        status=st.sampled_from(TOOL_RESULT_STATUSES),
    )
    def unmatched_tool_result(self, which, status):
        """Criterion 4 adversarial: a duplicate (already answered), a stale or foreign id, one for
        a call not drafted yet, one for a call still pending (before its execute), and the last
        awaited call of an earlier turn that was cut (R17, R18, an error or a disconnect), which
        must stay inert in every later turn, its generate live or not (R13; per-turn reset)."""
        call = {
            "answered": self.answered[-1] if self.answered else "c0",
            "unknown": "zz-9",
            "not_yet": f"c{self.call_seq + 1}",
            "pending": (self._pending_calls(self.s.turn_id) or ["c0"])[0],
            "cut": self.cut_calls[-1] if self.cut_calls else "c0",
        }[which]
        self._step(tool_result_event(self._advance(30), call, status))

    @rule()
    @precondition(lambda self: self.s.turn_id in self.awaiting)
    def tick_to_tool_result_deadline(self):
        at = self.result_deadline.get(self.s.turn_id, self.t + TOOL_RESULT_MS)
        self._step(E.Tick(t_ms=self._advance(max(1, at - self.t))))

    @rule(
        status=st.sampled_from(
            # T-0178 B3; B3a: "capped" / "failcapped" are the forced-step search refusals
            [
                "valid",
                "valid",
                None,
                "refused",
                "repairable",
                "asked",
                "declined",
                "capped",
                "failcapped",
            ]
        ),
        slot=st.integers(0, 1),
        says=st.booleans(),
        announce=st.booleans(),  # T-0178 B5
    )
    @precondition(lambda self: self.s.turn_id in self.awaiting)
    def next_step(self, status, slot, says, announce=False):
        """Drive one whole step: the result, a canvas, and the step's llm.done with or without a
        call. Uniform rules reach step 3 (the R16 report step) only by luck."""
        turn = self.s.turn_id
        self.result_for_the_awaited_call("dry_run")
        if self.s.turn_id != turn or self.s.active_generate is None:
            return
        if says:
            self.marked_commit("announce" if announce else "answer")
        tool = None if status is None else self._draft(status, "draft", slot)
        self._step(llm_done(self._advance(10), self.s.active_generate, tool))

    @rule(
        depth=st.integers(
            1, max(LOOP_BUDGETS) + 1
        ),  # T-0178: reaches the report step of every budget
        prefix=st.booleans(),
        ending=st.sampled_from(
            [
                "answer",
                "silent",
                "refused",
                "repairable",
                "takeover",
                "timeout",
                "stale",
                "duplicate",
                "interrupt",
                "open",
                "takeover_late",
                "timeout_late",
                "announce",  # T-0178 B5: a no-call step that only announces
                "asked",  # T-0178 B3: policy refusals still voice
                "declined",
                "capped",  # T-0178 B3a: a forced-step search refusal after a plain answer (voiced)
                "capped_announce",  # T-0178 B3a: ... after an announcement only (withheld, not_finished)
                "failcapped",
                "failcapped_announce",
            ]
        ),
    )
    @precondition(lambda self: self.s.state is TurnState.IDLE)
    def whole_loop_turn(self, depth, prefix, ending):
        """Walk a loop turn on purpose: step 1 (with or without a spoken prefix) executes a valid
        call, ``depth - 1`` further steps execute one each (``depth = 3`` reaches the R16 report
        step), then one ending. Measured on a correct in-process prototype of R11-R18 (2026-10-17,
        hypothesis as locked, 300 x 80): without this rule the traces reached step 3 zero times
        and R16 zero times in 26 600 events, so the budget and R17 properties were decided by luck.
        """
        self.whole_turn_up_to_thinking_or_speaking(text="Opening it. ", answers=prefix)
        if self.s.active_generate is None:
            return
        self._step(
            llm_done(self._advance(10), self.s.active_generate, self._draft("valid", "draft", 0))
        )
        if self.s.state is TurnState.SPEAKING:
            self.playback_drained(played=100)
        for _ in range(depth - 1):
            if self.s.turn_id not in self.awaiting:
                return
            if ending == "stale":
                self.unmatched_tool_result(which="unknown", status="dry_run")
            self.next_step(status="valid", slot=1, says=True)
        turn = self.s.turn_id
        if turn not in self.awaiting:
            if self.s.state is TurnState.SPEAKING:
                self.playback_drained(played=100)  # the R16 report step's speech
            return
        capped = ending.removesuffix("_announce")
        if capped in SEARCH_CAP_STATUSES:
            # T-0178 B3a: the forced answer step says something and drafts a search refused at the cap
            self.next_step(status=capped, slot=0, says=True, announce=ending.endswith("_announce"))
            if self.s.state is TurnState.SPEAKING:
                self.playback_drained(played=100)
        elif ending in (
            "answer",
            "silent",
            "refused",
            "repairable",
            "announce",
            "asked",
            "declined",
        ):
            status = None if ending in ("answer", "silent", "announce") else ending
            self.next_step(
                status=status, slot=0, says=ending != "silent", announce=ending == "announce"
            )
            if self.s.state is TurnState.SPEAKING:
                self.playback_drained(played=100)
        elif ending == "takeover":
            self.word(word="ok")
            self.takeover()
        elif ending == "timeout":
            self.tick_to_tool_result_deadline()
        elif ending == "duplicate":
            self.result_for_the_awaited_call(status="executed")
            self.unmatched_tool_result(which="answered", status="executed")
        elif ending == "interrupt":
            self.next_step(status=None, slot=0, says=True)
            if self.s.state is TurnState.SPEAKING:
                self.onset()
                self.takeover()
                self.playback_stopped(played=100, frac=0.5)
        elif ending == "open":
            self.result_for_the_awaited_call(status="dry_run")
            if self.s.active_generate is not None:
                self.marked_commit(kind="open")
                self._step(llm_done(self._advance(10), self.s.active_generate))
        elif ending in ("takeover_late", "timeout_late"):
            # R13 adversarial, on purpose: the cut turn's awaited call answers once the *next*
            # turn is generating. The uniform "cut" choice of unmatched_tool_result reaches that
            # moment only by luck (a takeover or R18, then a whole endpoint, then the result).
            if ending == "takeover_late":
                self.word(word="ok")
                self.takeover()
                if self.s.state is TurnState.LISTENING:
                    self.end_of_turn()
                    if self.s.t_eot_ms is not None:
                        self.flushed_matching()
                    self.tick(POLICY.eot_settle_ms + 1)
            else:
                self.tick_to_tool_result_deadline()
                if self.s.state is TurnState.IDLE:
                    self.whole_turn_up_to_thinking_or_speaking(text="Yes. ", answers=False)
            if self.s.state is TurnState.THINKING and self.s.active_generate is not None:
                self.unmatched_tool_result(which="cut", status="dry_run")

    # --- L7 ------------------------------------------------------------------------------------

    @invariant()
    def every_loop_turn_reaches_idle(self):
        """L7: from any state of a turn that executed a call, ordinary events end it: the result
        deadline, the model's answer, a backchannel, the drain."""
        s = self.s
        if s.state is TurnState.IDLE or not self.executed.get(s.turn_id):
            return
        start, t = s.turn_id, self.t
        for _ in range(12):
            if s.state is TurnState.IDLE or s.turn_id != start:
                return
            t += 1
            if getattr(s, "awaiting_result", None) is not None:
                t = max([t, *(x.at_ms for x in s.timers if x.name == "tool_result")])
                ev = E.Tick(t_ms=t)
            elif s.active_generate is not None:
                ev = E.LlmDone(t_ms=t, request_id=s.active_generate)
            elif s.state is TurnState.PAUSED:
                ev = E.UserBackchannel(t_ms=t, duration_ms=200)
            elif s.state is TurnState.INTERRUPTED:
                t += POLICY.interrupt_timeout_ms
                ev = E.Tick(t_ms=t)
            else:
                ev = E.PlaybackDrained(t_ms=t, played_ms=0, played_text="")
            s, _ = self.m.step(s, ev)
        assert s.state is TurnState.IDLE or s.turn_id != start, (
            f"L7: loop turn {start} stuck in {s.state.value} after 12 ordinary events"
        )

    # --- L10 (T-0178 B6) -----------------------------------------------------------------------

    @invariant()
    def a_budget_cancelled_turn_is_idle_with_at_most_one_system_clip(self):
        """L10: every turn with a ``tool.cancel(step_budget)`` reaches IDLE (in the cancelling tuple: nothing of the
        turn is queued at a step >= 2, B4b) and emits at most one system clip (``_step`` counts them per turn)."""
        for turn in self.budget_cut:
            assert self.system_clips.get(turn, 0) <= 1, (
                f"L10: {turn} played {self.system_clips[turn]}"
            )
        if self.s.turn_id in self.budget_cut:
            assert self.s.state is TurnState.IDLE, (
                f"L10: {self.s.turn_id} cancelled step_budget and is {self.s.state.value}"
            )


class TestToolLoopInvariants(ToolLoopModel.TestCase):  # type: ignore[name-defined,misc]
    """T-0178 B6: the speech ledger is re-scoped to the honest ending (B2/B3/B5) and L9/L10 are added; its other
    properties are unchanged and ``choose_budget`` mixes budgets 1-3. Hypothesis reports several distinct
    counterexamples as an ``ExceptionGroup``; when every one of them is an ``AssertionError`` it is re-raised as one
    ``AssertionError`` (one readable failure; kept from the rows' pending phase), never otherwise, so a crash still
    surfaces as itself. Nothing is caught when the run passes."""

    def runTest(self) -> None:
        try:
            super().runTest()
        except BaseExceptionGroup as group:
            if all(isinstance(e, AssertionError) for e in group.exceptions):
                raise AssertionError(
                    f"{len(group.exceptions)} distinct property failures; first: {group.exceptions[0]}"
                ) from group
            raise


TestToolLoopInvariants.settings = settings(max_examples=300, stateful_step_count=80, deadline=None)


# --- L8: with tool_loop off, the machine is the frozen post-T-0089 one -----------------------------

_L8_TOOL = (
    st.tuples(
        st.just("tool_done"),
        st.sampled_from(["valid", "valid", "repairable", "refused"]),
        st.integers(0, 1),
    ),
    st.tuples(st.just("repair_done"), st.sampled_from(["valid", "refused", None])),
)
#: a spoken prefix, then a call, then the drain (or the drain first): the rows R11 rewrites
_L8_COMPOSITE = (
    st.tuples(
        st.just("tool_complete"),
        st.sampled_from(TurnMachineModel.COMMIT_TEXTS),
        st.sampled_from(["valid", "valid", "repairable", "refused"]),
        st.booleans(),
    ),
    st.tuples(st.just("tool_think"), st.sampled_from(["valid", "repairable", "refused"])),
)
_L8_RESULT = (
    st.tuples(
        st.just("result"),
        st.sampled_from(["last", "pending", "unknown"]),
        st.sampled_from(TOOL_RESULT_STATUSES),
    ),
)
_L8_EV = st.one_of(*_ATOMIC, *_L8_TOOL, *(_COMPOSITE * 2), *(_L8_COMPOSITE * 3))
_L8_EV_RESULTS = st.one_of(_L8_EV, *(_L8_RESULT * 3))


def _l8_expand(recipe, s):
    kind = recipe[0]
    if kind == "tool_complete":
        _, text, status, done_first = recipe
        ends = [("tool_done", status, 0), ("drained",)]
        return [("speak", text), *(ends if done_first else ends[::-1])]
    if kind == "tool_think":
        lead = [] if s.state is TurnState.IDLE else [("error", "llm")]
        if s.state is TurnState.THINKING and s.active_generate is not None:
            lead = []
        else:
            lead.append(("turn",))
        return [*lead, ("tool_done", recipe[1], 1)]
    return _expand(recipe, s)


def _l8_draft(status, call_id, slot, source="draft"):
    name, args = TOOL_NAMES[slot], TOOL_ARGS[slot]
    if status == "refused":
        return tool_draft(call_id=call_id, status=status, reason="r", source=source)
    if status == "repairable":
        return tool_draft(call_id=call_id, status=status, name=name, reason="schema")
    return tool_draft(call_id=call_id, status=status, name=name, arguments=args, source=source)


def _l8_events(recipe, s, t, spoken, calls):
    kind = recipe[0]
    if kind == "tool_done":
        if s.active_generate is None:
            return []
        calls.append(f"c{len(calls) + 1}")
        return [llm_done(t + 10, s.active_generate, _l8_draft(recipe[1], calls[-1], recipe[2]))]
    if kind == "repair_done":
        if s.active_generate is None or s.active_generate != repair_id(s.turn_id):
            return []
        status = recipe[1]
        draft = None if status is None else _l8_draft(status, calls[-1], 0, source="repair")
        return [llm_done(t + 10, s.active_generate, draft)]
    if kind == "result":
        call = {
            "last": calls[-1] if calls else "c0",
            "pending": s.tool.call_id if s.tool is not None else "c0",
            "unknown": "zz-9",
        }[recipe[1]]
        return [tool_result_event(t + 30, call, recipe[2])]
    return _events_for(recipe, s, t, spoken)


def _l8_trace(live, old, recipes, *, max_events=60):
    """Step both machines from the same state on every event; commands and the whole state must
    be equal (L8: "``step`` equals a frozen post-T-0089 machine on every sequence")."""
    s = live.initial()
    assert s == old.initial()
    t, spoken, calls, n = 0, "", [], 0
    queue = list(recipes)
    while queue and n < max_events:
        recipe = queue.pop(0)
        parts = _l8_expand(recipe, s)
        if parts is not None:
            queue[:0] = parts
            continue
        for ev in _l8_events(recipe, s, t, spoken, calls):
            if n >= max_events:
                break
            n += 1
            t = max(t, ev.t_ms)
            new_state, new_cmds = live.step(s, ev)
            old_state, old_cmds = old.step(s, ev)
            assert tuple(new_cmds) == tuple(old_cmds), (
                f"L8: {ev.type} in {s.state.value}: {kinds(new_cmds)} != frozen {kinds(old_cmds)}"
            )
            assert new_state == old_state, f"L8: {ev.type} in {s.state.value} diverged"
            spoken += "".join(c.source for c in new_cmds if isinstance(c, C.TtsSpeak))
            s = new_state
            if s.state is TurnState.IDLE:
                spoken = ""


def _l8_run(live_policy, events, *, segmenter, examples: int) -> None:
    seg = sentence_segmenter if segmenter == "sentences" else None

    @settings(max_examples=examples, deadline=None)
    @given(recipes=st.lists(events, min_size=4, max_size=40))
    def trace(recipes):
        live = TurnMachine(live_policy, segment=seg)
        old = frozen_t0089.TurnMachine(frozen_t0089.TurnPolicy(thinking=True), segment=seg)
        _l8_trace(live, old, recipes)

    trace()


@pytest.mark.parametrize("segmenter", ["default", "sentences"])
def test_loop_off_differential_default_policy_guard(segmenter):
    """L8 guard, unmarked: the default ``TurnPolicy`` steps exactly as the frozen post-T-0089
    machine on sequences with tool drafts, repairs and every T-0089 row (no ``ToolResult``: the
    event does not exist yet). Passes before T-0112 on purpose; it stays green while the loop rows
    are added only if they are all behind ``tool_loop`` and its default is off (H-T111a-1).
    Regime: 100 random examples of 4-40 recipes, at most 60 events, per segmenter, CPU, pure."""
    _l8_run(TurnPolicy(thinking=True), _L8_EV, segmenter=segmenter, examples=100)


@pytest.mark.parametrize("segmenter", ["default", "sentences"])
def test_loop_off_differential(segmenter):
    """L8 / criterion 1 (negative): with ``tool_loop=False`` set explicitly, ``step`` equals the
    frozen post-T-0089 machine (``_frozen_turn_machine_t0089``) on every sequence, ``ToolResult``
    events included (matching the last call, the pending call, or nobody): commands exactly and
    the whole state. Regime: 300 random examples of 4-40 recipes, at most 60 events, per segmenter.
    """
    _l8_run(loop_policy(tool_loop=False), _L8_EV_RESULTS, segmenter=segmenter, examples=300)


def test_l8_frozen_copy_is_the_base_turn_machine_verbatim():
    """The L8 oracle is only an oracle while nobody edits it: sha256 of the copied body, computed
    by hand from ``git show 40c1279:local_astra/conductor/turn_machine.py | sed '1,17d'``; and,
    when the base commit is present, byte equality with it."""
    import hashlib
    import subprocess

    text = pathlib.Path(frozen_t0089.__file__).read_text()
    start, end = "dropped) ---\n", "# --- end of verbatim copy ---\n"
    body = text[text.index(start) + len(start) : text.index(end)].rstrip("\n") + "\n"
    assert hashlib.sha256(body.encode()).hexdigest() == (
        "3f1d5e1755e84d72528d40d2ff8ddd728a9c8198544d286b480028445b6ada9f"
    )
    assert frozen_t0089.BASE_SHA == "40c12794ecb871654319a1fd35db6aca69c414d6"
    shown = subprocess.run(
        ["git", "show", f"{frozen_t0089.BASE_SHA}:local_astra/conductor/turn_machine.py"],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).parent,
    )
    if shown.returncode != 0:
        pytest.skip(f"base commit {frozen_t0089.BASE_SHA} is not in this clone")
    assert body == "".join(shown.stdout.splitlines(keepends=True)[17:])


# --- T-0173 criterion 7 (P10): deadlines counted "from now" are never armed in the past -----------------------

#: The timers armed from the stepping event's own time. ``takeover`` is excluded: it is armed from ``onset_ms``, the
#: ADC clock (A13, out of T-0173's scope).
FROM_NOW_TIMERS = frozenset(
    {"tool_result", "tool_repair", "llm_timeout", "interrupt_timeout", "resume"}
)


class _OneClockMixin:
    """Adds one check to every step of the host model, and changes nothing else: while the event times stepped so
    far are non-decreasing, every ``ArmTimer`` in ``FROM_NOW_TIMERS`` has ``at_ms >=`` the largest ``t_ms`` stepped so
    far (inclusive of the arming event). ``checked`` counts the arms checked, per timer name, across a run."""

    checked: ClassVar[dict[str, int]] = {}

    def _step(self, ev):
        latest = max(getattr(self, "_p10_latest", 0), ev.t_ms)
        monotone = getattr(self, "_p10_monotone", True) and ev.t_ms >= getattr(
            self, "_p10_latest", 0
        )
        self._p10_latest, self._p10_monotone = latest, monotone
        cmds = super()._step(ev)
        if monotone:
            for c in cmds:
                if isinstance(c, C.ArmTimer) and c.name in FROM_NOW_TIMERS:
                    assert c.at_ms >= latest, (
                        f"T-0173 P10: {c.name} armed at {c.at_ms}, before the latest event time {latest} "
                        f"(on {ev.type} in {self.s.state.value})"
                    )
                    type(self).checked[c.name] = type(self).checked.get(c.name, 0) + 1
        return cmds


class _OneClockLoopModel(_OneClockMixin, ToolLoopModel):
    checked: ClassVar[dict[str, int]] = {}


class _OneClockToolModel(_OneClockMixin, ToolTurnModel):
    checked: ClassVar[dict[str, int]] = {}


# T-0178: the loop model's ledger is the honest ending (B6), red on BASE
def test_from_now_deadlines_never_armed_before_latest_event():
    """T-0173 criterion 7: on non-decreasing event times, no ``tool_result``/``tool_repair``/``llm_timeout``/
    ``interrupt_timeout``/``resume`` deadline is armed before the latest event time. Runs the loop model (loop on) and
    the tool model (loop off, repairs) with every one of their own checks, then requires that each of the five
    timers was actually armed and checked, so the property cannot pass vacuously. Holds on the base machine (it is
    pure and unchanged by T-0173). Regime: 150 + 150 examples, 60 steps each, CPU.

    T-0178: the loop model it runs carries T-0178's re-scoped speech ledger (B6); a Hypothesis ``ExceptionGroup`` of
    assertions is re-raised as one ``AssertionError`` (as ``TestToolLoopInvariants``)."""
    for model in (_OneClockLoopModel, _OneClockToolModel):
        model.checked = {}
        try:
            run_state_machine_as_test(
                model, settings=settings(max_examples=150, stateful_step_count=60, deadline=None)
            )
        except BaseExceptionGroup as group:
            if all(isinstance(e, AssertionError) for e in group.exceptions):
                raise AssertionError(f"{model.__name__}: {group.exceptions[0]}") from group
            raise
    seen = {
        name: _OneClockLoopModel.checked.get(name, 0) + _OneClockToolModel.checked.get(name, 0)
        for name in FROM_NOW_TIMERS
    }
    # ``resume`` (PAUSED + a short offset) is armed in only some runs of the models (measured: 0 checks in 1 of 9
    # runs), so it also gets one fixed trace: SPEAKING -> onset (PAUSED) -> short offset, all times non-decreasing.
    m = TurnMachine(POLICY)
    s, _ = to_thinking(m)
    s, _ = m.step(s, commit(s, "Ok, here it is.", t=3000))
    latest = 3000
    for ev in (
        E.UserSpeechOnset(t_ms=3400, onset_ms=3400),
        E.UserSpeechOffset(t_ms=3700, offset_ms=3700, speech_ms=300),
    ):
        latest = max(latest, ev.t_ms)
        s, cmds = m.step(s, ev)
        for c in cmds:
            if isinstance(c, C.ArmTimer) and c.name in FROM_NOW_TIMERS:
                assert c.at_ms >= latest, f"T-0173 P10: {c.name} armed at {c.at_ms} < {latest}"
                seen[c.name] += 1
    assert all(seen.values()), (
        f"T-0173 criterion 7: every from-now timer must be exercised; checked {seen}"
    )


# --- T-0192 B3: the checkout hand-back --------------------------------------------------------------------------------
#
# B2 (docs/02 §4 spec delta): ``tool.result(call_id = awaiting_result, hand_back = "checkout")`` in THINKING with the
# loop on ends the turn: cancel ``tool_result``, ``play_clip(at_checkout)``, ``commit_history(user_only = false)``,
# ``schedule_summary``, IDLE; any other hand-back is inert (R13). B3's properties, never weakened, on traces that mix
# hand-backs into the loop model (``HandBackLoopModel``): L7 (the turn reaches IDLE, in the hand-back's own tuple), L10
# extended (at most one *ending* clip per turn, ``at_checkout`` counted with T-0178/T-0179's), L1/L3 (no
# ``llm.continue``, ``tool.execute`` or ``llm.start`` for a handed-back turn), H8/L6 (what is stored is a join of the
# turn's played runs, so never the clip). L8 with hand-backs: loop off, the machine is the frozen da29ddbe one (which
# is the post-T-0191 machine: byte-identical at a9a11add). New differential: with no ``hand_back`` anywhere, the
# machine equals the frozen a9a11add (= da29ddbe) machine, loop on and off, still_working on and off.


def _frozen_t0192():
    """The frozen da29ddbe machine, imported on use, never at module level: BASE-tree runs of earlier tasks'
    differentials import this module against trees that predate the modules the copy imports (``announced_call``)."""
    from tests.unit import _frozen_turn_machine_t0192

    return _frozen_turn_machine_t0192


#: L10 extended: every ending clip (T-0178 B4a, T-0179 B-6a, T-0192 B1)
T0192_ENDING_CLIPS = frozenset({*SYSTEM_CLIPS, "lost_thread", "at_checkout"})
#: B4: the statuses the session hands back with
HAND_BACK_STATUSES = ("executed", "refused")
#: T-0182 B4's still_working deadline used by the traces (the profile minimum, 5 s), or off
T0192_STILL_WORKING = (None, 5_000)


def hand_back_event(t_ms: int, call_id: str, status: str = "executed"):
    assert "hand_back" in E.ToolResult.model_fields, (
        'T-0192 B1: events.ToolResult gains `hand_back: Literal["checkout"] | None = None`; its fields are '
        f"{sorted(E.ToolResult.model_fields)}"
    )
    return E.ToolResult(t_ms=t_ms, call_id=call_id, status=status, hand_back="checkout")


def is_hand_back(ev) -> bool:
    return is_tool_result(ev) and getattr(ev, "hand_back", None) is not None


class HandBackLoopModel(ToolLoopModel):
    """``ToolLoopModel`` (every one of its checks) plus hand-backs: the awaited call handed back (B2), unmatched
    hand-backs (R13), a whole loop turn walked to step 1..3 and handed back, optionally after a long wait that plays
    still_working (budgets 1-3, still_working on or off). The matched hand-back is the one event whose tuple the
    parent's R12 premise does not cover (it would demand an ``llm.continue``), so that event alone is checked here,
    against B2, with the parent's base checks and its turn-boundary bookkeeping; every other event, unmatched
    hand-backs included, goes through the parent unchanged (its R13 check: an unmatched result moves nothing).
    ``counts`` measures the power (matched hand-backs, at step >= 2, after the last allowed execute, over a playing
    still_working)."""

    counts: ClassVar[dict[str, int]] = {}

    @initialize(budget=st.sampled_from(LOOP_BUDGETS), sw=st.sampled_from(T0192_STILL_WORKING))
    def choose_budget(self, budget, sw):
        self.max_steps = budget
        self.m = TurnMachine(
            loop_policy(tool_loop=True, max_tool_steps=budget, still_working_after_ms=sw),
            segment=self._segment,
        )

    def __init__(self):
        super().__init__()
        self.ending_clips: dict[str, int] = {}
        self.handed_back: set[str] = set()
        self.hand_back_calls: list[str] = []

    def _count(self, key: str) -> None:
        type(self).counts[key] = type(self).counts.get(key, 0) + 1

    def _step(self, ev):
        before = self.s
        turn = before.turn_id
        matched = (
            is_hand_back(ev)
            and self.awaiting.get(turn) == ev.call_id
            and before.state is TurnState.THINKING
        )
        cmds = self._hand_back_step(ev) if matched else list(super()._step(ev))
        kinds_played = [c.kind for c in cmds if isinstance(c, C.PlayClip)]
        ends = [k for k in kinds_played if k in T0192_ENDING_CLIPS]
        if ends:
            self.ending_clips[turn] = self.ending_clips.get(turn, 0) + len(ends)
            assert self.ending_clips[turn] <= 1, (
                f"L10 (T-0192): {self.ending_clips[turn]} ending clips in {turn}: {kinds(cmds)}"
            )
        if not matched:
            assert "at_checkout" not in kinds_played, (
                f"B2 (guard width): at_checkout on {ev.type} in {before.state.value}: {kinds(cmds)}"
            )
        if turn in self.handed_back and not matched:
            # L1/L3 after B2: the handed-back turn asks the model nothing more and runs nothing more
            for c in cmds:
                assert not isinstance(c, C.LlmStart | C.TtsSpeak) and not is_continue(c), (
                    f"L1/L3 (T-0192): {kinds(cmds)} for {turn} after its hand-back"
                )
                assert type(c).__name__ != "ToolExecute", (
                    f"L4 (T-0192): {kinds(cmds)} after a hand-back"
                )
        return cmds

    def _hand_back_step(self, ev):
        before = self.s
        turn = before.turn_id
        k = self.step_of.get(turn, 1)
        self.check_step = k
        cmds = list(TurnMachineModel._step(self, ev))
        # L7: the hand-back ends the turn in its own tuple
        assert self.s.state is TurnState.IDLE, (
            f"L7/B2: a hand-back left {turn} {self.s.state.value}"
        )
        assert self.s.awaiting_result is None and self.s.active_generate is None
        assert not [c for c in self.s.timers if c.name == "tool_result"], self.s.timers
        # B2's tuple (a StopClip may lead only while a clip plays; the exact lead is pinned in the row module)
        rest = [c for c in cmds if not isinstance(c, C.StopClip)]
        assert rest == [
            C.CancelTimer(name="tool_result"),
            C.PlayClip(kind="at_checkout"),
            C.CommitHistory(user_only=False),
            C.ScheduleSummary(),
            C.HudState(state=TurnState.IDLE.value, turn_id=turn),
        ], f"B2: {kinds(cmds)}"
        assert before.clip_playing or len(rest) == len(cmds), (
            f"B2: StopClip with no clip playing: {kinds(cmds)}"
        )
        # H8/L6: what is stored is a join of the turn's played runs (never the clip, never withheld text)
        if self.s.history != before.history:
            entry = self.s.history[-1]
            assert not entry.interrupted, entry
            assert joins_played_runs(stored_assistant_text(entry), self.runs.get(turn, [])), (
                f"L6/H8 (T-0192): stored {entry.assistant_spoken!r}; played runs {self.runs.get(turn, [])}"
            )
            for marker in self.step_markers.get(turn, []):
                assert marker not in entry.assistant_spoken, f"L9/L6: withheld {marker} stored"
            self.l6_checked.append(entry)
        # the parent's turn-boundary bookkeeping for a turn that ends here (P8': every opened call ended once)
        self.closed.add(turn)
        for call in self.opened.get(turn, []):
            assert len(self.terminals.get(call, [])) == 1, (
                f"P8': call {call} of {turn} ended {len(self.terminals.get(call, []))} times"
            )
        self.awaiting.pop(turn, None)
        self.cut.add(turn)
        self.step_of.pop(turn, None)
        self.spoke_since_drain = False
        self.run_spoken = ""
        self.answered.append(ev.call_id)
        self.handed_back.add(turn)
        self.hand_back_calls.append(ev.call_id)
        self._count("matched")
        if k >= 2:
            self._count("step_k")
        if before.tool_steps_executed >= self.max_steps:
            self._count("last_execute")
        if before.clip_playing:
            self._count("over_still_working")
        return cmds

    # --- rules ---------------------------------------------------------------------------------

    @rule(status=st.sampled_from(HAND_BACK_STATUSES))
    @precondition(lambda self: self.s.turn_id in self.awaiting)
    def hand_back_for_the_awaited_call(self, status):
        self._step(hand_back_event(self._advance(30), self.awaiting[self.s.turn_id], status))

    @rule(
        which=st.sampled_from(["answered", "unknown", "not_yet", "pending", "cut", "handed"]),
        status=st.sampled_from(HAND_BACK_STATUSES),
    )
    def unmatched_hand_back(self, which, status):
        """R13 with ``hand_back``: a duplicate, a foreign id, a call not drafted yet, one still pending its execute, a
        cut turn's call and an already handed-back call, in every state the trace is in."""
        call = {
            "answered": self.answered[-1] if self.answered else "c0",
            "unknown": "zz-9",
            "not_yet": f"c{self.call_seq + 1}",
            "pending": (self._pending_calls(self.s.turn_id) or ["c0"])[0],
            "cut": self.cut_calls[-1] if self.cut_calls else "c0",
            "handed": self.hand_back_calls[-1] if self.hand_back_calls else "c0",
        }[which]
        self._step(hand_back_event(self._advance(30), call, status))

    @rule()
    @precondition(lambda self: self.s.turn_id in self.awaiting)
    def long_wait(self):
        """5.2 s while a result is awaited: still_working plays when it is on (its deadline counts from llm.start)."""
        self._step(E.Tick(t_ms=self._advance(5_200)))

    @rule(
        depth=st.integers(1, max(LOOP_BUDGETS)),
        prefix=st.booleans(),
        wait=st.booleans(),
        status=st.sampled_from(HAND_BACK_STATUSES),
    )
    @precondition(lambda self: self.s.state is TurnState.IDLE)
    def whole_hand_back_turn(self, depth, prefix, wait, status):
        """Walk a loop turn on purpose to step ``depth`` (each step executes; ``depth`` equal to the budget is the last
        allowed execute) and hand the awaited call back, after a long wait or not."""
        self.whole_turn_up_to_thinking_or_speaking(text="Opening it. ", answers=prefix)
        if self.s.active_generate is None:
            return
        self._step(
            llm_done(self._advance(10), self.s.active_generate, self._draft("valid", "draft", 0))
        )
        if self.s.state is TurnState.SPEAKING:
            self.playback_drained(played=100)
        for _ in range(depth - 1):
            if self.s.turn_id not in self.awaiting:
                return
            self.next_step(status="valid", slot=1, says=True)
        if self.s.turn_id not in self.awaiting:
            return
        if wait:
            self.long_wait()
        if self.s.turn_id in self.awaiting:
            self.hand_back_for_the_awaited_call(status)

    @invariant()
    def a_handed_back_turn_stays_idle(self):
        """L7 (T-0192): while a handed-back turn is the current one, the machine is IDLE."""
        if self.s.turn_id in self.handed_back:
            assert self.s.state is TurnState.IDLE, (
                f"L7: handed-back {self.s.turn_id} is {self.s.state.value}"
            )


def _run_hand_back_model(model, examples: int) -> None:
    try:
        run_state_machine_as_test(
            model,
            settings=settings(
                max_examples=examples,
                stateful_step_count=80,
                deadline=None,
                database=None,
                suppress_health_check=[HealthCheck.too_slow],
            ),
        )
    except BaseExceptionGroup as group:
        if all(isinstance(e, AssertionError) for e in group.exceptions):
            raise AssertionError(
                f"{len(group.exceptions)} distinct property failures; first: {group.exceptions[0]}"
            ) from group
        raise


def test_t0192_hand_back_invariants():
    """T-0192 B3: L7, L10 extended, L1/L3, H8/L6, R13 and every ``ToolLoopModel`` property on traces with hand-backs;
    then the power: matched hand-backs happened, at a step >= 2, after the last allowed execute and over a playing
    still_working. Regime: 200 examples x 80 steps, CPU, pure."""
    HandBackLoopModel.counts = {}
    _run_hand_back_model(HandBackLoopModel, 200)
    got = HandBackLoopModel.counts
    for key in ("matched", "step_k", "last_execute", "over_still_working"):
        assert got.get(key, 0) > 0, f"T-0192 B3 power: no hand-back reached {key!r}; counts {got}"


# --- L8 with hand-backs, and the no-hand_back differential (frozen da29ddbe) ----------------------------------------

_T0192_HAND_BACK = (
    st.tuples(
        st.just("hand_back"),
        st.sampled_from(["last", "pending", "unknown"]),
        st.sampled_from(HAND_BACK_STATUSES),
    ),
)
_T0192_LONG = (st.tuples(st.just("tick"), st.integers(1_000, 7_000)),)
_T0192_EV = st.one_of(_L8_EV_RESULTS, *(_T0192_LONG * 2))
_T0192_EV_HB = st.one_of(_T0192_EV, *(_T0192_HAND_BACK * 3))


def _t0192_trace(live, old, recipes, *, max_events=60):
    """``_l8_trace`` with ``hand_back`` recipes (a hand-back of the last call, the pending call or nobody): both machines
    step from the same state; commands exactly and the whole state must be equal."""
    s = live.initial()
    assert s == old.initial()
    t, spoken, calls, n = 0, "", [], 0
    queue = list(recipes)
    while queue and n < max_events:
        recipe = queue.pop(0)
        if recipe[0] == "hand_back":
            call = {
                "last": calls[-1] if calls else "c0",
                "pending": s.tool.call_id if s.tool is not None else "c0",
                "unknown": "zz-9",
            }[recipe[1]]
            events = [hand_back_event(t + 30, call, recipe[2])]
        else:
            parts = _l8_expand(recipe, s)
            if parts is not None:
                queue[:0] = parts
                continue
            events = _l8_events(recipe, s, t, spoken, calls)
        for ev in events:
            if n >= max_events:
                break
            n += 1
            t = max(t, ev.t_ms)
            new_state, new_cmds = live.step(s, ev)
            old_state, old_cmds = old.step(s, ev)
            assert tuple(new_cmds) == tuple(old_cmds), (
                f"T-0192: {ev.type} in {s.state.value}: {kinds(new_cmds)} != frozen {kinds(old_cmds)}"
            )
            assert new_state == old_state, f"T-0192: {ev.type} in {s.state.value} diverged"
            spoken += "".join(c.source for c in new_cmds if isinstance(c, C.TtsSpeak))
            s = new_state
            if s.state is TurnState.IDLE:
                spoken = ""


def _t0192_run(events, *, loop: bool, segmenter: str, examples: int) -> None:
    seg = sentence_segmenter if segmenter == "sentences" else None

    @settings(max_examples=examples, deadline=None)
    @given(
        recipes=st.lists(events, min_size=4, max_size=40),
        budget=st.sampled_from(LOOP_BUDGETS),
        sw=st.sampled_from(T0192_STILL_WORKING),
    )
    def trace(recipes, budget, sw):
        fields = {
            "thinking": True,
            "tool_loop": loop,
            "max_tool_steps": budget,
            "still_working_after_ms": sw,
        }
        live = TurnMachine(TurnPolicy(**fields), segment=seg)
        frozen_t0192 = _frozen_t0192()
        old = frozen_t0192.TurnMachine(frozen_t0192.TurnPolicy(**fields), segment=seg)
        _t0192_trace(live, old, recipes)

    trace()


@pytest.mark.parametrize("loop", [True, False], ids=["loop_on", "loop_off"])
@pytest.mark.parametrize("segmenter", ["default", "sentences"])
def test_t0192_without_hand_back_the_machine_is_frozen_a9a11add(loop, segmenter):
    """T-0192 B3, the new differential: on sequences with no ``hand_back`` (tool drafts, repairs, results, long
    waits), ``step`` equals the frozen a9a11add (= da29ddbe) machine, commands exactly and the whole state, loop on
    and off, budgets 1-3, still_working off or 5 s. Holds at BASE (the live file is the frozen one) and must keep
    holding. Regime: 150 random examples of 4-40 recipes, at most 60 events, per cell.

    What it does not cover (B-RV5): on the ``loop_on`` arm the ``_l8`` recipes almost never reach a result for the
    awaited call, so the loop-on R12 continue and the budget endings are not compared with the frozen machine here.
    Measured 2026-10-29 at 90691869 (150 examples, database off): loop on, 33 ``ToolResult`` events, 0 matching
    ``awaiting_result``, 1 for the pending call; loop off, 64 results. Matched loop-on results are covered by
    ``ToolLoopModel`` / ``HandBackLoopModel`` (properties, not this differential)."""
    _t0192_run(_T0192_EV, loop=loop, segmenter=segmenter, examples=150)


@pytest.mark.parametrize("segmenter", ["default", "sentences"])
def test_t0192_l8_loop_off_hand_back_is_the_frozen_machine(segmenter):
    """T-0192 B3, L8: with ``tool_loop=False`` every ``hand_back`` is inert, so on sequences that include hand-backs
    (of the last call, the pending call, nobody) ``step`` equals the frozen post-T-0191 (da29ddbe) machine. Pending:
    BASE cannot build a hand-back. Regime: 150 random examples of 4-40 recipes, at most 60 events, per segmenter."""
    _t0192_run(_T0192_EV_HB, loop=False, segmenter=segmenter, examples=150)


def test_t0192_frozen_copy_is_the_base_turn_machine_verbatim():
    """The B3 oracle is only an oracle while nobody edits it: sha256 of the copied body, computed from ``git show
    da29ddbe:local_astra/conductor/turn_machine.py | sed '1,17d'``; and, when the base commit is present, byte equality
    with it."""
    import hashlib
    import subprocess

    frozen_t0192 = _frozen_t0192()
    text = pathlib.Path(frozen_t0192.__file__).read_text()
    start, end = "dropped) ---\n", "# --- end of verbatim copy ---\n"
    body = text[text.index(start) + len(start) : text.index(end)].rstrip("\n") + "\n"
    assert hashlib.sha256(body.encode()).hexdigest() == (
        "84b20dc3b790d22c9fc53df3a05692b0f630596e892646418b5a7aa3e3831385"
    )
    assert frozen_t0192.BASE_SHA == "da29ddbe644f7ed4af21e83968aa83a8a5aa47c3"
    shown = subprocess.run(
        ["git", "show", f"{frozen_t0192.BASE_SHA}:local_astra/conductor/turn_machine.py"],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).parent,
    )
    if shown.returncode != 0:
        pytest.skip(f"base commit {frozen_t0192.BASE_SHA} is not in this clone")
    assert body == "".join(shown.stdout.splitlines(keepends=True)[17:])


# --- T-0193 A0: the user_stop rule's reach ---------------------------------------------------------------------------
#
# ``TurnMachineModel.user_stop`` (inherited by every model above) checks A0-2..A0-7 on each stop and
# ``a_stopped_turn_never_speaks_or_relistens`` watches the rest of the trace. ``UserStopLoopModel`` adds one driver that
# walks a loop turn to an awaited result (and on to step k) before stopping it, so the loop states are reached on
# purpose. The power row below requires every kind of stop to have happened.


class UserStopLoopModel(HandBackLoopModel):
    """``HandBackLoopModel`` (every loop, hand-back and still_working check) plus a driver: a loop turn walked to step
    ``depth`` (each step executes), optionally after a long wait that plays still_working, then stopped."""

    @rule(
        depth=st.integers(1, max(LOOP_BUDGETS)),
        prefix=st.booleans(),
        wait=st.booleans(),
        into_step=st.booleans(),
    )
    @precondition(lambda self: hasattr(E, "UserStop") and self.s.state is TurnState.IDLE)
    def loop_turn_then_stop(self, depth, prefix, wait, into_step):
        self.whole_turn_up_to_thinking_or_speaking(text="Opening it. ", answers=prefix)
        if self.s.active_generate is None:
            return
        self._step(
            llm_done(self._advance(10), self.s.active_generate, self._draft("valid", "draft", 0))
        )
        if self.s.state is TurnState.SPEAKING:
            self.playback_drained(played=100)
        for _ in range(depth - 1):
            if self.s.turn_id not in self.awaiting:
                break
            self.next_step(status="valid", slot=1, says=True)
        if wait and self.s.turn_id in self.awaiting:
            self.long_wait()
        if into_step and self.s.turn_id in self.awaiting:
            self.result_for_the_awaited_call("dry_run")
        self.user_stop(which="current", exit_="stopped", frac=0.5)


class UserStopToolModel(ToolTurnModel):
    """``ToolTurnModel`` (loop off; R1-R10, P1-P8) plus a driver: a tool turn whose draft starts the repair (or is
    pending, or refused), then stopped, so the stop meets a live repair and a pending call on purpose."""

    @rule(
        status=st.sampled_from(["valid", "repairable", "repairable", "refused"]),
        slot=st.integers(0, 1),
        exit_=st.sampled_from(["stopped", "timeout", "none"]),
    )
    @precondition(lambda self: hasattr(E, "UserStop") and self.s.state is TurnState.IDLE)
    def tool_turn_then_stop(self, status, slot, exit_):
        self.whole_tool_turn(status=status, slot=slot)
        self.user_stop(which="current", exit_=exit_, frac=0.5)


def test_t0193_user_stop_invariants():
    """T-0193 A0 (Hypothesis): every ``TurnMachineModel`` and loop property holds on traces with stops injected
    anywhere (current, previous, next or a foreign turn id), and the stops reached every case: a no-op by state and by
    id, THINKING (awaiting a result, with the repair live, with a clip playing), SPEAKING, PAUSED, and both exits of a
    stopped INTERRUPTED. Counters are process-wide and reset here. Regime: 200 + 200 examples x 40/80 steps, CPU,
    pure, database off."""
    TurnMachineModel.stop_counts = {}
    run_state_machine_as_test(
        TurnMachineModel,
        settings=settings(max_examples=200, stateful_step_count=40, deadline=None, database=None),
    )
    base_counts = dict(TurnMachineModel.stop_counts)
    _run_hand_back_model(UserStopLoopModel, 200)
    got = TurnMachineModel.stop_counts
    for key in (
        "noop_state",
        "noop_wrong_id",
        "thinking",
        "speaking",
        "paused",
        "exit_stopped",
        "exit_timeout",
        "clip",
    ):
        assert base_counts.get(key, 0) > 0, (
            f"T-0193 power (base model): no stop reached {key!r}; {base_counts}"
        )
    for key in ("awaiting", "thinking", "speaking"):
        assert got.get(key, 0) > base_counts.get(key, 0), (
            f"T-0193 power (loop model): no stop reached {key!r}; {got}"
        )
    run_state_machine_as_test(
        UserStopToolModel,
        settings=settings(max_examples=100, stateful_step_count=40, deadline=None, database=None),
    )
    assert TurnMachineModel.stop_counts.get("repair", 0) > 0, (
        f"T-0193 power: no stop with the repair live; {TurnMachineModel.stop_counts}"
    )
