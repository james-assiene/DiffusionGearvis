"""Frozen copy of ``main``'s ``local_astra/conductor/turn_machine.py`` for T-0089 P9 (the differential).

Source: commit e0174f9, blob sha 683bfd10cb8acedc64bad54965686b6621d45bbb
(``git rev-parse e0174f9:local_astra/conductor/turn_machine.py``). Copied verbatim below the
marker line; never edit it. P9 (``tests/unit/test_turn_invariants.py``) steps this machine and the
live one on the same event sequences whose ``llm.done`` events carry no ``tool`` and requires
identical output: a tool turn may add rows, never change one.
"""

# ruff: noqa
# --- verbatim copy of main from its line 18 on (its module docstring, lines 1-17, dropped) ---

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.commands import Command, TimerName
from local_astra.contracts.events import Event
from local_astra.contracts.turn import HistoryTurn, MachineState, Timer, TurnState, Word


@dataclass(frozen=True)
class TurnPolicy:
    """Tunables with the design-doc defaults (docs/01 §8, docs/02 §4)."""

    eot_settle_ms: int = 120  # ENDPOINTING → THINKING after all words ≤ t_eot and no onset
    eot_p_threshold: float = 0.6  # stt.end_of_turn below this is not an endpoint (docs/01 §8)
    owner_check_min_speech_ms: int = 1500  # speaker embeddings need ≥ 1.5 s (docs/01 D7)
    takeover_ms: int = 600  # PAUSED: speech longer than this is a takeover
    resume_max_ms: int = 600  # PAUSED: speech shorter than this + backchannel words → resume
    pause_latency_budget_ms: int = 150  # SPEAKING → PAUSED: playback.pause must land within
    interrupt_timeout_ms: int = 300  # INTERRUPTED without playback.stopped → estimate played_ms
    llm_timeout_ms: int = 20_000
    interim_update_ms: int = 4_000  # spoken "still working" update while thinking (docs/11 §1.1)
    played_word_margin_ms: int = 100  # a word counts as played if it started ≥ this before the cut
    ack_first_canvas_ema_ms: int = 300  # play ack clip iff thinking on or EMA(first canvas) > this
    first_canvas_ema_alpha: float = 0.3  # EMA weight of the newest first-canvas measurement
    thinking: bool = True  # session setting, never per request (docs/01 D11)
    max_history_turns: int = 3
    max_frames_per_turn: int = 3  # screens one turn may show the model (T-0059 D2)
    max_buffered_words: int = 256  # per word buffer; oldest dropped first


class Segmenter(Protocol):
    """Splits committed answer text into speakable sentences (T-0004 supplies the real one).

    Returns ``(sentences, remainder)`` where each sentence is ``(source, text)``: ``source`` is the
    exact slice of the input it consumed, ``text`` its TTS rendering. The machine *checks* the
    conservation law ``"".join(sources) + remainder == text`` on every call and refuses to speak
    when it does not hold — a segmenter is never trusted with what reaches TTS.
    """

    def __call__(self, text: str, *, final: bool) -> tuple[Sequence[tuple[str, str]], str]: ...


def default_segmenter(text: str, *, final: bool) -> tuple[Sequence[tuple[str, str]], str]:
    """No sentence splitting: the whole commit is one utterance (criterion 10)."""
    if not text:
        return [], ""
    return [(text, text)], ""


#: Fields that belong to one turn and are cleared when it ends or a new one opens.
_PER_TURN_RESET: dict[str, Any] = {
    "onset_ms": None,
    "t_eot_ms": None,
    "flush_t_ms": None,
    "llm_start_ms": None,
    "pause_onset_ms": None,
    "active_generate": None,
    "active_prefill": None,
    "committed_answer": "",
    "spoken_text": "",
    "held_text": "",
    "played_text": "",
    "utterances": (),
    "turn_phashes": (),
    "utterance_seq": 0,
    "fillers_this_turn": 0,
    "first_canvas_measured": False,
    "clip_playing": False,
    "llm_done": False,
    "tts_done": False,
    "playback_drained": False,
    "timers": (),
}


def _buffer(words: tuple[Word, ...], word: Word, cap: int) -> tuple[Word, ...]:
    """Append by *audio* order, ignore a re-delivered word (A8), keep the newest ``cap``."""
    if cap <= 0:
        return ()
    if word in words:
        return words
    i = len(words)
    while i > 0 and words[i - 1].start_ms > word.start_ms:
        i -= 1
    out = (*words[:i], word, *words[i:])
    return out[-cap:] if len(out) > cap else out


def _merge(words: tuple[Word, ...], more: tuple[Word, ...], cap: int) -> tuple[Word, ...]:
    for word in more:
        words = _buffer(words, word, cap)
    return words


def _transcript(words: tuple[Word, ...]) -> str:
    return " ".join(w.text for w in words)


def _timer_at(state: MachineState, name: TimerName) -> int | None:
    for timer in state.timers:
        if timer.name == name:
            return timer.at_ms
    return None


def _arm(state: MachineState, name: TimerName, at_ms: int) -> tuple[MachineState, C.ArmTimer]:
    timers = (*(t for t in state.timers if t.name != name), Timer(name=name, at_ms=at_ms))
    return state.model_copy(update={"timers": timers}), C.ArmTimer(name=name, at_ms=at_ms)


def _cancel(state: MachineState, *names: TimerName) -> tuple[MachineState, list[Command]]:
    """Cancel the armed timers among ``names`` (an unarmed one produces no command)."""
    live = {t.name for t in state.timers}
    hit = [name for name in names if name in live]
    if not hit:
        return state, []
    timers = tuple(t for t in state.timers if t.name not in hit)
    cmds: list[Command] = [C.CancelTimer(name=name) for name in hit]
    return state.model_copy(update={"timers": timers}), cmds


def _append_history(state: MachineState, turn: HistoryTurn, keep: int) -> tuple[HistoryTurn, ...]:
    if keep <= 0:
        return ()
    return (*state.history, turn)[-keep:]


class TurnMachine:
    """Stateless namespace around ``step``; instances only carry the policy and the segmenter."""

    def __init__(
        self, policy: TurnPolicy | None = None, *, segment: Segmenter | None = None
    ) -> None:
        self.policy = policy or TurnPolicy()
        self.segment: Segmenter = segment or default_segmenter

    def initial(self) -> MachineState:
        return MachineState()

    def step(self, state: MachineState, event: Event) -> tuple[MachineState, Sequence[Command]]:
        """Apply one event. Pure. Returns the new state and the commands to execute, in order."""
        new, cmds = self._dispatch(state, event)
        if new.state is not state.state:
            # criterion 21: exactly one hud.state per state change, carrying the turn now current
            # (A2). It goes last so the first command of a row stays the row's own (criterion 16).
            cmds = [*cmds, C.HudState(state=new.state.value, turn_id=new.turn_id)]
        return new, tuple(cmds)

    # --- dispatch -------------------------------------------------------------------------------

    def _dispatch(self, state: MachineState, event: Event) -> tuple[MachineState, list[Command]]:
        if isinstance(event, E.ClientDisconnected):
            return self._on_client_disconnected(state, event)
        if isinstance(event, E.LlmError):
            return self._on_llm_error(state, event)
        if isinstance(event, E.TtsError):
            return self._on_tts_error(state, event)
        if isinstance(event, E.SttError):
            return self._on_stt_error(state, event)
        if isinstance(event, E.SttWord):
            return self._on_word(state, event)
        if isinstance(event, E.SttEndOfTurn):
            return self._on_end_of_turn(state, event)
        if isinstance(event, E.SttFlushed):
            return self._on_flushed(state, event)
        if isinstance(event, E.SceneChanged):
            return self._on_scene_changed(state, event)
        if isinstance(event, E.UserSpeechOnset):
            return self._on_onset(state, event)
        if isinstance(event, E.UserTakeover):
            return self._on_takeover(state, event)
        if isinstance(event, E.UserBackchannel):
            return self._on_backchannel(state, event)
        if isinstance(event, E.UserSpeechOffset):
            return self._on_offset(state, event)
        if isinstance(event, E.LlmCommit):
            return self._on_commit(state, event)
        if isinstance(event, E.LlmDone):
            return self._on_llm_done(state, event)
        if isinstance(event, E.TtsDone):
            return state.model_copy(update={"tts_done": True}), []
        if isinstance(event, E.PlaybackStopped):
            return self._on_playback_stopped(state, event)
        if isinstance(event, E.PlaybackDrained):
            return self._on_playback_drained(state, event)
        if isinstance(event, E.Tick):
            return self._on_tick(state, event)
        # user.pause, user.not_owner, stt.step, llm.prefill_ready and ui.action carry no row in
        # §4: the owner veto and the HUD actions are the executor's (T-0007), the rest is telemetry.
        return state, []

    # --- turn lifecycle -------------------------------------------------------------------------

    def _open_turn(
        self,
        state: MachineState,
        *,
        onset_ms: int,
        keep_words: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> tuple[MachineState, str]:
        """Start a new turn: new id from ``turn_seq``, ``next_words`` seeds the transcript."""
        seq = state.turn_seq + 1
        turn_id = f"t{seq}"
        base = state.words if keep_words else ()
        words = _merge(base, state.next_words, self.policy.max_buffered_words)
        update: dict[str, Any] = {
            **_PER_TURN_RESET,
            "turn_seq": seq,
            "turn_id": turn_id,
            "onset_ms": onset_ms,
            "words": words,
            "next_words": (),
            "active_prefill": f"{turn_id}-prefill",
            **(extra or {}),
        }
        new = state.model_copy(update=update)
        return new, f"{turn_id}-prefill"

    def _end_turn(
        self, state: MachineState, *, extra: dict[str, Any] | None = None
    ) -> MachineState:
        """Back to IDLE with every per-turn field cleared (``next_words`` survives, criterion 4)."""
        update: dict[str, Any] = {
            **_PER_TURN_RESET,
            "state": TurnState.IDLE,
            "words": (),
            **(extra or {}),
        }
        return state.model_copy(update=update)

    def _abort_live(self, state: MachineState) -> list[Command]:
        """Abort what is still live, and only that (A16 as corrected).

        Every row that ends a turn owes an abort for the vLLM requests this session started and
        nobody else will cancel: otherwise the engine keeps denoising an orphan against
        ``--max-num-seqs 3`` (``.claude/rules/conductor.md``). The abort is keyed on what is live
        *now*, never on a remembered id — a row reached after ``_interrupt`` already paid it has
        ``active_generate is None`` and emits nothing, which is what criterion 21 requires.
        """
        cmds: list[Command] = []
        if state.active_generate is not None:
            cmds.append(C.LlmAbort(request_id=state.active_generate))
        if state.active_prefill is not None:
            cmds.append(C.PrefillAbort(request_id=state.active_prefill))
        return cmds

    def _commit_user_only(
        self, state: MachineState, *, force: bool = False
    ) -> tuple[MachineState, list[Command]]:
        """Store the user half of a turn that produced no speech (A11: nothing if it is empty).

        ``force`` keeps the ``commit_history`` command for a row whose criterion demands it even
        with an empty transcript (criterion 20(b) / A10); no history entry is ever invented.
        """
        user_text = _transcript(state.words)
        if not user_text:
            return state, [C.CommitHistory(user_only=True)] if force else []
        turn = HistoryTurn(user_text=user_text, assistant_spoken="", interrupted=False)
        history = _append_history(state, turn, self.policy.max_history_turns)
        return state.model_copy(update={"history": history}), [C.CommitHistory(user_only=True)]

    def _commit_played(self, state: MachineState) -> tuple[MachineState, list[Command]]:
        """Store what the user actually heard (never what was merely generated)."""
        user_text = _transcript(state.words)
        cmds: list[Command] = [C.CommitHistory(user_only=False)]
        if not user_text and not state.played_text:
            return state, cmds
        turn = HistoryTurn(
            user_text=user_text, assistant_spoken=state.played_text, interrupted=False
        )
        history = _append_history(state, turn, self.policy.max_history_turns)
        return state.model_copy(update={"history": history}), cmds

    # --- speech ---------------------------------------------------------------------------------

    def _speak(self, state: MachineState, *, final: bool) -> tuple[MachineState, list[Command]]:
        """Segment the unspoken committed tail and emit one ``tts.speak`` per sentence.

        The segmenter is *policed*, not trusted: a call that does not conserve its input raises
        ``ValueError`` and nothing is spoken (docs/02 §4 "never tts.speak with text the LLM has
        not committed"). The caller's state is untouched, because nothing here mutates.
        """
        pending = state.held_text
        if not pending:
            return state, []
        sentences, remainder = self.segment(pending, final=final)
        if "".join(source for source, _ in sentences) + remainder != pending:
            raise ValueError(
                "segmenter did not conserve the committed text "
                f"(final={final}): it must return sources that rebuild its input exactly"
            )
        cmds: list[Command] = []
        seq = state.utterance_seq
        utterances = list(state.utterances)
        spoken = state.spoken_text
        for source, text in sentences:
            seq += 1
            utterance_id = f"{state.turn_id}-u{seq}"
            cmds.append(C.TtsSpeak(utterance_id=utterance_id, text=text, source=source))
            utterances.append(utterance_id)
            spoken += source
        new = state.model_copy(
            update={
                "spoken_text": spoken,
                "held_text": remainder,
                "utterances": tuple(utterances),
                "utterance_seq": seq,
            }
        )
        return new, cmds

    # --- rows: perception, transcript -----------------------------------------------------------

    def _on_scene_changed(
        self, state: MachineState, ev: E.SceneChanged
    ) -> tuple[MachineState, list[Command]]:
        """LISTENING with a live prefill: record the screen and re-warm the prefill.

        T-0059 criteria 1-3: the screens of one turn accumulate in ``turn_phashes``, in arrival
        order and once each, so the prompt can carry them all before the transcript. A hash the
        turn already holds costs nothing — no snapshot, no re-issue, and the state is returned
        unchanged (criterion 2; a screen evicted from the live slot is no longer held and counts
        as new, ruling A4). While there is room the hash is *appended*, so every prefill already
        sent stays a byte-prefix of the next one (D1); once the turn is full the newest screen
        replaces the **last** entry only, which at ``max_frames_per_turn == 1`` is docs/01 D9's
        "the last snapshot wins" unchanged (criterion 5). The turn's first frame is the onset
        snapshot, whose hash no event carries (D3), so it fills a slot this tuple cannot show.
        """
        if state.state is not TurnState.LISTENING or state.active_prefill is None:
            return state, []  # criterion 2 (negative)
        if ev.phash in state.turn_phashes:
            return state, []  # T-0059 criterion 2: this screen is already in the prompt
        cap = self.policy.max_frames_per_turn
        slots = cap - 1 if cap > 1 else 1
        held = state.turn_phashes
        recorded = (*held, ev.phash) if len(held) < slots else (*held[:-1], ev.phash)
        new = state.model_copy(update={"turn_phashes": recorded})
        return new, [
            C.SnapshotFrame(at_ms=ev.t_ms),
            C.PrefillReissue(request_id=new.active_prefill),
        ]

    def _on_word(self, state: MachineState, ev: E.SttWord) -> tuple[MachineState, list[Command]]:
        """Criterion 4: attribution by audio time; a word after ``t_eot`` is the next turn's."""
        word = Word(text=ev.text, start_ms=ev.start_ms, end_ms=ev.end_ms)
        cap = self.policy.max_buffered_words
        frozen = state.t_eot_ms is not None
        this_turn = (state.state is TurnState.LISTENING and not frozen) or (
            state.t_eot_ms is not None and ev.start_ms <= state.t_eot_ms
        )
        if this_turn:
            new = state.model_copy(update={"words": _buffer(state.words, word, cap)})
        else:
            new = state.model_copy(update={"next_words": _buffer(state.next_words, word, cap)})
        if (
            state.state is TurnState.PAUSED
            and state.pause_onset_ms is not None
            and ev.start_ms >= state.pause_onset_ms
        ):
            # criterion 18: speech inside the pause, and no backchannel ended it first (A14)
            return self._interrupt(new, t_ms=ev.t_ms)
        return new, []

    def _on_end_of_turn(
        self, state: MachineState, ev: E.SttEndOfTurn
    ) -> tuple[MachineState, list[Command]]:
        if state.state is not TurnState.LISTENING or ev.p_end < self.policy.eot_p_threshold:
            return state, []  # criterion 3: below the threshold nothing happens
        new = state.model_copy(
            update={
                "state": TurnState.ENDPOINTING,
                "t_eot_ms": ev.t_eot_ms,
                "flush_t_ms": None,
            }
        )
        return new, [C.FreezeAfter(t_eot_ms=ev.t_eot_ms)]

    def _on_flushed(
        self, state: MachineState, ev: E.SttFlushed
    ) -> tuple[MachineState, list[Command]]:
        """The moshi ``Marker``: every word ≤ ``t_eot`` has been delivered (criterion 6)."""
        if (
            state.state is not TurnState.ENDPOINTING
            or ev.t_eot_ms != state.t_eot_ms
            or state.flush_t_ms is not None
        ):
            return state, []
        return state.model_copy(update={"flush_t_ms": ev.t_ms}), []

    # --- rows: the reflex lane ------------------------------------------------------------------

    def _on_onset(
        self, state: MachineState, ev: E.UserSpeechOnset
    ) -> tuple[MachineState, list[Command]]:
        if state.state is TurnState.IDLE:  # criterion 1
            new, prefill = self._open_turn(state, onset_ms=ev.onset_ms)
            new = new.model_copy(update={"state": TurnState.LISTENING})
            return new, [
                C.CaptionerAbort(),
                C.SnapshotFrame(at_ms=ev.onset_ms),
                C.PrefillOpen(request_id=prefill),
            ]
        if state.state is TurnState.ENDPOINTING:  # criterion 5
            new = state.model_copy(
                update={"state": TurnState.LISTENING, "t_eot_ms": None, "flush_t_ms": None}
            )
            return new, [C.Unfreeze()]
        if state.state is TurnState.THINKING:  # criterion 15: not yet a takeover
            new, arm = _arm(state, "takeover", ev.t_ms + self.policy.takeover_ms)
            return new.model_copy(update={"clip_playing": False}), [C.StopClip(), arm]
        if state.state is TurnState.SPEAKING:  # criterion 16: pause first, always
            new, arm = _arm(state, "takeover", ev.onset_ms + self.policy.takeover_ms)
            new = new.model_copy(update={"state": TurnState.PAUSED, "pause_onset_ms": ev.onset_ms})
            return new, [C.PlaybackPause(), arm]
        return state, []

    def _on_takeover(
        self, state: MachineState, ev: E.UserTakeover
    ) -> tuple[MachineState, list[Command]]:
        if state.state is TurnState.THINKING:  # criterion 15: §4's concatenate_transcript cell
            cmds: list[Command] = []
            if state.active_generate is not None:
                cmds.append(C.LlmAbort(request_id=state.active_generate))
            new, prefill = self._open_turn(state, onset_ms=ev.onset_ms, keep_words=True)
            new = new.model_copy(update={"state": TurnState.LISTENING})
            cmds += [C.SnapshotFrame(at_ms=ev.onset_ms), C.PrefillOpen(request_id=prefill)]
            return new, cmds
        if state.state in (TurnState.SPEAKING, TurnState.PAUSED):  # criterion 18
            pausing = {"pause_onset_ms": state.pause_onset_ms or ev.onset_ms}
            return self._interrupt(state.model_copy(update=pausing), t_ms=ev.t_ms)
        return state, []

    def _on_backchannel(
        self, state: MachineState, ev: E.UserBackchannel
    ) -> tuple[MachineState, list[Command]]:
        if state.state is not TurnState.PAUSED:
            return state, []
        return self._resume(state)  # criterion 17(a), literal per A14

    def _on_offset(
        self, state: MachineState, ev: E.UserSpeechOffset
    ) -> tuple[MachineState, list[Command]]:
        if state.state is not TurnState.PAUSED:
            return state, []
        if ev.speech_ms >= self.policy.resume_max_ms:  # criterion 18(b)
            return self._interrupt(state, t_ms=ev.t_ms)
        # criterion 17(b) with A12: the short offset owns the outcome — the takeover deadline is
        # cancelled so the two cannot fire on the same tick.
        new, cmds = _cancel(state, "takeover")
        new, arm = _arm(new, "resume", ev.t_ms + self.policy.resume_max_ms)
        return new, [*cmds, arm]

    def _interrupt(self, state: MachineState, *, t_ms: int) -> tuple[MachineState, list[Command]]:
        """PAUSED/SPEAKING → INTERRUPTED, once per interruption (criteria 18 and 21)."""
        cmds: list[Command] = []
        if state.active_generate is not None:
            cmds.append(C.LlmAbort(request_id=state.active_generate))
        cmds += [C.TtsCancel(), C.PlaybackFlush()]
        new, cancels = _cancel(state, "takeover", "resume")
        cmds += cancels
        new, arm = _arm(new, "interrupt_timeout", t_ms + self.policy.interrupt_timeout_ms)
        cmds.append(arm)
        new = new.model_copy(
            update={"state": TurnState.INTERRUPTED, "active_generate": None, "clip_playing": False}
        )
        return new, cmds

    def _resume(self, state: MachineState) -> tuple[MachineState, list[Command]]:
        """PAUSED → SPEAKING (criterion 17): the text held back while paused is spoken now."""
        cmds: list[Command] = [C.CancelTimer(name="takeover")]
        new = state.model_copy(
            update={"timers": tuple(t for t in state.timers if t.name != "takeover")}
        )
        new, cancels = _cancel(new, "resume")
        cmds += cancels
        cmds.append(C.PlaybackResume())
        new = new.model_copy(update={"state": TurnState.SPEAKING, "pause_onset_ms": None})
        new, speaks = self._speak(new, final=new.llm_done)
        cmds += speaks
        return self._maybe_complete(new, cmds)

    def _relisten(
        self, state: MachineState, *, played_ms: int | None, played_text: str, at_ms: int
    ) -> tuple[MachineState, list[Command]]:
        """INTERRUPTED → LISTENING (criterion 19): history keeps the played prefix and nothing more."""
        marker = f"{played_text} [interrupted]" if played_text else "[interrupted]"  # A6
        turn = HistoryTurn(
            user_text=_transcript(state.words), assistant_spoken=marker, interrupted=True
        )
        history = _append_history(state, turn, self.policy.max_history_turns)
        cmds: list[Command] = [
            C.TruncateHistoryToPlayed(played_ms=played_ms, played_text=played_text)
        ]
        onset_ms = state.pause_onset_ms if state.pause_onset_ms is not None else at_ms
        new, prefill = self._open_turn(
            state, onset_ms=onset_ms, extra={"history": history, "state": TurnState.LISTENING}
        )
        cmds += [C.SnapshotFrame(at_ms=at_ms), C.PrefillOpen(request_id=prefill)]
        return new, cmds

    # --- rows: the LLM --------------------------------------------------------------------------

    def _measure_first_canvas(self, state: MachineState, ev: E.LlmCommit) -> MachineState:
        """Criterion 8: the cross-turn EMA of first-canvas latency, on the first commit only."""
        if state.first_canvas_measured or state.llm_start_ms is None:
            return state
        measured = ev.t_ms - state.llm_start_ms
        alpha = self.policy.first_canvas_ema_alpha
        previous = state.first_canvas_ms
        ema = measured if previous is None else round(alpha * measured + (1 - alpha) * previous)
        return state.model_copy(update={"first_canvas_ms": ema, "first_canvas_measured": True})

    def _on_commit(
        self, state: MachineState, ev: E.LlmCommit
    ) -> tuple[MachineState, list[Command]]:
        if ev.request_id != state.active_generate:
            return state, []  # criterion 11: a stale request may not speak
        new = self._measure_first_canvas(state, ev)
        if ev.segment != "answer" or ev.thought_open:
            return new, []  # criterion 11: thoughts and unclosed canvases are never voiced
        new = new.model_copy(
            update={
                "committed_answer": new.committed_answer + ev.text,
                "held_text": new.held_text + ev.text,
            }
        )
        if new.state is TurnState.PAUSED:
            return new, []  # criterion 16: no synthesis a takeover could cancel
        if new.state is not TurnState.THINKING and new.state is not TurnState.SPEAKING:
            return new, []
        cmds: list[Command] = []
        if new.state is TurnState.THINKING:  # criterion 9 (A4: the row is keyed on the commit)
            new = new.model_copy(update={"state": TurnState.SPEAKING})
            if new.clip_playing:
                cmds.append(C.StopClip())
                new = new.model_copy(update={"clip_playing": False})
        new, speaks = self._speak(new, final=new.llm_done)
        return new, [*cmds, *speaks]

    def _on_llm_done(
        self, state: MachineState, ev: E.LlmDone
    ) -> tuple[MachineState, list[Command]]:
        if ev.request_id != state.active_generate:
            return state, []
        new, cmds = _cancel(state, "llm_timeout")  # criterion 12
        new = new.model_copy(update={"llm_done": True, "active_generate": None})
        if new.state is TurnState.THINKING or new.state is TurnState.SPEAKING:
            new, speaks = self._speak(new, final=True)
            cmds = [*speaks, *cmds]  # A3: only timers and hud.state may precede the flush
        if new.state is TurnState.THINKING:
            # criterion 12: nothing was committed, so the turn ends quietly
            new, history_cmds = self._commit_user_only(new)
            return self._end_turn(new), [*cmds, *history_cmds]
        return self._maybe_complete(new, cmds)

    def _maybe_complete(
        self, state: MachineState, cmds: list[Command]
    ) -> tuple[MachineState, list[Command]]:
        """Criterion 13: whichever of llm.done / playback.drained arrives last ends the turn."""
        if state.state is not TurnState.SPEAKING or not (state.llm_done and state.playback_drained):
            return state, cmds
        new, history_cmds = self._commit_played(state)
        return self._end_turn(new), [*cmds, *history_cmds, C.ScheduleSummary()]

    # --- rows: playback -------------------------------------------------------------------------

    def _on_playback_stopped(
        self, state: MachineState, ev: E.PlaybackStopped
    ) -> tuple[MachineState, list[Command]]:
        new = state.model_copy(update={"played_text": ev.played_text})
        if state.state is TurnState.INTERRUPTED:
            return self._relisten(
                new, played_ms=ev.played_ms, played_text=ev.played_text, at_ms=ev.t_ms
            )
        return new, []

    def _on_playback_drained(
        self, state: MachineState, ev: E.PlaybackDrained
    ) -> tuple[MachineState, list[Command]]:
        new = state.model_copy(update={"played_text": ev.played_text, "playback_drained": True})
        return self._maybe_complete(new, [])

    # --- rows: time -----------------------------------------------------------------------------

    def _on_tick(self, state: MachineState, ev: E.Tick) -> tuple[MachineState, list[Command]]:
        if state.state is TurnState.ENDPOINTING:
            settled = state.flush_t_ms is not None and (
                ev.t_ms >= state.flush_t_ms + self.policy.eot_settle_ms
            )
            if settled:
                return self._start_thinking(state, ev)
            return state, []
        if state.state is TurnState.THINKING:
            deadline = _timer_at(state, "llm_timeout")
            if deadline is not None and ev.t_ms >= deadline:  # criterion 14
                return self._fail_turn(state, message="llm timeout", user_only=True)
            return state, []
        if state.state is TurnState.PAUSED:
            resume_at = _timer_at(state, "resume")
            if resume_at is not None and ev.t_ms >= resume_at:  # criterion 17(b)
                return self._resume(state)
            takeover_at = _timer_at(state, "takeover")
            if takeover_at is not None and ev.t_ms >= takeover_at:  # criterion 18(c)
                return self._interrupt(state, t_ms=ev.t_ms)
            return state, []
        if state.state is TurnState.INTERRUPTED:
            deadline = _timer_at(state, "interrupt_timeout")
            if deadline is not None and ev.t_ms >= deadline:  # criterion 19, timeout path
                return self._relisten(state, played_ms=None, played_text="", at_ms=ev.t_ms)
            return state, []
        return state, []

    def _start_thinking(
        self, state: MachineState, ev: E.Tick
    ) -> tuple[MachineState, list[Command]]:
        """ENDPOINTING → THINKING (criteria 6 and 7): both signals seen, the window elapsed."""
        cmds: list[Command] = []
        if (
            state.onset_ms is not None
            and state.t_eot_ms is not None
            and state.t_eot_ms - state.onset_ms >= self.policy.owner_check_min_speech_ms
        ):
            cmds.append(C.CheckOwner(from_ms=state.onset_ms, to_ms=state.t_eot_ms))
        if state.active_prefill is not None:
            cmds.append(C.PrefillAbort(request_id=state.active_prefill))
        request_id = f"{state.turn_id}-gen"
        cmds.append(C.LlmStart(request_id=request_id, transcript=_transcript(state.words)))
        new = state.model_copy(
            update={
                "state": TurnState.THINKING,
                "active_prefill": None,
                "active_generate": request_id,
                "llm_start_ms": ev.t_ms,
            }
        )
        if (
            self.policy.thinking
            or new.first_canvas_ms is None
            or new.first_canvas_ms > self.policy.ack_first_canvas_ema_ms
        ):  # criterion 7 (A7: at most one clip per turn, nothing from T-0013)
            cmds.append(C.PlayClip(kind="ack"))
            new = new.model_copy(
                update={"fillers_this_turn": new.fillers_this_turn + 1, "clip_playing": True}
            )
        new, arm = _arm(new, "llm_timeout", ev.t_ms + self.policy.llm_timeout_ms)
        cmds.append(arm)
        return new, cmds

    # --- rows: errors and disconnect ------------------------------------------------------------

    def _fail_turn(
        self, state: MachineState, *, message: str, user_only: bool
    ) -> tuple[MachineState, list[Command]]:
        """End the turn on an error: hud.error, then whatever history the turn earned."""
        cmds: list[Command] = [C.HudError(message=message)]
        cmds += self._abort_live(state)
        if user_only:
            new, history_cmds = self._commit_user_only(state)
        else:
            new, history_cmds = self._commit_played(state)
        return self._end_turn(new), [*cmds, *history_cmds]

    def _on_llm_error(
        self, state: MachineState, ev: E.LlmError
    ) -> tuple[MachineState, list[Command]]:
        """Criterion 14 widened by A5/A9: the turn ends from every state, however it was spending."""
        if ev.request_id is not None and ev.request_id != state.active_generate:
            return state, []
        speaking = state.state in (TurnState.SPEAKING, TurnState.PAUSED, TurnState.INTERRUPTED)
        return self._fail_turn(state, message=ev.message, user_only=not speaking)

    def _on_tts_error(
        self, state: MachineState, ev: E.TtsError
    ) -> tuple[MachineState, list[Command]]:
        """Criterion 20(b): degrade always; end the turn only where speech was being produced."""
        cmds: list[Command] = [C.Degrade(mode="text_only")]
        new = state.model_copy(update={"degraded": True})
        if state.state not in (TurnState.SPEAKING, TurnState.PAUSED, TurnState.INTERRUPTED):
            return new, cmds
        cmds += self._abort_live(new)
        # A10: nothing was reported played, so there is nothing to commit but the user's half —
        # and never the "[interrupted]" marker, which would be a false claim about the turn.
        if new.played_text:
            new, history_cmds = self._commit_played(new)
        else:
            new, history_cmds = self._commit_user_only(new, force=True)
        return self._end_turn(new), [*cmds, *history_cmds]

    def _on_stt_error(
        self, state: MachineState, ev: E.SttError
    ) -> tuple[MachineState, list[Command]]:
        """Criterion 20(c): a dead transcriber ends a turn that was still listening, nothing else."""
        cmds: list[Command] = [C.HudError(message=ev.message)]
        if state.state not in (TurnState.LISTENING, TurnState.ENDPOINTING):
            return state, cmds
        cmds += self._abort_live(state)
        new, history_cmds = self._commit_user_only(state)  # A11: empty transcript stores nothing
        return self._end_turn(new), [*cmds, *history_cmds]

    def _on_client_disconnected(
        self, state: MachineState, ev: E.ClientDisconnected
    ) -> tuple[MachineState, list[Command]]:
        """Criterion 20(a): from any state, degrade to text-only and drop what was live."""
        cmds: list[Command] = [C.Degrade(mode="text_only")]
        cmds += self._abort_live(state)
        new = state.model_copy(update={"degraded": True})
        if new.played_text:
            new, history_cmds = self._commit_played(new)
        else:
            new, history_cmds = self._commit_user_only(new)
        return self._end_turn(new), [*cmds, *history_cmds]


def is_terminal_idle(state: MachineState) -> bool:
    return (
        state.state is TurnState.IDLE
        and state.active_generate is None
        and state.active_prefill is None
    )
