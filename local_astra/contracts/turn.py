"""Turn-machine state and the tutoring state carried across turns (docs/02 §4, docs/11 §2.3)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from local_astra.contracts.base import Contract


class TurnState(StrEnum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    ENDPOINTING = "ENDPOINTING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    PAUSED = "PAUSED"
    INTERRUPTED = "INTERRUPTED"


class Word(Contract):
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)


class HistoryTurn(Contract):
    """What is stored: user text and the assistant text that was actually *played*.

    ``interrupted`` marks a turn whose spoken text was cut; the stored text is the played prefix
    plus ``[interrupted]`` (docs/02 §4). Thinking is never stored.
    """

    user_text: str
    assistant_spoken: str
    interrupted: bool = False
    #: frame attached to this turn (perceptual hash; bytes live in the perception ring / memory
    #: thumbnails). Sent as an image while the turn is ≤ ``image_age_turns`` old, then replaced by
    #: ``frame_caption`` at ``commit_history`` (docs/02 §2.2).
    frame_phash: str | None = None
    frame_caption: str | None = None


#: How the session judged a tool call the model drafted (T-0089; computed by the session, T-0091).
ToolStatus = Literal["valid", "repairable", "refused"]
#: Which request produced the draft: pass 1 (``draft``) or the one repair request (``repair``).
ToolSource = Literal["draft", "repair"]


class ToolDraft(Contract):
    """A tool call carried by ``llm.done`` (T-0089 criterion 1). The machine trusts ``status``.

    ``valid`` has a name and arguments; ``repairable`` has a name and comes from pass 1 (a repair
    is never repaired); ``refused`` may have neither, and ``reason`` says why.
    """

    call_id: str
    status: ToolStatus
    name: str | None = None
    arguments: dict[str, Any] | None = None
    reason: str | None = None
    source: ToolSource = "draft"

    @model_validator(mode="after")
    def _shape(self) -> ToolDraft:
        if self.status == "valid" and (self.name is None or self.arguments is None):
            raise ValueError("a valid tool draft has a name and arguments")
        if self.status == "repairable" and self.name is None:
            raise ValueError("a repairable tool draft has a name")
        if self.status == "repairable" and self.source != "draft":
            raise ValueError("only a pass-1 draft is repairable: a repair is never repaired")
        return self


class Timer(Contract):
    name: str
    at_ms: int = Field(ge=0)


class MachineState(Contract):
    """Complete state of the pure machine. Everything a test needs to assert §4 invariants."""

    state: TurnState = TurnState.IDLE
    turn_id: str = "t0"
    turn_seq: int = 0
    #: audio time of the onset that opened the current turn
    onset_ms: int | None = None
    #: audio time of the end-of-turn currently being settled
    t_eot_ms: int | None = None
    words: tuple[Word, ...] = ()
    #: words whose audio time falls after ``t_eot_ms``: they belong to the *next* turn and seed its
    #: ``words`` when it opens (docs/02 §4, attribution by audio timestamp). Both word buffers are
    #: capped at ``TurnPolicy.max_buffered_words``, oldest dropped first.
    next_words: tuple[Word, ...] = ()
    #: perceptual hashes of the screens this turn has seen *change* while the user was speaking,
    #: in arrival order and once each (T-0059 D1). The turn's first frame is the onset snapshot,
    #: whose hash no event carries, so this holds at most ``TurnPolicy.max_frames_per_turn - 1``
    #: entries (one at cap 1) and the last one is the live slot. Hashes, not a counter: the state
    #: stays bounded over a long LISTENING (docs/02 §4).
    turn_phashes: tuple[str, ...] = ()
    #: at most one of each (invariant)
    active_generate: str | None = None
    active_prefill: str | None = None
    #: committed answer text for the current turn, in commit order
    committed_answer: str = ""
    #: text handed to TTS so far (must be a prefix-respecting slice of committed_answer)
    spoken_text: str = ""
    #: committed answer text the segmenter is still holding back: ``spoken_text + held_text ==
    #: committed_answer`` always. Flushed on ``llm.done`` and on resume from PAUSED.
    held_text: str = ""
    utterances: tuple[str, ...] = ()
    #: per-turn utterance counter; utterance ids derive from it and ``turn_id``, never from a clock
    utterance_seq: int = 0
    timers: tuple[Timer, ...] = ()
    history: tuple[HistoryTurn, ...] = ()
    fillers_this_turn: int = 0
    #: cross-turn EMA of first-canvas latency (docs/01 §8), updated on the *first* commit of a turn
    #: with ``alpha = TurnPolicy.first_canvas_ema_alpha`` and read by the ack-clip decision. It
    #: survives the turn; ``first_canvas_measured`` is the per-turn "already updated" flag.
    first_canvas_ms: int | None = None
    first_canvas_measured: bool = False
    #: delivery time of this turn's ``llm.start`` (the EMA measures the first canvas from it)
    llm_start_ms: int | None = None
    #: delivery time of the ``stt.flushed`` that confirmed ``t_eot_ms``; the settle window of the
    #: ENDPOINTING → THINKING row runs from here, never from state-entry time
    flush_t_ms: int | None = None
    #: audio onset of the barge-in that paused playback (T-0003 A13: the audio clock, not t_ms)
    pause_onset_ms: int | None = None
    #: what a playback event last reported as played; the only assistant text allowed into history
    played_text: str = ""
    llm_done: bool = False
    tts_done: bool = False
    playback_drained: bool = False
    #: an acknowledgement clip of this turn is playing and has not been stopped yet
    clip_playing: bool = False
    degraded: bool = False
    #: T-0089: the pending tool call of this turn (pass-1 draft, or the repaired one), until it
    #: ends in exactly one ``tool.execute`` or ``tool.cancel``
    tool: ToolDraft | None = None
    #: T-0089: repair requests started this turn (at most one, R2/R4)
    tool_repairs: int = Field(default=0, ge=0)
    #: T-0089 T15: a ``tts.speak`` was emitted after the last ``playback.drained`` of this turn, so
    #: that drain does not cover what is queued (set by every speak, cleared by every drain)
    spoken_after_drain: bool = False
    #: T-0112 (T-0111a §2), the tool-result loop; all per turn and inert with ``tool_loop`` off.
    #: The step whose generate is live or next (1 is the turn's own ``llm.start``).
    tool_step: int = Field(default=1, ge=1)
    #: ``tool.execute`` commands emitted this turn (at most ``TurnPolicy.max_tool_steps``)
    tool_steps_executed: int = Field(default=0, ge=0)
    #: the executed call whose ``tool.result`` the turn is waiting for, in THINKING (R11/R12)
    awaiting_result: str | None = None
    #: answer text committed at this step (>= 2): voiced only if the step ends without an
    #: executed call (R14/R15), otherwise withheld and never voiced or stored
    step_text: str = ""
    #: what earlier playback runs of this turn reported as played, joined with single spaces;
    #: history stores it joined with ``played_text`` (T-0111a §2 History)
    played_prior: str = ""
    #: T-0182 B4: the turn's one ``still_working`` clip has played (per turn). Excluded from dumps (B-R5), so
    #: ``MachineState`` dumps, goldens and BASE differentials are unchanged.
    still_working_played: bool = Field(default=False, exclude=True)
    #: T-0193 A0: this turn was ended by the HUD's stop (``user.stop``), so its INTERRUPTED exits go to IDLE instead
    #: of relistening (per turn). Excluded from dumps, as ``still_working_played``.
    stopped: bool = Field(default=False, exclude=True)


class TutorState(Contract):
    """Compact tutoring state re-injected each turn *after* the frame (docs/11 §2.3).

    Kept small on purpose: it rides in the query-dependent suffix of the prompt, so every token
    here is paid on every turn.
    """

    problem: str = ""
    solution: str = Field(
        default="", description="first solve, cached so drills stay single-canvas"
    )
    current_step: int = Field(default=0, ge=0)
    total_steps: int = Field(default=0, ge=0)
    misconceptions: tuple[str, ...] = ()
    hints_given: tuple[str, ...] = ()
    ladder_level: int = Field(default=0, ge=0, le=4, description="0 nudge … 4 worked step")

    def as_prompt_block(self) -> str:
        if not self.problem:
            return ""
        lines = [
            "[tutor state]",
            f"problem: {self.problem}",
            f"step: {self.current_step}/{self.total_steps}",
            f"ladder: {self.ladder_level}",
        ]
        if self.solution:
            lines.append(f"solution (do not reveal unless ladder=4): {self.solution}")
        if self.misconceptions:
            lines.append("misconceptions: " + "; ".join(self.misconceptions))
        if self.hints_given:
            lines.append("hints given: " + " | ".join(self.hints_given))
        return "\n".join(lines)
