"""Commands emitted by the pure turn machine and executed by ``conductor/session.py``.

The machine never calls a client; it returns a tuple of commands (docs/02 §4). Executors must
treat ``LlmAbort``, ``TtsCancel`` and ``PlaybackFlush`` as idempotent — the machine may emit
them more than once for different causes but never twice for the same cause.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter

from local_astra.contracts.base import Contract


class _Command(Contract):
    pass


# perception / prefill
class SnapshotFrame(_Command):
    """Capture the frame at ``at_ms`` (speech onset), not the latest one."""

    type: Literal["snapshot_frame"] = "snapshot_frame"
    at_ms: int = Field(ge=0)


class PrefillOpen(_Command):
    type: Literal["prefill_open"] = "prefill_open"
    request_id: str


class PrefillReissue(_Command):
    """Scene changed while listening: re-warm the prefix (rate limited by the executor)."""

    type: Literal["prefill_reissue"] = "prefill_reissue"
    request_id: str


class PrefillAbort(_Command):
    type: Literal["prefill_abort"] = "prefill_abort"
    request_id: str


# endpointing
class FreezeAfter(_Command):
    """Stop appending words with audio time > ``t_eot_ms`` to this turn's transcript."""

    type: Literal["freeze_after"] = "freeze_after"
    t_eot_ms: int = Field(ge=0)


class Unfreeze(_Command):
    type: Literal["unfreeze"] = "unfreeze"


class CheckOwner(_Command):
    """Run the owner-voice veto on the turn's audio (result arrives as ``user.not_owner``)."""

    type: Literal["check_owner"] = "check_owner"
    from_ms: int = Field(ge=0)
    to_ms: int = Field(ge=0)


# LLM
class LlmStart(_Command):
    type: Literal["llm.start"] = "llm.start"
    request_id: str
    transcript: str


class LlmAbort(_Command):
    type: Literal["llm.abort"] = "llm.abort"
    request_id: str


class LlmContinue(_Command):
    """Start step ``step`` (>= 2) of the turn's tool-result loop (T-0111a §2 R12).

    The executed call ``call_id`` has a result; the session builds the step prompt (the step-1
    prompt as sent, continued with the rendered calls and results) and starts ``request_id``.
    """

    type: Literal["llm.continue"] = "llm.continue"
    request_id: str
    step: int
    call_id: str


class CaptionerAbort(_Command):
    """Idle captioner (memory-lite) yields to a live turn."""

    type: Literal["captioner.abort"] = "captioner.abort"


# fillers / clips
#: T-0178 B4a: ``out_of_steps`` (a call past the step budget, or a report-step announcement) and ``not_finished`` (a
#: malformed step >= 2 draft) are the honest-ending clips: system audio, never history
ClipKind = Literal[
    "ack", "hmm", "still_working", "out_of_steps", "not_finished", "lost_thread", "at_checkout"
]


class PlayClip(_Command):
    """Acknowledgement / filler clip. ``source`` selects the contextual variant
    ("let me look at the screen…" vs "…at the camera", docs/01 D7/D11)."""

    type: Literal["play_clip"] = "play_clip"
    kind: ClipKind
    source: str | None = None


class StopClip(_Command):
    type: Literal["stop_clip"] = "stop_clip"


# timers (fire back as Tick; the machine compares deadlines itself)
TimerName = Literal[
    "eot_settle",
    "takeover",
    "resume",
    "llm_timeout",
    "interrupt_timeout",
    "interim_update",
    "tool_repair",
    "tool_result",
]


class ArmTimer(_Command):
    type: Literal["arm_timer"] = "arm_timer"
    name: TimerName
    at_ms: int = Field(ge=0)


class CancelTimer(_Command):
    type: Literal["cancel_timer"] = "cancel_timer"
    name: TimerName


# TTS / playback
class TtsSpeak(_Command):
    """Speak one sentence of *committed answer text*.

    ``source`` is the exact substring of the committed answer the sentence came from (the §4
    invariant is checked on it); ``text`` is the TTS-normalised rendering the speech gate produced
    (paths, identifiers and code fences rewritten — docs/03 §3), which may differ from ``source``.
    """

    type: Literal["tts.speak"] = "tts.speak"
    utterance_id: str
    text: str
    source: str
    voice: str = "default"


class TtsCancel(_Command):
    type: Literal["tts.cancel"] = "tts.cancel"


class PlaybackPause(_Command):
    type: Literal["playback.pause"] = "playback.pause"


class PlaybackResume(_Command):
    type: Literal["playback.resume"] = "playback.resume"


class PlaybackFlush(_Command):
    type: Literal["playback.flush"] = "playback.flush"


# history
class TruncateHistoryToPlayed(_Command):
    """HUD/executor side of a barge-in: strike through what was not played.

    ``played_ms`` is None when INTERRUPTED timed out without ``playback.stopped`` (the executor
    estimates from the playback clock).
    """

    type: Literal["truncate_history_to_played"] = "truncate_history_to_played"
    played_ms: int | None = Field(default=None, ge=0)
    played_text: str = ""


class CommitHistory(_Command):
    """Store (user_text, spoken_assistant_text); age images to captions; never store thinking."""

    type: Literal["commit_history"] = "commit_history"
    user_only: bool = False


class ScheduleSummary(_Command):
    type: Literal["schedule_summary"] = "schedule_summary"


# tools (T-0089; inert in the executor until T-0092)
class ToolRepairStart(_Command):
    """Start the one repair request for the pass-1 draft ``call_id`` (R2)."""

    type: Literal["tool.repair_start"] = "tool.repair_start"
    request_id: str
    draft_request_id: str
    call_id: str


class ToolExecute(_Command):
    """Run the valid call of a turn that completed uninterrupted and drained (R1, R3)."""

    type: Literal["tool.execute"] = "tool.execute"
    call_id: str
    turn_id: str
    name: str
    arguments: dict[str, Any]
    source: Literal["draft", "repair"]


ToolCancelReason = Literal[
    "interrupted",
    "takeover",
    "error",
    "disconnected",
    "refused",
    "repair_failed",
    "repair_timeout",
    "step_budget",
    "reset",
    "stopped",
]


class ToolCancel(_Command):
    """The pending call ``call_id`` will never run (R4-R8)."""

    type: Literal["tool.cancel"] = "tool.cancel"
    call_id: str
    reason: ToolCancelReason


# HUD
class HudState(_Command):
    type: Literal["hud.state"] = "hud.state"
    state: str
    turn_id: str


class HudError(_Command):
    type: Literal["hud.error"] = "hud.error"
    message: str


class Degrade(_Command):
    type: Literal["degrade"] = "degrade"
    mode: Literal["text_only", "no_perception"]


# session (T-0162 A-2)
class ResetSession(_Command):
    """Clear every piece of cross-turn session state and wipe long-term memory (A-3).

    ``ended_turn_id`` is the turn the reset cut, or ``None`` when no turn was open.
    """

    type: Literal["reset_session"] = "reset_session"
    ended_turn_id: str | None = None


Command = Annotated[
    SnapshotFrame
    | PrefillOpen
    | PrefillReissue
    | PrefillAbort
    | FreezeAfter
    | Unfreeze
    | CheckOwner
    | LlmStart
    | LlmAbort
    | LlmContinue
    | CaptionerAbort
    | PlayClip
    | StopClip
    | ArmTimer
    | CancelTimer
    | TtsSpeak
    | TtsCancel
    | PlaybackPause
    | PlaybackResume
    | PlaybackFlush
    | TruncateHistoryToPlayed
    | CommitHistory
    | ScheduleSummary
    | HudState
    | HudError
    | Degrade
    | ToolRepairStart
    | ToolExecute
    | ToolCancel
    | ResetSession,
    Field(discriminator="type"),
]

CommandAdapter: TypeAdapter[Command] = TypeAdapter(Command)

#: Commands whose executors must be idempotent (docs/02 §4 invariants).
IDEMPOTENT_COMMANDS: frozenset[str] = frozenset({"llm.abort", "tts.cancel", "playback.flush"})
