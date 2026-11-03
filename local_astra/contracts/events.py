"""Events consumed by the pure turn machine (docs/02 §4).

Every event carries ``t_ms``: the time on the session's audio clock at which the event is
delivered to the machine. Events that refer to *audio* (words, end-of-turn, onsets) also carry
the audio timestamp they describe, because STT words arrive ~0.5 s after their audio and can
land in a later state; attribution to a turn is always by audio timestamp (§4).

Producers: reflex lane (``user.*``), STT client (``stt.*``), perception (``scene.changed``),
LLM client (``llm.*``), TTS/playback (``tts.*``, ``playback.*``), HUD (``ui.action``,
``client.disconnected``) and the session clock (``tick``).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter

from local_astra.contracts.base import Contract
from local_astra.contracts.llm import Segment
from local_astra.contracts.turn import ToolDraft


class _Event(Contract):
    t_ms: int = Field(ge=0, description="delivery time on the session audio clock (ms)")


# --- reflex lane -------------------------------------------------------------------------------


class UserSpeechOnset(_Event):
    type: Literal["user.speech_onset"] = "user.speech_onset"
    onset_ms: int = Field(ge=0, description="audio time of the onset")


class UserBackchannel(_Event):
    """Short overlapping utterance classified as a backchannel ("mm-hm", "ok")."""

    type: Literal["user.backchannel"] = "user.backchannel"
    duration_ms: int = Field(ge=0)


class UserTakeover(_Event):
    """Overlapping speech classified as a real interruption."""

    type: Literal["user.takeover"] = "user.takeover"
    onset_ms: int = Field(ge=0)


class UserPause(_Event):
    """Turn predictor's probability that the user has finished (semantic VAD / smart-turn)."""

    type: Literal["user.pause"] = "user.pause"
    p_end: float = Field(ge=0.0, le=1.0)


class UserSpeechOffset(_Event):
    """Reflex VAD: the user stopped speaking (audio time). Feeds PAUSED → SPEAKING/INTERRUPTED."""

    type: Literal["user.speech_offset"] = "user.speech_offset"
    offset_ms: int = Field(ge=0)
    speech_ms: int = Field(ge=0, description="duration of the speech segment that just ended")


class UserNotOwner(_Event):
    """Owner-voice check vetoed the turn (needs ≥ 1.5 s of speech; runs at end-of-turn)."""

    type: Literal["user.not_owner"] = "user.not_owner"
    score: float


class UserText(_Event):
    """A question typed into the HUD page (T-0039 criterion 4).

    It is an *event* rather than a session method call so a typed question is ordered against the
    frames, ticks and microphone onsets already on the one pump: the foreign-onset guard depends
    on that ordering. ``text`` arrives already stripped and capped by ``protocol.AstraText``. The
    pure machine has **no row** for it; the session is what opens the turn.
    """

    type: Literal["user.text"] = "user.text"
    text: str


# --- STT ---------------------------------------------------------------------------------------


class SttWord(_Event):
    type: Literal["stt.word"] = "stt.word"
    text: str
    start_ms: int = Field(ge=0, description="audio start of the word")
    end_ms: int = Field(ge=0)


class SttStep(_Event):
    """moshi-server ``Step`` frame: pause-prediction heads."""

    type: Literal["stt.step"] = "stt.step"
    pause_heads: tuple[float, ...] = ()


class SttEndOfTurn(_Event):
    type: Literal["stt.end_of_turn"] = "stt.end_of_turn"
    t_eot_ms: int = Field(ge=0, description="audio time the end-of-turn refers to")
    p_end: float = Field(default=1.0, ge=0.0, le=1.0)


class SttFlushed(_Event):
    """moshi-server ``Marker`` echo: every word with audio time ≤ ``t_eot_ms`` has been delivered.

    This is the observable form of "all words ≤ t_eot received" in the ENDPOINTING → THINKING row.
    """

    type: Literal["stt.flushed"] = "stt.flushed"
    t_eot_ms: int = Field(ge=0)


class SttError(_Event):
    type: Literal["stt.error"] = "stt.error"
    message: str


# --- perception --------------------------------------------------------------------------------


class SceneChanged(_Event):
    type: Literal["scene.changed"] = "scene.changed"
    source: str
    phash: str


# --- LLM ---------------------------------------------------------------------------------------


class LlmCommit(_Event):
    type: Literal["llm.commit"] = "llm.commit"
    request_id: str
    canvas_idx: int = Field(ge=0)
    segment: Segment
    text: str
    thought_open: bool = False


class LlmDone(_Event):
    type: Literal["llm.done"] = "llm.done"
    request_id: str
    #: T-0089: the tool call the session parsed from this request's final canvas, if any
    tool: ToolDraft | None = None


class LlmError(_Event):
    type: Literal["llm.error"] = "llm.error"
    request_id: str | None = None
    message: str


#: How the session's executor ended an executed call (T-0111a §2). The machine reads only this.
ToolResultStatus = Literal["dry_run", "executed", "refused", "failed", "timeout"]


class ToolResult(_Event):
    """The executed call ``call_id`` has a result (T-0112, the tool-result loop, R12/R13).

    It carries **no text**: the result's content goes from the executor to the step prompt and
    the session log, never through the pure machine (T-0111a §2, criterion 11).
    """

    type: Literal["tool.result"] = "tool.result"
    call_id: str
    status: ToolResultStatus
    #: T-0192 B1: ``"checkout"`` when the browser reached a checkout page; the turn ends with the at_checkout clip
    #: (B2). Still no text: a closed literal. Left out of dumps when ``None`` (B-R4: byte-identical to BASE).
    hand_back: Literal["checkout"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class PrefillReady(_Event):
    type: Literal["llm.prefill_ready"] = "llm.prefill_ready"
    request_id: str
    prefix_tokens: int = Field(ge=0)


# --- TTS / playback ----------------------------------------------------------------------------


class TtsDone(_Event):
    """All audio of ``utterance_id`` has been *produced* (not necessarily played)."""

    type: Literal["tts.done"] = "tts.done"
    utterance_id: str


class TtsError(_Event):
    type: Literal["tts.error"] = "tts.error"
    utterance_id: str | None = None
    message: str


class PlaybackStopped(_Event):
    """Playback flushed; ``played_ms`` is the audio actually rendered to the device.

    ``played_text`` is the prefix of the spoken text the executor derives from word timings (a
    word counts as played if it started ≥ 100 ms before the cut, docs/02 §4); the pure machine
    stores exactly this in history, never what was generated.
    """

    type: Literal["playback.stopped"] = "playback.stopped"
    played_ms: int = Field(ge=0)
    played_text: str = ""


class PlaybackDrained(_Event):
    """Everything queued has been played (natural end of speaking)."""

    type: Literal["playback.drained"] = "playback.drained"
    played_ms: int = Field(ge=0)
    played_text: str = ""


# --- HUD / client ------------------------------------------------------------------------------


class UiAction(_Event):
    type: Literal["ui.action"] = "ui.action"
    surface_id: str
    name: str
    context: dict[str, Any] = Field(default_factory=dict)


class ClientDisconnected(_Event):
    type: Literal["client.disconnected"] = "client.disconnected"


class SessionReset(_Event):
    """The HUD's Restart (T-0162 A-2): end whatever turn is live and start a new session.

    The machine's row aborts what is live, cancels a pending call, stops speech, commits **no**
    history and returns to IDLE with ``history=()``; the session executes ``reset_session``.
    """

    type: Literal["session.reset"] = "session.reset"


class MicMuted(_Event):
    """The HUD's mic mute engaged (T-0181 A0, H-T181-2: "just drop it").

    ``t_ms`` is the session clock (P10) at the moment ``apply_mic_mute`` turns mute on; the session
    submits one per false-to-true transition. There is no unmute event: unmuting changes nothing
    the machine can see. In LISTENING, and in ENDPOINTING before the flush, the open spoken
    question is abandoned; everywhere else the event has no row.
    """

    type: Literal["mic.muted"] = "mic.muted"


class UserStop(_Event):
    """The HUD's "Interrupt Gearvis" button (T-0193 A0).

    ``t_ms`` is the session clock; ``turn_id`` is the turn the page showed when it was clicked. A stop for
    the current turn in THINKING ends it in IDLE; in SPEAKING/PAUSED it interrupts and the INTERRUPTED exit
    goes to IDLE (no relisten). Everywhere else, and for any other turn id, the event has no row.
    """

    type: Literal["user.stop"] = "user.stop"
    turn_id: str


class Tick(_Event):
    """Time advances only through ticks; timers fire when a tick passes their deadline."""

    type: Literal["tick"] = "tick"


Event = Annotated[
    UserSpeechOnset
    | UserBackchannel
    | UserTakeover
    | UserPause
    | UserSpeechOffset
    | UserNotOwner
    | UserText
    | SttWord
    | SttStep
    | SttEndOfTurn
    | SttFlushed
    | SttError
    | SceneChanged
    | LlmCommit
    | LlmDone
    | LlmError
    | ToolResult
    | PrefillReady
    | TtsDone
    | TtsError
    | PlaybackStopped
    | PlaybackDrained
    | UiAction
    | ClientDisconnected
    | SessionReset
    | MicMuted
    | UserStop
    | Tick,
    Field(discriminator="type"),
]

EventAdapter: TypeAdapter[Event] = TypeAdapter(Event)

EVENT_TYPES: tuple[type[_Event], ...] = (
    UserSpeechOnset,
    UserBackchannel,
    UserTakeover,
    UserPause,
    UserSpeechOffset,
    UserNotOwner,
    UserText,
    SttWord,
    SttStep,
    SttEndOfTurn,
    SttFlushed,
    SttError,
    SceneChanged,
    LlmCommit,
    LlmDone,
    LlmError,
    ToolResult,
    PrefillReady,
    TtsDone,
    TtsError,
    PlaybackStopped,
    PlaybackDrained,
    UiAction,
    ClientDisconnected,
    SessionReset,
    MicMuted,
    UserStop,
    Tick,
)
