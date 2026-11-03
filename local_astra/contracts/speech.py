"""Speech contracts: STT events from moshi-server, TTS utterances, word timings (docs/02 §5)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from local_astra.contracts.base import Contract


class SttWordEvent(Contract):
    """One recognised word, in *audio* time. ``end_estimated`` says who measured its end.

    ``False`` — the default, so every producer that is not ``MoshiSttClient`` keeps meaning "this
    end was measured" — is an ``EndWord`` the server actually sent. ``True`` is an end the client
    had to supply itself: the next word's arrival, the marker echo, ``close()``, a disconnect, or
    the bounded hold of ``STT_OPEN_WORD_HOLD_MS`` (T-0051 criteria 8 and 9). The distinction is
    load-bearing rather than diagnostic: the endpointer refuses to open a run on an estimated end
    (T-0051 criterion 12), because a word released by the hold mid-reply would otherwise become a
    barge-in through the STT tap. The flag lives on this contract and the endpointer acts on it;
    it does **not** reach the session log, whose ``stt.word`` row carries ``t_ms``, ``text``,
    ``start_ms`` and ``end_ms`` and nothing else (T-0051 criterion 8's negative, ruling B39), so
    the log still cannot say of ``see?`` on 2026-09-16 that its 21 040 ms duration was invented.
    Carrying the flag into the log is a later task.
    """

    type: Literal["word"] = "word"
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    end_estimated: bool = False


class SttStepEvent(Contract):
    type: Literal["step"] = "step"
    pause_heads: tuple[float, ...] = ()
    #: audio time of the step (moshi-server steps every 80 ms)
    t_ms: int = Field(ge=0)


class SttEndOfTurnEvent(Contract):
    type: Literal["end_of_turn"] = "end_of_turn"
    t_eot_ms: int = Field(ge=0)
    p_end: float = Field(default=1.0, ge=0.0, le=1.0)


class SttErrorEvent(Contract):
    type: Literal["error"] = "error"
    message: str


class SttFlushedEvent(Contract):
    """The ``Marker`` echo: every word up to ``t_eot_ms`` has been delivered (docs/02 §4 row 6).

    It travels on the same ordered stream as the words, so "after the flush" is a statement
    about the wire and never about a clock (T-0007a criterion 4).
    """

    type: Literal["flushed"] = "flushed"
    t_eot_ms: int = Field(ge=0)


SttEvent = Annotated[
    SttWordEvent | SttStepEvent | SttEndOfTurnEvent | SttErrorEvent | SttFlushedEvent,
    Field(discriminator="type"),
]
SttEventAdapter: TypeAdapter[SttEvent] = TypeAdapter(SttEvent)


class WordTiming(Contract):
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)


class TtsRequest(Contract):
    utterance_id: str
    text: str
    voice: str = "default"


class AudioFormat(Contract):
    sample_rate: int = Field(default=24000, ge=8000)
    channels: int = Field(default=1, ge=1, le=2)
    sample_format: Literal["s16le", "f32le"] = "s16le"


TTS_OUTPUT_FORMAT = AudioFormat(sample_rate=24000, channels=1, sample_format="s16le")
CAPTURE_FORMAT = AudioFormat(sample_rate=48000, channels=1, sample_format="f32le")
STT_INPUT_FORMAT = AudioFormat(sample_rate=24000, channels=1, sample_format="f32le")

#: Conductor audio block size (docs/06 R3): 10 ms at 48 kHz.
CAPTURE_BLOCK_MS = 10
