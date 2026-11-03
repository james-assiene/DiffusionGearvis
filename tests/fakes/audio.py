"""Fake ``AudioIO`` driven by the virtual clock: ``played_ms`` is exact by construction.

T-0006 ruling A4: the fake grows the seams the playback module grew — ``play(..., source=)``,
``mark_drain()``, ``playback_events()`` and ``played_text`` on the terminal events — so that
``tests/contract/test_audio_io_conformance.py`` can run the same behavioural assertions against
this fake and against ``PipeWireAudioIO`` (CLAUDE.md: an interface change changes the fake with
it). The rules implemented here are docs/02 §4 and its 2026-09-13 spec deltas:

- ``played_ms`` is clock time inside the run, clamped to what was enqueued;
- ``PlaybackDrained`` needs ``mark_drain()`` **and** the last sample — a queue that merely empties
  is an underrun, never a drain;
- ``played_text`` concatenates, with **no separator**, the sources of fully played utterances plus
  the word-boundary prefix of the partial one (a word counts as played if it started ≥ 100 ms
  before the cut), and a partial utterance whose source word count differs from its rendering
  contributes no text at all.

T-0030 ruling A2: the partial keeps the **source's own spacing** (``audio_io._word_prefix`` returns
``source[:index]``), so a source with a double space or a newline produces the same string here as
it does in ``local_astra/conductor/audio_io.py``. The real module is the contract; this fake
follows it (``tests/contract/test_audio_io_conformance.py`` is what holds the two together).

T-0021 adds the capture side, for the same reason: ``capture()`` now yields real 1920-byte blocks
(480 f32 at 48 kHz = 10 ms, ``CAPTURE_FORMAT``), ``capture_frames()`` yields blocks with the
fields ``CaptureBlock`` has, ``stt_blocks()`` yields 1920-sample 80 ms frames at 24 kHz
(``STT_INPUT_FORMAT``) and ``inject_xrun(ms)`` makes the raw timeline show the gap while the STT
timeline stays contiguous (silence substituted). The block *content* here is a placeholder — the
fake owns no anti-aliasing filter, and every assertion about the audio itself belongs to the unit
tests of ``local_astra/conductor/audio_resample.py``. What the fake does own is the *interface*:
block size, cadence, timestamps and contiguity, which is what
``tests/contract/test_audio_capture_conformance.py`` asserts against both implementations.
"""

from __future__ import annotations

import asyncio
import re
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from local_astra.contracts.events import PlaybackDrained, PlaybackStopped
from local_astra.contracts.speech import CAPTURE_BLOCK_MS, CAPTURE_FORMAT, STT_INPUT_FORMAT
from tests.conftest import VirtualClock
from tests.fakes.tts import FakeUtterance

#: docs/02 §4: "a word counts as played if it started ≥ 100 ms before the cut".
GUARD_BAND_MS = 100
#: criterion 8: the event queue is bounded, keeps the newest and counts the drops.
EVENT_SLOTS = 8

#: T-0021: 10 ms of 48 kHz mono f32 in, 80 ms of 24 kHz mono f32 out.
CAPTURE_RATE = CAPTURE_FORMAT.sample_rate  # 48000
STT_RATE = STT_INPUT_FORMAT.sample_rate  # 24000
CAPTURE_BLOCK_SAMPLES = CAPTURE_RATE * CAPTURE_BLOCK_MS // 1000  # 480
STT_BLOCK_SAMPLES = 1920
STT_BLOCK_MS = 1000 * STT_BLOCK_SAMPLES // STT_RATE  # 80


@dataclass(frozen=True)
class FakeCaptureBlock:
    """The fields of ``audio_capture.CaptureBlock`` (T-0021 criterion 2)."""

    samples: np.ndarray
    t_ms: float
    xrun: bool = False
    gated: bool = False
    seq: int = 0


@dataclass(frozen=True)
class FakeSttFrame:
    """The fields of ``audio_capture.SttFrame``: 1920 samples and criterion 10's two times."""

    samples: np.ndarray
    t_ms: float
    end_ms: float
    yielded_ms: float


@dataclass(frozen=True)
class FakeCaptureStats:
    """The fields of ``audio_capture.CaptureStats`` (T-0021 criterion 10)."""

    blocks: int
    xruns: int
    substituted_ms: float
    dropped_blocks: int
    gated_blocks: int
    quantum_ms: float
    device: str


@dataclass
class _Queued:
    utterance: Any
    source: str
    offset_ms: int
    duration_ms: int


class FakeAudioIO:
    """``capture_origin_ms`` (T-0199): the audio time of the first captured sample.

    ``None`` (the default) keeps every earlier row byte-identical: the capture clock starts at
    ``clock.now_ms`` when capture is armed, so audio time and session time coincide -- the T-0051
    B19 kindness that hid a whole class of two-clock bugs. A number makes the fake behave like the
    real ``audio_capture`` module, whose ``CaptureBlock.t_ms``/``SttFrame.t_ms`` count the ADC
    position from the open of the stream, independent of the session clock: blocks and STT frames
    take ``t_ms = capture_origin_ms + elapsed session time since the arm``. The arm (the first
    ``capture_frames()``/``stt_blocks()`` iteration) is still what starts the elapsed count, and
    ``yielded_ms`` stays on the session clock.
    """

    def __init__(self, clock: VirtualClock, *, capture_origin_ms: float | None = None) -> None:
        self.clock = clock
        self.capture_origin_ms = capture_origin_ms
        self.current: FakeUtterance | None = None
        self.started_ms: int | None = None
        self.paused_at_ms: int | None = None
        self.paused_total_ms = 0
        self.flushes: list[int] = []
        self.captured_blocks = 0
        self.queue: list[_Queued] = []
        self.enqueued_ms = 0
        self.dropped_events = 0
        self.underruns = 0
        self._events: deque[PlaybackStopped | PlaybackDrained] = deque(maxlen=EVENT_SLOTS)
        self._drain_marked = False
        self._terminal = False
        self._final = 0
        #: T-0092a: delivery hold (see ``hold_events``). Off by default, so every existing row
        #: sees exactly the delivery it saw before.
        self._events_held = False
        # -- the capture side (T-0021)
        self._cap_start_ms: int | None = None
        #: T-0199: the audio time of capture sample 0 (``_cap_start_ms`` unless ``capture_origin_ms`` is given)
        self._cap_t0_ms: float = 0.0
        self._cap_index = 0
        self._cap_blocks: list[FakeCaptureBlock] = []
        self._cap_frames: list[FakeSttFrame] = []
        self._cap_stt_buf = np.zeros(0, dtype=np.float32)
        self._cap_stt_ms = 0.0
        self._cap_pending_xrun_ms = 0.0
        self._cap_xruns = 0
        self._cap_substituted_ms = 0.0

    async def play(self, utterance: Any, *, source: str | None = None) -> None:
        if self.started_ms is None:
            self.started_ms = self.clock.now_ms
            self.paused_at_ms = None
            self.paused_total_ms = 0
            self.enqueued_ms = 0
            self.queue = []
            self._drain_marked = False
            self._terminal = False
        duration = int(getattr(utterance, "duration_ms", 0))
        self.queue.append(
            _Queued(
                utterance=utterance,
                source=source if source is not None else str(getattr(utterance, "text", "")),
                offset_ms=self.enqueued_ms,
                duration_ms=duration,
            )
        )
        self.enqueued_ms += duration
        self.current = utterance

    def played_ms(self) -> int:
        if self.started_ms is None:
            return 0
        end = self.paused_at_ms if self.paused_at_ms is not None else self.clock.now_ms
        return max(0, min(self.enqueued_ms, end - self.started_ms - self.paused_total_ms))

    async def pause(self) -> None:
        if self.paused_at_ms is None:
            self.paused_at_ms = self.clock.now_ms

    async def resume(self) -> None:
        if self.paused_at_ms is not None:
            self.paused_total_ms += self.clock.now_ms - self.paused_at_ms
            self.paused_at_ms = None

    async def flush(self) -> int:
        if self._terminal:
            return self._final
        played = self.played_ms()
        self.flushes.append(played)
        self._emit(
            PlaybackStopped(
                t_ms=self.clock.now_ms, played_ms=played, played_text=self.played_text(played)
            )
        )
        self._end_run(played)
        return played

    def mark_drain(self) -> None:
        """No further audio is coming for this run (T-0006 criterion 7)."""
        self._drain_marked = True

    async def playback_events(self) -> AsyncIterator[PlaybackStopped | PlaybackDrained]:
        while True:
            self._poll()
            if self._events and not self._events_held:
                yield self._events.popleft()
            else:
                await asyncio.sleep(0)

    # --- T-0092a: a late consumer -----------------------------------------------------------
    def hold_events(self) -> None:
        """Stop *delivering* terminal events; the run keeps going and still emits them.

        This models a consumer that reads late, nothing more. ``audio_io.PipeWireAudioIO``
        appends terminal events to a list that ``playback_events()`` drains only when its
        consumer next iterates, so an event can be emitted in one turn and read in the next.
        While held, the run's state machine is still advanced (``_poll`` runs, a drain is still
        emitted at ``mark_drain()`` + last sample, ``flush()`` still ends the run and stays
        idempotent once the run is terminal), the bounded queue still keeps the newest, and
        emission order is delivery order. Only the hand-over to the consumer waits.
        """
        self._events_held = True

    def release_events(self) -> None:
        """Deliver what was held, in emission order, and everything after it as it comes."""
        self._events_held = False

    def pending_events(self) -> tuple[PlaybackStopped | PlaybackDrained, ...]:
        """Terminal events emitted and not yet delivered, oldest first (a read-only view)."""
        self._poll()
        return tuple(self._events)

    def is_rendering(self, t_ms: float) -> bool:
        """True when the device is putting samples on the wire at ``t_ms`` (T-0006 criterion 10).

        The capture-side self-speech gate (T-0021) and the ``AudioIO`` widening T-0007 owes both
        name this seam, so the fake carries it too. A run that has ended, a run that has not
        started and a time past the last enqueued sample all answer False; a paused run answers
        False for every time at or after the pause, because nothing is being rendered then.
        """
        if self.started_ms is None or self._terminal:
            return False
        if self.paused_at_ms is not None and t_ms >= self.paused_at_ms:
            return False
        end_ms = self.started_ms + self.paused_total_ms + self.enqueued_ms
        return self.started_ms <= t_ms < end_ms

    def played_text(self, played_ms: int) -> str:
        parts: list[str] = []
        for item in self.queue:
            if played_ms >= item.offset_ms + item.duration_ms:
                parts.append(item.source)
                continue
            if played_ms <= item.offset_ms:
                break
            words = item.utterance.words() or []
            source_words = item.source.split()
            if len(source_words) != len(words):
                break  # normalisation mismatch: the partial utterance contributes nothing
            spoken = sum(
                1 for w in words if item.offset_ms + w.start_ms + GUARD_BAND_MS <= played_ms
            )
            if spoken:
                # T-0030 ruling A2: the *literal* spacing of the source, as
                # ``audio_io._word_prefix`` produces it (``source[:index]``) -- not a re-join, which
                # diverges from the real module on a double space or a newline.
                parts.append(re.match(rf"(?:\s*\S+){{{spoken}}}", item.source).group(0))
            break
        return "".join(parts)

    # --- the capture side (T-0021) ---------------------------------------------------------
    def inject_xrun(self, ms: float) -> None:
        """``ms`` of capture audio is lost: the raw timeline shows it, the STT one is filled."""
        self._cap_pending_xrun_ms += float(ms)

    def capture_stats(self) -> FakeCaptureStats:
        return FakeCaptureStats(
            blocks=self.captured_blocks,
            xruns=self._cap_xruns,
            substituted_ms=self._cap_substituted_ms,
            dropped_blocks=0,
            gated_blocks=0,
            quantum_ms=float(CAPTURE_BLOCK_MS),
            device="fake",
        )

    async def capture(self) -> AsyncIterator[bytes]:
        """``AudioIO.capture()``: 10 ms blocks of 48 kHz mono f32 as raw bytes (1920 of them)."""
        async for block in self.capture_frames():
            yield block.samples.astype(np.float32).tobytes()

    async def capture_frames(self) -> AsyncIterator[FakeCaptureBlock]:
        cursor = self._cap_arm()
        while True:
            self._cap_produce()
            if cursor < len(self._cap_blocks):
                yield self._cap_blocks[cursor]
                cursor += 1
            else:
                await asyncio.sleep(0)

    async def stt_blocks(self) -> AsyncIterator[FakeSttFrame]:
        cursor = self._cap_arm()
        while True:
            self._cap_produce()
            if cursor < len(self._cap_frames):
                yield self._cap_frames[cursor]
                cursor += 1
            else:
                await asyncio.sleep(0)

    def _cap_arm(self) -> int:
        if self._cap_start_ms is None:
            self._cap_start_ms = self.clock.now_ms
            origin = self.capture_origin_ms
            if origin is None:
                self._cap_t0_ms = self.clock.now_ms  # type: ignore[assignment]
                self._cap_stt_ms = float(self.clock.now_ms)
            else:
                self._cap_t0_ms = float(origin)
                self._cap_stt_ms = float(origin)
        return 0

    def _cap_produce(self) -> None:
        """Turn elapsed virtual time into 10 ms blocks, and those into 80 ms STT frames."""
        if self._cap_start_ms is None:
            return
        while True:
            gap_ms = self._cap_pending_xrun_ms
            gap_frames = round(gap_ms * CAPTURE_RATE / 1000.0)
            elapsed = self.clock.now_ms - self._cap_start_ms
            if elapsed * CAPTURE_RATE / 1000.0 < (
                self._cap_index + gap_frames + CAPTURE_BLOCK_SAMPLES
            ):
                return  # the ADC has not produced this block yet
            if gap_ms > 0.0:
                self._cap_pending_xrun_ms = 0.0
                self._cap_index += gap_frames
                self._cap_xruns += 1
            t_ms = self._cap_t0_ms + 1000.0 * self._cap_index / CAPTURE_RATE
            samples = self._cap_pattern(self._cap_index, CAPTURE_BLOCK_SAMPLES)
            self._cap_blocks.append(
                FakeCaptureBlock(
                    samples=samples,
                    t_ms=t_ms,
                    xrun=gap_ms > 0.0,
                    gated=False,
                    seq=self.captured_blocks,
                )
            )
            self.captured_blocks += 1
            self._cap_index += CAPTURE_BLOCK_SAMPLES
            if gap_ms > 0.0:  # the STT timeline has no hole: silence of the missing duration
                self._cap_push_stt(np.zeros(round(gap_ms * STT_RATE / 1000.0), dtype=np.float32))
                self._cap_substituted_ms += gap_ms
            self._cap_push_stt(0.5 * (samples[0::2] + samples[1::2]))

    def _cap_push_stt(self, samples: np.ndarray) -> None:
        self._cap_stt_buf = np.concatenate([self._cap_stt_buf, samples])
        while self._cap_stt_buf.size >= STT_BLOCK_SAMPLES:
            frame, self._cap_stt_buf = (
                self._cap_stt_buf[:STT_BLOCK_SAMPLES],
                self._cap_stt_buf[STT_BLOCK_SAMPLES:],
            )
            t_ms = self._cap_stt_ms
            self._cap_stt_ms += STT_BLOCK_MS
            self._cap_frames.append(
                FakeSttFrame(
                    samples=frame,
                    t_ms=t_ms,
                    end_ms=t_ms + STT_BLOCK_MS - 1000.0 / STT_RATE,
                    yielded_ms=float(self.clock.now_ms),
                )
            )

    @staticmethod
    def _cap_pattern(start: int, n: int) -> np.ndarray:
        """A placeholder signal, identifiable by absolute sample index (never silence)."""
        idx = np.arange(start, start + n, dtype=np.int64)
        return ((idx % 8191).astype(np.float64) / 8191.0 - 0.5).astype(np.float32)

    def _poll(self) -> None:
        """Emit the drain when, and only when, ``mark_drain()`` and the last sample have met."""
        if self.started_ms is None or self._terminal:
            return
        played = self.played_ms()
        if played < self.enqueued_ms:
            return
        if not self._drain_marked:
            self.underruns += 1
            return
        self._emit(
            PlaybackDrained(
                t_ms=self.clock.now_ms, played_ms=played, played_text=self.played_text(played)
            )
        )
        self._end_run(played)

    def _emit(self, event: PlaybackStopped | PlaybackDrained) -> None:
        if len(self._events) == EVENT_SLOTS:
            self.dropped_events += 1  # the oldest goes, never the newest (criterion 8)
        self._events.append(event)

    def _end_run(self, played: int) -> None:
        self._final = played
        self._terminal = True
        self.current = None
        self.started_ms = None
        self.paused_at_ms = None
