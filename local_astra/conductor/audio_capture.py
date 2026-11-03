"""Capture: the device's clock, 10 ms blocks, 48 -> 24 kHz, 80 ms STT frames, the gate (T-0021).

This is the only way audio enters the system. Everything here is plumbing around one number, the
capture twin of T-0006's ``played_ms``: **a block's ``t_ms`` is the device's ADC position**
(``AudioDevice.captured_ms()`` / the ``adc_ms`` handed to the delivery callback, PortAudio's
``inputBufferAdcTime``), never the moment a callback ran or a coroutine resumed. A device that
stalls and then bursts ten buffers at once must still produce ten blocks 10 ms apart, because the
ADC ran the whole time; timestamping at arrival gives ten near-identical stamps and silently
re-dates every word the STT will return (criterion 3). ``input_latency_ms`` is reported but never
added on top: ``captured_ms()`` already *is* the capture instant (ruling A4).

**Three streams, one source.** The device thread hands over buffers of whatever size the graph
negotiated (1024 frames here, not a multiple of a 10 ms block, so re-blocking is unconditional);
``_on_capture`` re-blocks and timestamps them and drops them in a bounded ring. From there:

* ``blocks()``/``bytes_blocks()`` -- the **raw** stream, never gated, never zeroed, never delayed.
  The reflex VAD reads it, and it is the only thing that can hear a barge-in while we are speaking
  (docs/02 section 4 ``SPEAKING | user_speech_onset | PAUSED``). A gated block is *marked*, not
  silenced, and an xrun is *shown*: ``t_ms`` steps by the real gap.
* ``stt_frames()`` -- the **STT tap**: gate -> silence, xrun -> substituted silence, 48 -> 24 kHz,
  1920-sample (80 ms) frames, which is what the Kyutai STT server reads. Its timeline is
  contiguous by construction because the delayed-stream model advances its clock only while audio
  arrives (measured: ``scripts/demo/_listen.py`` had to append real silence to flush it, or a wav
  that stopped dead transcribed to nothing at all). Substitution is capped at
  ``MAX_SUBSTITUTED_MS``; past that the stream raises ``SubstitutionLimitExceeded`` rather than
  manufacture more audio.

**The self-speech gate is on the STT tap only** (criterion 8, docs/02 section 4 spec delta
2026-09-13). ``hangover_ms`` defaults to ``DEFAULT_HANGOVER_MS`` = 60 ms with this regime, stated
because a number without one cannot be compared later (CLAUDE.md): T-0006's acoustic **echo-path
delay p95 is not measured yet** -- the acoustic rig is blocked on prerequisite P1
(``libportaudio2``), so the value stands in for it as T-0006's criterion 3 p95 budget (50 ms,
pause -> silence, null-sink rig) **plus one 10 ms block**, per criterion 8's "p95 + one block".
T-0023 measures the real echo path against speakers and a mic and revises this constant; until
then it is a budget, not a measurement.

**Threading (criterion 5, .claude/rules/conductor.md).** ``_on_capture`` runs on the device
thread: it takes one lock for a memory copy, awaits nothing, calls no asyncio primitive and
allocates nothing unbounded. The gate, the decimator and the framing run on the consumer side, so
the device thread never calls into the playback module and no lock is ever held across both.
Backpressure is a bounded ring of ``QUEUE_BLOCKS`` blocks that drops the **oldest** and counts the
drop once, at the ring (ruling A16b: three consumers reading one ring reported three times the
loss); the next block the consumer sees is from the live edge and its ``t_ms`` shows the true gap.
Each consumer's own bounded queue drops oldest too and counts that separately.

This module imports no device library (criterion 1: the device arrives by injection), opens no
socket, and does not write the turn record -- it reports ``CaptureStats`` and the two per-frame
times ``session.py`` (T-0007) needs for ``spans["audio.capture"]``.
"""

from __future__ import annotations

import asyncio
import math
import threading
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from local_astra.conductor.audio_device import DEVICE_RATE, AudioDevice
from local_astra.conductor.audio_resample import STT_RATE, Resampler48kTo24k
from local_astra.contracts.speech import CAPTURE_BLOCK_MS

#: 480 f32 = 10 ms at 48 kHz: what every raw block is, whatever the device hands over.
BLOCK_SAMPLES = DEVICE_RATE * CAPTURE_BLOCK_MS // 1000
#: The STT server reads 1920 samples (80 ms) of 24 kHz mono f32 at a time (``STT_INPUT_FORMAT``;
#: ``BLOCK = 1920`` in ``scripts/demo/_listen.py``).
STT_BLOCK_SAMPLES = 1920
STT_BLOCK_MS = 1000.0 * STT_BLOCK_SAMPLES / STT_RATE
#: Longest run of continuous substituted silence before the STT stream gives up (criterion 4).
MAX_SUBSTITUTED_MS = 500.0
#: Capture backpressure: 64 blocks = 640 ms, over criterion 5's floor of 32 (320 ms).
QUEUE_BLOCKS = 64
#: Default self-speech hangover. See the module docstring for the regime behind the number.
DEFAULT_HANGOVER_MS = 60.0
#: How long a consumer waits before asking the ring again. Small: it is the granularity of the
#: capture handoff, not of the event loop, and nothing blocks in between.
POLL_S = 0.0002
#: Half a sample at 48 kHz: below this, two capture timestamps are the same instant.
_TOLERANCE_MS = 0.5 * 1000.0 / DEVICE_RATE


class SubstitutionLimitExceeded(RuntimeError):
    """More than ``MAX_SUBSTITUTED_MS`` of audio in a row never arrived (criterion 4).

    Named, and raised on the STT stream only: past the cap the honest answer is that capture is
    broken, not another half second of manufactured silence. The raw stream keeps running.
    """


@dataclass(frozen=True)
class CaptureBlock:
    """10 ms of 48 kHz mono f32, stamped with the ADC instant of its **first** sample."""

    samples: npt.NDArray[np.float32]
    t_ms: float
    #: True on the first block after audio the device never gave us (criterion 4a).
    xrun: bool = False
    #: True when the self-speech gate silenced this block *in the STT stream*. The samples here
    #: are untouched: gating the raw stream would destroy barge-in (criterion 8, adversarial 2).
    gated: bool = False
    seq: int = 0


@dataclass(frozen=True)
class SttFrame:
    """1920 samples (80 ms) of 24 kHz mono f32 -- one STT-server block (criterion 7)."""

    samples: npt.NDArray[np.float32]
    #: Capture time of the frame's first sample.
    t_ms: float
    #: Capture time of its last sample; with ``yielded_ms`` this is criterion 10's pair.
    end_ms: float
    #: Capture-clock time at which the frame was handed to the consumer.
    yielded_ms: float


@dataclass(frozen=True)
class CaptureStats:
    """The snapshot criterion 10 names. The module reports; ``session.py`` writes the row."""

    blocks: int
    xruns: int
    substituted_ms: float
    dropped_blocks: int
    gated_blocks: int
    quantum_ms: float
    device: str


class SelfSpeechGate:
    """Pure, injected, on the STT tap only (criterion 8).

    A block is gated when the device was rendering our own audio at its capture timestamp, or
    when it starts within ``hangover_ms`` of the last block that was. The hangover runs from the
    last capture time we were *told* was rendering, because those timestamps are the only
    instants this object ever asks about (the caller hands them over in capture order); an
    implementation with no hangover reopens the gate in the 5 ms gap between two utterances and
    chatters one block of our own voice into the STT at every sentence boundary.
    """

    def __init__(
        self,
        is_rendering: Callable[[float], bool],
        hangover_ms: float = DEFAULT_HANGOVER_MS,
        *,
        block_ms: float = float(CAPTURE_BLOCK_MS),
    ) -> None:
        self._is_rendering = is_rendering
        self.hangover_ms = float(hangover_ms)
        self._block_ms = float(block_ms)
        self._open_until_ms = -math.inf

    def should_gate(self, t_ms: float) -> bool:
        rendering = bool(self._is_rendering(t_ms))
        if rendering:
            # Anchored on the *end* of the rendering block, not its start (ruling A16c): the
            # caller only ever names block starts, so an anchor on ``t_ms`` would protect
            # p95 past the last rendering block instead of the p95 + one block the constant
            # and the docstring both claim.
            self._open_until_ms = float(t_ms) + self._block_ms + self.hangover_ms
        return rendering or float(t_ms) < self._open_until_ms


@dataclass
class _Subscription:
    """One consumer's bounded view of the raw stream (criterion 5: drop oldest, count it).

    Its own drops are counted here and **not** added to ``CaptureStats.dropped_blocks``: that
    number is the ring's, counted once per block lost (ruling A16b). Three consumers reading one
    ring must not report three times the loss; a per-stream number is available here for the
    session if it ever needs one.
    """

    queue: deque[CaptureBlock]
    dropped: int = 0


@dataclass
class _SttState:
    """Per-consumer state of the STT tap: filter, frame carry and the two timelines."""

    decimator: Resampler48kTo24k = field(default_factory=Resampler48kTo24k)
    carry: npt.NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    #: Capture time of the next raw block the tap expects; a later one means a hole to fill.
    next_block_ms: float | None = None
    #: Capture time of the first sample of the frame being filled.
    frame_t_ms: float = 0.0


class CaptureStream:
    """The capture pipeline over an injected ``AudioDevice`` (criterion 1: no device library).

    One device object serves capture and playback, so ``block.t_ms`` and the playback module's
    ``rendered_ms()``/``is_rendering(t_ms)`` are directly comparable without a wall-clock
    subtraction.
    """

    def __init__(
        self,
        device: AudioDevice,
        *,
        gate: SelfSpeechGate | None = None,
        block_ms: int = CAPTURE_BLOCK_MS,
        queue_blocks: int = QUEUE_BLOCKS,
    ) -> None:
        self._device = device
        self._gate = gate
        self._block_ms = float(block_ms)
        self._block = round(block_ms * DEVICE_RATE / 1000)
        self._queue_blocks = queue_blocks

        self._lock = threading.Lock()
        #: Bounded, drop-oldest, counted: the device thread must never wait for a consumer.
        self._pending: deque[tuple[npt.NDArray[np.float32], float, bool, int]] = deque(
            maxlen=queue_blocks
        )
        self._subs: list[_Subscription] = []

        self._carry: npt.NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self._carry_t_ms = 0.0
        self._next_sample_ms: float | None = None
        self._flag_xrun = False

        self._started = False
        self._closed = False

        self.blocks_produced = 0
        self.xruns = 0
        self.dropped_blocks = 0
        self.gated_blocks = 0
        self.substituted_ms = 0.0

    # -- the device thread (criterion 5: no awaits, no asyncio, nothing unbounded) ---------
    def _on_capture(self, samples: npt.NDArray[np.float32], adc_ms: float, xrun: bool) -> None:
        """One device buffer: re-block it, stamp it from the capture clock, ring it.

        ``adc_ms`` is the capture time of the buffer's first sample. A jump in it (or the
        backend's own xrun flag) means audio the ADC took never reached us: the partial block in
        hand is dropped rather than spliced across the hole -- the raw stream never invents a
        sample -- and the next block carries ``xrun=True`` with the real gap in its ``t_ms``.
        """
        with self._lock:
            if self._closed:
                return
            block = self._block
            if self._next_sample_ms is None:
                self._carry_t_ms = adc_ms
            elif xrun or adc_ms - self._next_sample_ms > _TOLERANCE_MS:
                self.xruns += 1
                self._flag_xrun = True
                # Counted here, where the hole is *detected*, and once (ruling A16b): the STT
                # tap fills exactly this much silence, and counting it there instead makes the
                # number a multiple of however many consumers happen to be reading.
                gap_ms = max(0.0, adc_ms - self._next_sample_ms)
                self.substituted_ms += min(gap_ms, MAX_SUBSTITUTED_MS)
                self._carry = np.zeros(0, dtype=np.float32)
                self._carry_t_ms = adc_ms
            self._next_sample_ms = adc_ms + 1000.0 * samples.size / DEVICE_RATE
            carry = samples if self._carry.size == 0 else np.concatenate([self._carry, samples])
            at = 0
            while carry.size - at >= block:
                if len(self._pending) == self._queue_blocks:
                    self.dropped_blocks += 1
                self._pending.append(
                    (
                        carry[at : at + block],
                        self._carry_t_ms,
                        self._flag_xrun,
                        self.blocks_produced,
                    )
                )
                self._flag_xrun = False
                self._carry_t_ms += self._block_ms
                self.blocks_produced += 1
                at += block
            self._carry = carry[at:]

    # -- lifecycle ------------------------------------------------------------------------
    def _ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        self._device.set_capture_callback(self._on_capture)
        self._device.start_capture()

    def close(self) -> None:
        """Explicit end of stream: the device stops, the iterators finish, the tail is padded."""
        self._device.stop_capture()
        with self._lock:
            self._closed = True

    def stats(self) -> CaptureStats:
        with self._lock:
            return CaptureStats(
                blocks=self.blocks_produced,
                xruns=self.xruns,
                substituted_ms=self.substituted_ms,
                dropped_blocks=self.dropped_blocks,
                gated_blocks=self.gated_blocks,
                quantum_ms=float(self._device.quantum_ms),
                device=str(getattr(self._device, "sink", None) or type(self._device).__name__),
            )

    # -- fan-out ---------------------------------------------------------------------------
    def _subscribe(self) -> _Subscription:
        self._ensure_started()
        sub = _Subscription(queue=deque(maxlen=self._queue_blocks))
        with self._lock:
            self._subs.append(sub)
        return sub

    def _unsubscribe(self, sub: _Subscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    def _drain(self) -> None:
        """Move what the device thread produced to the consumers, gating once per block.

        Runs on the event loop thread and holds no lock while it calls the gate: the gate reads
        the playback module, which reads the device, and a lock taken across all three in the
        other order would close a cycle with the playback writer thread.
        """
        with self._lock:
            items = list(self._pending)
            self._pending.clear()
        if not items:
            return
        blocks = []
        gated_count = 0
        for samples, t_ms, xrun, seq in items:
            gated = self._gate.should_gate(t_ms) if self._gate is not None else False
            gated_count += int(gated)
            blocks.append(CaptureBlock(samples=samples, t_ms=t_ms, xrun=xrun, gated=gated, seq=seq))
        with self._lock:
            self.gated_blocks += gated_count
            for item in blocks:
                # Once per block lost, however many queues lost it (ruling A16b): three
                # consumers overflowing on the same audio reported three times the loss. The
                # per-consumer number stays on the subscription for whoever wants it.
                lost = False
                for sub in self._subs:
                    if len(sub.queue) == self._queue_blocks:
                        sub.dropped += 1
                        lost = True
                    sub.queue.append(item)
                self.dropped_blocks += int(lost)

    # -- the three streams -----------------------------------------------------------------
    async def blocks(self) -> AsyncIterator[CaptureBlock]:
        """The raw stream: every block the device captured, unmodified, at cadence."""
        sub = self._subscribe()
        try:
            while True:
                self._drain()
                if sub.queue:
                    yield sub.queue.popleft()
                    continue
                if self._closed:
                    return
                await asyncio.sleep(POLL_S)
        finally:
            self._unsubscribe(sub)

    async def bytes_blocks(self) -> AsyncIterator[bytes]:
        """``AudioIO.capture()``: the same blocks as 1920 bytes of little-endian f32."""
        async for block in self.blocks():
            yield np.asarray(block.samples, dtype="<f4").tobytes()

    async def stt_frames(self) -> AsyncIterator[SttFrame]:
        """The STT tap: gated and xrun-filled, 24 kHz, 1920 samples per frame."""
        sub = self._subscribe()
        state = _SttState()
        try:
            while True:
                self._drain()
                if sub.queue:
                    batch = list(sub.queue)
                    sub.queue.clear()
                    for frame in self._feed_stt(state, batch):
                        yield frame
                    continue
                if self._closed:
                    for frame in self._flush_stt(state):
                        yield frame
                    return
                await asyncio.sleep(POLL_S)
        finally:
            self._unsubscribe(sub)

    # -- the STT tap -----------------------------------------------------------------------
    def _feed_stt(self, state: _SttState, batch: Sequence[CaptureBlock]) -> Iterator[SttFrame]:
        """Blocks in, 80 ms frames out: substitute, gate, decimate, frame -- in that order.

        Contiguous blocks are decimated in one call. That is safe because ``Resampler48kTo24k``
        is bit-identical under any chunking (criterion 6), and it is what keeps a 60 s stream
        from costing 6000 filter calls.
        """
        run: list[npt.NDArray[np.float32]] = []
        for block in batch:
            if state.next_block_ms is None:
                state.next_block_ms = block.t_ms
                state.frame_t_ms = block.t_ms
            gap_ms = block.t_ms - state.next_block_ms
            if gap_ms > _TOLERANCE_MS:
                # The delayed-stream STT advances its clock only with audio, so a hole in the
                # capture becomes silence of exactly the missing duration -- up to the cap.
                filled_ms = min(gap_ms, MAX_SUBSTITUTED_MS)
                run.append(np.zeros(round(filled_ms * DEVICE_RATE / 1000.0), dtype=np.float32))
                if gap_ms > MAX_SUBSTITUTED_MS:
                    yield from self._emit(state, run)
                    raise SubstitutionLimitExceeded(
                        f"{gap_ms:.0f} ms of capture never arrived; substitution is capped at "
                        f"{MAX_SUBSTITUTED_MS:.0f} ms (T-0021 criterion 4)"
                    )
            state.next_block_ms = block.t_ms + self._block_ms
            run.append(
                np.zeros(block.samples.size, dtype=np.float32) if block.gated else block.samples
            )
        yield from self._emit(state, run)

    def _emit(self, state: _SttState, run: list[npt.NDArray[np.float32]]) -> Iterator[SttFrame]:
        if not run:
            return
        pcm = state.decimator.process(run[0] if len(run) == 1 else np.concatenate(run))
        run.clear()
        state.carry = pcm if state.carry.size == 0 else np.concatenate([state.carry, pcm])
        if state.carry.size < STT_BLOCK_SAMPLES:
            return
        yielded_ms = float(self._device.captured_ms())
        at = 0
        while state.carry.size - at >= STT_BLOCK_SAMPLES:
            yield self._frame(state, state.carry[at : at + STT_BLOCK_SAMPLES], yielded_ms)
            at += STT_BLOCK_SAMPLES
        state.carry = state.carry[at:]

    def _flush_stt(self, state: _SttState) -> Iterator[SttFrame]:
        """Criterion 7: the **only** place a partial frame is zero-padded is ``close()``."""
        if state.carry.size == 0:
            return
        padded = np.zeros(STT_BLOCK_SAMPLES, dtype=np.float32)
        padded[: state.carry.size] = state.carry
        state.carry = np.zeros(0, dtype=np.float32)
        yield self._frame(state, padded, float(self._device.captured_ms()))

    def _frame(
        self, state: _SttState, samples: npt.NDArray[np.float32], yielded_ms: float
    ) -> SttFrame:
        t_ms = state.frame_t_ms
        state.frame_t_ms += STT_BLOCK_MS
        return SttFrame(
            samples=np.asarray(samples, dtype=np.float32),
            t_ms=t_ms,
            end_ms=t_ms + STT_BLOCK_MS - 1000.0 / STT_RATE,
            yielded_ms=yielded_ms,
        )
