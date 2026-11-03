"""Conductor-owned playback: 48 kHz, 10 ms blocks, and the played clock (T-0006, docs/06 R3).

Barge-in correctness depends on ``flush()`` returning the audio actually *rendered*, so history
is truncated to what the user heard (the docs/02 section 4 invariant "the history never contains
assistant text that was not played"). Everything in this module is plumbing around one number.

**played_ms.** ``played_ms = clamp(device.rendered_ms() - rendered_at_run_start - paused_ms, 0,
enqueued_ms)``, where the run starts when the queue goes non-empty from idle (criterion 2). Two
properties of the device seam make that exact here:

* ``AudioDevice.rendered_ms()`` is a position *inside the stream this module wrote* (see
  ``audio_device.py``), so frames the device emitted while starved never count;
* the only thing ever written is the run's stream, in order, with no silence padding: ``pause()``
  stops writing rather than pushing zeros. The run's output is therefore the run's stream prefix,
  which collapses ``paused_ms`` to zero and leaves ``rendered - base`` as the played position.
  The ``enqueued_ms`` clamp catches the one exception, the sub-block zero padding of a final or
  faded block.

**Pause acts on the device.** The queue handed to the device is kept two blocks deep, so
``pause()`` and ``flush()`` only have to fade one block into silence to stop the sound inside the
50 ms criterion 3 allows -- not the 100+ ms an implementation that merely stops its producer
would leave playing. The fade is a raised cosine over one 10 ms block: no consecutive-sample step
above 0.1 full scale, and the samples it attenuates are consumed, never repeated on ``resume()``.

**Drain is declared, never inferred.** ``PlaybackDrained`` is emitted only after ``mark_drain()``
*and* the last sample has been rendered; a queue that empties because the TTS stalled is an
underrun and is counted as one (criterion 7). Without this, the turn machine's
``SPEAKING + playback.drained and llm.done -> IDLE`` row ends the turn mid-sentence.

**Threading (criterion 13, .claude/rules/conductor.md).** Three actors: the asyncio task that
pulls TTS chunks and resamples them (``_feed_loop``), a dedicated writer thread that hands 10 ms
blocks to the device (``_writer_loop``, no event loop anywhere in it), and the device's own
thread. They meet on one lock held for microseconds and on a bounded event ring that
``playback_events()`` drains; nothing here blocks the event loop.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from local_astra.conductor.audio_capture import (
    CaptureBlock,
    CaptureStats,
    CaptureStream,
    SelfSpeechGate,
    SttFrame,
)
from local_astra.conductor.audio_device import DEVICE_RATE, AudioDevice, open_device
from local_astra.conductor.audio_resample import Resampler24kTo48k, s16le_to_f32
from local_astra.contracts.events import PlaybackDrained, PlaybackStopped
from local_astra.contracts.protocols import TtsUtterance
from local_astra.contracts.speech import CAPTURE_BLOCK_MS, CAPTURE_FORMAT, WordTiming

#: A word counts as played if it started at least this long before the cut (docs/02 section 4).
GUARD_MS = 100.0
#: Fade applied by ``pause()`` and ``flush()``: one block, the top of criterion 3's 2-10 ms range.
FADE_BLOCKS = 1
#: Far-end reference kept for the capture gate and any future AEC (criterion 10). Bounded.
REFERENCE_RING_MS = 2000.0
#: Terminal events kept for a slow consumer (criterion 8: at least 8, newest kept, drops counted).
EVENT_SLOTS = 8
#: How long the writer thread lives with nothing to do before it exits (it is restarted by the
#: next ``play()``/``resume()``/``mark_drain()``; a leaked instance must not spin forever). It is
#: load-bearing and deliberately unchanged by T-0057: the drain and underrun rows need the writer
#: to still be there when the device clock moves later in the row. The *signalled* exit is
#: ``close()``; no caller, no exit.
WRITER_IDLE_EXIT_S = 2.0
#: How long ``close()`` waits for the writer to notice the signal. It normally returns within one
#: poll; the bound is T-0057 criterion 3(a)'s 100 ms, so a teardown is never held longer than the
#: budget and a thread that somehow outlives it is reported by the leak guard rather than hidden.
WRITER_JOIN_S = 0.1
#: Writer poll while there is something to do, and while there is not. The device is topped up
#: as it drains, so this is the granularity of the playback queue, not of the event loop.
POLL_ACTIVE_S = 0.0002
POLL_IDLE_S = 0.002
#: The feeder task exits after this long with no utterance; ``play()`` starts it again.
FEEDER_IDLE_EXIT_S = 5.0


def _fade_envelope(samples: int) -> npt.NDArray[np.float32]:
    """Raised cosine from 1 to exactly 0: continuous at both ends, so neither end clicks."""
    ramp = np.arange(samples, dtype=np.float64) / max(1, samples - 1)
    return np.asarray(0.5 * (1.0 + np.cos(np.pi * ramp)), dtype=np.float32)


def _word_prefix(source: str, words: int) -> str:
    """The first ``words`` whitespace-separated tokens of ``source``, spacing kept as written.

    A prefix of the committed answer, not a re-joined one: T-0003 criterion 10 makes the sources
    exact substrings of ``committed_answer`` and criterion 6 concatenates them with no separator.
    """
    if words <= 0:
        return ""
    seen, index, length = 0, 0, len(source)
    while index < length:
        while index < length and source[index].isspace():
            index += 1
        start = index
        while index < length and not source[index].isspace():
            index += 1
        if index > start:
            seen += 1
            if seen == words:
                return source[:index]
    return source


@dataclass
class _Utterance:
    """One ``play()`` call: what was queued, where it landed in the run, what it may contribute."""

    utterance: TtsUtterance
    source: str
    offset_ms: float = 0.0
    duration_ms: float = 0.0
    #: Set when the feeder reaches it, which is when its offset in the run is known.
    started: bool = False
    #: Set when its audio iterator is exhausted: only then can it be "fully played".
    complete: bool = False
    run_id: int = 0
    words: Sequence[WordTiming] | None = None
    from_tts_text: bool = False


@dataclass
class _Run:
    """One playback run: from an idle queue going non-empty to its terminal event."""

    run_id: int
    base_ms: float
    utterances: list[_Utterance] = field(default_factory=list)
    enqueued_ms: float = 0.0
    drain_marked: bool = False
    played_high_water_ms: float = 0.0


class PipeWireAudioIO:
    """``AudioIO`` over an injected ``AudioDevice`` (criterion 1: the device is never imported)."""

    #: Defaults of criterion 1; the instance takes both from the device it was given.
    block_ms = CAPTURE_BLOCK_MS
    fmt = CAPTURE_FORMAT

    def __init__(
        self, device: AudioDevice | None = None, *, gate: SelfSpeechGate | None = None
    ) -> None:
        self._device: AudioDevice = device if device is not None else open_device()
        self.block_ms = int(self._device.block_ms)
        #: The capture half (T-0021). One device object, so a capture block's ``t_ms`` and
        #: ``is_rendering(t_ms)`` are in one clock domain; and one ``is_rendering`` -- the gate
        #: reads this very instance, never a second view of the playback state (criterion 8).
        self._capture = CaptureStream(
            self._device,
            gate=gate if gate is not None else SelfSpeechGate(self.is_rendering),
            block_ms=self.block_ms,
        )
        self._block = round(self.block_ms * DEVICE_RATE / 1000)
        #: How much audio may sit in the device queue: one block. It is the part of "pause ->
        #: silence" and of the played clock's under-report that this module controls, so it is
        #: spent, not saved (criteria 2 and 3). The device's own buffer is the xrun cushion; the
        #: writer refills this one every 0.1 ms.
        self._target_lead_ms = float(self.block_ms)
        #: How much the device is holding that has not been rendered. A backend whose
        #: ``rendered_ms()`` already excludes its output latency (the real one does) must say so
        #: itself: ``out_ms - rendered_ms`` would then include that latency for ever and the
        #: writer would stop feeding it after one buffer.
        #: The backend counts the frames it rendered while starved and is the authority on
        #: xruns; ``rendered >= written`` cannot be, because a device that subtracts its output
        #: latency never gets there (ruling A11). ``underflow_frames`` is an ``AudioDevice``
        #: member since ruling A22, so there is no second path to fall back to (T-0021).
        self._underflow_frames_seen = 0
        self._fade = _fade_envelope(FADE_BLOCKS * self._block)

        self._lock = threading.RLock()
        self._resampler = Resampler24kTo48k()
        self._pending: deque[npt.NDArray[np.float32]] = deque()
        self._pending_samples = 0
        self._state = "idle"
        self._run: _Run | None = None
        self._runs = 0
        self._last_played_ms = 0
        self._out_ms = 0.0

        #: Counters the tests read (criteria 7 and 8).
        self.underruns = 0
        self.dropped_events = 0
        self._underrun_open = False

        self._events: deque[PlaybackStopped | PlaybackDrained] = deque()
        self._reference = np.zeros(round(REFERENCE_RING_MS * DEVICE_RATE / 1000.0), np.float32)
        self._reference_pos = 0
        self._reference_total = 0
        self._spans: deque[list[float]] = deque(maxlen=16)
        self._open_span: list[float] | None = None

        self._device_started = False
        self._writer: threading.Thread | None = None
        self._feeder: asyncio.Task[None] | None = None
        self._incoming: asyncio.Queue[_Utterance] | None = None
        #: The playback half's lifecycle (T-0057). ``_writer_stop`` is what the writer waits on
        #: between passes, so ``close()`` is noticed inside the poll rather than after it, and is
        #: sticky: a writer started after a ``close()`` would exit at the top of its loop anyway.
        #: ``_closed`` is therefore not there to stop a thread coming back (measured: it does not
        #: come back without it, ruling A13) but to stop a ``play()`` after the close **stranding
        #: a feeder task** -- one pending task without the guard, none with it.
        self._writer_stop = threading.Event()
        self._closed = False

    # -- what the device negotiated (criterion 1) -----------------------------------------
    @property
    def latency_ms(self) -> float:
        return float(self._device.latency_ms)

    @property
    def quantum_ms(self) -> float:
        """The quantum the graph gave us, asked of the device every time: it can change."""
        return float(self._device.quantum_ms)

    # -- AudioIO --------------------------------------------------------------------------
    async def play(self, utterance: TtsUtterance, *, source: str | None = None) -> None:
        """Queue an utterance. ``source`` is ``TtsSpeak.source``: what history may keep."""
        text = source if source is not None else str(getattr(utterance, "text", ""))
        with self._lock:
            if self._closed:
                # ``close()`` is terminal (the ``HudStaticServer.close()`` precedent): there is no
                # writer to render this, and ``_ensure_feeder()`` would leave a task waiting five
                # seconds on a queue nobody will ever drain -- the pending-task leak criterion
                # 3(b) is about, arriving through the one door close() does not guard.
                return
            if self._run is None:
                self._begin_run()
            run = self._run
            assert run is not None
            item = _Utterance(
                utterance=utterance,
                source=text,
                run_id=run.run_id,
                from_tts_text=source is None,
            )
            run.utterances.append(item)
        queue = self._ensure_feeder()
        queue.put_nowait(item)
        self._ensure_writer()

    async def pause(self) -> None:
        """Silence the device within one block and stop the played clock (criterion 3)."""
        with self._lock:
            if self._state != "playing":
                return
            self._state = "paused"
            self._cut_output(drop=False)
            # A pause with less than one block left is not an xrun, and neither is the silence
            # the device invents while paused: it is ours to withhold, not ours to have missed.
            self._underrun_open = False

    async def resume(self) -> None:
        """Continue from the sample the fade consumed: nothing is dropped, nothing repeated."""
        with self._lock:
            if self._state != "paused":
                return
            self._state = "playing"
        self._ensure_writer()

    async def flush(self) -> int:
        """Stop, drop what is queued, report the run total, emit exactly one PlaybackStopped."""
        with self._lock:
            if self._run is None:
                return self._last_played_ms
            if self._state == "playing":
                self._cut_output(drop=True)
            self._pending.clear()
            self._pending_samples = 0
            self._end_run(PlaybackStopped)
            return self._last_played_ms

    async def capture(self) -> AsyncIterator[bytes]:
        """``AudioIO.capture()``: 10 ms blocks of 48 kHz mono f32, 1920 bytes each (T-0021)."""
        async for block in self._capture.bytes_blocks():
            yield block

    # -- the capture seams T-0007 drives (T-0021 ruling A1; the protocol is not widened) ---
    async def capture_frames(self) -> AsyncIterator[CaptureBlock]:
        """The raw stream with its timestamps and marks. Never gated, never delayed."""
        async for block in self._capture.blocks():
            yield block

    async def stt_blocks(self) -> AsyncIterator[SttFrame]:
        """The STT tap: 24 kHz, 1920-sample frames, gated and xrun-filled (criteria 4, 7, 8)."""
        async for frame in self._capture.stt_frames():
            yield frame

    def capture_stats(self) -> CaptureStats:
        """Criterion 10's snapshot. This module reports numbers; ``session.py`` writes the row."""
        return self._capture.stats()

    def close_capture(self) -> None:
        """End of capture: the iterators finish and the STT tail is padded once (criterion 7)."""
        self._capture.close()

    # -- the playback half's lifecycle (T-0057) -------------------------------------------
    def close(self) -> None:
        """End of playback: the writer thread exits and is joined, the feeder task is cancelled.

        Until T-0057 the writer's only exit was ``WRITER_IDLE_EXIT_S`` of idleness and nothing
        joined it, so every caller that played a tone and went away left a thread reading the
        device ~4 000 times a second for two more seconds (docs/tasks/T-0057.md §Regime).

        Three things this is *not*, each of them a way to get the loop to exit that would be a
        behaviour change rather than a lifecycle one:

        * it is not ``flush()``. No run is ended, no ``PlaybackStopped`` and no
          ``PlaybackDrained`` is emitted, and ``played_ms`` does not move: a drain still needs
          ``mark_drain()`` *and* the last sample (docs/02 §4, T-0006 spec delta 1). A seam that
          emitted a terminal event would fire the turn machine's ``SPEAKING -> IDLE`` edge from a
          teardown;
        * it is not ``close_capture()``. The two halves share one device object for one clock
          domain, not one lifetime; the capture seam stays T-0023's;
        * it does not stop the device. Whoever opened it closes it (T-0040 owns shutdown), and
          stopping it here would move the rendered position this module reports;
        What makes stopping *mid-playback* safe is that the writer notices ``_writer_stop`` inside
        ``Event.wait``, which it does **outside** ``self._lock`` and never re-acquires, so the
        join has nothing to wait for. The join is kept outside the lock as well, because the
        writer takes it every pass and opens the stream inside the thread (ruling A14) -- but that
        placement is the lock discipline here, not what rescues this call, and no row
        discriminates it (ruling A13). On hardware it is not free either way: ``PortAudioDevice``
        can block in ``write`` for up to two seconds with the lock held, so the 100 ms bound is a
        fake-device bound, and wiring this into the session on the event loop would need more than
        a call (ruling A11, the named follow-up).

        Terminal and idempotent: a second call finds nothing to join, and a ``play()`` after it is
        refused -- not because the thread would come back (it would not; the stop flag is sticky)
        but because it would strand a feeder task. Synchronous, and callable with no running event
        loop -- the first caller is a pytest teardown, by which time the row's loop is closed.
        """
        with self._lock:
            self._closed = True
            self._writer_stop.set()
            writer, self._writer = self._writer, None
            feeder, self._feeder = self._feeder, None
        if writer is not None:
            writer.join(WRITER_JOIN_S)
        if feeder is not None and not feeder.done():
            self._cancel_feeder(feeder)

    def _cancel_feeder(self, feeder: asyncio.Task[None]) -> None:
        """Cancel the feeder task from wherever ``close()`` was called.

        ``_feed_loop`` exits on its own only after ``FEEDER_IDLE_EXIT_S`` with an empty queue, and
        a pending task is invisible to a thread guard: it surfaces its exception at collection
        time, attached to whatever is running then. Three callers exist and they need three
        different things -- the task's own loop (cancel directly, so the task is finished by the
        next scheduling point), another thread while that loop still turns (cancelling a task from
        off its loop is not thread-safe), and a teardown after the loop has been closed, where
        nothing can be scheduled and nothing will ever run again either.
        """
        loop = feeder.get_loop()
        if loop.is_closed():
            return
        try:
            running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:  # the documented way to ask: no loop is running in this thread
            running = None
        if running is loop:
            feeder.cancel()
        else:
            loop.call_soon_threadsafe(feeder.cancel)

    # -- the seams T-0007 drives (ruling A1) ----------------------------------------------
    def mark_drain(self) -> None:
        """Declare end of stream: no further audio for this run (criterion 7)."""
        with self._lock:
            if self._run is None:
                return
            self._run.drain_marked = True
        self._ensure_writer()

    def played_ms(self) -> float:
        with self._lock:
            return self._played_ms()

    async def playback_events(self) -> AsyncIterator[PlaybackStopped | PlaybackDrained]:
        """Terminal playback events, in emission order, for a single consumer (criterion 8)."""
        while True:
            with self._lock:
                batch = list(self._events)
                self._events.clear()
            for event in batch:
                yield event
            if not batch:
                await asyncio.sleep(POLL_ACTIVE_S)

    def playback_reference(self, window_ms: float) -> tuple[npt.NDArray[np.float32], float]:
        """The last ``window_ms`` of mixed output actually sent to the device (criterion 10)."""
        with self._lock:
            size = self._reference.size
            available = min(self._reference_total, size)
            count = min(available, max(0, round(window_ms * DEVICE_RATE / 1000.0)))
            start = (self._reference_pos - count) % size if size else 0
            first = min(count, size - start)
            window = np.concatenate(
                [self._reference[start : start + first], self._reference[: count - first]]
            )
            first_sample_ms = 1000.0 * (self._reference_total - count) / DEVICE_RATE
        return np.asarray(window, dtype=np.float32), first_sample_ms

    def is_rendering(self, t_ms: float) -> bool:
        """Was audio of ours going to the device at device time ``t_ms`` (criterion 10)?"""
        with self._lock:
            spans = list(self._spans)
            if self._open_span is not None:
                spans.append(self._open_span)
        # Half-open, like ``tests/fakes/audio.py``'s half of this seam: ``end`` is the instant
        # *after* the last rendered sample, so a capture block that starts exactly there caught
        # no audio of ours. With the closed form the gate's hangover was anchored one block late
        # (T-0021 ruling A16c pushes the anchor to the block's end, and the two together would
        # hold the gate shut a block longer than the hangover the docstring names).
        return any(start <= t_ms < end for start, end in spans)

    # -- the run ---------------------------------------------------------------------------
    def _begin_run(self) -> None:
        """A run starts when the queue goes non-empty from idle (criterion 2's ``run start``)."""
        self._runs += 1
        self._run = _Run(run_id=self._runs, base_ms=self._out_ms)
        self._state = "playing"
        self._resampler.reset()
        self._pending.clear()
        self._pending_samples = 0
        self._underrun_open = False
        # Frames the device invented between runs are not this run's xruns.
        self._underflow_frames_seen = int(self._device.underflow_frames)

    def _end_run(
        self, kind: type[PlaybackStopped] | type[PlaybackDrained], *, played: float | None = None
    ) -> None:
        run = self._run
        if run is None:
            return
        played = self._played_ms() if played is None else played
        # Truncated, not rounded: half a millisecond of over-reporting is still over-reporting.
        self._last_played_ms = int(played)
        text = self._played_text(run, played)
        self._run = None
        self._state = "idle"
        if self._open_span is not None:
            self._spans.append(self._open_span)
            self._open_span = None
        self._emit(
            kind(
                t_ms=int(max(0.0, self._device.rendered_ms())),
                played_ms=self._last_played_ms,
                played_text=text,
            )
        )

    def _played_ms(self) -> float:
        run = self._run
        if run is None:
            return float(self._last_played_ms)
        value = self._device.rendered_ms() - run.base_ms
        value = min(max(value, 0.0), run.enqueued_ms)
        run.played_high_water_ms = max(run.played_high_water_ms, value)
        return run.played_high_water_ms

    def _played_text(self, run: _Run, played: float) -> str:
        """Criterion 6: whole sources for what finished, a word prefix for what did not."""
        parts: list[str] = []
        for item in run.utterances:
            if not item.started:
                break
            if item.complete and played >= item.offset_ms + item.duration_ms - 0.5:
                parts.append(item.source)
                continue
            if played > item.offset_ms:
                parts.append(self._partial_text(item, played))
            break
        return "".join(parts)

    def _partial_text(self, item: _Utterance, played: float) -> str:
        """A word counts as played iff it started at least ``GUARD_MS`` before the cut.

        The normalisation ruling of criterion 6: if the source and the rendering do not have the
        same number of words there is no word-to-source mapping, so a partially played utterance
        contributes nothing at all ("3.5 apples" rendered as four words must never yield "3.5").
        """
        words = item.words
        tokens = item.source.split()
        if words is None or len(tokens) != len(words):
            return ""
        played_words = 0
        for word in words:
            if item.offset_ms + word.start_ms + GUARD_MS > played:
                break
            played_words += 1
        return _word_prefix(item.source, played_words)

    def _all_complete(self, run: _Run) -> bool:
        return all(item.complete for item in run.utterances)

    def _emit(self, event: PlaybackStopped | PlaybackDrained) -> None:
        self._events.append(event)
        while len(self._events) > EVENT_SLOTS:
            self._events.popleft()
            self.dropped_events += 1

    # -- the output path -------------------------------------------------------------------
    def _take(self, count: int, *, pad: bool = False) -> npt.NDArray[np.float32] | None:
        """One block of the stream, or None when there is not a whole one (unless ``pad``)."""
        if self._pending_samples == 0 or (self._pending_samples < count and not pad):
            return None
        block = np.zeros(count, dtype=np.float32)
        filled = 0
        while filled < count and self._pending:
            head = self._pending[0]
            take = min(count - filled, head.size)
            block[filled : filled + take] = head[:take]
            if take == head.size:
                self._pending.popleft()
            else:
                self._pending[0] = head[take:]
            filled += take
        self._pending_samples -= filled
        return block

    def _write_block(self, block: npt.NDArray[np.float32]) -> None:
        # When this block will be *heard*, on the capture clock (criterion 1, ruling A10). The
        # device plays what it is already holding first *and* holds the output latency after
        # that, and both are asked of the device rather than assumed: ``_out_ms - rendered_ms()``
        # is the whole un-rendered lead, queue plus DAC lag, because ``rendered_ms()`` already
        # subtracts that lag (audio handed to the DAC has not been heard yet). Measured, not the
        # configured ``latency_ms``: PortAudio re-measures the lag per callback (15.3 ms against
        # a negotiated 10 ms, T-0006 ruling A16) and another client can move the graph quantum.
        #
        # The ADC-side lag is deliberately *not* added on top: ``captured_ms()`` and the block
        # timestamps are ``inputBufferAdcTime``, which is already the instant the ADC took the
        # sample (ruling A4), so adding ``input_latency_ms`` would count the input path twice.
        # Residual to state beside the criterion-9 number: what is left unmodelled is the path
        # from the DAC to the ADC itself (the monitor link, or air), which is the echo-path
        # delay T-0023 measures and the hangover covers at the trailing edge.
        starts_ms = float(self._device.captured_ms()) + max(
            0.0, self._out_ms - float(self._device.rendered_ms())
        )
        self._device.write(np.asarray(block, dtype=np.float32))
        duration_ms = 1000.0 * block.size / DEVICE_RATE
        self._out_ms += duration_ms
        # Recording the span in the capture domain is what makes ``is_rendering(block.t_ms)``
        # answerable for a capture block without a wall-clock subtraction. With capture stopped
        # ``captured_ms()`` is 0 and the span degrades to "the un-rendered lead plus the sum of
        # the durations written since the span opened", which restarts near zero for every run
        # and is *not* the written-stream position T-0006 recorded (ruling A18): nothing reads
        # it that way, and only the capture path asks ``is_rendering`` at all.
        span = self._open_span
        if span is None or starts_ms > span[1] + self.block_ms:
            # A gap in the writing (a pause, a stalled TTS) ends the span: rendering did stop.
            if span is not None:
                self._spans.append(span)
            span = self._open_span = [starts_ms, starts_ms]
        span[1] = max(span[1], starts_ms) + duration_ms
        size = self._reference.size
        start = self._reference_pos % size
        first = min(block.size, size - start)
        self._reference[start : start + first] = block[:first]
        self._reference[: block.size - first] = block[first:]
        self._reference_pos = (start + block.size) % size
        self._reference_total += block.size

    def _cut_output(self, *, drop: bool) -> None:
        """Fade one block into silence so the device goes quiet without a click (criterion 3)."""
        block = self._take(self._block, pad=True)
        if block is not None:
            self._write_block(np.asarray(block * self._fade, dtype=np.float32))
        if drop:
            self._pending.clear()
            self._pending_samples = 0

    # -- the writer thread: no event loop below this line (criterion 13) -------------------
    def _ensure_writer(self) -> None:
        with self._lock:
            if self._closed or self._writer is not None:
                return
            self._writer = threading.Thread(
                target=self._writer_loop, name="astra-audio-writer", daemon=True
            )
            self._writer.start()

    def _writer_loop(self) -> None:
        """Top the device up block by block. Runs off the loop, holds the lock in microseconds."""
        idle_since = time.monotonic()
        while not self._writer_stop.is_set():
            # Opening the stream takes tens of milliseconds of PortAudio and ALSA, and ``play()``
            # takes the same lock on the event loop: doing it inside would hand the loop a stall
            # far beyond the 20 ms .claude/rules/conductor.md allows (ruling A14). Only this
            # thread touches ``_device_started``.
            if not self._device_started:
                self._device.start()
                self._device_started = True
            with self._lock:
                busy = self._writer_step()
                if busy:
                    idle_since = time.monotonic()
                elif time.monotonic() - idle_since > WRITER_IDLE_EXIT_S:
                    self._writer = None
                    return
                active = self._run is not None
            # Waited on the stop flag rather than slept: ``close()`` has to be noticed *inside*
            # the poll. With ``time.sleep`` the join would be bounded by the poll instead of by
            # the signal, and the caller is a teardown.
            if self._writer_stop.wait(POLL_ACTIVE_S if active else POLL_IDLE_S):
                return

    def _writer_step(self) -> bool:
        """One pass under the lock: keep the device fed, count xruns, notice the drain."""
        run = self._run
        # Read the device's xrun counter on every pass, whatever the state: frames it invented
        # while paused, idle or between runs are consumed here so they cannot be counted against
        # the next thing that plays.
        invented = self._consume_underflow()
        busy = False
        if run is not None and self._state == "playing":
            tail = run.drain_marked and self._all_complete(run)
            while float(self._device.queued_ms()) < self._target_lead_ms:
                block = self._take(self._block, pad=tail)
                if block is None:
                    break
                self._write_block(block)
                busy = True
            starved = (
                invented > 0
                and self._pending_samples < self._block
                and self._out_ms > run.base_ms
                and not tail
            )
            if starved and not self._underrun_open:
                self.underruns += 1
            self._underrun_open = starved
        if run is not None and run.drain_marked and self._is_drained(run):
            # A drain is the end of a *complete* stream: everything enqueued was handed to the
            # device and nothing cancels it, so the run total is the whole stream. ``flush()``
            # keeps the conservative device-clock number -- that is the cut the turn machine
            # writes into history, and the one the section 4 invariant is about.
            self._end_run(PlaybackDrained, played=run.enqueued_ms)
            busy = True
        return busy

    def _consume_underflow(self) -> int:
        """Frames the device invented since the last pass, per its own counter (ruling A11).

        A backend that counts what it had to put out while starved is the authority on xruns;
        ``rendered >= written`` cannot be, because a device that subtracts its output latency
        never gets there -- which is why ruling A22 put ``underflow_frames`` on the protocol and
        deleted the fallback that used to stand here. The answer is only turned into an
        ``underruns`` increment while a run is playing: a device fed nothing between turns is
        idle, not starved.
        """
        frames = int(self._device.underflow_frames)
        missing, self._underflow_frames_seen = frames - self._underflow_frames_seen, frames
        return max(0, missing)

    def _is_drained(self, run: _Run) -> bool:
        """Criterion 7: declared end of stream *and* the audio out. Never inferred.

        Three conjuncts, none of them spare:

        * ``_pending_samples == 0`` -- nothing of ours is still waiting for the writer. Without
          it, ``mark_drain()`` followed by ``pause()`` would drain with ``played_ms =
          enqueued_ms`` and write a whole utterance into history after a tenth of it was heard.
        * ``_all_complete(run)`` -- every utterance's audio iterator is exhausted. A TTS stall
          inside the last utterance after ``TtsDone`` is the normal ordering, and it is a stall,
          not an end of stream (ruling A17).
        * ``queued_ms() == 0`` -- the device has taken everything we wrote. This is the half that
          cannot be replaced by ``rendered_ms()``: a device with output latency subtracts its DAC
          lag, saturating below ``written``, and that lag is not a constant anyone may assume
          (``PortAudioDevice`` measures it per callback; another client can force the graph
          quantum). Asking the device is lag-independent by construction (ruling A15).

        What is outstanding when all three hold is inside the DAC and, since nothing flushes a
        drained run, certain to be heard. A run whose utterances yield no audio at all satisfies
        the three immediately and drains with ``played_ms = 0`` -- deliberately (ruling A21): an
        empty stream is still a finished one, and the turn machine's ``SPEAKING + drained -> IDLE``
        row would otherwise wait for an event that can never come.
        """
        return (
            self._pending_samples == 0
            and self._all_complete(run)
            and float(self._device.queued_ms()) <= 0.0
        )

    # -- the feeder task: TTS chunks in, 48 kHz blocks out ---------------------------------
    def _ensure_feeder(self) -> asyncio.Queue[_Utterance]:
        if self._incoming is None:
            self._incoming = asyncio.Queue()
        if self._feeder is None or self._feeder.done():
            self._feeder = asyncio.get_running_loop().create_task(self._feed_loop())
        return self._incoming

    async def _feed_loop(self) -> None:
        queue = self._incoming
        assert queue is not None
        while True:
            try:
                # ``asyncio.timeout`` rather than ``wait_for``: the task then waits on the queue's
                # own future instead of on a wrapper task, so ``close()``'s cancellation finishes
                # it at the next scheduling point rather than four hops later (measured 1 against
                # 4 on 3.11), and the pending-task leak has somewhere to be noticed. This is why
                # 3.12 reimplemented ``wait_for`` on top of it.
                async with asyncio.timeout(FEEDER_IDLE_EXIT_S):
                    item = await queue.get()
            except TimeoutError:
                # The two forms differ at the deadline boundary, and the difference is a hang
                # (ruling A10): an item enqueued in the same loop iteration as the deadline is
                # recovered by ``wait_for``, which re-checks the future after cancelling, and left
                # in the queue by ``asyncio.timeout``, which converts the cancellation at context
                # exit. Abandoning it strands the utterance in the run with ``complete=False``, so
                # ``_all_complete`` is false for ever and ``mark_drain()`` can never produce
                # ``PlaybackDrained``: the turn machine stays in SPEAKING (docs/02 section 4). The
                # idle exit is "the queue has been empty for FEEDER_IDLE_EXIT_S", so ask the queue.
                if not queue.empty():
                    continue
                return
            await self._feed(item)

    async def _feed(self, item: _Utterance) -> None:
        """Resample one utterance into the pending stream, in order, without holding the loop."""
        if not self._claim(item):
            await item.utterance.cancel()
            return
        stale = False
        chunks = item.utterance.audio()
        try:
            async for chunk in chunks:
                block = self._resampler.process(s16le_to_f32(chunk))
                stale = not self._append(item, block)
                if stale:
                    break
                self._ensure_writer()
        finally:
            # ``TtsUtterance.audio()`` is an AsyncIterator by contract; when the concrete one is
            # a generator, abandoning it mid-stream has to close it here and not at collection.
            aclose = getattr(chunks, "aclose", None)
            if aclose is not None:
                await aclose()
        if stale:
            await item.utterance.cancel()
            return
        with self._lock:
            item.complete = True
        self._ensure_writer()

    def _claim(self, item: _Utterance) -> bool:
        """Give the utterance its offset in the run, or report that its run is already over."""
        with self._lock:
            run = self._run
            if run is None or run.run_id != item.run_id:
                return False
            item.offset_ms = run.enqueued_ms
            item.words = item.utterance.words()
            item.started = True
            return True

    def _append(self, item: _Utterance, block: npt.NDArray[np.float32]) -> bool:
        with self._lock:
            run = self._run
            if run is None or run.run_id != item.run_id:
                return False
            self._pending.append(block)
            self._pending_samples += block.size
            added = 1000.0 * block.size / DEVICE_RATE
            item.duration_ms += added
            run.enqueued_ms += added
            return True
