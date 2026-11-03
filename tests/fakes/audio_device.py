"""Fake ``AudioDevice`` (T-0006 criterion 1) and the seams the audio tests share.

The unit tests for the playback path need three things that do not exist yet and must not be
faked *inside* a test module: a device double whose rendered position is controlled by the
virtual clock, a ``TtsUtterance`` carrying real PCM and hand-written word timings, and a couple
of signal-analysis oracles for the rig recordings.

**The device model, stated once.** ``FakeAudioDevice`` is a sink that renders in *virtual* time:

- ``write(block)`` appends a 10 ms block of 48 kHz f32 mono and records it (every block is kept,
  so the device's own memory of what it received is the monitor oracle for the unit tests);
- the device *takes* 1 ms of audio per elapsed ms while it is started and we have given it
  something to take; time it spends with nothing to take is counted in ``underflow_frames`` (the
  fake's equivalent of ``PortAudioDevice.underflow_frames``, T-0006 ruling A11) and is **not**
  credited to the stream;
- ``rendered_ms()`` is what the device took minus ``dac_lag_ms``: monotonic, never backwards,
  the "device's own rendered-frame position" of criterion 2. With the default ``dac_lag_ms=0`` it
  saturates at ``written_ms``; construct the device with a non-zero ``dac_lag_ms`` to model a
  real backend, whose rendered position saturates at ``written − lag`` because audio handed to
  the DAC has not left it yet (ruling A12) — a drain rule written as ``rendered >= written``
  never fires against such a device. **The lag may exceed ``latency_ms``** and on the hardware it
  does (ruling A16): ``PortAudioDevice`` measures the DAC lag per callback — 15.3 ms against a
  negotiated 10 ms — and another client can force the graph quantum to ≈ 21 ms, so any allowance
  computed from the *configured* latency is guesswork. That is why A15 makes the drain rule read
  ``queued_ms() == 0`` instead. It is deliberately *not* a function of what was written: a
  module reporting bytes handed to the buffer disagrees with it as soon as the queue runs ahead
  of the clock;
- ``queued_ms()`` is the flow-control seam (``written − taken``, the real backend's ``_fill``),
  and ``buffer_ms`` is how much of it the device tolerates — writes beyond it are accepted (a
  fake cannot block a test) but counted in ``overfills`` and visible in ``max_lead_ms``, so "keep
  the device queue shallow" is measurable;
- ``stall(ms)`` makes the device render ``ms`` of *something else* (an xrun): the clock advances,
  the rendered position does not.

**The capture model (T-0021 criterion 1), stated once.** The same device object also records:

- ``start_capture()``/``stop_capture()``/``captured_ms()`` — the ADC position integrates the
  virtual clock exactly like ``rendered_ms()`` does, so capture and playback share one clock base;
- delivery is by ``set_capture_callback(cb)`` **or** by polling ``read_capture()`` (the task
  accepts either), and both hand over ``(samples_f32, adc_ms, xrun)`` where ``adc_ms`` is the
  capture time of the buffer's *first* sample — the fake's ``inputBufferAdcTime``;
- ``capture_buffer_frames`` defaults to 1024 (the graph quantum on this box), which is not a
  multiple of a 10 ms block, so re-blocking is mandatory;
- ``stall_capture(ms)`` holds delivery and then bursts (no audio lost — criterion 3);
  ``xrun(ms)`` loses audio the ADC ran through (criterion 4). The two are not the same failure
  and the module must not treat them the same way.

Nothing here imports ``sounddevice`` (PortAudio is absent on this box) or touches a real device.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import importlib
import inspect
import json
import math
import threading
import time
import wave
import weakref
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from local_astra.contracts.speech import (
    CAPTURE_BLOCK_MS,
    STT_INPUT_FORMAT,
    TTS_OUTPUT_FORMAT,
    WordTiming,
)
from tests.conftest import VirtualClock

DEVICE_RATE = 48000
TTS_RATE = TTS_OUTPUT_FORMAT.sample_rate  # 24000
BLOCK_SAMPLES = DEVICE_RATE * CAPTURE_BLOCK_MS // 1000  # 480
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "audio"

#: Linear chirp parameters of ``chirp_5s_24k.wav`` (tests/fixtures/audio/MANIFEST.md). The
#: inverse ``t(f) = (f - f0) / k`` is the independent clock of the null-sink tests.
CHIRP_F0_HZ = 200.0
CHIRP_K_HZ_PER_S = 1160.0

# --- T-0021 (capture) --------------------------------------------------------------------

STT_RATE = STT_INPUT_FORMAT.sample_rate  # 24000
#: moshi-server 0.6.4 reads 1920 samples (80 ms) at a time (scripts/demo/_listen.py BLOCK).
STT_BLOCK_SAMPLES = 1920
STT_BLOCK_MS = 1000 * STT_BLOCK_SAMPLES // STT_RATE  # 80
#: 480 f32 = 1920 bytes: what ``capture()`` yields (T-0021 criterion 2).
CAPTURE_BLOCK_BYTES = 4 * BLOCK_SAMPLES
#: The graph default quantum on this box, and deliberately **not** a multiple of a 10 ms
#: block: a module that assumes the device hands it whole blocks re-blocks wrong.
DEFAULT_CAPTURE_BUFFER_FRAMES = 1024
WHY_CAPTURE = "T-0021: the capture path (owned file local_astra/conductor/audio_capture.py; docs/tasks/T-0021.md)"


def capture_pattern(start: int, n: int) -> np.ndarray:
    """The device's input signal as a pure function of the *absolute* sample index.

    Two sawtooths with coprime periods (8191 and 8192 samples), so the pattern repeats only
    after ~23 minutes: every sample of a capture run is identifiable by its index. "No sample
    is lost or duplicated" is then checked by comparing what the module yielded with
    ``capture_pattern(0, n)`` — a drop or a repeat shifts everything after it and the
    comparison fails. Element-wise, so the value at an index does not depend on the chunking.
    """
    idx = np.arange(start, start + n, dtype=np.int64)
    a = (idx % 8191).astype(np.float64) / 8191.0 - 0.5
    b = (idx % 8192).astype(np.float64) / 8192.0 - 0.5
    return (0.6 * a + 0.3 * b).astype(np.float32)


# --- the device double -------------------------------------------------------------------------


class FakeAudioDevice:
    """Virtual-clock ``AudioDevice``. See the module docstring for the rendering model."""

    def __init__(
        self,
        clock: VirtualClock | None = None,
        *,
        buffer_ms: float = 200.0,
        latency_ms: float = 20.0,
        dac_lag_ms: float = 0.0,
        block_ms: int = CAPTURE_BLOCK_MS,
        quantum_ms: float = float(CAPTURE_BLOCK_MS),
        sample_rate: int = DEVICE_RATE,
        channels: int = 1,
        input_latency_ms: float = 15.0,
        capture_buffer_frames: int = DEFAULT_CAPTURE_BUFFER_FRAMES,
        capture_source: Callable[[int, int], np.ndarray] = capture_pattern,
    ) -> None:
        self.clock = clock if clock is not None else VirtualClock()
        self.buffer_ms = buffer_ms
        self.latency_ms = latency_ms
        self.dac_lag_ms = dac_lag_ms
        self.block_ms = block_ms
        self.quantum_ms = quantum_ms
        self.sample_rate = sample_rate
        self.channels = channels

        self.started = False
        self.starts = 0
        self.stops = 0
        self.write_calls = 0
        self.blocks: list[np.ndarray] = []
        #: every block that was not 10 ms of 48 kHz f32 mono, with the reason (criterion 1)
        self.block_violations: list[str] = []
        self.overfills = 0
        self.max_lead_ms = 0.0
        #: frames the device had to emit as silence because we handed it nothing — the fake's
        #: equivalent of ``PortAudioDevice.underflow_frames`` (T-0006 ruling A11).
        self.underflow_frames = 0

        self._lock = threading.RLock()
        self._pos_ms = 0.0
        self._taken_ms = 0.0
        self._last_t_ms = float(self.clock.now_ms)
        self._stall_remaining_ms = 0.0

        # -- the input side (T-0021 criterion 1); see "The capture model" in the docstring
        self.input_latency_ms = input_latency_ms
        self.capture_buffer_frames = capture_buffer_frames
        self.capture_source = capture_source
        self.capturing = False
        self.capture_starts = 0
        self.capture_stops = 0
        self.capture_reads = 0
        self.delivered_buffers = 0
        self.delivered_frames = 0
        self.capture_start_ms: float | None = None
        self._adc_frames = 0
        self._deliver_index = 0
        self._lost_frames = 0
        self._flag_xrun = False
        self._hold_until_ms: float | None = None
        self._cap_last_t_ms = float(self.clock.now_ms)
        self._pending_buffers: deque[tuple[np.ndarray, float, bool]] = deque()
        self._capture_callback: Callable[[np.ndarray, float, bool], None] | None = None

    # -- AudioDevice protocol ---------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            self._integrate()
            self.started = True
            self.starts += 1

    def stop(self) -> None:
        with self._lock:
            self._integrate()
            self.started = False
            self.stops += 1

    def write(self, block: Any) -> None:
        with self._lock:
            self._integrate()
            self.write_calls += 1
            arr = np.asarray(block)
            if arr.dtype != np.float32:
                self.block_violations.append(f"dtype {arr.dtype} (want float32)")
            if arr.ndim != 1:
                self.block_violations.append(f"shape {arr.shape} (want mono 1-D)")
            if arr.size != BLOCK_SAMPLES:
                self.block_violations.append(
                    f"{arr.size} samples = {1000 * arr.size / self.sample_rate:.3f} ms "
                    f"(want {BLOCK_SAMPLES} = {self.block_ms} ms at {self.sample_rate} Hz)"
                )
            self.blocks.append(np.asarray(arr, dtype=np.float64).reshape(-1))
            lead = self.written_ms - self._taken_ms  # = queued_ms(), the real backend's ``_fill``
            self.max_lead_ms = max(self.max_lead_ms, lead)
            if lead > self.buffer_ms:
                self.overfills += 1

    def rendered_ms(self) -> float:
        with self._lock:
            self._integrate()
            return self._pos_ms

    def queued_ms(self) -> float:
        """Audio handed over that the device has not consumed yet (``PortAudioDevice._fill``).

        The flow-control seam, kept separate from ``rendered_ms()`` on purpose: under a DAC lag
        ``written - rendered`` never falls below ``dac_lag_ms``, so a writer that paced itself on
        that difference would stop feeding the device after one buffer.
        """
        with self._lock:
            self._integrate()
            return max(0.0, self.written_ms - self._taken_ms)

    # -- AudioDevice input side (T-0021 criterion 1) -----------------------------------------
    def start_capture(self) -> None:
        with self._lock:
            self._integrate_capture()
            self.capturing = True
            self.capture_starts += 1
            if self.capture_start_ms is None:
                self.capture_start_ms = float(self.clock.now_ms)

    def stop_capture(self) -> None:
        with self._lock:
            self._integrate_capture()
            self.capturing = False
            self.capture_stops += 1

    def captured_ms(self) -> float:
        """Device-reported ADC position in ms since ``start_capture()``: monotone, from the clock.

        Deliberately *not* a function of what was delivered: it keeps advancing through a
        delivery stall and through an xrun, which is what makes it the timestamp oracle.
        """
        with self._lock:
            self._integrate_capture()
            return 1000.0 * self._adc_frames / self.sample_rate

    def set_capture_callback(self, callback: Callable[[np.ndarray, float, bool], None]) -> None:
        """Delivery by callback: ``callback(samples_f32, adc_ms, xrun)`` off the event loop."""
        with self._lock:
            self._capture_callback = callback
            while self._pending_buffers:
                samples, adc_ms, xrun = self._pending_buffers.popleft()
                callback(samples, adc_ms, xrun)

    def read_capture(self) -> tuple[np.ndarray, float, bool] | None:
        """Delivery by bounded read: one buffer ``(samples_f32, adc_ms, xrun)``, or None.

        ``adc_ms`` is the capture-clock time of the buffer's **first** sample — the fake's
        ``time_info.inputBufferAdcTime``. Both delivery styles are offered because T-0021
        criterion 1 accepts either ("a delivery callback or bounded read").
        """
        with self._lock:
            self.capture_reads += 1
            if not self._pending_buffers:
                return None
            return self._pending_buffers.popleft()

    # -- test-side capture controls and oracles ----------------------------------------------
    def pump(self) -> None:
        """Hand over every buffer the device has ready. ``capture_out()`` calls this."""
        with self._lock:
            self._integrate_capture()
            if self._hold_until_ms is not None:
                if float(self.clock.now_ms) < self._hold_until_ms:
                    return
                self._hold_until_ms = None
            n = self.capture_buffer_frames
            while True:
                if self._lost_frames:
                    self._deliver_index += self._lost_frames
                    self._lost_frames = 0
                    self._flag_xrun = True
                if self._adc_frames - self._deliver_index < n:
                    return
                samples = np.asarray(self.capture_source(self._deliver_index, n), dtype=np.float32)
                adc_ms = 1000.0 * self._deliver_index / self.sample_rate
                xrun, self._flag_xrun = self._flag_xrun, False
                self._deliver_index += n
                self.delivered_buffers += 1
                self.delivered_frames += n
                if self._capture_callback is not None:
                    self._capture_callback(samples, adc_ms, xrun)
                else:
                    self._pending_buffers.append((samples, adc_ms, xrun))

    def stall_capture(self, ms: float) -> None:
        """Deliver nothing for ``ms`` of virtual time, then all of it at once (criterion 3).

        The ADC keeps running: no audio is lost, only its delivery is late. An implementation
        that stamps blocks when the callback ran cannot tell this apart from silence.
        """
        with self._lock:
            self._hold_until_ms = float(self.clock.now_ms) + float(ms)

    def xrun(self, ms: float) -> None:
        """``ms`` of audio never arrives (criterion 4): the ADC runs on, those frames are lost."""
        with self._lock:
            self._integrate_capture()
            self._lost_frames += round(float(ms) * self.sample_rate / 1000.0)

    def source_audio(self, n: int, start: int = 0) -> np.ndarray:
        """The audio the ADC produced for absolute sample indices ``[start, start + n)``."""
        return np.asarray(self.capture_source(start, n), dtype=np.float32)

    def true_capture_ms(self, index: int) -> float:
        """When sample ``index`` of the capture stream was taken, on the virtual clock.

        The independent oracle for criterion 3: it comes from the clock and the sample rate,
        never from anything the module reports.
        """
        base = 0.0 if self.capture_start_ms is None else self.capture_start_ms
        return base + 1000.0 * index / self.sample_rate

    def _integrate_capture(self) -> None:
        now = float(self.clock.now_ms)
        dt = max(0.0, now - self._cap_last_t_ms)
        self._cap_last_t_ms = now
        if self.capturing:
            self._adc_frames += round(dt * self.sample_rate / 1000.0)

    # -- test-side controls and oracles -----------------------------------------------------
    @property
    def written_ms(self) -> float:
        return 1000.0 * sum(b.size for b in self.blocks) / self.sample_rate

    def stall(self, ms: float) -> None:
        """Render ``ms`` of something else (an xrun): time passes, the stream does not advance."""
        with self._lock:
            self._integrate()
            self._stall_remaining_ms += ms

    def written_audio(self) -> np.ndarray:
        with self._lock:
            if not self.blocks:
                return np.zeros(0, dtype=np.float64)
            return np.concatenate(self.blocks)

    def rendered_audio(self) -> np.ndarray:
        """The prefix of the written stream the device has actually put out (the monitor)."""
        with self._lock:
            n = round(self.rendered_ms() * self.sample_rate / 1000.0)
            return self.written_audio()[:n]

    def last_nonzero_rendered_ms(self, threshold: float = 1e-4) -> float | None:
        x = self.rendered_audio()
        nz = np.flatnonzero(np.abs(x) > threshold)
        if nz.size == 0:
            return None
        return 1000.0 * (int(nz[-1]) + 1) / self.sample_rate

    def last_nonzero_written_ms(self, threshold: float = 1e-4) -> float | None:
        x = self.written_audio()
        nz = np.flatnonzero(np.abs(x) > threshold)
        if nz.size == 0:
            return None
        return 1000.0 * (int(nz[-1]) + 1) / self.sample_rate

    def _integrate(self) -> None:
        """Consume time: take what we were given, count what we were not, then subtract the lag."""
        now = float(self.clock.now_ms)
        dt = max(0.0, now - self._last_t_ms)
        self._last_t_ms = now
        if not self.started:
            return
        if self._stall_remaining_ms > 0.0:
            used = min(dt, self._stall_remaining_ms)
            self._stall_remaining_ms -= used
            dt -= used
        if dt <= 0.0:
            return
        taken = min(dt, max(0.0, self.written_ms - self._taken_ms))
        self._taken_ms += taken
        starved = dt - taken
        if starved > 0.0:
            self.underflow_frames += round(starved * self.sample_rate / 1000.0)
        self._pos_ms = max(self._pos_ms, max(0.0, self._taken_ms - self.dac_lag_ms))


# --- seams: the module under test does not exist yet ---------------------------------------------


def import_or_fail(name: str, why: str) -> ModuleType:
    """Import a module T-0006 must create; a missing module is an assertion, not a collect error."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:  # pragma: no cover - the pending path
        raise AssertionError(f"{name} does not exist ({why}): {exc}") from exc


def attr_or_fail(obj: object, name: str, why: str) -> Any:
    """Fetch a seam the task requires; absence is an AssertionError naming the criterion."""
    if not hasattr(obj, name):
        raise AssertionError(f"{obj!r} has no {name!r} ({why})")
    return getattr(obj, name)


def make_audio_io(device: FakeAudioDevice, **kwargs: Any) -> Any:
    """Build ``PipeWireAudioIO`` with the device injected (criterion 1: injection, default real)."""
    _install_audio_io_tracking()
    from local_astra.conductor.audio_io import PipeWireAudioIO

    try:
        return PipeWireAudioIO(device, **kwargs)
    except TypeError as exc:
        raise AssertionError(
            "T-0006 criterion 1: PipeWireAudioIO must take an AudioDevice by injection "
            f"(PipeWireAudioIO(device, **{kwargs!r}) raised {exc})"
        ) from exc


# --- T-0057: the playback module's lifecycle ----------------------------------------------------
#
# ``PipeWireAudioIO`` starts ``astra-audio-writer`` on the first ``play()``. Until T-0057 the
# thread's only exit was ``WRITER_IDLE_EXIT_S = 2.0`` s of idleness, so every row that played a
# tone handed the next row a live thread (measured on this tree: 7 960 reads of one device method
# in 1.99 s after the row ended). ``tests/conftest.py``'s ``astra_audio_modules_are_stopped``
# calls ``stop_tracked_audio_modules()`` in every row's teardown; the tracking below is what makes
# that reach the modules built *directly* by a row as well as the ones built here, which matters
# because ``tests/unit/test_null_audio.py`` and ``tests/unit/test_audio_measure_report.py``
# construct ``PipeWireAudioIO(device)`` themselves.
#
# The wrapper delegates: it calls the real ``__init__`` and records the instance, and knows
# nothing about its arguments (.claude/rules/tests.md, "a test double closes by delegation").

#: Names the stop seam may carry. T-0057 criterion 3 leaves the name to the implementer and names
#: ``close()`` (``HudStaticServer.close()``, T-0040) as the house precedent. This is a stated
#: contract the tests assume, *not* a ``dir()`` walk: nothing here is discovered by pattern and
#: then invoked (.claude/rules/tests.md). A seam under a different name is added to this tuple in
#: the same PR -- and if it is, say so in the review, because the list is the interface.
STOP_SEAM_NAMES = ("close", "stop", "close_playback", "stop_playback")

_TRACKED_AUDIO_IO: weakref.WeakSet[Any] = weakref.WeakSet()
_TRACKING_INSTALLED = False


def _install_audio_io_tracking() -> None:
    """Record every ``PipeWireAudioIO`` built in this process. Idempotent; never raises."""
    global _TRACKING_INSTALLED
    if _TRACKING_INSTALLED:
        return
    try:
        from local_astra.conductor.audio_io import PipeWireAudioIO
    except Exception:  # pragma: no cover - the module has to exist for any audio row to run
        return
    original = PipeWireAudioIO.__init__

    @functools.wraps(original)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        _TRACKED_AUDIO_IO.add(self)

    PipeWireAudioIO.__init__ = __init__  # type: ignore[method-assign]
    _TRACKING_INSTALLED = True


def audio_io_stop_seam(io: Any) -> Callable[[], Any] | None:
    """The bound stop seam of ``io``, or ``None`` while T-0057 has not added one.

    Looked up on the *class* and only among ``STOP_SEAM_NAMES``, and only accepted when it takes
    no argument beyond ``self``: a test that calls whatever it finds dictates the arity of every
    name that matches (.claude/rules/tests.md, the nine rows a private ``_refuse(self, source)``
    killed).
    """
    for name in STOP_SEAM_NAMES:
        fn = getattr(type(io), name, None)
        if fn is None or not callable(fn):
            continue
        params = list(inspect.signature(fn).parameters.values())[1:]
        required = [
            p
            for p in params
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if required:
            continue
        return getattr(io, name)  # type: ignore[no-any-return]
    return None


def stop_audio_module(io: Any) -> bool:
    """Call the stop seam. ``False`` means there is none yet (T-0057 criterion 3 unimplemented).

    Accepts a sync or an async seam. The teardown that calls this runs after pytest-asyncio has
    closed the row's loop, so an async seam is driven on a loop of its own -- state that as a
    constraint on the implementation rather than hiding it: a seam that needs the row's own loop
    to be still running cannot be called from a teardown at all.
    """
    seam = audio_io_stop_seam(io)
    if seam is None:
        return False
    result = seam()
    if inspect.isawaitable(result):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_drive(result))
        else:
            raise AssertionError(
                "the stop seam returned an awaitable while a loop is running; await it there"
            )
    return True


async def _drive(awaitable: Any) -> None:
    await awaitable


def stop_tracked_audio_modules() -> int:
    """Stop every playback module built since the last call; return how many were stopped."""
    stopped = 0
    for io in list(_TRACKED_AUDIO_IO):
        if stop_audio_module(io):
            stopped += 1
    _TRACKED_AUDIO_IO.clear()
    return stopped


_install_audio_io_tracking()


_UNSET = object()


async def maybe_await(value: Any) -> Any:
    """Accept a sync or async seam. ``played_ms()`` is sync in ``tests/fakes/audio.py`` today."""
    return await value if inspect.isawaitable(value) else value


async def play(io: Any, utterance: Any, *, source: Any = _UNSET) -> None:
    """``play(utterance, *, source: str | None = None)`` — T-0006 criterion 6."""
    fn = attr_or_fail(io, "play", "AudioIO protocol (contracts/protocols.py)")
    try:
        await maybe_await(fn(utterance) if source is _UNSET else fn(utterance, source=source))
    except TypeError as exc:
        raise AssertionError(
            f"T-0006 criterion 6: play(utterance, *, source=...) — call raised {exc}"
        ) from exc


async def played_ms(io: Any) -> float:
    value = await maybe_await(
        attr_or_fail(io, "played_ms", "T-0006 criterion 2: the device-side played clock")()
    )
    assert isinstance(value, int | float), f"played_ms() returned {value!r}"
    return float(value)


async def call(io: Any, name: str, why: str, *args: Any) -> Any:
    return await maybe_await(attr_or_fail(io, name, why)(*args))


class EventSink:
    """Collects ``playback_events()`` (criterion 8) in the background.

    ``arm()`` creates the iterator without consuming it — that is the "slow consumer" of
    criterion 8 — and ``start()`` begins draining it into ``events``.
    """

    WHY = "T-0006 criterion 8: playback_events() yields PlaybackStopped/PlaybackDrained"

    def __init__(self, io: Any, *, autostart: bool = True) -> None:
        self.io = io
        self.events: list[Any] = []
        self._it: Any = None
        self._task: asyncio.Task[None] | None = None
        if autostart:
            self.arm()
            self.start()

    def arm(self) -> None:
        if self._it is None:
            self._it = attr_or_fail(self.io, "playback_events", self.WHY)()

    def start(self) -> None:
        self.arm()
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        async for event in self._it:
            self.events.append(event)

    async def sync(self, rounds: int = 8) -> None:
        for _ in range(rounds):
            await asyncio.sleep(0.0005)

    def of_type(self, *types: type) -> list[Any]:
        return [e for e in self.events if isinstance(e, types)]

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(BaseException):
                await self._task


async def settle(device: FakeAudioDevice, *, quiet_rounds: int = 4, timeout_s: float = 2.0) -> None:
    """Give the module's writer (thread *or* task, criterion 13) room to top the device up.

    Returns once ``write()`` has been quiet for ``quiet_rounds`` scheduler turns. Real time only;
    the *device's* clock is virtual and moves only when a test says so.
    """
    deadline = time.monotonic() + timeout_s
    last = device.write_calls
    quiet = 0
    while time.monotonic() < deadline:
        await asyncio.sleep(0.0002)
        if device.write_calls != last:
            last = device.write_calls
            quiet = 0
        elif (quiet := quiet + 1) >= quiet_rounds:
            return


#: Never advance the virtual clock by more than one block at a time: a module that keeps the
#: device queue shallow (T-0006 implementation notes: 2-3 blocks) starves if the clock jumps
#: further than its lead, and the device would render silence the test never asked for.
MAX_STEP_MS = 10.0


async def play_out(device: FakeAudioDevice, ms: float, *, step_ms: float = MAX_STEP_MS) -> None:
    """Advance the device's virtual clock by ``ms``, letting the writer keep the device fed."""
    left = float(ms)
    await settle(device)
    while left > 0:
        step = min(step_ms, MAX_STEP_MS, left)
        device.clock.advance(int(step))
        left -= step
        await settle(device)


# --- driving the capture side (T-0021) ----------------------------------------------------------


def import_capture() -> ModuleType:
    """``local_astra.conductor.audio_capture`` — T-0021's owned file."""
    return import_or_fail("local_astra.conductor.audio_capture", WHY_CAPTURE)


class CaptureSink:
    """Drains one of the three capture iterators in the background, keeping everything.

    ``kind``: ``"raw"`` -> ``capture_frames()`` (``CaptureBlock``), ``"bytes"`` -> ``capture()``
    (the ``AudioIO`` protocol method, 1920-byte blocks), ``"stt"`` -> ``stt_blocks()``. All three
    are drained the way ``session.py`` will (T-0021 "Seam to session.py"): concurrently, by a
    consumer that is not the device thread. An exception raised by the iterator is kept in
    ``error`` instead of killing the test task — criterion 4's named error is checked there.
    """

    ITERATORS = {"raw": "capture_frames", "bytes": "capture", "stt": "stt_blocks"}

    def __init__(self, io: Any, kind: str = "raw", *, autostart: bool = True) -> None:
        self.io = io
        self.kind = kind
        self.items: list[Any] = []
        self.error: BaseException | None = None
        self.finished = False
        self._it: Any = None
        self._task: asyncio.Task[None] | None = None
        self._gate = asyncio.Event()
        self._gate.set()
        if autostart:
            self.arm()
            self.start()

    def hold(self) -> None:
        """Stop reading (criterion 5's slow consumer) without cancelling the iterator.

        The next item is not even requested while held, so everything the device produces has to
        wait in the module's own queue — which is where the bound and the drop policy live.
        """
        self._gate.clear()

    def release(self) -> None:
        self._gate.set()

    def arm(self) -> None:
        """Create the iterator without consuming it (criterion 5's slow consumer)."""
        if self._it is None:
            name = self.ITERATORS[self.kind]
            self._it = attr_or_fail(self.io, name, WHY_CAPTURE)()

    def start(self) -> None:
        self.arm()
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        iterator = self._it.__aiter__()
        try:
            while True:
                await self._gate.wait()
                self.items.append(await iterator.__anext__())
        except StopAsyncIteration:
            self.finished = True
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # surfaced by the test, never swallowed
            self.error = exc

    def raise_error(self) -> None:
        if self.error is not None:
            raise self.error

    def __len__(self) -> int:
        return len(self.items)

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(BaseException):
                await self._task


async def capture_settle(
    device: FakeAudioDevice,
    *sinks: CaptureSink,
    quiet_rounds: int = 5,
    timeout_s: float = 2.0,
) -> None:
    """Let the module's capture path (thread *or* task) drain what the device handed over.

    Quiescence is "no new device buffer and no new item in any sink for ``quiet_rounds``
    scheduler turns". Real time only: the *device's* clock is virtual and moves only when a test
    says so.
    """
    deadline = time.monotonic() + timeout_s
    last: tuple[Any, ...] | None = None
    quiet = 0
    while time.monotonic() < deadline:
        await asyncio.sleep(0.0003)
        state = (device.write_calls, device.delivered_buffers, tuple(len(s) for s in sinks))
        if state != last:
            last, quiet = state, 0
        elif (quiet := quiet + 1) >= quiet_rounds:
            return


async def capture_out(
    device: FakeAudioDevice,
    ms: float,
    *sinks: CaptureSink,
    step_ms: float = 10.0,
) -> None:
    """Advance the device's virtual clock by ``ms``, delivering capture buffers as they fill.

    ``step_ms`` is the granularity of the *clock*, not of delivery: the device still hands over
    whole ``capture_buffer_frames`` buffers, and ``stall_capture()`` still holds them back.
    """
    left = float(ms)
    await capture_settle(device, *sinks)
    while left > 0:
        step = min(float(step_ms), left)
        device.clock.advance(int(step))
        device.pump()
        left -= step
        await capture_settle(device, *sinks)


def samples_of(block: Any) -> np.ndarray:
    """``block.samples`` as a numpy array, with the field named in the failure (criterion 2)."""
    return np.asarray(attr_or_fail(block, "samples", WHY_CAPTURE))


def t_ms_of(block: Any) -> float:
    return float(attr_or_fail(block, "t_ms", WHY_CAPTURE))


def concat(blocks: Sequence[Any]) -> np.ndarray:
    if not blocks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([samples_of(b) for b in blocks])


async def capture_stats(io: Any) -> Any:
    """``capture_stats()`` — T-0021 criterion 10's snapshot (concrete class, not the protocol)."""
    return await maybe_await(attr_or_fail(io, "capture_stats", WHY_CAPTURE)())


def stat(stats: Any, name: str) -> Any:
    return attr_or_fail(stats, name, f"{WHY_CAPTURE} — criterion 10's CaptureStats.{name}")


# --- a TtsUtterance carrying real PCM ------------------------------------------------------------


def tone(duration_ms: float, freq_hz: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    n = round(duration_ms * TTS_RATE / 1000.0)
    t = np.arange(n, dtype=np.float64) / TTS_RATE
    return amplitude * np.sin(2.0 * math.pi * freq_hz * t)


def to_s16le(x: np.ndarray) -> bytes:
    return np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class PcmUtterance:
    """``TtsUtterance`` with real 24 kHz s16le audio, exact word timings and a stall control."""

    def __init__(
        self,
        utterance_id: str = "u1",
        *,
        samples: np.ndarray | None = None,
        duration_ms: float | None = None,
        words: Sequence[WordTiming] | None = None,
        text: str = "",
        chunk_ms: int = 80,
        hold_after_ms: float | None = None,
    ) -> None:
        if samples is None:
            samples = tone(duration_ms if duration_ms is not None else 1000.0)
        self.utterance_id = utterance_id
        self.samples = samples
        self.text = text
        self.chunk_ms = chunk_ms
        self.hold_after_ms = hold_after_ms
        self._words = list(words) if words is not None else None
        self.cancelled = False
        self.chunks_sent = 0
        self._released = asyncio.Event()
        self.first_audio_latency_ms_value: float | None = None

    @property
    def duration_ms(self) -> float:
        return 1000.0 * self.samples.size / TTS_RATE

    def release(self) -> None:
        """Let the audio iterator past ``hold_after_ms`` (end of a simulated TTS stall)."""
        self._released.set()

    async def audio(self) -> AsyncIterator[bytes]:
        per_chunk = round(self.chunk_ms * TTS_RATE / 1000.0)
        sent = 0
        while sent < self.samples.size and not self.cancelled:
            at_ms = 1000.0 * sent / TTS_RATE
            if self.hold_after_ms is not None and at_ms >= self.hold_after_ms:
                await self._released.wait()
            chunk = self.samples[sent : sent + per_chunk]
            sent += chunk.size
            self.chunks_sent += 1
            if self.first_audio_latency_ms_value is None:
                self.first_audio_latency_ms_value = 0.0
            yield to_s16le(chunk)

    def words(self) -> Sequence[WordTiming] | None:
        return self._words

    async def cancel(self) -> None:
        self.cancelled = True
        self._released.set()

    @property
    def first_audio_latency_ms(self) -> float | None:
        return self.first_audio_latency_ms_value


def words_of(pairs: Sequence[tuple[str, int, int]]) -> list[WordTiming]:
    return [WordTiming(text=t, start_ms=s, end_ms=e) for t, s, e in pairs]


# --- fixtures and signal oracles ------------------------------------------------------------------


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Stdlib reader (no soundfile, no sounddevice): mono s16le -> float64 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise AssertionError(f"{path.name}: expected s16, got {8 * w.getsampwidth()}-bit")
        rate = w.getframerate()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, rate


def chirp_fixture() -> tuple[np.ndarray, int]:
    return read_wav(FIXTURES / "chirp_5s_24k.wav")


def tts_fixture() -> tuple[np.ndarray, int, dict[str, Any]]:
    x, rate = read_wav(FIXTURES / "pocket_tts_sentence.wav")
    meta = json.loads((FIXTURES / "pocket_tts_sentence.words.json").read_text())
    return x, rate, meta


def chirp_position_ms(x: np.ndarray, rate: int, at_sample: int, window_ms: float = 20.0) -> float:
    """Where in ``chirp_5s_24k.wav`` a recorded sample sits, from its instantaneous frequency.

    The zero-crossing rate of the ``window_ms`` *ending* at ``at_sample`` gives the mean frequency
    over that window; the fixture's frequency law (MANIFEST.md) inverts it to the mid time of the
    crossings, and the remaining offset to the window end is added back. Rate-independent, so it
    reads a 48 kHz monitor recording of the 24 kHz fixture. Independent of anything the module
    under test reports.
    """
    w = max(32, round(window_ms * rate / 1000.0))
    hi = min(len(x), at_sample + 1)
    seg = x[max(0, hi - w) : hi]
    if seg.size < 32:
        raise AssertionError("window too short for a frequency estimate")
    crossings = np.flatnonzero(np.diff(np.signbit(seg)))
    if crossings.size < 3:
        raise AssertionError("too few zero crossings: the recording is silent or clipped here")
    first, last = int(crossings[0]), int(crossings[-1])
    freq = (crossings.size - 1) / (2.0 * (last - first) / rate)
    mid_ms = 1000.0 * (freq - CHIRP_F0_HZ) / CHIRP_K_HZ_PER_S
    to_end_ms = 1000.0 * (seg.size - 1 - 0.5 * (first + last)) / rate
    return mid_ms + to_end_ms


def rendered_ms_from_chirp(rec: np.ndarray, rate: int, threshold: float = 0.02) -> float:
    """How much of the chirp a recording contains, read off its own frequency ramp."""
    nz = np.flatnonzero(np.abs(rec) > threshold)
    if nz.size == 0:
        raise AssertionError("recording is silent: the rig is not rendering (criterion 11)")
    return chirp_position_ms(rec, rate, int(nz[-1]))


def chirp_reference_48k(seconds: float = 2.0) -> np.ndarray:
    """The first ``seconds`` of the chirp fixture at the device rate, as a correlation template."""
    x, rate = chirp_fixture()
    assert rate == TTS_RATE
    up = np.interp(np.arange(2 * x.size) / 2.0, np.arange(x.size), x)
    return up[: round(seconds * DEVICE_RATE)]


def chirp_correlation(rec: np.ndarray, rate: int, seconds: float = 2.0) -> float:
    """Normalised cross-correlation peak of a recording against the chirp fixture, in [0, 1].

    The criterion-11 gate (T-0006 ruling A7): a recording *of the chirp* scores ≈ 0.9, room noise
    or a neighbouring stream scores ≈ 0, and no energy threshold can tell those two apart — the
    C925e's room noise peaks at ~0.12, well above any floor a silent-sink test would use.
    """
    if rate != DEVICE_RATE:
        raise AssertionError(f"recording is {rate} Hz, expected {DEVICE_RATE}")
    ref = chirp_reference_48k(seconds)
    rec = np.asarray(rec, dtype=np.float64)
    if rec.size < ref.size:
        raise AssertionError(
            f"recording is {1000 * rec.size / rate:.0f} ms, shorter than the {seconds:.1f} s "
            f"correlation template"
        )
    lag = fft_best_lag(ref, rec)
    seg = rec[lag : lag + ref.size]
    if seg.size < ref.size:
        seg = np.pad(seg, (0, ref.size - seg.size))
    norm = float(np.linalg.norm(seg) * np.linalg.norm(ref))
    return 0.0 if norm == 0.0 else abs(float(np.dot(seg, ref))) / norm


def nonzero_regions(x: np.ndarray, rate: int, threshold: float = 1e-3) -> list[tuple[int, int]]:
    loud = np.abs(x) > threshold
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for i, a in enumerate(loud):
        if a and start is None:
            start = i
        elif not a and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(x)))
    return regions


def max_abs_step(x: np.ndarray) -> float:
    """Largest consecutive-sample step; a hard cut of a loud signal shows up here as a click."""
    return 0.0 if x.size < 2 else float(np.max(np.abs(np.diff(x))))


def residual_db(reference: np.ndarray, signal: np.ndarray) -> float:
    """10·log10(‖signal−reference‖² / ‖reference‖²) over the common length."""
    n = min(reference.size, signal.size)
    ref, sig = reference[:n], signal[:n]
    num = float(np.sum((sig - ref) ** 2))
    den = float(np.sum(ref**2))
    if den == 0.0:
        raise AssertionError("reference is silent")
    return 10.0 * math.log10(max(num, 1e-300) / den)


def best_lag(reference: np.ndarray, signal: np.ndarray, max_lag: int) -> int:
    """Lag (samples) maximising the cross-correlation, searched by brute force over ±max_lag."""
    n = min(reference.size, signal.size) - max_lag
    best, best_score = 0, -math.inf
    for lag in range(-max_lag, max_lag + 1):
        a = reference[max_lag : max_lag + n]
        b = signal[max_lag + lag : max_lag + lag + n]
        score = float(np.dot(a, b))
        if score > best_score:
            best, best_score = lag, score
    return best


# --- the PipeWire rig (integration only; nothing above this line touches a real device) ----------

#: The null sink ``scripts/audio_rig.sh`` creates (T-0006 criterion 11). ``pactl`` is not on this
#: box, so the rig is built with ``pw-cli``/``pw-loopback`` and verified with ``pw-dump``.
NULL_SINK = "astra-test-sink"


def pw_nodes() -> dict[str, int] | None:
    """``{node.name: id}`` from ``pw-dump``, or None if PipeWire tooling is not available."""
    import shutil
    import subprocess

    if shutil.which("pw-dump") is None:
        return None
    proc = subprocess.run(["pw-dump"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    nodes: dict[str, int] = {}
    for obj in json.loads(proc.stdout):
        if obj.get("type") != "PipeWire:Interface:Node":
            continue
        name = obj.get("info", {}).get("props", {}).get("node.name")
        if isinstance(name, str):
            nodes[name] = int(obj["id"])
    return nodes


def missing_node_reason(name: str, how: str) -> str | None:
    """Skip reason naming the device that is absent, or None when it is there.

    Never returns a reason for "the recording was silent" — that is criterion 11's failure, not a
    skip: the default sink on this box is a **Dummy Output** while the DP cable is unplugged, and
    a test that skips on silence would report success through a dead speaker.
    """
    nodes = pw_nodes()
    if nodes is None:
        return "pw-dump is not available: no PipeWire session to run the audio rig in"
    if name not in nodes:
        return f"PipeWire node {name!r} is not in pw-dump ({how})"
    return None


class MonitorRecording:
    """``pw-record`` on a node, as a context manager. The recording is the oracle.

    ``capture_sink`` (T-0006 ruling A7) is the difference between an oracle and a lie:
    ``pw-record --target <sink>`` alone does **not** record that sink's monitor — WirePlumber
    hands the stream the default *source* instead, which on this box is the C925e webcam mic
    whose room noise peaks around 0.12 and sails past an energy floor. ``-P
    stream.capture.sink=true`` is what makes the stream a monitor of the target sink (the same
    property ``scripts/audio_loopback.py`` sets). Pass ``capture_sink=False`` when the target
    really is a capture device (the acoustic loopback's microphone).
    """

    def __init__(self, target: str, path: Path, *, capture_sink: bool = True) -> None:
        self.target = target
        self.path = path
        self.capture_sink = capture_sink
        self._proc: Any = None

    def __enter__(self) -> MonitorRecording:
        import subprocess

        command = ["pw-record", "--target", self.target, "--rate", str(DEVICE_RATE)]
        if self.capture_sink:
            command += ["-P", "stream.capture.sink=true"]
        self._proc = subprocess.Popen(
            [*command, str(self.path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.3)  # let the stream connect before the first sample is played
        return self

    def __exit__(self, *exc: object) -> None:
        if self._proc is not None:
            time.sleep(0.2)
            self._proc.terminate()
            self._proc.wait(timeout=5)

    def read(self) -> tuple[np.ndarray, int]:
        x, rate = read_wav(self.path)
        if x.size == 0:
            raise AssertionError(
                f"pw-record captured nothing from {self.target!r} (criterion 11: the rig must "
                f"not pass silently)"
            )
        return x, rate


def default_sink_name() -> str | None:
    """The PipeWire default sink (``default.audio.sink`` metadata), or None if there is none."""
    import subprocess

    proc = subprocess.run(["pw-dump"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    for obj in json.loads(proc.stdout):
        if obj.get("type") != "PipeWire:Interface:Metadata":
            continue
        for entry in obj.get("metadata", []) or []:
            if entry.get("key") == "default.audio.sink":
                value = entry.get("value")
                if isinstance(value, dict):
                    return str(value.get("name"))
                if isinstance(value, str):
                    return value
    return None


def fft_best_lag(reference: np.ndarray, signal: np.ndarray) -> int:
    """Lag in samples that maximises the cross-correlation, via rfft (float64 oracle)."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(signal, dtype=np.float64)
    n = 1 << int(np.ceil(np.log2(a.size + b.size)))
    corr = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
    return int(np.argmax(corr[: b.size]))


def rendering_span_start_ms(io: Any, after_ms: float, window_ms: float = 3000.0) -> float:
    """The first capture-clock instant at or after ``after_ms`` at which we were rendering.

    Read through the **public** ``is_rendering(t_ms)`` — the one question the self-speech gate
    asks — so the number is the span start as the gate sees it, not a private field. Resolution is
    half a block; the caller's own timing is coarser than that.
    """
    step = 0.5 * float(CAPTURE_BLOCK_MS)
    probe = attr_or_fail(io, "is_rendering", "T-0006 criterion 10: the rendering span")
    t = float(after_ms)
    stop = t + float(window_ms)
    while t < stop:
        if bool(probe(t)):
            return t
        t += step
    raise AssertionError(
        f"no rendering span opened in the {window_ms:.0f} ms after {after_ms:.1f} ms: play() "
        f"never handed audio to the device, so there is nothing to measure the echo against"
    )


async def acoustic_delay_ms(
    io: Any,
    device: Any,
    utterance: Any,
    *,
    settle_s: float = 0.3,
    listen_s: float = 3.0,
    template_s: float = 2.0,
    heard_peak: float = 0.2,
) -> tuple[float, float]:
    """DAC → air → ADC for one chirp, on the module's own clocks (T-0023 ruling A4).

    Returns ``(delay_ms, peak)``. The chirp goes out through ``io.play()`` and comes back through
    the **same duplex device's** capture, so both ends are stamped on one clock: the delay is
    where the correlation peak sits on the capture clock minus where the rendering span for that
    chirp started. Nothing external times anything — the earlier method (two ``pw-record``
    processes started 0.3 s apart, monitor minus mic) had no shared time origin, and a sink that
    resumes late turns that start jitter into a plausible ≈300 ms "echo delay" (ruling A4).

    This is exactly the quantity the gate needs: the lag between handing a block to the DAC and
    hearing it at the ADC. The caller states the regime (sink, transducer, distance, floor, n).
    """
    raw = CaptureSink(io, "raw")
    try:
        await asyncio.sleep(settle_s)  # capture running, so the span start is inside the window
        before_ms = float(attr_or_fail(device, "captured_ms", "T-0021 ruling A4")())
        await play(io, utterance, source="chirp")
        await asyncio.sleep(listen_s)
        blocks = list(raw.items)
    finally:
        await raw.close()
    if not blocks:
        raise AssertionError("no capture blocks at all: the duplex device captured nothing")
    heard = np.asarray(concat(blocks), dtype=np.float64)
    loudest = float(np.max(np.abs(heard))) if heard.size else 0.0
    if loudest <= heard_peak:
        raise AssertionError(
            f"nothing audible came back: the capture peaks at {loudest:.3f} (floor {heard_peak}) "
            f"while the chirp was playing — the rig went silent mid-measurement"
        )
    lag = fft_best_lag(chirp_reference_48k(template_s), heard)
    heard_ms = t_ms_of(blocks[0]) + 1000.0 * lag / DEVICE_RATE
    span_start_ms = rendering_span_start_ms(io, before_ms)
    return heard_ms - span_start_ms, loudest
