"""A ``sounddevice`` module double, installed into ``sys.modules`` for the length of one test.

PortAudio is **not installed on this box** (T-0035 prerequisite P1): ``import sounddevice`` raises
``OSError`` because ``libportaudio2`` is missing. Group A of T-0035 nevertheless has to exercise
``PortAudioDevice.start()``, its callbacks and the environment it sets while a stream is
constructed, so this module stands in for the library:

* ``query_devices(device, kind)`` — the two latency hints ``start()`` reads;
* ``OutputStream`` / ``Stream`` — recording doubles that keep the callback, the constructor
  keyword arguments, and **the environment visible at construction time** (``PIPEWIRE_NODE``,
  ``PIPEWIRE_LATENCY``, ``PIPEWIRE_PROPS``). That snapshot is the only way to test T-0035
  criterion 6 ("``PIPEWIRE_NODE`` is reserved for the input half"), because the variable is read
  by libpipewire when the client connects and is gone again by the time ``start()`` returns;
* ``CallbackFlags`` — a status object with settable ``output_underflow`` / ``input_overflow``;
* ``Stream.drive(n_callbacks, frames=…, taken=…, status=…)`` — invokes the installed callback
  synchronously, **on a non-asyncio worker thread**, the way PortAudio's device thread does
  (`.claude/rules/conductor.md`: the callback never runs on the event loop);
* ``Driver`` / ``driving(module, …)`` — the same callback at **real-time pace** on one background
  thread (480 frames every 10 ms), because a writer that waits for a queue to empty can only be
  tested against a device that is draining it while the writer runs (T-0042); ``stop_when``
  makes that device stall for good, which is what ``write()``'s 2 s ceiling exists for.

Installation is a fixture (``sounddevice_stub``) and it removes the module again, so no other
test in the suite can see it: a stub that outlived its test would make
``tests/unit/test_audio_ownership.py``'s "the device module really is the backend" meaningless
and could hide a real import error elsewhere.

``drive(..., taken=…)`` needs somebody to put ``taken`` frames into the device's ring before the
callback runs — the ring belongs to the device, not to this double — so the test rig sets
``stream.prefill = lambda n: device.write(np.zeros(n, dtype=np.float32))``; ``wire_prefill()``
below does it for every caller.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Any

import numpy as np
import pytest

#: The environment variables PipeWire reads when a client connects (``_PipewireEnv``).
PIPEWIRE_VARS = ("PIPEWIRE_NODE", "PIPEWIRE_LATENCY", "PIPEWIRE_PROPS")


class CallbackFlags:
    """``sounddevice.CallbackFlags`` stand-in: the four flags PortAudio can raise."""

    def __init__(
        self,
        *,
        output_underflow: bool = False,
        output_overflow: bool = False,
        input_underflow: bool = False,
        input_overflow: bool = False,
    ) -> None:
        self.output_underflow = output_underflow
        self.output_overflow = output_overflow
        self.input_underflow = input_underflow
        self.input_overflow = input_overflow

    def __bool__(self) -> bool:
        return any(
            (
                self.output_underflow,
                self.output_overflow,
                self.input_underflow,
                self.input_overflow,
            )
        )

    def __repr__(self) -> str:
        raised = [name for name, value in vars(self).items() if value]
        return f"CallbackFlags({', '.join(raised) or 'clear'})"


class TimeInfo:
    """``time_info``: the three PortAudio timestamps the device reads, in seconds."""

    def __init__(self, current: float, dac_lag_s: float, adc: float) -> None:
        self.currentTime = current
        self.outputBufferDacTime = current + dac_lag_s
        self.inputBufferAdcTime = adc


class StubStream:
    """The recording double behind both ``OutputStream`` and ``Stream``."""

    def __init__(self, module: ModuleType, kind: str, kwargs: dict[str, Any]) -> None:
        self.module = module
        #: ``"output"`` for ``OutputStream``, ``"duplex"`` for ``Stream``.
        self.kind = kind
        self.kwargs = dict(kwargs)
        #: the environment libpipewire would have read when this client connected
        self.env: dict[str, str | None] = {key: os.environ.get(key) for key in PIPEWIRE_VARS}
        self.callback: Callable[..., None] | None = kwargs.get("callback")
        self.blocksize: int = int(kwargs.get("blocksize") or 0)
        self.samplerate: int = int(kwargs.get("samplerate") or 48000)
        self.channels: int = int(kwargs.get("channels") or 1)
        self.starts = 0
        self.stops = 0
        self.closes = 0
        self.callbacks = 0
        #: every thread the callback ran on, so a test can prove it was not the main thread
        self.callback_threads: list[str] = []
        #: what the device wrote out, per callback (mono, the first channel)
        self.out_blocks: list[np.ndarray] = []
        #: filled into ``indata`` on the duplex path; a constant is enough for these tests
        self.capture_value = 0.0
        #: set by the rig: puts ``n`` frames into the device's ring before a callback
        self.prefill: Callable[[int], None] | None = None
        self._t = 0.0
        self._adc = 0.0
        latency = self.module.negotiated_latency_s  # type: ignore[attr-defined]
        in_latency = self.module.negotiated_input_latency_s  # type: ignore[attr-defined]
        self.latency: float | tuple[float, float] = (
            (in_latency, latency) if kind == "duplex" else latency
        )

    # -- the sounddevice surface ----------------------------------------------------------
    def start(self) -> None:
        self.starts += 1

    def stop(self) -> None:
        self.stops += 1

    def close(self) -> None:
        self.closes += 1

    @property
    def active(self) -> bool:
        return self.starts > self.stops and self.closes == 0

    # -- the device thread ----------------------------------------------------------------
    def drive(
        self,
        n_callbacks: int = 1,
        *,
        frames: int | None = None,
        taken: int | None = None,
        status: CallbackFlags | bool | None = None,
        dac_lag_s: float = 0.0,
    ) -> None:
        """Run the installed callback ``n_callbacks`` times on a worker thread.

        ``taken`` asks the rig to make exactly that many frames available in the device's ring
        first; ``status`` is either a ``CallbackFlags`` or ``True`` for ``output_underflow``.
        """
        if self.callback is None:
            raise AssertionError("the stream was constructed without a callback")
        flags = CallbackFlags(output_underflow=True) if status is True else status
        flags = CallbackFlags() if flags is None else flags
        n_frames = int(frames if frames is not None else self.blocksize)
        for _ in range(n_callbacks):
            if taken is not None:
                if self.prefill is None:
                    raise AssertionError(
                        "drive(taken=…) needs stream.prefill (see the module docstring)"
                    )
                self.prefill(taken)
            error: list[BaseException] = []

            def run(error: list[BaseException] = error) -> None:
                try:
                    self.one_callback(n_frames, flags, dac_lag_s=dac_lag_s)
                except BaseException as exc:  # pragma: no cover - surfaced below
                    error.append(exc)

            thread = threading.Thread(target=run, name="stub-portaudio")
            thread.start()
            thread.join(timeout=5.0)
            if thread.is_alive():  # pragma: no cover - a deadlocked callback
                raise AssertionError("the device callback did not return within 5 s")
            if error:
                raise error[0]

    def one_callback(
        self,
        n_frames: int,
        flags: CallbackFlags,
        *,
        dac_lag_s: float = 0.0,
    ) -> None:
        """One callback, **on the calling thread** — which must not be the main thread.

        Shared by ``drive()`` (one worker thread per callback, synchronous) and by ``Driver``
        (one thread, real-time pace). Everything PortAudio's device thread does per callback is
        here: the two buffers, the three timestamps, the counters, and the record of what the
        device emitted.
        """
        if self.callback is None:
            raise AssertionError("the stream was constructed without a callback")
        outdata = np.zeros((n_frames, self.channels), dtype=np.float32)
        indata = np.full((n_frames, self.channels), self.capture_value, dtype=np.float32)
        info = TimeInfo(self._t, dac_lag_s, self._adc)
        args = (
            (indata, outdata, n_frames, info, flags)
            if self.kind == "duplex"
            else (outdata, n_frames, info, flags)
        )
        self.callback_threads.append(threading.current_thread().name)
        self.callback(*args)
        self.callbacks += 1
        self.out_blocks.append(np.array(outdata[:, 0]))
        self._t += n_frames / self.samplerate
        self._adc += n_frames / self.samplerate

    def emitted(self) -> np.ndarray:
        """Every frame this stream handed to the DAC, concatenated (mono)."""
        if not self.out_blocks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.out_blocks)


def make_module() -> ModuleType:
    """A fresh ``sounddevice`` double. Configure it *before* the device opens its stream."""
    module = ModuleType("sounddevice")
    module.__version__ = "0.5.6-stub"  # type: ignore[attr-defined]
    #: what ``stream.latency`` reports back — the *negotiated* value (never the request)
    module.negotiated_latency_s = 0.021  # type: ignore[attr-defined]
    module.negotiated_input_latency_s = 0.019  # type: ignore[attr-defined]
    #: what ``query_devices`` offers as the device's low-latency hints
    module.low_output_latency_s = 0.010  # type: ignore[attr-defined]
    module.low_input_latency_s = 0.010  # type: ignore[attr-defined]
    module.streams = []  # type: ignore[attr-defined]
    module.device_queries = []  # type: ignore[attr-defined]

    class PortAudioError(Exception):
        pass

    def query_devices(device: Any = None, kind: str | None = None) -> dict[str, Any]:
        module.device_queries.append((device, kind))  # type: ignore[attr-defined]
        return {
            "name": str(device) if device is not None else "default",
            "max_output_channels": 2,
            "max_input_channels": 2,
            "default_samplerate": 48000.0,
            "default_low_output_latency": module.low_output_latency_s,  # type: ignore[attr-defined]
            "default_high_output_latency": 0.096,
            "default_low_input_latency": module.low_input_latency_s,  # type: ignore[attr-defined]
            "default_high_input_latency": 0.096,
        }

    def output_stream(**kwargs: Any) -> StubStream:
        stream = StubStream(module, "output", kwargs)
        module.streams.append(stream)  # type: ignore[attr-defined]
        return stream

    def duplex_stream(**kwargs: Any) -> StubStream:
        stream = StubStream(module, "duplex", kwargs)
        module.streams.append(stream)  # type: ignore[attr-defined]
        return stream

    module.CallbackFlags = CallbackFlags  # type: ignore[attr-defined]
    module.PortAudioError = PortAudioError  # type: ignore[attr-defined]
    module.query_devices = query_devices  # type: ignore[attr-defined]
    module.OutputStream = output_stream  # type: ignore[attr-defined]
    module.Stream = duplex_stream  # type: ignore[attr-defined]
    module.InputStream = duplex_stream  # type: ignore[attr-defined]
    return module


@contextmanager
def installed() -> Iterator[ModuleType]:
    """Install the double as ``sounddevice`` for the body, then put ``sys.modules`` back."""
    module = make_module()
    had = "sounddevice" in sys.modules
    previous = sys.modules.get("sounddevice")
    sys.modules["sounddevice"] = module
    try:
        yield module
    finally:
        if had:
            sys.modules["sounddevice"] = previous  # type: ignore[assignment]
        else:
            sys.modules.pop("sounddevice", None)


@pytest.fixture
def sounddevice_stub() -> Iterator[ModuleType]:
    """The double for one test. Removed afterwards: no other test may see it."""
    with installed() as module:
        yield module


def last_stream(module: ModuleType) -> StubStream:
    """The stream the device opened most recently; its absence is an assertion, not an IndexError."""
    streams: list[StubStream] = module.streams  # type: ignore[attr-defined]
    if not streams:
        raise AssertionError("the device opened no stream at all")
    return streams[-1]


def wire_prefill(stream: StubStream, device: Any) -> StubStream:
    """Let ``drive(taken=…)`` put frames into *the device's* ring before each callback."""

    def prefill(n: int) -> None:
        if n:
            device.write(np.zeros(n, dtype=np.float32))

    stream.prefill = prefill
    return stream


#: PortAudio's device thread at 48 kHz with a 480-frame block: one callback every 10 ms.
CALLBACK_PERIOD_S = 0.010


class Driver:
    """A **device thread**: the stream's callback, once every ``period_s``, at real-time pace.

    ``drive()`` is the synchronous shape -- the test says when a callback happens. That cannot
    express the condition T-0042 is about: a device that *keeps taking audio while the writer is
    writing*, so that a writer which refills the queue it is waiting on never sees it empty. This
    driver is the asynchronous shape: it starts before the stream exists (the device opens it
    inside ``start()``), waits for it, and then consumes ``frames`` frames every ``period_s`` of
    **wall** time, exactly as PortAudio does, until it is stopped.

    ``stop_when`` is the device-failure shape: a predicate polled before every callback; the
    first time it is true the driver stops calling the callback for good, which is what a stalled
    PortAudio callback looks like from the writer's side (the queue stops shrinking and stays
    where it is). ``stop_when=lambda: True`` is a device that never consumes at all.

    Real time, not a virtual clock: ``write()``'s 2 s ceiling and the drain wait are wall-clock
    behaviours of the module under test, and a stub that lied about the clock would prove
    nothing about either. Keep the periods short.
    """

    def __init__(
        self,
        module: ModuleType,
        *,
        period_s: float = CALLBACK_PERIOD_S,
        frames: int | None = None,
        stop_when: Callable[[], bool] | None = None,
        wait_s: float = 5.0,
        name: str = "stub-portaudio-driver",
    ) -> None:
        self.module = module
        self.period_s = period_s
        self.frames = frames
        self.stop_when = stop_when
        self.wait_s = wait_s
        #: what went wrong on the driver thread, re-raised by ``driving()`` on the test's thread
        self.error: BaseException | None = None
        #: callbacks actually run (a callback the device starved still counts: it ran)
        self.callbacks = 0
        #: True once ``stop_when`` fired: from here on this device consumes nothing, for ever
        self.gave_up = False
        self.stream: StubStream | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> Driver:
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        stream = self._await_stream()
        if stream is None:
            return
        self.stream = stream
        frames = int(self.frames if self.frames is not None else stream.blocksize)
        flags = CallbackFlags()
        due = time.monotonic()
        while not self._stop.is_set():
            if self.stop_when is not None and self.stop_when():
                self.gave_up = True
                return
            try:
                stream.one_callback(frames, flags)
            except BaseException as exc:  # pragma: no cover - surfaced by driving()
                self.error = exc
                return
            self.callbacks += 1
            due += self.period_s
            now = time.monotonic()
            if due > now:
                self._stop.wait(due - now)
            else:
                due = now  # a slow box: carry on without bursting a batch of callbacks

    def _await_stream(self) -> StubStream | None:
        deadline = time.monotonic() + self.wait_s
        while not self._stop.is_set():
            streams: list[StubStream] = self.module.streams  # type: ignore[attr-defined]
            if streams:
                return streams[-1]
            if time.monotonic() > deadline:
                self.error = AssertionError(
                    f"no stream was opened within {self.wait_s} s, so this device thread had "
                    f"nothing to drive"
                )
                return None
            self._stop.wait(0.001)
        return None


@contextmanager
def driving(module: ModuleType, **kwargs: Any) -> Iterator[Driver]:
    """Run a ``Driver`` for the body, stop it afterwards, and surface what it hit."""
    driver = Driver(module, **kwargs).start()
    try:
        yield driver
    finally:
        driver.stop()
    if driver.error is not None:
        raise AssertionError(f"the stub device thread failed: {driver.error!r}")
