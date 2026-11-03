"""The playback device of ``astra session --silent``: it models time, it does not abolish it.

T-0039 criteria 15 and 16. ``NullAudioDevice`` is a complete ``AudioDevice``
(``conductor/audio_device.AudioDevice``, docs/02 §2.7) that puts no audio anywhere, so a session
runs on a box with no PortAudio and no sink at all, and nothing is audible. ``PipeWireAudioIO``
is handed one unchanged -- ``--silent`` is a **device swap, not a policy branch** (T-0039 §5),
which is what keeps the turn machine, the drain rule and the spoken pane byte-identical between a
silent run and an acoustic one.

**The rendering model, stated once.** The device consumes 1 ms of audio per 1 ms of the injected
clock, and never faster:

* ``write(block)`` hands over audio and **blocks while the modelled queue is full** -- the
  back-pressure a real device has. Without it this is an infinite sink: the writer would push a
  whole utterance in one pass, ``queued_ms()`` would mean nothing and ``played_ms`` would run
  ahead of anything anyone could have heard;
* ``rendered_ms()`` is what the device has taken out of the queue -- a position **inside the
  stream we wrote**, never the clock: a device fed nothing renders nothing, however long the
  clock runs;
* ``queued_ms()`` is what was handed over and not taken yet, and it is what ``audio_io``'s drain
  rule reads (T-0006 ruling A15). A device that renders instantly reports 0 the moment it is
  written, so every ``mark_drain()`` would drain before a word was "spoken" and every silent turn
  would end mid-sentence. That is the failure mode this module is written against;
* ``underflow_frames`` stays ``0``: a device that was given nothing to play is idle, not starved,
  and nobody may read an xrun out of a silent run;
* ``quantum_ms`` is ``0.0`` with ``quantum_source == "unknown"`` -- there is no graph here and no
  number to report, and a number without its provenance is not a number (T-0035 ruling A13).

Real-time pacing is the *decision* (T-0039 §5 and open ambiguity 3): a silent turn takes as long
as the speech would have, so ``played_ms``, ``played_text``, the unplayed strike and the drain
rule all keep their meaning. An instant-render mode would be a different flag with its own regime
word, never a change to ``silent``.

**The clock is injected** so the unit suite runs on ``tests/conftest.VirtualClock`` and the fast
suite is not paced by wall time; ``astra session --silent`` passes the session's own
``SessionClock``.

**The capture half is a no-op** (criterion 19, ambiguity 4 decided yes): ``--silent`` is one
device object for both directions, so there is no microphone. ``captured_ms()`` still integrates
the clock -- it is a clock, and the protocol calls it monotonic -- but no callback is ever
invoked and no audio is invented to feed one.

This module names no sound library and reads no PipeWire graph: ``audio_device.py`` stays the
single module in ``local_astra/`` allowed to name one (``tests/unit/test_audio_ownership.py``).
"""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np
import numpy.typing as npt

from local_astra.conductor.audio_device import CaptureCallback
from local_astra.contracts.speech import CAPTURE_BLOCK_MS, CAPTURE_FORMAT

DEVICE_RATE = CAPTURE_FORMAT.sample_rate

#: How much audio the modelled queue holds before ``write()`` blocks, in ms. Comfortably deeper
#: than ``PipeWireAudioIO._target_lead_ms`` (one block): the writer thread holds the module lock
#: across ``write()`` and the event loop takes that same lock in ``play()``, so a queue shallow
#: enough to block the normal writer would stall the loop far beyond the 20 ms
#: ``.claude/rules/conductor.md`` allows.
DEFAULT_BUFFER_MS = 200.0

#: How often a blocked ``write()`` re-reads the clock. Nothing signals it: the clock is advanced
#: by whoever owns it (a test thread, or wall time under ``SessionClock``), so the wait is a poll
#: with a timeout rather than a notification.
_POLL_S = 0.0005


class Clock(Protocol):
    """The millisecond clock this device is paced by (``session.SessionClock``, ``VirtualClock``)."""

    @property
    def now_ms(self) -> int: ...


class NullAudioDevice:
    """``AudioDevice`` that renders to nowhere at real-time rate. See the module docstring."""

    def __init__(
        self,
        clock: Clock,
        *,
        buffer_ms: float = DEFAULT_BUFFER_MS,
        block_ms: int = CAPTURE_BLOCK_MS,
        sample_rate: int = DEVICE_RATE,
    ) -> None:
        self.clock = clock
        self.buffer_ms = float(buffer_ms)
        self.block_ms = int(block_ms)
        self.sample_rate = int(sample_rate)
        self.channels = 1
        #: Nothing is handed to a DAC, so there is no lag to report and none to subtract: a
        #: silent device's rendered position saturates at what was written.
        self.latency_ms = 0.0
        self.input_latency_ms = 0.0
        #: No graph, no quantum -- and the provenance says so rather than suggesting a
        #: measurement (T-0035 ruling A13).
        self.quantum_ms = 0.0
        self.quantum_source = "unknown"
        #: A device that invents no frames counts none, for ever (criterion 15).
        self.underflow_frames = 0

        self._cv = threading.Condition()
        self._started = False
        self._written_ms = 0.0
        self._taken_ms = 0.0
        self._last_t_ms = float(self.clock.now_ms)

        self._capturing = False
        self._captured_ms = 0.0
        self._cap_last_t_ms = float(self.clock.now_ms)
        self._capture_callback: CaptureCallback | None = None

    # -- AudioDevice: the output half --------------------------------------------------------
    def start(self) -> None:
        with self._cv:
            self._integrate()
            self._started = True
            self._cv.notify_all()

    def stop(self) -> None:
        with self._cv:
            self._integrate()
            self._started = False

    def write(self, block: npt.NDArray[np.float32]) -> None:
        """Hand over one block, blocking while the modelled queue is full (criterion 15).

        An empty queue always accepts, whatever the block is: a block larger than ``buffer_ms``
        would otherwise wait for room that can never appear.
        """
        ms = 1000.0 * int(np.asarray(block).size) / self.sample_rate
        with self._cv:
            while True:
                self._integrate()
                queued = self._written_ms - self._taken_ms
                if queued <= 0.0 or queued + ms <= self.buffer_ms:
                    self._written_ms += ms
                    return
                self._cv.wait(_POLL_S)

    def rendered_ms(self) -> float:
        """Audio of *our* stream the device has taken out of the queue. Monotonic."""
        with self._cv:
            self._integrate()
            return self._taken_ms

    def queued_ms(self) -> float:
        """Audio handed over that the device has not taken yet -- the drain rule's number."""
        with self._cv:
            self._integrate()
            return max(0.0, self._written_ms - self._taken_ms)

    # -- AudioDevice: the input half (a no-op, criterion 19) ---------------------------------
    def start_capture(self) -> None:
        with self._cv:
            self._integrate_capture()
            self._capturing = True

    def stop_capture(self) -> None:
        with self._cv:
            self._integrate_capture()
            self._capturing = False

    def captured_ms(self) -> float:
        """The ADC position a silent device would have: the clock, never delivered audio."""
        with self._cv:
            self._integrate_capture()
            return self._captured_ms

    def set_capture_callback(self, callback: CaptureCallback) -> None:
        """Installed and never called: ``--silent`` opens no microphone (criterion 19)."""
        with self._cv:
            self._capture_callback = callback

    # -- the two integrators -----------------------------------------------------------------
    def _integrate(self) -> None:
        """Consume elapsed clock: take what we were given, and never more than that.

        Time the device spends with nothing to take is *not* credited to the stream and is not an
        underflow either -- it is an idle device, which is what a silent session mostly is.
        """
        now = float(self.clock.now_ms)
        dt = max(0.0, now - self._last_t_ms)
        self._last_t_ms = now
        if not self._started or dt <= 0.0:
            return
        self._taken_ms += min(dt, max(0.0, self._written_ms - self._taken_ms))

    def _integrate_capture(self) -> None:
        now = float(self.clock.now_ms)
        dt = max(0.0, now - self._cap_last_t_ms)
        self._cap_last_t_ms = now
        if self._capturing:
            self._captured_ms += dt
