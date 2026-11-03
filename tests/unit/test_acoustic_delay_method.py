"""T-0023 ruling A4: the echo-path delay method, on a synthetic echo with a known answer.

The number this task exists for is the DAC → air → ADC lag, and until now the only way to check
*how* it is measured was to stand in the room. These two tests give the method an oracle: a stub
duplex device whose capture carries the chirp back at a hand-chosen 137.5 ms after the rendering
span opened, and the helper the integration test and the locked script both use must return that
number. No hardware, no clock, no sleep longer than the event loop needs.

Why the method changed (ruling A4): the first version correlated the sink's monitor and the mic,
each recorded by its own ``pw-record`` process started ~0.3 s apart. Two processes with no shared
time origin cannot measure a 30 ms lag between them — a sink that resumes late turns their start
jitter into a plausible ≈300 ms "echo delay", and a hangover derived from it would blow past
criterion 5's 200 ms bound for a reason that is not in the room at all. Both ends are now stamped
on the module's own clocks: the capture block's ``t_ms`` (the ADC instant, T-0021 ruling A4) and
the rendering span read through the public ``is_rendering()`` — the one question the gate asks.

Regime: pure CPU, no device, no socket. ``profile=n/a canvas=n/a steps=n/a``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest

from local_astra.conductor.audio_capture import CaptureBlock
from local_astra.contracts.speech import CAPTURE_BLOCK_MS
from tests.fakes.audio_device import (
    BLOCK_SAMPLES,
    DEVICE_RATE,
    acoustic_delay_ms,
    chirp_reference_48k,
    rendering_span_start_ms,
)

#: Hand-chosen: not a multiple of a block, so a helper that rounds to blocks cannot pass by luck.
TRUE_DELAY_MS = 137.5
#: Where the capture clock happens to stand when the measurement starts (it is not zero in life).
CAPTURE_ORIGIN_MS = 4_321.0
#: The lead audio_io adds between "play() returned" and "the DAC renders it".
QUEUED_LEAD_MS = 60.0


class StubDuplexDevice:
    """Just enough of ``AudioDevice`` for the helper: an ADC position and a stop button."""

    def __init__(self, captured_ms: float) -> None:
        self._captured_ms = captured_ms
        self.stops = 0

    def captured_ms(self) -> float:
        return self._captured_ms

    def stop(self) -> None:
        self.stops += 1


class StubDuplexIo:
    """A duplex ``PipeWireAudioIO`` stand-in whose room echoes the chirp at a known delay.

    ``capture_frames()`` yields 10 ms blocks stamped on the capture clock; the chirp appears in
    them ``TRUE_DELAY_MS`` after the rendering span starts, on top of a little noise so the
    correlation has something to reject. ``is_rendering()`` answers for the span the way
    ``audio_io.py`` does — half-open, opening one queued lead after ``play()``.
    """

    def __init__(self, device: StubDuplexDevice, *, delay_ms: float, seconds: float = 4.0) -> None:
        self.device = device
        self.delay_ms = delay_ms
        self.span_start_ms = device.captured_ms() + QUEUED_LEAD_MS
        self.span_end_ms = self.span_start_ms + 5_000.0
        self.played: list[Any] = []
        rng = np.random.default_rng(0)
        total = round(seconds * DEVICE_RATE)
        self.audio = (0.01 * rng.standard_normal(total)).astype(np.float32)
        echo = chirp_reference_48k(2.5)
        at = round((QUEUED_LEAD_MS + delay_ms) * DEVICE_RATE / 1000.0)
        self.audio[at : at + echo.size] += 0.8 * echo[: max(0, total - at)]

    async def play(self, utterance: Any, *, source: str | None = None) -> None:
        self.played.append((utterance, source))

    def is_rendering(self, t_ms: float) -> bool:
        return self.span_start_ms <= t_ms < self.span_end_ms

    async def capture_frames(self) -> AsyncIterator[CaptureBlock]:
        for i in range(self.audio.size // BLOCK_SAMPLES):
            yield CaptureBlock(
                samples=self.audio[i * BLOCK_SAMPLES : (i + 1) * BLOCK_SAMPLES],
                t_ms=CAPTURE_ORIGIN_MS + i * float(CAPTURE_BLOCK_MS),
            )
            await asyncio.sleep(0)

    def close_capture(self) -> None:
        pass


def test_the_span_start_is_read_through_the_public_is_rendering() -> None:
    """``rendering_span_start_ms`` finds the span the *gate* sees, to half a block.

    It must not read a private field: the number the echo is measured against is the same
    ``is_rendering(t_ms)`` the self-speech gate asks about, or the delay and the gate disagree.
    """
    device = StubDuplexDevice(CAPTURE_ORIGIN_MS)
    io = StubDuplexIo(device, delay_ms=TRUE_DELAY_MS)

    found = rendering_span_start_ms(io, CAPTURE_ORIGIN_MS)
    assert found == pytest.approx(io.span_start_ms, abs=0.5 * CAPTURE_BLOCK_MS)
    assert io.is_rendering(found)
    assert not io.is_rendering(found - 0.5 * CAPTURE_BLOCK_MS)

    io.span_start_ms = io.span_end_ms = -1.0  # nothing ever rendered
    with pytest.raises(AssertionError) as caught:
        rendering_span_start_ms(io, CAPTURE_ORIGIN_MS, window_ms=100.0)
    assert "rendering span" in str(caught.value)


async def test_the_echo_delay_is_the_peak_on_the_capture_clock_minus_the_span_start() -> None:
    """A synthetic 137.5 ms echo path comes back as 137.5 ms, from one clock (ruling A4).

    The answer is hand-chosen, not read off the implementation, and it is deliberately not a whole
    number of 10 ms blocks. A helper that timed the two ends against different origins — the old
    monitor-minus-mic pair, or ``time.monotonic()`` around ``play()`` — would be off by the queued
    lead, which is 60 ms here and would show up immediately.
    """
    device = StubDuplexDevice(CAPTURE_ORIGIN_MS)
    io = StubDuplexIo(device, delay_ms=TRUE_DELAY_MS)

    delay, loudest = await acoustic_delay_ms(
        io, device, object(), settle_s=0.0, listen_s=0.05, heard_peak=0.2
    )

    assert loudest > 0.3, f"the stub room came back at {loudest:.3f}: the fixture is not audible"
    assert delay == pytest.approx(TRUE_DELAY_MS, abs=0.5 * CAPTURE_BLOCK_MS), (
        f"the helper reported {delay:.1f} ms for a hand-built {TRUE_DELAY_MS} ms echo path "
        f"(queued lead {QUEUED_LEAD_MS} ms, capture origin {CAPTURE_ORIGIN_MS} ms): the two ends "
        f"are not being read on one clock (T-0023 ruling A4)"
    )
    assert io.played, "the chirp was never played: the helper measured its own silence"


async def test_a_silent_room_is_an_error_not_a_delay_of_zero() -> None:
    """The guard that stops a dead rig from producing a confident number (criterion 2)."""
    device = StubDuplexDevice(CAPTURE_ORIGIN_MS)
    io = StubDuplexIo(device, delay_ms=TRUE_DELAY_MS)
    io.audio = np.zeros_like(io.audio)

    with pytest.raises(AssertionError) as caught:
        await acoustic_delay_ms(io, device, object(), settle_s=0.0, listen_s=0.05, heard_peak=0.2)
    assert "audible" in str(caught.value)
