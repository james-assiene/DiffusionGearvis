"""The device's xrun counter tells the truth: T-0035 criteria 1, 2 and 10(c) (audit finding F2).

`underflow_frames` is a **protocol** member (T-0006 ruling A22) and the only honest evidence that
a stream was starved: ``audio_io.underruns`` reads it as a delta (`audio_io.py:569-570`) and the
alternative test ``rendered >= written`` is unreachable on a device that subtracts its output
latency. Today it counts only *our* empty ring (``frames - taken``) and never looks at
``status.output_underflow``, so an xrun the backend itself reports is invisible — the failure mode
the 2026-09-14 first-contact audit found.

What criterion 1 asks for, and what every row below asserts:

* per callback, ``underflow_frames += max(frames - taken, frames if output_underflow else 0)`` —
  a callback the device flagged is suspect *in whole*, and the two causes are never added
  together;
* ``device_xruns: int`` counts the **callbacks** that carried the flag, so "our ring starved" and
  "the graph starved" stay distinguishable in a report;
* ``device_xruns`` lives on ``PortAudioDevice`` **only** — criterion 10(c) keeps the ``AudioDevice``
  protocol's member set unchanged, exactly as T-0006 did for ``played_ms()``.

Regime: no PortAudio on this box (prerequisite P1), so ``sounddevice`` is
``tests/fakes/sounddevice_stub.py`` installed into ``sys.modules`` for the length of each test and
removed afterwards; no graph is read (no node is requested, so criterion 4 makes no ``pw-dump``
call); 48 kHz mono f32, ``block_ms`` 10 → 480 frames; the monotonicity row is n=100 callbacks with
the mix shuffled by ``random.Random(0)``. Every expected total is written out by hand in the
docstring of its test.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from local_astra.conductor.audio_device import DEVICE_RATE, AudioDevice, PortAudioDevice
from tests.fakes.audio_device import attr_or_fail
from tests.fakes.sounddevice_stub import StubStream, installed, last_stream, wire_prefill

FRAMES = 480  # 10 ms at 48 kHz — CAPTURE_BLOCK_MS, the block the writer hands over


@contextmanager
def rig(**kwargs: Any) -> Iterator[tuple[PortAudioDevice, StubStream]]:
    """An open ``PortAudioDevice`` on the stub, with ``drive(taken=…)`` wired to its ring."""
    with installed() as stub:
        device = PortAudioDevice(block_ms=10, **kwargs)
        device.start()
        stream = wire_prefill(last_stream(stub), device)
        try:
            yield device, stream
        finally:
            device.stop()


def xruns(device: PortAudioDevice) -> int:
    return int(
        attr_or_fail(
            device,
            "device_xruns",
            "T-0035 criterion 1: callbacks the backend flagged `output_underflow`, counted "
            "apart from our own empty ring",
        )
    )


def test_a_device_xrun_with_a_full_ring_is_counted() -> None:
    """Adversarial, F2: the ring was full and the *graph* still starved (criterion 1).

    One callback, ``frames == taken == 480`` (we had every sample it asked for) and
    ``status.output_underflow`` set. The device flagged the whole callback, so the counter grows
    by 480 frames and ``device_xruns`` by exactly 1. Today's code computes ``frames - taken == 0``
    and reports nothing at all, which is why the only counter T-0006 ruling A22 put on the
    protocol cannot see a real xrun.
    """
    with rig() as (device, stream):
        before, before_x = device.underflow_frames, xruns(device)
        stream.drive(1, frames=FRAMES, taken=FRAMES, status=True)
        assert device.underflow_frames - before == FRAMES, (
            f"the backend reported output_underflow for a whole {FRAMES}-frame callback and "
            f"underflow_frames grew by {device.underflow_frames - before}: a device xrun is "
            f"invisible in the counter audio_io.underruns reads (criterion 1, finding F2)"
        )
        assert xruns(device) - before_x == 1, (
            f"device_xruns grew by {xruns(device) - before_x}, not 1: it counts *callbacks* the "
            f"device flagged, so 'our ring starved' and 'the graph starved' stay apart"
        )


def test_an_empty_ring_without_the_flag_still_counts_the_shortfall() -> None:
    """The behaviour that exists today is not lost (criterion 1, second half).

    120 of 480 frames available, no flag: the device had to invent 360 frames, and that is what
    the counter says. ``device_xruns`` stays 0 — the graph was fine, we were late.
    """
    with rig() as (device, stream):
        stream.drive(1, frames=FRAMES, taken=120, status=False)
        assert device.underflow_frames == FRAMES - 120 == 360, (
            f"underflow_frames is {device.underflow_frames}, expected 360 = 480 - 120 frames the "
            f"device had to invent because our ring was short (criterion 1)"
        )


def test_the_two_causes_are_not_double_counted() -> None:
    """Criterion 1: the increment is ``max(shortfall, frames)``, never their sum.

    120 frames available *and* the device flagged the callback: 480, not 840. Adding the two
    causes would make ``audio_io.underruns`` report more starved audio than the stream contains,
    and the number is quoted in T-0023's regime blocks.
    """
    with rig() as (device, stream):
        stream.drive(1, frames=FRAMES, taken=120, status=True)
        assert device.underflow_frames == FRAMES, (
            f"underflow_frames is {device.underflow_frames}; a flagged callback that was also "
            f"short counts {FRAMES} frames (max of the two causes), not 360 + 480 = 840"
        )
        assert xruns(device) == 1


def test_the_counter_is_monotone_across_a_hundred_mixed_callbacks() -> None:
    """Criterion 2: ``audio_io``'s delta read is never negative, and the total is hand-computed.

    n = 100 callbacks of 480 frames, the mix below shuffled by ``random.Random(0)`` (the order
    cannot change a sum of per-callback contributions, which is the point: the assertion is an
    arithmetic oracle, not a replay of the loop):

    ====================================  ===  =================  =======
    callback                                n  per callback       total
    ====================================  ===  =================  =======
    ring full, no flag                     50  0                        0
    ring full, flag set                    20  480                  9 600
    120 frames short of 480, no flag       20  360                  7 200
    120 frames short of 480, flag set      10  480                  4 800
    ====================================  ===  =================  =======

    Expected total: 9 600 + 7 200 + 4 800 = **21 600 frames** = 450.0 ms at 48 kHz, and
    ``device_xruns`` = 20 + 10 = **30**.
    """
    schedule = (
        [(FRAMES, False)] * 50 + [(FRAMES, True)] * 20 + [(120, False)] * 20 + [(120, True)] * 10
    )
    random.Random(0).shuffle(schedule)
    with rig() as (device, stream):
        previous = device.underflow_frames
        for index, (taken, flag) in enumerate(schedule):
            stream.drive(1, frames=FRAMES, taken=taken, status=flag)
            now = device.underflow_frames
            assert now >= previous, (
                f"underflow_frames went from {previous} to {now} at callback {index}: "
                f"audio_io.underruns reads it as a delta (criterion 2) and would report a "
                f"negative number of starved frames"
            )
            previous = now
        assert device.underflow_frames == 21_600, (
            f"underflow_frames is {device.underflow_frames}, expected 21600 frames "
            f"(20·480 + 20·360 + 10·480 = 450.0 ms at {DEVICE_RATE} Hz) — see the table above"
        )
        assert xruns(device) == 30, (
            f"device_xruns is {xruns(device)}, expected 30 flagged callbacks out of 100"
        )


def test_the_duplex_callback_counts_the_same_way() -> None:
    """Criterion 1 reaches the capture path too: the duplex callback delegates, so it must.

    The duplex stream is what every rig run uses (T-0021 criterion 1: one stream, one clock), so
    a fix that only touched the output-only callback would be invisible on the hardware this
    task exists to make honest. ``input_device`` makes the device duplex without naming a
    PipeWire node, so no ``pw-dump`` is involved (criterion 4).
    """
    with rig(input_device="default") as (device, stream):
        assert stream.kind == "duplex", "the device did not open a duplex stream"
        stream.drive(2, frames=FRAMES, taken=FRAMES, status=True)
        assert device.underflow_frames == 2 * FRAMES, (
            f"two flagged duplex callbacks of {FRAMES} frames left underflow_frames at "
            f"{device.underflow_frames}, expected {2 * FRAMES} (criterion 1)"
        )
        assert xruns(device) == 2


def test_the_new_counters_stay_off_the_audio_device_protocol() -> None:
    """Criterion 10(c): ``device_xruns`` and ``quantum_source`` are ``PortAudioDevice``-only.

    ``tests/unit/test_audio_capture_ownership.py`` pins the protocol's member set and must pass
    **unedited**; adding either attribute to ``AudioDevice`` would make every fake device in the
    tree stop conforming. They are backend self-reports, like ``played_ms()`` in T-0006.
    """
    declared = set(getattr(AudioDevice, "__protocol_attrs__", set()))
    for klass in AudioDevice.__mro__:
        declared |= set(getattr(klass, "__annotations__", {}))
    for name in ("device_xruns", "quantum_source"):
        assert name not in declared, (
            f"{name!r} was added to the AudioDevice protocol; criterion 10(c) keeps it on "
            f"PortAudioDevice only (the fakes and FakeAudioDevice do not have it)"
        )
    device = PortAudioDevice()
    for name in ("device_xruns", "quantum_source"):
        assert hasattr(device, name), (
            f"PortAudioDevice has no {name!r}: criterion 1/3 put both on the real backend, "
            f"before the stream is ever opened"
        )
    assert isinstance(device.device_xruns, int) and device.device_xruns == 0
    assert isinstance(device.quantum_source, str)
