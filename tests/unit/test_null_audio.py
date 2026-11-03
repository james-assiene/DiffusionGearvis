"""T-0039 group C: ``local_astra/conductor/null_audio.py`` -- a playback device that models time.

Criteria 15 and 16 of ``docs/tasks/T-0039.md``. ``NullAudioDevice`` is a complete ``AudioDevice``
(``conductor/audio_device.AudioDevice``, docs/02 section 2.7) that renders nothing and **paces the
rendering at real time on an injected clock**: 1 ms of audio per 1 ms of clock, never faster. The
whole point of the flag is that a silent turn takes as long as the speech would have, so
``played_ms``, ``played_text``, the unplayed strike and the drain rule keep their meaning
(docs/tasks/T-0039.md section 5, "The null device models time; it does not abolish it").

**The failure mode these rows are aimed at** (the task names it): a null device that renders
instantly. Every queue would be empty the moment it was written, so ``PipeWireAudioIO`` would
find its drain condition satisfied before a word was "spoken", every silent turn would be a
drain-before-speech, and every ``played_ms`` in ``turns.jsonl`` would be a fiction. The
adversarial rows at the bottom are the ones that catch it.

**Regime** (docs/tasks/T-0039.md section 6): virtual time only -- ``tests/conftest.VirtualClock``,
injected -- n=1 per row, no seed, no wall clock in any assertion. The one place wall time appears
is ``write()``'s blocking row, where a *thread* has to be observed not returning; the deadline
there is a generous ceiling, not a measurement. No number in this module is a latency claim.

Seams this pins (docs/tasks/T-0039.md section 3 names the module and the class; the rest is the
``AudioDevice`` protocol, which is not this task's to change):

    local_astra.conductor.null_audio.NullAudioDevice(clock=<Clock with .now_ms>)
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from local_astra.contracts.events import PlaybackDrained, PlaybackStopped
from tests.conftest import VirtualClock
from tests.fakes.audio_device import (
    EventSink,
    PcmUtterance,
    attr_or_fail,
    call,
    import_or_fail,
    play,
    played_ms,
    tone,
    words_of,
)

WHY = "T-0039 criterion 15: local_astra/conductor/null_audio.py provides NullAudioDevice"

DEVICE_RATE = 48_000
#: Never advance the device clock by more than one block: ``PipeWireAudioIO`` keeps one block of
#: lead (``_target_lead_ms``), so a bigger jump starves a device the test never meant to starve.
STEP_MS = 10.0

SOURCE = "one two three"
WORDS = words_of([("one", 0, 300), ("two", 400, 700), ("three", 800, 1100)])
UTTERANCE_MS = 1200.0


def null_device(clock: VirtualClock, **kwargs: Any) -> Any:
    """``NullAudioDevice(clock=...)`` -- a missing module or signature is an assertion."""
    module = import_or_fail("local_astra.conductor.null_audio", WHY)
    factory = attr_or_fail(module, "NullAudioDevice", WHY)
    try:
        return factory(clock=clock, **kwargs)
    except TypeError as exc:
        raise AssertionError(
            f"T-0039 criterion 15: NullAudioDevice(clock=...) takes its clock by injection so "
            f"the fast suite is not paced by wall time (section 5); the call raised {exc}"
        ) from exc


def require(device: Any, *names: str) -> None:
    missing = [name for name in names if not hasattr(device, name)]
    assert missing == [], (
        f"T-0039 criterion 15: NullAudioDevice is a *complete* AudioDevice "
        f"(conductor/audio_device.py); it is missing {missing}"
    )


def block(ms: float = STEP_MS) -> np.ndarray:
    """One block of mono f32 silence at 48 kHz -- what the writer hands a device."""
    return np.zeros(round(ms * DEVICE_RATE / 1000.0), dtype=np.float32)


class Writer(threading.Thread):
    """Writes blocks from a worker thread, counting the ones ``write()`` accepted.

    ``AudioDevice.write`` "may block while the device is full", so the writer cannot be the
    thread that advances the clock -- that is exactly the deadlock the real writer thread
    (``audio_io.py``'s ``_writer``) avoids by living off the event loop.
    """

    def __init__(self, device: Any, blocks: int, *, block_ms: float = STEP_MS) -> None:
        super().__init__(daemon=True)
        self.device = device
        self.blocks = blocks
        self.block_ms = block_ms
        self.accepted = 0
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            for _ in range(self.blocks):
                self.device.write(block(self.block_ms))
                self.accepted += 1
        except BaseException as exc:  # reported to the test thread, not swallowed
            self.error = exc

    def settled(self, *, quiet_s: float = 0.05, timeout_s: float = 2.0) -> int:
        """Wait until ``accepted`` has been still for ``quiet_s``; return it."""
        deadline = time.monotonic() + timeout_s
        last, since = self.accepted, time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(0.002)
            if self.accepted != last:
                last, since = self.accepted, time.monotonic()
            elif time.monotonic() - since >= quiet_s:
                break
        if self.error is not None:
            raise AssertionError(
                f"T-0039 criterion 15: write() of a {self.block_ms:.0f} ms block raised "
                f"{type(self.error).__name__}: {self.error}"
            )
        return self.accepted


# --- criterion 15: the protocol, and the two numbers that carry their provenance ----------------


def test_it_is_a_complete_audio_device() -> None:
    """Criterion 15: it satisfies ``AudioDevice`` -- the same protocol ``PortAudioDevice`` does."""
    from local_astra.conductor.audio_device import AudioDevice

    device = null_device(VirtualClock())
    assert isinstance(device, AudioDevice), (
        "criterion 15: NullAudioDevice must satisfy the runtime_checkable AudioDevice protocol "
        "(conductor/audio_device.py) -- PipeWireAudioIO is handed it unchanged, which is what "
        "keeps --silent a device swap and not a policy branch (section 5)"
    )


def test_the_quantum_is_zero_and_says_where_that_came_from() -> None:
    """Criterion 15 / T-0035 ruling A13: a number without its provenance is not a number."""
    device = null_device(VirtualClock())
    require(device, "quantum_ms", "quantum_source")
    assert device.quantum_ms == 0.0, (
        f"criterion 15: a device that drives no graph has no quantum to report; it says "
        f"{device.quantum_ms!r}"
    )
    assert device.quantum_source == "unknown", (
        f"criterion 15 (T-0035 ruling A13): 0.0 always comes with the provenance 'unknown', "
        f"never with a source that suggests somebody measured something; it says "
        f"{device.quantum_source!r}"
    )


def test_it_names_no_sound_library_and_no_pipewire_tool() -> None:
    """Criterion 15: it imports no ``sounddevice`` and touches no PipeWire graph.

    ``tests/unit/test_audio_ownership.py`` already fences ``local_astra/`` to one module that may
    name a sound library (``audio_device.py``); this is the same fence stated for the new file, so
    a "null" device that quietly opens a stream to discover a latency cannot pass.
    """
    path = Path(__file__).resolve().parents[2] / "local_astra" / "conductor" / "null_audio.py"
    assert path.is_file(), WHY
    code = "\n".join(
        line for line in path.read_text().splitlines() if not line.lstrip().startswith("#")
    )
    for forbidden in ("sounddevice", "pw-dump", "pw-cli", "pw-top", "subprocess"):
        assert forbidden not in code, (
            f"criterion 15: null_audio.py must name no sound library and no PipeWire tool; it "
            f"mentions {forbidden!r} outside a full-line comment"
        )


# --- criterion 15: rendering is paced by the injected clock, and by the audio --------------------


def test_an_idle_device_renders_nothing_however_long_the_clock_runs() -> None:
    """Criterion 15: 1 ms of *audio* per 1 ms of clock -- a device fed nothing renders nothing.

    A device whose ``rendered_ms()`` is "the clock" rather than "the audio" reports a position
    inside a stream it was never given, and ``audio_io``'s ``played_ms`` (clamped to
    ``enqueued_ms``) would start every run already finished.
    """
    clock = VirtualClock()
    device = null_device(clock)
    require(device, "start", "rendered_ms", "queued_ms", "underflow_frames")
    device.start()
    clock.advance(500)
    assert float(device.rendered_ms()) == 0.0, (
        f"criterion 15: nothing was written, so nothing can have been rendered; the device "
        f"reports {device.rendered_ms()!r} after 500 ms of clock"
    )
    assert float(device.queued_ms()) == 0.0, (
        f"criterion 15: an empty queue is 0 ms queued, not {device.queued_ms()!r}"
    )
    assert int(device.underflow_frames) == 0, (
        f"criterion 15: the null device never invents frames -- underflow_frames stays 0, it is "
        f"{device.underflow_frames}"
    )


def test_rendered_ms_advances_exactly_one_ms_per_ms_of_clock() -> None:
    """Criterion 15: the exact relation (section 6: an exact relation, not a tolerance).

    Written as *deltas* so it holds for a backend that subtracts an output latency: what is
    pinned is the rate, on a device that is kept fed, and that no advance of the clock renders
    more audio than the clock allowed.
    """
    clock = VirtualClock()
    device = null_device(clock)
    require(device, "start", "write", "rendered_ms", "queued_ms")
    device.start()
    writer = Writer(device, 200)  # 2 s of audio, offered; the device takes what it can hold
    writer.start()
    writer.settled()

    before = float(device.rendered_ms())
    trace: list[tuple[int, float]] = []
    for _ in range(20):  # 200 ms in one-block steps
        clock.advance(int(STEP_MS))
        writer.settled(quiet_s=0.01, timeout_s=1.0)
        trace.append((clock.now_ms, float(device.rendered_ms())))
    after = trace[-1][1]

    assert after - before == pytest.approx(200.0, abs=0.001), (
        f"criterion 15: 200 ms of clock renders exactly 200 ms of audio on a device that was "
        f"kept fed; it rendered {after - before} ms. Trace (clock_ms, rendered_ms): {trace}"
    )
    for now_ms, rendered in trace:
        assert rendered <= float(now_ms) + 0.001, (
            f"criterion 15: never faster than the clock -- at clock {now_ms} ms the device "
            f"claims {rendered} ms rendered. Trace: {trace}"
        )
    assert [r for _, r in trace] == sorted(r for _, r in trace), (
        f"AudioDevice.rendered_ms is monotonic and non-decreasing; trace: {trace}"
    )


def test_rendering_stops_at_the_audio_it_was_given_and_counts_no_underflow() -> None:
    """Criterion 15: rendered position never runs past what was written; ``underflow_frames`` 0.

    The second half is the criterion's own words ("``underflow_frames`` stays 0") and it is what
    distinguishes a silent *device* from a starved one: nobody may read an xrun out of a run that
    had nothing to play.
    """
    clock = VirtualClock()
    device = null_device(clock)
    require(device, "start", "write", "rendered_ms", "queued_ms", "underflow_frames")
    device.start()
    for _ in range(3):  # 30 ms, small enough that no modelled queue can be full
        device.write(block())
    for _ in range(50):  # 500 ms of clock against 30 ms of audio
        clock.advance(int(STEP_MS))

    rendered = float(device.rendered_ms())
    assert rendered == pytest.approx(30.0, abs=0.001), (
        f"criterion 15: 30 ms was written and 500 ms of clock passed; the device rendered "
        f"{rendered} ms -- a null device that renders wall time reports 500"
    )
    assert float(device.queued_ms()) == 0.0, (
        f"criterion 15: everything written has left the queue; it says {device.queued_ms()!r}"
    )
    assert int(device.underflow_frames) == 0, (
        f"criterion 15: underflow_frames stays 0 even across 470 ms with nothing to play; it is "
        f"{device.underflow_frames}"
    )


def test_queued_ms_falls_to_zero_only_as_the_modelled_queue_drains() -> None:
    """Criterion 15: ``queued_ms()`` is what the device has not taken yet, paced by the clock.

    ``audio_io._is_drained`` reads exactly this number ("the device has taken everything we
    wrote", ruling A15), so a device that reports 0 the instant it is written turns every
    ``mark_drain()`` into an immediate drain.
    """
    clock = VirtualClock()
    device = null_device(clock)
    require(device, "start", "write", "queued_ms")
    device.start()
    for _ in range(5):  # 50 ms
        device.write(block())

    assert float(device.queued_ms()) == pytest.approx(50.0, abs=0.001), (
        f"criterion 15: 50 ms was handed over and no clock has passed, so 50 ms is queued; the "
        f"device says {device.queued_ms()!r}"
    )
    clock.advance(20)
    assert float(device.queued_ms()) == pytest.approx(30.0, abs=0.001), (
        f"criterion 15: after 20 ms of clock, 30 ms is still queued; it says {device.queued_ms()!r}"
    )
    clock.advance(29)
    assert float(device.queued_ms()) > 0.0, (
        "criterion 15: one millisecond of audio is still to come, so the queue is not empty yet; "
        f"the device says {device.queued_ms()!r} at clock {clock.now_ms} ms"
    )
    clock.advance(1)
    assert float(device.queued_ms()) == 0.0, (
        f"criterion 15: the 50 ms has played out; the queue is empty, not {device.queued_ms()!r}"
    )


def test_write_blocks_the_writer_while_the_modelled_queue_is_full() -> None:
    """Criterion 15: ``write()`` blocks when full -- the back-pressure the real device has.

    Without it a "null" device is an infinite sink: ``audio_io``'s writer would hand it a whole
    utterance in one pass, ``queued_ms()`` would be meaningless and the run would be bounded by
    nothing. The ceiling below is wall time and is a deadline, never a measurement.
    """
    clock = VirtualClock()
    device = null_device(clock)
    require(device, "start", "write", "queued_ms")
    device.start()
    writer = Writer(device, 10_000)  # 100 s of audio offered against a modelled queue
    writer.start()
    filled = writer.settled()

    assert filled >= 1, "criterion 15: the device must accept at least one block"
    assert filled < 10_000, (
        f"criterion 15: write() must block while the modelled queue is full; the device "
        f"swallowed all {filled} blocks (100 s of audio) without the clock moving -- that is an "
        f"infinite sink, not a device"
    )
    queued_before = float(device.queued_ms())
    assert queued_before > 0.0, "the blocks it took are queued, not discarded"

    clock.advance(int(STEP_MS))
    released = writer.settled()
    assert released > filled, (
        f"criterion 15: one block of audio played out, so the blocked write must complete; the "
        f"writer is still stuck at {released} blocks (it was {filled} before the clock moved)"
    )
    assert float(device.queued_ms()) == pytest.approx(queued_before, abs=STEP_MS + 0.001), (
        "criterion 15: the queue refilled by what it released, not more -- it is "
        f"{device.queued_ms()!r} against {queued_before}"
    )


def test_capture_is_a_no_op_and_the_callback_is_never_invoked() -> None:
    """Criterion 15/19: ``--silent`` means no microphone; the capture half delivers nothing."""
    clock = VirtualClock()
    device = null_device(clock)
    require(device, "start_capture", "stop_capture", "captured_ms", "set_capture_callback")
    calls: list[tuple[Any, float, bool]] = []
    device.set_capture_callback(lambda samples, adc_ms, xrun: calls.append((samples, adc_ms, xrun)))
    device.start_capture()
    first = float(device.captured_ms())
    for _ in range(100):
        clock.advance(int(STEP_MS))
    assert calls == [], (
        f"criterion 15: capture is a no-op that delivers no callback; it delivered {len(calls)} "
        f"buffers over 1 s of clock (criterion 19: a --silent session opens no microphone)"
    )
    assert float(device.captured_ms()) >= first, (
        "AudioDevice.captured_ms is monotonic and non-decreasing"
    )
    device.stop_capture()
    assert calls == [], "...and nothing is flushed on the way out either"


# --- criterion 16 (adversarial): the drain rule under a device that renders nothing --------------


def utterance() -> PcmUtterance:
    return PcmUtterance("u1", samples=tone(UTTERANCE_MS), words=WORDS, text=SOURCE)


def make_io(device: Any) -> Any:
    from local_astra.conductor.audio_io import PipeWireAudioIO

    try:
        return PipeWireAudioIO(device)
    except TypeError as exc:
        raise AssertionError(
            f"criterion 16: PipeWireAudioIO(NullAudioDevice(...)) is the whole of --silent "
            f"(section 5: a device swap, not a policy branch); it raised {exc}"
        ) from exc


async def advance(clock: VirtualClock, ms: float, *, step_ms: float = STEP_MS) -> None:
    """Advance the injected clock in one-block steps, letting the writer thread keep up."""
    left = float(ms)
    await asyncio.sleep(0.002)
    while left > 0:
        step = min(step_ms, left)
        clock.advance(int(step))
        left -= step
        await asyncio.sleep(0.002)


async def test_a_queue_that_empties_under_the_null_device_is_not_a_drain() -> None:
    """Criterion 16 / docs/02 section 4 spec delta 1 (T-0006): drain is declared, never inferred.

    The adversarial row the task names: a 400 ms stall mid-run -- the device runs dry with no
    ``mark_drain()`` -- must produce no ``PlaybackDrained``. Under a null device this is the shape
    every silent turn has between utterances, and a drain here ends the turn mid-sentence
    (``SPEAKING + playback.drained and llm.done -> IDLE``).
    """
    clock = VirtualClock()
    io = make_io(null_device(clock))
    sink = EventSink(io)
    await play(io, utterance(), source=SOURCE)
    await advance(clock, UTTERANCE_MS + 400.0)
    await sink.sync()
    assert sink.of_type(PlaybackDrained) == [], (
        "criterion 16: everything queued played out and the writer stalled for 400 ms, but no "
        "mark_drain() was called -- that is a stall, not the end of a stream "
        f"(docs/02 section 4 spec delta 1). The module emitted {sink.of_type(PlaybackDrained)}"
    )
    await sink.close()


async def test_the_drain_waits_for_the_audio_time_of_the_last_block() -> None:
    """Criterion 16, the failure mode named in the task: an instantly-rendering null device.

    ``mark_drain()`` is called **while audio is still to come**. A device that renders instantly
    satisfies ``audio_io._is_drained`` (``queued_ms() == 0``) on the very next writer pass, and
    the turn ends before a word of it was "spoken". The drain may only arrive once the clock has
    carried the last block's audio time.
    """
    clock = VirtualClock()
    io = make_io(null_device(clock))
    sink = EventSink(io)
    await play(io, utterance(), source=SOURCE)

    await advance(clock, 300.0)
    await call(io, "mark_drain", "T-0006 criterion 7: the end of stream is declared")
    await advance(clock, 200.0)
    await sink.sync()
    early = sink.of_type(PlaybackDrained)
    assert early == [], (
        f"criterion 16: 500 ms of a {UTTERANCE_MS:.0f} ms utterance has been rendered and "
        f"mark_drain() was called at 300 ms; a drain now means the device reported an empty "
        f"queue for audio nobody has heard. Got {early}"
    )
    mid = await played_ms(io)
    assert 400.0 <= mid <= 560.0, (
        f"criterion 16: played_ms tracks the injected clock in silent mode (500 ms in); it is {mid}"
    )

    await advance(clock, UTTERANCE_MS - 500.0 + 100.0)
    await sink.sync()
    drained = sink.of_type(PlaybackDrained)
    assert len(drained) == 1, (
        f"criterion 16: once the last block's audio time has elapsed the run drains, exactly "
        f"once; got {drained}"
    )
    assert drained[0].played_text == SOURCE, (
        f"criterion 16: played_text is the committed source of what 'rendered' -- a silent run "
        f"still says what was spoken (docs/02 section 4: history never contains assistant text "
        f"that was not played). It said {drained[0].played_text!r}"
    )
    assert drained[0].played_ms == int(UTTERANCE_MS), (
        f"criterion 16: a drained run reports the enqueued total (T-0006 ruling A18); it "
        f"reported {drained[0].played_ms}"
    )
    await sink.close()


async def test_played_ms_tracks_the_clock_and_never_goes_backwards() -> None:
    """Criterion 16: the existing ``AudioIO`` rows hold over the null device."""
    clock = VirtualClock()
    io = make_io(null_device(clock))
    await play(io, utterance(), source=SOURCE)
    await advance(clock, 400.0)
    first = await played_ms(io)
    assert 360.0 <= first <= 420.0, (
        f"criterion 16: 400 ms of clock is 400 ms played in silent mode (real-time pacing is the "
        f"decision, section 5 and ambiguity 3); played_ms={first}"
    )
    await advance(clock, 300.0)
    second = await played_ms(io)
    assert second >= first, f"played_ms went backwards {first} -> {second}"
    assert second <= UTTERANCE_MS, f"played_ms={second} exceeds the utterance"


async def test_flush_is_idempotent_and_terminal_under_the_null_device() -> None:
    """Criterion 16: ``flush()`` is still one ``PlaybackStopped`` and the same number twice."""
    clock = VirtualClock()
    io = make_io(null_device(clock))
    sink = EventSink(io)
    await play(io, utterance(), source=SOURCE)
    await advance(clock, 500.0)

    first = await call(io, "flush", "AudioIO protocol")
    second = await call(io, "flush", "AudioIO protocol")
    await sink.sync()
    assert first == second, f"flush() is idempotent (contracts/commands.py); {first} != {second}"
    stopped = sink.of_type(PlaybackStopped)
    assert len(stopped) == 1, f"exactly one PlaybackStopped per run, got {stopped}"
    assert stopped[0].played_ms == first
    assert sink.of_type(PlaybackDrained) == [], "a flush is not a drain"
    await sink.close()
