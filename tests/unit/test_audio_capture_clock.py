"""The capture clock, the block cadence, xruns and backpressure: T-0021 criteria 1-5, 10.

Everything here is pure: ``FakeAudioDevice`` + ``VirtualClock``, no PortAudio, no PipeWire, no
rig (regime: *Unit* in docs/tasks/T-0021.md). The device's ADC position is the oracle — a module
that stamps a block when its callback ran, or when a coroutine resumed, disagrees with it the
moment delivery is late, which is the whole point of criterion 3.

Seams these tests fix, because the task names the behaviour and not the signatures (reported as
such): ``PipeWireAudioIO(device)`` gains ``capture_frames()`` yielding
``CaptureBlock(samples, t_ms, xrun, gated, seq)``, ``stt_blocks()``, ``capture_stats()`` and
``close_capture()``; the device gains ``start_capture()``/``stop_capture()``/``captured_ms()``/
``input_latency_ms`` plus one delivery style — ``set_capture_callback(cb)`` or ``read_capture()``,
both handing over ``(samples, adc_ms, xrun)`` (``tests/fakes/audio_device.py``). Sync or async is
accepted for the call-style seams (``maybe_await``); the names are not negotiable without a task
amendment.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.contracts.speech import CAPTURE_BLOCK_MS
from tests.conftest import VirtualClock
from tests.fakes.audio_device import (
    BLOCK_SAMPLES,
    DEVICE_RATE,
    STT_BLOCK_MS,
    STT_RATE,
    WHY_CAPTURE,
    CaptureSink,
    EventSink,
    FakeAudioDevice,
    PcmUtterance,
    attr_or_fail,
    call,
    capture_out,
    capture_pattern,
    capture_settle,
    capture_stats,
    concat,
    import_capture,
    make_audio_io,
    play,
    play_out,
    played_ms,
    samples_of,
    stat,
    t_ms_of,
    tone,
)

#: criterion 3's bound on the timestamp: |t_ms − true capture time| ≤ block_ms + input_latency_ms.
INPUT_LATENCY_MS = 15.0
TIMESTAMP_BOUND_MS = CAPTURE_BLOCK_MS + INPUT_LATENCY_MS
#: criterion 5: the queue holds at least 32 blocks (320 ms).
MIN_QUEUE_BLOCKS = 32
#: criterion 4: at most 500 ms of continuous substituted silence, then a named error.
MAX_SUBSTITUTED_MS = 500.0


def rig(clock: VirtualClock | None = None, **kwargs: object) -> tuple[object, FakeAudioDevice]:
    device = FakeAudioDevice(clock if clock is not None else VirtualClock(), **kwargs)  # type: ignore[arg-type]
    return make_audio_io(device), device


def loud_pattern(start: int, n: int) -> np.ndarray:
    """``capture_pattern`` lifted off zero (0.5 ± 0.03): never quiet, still index-identifiable.

    Used where a test has to tell *substituted silence* from *audio*: a signal that crosses zero
    would make "this region is silent" true by accident for a few hundred samples, and a
    decimator passes DC through unchanged, so the offset survives the 48 → 24 kHz stage.
    """
    return (0.5 + 0.06 * capture_pattern(start, n)).astype(np.float32)


def assert_cadence(blocks: list[object], why: str) -> None:
    """Consecutive non-xrun blocks are exactly ``CAPTURE_BLOCK_MS`` apart (criterion 3)."""
    stamps = [t_ms_of(b) for b in blocks]
    bad = [
        (i, stamps[i], stamps[i + 1])
        for i in range(len(stamps) - 1)
        if abs(stamps[i + 1] - stamps[i] - CAPTURE_BLOCK_MS) > 1e-6
    ]
    assert bad == [], (
        f"{why}: {len(bad)} of {len(stamps) - 1} consecutive blocks are not "
        f"{CAPTURE_BLOCK_MS} ms apart, first at index {bad[0][0]}: "
        f"{bad[0][1]:.3f} -> {bad[0][2]:.3f} ms (criterion 3)"
    )


async def test_block_timestamps_come_from_the_capture_clock_not_arrival() -> None:
    """Criterion 3, the headline: a 100 ms delivery stall, then ten blocks in one burst.

    No audio is lost — the ADC ran the whole time, the buffers were simply handed over late (the
    fake's ``stall_capture``). The ten blocks must be stamped 10 ms apart and must cover the
    stall window, because their timestamps come from the device's capture clock. An
    implementation that stamps with ``loop.time()``/``time.monotonic()`` when the callback ran
    gives ten nearly identical stamps at the burst: its consecutive differences are ~0 instead of
    10 ms, and its error against the true capture time is ~100 ms instead of ≤ 25 ms.

    The oracle is ``device.true_capture_ms(index)`` = capture start + index / 48 kHz, which comes
    from the virtual clock and the sample rate and owes the module nothing.
    """
    clock = VirtualClock()
    io, device = rig(clock, capture_buffer_frames=BLOCK_SAMPLES, input_latency_ms=INPUT_LATENCY_MS)
    sink = CaptureSink(io, "raw")

    await capture_out(device, 100.0, sink)
    assert len(sink) == 10, f"10 blocks in the first 100 ms, got {len(sink)}"

    device.stall_capture(100.0)
    await capture_out(device, 90.0, sink)
    assert len(sink) == 10, (
        f"the device delivered {len(sink) - 10} buffers during the stall — the burst case this "
        f"test exists for is not set up (tests/fakes/audio_device.py::stall_capture)"
    )

    await capture_out(device, 20.0, sink)  # the hold expires: ten buffers land at once
    blocks = list(sink.items)
    sink.raise_error()
    await sink.close()

    assert len(blocks) == 21, (
        f"210 ms of capture at {CAPTURE_BLOCK_MS} ms per block is 21 blocks, got {len(blocks)}: "
        f"the stalled audio was dropped rather than delivered late"
    )
    assert_cadence(blocks, "after a 100 ms delivery stall and a ten-block burst")

    errors = [
        (i, t_ms_of(b), device.true_capture_ms(i * BLOCK_SAMPLES))
        for i, b in enumerate(blocks)
        if abs(t_ms_of(b) - device.true_capture_ms(i * BLOCK_SAMPLES)) > TIMESTAMP_BOUND_MS
    ]
    assert errors == [], (
        f"{len(errors)} blocks are stamped further than {TIMESTAMP_BOUND_MS} ms "
        f"(block_ms + input_latency_ms) from when the ADC actually took them; first: block "
        f"{errors[0][0]} says t_ms={errors[0][1]:.1f} for audio captured at "
        f"{errors[0][2]:.1f} ms (criterion 3). Ten identical stamps here means the timestamp is "
        f"the arrival time of the burst, not the capture clock."
    )

    covered = [t_ms_of(b) for b in blocks[10:20]]
    assert max(covered) - min(covered) == pytest.approx(90.0, abs=1e-6), (
        f"the ten blocks captured during the stall span {max(covered) - min(covered):.1f} ms of "
        f"timeline (want 90 ms between the first and the last of ten 10 ms blocks); stamps: "
        f"{[round(c, 2) for c in covered]}"
    )
    for i, block in enumerate(blocks):
        expected = capture_pattern(i * BLOCK_SAMPLES, BLOCK_SAMPLES)
        assert np.array_equal(samples_of(block), expected), (
            f"block {i} does not carry the audio the ADC took at sample {i * BLOCK_SAMPLES}: "
            f"the burst was re-ordered or the stalled audio was replaced"
        )


async def test_blocks_are_exactly_10ms_whatever_the_device_quantum() -> None:
    """Criterion 2: the graph quantum here is 1024/48000 ≈ 21.33 ms; blocks are still 10 ms.

    Conservation, in the form the task states it: samples out + carry == samples in. The carry is
    whatever has not yet filled a block, and it is never yielded short.
    """
    clock = VirtualClock()
    io, device = rig(
        clock,
        capture_buffer_frames=1024,
        quantum_ms=1000.0 * 1024 / DEVICE_RATE,
    )
    sink = CaptureSink(io, "raw")
    await capture_out(device, 1000.0, sink)
    blocks = list(sink.items)
    stats = await capture_stats(io)
    sink.raise_error()
    await sink.close()

    delivered = device.delivered_frames
    assert delivered == 46 * 1024, (
        f"the fake handed over {delivered} frames; 1000 ms at 48 kHz is 48000 frames and "
        f"48000 // 1024 = 46 buffers (the test's own arithmetic is wrong if this fails)"
    )
    sizes = {int(samples_of(b).size) for b in blocks}
    assert sizes == {BLOCK_SAMPLES}, (
        f"blocks of {sorted(sizes)} samples came out of a device delivering 1024-frame buffers; "
        f"criterion 2 says always {BLOCK_SAMPLES} (= {CAPTURE_BLOCK_MS} ms at 48 kHz)"
    )
    assert len(blocks) == delivered // BLOCK_SAMPLES, (
        f"{len(blocks)} blocks for {delivered} delivered frames; "
        f"{delivered // BLOCK_SAMPLES} whole blocks fit (criterion 2)"
    )
    # Conservation by hand, not by restating the assertion above: 46 buffers of 1024 frames is
    # 47104 frames, and 47104 = 98 * 480 + 64. So 98 whole blocks come out and 64 samples stay
    # in the carry, never yielded short (criterion 2).
    carry = delivered - len(blocks) * BLOCK_SAMPLES
    assert (delivered, len(blocks), carry) == (47_104, 98, 64), (
        f"{len(blocks)} blocks and a carry of {carry} samples out of {delivered} delivered "
        f"frames; the arithmetic is 47104 = 98 x {BLOCK_SAMPLES} + 64 (samples out + carry == "
        f"samples in)"
    )
    assert np.array_equal(concat(blocks), capture_pattern(0, len(blocks) * BLOCK_SAMPLES)), (
        "the re-blocked stream is not the device's stream: a sample was lost, duplicated or "
        "re-ordered at a 1024-frame buffer boundary"
    )
    assert float(stat(stats, "quantum_ms")) == pytest.approx(1000.0 * 1024 / DEVICE_RATE), (
        f"capture_stats().quantum_ms is {stat(stats, 'quantum_ms')!r}; criterion 2 says the "
        f"**negotiated** quantum is reported, never the requested {CAPTURE_BLOCK_MS} ms"
    )
    assert int(stat(stats, "blocks")) == len(blocks), "criterion 10: CaptureStats.blocks"


@settings(
    max_examples=50,  # each example drives a real event loop (T-0006 ruling A3); regime amended
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    buffer_frames=st.sampled_from([64, 160, 480, 512, 1024, 2048, 4800]),
    tens_of_ms=st.integers(min_value=2, max_value=30),
)
async def test_no_sample_is_lost_or_duplicated_across_reblocking(
    buffer_frames: int, tens_of_ms: int
) -> None:
    """Criterion 2 as a property: any device buffer size, any stream length, same audio out."""
    clock = VirtualClock()
    io, device = rig(clock, capture_buffer_frames=buffer_frames)
    sink = CaptureSink(io, "raw")
    await capture_out(device, 10.0 * tens_of_ms, sink)
    blocks = list(sink.items)
    sink.raise_error()
    await sink.close()

    expected_blocks = device.delivered_frames // BLOCK_SAMPLES
    assert len(blocks) == expected_blocks, (
        f"buffer={buffer_frames} frames, {10 * tens_of_ms} ms: {len(blocks)} blocks for "
        f"{device.delivered_frames} delivered frames (want {expected_blocks})"
    )
    assert np.array_equal(concat(blocks), capture_pattern(0, expected_blocks * BLOCK_SAMPLES)), (
        f"buffer={buffer_frames} frames: the concatenated blocks are not the device's stream"
    )


async def test_no_short_or_long_block_is_ever_yielded_including_at_the_end() -> None:
    """Criterion 2, negative: a run that ends mid-block yields the whole blocks and no stub."""
    clock = VirtualClock()
    io, device = rig(clock, capture_buffer_frames=1024)
    sink = CaptureSink(io, "raw")
    await capture_out(device, 25.0, sink)
    await call(io, "close_capture", f"{WHY_CAPTURE} — criterion 7's explicit close()")
    await capture_settle(device, sink)
    blocks = list(sink.items)
    sink.raise_error()
    await sink.close()

    assert blocks, "nothing was captured at all"
    sizes = [int(samples_of(b).size) for b in blocks]
    assert set(sizes) == {BLOCK_SAMPLES}, (
        f"block sizes {sorted(set(sizes))} at close(): criterion 2 allows only {BLOCK_SAMPLES} "
        f"samples; a partial block is never yielded on the raw stream"
    )
    assert len(blocks) == device.delivered_frames // BLOCK_SAMPLES


async def test_xrun_substitutes_silence_in_the_stt_stream_and_marks_the_raw_one() -> None:
    """Criterion 4: the raw timeline is honest, the STT timeline has no hole.

    30 ms of audio never reaches us. The raw stream must say so — the next block is marked and
    its ``t_ms`` shows the real 40 ms step — while the STT stream must contain exactly 30 ms of
    substituted silence, because the Kyutai delayed-stream model advances its clock only while
    audio arrives (``scripts/demo/_listen.py`` had to append real silence to flush it).

    An implementation that simply skips the missing audio keeps every timestamp and every frame
    count identical to this one; what it cannot produce is the silent region in the middle of
    frame 2, or a non-zero ``substituted_ms``.
    """
    clock = VirtualClock()
    io, device = rig(clock, capture_buffer_frames=BLOCK_SAMPLES, capture_source=loud_pattern)
    raw = CaptureSink(io, "raw")
    stt = CaptureSink(io, "stt")

    await capture_out(device, 200.0, raw, stt)
    device.xrun(30.0)
    await capture_out(device, 230.0, raw, stt)

    blocks, frames = list(raw.items), list(stt.items)
    stats = await capture_stats(io)
    raw.raise_error()
    stt.raise_error()
    await raw.close()
    await stt.close()

    # -- (a) the raw stream never lies
    assert len(blocks) == 40, f"20 blocks before and 20 after a 30 ms hole, got {len(blocks)}"
    marked = [i for i, b in enumerate(blocks) if bool(getattr(b, "xrun", False))]
    assert marked == [20], (
        f"blocks {marked} carry xrun=True; exactly the first block after the hole (index 20) "
        f"must (criterion 4a)"
    )
    step = t_ms_of(blocks[20]) - t_ms_of(blocks[19])
    assert step == pytest.approx(40.0, abs=1e-6), (
        f"t_ms steps by {step:.1f} ms across a 30 ms hole (want {CAPTURE_BLOCK_MS} + 30 = 40): "
        f"the raw timeline hid the gap"
    )
    assert np.array_equal(samples_of(blocks[20]), loud_pattern(11040, BLOCK_SAMPLES)), (
        "the first block after the hole is not the audio the ADC took after the hole"
    )

    # -- (b) the STT stream has no hole
    assert len(frames) == 5, (
        f"430 ms of STT timeline is 5 whole {STT_BLOCK_MS} ms frames (400 ms) plus a 30 ms "
        f"carry, got {len(frames)} — an implementation that skipped the missing 30 ms has "
        f"400 ms of *audio*, so count alone does not separate them; the content below does"
    )
    stamps = [t_ms_of(f) for f in frames]
    assert stamps == pytest.approx([0.0, 80.0, 160.0, 240.0, 320.0], abs=TIMESTAMP_BOUND_MS), (
        f"STT frame stamps {stamps} are not a contiguous 80 ms timeline from the capture clock"
    )
    gaps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    assert gaps == pytest.approx([80.0] * 4, abs=1e-6), f"STT timeline is not contiguous: {gaps}"

    y = concat(frames)
    assert y.size == 5 * 1920
    lo = round(200.0 * STT_RATE / 1000.0)  # the hole starts at capture time 200 ms
    hi = round(230.0 * STT_RATE / 1000.0)
    # one block for a compensated-or-not group delay (criterion 6), plus 2 ms of filter ring
    guard = round((CAPTURE_BLOCK_MS + 2.0) * STT_RATE / 1000.0)
    inside = np.abs(y[lo + guard : hi - guard])
    assert float(inside.max()) <= 1e-3, (
        f"the 30 ms the device never captured is not silence in the STT stream (peak "
        f"{float(inside.max()):.4f}, want ≤ 1e-3 = −60 dBFS): the missing audio was skipped, so "
        f"real audio has been re-dated 30 ms early (criterion 4b)"
    )
    before = np.abs(y[lo - 10 * guard : lo - guard])
    after = np.abs(y[hi + guard : hi + 10 * guard])
    assert float(before.min()) > 0.2 and float(after.min()) > 0.2, (
        f"the audio around the hole is silent too (min {float(before.min()):.3f} / "
        f"{float(after.min()):.3f} of a 0.5 DC-offset source): the whole STT stream is silence, "
        f"so the assertion above proves nothing"
    )

    assert int(stat(stats, "xruns")) == 1, f"CaptureStats.xruns={stat(stats, 'xruns')!r}, want 1"
    assert float(stat(stats, "substituted_ms")) == pytest.approx(30.0, abs=1.0), (
        f"CaptureStats.substituted_ms={stat(stats, 'substituted_ms')!r}, want 30 ms — exactly "
        f"the missing duration (criterion 4b)"
    )


async def test_a_long_stall_errors_instead_of_manufacturing_audio() -> None:
    """Criterion 4's cap: 1 s of missing audio yields ≤ 500 ms of silence, then a named error."""
    module = import_capture()
    cap = getattr(module, "MAX_SUBSTITUTED_MS", MAX_SUBSTITUTED_MS)
    error_type = getattr(module, "SubstitutionLimitExceeded", None)
    assert isinstance(error_type, type) and issubclass(error_type, Exception), (
        "criterion 4 wants a **named** error when substitution is capped; "
        "audio_capture.SubstitutionLimitExceeded does not exist "
        f"(module has {sorted(n for n in vars(module) if not n.startswith('_'))[:12]})"
    )
    assert float(cap) == pytest.approx(500.0), f"MAX_SUBSTITUTED_MS={cap!r}, criterion 4 says 500"

    clock = VirtualClock()
    io, device = rig(clock, capture_buffer_frames=BLOCK_SAMPLES, capture_source=loud_pattern)
    raw = CaptureSink(io, "raw")
    stt = CaptureSink(io, "stt")
    await capture_out(device, 100.0, raw, stt)
    device.xrun(1000.0)
    await capture_out(device, 1200.0, raw, stt, step_ms=10.0)

    stats = await capture_stats(io)
    assert stt.error is not None, (
        f"the STT stream manufactured {float(stat(stats, 'substituted_ms')):.0f} ms of silence "
        f"for a 1000 ms hole and raised nothing; criterion 4 caps it at {cap:.0f} ms and then "
        f"surfaces a named error"
    )
    assert isinstance(stt.error, error_type), (
        f"stt_blocks() raised {type(stt.error).__name__}({stt.error}); criterion 4 wants "
        f"{error_type.__name__}"
    )
    assert float(stat(stats, "substituted_ms")) <= float(cap) + CAPTURE_BLOCK_MS, (
        f"{float(stat(stats, 'substituted_ms')):.0f} ms of audio was fabricated for a 1000 ms "
        f"hole; the cap is {cap:.0f} ms (+ one block of block granularity)"
    )
    assert raw.error is None, (
        f"the raw stream died with {raw.error!r} as well; criterion 4a and 8 keep it running — "
        f"the reflex VAD is the only thing that can still hear the user here"
    )
    assert len(raw.items) > 10, "the raw stream stopped delivering after the hole"
    await raw.close()
    await stt.close()


async def test_overflow_drops_oldest_and_the_gap_shows_in_t_ms() -> None:
    """Criterion 5: a slow consumer loses the **oldest** blocks, and the count says how many."""
    clock = VirtualClock()
    io, device = rig(clock, capture_buffer_frames=BLOCK_SAMPLES)
    sink = CaptureSink(io, "raw")

    await capture_out(device, 100.0, sink)
    before = len(sink)
    assert before == 10, f"{before} blocks in the first 100 ms"
    last_before = t_ms_of(sink.items[-1])

    sink.hold()  # the consumer stops reading; the device thread must not care
    await capture_out(device, 1000.0, sink)
    # one block of slack: the consumer may already have been suspended inside __anext__ when
    # hold() was called, so it can still take delivery of the block it had asked for.
    assert len(sink) <= before + 1, (
        f"the held consumer received {len(sink) - before} more blocks: hold() did not stop it"
    )
    before = len(sink)
    last_before = t_ms_of(sink.items[-1])
    sink.release()
    await capture_settle(device, sink)

    blocks = list(sink.items)
    stats = await capture_stats(io)
    sink.raise_error()
    await sink.close()

    produced = device.delivered_frames // BLOCK_SAMPLES
    dropped = int(stat(stats, "dropped_blocks"))
    assert dropped > 0, (
        f"{produced} blocks were captured while the consumer held for 1000 ms and none was "
        f"dropped: the queue is unbounded (criterion 5 wants ≥ {MIN_QUEUE_BLOCKS} slots, bounded)"
    )
    assert len(blocks) + dropped == produced, (
        f"conservation: {len(blocks)} delivered + {dropped} dropped != {produced} captured"
    )
    survived = len(blocks) - before
    assert survived >= MIN_QUEUE_BLOCKS, (
        f"only {survived} blocks survived the overflow; criterion 5 wants a queue of at least "
        f"{MIN_QUEUE_BLOCKS} blocks (320 ms)"
    )
    first_after = t_ms_of(blocks[before])
    assert first_after - last_before > 100.0, (
        f"after the overflow the consumer got a block from t_ms={first_after:.0f}, only "
        f"{first_after - last_before:.0f} ms after the last one it had: the queue dropped the "
        f"**newest** blocks and kept stale audio. Criterion 5 drops the oldest, so the next "
        f"block the consumer sees is recent and its t_ms shows the true gap."
    )
    assert t_ms_of(blocks[-1]) >= 1000.0, (
        f"the last block is from t_ms={t_ms_of(blocks[-1]):.0f}; 1100 ms of audio was captured, "
        f"so a drop-oldest queue ends at the live edge"
    )


async def test_a_held_subscriber_loses_the_oldest_blocks_and_the_fast_one_loses_nothing() -> None:
    """Criterion 5 with two subscribers (ruling A17; A16(b) made the queues per subscription).

    The session subscribes twice — the reflex VAD on one iterator, the STT tap on another — so
    the drop rule has to be tested where it actually runs: one consumer held while another keeps
    reading, for 2 s = 200 blocks. Criterion 5 asks for at least 32 slots (320 ms) and does not
    name an upper bound, so the hold is six times that: a queue that swallows 2 s of audio
    rather than dropping is not backpressure, it is the unbounded buffer the criterion forbids.
    The fast consumer is the oracle for the whole stream: it lost nothing, so what the
    held consumer is missing is what its own queue dropped, and *which end* it dropped is
    readable off the surviving ``t_ms`` values. Dropping the newest instead leaves the held
    consumer with stale audio from the moment it stopped reading and no gap at all — which is
    the mutation that survived the suite before this test existed.
    """
    clock = VirtualClock()
    io, device = rig(clock, capture_buffer_frames=BLOCK_SAMPLES)
    fast = CaptureSink(io, "raw")
    slow = CaptureSink(io, "raw")

    await capture_out(device, 50.0, fast, slow)
    assert len(fast) == len(slow) == 5, (
        f"the two subscribers saw {len(fast)} and {len(slow)} of the first 5 blocks: every "
        f"subscriber gets every block until its own queue overflows (criterion 5)"
    )
    slow.hold()
    await capture_out(device, 2000.0, fast, slow)  # 200 blocks: past any bounded queue
    slow.release()
    await capture_settle(device, fast, slow)

    fast_stamps = [t_ms_of(b) for b in fast.items]
    slow_stamps = [t_ms_of(b) for b in slow.items]
    stats = await capture_stats(io)
    fast.raise_error()
    slow.raise_error()
    await fast.close()
    await slow.close()

    produced = device.delivered_frames // BLOCK_SAMPLES
    assert produced == 205, f"2050 ms of capture is 205 blocks, the device delivered {produced}"
    assert len(fast_stamps) == produced, (
        f"the fast subscriber received {len(fast_stamps)} of {produced} blocks while another "
        f"subscriber was not reading: one slow consumer must not cost a fast one a single block "
        f"(criterion 5, ruling A16(b) — the queues are per subscription)"
    )
    fast_steps = [fast_stamps[i + 1] - fast_stamps[i] for i in range(len(fast_stamps) - 1)]
    assert fast_steps == pytest.approx([float(CAPTURE_BLOCK_MS)] * len(fast_steps), abs=1e-6), (
        f"the fast subscriber's timeline is not contiguous: {sorted(set(fast_steps))[:5]}"
    )

    assert len(slow_stamps) < produced, (
        f"the held subscriber received all {produced} blocks: nothing was dropped although it "
        f"stopped reading for 2 s, so its queue holds at least 2 s of audio (criterion 5 wants a "
        f"bounded ring of ~320 ms whose overflow is counted)"
    )
    assert slow_stamps[-1] == fast_stamps[-1], (
        f"the held subscriber's last block is from t_ms={slow_stamps[-1]:.0f} while the stream "
        f"reached t_ms={fast_stamps[-1]:.0f}: its queue dropped the **newest** blocks and kept "
        f"stale audio. Criterion 5 drops the oldest, so a consumer that falls behind comes back "
        f"at the live edge rather than 600 ms in the past"
    )
    gaps = [
        (i, slow_stamps[i + 1] - slow_stamps[i])
        for i in range(len(slow_stamps) - 1)
        if abs(slow_stamps[i + 1] - slow_stamps[i] - CAPTURE_BLOCK_MS) > 1e-6
    ]
    assert len(gaps) == 1, (
        f"the held subscriber's timeline has {len(gaps)} discontinuities {gaps[:3]}; exactly one "
        f"is expected, where its queue overflowed (criterion 5: the next delivered block's t_ms "
        f"shows the true gap)"
    )
    at, size = gaps[0]
    # 4 or 5: the consumer may already have been suspended inside __anext__ when hold() was
    # called, so it can still take delivery of the block it had asked for (the same slack the
    # single-consumer overflow case above allows).
    assert at in (4, 5) and size > 100.0, (
        f"the gap is {size:.0f} ms after block {at}; the consumer stopped reading after the "
        f"fifth block (t_ms={fast_stamps[4]:.0f}) and 2 s went by, so the hole must be there and "
        f"must be far larger than one block"
    )
    survived = len(slow_stamps) - 1 - at
    assert survived >= MIN_QUEUE_BLOCKS, (
        f"only {survived} blocks survived in the held subscriber's queue; criterion 5 wants at "
        f"least {MIN_QUEUE_BLOCKS} slots (320 ms)"
    )
    assert int(stat(stats, "dropped_blocks")) > 0, (
        "blocks were lost and dropped_blocks is 0: criterion 5 makes the drop visible (counted "
        "once at the ring, ruling A16(b))"
    )


async def test_the_capture_queue_and_the_event_queue_drop_at_the_ends_their_specs_name() -> None:
    """Criterion 5's asymmetry clause, tested in both directions.

    **Reported ambiguity.** T-0021 criterion 5 says dropping the oldest capture block is
    "deliberately the opposite of T-0006 criterion 8's event queue, which keeps the newest" — but
    dropping the oldest *is* keeping the newest, and T-0006's own test is named
    ``..._drops_oldest_never_newest...``. The two rules as written are the same rule. This test
    asserts each spec's operative sentence (capture: "the **oldest** block is dropped"; events:
    "the newest kept never the oldest") rather than the rationale, so neither queue can be
    "fixed" to match a misremembered version of the other. If the orchestrator intends the
    capture queue to keep the *oldest* blocks instead, that is a task amendment and this test
    changes with it.
    """
    clock = VirtualClock()
    io, device = rig(clock, capture_buffer_frames=BLOCK_SAMPLES)
    sink = CaptureSink(io, "raw")
    await capture_out(device, 50.0, sink)
    sink.hold()
    await capture_out(device, 800.0, sink)
    sink.release()
    await capture_settle(device, sink)
    kept = [t_ms_of(b) for b in sink.items[5:]]
    await sink.close()
    assert kept and max(kept) >= 800.0, (
        f"the capture queue kept blocks up to t_ms={max(kept) if kept else None}; it drops the "
        f"oldest, so the newest 320 ms are the ones that survive (criterion 5)"
    )

    # the other direction: the playback event queue keeps the newest terminal event (T-0006 c8)
    clock2 = VirtualClock()
    io2, device2 = rig(clock2, buffer_ms=200.0)
    assert device2.quantum_ms > 0.0
    events = EventSink(io2, autostart=False)
    events.arm()
    finals = []
    for i in range(10):
        await play(io2, PcmUtterance(f"u{i}", samples=tone(100.0)), source=f"u{i}")
        await play_out(device2, 100.0)
        finals.append(await call(io2, "flush", "T-0006 criterion 5"))
    events.start()
    await events.sync()
    seen = [e.played_ms for e in events.events]
    await events.close()
    assert seen, "no playback event survived at all"
    assert seen[-1] == finals[-1], (
        f"the last event the consumer sees reports {seen[-1]} ms, the last run played "
        f"{finals[-1]} ms: the event queue dropped the newest event (T-0006 criterion 8 keeps it)"
    )


async def test_capture_and_playback_share_one_device_and_one_clock_base() -> None:
    """Criterion 1: ``is_rendering(block.t_ms)`` is answerable without a wall-clock subtraction."""
    clock = VirtualClock()
    io, device = rig(clock, capture_buffer_frames=BLOCK_SAMPLES, buffer_ms=200.0)
    sink = CaptureSink(io, "raw")
    await capture_out(device, 100.0, sink)
    quiet_before = list(sink.items)

    await play(io, PcmUtterance(samples=tone(200.0)), source="two hundred")
    await capture_out(device, 200.0, sink)
    during = sink.items[len(quiet_before) :]
    await capture_out(device, 300.0, sink)
    after = sink.items[len(quiet_before) + len(during) :]
    sink.raise_error()
    await sink.close()

    assert device.capture_starts >= 1, "criterion 1: the module starts capture on the device"
    assert device.starts >= 1, "criterion 1: the same device object serves playback"

    async def rendering(block: object) -> bool:
        return bool(await call(io, "is_rendering", "T-0006 criterion 10", t_ms_of(block)))

    # all but the block adjacent to the onset: a module that samples the rendering position
    # once per device buffer cannot place the edge more precisely than one block (criterion 3's
    # own bound), and the criterion under test here is the shared clock base, not the edge.
    quiet_hits = [b for b in quiet_before[:-1] if await rendering(b)]
    assert quiet_hits == [], (
        f"{len(quiet_hits)} blocks captured before any audio was played report "
        f"is_rendering(t_ms) True: the two clocks do not share a base (criterion 1)"
    )
    hits = 0
    for block in during:
        hits += int(await rendering(block))
    assert hits >= 15, (
        f"only {hits} of the {len(during)} blocks captured while 200 ms of audio was rendering "
        f"report is_rendering(t_ms) True; a capture stamp and the playback clock must be "
        f"directly comparable (criterion 1)"
    )
    late = []
    for block in after[10:]:
        if await rendering(block):
            late.append(block)
    assert late == [], (
        f"{len(late)} blocks captured well after playback ended still report is_rendering True"
    )


async def test_stalling_one_side_does_not_move_the_other_sides_timestamps() -> None:
    """Criterion 1, the two-stream clause: one graph clock, never a subtraction of arrival times."""
    clock = VirtualClock()
    io, device = rig(clock, capture_buffer_frames=BLOCK_SAMPLES, buffer_ms=300.0)
    sink = CaptureSink(io, "raw")
    await play(io, PcmUtterance(samples=tone(2000.0)), source="long")

    await capture_out(device, 200.0, sink)
    device.stall(100.0)  # the *playback* side hiccups: the sink renders something else
    await capture_out(device, 200.0, sink)
    blocks = list(sink.items)
    assert_cadence(blocks, "while the playback side stalled")
    bad = [
        i
        for i, b in enumerate(blocks)
        if abs(t_ms_of(b) - device.true_capture_ms(i * BLOCK_SAMPLES)) > TIMESTAMP_BOUND_MS
    ]
    assert bad == [], f"a playback stall moved capture stamps at blocks {bad[:5]} (criterion 1)"

    rendered_before = device.rendered_ms()
    played_before = await played_ms(io)
    device.stall_capture(150.0)  # now the *capture* side hiccups
    await capture_out(device, 150.0, sink)
    played_after = await played_ms(io)
    assert played_after - played_before == pytest.approx(
        device.rendered_ms() - rendered_before, abs=20.0
    ), (
        f"played_ms moved by {played_after - played_before:.1f} ms while the device rendered "
        f"{device.rendered_ms() - rendered_before:.1f} ms and the *capture* side was stalled: "
        f"the playback clock is reading the capture side (criterion 1)"
    )
    sink.raise_error()
    await sink.close()


async def test_capture_stats_is_the_snapshot_criterion_10_names() -> None:
    """Criterion 10: seven fields, no new span, nothing invented."""
    clock = VirtualClock()
    io, device = rig(clock, capture_buffer_frames=BLOCK_SAMPLES)
    sink = CaptureSink(io, "raw")
    await capture_out(device, 200.0, sink)
    stats = await capture_stats(io)
    sink.raise_error()
    await sink.close()

    assert int(stat(stats, "blocks")) == len(sink.items) == 20
    assert int(stat(stats, "xruns")) == 0
    assert float(stat(stats, "substituted_ms")) == 0.0
    assert int(stat(stats, "dropped_blocks")) == 0
    assert int(stat(stats, "gated_blocks")) == 0
    assert float(stat(stats, "quantum_ms")) > 0.0
    assert isinstance(stat(stats, "device"), str) and stat(stats, "device"), (
        "CaptureStats.device names the device the numbers came from (criterion 10)"
    )


# --- the real backend's duplex callback, driven in-process (ruling A13) --------------------------
#
# No hardware and no PortAudio: ``PortAudioDevice`` imports ``sounddevice`` inside ``start()``
# only, so the callback PortAudio would call can be called by hand with a ``SimpleNamespace``
# ``time_info`` — exactly the shape PortAudio passes (``inputBufferAdcTime``,
# ``outputBufferDacTime``, ``currentTime`` in seconds) and a ``status`` with ``input_overflow``.
# What is under test is criterion 1's single clock: ``adc_ms`` (what a capture block is stamped
# with) and ``captured_ms()`` (what the module's playback span and ``yielded_ms`` are measured
# against) must be two readings of *one* clock, at one rate, from one base. The oracle is the ADC
# times fed in, which come from this test and owe the device nothing.

A13 = f"{WHY_CAPTURE} — ruling A13: _duplex_callback drives one ADC clock"
#: One PortAudio buffer in this rig: 480 frames = 10 ms at 48 kHz.
CB_FRAMES = BLOCK_SAMPLES
CB_MS = 1000.0 * CB_FRAMES / DEVICE_RATE
#: Where PortAudio's stream clock happens to be when the first callback runs. Nothing may depend
#: on the value (the origin is re-based away); an implementation that reads it as "unset" is the
#: subject of its own case below.
STREAM_T0 = 12.345


def streaming_device(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, list[tuple[float, bool, int]]]:
    """A ``PortAudioDevice`` with a source and a delivery callback, with no stream behind it.

    Opening the stream is the one thing that needs the library, so ``start()`` is neutralised for
    the duration; everything else — the callback, the counters, the clock — is the shipped code.
    Capture is *not* started here: two of the cases below care about what the device does between
    the stream opening and ``start_capture()``.
    """
    from local_astra.conductor.audio_device import PortAudioDevice

    try:
        device = PortAudioDevice(sink="astra-test-sink", source="astra-test-sink")
    except TypeError as exc:
        raise AssertionError(
            f"PortAudioDevice takes no capture source: ruling A1 fixes the constructor as "
            f"open_device(sink=, source=, block_ms=, latency_ms=) and criterion 1 wants one "
            f"device object serving both directions ({exc})"
        ) from exc
    delivered: list[tuple[float, bool, int]] = []
    attr_or_fail(device, "set_capture_callback", A13)(
        lambda samples, adc_ms, xrun: delivered.append((float(adc_ms), bool(xrun), samples.size))
    )
    monkeypatch.setattr(device, "start", lambda: None)
    return device, delivered


def duplex_device(monkeypatch: pytest.MonkeyPatch) -> tuple[object, list[tuple[float, bool, int]]]:
    """``streaming_device`` with capture already running — the ordinary case."""
    device, delivered = streaming_device(monkeypatch)
    attr_or_fail(device, "start_capture", A13)()
    return device, delivered


def fire(device: object, *, adc_s: float, frames: int = CB_FRAMES, overflow: bool = False) -> float:
    """One PortAudio callback at ADC time ``adc_s``; returns ``captured_ms()`` right after it."""
    callback = attr_or_fail(device, "_duplex_callback", A13)
    indata = np.zeros((frames, 1), dtype=np.float32)
    outdata = np.zeros((frames, 1), dtype=np.float32)
    time_info = SimpleNamespace(
        inputBufferAdcTime=adc_s,
        currentTime=adc_s + 0.005,
        outputBufferDacTime=adc_s + 0.020,
    )
    callback(indata, outdata, frames, time_info, SimpleNamespace(input_overflow=overflow))
    return float(attr_or_fail(device, "captured_ms", A13)())


def offsets(delivered: list[tuple[float, bool, int]], positions: list[float]) -> list[float]:
    """``captured_ms() − adc_ms`` per delivered buffer: constant iff both share a base and a rate."""
    return [
        round(pos - adc_ms, 6)
        for (adc_ms, _xrun, _n), pos in zip(delivered, positions, strict=True)
    ]


def test_the_duplex_callback_gives_both_capture_seams_one_base_and_one_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Criterion 1 + ruling A9 at the device seam: ``adc_ms`` and ``captured_ms()`` are one clock.

    Twenty consecutive buffers, nothing unusual. ``adc_ms`` must be the ADC time PortAudio
    reported, re-based (criterion 3), and ``captured_ms()`` must be the same clock read at the
    end of the buffer — so their difference is the *same* number every time and no larger than
    one buffer. Two clocks that happen to agree while nothing goes wrong are the defect ruling A9
    describes; the two tests below are where they come apart.
    """
    device, delivered = duplex_device(monkeypatch)
    positions = [fire(device, adc_s=STREAM_T0 + 0.010 * i) for i in range(20)]

    assert len(delivered) == 20, f"{len(delivered)} of 20 buffers reached the capture callback"
    stamps = [adc_ms for adc_ms, _x, _n in delivered]
    steps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    assert steps == pytest.approx([CB_MS] * len(steps), abs=1e-6), (
        f"adc_ms steps by {sorted(set(steps))[:4]} ms for buffers PortAudio dated exactly "
        f"{CB_MS} ms apart: the stamp is not inputBufferAdcTime (criterion 3)"
    )
    assert positions == sorted(positions), f"captured_ms() went backwards: {positions[:6]}"

    seen = offsets(delivered, positions)
    assert len(set(seen)) == 1, (
        f"captured_ms() − adc_ms takes {len(set(seen))} different values over 20 buffers "
        f"({sorted(set(seen))[:5]}): the two seams are not reading one clock at one rate "
        f"(ruling A9)"
    )
    assert 0.0 <= seen[0] <= 2 * CB_MS, (
        f"captured_ms() sits {seen[0]:.1f} ms from the adc_ms of the buffer just delivered; one "
        f"clock puts the two within a buffer of each other ({CB_MS} ms), because captured_ms() "
        f"is the ADC position at the *end* of that buffer (ruling A9)"
    )


def test_capture_starting_after_the_stream_does_not_split_the_adc_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ruling A9: the duplex stream runs 500 ms of playback-only before capture is asked for.

    That is the ordinary case on hardware — ``play()`` opens the stream, the session starts
    listening later — and it is where a ``captured_ms()`` counted from *delivered frames* takes a
    base of its own: it reads ~10 ms while the block it just delivered is stamped ~500 ms. The
    module compares the two (the gate's playback span, ``yielded_ms``), so a 490 ms split means
    the gate never fires and every ``audio.capture`` span is nonsense.
    """
    device, delivered = streaming_device(monkeypatch)

    for i in range(50):  # the stream is open and playing; capture has not been asked for
        fire(device, adc_s=STREAM_T0 + 0.010 * i)
    assert delivered == [], (
        f"{len(delivered)} buffers were handed to the capture path before start_capture(): the "
        f"input side must be silent until the module asks for it"
    )

    attr_or_fail(device, "start_capture", A13)()
    positions = [fire(device, adc_s=STREAM_T0 + 0.500 + 0.010 * i) for i in range(10)]

    assert len(delivered) == 10, f"{len(delivered)} of 10 buffers after start_capture()"
    seen = offsets(delivered, positions)
    assert len(set(seen)) == 1 and 0.0 <= seen[0] <= 2 * CB_MS, (
        f"capture began 500 ms after the stream and captured_ms() − adc_ms is now {seen[:3]} ms "
        f"(want one constant in [0, {2 * CB_MS:.0f}]). A captured_ms() counted from delivered "
        f"frames starts its own base at start_capture() while adc_ms keeps the stream's: the two "
        f"differ by the 500 ms of playback-only that preceded capture, so is_rendering(block.t_ms) "
        f"is asked about a time half a second away from the playback span (ruling A9, criterion 1)"
    )


def test_an_input_overflow_does_not_permanently_offset_the_capture_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ruling A9 and criterion 4: the ADC ran through the 100 ms nobody handed us.

    PortAudio reports the loss with ``status.input_overflow`` and dates the next buffer at the
    time the ADC actually took it — 100 ms later. A ``captured_ms()`` that counts delivered
    frames gains one buffer across that hole instead of eleven, and never catches up: every
    later block is dated 100 ms adrift of the playback clock it is compared with.
    """
    device, delivered = duplex_device(monkeypatch)

    before = [fire(device, adc_s=STREAM_T0 + 0.010 * i) for i in range(10)]  # 0 .. 90 ms
    gap = fire(device, adc_s=STREAM_T0 + 0.200, overflow=True)  # 100 ms never arrived
    after = [fire(device, adc_s=STREAM_T0 + 0.210 + 0.010 * i) for i in range(10)]
    positions = [*before, gap, *after]

    assert len(delivered) == 21, f"{len(delivered)} of 21 buffers delivered"
    assert [i for i, (_a, xrun, _n) in enumerate(delivered) if xrun] == [10], (
        f"the buffer after the overflow is not marked: status.input_overflow is the device's own "
        f"report that audio was lost, and criterion 4 is built on it "
        f"(marks: {[i for i, (_a, x, _n) in enumerate(delivered) if x]})"
    )
    advance = gap - before[-1]
    assert advance >= 100.0, (
        f"captured_ms() advanced {advance:.1f} ms across a 100 ms input overflow (want ≥ 110: "
        f"the 100 ms the ADC took while nobody was listening, plus the buffer just delivered). "
        f"A frame counter gains only {CB_MS:.0f} ms here and stays {100.0:.0f} ms behind the ADC "
        f"for the rest of the run (ruling A9)"
    )
    seen = offsets(delivered, positions)
    assert len(set(seen)) == 1, (
        f"captured_ms() − adc_ms was {seen[0]} ms before the overflow and {sorted(set(seen))} "
        f"across it: the hole moved one of the two clocks and not the other, which is a "
        f"permanent offset, not a hiccup (ruling A9)"
    )


def test_a_stream_clock_that_starts_at_zero_is_still_the_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Criterion 3: ``inputBufferAdcTime == 0.0`` is a time, not a missing value.

    PortAudio's stream clock is ``Pa_GetStreamTime``, which on the ALSA host API starts at zero,
    so the first callback of a stream can legitimately report 0.0. An implementation that anchors
    its origin on the first *truthy* timestamp silently takes the **second** callback as the
    base, and every block for the rest of the run is dated one buffer early — a constant error
    that no later test can see, because everything after it is self-consistent.
    """
    device, delivered = duplex_device(monkeypatch)
    for i in range(10):
        fire(device, adc_s=0.010 * i)

    stamps = [adc_ms for adc_ms, _x, _n in delivered]
    assert len(stamps) == 10, f"{len(stamps)} of 10 buffers delivered"
    steps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    assert steps == pytest.approx([CB_MS] * len(steps), abs=1e-6), (
        f"buffers PortAudio dated exactly {CB_MS} ms apart from a stream clock starting at 0.0 "
        f"came out {stamps[:4]}: a zero timestamp was read as 'no base yet', so the origin is "
        f"the second callback and every block is {CB_MS} ms early (criterion 3)"
    )
