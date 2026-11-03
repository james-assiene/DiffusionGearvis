"""The 80 ms STT framing: T-0021 criterion 7 (and criterion 10's two per-frame times).

moshi-server 0.6.4 reads 1920 samples of 24 kHz mono f32 at a time (``STT_INPUT_FORMAT``, and
``BLOCK = 1920`` in ``scripts/demo/_listen.py``). **The ruling this file pins down, so T-0007 can
be planned against it: T-0021 owns the 48 → 24 kHz resample and the 80 ms framing**;
``MoshiSttClient.feed(pcm_24k_f32, t_ms)`` is a consumer that does not resample, re-frame or
re-timestamp.

Seams fixed here (reported as such): ``stt_blocks()`` yields ``SttFrame(samples, t_ms, end_ms,
yielded_ms)`` — ``t_ms`` is the capture time of the frame's **first** sample (criterion 7),
``end_ms`` the capture time of its **last** one and ``yielded_ms`` the moment it was handed over
(criterion 10's pair) — and ``close_capture()`` is the explicit end of stream that is allowed to
zero-pad.
"""

from __future__ import annotations

import numpy as np
import pytest

from local_astra.contracts.speech import CAPTURE_BLOCK_MS, STT_INPUT_FORMAT
from tests.conftest import VirtualClock
from tests.fakes.audio_device import (
    BLOCK_SAMPLES,
    DEVICE_RATE,
    STT_BLOCK_MS,
    STT_BLOCK_SAMPLES,
    STT_RATE,
    WHY_CAPTURE,
    CaptureSink,
    FakeAudioDevice,
    attr_or_fail,
    call,
    capture_out,
    capture_pattern,
    capture_settle,
    concat,
    import_capture,
    import_or_fail,
    make_audio_io,
    samples_of,
    t_ms_of,
)

#: 80 ms of capture at 48 kHz: the input a single STT frame is made of.
FRAME_INPUT_SAMPLES = DEVICE_RATE * STT_BLOCK_MS // 1000  # 3840
BLOCKS_PER_FRAME = STT_BLOCK_MS // CAPTURE_BLOCK_MS  # 8


def rig(clock: VirtualClock | None = None, **kwargs: object) -> tuple[object, FakeAudioDevice]:
    device = FakeAudioDevice(clock if clock is not None else VirtualClock(), **kwargs)  # type: ignore[arg-type]
    return make_audio_io(device), device


def decimate_whole(x: np.ndarray) -> np.ndarray:
    """One-shot 48 → 24 kHz with the module's own stage — the framing's independent oracle.

    This is *differential* testing, not a re-implementation: the filter's correctness is the
    subject of ``tests/unit/test_audio_decimate.py`` (and its bit-identity across chunkings is
    asserted there). What is under test here is the framing, which must not lose, duplicate or
    re-order a sample relative to feeding the same audio through in one piece.
    """
    module = import_or_fail("local_astra.conductor.audio_resample", WHY_CAPTURE)
    stage = attr_or_fail(module, "Resampler48kTo24k", WHY_CAPTURE)()
    return np.asarray(attr_or_fail(stage, "process", WHY_CAPTURE)(np.asarray(x, np.float32)))


async def test_frames_are_1920_samples_of_24khz_mono_f32() -> None:
    """Criterion 7: moshi's block size, in moshi's format, always."""
    module = import_capture()
    constant = attr_or_fail(module, "STT_BLOCK_SAMPLES", f"{WHY_CAPTURE} — criterion 7")
    assert int(constant) == 1920, (
        f"audio_capture.STT_BLOCK_SAMPLES = {constant!r}; moshi-server 0.6.4 reads 1920 samples "
        f"(80 ms at {STT_RATE} Hz) per block"
    )

    io, device = rig(capture_buffer_frames=1024)
    stt = CaptureSink(io, "stt")
    await capture_out(device, 500.0, stt)
    frames = list(stt.items)
    stt.raise_error()
    await stt.close()

    assert len(frames) == 6, (
        f"500 ms of capture through a 1024-frame device is 6 whole {STT_BLOCK_MS} ms frames "
        f"(the 1024-frame quantum leaves less than a block outstanding), got {len(frames)}"
    )
    for i, frame in enumerate(frames):
        x = samples_of(frame)
        assert x.size == STT_BLOCK_SAMPLES, f"frame {i} has {x.size} samples, want 1920"
        assert x.dtype == np.float32, f"frame {i} is {x.dtype}; {STT_INPUT_FORMAT!r}"
        assert x.ndim == 1, f"frame {i} has shape {x.shape}; mono"


async def test_each_frame_carries_the_capture_time_of_its_first_sample() -> None:
    """Criterion 7: frame k starts where raw block 8k starts — both from the capture clock."""
    io, device = rig(capture_buffer_frames=BLOCK_SAMPLES)
    raw = CaptureSink(io, "raw")
    stt = CaptureSink(io, "stt")
    await capture_out(device, 500.0, raw, stt)
    blocks, frames = list(raw.items), list(stt.items)
    raw.raise_error()
    stt.raise_error()
    await raw.close()
    await stt.close()

    assert len(frames) >= 6 and len(blocks) >= 48
    for k, frame in enumerate(frames):
        expected = t_ms_of(blocks[k * BLOCKS_PER_FRAME])
        # one block of slack: criterion 6 allows the decimator's group delay to be compensated
        # in t_ms or documented, and either way it is constant and ≤ one block.
        assert abs(t_ms_of(frame) - expected) <= CAPTURE_BLOCK_MS, (
            f"STT frame {k} says t_ms={t_ms_of(frame):.1f}; its first sample is the first sample "
            f"of raw block {k * BLOCKS_PER_FRAME}, captured at {expected:.1f} ms (criterion 7)"
        )
    stamps = [t_ms_of(f) for f in frames]
    steps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    assert steps == pytest.approx([float(STT_BLOCK_MS)] * len(steps), abs=1e-6), (
        f"the STT timeline steps by {steps} ms; moshi's clock advances 80 ms per block and a "
        f"hole or an overlap re-dates every word after it (criterion 7)"
    )


async def test_the_frame_carries_the_two_times_the_audio_capture_span_needs() -> None:
    """Criterion 10: the capture time of the frame's last sample, and when it was yielded."""
    io, device = rig(capture_buffer_frames=BLOCK_SAMPLES)
    stt = CaptureSink(io, "stt")
    await capture_out(device, 300.0, stt)
    frames = list(stt.items)
    stt.raise_error()
    await stt.close()

    assert frames, "no STT frame at all"
    one_sample_ms = 1000.0 / STT_RATE
    for i, frame in enumerate(frames):
        end = float(attr_or_fail(frame, "end_ms", f"{WHY_CAPTURE} — criterion 10"))
        yielded = float(attr_or_fail(frame, "yielded_ms", f"{WHY_CAPTURE} — criterion 10"))
        span = end - t_ms_of(frame)
        assert span == pytest.approx(STT_BLOCK_MS - one_sample_ms, abs=one_sample_ms + 1e-6), (
            f"frame {i}: end_ms − t_ms = {span:.4f} ms; 1920 samples at {STT_RATE} Hz span "
            f"{STT_BLOCK_MS - one_sample_ms:.4f} ms from the first to the last one"
        )
        assert yielded >= end - CAPTURE_BLOCK_MS, (
            f"frame {i} was yielded at {yielded:.1f} ms, before its last sample was captured "
            f"({end:.1f} ms): yielded_ms is not on the capture clock (criterion 10)"
        )


async def test_a_partial_frame_is_padded_only_at_close() -> None:
    """Criterion 7: no short frame mid-stream; the tail is zero-padded at ``close_capture()``."""
    io, device = rig(capture_buffer_frames=BLOCK_SAMPLES)
    stt = CaptureSink(io, "stt")
    await capture_out(device, 100.0, stt)  # 100 ms = one 80 ms frame + 20 ms of carry

    assert len(stt) == 1, f"{len(stt)} frames after 100 ms: the 20 ms carry was flushed early"
    first = samples_of(stt.items[0])
    assert float(np.abs(first[-STT_RATE // 100 :]).max()) > 0.0, (
        "the last 10 ms of the first frame is digital silence: a partial frame was padded and "
        "yielded mid-stream (criterion 7 pads only at close())"
    )

    await call(io, "close_capture", f"{WHY_CAPTURE} — criterion 7's explicit close()")
    await capture_settle(device, stt)
    frames = list(stt.items)
    stt.raise_error()

    assert len(frames) == 2, f"close() must flush the carry as one padded frame, got {len(frames)}"
    tail = samples_of(frames[1])
    assert tail.size == STT_BLOCK_SAMPLES, f"the flushed frame is {tail.size} samples, want 1920"
    carry = 20 * STT_RATE // 1000  # 20 ms of real audio at 24 kHz
    assert float(np.abs(tail[carry:]).max()) == 0.0, (
        f"the padding after the {carry} real samples is not zero (peak "
        f"{float(np.abs(tail[carry:]).max()):.4f}): close() must pad with silence, not with "
        f"whatever was in the buffer"
    )
    assert float(np.abs(tail[: carry - 48]).max()) > 0.0, (
        "the flushed frame is all silence: the 20 ms of captured audio was dropped, not padded"
    )
    assert stt.finished, "close_capture() must end the stt_blocks() iterator"
    await stt.close()


async def test_no_sample_is_dropped_or_duplicated_over_60_seconds() -> None:
    """Criterion 7's conservation law, over the 60 s the task names (n ≈ 6000 blocks).

    Two independent checks: the *count* against hand arithmetic (60 s of 48 kHz is 2 880 000
    samples, so 1 440 000 at 24 kHz, so 750 whole frames), and the *content* against one-shot
    decimation of the same audio — framing may not move a sample.
    """
    io, device = rig(capture_buffer_frames=1024)
    stt = CaptureSink(io, "stt")
    await capture_out(device, 60_000.0, stt, step_ms=80.0)
    frames = list(stt.items)
    stt.raise_error()
    await stt.close()

    delivered = device.delivered_frames
    assert delivered == 2_812 * 1024, "60 s at 48 kHz is 2 880 000 frames = 2812 whole buffers"
    expected_frames = (delivered // 2) // STT_BLOCK_SAMPLES
    assert len(frames) == expected_frames, (
        f"{len(frames)} frames for {delivered} captured samples; {delivered // 2} samples at "
        f"24 kHz make {expected_frames} whole 1920-sample frames (criterion 7: nothing is "
        f"dropped, nothing is invented)"
    )
    stamps = [t_ms_of(f) for f in frames]
    drift = stamps[-1] - stamps[0] - STT_BLOCK_MS * (len(frames) - 1)
    assert drift == pytest.approx(0.0, abs=1e-6), (
        f"the STT timeline drifted {drift:.3f} ms over 60 s: {len(frames)} frames 80 ms apart "
        f"span {STT_BLOCK_MS * (len(frames) - 1)} ms"
    )

    y = concat(frames)
    oracle = decimate_whole(capture_pattern(0, delivered))
    assert np.array_equal(y, oracle[: y.size]), (
        f"the framed stream differs from the same audio decimated in one piece at sample "
        f"{int(np.argmax(y != oracle[: y.size]))} of {y.size}: framing lost, duplicated or "
        f"re-ordered samples"
    )
