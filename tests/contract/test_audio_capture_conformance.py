"""Both capture implementations answer the same questions: T-0021 (docs/02 §5, §4).

Two implementations of one interface run the same assertions: the real module driven by
``FakeAudioDevice``, and ``tests/fakes/audio.py::FakeAudioIO`` driven by the virtual clock. If
the fake and the module disagree about the block size, the cadence or what an xrun does to the
STT timeline, every conductor test written against the fake is a lie (CLAUDE.md: an interface
change changes the fake with it).

Parametrised **locally** on purpose: ``tests/contract/conftest.py``'s ``backend`` fixture marks
its ``real`` case ``gpu``, and nothing here needs a GPU (T-0006's ruling). The real-*device*
conformance of the same assertions is ``tests/integration/test_audio_capture_null_sink.py``
(markers ``integration`` + ``audio``).

What is deliberately **not** asserted here: the audio itself. The fake owns no anti-aliasing
filter, and the content of the 24 kHz stream is the subject of
``tests/unit/test_audio_decimate.py``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import pytest

from local_astra.contracts.protocols import AudioIO
from local_astra.contracts.speech import CAPTURE_BLOCK_MS
from tests.conftest import VirtualClock
from tests.fakes.audio import FakeAudioIO
from tests.fakes.audio_device import (
    BLOCK_SAMPLES,
    CAPTURE_BLOCK_BYTES,
    STT_BLOCK_MS,
    STT_BLOCK_SAMPLES,
    WHY_CAPTURE,
    CaptureSink,
    FakeAudioDevice,
    capture_out,
    capture_stats,
    make_audio_io,
    samples_of,
    stat,
    t_ms_of,
)

pytestmark = [pytest.mark.contract]

#: ``pending`` sits on the **module** case only, not on the module-level ``pytestmark``: the fake
#: half of every assertion below is implemented in this PR (``tests/fakes/audio.py``, CLAUDE.md
#: "an interface change changes the fake with it"), so marking it pending would make it
#: XPASS(strict) and fail the suite for being *right*. The module half is T-0021's to make green.
IMPLS = ["module", "fake"]
#: Seams T-0021 adds to the capture interface. ``contracts/protocols.py`` is not widened (that is
#: T-0017's file), so conformance is by presence — exactly as T-0006 did for ``played_ms()``.
T0021_SEAMS = ("capture_frames", "stt_blocks", "capture_stats")


async def settle_sinks(*sinks: CaptureSink, rounds: int = 5, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    last: tuple[int, ...] | None = None
    quiet = 0
    while time.monotonic() < deadline:
        await asyncio.sleep(0.0003)
        state = tuple(len(s) for s in sinks)
        if state != last:
            last, quiet = state, 0
        elif (quiet := quiet + 1) >= rounds:
            return


class Rig:
    """One of the two implementations, plus the two controls the assertions below need."""

    def __init__(self, impl: str) -> None:
        self.impl = impl
        self.clock = VirtualClock()
        self.sinks: list[CaptureSink] = []
        if impl == "module":
            self.device: FakeAudioDevice | None = FakeAudioDevice(
                self.clock, capture_buffer_frames=BLOCK_SAMPLES
            )
            self.io: Any = make_audio_io(self.device)
        else:
            self.device = None
            self.io = FakeAudioIO(self.clock)

    def sink(self, kind: str) -> CaptureSink:
        sink = CaptureSink(self.io, kind)
        self.sinks.append(sink)
        return sink

    async def advance(self, ms: float) -> None:
        if self.device is not None:
            await capture_out(self.device, ms, *self.sinks)
            return
        left = float(ms)
        await settle_sinks(*self.sinks)
        while left > 0:
            step = min(float(CAPTURE_BLOCK_MS), left)
            self.clock.advance(int(step))
            left -= step
            await settle_sinks(*self.sinks)

    def xrun(self, ms: float) -> None:
        if self.device is not None:
            self.device.xrun(ms)
        else:
            self.io.inject_xrun(ms)

    async def close(self) -> None:
        for sink in self.sinks:
            sink.raise_error()
            await sink.close()


@pytest.mark.parametrize("impl", IMPLS)
async def test_it_is_still_an_audio_io_with_the_capture_seams(impl: str) -> None:
    rig = Rig(impl)
    assert isinstance(rig.io, AudioIO), (
        f"{impl}: does not satisfy the runtime_checkable AudioIO protocol "
        f"(contracts/protocols.py); T-0021 does not widen it (criterion 12b)"
    )
    missing = [name for name in T0021_SEAMS if not hasattr(rig.io, name)]
    assert missing == [], f"{impl}: missing the T-0021 capture seams {missing} ({WHY_CAPTURE})"


@pytest.mark.parametrize("impl", IMPLS)
async def test_capture_yields_1920_byte_blocks(impl: str) -> None:
    """Criterion 2: the ``AudioIO`` protocol method yields ``bytes``, 480 f32 at 48 kHz."""
    rig = Rig(impl)
    sink = rig.sink("bytes")
    await rig.advance(200.0)
    blocks = list(sink.items)
    await rig.close()

    assert len(blocks) >= 19, f"{impl}: {len(blocks)} blocks in 200 ms (want ~20)"
    sizes = {len(b) for b in blocks}
    assert sizes == {CAPTURE_BLOCK_BYTES}, (
        f"{impl}: capture() yielded blocks of {sorted(sizes)} bytes; criterion 2 says always "
        f"{CAPTURE_BLOCK_BYTES} (= {BLOCK_SAMPLES} f32 = {CAPTURE_BLOCK_MS} ms at 48 kHz)"
    )
    kinds = {type(b).__name__ for b in blocks}
    assert kinds == {"bytes"}, (
        f"{impl}: capture() must yield bytes (AsyncIterator[bytes]), got {sorted(kinds)}"
    )
    frame = np.frombuffer(blocks[0], dtype="<f4")
    assert frame.size == BLOCK_SAMPLES and bool(np.all(np.isfinite(frame))), (
        f"{impl}: the bytes do not decode as {BLOCK_SAMPLES} little-endian f32 samples"
    )


@pytest.mark.parametrize("impl", IMPLS)
async def test_block_timestamps_are_monotone_and_one_block_apart(impl: str) -> None:
    """Criteria 2 and 3, shared: the raw timeline is a clock, not a counter of callbacks."""
    rig = Rig(impl)
    sink = rig.sink("raw")
    await rig.advance(300.0)
    stamps = [t_ms_of(b) for b in sink.items]
    sizes = {int(samples_of(b).size) for b in sink.items}
    await rig.close()

    assert len(stamps) >= 29, f"{impl}: {len(stamps)} blocks in 300 ms"
    assert sizes == {BLOCK_SAMPLES}, f"{impl}: block sizes {sorted(sizes)}"
    assert stamps == sorted(stamps), f"{impl}: t_ms went backwards: {stamps[:8]}"
    steps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    assert steps == pytest.approx([float(CAPTURE_BLOCK_MS)] * len(steps), abs=1e-6), (
        f"{impl}: consecutive non-xrun blocks must be exactly {CAPTURE_BLOCK_MS} ms apart, got "
        f"{sorted(set(steps))[:5]}"
    )


@pytest.mark.parametrize("impl", IMPLS)
async def test_an_xrun_marks_the_raw_stream_and_leaves_the_stt_timeline_contiguous(
    impl: str,
) -> None:
    """Criterion 4, shared by both implementations (docs/02 §4 spec delta 2)."""
    rig = Rig(impl)
    raw = rig.sink("raw")
    stt = rig.sink("stt")
    await rig.advance(200.0)
    before = len(raw)
    rig.xrun(30.0)
    await rig.advance(300.0)
    blocks, frames = list(raw.items), list(stt.items)
    stats = await capture_stats(rig.io)
    await rig.close()

    marked = [i for i, b in enumerate(blocks) if bool(getattr(b, "xrun", False))]
    assert len(marked) == 1, f"{impl}: {len(marked)} blocks marked xrun for one 30 ms hole"
    at = marked[0]
    assert abs(at - before) <= 1, f"{impl}: the mark is on block {at}, the hole was after {before}"
    step = t_ms_of(blocks[at]) - t_ms_of(blocks[at - 1])
    assert step == pytest.approx(CAPTURE_BLOCK_MS + 30.0, abs=1e-6), (
        f"{impl}: t_ms steps by {step:.1f} ms across a 30 ms hole (want 40): criterion 4a says "
        f"the raw timeline shows the real gap"
    )

    stamps = [t_ms_of(f) for f in frames]
    steps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    assert steps == pytest.approx([float(STT_BLOCK_MS)] * len(steps), abs=1e-6), (
        f"{impl}: the STT timeline has a hole ({sorted(set(steps))[:5]}); criterion 4b "
        f"substitutes silence so the delayed-stream clock keeps advancing"
    )
    assert {int(samples_of(f).size) for f in frames} == {STT_BLOCK_SAMPLES}, (
        f"{impl}: STT frames must be {STT_BLOCK_SAMPLES} samples (the STT server's block size)"
    )
    assert int(stat(stats, "xruns")) == 1, f"{impl}: CaptureStats.xruns"
    assert float(stat(stats, "substituted_ms")) == pytest.approx(30.0, abs=1.0), (
        f"{impl}: {float(stat(stats, 'substituted_ms'))} ms substituted for a 30 ms hole"
    )
