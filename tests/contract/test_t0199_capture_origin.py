"""T-0199 fake change, the real-service half: the capture clock starts at the ADC, not at the session clock.

``tests/fakes/audio.py::FakeAudioIO`` grew ``capture_origin_ms`` (T-0199, the T-0051 B19 lesson): by default the fake's
capture clock starts at ``clock.now_ms`` when capture is armed, which makes audio time and session time coincide and
hid the onset-frame bug. ``.claude/rules/tests.md``: a behaviour added to a fake needs a row showing the real module has
it. The real module here is ``PipeWireAudioIO`` (``local_astra/conductor/audio_io.py`` + ``audio_capture.py``) driven
by ``tests/fakes/audio_device.FakeAudioDevice`` on a virtual clock -- the harness
``tests/contract/test_audio_capture_conformance.py`` already uses for "module" -- **not** ``sounddevice_stub`` +
``PortAudioDevice`` as the task file suggests: the stub drives callbacks on a worker thread at real-time pace and has no
virtual session clock to start late, so it cannot state "armed at session 11 400". Flagged to the reviewer.

Parametrised locally over ``backend`` (``real``, ``fake``) and not marked ``gpu``: nothing here needs a GPU, so the
real half runs in the fast suite (the conformance module's ruling). Regime: virtual clock started at 11 400 ms,
FakeAudioDevice with one 10 ms buffer per callback, 400 ms of capture, n = 1.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.conftest import VirtualClock
from tests.fakes.audio import FakeAudioIO
from tests.fakes.audio_device import (
    BLOCK_SAMPLES,
    CaptureSink,
    FakeAudioDevice,
    capture_out,
    make_audio_io,
    t_ms_of,
)

pytestmark = [pytest.mark.contract]

#: the session time at which capture is armed (the task's regime: James's capture opened ~11 s into the session)
ARM_MS = 11_400
#: 24 kHz, 1920 samples: one STT frame is 80 ms of audio
STT_FRAME_MS = 80.0


async def _settle(sink: CaptureSink) -> None:
    """Let the sink's background drain run until it stops growing (the fake yields with ``sleep(0)``)."""
    last, quiet = -1, 0
    while quiet < 5:
        await asyncio.sleep(0)
        quiet = quiet + 1 if len(sink) == last else 0
        last = len(sink)


async def _first_frames(backend: str) -> list[float]:
    clock = VirtualClock(now_ms=ARM_MS)
    if backend == "real":
        device = FakeAudioDevice(clock, capture_buffer_frames=BLOCK_SAMPLES)
        io: Any = make_audio_io(device)
        sink = CaptureSink(io, "stt")
        await capture_out(device, 400, sink)
    else:
        io = FakeAudioIO(clock, capture_origin_ms=0)
        sink = CaptureSink(io, "stt")
        await _settle(sink)
        for _ in range(40):
            clock.advance(10)
            await _settle(sink)
    frames = [t_ms_of(item) for item in sink.items]
    sink.raise_error()
    await sink.close()
    return frames


@pytest.mark.parametrize("backend", ["real", "fake"])
async def test_first_stt_frame_t_ms_is_the_capture_origin_not_the_session_clock(
    backend: str,
) -> None:
    """Capture armed at session 11 400: the first STT frame's ``t_ms`` is 0 (the ADC position at open), the next
    ones step by 80 ms, and no frame is stamped on the session clock (``>= 11 400``). Expected values by hand."""
    frames = await _first_frames(backend)
    assert len(frames) >= 3, f"{backend}: 400 ms of capture gave {len(frames)} STT frames"
    assert frames[:3] == [0.0, STT_FRAME_MS, 2 * STT_FRAME_MS], (
        f"{backend}: STT frames count audio from the capture open (0, 80, 160), not from the session clock; "
        f"got {frames[:3]}"
    )
    assert all(t < ARM_MS for t in frames), (
        f"{backend}: a frame was stamped on the session clock; {frames}"
    )
