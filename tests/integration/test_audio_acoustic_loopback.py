"""Speaker → microphone loopback: T-0006 criteria 10 and 11.

What this module produces is the one number T-0006b cannot get later for free: the **echo path
delay** of this room, and the hangover the capture gate will need (= p95 + one block). v1 ships
no AEC (``Profile.aec == "none"``, docs/01 R4, docs/06 R3), so ERLE, double-talk and any APM
judgement are explicitly *not* here (docs/03 B15, deferred with T-0006b).

Markers: ``integration`` + ``audio`` + ``slow``. No GPU, no human in the room.

**The Dummy-Output trap (criterion 11).** At planning time the box's default sink was
``auto_null`` ("Dummy Output") because the DP cable was unplugged, and the only capture device is
the Logitech C925e webcam mic. A test that "finds no echo" through a dead speaker and passes is
worse than no test, so: a missing or dummy sink **skips with the device named**, and a rig that is
present but renders nothing **fails**. ``test_calibration_tone_is_heard_or_the_test_errors`` runs
first in this module on purpose.

Regime to quote with every number (CLAUDE.md): box ``gpu-box``, PipeWire 1.0.5,
sink named in the report (``ASTRA_ACOUSTIC_SINK``), volume as reported by ``wpctl``, Logitech
C925e mic, mic-to-speaker distance in cm, room noise floor, n = 10, **speakers** (this rig has no
headset path), profile=lean, vLLM and STT not running.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pytest

from local_astra.contracts.speech import CAPTURE_BLOCK_MS
from tests.fakes.audio_device import (
    DEVICE_RATE,
    MonitorRecording,
    PcmUtterance,
    attr_or_fail,
    call,
    chirp_fixture,
    chirp_reference_48k,
    default_sink_name,
    fft_best_lag,
    import_or_fail,
    missing_node_reason,
    pw_nodes,
    rendered_ms_from_chirp,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.audio,
    pytest.mark.slow,
]

#: Set this to the node name of a sink that is connected to a speaker. There is no way to detect
#: "a speaker is plugged in" from software, which is exactly why criterion 11 exists.
SINK_ENV = "ASTRA_ACOUSTIC_SINK"
DUMMY_SINKS = {"auto_null", "Dummy Output"}
N = 10
#: The mic is 16 kHz-band-limited USB audio in a quiet room; the ladder scripts measured ~0.116
#: peak of ambient noise here (``scripts/demo/_listen.py``), so anything above 0.2 is signal.
HEARD_PEAK = 0.2


def sink_or_skip() -> str:
    sink = os.environ.get(SINK_ENV)
    if sink is None:
        current = default_sink_name()
        if current is None or current in DUMMY_SINKS:
            pytest.skip(
                f"the PipeWire default sink is {current!r} (a dummy: no speaker, the DP cable is "
                f"unplugged) — set {SINK_ENV}=<sink node name> to run the acoustic loopback"
            )
        sink = current
    reason = missing_node_reason(sink, f"{SINK_ENV}={sink}")
    if reason is not None:
        pytest.skip(reason)
    return sink


def mic_or_skip() -> str:
    nodes = pw_nodes() or {}
    mics = [name for name in nodes if name.startswith("alsa_input.")]
    if not mics:
        pytest.skip("no ALSA capture node in pw-dump (the Logitech C925e mic is the only one)")
    return mics[0]


def build_io(sink: str) -> object:
    from local_astra.conductor.audio_io import PipeWireAudioIO

    module = import_or_fail("local_astra.conductor.audio_device", "T-0006 criterion 1")
    open_device = attr_or_fail(module, "open_device", "criterion 1: the real PipeWire backend")
    return PipeWireAudioIO(open_device(sink=sink, block_ms=CAPTURE_BLOCK_MS, latency_ms=30.0))


def chirp_utterance(uid: str = "chirp") -> PcmUtterance:
    samples, rate = chirp_fixture()
    assert rate == 24000
    return PcmUtterance(uid, samples=samples, text="chirp")


def p(values: list[float], q: float) -> float:
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(q * 100) - 1])


async def test_calibration_tone_is_heard_or_the_test_errors(tmp_path: Path) -> None:
    """Criterion 11: prove the speaker is audible to the mic before believing any echo number.

    Runs first in the module. If the lead-in chirp is not in the recording above the energy
    floor, this **fails** — it never skips, because "no echo found" through a dead speaker looks
    exactly like a perfect result. The room's noise floor is measured in the same breath: every
    acoustic number in the report carries it (CLAUDE.md regime).
    """
    sink, mic = sink_or_skip(), mic_or_skip()

    with MonitorRecording(mic, tmp_path / "floor.wav", capture_sink=False) as silence:
        await asyncio.sleep(1.0)
    quiet, _ = silence.read()
    floor = float(np.max(np.abs(quiet)))
    assert floor < HEARD_PEAK, (
        f"the room's noise floor peaks at {floor:.3f} on {mic!r}, at or above the {HEARD_PEAK} "
        f"threshold these tests call 'heard': every echo number would be noise"
    )

    with MonitorRecording(mic, tmp_path / "calib.wav", capture_sink=False) as rec:
        io = build_io(sink)
        await play_chirp(io)
        await asyncio.sleep(2.0)
        await call(io, "flush", "AudioIO protocol")
    x, rate = rec.read()
    peak = float(np.max(np.abs(x)))
    assert peak > HEARD_PEAK, (
        f"the calibration chirp played through {sink!r} peaked at {peak:.3f} on {mic!r} — below "
        f"the {HEARD_PEAK} energy floor (noise floor {floor:.3f}). The speaker is muted, "
        f"unplugged or not the sink the module opened; every echo number measured on this rig "
        f"would be a lie (criterion 11)."
    )
    assert rate == DEVICE_RATE
    print(f"\ncalibration: sink={sink!r} mic={mic!r} peak={peak:.3f} noise floor={floor:.3f}")


async def play_chirp(io: object, uid: str = "chirp") -> None:
    from tests.fakes.audio_device import play

    await play(io, chirp_utterance(uid), source="chirp")


async def test_echo_path_delay_is_measured_and_reported(tmp_path: Path) -> None:
    """Criterion 10: speaker → mic delay, n = 10, p50/p95, and the gate's hangover.

    The delay is the lag that maximises the cross-correlation between the chirp as sent and the
    chirp as heard — a linear chirp has a sharp autocorrelation peak, which is why the fixture is
    a chirp and not a tone. The number T-0006b needs is ``p95 + one block``.
    """
    sink, mic = sink_or_skip(), mic_or_skip()
    at_48k = chirp_reference_48k(2.0)  # 2 s of the chirp is plenty for the correlation peak

    delays: list[float] = []
    for i in range(N):
        with MonitorRecording(mic, tmp_path / f"echo{i}.wav", capture_sink=False) as rec:
            io = build_io(sink)
            started = time.monotonic()
            await play_chirp(io, f"e{i}")
            await asyncio.sleep(2.5)
            await call(io, "flush", "AudioIO protocol")
        heard, rate = rec.read()
        assert rate == DEVICE_RATE
        assert float(np.max(np.abs(heard))) > HEARD_PEAK, (
            f"run {i}: nothing audible on {mic!r} (criterion 11)"
        )
        lag_samples = fft_best_lag(at_48k, heard)
        delays.append(1000.0 * lag_samples / rate - 1000.0 * (0.0))
        assert 0.0 <= delays[-1] <= 1000.0, (
            f"run {i}: echo path delay of {delays[-1]:.1f} ms is not physical — the correlation "
            f"peak did not find the chirp (elapsed {time.monotonic() - started:.1f} s)"
        )
    hangover_ms = p(delays, 0.95) + CAPTURE_BLOCK_MS
    print(
        f"\necho path delay (ms), sink={sink!r} mic={mic!r} n={N}, speakers: "
        f"p50={p(delays, 0.5):.1f} p95={p(delays, 0.95):.1f} "
        f"spread={max(delays) - min(delays):.1f}; hangover for T-0006b's gate = "
        f"{hangover_ms:.1f} ms (p95 + one {CAPTURE_BLOCK_MS} ms block)"
    )
    assert statistics.pstdev(delays) <= 50.0, (
        f"the echo path delay varies by {statistics.pstdev(delays):.1f} ms across {N} runs; a "
        f"gate hangover derived from it would be meaningless"
    )


async def test_played_ms_matches_the_microphone(tmp_path: Path) -> None:
    """The real device path's clock, checked through the air.

    The fixed acoustic delay cancels: what is compared is *how much of the chirp* the microphone
    heard (read off its frequency ramp, which the room does not change) against what ``flush()``
    reported, not when either happened.
    """
    sink, mic = sink_or_skip(), mic_or_skip()
    with MonitorRecording(mic, tmp_path / "clock.wav", capture_sink=False) as rec:
        io = build_io(sink)
        await play_chirp(io, "clock")
        await asyncio.sleep(2.0)
        reported = float(await call(io, "flush", "AudioIO protocol"))
    heard, rate = rec.read()
    assert float(np.max(np.abs(heard))) > HEARD_PEAK, f"nothing audible on {mic!r} (criterion 11)"
    content_ms = rendered_ms_from_chirp(heard, rate, threshold=HEARD_PEAK / 2.0)
    assert reported - content_ms <= 20.0 + 30.0, (
        f"flush() reported {reported:.0f} ms but the microphone heard {content_ms:.0f} ms of the "
        f"chirp: over-reporting (criterion 2; 30 ms of the budget is the acoustic estimate)"
    )
    assert content_ms - reported <= 40.0 + 30.0, (
        f"flush() reported {reported:.0f} ms, the microphone heard {content_ms:.0f} ms"
    )
    print(f"\nplayed_ms={reported:.0f} ms vs microphone {content_ms:.0f} ms (sink={sink!r})")
