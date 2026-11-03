"""The capture path against a real PipeWire device: T-0021 criteria 1-3, 9, 11, 12(f).

Markers: ``integration`` + ``audio``, **not** ``gpu`` — a null sink and its monitor, no GPU, no
microphone, no speaker, no display (``tests/integration/README.md``; T-0006 ruling A4).

The oracle is the monitor of ``astra-test-sink``: it is a **bit-exact echo path**, which is the
strictest possible test of a self-speech gate and needs nobody in the room. What the module says
it captured is compared with what was played into the sink — the playback reference ring
(T-0006's ``playback_reference``) and an independent ``pw-record`` capture, never the module's
own account of itself.

Two failure modes are kept apart on purpose (criterion 11): **no rig** is a skip naming the node
that is missing; **a rig that renders nothing** is a failure. The default sink on this box is a
"Dummy Output" while the DP cable is unplugged, so "no echo found" and "no audio path at all"
look identical unless something checks — ``test_rig_is_alive_or_the_test_errors`` is that
something, it runs first, and it never skips and never passes quietly.

Regime for every number here: box ``gpu-box``, PipeWire 1.0.5 / libpipewire 1.0.5,
null sink ``astra-test-sink`` with its ``.monitor`` as the capture source (``scripts/audio_rig.sh``),
requested quantum 480/48000 with the negotiated one reported, 48 kHz mono f32 capture → 24 kHz
f32 STT frames, source ``tests/fixtures/audio/chirp_5s_24k.wav``, ``profile=lean``,
``audio_path=loopback``, vLLM and STT not running, no other PipeWire clients, n ≥ 3000 blocks
over a 60 s run, seed = 0, git tree clean. p50 and p95 are reported, never a mean.
"""

from __future__ import annotations

import asyncio
import resource
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from local_astra.contracts.speech import CAPTURE_BLOCK_MS
from tests.fakes.audio_device import (
    BLOCK_SAMPLES,
    DEVICE_RATE,
    NULL_SINK,
    STT_BLOCK_MS,
    STT_BLOCK_SAMPLES,
    STT_RATE,
    WHY_CAPTURE,
    CaptureSink,
    MonitorRecording,
    attr_or_fail,
    capture_stats,
    chirp_fixture,
    concat,
    fft_best_lag,
    import_or_fail,
    maybe_await,
    missing_node_reason,
    read_wav,
    residual_db,
    samples_of,
    stat,
    t_ms_of,
)

pytestmark = [pytest.mark.integration, pytest.mark.audio]

HOW = "run `bash scripts/audio_rig.sh up` (T-0021 criterion 11, inherited from T-0006)"
#: Ruling A11/F6: ``<sink>.monitor`` is a pipewire-pulse name and is not a PipeWire node —
#: neither ``pw-record --target`` nor the device can resolve it. Both sides name the *node*
#: and ask for its monitor with ``stream.capture.sink=true`` (T-0006's convention, the
#: property ``MonitorRecording`` sets and the one ``open_device(source=...)`` routes through
#: the PipeWire environment).
MONITOR_OF = NULL_SINK
CHIRP = Path("tests/fixtures/audio/chirp_5s_24k.wav")
#: criterion 11: the rig is alive if the monitor carries the calibration chirp above this floor.
ALIVE_DBFS = -40.0
#: criterion 9: the gated STT stream is silence.
GATED_DBFS = -60.0
#: criterion 9: the ungated positive control must be loud, or "silence" proves nothing.
CONTROL_DBFS = -20.0
#: criterion 11: the module's slice of the capture budget (docs/01 §8 "Mic capture + AEC +
#: resample — 20 ms"). The structural +80 ms of *filling* a 1920-sample frame is the STT
#: server's block size, not this module's latency: it is excluded here and stated in the report.
HANDOFF_P95_MS = 20.0
SEED = 0


@pytest.fixture(autouse=True)
def _rig() -> None:
    reason = missing_node_reason(NULL_SINK, HOW)
    if reason is not None:
        pytest.skip(reason)


def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))) if x.size else 0.0
    return 20.0 * np.log10(max(rms, 1e-12))


def p(values: list[float], q: float) -> float:
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(q * 100) - 1])


def open_capture_device(**kwargs: Any) -> Any:
    """The real backend, injected as criterion 1 requires. Imported *inside* the test: no device
    library may be imported at collection time (PortAudio may not be installed — P1)."""
    module = import_or_fail("local_astra.conductor.audio_device", WHY_CAPTURE)
    open_device = attr_or_fail(module, "open_device", "T-0006 ruling A1: the real backend")
    try:
        return open_device(
            sink=NULL_SINK,
            source=MONITOR_OF,  # the node, not "<sink>.monitor" (ruling A11)
            block_ms=CAPTURE_BLOCK_MS,
            latency_ms=30.0,
            **kwargs,
        )
    except TypeError as exc:
        pytest.fail(
            f"open_device() does not take the capture source yet: T-0021 ruling A1 fixes the "
            f"signature as open_device(sink=, source=, block_ms=, latency_ms=), and criterion 1 "
            f"wants capture and playback served by one device object ({exc})"
        )
    except (OSError, RuntimeError) as exc:
        # T-0006 merged the sounddevice backend: open_device() wraps the missing shared library
        # in a RuntimeError naming prerequisite P1, so both types are caught here.
        if "PortAudio" in str(exc) or "libportaudio2" in str(exc):
            pytest.fail(
                f"prerequisite P1: the sounddevice backend needs libportaudio2 and it is not "
                f"installed (`sudo apt-get install -y libportaudio2`). This is an error, not a "
                f"skip: T-0021 criterion 11 — a rig-dependent test must never report success "
                f"through a path that cannot open. ({exc})"
            )
        raise


def build_io(**kwargs: Any) -> tuple[Any, Any]:
    from local_astra.conductor.audio_io import PipeWireAudioIO

    device = open_capture_device()
    return PipeWireAudioIO(device, **kwargs), device


def gate_off() -> Any:
    """The ungated positive control of criterion 9: a gate that never closes."""
    module = import_or_fail("local_astra.conductor.audio_capture", WHY_CAPTURE)
    cls = attr_or_fail(module, "SelfSpeechGate", "criterion 8")
    return cls(lambda _t_ms: False, 0.0)


def play_chirp_into_the_sink() -> subprocess.Popen[bytes]:
    """An audio source that is not the module under test (criterion 11's independence)."""
    return subprocess.Popen(
        ["pw-play", "--target", NULL_SINK, str(CHIRP)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def test_rig_is_alive_or_the_test_errors(tmp_path: Path) -> None:
    """Criterion 11, first in the module: prove the path renders before believing any number.

    Not marked ``pending``: it tests the rig, not T-0021, and it must be able to pass today.
    """
    with MonitorRecording(MONITOR_OF, tmp_path / "calibration.wav", capture_sink=True):
        proc = play_chirp_into_the_sink()
        assert proc.wait(timeout=30) == 0, (
            f"pw-play could not play {CHIRP} into {NULL_SINK!r}: "
            f"{(proc.stderr.read() if proc.stderr else b'').decode()[:400]}"
        )
    recorded, rate = read_wav(tmp_path / "calibration.wav")
    level = dbfs(recorded)
    assert rate == DEVICE_RATE, f"the monitor recorded at {rate} Hz, expected {DEVICE_RATE}"
    if level <= ALIVE_DBFS:
        pytest.fail(
            f"the monitor of {MONITOR_OF} carries {level:.1f} dBFS while the calibration chirp "
            f"was playing "
            f"(floor {ALIVE_DBFS} dBFS). The rig is not rendering — every echo/gate number "
            f"measured here would be 'no energy found' through a dead path (criterion 11). "
            f"This is a failure, never a skip."
        )


async def test_captured_monitor_reproduces_the_played_chirp(tmp_path: Path) -> None:
    """Criteria 1 and 9's precondition: the capture path really carries the sink's audio.

    The oracle is T-0006's ``playback_reference`` — the exact mixed output handed to the device —
    aligned to the capture by cross-correlation. A monitor is a bit-exact echo path, so after
    the fixed monitor delay the residual is quantisation only.
    """
    io, _device = build_io(gate=gate_off())
    raw = CaptureSink(io, "raw")
    samples, rate = chirp_fixture()
    assert rate == 24_000
    from tests.fakes.audio_device import PcmUtterance

    await attr_or_fail(io, "play", "AudioIO protocol")(PcmUtterance("chirp", samples=samples))
    await asyncio.sleep(2.0)
    captured = np.asarray(concat(list(raw.items)), dtype=np.float64)
    reference, _first_ms = await maybe_await(
        attr_or_fail(io, "playback_reference", "T-0006 criterion 10")(2000.0)
    )
    await raw.close()

    reference = np.asarray(reference, dtype=np.float64)
    assert captured.size > DEVICE_RATE, f"only {captured.size} samples captured in 2 s"
    lag = fft_best_lag(reference, captured)
    aligned = captured[lag : lag + reference.size]
    residual = residual_db(reference[: aligned.size], aligned)
    assert residual <= -40.0, (
        f"the monitor capture differs from what was sent to the sink by {residual:.1f} dB "
        f"(want ≤ −40 dB) at lag {1000.0 * lag / DEVICE_RATE:.1f} ms: the capture path is not "
        f"carrying the sink's audio (criterion 9 would then prove nothing)"
    )


async def test_capture_timestamps_track_the_playback_clock() -> None:
    """Criteria 1 and 3 on real hardware: one clock base, constant lag, jitter p95 ≤ 10 ms."""
    io, _device = build_io(gate=gate_off())
    raw = CaptureSink(io, "raw")
    samples, _rate = chirp_fixture()
    from tests.fakes.audio_device import PcmUtterance

    await attr_or_fail(io, "play", "AudioIO protocol")(PcmUtterance("chirp", samples=samples))
    await asyncio.sleep(3.0)
    blocks = list(raw.items)
    await raw.close()

    assert len(blocks) >= 200, f"{len(blocks)} blocks in 3 s of capture"
    window = 10  # 100 ms of capture per measurement
    lags_ms: list[float] = []
    reference, first_ms = await maybe_await(
        attr_or_fail(io, "playback_reference", "T-0006 criterion 10")(3000.0)
    )
    reference = np.asarray(reference, dtype=np.float64)
    for i in range(0, len(blocks) - window, window):
        chunk = np.asarray(concat(blocks[i : i + window]), dtype=np.float64)
        if dbfs(chunk) < -50.0:
            continue
        lag = fft_best_lag(reference, chunk)
        # where the module says this chunk was captured, minus where its audio sits in the
        # reference: a constant difference is one clock base, a drifting one is two.
        lags_ms.append(t_ms_of(blocks[i]) - (first_ms + 1000.0 * lag / DEVICE_RATE))
    assert len(lags_ms) >= 20, f"only {len(lags_ms)} usable windows (n ≥ 20 wanted)"

    median = statistics.median(lags_ms)
    jitter = sorted(abs(x - median) for x in lags_ms)
    jitter_p95 = p(jitter, 0.95)
    assert jitter_p95 <= 10.0, (
        f"the lag between the capture stamps and the playback clock varies by {jitter_p95:.1f} ms "
        f"p95 around {median:.1f} ms (n={len(lags_ms)}, seed={SEED}); criteria 1 and 3 want one "
        f"clock base and a constant offset, not two clocks subtracted from each other"
    )


async def test_blocks_stay_10ms_at_the_negotiated_quantum() -> None:
    """Criterion 2 on the graph: the quantum is whatever PipeWire granted; blocks are 10 ms."""
    io, device = build_io()
    raw = CaptureSink(io, "raw")
    await asyncio.sleep(2.0)
    blocks = list(raw.items)
    stats = await capture_stats(io)
    await raw.close()

    quantum = float(stat(stats, "quantum_ms"))
    negotiated = float(attr_or_fail(device, "quantum_ms", "T-0006 criterion 1"))
    assert quantum == pytest.approx(negotiated, abs=1e-6), (
        f"capture_stats() reports {quantum} ms while the device negotiated {negotiated} ms: "
        f"criterion 2 reports the negotiated quantum, never the requested 480/48000"
    )
    sizes = {int(samples_of(b).size) for b in blocks}
    assert sizes == {BLOCK_SAMPLES}, f"block sizes {sorted(sizes)} at a {quantum} ms quantum"
    stamps = [t_ms_of(b) for b in blocks]
    steps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    assert steps == pytest.approx([float(CAPTURE_BLOCK_MS)] * len(steps), abs=1e-3), (
        f"the cadence wandered at a {quantum} ms quantum: {sorted(set(steps))[:5]}"
    )


async def test_gate_silences_the_stt_stream_with_the_ungated_positive_control() -> None:
    """Criterion 9: a bit-exact echo path, gated and ungated — the control is mandatory.

    Without the second half, "the STT stream is silent" is equally true of a capture path that
    produces nothing at all.
    """
    samples, _rate = chirp_fixture()
    from tests.fakes.audio_device import PcmUtterance

    async def run(gate: Any) -> float:
        io, _device = build_io(**({} if gate is None else {"gate": gate}))
        stt = CaptureSink(io, "stt")
        await attr_or_fail(io, "play", "AudioIO protocol")(
            PcmUtterance("chirp", samples=samples[: 24_000 * 2])
        )
        await asyncio.sleep(2.5)
        frames = list(stt.items)
        await stt.close()
        assert frames, "no STT frames at all"
        # the playback window only: the first and last 200 ms are lead-in and hangover
        y = np.asarray(concat(frames), dtype=np.float64)
        lo, hi = round(0.3 * STT_RATE), round(1.9 * STT_RATE)
        assert hi <= y.size, f"only {y.size / STT_RATE:.1f} s of STT audio"
        return dbfs(y[lo:hi])

    ungated = await run(gate_off())
    assert ungated >= CONTROL_DBFS, (
        f"the **positive control** failed: with the gate off the monitor's echo of our own "
        f"playback reaches the STT tap at only {ungated:.1f} dBFS (want ≥ {CONTROL_DBFS}). "
        f"Nothing below this line means anything until it does — a dead capture path is silent "
        f"too (criterion 9)"
    )
    gated = await run(None)
    assert gated <= GATED_DBFS, (
        f"with the gate on the STT tap still hears {gated:.1f} dBFS of our own voice through a "
        f"bit-exact echo path (want ≤ {GATED_DBFS}); the ungated control measured "
        f"{ungated:.1f} dBFS, so the path is alive (criterion 9)"
    )


@pytest.mark.slow
async def test_capture_to_stt_handoff_p95_and_zero_xruns_over_60s() -> None:
    """Criterion 11: n ≥ 3000 blocks, seed 0 — p95 handoff ≤ 20 ms, no xrun, no drop, flat RSS.

    The delay measured is from a frame's **last** sample's capture time to the moment
    ``stt_blocks()`` yielded it. The 80 ms it takes to *fill* the frame is the STT server's block
    size and is deliberately not in this number; it is stated in the report instead.
    """
    io, _device = build_io()
    raw = CaptureSink(io, "raw")
    stt = CaptureSink(io, "stt")
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.monotonic()
    while time.monotonic() - started < 60.0:
        await asyncio.sleep(0.5)
    frames = list(stt.items)
    blocks = list(raw.items)
    stats = await capture_stats(io)
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    await raw.close()
    await stt.close()

    assert len(blocks) >= 3000, f"n = {len(blocks)} blocks in 60 s (criterion 11 wants ≥ 3000)"
    assert len(frames) >= 700, f"{len(frames)} STT frames in 60 s (60000/{STT_BLOCK_MS} ≈ 750)"
    delays = [
        float(attr_or_fail(f, "yielded_ms", "criterion 10"))
        - float(attr_or_fail(f, "end_ms", "criterion 10"))
        for f in frames
    ]
    p50, p95 = p(delays, 0.5), p(delays, 0.95)
    assert p95 <= HANDOFF_P95_MS, (
        f"capture → stt_blocks() handoff p95 = {p95:.1f} ms (p50 {p50:.1f} ms, n={len(frames)}, "
        f"seed={SEED}); docs/01 §8 allots 20 ms to mic capture + resample. A threshold that "
        f"cannot be met is a spec change with its measurement, never an edited assertion"
    )
    assert int(stat(stats, "xruns")) == 0, f"{stat(stats, 'xruns')} xruns in a 60 s idle run"
    assert int(stat(stats, "dropped_blocks")) == 0, f"{stat(stats, 'dropped_blocks')} blocks lost"
    assert {int(samples_of(f).size) for f in frames} == {STT_BLOCK_SAMPLES}
    grew_mb = (rss_after - rss_before) / 1024.0
    assert grew_mb < 5.0, (
        f"RSS grew {grew_mb:.1f} MB over 60 s of capture: the rings are not bounded (criterion 11)"
    )


async def test_event_loop_lag_under_duplex() -> None:
    """Criterion 12(f): 10 s of capture **and** playback at once, loop lag p99 ≤ 20 ms."""
    io, _device = build_io()
    raw = CaptureSink(io, "raw")
    stt = CaptureSink(io, "stt")
    samples, _rate = chirp_fixture()
    from tests.fakes.audio_device import PcmUtterance

    await attr_or_fail(io, "play", "AudioIO protocol")(PcmUtterance("chirp", samples=samples))

    lags: list[float] = []
    tick = 0.01
    previous = time.monotonic()
    while time.monotonic() - previous < 10.0 or len(lags) < 500:
        before = time.monotonic()
        await asyncio.sleep(tick)
        lags.append(1000.0 * (time.monotonic() - before - tick))
        if len(lags) >= 1000:
            break
    blocks = list(raw.items)
    await raw.close()
    await stt.close()

    assert len(blocks) >= 500, f"only {len(blocks)} blocks captured during the duplex run"
    lag_p99 = p(lags, 0.99)
    assert lag_p99 <= 20.0, (
        f"event-loop lag p99 = {lag_p99:.1f} ms over {len(lags)} samples of duplex capture + "
        f"playback (.claude/rules/conductor.md: never block the loop for more than 20 ms). "
        f"Duplex is the new case and the likeliest regression (criterion 12f)"
    )
