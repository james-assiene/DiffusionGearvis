"""The playback path against a real PipeWire device: T-0006 criteria 1-6, 9, 13.

Markers: ``integration`` + ``audio``, **not** ``gpu`` — this rig is a null sink and a monitor
capture, no GPU is involved (``tests/integration/README.md`` still says every integration module
also carries ``gpu``; that line needs the correction T-0006 asks for, and it is not mine to make
in this PR — see the report).

The oracle is the monitor recording of ``astra-test-sink`` made with ``pw-record``: what the
module *says* it played is compared with what the sink actually received, and the chirp fixture's
frequency ramp (tests/fixtures/audio/MANIFEST.md) turns that recording into a clock that owes the
module nothing.

Two failure modes are kept apart on purpose (criterion 11): **no rig** is a skip that names the
node that is missing, **a rig that renders nothing** is a failure. At planning time the box's
default sink was a "Dummy Output" — a test that skipped on silence would report success through a
dead speaker.

Regime for every number here: box ``gpu-box``, PipeWire 1.0.5, null sink
``astra-test-sink`` + ``pw-record`` monitor capture, requested quantum 480/48000 (the negotiated
one is reported, criterion 1), 48 kHz mono f32, source ``tests/fixtures/audio/chirp_5s_24k.wav``
and ``pocket_tts_sentence.wav``, profile=lean, vLLM and STT not running, n = 30, seed = 0.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from local_astra.contracts.speech import CAPTURE_BLOCK_MS
from tests.fakes.audio_device import (
    DEVICE_RATE,
    NULL_SINK,
    EventSink,
    MonitorRecording,
    PcmUtterance,
    attr_or_fail,
    call,
    chirp_correlation,
    chirp_fixture,
    import_or_fail,
    max_abs_step,
    missing_node_reason,
    nonzero_regions,
    play,
    played_ms,
    rendered_ms_from_chirp,
    residual_db,
    tts_fixture,
    words_of,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.audio,
]

HOW = "run `bash scripts/audio_rig.sh up` (T-0006 criterion 11)"
#: The chirp oracle is accurate to ~8 ms on this fixture (checked while authoring, by evaluating
#: it at known offsets), so the criterion-2 bounds are widened by that much and no more.
ORACLE_MS = 8.0
OVER_MS, UNDER_MS = 20.0, 40.0
N = 30
SEED = 0


@pytest.fixture(autouse=True)
def _rig() -> None:
    reason = missing_node_reason(NULL_SINK, HOW)
    if reason is not None:
        pytest.skip(reason)


def build_io() -> tuple[object, object]:
    """The real backend, injected exactly as criterion 1 requires. Imported inside the test:
    ``sounddevice`` must never be imported at collection time (PortAudio is not installed)."""
    from local_astra.conductor.audio_io import PipeWireAudioIO

    module = import_or_fail("local_astra.conductor.audio_device", "T-0006 criterion 1")
    open_device = attr_or_fail(module, "open_device", "criterion 1: the real PipeWire backend")
    device = open_device(sink=NULL_SINK, block_ms=CAPTURE_BLOCK_MS, latency_ms=30.0)
    return PipeWireAudioIO(device), device


def chirp_utterance(uid: str = "chirp") -> PcmUtterance:
    samples, rate = chirp_fixture()
    assert rate == 24000
    return PcmUtterance(uid, samples=samples, text="chirp")


def p(values: list[float], q: float) -> float:
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(q * 100) - 1])


def test_the_sink_exists_and_the_monitor_records_what_is_played(tmp_path: Path) -> None:
    """Criterion 11: prove the rig renders *this* chirp before believing any number from it.

    The gate is the correlation with the chirp fixture, not the recording's energy (ruling A7).
    An energy floor cannot tell a monitor of the sink from ``pw-record``'s fallback to the
    default *source*: the C925e's room noise peaks around 0.12 and passes any floor low enough
    to be safe, while scoring ≈ 0.05 against the chirp template. A recording of the chirp scores
    ≈ 1.0 whatever the volume.
    """
    with MonitorRecording(NULL_SINK, tmp_path / "calib.wav") as rec:
        subprocess.run(
            ["pw-play", "--target", NULL_SINK, str(Path("tests/fixtures/audio/chirp_5s_24k.wav"))],
            check=True,
            timeout=30,
            stdout=subprocess.DEVNULL,
        )
    x, rate = rec.read()
    assert rate == DEVICE_RATE
    rho = chirp_correlation(x, rate)
    assert rho >= 0.3, (
        f"the recording of {NULL_SINK!r}'s monitor correlates {rho:.3f} with the chirp that was "
        f"playing (peak {float(np.max(np.abs(x))):.3f}): it is room noise from the default "
        f"source, or the sink rendered nothing. The rig is not usable (criterion 11 — this must "
        f"fail, never skip)"
    )
    heard = rendered_ms_from_chirp(x, rate)
    assert 4800.0 <= heard <= 5200.0, f"the monitor holds {heard:.0f} ms of the 5000 ms chirp"


async def test_played_ms_matches_the_monitor_within_20ms(tmp_path: Path) -> None:
    """Criterion 2 on real hardware: n = 30 flush points, seed = 0.

    The comparison is against the chirp's own frequency ramp in the monitor recording, not
    against elapsed wall time, so the device's buffering is inside the measurement.
    """
    rng = random.Random(SEED)
    errors: list[float] = []
    stops: list[float] = []
    for i in range(N):
        cut_ms = rng.uniform(300.0, 4500.0)
        with MonitorRecording(NULL_SINK, tmp_path / f"cut{i}.wav") as rec:
            io, _ = build_io()
            await play(io, chirp_utterance(f"c{i}"), source="chirp")
            play_returned = time.monotonic()
            await asyncio.sleep(cut_ms / 1000.0)
            called = time.monotonic()
            reported = float(await call(io, "flush", "AudioIO protocol"))
        heard = rendered_ms_from_chirp(*rec.read())
        errors.append(reported - heard)
        stops.append(heard - 1000.0 * (called - play_returned))
        assert reported - heard <= OVER_MS + ORACLE_MS, (
            f"cut {cut_ms:.0f} ms: flush() reported {reported:.0f} ms but the monitor holds "
            f"{heard:.0f} ms — over-reporting corrupts history (criterion 2)"
        )
        assert heard - reported <= UNDER_MS + ORACLE_MS, (
            f"cut {cut_ms:.0f} ms: flush() reported {reported:.0f} ms, the monitor holds "
            f"{heard:.0f} ms — under-reporting beyond latency + one block (criterion 2)"
        )
    assert p(stops, 0.95) <= 50.0, (
        f"flush() → silence p95 = {p(stops, 0.95):.1f} ms (criterion 9 allows 50 ms). Like the "
        f"pause measurement this is a lower bound, so it cannot fail spuriously."
    )
    print(
        f"\nplayed_ms − monitor (ms), n={N}, seed={SEED}: p50={p(errors, 0.5):.1f} "
        f"p95={p(errors, 0.95):.1f} max={max(errors):.1f} min={min(errors):.1f}"
        f"\nflush → silence (ms, lower bound), n={N}: p50={p(stops, 0.5):.1f} "
        f"p95={p(stops, 0.95):.1f}"
    )


async def test_pause_silences_the_output_and_does_not_click(tmp_path: Path) -> None:
    """Criteria 3 and 9: p95 ≤ 50 ms / p50 ≤ 30 ms to silence, and no step above 0.1.

    **How the latency is measured, and why it cannot fail spuriously.** The monitor recording is
    not aligned to the wall clock (``pw-record`` connects when it connects), so the quantity
    measured is ``chirp position of the last audible sample − wall time from play() returning to
    the pause() call``. The first sample leaves the sink somewhere between those two instants, so
    this is a *lower bound* on the true pause latency (it understates it by the startup delay, at
    most ``latency_ms + block_ms`` ≤ 40 ms). A correct implementation therefore cannot fail this
    test, while the failure the criterion names — ≥ 100 ms of queued audio playing out — still
    exceeds 50 ms after the subtraction. The authoritative number is the direct measurement of
    ``scripts/audio_loopback.py --measure pause --n 30 --seed 0`` (criterion 9's evidence).
    """
    rng = random.Random(SEED)
    latencies: list[float] = []
    for i in range(N):
        at_ms = rng.uniform(500.0, 4000.0)
        with MonitorRecording(NULL_SINK, tmp_path / f"pause{i}.wav") as rec:
            io, _ = build_io()
            await play(io, chirp_utterance(f"p{i}"), source="chirp")
            play_returned = time.monotonic()
            await asyncio.sleep(at_ms / 1000.0)
            called = time.monotonic()
            await call(io, "pause", "AudioIO protocol")
            await asyncio.sleep(0.5)
            await call(io, "flush", "AudioIO protocol")
        x, rate = rec.read()
        loud = np.flatnonzero(np.abs(x) > 0.02)
        assert loud.size > 0, "nothing was rendered at all (criterion 11)"
        heard_ms = rendered_ms_from_chirp(x, rate)
        latencies.append(heard_ms - 1000.0 * (called - play_returned))
        step = max_abs_step(np.concatenate([x[: int(loud[-1]) + 1], np.zeros(1)]))
        assert step <= 0.1, (
            f"pause at {at_ms:.0f} ms cut the waveform with a step of {step:.3f} full-scale — "
            f"criterion 3 wants a 2-10 ms fade"
        )
    assert p(latencies, 0.95) <= 50.0, (
        f"pause → silence p95 = {p(latencies, 0.95):.1f} ms (criterion 9 allows 50 ms; that is "
        f"the module's share of the 150 ms docs/02 §4 gives playback.pause). An implementation "
        f"that only stops the producer leaves the queued audio to play out and lands here."
    )
    assert p(latencies, 0.5) <= 30.0, f"pause → silence p50 = {p(latencies, 0.5):.1f} ms"
    print(
        f"\npause → silence (ms, lower bound), n={N}, seed={SEED}: p50={p(latencies, 0.5):.1f} "
        f"p95={p(latencies, 0.95):.1f}"
    )


async def test_resume_loses_no_sample(tmp_path: Path) -> None:
    """Criterion 4: the non-silent regions of the monitor concatenate back into the source."""
    paused_s = 0.5
    with MonitorRecording(NULL_SINK, tmp_path / "resume.wav") as rec:
        io, _ = build_io()
        await play(io, chirp_utterance("r"), source="chirp")
        await asyncio.sleep(1.0)
        await call(io, "pause", "AudioIO protocol")
        await asyncio.sleep(paused_s)
        await call(io, "resume", "AudioIO protocol")
        await asyncio.sleep(1.5)
        await call(io, "flush", "AudioIO protocol")

    x, rate = rec.read()
    regions = nonzero_regions(x, rate, threshold=0.02)
    regions = [(a, b) for a, b in regions if (b - a) > rate // 50]
    assert len(regions) == 2, f"expected sound, silence, sound; found {len(regions)} regions"
    stitched = np.concatenate([x[a:b] for a, b in regions])

    source, src_rate = chirp_fixture()
    assert src_rate == 24000
    expected = np.interp(  # 24 k → 48 k for comparison only (float64 oracle, not the module's)
        np.arange(2 * source.size) / 2.0, np.arange(source.size), source
    )[: stitched.size]
    gap_ms = 1000.0 * (regions[1][0] - regions[0][1]) / rate
    assert gap_ms - 1000.0 * paused_s <= 50.0, (
        f"the silence lasted {gap_ms:.0f} ms for a {1000.0 * paused_s:.0f} ms pause: resume() "
        f"needed more than the 50 ms p95 of criterion 9 to put the first sample out"
    )
    residual = residual_db(expected, stitched)
    assert residual <= -40.0, (
        f"the concatenated non-silent regions differ from the source by {residual:.1f} dB: "
        f"resume dropped or repeated samples (criterion 4)"
    )


async def test_played_text_matches_what_the_monitor_contains(tmp_path: Path) -> None:
    """Criterion 6 against the Pocket TTS fixture: the words in history are the words heard."""
    samples, rate, meta = tts_fixture()
    assert rate == 24000
    words = words_of([(w["text"], w["start_ms"], w["end_ms"]) for w in meta["words"]])
    source = meta["text"]

    with MonitorRecording(NULL_SINK, tmp_path / "text.wav") as rec:
        io, _ = build_io()
        sink = EventSink(io)
        await play(
            io,
            PcmUtterance("tts", samples=samples, words=words, text=source),
            source=source,
        )
        # cut at 2300 ms: "three," started at 1780 (+100 = 1880 ≤ 2300), "four," starts at 2510.
        await asyncio.sleep(2.3)
        reported = float(await call(io, "flush", "AudioIO protocol"))
        await sink.sync()

    played_text = sink.events[-1].played_text
    assert played_text == "one, two, three,", f"criterion 6: got {played_text!r}"
    x, srate = rec.read()
    heard = [r for r in nonzero_regions(x, srate, 0.02) if (r[1] - r[0]) > srate // 50]
    merged = 1
    for before, after in itertools.pairwise(heard):
        if (after[0] - before[1]) > srate // 5:  # 200 ms of silence separates two words
            merged += 1
    assert merged == len(played_text.split()), (
        f"the monitor contains {merged} spoken words but played_text claims "
        f"{len(played_text.split())}: {played_text!r}"
    )
    assert 2200.0 <= reported <= 2400.0, f"played_ms={reported:.0f} for a 2300 ms cut"
    await sink.close()


async def test_negotiated_quantum_is_reported_not_assumed() -> None:
    """Criterion 1: PipeWire's graph default is 1024/48000 ≈ 21 ms, not the 10 ms we ask for."""
    io, device = build_io()
    reported = float(attr_or_fail(io, "quantum_ms", "criterion 1: the negotiated quantum"))
    assert reported == pytest.approx(float(device.quantum_ms)), (
        "the module must report the device's negotiated quantum, not its own request"
    )
    dump = json.loads(subprocess.run(["pw-dump"], capture_output=True, text=True).stdout)
    graph = [
        obj["info"]["props"]
        for obj in dump
        if obj.get("info", {}).get("props", {}).get("node.name") == NULL_SINK
    ]
    assert graph, f"{NULL_SINK} vanished from pw-dump mid-test"
    quantum = graph[0].get("clock.quantum")
    rate = graph[0].get("clock.rate")
    if quantum and rate:
        assert reported == pytest.approx(1000.0 * quantum / rate, abs=0.5), (
            f"module reports {reported:.3f} ms, the graph runs at {quantum}/{rate} = "
            f"{1000.0 * quantum / rate:.3f} ms"
        )
    assert int(attr_or_fail(io, "block_ms", "criterion 1")) == CAPTURE_BLOCK_MS
    latency = float(attr_or_fail(io, "latency_ms", "criterion 1: configured output latency"))
    assert latency == pytest.approx(float(device.latency_ms)), (
        "the module must report the device's latency, not the one it asked for"
    )
    assert latency <= 30.0, (
        f"the negotiated output latency is {latency:.1f} ms; criterion 1 caps it at 30 ms (it is "
        f"also the under-report allowance of criterion 2)"
    )
    await call(io, "flush", "AudioIO protocol")


async def test_event_loop_lag_under_playback() -> None:
    """Criterion 13: 10 s of playback, event-loop lag p99 ≤ 20 ms (.claude/rules/conductor.md)."""
    io, _ = build_io()
    await play(io, chirp_utterance("lag"), source="chirp")
    lags: list[float] = []
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        before = time.monotonic()
        await asyncio.sleep(0.005)
        lags.append(1000.0 * (time.monotonic() - before - 0.005))
    await call(io, "flush", "AudioIO protocol")
    assert len(lags) > 500, "the sampler did not run often enough to say anything"
    assert p(lags, 0.99) <= 20.0, (
        f"event-loop lag p99 = {p(lags, 0.99):.1f} ms during playback; the device writer must "
        f"run off the loop (criterion 13)"
    )
    print(f"\nevent-loop lag (ms) over 10 s: p50={p(lags, 0.5):.2f} p99={p(lags, 0.99):.2f}")


async def test_played_ms_and_flush_agree_with_the_reference_ring(tmp_path: Path) -> None:
    """Criterion 10 on hardware: the reference is the audio the sink actually received."""
    with MonitorRecording(NULL_SINK, tmp_path / "ref.wav") as rec:
        io, _ = build_io()
        await play(io, chirp_utterance("ref"), source="chirp")
        await asyncio.sleep(2.0)
        reference, first_ms = await call(io, "playback_reference", "criterion 10", 500.0)
        reported = await played_ms(io)
        await call(io, "flush", "AudioIO protocol")

    reference = np.asarray(reference, dtype=np.float64)
    assert reference.size == pytest.approx(0.5 * DEVICE_RATE, abs=DEVICE_RATE // 100)
    assert first_ms <= reported, "the reference window cannot start after the played position"
    x, rate = rec.read()
    heard_end = rendered_ms_from_chirp(x, rate)
    ref_end = chirp_ms_of(reference, DEVICE_RATE)
    assert abs(ref_end - min(first_ms + 500.0, heard_end)) <= 60.0, (
        f"the reference ring ends at {ref_end:.0f} ms of the chirp but the sink received up to "
        f"{heard_end:.0f} ms (criterion 10: the reference is the real mixed output)"
    )


def chirp_ms_of(x: np.ndarray, rate: int) -> float:
    return rendered_ms_from_chirp(x, rate)
