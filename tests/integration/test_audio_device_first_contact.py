"""First contact with PortAudio, on the null-sink rig: T-0035 group B, criteria 11-14.

Markers: ``integration`` + ``audio`` — this rig is a null sink and its monitor, **no GPU** and
nothing acoustic (the speaker/headset path is T-0023's and needs ``/launch``). Group B is
otherwise ungated, like the rest of the null-sink rig.

Every row here **fails, never skips**, when the rig is up and PortAudio is present: a skip on
"the device produced nothing" is how a dead path reports success (T-0006 criterion 11). It skips
only when the rig is genuinely absent — ``libportaudio2`` not installed (prerequisite P1, a
sysadmin step) or the node ``astra-test-sink`` not in ``pw-dump`` — and the skip reason names
which.

Regime for every number recorded here (CLAUDE.md: state the whole regime): box
``gpu-box``, 32 cores, Linux 7.0.0-31-generic, libpipewire **1.0.5**, sink
``astra-test-sink`` (null sink from ``scripts/audio_rig.sh up``), source
``astra-test-sink.monitor`` with ``stream.capture.sink=true``; transducer **none** (monitor
loopback); 48 kHz mono f32; requested output/input latency 30 ms, ``block_ms`` 10; chirp 2 s;
n = 3 per criterion, seed 0; co-residents (vLLM, ``astra-stt``) recorded in the report because
another client can force the graph quantum; ``git status`` clean and the sha recorded before any
number is quoted.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

import numpy as np
import pytest

from tests.fakes.audio_device import attr_or_fail, import_or_fail, missing_node_reason

pytestmark = [pytest.mark.integration, pytest.mark.audio]

HOW = "run `bash scripts/audio_rig.sh up` (T-0006 criterion 11)"
SINK = "astra-test-sink"
MONITOR = f"{SINK}.monitor"
ABSENT = "astra-no-such-sink"
#: the C925e's room noise peaks around 0.12; a monitor of the chirp is far above it
WEBCAM_FLOOR = 0.2
SETTLE_CEILING_S = 2.0
N = 3


@pytest.fixture(autouse=True)
def _rig() -> None:
    """Skip only on a genuinely absent rig, and name what is missing."""
    try:
        import sounddevice  # noqa: F401
    except Exception as exc:  # OSError without libportaudio2
        pytest.skip(f"PortAudio is not installed (T-0035 prerequisite P1: libportaudio2): {exc}")
    reason = missing_node_reason(SINK, HOW)
    if reason is not None:
        pytest.skip(reason)


def device_module() -> Any:
    return import_or_fail("local_astra.conductor.audio_device", "T-0035 criteria 1-7")


def open_device(**kwargs: Any) -> Any:
    return attr_or_fail(device_module(), "open_device", "T-0006 criterion 1")(**kwargs)


def chirp(seconds: float = 2.0, rate: int = 48000) -> np.ndarray:
    """A 200 → 6000 Hz linear chirp, the same shape as tests/fixtures/audio/chirp_5s_24k.wav."""
    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    k = (6000.0 - 200.0) / seconds
    return (0.5 * np.sin(2 * np.pi * (200.0 * t + 0.5 * k * t * t))).astype(np.float32)


def play_and_capture(device: Any, seconds: float = 2.0) -> float:
    """Play the chirp through the device while capturing; return the captured peak."""
    captured: list[np.ndarray] = []
    device.set_capture_callback(lambda samples, adc_ms, xrun: captured.append(np.array(samples)))
    device.start_capture()
    block = round(device.block_ms * 48000 / 1000)
    samples = chirp(seconds)
    for start in range(0, samples.size - block, block):
        device.write(samples[start : start + block])
    deadline = time.monotonic() + seconds + 2.0
    while device.queued_ms() > 0.0 and time.monotonic() < deadline:
        time.sleep(0.01)
    device.stop_capture()
    return float(np.max(np.abs(np.concatenate(captured)))) if captured else 0.0


@pytest.mark.pending(task="T-0035")
def test_the_rig_stream_links_to_the_null_sink_and_its_monitor() -> None:
    """Criterion 11: the whole path, once, on the rig — and it says what it did.

    n = 3 runs. Each opens the duplex device on ``astra-test-sink`` and its monitor, plays the
    2 s chirp and reads the peak back off the monitor. A peak at or below the webcam floor means
    the capture half is not on the monitor at all (criterion 8's discriminator), which is a
    failure, not a skip.
    """
    peaks: list[float] = []
    for _ in range(N):
        device = open_device(sink=SINK, source=MONITOR, block_ms=10, latency_ms=30.0)
        try:
            peaks.append(play_and_capture(device))
            assert device.underflow_frames == 0, (
                f"underflow_frames={device.underflow_frames} over a 2 s chirp on an idle rig "
                f"(criterion 11); device_xruns={attr_or_fail(device, 'device_xruns', 'crit. 1')}"
            )
        finally:
            device.stop()
    assert min(peaks) > WEBCAM_FLOOR, (
        f"captured peaks {peaks} over {N} runs; a monitor capture of the chirp is ≈ 0.5 and the "
        f"webcam mic's room noise peaks ≈ 0.12 — this is not the monitor (criteria 11, 13)"
    )


@pytest.mark.pending(task="T-0035")
def test_an_absent_sink_raises_instead_of_playing_into_auto_null() -> None:
    """Adversarial, criterion 12 — the row that proves the whole task on hardware.

    Today this plays happily into ``auto_null`` for as long as you let it. It must refuse inside
    the settle window, naming what was requested and what the stream actually landed on.
    """
    module = device_module()
    start = time.monotonic()
    with pytest.raises(Exception) as caught:
        device = open_device(sink=ABSENT, block_ms=10, latency_ms=30.0)
        try:
            device.start()
        finally:
            device.stop()
    elapsed = time.monotonic() - start
    assert type(caught.value).__module__ == module.__name__, (
        f"{type(caught.value).__name__} is not a named error of audio_device (criteria 4, 5)"
    )
    assert ABSENT in str(caught.value), f"the error does not name {ABSENT!r}: {caught.value}"
    assert elapsed < SETTLE_CEILING_S, f"the refusal took {elapsed:.2f} s (settle window 500 ms)"


@pytest.mark.pending(task="T-0035")
def test_a_monitor_capture_of_an_absent_node_raises_before_anyone_reads_a_zero_peak() -> None:
    """Criterion 13: the trap that cost the audit an afternoon.

    A monitor capture of a node that is not there yields seconds of exactly 0.0 with no error —
    indistinguishable from a working rig in the dark. Criterion 4/5 must reject it before any
    peak is read.
    """
    with pytest.raises(Exception) as caught:
        device = open_device(sink=SINK, source=f"{ABSENT}.monitor", block_ms=10)
        try:
            device.start()
        finally:
            device.stop()
    assert ABSENT in str(caught.value), f"the error does not name the absent source: {caught.value}"


@pytest.mark.pending(task="T-0035")
def test_the_negotiated_quantum_is_read_from_the_graph() -> None:
    """Criterion 11 / criterion 3 on the real graph: a quantum with a provenance.

    The value itself is recorded with its regime (another client — vLLM, ``astra-stt``, a browser
    — can force the graph quantum at any moment), so the assertion is only that the device read
    *something* from the graph and that it is a plausible quantum.
    """
    device = open_device(sink=SINK, source=MONITOR, block_ms=10, latency_ms=30.0)
    try:
        device.start()
        source = attr_or_fail(device, "quantum_source", "T-0035 criterion 3")
        print(
            f"negotiated quantum: {device.quantum_ms:.3f} ms from {source!r}; "
            f"latency out {device.latency_ms:.1f} ms, in {device.input_latency_ms:.1f} ms; "
            f"regime: null-sink rig, 48 kHz mono f32, block 10 ms, libpipewire 1.0.5"
        )
        assert source != "unknown", (
            "the graph would not say what quantum it negotiated while the rig is up: "
            "criterion 11 expects a source of 'pw-dump' or 'callback'"
        )
        assert 0.0 < device.quantum_ms <= 100.0, (
            f"quantum_ms={device.quantum_ms} is not a plausible graph quantum"
        )
    finally:
        device.stop()


@pytest.mark.pending(task="T-0035")
def test_a_starved_graph_shows_up_in_device_xruns() -> None:
    """Criterion 14: the counter T-0006 ruling A22 asked for, measured once.

    The stream is started and then deliberately not fed for 500 ms — the graph asks for audio
    every quantum and we hand it nothing. ``device_xruns`` must count the flagged callbacks and
    ``underflow_frames`` must grow by roughly the starved frames (24 000 at 48 kHz, ± 30 %: the
    device's first callbacks land before the starvation window opens).
    """
    device = open_device(sink=SINK, block_ms=10, latency_ms=30.0)
    try:
        device.start()
        block = round(device.block_ms * 48000 / 1000)
        for _ in range(5):
            device.write(np.zeros(block, dtype=np.float32))
        while device.queued_ms() > 0.0:
            time.sleep(0.005)
        before, before_x = device.underflow_frames, device.device_xruns
        time.sleep(0.5)
        starved = device.underflow_frames - before
        xruns = device.device_xruns - before_x
        print(f"starved {starved} frames in {xruns} flagged callbacks over 500 ms")
        assert xruns > 0, (
            "500 ms of a completely unfed stream produced no flagged callback: the backend's "
            "xrun flag is still not being read (criterion 14, finding F2)"
        )
        assert 0.7 * 24_000 <= starved <= 1.3 * 24_000, (
            f"underflow_frames grew by {starved} over 500 ms of starvation, expected ≈ 24000 "
            f"frames at 48 kHz"
        )
    finally:
        device.stop()


@pytest.mark.pending(task="T-0035")
def test_the_smoke_script_runs_on_the_rig() -> None:
    """Criterion 11: ``scripts/demo/audio_smoke.py`` is the two-minute check, so it is run.

    Exit 0 on the rig, and non-zero for an absent sink (criterion 12) — the same discriminator
    the unit tests exercise in dry-run, here against the real device.
    """
    ok = subprocess.run(
        ["uv", "run", "python", "scripts/demo/audio_smoke.py", "--sink", SINK, "--source", MONITOR],
        capture_output=True,
        text=True,
        timeout=120,
    )
    print(ok.stdout)
    assert ok.returncode == 0, f"audio_smoke.py failed on the rig:\n{ok.stdout}\n{ok.stderr}"
    bad = subprocess.run(
        ["uv", "run", "python", "scripts/demo/audio_smoke.py", "--sink", ABSENT],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert bad.returncode != 0, (
        f"audio_smoke.py exited 0 for an absent sink — the auto_null trap is still silent "
        f"(criterion 12):\n{bad.stdout}"
    )
