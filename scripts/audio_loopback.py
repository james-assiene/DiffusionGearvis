#!/usr/bin/env python3
"""T-0006 rig helper: PipeWire node lookup, the calibration check, and the pause/flush numbers.

Three jobs, all of them pactl-free (``pactl`` is not installed on this box) and all of them
writing only to ``$TMPDIR``:

``--sink-ids NAME``
    The PipeWire node ids of ``NAME`` from ``pw-dump``, one per line. ``scripts/audio_rig.sh``
    uses it to create and destroy the null sink without a JSON parser in shell.

``--calibrate``
    Play the chirp fixture into the sink and check that the sink's own monitor recorded it above
    an energy floor. **Exits non-zero with a named reason** when it did not (criterion 11): the
    default sink on this box is a Dummy Output while the DP cable is unplugged, and a rig that
    renders nothing must never look like a rig that found no echo.

``--measure pause|flush --n 30 --seed 0``
    Criterion 9's authoritative number: the time from the call to the last audible sample on the
    monitor, p50 and p95 over n points. The measurement drives ``PipeWireAudioIO`` through the
    real backend, so it measures the module, not ``pw-play``.

Regime for anything this prints (CLAUDE.md): box, PipeWire version, sink, requested quantum
480/48000 with the negotiated one reported by the module, 48 kHz mono f32, the chirp fixture,
profile=lean, n and seed as given on the command line, git tree clean.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

REPO = Path(__file__).resolve().parent.parent
CHIRP = REPO / "tests" / "fixtures" / "audio" / "chirp_5s_24k.wav"
DEVICE_RATE = 48000
#: The chirp fixture's frequency law (tests/fixtures/audio/MANIFEST.md).
CHIRP_F0_HZ = 200.0
CHIRP_K_HZ_PER_S = 1160.0
HEARD_PEAK = 0.02


def pw_nodes() -> list[dict]:
    proc = subprocess.run(["pw-dump"], capture_output=True, text=True, check=True)
    return [o for o in json.loads(proc.stdout) if o.get("type") == "PipeWire:Interface:Node"]


def sink_ids(name: str) -> list[int]:
    return [
        int(obj["id"])
        for obj in pw_nodes()
        if obj.get("info", {}).get("props", {}).get("node.name") == name
    ]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    # Imported here, not at the top: `--rig up/down/status` and `--sink-ids` must run under the
    # system python (that is what scripts/audio_rig.sh uses); only the measurements need numpy.
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, rate


def chirp_ms(rec: np.ndarray, rate: int, threshold: float = HEARD_PEAK) -> float:
    """How much of the chirp a recording holds, read off its own frequency ramp."""
    import numpy as np

    loud = np.flatnonzero(np.abs(rec) > threshold)
    if loud.size == 0:
        raise SystemExit("the recording is silent: the rig is not rendering (criterion 11)")
    window = max(32, round(0.02 * rate))
    end = int(loud[-1]) + 1
    seg = rec[max(0, end - window) : end]
    crossings = np.flatnonzero(np.diff(np.signbit(seg)))
    if crossings.size < 3:
        raise SystemExit("too few zero crossings at the end of the recording to read its clock")
    first, last = int(crossings[0]), int(crossings[-1])
    freq = (crossings.size - 1) / (2.0 * (last - first) / rate)
    mid_ms = 1000.0 * (freq - CHIRP_F0_HZ) / CHIRP_K_HZ_PER_S
    return mid_ms + 1000.0 * (seg.size - 1 - 0.5 * (first + last)) / rate


class Monitor:
    """``pw-record`` on a node's monitor for the duration of a with-block."""

    def __init__(self, target: str, path: Path) -> None:
        self.target = target
        self.path = path
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> Monitor:
        # ``stream.capture.sink=true`` is what makes this the *monitor* of a sink. Without it the
        # session manager finds no source called ``astra-test-sink`` and links the stream to the
        # default source instead -- on this box the webcam mic, which records the room at a peak
        # of ~0.12 and looks exactly like a rig that is working.
        self._proc = subprocess.Popen(
            [
                "pw-record",
                "-P",
                "stream.capture.sink=true",
                "--target",
                self.target,
                "--rate",
                str(DEVICE_RATE),
                str(self.path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.3)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._proc is not None:
            time.sleep(0.2)
            self._proc.terminate()
            self._proc.wait(timeout=5)

    def read(self) -> tuple[np.ndarray, int]:
        x, rate = read_wav(self.path)
        if x.size == 0:
            raise SystemExit(f"pw-record captured nothing from {self.target}")
        return x, rate


def utterance(path: Path):
    """A minimal ``TtsUtterance`` over a wav fixture: 24 kHz s16le in 80 ms chunks."""
    from local_astra.contracts.speech import TTS_OUTPUT_FORMAT

    with wave.open(str(path), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
        assert handle.getframerate() == TTS_OUTPUT_FORMAT.sample_rate

    class WavUtterance:
        utterance_id = "chirp"
        text = "chirp"
        first_audio_latency_ms = None

        async def audio(self):
            step = 2 * 80 * TTS_OUTPUT_FORMAT.sample_rate // 1000
            for start in range(0, len(raw), step):
                yield raw[start : start + step]

        def words(self):
            return None

        async def cancel(self) -> None:
            return None

    return WavUtterance()


def build_io(sink: str):
    from local_astra.conductor.audio_device import open_device
    from local_astra.conductor.audio_io import PipeWireAudioIO

    return PipeWireAudioIO(open_device(sink=sink, block_ms=10, latency_ms=30.0))


def percentile(values: list[float], q: float) -> float:
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(q * 100) - 1])


async def calibrate(sink: str, tmp: Path) -> int:
    """Criterion 11: prove the sink renders, or exit non-zero saying exactly what was wrong."""
    import numpy as np

    with Monitor(sink, tmp / "calib.wav") as rec:
        io = build_io(sink)
        await io.play(utterance(CHIRP), source="chirp")
        await asyncio.sleep(2.0)
        reported = await io.flush()
    x, rate = rec.read()
    peak = float(np.max(np.abs(x)))
    if peak <= HEARD_PEAK:
        print(
            f"CALIBRATION FAILED: the monitor of {sink} peaked at {peak:.4f} while the chirp was "
            f"playing. The sink is not rendering (a Dummy Output, a muted node, or the module "
            f"opened a different device). No number measured on this rig can be believed.",
            file=sys.stderr,
        )
        return 1
    print(
        f"calibration ok: sink={sink} peak={peak:.3f} monitor_holds={chirp_ms(x, rate):.0f} ms "
        f"module_reported={reported} ms quantum={io.quantum_ms:.3f} ms latency={io.latency_ms:.1f} ms"
    )
    return 0


async def measure(sink: str, what: str, n: int, seed: int, tmp: Path) -> int:
    """Criterion 9: time from the call to the last audible sample on the monitor, p50 and p95."""
    import numpy as np

    rng = random.Random(seed)
    values: list[float] = []
    for i in range(n):
        at_ms = rng.uniform(500.0, 4000.0)
        with Monitor(sink, tmp / f"{what}{i}.wav") as rec:
            io = build_io(sink)
            await io.play(utterance(CHIRP), source="chirp")
            await asyncio.sleep(at_ms / 1000.0)
            if what == "pause":
                await io.pause()
            else:
                await io.flush()
            await asyncio.sleep(0.4)
            await io.flush()
        x, rate = rec.read()
        loud = np.flatnonzero(np.abs(x) > HEARD_PEAK)
        if loud.size == 0:
            print(f"run {i}: nothing was rendered (criterion 11)", file=sys.stderr)
            return 1
        # Lower bound, exactly as tests/integration/test_audio_null_sink.py argues: the monitor
        # is not aligned to the wall clock, so the startup delay (<= latency + one block) is
        # subtracted out of the answer rather than added to it.
        values.append(chirp_ms(x, rate) - at_ms)
    print(
        f"{what} -> silence (ms), sink={sink} n={n} seed={seed}: "
        f"p50={percentile(values, 0.5):.1f} p95={percentile(values, 0.95):.1f} "
        f"min={min(values):.1f} max={max(values):.1f}"
    )
    return 0


#: ``pw-cli create-node`` properties of a null sink. ``object.linger`` keeps the node alive after
#: pw-cli exits -- without it the sink dies with the process that made it and every rig test
#: skips. There is no ``pactl`` on this box, so this is how the sink is made (criterion 11).
NULL_SINK_PROPS = " ".join(
    [
        "factory.name=support.null-audio-sink",
        "node.name={name}",
        "node.description={name}",
        "media.class=Audio/Sink",
        "object.linger=true",
        "audio.position=[FL FR]",
        "monitor.channel-volumes=true",
    ]
)


def rig(action: str, sink: str) -> int:
    """Create, destroy or report the null sink, verifying the result in ``pw-dump`` every time."""
    ids = sink_ids(sink)
    if action == "status":
        print(f"{sink} is up: {ids}" if ids else f"{sink} is not up")
        return 0
    if action == "up":
        if ids:
            print(f"{sink} is already up: {ids}")
            return 0
        props = "{ " + NULL_SINK_PROPS.format(name=sink) + " }"
        subprocess.run(["pw-cli", "create-node", "adapter", props], check=True, capture_output=True)
        for _ in range(50):
            ids = sink_ids(sink)
            if ids:
                print(f"{sink} is up: {ids}")
                return 0
            time.sleep(0.1)
        print(f"failed to create {sink}: pw-dump does not list it", file=sys.stderr)
        return 1
    if action == "down":
        if not ids:
            print(f"{sink} is not up")
            return 0
        for node_id in ids:
            subprocess.run(["pw-cli", "destroy", str(node_id)], check=False, capture_output=True)
        time.sleep(0.3)
        if sink_ids(sink):
            print(f"failed to destroy {sink}", file=sys.stderr)
            return 1
        print(f"{sink} is down")
        return 0
    print(f"unknown rig action {action}", file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sink", default="astra-test-sink")
    parser.add_argument("--sink-ids", metavar="NAME")
    parser.add_argument("--rig", choices=["up", "down", "status"])
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--measure", choices=["pause", "flush"])
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.rig:
        return rig(args.rig, args.sink)
    if args.sink_ids:
        for node_id in sink_ids(args.sink_ids):
            print(node_id)
        return 0
    with tempfile.TemporaryDirectory(prefix="astra-audio-") as tmpdir:
        tmp = Path(tmpdir)
        if args.calibrate:
            return asyncio.run(calibrate(args.sink, tmp))
        if args.measure:
            code = asyncio.run(calibrate(args.sink, tmp))
            if code != 0:
                return code
            return asyncio.run(measure(args.sink, args.measure, args.n, args.seed, tmp))
    parser.error("nothing to do: pass --sink-ids, --calibrate or --measure")
    return 2


if __name__ == "__main__":
    sys.exit(main())
