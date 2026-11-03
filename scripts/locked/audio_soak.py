#!/usr/bin/env python3
"""T-0023 criterion 7: the 10-minute duplex run on the null-sink rig (docs/03 §7 B15).

``/launch``-gated (``scripts/locked/``): the run spends ten minutes of hardware time. The
thresholds are **not** here -- they live in ``tests/integration/test_audio_soak.py``, which reads
the JSON report this script writes. That split keeps the numbers re-checkable by a reviewer
without playing ten more minutes of audio.

What is measured, and from where:

* the counters come from ``CaptureStats`` snapshots taken every ``SAMPLE_PERIOD_S``, differenced
  end-to-start (they are cumulative) and never re-derived from the audio -- an xrun is what the
  device said it was. ``underruns`` is the playback side's own counter (ruling A1);
* the capture -> STT handoff is ``SttFrame.yielded_ms - SttFrame.end_ms``: the time between the
  ADC taking the frame's **last** sample and the consumer receiving the frame. The structural
  80 ms of *filling* a 1920-sample frame is the STT server's block size, not this module's
  latency, so it is reported as ``structural_frame_fill_ms`` and excluded, never hidden inside
  the number (criterion 7);
* the event-loop lag is the overshoot of a 10 ms sleep on the same loop that drives capture
  (``.claude/rules/conductor.md``'s p99 <= 20 ms);
* RSS is sampled from ``/proc/self/status`` every ``SAMPLE_PERIOD_S`` and its growth is the
  **least-squares slope** over those samples -- never ``last - first``, which reports a leak as a
  saving whenever the collector happens to run just before the final sample. The measurement's
  own accumulators live inside that RSS, so they are packed into ``array('d')`` and their share
  is reported as ``rss.harness_share_mb_per_min`` (ruling A13: in Python lists they were ~0.27
  MB/min of the < 1 MB/min bound, a quarter of the budget spent by the measurement on itself);
* a run shorter than the declared duration is refused, not reported (``check_duration``).

Regime (docs/tasks/T-0023.md §Regime, null-sink rig): box, PipeWire version, sink
``astra-test-sink`` via ``scripts/audio_rig.sh``, requested quantum 480/48000 with the negotiated
one and the measured DAC lag in the report, 48 kHz mono f32 capture -> 24 kHz f32 STT frames,
profile lean, gate on, duty cycle 4 s speech / 6 s silence, duration 600 s, seed 0, co-residency
stated, git tree clean (this script refuses to measure on a dirty tree).

The pure helpers below are unit-tested without hardware in
``tests/unit/test_audio_measure_report.py``; the regime block, ``emit_report`` and
``percentiles`` are the acoustic script's, loaded by path because ``scripts/locked/`` is a
``/launch``-gated directory and not a package.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import random
import statistics
import sys
import time
import wave
from array import array
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "audio"
SENTENCE = FIXTURES / "pocket_tts_sentence.wav"

#: docs/03 §7 B15's line, and §Regime's duty cycle.
DECLARED_DURATION_S = 600.0
SPEECH_S = 4.0
SILENCE_S = 6.0
#: Criterion 7: one snapshot every 10 s, n >= 60 over the declared run.
SAMPLE_PERIOD_S = 10.0
#: The structural cost of *filling* a 1920-sample frame at 24 kHz: stated and excluded.
STRUCTURAL_FRAME_FILL_MS = 80.0
#: 1 MB = 1024 * 1024 bytes (ruling A1), per minute.
MB = 1024 * 1024
#: The sleep whose overshoot is the event-loop lag.
LAG_TICK_S = 0.01
TTS_RATE = 24000


def _sibling(name: str) -> ModuleType:
    """Load another ``scripts/locked`` script by path: this directory is not a package."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"astra_locked_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} is not importable by path: no loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_acoustic = _sibling("audio_acoustic")

#: Re-exported, same semantics: one regime contract for both reports (criterion 1).
REGIME_FIELDS: tuple[str, ...] = _acoustic.REGIME_FIELDS
RegimeError = _acoustic.RegimeError
ReportError = _acoustic.ReportError
emit_report = _acoustic.emit_report
percentiles = _acoustic.percentiles


class ShortRunError(ValueError):
    """The run was shorter than it declared: a 20 s run is not the 10-minute number."""


# --- criterion 7's arithmetic (pure; unit-tested without hardware) ------------------------------

#: The counters the soak reads off ``CaptureStats``. It re-derives none of them from the audio.
COUNTER_FIELDS = ("blocks", "xruns", "substituted_ms", "dropped_blocks", "gated_blocks")


def counters_from_snapshots(snapshots: Sequence[Any]) -> dict[str, float]:
    """Cumulative ``CaptureStats`` snapshots in, the run's counters out: ``last - first``.

    The snapshots are cumulative and the first one marks the start of the run, so ``last``
    alone would attribute anything before it to this run. A counter that *decreases* between two
    consecutive snapshots is a reset, which makes
    ``last - first`` an undercount, so it is reported as an error naming the counter rather than
    quietly folded into a smaller number.

    One argument, ever: criterion 7 says the soak **reads** the counters. Handing this function
    audio, or a second opinion about what an xrun is, is how a soak starts disagreeing with the
    device about what happened.
    """
    rows = list(snapshots)
    if not rows:
        raise ReportError(
            "no CaptureStats snapshots: there is no run to report (T-0023 criterion 7 samples "
            f"one every {SAMPLE_PERIOD_S:.0f} s)"
        )
    for name in COUNTER_FIELDS:
        series = [float(getattr(row, name)) for row in rows]
        for index in range(1, len(series)):
            if series[index] < series[index - 1]:
                raise ReportError(
                    f"the cumulative counter {name!r} fell from {series[index - 1]} to "
                    f"{series[index]} between snapshots {index - 1} and {index}: a counter that "
                    f"resets makes last - first an undercount, so this run is not reportable"
                )
    return {
        name: float(getattr(rows[-1], name)) - float(getattr(rows[0], name))
        for name in COUNTER_FIELDS
    }


def rss_growth_mb_per_min(samples: Sequence[tuple[float, float]]) -> float:
    """Least-squares slope of ``(t_s, rss_bytes)``, in MB per minute. **Never last - first.**

    ``last - first`` on a rising series whose final sample lands just after a collection is
    negative: a leak reported as a saving. The slope over all the samples is not fooled by one
    dip, which is why criterion 7 asks for a fit.
    """
    points = [(float(t), float(y)) for t, y in samples]
    if len(points) < 2:
        raise ValueError(
            f"an RSS growth needs at least two samples, got {len(points)}: criterion 7 wants one "
            f"every {SAMPLE_PERIOD_S:.0f} s over the run (n >= 60)"
        )
    n = len(points)
    mean_t = sum(t for t, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    covariance = sum((t - mean_t) * (y - mean_y) for t, y in points)
    variance = sum((t - mean_t) ** 2 for t, _ in points)
    if variance == 0.0:
        raise ValueError("every RSS sample carries the same timestamp: there is no slope to fit")
    return (covariance / variance) * 60.0 / MB


def check_duration(measured_s: float, declared_s: float, *, tolerance_s: float = 1.0) -> None:
    """Refuse a run that did not last what it says it did (criterion 7). Longer is fine."""
    if float(measured_s) < float(declared_s) - float(tolerance_s):
        raise ShortRunError(
            f"the run lasted {float(measured_s):.1f} s of a declared {float(declared_s):.1f} s "
            f"(tolerance {tolerance_s:.1f} s): a short run is refused, never reported as the "
            f"10-minute number (T-0023 criterion 7)"
        )


# --- the run (hardware; /launch-gated) ----------------------------------------------------------


def rss_bytes() -> float:
    """This process's resident set, from ``/proc/self/status`` (kB there, bytes here)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) * 1024.0
    raise ReportError("/proc/self/status has no VmRSS line: RSS growth cannot be measured here")


def speech_utterance(seconds: float, rng: random.Random) -> Any:
    """``seconds`` of the sentence fixture from a seeded offset -- the duty cycle's speech half."""
    with wave.open(str(SENTENCE), "rb") as handle:
        if handle.getframerate() != TTS_RATE:
            raise ReportError(f"{SENTENCE.name} is {handle.getframerate()} Hz, expected {TTS_RATE}")
        raw = handle.readframes(handle.getnframes())
    want = int(seconds * TTS_RATE) * 2
    start = 2 * rng.randrange(max(1, len(raw) // 2))
    body = (raw[start:] + raw) * (1 + want // max(1, len(raw)))
    pcm = body[:want]

    class WavUtterance:
        utterance_id = "soak"
        text = ""
        first_audio_latency_ms = None

        async def audio(self) -> Any:
            step = 2 * 80 * TTS_RATE // 1000
            for at in range(0, len(pcm), step):
                yield pcm[at : at + step]

        def words(self) -> None:
            return None

        async def cancel(self) -> None:
            return None

    return WavUtterance()


async def soak(args: argparse.Namespace) -> dict[str, Any]:
    """600 s of continuous capture with playback at 4 s / 6 s, gate on, on the null-sink rig."""
    io, device = _acoustic.build_io(args.sink, args.source)
    rng = random.Random(args.seed)
    # ``array('d')``, not a list (ruling A13): 60 000 loop-lag samples plus ~7 500 handoff
    # samples in a Python list cost ~0.27 MB/min of the < 1 MB/min RSS bound this run is
    # measuring -- a quarter of the budget spent by the measurement on itself. Packed doubles
    # are 8 bytes each with no object header, and what the harness still costs is reported
    # beside the growth as ``harness_share_mb_per_min`` rather than left in the number.
    handoffs = array("d")
    lags = array("d")
    snapshots: list[Any] = []
    rss: list[tuple[float, float]] = []
    stop = asyncio.Event()
    started = time.monotonic()

    async def pump() -> None:
        """The STT tap, consumed at cadence: the handoff is the frame's own pair of times."""
        async for frame in io.stt_blocks():
            handoffs.append(float(frame.yielded_ms) - float(frame.end_ms))

    async def lag_meter() -> None:
        while not stop.is_set():
            before = time.perf_counter()
            await asyncio.sleep(LAG_TICK_S)
            lags.append(1000.0 * (time.perf_counter() - before - LAG_TICK_S))

    async def sampler() -> None:
        while not stop.is_set():
            snapshots.append(io.capture_stats())
            rss.append((time.monotonic() - started, rss_bytes()))
            await asyncio.sleep(SAMPLE_PERIOD_S)

    async def duty_cycle() -> None:
        while not stop.is_set():
            await io.play(speech_utterance(SPEECH_S, rng), source="soak")
            # Ruling A5: declare end of stream for **every** utterance. Without it the run never
            # drains, and each of the ~60 silences is counted as an underrun (playback with an
            # empty queue and no drain mark is exactly T-0006 criterion 7's underrun), which
            # makes criterion 7's `underruns == 0` unreachable by construction.
            io.mark_drain()
            await asyncio.sleep(SPEECH_S + SILENCE_S)

    tasks = [asyncio.create_task(coro()) for coro in (pump, lag_meter, sampler, duty_cycle)]
    try:
        await asyncio.sleep(args.duration)
        stop.set()
        snapshots.append(io.capture_stats())
        rss.append((time.monotonic() - started, rss_bytes()))
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        facts = _acoustic.device_lag_facts(device)  # read while the stream is still up
        _acoustic.close_io(io, device)  # ruling A9: the device stops, not only the capture
    measured_s = time.monotonic() - started
    check_duration(measured_s, args.duration)

    counters = counters_from_snapshots(snapshots)
    handoff = percentiles(handoffs)
    lag = percentiles(lags)
    # Inclusive quantiles, the same family as ``percentiles`` (which reports p50/p95); the p99 is
    # the one ``.claude/rules/conductor.md`` bounds and nothing else here needs it.
    p99 = statistics.quantiles(sorted(lags), n=100, method="inclusive")[98]
    # What the harness itself accumulated, as a rate over the run: the samples it holds are
    # inside the RSS it is measuring, so the share is stated rather than subtracted (ruling A13).
    harness_bytes = float(
        lags.itemsize * len(lags) + handoffs.itemsize * len(handoffs) + 2 * 8 * len(rss)
    )
    harness_share = harness_bytes / MB / (measured_s / 60.0)
    return {
        "counters": {
            **{name: counters[name] for name in COUNTER_FIELDS},
            "underruns": float(io.underruns),
        },
        "handoff_ms": {
            "p50": handoff["p50"],
            "p95": handoff["p95"],
            "n": handoff["n"],
            "structural_frame_fill_ms": STRUCTURAL_FRAME_FILL_MS,
        },
        "loop_lag_ms": {"p50": lag["p50"], "p99": float(p99), "n": lag["n"]},
        "rss": {
            "samples": [[t, y] for t, y in rss],
            "growth_mb_per_min": rss_growth_mb_per_min(rss),
            "n": len(rss),
            "harness_bytes": harness_bytes,
            "harness_share_mb_per_min": harness_share,
            "harness_note": (
                "the measurement's own accumulators (array('d') loop-lag and handoff samples, "
                "plus the RSS series) are inside this RSS: stated, not subtracted"
            ),
        },
        "duration_s": measured_s,
        "declared_duration_s": float(args.duration),
        "_facts": facts,
    }


#: The STT is not in this loop at all: the soak measures capture -> tap, not capture -> server.
#: A complete sub-block that says so beats four fields copied from a server nobody started
#: (ruling A7).
STT_NOT_USED: dict[str, Any] = {"used": False}


def soak_regime(args: argparse.Namespace, source: str, **measured: Any) -> dict[str, Any]:
    """The §Regime block for the run: validated before the 600 s, completed after (ruling A7)."""
    return _acoustic.build_regime(
        argparse.Namespace(
            transducer=args.transducer,
            distance_cm=args.distance_cm,
            n=measured.pop("n", ""),
            seed=args.seed,
        ),
        args.sink,
        source,
        gate="on",
        stt=STT_NOT_USED,
        **measured,
    )


async def run(args: argparse.Namespace) -> int:
    _acoustic.refuse_dirty_tree()
    if not _acoustic.node_ids(args.sink):
        raise ReportError(
            f"the sink {args.sink!r} is not in pw-dump: bring the rig up with "
            f"`bash scripts/audio_rig.sh up` before the soak (T-0021)"
        )
    source = args.source or _acoustic.default_source()
    if not source:
        raise RegimeError("no capture node in pw-dump (P3: the Logitech C925e is the only mic)")
    args.source = source

    # Ruling A7: everything that can be missing is checked here, in a second, and not after ten
    # minutes of audio. ``n`` is the block count, which only the run can know, so it is checked
    # with the rest of the measured fields at the end.
    _acoustic.require_regime(
        soak_regime(args, source),
        [name for name in REGIME_FIELDS if name not in (*_acoustic.MEASURED_REGIME_FIELDS, "n")],
    )

    measured = await soak(args)
    facts = measured.pop("_facts")
    regime = soak_regime(
        args,
        source,
        n=int(measured["counters"]["blocks"]),
        floor=args.noise_floor_peak,
        duration_s=measured["duration_s"],
        facts=facts,
    )
    _acoustic.require_regime(regime)
    # Ruling A8: criterion 1 is per number. This report is one run, so every section's regime is
    # this same block -- stated, not implied, because the sections are read separately.
    for name in ("counters", "handoff_ms", "loop_lag_ms", "rss"):
        measured[name]["regime"] = regime
    report = {"regime": regime, **measured}

    buffer = StringIO()
    code = emit_report(report, buffer, sys.stderr)
    if code != 0:
        return code
    out_path = Path(args.out) if args.out else None
    if out_path is None:
        sys.stdout.write(buffer.getvalue())
    else:
        out_path.write_text(buffer.getvalue())
    counters = report["counters"]
    print(
        f"soak: sink={args.sink!r} source={source!r} duration={measured['duration_s']:.0f} s "
        f"declared={measured['declared_duration_s']:.0f} s duty={SPEECH_S:g}s/{SILENCE_S:g}s "
        f"gate=on seed={args.seed} blocks={counters['blocks']:.0f} xruns={counters['xruns']:.0f} "
        f"dropped={counters['dropped_blocks']:.0f} substituted={counters['substituted_ms']:.0f} ms "
        f"underruns={counters['underruns']:.0f} handoff p50={report['handoff_ms']['p50']:.1f} "
        f"p95={report['handoff_ms']['p95']:.1f} ms (excl. the structural "
        f"{STRUCTURAL_FRAME_FILL_MS:.0f} ms frame fill) loop lag p99="
        f"{report['loop_lag_ms']['p99']:.1f} ms RSS {report['rss']['growth_mb_per_min']:.3f} "
        f"MB/min over n={report['rss']['n']} (the harness's own samples are "
        f"{report['rss']['harness_share_mb_per_min']:.3f} MB/min of that) "
        f"co_resident={regime['co_resident']!r} "
        f"git={regime['git_sha']} -> {out_path if out_path is not None else 'stdout'}",
        file=sys.stderr,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T-0023 criterion 7: the 10-minute duplex run.")
    parser.add_argument("--duration", type=float, default=DECLARED_DURATION_S)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sink", default=os.environ.get("ASTRA_SOAK_SINK", "astra-test-sink"))
    parser.add_argument("--source", default=None)
    parser.add_argument("--transducer", choices=["speakers", "headset"], default="speakers")
    parser.add_argument(
        "--distance-cm",
        type=float,
        default=0.0,
        help="the null-sink rig never leaves the box; 0 is a value, and it is stated",
    )
    parser.add_argument(
        "--noise-floor-peak",
        type=float,
        default=0.0,
        help="the null-sink rig is digital: its floor is 0.0, stated rather than omitted",
    )
    parser.add_argument("--out", default=os.environ.get("ASTRA_SOAK_REPORT") or None)
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run(args))
    except (RegimeError, ReportError, ShortRunError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
