"""T-0023 criterion 7: the 10-minute duplex run, judged from the report the script wrote.

Markers: ``integration`` + ``audio`` + ``slow``, **never** ``gpu``. The 600 s run lives in
``scripts/locked/audio_soak.py`` (``/launch``-gated, it spends hardware time); the thresholds live
here. That split is deliberate: the numbers stay checkable — and re-checkable by a reviewer —
without playing ten more minutes of audio, and the run itself cannot happen without a token.

Regime (T-0023 §Regime, null-sink rig): box ``gpu-box``, PipeWire 1.0.5, sink
``astra-test-sink`` via ``scripts/audio_rig.sh``, requested quantum 480/48000 with the negotiated
one and the measured DAC lag in the report, 48 kHz mono f32 capture → 24 kHz f32 STT frames,
``profile=lean``, ``audio_path=loopback``, gate on, duty cycle 4 s speech / 6 s silence,
duration 600 s, n ≥ 60 000 blocks, seed 0, vLLM and ``astra-stt`` co-residency stated by the
report, no other PipeWire clients, git tree clean.

The run command and the pytest invocation that reads its report are in ``docs/tasks/T-0023.md``
§Evidence required (stage 5); both go through ``/launch``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.unit.test_audio_measure_report import MB, REQUIRED_REGIME_FIELDS

pytestmark = [
    pytest.mark.integration,
    pytest.mark.audio,
    pytest.mark.slow,
]

REPORT_ENV = "ASTRA_SOAK_REPORT"
DECLARED_S = 600.0
#: docs/01 §8 "Mic capture + AEC + resample — 20 ms".
HANDOFF_P95_MS = 20.0
#: ``.claude/rules/conductor.md``.
LOOP_LAG_P99_MS = 20.0
#: docs/03 §7 B15's measurable line.
RSS_GROWTH_MB_PER_MIN = 1.0
#: The structural cost of *filling* a 1920-sample frame at 24 kHz. It is the STT server's block
#: size, not this module's latency: criterion 7 wants it stated and excluded, never hidden inside
#: the handoff number.
STRUCTURAL_FRAME_FILL_MS = 80.0
#: Criterion 7's sampling rule: every 10 s over 600 s.
RSS_SAMPLE_PERIOD_S = 10.0
MIN_RSS_SAMPLES = 60


def soak_report() -> dict[str, Any]:
    path = os.environ.get(REPORT_ENV)
    if not path:
        pytest.fail(
            f"{REPORT_ENV} is unset: run the locked soak script for {DECLARED_S:.0f} s at seed 0 "
            f"with --out $TMPDIR/soak.json (docs/tasks/T-0023.md §Evidence, stage 5; it needs a "
            f"/launch token) and point {REPORT_ENV} at the file. This is an error, never a skip: "
            f"a duplex budget nobody measured is not a passing budget (T-0023 criterion 7)"
        )
    file = Path(path)
    if not file.exists():
        pytest.fail(f"{REPORT_ENV}={path!r} does not exist: no soak report was written")
    return dict(json.loads(file.read_text()))


def field(report: dict[str, Any], *path: str) -> Any:
    """A report field, or a failure naming the exact path that is missing (criterion 1)."""
    node: Any = report
    seen: list[str] = []
    for key in path:
        if not isinstance(node, dict) or key not in node:
            pytest.fail(
                f"the soak report has no {'.'.join(path)}: {'.'.join(seen) or '<root>'} holds "
                f"{sorted(node) if isinstance(node, dict) else type(node).__name__}. The report "
                f"shape is pinned in tests/unit/test_audio_measure_report.py's module docstring"
            )
        node = node[key]
        seen.append(key)
    return node


def test_soak_report_meets_the_duplex_budget() -> None:
    """Criterion 7's six numbers plus a complete regime block, from a 600 s run.

    The counters come from the ``CaptureStats`` snapshots (they are not re-derived from audio),
    the handoff p95 excludes the structural 80 ms of filling a frame and says so, and the RSS
    growth is re-fitted here from the report's own samples — a report whose ``growth_mb_per_min``
    does not match a least-squares fit over its samples was computed as ``last − first``, which
    reports a leak as a saving whenever the collector ran just before the final sample.
    """
    report = soak_report()

    regime = field(report, "regime")
    missing = [f for f in REQUIRED_REGIME_FIELDS if f not in regime or regime[f] in (None, "", [])]
    assert not missing, f"the soak report's regime block is missing or blank at {missing}"
    assert str(regime["gate"]) == "on", f"the soak runs with the gate on, not {regime['gate']!r}"
    assert int(regime["seed"]) == 0

    duration = float(field(report, "duration_s"))
    declared = float(field(report, "declared_duration_s"))
    assert declared == DECLARED_S, f"the run declared {declared} s, not the 10 minutes of B15"
    assert duration >= declared - 1.0, (
        f"the run lasted {duration:.1f} s of a declared {declared:.0f} s: a short run is refused, "
        f"not reported as the 10-minute number (criterion 7)"
    )

    counters = field(report, "counters")
    for name in ("xruns", "dropped_blocks", "substituted_ms", "underruns"):
        assert float(field(report, "counters", name)) == 0.0, (
            f"{name} = {counters[name]} over {duration:.0f} s of continuous duplex audio "
            f"(docs/03 §7 B15 wants zero, gate on, duty cycle 4 s / 6 s, seed 0)"
        )
    assert int(counters["blocks"]) >= 60_000, (
        f"{counters['blocks']} capture blocks in {duration:.0f} s: at 10 ms a block, a 600 s run "
        f"is n ≥ 60 000 (§Regime). Fewer means capture stopped somewhere in the run"
    )

    handoff_p95 = float(field(report, "handoff_ms", "p95"))
    handoff_p50 = float(field(report, "handoff_ms", "p50"))
    assert float(field(report, "handoff_ms", "structural_frame_fill_ms")) == pytest.approx(
        STRUCTURAL_FRAME_FILL_MS
    ), (
        "the structural +80 ms of filling a 1920-sample frame must be stated in the report and "
        "excluded from the handoff number, never hidden inside it (criterion 7)"
    )
    assert handoff_p95 <= HANDOFF_P95_MS, (
        f"capture → STT handoff p95 = {handoff_p95:.1f} ms (p50 {handoff_p50:.1f} ms, "
        f"n={field(report, 'handoff_ms', 'n')}), over docs/01 §8's 20 ms capture line. That is a "
        f"spec change raised with the measurement and its regime, never an edited assertion"
    )

    lag_p99 = float(field(report, "loop_lag_ms", "p99"))
    assert lag_p99 <= LOOP_LAG_P99_MS, (
        f"event-loop lag p99 = {lag_p99:.1f} ms over the run (.claude/rules/conductor.md: 20 ms)"
    )

    samples = [(float(t), float(rss)) for t, rss in field(report, "rss", "samples")]
    assert len(samples) >= MIN_RSS_SAMPLES, (
        f"{len(samples)} RSS samples over {duration:.0f} s: criterion 7 wants one every "
        f"{RSS_SAMPLE_PERIOD_S:.0f} s, n ≥ {MIN_RSS_SAMPLES}, and a fit over them"
    )
    t = np.array([s[0] for s in samples], dtype=np.float64)
    y = np.array([s[1] for s in samples], dtype=np.float64)
    fitted = float(np.polyfit(t, y, 1)[0]) * 60.0 / MB
    reported = float(field(report, "rss", "growth_mb_per_min"))
    assert reported == pytest.approx(fitted, rel=1e-3, abs=1e-6), (
        f"the report says RSS grows {reported:.3f} MB/min but a least-squares fit over its own "
        f"{len(samples)} samples gives {fitted:.3f} MB/min: the growth was not computed as a fit "
        f"(last − first over these samples is "
        f"{(y[-1] - y[0]) * 60.0 / (MB * (t[-1] - t[0])):.3f} MB/min)"
    )
    assert fitted < RSS_GROWTH_MB_PER_MIN, (
        f"RSS grows {fitted:.3f} MB/min over {duration:.0f} s (least-squares fit, n={len(samples)}"
        f", every {RSS_SAMPLE_PERIOD_S:.0f} s): docs/03 §7 B15 allows < {RSS_GROWTH_MB_PER_MIN}"
    )

    print(
        f"\nsoak: duration={duration:.0f} s gate={regime['gate']} duty=4s/6s seed={regime['seed']} "
        f"sink={regime['sink']!r} quantum={regime['negotiated_quantum_ms']} ms "
        f"dac_lag={regime['measured_dac_lag_ms']} ms co_resident={regime['co_resident']!r} | "
        f"xruns={counters['xruns']} dropped={counters['dropped_blocks']} "
        f"substituted={counters['substituted_ms']} underruns={counters['underruns']} | "
        f"handoff p50={handoff_p50:.1f} p95={handoff_p95:.1f} ms (excl. the structural "
        f"{STRUCTURAL_FRAME_FILL_MS:.0f} ms frame fill) | loop lag p99={lag_p99:.1f} ms | "
        f"RSS {fitted:.3f} MB/min over n={len(samples)}"
    )
