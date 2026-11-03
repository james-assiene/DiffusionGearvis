"""T-0047 criterion 6 — the level tap costs nothing the capture pump can feel.

**This is a guard, not a benchmark.** The pump has ~80 ms per frame and a ~20 ms budget before
anything it does is felt (`.claude/rules/conductor.md`), so the ceiling is deliberately generous -
roughly three orders of magnitude under the frame - and a loaded box does not make it flaky. No
number from this module may be quoted as a latency measurement.

Two things are guarded here, and ruling A9 is why they are named separately:

* **the ceiling** (criterion 6) catches a tap that is gross per frame - ``np.quantile`` on every
  frame (~26 µs, ruling A6), a second conversion of the 1920 samples, a Python loop over them;
* **the growth** (criterion 6's negative, "no allocation proportional to the number of frames
  seen") catches an accidentally quadratic window by comparing the first 100 frames' median with
  the last 100's *in the same run*. The ceiling alone does **not** catch it: the reviewer measured
  an unbounded-window mutant at p95 28 µs, comfortably inside the 50 µs ceiling, while its last-100
  median is 4.5-5.7x its first-100 median (measured on this box, five repeats, load 5.9, numpy
  2.4.6: 4.3 -> 21-26 µs mutant, 4.3 -> 4.3 µs unmutated). The window's *size* is pinned
  independently and without timing by
  ``test_listen_level.py::test_the_meter_is_bounded_after_100k_observations``, which is the
  stronger of the two; this row catches the cost of a window that grows in any other way.

Regime (docs/tasks/T-0047.md *Regime*, criterion 6): this box, CPU only, no GPU service touched;
n = 1000 frames of 1920 float32 at 24 kHz (80 ms), synthetic signal from
``np.random.default_rng(0)``; one warm-up frame outside the measurement; p95 of
``time.perf_counter_ns()`` deltas around ``rms_dbfs`` + ``LevelMeter.observe`` together; the numpy
version and the measured p50/p95 are printed in the failure message.
"""

from __future__ import annotations

import time

import numpy as np

from local_astra.conductor.audio_capture import STT_BLOCK_SAMPLES
from tests.unit import _level_rig as rig

#: Criterion 6. 50 us is ~0.06 % of the 80 ms frame: a real regression is orders of magnitude,
#: not percent, so the headroom costs nothing in sensitivity.
CEILING_US = 50.0
FRAMES = 1000
FRAME_MS = 80.0

#: How much slower the last 100 frames may be than the first 100 before the per-frame cost counts
#: as *growing*. Unmutated the ratio is 0.95-1.01 over five repeats on a box at load 5.9; the
#: unbounded-window mutant is 4.5-5.7. 2x plus 3 us of absolute slack sits between the two with
#: room for a box whose clock changes mid-run, which is the only way this row could be flaky.
GROWTH_RATIO = 2.0
GROWTH_SLACK_US = 3.0


def test_observe_and_rms_dbfs_cost_under_50us_p95() -> None:
    rms_dbfs = rig.attr("rms_dbfs", "criterion 6")
    meter = rig.meter()

    rng = np.random.default_rng(0)
    signal = (0.1 * rng.standard_normal((FRAMES, STT_BLOCK_SAMPLES))).astype(np.float32)
    assert signal.shape[1] == 1920

    # One warm-up frame: the first call pays for numpy's lazy imports and the window's allocation.
    meter.observe(rms_dbfs(signal[0]), -FRAME_MS)

    deltas_ns: list[int] = []
    for i in range(FRAMES):
        frame = signal[i]
        t0 = time.perf_counter_ns()
        meter.observe(rms_dbfs(frame), i * FRAME_MS)
        deltas_ns.append(time.perf_counter_ns() - t0)

    ordered = sorted(deltas_ns)
    p50_us = ordered[len(ordered) // 2] / 1000.0
    p95_us = ordered[int(0.95 * len(ordered))] / 1000.0
    first_100 = sorted(deltas_ns[:100])[50] / 1000.0
    last_100 = sorted(deltas_ns[-100:])[50] / 1000.0
    regime = f"n={FRAMES}, seed 0, numpy {np.__version__}"
    assert p95_us <= CEILING_US, (
        f"criterion 6: p95 {p95_us:.1f} us > {CEILING_US} us per 1920-sample frame "
        f"(p50 {p50_us:.1f} us; {regime})"
    )
    assert last_100 <= GROWTH_RATIO * first_100 + GROWTH_SLACK_US, (
        f"criterion 6's negative: the per-frame cost grows with the number of frames seen. "
        f"Median of the first 100 frames {first_100:.1f} us, of the last 100 {last_100:.1f} us "
        f"({last_100 / first_100:.1f}x, ceiling {GROWTH_RATIO}x + {GROWTH_SLACK_US} us) - that is "
        f"a window that is re-sorted, re-allocated or re-scanned from the start of the session. "
        f"p95 {p95_us:.1f} us, which can stay inside the {CEILING_US} us ceiling while this fails "
        f"(ruling A9); {regime}"
    )
