"""B13 statistics: constrained-decoding per-step overhead with an A/A' control (T-0070 criterion 11,
ruling E10).

The microbenchmark (T-0071) interleaves per-step times of A (unconstrained), B (constrained) and A'
(unconstrained again) and hands the three paired sequences here. Pure statistics, stdlib only.

- ``delta_pct``: 100 x (median over pairs of B_i / A_i - 1), the overhead of B over A;
- ``ci_low_pct``/``ci_high_pct``: a distribution-free 95 % interval on that median from the sign test
  (order statistics of the paired ratios; exact binomial, no random numbers), in the same units;
- ``control_pct``: 100 x |median over pairs of A'_i / A_i - 1|, the size of a difference that is nothing
  but noise and drift.

Verdict (E10: a confidence statement, not a point estimate):

- ``not_resolvable`` whenever ``control_pct >= |delta_pct|`` -- never ``pass`` then (the B16 pattern);
- ``pass`` only if ``n >= MIN_PAIRS`` and the interval's upper bound is ``<= threshold_pct``;
- ``fail`` if the interval's lower bound is ``> threshold_pct``;
- ``not_resolvable`` otherwise (the interval straddles the threshold, or too few pairs to pass).

Thresholds come from the caller (T-0062 decision 8: 15 % at bucket <= 128, 10 % at <= 64).
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections.abc import Sequence
from typing import Literal

__all__ = ["MIN_PAIRS", "B13Result", "evaluate", "sign_test_interval"]

Verdict = Literal["pass", "fail", "not_resolvable"]
#: E10: fewer pairs than this can never pass.
MIN_PAIRS = 20
ALPHA = 0.05


@dataclasses.dataclass(frozen=True)
class B13Result:
    verdict: Verdict
    delta_pct: float
    ci_low_pct: float
    ci_high_pct: float
    control_pct: float
    threshold_pct: float
    n: int


def _ratios_pct(num: Sequence[float], den: Sequence[float]) -> list[float]:
    return [100.0 * (x / y - 1.0) for x, y in zip(num, den, strict=True)]


def sign_test_interval(values: Sequence[float], alpha: float = ALPHA) -> tuple[float, float]:
    """Distribution-free ``1 - alpha`` interval on the median: ``[x_(k), x_(n-k+1)]`` with the largest
    ``k`` such that ``2 * P(Binom(n, 1/2) <= k - 1) <= alpha``. ``(-inf, inf)`` when ``n`` is too small
    for any such ``k``."""
    xs = sorted(values)
    n = len(xs)
    k = 0
    cdf = 0.0
    for j in range(n + 1):
        cdf += math.comb(n, j) / 2.0**n  # P(Binom <= j)
        if 2.0 * cdf <= alpha:
            k = j + 1
        else:
            break
    if k < 1:
        return (-math.inf, math.inf)
    return (xs[k - 1], xs[n - k])


def evaluate(
    a: Sequence[float],
    b: Sequence[float],
    a_prime: Sequence[float],
    *,
    threshold_pct: float,
) -> B13Result:
    """Paired A/B per-step times with the A/A' control; the sequences are paired by index."""
    n = len(a)
    if n == 0 or len(b) != n or len(a_prime) != n:
        raise ValueError(
            f"A, B and A' must be non-empty and paired: {len(a)}, {len(b)}, {len(a_prime)}"
        )
    if any(x <= 0 for x in a):
        raise ValueError("A's per-step times must be positive")
    overhead = _ratios_pct(b, a)
    delta = statistics.median(overhead)
    control = abs(statistics.median(_ratios_pct(a_prime, a)))
    lo, hi = sign_test_interval(overhead)
    verdict: Verdict
    if control >= abs(delta):
        verdict = "not_resolvable"
    elif n >= MIN_PAIRS and hi <= threshold_pct:
        verdict = "pass"
    elif lo > threshold_pct:
        verdict = "fail"
    else:
        verdict = "not_resolvable"
    return B13Result(verdict, delta, lo, hi, control, float(threshold_pct), n)
