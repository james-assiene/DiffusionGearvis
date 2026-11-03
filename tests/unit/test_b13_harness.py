"""T-0070 criterion 11: B13 statistics (``local_astra/bench/b13_cd.py``), fast suite, pure statistics.

Paired interleaved A/B per-step times with an A/A' control. The overhead delta is B against A in per
cent; the control is A' against A. ``not_resolvable`` whenever the control's magnitude is >= |delta|, and
never ``pass`` then (the B16 pattern); otherwise ``pass`` iff delta <= the threshold.

Assumed API (ruling): ``b13_cd.evaluate(a, b, a_prime, *, threshold_pct) -> result`` with ``verdict`` in
{"pass", "fail", "not_resolvable"}, ``delta_pct`` and ``control_pct``; the three sequences are paired by
index and must have equal length. Expected values are hand-computed on constant series (every location
estimator agrees there) and on one seeded noisy series. n = 30 pairs (T-0071's regime).
"""

from __future__ import annotations

import importlib
import random
from types import ModuleType

import pytest

N = 30


def b13() -> ModuleType:
    try:
        return importlib.import_module("local_astra.bench.b13_cd")
    except ModuleNotFoundError as exc:
        if (exc.name or "").startswith("local_astra.bench"):
            pytest.fail(f"T-0070 not implemented: local_astra.bench.b13_cd does not exist ({exc})")
        raise


def ev(a: float, b: float, a_prime: float, threshold: float = 15.0):  # type: ignore[no-untyped-def]
    return b13().evaluate([a] * N, [b] * N, [a_prime] * N, threshold_pct=threshold)


def test_clean_overhead_under_threshold_passes() -> None:
    r = ev(10.0, 10.5, 10.0)
    assert r.verdict == "pass"
    assert r.delta_pct == pytest.approx(5.0, rel=1e-9)
    assert r.control_pct == pytest.approx(0.0, abs=1e-9)


def test_overhead_over_threshold_fails() -> None:
    r = ev(10.0, 12.0, 10.0)
    assert r.verdict == "fail"
    assert r.delta_pct == pytest.approx(20.0, rel=1e-9)


def test_adversarial_one_percent_with_four_percent_control_is_not_resolvable() -> None:
    """The task's adversarial row: a true 1 % overhead under a 4 % A/A' spread must not pass."""
    r = ev(10.0, 10.1, 10.4)
    assert r.verdict == "not_resolvable"
    assert r.verdict != "pass"
    assert r.delta_pct == pytest.approx(1.0, rel=1e-9)
    assert r.control_pct == pytest.approx(4.0, rel=1e-9)


def test_adversarial_seeded_noise_one_percent_under_four_percent_control() -> None:
    rng = random.Random(0)
    a = [10.0 + rng.gauss(0, 0.02) for _ in range(N)]
    b = [10.1 + rng.gauss(0, 0.02) for _ in range(N)]
    a_prime = [10.4 + rng.gauss(0, 0.02) for _ in range(N)]
    r = b13().evaluate(a, b, a_prime, threshold_pct=15.0)
    assert r.verdict == "not_resolvable"


def test_control_equal_to_delta_is_not_resolvable() -> None:
    """Boundary: the rule is control >= |delta|."""
    assert ev(10.0, 10.4, 10.4).verdict == "not_resolvable"


def test_control_magnitude_counts_when_a_prime_is_faster() -> None:
    assert ev(10.0, 10.1, 9.6).verdict == "not_resolvable"


def test_guard_not_too_wide_small_control_below_delta_resolves() -> None:
    """Companion (tests.md: a guard needs a row for being too wide): control 1 %, delta 5 % -> pass."""
    assert ev(10.0, 10.5, 10.1).verdict == "pass"


def test_large_control_but_larger_delta_still_fails() -> None:
    assert ev(10.0, 12.0, 10.4).verdict == "fail"


def test_negative_delta_beyond_control_passes() -> None:
    """B faster than A by 5 %, control 1 %: resolvable, and an overhead of -5 % is under any threshold."""
    assert ev(10.0, 9.5, 10.1).verdict == "pass"


def test_threshold_is_the_argument_not_a_constant() -> None:
    """8 % overhead, no control: passes at 15 % (bucket <= 128), fails at 5 %."""
    assert ev(10.0, 10.8, 10.0, threshold=15.0).verdict == "pass"
    assert ev(10.0, 10.8, 10.0, threshold=5.0).verdict == "fail"


def test_unpaired_lengths_are_refused() -> None:
    with pytest.raises(ValueError):
        b13().evaluate([10.0] * N, [10.0] * (N - 1), [10.0] * N, threshold_pct=15.0)


# ------------------------------------------------------------------ E10: a confidence statement (review)


def _series(seed: int, n: int, overhead: float, jitter: float, control: float = 0.0):  # type: ignore[no-untyped-def]
    """Seeded synthetic per-pair step times: base 10 ms with multiplicative jitter ``jitter`` (sd)."""
    rng = random.Random(seed)
    a = [10.0 * (1 + rng.gauss(0, jitter)) for _ in range(n)]
    b = [x * (1 + overhead) * (1 + rng.gauss(0, jitter)) for x in a]
    a_prime = [x * (1 + control) * (1 + rng.gauss(0, jitter)) for x in a]
    return a, b, a_prime


def test_e10_twelve_percent_true_overhead_never_passes_a_ten_percent_threshold() -> None:
    """The reviewer's case: +12 % true overhead, 5 % jitter, n = 30, threshold 10 %. Under the old
    point-estimate rule this passed 17.6-28.9 % of the time; seeds 0-49 must never pass."""
    passed = []
    for seed in range(50):
        a, b, a_prime = _series(seed, 30, 0.12, 0.05)
        if b13().evaluate(a, b, a_prime, threshold_pct=10.0).verdict == "pass":
            passed.append(seed)
    assert passed == []


def test_e10_fewer_than_20_pairs_never_pass() -> None:
    """A clear 2 % overhead with 0.2 % jitter against a 10 % threshold, but n = 19."""
    a, b, a_prime = _series(1, 19, 0.02, 0.002)
    assert b13().evaluate(a, b, a_prime, threshold_pct=10.0).verdict != "pass"


def test_e10_twenty_pairs_is_enough() -> None:
    """Companion (not too wide): the same data at n = 20 passes."""
    a, b, a_prime = _series(1, 20, 0.02, 0.002)
    assert b13().evaluate(a, b, a_prime, threshold_pct=10.0).verdict == "pass"


def test_e10_clear_two_percent_at_n40_passes() -> None:
    a, b, a_prime = _series(2, 40, 0.02, 0.002)
    r = b13().evaluate(a, b, a_prime, threshold_pct=10.0)
    assert r.verdict == "pass"
    assert r.delta_pct == pytest.approx(2.0, abs=0.5)


def test_e10_clearly_over_threshold_fails() -> None:
    """+20 % with 0.5 % jitter at n = 30: the whole interval is above 10 %."""
    a, b, a_prime = _series(3, 30, 0.20, 0.005)
    assert b13().evaluate(a, b, a_prime, threshold_pct=10.0).verdict == "fail"


def test_e10_sign_test_interval_hand_values_n20() -> None:
    """Distribution-free 95 % interval on the median, n = 20. By hand: P(Binom(20, 1/2) <= 5) =
    (1+20+190+1140+4845+15504)/2^20 = 21700/1048576 = 0.0207, doubled 0.0414 <= 0.05; adding j = 6
    (38760) gives 60460/1048576 = 0.0577, doubled 0.115 > 0.05. So k = 6 and the interval is the 6th and
    15th order statistics. (At alpha = 0.2, k = 7: the 7th and 14th -- a different answer.)"""
    values = [float((7 * i) % 20) for i in range(20)]  # a permutation of 0..19, unsorted
    assert sorted(values) == [float(v) for v in range(20)]
    lo, hi = b13().sign_test_interval(values)
    assert (lo, hi) == (5.0, 14.0)
