"""T-0070 criterion 7: Wilson 95 % intervals, exact McNemar on paired per-record outcomes, Bonferroni
over 10, and the "both parsed" restriction reported alongside.

Oracles, independent of the implementation: Wilson intervals published in diffgemma-fa docs/RESULTS.md
at 27c7d0d (line 87, three decimals) and the closed form of the k = 0 upper limit, z^2 / (n + z^2);
McNemar p-values computed by hand from binomial coefficients (two-sided, 2 * sum_{i<=min(b,c)} C(n,i)
/ 2^n, clipped to 1). Pure statistics, CPU.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from tests.cd import _eval_support as es


@pytest.mark.parametrize(
    ("k", "n", "lo", "hi"),
    [
        (82, 130, 0.545, 0.709),  # unconstrained schema valid
        (65, 130, 0.415, 0.585),  # unconstrained exact call
        (0, 130, 0.000, 0.029),  # cs of the unconstrained arm
        (126, 130, 0.924, 0.988),  # parse rate
        (186, 291, 0.583, 0.692),  # arg accuracy over arg_total = 291
    ],
)
def test_wilson_matches_published_intervals(k: int, n: int, lo: float, hi: float) -> None:
    got = es.stats().wilson(k, n)
    assert (round(got[0], 3), round(got[1], 3)) == (lo, hi)


def test_wilson_boundaries_closed_form() -> None:
    """k = 0: lower 0, upper z^2/(n+z^2) exactly; k = n mirrors it. 1.96^2 = 3.8416."""
    lo, hi = es.stats().wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(3.8416 / 13.8416, rel=1e-9)
    lo, hi = es.stats().wilson(10, 10)
    assert hi == pytest.approx(1.0, abs=1e-12)
    assert lo == pytest.approx(10 / 13.8416, rel=1e-9)


@pytest.mark.parametrize(
    ("b", "c", "p"),
    [
        (0, 0, Fraction(1)),
        (0, 5, Fraction(2, 32)),
        (1, 5, Fraction(2 * (1 + 6), 64)),
        (5, 1, Fraction(2 * (1 + 6), 64)),
        (3, 3, Fraction(1)),  # 2 * 42/64 > 1, clipped
        (0, 10, Fraction(2, 1024)),
        (3, 9, Fraction(2 * (1 + 12 + 66 + 220), 4096)),
    ],
)
def test_mcnemar_exact_hand_values(b: int, c: int, p: Fraction) -> None:
    assert es.stats().mcnemar_exact(b, c) == pytest.approx(float(p), rel=1e-12, abs=0)


def _table() -> tuple[dict[str, dict[str, bool]], dict[str, dict[str, bool]]]:
    """17 paired records. Full table: A-only successes b = 3 (B unparsed on all three), B-only c = 9
    (both parsed), 5 concordant (3 both parsed). Both-parsed restriction: n = 12, b = 0, c = 9."""
    a: dict[str, dict[str, bool]] = {}
    bb: dict[str, dict[str, bool]] = {}
    for i in range(9):
        a[f"c{i}"], bb[f"c{i}"] = {"ok": False, "parsed": True}, {"ok": True, "parsed": True}
    for i in range(3):
        a[f"b{i}"], bb[f"b{i}"] = {"ok": True, "parsed": True}, {"ok": False, "parsed": False}
    for i in range(3):
        a[f"s{i}"], bb[f"s{i}"] = {"ok": True, "parsed": True}, {"ok": True, "parsed": True}
    a["f0"], bb["f0"] = {"ok": False, "parsed": False}, {"ok": False, "parsed": True}
    a["f1"], bb["f1"] = {"ok": False, "parsed": True}, {"ok": False, "parsed": False}
    return a, bb


def test_paired_comparison_counts_p_and_bonferroni() -> None:
    a, b = _table()
    r = es.stats().paired(a, b)
    assert (r.n, r.b, r.c) == (17, 3, 9)
    p_full = 2 * (1 + 12 + 66 + 220) / 4096
    assert r.p == pytest.approx(p_full, rel=1e-12)
    # T-0076 Q2: the Bonferroni family is 6 pre-registered pairs x 2 tasks = 12 (was 10)
    assert r.p_bonferroni == pytest.approx(min(1.0, 12 * p_full), rel=1e-12)
    bp = r.both_parsed
    assert (bp.n, bp.b, bp.c) == (12, 0, 9)
    assert bp.p == pytest.approx(2 / 512, rel=1e-12)
    assert bp.p_bonferroni == pytest.approx(12 * 2 / 512, rel=1e-12)


def test_paired_refuses_unpaired_records() -> None:
    """Pairing is by record id; a record missing from one arm is an error, never silently dropped."""
    a, b = _table()
    del b["c0"]
    with pytest.raises(ValueError):
        es.stats().paired(a, b)


def test_preregistered_pairs_and_bonferroni_family() -> None:
    s = es.stats()
    assert set(map(tuple, s.PREREGISTERED_PAIRS)) == {
        ("perstep_mar", "end_projection"),
        ("perstep_mar", "unconstrained"),
        ("end_projection", "unconstrained"),
        ("perstep_mar", "perstep_mf"),
        ("perstep_mar_sc", "perstep_mar"),
        (
            "perstep_mar_qargmax",
            "perstep_mar",
        ),  # T-0076 Q2, fixed before T-0071c touches a test cut
    }
    assert len(s.PREREGISTERED_PAIRS) == 6
    assert s.N_COMPARISONS == 12
    # Q2: the SC twin is reported with its Wilson CI, never significance-tested
    assert not [p for p in s.PREREGISTERED_PAIRS if "perstep_mar_qargmax_sc" in p]
    # the family size drives the correction: a p of 0.004 is 0.048 after Bonferroni over 12
    assert s.bonferroni(0.004) == pytest.approx(0.048, rel=1e-12)


def test_exact_mcnemar_is_not_the_chi_square_approximation() -> None:
    """b = 0, c = 6: exact p = 2/64 = 0.03125; chi-square (1 d.o.f., no continuity) gives p ~ 0.0143."""
    exact = es.stats().mcnemar_exact(0, 6)
    assert exact == pytest.approx(0.03125, rel=1e-12)
    chi2 = math.erfc(math.sqrt(6.0 / 2))  # P(chi2_1 > 6)
    assert abs(exact - chi2) > 0.01
