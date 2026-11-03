"""T-0065 criteria 3–5: non-subtractive complement, no spurious Z = 0, no NaN/Inf.

Oracles: ``Fraction`` (logits = n·ln 2, so p is exactly rational), float64 log-space string
enumeration, and an entry-by-entry float64 logsumexp for ``log_matmul``. The subtractive-complement
and single-shift ``log_matmul`` mutants appear as named *controls*: each control asserts that the
mutant formula is red on the same input, which proves the input discriminates.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from tests.cd.oracle.enumerate import (
    Toy,
    enumerate_log,
    log_fraction,
    log_matmul_bruteforce,
    log_softmax64,
)
from tests.cd.unit._core_support import (
    SENTINEL_MAX,
    core,
    random_logits,
    random_toy,
    run_core,
    tables_for,
)

LN2 = math.log(2.0)

# ----------------------------------------------------------------------------- criterion 3


def _neg_class_case(n_hot: int):
    """V = 6; the negated class stores N_c = {0, 1}; tokens 0/1 carry 2^n_hot, 2^(n_hot-2); others 1."""
    V = 6
    exps = [n_hot, n_hot - 2, 0, 0, 0, 0]
    w = [Fraction(2) ** e for e in exps]
    tot = sum(w)
    true_mass = sum(w[2:]) / tot
    assert true_mass <= Fraction(1, 10**7)  # complement set N_c holds ≥ 1 − 1e-7 of the mass
    logits = [e * LN2 for e in exps]
    toy = Toy(1, V, ((frozenset({0, 1}), True),), ((0, 0, 0),), frozenset({0}))
    return toy, logits, true_mass


@pytest.mark.parametrize("n_hot", [26, 60, 200], ids=lambda n: f"span{n * LN2:.0f}nats")
def test_negated_class_log_weight_matches_fraction(n_hot):
    import torch

    toy, logits, true_mass = _neg_class_case(n_hot)
    expected = log_fraction(true_mass)
    x = torch.tensor([logits], dtype=torch.float32)
    got = float(core("emission").class_log_weights(x, tables_for(toy))[0, 0])
    assert math.isfinite(got), got
    assert abs(math.expm1(got - expected)) <= 1e-4, (got, expected)

    # control: the subtractive form log(1 − Σ_{N_c} p) in the same precision is demonstrably red
    p = torch.softmax(x, dim=-1)[0]
    sub = float(torch.log(1.0 - (p[0] + p[1])))
    assert not (math.isfinite(sub) and abs(math.expm1(sub - expected)) <= 1e-4), (
        f"control failed to be red: subtractive gave {sub}, oracle {expected}"
    )


def test_negated_class_through_the_whole_core_fp32():
    """A 1-state loop on the negated class: log Z = L · log(class mass), by hand."""
    import torch

    toy, logits, true_mass = _neg_class_case(60)
    L = 5
    res, _ = run_core(torch.tensor([logits] * L, dtype=torch.float32), toy, [0], [0])
    expected = L * log_fraction(true_mass)
    assert abs(float(res.log_z) - expected) <= 1e-4 * max(1.0, abs(expected))
    q = res.q[0].double().tolist()
    # the posterior lives on tokens 2..5 only, uniformly (equal logits)
    assert q[0] == 0.0 and q[1] == 0.0
    assert max(abs(q[v] - 0.25) for v in range(2, 6)) <= 1e-6


# -------------------------------------------------------------- log_matmul (criterion 3/4 support)


def _single_shift(a, b):
    """Named mutant control: one global shift per operand."""
    import torch

    ma, mb = a.max(), b.max()
    return torch.log(torch.exp(a - ma) @ torch.exp(b - mb)) + ma + mb


def _row_col_shift(a, b):
    """Named mutant control: row max of A plus column max of B (still not pairwise)."""
    import torch

    ma = a.max(dim=-1, keepdim=True).values
    mb = b.max(dim=-2, keepdim=True).values
    return torch.log(torch.exp(a - ma) @ torch.exp(b - mb)) + ma + mb


@pytest.mark.parametrize("dtype_name", ["float64", "float32"])
def test_log_matmul_pairwise_anchor(dtype_name):
    import torch

    dtype = getattr(torch, dtype_name)
    A = [[0.0, -800.0, -1e30], [-900.0, -905.0, -1e30], [-1e30, -1e30, -1e30]]
    B = [[-800.0, 0.0, -2.0], [0.0, -1e30, -7.0], [-1.0, -4.0, -900.0]]
    expected = log_matmul_bruteforce(A, B)
    a, b = torch.tensor(A, dtype=dtype), torch.tensor(B, dtype=dtype)
    got = core("semiring").log_matmul(a, b).double().tolist()
    tol_abs = 1e-10 if dtype_name == "float64" else None
    for i in range(3):
        for k in range(3):
            e, g = expected[i][k], got[i][k]
            assert math.isfinite(g), (i, k, g)
            if e < -1e28:  # an all-sentinel sum is "zero": stays at or below the sentinel line
                assert g <= SENTINEL_MAX, (i, k, g)
                continue
            tol = tol_abs if tol_abs is not None else 1e-4 * max(1.0, abs(e))
            assert abs(g - e) <= tol, (i, k, g, e)
    # batched over leading dims
    batched = core("semiring").log_matmul(torch.stack([a, a]), torch.stack([b, b]))
    assert tuple(batched.shape) == (2, 3, 3)

    # controls: both shift mutants lose entry (0, 0) = log(2)·e^-800 on this input
    for mutant in (_single_shift, _row_col_shift):
        m00 = float(mutant(a, b)[0, 0])
        assert not (math.isfinite(m00) and abs(m00 - expected[0][0]) <= 1e-3), mutant.__name__


# ----------------------------------------------------------------------------- criterion 4


def _two_band_toy() -> Toy:
    """0 -1-> 1 -1-> 2 -2-> 3 (ACC); ACC has the unscored Σ tail (negated class, empty N_c)."""
    return Toy(
        4,
        4,
        ((frozenset({1}), False), (frozenset({2}), False), (frozenset(), True)),
        ((0, 1, 0), (1, 2, 0), (2, 3, 1), (3, 3, 2)),
        frozenset({3}),
    )


@pytest.mark.parametrize("dtype_name", ["float64", "float32"])
def test_two_band_no_spurious_zero_and_i_invariance(dtype_name):
    """live_simple_106-63-0 regime: tail at 0 nats, the genuine path ~850 nats below."""
    import torch

    L = 8
    toy = _two_band_toy()
    logits = [[0.0, -283.0, -283.0, 0.0] for _ in range(L)]
    ora = enumerate_log(log_softmax64(logits), toy, [0], [3])
    assert -860.0 < ora.log_z < -840.0  # the regime is what it claims to be
    res, _ = run_core(torch.tensor(logits, dtype=getattr(torch, dtype_name)), toy, [0], [3])
    log_z = float(res.log_z)
    assert math.isfinite(log_z) and log_z > SENTINEL_MAX, "spurious Z = 0"
    tol = 1e-10 if dtype_name == "float64" else 1e-4 * max(1.0, abs(ora.log_z))
    assert abs(log_z - ora.log_z) <= tol, (log_z, ora.log_z)
    la, lb = res.log_a.double(), res.log_b.double()
    for i in range(L + 1):
        zi = float(torch.logsumexp(la[i] + lb[i], dim=0))
        assert abs(zi - ora.log_z) <= tol, (
            f"log(a_i·b_i) not i-invariant at i={i}: {zi} vs {ora.log_z}"
        )


@pytest.mark.parametrize("span", [30.0, 150.0, 250.0])
def test_no_spurious_zero_random_sweep_fp32(span):
    """n = 12 random tiny NFAs per span (seeded); oracle Z > 0 ⇒ finite log Z matching the oracle."""
    import torch

    rng = random.Random(int(span))
    checked = 0
    while checked < 12:
        toy = random_toy(rng, V=3, S=4, nfa=True)
        L = 6
        logits = random_logits(rng, L, 3, span=span)
        start = [0, 1]
        end = sorted(toy.finals)
        ora = enumerate_log(log_softmax64(logits), toy, start, end)
        if ora.log_z == -math.inf:
            continue
        checked += 1
        res, _ = run_core(torch.tensor(logits, dtype=torch.float32), toy, start, end)
        log_z = float(res.log_z)
        assert math.isfinite(log_z) and log_z > SENTINEL_MAX, (checked, ora.log_z)
        assert abs(log_z - ora.log_z) <= 1e-4 * max(1.0, abs(ora.log_z)), (log_z, ora.log_z)
        la, lb = res.log_a.double(), res.log_b.double()
        for i in range(L + 1):
            zi = float(torch.logsumexp(la[i] + lb[i], dim=0))
            assert abs(zi - ora.log_z) <= 1e-4 * max(1.0, abs(ora.log_z)), (i, zi, ora.log_z)


# ----------------------------------------------------------------------------- criterion 5


def _filtered_case():
    """V = 5, L = 4. State 0 --{0,1,2}--> 1, 1 --{3}--> 1 (only token 3), 1 --{1}--> 0; F = {1}."""
    toy = Toy(
        2,
        5,
        ((frozenset({0, 1, 2}), False), (frozenset({3}), False), (frozenset({1}), False)),
        ((0, 1, 0), (1, 1, 1), (1, 0, 2)),
        frozenset({1}),
    )
    ninf = float("-inf")
    logits = [
        [1.0, ninf, 0.5, 2.0, -1e30],  # top-k filtered + finite sentinel
        [ninf, -1e30, 0.0, 3.0, ninf],  # token 1 (the only way back to 0) is filtered
        [0.0, 1.0, ninf, 0.2, -1e20],
        [ninf, ninf, ninf, 0.0, ninf],  # only token 3 survives the filter
    ]
    return toy, logits


@pytest.mark.parametrize("dtype_name", ["float64", "float32"])
def test_filtered_logits_no_nan_inf_and_exact_zeros(dtype_name):
    import torch

    toy, logits = _filtered_case()
    L, V = 4, 5
    ora = enumerate_log(log_softmax64(logits), toy, [0], [1])
    assert math.isfinite(ora.log_z)
    res, _ = run_core(torch.tensor(logits, dtype=getattr(torch, dtype_name)), toy, [0], [1])
    for name in ("log_z", "q", "entropy", "log_a", "log_b"):
        t = getattr(res, name)
        assert bool(torch.isfinite(t).all()), f"{name} has NaN/Inf: {t}"
    q = res.q.double().tolist()
    h = res.entropy.double().tolist()
    for i in range(L):
        for v in range(V):
            if ora.q[i][v] == 0.0:
                assert q[i][v] == 0.0, (
                    f"forbidden token ({i},{v}) has q = {q[i][v]!r}, not exactly 0"
                )
        assert h[i] >= 0.0, (i, h[i])
        assert abs(h[i] - ora.entropy[i]) <= 1e-5, (i, h[i], ora.entropy[i])
    # position 3 admits exactly one token: q = 1 there, and the forbidden ones add exactly 0 to H
    assert q[3][3] == pytest.approx(1.0, abs=1e-6)
    assert abs(h[3]) <= 1e-6, h[3]


def test_forbidden_tokens_contribute_exactly_zero_entropy():
    """Every token but one is forbidden by the automaton at every position: forbidden q is exactly 0,
    H(q_i) is 0 up to the rounding of the one allowed q (≈ 1), never NaN, never negative."""
    import torch

    toy = Toy(1, 6, ((frozenset({4}), False),), ((0, 0, 0),), frozenset({0}))
    logits = torch.tensor(random_logits(random.Random(3), 3, 6, span=40.0), dtype=torch.float32)
    res, _ = run_core(logits, toy, [0], [0])
    q = res.q.double().tolist()
    for i in range(3):
        assert [q[i][v] for v in (0, 1, 2, 3, 5)] == [0.0] * 5, q[i]
        assert abs(q[i][4] - 1.0) <= 1e-6, q[i][4]
    h = res.entropy.double().tolist()
    assert all(0.0 <= x <= 1e-6 for x in h), h
