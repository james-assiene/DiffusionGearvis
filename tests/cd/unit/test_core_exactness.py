"""T-0065 criterion 1: exactness on tiny instances against brute-force enumeration (SPEC §6.1).

V ≤ 6, L ≤ 8, |S| ≤ 5; DFAs and NFAs with parallel overlapping labels; non-singleton start sets;
budget terminal factors ``b = 1[d(s) ≤ R]`` (R drawn so b is neither F nor every state in general).
Oracle: ``tests/cd/oracle/enumerate.py`` (string enumeration with integer path multiplicities),
float64 log space; itself cross-checked against the exact ``Fraction`` enumerator below.
"""

from __future__ import annotations

import functools
import math
import random
from fractions import Fraction

import pytest

from tests.cd.oracle.enumerate import (
    distance_to_final,
    enumerate_fraction,
    enumerate_log,
    log_fraction,
    log_softmax64,
)
from tests.cd.unit._core_support import dfa_toy, random_logits, random_toy, run_core

# (V, L, S, kind): V^L kept ≤ 6561 so enumeration stays cheap; together they reach every bound.
COMBOS = [
    (2, 8, 3, "dfa"),
    (3, 8, 5, "nfa"),
    (4, 6, 5, "dfa"),
    (4, 6, 5, "nfa"),
    (5, 5, 4, "nfa"),
    (6, 4, 5, "dfa"),
    (6, 4, 4, "nfa"),
    (3, 1, 2, "dfa"),
    (6, 1, 5, "nfa"),
]
SEEDS = (0, 1, 2)
TOL = {
    "float64": {"logz_abs": 1e-10, "q": 1e-12, "h": None},
    "float32": {"logz_rel": 1e-4, "q": 1e-6, "h": 1e-5},
}


@functools.cache
def instance(combo: tuple[int, int, int, str], seed: int):
    V, L, S, kind = combo
    rng = random.Random(hash((combo, seed)) & 0xFFFFFFFF)
    for _ in range(200):
        toy = random_toy(rng, V=V, S=S, nfa=True) if kind == "nfa" else dfa_toy(rng, V=V, S=S)
        d = distance_to_final(toy)
        live = [s for s in range(S) if d[s] < math.inf]
        start = sorted(rng.sample(range(S), rng.randint(min(2, S), S)))  # non-singleton when S ≥ 2
        R = rng.choice([0, 1, 2, L])
        end = [s for s in range(S) if d[s] <= R]
        if not live or not end:
            continue
        logits = random_logits(rng, L, V, span=8.0)
        ora = enumerate_log(log_softmax64(logits), toy, start, end)
        if ora.log_z == -math.inf:
            continue
        return toy, start, end, logits, ora
    raise AssertionError(f"no feasible instance for {combo} seed {seed}")


@pytest.mark.parametrize("dtype_name", ["float64", "float32"])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("combo", COMBOS, ids=lambda c: f"V{c[0]}-L{c[1]}-S{c[2]}-{c[3]}")
def test_core_matches_bruteforce(combo, seed, dtype_name):
    import torch

    toy, start, end, logits, ora = instance(combo, seed)
    dtype = getattr(torch, dtype_name)
    res, _ = run_core(torch.tensor(logits, dtype=dtype), toy, start, end)
    tol = TOL[dtype_name]
    log_z = float(res.log_z)
    if dtype_name == "float64":
        assert abs(log_z - ora.log_z) <= tol["logz_abs"], (log_z, ora.log_z)
    else:
        assert abs(log_z - ora.log_z) <= tol["logz_rel"] * max(1.0, abs(ora.log_z)), (
            log_z,
            ora.log_z,
        )
    q = res.q.double().tolist()
    L, V = combo[1], combo[0]
    dq = max(abs(q[i][v] - ora.q[i][v]) for i in range(L) for v in range(V))
    assert dq <= tol["q"], f"max |Δq| = {dq:.3e}"
    if tol["h"] is not None:
        h = res.entropy.double().tolist()
        dh = max(abs(h[i] - ora.entropy[i]) for i in range(L))
        assert dh <= tol["h"], f"max |ΔH(q_i)| = {dh:.3e} nats"


@pytest.mark.parametrize("seed", range(4))
def test_core_matches_fraction_oracle_exactly_fp64(seed):
    """Rational p = k/Σk (logits = log k): fp64 core against the exact Fraction enumeration."""
    import torch

    rng = random.Random(1000 + seed)
    V, L, S = 4, 5, 4
    for _ in range(200):
        toy = random_toy(rng, V=V, S=S, nfa=True)
        start = sorted(rng.sample(range(S), 2))
        end = sorted(toy.finals)
        ks = [[rng.randint(1, 9) for _ in range(V)] for _ in range(L)]
        p = [[Fraction(k, sum(row)) for k in row] for row in ks]
        Z, qf = enumerate_fraction(p, toy, start, end)
        if Z:
            break
    logits = torch.tensor([[math.log(k) for k in row] for row in ks], dtype=torch.float64)
    res, _ = run_core(logits, toy, start, end)
    assert abs(float(res.log_z) - log_fraction(Z)) <= 1e-10
    q = res.q.tolist()
    assert max(abs(q[i][v] - float(qf[i][v])) for i in range(L) for v in range(V)) <= 1e-12


@pytest.mark.parametrize("seed", range(3))
def test_nfa_multiplicity_is_counted(seed):
    """Adversarial: two parallel edges with the same label double that string's weight (SPEC §2.2).

    An ``∃``-style transfer (max/or over parallel edges instead of a sum) gives log Z off by log 2.
    """
    import torch

    from tests.cd.oracle.enumerate import Toy

    V, L = 3, 4
    toy = Toy(
        n_states=2,
        vocab_size=V,
        classes=((frozenset({0, 1}), False), (frozenset({1, 2}), False), (frozenset({2}), True)),
        edges=((0, 1, 0), (0, 1, 1), (1, 0, 2), (1, 1, 1)),
        finals=frozenset({1}),
    )
    assert not toy.is_dfa()
    rng = random.Random(seed)
    logits = random_logits(rng, L, V, span=4.0)
    ora = enumerate_log(log_softmax64(logits), toy, [0], [1])
    res, tables = run_core(torch.tensor(logits, dtype=torch.float64), toy, [0], [1])
    assert tables.is_dfa is False
    assert abs(float(res.log_z) - ora.log_z) <= 1e-10


def test_oracle_self_check_log_vs_fraction():
    """Support row (not an acceptance row): the float64 log enumerator agrees with Fraction."""
    rng = random.Random(7)
    toy = random_toy(rng, V=3, S=3, nfa=True)
    L = 4
    ks = [[rng.randint(1, 6) for _ in range(3)] for _ in range(L)]
    p = [[Fraction(k, sum(row)) for k in row] for row in ks]
    start, end = [0, 1], sorted(toy.finals)
    Z, qf = enumerate_fraction(p, toy, start, end)
    ora = enumerate_log(
        [[math.log(k) - math.log(sum(row)) for k in row] for row in ks], toy, start, end
    )
    if Z:
        assert abs(ora.log_z - log_fraction(Z)) <= 1e-12
        assert max(abs(ora.q[i][v] - float(qf[i][v])) for i in range(L) for v in range(3)) <= 1e-14
    else:
        assert ora.log_z == -math.inf
