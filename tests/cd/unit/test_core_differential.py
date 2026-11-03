"""T-0065 criterion 2: fp32 core vs the float64 reference at production shape, on CPU (slow).

L = 256, V = 262 144, buckets 64 and 128, synthetic automata shaped like compiled grammars
(interned positive classes of ~30 ids, negated string-body classes storing ~800-id ``N_c``,
back edges, parallel overlapping edges, the unscored Σ tail at F). Regimes: T = 0.8 and 0.4 on
softcapped logits ``30·tanh(x/30)/T`` (≤ 150 nats per position), synthetic 250-nat spans, the model's
mass inside a narrow negated class's ``N_c``, a start set that includes the tail, and ``L_eff < L``.

The bar is relative to stock: per position, the fp32 error of H(q_i) must be ≤ 10× the fp32 error of
the stock H(p_i) on the same logits + 1e-6, and likewise max_v |Δq_i| ≤ 10·max_v |Δp_i| + 1e-6.
Arbiter: ``tests/cd/oracle/dense64.py`` (float64, direct class sums, log-space forward–backward).

Gap (reported, not guessed): "automata from T-0063" are not covered here — T-0063's compiler is not
implemented on this branch and its API is not fixed yet. Add a row over its four cuts when it lands.
"""

from __future__ import annotations

import functools
import math
import random

import numpy as np
import pytest

from tests.cd.oracle.dense64 import forward_backward64
from tests.cd.oracle.enumerate import Toy, distance_to_final
from tests.cd.unit._core_support import SENTINEL_MAX, core, mask, run_core

pytestmark = pytest.mark.slow

L, V = 256, 262_144


def production_toy(rng: random.Random, n: int) -> tuple[Toy, list[int]]:
    """Returns the automaton and the union of every negated class's stored ``N_c``."""
    pos_pool = [frozenset(rng.sample(range(V), 30)) for _ in range(6)]
    neg_pool = [frozenset(rng.sample(range(V), 800)) for _ in range(3)]
    classes: list[tuple[frozenset[int], bool]] = [(c, False) for c in pos_pool] + [
        (c, True) for c in neg_pool
    ]
    classes.append((frozenset(), True))  # Σ: the unscored tail
    sigma = len(classes) - 1
    edges: list[tuple[int, int, int]] = []
    for s in range(n - 1):
        edges.append((s, s + 1, rng.randrange(6)))
        if s % 3 == 0:
            edges.append((s, s, 6 + rng.randrange(3)))  # string body: everything but N_c
        if s % 5 == 1 and s > 2:
            edges.append((s, rng.randrange(s - 1), rng.randrange(6)))  # back edge
        if s % 7 == 2:
            edges.append((s, s + 1, rng.randrange(6)))  # parallel edge, possibly overlapping labels
    edges.append((n - 1, n - 1, sigma))
    n_c = sorted(set().union(*neg_pool))
    return Toy(n, V, tuple(classes), tuple(edges), frozenset({n - 1})), n_c


def _logits(regime: str, seed: int, n_c: list[int]):
    import torch

    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(L, V, generator=g) * 8.0
    spikes = torch.randint(0, V, (L, 50), generator=g)
    raw.scatter_add_(1, spikes, torch.full((L, 50), 60.0))
    if regime == "span250":
        return -250.0 * torch.rand(L, V, generator=g)
    temp = 0.8 if regime in ("T0.8", "l_eff") else 0.4
    x = 30.0 * torch.tanh(raw / 30.0)
    if regime == "narrow_neg":
        x[:, torch.tensor(n_c)] = 30.0  # the model's mass sits inside N_c
    return x / temp


REGIMES = ["T0.8", "T0.4", "span250", "narrow_neg", "tail", "l_eff"]
BUCKETS = {64: 50, 128: 100}  # bucket -> live states
#: seeds per regime. Seed 0 is the original instance of each row. narrow_neg and T0.4 are where the
#: fp32 H(q) error sits closest to the stock bar, so one seed under-samples them (review finding 3,
#: 2026-09-29: narrow_neg @ 64 passed on seed 0 and failed on 4 of 5 other seeds).
SEEDS = {"T0.4": range(5), "narrow_neg": range(5)}
ROWS = [
    pytest.param(regime, bucket, n, seed, id=f"{regime}-b{bucket}-seed{seed}")
    for regime in REGIMES
    for bucket, n in BUCKETS.items()
    for seed in SEEDS.get(regime, range(1))
]


@functools.lru_cache(maxsize=2)  # one reference holds ~1 GiB of float64 [L, V]; keep at most two
def _instance(regime: str, bucket: int, n_states: int, seed: int):
    """(toy, logits, start, end, l_eff, float64 reference), cached per (regime, bucket, seed)."""
    rng = random.Random(f"{regime}-{bucket}" if seed == 0 else f"{regime}-{bucket}-seed{seed}")
    toy, n_c = production_toy(rng, n_states)
    logits = _logits(regime, seed=bucket + REGIMES.index(regime) + 1000 * seed, n_c=n_c)
    l_eff = 100 if regime == "l_eff" else L
    d = distance_to_final(toy)
    start = [0, n_states - 1, n_states - 2] if regime == "tail" else [0]
    end = [s for s in range(n_states) if d[s] <= 40]
    ref = forward_backward64(
        logits.double().numpy(),
        toy,
        mask(start, n_states).numpy(),
        mask(end, n_states).numpy(),
        l_eff,
    )
    return toy, logits, start, end, l_eff, ref


@pytest.mark.parametrize(("regime", "bucket", "n_states", "seed"), ROWS)
def test_fp32_core_vs_float64_reference(regime, bucket, n_states, seed):
    import torch

    for name in ("emission", "tree", "marginals"):  # fail fast, before the float64 oracle runs
        core(name)
    toy, logits, start, end, l_eff, ref = _instance(regime, bucket, n_states, seed)
    assert math.isfinite(ref.log_z), "the instance must be feasible"

    res, _ = run_core(logits, toy, start, end, l_eff=l_eff, bucket=bucket)
    log_z = float(res.log_z)
    assert math.isfinite(log_z) and log_z > SENTINEL_MAX, "spurious Z = 0"
    assert abs(log_z - ref.log_z) <= 1e-4 * max(1.0, abs(ref.log_z)), (log_z, ref.log_z)

    # stock fp32 errors on the same logits (what vLLM already tolerates)
    lp32 = torch.log_softmax(logits[:l_eff], dim=-1)
    p32 = lp32.exp()
    h_stock32 = -(p32 * lp32).sum(-1).double().numpy()
    p64 = np.exp(ref.logp[:l_eff])
    h_p64 = -np.sum(np.where(p64 > 0, p64 * ref.logp[:l_eff], 0.0), axis=1)
    err_h_stock = np.abs(h_stock32 - h_p64)
    err_p_stock = np.max(np.abs(p32.double().numpy() - p64), axis=1)

    q32 = res.q[:l_eff].double().numpy()
    h32 = res.entropy[:l_eff].double().numpy()
    err_h = np.abs(h32 - ref.entropy)
    err_q = np.max(np.abs(q32 - ref.q), axis=1)
    bad_h = np.nonzero(err_h > 10 * err_h_stock + 1e-6)[0]
    bad_q = np.nonzero(err_q > 10 * err_p_stock + 1e-6)[0]
    assert bad_h.size == 0, (
        f"H(q) worse than 10x stock at {bad_h.size} positions; first i={bad_h[0]}: "
        f"{err_h[bad_h[0]]:.3e} vs stock {err_h_stock[bad_h[0]]:.3e}"
    )
    assert bad_q.size == 0, (
        f"q worse than 10x stock at {bad_q.size} positions; first i={bad_q[0]}: "
        f"{err_q[bad_q[0]]:.3e} vs stock {err_p_stock[bad_q[0]]:.3e}"
    )
    assert np.all(h32 >= 0.0)
