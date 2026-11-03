"""T-0066 criteria 4 and 6: the constrained MAP is exact, in C, temperature-invariant; production regimes.

Criterion 4 oracle: brute force over strings (``enumerate.support_scores``: every x with N(x) > 0 and
its score ``Σ log p_i(x_i)``); instances are resampled until the maximum is unique by a margin ≥ 1e-3
nats (tie-free, as the task asks, so no tie-break is specified). Tolerance on the score: 1e-5 relative
(fp32) / 1e-12 relative (fp64), relative to ``max(1, |score|)``.

Criterion 6 oracle: ``oracle/viterbi64.map_score64`` (float64 max-plus over true-member class maxima) on
T-0065 criterion 2's synthetic production automata and regimes (L = 256, V = 262 144, buckets 64/128,
T 0.8/0.4 softcapped, 250-nat spans, narrow negated classes, the tail start, L_eff < L), plus the
two-band case at tiny shape. Compared by score (the narrow_neg regime has exact logit ties), and the
returned string must itself attain that score and be accepted by host replay.
"""

from __future__ import annotations

import functools
import math
import random

import pytest

from tests.cd.oracle.enumerate import (
    Toy,
    distance_to_final,
    log_softmax64,
    support_scores,
)
from tests.cd.unit._core_support import dfa_toy, random_logits, random_toy
from tests.cd.unit._draw_support import (
    Replay,
    assert_draw_consistent,
    cdmod,
    draw,
    exists_control_nfa,
    generator,
    prepare,
    run_map,
)

COMBOS = [(3, 5, 4, "dfa"), (4, 4, 5, "dfa"), (3, 5, 4, "nfa"), (4, 4, 5, "nfa"), (5, 3, 4, "nfa")]
SEEDS = range(4)
REL = {"float32": 1e-5, "float64": 1e-12}


def _tie_free(scores: dict, margin: float = 1e-3):
    ranked = sorted(scores.items(), key=lambda kv: kv[1][1], reverse=True)
    if len(ranked) >= 2 and ranked[0][1][1] - ranked[1][1][1] < margin:
        return None
    return ranked[0][0], ranked[0][1][1]


@functools.cache
def instance(combo, seed, softcap: bool = False):
    V, L, S, kind = combo
    rng = random.Random(f"t0066-map-{combo}-{seed}-{softcap}")
    for _ in range(500):
        toy = random_toy(rng, V=V, S=S, nfa=True) if kind == "nfa" else dfa_toy(rng, V=V, S=S)
        d = distance_to_final(toy)
        start = sorted(rng.sample(range(S), rng.randint(2, S)))
        R = rng.choice([0, 1, 2, L])
        end = [s for s in range(S) if d[s] <= R]
        if not end:
            continue
        raw = random_logits(rng, L, V, span=40.0 if softcap else 6.0)
        logits = [[30.0 * math.tanh(x / 30.0) for x in row] for row in raw] if softcap else raw
        scores = support_scores(log_softmax64(logits), toy, start, end)
        if len(scores) < 2:
            continue
        best = _tie_free(scores)
        if best is None:
            continue
        return toy, start, end, logits, best
    raise AssertionError(f"no tie-free instance for {combo} seed {seed}")


def _score_tol(dtype_name, s):
    return REL[dtype_name] * max(1.0, abs(s))


@pytest.mark.parametrize("dtype_name", ["float64", "float32"])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("combo", COMBOS, ids=lambda c: f"V{c[0]}-L{c[1]}-S{c[2]}-{c[3]}")
def test_map_equals_bruteforce_argmax(combo, seed, dtype_name):
    import torch

    toy, start, end, logits, (x_best, s_best) = instance(combo, seed)
    lg = torch.tensor(logits, dtype=getattr(torch, dtype_name))
    prep = prepare(lg, toy, start, end)
    res = run_map(prep)
    toks = tuple(int(t) for t in res.tokens.tolist())
    assert not bool(res.empty)
    assert Replay(toy).accepts(start, end, toks), f"MAP {toks} is not in C"
    assert toks == x_best, f"MAP {toks} != brute-force argmax {x_best}"
    s = float(res.score)
    assert abs(s - s_best) <= _score_tol(dtype_name, s_best), (s, s_best)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("combo", COMBOS[1:4], ids=lambda c: f"V{c[0]}-L{c[1]}-S{c[2]}-{c[3]}")
def test_map_is_temperature_invariant(combo, seed):
    """SPEC §2.7: the same softcapped logits at T = 0.8 and T = 0.4 give the same MAP (fp32)."""
    import torch

    toy, start, end, logits, (x_best, _) = instance(combo, seed, softcap=True)
    base = torch.tensor(logits, dtype=torch.float32)
    prep = prepare(base / 0.8, toy, start, end)
    hot = run_map(prep, base / 0.8)
    cold = run_map(prep, base / 0.4)
    t_hot = tuple(int(t) for t in hot.tokens.tolist())
    t_cold = tuple(int(t) for t in cold.tokens.tolist())
    assert t_hot == t_cold, f"MAP changed with temperature: {t_hot} vs {t_cold}"
    assert t_hot == x_best
    # the score is not temperature-invariant: each matches its own brute force
    for T, r in ((0.8, hot), (0.4, cold)):
        ref = support_scores(
            log_softmax64([[v / T for v in row] for row in logits]), toy, start, end
        )
        s_ref = max(s for _, s in ref.values())
        assert abs(float(r.score) - s_ref) <= _score_tol("float32", s_ref), (
            T,
            float(r.score),
            s_ref,
        )


def test_map_on_nfa_parallel_edges_hand_case():
    """exists_control_nfa (parallel, overlapping labels on both pairs): token 2 dominates p_0 but is in
    no label out of state 0. By hand: argmax over C = (0, 2, 1) (0 beats 1 at position 0; 2 is best at
    position 1; position 2 is the Σ tail, argmax 1)."""
    import torch

    toy = exists_control_nfa()
    logits = [[1.0, 0.5, 3.0], [0.2, 0.1, 0.9], [0.0, 2.0, 0.3]]  # token 2 at 0 is forbidden
    prep = prepare(torch.tensor(logits, dtype=torch.float32), toy, [0], [2])
    res = run_map(prep)
    assert tuple(res.tokens.tolist()) == (0, 2, 1)


def test_map_never_selects_sentinel_or_filtered_tokens():
    """Adversarial: the per-position argmax is forbidden everywhere; valid tokens sit at -1e4 (finite,
    far below; -1e20 would swamp fp64 and tie the strings) or next to -inf/-1e30 filtered entries. The MAP must still be the brute-force valid string."""
    import torch

    # 0 --{0,1,2}--> 1, 1 --{3}--> 1, 1 --{1}--> 0; F = {1}; token 4 is in no label
    toy = Toy(
        2,
        5,
        ((frozenset({0, 1, 2}), False), (frozenset({3}), False), (frozenset({1}), False)),
        ((0, 1, 0), (1, 1, 1), (1, 0, 2)),
        frozenset({1}),
    )
    ninf = float("-inf")
    logits = [
        [1.0, ninf, 0.5, 2.0, 50.0],  # token 4 dominates but is never valid
        [ninf, -1e30, 0.0, -1e4, 60.0],  # stay in F: token 3 at -1e4; leave F: token 1 at -1e30
        [0.0, 1.0, ninf, 0.2, 70.0],
        [ninf, ninf, ninf, 0.0, 80.0],
    ]
    scores = support_scores(log_softmax64(logits), toy, [0], [1])
    x_best, s_best = _tie_free(scores)
    prep = prepare(torch.tensor(logits, dtype=torch.float64), toy, [0], [1])
    res = run_map(prep)
    toks = tuple(int(t) for t in res.tokens.tolist())
    assert 4 not in toks
    assert all(math.isfinite(logits[i][v]) for i, v in enumerate(toks)), toks
    assert toks == x_best, (toks, x_best)
    assert abs(float(res.score) - s_best) <= _score_tol("float64", s_best)


def test_map_and_draw_report_empty_when_filtering_removes_every_valid_token():
    """Every valid token at position 1 is -inf (top-k filtered): Z = 0. The max-plus sentinel must not
    be 'selected' as if it were a path: both functions report empty."""
    import torch

    toy = Toy(
        2,
        3,
        ((frozenset({0}), False), (frozenset({1}), False)),
        ((0, 1, 0), (1, 1, 1)),
        frozenset({1}),
    )
    ninf = float("-inf")
    logits = torch.tensor([[0.0, 1.0, 2.0], [0.0, ninf, 5.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    prep = prepare(logits, toy, [0], [1])
    assert bool(run_map(prep).empty)
    assert bool(draw(prep, generator(0)).empty)


def test_map_with_exact_ties_is_a_maximiser_and_deterministic():
    """Ties are allowed any deterministic break (task note): the result is *a* maximiser and repeatable."""
    import torch

    rng = random.Random(77)
    toy = random_toy(rng, V=3, S=4, nfa=True)
    d = distance_to_final(toy)
    end = [s for s in range(4) if d[s] <= 1]
    logits = [[0.0, 0.0, 0.0] for _ in range(4)]
    scores = support_scores(log_softmax64(logits), toy, [0, 1], end)
    assert len(scores) >= 2, "the instance must have several (tied) strings in C"
    best = max(s for _, s in scores.values())
    maximisers = {x for x, (_, s) in scores.items() if abs(s - best) < 1e-12}
    prep = prepare(torch.tensor(logits, dtype=torch.float32), toy, [0, 1], end)
    first = tuple(run_map(prep).tokens.tolist())
    assert first in maximisers
    assert all(tuple(run_map(prep).tokens.tolist()) == first for _ in range(3))


# ---------------------------------------------------------------------------- criterion 6


def test_two_band_map_and_draw():
    """live_simple_106-63-0 regime (fp32): the tail sits at 0 nats, the only path in C ~850 nats below.
    The draw and the MAP both return that path; the MAP score matches the oracle."""
    import torch

    from tests.cd.oracle.viterbi64 import map_score64

    toy = Toy(
        4,
        4,
        ((frozenset({1}), False), (frozenset({2}), False), (frozenset(), True)),
        ((0, 1, 0), (1, 2, 0), (2, 3, 1), (3, 3, 2)),
        frozenset({3}),
    )
    L = 8
    logits = [[0.0, -283.0, -283.0, 0.0] for _ in range(L)]
    prep = prepare(torch.tensor(logits, dtype=torch.float32), toy, [0], [3])
    res = run_map(prep)
    toks = tuple(res.tokens.tolist())
    assert toks[:3] == (1, 1, 2), toks
    assert set(toks[3:]) <= {0, 3}, toks  # the tail's two tied maxima
    import numpy as np

    s_ref = map_score64(np.array(logits), toy, [0], [3], L)
    assert -860 < s_ref < -840
    assert abs(float(res.score) - s_ref) <= _score_tol("float32", s_ref)
    g = generator(1)
    for _ in range(20):
        toks = assert_draw_consistent(draw(prep, g), prep)
        assert tuple(toks[:3]) == (1, 1, 2)


PROD_REGIMES = ["T0.8", "T0.4", "span250", "narrow_neg", "tail", "l_eff"]


@pytest.mark.slow
@pytest.mark.parametrize("bucket", [64, 128])
@pytest.mark.parametrize("regime", PROD_REGIMES)
def test_production_regimes_map_score_and_draw_support(regime, bucket):
    import numpy as np

    from tests.cd.oracle.viterbi64 import map_score64, string_score64
    from tests.cd.unit.test_core_differential import REGIMES, _logits, production_toy

    cdmod("mapdecode")
    cdmod("draw")
    n_states = {64: 50, 128: 100}[bucket]
    rng = random.Random(f"t0066-{regime}-{bucket}")
    toy, n_c = production_toy(rng, n_states)
    logits = _logits(regime, seed=7 * bucket + REGIMES.index(regime), n_c=n_c)
    l_eff = 100 if regime == "l_eff" else 256
    d = distance_to_final(toy)
    start = [0, n_states - 1, n_states - 2] if regime == "tail" else [0]
    end = [s for s in range(n_states) if d[s] <= 40]
    prep = prepare(logits, toy, start, end, l_eff=l_eff, bucket=bucket)

    res = run_map(prep)
    assert not bool(res.empty)
    toks = [int(t) for t in res.tokens.tolist()]
    rep = Replay(toy)
    assert rep.accepts(start, end, toks[:l_eff]), "MAP is not in C"
    lg64 = logits.double().numpy()
    s_ref = map_score64(lg64, toy, start, end, l_eff)
    assert np.isfinite(s_ref)
    tol = _score_tol("float32", s_ref)
    assert abs(float(res.score) - s_ref) <= tol, (float(res.score), s_ref)
    s_str = string_score64(lg64, toks, l_eff)
    assert abs(s_str - s_ref) <= tol, f"returned string scores {s_str}, optimum {s_ref}"

    g = generator(bucket + len(regime))
    for _ in range(20):
        assert_draw_consistent(draw(prep, g), prep, rep)
