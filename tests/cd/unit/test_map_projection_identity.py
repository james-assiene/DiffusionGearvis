"""T-0066 criterion 5 (adversarial): the projection identity the Sudoku end-projection control rests on.

If the per-position argmax x* is in C (under the budget), the MAP is x* **exactly**; if x* ∉ C, the MAP
differs from x* and is in C. The failure this is aimed at: a MAP that returns a *different valid* string
when x* is already valid -- a tie-break or an off-by-one position shift. The hand-built "exactly one b"
language makes every shift of x* valid and scores the neighbours 0.01 nats below it, so a shifted
recovery is seen at the first position, the middle and the last scored position (L_eff < L included).

Rows: hand-built; random tiny DFAs/NFAs (x* planted from the enumerated support, or planted outside
it); the synthetic production automaton at L = 256, V = 262 144; T-0063's compiled Sudoku automata
(``tests/cd/_compile_support``, merged at 7244fef).
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from tests.cd.oracle.enumerate import Toy, distance_to_final, log_softmax64, support_scores
from tests.cd.unit._core_support import dfa_toy, random_logits, random_toy
from tests.cd.unit._draw_support import Replay, cdmod, need_t0063, one_dot_toy, prepare, run_map


def _one_b_toy() -> Toy:
    """V = {0: 'a', 1: 'b'}; C = strings with exactly one b. 0 -a-> 0, 0 -b-> 1, 1 -a-> 1; F = {1}."""
    return Toy(
        2,
        2,
        ((frozenset({0}), False), (frozenset({1}), False)),
        ((0, 0, 0), (0, 1, 1), (1, 1, 0)),
        frozenset({1}),
    )


def _map_tokens(prep):
    res = run_map(prep)
    assert not bool(res.empty)
    return tuple(int(t) for t in res.tokens.tolist())


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize(
    ("L", "l_eff", "j"), [(8, 8, 0), (8, 8, 4), (8, 8, 7), (8, 6, 5), (8, 6, 2)]
)
def test_valid_argmax_is_returned_exactly_one_b(L, l_eff, j, dtype_name):
    """x* = a…a b a…a with b at j < L_eff; b is 0.01 nats below a at j±1 (the tempting shifts)."""
    import torch

    logits = [[0.0, -2.0] for _ in range(L)]
    logits[j] = [0.0, 0.5]
    for k in (j - 1, j + 1):
        if 0 <= k < L:
            logits[k] = [0.0, -0.01]
    x_star = tuple(max(range(2), key=lambda v: row[v]) for row in logits)
    assert x_star[:l_eff].count(1) == 1 and x_star[j] == 1  # x* ∈ C by hand (prefix < L_eff)
    prep = prepare(
        torch.tensor(logits, dtype=getattr(torch, dtype_name)), _one_b_toy(), [0], [1], l_eff=l_eff
    )
    got = _map_tokens(prep)
    assert got[:l_eff] == x_star[:l_eff], f"x* ∈ C but MAP = {got[:l_eff]} (x* = {x_star[:l_eff]})"


def test_invalid_argmax_two_bs_projects_into_c():
    """x* has b at 2 and 5 (∉ C). By hand the MAP keeps the stronger b (at 5: +0.9 vs +0.4)."""
    import torch

    logits = [[0.0, -2.0] for _ in range(8)]
    logits[2] = [0.0, 0.4]
    logits[5] = [0.0, 0.9]
    prep = prepare(torch.tensor(logits, dtype=torch.float32), _one_b_toy(), [0], [1])
    got = _map_tokens(prep)
    assert got != (0, 0, 1, 0, 0, 1, 0, 0)
    assert got == (0, 0, 0, 0, 0, 1, 0, 0)


def test_invalid_argmax_one_dot():
    """The paper's case: per-position argmax ".." = (1, 1) is valid per position, not jointly.
    p_0('.') = 0.8, p_1('.') = 0.7: (1, 0) scores 0.8·0.3 = 0.24 > (0, 1) = 0.2·0.7 = 0.14."""
    import torch

    logits = [[0.0, math.log(4.0)], [0.0, math.log(7.0 / 3.0)]]
    prep = prepare(torch.tensor(logits, dtype=torch.float32), one_dot_toy(), [0], [3])
    got = _map_tokens(prep)
    assert got != (1, 1)
    assert got == (1, 0)


def _random_case(kind: str, seed: int, plant_valid: bool):
    V, L, S = (3, 5, 4) if kind == "dfa" else (4, 4, 5)
    rng = random.Random(f"t0066-proj-{kind}-{seed}-{plant_valid}")
    for _ in range(500):
        toy = random_toy(rng, V=V, S=S, nfa=True) if kind == "nfa" else dfa_toy(rng, V=V, S=S)
        d = distance_to_final(toy)
        start = sorted(rng.sample(range(S), rng.randint(1, S)))
        end = [s for s in range(S) if d[s] <= rng.choice([0, 1, L])]
        if not end:
            continue
        logits = random_logits(rng, L, V, span=6.0)
        support = support_scores(log_softmax64(logits), toy, start, end)
        outside = [x for x in itertools.product(range(V), repeat=L) if x not in support]
        if not support or not outside:
            continue
        x_star = rng.choice(sorted(support)) if plant_valid else rng.choice(outside)
        for i, v in enumerate(x_star):  # plant x* as the strict per-position argmax
            logits[i][v] = max(logits[i]) + rng.uniform(0.5, 2.0)
        support = support_scores(log_softmax64(logits), toy, start, end)
        return toy, start, end, logits, x_star, support
    raise AssertionError(f"no case for {kind} seed {seed}")


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("kind", ["dfa", "nfa"])
def test_random_valid_argmax_is_returned_exactly(kind, seed):
    import torch

    toy, start, end, logits, x_star, support = _random_case(kind, seed, plant_valid=True)
    assert x_star in support
    prep = prepare(torch.tensor(logits, dtype=torch.float32), toy, start, end)
    assert _map_tokens(prep) == x_star


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("kind", ["dfa", "nfa"])
def test_random_invalid_argmax_projects_into_c(kind, seed):
    import torch

    toy, start, end, logits, x_star, support = _random_case(kind, seed, plant_valid=False)
    assert x_star not in support
    prep = prepare(torch.tensor(logits, dtype=torch.float32), toy, start, end)
    got = _map_tokens(prep)
    assert got != x_star
    assert got in support, f"MAP {got} not in C"
    best = max(s for _, s in support.values())
    assert support[got][1] >= best - 1e-4 * max(1.0, abs(best)), "MAP ∈ C but not the maximiser"


def _plant(logits, tokens, bonus: float = 5.0):
    import torch

    idx = torch.arange(len(tokens))
    t = torch.tensor(tokens)
    logits[idx, t] = logits.max(dim=1).values + bonus
    return logits


@pytest.mark.slow
@pytest.mark.parametrize("l_eff", [256, 200])
def test_production_shape_projection_identity(l_eff):
    """Synthetic production automaton (bucket 64), L = 256, V = 262 144: a random accepted prefix planted
    as the argmax is returned exactly; one token overwritten so replay rejects -> MAP ≠ x*, MAP ∈ C."""
    import torch

    from tests.cd.unit.test_core_differential import production_toy

    cdmod("mapdecode")
    L, V = 256, 262_144
    rng = random.Random(f"t0066-proj-prod-{l_eff}")
    toy, _ = production_toy(rng, 50)
    d = distance_to_final(toy)
    end = [s for s in range(50) if d[s] <= 40]
    rep = Replay(toy)
    y = rep.random_accepted([0], end, l_eff, rng)
    assert y is not None
    tail = [rng.randrange(V) for _ in range(L - l_eff)]
    x_star = y + tail
    g = torch.Generator().manual_seed(l_eff)
    base = 30.0 * torch.tanh(torch.randn(L, V, generator=g) * 8.0 / 30.0) / 0.8
    prep = prepare(_plant(base.clone(), x_star), toy, [0], end, l_eff=l_eff, bucket=64)
    got = _map_tokens(prep)
    assert got[:l_eff] == tuple(y), "x* ∈ C but the MAP differs"
    first_diff = next((i for i in range(l_eff) if got[i] != y[i]), None)
    assert first_diff is None, first_diff

    # overwrite one position so that x* leaves C
    bad = list(y)
    for k in range(l_eff):  # the Σ tail and string bodies admit almost anything; scan from 0
        for v in (rng.randrange(V) for _ in range(50)):
            trial = [*bad[:k], v, *bad[k + 1 :]]
            if not rep.accepts([0], end, trial):
                bad = trial
                break
        else:
            continue
        break
    assert not rep.accepts([0], end, bad), "could not build an invalid argmax"
    prep_bad = prepare(_plant(base.clone(), bad + tail), toy, [0], end, l_eff=l_eff, bucket=64)
    got_bad = _map_tokens(prep_bad)
    assert got_bad[:l_eff] != tuple(bad)
    assert rep.accepts([0], end, got_bad[:l_eff]), "MAP not in C"


@pytest.mark.slow
@pytest.mark.parametrize("which", range(3))
def test_sudoku_projection_identity(which):
    """T-0063's compiled Sudoku automaton (test cut): a random accepted canvas planted as the argmax is
    returned exactly; the same canvas with one given overwritten by another digit (a position whose only
    outgoing label is a positive class strictly inside a blank cell's class) leaves C, and the
    MAP differs from it and is in C."""
    import torch

    from tests.cd.unit.test_draw_support_and_rng import start_of, toy_from_automaton

    cs = need_t0063()
    cdmod("mapdecode")
    L, V = 256, 262_144
    _, puzzle, _ = cs.sudoku_cut("test")[which]
    aut = cs.compiled_sudoku(puzzle)
    toy = toy_from_automaton(aut)
    start, end = start_of(aut), sorted(toy.finals)
    rep = Replay(toy)
    rng = random.Random(f"t0066-sudoku-{which}")
    y = rep.random_accepted(start, end, L, rng)
    assert y is not None
    g = torch.Generator().manual_seed(which)
    base = 30.0 * torch.tanh(torch.randn(L, V, generator=g) * 8.0 / 30.0) / 0.8
    prep = prepare(_plant(base.clone(), y), toy, start, end, bucket=64)
    assert _map_tokens(prep) == tuple(y), "valid Sudoku argmax not returned exactly"

    # A given is a forced positive class (the digit's tokenisations, e.g. "2" and " 2") that is a strict
    # subset of a blank cell's class (every digit's tokenisations); overwrite it with a member of the
    # blank class outside the given's class, i.e. another digit.
    pos = {c: st for c, (st, neg) in enumerate(toy.classes) if not neg and st}
    bad = None
    for k in range(L):
        states = rep.run(start, y[:k])
        outs = [(dd, c) for s in states for dd, c in rep.out.get(s, ())]
        if len(outs) != 1 or outs[0][1] not in pos:
            continue
        given = pos[outs[0][1]]
        blanks = [st for st in pos.values() if given < st]
        if not blanks:
            continue
        alts = sorted(set().union(*blanks) - given)
        bad = [*y[:k], alts[0], *y[k + 1 :]]
        break
    assert bad is not None and not rep.accepts(start, end, bad), "no given to overwrite"
    got = _map_tokens(prepare(_plant(base.clone(), bad), toy, start, end, bucket=64))
    assert got != tuple(bad)
    assert rep.accepts(start, end, list(got)), "MAP not in C"
