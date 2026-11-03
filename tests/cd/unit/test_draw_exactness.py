"""T-0066 criterion 1: the joint draw is distributed exactly as the constrained posterior (slow).

n = 20 000 draws per instance, fp32 (the live dtype), fixed generator seeds. Oracle:
``tests/cd/oracle/enumerate.string_posterior`` -- ``P(x) ∝ N(x)·Π_i p_i(x_i)`` by string enumeration with
integer path multiplicities (SPEC §2.2, §2.6 eq. 8). Positions ``≥ L_eff`` are expected to be drawn
independently from ``p_i`` (criterion 2's second sentence), so the ``L_eff < L`` instance's expected
distribution over whole canvases is ``P(prefix)·Π_{i ≥ L_eff} p_i(x_i)``.

Test: χ² goodness of fit with cells pooled to expected ≥ 5, plus a two-sided exact binomial test per
pooled cell; Bonferroni over every instance and cell at family α = 1e-3 (α / #instances per instance,
split evenly across its χ² and its cells). Any draw outside the support fails outright.

Control (named, must be red): the ∃-form token draw (SPEC §2.6 eq. 8's warning) -- the tokens redrawn
from the implementation's own boundary states with ``p_i(v)·1[∃e: s_i → s_{i+1}, v ∈ label(e)]`` -- is
rejected by the same test on the NFA instance. That proves the harness can see the multiplicity bug.
"""

from __future__ import annotations

import functools
import itertools
import math
import random
from collections import Counter

import pytest

from tests.cd.oracle.enumerate import distance_to_final, log_softmax64, string_posterior
from tests.cd.unit._core_support import dfa_toy, random_logits, random_toy
from tests.cd.unit._draw_support import (
    Replay,
    cdmod,
    cell_pvalues,
    chi2_gof,
    draw,
    exists_control_nfa,
    generator,
    one_dot_toy,
    prepare,
)

pytestmark = pytest.mark.slow

N_DRAWS = 20_000
FAMILY_ALPHA = 1e-3


def _random_instance(kind: str, V: int, L: int, S: int, seed: int, l_eff: int | None = None):
    rng = random.Random(f"t0066-draw-{kind}-{V}-{L}-{S}-{seed}")
    le = L if l_eff is None else l_eff
    for _ in range(500):
        toy = random_toy(rng, V=V, S=S, nfa=True) if kind == "nfa" else dfa_toy(rng, V=V, S=S)
        d = distance_to_final(toy)
        start = sorted(rng.sample(range(S), rng.randint(2, S)))  # non-singleton start set
        R = rng.choice([0, 1, 2])  # budget terminal b = 1[d(s) ≤ R]
        end = [s for s in range(S) if d[s] <= R]
        if not end:
            continue
        logits = random_logits(rng, L, V, span=3.0)
        post = string_posterior(log_softmax64(logits), toy, start, end, le)
        # a non-degenerate posterior: several strings, none dominating
        if len(post) >= 4 and max(post.values()) < 0.6:
            return toy, start, end, logits, le
    raise AssertionError(f"no usable instance for {kind} V{V} L{L} S{S} seed {seed}")


def _hand(name: str):
    if name == "exists-nfa":
        toy = exists_control_nfa()
        # p_0 favours token 1, p_1 favours token 0: the ∃ form and eq. 8 disagree strongly
        logits = [[0.0, 0.7, -0.3], [0.8, 0.2, 0.0], [0.1, 0.0, 0.3]]
        return toy, [0], [2], logits, 3
    if name == "one-dot":
        toy = one_dot_toy()
        # per-position argmax is ".." (1, 1), valid per position but not jointly
        logits = [[0.0, math.log(4.0)], [0.0, math.log(7.0 / 3.0)]]
        return toy, [0], [3], logits, 2
    raise KeyError(name)


INSTANCES = {
    "dfa-V3-L4-S4": lambda: _random_instance("dfa", 3, 4, 4, 0),
    "dfa-V4-L3-S5": lambda: _random_instance("dfa", 4, 3, 5, 1),
    "nfa-V3-L4-S4": lambda: _random_instance("nfa", 3, 4, 4, 2),
    "nfa-V4-L3-S5": lambda: _random_instance("nfa", 4, 3, 5, 3),
    "nfa-V3-L5-Leff3": lambda: _random_instance("nfa", 3, 5, 4, 4, l_eff=3),
    "exists-nfa": lambda: _hand("exists-nfa"),
    "one-dot": lambda: _hand("one-dot"),
}
ALPHA_INSTANCE = FAMILY_ALPHA / len(INSTANCES)


def _expected(toy, start, end, logits, le):
    """Exact distribution over whole canvases (tail positions independent from p)."""
    logp = log_softmax64(logits)
    post = string_posterior(logp, toy, start, end, le)
    L, V = len(logits), toy.vocab_size
    if le == L:
        return post
    out = {}
    for tail in itertools.product(range(V), repeat=L - le):
        pt = math.prod(math.exp(logp[le + j][v]) for j, v in enumerate(tail))
        for x, px in post.items():
            out[x + tail] = px * pt
    return out


@functools.cache
def _draws(name: str):
    """``(expected, counts, states_list, tokens_list)`` for N_DRAWS fp32 draws, cached per instance."""
    import torch

    cdmod("draw")  # fail fast with the T-0066 message before any work
    toy, start, end, logits, le = INSTANCES[name]()
    prep = prepare(torch.tensor(logits, dtype=torch.float32), toy, start, end, l_eff=le)
    g = generator(0x66 + sum(map(ord, name)))
    rep = Replay(toy)
    counts: Counter[tuple[int, ...]] = Counter()
    states_all = []
    for _ in range(N_DRAWS):
        res = draw(prep, g)
        toks = tuple(int(t) for t in res.tokens.tolist())
        states_all.append(tuple(int(s) for s in res.states.tolist()))
        assert not bool(res.empty)
        assert rep.accepts(start, end, toks[:le]), f"draw outside C: {toks}"
        counts[toks] += 1
    return _expected(toy, start, end, logits, le), counts, tuple(states_all), (toy, logits, le)


def _gof_verdict(expected, counts):
    stat, df, p, cells = chi2_gof(counts, expected, N_DRAWS)
    per_test = ALPHA_INSTANCE / (1 + len(cells))
    cell_p = cell_pvalues(cells, N_DRAWS)
    worst = min(cell_p, key=lambda t: t[1])
    return p, per_test, worst, stat, df


@pytest.mark.parametrize("name", list(INSTANCES))
def test_joint_draw_matches_enumerated_posterior(name):
    expected, counts, _, _ = _draws(name)
    outside = [x for x in counts if x not in expected]
    assert not outside, f"draws outside the posterior support: {outside[:5]}"
    p, per_test, worst, stat, df = _gof_verdict(expected, counts)
    assert p >= per_test, (
        f"{name}: χ² = {stat:.1f} on {df} df, p = {p:.3e} < {per_test:.3e} (n = {N_DRAWS})"
    )
    assert worst[1] >= per_test, (
        f"{name}: cell {worst[0]} binomial p = {worst[1]:.3e} < {per_test:.3e} (n = {N_DRAWS})"
    )


def test_one_dot_never_draws_the_per_position_valid_string():
    """Adversarial: ".." / "11" are valid per position, never jointly. (1, 0) has 24/38 of the mass."""
    expected, counts, _, _ = _draws("one-dot")
    assert set(counts) <= {(0, 1), (1, 0)}, dict(counts)
    assert abs(expected[(1, 0)] - 0.8 * 0.3 / (0.8 * 0.3 + 0.2 * 0.7)) < 1e-12  # hand value


def test_exists_form_control_is_rejected_on_the_nfa():
    """Named control: tokens redrawn with the ∃ indicator from the implementation's own states."""
    import torch

    expected, _, states_all, (toy, logits, le) = _draws("exists-nfa")
    labels: dict[tuple[int, int], set[int]] = {}
    for s, d, c in toy.edges:
        labels.setdefault((s, d), set()).update(toy.members(c))
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=-1)
    g = torch.Generator().manual_seed(12345)
    counts: Counter[tuple[int, ...]] = Counter()
    for st in states_all:
        toks = []
        for i in range(le):
            allowed = torch.zeros(toy.vocab_size, dtype=torch.float64)
            allowed[sorted(labels[(st[i], st[i + 1])])] = 1.0  # ∃ form: multiplicity dropped
            toks.append(int(torch.multinomial(probs[i] * allowed, 1, generator=g)))
        counts[tuple(toks)] += 1
    p, per_test, worst, _, _ = _gof_verdict(expected, counts)
    assert p < per_test or worst[1] < per_test, (
        f"the ∃-form control passed the χ² test (p = {p:.3e}); the harness cannot see eq. 8's bug"
    )


# ---------------------------------------------------------------------------- the token draw at large V

V_BLOCKS = 2100  # > two 1024-id blocks and a partial third (52 ids), if the draw works in blocks
N_BLOCK_DRAWS = 20_000
L_BLOCKS, LEFF_BLOCKS = 8, 6  # positions < L_eff constrained (Σ admits all), ≥ L_eff drawn from p


def _block_logits():
    """p over 2100 ids, shaped against a blocked inverse CDF (block edges at 1024, 2048):
    zero-weight ids at 1023, 1024, 2047 (block edges); block 1 mostly zero (3 live ids); a zero tail
    2090..2099 inside the partial last block; block 0 dense with a spread of weights."""
    rng = random.Random("t0066-blocks")
    ninf = float("-inf")
    row = [rng.uniform(-2.0, 0.0) for _ in range(1024)]
    row[1023] = ninf
    blk1 = [ninf] * 1024
    for v, lg in ((1030, 0.5), (1500, 2.0), (2000, 1.0)):
        blk1[v - 1024] = lg
    row += blk1
    row += [rng.uniform(-1.0, 1.0) for _ in range(2048, 2090)] + [ninf] * 10
    assert len(row) == V_BLOCKS
    return row


def test_token_draw_over_many_blocks_matches_p():
    """One-state Σ automaton (every token admitted, multiplicity 1), so each position's token is
    distributed exactly as p_i = softmax(logits_i), independently. 20 000 draws × 8 identical positions
    (160 000 tokens, fp32). χ² over ids (pooled to expected ≥ 5) and over the three 1024-id blocks, and
    a binomial test per pooled cell; Bonferroni at family α = 1e-3 over all of them. Zero-weight ids are
    never drawn. Aimed at a blocked inverse CDF: block masses by max instead of sum, one uniform reused
    for both levels, off-by-one at a block edge or in the partial last block."""
    import torch

    from tests.cd.oracle.enumerate import Toy

    cdmod("draw")
    row = _block_logits()
    toy = Toy(1, V_BLOCKS, ((frozenset(), True),), ((0, 0, 0),), frozenset({0}))
    logits = torch.tensor([row] * L_BLOCKS, dtype=torch.float32)
    prep = prepare(logits, toy, [0], [0], l_eff=LEFF_BLOCKS)
    g = generator(0xB10C)
    counts: Counter[int] = Counter()
    for _ in range(N_BLOCK_DRAWS):
        res = draw(prep, g)
        assert not bool(res.empty)
        counts.update(int(t) for t in res.tokens.tolist())
    n = N_BLOCK_DRAWS * L_BLOCKS

    lp = log_softmax64([row])[0]
    probs = {v: math.exp(x) for v, x in enumerate(lp) if x != float("-inf")}
    zero = [v for v in counts if v not in probs]
    assert not zero, (
        f"zero-weight ids drawn: {sorted(zero)[:10]} (counts {[counts[v] for v in sorted(zero)[:10]]})"
    )

    stat, df, p, cells = chi2_gof(counts, probs, n)
    blk_counts = Counter({b: sum(c for v, c in counts.items() if v // 1024 == b) for b in range(3)})
    blk_probs = {b: math.fsum(q for v, q in probs.items() if v // 1024 == b) for b in range(3)}
    bstat, bdf, bp, bcells = chi2_gof(blk_counts, blk_probs, n)
    cp = cell_pvalues(cells, n)
    per_test = FAMILY_ALPHA / (2 + len(cp))
    assert bp >= per_test, (
        f"block masses wrong: χ² = {bstat:.1f} on {bdf} df, p = {bp:.3e}; observed "
        f"{[o for _, o, _ in bcells]} vs expected {[round(e) for _, _, e in bcells]}"
    )
    assert p >= per_test, f"ids: χ² = {stat:.1f} on {df} df, p = {p:.3e} < {per_test:.3e} (n = {n})"
    worst = min(cp, key=lambda t: t[1])
    assert worst[1] >= per_test, f"cell {worst[0]}: binomial p = {worst[1]:.3e} < {per_test:.3e}"
