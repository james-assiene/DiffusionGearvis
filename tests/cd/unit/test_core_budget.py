"""T-0065 criteria 6–7: budget-aware terminal factor and δ* (SPEC §3.1b, §3.5 trap 1).

Expected values are hand-worked on a chain automaton whose ``d(s)`` is obvious, and the multi-canvas
guarantee is checked by an adversary that picks, inside the support computed by the brute-force
oracle, the canvas that makes the *least* progress. With ``R_after`` one canvas (or one token) too
generous that adversary keeps the run out of F and the run does not terminate.
"""

from __future__ import annotations

import math
import random

import pytest

from tests.cd.oracle.enumerate import Toy, advance, distance_to_final, enumerate_log
from tests.cd.unit._core_support import SENTINEL_MAX, core, dfa_toy, mask, random_toy, tables_for

A, B, STOP = 0, 1, 2  # token ids of the chain automaton
UNREACHABLE = 2**30


def chain_toy(n: int = 6) -> Toy:
    """States 0..n plus a dead state n+1. 'a' advances, 'b' stalls (states < n), F = {n} with the Σ tail.

    ``d(s) = n − s`` for s ≤ n. A premature STOP (from any s < n) enters the dead state n+1, which
    cannot reach F. V = 3.
    """
    classes = (
        (frozenset({A}), False),
        (frozenset({B}), False),
        (frozenset(), True),
        (frozenset({STOP}), False),
    )
    edges = [(s, s + 1, 0) for s in range(n)] + [(s, s, 1) for s in range(n)] + [(n, n, 2)]
    edges += [(s, n + 1, 3) for s in range(n)]  # a premature stop leads to a dead state
    return Toy(n + 2, 3, classes, tuple(edges), frozenset({n}))


def stop_toy() -> Toy:
    """0 -a-> 1; 1 -STOP-> 2 (F); 1 -3-> 2 (a second stop id); 2 has no outgoing edges. V = 4."""
    classes = ((frozenset({A}), False), (frozenset({STOP, 3}), False))
    return Toy(3, 4, classes, ((0, 1, 0), (1, 2, 1)), frozenset({2}))


def as_set(t) -> set[int]:
    return {i for i, x in enumerate(t.tolist()) if x}


# ----------------------------------------------------------------------------- criterion 6


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({"canvas_len": 256, "valid_canvas_len": 256, "max_tokens": 300, "emitted": 256}, 44),
        ({"canvas_len": 256, "valid_canvas_len": 256, "max_tokens": 300, "emitted": 0}, 256),
        ({"canvas_len": 256, "valid_canvas_len": 200, "max_tokens": 1000, "emitted": 0}, 200),
        ({"canvas_len": 256, "valid_canvas_len": 100, "max_tokens": 330, "emitted": 256}, 74),
    ],
)
def test_effective_length(kw, expected):
    assert core("budget").effective_length(**kw) == expected


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({"max_tokens": 300, "emitted": 0, "headroom": 10**6, "l_eff": 256}, 44),
        ({"max_tokens": 300, "emitted": 256, "headroom": 10**6, "l_eff": 44}, 0),
        ({"max_tokens": 1000, "emitted": 0, "headroom": 100, "l_eff": 50}, 50),  # headroom binds
        (
            {"max_tokens": 1000, "emitted": 512, "headroom": 600, "l_eff": 256},
            232,
        ),  # max_tokens binds
    ],
)
def test_remaining_after_subtracts_the_canvas(kw, expected):
    assert core("budget").remaining_after(**kw) == expected


def test_distance_to_final_matches_bfs_oracle():
    rng = random.Random(11)
    for k in range(20):
        toy = random_toy(rng, V=4, S=5, nfa=bool(k % 2)) if k % 3 else dfa_toy(rng, V=4, S=5)
        tables = tables_for(toy, bucket=8)
        got = tables.distance_to_final.tolist()
        want = distance_to_final(toy)
        for s in range(8):
            if s >= toy.n_states or want[s] == math.inf:
                assert got[s] >= UNREACHABLE, (k, s, got[s])
            else:
                assert got[s] == want[s], (k, s, got[s], want[s])


@pytest.mark.parametrize(
    ("r_after", "expected"),
    [
        (0, {6}),  # R = 0 means b = F
        (2, {4, 5, 6}),  # boundary: d = 2 admitted, d = 3 not
        (5, {1, 2, 3, 4, 5, 6}),
        (10**6, {0, 1, 2, 3, 4, 5, 6}),  # large R: every live state, never the dead or padded ones
    ],
)
def test_terminal_factor(r_after, expected):
    tables = tables_for(chain_toy(), bucket=16)
    b = core("budget").terminal_factor(tables, r_after)
    assert b.dtype.is_floating_point is False and str(b.dtype) == "torch.bool"
    assert tuple(b.shape) == (16,)
    assert as_set(b) == expected


def test_max_tokens_300_second_canvas_ends_in_f():
    """CL = 256, max_tokens = 300: canvas 1 has R_after = 44; canvas 2 has L_eff = 44 and b = F."""
    bud = core("budget")
    toy = Toy(
        50,
        3,
        ((frozenset({A}), False), (frozenset(), True)),
        tuple([(s, s + 1, 0) for s in range(49)] + [(49, 49, 1)]),
        frozenset({49}),
    )
    tables = tables_for(toy, bucket=64)
    l0 = bud.effective_length(canvas_len=256, valid_canvas_len=256, max_tokens=300, emitted=0)
    r0 = bud.remaining_after(max_tokens=300, emitted=0, headroom=10**6, l_eff=l0)
    assert (l0, r0) == (256, 44)
    assert as_set(bud.terminal_factor(tables, r0)) == set(range(5, 50))  # d(s) = 49 − s ≤ 44
    l1 = bud.effective_length(canvas_len=256, valid_canvas_len=256, max_tokens=300, emitted=256)
    r1 = bud.remaining_after(max_tokens=300, emitted=256, headroom=10**6, l_eff=l1)
    assert (l1, r1) == (44, 0)
    assert as_set(bud.terminal_factor(tables, r1)) == {49}


def _run_blocks(
    max_tokens: int, cl: int, *, adversarial: bool = True
) -> tuple[set[int], list[str]]:
    """Thread A_k across canvases using the implementation's budget + δ*, with the oracle's support.

    Each canvas is drawn by an adversary from the exact support of the constrained posterior
    (a_start = A_k, b = terminal_factor(R_after)); it picks the canvas whose δ* set is farthest from F.
    """
    bud = core("budget")
    toy = chain_toy()
    tables = tables_for(toy, bucket=8)
    d = distance_to_final(toy)
    states = {0}
    emitted = 0
    log: list[str] = []
    import torch

    while emitted < max_tokens:
        l_eff = bud.effective_length(
            canvas_len=cl, valid_canvas_len=cl, max_tokens=max_tokens, emitted=emitted
        )
        r = bud.remaining_after(max_tokens=max_tokens, emitted=emitted, headroom=10**6, l_eff=l_eff)
        end = as_set(bud.terminal_factor(tables, r))
        ora = enumerate_log([[0.0] * 3 for _ in range(l_eff)], toy, sorted(states), sorted(end))
        assert ora.support, (
            f"Z = 0 at emitted={emitted} (A={states}, R_after={r}): budget too strict"
        )
        a_k = frozenset(states)
        # the adversary: least progress towards F (largest min d over the reached set), then lexicographic
        canvas = max(
            sorted(ora.support), key=lambda x, a_k=a_k: (min(d[s] for s in advance(toy, a_k, x)), x)
        )
        res = bud.delta_star(
            tables, mask(states, 8), torch.tensor(canvas, dtype=torch.int64), [STOP]
        )
        states = as_set(res.states)
        log.append(f"emitted={emitted} l_eff={l_eff} R={r} canvas={canvas} -> {sorted(states)}")
        assert states, "δ* emptied inside the posterior's support"
        emitted += l_eff
    return states, log


@pytest.mark.parametrize("max_tokens", [9, 8, 7])
def test_adversarial_multi_canvas_run_terminates_in_f(max_tokens):
    """d(0) = 6, CL = 3. An R_after one canvas (or one token) too generous lets the adversary stall
    and the run ends outside F — the off-by-one-canvas defect of SPEC §3.5 trap 1 ('did not terminate')."""
    states, log = _run_blocks(max_tokens, 3)
    assert states & {6}, "run did not terminate in F:\n" + "\n".join(log)


def test_tight_budget_is_still_feasible():
    """The other side of the guard: max_tokens = d(0) exactly must leave a non-empty support every canvas."""
    states, log = _run_blocks(6, 3)
    assert states & {6}, "\n".join(log)


def test_initial_infeasibility_is_a_typed_host_signal():
    bud = core("budget")
    tables = tables_for(chain_toy(), bucket=8)
    bad = bud.initial_infeasibility(tables, mask({0}, 8), max_tokens=5)
    assert bad is not None and type(bad).__name__ == "BudgetInfeasible"
    assert (bad.min_distance, bad.remaining) == (6, 5)
    assert bud.initial_infeasibility(tables, mask({0}, 8), max_tokens=6) is None  # boundary: d == R
    assert (
        bud.initial_infeasibility(tables, mask({0, 4}, 8), max_tokens=2) is None
    )  # min over the set
    bad2 = bud.initial_infeasibility(tables, mask({0, 4}, 8), max_tokens=1)
    assert bad2 is not None and bad2.min_distance == 2
    bad3 = bud.initial_infeasibility(tables, mask({0}, 8), max_tokens=10, emitted=5)
    assert bad3 is not None and bad3.remaining == 5


def test_positions_at_or_after_l_eff_are_identity():
    """Criterion 6: positions ≥ L_eff are identity (unscored) transitions, and log Z sees only L_eff."""
    import torch

    from tests.cd.oracle.enumerate import log_softmax64
    from tests.cd.unit._core_support import random_logits, run_core

    toy = chain_toy(3)
    logits = random_logits(random.Random(2), 7, 3, span=20.0)
    tables = tables_for(toy, bucket=8)
    logM = core("emission").log_transfer(torch.tensor(logits, dtype=torch.float64), tables, 4)
    for i in range(4, 7):
        diag = torch.diagonal(logM[i]).tolist()
        assert diag == [0.0] * 8, (i, diag)
        off = logM[i].masked_fill(torch.eye(8, dtype=torch.bool), -2e30)
        assert float(off.max()) <= SENTINEL_MAX, (i, float(off.max()))
        assert bool(torch.isfinite(logM[i]).all())
    ora = enumerate_log(log_softmax64(logits), toy, [0], [3], l_eff=4)
    res, _ = run_core(torch.tensor(logits, dtype=torch.float64), toy, [0], [3], l_eff=4, bucket=8)
    assert abs(float(res.log_z) - ora.log_z) <= 1e-10, (float(res.log_z), ora.log_z)


# ----------------------------------------------------------------------------- criterion 7


def test_delta_star_walks_the_automaton():
    import torch

    tables = tables_for(chain_toy(), bucket=8)
    res = core("budget").delta_star(
        tables, mask({0}, 8), torch.tensor([A, B, A, A, B], dtype=torch.int64), [STOP]
    )
    assert str(res.states.dtype) == "torch.bool" and tuple(res.states.shape) == (8,)
    assert as_set(res.states) == {3}
    assert not bool(res.empty)


def test_delta_star_on_an_nfa_returns_the_set():
    import torch

    rng = random.Random(5)
    for _ in range(10):
        toy = random_toy(rng, V=4, S=5, nfa=True)
        ids = [rng.randrange(4) for _ in range(4)]
        start = {0, 2}
        tables = tables_for(toy, bucket=8)
        res = core("budget").delta_star(
            tables, mask(start, 8), torch.tensor(ids, dtype=torch.int64), []
        )
        assert as_set(res.states) == advance(toy, start, ids)
        assert bool(res.empty) == (not advance(toy, start, ids))


@pytest.mark.parametrize("stop", [STOP, 3])
def test_delta_star_consumes_through_the_first_stop_id_only(stop):
    """Tail after the first stop id is not consumed (here consuming it would empty the set)."""
    import torch

    tables = tables_for(stop_toy(), bucket=4)
    ids = torch.tensor([A, stop, A, A, STOP, 3], dtype=torch.int64)
    res = core("budget").delta_star(tables, mask({0}, 4), ids, [STOP, 3])
    assert as_set(res.states) == {2}  # not {1} (stop excluded) and not ∅ (tail consumed)
    assert not bool(res.empty)


def test_delta_star_empty_is_reported_never_substituted():
    import torch

    tables = tables_for(stop_toy(), bucket=4)
    prev = mask({0}, 4)
    res = core("budget").delta_star(
        tables, prev, torch.tensor([STOP], dtype=torch.int64), [STOP, 3]
    )
    assert as_set(res.states) == set(), "empty result substituted (SPEC §3.1b closure-2 fix)"
    assert bool(res.empty)
    assert prev.tolist() == [True, False, False, False]  # the caller's set is not mutated either
