"""T-0066 criteria 2–3: draw support at production shape, empty support reported, generator discipline.

Criterion 2 on T-0063's compiled automata (5 Sudoku test-cut puzzles and the first 5 BFCL test-cut records
with |S| < 128, compiled by ``tests/cd/_compile_support`` with ``compile_bfcl``'s default options, i.e.
the pretty whitespace; 10⁴ draws in total). The synthetic production-shape rows
(``tests/cd/unit/test_core_differential.production_toy``: interned classes, narrow negated string-body
classes, back edges, parallel edges, the Σ tail) run now.
"""

from __future__ import annotations

import random

import pytest

from tests.cd.oracle.enumerate import Toy, distance_to_final
from tests.cd.unit._core_support import random_toy
from tests.cd.unit._draw_support import (
    Replay,
    assert_draw_consistent,
    cdmod,
    draw,
    generator,
    need_t0063,
    one_dot_toy,
    prepare,
    run_map,
)

L_PROD, V_PROD = 256, 262_144


def _softcapped(seed: int, L: int, V: int, temp: float = 0.8):
    import torch

    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(L, V, generator=g) * 8.0
    return 30.0 * torch.tanh(raw / 30.0) / temp


# ---------------------------------------------------------------------------- criterion 2


@pytest.mark.slow
@pytest.mark.parametrize(("bucket", "n_states"), [(64, 50), (128, 100)])
@pytest.mark.parametrize("l_eff", [L_PROD, 200])
def test_draw_support_synthetic_production_shape(bucket, n_states, l_eff):
    """L = 256, V = 262 144, softcapped T = 0.8 logits, 100 draws per row: every draw is accepted by host
    replay on positions < L_eff with B = 1[d(s) ≤ 40]; states read the tokens."""
    import torch

    from tests.cd.unit.test_core_differential import production_toy

    cdmod("draw")
    rng = random.Random(f"t0066-support-{bucket}-{l_eff}")
    toy, _ = production_toy(rng, n_states)
    d = distance_to_final(toy)
    end = [s for s in range(n_states) if d[s] <= 40]
    logits = _softcapped(bucket + l_eff, L_PROD, V_PROD)
    prep = prepare(logits, toy, [0], end, l_eff=l_eff, bucket=bucket)
    rep = Replay(toy)
    g = generator(bucket * 1000 + l_eff)
    for _ in range(100):
        assert_draw_consistent(draw(prep, g), prep, rep)
    assert logits.dtype == torch.float32


def _compiled_rows():
    """(label, automaton) over T-0063's compiled Sudoku test cut and small BFCL records."""
    cs = need_t0063()

    rows = [
        (f"sudoku-{rid}", cs.compiled_sudoku(puzzle))
        for rid, puzzle, _ in cs.sudoku_cut("test")[:5]
    ]
    for rid in cs.bfcl_test_cut_ids():
        aut = cs.compiled_bfcl(rid)
        if len(aut.is_final) <= 127:
            rows.append((f"bfcl-{rid}", aut))
        if len(rows) == 10:
            break
    return rows


def toy_from_automaton(aut) -> Toy:
    """T-0063's artifact arrays (edge_src/dst/class, class CSR with polarity, finals) as a ``Toy``."""
    import numpy as np

    t = aut.tables
    classes = []
    for c in range(int(t.n_classes)):
        lo, hi = int(t.sum_indptr[c]), int(t.sum_indptr[c + 1])
        classes.append((frozenset(int(x) for x in t.sum_indices[lo:hi]), bool(t.sum_is_neg[c])))
    edges = tuple(
        (int(s), int(d), int(c))
        for s, d, c in zip(aut.edge_src, aut.edge_dst, aut.edge_class, strict=True)
    )
    finals = frozenset(int(i) for i in np.nonzero(np.asarray(aut.is_final))[0])
    return Toy(len(aut.is_final), 262_144, tuple(classes), edges, finals)


def start_of(aut) -> list[int]:
    import numpy as np

    return [int(i) for i in np.nonzero(np.asarray(aut.start_vector))[0]]


@pytest.mark.slow
def test_draw_support_on_t0063_automata():
    """10⁴ draws in total (1 000 per automaton, 5 Sudoku + 5 BFCL with |S| < 128) at L = 256 with random
    softcapped logits (T = 0.8); B = F; L_eff = 256 for even rows and 200 for odd. All accepted."""
    rows = _compiled_rows()
    cdmod("draw")
    for k, (_label, aut) in enumerate(rows):
        toy = toy_from_automaton(aut)
        bucket = 64 if toy.n_states < 64 else 128
        l_eff = L_PROD if k % 2 == 0 else 200
        prep = prepare(
            _softcapped(k, L_PROD, V_PROD),
            toy,
            start_of(aut),
            sorted(toy.finals),
            l_eff=l_eff,
            bucket=bucket,
        )
        rep = Replay(toy)
        g = generator(k)
        for _ in range(1_000):
            assert_draw_consistent(draw(prep, g), prep, rep)


# ---------------------------------------------------------------------------- empty support


def _empty_cases():
    """(toy, start, end, L): one-dot at L = 3 (C has only length-2 strings), and an unreachable end."""
    iso = Toy(3, 3, ((frozenset({0, 1}), False),), ((0, 1, 0), (1, 1, 0)), frozenset({2}))
    return {"one-dot-L3": (one_dot_toy(), [0], [3], 3), "unreachable-final": (iso, [0], [2], 4)}


@pytest.mark.parametrize("case", ["one-dot-L3", "unreachable-final"])
def test_empty_support_is_reported_not_substituted(case):
    """Z = 0 is reported through ``empty`` (a tensor flag), for the draw and the MAP; the feasible
    neighbour (one-dot at L = 2) reports not-empty (the flag is not stuck on)."""
    import torch

    toy, start, end, L = _empty_cases()[case]
    logits = torch.zeros(L, toy.vocab_size, dtype=torch.float32)
    prep = prepare(logits, toy, start, end)
    res = draw(prep, generator(0))
    assert isinstance(res.empty, torch.Tensor), "empty must be a device flag, not a host bool"
    assert bool(res.empty), "empty support drawn from anyway (substituted)"
    assert bool(run_map(prep).empty), "MAP on an empty language did not report empty"

    ok = prepare(torch.zeros(2, 2, dtype=torch.float32), one_dot_toy(), [0], [3])
    assert not bool(draw(ok, generator(0)).empty)
    assert not bool(run_map(ok).empty)


# ---------------------------------------------------------------------------- criterion 3


def _rng_instance():
    import torch

    rng = random.Random(33)
    toy = random_toy(rng, V=6, S=4, nfa=True)
    d = distance_to_final(toy)
    end = [s for s in range(4) if d[s] <= 2]
    logits = torch.tensor([[rng.uniform(-1, 0) for _ in range(6)] for _ in range(8)])
    return prepare(logits, toy, [0, 1], end, l_eff=6, bucket=8)


def test_draw_leaves_global_rng_untouched_and_consumes_the_generator():
    import numpy as np
    import torch

    prep = _rng_instance()
    g = generator(5)
    g_before = g.get_state().clone()
    before = (
        torch.random.get_rng_state().clone(),
        np.random.get_state()[1].copy(),
        random.getstate(),
    )
    for _ in range(5):
        assert_draw_consistent(draw(prep, g), prep)
    assert torch.equal(before[0], torch.random.get_rng_state()), "torch global RNG consumed"
    assert (before[1] == np.random.get_state()[1]).all(), "numpy global RNG consumed"
    assert before[2] == random.getstate(), "python global RNG consumed"
    assert not torch.equal(g_before, g.get_state()), "the passed generator was never consumed"


def test_same_generator_state_gives_identical_draw():
    import torch

    prep = _rng_instance()
    g = generator(11)
    state = g.get_state().clone()
    first = [draw(prep, g) for _ in range(3)]
    g.set_state(state)
    again = [draw(prep, g) for _ in range(3)]
    fresh = generator(11)
    third = [draw(prep, fresh) for _ in range(3)]
    for a, b, c in zip(first, again, third, strict=True):
        assert torch.equal(a.tokens, b.tokens) and torch.equal(a.states, b.states)
        assert torch.equal(a.tokens, c.tokens) and torch.equal(a.states, c.states)
    # and the draw is not constant: the generator actually drives it (non-vacuity)
    distinct = {tuple(draw(prep, generator(s)).tokens.tolist()) for s in range(30)}
    assert len(distinct) >= 3, f"30 seeds gave {len(distinct)} distinct draws"


RANDOM_OPS = {
    "rand",
    "rand_like",
    "randn",
    "randn_like",
    "randint",
    "randint_like",
    "randperm",
    "multinomial",
    "bernoulli",
    "bernoulli_",
    "normal",
    "normal_",
    "uniform_",
    "exponential_",
    "geometric_",
    "cauchy_",
    "log_normal_",
    "random_",
    "poisson",
}


def random_op_spy():
    from torch.overrides import TorchFunctionMode

    seen: list[tuple[str, object]] = []

    class Spy(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            name = getattr(func, "__name__", str(func))
            if name in RANDOM_OPS:
                seen.append((name, (kwargs or {}).get("generator")))
            return func(*args, **(kwargs or {}))

    return seen, Spy


def test_every_random_op_in_the_draw_uses_the_passed_generator():
    prep = _rng_instance()
    g = generator(3)
    seen, Spy = random_op_spy()
    with Spy():
        draw(prep, g)
    assert seen, "the draw issued no random op the spy can see (or the spy is dead)"
    stray = [(n, gen) for n, gen in seen if gen is not g]
    assert not stray, f"random ops not driven by the passed generator: {stray}"
