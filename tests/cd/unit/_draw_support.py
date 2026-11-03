"""Shared helpers for the T-0066 rows (joint draw and constrained MAP), and the API ruling the tests fix.

Ruling T-0066 R1 (test-author, 2026-09-29; to be recorded in docs/tasks/T-0066.md by its owner).
Module root as T-0065 R1 (``astra_vllm_plugin.cd.*``); one constrained slot per call; tensors carry the
dtype of ``logits``. The ``tree`` argument is exactly what T-0065's ``tree.up_sweep(log_transfer(...))``
returns: the **ungauged** pass-1 tree (T-0062 decision 10). Either function may add keyword-only
parameters *with defaults*; the tests never pass them.

``draw``
  ``joint_draw(logits[L,V], tables, tree, a_start[bucket] bool, b_final[bucket] bool, l_eff, *,
  generator: torch.Generator) -> CdDraw`` with
  ``tokens``: int64 ``[L]`` -- positions ``< l_eff`` drawn jointly from the constrained posterior
  ``P(x) ∝ N(x)·Π p_i(x_i)`` (SPEC §2.6: boundary pair, midpoint recursion, eq. 8 multiplicity-weighted
  token draw), positions ``≥ l_eff`` drawn independently from ``p_i`` unconstrained;
  ``states``: int64 ``[L+1]`` -- the drawn boundary states, ``states[i]`` before position ``i``
  (``states[0] ∈ A``, ``states[l_eff] ∈ B``, an edge ``states[i] → states[i+1]`` admits ``tokens[i]``);
  ``empty``: 0-d bool tensor, truthy iff the constrained support is empty (Z = 0). Randomness comes only
  from ``generator``.
``mapdecode``
  ``constrained_map(logits[L,V], tables, a_start[bucket] bool, b_final[bucket] bool, l_eff) -> CdMap``
  with ``tokens``: int64 ``[L]`` (positions ``< l_eff`` are ``argmax_{x ∈ C} Σ_i log p_i(x_i)``; positions
  ``≥ l_eff`` unspecified), ``score``: 0-d, that maximum ``Σ_{i<l_eff} log softmax(logits_i)[x_i]``
  (SPEC §2.7 eq. 9; no multiplicity term), ``empty``: 0-d bool tensor as above. No randomness.
  Whether it shares the sum tree is the implementer's choice; the tests do not pass one.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import random
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from types import ModuleType
from typing import Any

import pytest

from tests.cd.oracle.enumerate import Toy
from tests.cd.unit._core_support import PLUGIN_SRC, core, mask, tables_for

T0066_MODULES = ("draw", "mapdecode")


def cdmod(name: str) -> ModuleType:
    """Import a T-0066 module inside a test, so absence reads as 'T-0066 not implemented'."""
    if str(PLUGIN_SRC) not in sys.path:
        sys.path.insert(0, str(PLUGIN_SRC))
    try:
        return importlib.import_module(f"astra_vllm_plugin.cd.{name}")
    except ModuleNotFoundError as exc:
        pytest.fail(f"T-0066 not implemented: astra_vllm_plugin.cd.{name} is missing ({exc})")


@dataclasses.dataclass
class Prepared:
    logits: Any
    toy: Toy
    start: list[int]
    end: list[int]
    tables: Any
    tree: Any
    a: Any
    b: Any
    l_eff: int


def prepare(
    logits: Any,
    toy: Toy,
    start: Iterable[int],
    end: Iterable[int],
    *,
    l_eff: int | None = None,
    bucket: int | None = None,
    tables: Any = None,
) -> Prepared:
    """T-0065's pipeline up to the retained (ungauged) tree, as the caller will hold it."""
    L = logits.shape[0]
    l_eff = L if l_eff is None else l_eff
    tables = tables_for(toy, bucket=bucket) if tables is None else tables
    B = tables.bucket
    logM = core("emission").log_transfer(logits, tables, l_eff)
    tree = core("tree").up_sweep(logM)
    start, end = sorted(start), sorted(end)
    return Prepared(logits, toy, start, end, tables, tree, mask(start, B), mask(end, B), l_eff)


def draw(prep: Prepared, generator: Any) -> Any:
    return cdmod("draw").joint_draw(
        prep.logits, prep.tables, prep.tree, prep.a, prep.b, prep.l_eff, generator=generator
    )


def run_map(prep: Prepared, logits: Any = None) -> Any:
    lg = prep.logits if logits is None else logits
    return cdmod("mapdecode").constrained_map(lg, prep.tables, prep.a, prep.b, prep.l_eff)


def generator(seed: int) -> Any:
    import torch

    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------------------- host replay


class Replay:
    """Set-of-states replay over a ``Toy`` without materialising complements (fine at V = 262 144)."""

    def __init__(self, toy: Toy) -> None:
        self.toy = toy
        self.out: dict[int, list[tuple[int, int]]] = {}
        for s, d, c in toy.edges:
            self.out.setdefault(s, []).append((d, c))

    def member(self, c: int, v: int) -> bool:
        stored, neg = self.toy.classes[c]
        return (v in stored) != neg

    def run(self, start: Iterable[int], ids: Sequence[int]) -> set[int]:
        cur = set(start)
        for v in ids:
            cur = {d for s in cur for d, c in self.out.get(s, ()) if self.member(c, v)}
            if not cur:
                break
        return cur

    def accepts(self, start: Iterable[int], end: Iterable[int], ids: Sequence[int]) -> bool:
        return bool(self.run(start, ids) & set(end))

    def edge_admits(self, s: int, d: int, v: int) -> bool:
        return any(dd == d and self.member(c, v) for dd, c in self.out.get(s, ()))

    def exact_reach(self, end: Iterable[int], k_max: int) -> list[set[int]]:
        """``reach[k]`` = states with a path of exactly ``k`` tokens (non-empty labels) into ``end``."""
        V = self.toy.vocab_size
        nonempty = {
            c
            for c, (stored, neg) in enumerate(self.toy.classes)
            if (V - len(stored) if neg else len(stored))
        }
        reach = [set(end)]
        for _ in range(k_max):
            prev = reach[-1]
            reach.append({s for s, d, c in self.toy.edges if d in prev and c in nonempty})
        return reach

    def sample_member(self, c: int, rng: random.Random) -> int:
        stored, neg = self.toy.classes[c]
        if not neg:
            return rng.choice(sorted(stored))
        while True:
            v = rng.randrange(self.toy.vocab_size)
            if v not in stored:
                return v

    def random_accepted(
        self, start: Iterable[int], end: Iterable[int], length: int, rng: random.Random
    ) -> list[int] | None:
        """A random string of exactly ``length`` tokens accepted from ``start`` into ``end``."""
        reach = self.exact_reach(end, length)
        cands = sorted(set(start) & reach[length])
        if not cands:
            return None
        s = rng.choice(cands)
        out: list[int] = []
        for k in range(length, 0, -1):
            edges = [(d, c) for d, c in self.out.get(s, ()) if d in reach[k - 1]]
            edges = [(d, c) for d, c in edges if self._nonempty(c)]
            d, c = rng.choice(edges)
            out.append(self.sample_member(c, rng))
            s = d
        return out

    def _nonempty(self, c: int) -> bool:
        stored, neg = self.toy.classes[c]
        return (self.toy.vocab_size - len(stored)) > 0 if neg else len(stored) > 0


def assert_draw_consistent(res: Any, prep: Prepared, rep: Replay | None = None) -> list[int]:
    """Tokens accepted by host replay on ``< l_eff``; states form a valid path that reads them."""
    rep = rep or Replay(prep.toy)
    toks = [int(t) for t in res.tokens.tolist()]
    states = [int(s) for s in res.states.tolist()]
    L = int(prep.logits.shape[0])
    assert len(toks) == L and len(states) == L + 1, (len(toks), len(states))
    assert not bool(res.empty), "empty reported on a feasible instance"
    le = prep.l_eff
    assert rep.accepts(prep.start, prep.end, toks[:le]), (
        f"draw rejected by host replay: {toks[:le]}"
    )
    assert states[0] in prep.start, f"s_0 = {states[0]} not in A = {prep.start}"
    assert states[le] in prep.end, f"s_Leff = {states[le]} not in B = {prep.end}"
    for i in range(le):
        assert rep.edge_admits(states[i], states[i + 1], toks[i]), (
            f"position {i}: no edge {states[i]} -> {states[i + 1]} admits token {toks[i]}"
        )
    return toks


# ---------------------------------------------------------------------------- statistics


def chi2_gof(
    counts: Counter[Any], probs: dict[Any, float], n: int
) -> tuple[float, int, float, list[tuple[Any, int, float]]]:
    """χ² goodness of fit with cells pooled (smallest first) until every expected count is ≥ 5.

    Returns ``(statistic, df, p_value, cells)`` where ``cells`` lists ``(key, observed, expected)``
    after pooling (a pooled cell's key is the tuple of its members). ``counts`` keys outside ``probs``
    are the caller's to reject first (a draw outside the support is not a statistical question).
    """
    from scipy.stats import chi2

    items = sorted(((probs[k] * n, k) for k in probs), key=lambda t: t[0])
    cells: list[tuple[Any, int, float]] = []
    pool_keys: list[Any] = []
    pool_e = 0.0
    pool_o = 0
    for e, k in items:
        if pool_e < 5.0 and (pool_keys or e < 5.0):
            pool_keys.append(k)
            pool_e += e
            pool_o += counts.get(k, 0)
            continue
        cells.append((k, counts.get(k, 0), e))
    if pool_keys:
        if pool_e < 5.0 and cells:
            k, o, e = cells.pop(0)
            pool_keys.append(k)
            pool_e += e
            pool_o += o
        cells.append((tuple(pool_keys), pool_o, pool_e))
    stat = math.fsum((o - e) ** 2 / e for _, o, e in cells)
    df = len(cells) - 1
    p = float(chi2.sf(stat, df)) if df > 0 else 1.0
    return stat, df, p, cells


def cell_pvalues(cells: list[tuple[Any, int, float]], n: int) -> list[tuple[Any, float]]:
    """Two-sided exact binomial p-value per pooled cell (a localized deviation the omnibus χ² dilutes)."""
    from scipy.stats import binomtest

    return [(k, float(binomtest(o, n, e / n).pvalue)) for k, o, e in cells]


# ---------------------------------------------------------------------------- hand-built automata


def one_dot_toy() -> Toy:
    """The paper's "1." / ".1" case. V = {0: '1', 1: '.'}; C = {"1.", ".1"} = {(0, 1), (1, 0)}.

    Every token is valid at every position in *some* string of C, so a per-position mask admits "11"
    and ".." -- valid per position, not jointly.
    """
    return Toy(
        4,
        2,
        ((frozenset({0}), False), (frozenset({1}), False)),
        ((0, 1, 0), (0, 2, 1), (1, 3, 1), (2, 3, 0)),
        frozenset({3}),
    )


def exists_control_nfa() -> Toy:
    """NFA with parallel, overlapping-label edges (SPEC §2.6 eq. 8's counterexample shape).

    0 -{0,1}-> 1 and 0 -{0}-> 1 (token 0 has multiplicity 2 on the pair (0, 1));
    1 -{0,1,2}-> 2 and 1 -{2}-> 2 (token 2 has multiplicity 2 on (1, 2)); 2 -Σ-> 2; F = {2}.
    """
    return Toy(
        3,
        3,
        (
            (frozenset({0, 1}), False),
            (frozenset({0}), False),
            (frozenset(), True),  # Σ
            (frozenset({2}), False),
        ),
        ((0, 1, 0), (0, 1, 1), (1, 2, 2), (1, 2, 3), (2, 2, 2)),
        frozenset({2}),
    )


def need_t0063() -> ModuleType:
    """T-0063's test support (compiled automata). Its absence is an ``AssertionError`` so the rows that
    wait on it run as strict xfail (``xfail(strict=True, raises=AssertionError)`` naming T-0063; only
    NotImplementedError/AssertionError there)."""
    try:
        return importlib.import_module("tests.cd._compile_support")
    except ImportError as exc:
        raise AssertionError(
            f"T-0063 not merged into the integration branch: tests.cd._compile_support ({exc})"
        ) from exc
