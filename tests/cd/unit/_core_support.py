"""Shared helpers for the T-0065 core tests, and the API ruling the tests fix.

Ruling T-0065 R1 (test-author, 2026-09-29; to be recorded in docs/tasks/T-0065.md by the owner of that file).
Module root: ``plugin/astra_vllm_plugin/astra_vllm_plugin/cd/`` (imported as ``astra_vllm_plugin.cd.*``).
One constrained slot per call; tensors carry the dtype of ``logits`` (fp32 live, fp64 for tests).

``emission``
  ``build_tables(*, n_states, vocab_size, class_members, class_is_neg, edge_src, edge_dst, edge_class,
  finals, bucket=None, device="cpu") -> CdTables``. ``class_members[c]`` is the *stored* id list: the
  true set when ``class_is_neg[c]`` is False, its complement ``N_c`` when True (SPEC §4.4 layer 2).
  ``bucket`` (default: ``n_states``) pads the state axis; padded states are dead.
  ``CdTables`` exposes ``n_states: int``, ``bucket: int``, ``is_dfa: bool`` and
  ``distance_to_final``: int64 tensor ``[bucket]``, ``d(s)`` by reverse BFS, and ``≥ 2**30`` where F
  is unreachable (padded states included).
  ``class_log_weights(logits[L,V], tables) -> [L,C]``: ``log Σ_{v ∈ S_c} softmax(logits_i)_v`` over the
  true member set, never by subtraction.
  ``log_transfer(logits[L,V], tables, l_eff) -> [L,bucket,bucket]``: ``log M_i`` (edge multiplicity:
  parallel edges add); positions ``i ≥ l_eff`` are the identity (0 on the diagonal). Zero entries
  are the finite sentinel (≤ -1e29), never -inf or NaN.
``semiring``
  ``log_matmul(A[...,n,m], B[...,m,k]) -> [...,n,k]``: log-semiring product, batched over leading dims.
``tree``
  ``up_sweep(logM[L,S,S]) -> LogTree`` retaining the aligned dyadic nodes. ``LogTree.levels`` is a
  sequence of tensors, ``levels[0]`` the leaves ``[L,S,S]``, each next level half as long, the last
  ``[1,S,S]``. Every retained node is normalised (``LogTree`` may carry per-node log-scales).
``marginals``
  ``constrained_marginals(logits[L,V], tables, tree, a_start[bucket] bool, b_final[bucket] bool,
  l_eff) -> CdMarginals`` with ``log_z`` (0-d), ``q [L,V]``, ``entropy [L]`` (nats; rows ``≥ l_eff``
  unspecified), ``log_a``/``log_b`` ``[L+1,bucket]`` absolute log forward/backward vectors (``log_a[i]``
  before position ``i``; zero → sentinel).
``budget``
  ``effective_length(*, canvas_len, valid_canvas_len, max_tokens, emitted) -> int``;
  ``remaining_after(*, max_tokens, emitted, headroom, l_eff) -> int``;
  ``terminal_factor(tables, r_after) -> bool[bucket]``;
  ``initial_infeasibility(tables, a_start, *, max_tokens, emitted=0) -> BudgetInfeasible | None``
  (host check at ``add_request``; ``BudgetInfeasible`` has ``min_distance`` and ``remaining``);
  ``delta_star(tables, states[bucket] bool, committed_ids int64[n], stop_ids) -> DeltaStar`` with
  ``states`` (bool ``[bucket]``) and ``empty`` (truthy iff no state survives).
"""

from __future__ import annotations

import importlib
import math
import os
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.cd.oracle.enumerate import Toy

#: ``ASTRA_CD_PLUGIN_SRC`` points the tests at a copy (mutation harnesses); default is this tree.
PLUGIN_SRC = Path(
    os.environ.get(
        "ASTRA_CD_PLUGIN_SRC",
        str(Path(__file__).resolve().parents[3] / "plugin" / "astra_vllm_plugin"),
    )
)
CD_DIR = PLUGIN_SRC / "astra_vllm_plugin" / "cd"
CORE_MODULES = ("emission", "semiring", "tree", "marginals", "budget")
SENTINEL_MAX = -1e29  # any stored "log 0" must be at or below this and finite


def core(name: str) -> ModuleType:
    """Import ``astra_vllm_plugin.cd.<name>`` inside a test, so absence reads as 'not implemented'."""
    if str(PLUGIN_SRC) not in sys.path:
        sys.path.insert(0, str(PLUGIN_SRC))
    try:
        return importlib.import_module(f"astra_vllm_plugin.cd.{name}")
    except ModuleNotFoundError as exc:
        pytest.fail(f"T-0065 not implemented: astra_vllm_plugin.cd.{name} is missing ({exc})")


def tables_for(toy: Toy, *, bucket: int | None = None) -> Any:
    em = core("emission")
    return em.build_tables(
        n_states=toy.n_states,
        vocab_size=toy.vocab_size,
        class_members=[sorted(stored) for stored, _ in toy.classes],
        class_is_neg=[neg for _, neg in toy.classes],
        edge_src=[e[0] for e in toy.edges],
        edge_dst=[e[1] for e in toy.edges],
        edge_class=[e[2] for e in toy.edges],
        finals=sorted(toy.finals),
        bucket=bucket,
        device="cpu",
    )


def mask(indices: Any, size: int) -> Any:
    import torch

    m = torch.zeros(size, dtype=torch.bool)
    for s in indices:
        m[s] = True
    return m


def run_core(
    logits: Any, toy: Toy, start: Any, end: Any, l_eff: int | None = None, bucket: int | None = None
):
    """The three-stage pipeline of ruling R1: log_transfer -> up_sweep -> constrained_marginals."""
    L = logits.shape[0]
    l_eff = L if l_eff is None else l_eff
    tables = tables_for(toy, bucket=bucket)
    B = tables.bucket
    logM = core("emission").log_transfer(logits, tables, l_eff)
    tree = core("tree").up_sweep(logM)
    res = core("marginals").constrained_marginals(
        logits, tables, tree, mask(start, B), mask(end, B), l_eff
    )
    return res, tables


# ---------------------------------------------------------------------------- instance generators


def random_toy(rng: random.Random, *, V: int, S: int, nfa: bool, neg: bool = True) -> Toy:
    """Random automaton with mixed-polarity classes. ``nfa`` adds parallel edges with overlapping labels."""
    classes: list[tuple[frozenset[int], bool]] = []
    edges: list[tuple[int, int, int]] = []
    for s in range(S):
        n_out = rng.randint(1, 3)
        for _ in range(n_out):
            d = rng.randrange(S)
            k = rng.randint(1, max(1, V - 1))
            members = frozenset(rng.sample(range(V), k))
            if neg and rng.random() < 0.4 and len(members) < V:
                stored = frozenset(range(V)) - members
                classes.append((stored, True))
            else:
                classes.append((members, False))
            edges.append((s, d, len(classes) - 1))
        if nfa:
            # a parallel edge to an existing destination, labels overlapping the first one's
            s0, d0, c0 = next(e for e in edges if e[0] == s)
            mem0 = sorted(Toy(S, V, tuple(classes), tuple(edges), frozenset()).members(c0))
            classes.append((frozenset(mem0[: max(1, len(mem0) // 2 + 1)]), False))
            edges.append((s0, d0, len(classes) - 1))
    finals = frozenset(rng.sample(range(S), rng.randint(1, max(1, S // 2))))
    toy = Toy(S, V, tuple(classes), tuple(edges), finals)
    if nfa:
        assert not toy.is_dfa()
    return toy


def dfa_toy(rng: random.Random, *, V: int, S: int) -> Toy:
    """A random complete-ish DFA: per state, the vocabulary partitioned among destinations."""
    classes: list[tuple[frozenset[int], bool]] = []
    edges: list[tuple[int, int, int]] = []
    for s in range(S):
        toks = list(range(V))
        rng.shuffle(toks)
        n_parts = rng.randint(1, min(3, V))
        cuts = sorted(rng.sample(range(1, V), n_parts - 1)) if n_parts > 1 else []
        parts = [toks[a:b] for a, b in zip([0, *cuts], [*cuts, V], strict=True)]
        dests = (
            rng.sample(range(S), min(len(parts), S))
            if len(parts) <= S
            else [rng.randrange(S) for _ in parts]
        )
        for part, d in zip(parts, dests, strict=True):
            if rng.random() < 0.3:  # drop a part: tokens forbidden from this state
                continue
            members = frozenset(part)
            if rng.random() < 0.4 and len(members) > V // 2:
                classes.append((frozenset(range(V)) - members, True))
            else:
                classes.append((members, False))
            edges.append((s, d, len(classes) - 1))
    finals = frozenset(rng.sample(range(S), rng.randint(1, max(1, S // 2))))
    toy = Toy(S, V, tuple(classes), tuple(edges), finals)
    assert toy.is_dfa()
    return toy


def random_logits(rng: random.Random, L: int, V: int, span: float) -> list[list[float]]:
    return [[rng.uniform(-span, 0.0) for _ in range(V)] for _ in range(L)]


def lse(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(math.fsum(math.exp(x - m) for x in xs))
