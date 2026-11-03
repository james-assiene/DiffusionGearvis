"""Shared helpers for the T-0072 rows (δ* scan, stop-id cache, pass-through keywords, op budgets).

API assumed (T-0072 criteria 1-4; test-author, 2026-09-30):
  ``budget.delta_star(tables, states, committed_ids, stop_ids) -> DeltaStar`` keeps T-0065 R1's signature.
  New keyword-only parameters, each defaulting to today's behaviour; the value passed is *the object the
  function would have computed* (implementation note 3):
    ``emission.log_transfer(logits, tables, l_eff, *, class_mass=None)`` -- ``emission.class_mass(logits, tables)``
    ``marginals.constrained_marginals(..., l_eff, *, class_mass=None)`` -- the same ``ClassMass``
    ``draw.joint_draw(..., l_eff, *, generator, class_log_weights=None)`` -- ``emission.class_log_weights(logits, tables)``
    ``mapdecode.constrained_map(..., l_eff, *, class_max=None)`` -- the ``(max, argmax)`` pair
      ``mapdecode.class_max`` returns for these logits (the rows capture it from a call without the keyword).
  A call with a keyword the function does not accept yet fails the row as "T-0072 not implemented".
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.cd.oracle import delta_star_seq
from tests.cd.unit._core_support import core

STOPS = (1, 50, 106)  # T-0063's augmentation (EOS = 1)


def new_kw(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with T-0072's new keywords; an unknown keyword reads as 'not implemented'."""
    try:
        return fn(*args, **kwargs)
    except TypeError as exc:
        msg = str(exc)
        if "unexpected keyword" in msg or "got an unexpected" in msg:
            pytest.fail(
                f"T-0072 not implemented: {getattr(fn, '__qualname__', fn)} does not accept "
                f"{sorted(kwargs)} ({msg})"
            )
        raise


def seq(tables: Any, states: Any, ids: Any, stops: Sequence[int]) -> Any:
    return delta_star_seq.delta_star(tables, states, ids, stops)


def scan(tables: Any, states: Any, ids: Any, stops: Sequence[int]) -> Any:
    return core("budget").delta_star(tables, states, ids, stops)


def identical(got: Any, want: Any) -> list[str]:
    """Criterion 1's comparison: ``torch.equal`` plus dtype, shape and device, on ``states`` and ``empty``."""
    bad = []
    for name in ("states", "empty"):
        g, w = torch.as_tensor(getattr(got, name)), getattr(want, name)
        if g.dtype != w.dtype or g.shape != w.shape or g.device != w.device:
            bad.append(
                f"{name}: {g.dtype}{tuple(g.shape)}@{g.device} != {w.dtype}{tuple(w.shape)}@{w.device}"
            )
        elif not torch.equal(g, w):
            bad.append(f"{name}: values differ")
    return bad


# --------------------------------------------------------------------------- compiled automata -> tables


def tables_from_compiled(a: Any, bucket: int, device: str = "cpu") -> Any:
    """``build_tables`` from a T-0063 compiled automaton at ``bucket`` (as ``requests._tables_for`` does)."""
    t = a.tables
    indptr, indices = np.asarray(t.sum_indptr), np.asarray(t.sum_indices)
    n_classes = len(indptr) - 1
    members = [indices[int(indptr[c]) : int(indptr[c + 1])] for c in range(n_classes)]
    n = int(a.n_states)
    return core("emission").build_tables(
        n_states=n,
        vocab_size=int(a.vocab_size),
        class_members=members,
        class_is_neg=[bool(x) for x in np.asarray(t.sum_is_neg)],
        edge_src=np.asarray(a.edge_src).astype(np.int64),
        edge_dst=np.asarray(a.edge_dst).astype(np.int64),
        edge_class=np.asarray(a.edge_class).astype(np.int64),
        finals=np.flatnonzero(np.asarray(a.is_final)[:n]).astype(np.int64),
        bucket=bucket,
        device=device,
    )


def start_of(a: Any, bucket: int) -> torch.Tensor:
    sv = np.asarray(a.start_vector, dtype=bool)
    out = torch.zeros(bucket, dtype=torch.bool)
    out[: len(sv)] = torch.from_numpy(sv[:bucket].copy())
    return out


def walk_ids(a: Any, L: int, seed: int, *, avoid: Sequence[int] = STOPS) -> list[int]:
    """``L`` ids following one path of ``a`` from its start while one exists (positive classes only,
    avoiding ``avoid``), then random ids. Keeps the set alive for a while so the rows are not all-dead."""
    rng = random.Random(seed)
    t = a.tables
    indptr, indices, neg = (
        np.asarray(t.sum_indptr),
        np.asarray(t.sum_indices),
        np.asarray(t.sum_is_neg),
    )
    src, dst, cls = np.asarray(a.edge_src), np.asarray(a.edge_dst), np.asarray(a.edge_class)
    starts = np.flatnonzero(np.asarray(a.start_vector))
    s = int(starts[0]) if len(starts) else 0
    V = int(a.vocab_size)
    out: list[int] = []
    for _ in range(L):
        opts = []
        for e in np.flatnonzero(src == s):
            c = int(cls[e])
            if neg[c]:
                stored = {int(x) for x in indices[int(indptr[c]) : int(indptr[c + 1])]}
                pick = [
                    v
                    for v in (rng.randrange(V) for _ in range(8))
                    if v not in stored and v not in avoid
                ]
                if pick:
                    opts.append((pick[:1], int(dst[e])))
                continue
            mem = [
                int(x) for x in indices[int(indptr[c]) : int(indptr[c + 1])] if int(x) not in avoid
            ]
            if mem:
                opts.append((mem, int(dst[e])))
        if opts:
            mem, d = rng.choice(opts)
            out.append(rng.choice(mem))
            s = d
        else:
            out.append(rng.randrange(V))
    return out


# --------------------------------------------------------------------------- instruments


class OpCount(TorchDispatchMode):
    """Every ATen op dispatched while active."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: list[str] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def count_ops(fn: Callable[[], Any]) -> tuple[int, Any]:
    with OpCount() as m:
        out = fn()
    return len(m.ops), out


class LVOps(TorchDispatchMode):
    """ATen ops that read or write a tensor with an ``[L, V]`` (or larger ``L·V``-element) shape."""

    def __init__(self, L: int, V: int) -> None:
        super().__init__()
        self.L, self.V = L, V
        self.hits: list[str] = []

    def _lv(self, x: Any) -> bool:
        return (
            isinstance(x, torch.Tensor)
            and x.dim() >= 2
            and x.shape[-1] == self.V
            and x.numel() >= self.L * self.V
        )

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        flat = list(args) + list((kwargs or {}).values())
        flat += list(out) if isinstance(out, (tuple, list)) else [out]
        if any(self._lv(x) for x in flat):
            self.hits.append(str(func))
        return out
