"""T-0072 memory rows (review follow-up to criteria 2 and 5; T-0065 criterion 8 / R7's 256 MiB cap).

Two instruments, both ``TorchDispatchMode`` on CPU:
- **largest tensor**: the biggest single op output (``numel · element_size``);
- **peak live bytes**: the maximum, over the call, of the bytes held by storages created inside it and still
  referenced by a live tensor (tracked per storage pointer; each output tensor registers a ``weakref.finalize``
  that drops its reference). Inputs built before the call (tables, logits) are not counted.

Rows:
- ``mapdecode.constrained_map`` at L 256, bucket 128 (127 live states), V 4096: largest tensor ≤ 64 MiB (the
  max-plus chunk); peak live ≤ 64 MiB (one chunk temporary) + 2·(2L−1)·B²·4 B (leaves plus the retained max-plus
  tree, which do not depend on the chunking) + 16 MiB slack. Two live 64 MiB temporaries, or one 256 MiB chunk,
  exceed it.
- ``budget.delta_star`` at S = 511 (bucket 512), L 256 and L 512 (L 512 proves the scan is chunked: an unchunked
  ``[L/2, B, B]`` fp32 level is 256 MiB there): largest tensor ≤ 128 MiB and peak live ≤ 192 MiB (0.75 × the
  256 MiB cap: "well under").
- T-0073 (review): the order-fixed segment layout's **upper-level blow-up** — one negated class whose true set covers
  ``U`` plus many single-token positive classes (``C`` = 4 400 at L 256, 2 200 at L 512; V 16 384; two states).
  Every one-member class gets a padded level-0 block, so an unchunked gather of the logits over the layout is
  ``L × n_blocks × block`` fp32 (275 MiB at L 256 on d913f95). ``class_mass`` and the pipeline
  (``log_transfer`` -> ``up_sweep`` -> ``constrained_marginals``): largest single tensor ≤ 64 MiB (the map row's cap).
Liveness: both trackers see a planted allocation and its release.

Regime: CPU, torch 2.13.0+cpu, fp32, seeds fixed per row.
"""

from __future__ import annotations

import random
import weakref
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.cd.unit._core_support import core, mask, random_toy, tables_for
from tests.cd.unit._perf_support import STOPS

MIB = 1 << 20


class MemTracker(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.largest = (0, "")
        self.live: dict[int, list[int]] = {}  # storage ptr -> [refcount, nbytes]
        self.cur = 0
        self.peak = 0

    def _drop(self, ptr: int) -> None:
        ent = self.live.get(ptr)
        if ent is None:
            return
        ent[0] -= 1
        if ent[0] == 0:
            self.cur -= ent[1]
            del self.live[ptr]

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        for o in out if isinstance(out, (tuple, list)) else [out]:
            if not isinstance(o, torch.Tensor):
                continue
            n = o.numel() * o.element_size()
            if n > self.largest[0]:
                self.largest = (n, str(func))
            st = o.untyped_storage()
            ptr, nbytes = st.data_ptr(), st.nbytes()
            if ptr == 0:
                continue
            ent = self.live.get(ptr)
            if ent is None:
                self.live[ptr] = [1, nbytes]
                self.cur += nbytes
            else:
                ent[0] += 1
            weakref.finalize(o, self._drop, ptr)
            self.peak = max(self.peak, self.cur)
        return out


def test_trackers_are_live() -> None:
    m = MemTracker()
    with m:
        a = torch.empty(8 * MIB // 4, dtype=torch.float32)
        b = a.view(-1, 2)
        del a, b
        c = torch.empty(4 * MIB // 4, dtype=torch.float32)
        del c
    assert m.largest[0] == 8 * MIB, m.largest
    assert m.peak == 8 * MIB, (
        f"peak {m.peak / MIB} MiB: a released tensor was still counted (or a view twice)"
    )
    assert m.cur == 0 and not m.live


def _map_case() -> dict[str, Any]:
    L, V, bucket = 256, 4096, 128
    toy = random_toy(random.Random(31), V=V, S=bucket - 1, nfa=False)
    g = torch.Generator().manual_seed(31)
    return {
        "x": torch.randn(L, V, generator=g) * 4.0,
        "tables": tables_for(toy, bucket=bucket),
        "a": mask({0}, bucket),
        "b": mask(sorted(toy.finals), bucket),
        "L": L,
        "B": bucket,
    }


def test_constrained_map_largest_tensor_at_most_64_mib_at_l256_bucket128() -> None:
    c = _map_case()
    md = core("mapdecode")
    m = MemTracker()
    with m:
        md.constrained_map(c["x"], c["tables"], c["a"], c["b"], c["L"])
    n, op = m.largest
    assert n <= 64 * MIB, f"largest single tensor {n / MIB:.1f} MiB from {op} (cap 64 MiB)"


def test_constrained_map_peak_live_bytes_one_chunk_temporary() -> None:
    c = _map_case()
    L, B = c["L"], c["B"]
    bound = 64 * MIB + 2 * (2 * L - 1) * B * B * 4 + 16 * MIB
    md = core("mapdecode")
    m = MemTracker()
    with m:
        md.constrained_map(c["x"], c["tables"], c["a"], c["b"], L)
    assert m.peak <= bound, f"peak live {m.peak / MIB:.1f} MiB > {bound / MIB:.1f} MiB"


@pytest.mark.parametrize("L", [256, 512])
def test_delta_star_memory_at_s511(L: int) -> None:
    bucket = 512
    toy = random_toy(random.Random(41), V=64, S=bucket - 1, nfa=True)
    tables = tables_for(toy, bucket=bucket)
    a0 = torch.zeros(bucket, dtype=torch.bool)
    a0[:64] = True
    rng = random.Random(42)
    ids = torch.tensor([rng.randrange(64) for _ in range(L)], dtype=torch.int64)
    bud = core("budget")
    bud.delta_star(tables, a0, ids, STOPS)  # warm the stop-id cache
    m = MemTracker()
    with m:
        bud.delta_star(tables, a0, ids, STOPS)
    n, op = m.largest
    assert n <= 128 * MIB, f"L {L}: largest single tensor {n / MIB:.1f} MiB from {op} (cap 128 MiB)"
    assert m.peak <= 192 * MIB, f"L {L}: peak live {m.peak / MIB:.1f} MiB (cap 192 MiB)"


def _many_small_classes(L: int, n_small: int, V: int = 16_384) -> dict[str, Any]:
    """``n_small`` single-token positive classes ({1}, {2}, ...) and one negated class storing nothing (its true
    set is all of V, so its members inside U are all of U); state 0 -> 1 on every small class, 1 -> 1 on the
    negated one, F = {1}."""
    members: list[list[int]] = [[v] for v in range(1, n_small + 1)] + [[]]
    neg = [False] * n_small + [True]
    em = core("emission")
    tables = em.build_tables(
        n_states=2,
        vocab_size=V,
        class_members=members,
        class_is_neg=neg,
        edge_src=[0] * n_small + [1],
        edge_dst=[1] * (n_small + 1),
        edge_class=list(range(n_small + 1)),
        finals=[1],
    )
    g = torch.Generator().manual_seed(0)
    return {"x": torch.randn(L, V, generator=g) * 4.0, "tables": tables, "L": L}


@pytest.mark.parametrize(("L", "n_small"), [(256, 4400), (512, 2200)])
def test_many_single_token_classes_largest_tensor_at_most_64_mib(L: int, n_small: int) -> None:
    c = _many_small_classes(L, n_small)
    x, t = c["x"], c["tables"]
    em, tr, mg = core("emission"), core("tree"), core("marginals")
    a, b = mask({0}, t.bucket), mask({1}, t.bucket)
    m = MemTracker()
    with m:
        em.class_mass(x, t)
        tree = tr.up_sweep(em.log_transfer(x, t, L))
        mg.constrained_marginals(x, t, tree, a, b, L)
    n, op = m.largest
    assert n <= 64 * MIB, (
        f"L {L}, {n_small} single-token classes: largest single tensor {n / MIB:.1f} MiB from {op} (cap 64 MiB)"
    )
