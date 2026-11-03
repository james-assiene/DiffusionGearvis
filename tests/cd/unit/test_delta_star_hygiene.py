"""T-0072 criterion 2: δ* is log-depth and sync-free.

At bucket 128 on CPU ``budget.delta_star`` issues ≤ 160 dispatched ATen ops at L 256 and no more than its own
L 64 count + 40 (``TorchDispatchMode``); zero synchronizing calls (T-0065 criterion 8's spy pair); at L 256 and
bucket 512 no single allocation exceeds 256 MiB (T-0065 criterion 8 / R7). Each row carries a liveness
control: the frozen sequential δ* (``tests/cd/oracle/delta_star_seq.py``) must exceed the op limits, the sync
spy must see a planted ``.item()``, and the allocation spy must see a planted large allocation.

Pre-change counts (measured 2026-09-30 at f9304b1 on the frozen copy, which is the pre-change code; CPU,
torch 2.13.0+cpu, BFCL ``live_simple_2-2-0`` JSON whitespace, bucket 128, stop ids (1, 50, 106), walked ids,
n = 1, deterministic): ``SEQ_OPS`` below. Thresholds: ``OPS_L256_MAX = 160``; L 256 count ≤ L 64 count + 40.
Automaton choice: the criterion names only the bucket; op counts of both implementations do not depend on
the automaton's content at a fixed bucket, and this is the criterion-5 automaton.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.cd.unit._core_support import random_toy, tables_for
from tests.cd.unit._perf_support import (
    STOPS,
    count_ops,
    scan,
    seq,
    start_of,
    tables_from_compiled,
    walk_ids,
)
from tests.cd.unit.test_core_hygiene import _sync_spies

MIB = 1 << 20
OPS_L256_MAX = 160
OPS_GROWTH_MAX = 40  # L 256 count − L 64 count
ALLOC_MAX = 256 * MIB
#: Frozen sequential δ* at bucket 128 (see the module docstring); measured, never edited.
SEQ_OPS = {64: 616, 256: 2344}


def _bfcl128() -> Any:
    from tests.cd.unit._draw_support import need_t0063

    cs = need_t0063()
    a = cs.grammars().compile_bfcl(cs.bfcl_function("live_simple_2-2-0"), whitespace="json")
    assert int(a.n_states_bucket) == 128, "fixture precondition: bucket 128"
    return a


def _args(a: Any, bucket: int, L: int) -> tuple[Any, torch.Tensor, torch.Tensor]:
    tables = tables_from_compiled(a, bucket)
    return tables, start_of(a, bucket), torch.tensor(walk_ids(a, L, 5), dtype=torch.int64)


def _ops(fn: Any, a: Any, L: int, bucket: int = 128) -> int:
    tables, a0, ids = _args(a, bucket, L)
    fn(tables, a0, ids, STOPS)  # warm: the stop-id cache (criterion 3) is filled on the first call
    n, _ = count_ops(lambda: fn(tables, a0, ids, STOPS))
    return n


def test_frozen_sequential_delta_star_exceeds_the_limits() -> None:
    """Liveness control: the pre-change δ* fails both op limits, so the rows below can fail."""
    a = _bfcl128()
    s64, s256 = _ops(seq, a, 64), _ops(seq, a, 256)
    assert (s64, s256) == (SEQ_OPS[64], SEQ_OPS[256]), f"measured {s64}, {s256}: SEQ_OPS is stale"
    assert s256 > OPS_L256_MAX and s256 - s64 > OPS_GROWTH_MAX


def test_delta_star_at_l256_bucket128_is_at_most_160_ops() -> None:
    n = _ops(scan, _bfcl128(), 256)
    assert n <= OPS_L256_MAX, f"δ* issued {n} ATen ops at L 256, bucket 128 (limit {OPS_L256_MAX})"


def test_delta_star_op_count_grows_at_most_40_from_l64_to_l256() -> None:
    a = _bfcl128()
    n64, n256 = _ops(scan, a, 64), _ops(scan, a, 256)
    assert n256 <= n64 + OPS_GROWTH_MAX, (
        f"L 64: {n64} ops, L 256: {n256} ops (> L 64 + {OPS_GROWTH_MAX})"
    )


@pytest.mark.parametrize("L", [1, 65, 256])
def test_delta_star_issues_no_synchronizing_call(L: int, monkeypatch: pytest.MonkeyPatch) -> None:
    a = _bfcl128()
    tables, a0, ids = _args(a, 128, L)
    seen, FnSpy, AtenSpy = _sync_spies()
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *x, **k: seen.append("torch.cuda.synchronize")
    )
    scan(tables, a0, ids, STOPS)  # the first call may build the stop-id cache (criterion 3)
    with FnSpy(), AtenSpy():
        res = scan(tables, a0, ids, STOPS)
    assert not seen, f"synchronizing calls in δ*: {sorted(set(seen))}"
    with FnSpy(), AtenSpy():
        res.empty.item()
    assert seen, "liveness: the spy failed to see .item()"


class _AllocSpy(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.worst = (0, "")

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        for o in out if isinstance(out, (tuple, list)) else [out]:
            if isinstance(o, torch.Tensor):
                n = o.numel() * o.element_size()
                if n > self.worst[0]:
                    self.worst = (n, str(func))
        return out


def test_delta_star_allocation_cap_at_l256_bucket512() -> None:
    import random

    toy = random_toy(random.Random(11), V=64, S=300, nfa=True)
    tables = tables_for(toy, bucket=512)
    a0 = torch.zeros(512, dtype=torch.bool)
    a0[:40] = True
    ids = torch.tensor([random.Random(12).randrange(64) for _ in range(256)], dtype=torch.int64)
    scan(tables, a0, ids, STOPS)
    spy = _AllocSpy()
    with spy:
        scan(tables, a0, ids, STOPS)
    n, op = spy.worst
    assert n <= ALLOC_MAX, f"largest single allocation {n / MIB:.0f} MiB from {op}"
    live = _AllocSpy()
    with live:
        torch.empty(ALLOC_MAX // 4 + 1, dtype=torch.float32)
    assert live.worst[0] > ALLOC_MAX, "liveness: the allocation spy missed a > 256 MiB allocation"
