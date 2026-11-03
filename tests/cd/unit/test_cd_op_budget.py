"""T-0072 criterion 5: the whole-call op budget of one ``end_projection`` constrained call.

Regime (task "Regime", CPU): T-0067's CPU stand-in (``tests/cd/fakes/diffusion_sampler.py``, the vendored
v0.29.0 ``DiffusionSampler``), ``cd`` group, torch 2.13.0+cpu, seed 0, n = 1 (op counts are deterministic).
One constrained slot (``end_projection``, T-0063 automaton through T-0064's wire payload) plus one
unconstrained slot per call; CL 64 and CL 256; ``max_tokens`` 4096 so every call has one ``L_eff``
candidate. A call's **CD-added ops** are every ATen op dispatched inside ``sampler(logits, batch)``
(``TorchDispatchMode``) minus those dispatched inside the delegated stock ``DiffusionSampler.__call__``.
The count is the maximum over decode calls 2-5 of a canvas (call 1 is a warm-up: tables and caches are
built there; later calls include the previous call's report flush).

Ruling U1 (test-author, 2026-09-30; measured on branch ``T-0072-cd-perf`` at ``f9304b1``, the pre-change
code, i.e. the merge of T-0067 into the integration branch): the pre-change counts are ``PRE_CHANGE`` below.
The threshold is ``≤ 40 %`` of the pre-change count at CL 256 (criterion 5); it is never edited afterwards.
Automata: BFCL ``live_simple_2-2-0`` under JSON whitespace (bucket 128, the criterion's bucket) and the first
Sudoku of the T-0063 test cut (bucket 64: every T-0063 Sudoku automaton has 32 states, so no Sudoku automaton
has bucket 128; the row is held to the same 40 % rule against its own pre-change count).
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.cd import _wire_support as ws
from tests.cd.fakes.diffusion_sampler import Cfg, World, random_live_logits
from tests.cd.unit._draw_support import need_t0063
from tests.fixtures.vllm_v0_29_0 import diffusion_sampler as stock

BFCL_ID = "live_simple_2-2-0"
CL_VALUES = (64, 256)
#: U1: pre-change CD-added dispatched ops per call (max over decode calls 2-5; all four calls equal), measured
#: 2026-09-30 at f9304b1 (budget.py sha256 f4b59e2a...), torch 2.13.0+cpu, seed 0, n = 1. Never edit.
#: Thresholds: bfcl_json CL 256 <= floor(0.4 * 3733) = 1493; sudoku CL 256 <= floor(0.4 * 2838) = 1135.
PRE_CHANGE: dict[tuple[str, int], int] = {
    ("bfcl_json", 64): 1179,
    ("bfcl_json", 256): 3733,
    ("sudoku", 64): 976,
    ("sudoku", 256): 2838,
}
#: criterion 5: CD-added ops at CL 256 ≤ 40 % of the pre-change count (floor, so "≤ 40 %" is exact)
BUDGET_FRACTION = 0.40
LIVE_IDS = tuple(range(1000, 1008))


class _Count:
    inside = 0
    outside: list[str] = []


class _CountingStock(stock.DiffusionSampler):  # type: ignore[misc]
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        _Count.inside += 1
        try:
            return super().__call__(*args, **kwargs)
        finally:
            _Count.inside -= 1


class _Ops(TorchDispatchMode):
    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if not _Count.inside:
            _Count.outside.append(str(func))
        return func(*args, **(kwargs or {}))


def _automaton(kind: str) -> Any:
    cs = need_t0063()
    if kind == "bfcl_json":
        a = cs.grammars().compile_bfcl(cs.bfcl_function(BFCL_ID), whitespace="json")
        assert int(a.n_states_bucket) == 128, "fixture precondition: bucket 128 (criterion 5)"
    else:
        a = cs.compiled_sudoku(cs.sudoku_cut("test")[0][1])
    return a


def cd_added_ops(kind: str, CL: int) -> tuple[int, list[int]]:
    """(max, per-call list) of CD-added dispatched ops over decode calls 2-5."""
    payload = ws.encode(_automaton(kind))
    xargs = ws.raw_xargs(
        payload, mode="end_projection", self_conditioning="model", record_tag=f"t0072-u1-{kind}"
    )
    w = World(Cfg(CL=CL), cd=True, seed=0, sampler_cls=_CountingStock)
    w.add("c", xargs=xargs)
    w.add("u")
    w.prefill()

    def logits(call: int, req: str, n: int) -> torch.Tensor:
        return random_live_logits(10 * call + len(req), LIVE_IDS, n)

    w.decode(logits)  # warm-up call
    counts = []

    @contextlib.contextmanager
    def ctx() -> Any:
        with _Ops():
            yield

    for _ in range(4):
        _Count.outside = []
        w.call_ctx = ctx
        try:
            w.decode(logits)
        finally:
            w.call_ctx = contextlib.nullcontext
        counts.append(len(_Count.outside))
    return max(counts), counts


def test_counter_is_live_and_excludes_the_inner_call() -> None:
    """Instrument self-test: an unconstrained-only call adds zero ops (T-0067 criterion 3), so the
    subtraction is exact; and the same mode counts a stray op outside the inner call."""
    w = World(Cfg(CL=4), cd=True, seed=0, sampler_cls=_CountingStock)
    w.add("u")
    w.prefill()
    _Count.outside = []
    w.call_ctx = _Ops
    w.decode(lambda c, r, n: random_live_logits(c, LIVE_IDS, n))
    w.call_ctx = contextlib.nullcontext
    assert _Count.outside == []
    with _Ops():
        torch.zeros(1).add_(1)
    assert _Count.outside, "the op counter saw nothing"


@pytest.mark.parametrize("kind", ["bfcl_json", "sudoku"])
def test_whole_call_cd_added_ops_at_cl256_within_40_percent_of_u1(kind: str) -> None:
    pre = PRE_CHANGE[(kind, 256)]
    assert pre > 0, "ruling U1 not recorded: PRE_CHANGE is unset"
    limit = math.floor(BUDGET_FRACTION * pre)
    got, per_call = cd_added_ops(kind, 256)
    assert got <= limit, (
        f"{kind} CL 256 bucket {'128' if kind == 'bfcl_json' else '64'}: {got} CD-added ops per call "
        f"(calls 2-5: {per_call}) > {limit} = 40 % of the pre-change {pre} (ruling U1)"
    )
