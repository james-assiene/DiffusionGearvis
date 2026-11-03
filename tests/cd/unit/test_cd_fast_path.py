"""T-0067 criterion 3: the unconstrained fast path does nothing (primary adversarial row).

With no constrained **decode** slot in the batch, ``__call__`` returns the inner ``SamplerOutput`` object
itself, having issued zero torch operations outside the delegated inner call (``TorchFunctionMode`` and
``TorchDispatchMode``) and zero host syncs; the decision's host cost is p99 ≤ 20 µs (T-0062 decision 8).
"""

from __future__ import annotations

import contextlib
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.cd.fakes.diffusion_sampler import (
    Cfg,
    FakeModelStateBase,
    World,
    make_cd_model_state_cls,
    random_live_logits,
    sampling_params,
)
from tests.cd.unit._sampler_support import G1, LIVE, Grammar, SyncSpy, cd_xargs
from tests.fixtures.vllm_v0_29_0 import diffusion_sampler as stock


class Counter:
    def __init__(self) -> None:
        self.inside = 0
        self.outside: list[str] = []
        self.returned: list[Any] = []


COUNTER = Counter()


class CountingDiffusionSampler(stock.DiffusionSampler):
    """The stock sampler, marking the extent of the delegated call."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        COUNTER.inside += 1
        try:
            out = super().__call__(*args, **kwargs)
        finally:
            COUNTER.inside -= 1
        COUNTER.returned.append(out)
        return out


class _FnMode(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if not COUNTER.inside:
            COUNTER.outside.append(f"fn:{getattr(func, '__name__', func)}")
        return func(*args, **(kwargs or {}))


class _DispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if not COUNTER.inside:
            COUNTER.outside.append(f"aten:{func}")
        return func(*args, **(kwargs or {}))


def _measured(w: World, fn: Any) -> tuple[Any, list[str], list[str]]:
    """Run ``fn`` with the instruments around the sampler call only (``World.call_ctx``)."""
    COUNTER.outside.clear()
    COUNTER.returned.clear()
    spy = SyncSpy()

    @contextlib.contextmanager
    def ctx() -> Any:
        with _DispatchMode(), _FnMode(), spy:
            yield

    w.call_ctx = ctx
    try:
        snap = fn()
    finally:
        w.call_ctx = contextlib.nullcontext
    return snap, list(COUNTER.outside), spy.hits


def _logits(c: int, r: str, n: int) -> torch.Tensor:
    return random_live_logits(100 * c + len(r), LIVE, n)


def test_counter_is_not_vacuous() -> None:
    """Instrument self-test: one stray op outside the inner call is seen, one inside is not."""
    COUNTER.outside.clear()
    with _DispatchMode(), _FnMode():
        torch.zeros(1).add_(1)
    assert COUNTER.outside, "the op counter saw nothing"
    spy = SyncSpy()
    with spy:
        torch.ones(1).item()
        bool(torch.ones(1))
        torch.ones(2).tolist()
        torch.ones(2).to("cpu")  # same-device: not a sync (ruling S8)
    assert "item" in spy.hits and "__bool__" in spy.hits and "tolist" in spy.hits
    assert not [h for h in spy.hits if h.startswith("to(")], "same-device .to must not count (S8)"


def _world(**kw: Any) -> World:
    return World(Cfg(CL=4), cd=True, seed=0, sampler_cls=CountingDiffusionSampler, **kw)


def test_all_unconstrained_batch_issues_zero_ops_and_returns_the_inner_object() -> None:
    w = _world()
    for r in ("u1", "u2", "u3"):
        w.add(r)
    snap, ops, syncs = _measured(w, w.prefill)
    assert ops == [] and syncs == [], (ops[:10], syncs)
    assert snap.returned is COUNTER.returned[-1]
    for _ in range(6):  # denoise, converge and commit steps
        snap, ops, syncs = _measured(w, lambda: w.decode(_logits))
        assert ops == [], f"torch ops outside the inner call on an unconstrained batch: {ops[:10]}"
        assert syncs == [], f"host syncs on an unconstrained batch: {syncs}"
        assert len(COUNTER.returned) == 1 and snap.returned is COUNTER.returned[0], (
            "__call__ must return the inner SamplerOutput object itself"
        )


def test_constrained_request_not_scheduled_keeps_the_fast_path() -> None:
    g = Grammar(G1, "t0067-fast-g1")
    w = _world()
    w.add("c", xargs=cd_xargs(g, "fast-c"))
    for r in ("u1", "u2"):
        w.add(r)
    w.prefill()
    for _ in range(4):
        snap, ops, syncs = _measured(w, lambda: w.decode(_logits, only=("u1", "u2")))
        assert ops == [] and syncs == [], (ops[:10], syncs)
        assert snap.returned is COUNTER.returned[0]


def test_prefill_only_call_with_a_constrained_request_keeps_the_fast_path() -> None:
    g = Grammar(G1, "t0067-fast-g1")
    w = _world()
    w.add("u1")
    w.add("c", xargs=cd_xargs(g, "fast-p"))
    snap, ops, syncs = _measured(w, w.prefill)
    assert ops == [] and syncs == [], (ops[:10], syncs)
    assert snap.returned is COUNTER.returned[0]


class _StubInner:
    """A near-zero-cost inner: isolates the wrapper's own decision cost."""

    def __init__(self) -> None:
        self.sampling_states = SimpleNamespace(
            add_request=lambda *a: None, apply_staged_writes=lambda: None
        )
        self.req_states = SimpleNamespace()
        self.logits_mode = False
        self.diffusion_states = SimpleNamespace()
        self.out = object()

    def add_request(self, req_idx: int, prompt_len: int, sampling_params: Any) -> None:
        pass

    def apply_staged_writes(self) -> None:
        pass

    @property
    def penalties_state(self) -> Any:
        return None

    def __call__(self, logits: Any, input_batch: Any, draft_logits: Any = None) -> Any:
        return self.out


def test_fast_path_decision_p99_under_20us() -> None:
    """Realistic stand-in: 32 decode requests, 4 constrained requests admitted but not scheduled."""

    class Base(FakeModelStateBase):
        def custom_sampler(self, sampler: Any) -> Any:
            inner = _StubInner()
            self.constructed.append(inner)
            return inner, None

    g = Grammar(G1, "t0067-fast-g1")
    cfg = Cfg(CL=256, max_num_reqs=64)
    ms = make_cd_model_state_cls()(Base)(cfg)
    wrapper, _ = ms.custom_sampler(object())
    for i in range(4):
        slot = 60 + i
        ms.add_request(slot, SimpleNamespace(req_id=f"c{i}", prompt_token_ids=[2] * 5))
        wrapper.add_request(slot, 5, sampling_params(extra_args=cd_xargs(g, f"bench-{i}")))
    n = 32
    slots = np.arange(n, dtype=np.int64)
    cu = (np.arange(n + 1) * cfg.CL).astype(np.int32)
    batch = SimpleNamespace(
        num_reqs=n,
        idx_mapping=torch.from_numpy(slots.copy()),
        idx_mapping_np=slots,
        num_draft_tokens=int(cu[-1]),
        cu_num_logits_np=cu,
        query_start_loc_np=cu,
        num_scheduled_tokens=np.full(n, cfg.CL, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(n, dtype=np.int32),
        prefill_len_np=np.full(n, 5, dtype=np.int32),
    )
    logits = torch.empty(0)
    inner = ms.constructed[-1]
    for _ in range(200):
        assert wrapper(logits, batch) is inner.out
    p99s = []
    for _rep in range(5):
        ts = []
        for _ in range(2000):
            t0 = time.perf_counter_ns()
            wrapper(logits, batch)
            ts.append(time.perf_counter_ns() - t0)
        p99s.append(float(np.percentile(ts, 99)) / 1e3)
    best = min(p99s)
    assert best <= 20.0, (
        f"fast-path decision p99 = {best:.1f} µs (best of 5×2000; all {p99s}) > 20 µs"
    )
