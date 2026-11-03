"""T-0069 criterion 3's CUDA row (tier 2): Tap ∘ CD over the image's compiled sampler under sync-debug "error".

Run inside the pinned container by ``tests/contract/test_vllm_cd_wiring_gpu.py`` (``ASTRA_CD_DEVICE=cuda
ASTRA_CD_STOCK=image ASTRA_CD_REQUIRE_CUDA=1``, repo ``:ro``, ``--network none``, ``--noconftest -o addopts= -p
no:cacheprovider``, ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``, packages from the image only); the host pins the pass
count (``CD_TAP_CUDA_PASSED``). On a box without CUDA the rows skip, and fail under ``ASTRA_CD_REQUIRE_CUDA=1``.

The composition is built from the two real factories, ``make_model_state_cls(make_cd_model_state_cls(base),
tap=...)``, over ``tests/cd/fakes/diffusion_sampler.py``'s ``FakeModelStateBase``, whose ``custom_sampler``
constructs the image's ``DiffusionSampler`` (the image's real ``DiffusionGemmaModelState`` needs an engine; the
probe of ``tests/container_scripts/astra_cd_wiring_probe.py`` checks ``model.py``'s composition in the image).

Conventions (T-0011 16(a), T-0067 tier 2): ≥ 12 stock-only warm-up calls with sync-debug **off** first (inductor
compiles the image's sample step there and its compile syncs); harness snapshots (``.clone().cpu()``, entry phases,
the tap drain) run **outside** the strict context, which wraps only the sampler calls and the release; stock's own
request bookkeeping (``DiffusionGemmaRequestStates.add_request``/``remove_request`` write a Python scalar into a CUDA
row) runs relaxed so a raise under "error" can only be the plugin's. Binds run under "error" too (parity with T-0011
16(a)): ``bind_ctx`` wraps only the model state's ``add_request`` -- the tap's ``on_bind`` (a device ``zero_``) and CD's
pure-Python ``_bind`` -- including one slot-reuse bind inside the strict window. CD's automaton upload happens in the
sampler's ``add_request``, which no context wraps (criterion 3 names the calls and the release). A raise inside the tap or CD is swallowed by their
catch-alls, so the witnesses are ``tap.errors == 0`` and no CD error record (an ``internal`` record is a CD-side raise,
T-0067 S9).

Regime: image torch/CUDA, RTX 5090, V = 262 144, CL 4, 8 slots, ST 2, 48 steps, always confident, seed 0, one
constrained slot (G1, end_projection) + two unconstrained, tap hz 1e6 on a virtual clock (every call publishes).
Rows: 2 (the sync-debug liveness control + the 50-call row).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

import pytest
import torch

pytestmark = pytest.mark.gpu

WARM_UP_CALLS = 14  # >= 12 (criterion 3)
STRICT_CALLS = 50  # >= 50 (criterion 3)


@pytest.fixture(autouse=True)
def _need_cuda() -> None:
    if not torch.cuda.is_available():
        why = (
            "CUDA rows run inside the pinned container (tests/contract/test_vllm_cd_wiring_gpu.py)"
        )
        if os.environ.get("ASTRA_CD_REQUIRE_CUDA") == "1":
            pytest.fail(f"ASTRA_CD_REQUIRE_CUDA=1 but CUDA is not available: {why}")
        pytest.skip(why)


@contextlib.contextmanager
def _strict() -> Iterator[None]:
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(0)


@contextlib.contextmanager
def _relaxed() -> Iterator[None]:
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode(0)
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(prev)


def _relax_stock_bookkeeping(w: Any) -> None:
    st = w.ms.diffusion_states
    for name in ("add_request", "remove_request"):
        fn = getattr(st, name)

        def relaxed(*a: Any, _fn: Any = fn, **k: Any) -> Any:
            with _relaxed():
                return _fn(*a, **k)

        setattr(st, name, relaxed)


def test_sync_debug_error_really_raises_on_a_device_read() -> None:
    """Liveness: under the strict context a CUDA ``.item()`` raises, so "0 raises" below means something."""
    t = torch.ones(2, device="cuda")
    with pytest.raises(RuntimeError), _strict():
        t[0].item()


def test_fifty_tap_cd_calls_with_a_constrained_slot_under_sync_debug_error(
    caplog: Any, capsys: Any
) -> None:
    from tests.cd import _tap_cd_world as tw
    from tests.cd.fakes.diffusion_sampler import DEVICE, Cfg, FakeModelStateBase
    from tests.cd.unit._sampler_support import G1, Grammar, Records, cd_xargs
    from tests.cd.unit._scenario import canvas_logits

    assert DEVICE.type == "cuda", "run with ASTRA_CD_DEVICE=cuda"
    cfg = Cfg()
    fn = canvas_logits({"c": 91, "u1": 92, "u2": 93}, period=3)

    # warm-up: stock only, sync-debug off, the measured batch shape
    assert torch.cuda.get_sync_debug_mode() == 0
    warm = tw.World(cfg, FakeModelStateBase, seed=0)
    for r in ("c", "u1", "u2"):
        warm.add(r)
    warm.prefill()
    for _ in range(WARM_UP_CALLS):
        warm.decode(fn)
    torch.cuda.synchronize()

    records = Records(caplog, capsys)
    make = tw.plugin_attr("model_state", "make_model_state_cls", "T-0011")
    make_cd = tw.plugin_attr("cd.model_state", "make_cd_model_state_cls", "T-0067")
    factory = tw.TapFactory()
    w = tw.World(cfg, make(make_cd(FakeModelStateBase), tap=factory), factory=factory, seed=0)
    assert tw.chain_names(w.sampler) == [
        "TapSampler",
        "ConstrainedDiffusionSampler",
        "DiffusionSampler",
    ]
    _relax_stock_bookkeeping(w)
    w.bind_ctx = _strict  # ms.add_request only: tap on_bind + CD _bind
    g = Grammar(G1, "t0069-cuda-g1")
    STOPS = {1, 50, 106}

    def admit(k: int) -> tuple[str, int]:
        name = f"c{k}"
        return name, w.add(name, xargs=cd_xargs(g, f"t69-cuda-{name}"))

    # Engine order: a request whose committed canvas holds a stop id is finished and removed, never scheduled
    # again (tier 2 at 91baefa kept a CD-ended slot scheduled, which re-committed its EOS canvas every call).
    # The constrained slot is refilled with a fresh constrained request, so every strict call has one.
    k = 0
    live, slot_c = admit(k)
    names = [live]
    w.add("u1")
    w.add("u2")
    w.prefill()
    torch.cuda.synchronize()
    commits: dict[str, int] = {live: 0}
    w.call_ctx = _strict
    for _ in range(STRICT_CALLS):
        s = w.decode(fn)  # snapshot and drain run after the strict context closes
        ids = s.committed(live)
        if ids is not None:
            commits[live] += 1
            if STOPS & set(ids):
                w.remove(live)  # release under "error"
                k += 1
                live, slot_c = admit(k)  # bind under "error"
                names.append(live)
                commits[live] = 0
                w.call_ctx = contextlib.nullcontext  # stock's prefill is not under test here
                w.prefill()
                w.call_ctx = _strict
    c = live
    w.remove(c)  # the release flush (tap latch + CD records), under "error" too
    assert w.add("u3") == slot_c, (
        "fixture: u3 must reuse c's slot"
    )  # a slot-reuse bind under "error"
    w.call_ctx = contextlib.nullcontext  # stock's prefill is not under test here
    w.prefill()
    w.call_ctx = _strict
    for _ in range(3):
        w.decode(fn)
    w.call_ctx = contextlib.nullcontext
    w.bind_ctx = contextlib.nullcontext
    torch.cuda.synchronize()
    w.drain()

    assert w.tap is not None and w.tap.errors == 0, (
        f"the tap raised under sync-debug 'error': {w.tap.errors}"
    )
    errors = [r for r in records.all() if r.get("error") is not None]
    assert errors == [], (
        f"CD error records under sync-debug 'error' (internal = a CD-side raise): {errors[:3]}"
    )
    assert sum(commits.values()) >= 2, (
        f"fixture: the constrained slot must commit >= 2 canvases in the strict window ({commits})"
    )
    for name in names:
        mine = sorted(records.tagged(f"t69-cuda-{name}"), key=lambda r: r["canvas_idx"])
        frames = sorted(tw.commit_frames(w.frames, name), key=lambda f: f["canvas_idx"])
        assert [list(f["committed_tokens"]) for f in frames] == [
            list(r["committed_ids"]) for r in mine
        ], f"{name}: commit frames != CD records under the CUDA composition"
        assert len(mine) == commits[name], f"{name}: {commits[name]} commits, {len(mine)} records"
    assert w.tap.commit_overrun == 0, f"commit_overrun {w.tap.commit_overrun}"
