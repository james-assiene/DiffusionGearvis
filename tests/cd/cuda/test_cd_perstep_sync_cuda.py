"""T-0071a criterion 1 (a1, tier 2): the per-step arms make no host sync, Tap ∘ CD over the image's compiled sampler.

Run inside the pinned container by ``tests/contract/test_vllm_cd_perstep_sync_gpu.py`` through the CD tier-2 runner
(``tests/container_scripts/astra_cd_tier2_runner.py``: exits 3 before pytest unless CUDA on an RTX 5090) with
``ASTRA_CD_DEVICE=cuda ASTRA_CD_STOCK=image ASTRA_CD_REQUIRE_CUDA=1``, repo ``:ro``, ``--network none``,
``--noconftest -o addopts= -p no:cacheprovider``, ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``, packages from the image only;
the host pins the exact pass count (``CD_PERSTEP_SYNC_PASSED``). Without CUDA the rows skip, and fail under
``ASTRA_CD_REQUIRE_CUDA=1``.

Criterion 1: per per-step arm (``perstep_mf``, ``perstep_mar``, ``perstep_mar`` + constrained SC, and T-0076's
``perstep_mar_qargmax`` ± constrained SC) and per bucket
(64, 128): ≥ 12 stock-only warm-up calls with sync-debug **off** (inductor compiles the image's step there; T-0069
O5: CD builds its device constants at wrap or bind time, so the warm-up stays stock-only), then ≥ 50 calls under
``torch.cuda.set_sync_debug_mode("error")`` with one constrained slot, commits and a release included. Pass = 0
raises. Conventions are T-0069 criterion 3's (``tests/cd/cuda/test_cd_tap_composition_cuda.py``), reusing
``tests/cd/_tap_cd_world.py`` (a module, not a conftest):

- the composition is ``make_model_state_cls(make_cd_model_state_cls(FakeModelStateBase), tap=...)``, the chain is
  asserted to be ``TapSampler → ConstrainedDiffusionSampler → DiffusionSampler``;
- binds run under "error" too, and only the model state's ``add_request`` is wrapped (``bind_ctx``: the tap's
  ``on_bind`` and CD's ``_bind``), including one slot-reuse bind. CD's admission -- the sampler's ``add_request``, which
  uploads the automaton and makes one host sync by design -- runs relaxed, outside every strict context (ruling R5);
  that admission sync is part of B's per-request cost in the a2 metric, not a per-step sync (H1). Stock's own request
  bookkeeping and stock's prefill run relaxed too;
- harness snapshots (``.clone().cpu()``, entry phases, the tap drain, building the logits) run **outside** the
  strict context, which wraps only the sampler calls and the releases;
- a request is removed after the commit that holds a stop id (the engine's stop check; T-0069 O6's correction: a
  CD-ended slot left scheduled re-commits its EOS canvas every call) and its slot is refilled by a fresh constrained
  request, so every strict call has one constrained slot and the window holds several releases and binds;
- a raise inside the tap or CD is swallowed by their catch-alls, so the witnesses are ``tap.errors == 0`` and no CD
  error record (``internal`` is a CD-side raise, T-0067 S9). The records must also name the arm and the bucket and
  carry ``mean_accepted_per_step`` (a per-step record, T-0068 P7), so a request quietly served by another arm cannot
  pass.

The language at each bucket is hand-built so the vendored ``n_states_bucket`` is exactly the bucket: state 0 is final
and loops on the live ids 1000..1007; a cycle 0 → 1 → … → k−1 → 0 on ids 2000 + i pads the state count (every
state co-reachable, none merged by minimisation). The logits put EOS (1) among the live ids, so the constrained MAP
commits a stop within a few calls.

Regime: image torch/CUDA, RTX 5090, V 262 144, CL 256, hidden 64, 4 slots, ST 1, 48 steps, Cfg's bound 1.0 and
always-confident threshold (convergence = stability), TF32 "high" set globally, seed 0, one constrained slot + one
unconstrained, tap hz 1e6 on a virtual clock (every call publishes).
Rows: 11 (the sync-debug liveness control + 5 arms × 2 buckets; T-0076 criterion 11 added ``perstep_mar_qargmax``
and its SC twin).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

import pytest
import torch

pytestmark = pytest.mark.gpu

WARM_UP_CALLS = 14  # >= 12 (criterion 1)
STRICT_CALLS = 50  # >= 50 (criterion 1)
BUCKETS = (64, 128)
ARMS = (
    "mf",
    "mar",
    "mar_sc",
    "qa",
    "qa_sc",
)  # tests/cd/unit/_perstep_support.ALL_ARMS keys (T-0076: + qa)
LIVE_IDS = tuple(range(1000, 1008))
EOS = 1
STOPS = frozenset({1, 50, 106})


@pytest.fixture(autouse=True)
def _need_cuda() -> Iterator[None]:
    if not torch.cuda.is_available():
        why = "CUDA rows run inside the pinned container (tests/contract/test_vllm_cd_perstep_sync_gpu.py)"
        if os.environ.get("ASTRA_CD_REQUIRE_CUDA") == "1":
            pytest.fail(f"ASTRA_CD_REQUIRE_CUDA=1 but CUDA is not available: {why}")
        pytest.skip(why)
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    print(f"ASTRA_TIER2_MATMUL_PRECISION {torch.get_float32_matmul_precision()}")
    yield
    torch.set_float32_matmul_precision(prev)


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
    """Stock's ``DiffusionGemmaRequestStates.add_request``/``remove_request`` write a Python scalar into a CUDA row;
    that is stock's, not the plugin's, and the engine does it outside the sampler call."""
    st = w.ms.diffusion_states
    for name in ("add_request", "remove_request"):
        fn = getattr(st, name)

        def relaxed(*a: Any, _fn: Any = fn, **k: Any) -> Any:
            with _relaxed():
                return _fn(*a, **k)

        setattr(st, name, relaxed)


def _grammar_at_bucket(bucket: int) -> Any:
    from tests.cd.unit._sampler_support import Grammar

    for k in range(bucket - 4, bucket + 1):
        edges = [(0, t, 0) for t in LIVE_IDS]
        edges += [(i, 2000 + i, (i + 1) % k) for i in range(k)]
        g = Grammar((k, edges, {0}), f"t71a-loop{k}")
        if g.bucket == bucket:
            return g
    raise AssertionError(f"fixture: no padded loop language has n_states_bucket == {bucket}")


def _logits_fn(period: int = 3) -> Any:
    """``[CL, V]`` on the device (built before the strict context opens): floor −30 except the live ids and EOS,
    uniform in [0, 6), redrawn every ``period`` calls per request (seeded)."""
    from tests.cd.fakes.diffusion_sampler import DEVICE, V

    live = [*LIVE_IDS, EOS]
    seeds = {"u1": 92, "u2": 93}

    def fn(call: int, req: str, n: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(1000 * seeds.get(req, 91) + call // period)
        vals = 6.0 * torch.rand(n, len(live), generator=g)
        x = torch.full((n, V), -30.0, dtype=torch.float32, device=DEVICE)
        x[:, live] = vals.to(DEVICE)
        return x

    return fn


def test_sync_debug_error_really_raises_on_a_device_read() -> None:
    """Liveness: under the strict context a CUDA ``.item()`` raises, so "0 raises" below means something."""
    t = torch.ones(2, device="cuda")
    with pytest.raises(RuntimeError), _strict():
        t[0].item()


@pytest.mark.parametrize("bucket", BUCKETS)
@pytest.mark.parametrize("arm", ARMS)
def test_fifty_perstep_calls_with_a_constrained_slot_under_sync_debug_error(
    arm: str, bucket: int, caplog: Any, capsys: Any
) -> None:
    from tests.cd import _tap_cd_world as tw
    from tests.cd.fakes.diffusion_sampler import DEVICE, Cfg, FakeModelStateBase
    from tests.cd.unit import _perstep_support as ps
    from tests.cd.unit._sampler_support import Records

    assert DEVICE.type == "cuda", "run with ASTRA_CD_DEVICE=cuda"
    mode, _sc = ps.ALL_ARMS[arm]
    g = _grammar_at_bucket(bucket)
    cfg = Cfg(CL=256, max_num_reqs=4, hidden=64)
    fn = _logits_fn()

    # warm-up: stock only, sync-debug off, the measured batch shape (one would-be constrained slot + one other)
    assert torch.cuda.get_sync_debug_mode() == 0
    warm = tw.World(cfg, FakeModelStateBase, seed=0)
    for r in ("c", "u1"):
        warm.add(r)
    warm.prefill()
    for _ in range(WARM_UP_CALLS):
        warm.decode(fn)
    torch.cuda.synchronize()
    del warm

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

    def tag(name: str) -> str:
        return f"t71a-{arm}-{bucket}-{name}"

    def admit(k: int) -> tuple[str, int]:
        name = f"c{k}"
        return name, w.add(name, xargs=ps.ps_xargs(g, tag(name), arm))

    def prefill_relaxed() -> None:
        w.call_ctx = contextlib.nullcontext  # stock's prefill is not under test here
        w.prefill()
        w.call_ctx = _strict

    k = 0
    live, slot_c = admit(k)
    names = [live]
    w.add("u1")
    w.prefill()
    torch.cuda.synchronize()
    commits: dict[str, int] = {live: 0}
    stop_commits = 0
    w.call_ctx = _strict
    for _ in range(STRICT_CALLS):
        s = w.decode(fn)  # snapshot and drain run after the strict context closes
        ids = s.committed(live)
        if ids is not None:
            commits[live] += 1
            if STOPS & set(ids):
                stop_commits += 1
                w.remove(live)  # the release, under "error"
                k += 1
                live, slot_c = admit(
                    k
                )  # ms.add_request under "error"; sampler.add_request relaxed (R5)
                names.append(live)
                commits[live] = 0
                prefill_relaxed()
    w.remove(live)  # the final release flush (tap latch + CD records), under "error"
    assert w.add("u3") == slot_c, "fixture: u3 must reuse the constrained slot"  # slot-reuse bind
    prefill_relaxed()
    for _ in range(3):
        w.decode(fn)
    w.call_ctx = contextlib.nullcontext
    w.bind_ctx = contextlib.nullcontext
    torch.cuda.synchronize()
    w.drain()

    assert w.tap is not None and w.tap.errors == 0, (
        f"{arm} bucket {bucket}: the tap raised under sync-debug 'error': {w.tap.errors}"
    )
    errors = [r for r in records.all() if r.get("error") is not None]
    assert errors == [], (
        f"{arm} bucket {bucket}: CD error records under sync-debug 'error' (internal = a CD-side raise): {errors[:3]}"
    )
    assert sum(commits.values()) >= 2 and stop_commits >= 1, (
        f"fixture: the constrained slots must commit >= 2 canvases, one with a stop, in the strict window "
        f"({commits}, stop commits {stop_commits})"
    )
    for name in names:
        mine = sorted(records.tagged(tag(name)), key=lambda r: r["canvas_idx"])
        frames = sorted(tw.commit_frames(w.frames, name), key=lambda f: f["canvas_idx"])
        assert [list(f["committed_tokens"]) for f in frames] == [
            list(r["committed_ids"]) for r in mine
        ], f"{name}: commit frames != CD records under the CUDA composition"
        assert len(mine) == commits[name], f"{name}: {commits[name]} commits, {len(mine)} records"
        for r in mine:
            assert r["arm"] == mode and r["bucket"] == bucket, (
                f"{name}: record arm {r['arm']} bucket {r['bucket']}, want {mode} at {bucket}"
            )
            assert r["mean_accepted_per_step"] is not None, f"{name}: not a per-step record: {r}"
    assert w.tap.commit_overrun == 0, f"commit_overrun {w.tap.commit_overrun}"
