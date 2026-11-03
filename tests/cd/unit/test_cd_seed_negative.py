"""T-0081 criteria 6 and 7: seeding changes nothing it does not own, and a preempted seeded request ends loudly.

Every row carries a non-vacuity premise (the seeded request in the same engine *was* seeded), so an implementation
that does nothing cannot pass by default. The no-seed golden traces themselves stay pinned by
``tests/cd/unit/test_cd_perstep_golden.py`` (unchanged); the row here re-runs them in a process that has just served
a seeded request, so process-global seed state (a module-level generator or cache) is caught.

- mixed-batch identity: unconstrained slots and the global RNG equal the stock run, call by call, with the seeded
  slot's prefill completing on the prefill-only path or inside a decode call;
- fast path (SD6): no torch op outside the inner call and the inner object returned when no seeded slot completes
  its prefill (an unseeded constrained completion, a seeded mid-chunk, unconstrained decode after a seed);
- no host sync: ``SyncSpy`` plus the D6/O5 detector (Python scalar written into a tensor, tensor built from host data
  in a ``cd/`` frame) around the seeding calls;
- no record-schema change (SD7);
- preemption (SD8): S10 by tombstone and by the prefill hint.

Regime: CPU fp32, vendored v0.29.0 stock step, V = 262 144, CL 8, ``max_denoising_steps`` 6, ST 2, global seed per row.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest
import torch

from tests.cd import _wire_support as ws
from tests.cd.fakes.diffusion_sampler import Cfg, World, rows_differ, stock
from tests.cd.unit import _golden_support as gs
from tests.cd.unit import _perstep_support as ps
from tests.cd.unit import _seed_support as sd
from tests.cd.unit._sampler_support import LIVE, Records, SyncSpy

CL = 8
LOGITS = sd.fixed_logits([{t: 3.0 + 0.1 * k for k, t in enumerate(LIVE)}])
RECORD_FIELDS = frozenset(
    {
        "record_tag",
        "arm",
        "canvas_idx",
        "committed_ids",
        "state_set_after",
        "in_final",
        "steps",
        "bucket",
        "stop_pos",
        "error",
        "mean_accepted_per_step",
        "shadow_argmax_ids",
        "argmax_in_language",
        "map_equals_argmax",
    }
)


def cfg(**kw: Any) -> Cfg:
    base: dict[str, Any] = dict(CL=CL, max_denoising_steps=6, stability_threshold=1)
    base.update(kw)
    return Cfg(**base)


def xargs(tag: str, mode: str = "end_projection") -> dict[str, Any]:
    g = ps.loop_grammar(LIVE, "t0081-neg-loop")
    return ws.raw_xargs(g.payload, mode=mode, self_conditioning="model", record_tag=tag)


def seeded(tag: str, mode: str = "end_projection", **kw: Any) -> dict[str, Any]:
    return sd.with_seed(xargs(tag, mode), sd.DRAFT8, **kw)


# --------------------------------------------------------------------------- mixed-batch identity and global RNG


@pytest.mark.parametrize("mode", ["end_projection", "perstep_mar"])
@pytest.mark.parametrize("where", ["prefill_only", "decode_call"])
def test_unconstrained_slots_and_global_rng_equal_stock_around_a_seeded_slot(
    mode: str, where: str
) -> None:
    def run(cd: bool) -> list[Any]:
        w = World(cfg(), cd=cd, seed=17)
        snaps = []
        w.add("u1", top_k=3)
        w.add("u2")
        if where == "prefill_only":
            w.add("s", xargs=seeded("t0081-neg-mix", mode, start=2) if cd else None)
            snaps.append(w.prefill())
        else:
            snaps.append(w.prefill())
            snaps.append(w.decode(LOGITS))
            w.add("s", xargs=seeded("t0081-neg-mix", mode, start=2) if cd else None)
            snaps.append(
                sd.call(w, [sd.Row("u1"), sd.Row("s", prefill=True), sd.Row("u2")], LOGITS)
            )
        for _ in range(10):
            snaps.append(w.decode(LOGITS))
        return snaps

    ref, got = run(False), run(True)
    seeded_snap = got[0] if where == "prefill_only" else got[2]
    assert sd.row(seeded_snap, "canvas", "s") == list(sd.DRAFT8), "premise: the slot was seeded"
    phases: set[bool] = set()
    for i, (a, b) in enumerate(zip(got, ref, strict=True)):
        assert torch.equal(a.rng_after, b.rng_after), f"call {i}: global RNG differs from stock"
        for r in ("u1", "u2"):
            if r in a.order:
                bad = rows_differ(a, b, r)
                assert bad == [], f"call {i}: {r} differs from stock in {bad}"
                if r in a.entered_commit:
                    phases.add(a.entered_commit[r])
    assert phases == {True, False}, f"fixture: denoise and commit both covered ({phases})"


# --------------------------------------------------------------------------- fast path: zero ops


def _world() -> World:
    return World(
        cfg(), cd=True, seed=0, sampler_cls=sd.counting_sampler_cls(stock.DiffusionSampler)
    )


def _measured(w: World, fn: Any) -> tuple[Any, list[str], list[str]]:
    spy = SyncSpy()
    out: dict[str, Any] = {}

    @contextlib.contextmanager
    def ctx() -> Any:
        with sd.count_ops() as counter, spy:
            yield
        out["ops"] = list(counter.outside)
        out["ret"] = list(counter.returned)

    w.call_ctx = ctx
    try:
        snap = fn()
    finally:
        w.call_ctx = contextlib.nullcontext
    assert len(out["ret"]) == 1, "fixture: one inner call per sampler call"
    out["inner"] = out["ret"][0]
    snap.inner_out = out["inner"]
    return snap, out["ops"], spy.hits


def test_fast_path_holds_whenever_no_seeded_slot_completes_its_prefill() -> None:
    w = _world()
    # the seeding call itself is not on the fast path (liveness of the counter for this hook)
    w.add("s", xargs=seeded("t0081-fp-s"))
    snap, ops, _ = _measured(w, w.prefill)
    assert sd.row(snap, "canvas", "s") == list(sd.DRAFT8), "premise: the slot was seeded"
    assert ops, "the seeding call issued no torch op outside the inner call: the counter is blind"
    w.remove("s")
    # (a) an unseeded constrained request completing on the prefill-only path
    w.add("c", xargs=xargs("t0081-fp-c"))
    snap, ops, syncs = _measured(w, w.prefill)
    assert ops == [] and syncs == [], ("unseeded constrained completion", ops[:10], syncs)
    assert snap.returned is snap.inner_out
    # (b) a seeded request's first chunk (no completion)
    w.add("m", xargs=seeded("t0081-fp-m"))
    snap, ops, syncs = _measured(
        w, lambda: sd.call(w, [sd.Row("m", prefill=True, computed=0, scheduled=2)])
    )
    assert ops == [] and syncs == [], ("seeded mid-chunk", ops[:10], syncs)
    assert snap.returned is snap.inner_out
    # (c) unconstrained decode while the seeded request waits mid-chunk; (d) its completion seeds
    w.add("u")
    sd.call(w, [sd.Row("u", prefill=True)])
    for _ in range(3):
        snap, ops, syncs = _measured(w, lambda: sd.call(w, [sd.Row("u")], LOGITS))
        assert ops == [] and syncs == [], ("unconstrained decode", ops[:10], syncs)
        assert snap.returned is snap.inner_out
    snap = sd.call(w, [sd.Row("m", prefill=True, computed=2, scheduled=3)])
    assert sd.row(snap, "canvas", "m") == list(sd.DRAFT8)


# --------------------------------------------------------------------------- no host sync


@pytest.mark.parametrize("where", ["prefill_only", "decode_call"])
def test_seeding_makes_no_host_sync_and_no_host_write(where: str) -> None:
    live = sd.HostWrites()
    ns: dict[str, Any] = {"torch": torch}
    exec(compile("def poke(t):\n    t[0] = True\n", "/x/astra_vllm_plugin/cd/seed.py", "exec"), ns)
    with live:
        ns["poke"](torch.zeros(2, dtype=torch.bool))
    assert live.hits, "detector dead: a scalar write in a cd/ frame was not seen"

    w = World(cfg(), cd=True, seed=1)
    spy, writes = SyncSpy(), sd.HostWrites()

    @contextlib.contextmanager
    def ctx() -> Any:
        with spy, writes:
            yield

    w.add("u")
    if where == "prefill_only":
        w.add("s", xargs=seeded("t0081-sync", "perstep_mar", noise=0.5, start=3))
        w.call_ctx = ctx
        snap = w.prefill()
    else:
        w.prefill()
        w.add("s", xargs=seeded("t0081-sync", "perstep_mar", noise=0.5, start=3))
        w.call_ctx = ctx
        snap = sd.call(w, [sd.Row("u"), sd.Row("s", prefill=True)], LOGITS)
    w.call_ctx = contextlib.nullcontext
    can = sd.row(snap, "canvas", "s")
    assert sum(a == b for a, b in zip(can, sd.DRAFT8, strict=True)) == 4, (
        f"premise: seeded at noise 0.5: {can}"
    )
    assert sd.row(snap, "step", "s") == [3]
    assert spy.hits == [], f"host syncs in the seeding call: {spy.hits}"
    assert writes.hits == [], (
        f"host data written by CD inside the call (T-0073 D6 / O5): {writes.hits}"
    )


# --------------------------------------------------------------------------- golden traces after a seeded request


@pytest.fixture(scope="module")
def golden() -> dict[str, dict[str, Any]]:
    return torch.load(gs.FIXTURE, weights_only=True)  # type: ignore[no-any-return]


@pytest.mark.parametrize("key", [f"{a}/{s}" for a in gs.GOLDEN_ARMS for s in gs.SCENARIOS])
def test_unseeded_requests_stay_bit_identical_to_the_golden_traces_after_a_seeded_one(
    key: str, golden: dict[str, dict[str, Any]]
) -> None:
    w = World(cfg(), cd=True, seed=0)
    w.add("s", xargs=seeded("t0081-gold-s", "perstep_mar", noise=0.5))
    snap = w.prefill()
    assert sum(a == b for a, b in zip(sd.row(snap, "canvas", "s"), sd.DRAFT8, strict=True)) == 4, (
        "premise"
    )
    w.decode(LOGITS)
    w.remove("s")
    arm, scen = key.split("/")
    got, want = gs.trace(arm, scen), golden[key]
    assert len(got["calls"]) == len(want["calls"])
    for i, (g, wnt) in enumerate(zip(got["calls"], want["calls"], strict=True)):
        for name, v in wnt.items():
            if name == "ops":
                assert g[name] == v, f"{key} call {i}: op digest moved"
            else:
                assert torch.equal(sd_bits(g[name]), sd_bits(v)), f"{key} call {i}: {name} differs"
    assert got["records"] == want["records"]


def sd_bits(t: torch.Tensor) -> torch.Tensor:
    from tests.cd.fakes.diffusion_sampler import bits

    return bits(t)


# --------------------------------------------------------------------------- no record-schema change


def test_record_schema_unchanged_and_seeded_records_carry_no_seed(caplog: Any, capsys: Any) -> None:
    records = Records(caplog, capsys)
    schema = ws.plugin("record_schema")
    assert frozenset(schema.RECORD_FIELDS) == RECORD_FIELDS, sorted(
        set(schema.RECORD_FIELDS) ^ RECORD_FIELDS
    )
    assert frozenset(ws.contracts_cd().CdCanvasRecord.model_fields) == RECORD_FIELDS
    w = World(cfg(), cd=True, seed=2)
    w.add("s", xargs=seeded("t0081-rec", start=1))
    snap = w.prefill()
    assert sd.row(snap, "canvas", "s") == list(sd.DRAFT8), "premise: seeded"
    for _ in range(6):
        if w.decode(LOGITS).committed("s") is not None:
            break
    w.remove("s")
    mine = records.tagged("t0081-rec")
    assert mine and all(r["error"] is None for r in mine), mine
    for r in mine:
        assert set(r) == RECORD_FIELDS, sorted(set(r) ^ RECORD_FIELDS)
        ws.contracts_cd().CdCanvasRecord.model_validate(r)


# --------------------------------------------------------------------------- criterion 7: preemption


@pytest.mark.parametrize("order", ["model_state_first", "sampler_first"])
@pytest.mark.parametrize("how", ["tombstone", "prefill_hint"])
def test_a_preempted_seeded_request_ends_loudly_and_is_never_reseeded(
    how: str, order: str, caplog: Any, capsys: Any
) -> None:
    """Both admission orders the CD sampler documents: with the sampler's ``add_request`` first, the model state's
    bind marks the preemption *after* the seed was attached, so the seed hook itself must skip a request that is no
    longer live (review A F1)."""
    records = Records(caplog, capsys)
    first_kw = {"sampler_first": order == "sampler_first"}
    w = World(cfg(), cd=True, seed=4)
    x = seeded("t0081-pre", noise=0.0, start=2)
    if how == "tombstone":
        w.add("c", xargs=x, **first_kw)
        first = w.prefill()
        assert sd.row(first, "canvas", "c") == list(sd.DRAFT8), (
            "premise: the first admission was seeded"
        )
        w.decode(LOGITS)
        w.decode(LOGITS)
        w.remove("c")  # preempted
        w.add("c", xargs=x, **first_kw)  # resumed: same id, same options
    else:
        w.add("ctl", xargs=seeded("t0081-pre-ctl"), **first_kw)
        w.add(
            "c",
            xargs=x,
            prefill_token_ids=[2] * w.cfg.prompt_len + [1000, 1003],
            **first_kw,
        )
    resumed = w.prefill()
    if how == "prefill_hint":
        assert sd.row(resumed, "canvas", "ctl") == list(sd.DRAFT8), (
            "premise: an ordinary request is seeded"
        )
    can = sd.row(resumed, "canvas", "c")
    assert sum(a == b for a, b in zip(can, sd.DRAFT8, strict=True)) <= 1, (
        f"the resumed request was re-seeded: {can}"
    )
    assert sd.row(resumed, "step", "c") == [0], "the resumed request carries the seed's start step"
    commits = []
    for _ in range(3):
        s = w.decode(LOGITS)
        if s.committed("c") is not None:
            commits.append(s.committed("c"))
    assert commits and set(commits[0]) == {1}, f"not ended with an EOS canvas: {commits}"
    w.remove("c")
    errs = [r["error"] for r in records.tagged("t0081-pre") if r.get("error")]
    assert errs == ["preempted"], errs


# --------------------------------------------------------------------------- S9: the seed write never raises


def test_a_raising_seed_write_ends_the_request_loudly_and_nothing_escapes(
    monkeypatch: pytest.MonkeyPatch, caplog: Any, capsys: Any
) -> None:
    """Ruling S9 around the seed write (review A F2): ``seed.apply`` raising (here an OOM) must not escape the
    sampler call; the request commits an all-EOS canvas and its one error record says ``internal``; the
    unconstrained neighbour keeps decoding."""
    records = Records(caplog, capsys)
    seed_mod = sd.seed_module()
    calls = {"n": 0}

    def boom(*args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        raise torch.OutOfMemoryError("t0081 probe: seed write failed")

    monkeypatch.setattr(seed_mod, "apply", boom)
    w = World(cfg(), cd=True, seed=1)
    w.add("s", xargs=seeded("t0081-s9"))
    w.add("u")
    w.prefill()  # must not raise
    assert calls["n"] == 1, f"cannot inject: seed.apply was called {calls['n']} times"
    commits, u_commits = [], 0
    for _ in range(4):
        snap = w.decode(LOGITS)
        if snap.committed("s") is not None:
            commits.append(snap.committed("s"))
        u_commits += snap.committed("u") is not None
    assert commits and set(commits[0]) == {1}, f"not ended with an all-EOS canvas: {commits}"
    w.remove("s")
    errs = [r["error"] for r in records.tagged("t0081-s9") if r.get("error")]
    assert errs == ["internal"], errs
    assert calls["n"] == 1, "the seed write was retried"
    assert u_commits >= 1, "the unconstrained neighbour stopped decoding"
