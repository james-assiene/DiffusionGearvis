"""T-0068 criterion 6 (negative): T-0067 criteria 3, 4 and 9 -- and 7's ≤ 2-call loud failures -- re-run with the
per-step arms active in the constrained slot. The module extends ``test_cd_mixed_batch_identity.py`` and its
siblings (task file: "extend with per-step arms; name the module at activation") without editing them.

* 3: zero torch ops / syncs outside the inner call when no scheduled decode slot is constrained, with per-step
  requests registered.
* 4: {per-step constrained, unconstrained, unconstrained (top-k / top-p)}: every unconstrained slot's state and
  output rows, and the global RNG after each call, are bit-identical to the stock run with the ``astra_cd_*`` keys
  removed. Compared only over the calls in which both runs schedule the same set of requests: nobody is removed
  here, so that is every call. Non-vacuity: the per-step request runs (no error record) and commits something other
  than stock.
* 9: 200 decode calls over {mf, mar, mar + constrained SC, unconstrained} with no host sync on the sampler thread.
* 7: an injected log Z = −inf at a per-step denoise step and an infeasible budget end loudly within ≤ 2 calls;
  a feasible per-step request never trips.

T-0076 criterion 10: every row runs ``perstep_mar_qargmax`` and its SC twin as well (``ps.ALL_ARMS``), and the
sync row below has a qargmax slot and a qargmax + SC slot beside the T-0068 ones.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tests.cd.fakes.diffusion_sampler import Cfg, World, rows_differ
from tests.cd.unit import _perstep_support as ps
from tests.cd.unit._sampler_support import CHAIN6, G1, G2, Grammar, Records, SyncSpy, brute_map
from tests.cd.unit._scenario import Spec, canvas_logits, run_pair
from tests.cd.unit.test_cd_fast_path import COUNTER, _logits, _measured, _world
from tests.cd.unit.test_cd_loud_failures import _expect_loud

#: T-0076 criterion 10: the qargmax arms (± constrained SC) run every row of this module too
ARMS = sorted(ps.ALL_ARMS)


@pytest.fixture(scope="module")
def g1() -> Grammar:
    return Grammar(G1, "t0068-neg-g1")


# --------------------------------------------------------------------------- T-0067 criterion 3


@pytest.mark.parametrize("arm", ARMS)
def test_fast_path_zero_ops_with_perstep_requests_registered(arm: str, g1: Grammar) -> None:
    w = _world()
    w.add("c", xargs=ps.ps_xargs(g1, f"fast-{arm}", arm))
    w.add("u1")
    w.add("u2")
    snap, ops, syncs = _measured(w, w.prefill)
    assert ops == [] and syncs == [], (ops[:10], syncs)
    for _ in range(4):
        snap, ops, syncs = _measured(w, lambda: w.decode(_logits, only=("u1", "u2")))
        assert ops == [] and syncs == [], (ops[:10], syncs)
        assert snap.returned is COUNTER.returned[0]
    # the per-step request is live: scheduled, it leaves the fast path (both directions of the decision)
    snap, ops, _ = _measured(w, lambda: w.decode(_logits))
    assert ops, "a scheduled per-step slot must take the constrained path"


# --------------------------------------------------------------------------- T-0067 criterion 4


CASES = {
    "stability": (Cfg(CL=4, stability_threshold=1), {}, {}, {}),
    "cap": (Cfg(CL=4, confidence_threshold=0.0, max_denoising_steps=5), {}, {}, {}),
    "overrides": (
        Cfg(CL=4, stability_threshold=1, max_denoising_steps=6),
        {},
        {},
        {"entropy_bound": 0.3, "confidence_threshold": 2.5},
    ),
    "topk_topp_on_unconstrained": (
        Cfg(CL=4, stability_threshold=1, max_denoising_steps=5),
        {"top_k": 3},
        {"top_p": 0.7},
        {},
    ),
}


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("case", sorted(CASES))
def test_unconstrained_slots_bit_identical_with_perstep_arm(
    case: str, arm: str, g1: Grammar, caplog: Any, capsys: Any
) -> None:
    records = Records(caplog, capsys)
    cfg, p1, p2, extra = CASES[case]
    tag = f"mixed-{arm}-{case}"
    specs = [
        Spec("c", xargs=ps.ps_xargs(g1, tag, arm, **extra)),
        Spec("u1", params=p1),
        Spec("u2", params=p2),
    ]
    fn = canvas_logits({"c": 11, "u1": 12, "u2": 13}, period=3)
    a, b = run_pair(cfg, specs, fn, n_calls=14)
    phases: set[str] = set()
    for i, (sa, sb) in enumerate(zip(a, b, strict=True)):
        assert sa.order == sb.order, (
            f"call {i}: scheduled sets differ; the comparison window is closed"
        )
        for r in ("u1", "u2"):
            bad = rows_differ(sa, sb, r)
            assert bad == [], f"{arm} call {i}: unconstrained slot {r} differs from stock in {bad}"
            phases.add("commit" if sa.entered_commit[r] else "denoise")
        assert torch.equal(sa.rng_after, sb.rng_after), (
            f"{arm} call {i}: global RNG differs from stock"
        )
    assert phases == {"commit", "denoise"}, f"fixture: phases covered {phases}"
    errs = [r for r in records.tagged(tag) if r.get("error") is not None]
    assert errs == [], (
        f"non-vacuity: the per-step request ended loudly ({errs[:1]}), so it never ran"
    )
    cd = [s.committed("c") for s in a if s.committed("c") is not None]
    st = [s.committed("c") for s in b if s.committed("c") is not None]
    assert cd and cd != st, "non-vacuity: the per-step slot committed exactly what stock did"


# --------------------------------------------------------------------------- T-0067 criterion 9


def test_two_hundred_perstep_steps_without_a_host_sync(caplog: Any, capsys: Any) -> None:
    records = Records(caplog, capsys)
    g1, g2 = Grammar(G1, "t0068-sync-g1"), Grammar(G2, "t0068-sync-g2")
    w = World(Cfg(CL=4, stability_threshold=1, max_denoising_steps=6), cd=True)
    w.add("mf", xargs=ps.ps_xargs(g1, "sync-mf", "mf"))
    w.add("mar", xargs=ps.ps_xargs(g2, "sync-mar", "mar"))
    w.add("sc", xargs=ps.ps_xargs(g1, "sync-sc", "mar_sc"))
    w.add("qa", xargs=ps.ps_xargs(g2, "sync-qa", "qa"))  # T-0076 criterion 10
    w.add("qa_sc", xargs=ps.ps_xargs(g1, "sync-qa-sc", "qa_sc"))
    w.add("u")
    fn = canvas_logits({"mf": 81, "mar": 82, "sc": 84, "qa": 85, "qa_sc": 86, "u": 83}, period=3)
    spy = SyncSpy()
    w.call_ctx = lambda: spy
    w.prefill()
    commits = 0
    hits: list[tuple[int, list[str]]] = []
    for i in range(200):
        n0 = len(spy.hits)
        s = w.decode(fn)
        if len(spy.hits) > n0:
            hits.append((i, spy.hits[n0:]))
        commits += sum(1 for r in ("mf", "mar", "sc", "qa", "qa_sc") if s.committed(r) is not None)
    assert hits == [], f"host syncs on the sampler thread (call, ops): {hits[:5]}"
    assert commits >= 100, f"fixture: only {commits} constrained commits in 200 calls"
    tags = ("sync-mf", "sync-mar", "sync-sc", "sync-qa", "sync-qa-sc")
    recs = [r for r in records.all() if r.get("record_tag") in tags]
    assert {r["record_tag"] for r in recs} == set(tags), "a per-step slot never committed"
    assert {r["arm"] for r in recs if r["record_tag"].startswith("sync-qa")} == {
        "perstep_mar_qargmax"
    }
    assert [r for r in recs if r.get("error")] == [], "a per-step request ended loudly"
    assert len(recs) >= commits - 10, f"{len(recs)} records for {commits} commits (lag ≤ 2 calls)"
    assert all(r.get("mean_accepted_per_step") is not None for r in recs)


# --------------------------------------------------------------------------- T-0067 criterion 7


@pytest.mark.parametrize("arm", ARMS)
def test_log_z_neg_inf_at_a_perstep_denoise_step_is_loud(
    arm: str, g1: Grammar, caplog: Any, capsys: Any
) -> None:
    """Never confident, cap far away: the 4th denoise call's logits admit only 1007, which G1 never reads
    (Z = 0). A per-step arm needs q at every step, so the failure is detected at that step (not at a
    later convergence) and the request ends on EOS within ≤ 2 calls."""
    cfg = Cfg(CL=4, confidence_threshold=0.0, max_denoising_steps=40)
    base = canvas_logits({"c": 61, "u1": 62, "u2": 63}, period=1)
    trigger = 3
    inject = trigger + 1

    def fn(call: int, req: str, n: int) -> torch.Tensor:
        x = base(call, req, n)
        if req == "c" and call == inject:
            x = torch.full_like(x, float("-inf"))
            x[:, 1007] = 0.0
        return x

    for c in range(1, inject):
        brute_map(g1, base(c, "c", 4), {0}, 4, 10**6)
    tag = f"loud-z0-{arm}"
    _expect_loud(
        cfg,
        Spec("c", xargs=ps.ps_xargs(g1, tag, arm)),
        fn,
        trigger=trigger,
        records=Records(caplog, capsys),
        code="z0",
        tag=tag,
        n_calls=8,
    )


@pytest.mark.parametrize("arm", ARMS)
def test_infeasible_budget_is_loud_for_perstep(arm: str, caplog: Any, capsys: Any) -> None:
    chain6 = Grammar(CHAIN6, "t0068-chain6")
    tag = f"loud-budget-{arm}"
    _expect_loud(
        Cfg(CL=4, stability_threshold=1),
        Spec("c", xargs=ps.ps_xargs(chain6, tag, arm), params={"max_tokens": 3}),
        canvas_logits({"c": 11, "u1": 12, "u2": 13}, period=3),
        trigger=0,
        records=Records(caplog, capsys),
        code="infeasible",
        tag=tag,
    )


@pytest.mark.parametrize("arm", ARMS)
def test_tightest_feasible_budget_does_not_trip_for_perstep(
    arm: str, caplog: Any, capsys: Any
) -> None:
    """The guard's other side: max_tokens = 7 = d(start) is feasible; the chain is committed, no error."""
    chain6 = Grammar(CHAIN6, "t0068-chain6-ok")
    records = Records(caplog, capsys)
    specs = [
        Spec("c", xargs=ps.ps_xargs(chain6, f"ok-{arm}", arm), params={"max_tokens": 7}),
        Spec("u1"),
    ]
    a, _b = run_pair(
        Cfg(CL=4, stability_threshold=1, max_denoising_steps=6),
        specs,
        canvas_logits({"c": 11, "u1": 12}, period=3),
        n_calls=9,
    )
    commits = [s.committed("c") for s in a if s.committed("c") is not None]
    assert commits and commits[0] == [1000, 1001, 1002, 1003], commits
    assert [r for r in records.all() if r.get("error")] == []
