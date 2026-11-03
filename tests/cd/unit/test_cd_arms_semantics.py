"""T-0068 criteria 1, 5 and 7: the per-step arms' semantics, convergence sanity, and their records.

Everything runs through T-0067's CPU stand-in (vendored v0.29.0 stock step, ``tests/cd/fakes/diffusion_sampler.py``)
with one constrained slot plus one unconstrained companion. Expected values are hand-built: product languages
(position ``i`` admits exactly ``S_i``) make ``q_i`` = ``p_i`` restricted to ``S_i`` exactly, computed in fp64 here;
the MAP comes from ``_sampler_support.brute_map`` (fp64 enumeration). Each row states its expected accept mask as
a literal and asserts its premise (entropies, margins to the bound) so a tie or rounding cannot decide it.
See ``_perstep_support`` for the API assumed (P1-P8). V = 262 144, CL = 2-8, n and seeds per row.

Observables: after a denoise call that does not converge, ``canvas[slot]`` is the fed-back canvas
``where(accept, x̂, uniform ids)``. An accepted position holds a token of its constrained support; a rejected one
holds a uniform id, which is outside the live and stop ids with probability 1 − 11/262 144.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pytest
import torch

from tests.cd.fakes.diffusion_sampler import Cfg, Snap, World, random_live_logits
from tests.cd.unit import _perstep_support as ps
from tests.cd.unit._sampler_support import (
    G1,
    LIVE,
    Grammar,
    Records,
    argmax_ids,
    brute_map,
    validate,
)
from tests.cd.unit._scenario import Spec, commit_calls, convergence_calls, run_pair

BIG = 10**6
ARMS = sorted(ps.ARMS)
LogitsFn = Callable[[int, str, int], torch.Tensor]


def t1(**kw: Any) -> Cfg:
    """T = 1 exactly (t_min = t_max = 1), stability_threshold 1 (ST = 2), a cap far away unless given."""
    base: dict[str, Any] = dict(
        CL=4, t_min=1.0, t_max=1.0, stability_threshold=1, max_denoising_steps=1000
    )
    base.update(kw)
    return Cfg(**base)


def run_cd(
    cfg: Cfg,
    xargs: Mapping[str, Any],
    fn: LogitsFn,
    n_calls: int,
    *,
    max_tokens: int = 4096,
) -> tuple[World, list[Snap]]:
    w = World(cfg, cd=True, seed=0)
    w.add("c", xargs=xargs, max_tokens=max_tokens)
    w.add("u")
    w.prefill()
    return w, [w.decode(fn) for _ in range(n_calls)]


def const(x_c: torch.Tensor) -> LogitsFn:
    def fn(call: int, req: str, n: int) -> torch.Tensor:
        if req == "c":
            return x_c[:n].clone()
        return random_live_logits(500 + call, LIVE, n)

    return fn


def canvas(s: Snap) -> list[int]:
    return [int(t) for t in s.row("canvas", "c")]


def assert_fed_back(
    s: Snap, supports: Sequence[Sequence[int]], accepted: Sequence[bool], what: str
) -> None:
    got = canvas(s)
    for i, (sup, acc) in enumerate(zip(supports, accepted, strict=True)):
        if acc:
            assert got[i] in set(sup), (
                f"{what}: position {i} is accepted, so it must hold the joint draw x̂_{i} ∈ {sorted(sup)}; "
                f"canvas = {got}"
            )
        else:
            assert ps.renoised(got[i]), (
                f"{what}: position {i} is rejected, so it must hold a uniform renoise id; canvas = {got}"
            )
    slot = s.slots["c"]
    CL = len(got)
    assert s.draft_tokens[slot, :CL].tolist() == got, f"{what}: draft_tokens[slot, :CL] != canvas"


# --------------------------------------------------------------------------- fixture F-A (forced positions)

#: S_0 excludes 1000, which the model prefers at position 0; positions 1-3 are forced (H(q) = 0) under flat
#: logits over 8 / 6 / 4 live ids (H(p) = ln 8, ln 6, ln 4).
FA_SETS = [list(range(1001, 1008)), [1001], [1002], [1003]]
FA_ROWS = [
    {1000: 12.0, 1001: 1.0, **ps.flat(range(1002, 1008))},
    ps.flat(range(1000, 1008)),
    ps.flat(range(1000, 1006)),
    ps.flat(range(1000, 1004)),
]
FA_MASK = {"mf": [True, False, False, True], "mar": [True] * 4, "mar_sc": [True] * 4}
FA_MASK_BOUND2 = {"mf": [True, False, True, True], "mar": [True] * 4, "mar_sc": [True] * 4}


@pytest.fixture(scope="module")
def fa() -> Grammar:
    return ps.product_grammar(FA_SETS, "t0068-fa")


def _fa_entropies(x: torch.Tensor) -> tuple[list[float], list[float]]:
    hp = [ps.entropy64(ps.p64(x[i])) for i in range(4)]
    hq = [ps.entropy64(ps.q_product64(x[i], FA_SETS[i])) for i in range(4)]
    return hp, hq


def test_fixture_fa_premise() -> None:
    x = ps.logits_rows(FA_ROWS)
    hp, hq = _fa_entropies(x)
    assert hp[0] < 1e-3 and abs(hp[1] - math.log(8)) < 1e-3 and abs(hp[3] - math.log(4)) < 1e-3
    assert hq[1] == hq[2] == hq[3] == 0.0 and hq[0] > 1.0
    for bound, masks in ((0.5, FA_MASK), (2.0, FA_MASK_BOUND2)):
        assert ps.stock_rule(hp, bound) == masks["mf"] and ps.margin(hp, bound) > 0.05
        assert ps.stock_rule(hq, bound) == masks["mar"] and ps.margin(hq, bound) > 0.05


@pytest.mark.parametrize("arm", ARMS)
def test_fed_back_canvas_and_accept_signal(arm: str, fa: Grammar) -> None:
    """Accepted positions hold x̂ (never the out-of-language 1000 the model prefers); rejected ones a uniform
    id. Mf accepts on H(p) → {0, 3}; Mar on H(q) → all four (the forced positions have H(q) = 0)."""
    x = ps.logits_rows(FA_ROWS)
    xargs = ps.ps_xargs(fa, f"fa-{arm}", arm, confidence_threshold=ps.NEVER_CONFIDENT)
    _, snaps = run_cd(t1(entropy_bound=0.5), xargs, const(x), 4)
    for k, s in enumerate(snaps):
        assert s.committed("c") is None and not bool(s.row("is_encoder_phase", "c")), (
            f"call {k}: the slot entered its commit step (converged, or the per-step request was refused)"
        )
        assert_fed_back(s, FA_SETS, FA_MASK[arm], f"{arm} call {k}")


def test_end_projection_trajectory_is_stock_on_fixture_fa(fa: Grammar) -> None:
    """Discriminator for the row above: the stock trajectory feeds back the out-of-language 1000 at position 0
    (p ≈ 1 − 4e-5), which no per-step arm may do."""
    x = ps.logits_rows(FA_ROWS)
    _, snaps = run_cd(
        t1(entropy_bound=0.5, confidence_threshold=0.0), ps.ep_xargs(fa, "fa-ep"), const(x), 3
    )
    assert [canvas(s)[0] for s in snaps] == [1000, 1000, 1000]


@pytest.mark.parametrize(
    ("cfg_bound", "override", "want"),
    [(0.5, None, "0.5"), (0.5, 2.0, "2.0"), (2.0, None, "2.0"), (2.0, 0.5, "0.5")],
    ids=["default-0.5", "override-2.0", "default-2.0", "override-0.5"],
)
def test_bound_comes_from_request_else_stock(
    cfg_bound: float, override: float | None, want: str, fa: Grammar
) -> None:
    """Both directions: an override must be honoured, and without one the inner sampler's bound applies."""
    x = ps.logits_rows(FA_ROWS)
    extra: dict[str, Any] = {"confidence_threshold": ps.NEVER_CONFIDENT}
    if override is not None:
        extra["entropy_bound"] = override
    xargs = ps.ps_xargs(fa, "fa-bound", "mf", **extra)
    _, snaps = run_cd(t1(entropy_bound=cfg_bound), xargs, const(x), 2)
    mask = FA_MASK["mf"] if want == "0.5" else FA_MASK_BOUND2["mf"]
    for k, s in enumerate(snaps):
        assert_fed_back(s, FA_SETS, mask, f"mf bound={want} call {k}")


# --------------------------------------------------------------------------- confident


#: every position admits {1000, 1001} under logits flat over the 8 live ids: H(q) = ln 2, H(p) = ln 8
FC_SETS = [[1000, 1001]] * 4


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize(
    ("cfg_thr", "override"), [(1.0, None), (1.0, 0.5), (0.5, None), (0.5, 1.0)]
)
def test_confident_is_mean_entropy_of_the_arm(
    arm: str, cfg_thr: float, override: float | None
) -> None:
    g = ps.product_grammar(FC_SETS, "t0068-fc")
    x = ps.logits_rows([ps.flat(LIVE)] * 4)
    hq = np.mean([ps.entropy64(ps.q_product64(x[i], FC_SETS[i])) for i in range(4)])
    hp = np.mean([ps.entropy64(ps.p64(x[i])) for i in range(4)])
    assert abs(hq - math.log(2)) < 1e-6 and abs(hp - math.log(8)) < 1e-3
    thr = cfg_thr if override is None else override
    want = (hq if arm != "mf" else hp) < thr
    extra = {} if override is None else {"confidence_threshold": override}
    xargs = ps.ps_xargs(g, f"fc-{arm}", arm, **extra)
    _, snaps = run_cd(
        t1(confidence_threshold=cfg_thr, max_denoising_steps=1000), xargs, const(x), 1
    )
    got = bool(snaps[0].row("confident", "c"))
    assert got == want, (
        f"{arm}: confident = {got}, expected mean {'H(p)' if arm == 'mf' else 'H(q)'} "
        f"({hp if arm == 'mf' else hq:.3f}) < {thr}"
    )


# --------------------------------------------------------------------------- the joint draw is what is fed back


@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
def test_fed_back_canvas_is_a_joint_draw(arm: str) -> None:
    """XOR language {ab, ba}, every position accepted: the fed-back canvas is x̂ ~ p(x | C). n = 200 calls,
    CL = 2, T = 1. Oracle: P(ab) = p0(a)p1(b) / (p0(a)p1(b) + p0(b)p1(a)) = 0.7 by construction. A per-position
    (marginal) sampler produces aa / bb (each marginal is (0.7, 0.3) / (0.3, 0.7)); the stock sampler too."""
    a, b = 1000, 1001
    g = ps.xor_grammar(a, b, "t0068-xor")
    s = math.log(7 / 3)
    x = ps.logits_rows([{a: s / 2, b: 0.0}, {a: 0.0, b: s / 2}])
    p0, p1 = ps.p64(x[0]), ps.p64(x[1])
    w_ab, w_ba = p0[a] * p1[b], p0[b] * p1[a]
    p_ab = w_ab / (w_ab + w_ba)
    assert abs(p_ab - 0.7) < 1e-6
    xargs = ps.ps_xargs(
        g, f"xor-{arm}", arm, entropy_bound=ps.ACCEPT_ALL, confidence_threshold=ps.NEVER_CONFIDENT
    )
    n = 200
    _, snaps = run_cd(t1(CL=2), xargs, const(x), n)
    seen = [tuple(canvas(sn)) for sn in snaps]
    bad = [c for c in seen if c not in ((a, b), (b, a))]
    assert bad == [], f"{arm}: fed-back canvases outside the language (not a joint draw): {bad[:5]}"
    k = sum(1 for c in seen if c == (a, b))
    z = (k / n - p_ab) / math.sqrt(p_ab * (1 - p_ab) / n)
    assert abs(z) < 4.5, f"{arm}: P(ab) = {k / n:.3f} over n = {n}, exact {p_ab:.3f} (z = {z:.2f})"


@pytest.mark.parametrize("arm", ARMS)
def test_renoise_ids_come_from_cd_generator(arm: str) -> None:
    """A tiny bound accepts exactly one position per step; the other CL − 1 are renoised. Those ids are
    uniform over V, not stock's global-RNG ``randint`` ids, and the global RNG ends where stock's does."""
    g = ps.loop_grammar(LIVE, "t0068-loop")
    CL = 8
    x = ps.logits_rows([ps.flat(LIVE)] * CL)
    xargs = ps.ps_xargs(
        g, f"rn-{arm}", arm, entropy_bound=ps.TINY_BOUND, confidence_threshold=ps.NEVER_CONFIDENT
    )
    specs = [Spec("c", xargs=xargs), Spec("u")]
    a, b = run_pair(t1(CL=CL), specs, const(x), n_calls=12)
    ids: list[int] = []
    for k, (sa, sb) in enumerate(zip(a, b, strict=True)):
        assert torch.equal(sa.rng_after, sb.rng_after), f"call {k}: CD moved the global RNG"
        ca = canvas(sa)
        cb = [int(t) for t in sb.row("canvas", "c")]
        live = [i for i, t in enumerate(ca) if not ps.renoised(t)]
        assert len(live) == 1, (
            f"{arm} call {k}: {len(live)} accepted positions, expected exactly 1: {ca}"
        )
        both = [i for i in range(CL) if ps.renoised(ca[i]) and ps.renoised(cb[i])]
        same = [i for i in both if ca[i] == cb[i]]
        assert same == [], f"{arm} call {k}: renoise ids equal stock's global-RNG ids at {same}"
        ids += [ca[i] for i in range(CL) if i not in live]
    assert all(0 <= t < 262_144 for t in ids)
    assert len(set(ids)) >= 0.95 * len(ids) and max(ids) - min(ids) > 262_144 // 2, (
        "renoise ids are not spread over the vocabulary"
    )


# --------------------------------------------------------------------------- history, argmax_canvas, convergence


@pytest.mark.parametrize("arm", ARMS)
def test_history_and_argmax_canvas_hold_the_constrained_map(arm: str) -> None:
    g = Grammar(G1, "t0068-g1")
    fn = lambda call, req, n: random_live_logits(7000 + call, LIVE, n)  # noqa: E731
    outside = [not g.accepts({0}, argmax_ids(fn(c, "c", 4)), BIG) for c in range(1, 6)]
    assert sum(outside) >= 2, "fixture: the argmax must leave G1 on some calls"
    xargs = ps.ps_xargs(g, f"hist-{arm}", arm, confidence_threshold=ps.NEVER_CONFIDENT)
    _, snaps = run_cd(t1(), xargs, fn, 5)
    ST = snaps[0].state["accepted_canvas_history"].shape[1]
    for j, s in enumerate(snaps):
        want, _sc, gap = brute_map(g, fn(s.call, "c", 4), {0}, 4, BIG)
        assert gap > 1e-6
        assert [int(t) for t in s.row("argmax_canvas", "c")] == want, (
            f"{arm} call {j}: argmax_canvas != MAP"
        )
        hist = s.row("accepted_canvas_history", "c")[j % ST]
        assert [int(t) for t in hist] == want, f"{arm} call {j}: history row {j % ST} != MAP"


#: argmax alternates between two out-of-language ids every call; the constrained MAP is the literal
LITERAL = [1000, 1001, 1002, 1003]


def _alternating(call: int, req: str, n: int) -> torch.Tensor:
    if req != "c":
        return random_live_logits(900 + call, LIVE, n)
    peak = 1004 if call % 2 else 1005
    return ps.logits_rows([{**ps.flat(LITERAL), peak: 3.0}] * n)


@pytest.mark.parametrize("arm", ARMS)
def test_convergence_on_stable_map_history_not_argmax(arm: str) -> None:
    """The MAP history is stable from the start, the argmax history never is: the arm converges after ST
    denoise steps (canvas ← MAP) and commits the literal; stock (same batch, keys removed) runs to the cap.
    At the converging call ``sc_embeds`` is zero (``sc_keep`` from CD's own convergence) although stock's
    is not."""
    g = ps.product_grammar([[t] for t in LITERAL], "t0068-lit")
    cap = 8
    xargs = ps.ps_xargs(g, f"conv-{arm}", arm, confidence_threshold=ps.ALWAYS_CONFIDENT)
    specs = [Spec("c", xargs=xargs), Spec("u")]
    a, b = run_pair(t1(max_denoising_steps=cap, confidence_threshold=1e9), specs, _alternating, 6)
    ST = a[0].state["accepted_canvas_history"].shape[1]
    assert convergence_calls(b, "c")[:1] == [], "fixture: stock must not converge before the cap"
    conv = convergence_calls(a, "c")
    assert conv[:1] == [ST - 1], f"{arm}: first convergence at call {conv[:1]}, expected {ST - 1}"
    s = a[ST - 1]
    assert canvas(s) == LITERAL, f"{arm}: canvas at convergence {canvas(s)} != MAP {LITERAL}"
    assert commit_calls(a, "c")[:1] == [ST] and a[ST].committed("c") == LITERAL
    assert torch.count_nonzero(s.row("self_conditioning_embeds", "c")) == 0, (
        f"{arm}: sc_embeds must be zero on the converging call (sc_keep from CD's convergence)"
    )
    assert torch.count_nonzero(b[ST - 1].row("self_conditioning_embeds", "c")) > 0, (
        "fixture premise"
    )


@pytest.mark.parametrize("arm", ARMS)
def test_step_cap_commits_the_map_of_that_step(arm: str) -> None:
    g = Grammar(G1, "t0068-g1-cap")
    cap = 4
    fn = lambda call, req, n: random_live_logits(8100 + call, LIVE, n)  # noqa: E731
    xargs = ps.ps_xargs(g, f"cap-{arm}", arm, confidence_threshold=ps.NEVER_CONFIDENT)
    _, snaps = run_cd(t1(max_denoising_steps=cap), xargs, fn, cap + 1)
    assert convergence_calls(snaps, "c")[:1] == [cap - 1]
    want = brute_map(g, fn(snaps[cap - 1].call, "c", 4), {0}, 4, BIG)[0]
    prev = brute_map(g, fn(snaps[cap - 2].call, "c", 4), {0}, 4, BIG)[0]
    assert want != prev, "fixture: the MAPs of consecutive steps must differ"
    assert snaps[cap].committed("c") == want


# --------------------------------------------------------------------------- self-conditioning


FSC_SETS = [[1000, 1001]] * 4


@pytest.mark.parametrize("arm", ARMS)
def test_sc_embeds_p_or_q_times_embedding(arm: str) -> None:
    """Non-converging denoise call: ``sc_embeds[slot] = (π @ E) * normalizer`` with π = p (Mf, Mar) or q (Mar +
    constrained SC). fp64 oracle; the fixture makes p·E and q·E differ by far more than the tolerance."""
    g = ps.product_grammar(FSC_SETS, "t0068-sc")
    x = ps.logits_rows([ps.flat(LIVE)] * 4)
    xargs = ps.ps_xargs(g, f"sc-{arm}", arm, confidence_threshold=ps.NEVER_CONFIDENT)
    w, snaps = run_cd(t1(), xargs, const(x), 2)
    E = w.ms.embed_weight.to(torch.float64).cpu().numpy()
    norm = float(w.ms.normalizer)
    pE = np.stack([ps.p64(x[i]) @ E for i in range(4)]) * norm
    qE = np.stack([ps.q_product64(x[i], FSC_SETS[i]) @ E for i in range(4)]) * norm
    assert np.abs(pE - qE).max() > 1e-3, "fixture: p·E and q·E must differ"
    want = qE if arm == "mar_sc" else pE
    for k, s in enumerate(snaps):
        got = s.row("self_conditioning_embeds", "c").to(torch.float64).numpy()
        err = np.abs(got - want).max()
        assert err < 1e-5, (
            f"{arm} call {k}: sc_embeds off by {err:.2e} from {'q' if arm == 'mar_sc' else 'p'}·E"
        )


# --------------------------------------------------------------------------- temperature (P3)


@pytest.mark.parametrize(("T", "all_accepted"), [(0.5, True), (1.0, False)])
def test_entropy_is_taken_at_stock_temperature(T: float, all_accepted: bool) -> None:
    """Mar, bound 0.45: two positions over {1000: 0, 1001: 1}: H(q) = 0.362 at T = 0.5 (all accepted), 0.582 at
    T = 1 (one of them rejected). Pins P3 in both directions."""
    sets = [[1000, 1001], [1000, 1001], [1002], [1003]]
    g = ps.product_grammar(sets, "t0068-temp")
    x = ps.logits_rows([{1000: 0.0, 1001: 1.0}] * 2 + [ps.flat([1002]), ps.flat([1003])])
    h = [ps.entropy64(ps.q_product64(x[i], sets[i], T)) for i in range(4)]
    assert all(ps.stock_rule(h, 0.45)) == all_accepted and ps.margin(h, 0.45) > 0.05
    xargs = ps.ps_xargs(g, "temp", "mar", confidence_threshold=ps.NEVER_CONFIDENT)
    _, snaps = run_cd(t1(t_min=T, t_max=T, entropy_bound=0.45), xargs, const(x), 3)
    for k, s in enumerate(snaps):
        c = canvas(s)
        ok = [c[i] in (1000, 1001) for i in (0, 1)]
        if all_accepted:
            assert all(ok), f"T={T} call {k}: {c}: H(q) at stock temperature accepts both"
        else:
            assert ok.count(True) == 1, (
                f"T={T} call {k}: {c}: exactly one of positions 0, 1 is rejected"
            )


# --------------------------------------------------------------------------- criterion 5


@pytest.mark.parametrize("arm", ARMS)
def test_literal_automaton_converges_within_st_plus_one(arm: str) -> None:
    """Criterion 5: a single-string language under random logits; stock bound/threshold 0.1 / 0.005.
    Mar (H(q) = 0) converges within ST + 1 steps and commits the literal. Mf (H(p) high) is never confident
    and runs to the cap: the discriminating direction."""
    g = ps.product_grammar([[t] for t in LITERAL], "t0068-lit5")
    fn = lambda call, req, n: random_live_logits(4200 + call, LIVE, n)  # noqa: E731
    assert argmax_ids(fn(1, "c", 4)) != LITERAL
    cap = 8
    cfg = Cfg(
        CL=4,
        stability_threshold=1,
        entropy_bound=0.1,
        confidence_threshold=0.005,
        max_denoising_steps=cap,
    )
    _, snaps = run_cd(cfg, ps.ps_xargs(g, f"lit-{arm}", arm), fn, cap + 1)
    ST = snaps[0].state["accepted_canvas_history"].shape[1]
    commits = commit_calls(snaps, "c")
    assert commits, f"{arm}: no commit within {cap + 1} calls"
    if arm == "mf":
        assert commits[0] == cap, (
            f"mf: committed at call {commits[0]}; H(p) is never confident, cap is {cap}"
        )
    else:
        assert commits[0] <= ST, f"{arm}: committed at call {commits[0]} > ST = {ST}"
    assert snaps[commits[0]].committed("c") == LITERAL


def _peaked(call: int, req: str, n: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(3000 + call // 3 + (0 if req == "c" else 77))
    top = torch.randint(0, len(LIVE), (n,), generator=g)
    return ps.logits_rows([{**ps.flat(LIVE), LIVE[int(t)]: 20.0} for t in top])


def test_mf_converges_like_stock_when_the_argmax_is_in_the_language() -> None:
    """Criterion 5: loop language over the live ids (every argmax admitted), p ≈ one-hot (deterministic),
    stock bound/threshold. Mf converges on the same calls as stock and commits the same canvases."""
    g = ps.loop_grammar(LIVE, "t0068-loop5")
    cfg = Cfg(
        CL=4,
        stability_threshold=1,
        entropy_bound=0.1,
        confidence_threshold=0.005,
        max_denoising_steps=48,
    )
    specs = [Spec("c", xargs=ps.ps_xargs(g, "mf5", "mf")), Spec("u")]
    a, b = run_pair(cfg, specs, _peaked, n_calls=10)
    cb = convergence_calls(b, "c")
    assert len(cb) >= 2 and cb[0] < 47, "fixture: stock must converge on stability"
    assert convergence_calls(a, "c")[:3] == cb[:3]
    assert [a[i].committed("c") for i in commit_calls(a, "c")[:2]] == [
        b[i].committed("c") for i in commit_calls(b, "c")[:2]
    ]


# --------------------------------------------------------------------------- criterion 7: records


def _fa_two_steps(call: int, req: str, n: int) -> torch.Tensor:
    """Call 0: F-A (Mf accepts 2). Call 1 on: position 2 becomes peaked on its forced id (Mf accepts 3)."""
    if req != "c":
        return random_live_logits(600 + call, LIVE, n)
    rows = list(FA_ROWS)
    if call >= 2:  # decode call 0 is sampler call 1 (the prefill is call 0)
        rows[2] = {**ps.flat(range(1000, 1006)), 1002: 12.0}
    return ps.logits_rows(rows)


@pytest.mark.parametrize(("arm", "mean"), [("mf", 2.5), ("mar", 4.0), ("mar_sc", 4.0)])
def test_record_mean_accepted_per_step_and_steps(
    arm: str, mean: float, fa: Grammar, caplog: Any, capsys: Any
) -> None:
    records = Records(caplog, capsys)
    hp0, hq0 = _fa_entropies(_fa_two_steps(1, "c", 4))
    hp1, hq1 = _fa_entropies(_fa_two_steps(2, "c", 4))
    counts = {
        "mf": [sum(ps.stock_rule(hp0, 0.5)), sum(ps.stock_rule(hp1, 0.5))],
        "mar": [sum(ps.stock_rule(hq0, 0.5)), sum(ps.stock_rule(hq1, 0.5))],
    }
    counts["mar_sc"] = counts["mar"]
    assert counts[arm] == ([2, 3] if arm == "mf" else [4, 4]) and ps.margin(hp1, 0.5) > 0.05
    tag = f"rec-{arm}"
    g = fa
    xargs = ps.ps_xargs(g, tag, arm, confidence_threshold=ps.ALWAYS_CONFIDENT)
    _, snaps = run_cd(t1(entropy_bound=0.5), xargs, _fa_two_steps, 6)
    assert commit_calls(snaps, "c")[:1] == [2], (
        "fixture: the MAP is stable, so the commit is at call ST = 2"
    )
    want_ids = brute_map(g, _fa_two_steps(2, "c", 4), {0}, 4, BIG)[0]
    recs = [r for r in records.tagged(tag) if r.get("canvas_idx") == 0]
    assert len(recs) == 1, f"expected one record for canvas 0, got {recs}"
    rec = recs[0]
    validate(rec)
    assert rec["error"] is None and rec["arm"] == ps.ARMS[arm][0]
    assert list(rec["committed_ids"]) == want_ids
    assert rec["steps"] == 2, f"steps = {rec['steps']}, the canvas had 2 denoise steps"
    got = ps.record_count_field(rec)
    assert got is not None and abs(got - mean) < 1e-9, (
        f"{arm}: mean_accepted_per_step = {got}, expected {mean} (per-step counts {counts[arm]})"
    )
    for name in ("shadow_argmax_ids", "argmax_in_language", "map_equals_argmax"):
        assert rec[name] is None
