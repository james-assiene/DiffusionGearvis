"""T-0076 criteria 2, 3, 4 and 7: the settling arm ``perstep_mar_qargmax`` (history = per-position argmax of the
constrained marginals q; the joint constrained MAP over p only at commit) and its SC twin.

Everything runs through T-0067's CPU stand-in (vendored v0.29.0 stock step, ``tests/cd/fakes/diffusion_sampler.py``)
with one constrained slot plus one unconstrained companion, the arm chosen only by the wire key
``astra_cd_mode="perstep_mar_qargmax"`` (ruling P1). Expected values are hand-built: q from the fp64 enumeration
oracle ``_perstep_support.brute_q``, the MAP from ``_sampler_support.brute_map``. V = 262 144; T = 1 (t_min = t_max
= 1) unless stated; ST = 2 (stability_threshold 1).

Instance C3 (criteria 3, 4): ids a b c d e = 1000..1004; language {ad, ae, bc}; p₀ = (a .5, b .5),
p₁ = (c .4, d .3, e .3). Strings: ad .15, ae .15, bc .20 -> q₀ = (a .6, b .4), q₁ = (c .4, d .3, e .3), so the
argmax of q is ``ac`` -- **not in the language** -- and the MAP is ``bc``.

The history row written at a denoise call is ``accepted_canvas_history[(len − 1) mod ST]`` after the call
(stock and CD both advance the length by one per denoise step).

Mutants these rows kill: history = MAP (``bc``), = the joint draw, = the fed-back canvas (C3 rows and the
draw ≠ argmax row); last index on ties (tie row); argmax over p instead of q
(``test_history_is_argmax_of_q_not_of_p``); argmax-of-q committed or used
for δ* at convergence (commit row: ``ac`` is out of language, δ*(ac) = ∅ would end loudly); rows ≥ L_eff not
``perstep_mar``'s (L_eff < CL rows); an extra generator draw or a changed accept/confident/SC (bit-identity rows).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch

from tests.cd import _tap_cd_world as tw
from tests.cd.fakes.diffusion_sampler import Cfg, Snap, World, bits, random_live_logits
from tests.cd.unit import _perstep_support as ps
from tests.cd.unit._sampler_support import G1, LIVE, Grammar, Records, brute_map, validate
from tests.cd.unit.test_cd_arms_semantics import const, t1
from tests.fakes.diffusion_sampler import validate_canvas_step
from tests.stock.world import frame_mismatches

QA, QA_SC = "qa", "qa_sc"
QA_MODE = "perstep_mar_qargmax"
A, B, C, D, E = 1000, 1001, 1002, 1003, 1004
#: the twin of each new arm: identical except the history row
TWIN = {QA: "mar", QA_SC: "mar_sc"}


def c3_grammar(name: str = "t0076-c3") -> Grammar:
    return Grammar((4, [(0, A, 1), (0, B, 2), (1, D, 3), (1, E, 3), (2, C, 3)], {3}), name)


def c3_rows(extra: Sequence[Mapping[int, float]] = ()) -> list[dict[int, float]]:
    return [
        {A: math.log(0.5), B: math.log(0.5)},
        {C: math.log(0.4), D: math.log(0.3), E: math.log(0.3)},
        *[dict(r) for r in extra],
    ]


def run(
    cfg: Cfg,
    xargs: Mapping[str, Any],
    fn: Any,
    n_calls: int,
    *,
    max_tokens: int = 4096,
    valid: int | None = None,
    seed: int = 0,
) -> tuple[World, list[Snap]]:
    w = World(cfg, cd=True, seed=seed)
    w.add("c", xargs=xargs, max_tokens=max_tokens)
    w.add("u")
    if valid is not None:
        w.reqs["c"].valid_len = valid
    w.prefill()
    snaps = [w.decode(fn)]
    require_admitted(snaps[0], xargs)
    snaps += [w.decode(fn) for _ in range(n_calls - 1)]
    return w, snaps


def require_admitted(s: Snap, xargs: Mapping[str, Any], req: str = "c") -> None:
    """A refused constrained request is forced to an all-EOS canvas in its commit phase on its first call;
    say so plainly instead of failing on a downstream row."""
    if xargs.get("astra_cd_mode") != QA_MODE:
        return
    slot = s.slots[req]
    canvas_row = s.state["canvas"][slot]
    if bool(s.state["is_encoder_phase"][slot]) and bool((canvas_row == 1).all()):
        pytest.fail(
            "T-0076 not implemented: the perstep_mar_qargmax request was refused at add_request "
            "(mode not accepted; forced EOS canvas)"
        )


def hist_row(s: Snap) -> list[int]:
    hist = s.row("accepted_canvas_history", "c")
    hl = int(s.row("accepted_canvas_history_len", "c"))
    assert hl >= 1, "fixture: no history row written yet"
    return [int(t) for t in hist[(hl - 1) % hist.shape[0]]]


def row(s: Snap, name: str) -> list[int]:
    return [int(t) for t in s.row(name, "c")]


def phase(s: Snap) -> bool:
    return bool(s.row("is_encoder_phase", "c"))


def cd_generator_state(w: World) -> torch.Tensor:
    gens = getattr(w.sampler, "_generators", None)
    assert gens, "fixture: the CD sampler holds no per-step generator (T-0068 ruling P6)"
    return next(iter(gens.values())).get_state().clone()


# --------------------------------------------------------------------------- the C3 premise


def test_c3_premise_q_and_map_by_hand() -> None:
    g = c3_grammar()
    x = ps.logits_rows(c3_rows())
    q = ps.brute_q(g, x, [0], 2, 4000)
    assert q[0][A] == pytest.approx(0.6, abs=1e-6) and q[0][B] == pytest.approx(0.4, abs=1e-6)
    assert q[1][C] == pytest.approx(0.4, abs=1e-6)
    assert q[1][D] == pytest.approx(0.3, abs=1e-6) and q[1][E] == pytest.approx(0.3, abs=1e-6)
    assert ps.argmax_q(q) == [A, C]
    assert not g.accepts([0], [A, C], 4000), "premise: argmax-of-q ac is out of the language"
    tokens, _, gap = brute_map(g, x, [0], 2, 4000)
    assert tokens == [B, C] and gap > 0.2, (tokens, gap)


# --------------------------------------------------------------------------- criterion 3


@pytest.mark.parametrize("arm", [QA, QA_SC])
def test_history_is_argmax_of_q_not_the_map(arm: str) -> None:
    """Criterion 3: on C3, the new arm's history row and ``argmax_canvas`` are ``ac`` at every non-converging
    denoise call; ``perstep_mar``'s (same knobs) are the MAP ``bc``. The joint draw and the fed-back canvas
    can never be ``ac`` (out of the language), so "history = draw/fed" fails too."""
    g = c3_grammar()
    x = ps.logits_rows(c3_rows())
    for a, want in ((arm, [A, C]), (TWIN[arm], [B, C])):
        xargs = ps.ps_xargs(g, f"c3-{a}", a, confidence_threshold=ps.NEVER_CONFIDENT)
        _, snaps = run(t1(CL=2), xargs, const(x), 3)
        for k, s in enumerate(snaps):
            assert s.committed("c") is None and not phase(s), (
                f"{a} call {k}: entered its commit step (never confident; the per-step request was refused?)"
            )
            assert hist_row(s) == want, f"{a} call {k}: history row {hist_row(s)}, want {want}"
            assert row(s, "argmax_canvas") == want, (
                f"{a} call {k}: argmax_canvas {row(s, 'argmax_canvas')}, want {want}"
            )
            assert row(s, "canvas") != [A, C], f"{a} call {k}: the fed-back canvas is ac"


def test_history_is_argmax_of_q_not_of_p() -> None:
    """Discriminator for "argmax over p": p₁ prefers d (.5) over c (.3), but q₁ prefers c.

    Language {ac, bc, bd}; p₀ = (a .9, b .1), p₁ = (c .3, d .5, e .2): strings ac .27, bc .03, bd .05 ->
    q₀ = (a .771, b .229), q₁ = (c .857, d .143): argmax q = ``ac``; argmax p restricted per position = ``ad``
    (out of language); MAP = ``ac``. So this row pins q, not p, and agrees with the MAP (the C3 rows split them)."""
    g = Grammar((4, [(0, A, 1), (0, B, 2), (1, C, 3), (2, C, 3), (2, D, 3)], {3}), "t0076-qp")
    x = ps.logits_rows(
        [
            {A: math.log(0.9), B: math.log(0.1)},
            {C: math.log(0.3), D: math.log(0.5), E: math.log(0.2)},
        ]
    )
    q = ps.brute_q(g, x, [0], 2, 4000)
    assert ps.argmax_q(q) == [A, C] and q[1][C] == pytest.approx(
        0.27 / 0.35 + 0.03 / 0.35, abs=1e-6
    )
    xargs = ps.ps_xargs(g, "qp-qa", QA, confidence_threshold=ps.NEVER_CONFIDENT)
    _, snaps = run(t1(CL=2), xargs, const(x), 2)
    assert [hist_row(s) for s in snaps] == [[A, C], [A, C]]


# --------------------------------------------------------------------------- criterion 4 (adversarial)


@pytest.mark.parametrize("arm", [QA, QA_SC])
def test_commit_is_the_map_when_argmax_of_q_is_out_of_language(
    arm: str, caplog: Any, capsys: Any
) -> None:
    """Criterion 4: C3 with fixed logits and an always-confident threshold. Call 0: history ``ac``, not converged
    (one row). Call 1 (step ST = 2): history ``ac ac`` is stable and confident -> converges with
    ``argmax_canvas = canvas = MAP = bc``. Call 2 commits ``bc`` (host replay accepts), never ``ac``; δ* is taken of
    the MAP (δ*(ac) = ∅ would end the request loudly at the next canvas); no error record."""
    records = Records(caplog, capsys)
    g = c3_grammar()
    tag = f"c4-{arm}"
    x = ps.logits_rows(c3_rows())
    xargs = ps.ps_xargs(g, tag, arm, confidence_threshold=ps.ALWAYS_CONFIDENT)
    _, snaps = run(t1(CL=2), xargs, const(x), 6)
    s0, s1, s2 = snaps[:3]
    assert not phase(s0) and hist_row(s0) == [A, C] and row(s0, "argmax_canvas") == [A, C]
    assert phase(s1), "call 1 (step ST): a stable, confident argmax-of-q history must converge"
    assert hist_row(s1) == [A, C], "convergence is decided on the argmax-of-q history"
    assert row(s1, "argmax_canvas") == [B, C] and row(s1, "canvas") == [B, C], (
        f"at convergence argmax_canvas = canvas = MAP bc; got {row(s1, 'argmax_canvas')}, {row(s1, 'canvas')}"
    )
    slot = s1.slots["c"]
    assert s1.draft_tokens[slot, :2].tolist() == [B, C]
    assert s2.committed("c") == [B, C], f"committed {s2.committed('c')}, want the MAP bc"
    assert g.accepts([0], [B, C], 4000)
    committed = [s.committed("c") for s in snaps if s.committed("c") is not None]
    assert [A, C] not in committed
    recs = sorted(records.tagged(tag), key=lambda r: r["canvas_idx"])
    assert recs and all(r["error"] is None for r in recs), f"a loud end: {recs}"
    assert recs[0]["committed_ids"] == [B, C]
    if len(recs) > 1:
        assert recs[1]["committed_ids"][0] in (1, 50, 106), (
            "after bc only a stop is viable (δ* of the MAP)"
        )


# --------------------------------------------------------------------------- criterion 2: tie, L_eff < CL


def test_a_tie_in_q_takes_the_first_index() -> None:
    """Adversarial: position 0 admits a and b with identical logits and the same next state, so q₀(a) = q₀(b)
    exactly; the row takes a (the first index). Positions 1-2 are C3 shifted (a'=1002, b'=1003 / c'=1004,
    d'=1005, e'=1006) so the row is argmax-of-q there (a'c'), not the MAP (b'c')."""
    ap, bp, cp, dp, ep = 1002, 1003, 1004, 1005, 1006
    g = Grammar(
        (
            5,
            [(0, A, 1), (0, B, 1), (1, ap, 2), (1, bp, 3), (2, dp, 4), (2, ep, 4), (3, cp, 4)],
            {4},
        ),
        "t0076-tie",
    )
    x = ps.logits_rows(
        [
            {A: 0.0, B: 0.0},
            {ap: math.log(0.5), bp: math.log(0.5)},
            {cp: math.log(0.4), dp: math.log(0.3), ep: math.log(0.3)},
        ]
    )
    q = ps.brute_q(g, x, [0], 3, 4000)
    assert q[0][A] == q[0][B], "premise: an exact tie at position 0"
    assert ps.argmax_q(q) == [A, ap, cp]
    xargs = ps.ps_xargs(g, "tie-qa", QA, confidence_threshold=ps.NEVER_CONFIDENT)
    _, snaps = run(t1(CL=3), xargs, const(x), 2)
    for k, s in enumerate(snaps):
        assert hist_row(s) == [A, ap, cp], (
            f"call {k}: {hist_row(s)} (first index on the q tie, then a'c')"
        )


@pytest.mark.parametrize("variant", ["budget", "pad"])
@pytest.mark.parametrize("arm", [QA, QA_SC])
def test_rows_beyond_l_eff_are_perstep_mars(arm: str, variant: str) -> None:
    """Adversarial, L_eff = 3 < CL = 4: by budget (``max_tokens`` 3, valid 4) or by a padded canvas (valid 3).
    Positions 0-2 are argmax-of-q (``a c EOS``; after C3's final state only a stop is viable, EOS preferred);
    position 3 equals ``perstep_mar``'s row on the same logits (``where(i < L_eff, argmax q, map_row)``)."""
    g = c3_grammar(f"t0076-leff-{variant}")
    x = ps.logits_rows(c3_rows([{1: 2.0}, {1005: 3.0, 1006: 1.0}]))
    kw: dict[str, Any] = {"max_tokens": 3} if variant == "budget" else {"valid": 3}
    got: dict[str, list[list[int]]] = {}
    for a in (arm, TWIN[arm]):
        xargs = ps.ps_xargs(g, f"leff-{a}", a, confidence_threshold=ps.NEVER_CONFIDENT)
        _, snaps = run(t1(CL=4), xargs, const(x), 2, **kw)
        got[a] = [hist_row(s) for s in snaps]
        for k, s in enumerate(snaps):
            assert not phase(s), f"{a} call {k}: converged or refused"
            assert row(s, "argmax_canvas") == got[a][k], (
                f"{a} call {k}: argmax_canvas != history row"
            )
    for k in range(2):
        assert got[arm][k][:3] == [A, C, 1], f"call {k}: scored rows {got[arm][k][:3]} != a c EOS"
        assert got[TWIN[arm]][k][:3] == [B, C, 1], f"call {k}: perstep_mar's scored rows"
        assert got[arm][k][3] == got[TWIN[arm]][k][3], (
            f"call {k}: position 3 ≥ L_eff is {got[arm][k][3]}, perstep_mar's is {got[TWIN[arm]][k][3]}"
        )


def test_history_is_not_the_joint_draw() -> None:
    """Adversarial (kills "history = fed / x̂"): a product language where argmax-of-q is in the language
    (= the MAP), with every position accepted so the fed-back canvas is the joint draw x̂. Over 8 calls the
    draw differs from argmax-of-q at least once (fixture premise), while the history row never does."""
    sets = [[A, B], [C, D]]
    g = ps.product_grammar(sets, "t0076-draw")
    x = ps.logits_rows(
        [{A: math.log(0.6), B: math.log(0.4)}, {C: math.log(0.55), D: math.log(0.45)}]
    )
    xargs = ps.ps_xargs(
        g, "draw-qa", QA, entropy_bound=ps.ACCEPT_ALL, confidence_threshold=ps.NEVER_CONFIDENT
    )
    _, snaps = run(t1(CL=2), xargs, const(x), 8)
    draws = [row(s, "canvas") for s in snaps]
    assert all(d[0] in (A, B) and d[1] in (C, D) for d in draws), (
        f"premise: every position fed x̂ {draws}"
    )
    assert any(d != [A, C] for d in draws), f"premise: the draw never left argmax-of-q {draws}"
    assert [hist_row(s) for s in snaps] == [[A, C]] * 8


# --------------------------------------------------------------------------- criterion 2: bit-identity with Mar


def _random_fn(seed: int) -> Any:
    """Live logits uniform in [0, 1.5] (flat enough that argmax-of-q and the MAP part on canvas 0)."""

    def fn(call: int, req: str, n: int) -> torch.Tensor:
        base = 10_000 * seed + (call if req == "c" else 500 + call)
        return random_live_logits(base, LIVE, n, hi=1.5)

    return fn


IDENTICAL_ROWS = ("canvas", "confident", "self_conditioning_embeds", "step", "is_encoder_phase")
#: seeds whose canvas 0 has a call (0-2) where the fp64 argmax of q differs from the MAP (checked in-row)
SEEDS = (1, 3, 5, 7)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("arm", [QA, QA_SC])
def test_everything_but_the_history_is_perstep_mars(
    arm: str, seed: int, caplog: Any, capsys: Any
) -> None:
    """Criterion 2: G1, CL 4, random logits per call, bound 0.5, never confident and a step cap of 4, so both
    arms converge only by the cap -- at the same calls -- and commit the MAP. Over 15 calls (≥ 3 canvases) the
    fed-back canvas, ``confident``, ``sc_embeds``, step, phase, ``draft_tokens``, the committed ids, the CD
    generator state and the records (``steps``, ``mean_accepted_per_step``, ...) are bit-identical to the twin
    Mar arm's; at a converging call ``argmax_canvas`` is the MAP for both. On canvas 0 (calls 0-2, start {0},
    L_eff = CL) the new arm's history row equals the fp64 argmax of q (``brute_q``), and on at least one of those
    calls that differs from the MAP (non-vacuity, asserted on the oracle)."""
    records = Records(caplog, capsys)
    g = Grammar(G1, f"t0076-ident-{seed}")
    cfg = t1(CL=4, max_denoising_steps=4, entropy_bound=0.5)
    fn = _random_fn(seed)
    out: dict[str, tuple[list[Snap], list[torch.Tensor]]] = {}
    for a in (arm, TWIN[arm]):
        xargs = ps.ps_xargs(g, f"id-{a}-{seed}", a, confidence_threshold=ps.NEVER_CONFIDENT)
        w = World(cfg, cd=True, seed=seed)
        w.add("c", xargs=xargs)
        w.add("u")
        w.prefill()
        snaps, gens = [], []
        for _ in range(15):
            snaps.append(w.decode(fn))
            if len(snaps) == 1:
                require_admitted(snaps[0], xargs)
            gens.append(cd_generator_state(w))
        w.decode(fn)  # flush the records of the last call
        out[a] = (snaps, gens)
    (sa, ga), (sb, gb) = out[arm], out[TWIN[arm]]
    # canvas 0 against the oracle
    parted = 0
    for k in range(3):
        s = sa[k]
        assert not phase(s) and s.committed("c") is None, f"fixture: call {k} is not a denoise call"
        xk = fn(s.call, "c", 4)
        want = ps.argmax_q(ps.brute_q(g, xk, [0], 4, 4096 - 4), min_gap=1e-4)
        parted += want != brute_map(g, xk, [0], 4, 4096 - 4)[0]
        assert hist_row(s) == want, f"call {k}: history {hist_row(s)}, fp64 argmax of q {want}"
        assert row(s, "argmax_canvas") == want, f"call {k}: argmax_canvas != argmax of q"
    assert parted >= 1, f"premise (seed {seed}): argmax of q equals the MAP on every canvas-0 call"
    bad = []
    for k, (x, y) in enumerate(zip(sa, sb, strict=True)):
        for name in IDENTICAL_ROWS:
            if not torch.equal(bits(x.row(name, "c")), bits(y.row(name, "c"))):
                bad.append(f"call {k}: {name}")
        if not torch.equal(x.draft_tokens[x.slots["c"]], y.draft_tokens[y.slots["c"]]):
            bad.append(f"call {k}: draft_tokens")
        if x.committed("c") != y.committed("c"):
            bad.append(f"call {k}: committed {x.committed('c')} vs {y.committed('c')}")
        if not torch.equal(ga[k], gb[k]):
            bad.append(f"call {k}: CD generator state (consumption differs)")
        converging = phase(x) and not x.entered_commit["c"]
        if converging and row(x, "argmax_canvas") != row(y, "argmax_canvas"):
            bad.append(f"call {k}: argmax_canvas at convergence is not the MAP")
    assert not bad, f"{arm} vs {TWIN[arm]} (seed {seed}):\n" + "\n".join(bad[:20])
    ra = sorted(records.tagged(f"id-{arm}-{seed}"), key=lambda r: r["canvas_idx"])
    rb = sorted(records.tagged(f"id-{TWIN[arm]}-{seed}"), key=lambda r: r["canvas_idx"])
    assert len(ra) >= 3 and len(ra) == len(rb), (len(ra), len(rb))
    skip = {"record_tag", "arm"}
    for x, y in zip(ra, rb, strict=True):
        assert {k: v for k, v in x.items() if k not in skip} == {
            k: v for k, v in y.items() if k not in skip
        }, (x, y)
        validate(x)
        assert x["arm"] == QA_MODE and x["error"] is None, x
        assert x["steps"] == 4 and x["mean_accepted_per_step"] is not None, x
        assert x["shadow_argmax_ids"] is None and x["map_equals_argmax"] is None, x


# --------------------------------------------------------------------------- admission (P13 for the new mode)


def test_overrides_are_admitted_and_logprobs_refused(caplog: Any, capsys: Any) -> None:
    """Criterion 1 at ``add_request``: the new mode with ``entropy_bound``/``confidence_threshold`` is admitted
    (no error, decodes); with ``logprobs`` it is refused loudly (``unsupported_option``), as every constrained
    arm is."""
    records = Records(caplog, capsys)
    g = c3_grammar("t0076-admit")
    x = ps.logits_rows(c3_rows())
    ok = ps.ps_xargs(g, "adm-ok", QA, entropy_bound=0.2, confidence_threshold=ps.NEVER_CONFIDENT)
    _, snaps = run(t1(CL=2), ok, const(x), 3)
    assert all(not phase(s) for s in snaps), (
        "an admitted request with a never-confident override converged"
    )
    assert hist_row(snaps[-1]) == [A, C]
    w = World(t1(CL=2), cd=True)
    w.add("c", xargs=ps.ps_xargs(g, "adm-lp", QA), logprobs=1)
    w.prefill()
    for _ in range(3):
        w.decode(const(x))
    errs = [r for r in records.tagged("adm-lp") if r["error"] is not None]
    assert [r["error"] for r in errs] == ["unsupported_option"], records.tagged("adm-lp")
    assert records.tagged("adm-ok") == [] or all(
        r["error"] is None for r in records.tagged("adm-ok")
    )


# --------------------------------------------------------------------------- criterion 7: tap frames


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, tw.TapRouter]:
    return tw.served_model(monkeypatch)


def test_tap_denoise_frames_carry_argmax_of_q_and_the_historys_stable(
    served: tuple[Any, tw.TapRouter],
) -> None:
    """Criterion 7: C3 under the served Tap ∘ CD ∘ stock composition, always confident. Every frame validates (R10)
    and equals its call's rows; call 0's denoise frame shows ``argmax = ac`` (argmax of q); call 1 (converging)
    shows the MAP ``bc`` with ``stable`` computed from the ``ac ac`` history; call 2's commit frame carries
    ``bc``."""
    astra, router = served
    g = c3_grammar("t0076-tap")
    cfg = t1(CL=2, max_denoising_steps=48)
    w = tw.World(cfg, astra.get_model_state_cls(), factory=tw.TapFactory(), router=router)
    names = tw.chain_names(w.sampler)
    assert names == ["TapSampler", "ConstrainedDiffusionSampler", "DiffusionSampler"], names
    x = ps.logits_rows(c3_rows())
    w.add("c", xargs=ps.ps_xargs(g, "tap-qa", QA, confidence_threshold=ps.ALWAYS_CONFIDENT))
    w.prefill()
    snaps = [w.decode(const(x))]
    require_admitted(snaps[0], {"astra_cd_mode": QA_MODE})
    snaps += [w.decode(const(x)) for _ in range(2)]
    frames = []
    for k, s in enumerate(snaps):
        mine = [f for f in s.frames if f.get("request_id") == "c"]  # type: ignore[attr-defined]
        assert len(mine) == 1, f"call {k}: {len(mine)} frames"
        validate_canvas_step(mine[0])
        bad = frame_mismatches(mine[0], s, "c", canvas_idx=0, max_steps=cfg.max_denoising_steps)
        assert bad == [], f"call {k}: {bad}"
        frames.append(mine[0])
    assert frames[0]["phase"] == "denoise" and list(frames[0]["argmax"]) == [A, C]
    assert frames[1]["phase"] == "denoise" and list(frames[1]["argmax"]) == [B, C]
    assert hist_row(snaps[1]) == [A, C], "the stable mask is computed on the argmax-of-q history"
    assert frames[2]["phase"] == "commit" and list(frames[2]["committed_tokens"]) == [B, C]


# --------------------------------------------------------------------------- lagged reports (review A, I1/T1)

#: (seed, max_tokens) matrix of review A's probe: CL 4, so canvas 1 (and 2) get L_eff < CL from the budget, and
#: with reports read ≥ 2 calls late the sampler enumerates several L_eff candidates and selects on the device
LAG_SEEDS = (1, 2, 3, 4, 5, 6)
LAG_MAX_TOKENS = (6, 7, 9, 10)
LAG_CL = 4
#: the oracle row's subset (brute_q enumerates 11^L_eff strings per call in Python)
LAG_ORACLE_SEEDS = (1, 2, 3)
LAG_ORACLE_MAX_TOKENS = (7, 10)


def _lag_run(
    arm: str, seed: int, max_tokens: int, lag: bool, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[dict[str, Any]], Any]:
    """One request (G1, CL 4, step cap 3, ST 2, never confident, bound 0.5, T = 1) for 14 calls or until the
    budget is spent; reports read on time or ≥ 2 calls late (T-0067's ``_lagging``). Per call: the history row,
    ``argmax_canvas``, the committed ids, the L_eff candidates the sampler enumerated, and the emitted count
    before the call."""
    from tests.cd.fakes.diffusion_sampler import cd_module
    from tests.cd.unit.test_cd_end_projection import _lagging

    sm = cd_module("sampler")
    orig = sm._l_eff_candidates
    seen: list[tuple[int, ...]] = []

    def spy(req: Any, valid: int, CL: int) -> list[int]:
        c = orig(req, valid, CL)
        seen.append(tuple(c))
        return c

    monkeypatch.setattr(sm, "_l_eff_candidates", spy)
    g = Grammar(G1, f"t0076-lag-{seed}")
    xargs = ps.ps_xargs(
        g,
        f"lag-{arm}-{seed}-{max_tokens}-{int(lag)}",
        arm,
        confidence_threshold=ps.NEVER_CONFIDENT,
        entropy_bound=0.5,
    )
    w = World(t1(CL=LAG_CL, max_denoising_steps=3), cd=True, seed=seed)
    w.add("c", xargs=xargs, max_tokens=max_tokens)
    if lag:
        w.call_ctx = _lagging(w)
    w.prefill()

    def fn(call: int, req: str, n: int) -> torch.Tensor:
        return random_live_logits(7000 * seed + call, LIVE, n, hi=1.5)

    out: list[dict[str, Any]] = []
    emitted = 0
    for k in range(14):
        n0 = len(seen)
        s = w.decode(fn)
        if k == 0:
            require_admitted(s, xargs)
        hl = int(s.row("accepted_canvas_history_len", "c"))
        h = s.row("accepted_canvas_history", "c")
        ids = s.committed("c")
        out.append(
            {
                "call": s.call,
                "entered_commit": s.entered_commit["c"],
                "hist": [int(t) for t in h[(hl - 1) % h.shape[0]]] if hl else None,
                "argmax": row(s, "argmax_canvas"),
                "committed": ids,
                "cands": seen[n0:],
                "emitted": emitted,
            }
        )
        if ids is not None:
            emitted += len(ids)
            if emitted >= max_tokens:
                break
    return out, (g, fn)


@pytest.mark.parametrize("max_tokens", LAG_MAX_TOKENS)
@pytest.mark.parametrize("arm", [QA, QA_SC])
def test_history_is_invariant_to_late_reports(
    arm: str, max_tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review A (I1/T1): reading the device reports ≥ 2 calls late (the normal CUDA path) changes only how many
    L_eff candidates the host enumerates, never the result: on every call the history row, ``argmax_canvas`` and
    the committed ids equal the on-time run's (seeds 1-6). Liveness: the lagged runs enumerate > 1 candidate
    on some call. Kills M1 (history from the first candidate) and M2 (from the last); ``perstep_mar`` passes
    this row by construction (``TWIN`` control below)."""
    bad: list[str] = []
    multi = 0
    for seed in LAG_SEEDS:
        a, _ = _lag_run(arm, seed, max_tokens, False, monkeypatch)
        b, _ = _lag_run(arm, seed, max_tokens, True, monkeypatch)
        multi += sum(1 for c in b if any(len(x) > 1 for x in c["cands"]))
        strip = [{k: v for k, v in c.items() if k != "cands"} for c in a]
        strip_b = [{k: v for k, v in c.items() if k != "cands"} for c in b]
        if strip != strip_b:
            first = next(
                (i for i, (x, y) in enumerate(zip(strip, strip_b, strict=False)) if x != y),
                min(len(strip), len(strip_b)),
            )
            bad.append(
                f"seed {seed}: first difference at call {first}: on time "
                f"{strip[first] if first < len(strip) else None}, lagged "
                f"{strip_b[first] if first < len(strip_b) else None} (cands {b[first]['cands'] if first < len(b) else None})"
            )
    assert multi > 0, "liveness: no lagged call enumerated more than one L_eff candidate"
    assert not bad, f"{arm} max_tokens {max_tokens}: lagged != on time\n" + "\n".join(bad)


@pytest.mark.parametrize("max_tokens", (7, 10))
def test_lag_control_perstep_mar_is_invariant(
    max_tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control for the row above: the same probe on ``perstep_mar`` (history = MAP, selected per candidate)."""
    for seed in LAG_SEEDS:
        a, _ = _lag_run("mar", seed, max_tokens, False, monkeypatch)
        b, _ = _lag_run("mar", seed, max_tokens, True, monkeypatch)
        strip = lambda rows: [{k: v for k, v in c.items() if k != "cands"} for c in rows]  # noqa: E731
        assert strip(a) == strip(b), f"seed {seed}: perstep_mar is not lag-invariant"


@pytest.mark.parametrize("arm", [QA, QA_SC])
def test_lagged_history_is_argmax_of_q_at_the_true_l_eff(
    arm: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review A (I1/T1), against the fp64 oracle: with reports ≥ 2 calls late, on every denoise call of a canvas
    whose true ``L_eff = min(CL, max_tokens − emitted)`` is ≥ 1, the history row on ``[0, L_eff)`` equals the argmax
    of ``brute_q`` at that L_eff (start = host replay of the committed ids, ``R_after = max_tokens − emitted −
    L_eff``). Liveness: some checked call enumerated a candidate list whose first entry is not the true L_eff
    (kills M1, keep the first candidate) and some whose last entry is not (kills M2, always the last)."""
    checked = not_first = not_last = 0
    bad: list[str] = []
    for seed in LAG_ORACLE_SEEDS:
        for max_tokens in LAG_ORACLE_MAX_TOKENS:
            calls, (g, fn) = _lag_run(arm, seed, max_tokens, True, monkeypatch)
            start: set[Any] = {0}
            for c in calls:
                l_eff = max(0, min(LAG_CL, max_tokens - c["emitted"]))
                if not c["entered_commit"] and l_eff >= 1 and c["hist"] is not None:
                    x = fn(c["call"], "c", LAG_CL)
                    r_after = max_tokens - c["emitted"] - l_eff
                    try:
                        want = ps.argmax_q(
                            ps.brute_q(g, x, sorted(start, key=str), l_eff, r_after), min_gap=1e-4
                        )
                    except AssertionError:
                        want = None  # a near-tie on the oracle: not decidable by rounding, skip
                    if want is not None:
                        checked += 1
                        if c["hist"][:l_eff] != want:
                            bad.append(
                                f"seed {seed} max_tokens {max_tokens} call {c['call']}: history "
                                f"{c['hist']} != argmax q {want} at L_eff {l_eff} (cands {c['cands']})"
                            )
                        for cand in c["cands"]:
                            if len(cand) > 1:
                                not_first += cand[0] != l_eff
                                not_last += cand[-1] != l_eff
                if c["committed"] is not None:
                    start = g.run(start, c["committed"])
    assert checked >= 15, f"fixture: only {checked} oracle-checked calls"
    assert not_first > 0 and not_last > 0, (
        f"liveness: candidate lists where the true L_eff is not first {not_first}, not last {not_last}"
    )
    assert not bad, f"{arm}: {len(bad)} lagged history rows off the oracle:\n" + "\n".join(bad[:10])
