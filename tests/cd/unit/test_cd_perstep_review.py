"""T-0068 review rows (two reviews, 2026-09-30): mutants that survived the first rows.

1. P4: positions ≥ L_eff use H(p) (and p for self-conditioning), never the marginals' tail rows.
2. Lagged reports: the per-step L_eff candidate the device picks is the one whose H(q) and joint draw are used.
3. Criterion 7 on later and short canvases: ``mean_accepted_per_step`` counts [0, L_eff) only and restarts per canvas.
4. ``hist_len >= ST``: two consecutive canvases with the same MAP must not converge after one step.
5. P1/P13 admission: end_projection with bound/threshold overrides, and a per-step arm with logprobs/top_k/top_p,
   end loudly as ``unsupported_option``; a per-step arm with overrides does not.
6. P3: the temperature is stock's at the slot's step **on entry**.
7. A truncated canvas (``valid_canvas_len < CL``): padded rows are stock's uniform rows (H = log V) in ``confident``
   and the accept rule, and an accepted padded row holds a uniform id.
8. Per-step ``empty_delta`` at the commit is loud (δ* injection as T-0067's row).

Expected values are hand-derived (stated per row) from product languages whose ``q`` is exact; T = 1 unless stated;
CPU stand-in, CL = 4, V = 262 144, stability_threshold 1 (ST = 2).
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from tests.cd.fakes.diffusion_sampler import EOS, Cfg, Snap, World, cd_module
from tests.cd.unit import _perstep_support as ps
from tests.cd.unit._sampler_support import G1, LIVE, Grammar, Records, patch_everywhere
from tests.cd.unit._scenario import Spec, canvas_logits, commit_calls
from tests.cd.unit.test_cd_arms_semantics import canvas, t1
from tests.cd.unit.test_cd_end_projection import _lagging
from tests.cd.unit.test_cd_loud_failures import _expect_loud

MAR_ARMS = ["mar", "mar_sc"]
ARMS = sorted(ps.ARMS)


def _drive(
    cfg: Cfg,
    xargs: dict[str, Any],
    fn: Any,
    n: int,
    *,
    max_tokens: int = 4096,
    lag: bool = False,
    valid_len: int | None = None,
) -> list[Snap]:
    w = World(cfg, cd=True, seed=0)
    w.add("c", xargs=xargs, max_tokens=max_tokens)
    if valid_len is not None:
        w.reqs["c"].valid_len = valid_len
    if lag:
        w.call_ctx = _lagging(w)
    w.prefill()
    return [w.decode(fn) for _ in range(n)]


# --------------------------------------------------------------------------- 1 + 2: the tail beyond L_eff, lag


#: chain 1000..1004, F after 5 ids: d(start) = 6 = max_tokens. Canvas 0 (L_eff 4) is forced to 1000..1003;
#: canvas 1 has L_eff = 2 and R_after = 0: position 0 forced to 1004, position 1 forced to a stop (EOS dominates).
CHAIN5 = [[1000], [1001], [1002], [1003], [1004]]
#: every call: rows 0-1 flat over the live ids with EOS at +10 on row 1; row 2 flat over 8 live ids (H(p) = ln 8),
#: row 3 flat over 4 (H(p) = ln 4)
TAIL_ROWS = [
    ps.flat(LIVE),
    {**ps.flat(LIVE), EOS: 10.0},
    ps.flat(LIVE),
    ps.flat(LIVE[:4]),
]
#: canvas 1, bound 0.5: h = [H(q_0) = 0, H(q_1) ≈ 0, ln 8, ln 4] → sorted 0, 0, ln 4 (row 3), ln 8 (row 2)
TAIL_MASK = [True, True, False, True]


def test_tail_premise() -> None:
    x = ps.logits_rows(TAIL_ROWS)
    h = [
        0.0,
        ps.entropy64(ps.q_product64(x[1], [1, 50, 106])),
        ps.entropy64(ps.p64(x[2])),
        ps.entropy64(ps.p64(x[3])),
    ]
    assert h[1] < 1e-3 and abs(h[2] - math.log(8)) < 1e-3 and abs(h[3] - math.log(4)) < 1e-3
    assert ps.stock_rule(h, 0.5) == TAIL_MASK and ps.margin(h, 0.5) > 0.05
    assert sum(h) / 4 > 0.5 + 0.3  # canvas 1 is not confident at threshold 0.5


@pytest.mark.parametrize("lag", [False, True], ids=["read", "lagged"])
@pytest.mark.parametrize("arm", MAR_ARMS)
def test_positions_beyond_l_eff_use_h_of_p(arm: str, lag: bool) -> None:
    """max_tokens = CL + 2: canvas 1 has L_eff = 2. Its first denoise call: mask [T, T, F, T] with rows 0-1 holding
    the forced 1004 and EOS, row 3 a live id, row 2 a uniform id; ``confident`` False (mean ≈ 0.87 > 0.5).
    ``lagged``: every report is read ≥ 2 calls late (T-0067's ``_lagging``), so several L_eff candidates are
    computed and the device must use the right one's H(q) and draw."""
    g = ps.product_grammar(CHAIN5, "t0068-chain5")
    x = ps.logits_rows(TAIL_ROWS)
    xargs = ps.ps_xargs(g, f"tail-{arm}-{lag}", arm, confidence_threshold=0.5)
    snaps = _drive(
        t1(entropy_bound=0.5, max_denoising_steps=30),
        xargs,
        lambda c, r, n: x[:n].clone(),
        7,
        max_tokens=6,
        lag=lag,
    )
    commits = commit_calls(snaps, "c")
    assert commits[:1] == [2], (
        f"canvas 0 (H(q) = 0, confident) must commit at call 2, commits {commits}"
    )
    assert snaps[2].committed("c") == [1000, 1001, 1002, 1003]
    for k in (3, 4, 5):
        s = snaps[k]
        assert s.committed("c") is None and not bool(s.row("is_encoder_phase", "c")), (
            f"{arm} call {k}: canvas 1 converged; its tail H(p) keeps it unconfident"
        )
        assert not bool(s.row("confident", "c")), (
            f"{arm} call {k}: confident on canvas 1 (tail must be H(p))"
        )
        c = canvas(s)
        assert c[0] == 1004 and c[1] == EOS, (
            f"{arm} call {k}: scored rows must hold the forced draw: {c}"
        )
        assert c[3] in LIVE[:4] and ps.renoised(c[2]), (
            f"{arm} call {k}: tail mask must be [F, T]: {c}"
        )


# --------------------------------------------------------------------------- 3: records on later / short canvases


@pytest.mark.parametrize("arm", ARMS)
def test_record_means_over_three_canvases(arm: str, caplog: Any, capsys: Any) -> None:
    """Chain 1000..1008 (F after 9 ids), max_tokens = 2 CL + 2 = 10, every row flat over the live ids with EOS at +10
    (so every H(p) ≈ 0.004 and every H(q) ≈ 0; always confident): canvases 0 and 1 are forced (L_eff 4, all four
    accepted each step: mean 4.0); canvas 2 has L_eff 2 (1008, then a stop): its two scored rows are accepted and
    the accepted tail rows do not count: mean 2.0. Each canvas has 2 denoise steps."""
    g = ps.product_grammar([[1000 + i] for i in range(9)], "t0068-chain9")
    x = ps.logits_rows([{**ps.flat(LIVE), EOS: 10.0}] * 4)
    records = Records(caplog, capsys)
    tag = f"rec3-{arm}"
    xargs = ps.ps_xargs(g, tag, arm, confidence_threshold=2.0)
    snaps = _drive(t1(entropy_bound=0.5), xargs, lambda c, r, n: x[:n].clone(), 11, max_tokens=10)
    assert commit_calls(snaps, "c")[:3] == [2, 5, 8], commit_calls(snaps, "c")
    by_idx = {r["canvas_idx"]: r for r in records.tagged(tag)}
    assert sorted(by_idx)[:3] == [0, 1, 2], by_idx
    for k, want in ((0, 4.0), (1, 4.0), (2, 2.0)):
        rec = by_idx[k]
        assert rec["error"] is None and rec["steps"] == 2, rec
        got = rec["mean_accepted_per_step"]
        assert got is not None and abs(got - want) < 1e-9, (
            f"{arm} canvas {k}: mean {got}, expected {want}"
        )
    assert list(by_idx[2]["committed_ids"])[:1] == [1008]


# --------------------------------------------------------------------------- 4: hist_len >= ST


@pytest.mark.parametrize("arm", ARMS)
def test_same_map_on_consecutive_canvases_needs_st_steps(arm: str) -> None:
    """Loop language over the live ids, constant logits peaked on 1003: every canvas's MAP is the same row, so the
    history rows left by canvas k already agree with canvas k+1's first MAP. Convergence still needs ST = 2 denoise
    steps per canvas: commits at calls 2, 5, 8 (a missing ``hist_len >= ST`` guard commits at 2, 4, 6)."""
    g = ps.loop_grammar(LIVE, "t0068-loop-same")
    x = ps.logits_rows([{**ps.flat(LIVE), 1003: 20.0}] * 4)
    xargs = ps.ps_xargs(g, f"same-{arm}", arm, confidence_threshold=ps.ALWAYS_CONFIDENT)
    snaps = _drive(t1(), xargs, lambda c, r, n: x[:n].clone(), 9)
    assert commit_calls(snaps, "c")[:3] == [2, 5, 8], commit_calls(snaps, "c")
    assert all(snaps[i].committed("c") == [1003] * 4 for i in (2, 5, 8))


# --------------------------------------------------------------------------- 5: admission (P1 / P13)


def _first_error(
    arm_mode: tuple[str, str],
    extra: dict[str, Any],
    params: dict[str, Any],
    tag: str,
    caplog: Any,
    capsys: Any,
) -> list[str]:
    g = Grammar(G1, "t0068-adm")
    records = Records(caplog, capsys)
    from tests.cd import _wire_support as ws

    mode, sc = arm_mode
    xargs = ws.raw_xargs(g.payload, mode=mode, self_conditioning=sc, record_tag=tag, **extra)
    w = World(Cfg(CL=4, stability_threshold=1), cd=True, seed=0)
    w.add("c", xargs=xargs, **params)
    w.prefill()
    fn = canvas_logits({"c": 5}, period=3)
    for _ in range(4):
        w.decode(fn)
    w.remove("c")
    return [r["error"] for r in records.tagged(tag) if r.get("error")]


@pytest.mark.parametrize(
    "extra", [{"entropy_bound": 0.3}, {"confidence_threshold": 0.2}], ids=["bound", "thr"]
)
def test_end_projection_with_overrides_is_unsupported(
    extra: dict[str, Any], caplog: Any, capsys: Any
) -> None:
    errs = _first_error(("end_projection", "model"), extra, {}, "adm-ep", caplog, capsys)
    assert errs == ["unsupported_option"], errs


@pytest.mark.parametrize(
    "params", [{"logprobs": 1}, {"top_k": 3}, {"top_p": 0.7}], ids=["logprobs", "top_k", "top_p"]
)
@pytest.mark.parametrize("arm", ARMS)
def test_perstep_with_unsupported_sampling_options_is_loud(
    arm: str, params: dict[str, Any], caplog: Any, capsys: Any
) -> None:
    errs = _first_error(ps.ARMS[arm], {}, params, f"adm-{arm}", caplog, capsys)
    assert errs == ["unsupported_option"], errs


@pytest.mark.parametrize("arm", ARMS)
def test_perstep_with_overrides_is_admitted(arm: str, caplog: Any, capsys: Any) -> None:
    """The guard's other side: bound and threshold overrides are the per-step arms' own knobs (P1/P2)."""
    extra = {"entropy_bound": 0.3, "confidence_threshold": 0.2}
    assert _first_error(ps.ARMS[arm], extra, {}, f"adm-ok-{arm}", caplog, capsys) == []


# --------------------------------------------------------------------------- 6: temperature at the entry step


@pytest.mark.parametrize("arm", MAR_ARMS)
def test_temperature_uses_the_entry_step(arm: str) -> None:
    """t_min 0.5, t_max 1.0, max_denoising_steps 2: the first denoise call enters at step 0, T = 1.0
    (T = 0.75 if the step were taken after its increment). Rows 0-1 over {1000: 0, 1001: 1}: H(q) = 0.582 at
    T = 1 and 0.512 at T = 0.75; bound 0.55 → exactly one of rows 0, 1 rejected at T = 1, none at 0.75."""
    sets = [[1000, 1001], [1000, 1001], [1002], [1003]]
    g = ps.product_grammar(sets, "t0068-temp-entry")
    x = ps.logits_rows([{1000: 0.0, 1001: 1.0}] * 2 + [ps.flat([1002]), ps.flat([1003])])
    h1 = [ps.entropy64(ps.q_product64(x[i], sets[i], 1.0)) for i in range(4)]
    h75 = [ps.entropy64(ps.q_product64(x[i], sets[i], 0.75)) for i in range(4)]
    assert sum(ps.stock_rule(h1, 0.55)) == 3 and all(ps.stock_rule(h75, 0.55))
    assert ps.margin(h1, 0.55) > 0.02 and ps.margin(h75, 0.55) > 0.02
    xargs = ps.ps_xargs(g, f"tentry-{arm}", arm, confidence_threshold=ps.NEVER_CONFIDENT)
    for seed_call in range(3):  # three independent first calls (fresh worlds)
        snaps = _drive(
            Cfg(
                CL=4,
                stability_threshold=1,
                t_min=0.5,
                t_max=1.0,
                max_denoising_steps=2,
                entropy_bound=0.55,
            ),
            xargs,
            lambda c, r, n: x[:n].clone(),
            1,
        )
        c = canvas(snaps[0])
        ok = [c[i] in (1000, 1001) for i in (0, 1)]
        assert ok.count(True) == 1, (
            f"{arm} run {seed_call}: {c}: T must be stock's at step 0 on entry (1.0)"
        )


# --------------------------------------------------------------------------- 7: truncated canvas


@pytest.mark.parametrize("arm", MAR_ARMS)
def test_truncated_canvas_pads_like_stock(arm: str) -> None:
    """valid_canvas_len 3 < CL 4 (L_eff 3, R_after 0): rows 0-1 forced (1000, 1001), row 2 a stop (EOS at +10);
    the padded row 3 is stock's zero-logit row, uniform, H = log V = 12.48. Every row accepted (bound 1e6).
    ``confident`` = mean over CL = 12.48 / 4 > 1.0: False (a pad H of 0, or a mean over the valid rows only,
    makes it True); row 3 holds a uniform id, not a constant 0."""
    g = ps.product_grammar([[1000], [1001]], "t0068-trunc")
    x = ps.logits_rows([ps.flat(LIVE), ps.flat(LIVE), {**ps.flat(LIVE), EOS: 10.0}])
    xargs = ps.ps_xargs(
        g, f"trunc-{arm}", arm, entropy_bound=ps.ACCEPT_ALL, confidence_threshold=1.0
    )
    snaps = _drive(t1(max_denoising_steps=30), xargs, lambda c, r, n: x[:n].clone(), 4, valid_len=3)
    pads = []
    for k, s in enumerate(snaps):
        assert s.committed("c") is None and not bool(s.row("confident", "c")), (
            f"{arm} call {k}: confident/converged; the padded row's H = log V must count in the mean"
        )
        c = canvas(s)
        assert c[:3] == [1000, 1001, EOS], f"{arm} call {k}: {c}"
        pads.append(c[3])
    assert len(set(pads)) == len(pads) and 0 not in pads, (
        f"padded row must hold uniform ids, got {pads}"
    )


# --------------------------------------------------------------------------- 8: empty δ* at a per-step commit


@pytest.mark.parametrize("arm", ARMS)
def test_perstep_empty_delta_star_is_loud(
    arm: str, caplog: Any, capsys: Any, monkeypatch: Any
) -> None:
    budget = cd_module("budget")
    cd_module("sampler")
    ps.steps_module()
    calls = {"n": 0}

    def empty_delta_star(tables: Any, states: Any, committed_ids: Any, stop_ids: Any) -> Any:
        calls["n"] += 1
        z = torch.zeros_like(states, dtype=torch.bool)
        return budget.DeltaStar(states=z, empty=torch.tensor(True))

    patch_everywhere(monkeypatch, "delta_star", empty_delta_star)
    tag = f"loud-delta-{arm}"
    g = Grammar(G1, "t0068-delta")
    _expect_loud(
        Cfg(CL=4, stability_threshold=1),
        Spec("c", xargs=ps.ps_xargs(g, tag, arm, confidence_threshold=ps.ALWAYS_CONFIDENT)),
        canvas_logits({"c": 11, "u1": 12, "u2": 13}, period=3),
        trigger=2,
        records=Records(caplog, capsys),
        code="empty_delta",
        tag=tag,
    )
    assert calls["n"] >= 1, "cannot inject: δ* was not computed through T-0065's budget.delta_star"
