"""T-0077c criterion 4 on CPU, for the **spec order** itself (pure torch; no kernel): the order the kernel is pinned to
meets N4's bars, and the inputs can tell the realistic wrong orders apart.

The spec emulation is ``_frozen_t0077c.spec_log_matmul`` (transcribed from the task's "Implementation notes", steps
1-5). The kernel is compared with it bit for bit on CUDA (3a) and under the interpreter (3c); this module shows that
what the kernel is pinned to is worth being pinned to, and that the instances have the power to kill a wrong order:

- **Anchors** (the emulation is the task's spec, not an approximation of something else): a hand row; ``m ∈ {1, 2}``
  bit-identical to the branch-base torch ``log_matmul`` (a sum of ≤ 2 terms has one order, and TwoSum's ``s + c``
  rounds back to ``s``), so the per-term formulas are ``semiring``'s; the dead set (output == sentinel) and NaN
  positions equal to the torch path's on every instance; the fp32 ``DEAD`` comparison at fp32(DEAD) ± 1 ulp; T-0065's
  generator reproduced.
- **Bars** (criterion 4 against fp64 + ``math.fsum`` on fp32-rounded inputs, T-0065's metrics): compensated wide max
  ≤ 1.0 ulp and ≤ 5 % > 0.51 ulp; gauged mean ≤ 0.75 and max ≤ 2.5 ulp(1.0); at inner {32, 91, 205, 400, 512},
  outer 64, seeds 0 and 1 (O8: inner 32 gated); two-band (every live fp64 entry live and within the wide bar, no
  spurious sentinel); all-sentinel rows/columns exactly the sentinel. Plain (O6): max ≤ and mean ≤ the branch-base
  torch CPU plain path per (regime, inner, seed); the share > 0.51 ulp is printed.
- **Power controls** (task "Tests to write", (i)-(ix), plus (iii-b)): each differs bitwise from the spec on ≥ 1
  pinned instance; the precision ones also fail a bar: (i) plain sequential fails gauged at inner ≥ 91 and the wide
  5 % bar at inner ≥ 205; (iv) fails wide at T-0065's inner 64/128; (v) fails gauged at T-0065's inner 300; (vi)
  turns live two-band entries dead; and ``compensated=False`` fails the compensated bars on T-0065's inputs.

Regime: CPU, torch 2.13.0+cpu, numpy 2.4.6 (``uv.lock``), fp32, ``OMP_NUM_THREADS=4``; T-0065's generator, outer 64,
30 % sentinels, seeds 0/1; near-tie pins in ``_frozen_t0077c.NEAR_TIE_PINS``. Measured 2026-10-05 (test-author):
spec gauged mean/max at inner 32…512 = 0.64/2.16, 0.39/2.10, 0.34/1.44, 0.30/1.19, 0.31/1.16 (seed max), wide ≤ 0.58 ulp
/ ≤ 2.0 % — the planner's numbers, reproduced.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tests.cd.unit import _frozen_t0077c as c

ROUTES = {"torch_cpu": (torch.exp, torch.log), "np64_O2": (c.np64_exp, c.np64_log)}
SENT = c.SENTINEL32


def _cell(regime: str, inner: int, seed: int) -> str:
    return f"{regime}-inner{inner}-seed{seed}"


CELLS = [(r, i, s) for r in c.REGIMES for i in c.INNERS for s in c.SEEDS]


def spec(
    a: torch.Tensor, b: torch.Tensor, compensated: bool, route: str = "torch_cpu"
) -> torch.Tensor:
    ex, lg = ROUTES[route]
    return c.spec_log_matmul(a, b, compensated=compensated, exp=ex, log=lg)


def torch_path(a: torch.Tensor, b: torch.Tensor, compensated: bool) -> torch.Tensor:
    return c.frozen("semiring").log_matmul(a, b, compensated=compensated)


# --------------------------------------------------------------------------- anchors


def test_t0065_generator_and_oracle_are_reproduced() -> None:
    """``t0065_operands``/``lse_oracle`` are T-0065's ``_operands``/``_lse_oracle`` (same draws, same oracle)."""
    from tests.cd.unit import test_core_compensated as t65

    for lo, hi, inner, seed in ((-24.0, -16.0, 64, 0), (-3.35, -2.35, 300, 1)):
        a, b = c.t0065_operands(lo, hi, inner, seed)
        a0, b0 = t65._operands(lo, hi, inner, seed)
        assert torch.equal(a, a0) and torch.equal(b, b0)
        got = c.lse_oracle(a.unsqueeze(0), b.unsqueeze(0))[0]
        assert np.array_equal(got, t65._lse_oracle(a0, b0))


def test_hand_row() -> None:
    """``[0, 0] ⊗ [0, 0]ᵀ`` = log 2; ``[0, −1e30] ⊗ [−1e30, 0]ᵀ`` = sentinel; ``[x] ⊗ [y]`` with one term is ``x + y``
    within 1 ulp (S_tot = exp(err) ≈ 1). By hand, fp64."""
    z = torch.zeros
    for comp in (True, False):
        out = spec(z(1, 1, 2), z(1, 2, 1), comp)
        assert abs(float(out) - math.log(2.0)) <= 2.0**-24, float(out)
        dead = spec(torch.tensor([[[0.0, SENT]]]), torch.tensor([[[SENT], [0.0]]]), comp)
        assert dead.view(torch.int32).item() == torch.tensor(SENT).view(torch.int32).item()
        one = spec(torch.tensor([[[-3.25]]]), torch.tensor([[[-1.5]]]), comp)
        assert float(one) == -4.75  # exact sum, S_tot = exp(0) = 1, log 1 = 0


@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("compensated", [True, False], ids=["comp", "plain"])
def test_spec_equals_the_torch_path_where_the_sum_has_one_order(m: int, compensated: bool) -> None:
    """A sum of ≤ 2 non-negative terms has one order and TwoSum's ``fl(s + c)`` returns ``s``: the spec must then be
    the branch-base ``semiring.log_matmul`` bit for bit (same anchor, same per-term formulas, same ``_finish``) —
    the anchor that the emulation transcribes ``semiring``'s arithmetic, not something else."""
    for regime in c.REGIMES:
        lo, hi = c.REGIMES[regime]
        a, b = c.t0065_operands(lo, hi, m, 3, n=48)
        a3, b3 = a.unsqueeze(0), b.unsqueeze(0)
        assert c.bit_diff(torch_path(a3, b3, compensated), spec(a3, b3, compensated)) is None, (
            regime
        )


def _instances() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    out = {f"two_band{m}": c.two_band(m, 0) for m in (32, 91, 205)}
    out |= {f"all_sentinel{m}": c.all_sentinel(m, 0) for m in (1, 3, 33)}
    out |= {f"nan_inf{m}": c.nan_inf(m, 0) for m in (3, 33)}
    out["near_dead"] = c.near_dead()
    out["live_block"] = c.live_block_views(3, 33, 40, 0)
    out["vec_mat"] = c.vec_mat(3, 33, 40, 0)
    out["mat_vec"] = c.mat_vec(3, 33, 40, 0)
    out["broadcast"] = c.broadcast_batch(3, 5, 7, 4, 0)
    out |= {f"near_tie{m}": c.near_tie_pinned(m) for m in c.NEAR_TIE_PINS}
    return out


INSTANCES = _instances()


@pytest.mark.parametrize("name", list(INSTANCES))
@pytest.mark.parametrize("compensated", [True, False], ids=["comp", "plain"])
def test_dead_set_and_nan_positions_follow_the_torch_path(name: str, compensated: bool) -> None:
    """Task notes: NaN/±inf follow the torch path (same per-term ops); criterion 4: the dead set (output ==
    sentinel) equals the torch path's."""
    a, b = INSTANCES[name]
    want, got = torch_path(a, b, compensated), spec(a, b, compensated)
    assert torch.equal(torch.isnan(want), torch.isnan(got)), name
    assert torch.equal(want == SENT, got == SENT), name


def test_dead_threshold_is_the_fp32_comparison() -> None:
    """``live = mx > DEAD`` with DEAD as fp32: an anchor of exactly fp32(DEAD) and one ulp below are dead, one ulp
    above is live (by hand)."""
    a, b = c.near_dead()
    out = spec(a, b, True)
    live = (out != SENT)[0, :, 0].tolist()
    assert live == [False, True, False, True], live


@pytest.mark.parametrize("m", [1, 3, 33])
def test_all_sentinel_rows_and_columns_are_exactly_the_sentinel(m: int) -> None:
    """Criterion 4: all-sentinel rows of ``a`` (1, 7), columns of ``b`` (0, 5, k − 1) and a row/column pair with
    disjoint live states → exactly the sentinel's bits; every entry the fp64 brute force calls live is live."""
    a, b = c.all_sentinel(m, 0)
    dead = torch.from_numpy(c.lse_oracle(a, b) <= -1e29)
    assert bool(dead[:, [1, 7], :].all()) and bool(dead[:, :, [0, 5, b.shape[2] - 1]].all())
    assert bool(dead[:, 3, 2].all()) and not bool(dead.all())
    sent = torch.tensor(SENT).view(torch.int32)
    for comp in (True, False):
        out = spec(a, b, comp)
        assert bool((out.view(torch.int32)[dead] == sent).all()), "a dead entry is not the sentinel"
        assert bool((out[~dead] > c.DEAD32).all()), "a live entry came out dead"


# --------------------------------------------------------------------------- bars (criterion 4)


@pytest.mark.parametrize(("regime", "inner", "seed"), CELLS, ids=[_cell(*x) for x in CELLS])
def test_spec_compensated_meets_the_n4_bar(regime: str, inner: int, seed: int) -> None:
    a, b = c.regime_instance(regime, inner, seed)
    exp = c.regime_oracle(regime, inner, seed)
    got = spec(a, b, True)
    ok, msg = (c.wide_ok if regime == "wide" else c.gauged_ok)(got, exp)
    print(f"ASTRA_T0077C_SPEC {_cell(regime, inner, seed)} compensated {msg}")
    assert ok, f"spec order, compensated, {_cell(regime, inner, seed)}: {msg}"


@pytest.mark.parametrize(("regime", "inner", "seed"), CELLS, ids=[_cell(*x) for x in CELLS])
def test_spec_plain_is_no_worse_than_the_torch_cpu_plain_path(
    regime: str, inner: int, seed: int
) -> None:
    """O6 as ruled by the human (2026-10-06): per (regime, inner, seed) max ≤ the branch-base torch plain path's (same
    input, CPU) on every regime; mean ≤ gated on gauged only; on wide the mean is reported (printed), not gated. Unit:
    wide fp32 ulp of the oracle value, gauged ulp(1.0); the share > 0.51 ulp is printed."""
    a, b = c.regime_instance(regime, inner, seed)
    exp = c.regime_oracle(regime, inner, seed)
    unit = c.ulp32(exp) if regime == "wide" else c.ULP1

    def stats(t: torch.Tensor) -> tuple[float, float, float]:
        u = np.abs(t.double().numpy() - exp) / unit
        return float(u.max()), float(u.mean()), float(np.mean(u > 0.51))

    s, t = stats(spec(a, b, False)), stats(torch_path(a, b, False))
    print(
        f"ASTRA_T0077C_PLAIN {_cell(regime, inner, seed)} spec max/mean/share {s[0]:.4f}/{s[1]:.5f}/{s[2]:.4f}"
        f" torch {t[0]:.4f}/{t[1]:.5f}/{t[2]:.4f}"
    )
    assert s[0] <= t[0], f"plain spec max {s[0]} > torch plain max {t[0]}"
    if regime == "gauged":  # O6 (human, 2026-10-06): the wide mean is reported above, not gated
        assert s[1] <= t[1], f"plain spec mean {s[1]} > torch plain mean {t[1]} (gauged)"


@pytest.mark.parametrize("m", [32, 91, 205])
@pytest.mark.parametrize("seed", list(c.SEEDS))
def test_two_band_every_live_entry_is_live_and_within_the_wide_bar(m: int, seed: int) -> None:
    """Criterion 4, two-band (``live_simple_106-63-0`` regime): no spurious sentinel; the live entries meet the wide
    bar against fp64."""
    a, b = c.two_band(m, seed)
    exp = c.lse_oracle(a, b)
    live = exp > -1e29
    assert live.any() and (exp[live] < -800).any() and (exp[live] > -30).any(), "both bands present"
    got = spec(a, b, True)
    g = got.double().numpy()
    spurious = int((live & ~(np.isfinite(g) & (g > c.DEAD32))).sum())
    assert spurious == 0, f"{spurious} live fp64 entries came out dead"
    u = np.abs(g[live] - exp[live]) / c.ulp32(exp[live])
    assert u.max() <= 1.0 and np.mean(u > 0.51) <= 0.05, (u.max(), np.mean(u > 0.51))


# --------------------------------------------------------------------------- power controls


def _differs(
    name: str, compensated: bool, route: str, insts: dict[str, tuple[torch.Tensor, torch.Tensor]]
) -> list[str]:
    ex, lg = ROUTES[route]
    out = []
    for iname, (a, b) in insts.items():
        want = spec(a, b, compensated, route)
        got = c.MUTANTS[name](a, b, compensated=compensated, exp=ex, log=lg)
        if c.bit_diff(want, got) is not None:
            out.append(iname)
    return out


def _power_instances() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    out = {f"{r}{i}": c.regime_instance(r, i, 0) for r in c.REGIMES for i in (32, 91)}
    out |= {f"near_tie{m}": c.near_tie_pinned(m) for m in c.NEAR_TIE_PINS}
    out["two_band91"] = c.two_band(91, 0)
    return out


#: the mutants that exist in each variant ((iv) and (v) change only compensated arithmetic)
POWER = [(n, True) for n in c.MUTANTS] + [
    (n, False) for n in c.MUTANTS if n not in ("iv_ab_error_dropped", "v_residual_dropped")
]


@pytest.mark.parametrize("route", list(ROUTES))
@pytest.mark.parametrize(
    ("name", "compensated"), POWER, ids=[f"{n}-{'comp' if cp else 'plain'}" for n, cp in POWER]
)
def test_power_control_each_mutant_differs_bitwise(
    name: str, compensated: bool, route: str
) -> None:
    """Without this, "bit for bit equal to the spec" proves nothing: every mutant differs on at least one pinned
    instance, under torch's CPU exp/log and under the interpreter's O2 route."""
    insts = _power_instances()
    differ = _differs(name, compensated, route, insts)
    print(
        f"ASTRA_T0077C_CONTROL {name} {'comp' if compensated else 'plain'} {route}: differs on {differ}"
    )
    assert differ, f"{name} gives the spec's bits on every pinned instance: no power"


@pytest.mark.parametrize("route", list(ROUTES))
@pytest.mark.parametrize(
    "name", ["iii_b_compensated_pair_tree", "vii_online_rescaled", "viii_j_descending"]
)
@pytest.mark.parametrize("compensated", [True, False], ids=["comp", "plain"])
def test_power_control_near_tie_pins_separate_the_precise_orders(
    name: str, compensated: bool, route: str
) -> None:
    """Reviewer A's T-0077b lesson: orders as precise as the spec give the same bits on smooth inputs (0 of 82k
    entries); each pinned near-tie set must contain a row where the order shows."""
    hits = {
        m: _differs(name, compensated, route, {f"near_tie{m}": c.near_tie_pinned(m)})
        for m in c.NEAR_TIE_PINS
    }
    assert any(hits.values()), hits
    # and they really are sums of a few live terms each, with the anchor at 0
    for m in c.NEAR_TIE_PINS:
        a, _ = c.near_tie_pinned(m)
        assert bool(((a > c.DEAD32).sum(-1) == 4).all()) and bool((a.amax(-1) == 0).all())


@pytest.mark.parametrize("inner", [91, 205, 400, 512])
@pytest.mark.parametrize("seed", list(c.SEEDS))
def test_power_control_plain_sequential_fails_the_gauged_bar(inner: int, seed: int) -> None:
    """(i) the carried error dropped fails the gauged bar at inner ≥ 91."""
    a, b = c.regime_instance("gauged", inner, seed)
    ok, msg = c.gauged_ok(
        c.MUTANTS["i_plain_sequential"](a, b, compensated=True),
        c.regime_oracle("gauged", inner, seed),
    )
    assert not ok, f"(i) passes the gauged bar at inner {inner}: {msg}"


@pytest.mark.parametrize("inner", [205, 400, 512])
@pytest.mark.parametrize("seed", list(c.SEEDS))
def test_power_control_plain_sequential_fails_the_wide_share_bar(inner: int, seed: int) -> None:
    """(i) fails the wide 5 % bar at inner ≥ 205."""
    a, b = c.regime_instance("wide", inner, seed)
    _, frac = c.wide_stats(
        c.MUTANTS["i_plain_sequential"](a, b, compensated=True),
        c.regime_oracle("wide", inner, seed),
    )
    assert frac > 0.05, f"(i) wide share {frac:.2%} ≤ 5 % at inner {inner}"


def _t65(regime: str, inner: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    lo, hi = c.REGIMES[regime]
    a, b = c.t0065_operands(lo, hi, inner, seed)
    a3, b3 = a.unsqueeze(0), b.unsqueeze(0)
    return a3, b3, c.lse_oracle(a3, b3)


@pytest.mark.parametrize("inner", [64, 128])
@pytest.mark.parametrize("seed", list(c.SEEDS))
def test_power_control_ab_error_dropped_fails_wide(inner: int, seed: int) -> None:
    """(iv) on T-0065's wide inputs; the spec passes on the same inputs."""
    a, b, exp = _t65("wide", inner, seed)
    assert c.wide_ok(spec(a, b, True), exp)[0]
    ok, msg = c.wide_ok(c.MUTANTS["iv_ab_error_dropped"](a, b, compensated=True), exp)
    assert not ok, msg


@pytest.mark.parametrize("seed", list(c.SEEDS))
def test_power_control_residual_dropped_fails_gauged_300(seed: int) -> None:
    """(v) on T-0065's gauged inner-300 input; the spec passes on the same input."""
    a, b, exp = _t65("gauged", 300, seed)
    assert c.gauged_ok(spec(a, b, True), exp)[0]
    ok, msg = c.gauged_ok(c.MUTANTS["v_residual_dropped"](a, b, compensated=True), exp)
    assert not ok, msg


@pytest.mark.parametrize(("regime", "inner"), [("wide", 64), ("wide", 128), ("gauged", 300)])
@pytest.mark.parametrize("seed", list(c.SEEDS))
def test_power_control_plain_fails_the_compensated_bar(regime: str, inner: int, seed: int) -> None:
    """Criterion 4's power control on the inputs: ``compensated=False`` fails the compensated bar (wide inner 64/128,
    gauged inner 300); the CUDA module asks the same of the kernel."""
    a, b, exp = _t65(regime, inner, seed)
    ok, msg = (c.wide_ok if regime == "wide" else c.gauged_ok)(spec(a, b, False), exp)
    assert not ok, msg


@pytest.mark.parametrize("m", [32, 91, 205])
def test_power_control_rowcol_anchors_kill_two_band_entries(m: int) -> None:
    """(vi) separate row-max(A) + col-max(B) anchors: live fp64 entries of the two-band instance come out dead."""
    a, b = c.two_band(m, 0)
    exp = c.lse_oracle(a, b)
    g = c.MUTANTS["vi_rowcol_anchors"](a, b, compensated=True).double().numpy()
    live = exp > -1e29
    spurious = int((live & ~(np.isfinite(g) & (g > c.DEAD32))).sum())
    assert spurious > 0, "(vi) keeps every two-band entry live: the instance has no power"
