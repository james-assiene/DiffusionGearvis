"""T-0065 criterion 2 support: direct rows for the compensated arithmetic of ``semiring``.

The fp32 H(q) bar of criterion 2 is met only because ``log_matmul(compensated=True)`` (TwoSum error of
every ``A_ij + B_jk`` carried into the exponent, and a first-order residual on ``log s`` in the
finish) and ``compensated_sum`` (pairwise TwoSum) remove roundings that the differential rows are
too coarse to see one at a time (re-review 2026-09-29: each piece could be dropped with all 28 rows
green). These rows pin each piece directly against float64 / ``math.fsum`` oracles, each with a named
uncompensated control that is red on the same input.

Inputs, and why these (measured on 0440e65; ulp = fp32 ulp of the oracle value):
- ``wide`` entries in [-24, -16] (30 % sentinels, inner dim 64/128): |A_ij + B_jk| rounds on a grid
  coarser than either operand, so the TwoSum error is ≈ 0.5 ulp of the result. Compensated: max
  ≈ 0.59 ulp, ≤ 2 % of entries > 0.51 ulp; without the error term: max ≥ 1.17 ulp, ~16 % > 0.51.
- ``gauged`` entries in [-3.35, -2.35], inner dim 300: results near 0 with ``log s ≈ −max`` (the pass-2
  regime), where the residual on ``log s`` matters. Absolute error in ulp(1.0) = 2^-23: compensated
  mean ≈ 0.48, max ≈ 2.06; without the residual mean ≈ 1.05, max ≥ 3.2.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.cd.unit._core_support import CORE_MODULES, core, mask, random_toy, tables_for

ULP1 = 2.0**-23  # fp32 ulp of 1.0


def _lse_oracle(a, b) -> np.ndarray:
    """float64: ``log Σ_j exp(A_ij + B_jk)``, the sum of two fp32 values is exact in float64."""
    t = a.double().numpy()[:, :, None] + b.double().numpy()[None, :, :]
    live = t > -1e29
    m = np.where(live, t, -np.inf).max(axis=1)
    s = np.where(live, np.exp(t - m[:, None, :]), 0.0)
    out = np.empty(m.shape)
    for i in range(m.shape[0]):
        for k in range(m.shape[1]):
            out[i, k] = m[i, k] + math.log(math.fsum(s[i, :, k])) if np.isfinite(m[i, k]) else -1e30
    return out


def _operands(lo: float, hi: float, inner: int, seed: int, n: int = 64):
    import torch

    g = torch.Generator().manual_seed(seed)
    a = (lo + (hi - lo) * torch.rand(n, inner, generator=g)).float()
    b = (lo + (hi - lo) * torch.rand(inner, n, generator=g)).float()
    a[torch.rand(n, inner, generator=g) < 0.3] = -1e30
    b[torch.rand(inner, n, generator=g) < 0.3] = -1e30
    return a, b


def _ulp32(x: np.ndarray) -> np.ndarray:
    import torch

    e = torch.tensor(x, dtype=torch.float32).abs()
    return (torch.nextafter(e, torch.tensor(float("inf"))) - e).double().numpy()


def _wide_ok(got, exp) -> tuple[bool, str]:
    u = np.abs(got.double().numpy() - exp) / _ulp32(exp)
    frac = float(np.mean(u > 0.51))
    return bool(
        u.max() <= 1.0 and frac <= 0.05
    ), f"max {u.max():.2f} ulp, {frac:.1%} of entries > 0.51 ulp"


@pytest.mark.parametrize("seed", [0, 1], ids=lambda s: f"seed{s}")
@pytest.mark.parametrize("inner", [64, 128], ids=lambda m: f"inner{m}")
def test_log_matmul_compensated_within_one_ulp_wide(inner, seed):
    sr = core("semiring")
    a, b = _operands(-24.0, -16.0, inner, seed)
    exp = _lse_oracle(a, b)
    got = sr.log_matmul(a, b, compensated=True)
    ok, msg = _wide_ok(got, exp)
    assert ok, f"compensated log_matmul: {msg} (bar: max ≤ 1 ulp, ≤ 5 % > 0.51 ulp)"
    ok_c, msg_c = _wide_ok(sr.log_matmul(a, b, compensated=False), exp)
    assert not ok_c, f"control (compensated=False) is not red on this input: {msg_c}"


def _gauged_stats(got, exp) -> tuple[float, float]:
    err = np.abs(got.double().numpy() - exp) / ULP1
    return float(err.mean()), float(err.max())


@pytest.mark.parametrize("seed", [0, 1], ids=lambda s: f"seed{s}")
def test_log_matmul_compensated_gauged_near_zero(seed):
    sr = core("semiring")
    a, b = _operands(-3.35, -2.35, 300, seed)
    exp = _lse_oracle(a, b)
    assert np.abs(exp).max() < 1.0  # the regime: results near 0, log s ≈ −max
    mean, mx = _gauged_stats(sr.log_matmul(a, b, compensated=True), exp)
    assert mean <= 0.75 and mx <= 2.5, (
        f"compensated log_matmul: mean {mean:.2f}, max {mx:.2f} ulp(1.0) (bar: 0.75 / 2.5)"
    )
    c_mean, c_max = _gauged_stats(sr.log_matmul(a, b, compensated=False), exp)
    assert c_mean > 0.75 or c_max > 2.5, f"control not red: mean {c_mean:.2f}, max {c_max:.2f}"


def _fsum_ulps(got, x) -> tuple[float, float]:
    import torch

    errs = []
    for g, row in zip(got.double().tolist(), x, strict=True):
        e = math.fsum(row.double().tolist())
        e32 = torch.tensor(e, dtype=torch.float32).abs()
        ulp = float(torch.nextafter(e32, torch.tensor(float("inf"))) - e32)
        errs.append(abs(g - e) / ulp)
    return max(errs), sum(errs) / len(errs)


def _cancelling(seed: int, rows: int = 8, n_big: int = 2048, n_small: int = 2048):
    """Big values of both signs that cancel exactly (±1e2..1e6) around a small positive remainder."""
    import torch

    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(rows):
        sign = torch.where(torch.rand(n_big, generator=g) < 0.5, 1.0, -1.0)
        big = (10 ** (2 + 4 * torch.rand(n_big, generator=g))).float() * sign
        x = torch.cat([big, -big, torch.rand(n_small, generator=g)])
        out.append(x[torch.randperm(x.numel(), generator=g)])
    return torch.stack(out)


@pytest.mark.parametrize("seed", [0, 1], ids=lambda s: f"seed{s}")
def test_compensated_sum_under_cancellation(seed):
    x = _cancelling(seed)
    mx, _ = _fsum_ulps(core("semiring").compensated_sum(x), x)
    assert mx <= 1.0, f"compensated_sum: max {mx:.2f} ulp vs math.fsum"
    c_mx, _ = _fsum_ulps(x.sum(dim=-1), x)
    assert c_mx > 1.0, f"control (naive fp32 sum) is not red: {c_mx:.2f} ulp"


@pytest.mark.parametrize("seed", [0, 1], ids=lambda s: f"seed{s}")
def test_compensated_sum_over_a_vocabulary_row(seed):
    """V = 262 144 positive terms spanning 20 nats (a q̃ row): plain fp32 reduction is ~1 ulp,
    compensated ≤ ~0.5 (measured max 0.47, mean ≤ 0.35; without the carried error max ≥ 0.91,
    mean ≥ 0.53)."""
    import torch

    g = torch.Generator().manual_seed(seed)
    x = torch.exp(-20.0 * torch.rand(4, 262_144, generator=g)).float()
    mx, mean = _fsum_ulps(core("semiring").compensated_sum(x), x)
    assert mx <= 0.75 and mean <= 0.45, f"compensated_sum: max {mx:.2f}, mean {mean:.3f} ulp"
    c_mx, c_mean = _fsum_ulps(x.sum(dim=-1), x)
    assert c_mx > 0.75 or c_mean > 0.45, f"control not red: max {c_mx:.2f}, mean {c_mean:.3f}"


def test_pass2_uses_compensated_log_matmul(monkeypatch):
    """A refactor must not silently drop the compensated option from constrained_marginals' pass 2.

    Every ``log_matmul`` name bound in the core modules is wrapped (delegating to the real one) and
    the ``compensated`` flag of each call is recorded.
    """
    import random

    import torch

    calls: list[bool] = []
    real = core("semiring").log_matmul

    def spy(*args, **kwargs):
        calls.append(bool(kwargs.get("compensated", False)))
        return real(*args, **kwargs)

    for name in CORE_MODULES:
        mod = core(name)
        if getattr(mod, "log_matmul", None) is real:
            monkeypatch.setattr(mod, "log_matmul", spy)
    rng = random.Random(4)
    toy = random_toy(rng, V=16, S=5, nfa=True)
    logits = torch.tensor([[rng.uniform(-10, 0) for _ in range(16)] for _ in range(8)])
    tables = tables_for(toy, bucket=8)
    log_m = core("emission").log_transfer(logits, tables, 8)
    tree = core("tree").up_sweep(log_m)
    calls.clear()
    core("marginals").constrained_marginals(
        logits, tables, tree, mask({0, 1}, 8), mask(sorted(toy.finals), 8), 8
    )
    assert calls, "constrained_marginals made no log_matmul call through the spied names"
    assert any(calls), (
        f"no compensated log_matmul call in constrained_marginals ({len(calls)} calls)"
    )
