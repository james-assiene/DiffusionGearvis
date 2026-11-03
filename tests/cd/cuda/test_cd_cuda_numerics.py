"""T-0073 criterion 5 and ruling D4 (tier 2, ``-m gpu``): stricter CUDA rows, run inside the pinned container by
``tests/contract/test_vllm_cd_numerics_gpu.py::test_criterion5_numerics_module_on_cuda`` (no engine boot).

- **Repeatability (D4's bonus property):** 20 same-process repeats on the device of ``constrained_marginals``,
  ``joint_draw`` (fresh generator, same seed), ``constrained_map`` and one per-step sampler call are bitwise
  identical (``tests/cd/unit/_repeat_support.py``; the CPU twin is ``tests/cd/unit/test_cd_repeatability.py``).
  Launch 2 (probe v2, 255a5e9) saw the first difference in ``class_mass`` on every repeat before the fix.
- **(a) on every repeat:** the c2 instance at T 0.4 and 0.8 meets the D2 bar on each of 20 repeats (before the fix
  the worst err/bar ranged 0.44–1.98 across repeats: pass/fail by chance).
- **c5 against the oracle:** the corrected c5 fixture (seed 13, non-empty language) — q and H(q) within T-0065 c1's
  fp32 tolerances (dq ≤ 1e-6, dH ≤ 1e-5) of the enumerator, every q row sums to 1 ± 1e-6 (an all-zero q fails).
- **c2 over seeds 0–7 × T {0.8, 0.4} × {plain, narrow_neg}** at L 256, V 4096 with the 10×-stock + 1e-6 bar, D2
  scoring (oracle and bar reference on the fp32-rounded logits). ``narrow_neg``: the model's mass sits inside a
  negated class's stored complement ``N_c`` (T-0065's regime of that name): the ids of the smallest non-empty
  ``N_c`` of the toy's negated classes are set to the softcap ceiling 30 before dividing by T.
- **Compensated ``log_matmul`` on CUDA at the CPU bars** of ``tests/cd/unit/test_core_compensated.py`` (``wide``:
  max ≤ 1 ulp and ≤ 5 % > 0.51 ulp, uncompensated control red; ``gauged``: mean ≤ 0.75, max ≤ 2.5 ulp(1.0),
  control red), over seeds 0–3. Its operands are generated in fp32, so the float64 oracle already sees the rounded
  inputs the core receives.

TF32 "high" is set globally for the module, as in the T-0067 repeats.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch

from tests.cd._quiet_logs import quiet_torch_logs
from tests.cd.cuda.test_cd_cuda_repeats import C5_SEED, accepted_over_finite_ids, c5_fixture
from tests.cd.oracle import dense64
from tests.cd.oracle import enumerate as en
from tests.cd.oracle.enumerate import Toy
from tests.cd.unit import _repeat_support as rs
from tests.cd.unit._core_support import core, random_toy
from tests.cd.unit.test_core_compensated import _gauged_stats, _lse_oracle, _operands, _wide_ok

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def quiet_torch_trace_per_row() -> Iterator[None]:
    """T-0077c: keep torch.__trace DEBUG lines (torch.compile) out of pytest's captured logs, so a red row's message
    survives in the wrapper's evidence; restored after each row (``tests/cd/_quiet_logs.py``)."""
    with quiet_torch_logs():
        yield


N_REPEATS = 20


@pytest.fixture(autouse=True, scope="module")
def tf32_high() -> Iterator[None]:
    if not torch.cuda.is_available():
        why = "CUDA rows run inside the pinned container (tests/contract/test_vllm_cd_numerics_gpu.py)"
        if os.environ.get("ASTRA_CD_REQUIRE_CUDA") == "1":
            pytest.fail(f"ASTRA_CD_REQUIRE_CUDA=1 but CUDA is not available: {why}")
        pytest.skip(why)
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    print(f"ASTRA_TIER2_MATMUL_PRECISION {torch.get_float32_matmul_precision()}")
    yield
    torch.set_float32_matmul_precision(prev)


DEV = torch.device("cuda")


# --------------------------------------------------------------------------- D4: repeatability


@pytest.mark.parametrize("check", sorted(rs.CHECKS))
def test_core_call_is_bitwise_repeatable_on_cuda(check: str) -> None:
    runs = rs.CHECKS[check](DEV, N_REPEATS)
    bad = rs.compare_runs(runs)
    assert bad == [], f"{check}: {len(bad)} differences over {N_REPEATS} repeats, e.g. {bad[:8]}"


def test_per_step_sampler_call_is_bitwise_repeatable_on_cuda() -> None:
    from tests.cd.fakes.diffusion_sampler import DEVICE

    assert DEVICE.type == DEV.type, "run with ASTRA_CD_DEVICE=cuda"
    runs = rs.sampler_runs(N_REPEATS)
    assert any(".c.out.num_sampled" in k and bool(v.any()) for k, v in runs[0].items()), (
        "fixture: the constrained request never committed in the compared calls"
    )
    bad = rs.compare_runs(runs)
    assert bad == [], f"sampler: {len(bad)} differences over {N_REPEATS} repeats, e.g. {bad[:8]}"


# --------------------------------------------------------------------------- the c2 family (D2 scoring)


def _tables(toy: Toy) -> Any:
    return rs.tables(toy, DEV)


def _pipeline(x32: torch.Tensor, toy: Toy, L: int) -> Any:
    t = _tables(toy)
    tree = core("tree").up_sweep(core("emission").log_transfer(x32, t, L))
    return core("marginals").constrained_marginals(
        x32, t, tree, rs.vec([0], t.bucket, DEV), rs.vec(sorted(toy.finals), t.bucket, DEV), L
    )


def c2_instance(
    seed_rng: int, seed_gen: int, temp: float, regime: str
) -> tuple[Toy, torch.Tensor, int]:
    """(toy, float64 logits = the fp32 round (D2), L). ``regime``: "plain" or "narrow_neg"."""
    rng = random.Random(seed_rng)
    V, L = 4096, 256
    toy = random_toy(rng, V=V, S=12, nfa=True)
    g = torch.Generator().manual_seed(seed_gen)
    raw = torch.randn(L, V, generator=g, dtype=torch.float64) * 40.0
    x = 30.0 * torch.tanh(raw / 30.0)
    if regime == "narrow_neg":
        stored = [sorted(s) for s, neg in toy.classes if neg and s]
        if not stored:
            pytest.fail("fixture: the toy has no negated class with a non-empty N_c")
        x[:, min(stored, key=len)] = 30.0
    elif regime != "plain":
        raise ValueError(regime)
    return toy, (x / temp).to(torch.float32).to(torch.float64), L


def c2_err_over_bar(toy: Toy, x64: torch.Tensor, L: int, h: torch.Tensor) -> np.ndarray:
    """Per-position |H(q) − oracle| / (10·|stock fp32 H(p) − fp64 H(p)| + 1e-6); ``x64`` is already rounded."""
    start, end = np.zeros(toy.n_states, bool), np.zeros(toy.n_states, bool)
    start[0] = True
    end[sorted(toy.finals)] = True
    o = dense64.forward_backward64(x64.numpy(), toy, start, end, L)
    if not math.isfinite(o.log_z):
        pytest.fail("fixture: empty language on this instance")
    lp = torch.log_softmax(x64.to(torch.float32).to(DEV), dim=-1)
    hp32 = -(lp.exp() * lp).sum(-1).double().cpu().numpy()
    lp64 = torch.log_softmax(x64, dim=-1)
    hp64 = -(lp64.exp() * lp64).sum(-1).numpy()
    bar = 10.0 * np.abs(hp32 - hp64) + 1e-6
    return np.abs(h.double().cpu().numpy()[:L] - o.entropy) / bar  # type: ignore[no-any-return]


@pytest.mark.parametrize("temp", [0.8, 0.4])
def test_a_instance_meets_the_d2_bar_on_every_repeat(temp: float) -> None:
    toy, x64, L = c2_instance(7, 3, temp, "plain")
    x32 = x64.to(torch.float32).to(DEV)
    worst = []
    for _ in range(N_REPEATS):
        worst.append(float(np.max(c2_err_over_bar(toy, x64, L, _pipeline(x32, toy, L).entropy))))
    assert max(worst) <= 1.0, f"T {temp}: worst err/bar per repeat {[round(w, 3) for w in worst]}"


@pytest.mark.parametrize("regime", ["plain", "narrow_neg"])
@pytest.mark.parametrize("temp", [0.8, 0.4])
@pytest.mark.parametrize("seed", range(8))
def test_c2_differential_seeds_on_cuda(seed: int, temp: float, regime: str) -> None:
    toy, x64, L = c2_instance(seed, seed, temp, regime)
    r = c2_err_over_bar(toy, x64, L, _pipeline(x64.to(torch.float32).to(DEV), toy, L).entropy)
    assert np.all(r <= 1.0), (
        f"seed {seed} T {temp} {regime}: worst err/bar {float(np.max(r)):.2f} at {int(np.argmax(r))}"
    )


# --------------------------------------------------------------------------- c5 against the oracle


def test_c5_q_and_entropy_match_the_oracle_on_cuda() -> None:
    toy, filt, L, _sid = c5_fixture(C5_SEED)
    start = [0]
    assert accepted_over_finite_ids(toy, filt, start, L), "fixture: empty language"
    o = en.enumerate_log(
        en.log_softmax64(filt.double().tolist()), toy, start, sorted(toy.finals), L
    )
    assert math.isfinite(o.log_z)
    t = _tables(toy)
    x = filt.to(DEV)
    tree = core("tree").up_sweep(core("emission").log_transfer(x, t, L))
    res = core("marginals").constrained_marginals(
        x, t, tree, rs.vec(start, t.bucket, DEV), rs.vec(sorted(toy.finals), t.bucket, DEV), L
    )
    q = res.q.double().cpu().numpy()[:L]
    h = res.entropy.double().cpu().numpy()[:L]
    dq = float(np.max(np.abs(q - np.array(o.q))))
    dh = float(np.max(np.abs(h - np.array(o.entropy))))
    sums = q.sum(axis=-1)
    assert np.all(np.abs(sums - 1.0) <= 1e-6), f"q rows do not sum to 1: {sums}"
    assert dq <= 1e-6 and dh <= 1e-5, (dq, dh)


# --------------------------------------------------------------------------- compensated log_matmul at CPU bars


def _on_dev(fn: Any, a: torch.Tensor, b: torch.Tensor, compensated: bool) -> torch.Tensor:
    return fn(a.to(DEV), b.to(DEV), compensated=compensated).cpu()  # type: ignore[no-any-return]


@pytest.mark.parametrize("seed", range(4), ids=lambda s: f"seed{s}")
@pytest.mark.parametrize("inner", [64, 128], ids=lambda m: f"inner{m}")
def test_log_matmul_compensated_wide_on_cuda(inner: int, seed: int) -> None:
    lm = core("semiring").log_matmul
    a, b = _operands(-24.0, -16.0, inner, seed)
    exp = _lse_oracle(a, b)
    ok, msg = _wide_ok(_on_dev(lm, a, b, True), exp)
    assert ok, f"compensated log_matmul on CUDA: {msg} (bar: max ≤ 1 ulp, ≤ 5 % > 0.51 ulp)"
    # ruling P7 (T-0077c O9): on CUDA compensated=False runs the compensated kernel; the control is the torch path
    ok_c, msg_c = _wide_ok(lm(a.to(DEV), b.to(DEV), compensated=False, kernels=False).cpu(), exp)
    assert not ok_c, f"control (compensated=False) is not red on CUDA: {msg_c}"


@pytest.mark.parametrize("seed", range(4), ids=lambda s: f"seed{s}")
def test_log_matmul_compensated_gauged_on_cuda(seed: int) -> None:
    lm = core("semiring").log_matmul
    a, b = _operands(-3.35, -2.35, 300, seed)
    exp = _lse_oracle(a, b)
    assert np.abs(exp).max() < 1.0
    mean, mx = _gauged_stats(_on_dev(lm, a, b, True), exp)
    assert mean <= 0.75 and mx <= 2.5, (
        f"compensated on CUDA: mean {mean:.2f}, max {mx:.2f} (bar 0.75 / 2.5)"
    )
    # ruling P7 (T-0077c O9): the uncompensated control is the torch path
    c_mean, c_max = _gauged_stats(
        lm(a.to(DEV), b.to(DEV), compensated=False, kernels=False).cpu(), exp
    )
    assert c_mean > 0.75 or c_max > 2.5, (
        f"control not red on CUDA: mean {c_mean:.2f}, max {c_max:.2f}"
    )
