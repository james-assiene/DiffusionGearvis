"""T-0073 (a), on CPU: the instance of ``test_t0065_c2_differential_vs_stock_error_on_cuda`` that escaped T-0065's
seeded CPU rows (ruling D0, docs/tasks/T-0073.md).

Instance, copied from the CUDA row: L 256, V 4096, ``random_toy(random.Random(7), V=4096, S=12, nfa=True)``,
logits ``torch.Generator().manual_seed(3)`` ``randn × 40``, softcapped ``30·tanh(x/30)`` then divided by T, fp32.
Bar (T-0065 criterion 2): per position, |H(q) − H_oracle| ≤ 10 × |H(p)_fp32 − H(p)_fp64| + 1e-6, stock fp32 H(p)
on the same logits, oracle float64 (``tests/cd/oracle/dense64``). T-0073 D2 (human, 2026-09-30): the oracle and the
bar's fp64 H(p) both take the fp32-rounded logits the core receives. Measured on host CPU (torch 2.13.0+cpu): with the
unrounded-input oracle T 0.4 was 1.18 (red) — an oracle input mismatch (D1.1), not a core error; with D2 it is ~0.44.
CUDA gave 2.62 (row as written) in the tier-2 run 2026-09-30. About 1 s per row.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch

from tests.cd.oracle import dense64
from tests.cd.unit._core_support import random_toy, run_core


@pytest.mark.parametrize("temp", [0.8, 0.4])
def test_c2_cuda_row_instance_meets_the_differential_bar_on_cpu(temp: float) -> None:
    rng = random.Random(7)
    V, L = 4096, 256
    toy = random_toy(rng, V=V, S=12, nfa=True)
    g = torch.Generator().manual_seed(3)
    raw = torch.randn(L, V, generator=g, dtype=torch.float64) * 40.0
    # T-0073 D2: oracle and bar reference on the fp32-rounded inputs
    x64 = ((30.0 * torch.tanh(raw / 30.0)) / temp).to(torch.float32).to(torch.float64)
    start, end = np.zeros(toy.n_states, bool), np.zeros(toy.n_states, bool)
    start[0] = True
    end[sorted(toy.finals)] = True
    o = dense64.forward_backward64(x64.numpy(), toy, start, end, L)
    if not math.isfinite(o.log_z):
        pytest.fail("fixture: empty language on this instance")
    x32 = x64.to(torch.float32)
    res, _tables = run_core(x32, toy, [0], sorted(toy.finals))
    lp = torch.log_softmax(x32, dim=-1)
    hp32 = -(lp.exp() * lp).sum(-1).double().numpy()
    lp64 = torch.log_softmax(x64, dim=-1)
    hp64 = -(lp64.exp() * lp64).sum(-1).numpy()
    bar = 10.0 * np.abs(hp32 - hp64) + 1e-6
    err = np.abs(res.entropy.double().numpy()[:L] - o.entropy)
    r = err / bar
    assert np.all(r <= 1.0), (
        f"T {temp}: worst H(q) error / bar = {float(np.max(r)):.2f} at position {int(np.argmax(r))} "
        f"({int(np.sum(r > 1))} of {L} positions over the bar)"
    )
