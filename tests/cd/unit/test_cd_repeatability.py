"""T-0073 D4 (bonus property), on CPU: constrained decoding outputs are bitwise repeatable on the same device.

Three repeats in one process of ``constrained_marginals`` (via log_transfer -> up_sweep), ``joint_draw`` (fresh
generator, same seed), ``constrained_map`` and the per-step ``ConstrainedDiffusionSampler`` (fresh fake engine, same
global seed). Instance and comparison: ``tests/cd/unit/_repeat_support.py``. The CUDA counterpart (20 repeats) is
in ``tests/cd/cuda/test_cd_cuda_numerics.py``; CPU ``index_add_`` is expected to be order-stable already, so these
rows may be green before the D4 fix — they pin that the fix keeps it so.
"""

from __future__ import annotations

import pytest
import torch

from tests.cd.unit import _repeat_support as rs

CPU = torch.device("cpu")
N = 3


@pytest.mark.parametrize("check", sorted(rs.CHECKS))
def test_core_call_is_bitwise_repeatable_on_cpu(check: str) -> None:
    runs = rs.CHECKS[check](CPU, N)
    assert len(runs) == N and runs[0], "fixture: no outputs compared"
    bad = rs.compare_runs(runs)
    assert bad == [], f"{check} is not bitwise repeatable on CPU: {bad[:8]}"


def test_draw_repeatability_is_not_vacuous() -> None:
    """Four draws within one run must not all be the same canvas (else 'repeatable' could be a constant)."""
    run = rs.draw_runs(CPU, 1)[0]
    canvases = {tuple(run[f"draw{k}.tokens"].tolist()) for k in range(4)}
    assert len(canvases) > 1, (
        "the four draws are identical: the repeatability rows compare a constant"
    )


def test_per_step_sampler_call_is_bitwise_repeatable_on_cpu() -> None:
    runs = rs.sampler_runs(N)
    assert any(
        k.startswith("call") and ".c.out.num_sampled" in k and bool(v.any())
        for k, v in runs[0].items()
    ), "fixture: the constrained request never committed in the compared calls"
    bad = rs.compare_runs(runs)
    assert bad == [], f"the per-step sampler call is not bitwise repeatable on CPU: {bad[:8]}"
