"""T-0068 criterion 8 (tier 2, run inside the pinned container by ``tests/contract/test_vllm_cd_perstep_gpu.py``).

Peak added device memory of a ``perstep_mar`` + constrained-SC call at CL 256, buckets 128 and 256: the CD call's
``max_memory_allocated`` peak minus the stock call's on the same batch (T-0067 ruling S7's definition), ≤ 1.5 GiB.
The measured number is printed as ``ASTRA_CD_PERSTEP_PEAK <json>`` for the launch log (regime: CL 256, hidden 64,
fp32 embedding, one constrained slot, TF32 "high" set globally, max over 3 denoise calls).
Criterion 1 on CUDA is the host row re-running ``test_cd_arms_semantics.py`` / ``test_cd_mar_vs_mf.py`` there.

T-0076 criterion 12 (rows below the T-0068 one): the same peak-added-memory bound for ``perstep_mar_qargmax`` +
constrained SC (CL 256, buckets 128 and 256, ≤ 1.5 GiB); on CUDA the history row is the argmax of q -- C3 (language
{ad, ae, bc}, argmax of q ``ac`` out of language, MAP ``bc``, ``tests/cd/unit/test_cd_qargmax_arm.py``) gives ``ac``
until convergence and commits the MAP ``bc``, in the language; and on G1 with random logits (seeds 1, 3) the canvas-0
history rows equal the fp64 argmax of q (``brute_q``, every winner ≥ 1e-4 ahead, so CUDA rounding cannot decide it).
Rows: 2 (T-0068) + 2 (qargmax memory) + 2 (C3, ± SC) + 2 (G1 oracle) = 8.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from tests.cd import _wire_support as ws
from tests.cd.unit import _perstep_support as ps


@pytest.fixture(autouse=True)
def _cuda_high_precision() -> object:
    if not torch.cuda.is_available():
        why = (
            "CUDA rows run inside the pinned container (tests/contract/test_vllm_cd_perstep_gpu.py)"
        )
        if os.environ.get("ASTRA_CD_REQUIRE_CUDA") == "1":
            pytest.fail(f"ASTRA_CD_REQUIRE_CUDA=1 but CUDA is not available: {why}")
        pytest.skip(why)
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    print(f"ASTRA_TIER2_MATMUL_PRECISION {torch.get_float32_matmul_precision()}")
    yield
    torch.set_float32_matmul_precision(prev)


def _peak_added(bucket: int, mode: str) -> int:
    from tests.cd.fakes.diffusion_sampler import Cfg, World, random_live_logits

    compiled = ws.with_bucket(bucket)
    payload = ws.encode(compiled)
    xargs = ws.raw_xargs(
        payload, mode=mode, self_conditioning="constrained", record_tag=f"mem-{mode}-{bucket}"
    )
    cfg = Cfg(CL=256, max_num_reqs=4, hidden=64, stability_threshold=1, confidence_threshold=0.0)
    live = list(range(1000, 1000 + min(bucket, 200)))

    def fn(call: int, req: str, n: int) -> torch.Tensor:
        return random_live_logits(5 + call, live, n)

    peaks: dict[bool, int] = {}
    for cd in (False, True):
        w = World(cfg, cd=cd, seed=0)
        w.add("c", xargs=xargs if cd else None, max_tokens=4096)
        w.prefill()
        first = w.decode(fn)
        if cd:
            slot = first.slots["c"]
            assert not (
                bool(first.state["is_encoder_phase"][slot])
                and bool((first.state["canvas"][slot] == 1).all())
            ), f"{mode}: the request was refused (forced EOS canvas)"
        worst = 0
        for _ in range(3):
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            w.decode(fn)
            torch.cuda.synchronize()
            worst = max(worst, torch.cuda.max_memory_allocated() - base)
        peaks[cd] = worst
    return peaks[True] - peaks[False]


@pytest.mark.parametrize("bucket", [128, 256])
def test_qargmax_constrained_sc_peak_added_memory(bucket: int) -> None:
    """T-0076 criterion 12: ``perstep_mar_qargmax`` + constrained SC, CL 256, added peak ≤ 1.5 GiB."""
    from tests.cd.fakes.diffusion_sampler import DEVICE

    assert DEVICE.type == "cuda", "run with ASTRA_CD_DEVICE=cuda"
    added = _peak_added(bucket, "perstep_mar_qargmax")
    print(
        "ASTRA_CD_PERSTEP_PEAK "
        + json.dumps(
            {"mode": "perstep_mar_qargmax", "bucket": bucket, "CL": 256, "added_bytes": added}
        )
    )
    assert added <= 1.5 * (1 << 30), (
        f"qargmax added peak {added / (1 << 30):.2f} GiB > 1.5 GiB (CL 256, bucket {bucket})"
    )


@pytest.mark.parametrize("arm", ["qa", "qa_sc"])
def test_qargmax_history_and_commit_on_cuda_c3(arm: str) -> None:
    from tests.cd.unit import test_cd_qargmax_arm as qa

    x = ps.logits_rows(qa.c3_rows())
    g = qa.c3_grammar(f"t0076-cuda-{arm}")
    xargs = ps.ps_xargs(g, f"cuda-c3-{arm}", arm, confidence_threshold=ps.ALWAYS_CONFIDENT)
    _, snaps = qa.run(qa.t1(CL=2), xargs, qa.const(x), 3)
    assert qa.hist_row(snaps[0]) == [qa.A, qa.C] and not qa.phase(snaps[0])
    assert qa.phase(snaps[1]) and qa.row(snaps[1], "canvas") == [qa.B, qa.C]
    assert snaps[2].committed("c") == [qa.B, qa.C]
    assert g.accepts([0], [qa.B, qa.C], 4000)


@pytest.mark.parametrize("seed", [1, 3])
def test_qargmax_history_equals_argmax_of_q_on_cuda_g1(seed: int) -> None:
    from tests.cd.unit import test_cd_qargmax_arm as qa
    from tests.cd.unit._sampler_support import G1, Grammar

    g = Grammar(G1, f"t0076-cuda-g1-{seed}")
    fn = qa._random_fn(seed)
    xargs = ps.ps_xargs(g, f"cuda-g1-{seed}", "qa", confidence_threshold=ps.NEVER_CONFIDENT)
    _, snaps = qa.run(qa.t1(CL=4, max_denoising_steps=4, entropy_bound=0.5), xargs, fn, 3)
    for s in snaps:
        want = ps.argmax_q(ps.brute_q(g, fn(s.call, "c", 4), [0], 4, 4096 - 4), min_gap=1e-4)
        assert qa.hist_row(s) == want, f"call {s.call}: CUDA history {qa.hist_row(s)}, fp64 {want}"


@pytest.mark.parametrize("bucket", [128, 256])
def test_perstep_mar_constrained_sc_peak_added_memory(bucket: int) -> None:
    from tests.cd.fakes.diffusion_sampler import DEVICE, Cfg, World, random_live_logits

    assert DEVICE.type == "cuda", "run with ASTRA_CD_DEVICE=cuda"
    compiled = ws.with_bucket(bucket)
    payload = ws.encode(compiled)
    xargs = ws.raw_xargs(
        payload, mode="perstep_mar", self_conditioning="constrained", record_tag=f"mem-{bucket}"
    )
    cfg = Cfg(CL=256, max_num_reqs=4, hidden=64, stability_threshold=1, confidence_threshold=0.0)
    live = list(range(1000, 1000 + min(bucket, 200)))

    def fn(call: int, req: str, n: int) -> torch.Tensor:
        return random_live_logits(5 + call, live, n)

    peaks: dict[bool, int] = {}
    for cd in (False, True):
        w = World(cfg, cd=cd, seed=0)
        w.add("c", xargs=xargs if cd else None, max_tokens=4096)
        w.prefill()
        w.decode(fn)
        worst = 0
        for _ in range(3):
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            w.decode(fn)
            torch.cuda.synchronize()
            worst = max(worst, torch.cuda.max_memory_allocated() - base)
        peaks[cd] = worst
    added = peaks[True] - peaks[False]
    print(
        "ASTRA_CD_PERSTEP_PEAK " + json.dumps({"bucket": bucket, "CL": 256, "added_bytes": added})
    )
    assert added <= 1.5 * (1 << 30), (
        f"added peak {added / (1 << 30):.2f} GiB > 1.5 GiB (CL 256, bucket {bucket})"
    )
    assert ps.ARMS["mar_sc"] == ("perstep_mar", "constrained")
