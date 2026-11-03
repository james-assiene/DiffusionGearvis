"""T-0072 microbench probe (criterion 7, note 6), run INSIDE the pinned vLLM container (no engine boot).

Reports facts only: one line ``ASTRA_PROBE_JSON {...}`` per bucket (64, 128, 256, 512); every assertion lives
in ``tests/contract/test_vllm_cd_perf_gpu.py``. T-0071's ``scripts/gpu/cd_microbench.py`` may import
``measure`` / ``regime``; this task does not touch that script.

Needs ``/repo`` on ``PYTHONPATH`` together with ``/repo/plugin/astra_vllm_plugin`` (the plugin's ``cd`` package
and the frozen sequential δ* in ``tests/cd/oracle/delta_star_seq.py``). Regime (task "Regime", tier 2): fp32,
``torch.set_float32_matmul_precision("high")`` set globally, CL 256, V 262 144, seed 0, median and p95 over
200 calls after 20 warm-up calls, CUDA events. The automaton per bucket is synthetic (numpy, seeded): a random
NFA with ``bucket − 1`` states, 3 out-edges per state, positive classes of 1-64 ids and 10 % negated classes,
so no compiler is needed inside the image.

Components (milliseconds, per call): ``class_max`` (``mapdecode.class_max`` on the max-normalised logits),
``map_sweep`` (``mapdecode.max_up_sweep`` over ``[CL, B, B]`` leaves), ``delta_star_scan`` (``budget.delta_star``,
the change under test), ``delta_star_seq`` (the frozen pre-change δ*), ``stop_id_build`` (one host-to-device
build of the stop-id tensor, the cost the cache removes), and ``constrained_map`` (whole call) for context.
Each component also reports ``peak_mem_delta_bytes``: ``torch.cuda.max_memory_allocated()`` over one call after
``reset_peak_memory_stats()``, minus ``memory_allocated()`` before it.
Regime fields: ``container_digest`` and ``commit`` come from the host (env ``ASTRA_CONTAINER_DIGEST``,
``ASTRA_COMMIT``); versions, device name and matmul precision from the process.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

MARKER = "ASTRA_PROBE_JSON"
BUCKETS = (64, 128, 256, 512)
STOP_IDS = (1, 50, 106)
V = 262_144


def regime() -> dict[str, Any]:
    import torch

    try:
        import vllm

        vllm_version = vllm.__version__
    except Exception:  # pragma: no cover - reported, not fatal
        vllm_version = None
    return {
        "container_digest": os.environ.get("ASTRA_CONTAINER_DIGEST"),
        "commit": os.environ.get("ASTRA_COMMIT"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": vllm_version,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "matmul_precision": torch.get_float32_matmul_precision(),
        "dtype": "float32",
        "seed": 0,
    }


def synthetic_tables(bucket: int, device: Any, seed: int = 0) -> tuple[Any, Any]:
    import numpy as np
    from astra_vllm_plugin.cd import emission

    rng = np.random.default_rng(seed + bucket)
    S = bucket - 1
    members, neg, src, dst, cls = [], [], [], [], []
    for s in range(S):
        for _ in range(3):
            k = int(rng.integers(1, 65))
            ids = np.sort(rng.choice(V, size=k, replace=False)).astype(np.int64)
            members.append(ids)
            neg.append(bool(rng.random() < 0.1))
            src.append(s)
            dst.append(int(rng.integers(0, S)))
            cls.append(len(members) - 1)
    tables = emission.build_tables(
        n_states=S,
        vocab_size=V,
        class_members=members,
        class_is_neg=neg,
        edge_src=np.array(src, dtype=np.int64),
        edge_dst=np.array(dst, dtype=np.int64),
        edge_class=np.array(cls, dtype=np.int64),
        finals=np.arange(0, S, 7, dtype=np.int64),
        bucket=bucket,
        device=device,
    )
    return tables, members


def _timed(fn: Any, n: int, warmup: int) -> dict[str, float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    # peak added device memory of one call: max_memory_allocated() after a reset, minus what was live before
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - base
    return {
        "peak_mem_delta_bytes": int(peak_delta),
        "median_ms": times[len(times) // 2],
        "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
    }


def measure(
    bucket: int, *, CL: int = 256, n: int = 200, warmup: int = 20, seed: int = 0
) -> dict[str, Any]:
    import torch
    from astra_vllm_plugin.cd import budget, mapdecode

    from tests.cd.oracle import delta_star_seq

    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    tables, _members = synthetic_tables(bucket, dev, seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    logits = (torch.randn(CL, V, generator=g) * 4.0).to(dev)
    a0 = torch.zeros(bucket, dtype=torch.bool, device=dev)
    a0[0] = True
    b = torch.ones(bucket, dtype=torch.bool, device=dev)
    ids = torch.randint(0, V, (CL,), generator=g).to(dev)
    row_max, row_arg = logits.max(dim=1)
    reg = logits - row_max.unsqueeze(1)
    leaves = (torch.rand(CL, bucket, bucket, generator=g) * -8.0).to(dev)
    out: dict[str, Any] = {"bucket": bucket, "CL": CL, "n": n, "warmup": warmup, **regime()}
    out["components"] = {
        "class_max": _timed(lambda: mapdecode.class_max(reg, tables, row_arg), n, warmup),
        "map_sweep": _timed(lambda: mapdecode.max_up_sweep(leaves), n, warmup),
        "delta_star_scan": _timed(lambda: budget.delta_star(tables, a0, ids, STOP_IDS), n, warmup),
        "delta_star_seq": _timed(
            lambda: delta_star_seq.delta_star(tables, a0, ids, STOP_IDS), n, warmup
        ),
        "stop_id_build": _timed(
            lambda: torch.as_tensor(list(STOP_IDS), dtype=torch.int64, device=dev), n, warmup
        ),
        "constrained_map": _timed(
            lambda: mapdecode.constrained_map(logits, tables, a0, b, CL), n, warmup
        ),
    }
    return out


def main() -> None:
    buckets = [int(x) for x in sys.argv[1:]] or list(BUCKETS)
    for bucket in buckets:
        try:
            facts = measure(bucket)
        except Exception:
            facts = {"bucket": bucket, "error": traceback.format_exc()}
        print(MARKER + json.dumps(facts), flush=True)


if __name__ == "__main__":
    main()
