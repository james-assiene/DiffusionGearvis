"""T-0072 criterion 7 (tier 2, ``-m gpu``, ``/launch``, container only, no engine boot).

- ``test_criterion1_rows_hold_on_cuda``: ``tests/cd/cuda/test_cd_perf_cuda.py`` through T-0067's tier-2 helper
  (``test_vllm_cd_cuda_gpu._pytest_in_container`` and ``astra_cd_tier2_runner.py``: 5090 preflight, image-only
  imports, ``ASTRA_CD_REQUIRE_CUDA=1`` so the module fails rather than skips without CUDA, the regime printed,
  and exactly ``PERF_CUDA_PASSED`` passes with nothing skipped). Criterion 1's hand-built and adversarial rows and
  every T-0063 cut automaton on CUDA against the frozen sequential δ*, TF32 "high" set globally, and the stop-id
  cache without host-to-device copies. The cut automata are exported on the host from T-0063's compile cache to
  a temporary directory of ``.npz`` files mounted read-only at ``/cutnpz`` (the image has no compiler).
- ``test_probe_*``: ``tests/container_scripts/astra_cd_perf_probe.py`` prints one JSON line per bucket
  (64/128/256/512) with the note-6 component breakdown, ``peak_mem_delta_bytes`` per component and its regime.
  The host adds its own regime (image digest, driver, host HEAD and dirty flag), prints every line and writes
  them to ``logs/t0072-perf-probe-<sha>.json``. Every bucket must be present, the device must be the 5090, and
  the δ* scan's median is ≤ the frozen sequential δ*'s at buckets 64 and 128 (256 and 512 reported only).
  Regime: CL 256, fp32, matmul precision "high", median/p95 over 200 calls after 20 warm-up (CUDA events),
  seed 0, synthetic NFA per bucket.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests import vllm_container as vc
from tests.contract import test_vllm_cd_cuda_gpu as tier2

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "logs"
PROBE = "/repo/tests/container_scripts/astra_cd_perf_probe.py"
BUCKETS = (64, 128, 256, 512)
PERF_CUDA = "tests/cd/cuda/test_cd_perf_cuda.py"
#: 43 = hand-built NFAs: buckets {None, 64, 128, 512} x L {0,1,2,3,63,64,65,255,256} (36)
#:      + odd carry and order: L {63, 64, 65, 255, 256} (5) + every cut automaton (1) + stop-id cache H2D (1)
PERF_CUDA_PASSED = 43
#: every T-0063 cut automaton in criterion 1's ladder: 810 minus the 17 with bucket > 512 (OVER_LADDER)
CUT_AUTOMATA = 793
COMPONENTS = (
    "class_max",
    "map_sweep",
    "delta_star_scan",
    "delta_star_seq",
    "stop_id_build",
    "constrained_map",
)
MEMORY_COMPONENTS = ("class_max", "map_sweep", "delta_star_scan", "delta_star_seq")
REGIME_FIELDS = (
    "container_digest",
    "commit",
    "torch",
    "cuda",
    "device",
    "matmul_precision",
    "CL",
    "n",
    "warmup",
)


def _export_cut_npz(dest: Path) -> int:
    """Every T-0063 cut automaton's arrays, from the host compile cache (test-side glue, no plugin code)."""
    import numpy as np

    import tests.cd.unit.test_delta_star_scan as S

    n = 0
    for kind, cut, rid in S._cut_ids():
        a = S._compiled(kind, cut, rid)
        np.savez(
            dest / f"{kind}__{cut}__{rid}.npz",
            n_states=a.n_states,
            n_states_bucket=a.n_states_bucket,
            vocab_size=a.vocab_size,
            edge_src=a.edge_src,
            edge_dst=a.edge_dst,
            edge_class=a.edge_class,
            is_final=a.is_final,
            start_vector=a.start_vector,
            sum_indptr=a.tables.sum_indptr,
            sum_indices=a.tables.sum_indices,
            sum_is_neg=a.tables.sum_is_neg,
        )
        n += 1
    return n


def test_criterion1_rows_hold_on_cuda() -> None:
    vc.require_container()
    with tempfile.TemporaryDirectory(prefix="t0072-cut-npz-") as tmp:
        n = _export_cut_npz(Path(tmp))
        assert n == CUT_AUTOMATA, f"exported {n} cut automata, expected {CUT_AUTOMATA}"
        stdout = tier2._pytest_in_container(
            PERF_CUDA,
            PERF_CUDA_PASSED,
            {"ASTRA_CD_CUT_NPZ": "/cutnpz"},
            mounts=[(tmp, "/cutnpz")],
        )
    assert f"{tier2.PRECISION_MARKER} high" in stdout, (
        "the rows ran without TF32 'high' set globally"
    )


@pytest.fixture(scope="module")
def probe_lines() -> dict[int, dict[str, Any]]:
    image = vc.require_container()
    head = tier2._host(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    dirty = bool(tier2._host(["git", "-C", str(REPO), "status", "--porcelain"]))
    driver = tier2._host(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    digest = tier2._image_digest(image)
    argv = ["docker", "run", "--rm", "--gpus", "all", "--network", "none"]
    argv += ["-v", f"{REPO}:/repo:ro", "-w", "/repo"]
    for k, v in {
        "PYTHONPATH": "/repo:/repo/plugin/astra_vllm_plugin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "ASTRA_CONTAINER_DIGEST": digest,
        "ASTRA_COMMIT": head,
    }.items():
        argv += ["-e", f"{k}={v}"]
    argv += ["--entrypoint", "python3", image, PROBE]
    proc = vc._run(argv, timeout=3600)
    host_regime = {
        "image": image,
        "image_digest": digest,
        "driver": driver,
        "host_head": head,
        "host_tree_dirty": dirty,
    }
    out: dict[int, dict[str, Any]] = {}
    for line in proc.stdout.splitlines():
        if line.startswith(vc.PROBE_MARKER):
            facts = json.loads(line[len(vc.PROBE_MARKER) :])
            facts["host_regime"] = host_regime
            out[int(facts["bucket"])] = facts
    print(f"T-0072 perf probe regime: {json.dumps(host_regime, sort_keys=True)}")
    for bucket in sorted(out):
        print(f"T-0072 perf probe: {json.dumps(out[bucket], sort_keys=True)}")
    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"t0072-perf-probe-{head[:12]}{'-dirty' if dirty else ''}.json"
    log.write_text(
        json.dumps({"host_regime": host_regime, "lines": [out[b] for b in sorted(out)]}, indent=1)
    )
    print(f"T-0072 perf probe written to {log}")
    assert proc.returncode == 0, f"probe exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    missing = sorted(set(BUCKETS) - set(out))
    assert not missing, f"probe lines missing for buckets {missing}\n{proc.stdout}\n{proc.stderr}"
    return out


def test_probe_emits_one_json_line_per_bucket_with_the_breakdown(
    probe_lines: dict[int, dict[str, Any]],
) -> None:
    assert sorted(probe_lines) == list(BUCKETS)
    for bucket, facts in probe_lines.items():
        assert "error" not in facts, f"bucket {bucket}: {facts['error']}"
        assert "5090" in str(facts.get("device")), (
            f"bucket {bucket}: not on the 5090: {facts.get('device')}"
        )
        missing = [f for f in REGIME_FIELDS if facts.get(f) in (None, "")]
        assert not missing, f"bucket {bucket}: regime fields missing {missing}"
        assert facts["matmul_precision"] == "high" and facts["CL"] == 256
        for c in COMPONENTS:
            comp = facts["components"][c]
            assert comp["median_ms"] > 0 and comp["p95_ms"] >= comp["median_ms"], (bucket, c, comp)
        for c in MEMORY_COMPONENTS:
            mem = facts["components"][c].get("peak_mem_delta_bytes")
            assert isinstance(mem, int) and mem >= 0, (bucket, c, mem)


@pytest.mark.parametrize("bucket", [64, 128])
def test_scan_median_not_slower_than_sequential(
    probe_lines: dict[int, dict[str, Any]], bucket: int
) -> None:
    comp = probe_lines[bucket]["components"]
    scan_ms, seq_ms = comp["delta_star_scan"]["median_ms"], comp["delta_star_seq"]["median_ms"]
    assert scan_ms <= seq_ms, (
        f"bucket {bucket}, CL 256: δ* scan median {scan_ms:.3f} ms > sequential {seq_ms:.3f} ms"
    )
