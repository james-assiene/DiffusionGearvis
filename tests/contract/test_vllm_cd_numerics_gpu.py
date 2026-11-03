"""T-0073 criteria 2 and 5 (tier 2, ``-m gpu``, ``/launch``): the numerics probe and the stricter CUDA module, run
**inside** the pinned vLLM container on the RTX 5090. No engine boots.

The container gets the repository read-only at ``/repo`` and nothing else: **no** ``/hostsite`` (the host's cp311
scipy broke imports in the image's Python 3.12, T-0067 tier-2 run 2026-09-30), ``--network none``. The probe
imports torch, numpy, the stdlib, pytest (for the rows' instance builders; the image ships 9.1.1) and the plugin
from the source tree; ``PYTHONDONTWRITEBYTECODE=1`` because the mount is read-only.

Rows:
- ``test_numerics_probe_on_the_5090`` (criterion 2, launch 1): runs ``astra_cd_numerics_probe.py --device cuda``,
  takes the JSON from stdout (the one ``ASTRA_T0073_PROBE_JSON`` line), adds the host regime (image digest,
  driver, ``git rev-parse HEAD``, dirty flag) and writes ``logs/t0073-numerics-probe-<sha>.json`` (``logs/`` is
  gitignored). Fails when any section is missing, errored or skipped (``tests/numerics_probe_schema.py``), when
  the device is not a 5090, or when the container exits non-zero. Nothing under ``plugin/`` is touched.
- ``test_criterion5_numerics_module_on_cuda`` (criterion 5 and D4, acceptance launch):
  ``tests/cd/cuda/test_cd_cuda_numerics.py`` through T-0067's runner (``astra_cd_tier2_runner.py``: 5090 preflight,
  regime line), ``ASTRA_CD_DEVICE=cuda ASTRA_CD_STOCK=image``; the inner summary must be exactly
  ``NUMERICS_PASSED`` passes and nothing else.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests import vllm_container as vc
from tests.contract._evidence import evidence
from tests.numerics_probe_schema import extract, problems

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[2]
PROBE = "/repo/tests/container_scripts/astra_cd_numerics_probe.py"
LOGS = REPO / "logs"
NUMERICS_MODULE = "tests/cd/cuda/test_cd_cuda_numerics.py"
RUNNER = "/repo/tests/container_scripts/astra_cd_tier2_runner.py"
RUNNER_REGIME = "ASTRA_TIER2_REGIME"
#: 51 = repeatability x {constrained_map, constrained_marginals, joint_draw} (3) + per-step sampler (1)
#:      + (a) on every repeat x T {0.8, 0.4} (2) + c2 x seeds 0-7 x T {0.8, 0.4} x {plain, narrow_neg} (32)
#:      + c5 vs oracle (1) + compensated wide x inner {64, 128} x seeds 0-3 (8) + gauged x seeds 0-3 (4)
NUMERICS_PASSED = 51


def _host(argv: list[str]) -> str:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode == 0, f"{argv} exited {proc.returncode}: {proc.stderr}"
    return proc.stdout.strip()


def _image_digest(image: str) -> str:
    fmt = "{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}"
    proc = vc._run(["docker", "image", "inspect", "--format", fmt, image], timeout=60)
    assert proc.returncode == 0 and proc.stdout.strip(), (
        f"docker image inspect {image}: {proc.stderr}"
    )
    return proc.stdout.strip()


def _docker_argv(image: str, env: dict[str, str], entry: list[str]) -> list[str]:
    argv = ["docker", "run", "--rm", "--gpus", "all", "--network", "none"]
    argv += ["-v", f"{REPO}:/repo:ro", "-w", "/repo"]
    base = {
        "PYTHONPATH": "/repo:/repo/plugin/astra_vllm_plugin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        **env,
    }
    for k, v in base.items():
        argv += ["-e", f"{k}={v}"]
    return [*argv, "--entrypoint", "python3", image, *entry]


def _host_regime(image: str) -> dict[str, Any]:
    return {
        "image": image,
        "image_digest": _image_digest(image),
        "driver": _host(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "host_head": _host(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "host_tree_dirty": bool(_host(["git", "-C", str(REPO), "status", "--porcelain"])),
    }


def test_numerics_probe_on_the_5090() -> None:
    image = vc.require_container()
    host = _host_regime(image)
    proc = vc._run(_docker_argv(image, {}, [PROBE, "--device", "cuda"]), timeout=3600)
    out = proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else "")
    payload = extract(proc.stdout)
    payload["host"] = host
    sha = host["host_head"][:7] + ("-dirty" if host["host_tree_dirty"] else "")
    LOGS.mkdir(exist_ok=True)
    dest = LOGS / f"t0073-numerics-probe-{sha}.json"
    dest.write_text(json.dumps(payload, indent=1, sort_keys=True))
    reg = payload.get("regime", {})
    line = {
        **host,
        **{
            k: reg.get(k) for k in ("torch", "torch_cuda", "python", "device_name", "total_seconds")
        },
    }
    print(f"T-0073 probe regime: {json.dumps(line, sort_keys=True)}")
    print(f"T-0073 probe JSON written to {dest} ({dest.stat().st_size} bytes)")
    print(f"--- container stderr (section timings) ---\n{proc.stderr[-6000:]}")

    assert proc.returncode == 0, f"probe exited {proc.returncode}\n{out[-20000:]}"
    bad = problems(payload)
    assert bad == [], "incomplete probe payload:\n" + "\n".join(bad)
    assert reg.get("cuda_available") is True and "5090" in str(reg.get("device_name")), (
        f"not on the 5090: {reg.get('device_name')}"
    )
    assert not host["host_tree_dirty"], (
        "launch evidence from a dirty tree (CLAUDE.md: never against a dirty tree)"
    )


def test_criterion5_numerics_module_on_cuda() -> None:
    from tests.pytest_summary import require_exact

    image = vc.require_container()
    host = _host_regime(image)
    # inside the container, as T-0067's wrapper does: no host conftests, no host addopts (so no -m filter)
    args = ["-rA", "-p", "no:cacheprovider", "--noconftest", "-o", "addopts=", NUMERICS_MODULE]
    env = {
        "ASTRA_CD_TESTS": "1",
        "ASTRA_CD_REQUIRE_CUDA": "1",
        "ASTRA_CD_DEVICE": "cuda",
        "ASTRA_CD_STOCK": "image",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    proc = vc._run(_docker_argv(image, env, [RUNNER, *args]), timeout=3600)
    inner = next(
        (
            json.loads(ln[len(RUNNER_REGIME) + 1 :])
            for ln in proc.stdout.splitlines()
            if ln.startswith(RUNNER_REGIME + " ")
        ),
        None,
    )
    print(f"T-0073 criterion-5 regime: {json.dumps({**host, 'inner': inner}, sort_keys=True)}")
    # evidence extract, not a raw tail (T-0077c: a torch.__trace flood pushed red rows' messages out of the window)
    print(f"--- container pytest evidence ({NUMERICS_MODULE}) ---\n{evidence(proc.stdout)}")
    if proc.returncode != 0:
        print(
            f"--- container pytest output tail (exit {proc.returncode}) ---\n{proc.stdout[-8000:]}"
        )
    assert inner is not None, f"the runner printed no {RUNNER_REGIME} line\n{proc.stdout[-8000:]}"
    assert inner["cuda_available"] and "5090" in inner["device"], f"not on the 5090: {inner}"
    assert proc.returncode == 0, evidence(proc.stdout) + "\n" + proc.stderr[-4000:]
    require_exact(proc.stdout, NUMERICS_PASSED)
    assert "ASTRA_TIER2_MATMUL_PRECISION high" in proc.stdout, "TF32 'high' was not reported set"
    assert not host["host_tree_dirty"], (
        "launch evidence from a dirty tree (CLAUDE.md: never against a dirty tree)"
    )
