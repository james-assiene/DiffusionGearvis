"""T-0011 criterion 16 (tier 2, ``-m gpu``, human ``/launch``): the tier-1b module on CUDA with the image's
compiled sampler, run by pytest **inside** the pinned container. No engine boots.

Conventions (S14/S15, as T-0062's tier-2 harness): the repository is mounted read-only at ``/repo``; **nothing
from the host venv** enters the container -- pytest, numpy, torch, vllm and pydantic all come from the image's
``/usr/`` site-packages, and the row asserts where pytest came from; ``--noconftest`` keeps the host's conftests
out (helpers live in ``tests/stock/world.py``), ``-o addopts=`` drops host pytest options,
``-p no:cacheprovider`` and ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``; ``--network none``.

No vacuous pass. The runner (``tests/container_scripts/astra_cd_tier2_runner.py``, shared byte-identically with
T-0062, R8) exits 3 before pytest unless it finds CUDA on an RTX 5090; ``ASTRA_STOCK_REQUIRE_CUDA=1`` turns the
module's no-CUDA skip into a failure; and the host demands **exactly** ``STOCK_TAP_PASSED`` passes and nothing
else but warnings (``tests/pytest_summary.require_exact``). Warm-up before sync-debug is inside the module
(criterion 16(a): ≥ 12 stock-only calls, sync-debug off). One regime line per row: image digest, driver,
torch/CUDA/Python/vllm/pytest versions and where pytest came from, device, host HEAD and the dirty flag.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests import vllm_container as vc
from tests.pytest_summary import require_exact

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[2]
RUNNER = "/repo/tests/container_scripts/astra_cd_tier2_runner.py"
REGIME_MARKER = "ASTRA_TIER2_REGIME"

STOCK_TAP = "tests/stock/test_tap_stock.py"
#: 18 = S8 liveness on real torch x {.item, .tolist, .numpy, bool, int, float, .nonzero} (7)
#:      + device->host moves only when blocking (1)
#:      + value reads count only on the world's device (a CPU tensor's .item on CUDA is not a sync) (1)
#:      + each recorded sync is attributed to the tap or to stock, with its call site (1)
#:      + criterion 15: frames equal the rows, exactly-once under the limiter, bit-identity, zero syncs (4)
#:      + R15: a canvas closed by the step cap publishes valid frames (1)
#:      + R13: the backend's CUDA events are blocking (1)
#:      + criterion 16: (a) 50 calls under sync-debug "error" after warm-up, (b) the lag really happens (2)
#: adding or removing a row in tests/stock/test_tap_stock.py means updating this constant
STOCK_TAP_PASSED = 18


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


def _inner_regime(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        if line.startswith(REGIME_MARKER + " "):
            return json.loads(line[len(REGIME_MARKER) + 1 :])  # type: ignore[no-any-return]
    raise AssertionError(f"the container printed no {REGIME_MARKER} line\n{stdout[-20000:]}")


def _pytest_in_container(
    target: str, expected_passed: int, env: dict[str, str], timeout: int = 3600
) -> str:
    image = vc.require_container()
    head = _host(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    dirty = bool(_host(["git", "-C", str(REPO), "status", "--porcelain"]))
    driver = _host(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    digest = _image_digest(image)

    argv = ["docker", "run", "--rm", "--gpus", "all", "--network", "none"]
    argv += ["-v", f"{REPO}:/repo:ro", "-w", "/repo"]
    base_env = {
        "PYTHONPATH": "/repo:/repo/plugin/astra_vllm_plugin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "ASTRA_STOCK_TESTS": "1",
        "ASTRA_STOCK_REQUIRE_CUDA": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        **env,
    }
    for k, v in base_env.items():
        argv += ["-e", f"{k}={v}"]
    args = ["-rA", "-p", "no:cacheprovider", "--noconftest", "-o", "addopts=", target]
    argv += ["--entrypoint", "python3", image, RUNNER, *args]
    proc = vc._run(argv, timeout=timeout)
    out = proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else "")

    inner = _inner_regime(proc.stdout)
    regime = {
        "target": target,
        "image": image,
        "image_digest": digest,
        "driver": driver,
        **inner,
        "host_head": head,
        "host_tree_dirty": dirty,
        "env": env,
    }
    print(f"T-0011 tier-2 regime: {json.dumps(regime, sort_keys=True)}")
    print(f"--- container pytest output ({target}) ---\n{out}\n--- end container output ---")

    assert proc.returncode == 0, (
        f"container pytest over {target} exited {proc.returncode}\n{out[-20000:]}"
    )
    assert inner["cuda_available"] and "5090" in inner["device"], f"not on the 5090: {inner}"
    assert str(inner["pytest_from"]).startswith("/usr/"), (
        f"pytest did not come from the image's site-packages: {inner['pytest_from']}"
    )
    require_exact(proc.stdout, expected_passed)
    return proc.stdout


def test_criterion16_the_tier1b_module_on_cuda_with_the_images_compiled_sampler() -> None:
    _pytest_in_container(
        STOCK_TAP,
        STOCK_TAP_PASSED,
        {"ASTRA_STOCK": "image", "ASTRA_STOCK_DEVICE": "cuda"},
    )
