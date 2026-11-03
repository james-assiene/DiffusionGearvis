"""T-0067 criterion 12 (tier 2, ``-m gpu``, ``/launch``): the CUDA repeats, run by pytest **inside** the pinned
container. No engine boots: the container imports torch, vllm and the plugin, nothing else.

The container gets the repository read-only at ``/repo`` (tests, oracles, fixtures, plugin source, and
``local_astra`` for T-0064's wire encoder) and **nothing from the host venv** (ruling S7 as amended after the first
tier-2 run, 2026-09-30, log ``logs/t0067-tier2-295c13b.log``). S7 first appended the host's site-packages as
``/hostsite``; the image has no scipy, so transformers (imported by vllm, imported by
``tests/cd/fakes/diffusion_sampler.py``) found the host's cp311 scipy there and failed to import it under the
image's Python 3.12. Every third-party module the inner modules import comes from the image instead: pytest 9.1.1
(with pluggy, iniconfig, packaging, pygments), numpy, torch, vllm, transformers, pydantic, tokenizers and
outlines_core are all in ``/usr/local/lib/python3.12/dist-packages`` of ``vllm/vllm-openai:v0.29.0``; scipy and
hypothesis are not, and nothing these modules import at module level or call on these rows needs them (scipy is
only imported lazily inside ``tests/cd/unit/_draw_support.py``'s ``chi2_gof``/``cell_pvalues``, which the CUDA rows
do not call; the repeats compute the chi-square tail with ``torch.special.gammaincc``). ``--noconftest`` keeps the
host's conftests out, ``-o addopts=`` drops the host's pytest options (so no ``-m`` filter applies inside), and
``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`` keeps any plugin the image ships from loading.

No vacuous pass. The runner (``tests/container_scripts/astra_cd_tier2_runner.py``) first requires CUDA on an
RTX 5090, else exits non-zero before pytest runs; ``ASTRA_CD_REQUIRE_CUDA=1`` turns the repeats module's
no-CUDA skip into a failure; and the host requires the inner summary to be **exactly** the pinned number of passes
with no skipped / xfailed / xpassed / deselected / failed / errors (warnings allowed). The expected counts are
pinned below from the modules' definitions; adding or removing a row there means updating the constant.

Evidence. Each row prints one ``T-0067 tier-2 regime:`` line (image digest, driver, torch, torch CUDA, Python,
vllm, pytest and where it was imported from, device, fp32 matmul precision at start and as set in the rows, host
``git rev-parse HEAD`` and whether the tree was dirty) and the container's full pytest output, so both appear
under ``-rA`` / ``-s`` and in the launch log.

Rows:
- ``test_criterion4_mixed_batch_identity_with_the_real_compiled_step``:
  ``tests/cd/unit/test_cd_mixed_batch_identity.py`` with ``ASTRA_CD_DEVICE=cuda ASTRA_CD_STOCK=image`` (the
  image's compiled ``DiffusionSampler`` and ``SamplingStates`` in place of the vendored eager copy).
- ``test_core_repeats_and_memory_on_cuda``: ``tests/cd/cuda/test_cd_cuda_repeats.py`` (T-0065 c1-5, T-0066
  c1/4/5, compensated rows eager and compiled, TF32 "high" set globally; peak added memory <= 1.5 GiB at CL 256).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from tests import vllm_container as vc
from tests.contract._evidence import evidence
from tests.pytest_summary import require_exact

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[2]
RUNNER = "/repo/tests/container_scripts/astra_cd_tier2_runner.py"
REGIME_MARKER = "ASTRA_TIER2_REGIME"
#: printed by the repeats module's autouse fixture after it sets TF32 "high"
PRECISION_MARKER = "ASTRA_TIER2_MATMUL_PRECISION"

MIXED_BATCH = "tests/cd/unit/test_cd_mixed_batch_identity.py"
#: 3 = one parametrized row x CASES {cap, stability, topk_topp_on_unconstrained}
MIXED_BATCH_PASSED = 3
REPEATS = "tests/cd/cuda/test_cd_cuda_repeats.py"
#: 18 = c1 x {fp64, fp32} (2) + c2 x temp {0.8, 0.4} (2) + c3, c4, c5 (3)
#:      + c5 empty language detected (1, T-0073 D0) + T-0066 c1, c4, c5 (3)
#:      + compensated log-matmul x {eager, compiled} (2) + compensated sum x {eager, compiled} (2)
#:      + peak added memory (1) + lagged reports synced (1) + lagged reports really lag (1)
REPEATS_PASSED = 18


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
    target: str,
    expected_passed: int,
    env: dict[str, str],
    timeout: int = 3600,
    *,
    mounts: Sequence[tuple[str, str]] = (),
) -> str:
    """``mounts``: extra ``(host, container)`` directories, mounted read-only (T-0072: the exported cut automata)."""
    image = vc.require_container()
    head = _host(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    dirty = bool(_host(["git", "-C", str(REPO), "status", "--porcelain"]))
    driver = _host(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    digest = _image_digest(image)

    argv = ["docker", "run", "--rm", "--gpus", "all", "--network", "none"]
    argv += ["-v", f"{REPO}:/repo:ro", "-w", "/repo"]
    for host_dir, container_dir in mounts:
        argv += ["-v", f"{host_dir}:{container_dir}:ro"]
    base_env = {
        "PYTHONPATH": "/repo:/repo/plugin/astra_vllm_plugin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "ASTRA_CD_TESTS": "1",
        "ASTRA_CD_REQUIRE_CUDA": "1",
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
    precision_in_rows = sorted(
        {ln.split()[1] for ln in proc.stdout.splitlines() if ln.startswith(PRECISION_MARKER + " ")}
    )
    regime = {
        "target": target,
        "image": image,
        "image_digest": digest,
        "driver": driver,
        **inner,
        "fp32_matmul_precision_in_rows": precision_in_rows or "not set by the module",
        "host_head": head,
        "host_tree_dirty": dirty,
        "env": {k: v for k, v in env.items()},
    }
    print(f"T-0067 tier-2 regime: {json.dumps(regime, sort_keys=True)}")
    # evidence extract, not the raw output (T-0077c: a torch.__trace flood pushed red rows' messages out of the window)
    print(
        f"--- container pytest evidence ({target}) ---\n{evidence(proc.stdout)}\n--- end container evidence ---"
    )
    if proc.returncode != 0:
        print(f"--- container pytest output tail (exit {proc.returncode}) ---\n{out[-8000:]}")

    assert proc.returncode == 0, (
        f"container pytest over {target} exited {proc.returncode}\n{evidence(proc.stdout)}\n{proc.stderr[-4000:]}"
    )
    assert inner["cuda_available"] and "5090" in inner["device"], f"not on the 5090: {inner}"
    assert str(inner["pytest_from"]).startswith("/usr/"), (
        f"pytest did not come from the image's site-packages: {inner['pytest_from']}"
    )
    require_exact(proc.stdout, expected_passed)
    return proc.stdout


def test_criterion4_mixed_batch_identity_with_the_real_compiled_step() -> None:
    _pytest_in_container(
        MIXED_BATCH, MIXED_BATCH_PASSED, {"ASTRA_CD_DEVICE": "cuda", "ASTRA_CD_STOCK": "image"}
    )


def test_core_repeats_and_memory_on_cuda() -> None:
    stdout = _pytest_in_container(
        REPEATS, REPEATS_PASSED, {"ASTRA_CD_DEVICE": "cuda", "ASTRA_CD_STOCK": "image"}
    )
    assert f"{PRECISION_MARKER} high" in stdout, (
        "the repeats ran without TF32 'high' reported set globally (criterion 12's landmine)"
    )
