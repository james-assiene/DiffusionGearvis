"""T-0069 criterion 3 (tier 2, ``-m gpu``, ``/launch`` 1; container only, no engine boot).

In its own module so T-0011's ``test_vllm_tap_fields_gpu.py`` stays unedited. Two halves:

1. **The probe** (``tests/container_scripts/astra_cd_wiring_probe.py``, run in ``vllm/vllm-openai:v0.29.0`` with
   ``--gpus all --network none``, the probe directory and the plugin source mounted read-only, nothing from the host
   venv): ``AstraDiffusionGemma.get_model_state_cls()`` in the image is ``AstraDiffusionGemmaModelState`` whose base
   is ``Cd<stock name>`` whose base is the image's stock model state; it is cached; ``custom_sampler`` yields
   ``TapSampler → ConstrainedDiffusionSampler → stock DiffusionSampler`` by identity (and stock's second element);
   a failing ``cd`` import fails the build (no fallback).
2. **The CUDA row** (``tests/cd/cuda/test_cd_tap_composition_cuda.py``) run by pytest inside the container through
   the CD tier-2 runner (``tests/container_scripts/astra_cd_tier2_runner.py``: exits 3 before pytest unless CUDA on
   an RTX 5090), repo ``:ro`` at ``/repo``, ``--noconftest -o addopts= -p no:cacheprovider``,
   ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``, ``ASTRA_CD_REQUIRE_CUDA=1`` (a skip is a failure), and the host demands
   **exactly** ``CD_TAP_CUDA_PASSED`` passes (``tests/pytest_summary.require_exact``).

Fences (criterion 3: "all passing unedited"): ``test_vllm_shim_gpu.py``, T-0011 criterion 11 and criterion 12 except its
``cd/**`` parametrisation (ruling O1: ``--deselect``ed by name in the launch), and T-0067's coupling module run in the
same launch; *unedited* is checked on CPU by ``tests/contract/test_vllm_cd_wiring_scope.py`` over T-0069's closed
range (moved there by T-0094b; git only, no container).

Fail, not skip: a probe that cannot run, prints no line or reports an error fails. Only a box with no docker or no
pinned image skips (``tests/vllm_container.require_container``), as for every tier-2 module.

Every row prints one ``T-0069 tier-2 regime [<row>]:`` line: image and digest, driver, the container's
torch/CUDA/Python/vllm/pytest versions and where they came from, the device, host HEAD and the dirty flag (git-only
rows print the host half).
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
DG = "vllm.model_executor.models.diffusion_gemma"

TAP_CD_CUDA = "tests/cd/cuda/test_cd_tap_composition_cuda.py"
#: 2 = sync-debug "error" really raises on a CUDA .item() (liveness, 1)
#:     + >= 12 stock-only warm-up calls, then 50 Tap ∘ CD calls with one constrained slot (commits included), its
#:       release, a slot-reuse bind and the binds under set_sync_debug_mode("error"): tap.errors == 0, no CD
#:       error record (1)
#: adding or removing a row in tests/cd/cuda/test_cd_tap_composition_cuda.py means updating this constant
CD_TAP_CUDA_PASSED = 2


def _host_or_note(argv: list[str]) -> str:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
    return (
        proc.stdout.strip()
        if proc.returncode == 0
        else f"<{argv[0]} failed: {proc.stderr.strip()}>"
    )


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


@pytest.fixture(scope="module")
def host_regime() -> dict[str, Any]:
    return {
        "host_head": _host_or_note(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "host_tree_dirty": bool(_host_or_note(["git", "-C", str(REPO), "status", "--porcelain"])),
        "driver": _host_or_note(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
    }


@pytest.fixture(scope="module")
def probe() -> dict[str, Any]:
    image = vc.require_container()
    argv = [
        "docker", "run", "--rm", "--gpus", "all", "--network", "none",
        "-v", f"{vc.SCRIPTS}:/astra_scripts:ro",
        "-v", f"{vc.PLUGIN_DIR}:/plugin:ro",
        "-e", "PYTHONPATH=/plugin", "-e", "HF_HUB_OFFLINE=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "--entrypoint", "python3", image, "/astra_scripts/astra_cd_wiring_probe.py",
    ]  # fmt: skip
    proc = vc._run(argv, timeout=900)
    for line in proc.stdout.splitlines():
        if line.startswith(vc.PROBE_MARKER):
            facts: dict[str, Any] = json.loads(line[len(vc.PROBE_MARKER) :])
            break
    else:
        pytest.fail(
            f"astra_cd_wiring_probe.py printed no probe line in {image}\n{proc.stdout}\n{proc.stderr}"
        )
    if "error" in facts:
        pytest.fail(f"the CD wiring probe failed inside the pinned container:\n{facts['error']}")
    facts["image"] = image
    facts["image_digest"] = _image_digest(image)
    return facts


@pytest.fixture(autouse=True)
def _regime_line(request: pytest.FixtureRequest, host_regime: dict[str, Any]) -> None:
    regime = dict(host_regime)
    if "probe" in request.fixturenames:
        p = request.getfixturevalue("probe")
        regime.update(
            image=p["image"],
            image_digest=p["image_digest"],
            container=p.get("regime", "<probe printed none>"),
        )
    else:
        regime["container"] = "<started by the row itself, or git facts only>"
    print(f"T-0069 tier-2 regime [{request.node.name}]: {json.dumps(regime, sort_keys=True)}")


def _v(probe: dict[str, Any]) -> str:
    return f"the CD wiring drifted from vLLM {probe.get('versions', {}).get('vllm')}"


# --------------------------------------------------------------------------- the probe


def test_the_served_model_state_is_tap_over_cd_over_the_images_stock(probe: dict[str, Any]) -> None:
    ms = probe["model_state"]
    assert "error" not in ms, (
        f"AstraDiffusionGemma.get_model_state_cls() raised:\n{ms.get('error')}"
    )
    stock = probe["stock_model_state"]
    stock_name = stock.rsplit(".", 1)[1]
    assert ms["name"] == "AstraDiffusionGemmaModelState", f"{_v(probe)}: outer class {ms['name']}"
    assert len(ms["bases"]) == 1 and ms["base_name"] == f"Cd{stock_name}", (
        f"{_v(probe)}: the outer class's bases are {ms['bases']}, want one Cd{stock_name}"
    )
    assert ms["base_bases"] == [stock] and ms["base_base_is_stock"] is True, (
        f"{_v(probe)}: Cd{stock_name}'s bases are {ms['base_bases']}, want the image's {stock}"
    )
    assert ms["mro"][1:3] == [ms["bases"][0], stock], ms["mro"]
    assert ms["cached"] is True, "get_model_state_cls() must be cached (R9)"


def test_custom_sampler_yields_tap_over_cd_over_the_stock_sampler_by_identity(
    probe: dict[str, Any],
) -> None:
    ch = probe["chain"]
    assert "error" not in ch, f"custom_sampler on the served model state raised:\n{ch.get('error')}"
    assert ch["stock_custom_sampler_calls"] == 1 and ch["stock_got_the_runner_sampler"], ch
    assert ch.get("types") == [
        "astra_vllm_plugin.sampler.TapSampler",
        "astra_vllm_plugin.cd.sampler.ConstrainedDiffusionSampler",
        f"{DG}.DiffusionSampler",
    ], f"{_v(probe)}: the served sampler chain is {ch.get('types', ch)}"
    assert ch["innermost_is_stocks"] is True, "the innermost sampler is not the one stock returned"
    assert ch["second_is_stocks"] is True, "the second element is not the one stock returned"


def test_a_failing_cd_import_fails_the_build_in_the_image(probe: dict[str, Any]) -> None:
    nf = probe["no_fallback"]
    assert "error" not in nf, nf
    assert nf["raised"] in ("ImportError", "ModuleNotFoundError"), (
        f"with astra_vllm_plugin.cd.model_state unimportable, get_model_state_cls() {nf}: a silent fallback"
    )


# --------------------------------------------------------------------------- the CUDA row in the container


def _inner_regime(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        if line.startswith(REGIME_MARKER + " "):
            return json.loads(line[len(REGIME_MARKER) + 1 :])  # type: ignore[no-any-return]
    raise AssertionError(f"the container printed no {REGIME_MARKER} line\n{stdout[-20000:]}")


def test_tap_over_cd_on_the_images_compiled_sampler_under_sync_debug_error() -> None:
    image = vc.require_container()
    head = _host(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    dirty = bool(_host(["git", "-C", str(REPO), "status", "--porcelain"]))
    driver = _host(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    digest = _image_digest(image)
    env = {
        "PYTHONPATH": "/repo:/repo/plugin/astra_vllm_plugin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "ASTRA_CD_TESTS": "1",
        "ASTRA_CD_REQUIRE_CUDA": "1",
        "ASTRA_CD_DEVICE": "cuda",
        "ASTRA_CD_STOCK": "image",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    argv = ["docker", "run", "--rm", "--gpus", "all", "--network", "none"]
    argv += ["-v", f"{REPO}:/repo:ro", "-w", "/repo"]
    for k, v in env.items():
        argv += ["-e", f"{k}={v}"]
    args = ["-rA", "-p", "no:cacheprovider", "--noconftest", "-o", "addopts=", TAP_CD_CUDA]
    argv += ["--entrypoint", "python3", image, RUNNER, *args]
    proc = vc._run(argv, timeout=3600)
    out = proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else "")
    inner = _inner_regime(proc.stdout)
    regime = {
        "target": TAP_CD_CUDA,
        "image": image,
        "image_digest": digest,
        "driver": driver,
        **inner,
        "host_head": head,
        "host_tree_dirty": dirty,
    }
    print(f"T-0069 tier-2 regime: {json.dumps(regime, sort_keys=True)}")
    print(f"--- container pytest output ({TAP_CD_CUDA}) ---\n{out}\n--- end container output ---")
    assert proc.returncode == 0, (
        f"container pytest over {TAP_CD_CUDA} exited {proc.returncode}\n{out[-20000:]}"
    )
    assert inner["cuda_available"] and "5090" in inner["device"], f"not on the 5090: {inner}"
    assert str(inner["pytest_from"]).startswith("/usr/"), (
        f"pytest did not come from the image's site-packages: {inner['pytest_from']}"
    )
    require_exact(proc.stdout, CD_TAP_CUDA_PASSED)
