"""T-0011 criteria 11-12 (tier 2, ``-m gpu``, human ``/launch``; container only, no engine boot).

The tap is coupled to v0.29.0 internals; this pins the coupling to the image. Facts come from
``tests/container_scripts/astra_tap_probe.py`` run in ``vllm/vllm-openai:v0.29.0`` with ``--gpus all
--network none``, the probe script and the plugin source mounted read-only; every assertion is here and every
failure message names the vLLM version ("the tap coupling drifted from vLLM X").

Criterion 12's first fence -- ``tests/contract/test_vllm_shim_gpu.py`` passes unmodified -- is that module's own
run in the same tier-2 launch (the host run covers the three modules: tap fields, shim, tap CUDA); its
"unmodified" half, and the byte-identity of ``services/vllm.sh`` and of ``plugin/.../cd/**``, are git facts
checked on CPU over T-0011's closed range in ``tests/contract/test_vllm_tap_fields_scope.py`` (T-0094). A probe
that cannot run fails (``pytest.fail``); only a box with no docker or no pinned image skips
(``tests/vllm_container.require_container``), as for every other tier-2 module.

Every row prints one regime line (S15, as ``test_vllm_tap_cuda_gpu.py``): image and digest, driver, the
container's torch/CUDA/Python/vllm/pytest versions and where each came from, the device, host HEAD and the dirty
flag. The git-only rows of criterion 12 print the host half (they do not start the container).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests import vllm_container as vc
from tests.unit.test_plugin_wheel import CONTAINER_MODULES

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "vllm_v0_29_0"
CD_DIR = "plugin/astra_vllm_plugin/astra_vllm_plugin/cd"
#: T-0015's three members of AstraDiffusionGemma; T-0011 adds exactly one (R9)
T0015_MEMBERS = {"_enable_mm_lora", "_parse_and_validate_audio_input", "_process_audio_input"}


def _probe_run() -> dict[str, Any]:
    image = vc.require_container()
    arg = json.dumps(
        {
            "manifest": json.loads((FIXTURE / "manifest.json").read_text()),
            "modules": sorted(CONTAINER_MODULES),
        }
    )
    argv = [
        "docker", "run", "--rm", "--gpus", "all", "--network", "none",
        "-v", f"{vc.SCRIPTS}:/astra_scripts:ro",
        "-v", f"{vc.PLUGIN_DIR}:/plugin:ro",
        "-e", "PYTHONPATH=/plugin", "-e", "HF_HUB_OFFLINE=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "--entrypoint", "python3", image, "/astra_scripts/astra_tap_probe.py", arg,
    ]  # fmt: skip
    proc = vc._run(argv, timeout=900)
    for line in proc.stdout.splitlines():
        if line.startswith(vc.PROBE_MARKER):
            return json.loads(line[len(vc.PROBE_MARKER) :])  # type: ignore[no-any-return]
    pytest.fail(
        f"astra_tap_probe.py printed no probe line in {image}\n{proc.stdout}\n{proc.stderr}"
    )


@pytest.fixture(scope="module")
def probe() -> dict[str, Any]:
    facts = _probe_run()
    if "error" in facts:
        pytest.fail(f"the tap probe failed inside the pinned container:\n{facts['error']}")
    print(f"T-0011 tier-2 probe regime: vllm {facts['versions']['vllm']}, python {facts['python']}")
    return facts


def _host(argv: list[str]) -> str:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
    return (
        proc.stdout.strip()
        if proc.returncode == 0
        else f"<{argv[0]} failed: {proc.stderr.strip()}>"
    )


@pytest.fixture(scope="module")
def host_regime() -> dict[str, Any]:
    image = vc.require_container()
    fmt = "{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}"
    return {
        "image": image,
        "image_digest": _host(["docker", "image", "inspect", "--format", fmt, image]),
        "driver": _host(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "host_head": _host(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "host_tree_dirty": bool(_host(["git", "-C", str(REPO), "status", "--porcelain"])),
    }


@pytest.fixture(autouse=True)
def _regime_line(request: pytest.FixtureRequest, host_regime: dict[str, Any]) -> None:
    """One regime line per row; the container half comes from the probe when the row uses it."""
    regime = dict(host_regime)
    if "probe" in request.fixturenames:
        regime["container"] = request.getfixturevalue("probe").get("regime", "<probe printed none>")
    else:
        regime["container"] = "<not started by this row: git facts only>"
    print(f"T-0011 tier-2 regime [{request.node.name}]: {json.dumps(regime, sort_keys=True)}")


def _v(probe: dict[str, Any]) -> str:
    return f"the tap coupling drifted from vLLM {probe['versions']['vllm']}"


# --------------------------------------------------------------------------- criterion 11


def test_get_model_state_cls_is_a_staticmethod_and_custom_sampler_returns_a_two_tuple(
    probe: dict[str, Any],
) -> None:
    assert probe["stock_get_model_state_cls_is_staticmethod"], f"{_v(probe)}: get_model_state_cls"
    ar = probe["custom_sampler_return_arities"]
    tuples = [a for a in ar if a != "None"]
    assert tuples and all(a == 2 for a in tuples), f"{_v(probe)}: custom_sampler returns {ar}"


def test_the_sampler_call_signature(probe: dict[str, Any]) -> None:
    assert probe["call_signature"] == [
        ["self", None],
        ["logits", None],
        ["input_batch", None],
        ["draft_logits", "None"],
    ], f"{_v(probe)}: DiffusionSampler.__call__ is {probe['call_signature']}"


def test_every_field_the_tap_reads_exists(probe: dict[str, Any]) -> None:
    assert probe["missing_classes"] == {}, f"{_v(probe)}: {probe['missing_classes']}"
    missing = {c: [n for n, ok in m.items() if not ok] for c, m in probe["attrs"].items()}
    missing = {c: ns for c, ns in missing.items() if ns}
    assert missing == {}, f"{_v(probe)}: {missing}"


def test_num_sampled_is_a_cuda_tensor_after_a_call(probe: dict[str, Any]) -> None:
    """R4: the reason the commit cannot be read on the host -- re-proved in the image every run."""
    dev = probe["num_sampled_device"]
    assert "error" not in dev, f"{_v(probe)}: the toy stock call failed:\n{dev.get('error')}"
    for kind in ("prefill", "decode"):
        assert "error" not in dev[kind], f"{_v(probe)}: {kind} call failed: {dev[kind]['error']}"
        assert dev[kind]["num_sampled"].startswith("cuda"), f"{_v(probe)}: {kind} {dev[kind]}"
        assert dev[kind]["sampled_token_ids"].startswith("cuda"), f"{_v(probe)}: {kind} {dev[kind]}"


def test_the_manifest_segments_equal_the_images_source(probe: dict[str, Any]) -> None:
    """T-0067's rule: the vendored oracle (tier 1b) is the image's code, segment by segment."""
    man = json.loads((FIXTURE / "manifest.json").read_text())
    got = probe["segments"]
    bad = {
        name: got.get(name)
        for name, seg in man["segments"].items()
        if got.get(name, {}).get("sha256") != seg["sha256"]
    }
    assert bad == {}, f"{_v(probe)}: vendored segments no longer match the image: {bad}"


def test_every_image_module_on_criterion_10s_list_imports(probe: dict[str, Any]) -> None:
    bad = {m: r for m, r in probe["modules"].items() if r != "ok"}
    assert set(probe["modules"]) == set(CONTAINER_MODULES)
    assert bad == {}, f"{_v(probe)}: modules the plugin imports are missing in the image: {bad}"


def test_the_scaffolds_guessed_names_are_still_absent(probe: dict[str, Any]) -> None:
    neg = probe["negatives"]
    assert neg["DiffusionSampler.states"] is False, f"{_v(probe)}: DiffusionSampler.states appeared"
    assert neg["DiffusionSampler.remove_request"] is False, f"{_v(probe)}: remove_request appeared"
    assert neg["accept_fields"] == [], (
        f"{_v(probe)}: an acceptance field appeared ({neg['accept_fields']}); criterion 6 says accepted is "
        "None because none existed -- re-plan before publishing it"
    )
    assert probe["has_DiffusionGemmaForCausalLM"] is False, (
        f"{_v(probe)}: DiffusionGemmaForCausalLM exists"
    )


# --------------------------------------------------------------------------- criterion 12 (+ R9)


def test_astra_diffusion_gemma_adds_only_the_model_state_seam(probe: dict[str, Any]) -> None:
    bare = probe["bare_subclass_own_members"]
    assert isinstance(bare, list), (
        f"the probe could not build a bare subclass of the same bases: {bare}"
    )
    # class machinery (ABCMeta's _abc_impl, typing.Protocol's _is_protocol, ...) is on a bare
    # subclass of the same bases too; everything else AstraDiffusionGemma owns, the plugin wrote
    machinery = set(bare)
    own = set(probe["astra_own_members"]) - machinery
    assert own == T0015_MEMBERS | {"get_model_state_cls"}, (
        f"AstraDiffusionGemma's own members beyond class machinery {sorted(machinery)}: "
        f"{sorted(own)}"
    )
    assert not (machinery & (T0015_MEMBERS | {"get_model_state_cls"})), (
        f"a bare subclass already has {sorted(machinery)}: the oracle would hide a real member"
    )
    assert probe["astra_gms_is_staticmethod"], "get_model_state_cls must be a staticmethod (R9)"
    assert probe["process_image_input_is_gemma4s"], (
        "_process_image_input is no longer Gemma4's function"
    )
    ms = probe["astra_model_state"]
    assert "error" not in ms, (
        f"AstraDiffusionGemma.get_model_state_cls() raised:\n{ms.get('error')}"
    )
    assert ms == {
        "name": "AstraDiffusionGemmaModelState",
        "subclasses_stock": True,
        "cached": True,
    }, ms
