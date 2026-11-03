"""T-0015 criterion 3, host-side: the shim audit's logic on a synthetic source tree.

The real audit runs inside the pinned vLLM container (``tests/contract/test_vllm_shim_gpu.py``,
``gpu``). Its logic does not need vLLM, though, and the defect found in the closing round
(2026-10-02) was in that logic: resolution was computed against the *shim* (``dir(astra_cls)``),
so the shim's own audio stubs counted as resolved before the upstream-derived modality
exclusion ran, and "goes red if DiffusionGemma ever declares audio" could never fire.

These rows rebuild the upstream shape in miniature -- a donor class, a stock class that borrows
its methods by class-level assignment and declares its modalities in a processing-info class,
and a shim subclass with one piece of state and one stub -- and run the probe's own
``audit_section`` plus the gpu test's own verdict (``check_audit``) over it. Expected outcomes
are fixed by the task text, not by running the code: with ``{"image", "video"}`` declared the
audit passes; with ``"audio"`` added it must fail naming the stub and the modality.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.contract.test_vllm_shim_gpu import check_audit

PROBE_PATH = Path(__file__).resolve().parents[1] / "container_scripts" / "astra_shim_probe.py"

DONOR_SRC = """
class Gemma4ForConditionalGeneration:
    def __init__(self):
        self.vision_tower = object()
        self._enable_mm_lora = False

    def _process_image_input(self, image_input):
        if self._enable_mm_lora:
            return self.vision_tower
        return self.vision_tower

    def _process_audio_input(self, audio_input):
        return self.audio_tower

    def embed_multimodal(self, modality, data):
        if modality == "image":
            return self._process_image_input(data)
        if modality == "video":
            return self.vision_tower
        if modality == "audio":
            return self._process_audio_input(data)
        return None
"""

STOCK_SRC_TEMPLATE = """
from {donor} import Gemma4ForConditionalGeneration


class DiffusionGemmaProcessingInfo:
    def get_supported_mm_limits(self):
        return {limits}


class DiffusionGemmaForConditionalGeneration:
    def __init__(self):
        self.vision_tower = object()

    _process_image_input = Gemma4ForConditionalGeneration._process_image_input
    embed_multimodal = Gemma4ForConditionalGeneration.embed_multimodal
"""

IMAGE_VIDEO = '{"image": None, "video": None}'
WITH_AUDIO = '{"image": None, "video": None, "audio": None}'

# The shape the real shim has: plain state for the reachable defect, a raising stub for the
# audio-only sibling.
SHIM_REAL_SHAPE = """
class AstraDiffusionGemma(DiffusionGemmaForConditionalGeneration):
    _enable_mm_lora = False

    def _process_audio_input(self, audio_input):
        raise NotImplementedError("no audio tower")
"""


def _load(name: str, src: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(src))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # inspect.getsource(cls) finds the file through sys.modules[cls.__module__].
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load("astra_shim_probe_under_test", PROBE_PATH.read_text(), tmp_path, monkeypatch)


def _audit(
    probe: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    limits: str,
    shim_body: str,
) -> dict[str, Any]:
    """Build donor/stock/shim modules on disk and run the probe's audit over them."""
    tag = f"t{abs(hash((limits, shim_body))) % 10**8}"
    donor_name, stock_name = f"synth_donor_{tag}", f"synth_stock_{tag}"
    donor = _load(donor_name, DONOR_SRC, tmp_path, monkeypatch)
    stock_src = STOCK_SRC_TEMPLATE.format(donor=donor_name, limits=limits)
    stock = _load(stock_name, stock_src, tmp_path, monkeypatch)
    shim = _load(
        f"synth_shim_{tag}",
        f"from {stock_name} import DiffusionGemmaForConditionalGeneration\n" + shim_body,
        tmp_path,
        monkeypatch,
    )
    supported = probe.supported_modalities(stock.DiffusionGemmaProcessingInfo)
    result: dict[str, Any] = probe.audit_section(
        shim.AstraDiffusionGemma,
        stock.DiffusionGemmaForConditionalGeneration,
        inspect.getsource(stock),
        inspect.getsource(donor),
        supported,
    )
    return result


def test_real_shape_passes_with_state_reachable_and_audio_dead(
    probe: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(probe, tmp_path, monkeypatch, limits=IMAGE_VIDEO, shim_body=SHIM_REAL_SHAPE)

    assert audit["supported_modalities"] == ["image", "video"]
    assert audit["borrowed"] == {
        "_process_image_input": "_process_image_input",
        "embed_multimodal": "embed_multimodal",
    }
    lora = audit["methods"]["_process_image_input"]["unresolved"]["_enable_mm_lora"]
    # Unresolved on the stock class, reachable (unguarded), and answered by shim *state*.
    assert lora["excluded_as_dead"] is False
    assert lora["reachable_sites"] == [[]]  # one unguarded read: the `if` test
    assert lora["supplied_by_shim"] == "state"

    embed = audit["methods"]["embed_multimodal"]
    assert embed["dead_modality_literals"] == ["audio"]
    audio = embed["unresolved"]["_process_audio_input"]
    # Present on the shim (it is in dir()), yet still audited: dead by upstream's declaration,
    # and recorded as a stub, not silently counted as resolved.
    assert audio["supplied_by_shim"] == "behaviour"
    assert audio["excluded_as_dead"] is True
    # Things the stock class does have never show up as unresolved.
    assert "vision_tower" not in embed["unresolved"]
    assert "_process_image_input" not in embed["unresolved"]

    check_audit(audit)  # must not raise


def test_declaring_audio_turns_the_stub_red_naming_it_and_the_modality(
    probe: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The closing-round defect: before the fix this audit passed, because dir(shim) hid the stub."""
    audit = _audit(probe, tmp_path, monkeypatch, limits=WITH_AUDIO, shim_body=SHIM_REAL_SHAPE)

    assert audit["supported_modalities"] == ["audio", "image", "video"]
    audio = audit["methods"]["embed_multimodal"]["unresolved"]["_process_audio_input"]
    assert audio["excluded_as_dead"] is False
    assert audio["reachable_modalities"] == ["audio"]

    with pytest.raises(AssertionError) as err:
        check_audit(audit)
    message = str(err.value)
    assert "self._process_audio_input" in message
    assert "'audio'" in message
    assert "embed_multimodal" in message
    assert "stub" in message


def test_audio_declared_and_shim_does_not_stub_it_fails_as_missing(
    probe: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shim = """
class AstraDiffusionGemma(DiffusionGemmaForConditionalGeneration):
    _enable_mm_lora = False
"""
    audit = _audit(probe, tmp_path, monkeypatch, limits=WITH_AUDIO, shim_body=shim)
    with pytest.raises(AssertionError, match="do not resolve on the shim") as err:
        check_audit(audit)
    assert "self._process_audio_input" in str(err.value)


def test_shim_without_the_state_fails_naming_enable_mm_lora(
    probe: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shim = """
class AstraDiffusionGemma(DiffusionGemmaForConditionalGeneration):
    def _process_audio_input(self, audio_input):
        raise NotImplementedError("no audio tower")
"""
    audit = _audit(probe, tmp_path, monkeypatch, limits=IMAGE_VIDEO, shim_body=shim)
    with pytest.raises(AssertionError, match="do not resolve on the shim") as err:
        check_audit(audit)
    assert "self._enable_mm_lora" in str(err.value)
    assert "_process_image_input" in str(err.value)


@pytest.mark.parametrize(
    "member",
    [
        "    def _enable_mm_lora(self):\n        return False\n",
        "    @property\n    def _enable_mm_lora(self):\n        return False\n",
        "    _enable_mm_lora = staticmethod(lambda: False)\n",
    ],
    ids=["method", "property", "staticmethod"],
)
def test_live_attribute_answered_by_behaviour_not_state_fails(
    probe: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, member: str
) -> None:
    """Only plain non-callable state counts as a fix; a callable on a live branch is a stub."""
    shim = (
        "\nclass AstraDiffusionGemma(DiffusionGemmaForConditionalGeneration):\n"
        + member
        + "\n    def _process_audio_input(self, audio_input):\n"
        + "        raise NotImplementedError('no audio tower')\n"
    )
    audit = _audit(probe, tmp_path, monkeypatch, limits=IMAGE_VIDEO, shim_body=shim)
    lora = audit["methods"]["_process_image_input"]["unresolved"]["_enable_mm_lora"]
    assert lora["supplied_by_shim"] == "behaviour"
    with pytest.raises(AssertionError) as err:
        check_audit(audit)
    message = str(err.value)
    assert "self._enable_mm_lora" in message
    assert "no modality guard (always live)" in message


def test_state_assigned_in_shim_init_counts_as_state(
    probe: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shim = """
class AstraDiffusionGemma(DiffusionGemmaForConditionalGeneration):
    def __init__(self):
        super().__init__()
        self._enable_mm_lora = False
"""
    audit = _audit(probe, tmp_path, monkeypatch, limits=IMAGE_VIDEO, shim_body=shim)
    lora = audit["methods"]["_process_image_input"]["unresolved"]["_enable_mm_lora"]
    assert lora["supplied_by_shim"] == "state"
    # Audio is dead and unstubbed here: excluded, so the audit passes.
    assert audit["methods"]["embed_multimodal"]["unresolved"]["_process_audio_input"] == {
        "sites": 1,
        "guards": [["audio"]],
        "reachable_sites": [],
        "reachable_modalities": [],
        "excluded_as_dead": True,
        "supplied_by_shim": None,
    }
    check_audit(audit)
