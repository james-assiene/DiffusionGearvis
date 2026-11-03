"""T-0068 criterion 8 (tier 2, ``-m gpu``, ``/launch``, no engine boot): the per-step arms on CUDA.

Same mechanism as T-0067's ``test_vllm_cd_cuda_gpu.py`` (repository read-only at ``/repo``, nothing from the host
venv on the container's path -- pytest and every import come from the image, ruling S7 as amended --
``--noconftest``, exact pass counts, no skip / xfail / deselect accepted), with the
runner named by the T-0068 task file (``tests/container_scripts/astra_cd_perstep_probe.py``).

Rows:
- ``test_criterion1_arms_semantics_on_cuda``: ``test_cd_arms_semantics.py`` and ``test_cd_mar_vs_mf.py`` with
  ``ASTRA_CD_DEVICE=cuda ASTRA_CD_STOCK=image`` (the image's compiled stock step), 48 + 8 rows.
- ``test_perstep_peak_added_memory_on_cuda``: ``tests/cd/cuda/test_cd_perstep_cuda.py`` (8 rows: buckets 128, 256 for
  ``perstep_mar`` + constrained SC and for ``perstep_mar_qargmax`` + constrained SC at CL 256, added peak ≤ 1.5 GiB,
  TF32 "high" set globally; plus T-0076 criterion 12's CUDA semantics rows).
"""

from __future__ import annotations

import pytest

from tests.contract import test_vllm_cd_cuda_gpu as tier2

pytestmark = pytest.mark.gpu

PROBE = "/repo/tests/container_scripts/astra_cd_perstep_probe.py"
#: exact pass counts from the modules' definitions; a row added there means updating the constant
ARMS_MODULES = {
    "tests/cd/unit/test_cd_arms_semantics.py": 48,
    "tests/cd/unit/test_cd_mar_vs_mf.py": 8,
}
MEMORY = "tests/cd/cuda/test_cd_perstep_cuda.py"
#: 8 = buckets {128, 256} for perstep_mar + SC (T-0068 c8) and for perstep_mar_qargmax + SC (T-0076 c12), plus
#:     T-0076 c12's CUDA semantics: C3 (± SC) and the G1 fp64-oracle rows (seeds 1, 3)
MEMORY_PASSED = 8
CUDA_ENV = {"ASTRA_CD_DEVICE": "cuda", "ASTRA_CD_STOCK": "image"}


@pytest.fixture(autouse=True)
def _perstep_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tier2, "RUNNER", PROBE)


@pytest.mark.parametrize("module", sorted(ARMS_MODULES))
def test_criterion1_arms_semantics_on_cuda(module: str) -> None:
    tier2._pytest_in_container(module, ARMS_MODULES[module], CUDA_ENV)


def test_perstep_peak_added_memory_on_cuda() -> None:
    stdout = tier2._pytest_in_container(MEMORY, MEMORY_PASSED, CUDA_ENV)
    assert f"{tier2.PRECISION_MARKER} high" in stdout
    peaks = [ln for ln in stdout.splitlines() if ln.startswith("ASTRA_CD_PERSTEP_PEAK ")]
    assert len(peaks) == 4, f"expected four peak lines (2 modes x 2 buckets), got {peaks}"
