"""T-0071a criterion 1 (a1, tier 2, ``-m gpu``, ``/launch``; container only, no engine boot): the per-step sync rows.

Runs ``tests/cd/cuda/test_cd_perstep_sync_cuda.py`` by pytest **inside** ``vllm/vllm-openai:v0.29.0`` through the CD
tier-2 runner (``tests/container_scripts/astra_cd_tier2_runner.py``: exits 3 before pytest unless CUDA on an RTX
5090), with T-0067's mechanism (``tests/contract/test_vllm_cd_cuda_gpu._pytest_in_container``): repo ``:ro`` at
``/repo``, ``--gpus all --network none``, ``--noconftest -o addopts= -p no:cacheprovider``,
``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``, ``ASTRA_CD_REQUIRE_CUDA=1`` (a skip is a failure), packages from the image
only, ``ASTRA_CD_DEVICE=cuda ASTRA_CD_STOCK=image`` (the image's compiled ``DiffusionSampler``). The host demands
**exactly** ``CD_PERSTEP_SYNC_PASSED`` passes (``tests/pytest_summary.require_exact``: no skip, xfail, deselect,
failure or error) and prints one regime line with the image digest, driver, the container's versions, host HEAD
and the dirty flag.

A raise here is a defect in ``cd/**``: its fix is its own task and a2 does not run until it lands (T-0069 O5).
In its own module so T-0069's ``test_vllm_cd_wiring_gpu.py`` and T-0068's ``test_vllm_cd_perstep_gpu.py`` stay
unedited.
"""

from __future__ import annotations

import pytest

from tests.contract import test_vllm_cd_cuda_gpu as tier2

pytestmark = pytest.mark.gpu

PERSTEP_SYNC_CUDA = "tests/cd/cuda/test_cd_perstep_sync_cuda.py"
#: 11 = sync-debug "error" really raises on a CUDA .item() (liveness, 1)
#:     + arms {perstep_mf, perstep_mar, perstep_mar + constrained SC, perstep_mar_qargmax ± constrained SC
#:       (T-0076 criterion 11)} x buckets {64, 128}: >= 12 stock-only
#:       warm-up calls with sync-debug off, then 50 Tap ∘ CD calls with one constrained slot (commits and
#:       stop-commit releases under "error"; binds under "error" cover ms.add_request only -- the tap's on_bind and
#:       CD's _bind, incl. a slot-reuse bind -- while CD's admission, sampler.add_request with its one host sync by
#:       design, runs relaxed, ruling R5): tap.errors == 0, no CD error record, records of the arm and bucket (10)
#: adding or removing a row in tests/cd/cuda/test_cd_perstep_sync_cuda.py means updating this constant
CD_PERSTEP_SYNC_PASSED = 11
CUDA_ENV = {"ASTRA_CD_DEVICE": "cuda", "ASTRA_CD_STOCK": "image"}


def test_perstep_arms_on_the_images_compiled_sampler_under_sync_debug_error() -> None:
    stdout = tier2._pytest_in_container(PERSTEP_SYNC_CUDA, CD_PERSTEP_SYNC_PASSED, CUDA_ENV)
    assert f"{tier2.PRECISION_MARKER} high" in stdout, (
        "the rows ran without TF32 'high' reported set globally (the a1 regime)"
    )
