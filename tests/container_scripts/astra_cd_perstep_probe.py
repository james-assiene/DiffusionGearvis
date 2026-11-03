"""T-0068 tier-2 probe, run INSIDE the pinned vLLM container (no engine boot) by
``tests/contract/test_vllm_cd_perstep_gpu.py``.

It is the T-0067 tier-2 runner (``astra_cd_tier2_runner.py``: CUDA-on-a-5090 preflight, one
``ASTRA_TIER2_REGIME <json>`` line, then pytest over ``argv[1:]``) under the name the T-0068 task file gives;
the per-step rows themselves live in ``tests/cd/cuda/test_cd_perstep_cuda.py`` and the unit modules re-run on CUDA.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    runner = Path(__file__).resolve().parent / "astra_cd_tier2_runner.py"
    sys.argv[0] = str(runner)
    runpy.run_path(str(runner), run_name="__main__")
