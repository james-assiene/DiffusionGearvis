"""T-0067 tier-2 runner, run INSIDE the pinned vLLM container (never on the host; no engine boot).

Driven by ``tests/contract/test_vllm_cd_cuda_gpu.py`` with the repository mounted read-only at ``/repo``.

1. Preflight: ``torch.cuda.is_available()`` and a device name containing "5090", else exit 3 before pytest runs
   (a CPU-only container must fail the host row, never skip its way to exit 0).
2. Prints one ``ASTRA_TIER2_REGIME <json>`` line: torch / CUDA / Python / vllm / pytest versions, where pytest
   was imported from, the device name and ``torch.get_float32_matmul_precision()`` as the run starts.
3. Runs pytest over ``argv[1:]`` and exits with its code. The host parses the summary line; exit 0 is not enough.

Nothing from the host venv is on ``sys.path`` (ruling S7 as amended 2026-09-30): pytest and every third-party
module the inner tests import come from the image. An earlier ``/hostsite`` append let transformers pick up the
host's cp311 scipy, which cannot load under the image's Python 3.12.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import sys

REGIME_MARKER = "ASTRA_TIER2_REGIME"
PREFLIGHT_FAILED = 3


def _version(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "absent"


def main() -> int:
    import torch

    cuda = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda else "none"
    import pytest

    regime = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "vllm": _version("vllm"),
        "pytest": pytest.__version__,
        "pytest_from": pytest.__file__,
        "cuda_available": cuda,
        "device": name,
        "fp32_matmul_precision_at_start": torch.get_float32_matmul_precision(),
    }
    print(f"{REGIME_MARKER} {json.dumps(regime, sort_keys=True)}", flush=True)
    if not cuda or "5090" not in name:
        print(
            f"PREFLIGHT FAILED: need CUDA on an RTX 5090, got cuda_available={cuda} device={name!r}",
            flush=True,
        )
        return PREFLIGHT_FAILED
    return int(pytest.main(sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(main())
