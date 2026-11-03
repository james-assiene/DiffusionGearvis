"""The vendored v0.29.0 stock code under ``tests/fixtures/vllm_v0_29_0/`` is byte-identical to the pinned
image's segments recorded in ``manifest.json`` (tier 2's ``test_vllm_cd_coupling_gpu.py`` checks the image
side). A local edit inside a ``# >>> vendored`` block fails here."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "vllm_v0_29_0"
MAN = json.loads((FIX / "manifest.json").read_text())
BLOCK = re.compile(r"# >>> vendored (\S+)\n(.*?)# <<< vendored \1\n", re.S)


def _blocks() -> dict[str, str]:
    out: dict[str, str] = {}
    for f in ("diffusion_gemma_step.py", "diffusion_sampler.py"):
        for m in BLOCK.finditer((FIX / f).read_text()):
            out[m.group(1)] = m.group(2)
    return out


@pytest.mark.parametrize("name", sorted(MAN["segments"]))
def test_vendored_segment_matches_manifest(name: str) -> None:
    blocks = _blocks()
    assert name in blocks, f"segment {name} is not vendored"
    assert hashlib.sha256(blocks[name].encode()).hexdigest() == MAN["segments"][name]["sha256"]


def test_step_decorator_is_the_only_omission() -> None:
    lines = (FIX / "diffusion_gemma_step.py").read_text().splitlines()
    assert not [ln for ln in lines if ln.startswith("@torch.compile")], (
        "the step must run eagerly on CPU"
    )
