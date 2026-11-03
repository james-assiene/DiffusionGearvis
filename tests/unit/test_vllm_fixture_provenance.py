"""T-0011 R8: the vendored v0.29.0 stock code under ``tests/fixtures/vllm_v0_29_0/`` is byte-identical,
segment by segment, to the pinned image's source as recorded in ``manifest.json``.

Torch-free (reads text, hashes it; never imports the vendored modules), so it runs in the fast suite.
The image side -- ``manifest.json`` equals the image's source -- is criterion 11's
``tests/contract/test_vllm_tap_fields_gpu.py`` (``-m gpu``). A local edit inside a ``# >>> vendored``
block, a segment that went missing, or a manifest entry nobody vendored fails here.

Green on arrival by design: this row guards the R8 copy (taken from ``T-0062-perstep-cd`` @ e10bcd0),
not the implementation; the fixture is what tier 1b (``tests/stock``) treats as the oracle.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "vllm_v0_29_0"
MAN = json.loads((FIX / "manifest.json").read_text())
BLOCK = re.compile(r"# >>> vendored (\S+)\n(.*?)# <<< vendored \1\n", re.S)
VENDORED_FILES = ("diffusion_gemma_step.py", "diffusion_sampler.py")
#: v0.29.0 @ sha256:c2914767..., the pin in services/versions.lock (re-derive after a vLLM bump)
IMAGE = (
    "vllm/vllm-openai:v0.29.0@sha256:"
    "c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1"
)


def _blocks() -> dict[str, str]:
    out: dict[str, str] = {}
    for f in VENDORED_FILES:
        for m in BLOCK.finditer((FIX / f).read_text()):
            assert m.group(1) not in out, f"segment {m.group(1)} vendored twice"
            out[m.group(1)] = m.group(2)
    return out


@pytest.mark.parametrize("name", sorted(MAN["segments"]))
def test_vendored_segment_hashes_to_the_manifest(name: str) -> None:
    blocks = _blocks()
    assert name in blocks, f"segment {name} is in manifest.json but not vendored"
    got = hashlib.sha256(blocks[name].encode()).hexdigest()
    assert got == MAN["segments"][name]["sha256"], (
        f"segment {name} differs from the image's source ({got}); the vendored code was edited"
    )


def test_every_vendored_segment_is_in_the_manifest() -> None:
    assert set(_blocks()) == set(MAN["segments"]), "vendored blocks and manifest segments differ"


def test_the_manifest_names_the_pinned_image() -> None:
    assert MAN["image"] == IMAGE
    lock = (Path(__file__).resolve().parents[2] / "services" / "versions.lock").read_text()
    tag = IMAGE.split("@")[0]
    assert f'vllm_container = "{tag}"' in lock, (
        f"services/versions.lock no longer pins {tag}, the image the fixture was extracted from "
        "(the lock pins the tag; the digest is recorded in manifest.json)"
    )


def test_the_step_decorator_is_the_only_omission() -> None:
    lines = (FIX / "diffusion_gemma_step.py").read_text().splitlines()
    assert not [ln for ln in lines if ln.startswith("@torch.compile")], (
        "the step must run eagerly on CPU (tier 1b)"
    )


def test_the_hash_check_is_live() -> None:
    """The instrument fires: a one-character edit inside a block changes its hash."""
    name, text = next(iter(_blocks().items()))
    assert hashlib.sha256((text + " ").encode()).hexdigest() != MAN["segments"][name]["sha256"]
