"""The grammar vocabulary, read from the HF ``tokenizer.json`` of the pinned model revision (T-0063).

diffgemma-fa built its ``outlines_core.Vocabulary`` from the JAX ``gemma`` package's SentencePiece model.
This port reads the same pieces from ``tokenizer.json`` of ``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` at
the revision pinned in ``services/versions.lock``; the piece -> bytes rule is unchanged and lives in the
vendored ``dgfa_compile/vocab.py`` (see ``_vendor/PROVENANCE.md``). This module owns where the file is,
its sha256 (the automaton cache key's ``tokenizer_hash``) and the stop set from ``generation_config.json``.

The file is read from the local Hugging Face cache only; nothing here touches the network.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from local_astra.decoding._vendor.dgfa_compile import vocab as _vendored

__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "TOKENIZER_SHA256",
    "build_vocabulary",
    "encode",
    "load_tokenizer",
    "snapshot_dir",
    "stop_token_ids",
    "tokenizer_hash",
    "tokenizer_json_path",
]

MODEL_ID = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
#: services/versions.lock
MODEL_REVISION = "2ea837236295d617ac27f8c17d61228081932c40"
#: sha256 of that revision's ``tokenizer.json`` (measured 2026-09-29). A different file means a
#: different vocabulary, so every automaton compiled against it would be silently wrong.
TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"


def _hf_hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def snapshot_dir() -> Path:
    """The pinned snapshot directory in the local HF cache."""
    return _hf_hub_cache() / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / MODEL_REVISION


def tokenizer_json_path() -> Path:
    path = snapshot_dir() / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing: fetch the tokenizer files of {MODEL_ID} rev {MODEL_REVISION}"
        )
    return path


@functools.cache
def tokenizer_hash() -> str:
    """sha256 of the pinned ``tokenizer.json``; raises if the file on disk is not the pinned one."""
    digest = hashlib.sha256(tokenizer_json_path().read_bytes()).hexdigest()
    if digest != TOKENIZER_SHA256:
        raise ValueError(
            f"{tokenizer_json_path()} has sha256 {digest}, not the pinned {TOKENIZER_SHA256}"
        )
    return digest


@functools.cache
def load_tokenizer() -> Any:
    """The ``tokenizers.Tokenizer`` of the pinned revision (hash-checked)."""
    from tokenizers import Tokenizer

    tokenizer_hash()
    return Tokenizer.from_file(str(tokenizer_json_path()))


@functools.cache
def stop_token_ids() -> tuple[int, ...]:
    """``eos_token_id`` of the pinned ``generation_config.json``, in file order ([1, 106, 50])."""
    cfg = json.loads((snapshot_dir() / "generation_config.json").read_text())
    ids = cfg["eos_token_id"]
    return tuple(int(t) for t in (ids if isinstance(ids, list) else [ids]))


def build_vocabulary() -> Any:
    """The ``outlines_core.Vocabulary`` every lift uses (cached by the vendored module).

    Reserved ids ``{0, 1, 50, 100, 101, 105, 106}`` and the four control pieces are excluded.
    """
    return _vendored.build_vocabulary()


def encode(text: str) -> list[int]:
    """HF tokenisation of ``text``: special pieces parsed, no BOS."""
    return list(load_tokenizer().encode(text, add_special_tokens=False).ids)
