"""Stdlib host detokeniser of the pinned ``tokenizer.json`` (T-0088a criterion 2).

Turns streamed token ids into exactly the text ``tokenizers.Tokenizer.decode(ids, skip_special_tokens=False)``
gives on the pinned file (sha256 ``cc8d3a0c…``, rev ``2ea8372…``); ``tests/cd/unit/test_host_detok_parity.py``
is the oracle. Only the decoder side is read: there is no host encoding (seed tokenisation stays the engine's
``/tokenize``, H9). ``tokenizers`` stays out of the conductor (T-0063 criterion 2).

The decoder this reads is refused unless it is exactly ``Sequence[Replace("▁"→" "), ByteFallback, Fuse]``:

- ``Replace`` turns every ``▁`` of a piece into a space (no stripping);
- ``ByteFallback``: consecutive ``<0xNN>`` pieces form one byte run, ended by any other piece or by the end of
  the ids. A run that is valid UTF-8 as a whole decodes to its string; a run that is not becomes one U+FFFD per
  byte of the *whole* run (HF semantics, not ``bytes.decode("utf-8", "replace")``);
- ``Fuse`` concatenates.

Streaming (``stream()``): a byte run is held until a non-byte piece arrives or ``finish()`` is called, even when
it is already valid UTF-8, because a later byte can still turn the whole run into U+FFFD (ruling D4). Every
other piece, specials included, is emitted whole by the ``feed`` that carries it. Each id costs O(1) amortised,
so the stream is linear in the ids fed.

An id outside the loaded vocabulary raises ``UnknownTokenId`` (``tokenizers`` silently drops it; the host never
guesses, ruling D5).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

from local_astra.toolcall.special_tokens import SPECIAL_IDS

__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "TOKENIZER_SHA256",
    "DetokStream",
    "HostDetokenizer",
    "TokenizerFileError",
    "UnknownTokenId",
]

#: The pinned model and revision (``services/versions.lock``); the same values as ``decoding.vocab``, restated
#: because the conductor must not import ``local_astra.decoding``.
MODEL_ID: Final = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
MODEL_REVISION: Final = "2ea837236295d617ac27f8c17d61228081932c40"
TOKENIZER_SHA256: Final = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"

_EXPECTED_DECODER: Final[dict[str, Any]] = {
    "type": "Sequence",
    "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
    ],
}

_HEX: Final = frozenset("0123456789abcdefABCDEF")


class UnknownTokenId(LookupError):
    """An id that is not in the loaded vocabulary."""


class TokenizerFileError(ValueError):
    """The tokenizer file is not the one this detokeniser can decode exactly (the message names file and reason)."""


def _hf_hub_cache() -> Path:
    """The ``decoding.vocab._hf_hub_cache`` rule: ``HF_HUB_CACHE``, else ``HF_HOME/hub``, else ``~/.cache``."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def pinned_tokenizer_path() -> Path:
    """``tokenizer.json`` of the pinned revision in the local HF cache (it may not exist)."""
    snapshot = (
        _hf_hub_cache() / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / MODEL_REVISION
    )
    return snapshot / "tokenizer.json"


def _byte_of(piece: str) -> int | None:
    """The byte of a ``<0xNN>`` piece, by the HF ``ByteFallback`` rule (6 bytes long, ``<0x`` … ``>``, the middle
    two parsed as base-16 ``u8``, which admits one leading ``+``); ``None`` for any other piece."""
    raw = piece.encode("utf-8")
    if len(raw) != 6 or not raw.startswith(b"<0x") or not raw.endswith(b">"):
        return None
    mid = piece[3:5]
    digits = mid[1:] if mid.startswith("+") else mid
    if not digits or any(ch not in _HEX for ch in digits):
        return None
    return int(digits, 16)


def _flush_run(run: bytearray) -> str:
    try:
        return run.decode("utf-8")
    except UnicodeDecodeError:
        return "�" * len(run)


class DetokStream:
    """Incremental decoding: ``"".join(feed(c) for c in chunks) + finish()`` equals ``decode`` of all ids."""

    __slots__ = ("_pieces", "_run")

    def __init__(self, pieces: list[str | int | None]) -> None:
        self._pieces = pieces
        self._run = bytearray()

    def feed(self, ids: Iterable[int]) -> str:
        """Decode ``ids``; an open byte run at the end is held. Raises ``UnknownTokenId`` (state unchanged)."""
        pieces = self._pieces
        n = len(pieces)
        resolved: list[str | int] = []
        for i in ids:
            p = pieces[i] if 0 <= i < n else None
            if p is None:
                raise UnknownTokenId(f"token id {i!r} is not in the loaded vocabulary")
            resolved.append(p)
        run = self._run
        out: list[str] = []
        for p in resolved:
            if isinstance(p, int):
                run.append(p)
                continue
            if run:
                out.append(_flush_run(run))
                run.clear()
            out.append(p)
        return "".join(out)

    def finish(self) -> str:
        """Flush a held byte run (a truncated one renders as U+FFFD per byte)."""
        if not self._run:
            return ""
        text = _flush_run(self._run)
        self._run.clear()
        return text


class HostDetokenizer:
    """The decoder of one ``tokenizer.json``, checked against its expected sha256 and decoder shape."""

    def __init__(self, path: Path | str, expected_sha256: str) -> None:
        path = Path(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_sha256:
            raise TokenizerFileError(
                f"{path}: sha256 {digest} is not the expected {expected_sha256}"
            )
        try:
            tok = json.loads(raw)
        except ValueError as exc:
            raise TokenizerFileError(f"{path}: not valid JSON ({exc})") from exc
        model = tok.get("model") if isinstance(tok, dict) else None
        if not isinstance(model, dict) or model.get("type") != "BPE":
            raise TokenizerFileError(f"{path}: the model is not BPE")
        if model.get("byte_fallback") is not True:
            raise TokenizerFileError(f"{path}: the BPE model does not have byte_fallback: true")
        if tok.get("decoder") != _EXPECTED_DECODER:
            raise TokenizerFileError(
                f"{path}: the decoder is not exactly Sequence[Replace('▁'->' '), ByteFallback, Fuse]"
            )
        added = tok.get("added_tokens")
        if not isinstance(added, list):
            raise TokenizerFileError(f"{path}: added_tokens is missing")
        specials = sorted((a["id"], a["content"]) for a in added if a.get("special"))
        if specials != sorted(SPECIAL_IDS.items()):
            raise TokenizerFileError(
                f"{path}: the special added tokens are not the frozen SPECIAL_TOKENS table"
            )
        vocab = model.get("vocab")
        if not isinstance(vocab, dict):
            raise TokenizerFileError(f"{path}: the BPE model has no vocab")
        top = max([*vocab.values(), *(a["id"] for a in added)], default=-1)
        pieces: list[str | int | None] = [None] * (top + 1)
        #: T-0159: the literal of each byte-fallback id as written in the file (``pieces()`` only)
        byte_literals: dict[int, str] = {}
        for piece, i in vocab.items():
            byte = _byte_of(piece)
            pieces[i] = piece.replace("▁", " ") if byte is None else byte
            if byte is not None:
                byte_literals[i] = piece
        for (
            a
        ) in added:  # the added vocabulary is looked up first; added tokens are emitted verbatim
            pieces[a["id"]] = a["content"]
        self.path = path
        self._pieces = pieces
        self._byte_literals = byte_literals

    @classmethod
    def load(cls) -> HostDetokenizer:
        """The pinned ``tokenizer.json`` from the local HF cache (no network). Raises naming the path."""
        path = pinned_tokenizer_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing: fetch the tokenizer files of {MODEL_ID} rev {MODEL_REVISION}"
            )
        return cls(path, TOKENIZER_SHA256)

    def stream(self) -> DetokStream:
        return DetokStream(self._pieces)

    def decode(self, ids: Iterable[int]) -> str:
        s = self.stream()
        return s.feed(ids) + s.finish()

    def pieces(self, ids: Iterable[int]) -> tuple[str, ...]:
        """One display piece per id (T-0159 A2), for the denoising view; a pure per-id lookup.

        An ordinary piece has every ``▁`` replaced by a space; an added token (the specials
        included) is its content verbatim; a byte-fallback id is its own ``<0xNN>`` literal from
        the file, never merged with its neighbours (one position, one piece). So for ids with no
        byte-fallback id, ``"".join(pieces(ids)) == decode(ids)``. Raises ``UnknownTokenId``.
        """
        table = self._pieces
        n = len(table)
        out: list[str] = []
        for i in ids:
            p = table[i] if 0 <= i < n else None
            if p is None:
                raise UnknownTokenId(f"token id {i!r} is not in the loaded vocabulary")
            out.append(p if isinstance(p, str) else self._byte_literals[i])
        return tuple(out)
