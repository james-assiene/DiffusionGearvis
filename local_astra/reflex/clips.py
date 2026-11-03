"""The clip registry (T-0182 A1): where each clip's asset lives, which kinds play by default, and whether an asset
is current.

A clip is a fixed, pre-rendered system utterance (``Session._clip_pcm`` reads ``<data_dir>/clips/<kind>.s16le``, raw
24 kHz mono s16le). ``scripts/render_clips.py`` renders the assets once through Pocket TTS and records each in
``<data_dir>/clips/manifest.json``; ``doctor``'s ``clips`` row reads :func:`status` for the default-enabled kinds.

Pure apart from file reads. "Registered" is computed at call time: the kind is in ``get_args(ClipKind)`` **and** has a
``CLIP_TEXT`` entry, both read from their modules on every call.
"""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, get_args

import pydantic

from local_astra.contracts.base import Contract

ENDING_KINDS: frozenset[str] = frozenset(
    {"out_of_steps", "not_finished", "lost_thread", "at_checkout"}
)
PROGRESS_KINDS: frozenset[str] = frozenset({"still_working"})
FILLER_KINDS: frozenset[str] = frozenset({"ack", "hmm"})

SAMPLE_RATE = 24_000
MANIFEST_NAME = "manifest.json"
LOCK_PATH = Path(__file__).resolve().parents[2] / "services" / "versions.lock"

ClipStatus = Literal["current", "missing", "stale"]


def clips_dir(data_dir: Path | str) -> Path:
    return Path(data_dir) / "clips"


def clip_path(data_dir: Path | str, kind: str) -> Path:
    """The file ``Session._clip_pcm(kind)`` reads."""
    return clips_dir(data_dir) / f"{kind}.s16le"


def manifest_path(data_dir: Path | str) -> Path:
    return clips_dir(data_dir) / MANIFEST_NAME


def registered() -> dict[str, str]:
    """``{kind: text}`` for every kind in ``ClipKind`` that has a ``CLIP_TEXT`` entry, read now (not at import)."""
    from local_astra.conductor import session
    from local_astra.contracts import commands

    kinds = get_args(commands.ClipKind)
    return {kind: session.CLIP_TEXT[kind] for kind in kinds if kind in session.CLIP_TEXT}


def default_enabled(registered: Iterable[str], *, filler_clips: bool = False) -> frozenset[str]:
    """The kinds a session can play by default (T-0182 B6): the ending kinds and ``still_working`` (live tool loops),
    plus ``ack``/``hmm`` only when ``filler_clips`` is on; restricted to ``registered``."""
    enabled = ENDING_KINDS | PROGRESS_KINDS | (FILLER_KINDS if filler_clips else frozenset())
    return enabled & frozenset(registered)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tts_pin(lock_path: Path | None = None) -> str:
    """The ``pocket_tts`` pin from ``services/versions.lock``."""
    with (lock_path or LOCK_PATH).open("rb") as handle:
        return str(tomllib.load(handle)["speech"]["pocket_tts"])


class ClipEntry(Contract):
    text: str
    text_sha256: str
    pcm_sha256: str
    bytes: int
    duration_ms: int
    peak_dbfs: float
    rms_dbfs: float
    sample_rate: Literal[24000]
    channels: Literal[1]
    sample_format: Literal["s16le"]
    tts: str
    rendered_at: str
    git_commit: str


class ClipManifest(Contract):
    """``<data_dir>/clips/manifest.json``: ``{"clips": {kind: entry}}``."""

    clips: dict[str, ClipEntry]


def read_manifest(data_dir: Path | str) -> ClipManifest | None:
    """The manifest, or ``None`` when it is absent or unreadable."""
    path = manifest_path(data_dir)
    if not path.is_file():
        return None
    try:
        return ClipManifest.model_validate_json(path.read_bytes())
    except (OSError, ValueError, pydantic.ValidationError):
        return None


def status(data_dir: Path | str, kind: str, text: str, tts_pin: str) -> ClipStatus:
    """``missing`` without an asset; ``stale`` without a manifest entry or with another text or pin; else ``current``.
    ``rendered_at`` and ``git_commit`` are provenance, not staleness."""
    if not clip_path(data_dir, kind).is_file():
        return "missing"
    manifest = read_manifest(data_dir)
    entry = manifest.clips.get(kind) if manifest is not None else None
    if entry is None or entry.text_sha256 != text_sha256(text) or entry.tts != tts_pin:
        return "stale"
    return "current"
