#!/usr/bin/env python3
"""Render the clip assets once through the running Pocket TTS (T-0182 A2).

Usage: ``uv run python scripts/render_clips.py [--data-dir D] [--profile P] [--kinds K ...] [--force]``

Each kind is rendered from ``CLIP_TEXT[kind]`` verbatim through ``PocketTtsClient(profile.services)`` with
``DEFAULT_VOICE`` (the same client, voice and format as speech) into ``<data_dir>/clips/<kind>.s16le``: raw 24 kHz
mono s16le, the concatenated ``audio()`` chunks, with no header, resampling, gain, normalisation or trim. The manifest
``<data_dir>/clips/manifest.json`` records what was rendered (``local_astra.reflex.clips.ClipManifest``).

- **Which kinds:** by default ``default_enabled(registered)``, the ending kinds that are registered. ``--kinds`` names
  others (ack, hmm, still_working are rendered only when named); an unregistered kind is refused (exit 2).
- **Idempotent:** a ``current`` kind is skipped unless ``--force``. One line per kind: ``rendered``,
  ``skipped (current)`` or ``failed: <reason>``.
- **Validated, then atomic:** a body must be non-empty with an even byte count, last 0.3-6.0 s, not be all zeros
  and hold no full-scale sample (32767 or -32768). A valid body is written as ``.tmp`` and renamed; the manifest is
  rewritten the same way, after the assets. A failed kind writes nothing, and the run exits 1.

Needs ``services/tts.sh`` up; CPU only; the only socket is the loopback TTS (``Services`` validates the URL).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import math
import os
import struct
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from local_astra.conductor.tts_client import (
    DEFAULT_VOICE,
    PocketTtsClient,
    TtsClientError,
)
from local_astra.config import load
from local_astra.memory import store
from local_astra.reflex import clips

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_SAMPLES = 7_200  # 0.3 s at 24 kHz, inclusive
MAX_SAMPLES = 144_000  # 6.0 s at 24 kHz, inclusive

__all__ = ["PocketTtsClient", "main"]


class InvalidClip(Exception):
    pass


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _dbfs(level: float) -> float:
    return 20 * math.log10(level / 32768)


def validate(pcm: bytes) -> tuple[float, float]:
    """A2's rules; returns ``(peak_dbfs, rms_dbfs)`` of a valid body, raises :class:`InvalidClip` otherwise."""
    if not pcm:
        raise InvalidClip("empty body")
    if len(pcm) % 2:
        raise InvalidClip(f"odd byte count ({len(pcm)})")
    n = len(pcm) // 2
    if not MIN_SAMPLES <= n <= MAX_SAMPLES:
        raise InvalidClip(f"duration {n / clips.SAMPLE_RATE:.3f} s is outside 0.3-6.0 s")
    values = struct.unpack(f"<{n}h", pcm)
    peak = max(abs(min(values)), abs(max(values)))
    if peak == 0:
        raise InvalidClip("all zeros")
    if max(values) == 32767 or min(values) == -32768:
        raise InvalidClip("a full-scale sample (clipped)")
    rms = math.sqrt(sum(v * v for v in values) / n)
    return _dbfs(peak), _dbfs(rms)


async def _synthesise(client: Any, kind: str, text: str) -> bytes:
    utterance = await client.speak(f"clip-{kind}", text, DEFAULT_VOICE)
    chunks = [chunk async for chunk in utterance.audio()]
    return b"".join(chunks)


def _write_atomic(path: Path, body: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(body)
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="render_clips.py", description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, default=None, help="default: memory.store.DEFAULT_DIR")
    p.add_argument("--profile", default=None, help="the astra.toml profile whose services to use")
    p.add_argument("--config", type=Path, default=Path("astra.toml"))
    p.add_argument(
        "--kinds", nargs="+", default=None, help="kinds to render (default: the default set)"
    )
    p.add_argument("--force", action="store_true", help="re-render kinds that are current")
    return p.parse_args(list(argv) if argv is not None else None)


async def _render(
    services: Any, data_dir: Path, todo: dict[str, str], force: bool, pin: str
) -> int:
    failed = False
    pending = {
        k: t for k, t in todo.items() if force or clips.status(data_dir, k, t, pin) != "current"
    }
    for kind in todo:
        if kind not in pending:
            print(f"{kind}: skipped (current)")
    if not pending:
        return 0
    try:
        client = PocketTtsClient(services)
    except ValueError as exc:
        print(f"render_clips: refused: {exc}", file=sys.stderr)
        return 2
    rendered: dict[str, clips.ClipEntry] = {}
    commit = _git_commit()
    try:
        for kind, text in pending.items():
            try:
                pcm = await _synthesise(client, kind, text)
                peak, rms = validate(pcm)
                path = clips.clip_path(data_dir, kind)
                path.parent.mkdir(parents=True, exist_ok=True)
                _write_atomic(path, pcm)
            except (TtsClientError, InvalidClip, OSError) as exc:
                failed = True
                print(f"{kind}: failed: {type(exc).__name__}: {exc}")
                continue
            rendered[kind] = clips.ClipEntry(
                text=text,
                text_sha256=clips.text_sha256(text),
                pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                bytes=len(pcm),
                duration_ms=len(pcm) // 2 * 1000 // clips.SAMPLE_RATE,
                peak_dbfs=peak,
                rms_dbfs=rms,
                sample_rate=24000,
                channels=1,
                sample_format="s16le",
                tts=pin,
                rendered_at=datetime.now(UTC).isoformat(timespec="seconds"),
                git_commit=commit,
            )
            print(
                f"{kind}: rendered ({len(pcm) // 48} ms, peak {peak:.1f} dBFS, rms {rms:.1f} dBFS)"
            )
    finally:
        await client.aclose()
    if rendered:
        old = clips.read_manifest(data_dir)
        entries = {**(old.clips if old is not None else {}), **rendered}
        manifest = clips.ClipManifest(clips=dict(sorted(entries.items())))
        try:
            _write_atomic(
                clips.manifest_path(data_dir), manifest.model_dump_json(indent=2).encode()
            )
        except OSError as exc:
            print(f"render_clips: failed to write the manifest: {exc}", file=sys.stderr)
            return 1
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    registry = clips.registered()
    if args.kinds is None:
        kinds = sorted(clips.default_enabled(registry))
    else:
        kinds = [k for item in args.kinds for k in item.split(",") if k]
        unknown = [k for k in kinds if k not in registry]
        if unknown:
            print(
                f"render_clips: not a registered clip kind: {', '.join(unknown)} "
                f"(registered: {', '.join(sorted(registry))})",
                file=sys.stderr,
            )
            return 2
    if not kinds:
        print("render_clips: no default clips registered; nothing to render")
        return 0
    cfg = load(args.config)
    if args.profile is not None and args.profile not in cfg.profiles:
        print(
            f"render_clips: no profile {args.profile!r} in {args.config} (it has {sorted(cfg.profiles)})",
            file=sys.stderr,
        )
        return 2
    profile = cfg.profiles[args.profile] if args.profile else cfg.active()
    data_dir = args.data_dir if args.data_dir is not None else store.DEFAULT_DIR
    todo = {kind: registry[kind] for kind in dict.fromkeys(kinds)}
    return asyncio.run(_render(profile.services, data_dir, todo, args.force, clips.tts_pin()))


if __name__ == "__main__":
    sys.exit(main())
