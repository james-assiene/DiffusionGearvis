#!/usr/bin/env python3
"""Generator for ``tests/fixtures/audio/`` (provenance: see MANIFEST.md).

Two subcommands, both writing next to this file:

``chirp``
    ``chirp_5s_24k.wav`` — a deterministic linear chirp, no RNG, no service. The instantaneous
    frequency is an affine function of time, so a recording of it carries its own clock: the
    frequency of the last rendered sample says how many ms were rendered, independently of
    anything the module under test reports (docs/tasks/T-0006.md criterion 2).

``tts``
    ``pocket_tts_sentence.wav`` + ``pocket_tts_sentence.words.json`` — one sentence rendered by
    the Pocket TTS service on CPU (``services/tts.sh``), plus word timings recovered from the
    rendering by energy segmentation (the HTTP API returns audio only). The sentence is chosen so
    the segmentation is unambiguous: five words separated by >= 250 ms of silence. The script
    refuses to write the fixture if the number of segments found is not the number of words.

Usage:
    uv run python tests/fixtures/audio/make_fixtures.py chirp
    bash services/tts.sh &   # or an already-running Pocket TTS on 127.0.0.1:8081
    uv run python tests/fixtures/audio/make_fixtures.py tts
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SR = 24000

CHIRP_SECONDS = 5.0
CHIRP_F0_HZ = 200.0
CHIRP_F1_HZ = 6000.0
CHIRP_AMPLITUDE = 0.5

TTS_URL = "http://127.0.0.1:8081/tts"
TTS_SENTENCE = "one, two, three, four, five."
TTS_WORDS = ("one,", "two,", "three,", "four,", "five.")
#: energy segmentation of the rendering (10 ms frames, peak |x| per frame)
SEG_FRAME_MS = 10
SEG_THRESHOLD = 0.02
SEG_MERGE_GAP_MS = 80


def write_wav_s16(path: Path, x: np.ndarray) -> None:
    """Write float64 mono in [-1, 1] as 24 kHz s16le (the TTS output format, contracts/speech)."""
    if np.max(np.abs(x)) > 1.0:
        raise SystemExit("fixture would clip")
    pcm = np.round(x * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read_wav_s16(raw: bytes) -> np.ndarray:
    i = raw.index(b"data")
    return np.frombuffer(raw[i + 8 :], dtype="<i2").astype(np.float64) / 32768.0


def make_chirp() -> None:
    n = int(CHIRP_SECONDS * SR)
    t = np.arange(n, dtype=np.float64) / SR
    k = (CHIRP_F1_HZ - CHIRP_F0_HZ) / CHIRP_SECONDS
    phase = 2.0 * np.pi * (CHIRP_F0_HZ * t + 0.5 * k * t * t)
    write_wav_s16(HERE / "chirp_5s_24k.wav", CHIRP_AMPLITUDE * np.sin(phase))


def segment(x: np.ndarray) -> list[tuple[int, int]]:
    f = SEG_FRAME_MS * SR // 1000
    n = len(x) // f
    loud = np.abs(x[: n * f].reshape(n, f)).max(axis=1) > SEG_THRESHOLD
    segs: list[tuple[int, int]] = []
    start: int | None = None
    for k, a in enumerate(loud):
        if a and start is None:
            start = k
        elif not a and start is not None:
            segs.append((start * SEG_FRAME_MS, k * SEG_FRAME_MS))
            start = None
    if start is not None:
        segs.append((start * SEG_FRAME_MS, n * SEG_FRAME_MS))
    merged: list[tuple[int, int]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < SEG_MERGE_GAP_MS:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    return merged


def make_tts() -> None:
    import httpx

    resp = httpx.post(TTS_URL, data={"text": TTS_SENTENCE}, timeout=300.0)
    resp.raise_for_status()
    x = read_wav_s16(resp.content)
    segs = segment(x)
    if len(segs) != len(TTS_WORDS):
        raise SystemExit(f"segmentation found {len(segs)} segments for {len(TTS_WORDS)} words")
    write_wav_s16(HERE / "pocket_tts_sentence.wav", x)
    (HERE / "pocket_tts_sentence.words.json").write_text(
        json.dumps(
            {
                "text": TTS_SENTENCE,
                "sample_rate": SR,
                "duration_ms": round(1000 * len(x) / SR),
                "words": [
                    {"text": w, "start_ms": s, "end_ms": e}
                    for w, (s, e) in zip(TTS_WORDS, segs, strict=True)
                ],
            },
            indent=1,
        )
        + "\n"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("what", choices=["chirp", "tts"])
    args = p.parse_args()
    {"chirp": make_chirp, "tts": make_tts}[args.what]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
