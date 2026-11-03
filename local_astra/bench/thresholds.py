"""Bench thresholds from docs/03 §7. Changing one requires a docs/tasks/ entry (CLAUDE.md)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Threshold:
    bench: str
    metric: str
    limit: float
    unit: str
    note: str = ""


THRESHOLDS: tuple[Threshold, ...] = (
    Threshold(
        "B3", "last_word_to_first_audio_p50_ms", 1500, "ms", "G1 p50; regime lean, canvas 256"
    ),
    Threshold("B3", "last_word_to_first_audio_p95_ms", 2200, "ms", "G1 p95"),
    Threshold("B4", "barge_in_pause_p95_ms", 300, "ms", "speech onset → playback paused"),
    Threshold(
        "B7", "vram_total_gib", 29.0, "GiB", "lean profile incl. sampler transient head-room"
    ),
    Threshold("B11", "prefix_cache_hit_ratio", 0.9, "ratio", "prefill-then-generate; ≥"),
    Threshold("B16", "tap_overhead_pct", 3.0, "%", "plugin per-step tap vs no plugin"),
    Threshold(
        "B13",
        "cd_step_overhead_pct_b64",
        10.0,
        "%",
        "perstep_mar per-step vs unconstrained, bucket 64; T-0062 decision 8, T-0071a",
    ),
    Threshold(
        "B13",
        "cd_step_overhead_pct_b128",
        15.0,
        "%",
        "perstep_mar per-step vs unconstrained, bucket <= 128; T-0062 decision 8, T-0071a",
    ),
)
