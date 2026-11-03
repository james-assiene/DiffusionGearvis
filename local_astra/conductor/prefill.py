"""Anticipatory prefill bookkeeping (docs/01 D9). Task T-0005.

Opened at speech onset with the prefix (system, history, current frame); re-issued rate-limited
on scene change; aborted at end-of-turn before ``llm.start``. The final request's prefix must
tokenise identically (tested on token ids after templating; the last partial block may miss).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillPolicy:
    reissue_min_interval_ms: int = 1_000
    max_reissues_per_turn: int = 3
