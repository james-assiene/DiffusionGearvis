"""One window of mic input and recogniser activity, for the ``mic.level`` and ``mic.unmute_check`` notes (T-0195 B).

Log-only diagnostics: nothing here reaches the model, the HUD, history, TTS or ``turns.jsonl``. The session builds one
:class:`InputWindow` per row and hands it, in order, every captured frame (``live_frame`` for one the mic gate passes
live, ``muted_frame`` for one the mute feeds as zeros: B3, muted samples are never measured), every recogniser word and
step at ``_on_stt_event`` entry, and every endpointer onset; ``summary()`` is the row's measurement fields.

Pure, as `.claude/rules/conductor.md` asks of the reflex lane: no clock, no I/O, no logging, no asyncio. Frame time
arrives as the frame's own capture ``t_ms``. State is O(1) per window (B7): counters, sums and extremes, no buffers.

Levels are over the *qualifying* frames only: live, not all exactly zero, all finite (reading B-R6). ``rms_dbfs`` is
``10 log10`` of the mean square over all their samples; ``peak_dbfs`` is ``20 log10`` of the largest ``|sample|``;
``max_frame_rms_dbfs``/``min_frame_rms_dbfs`` are the extremes of :func:`local_astra.reflex.level.rms_dbfs` per frame.
Every level is a finite float floored at :data:`SILENCE_DBFS`, or ``None`` when no frame qualifies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from local_astra.reflex.level import SILENCE_DBFS, rms_dbfs
from local_astra.reflex.turn_predictor import PAUSE_HEAD_INDEX

#: B1: one periodic ``mic.level`` row per this much capture time (frame clock)
LEVEL_WINDOW_MS = 10_000
#: B4: the ``mic.unmute_check`` covers this much capture time from the unmute frame
UNMUTE_CHECK_MS = 5_000


def _floored_db(value: float) -> float:
    """``10 log10(value)`` floored at :data:`SILENCE_DBFS`; a non-positive value is the floor."""
    if value <= 0.0:
        return SILENCE_DBFS
    return max(10.0 * math.log10(value), SILENCE_DBFS)


class InputWindow:
    """One window's counts and level statistics; a new window is a new object (nothing carries across rows)."""

    __slots__ = (
        "_first_t_ms",
        "_frame_db_max",
        "_frame_db_min",
        "_frames",
        "_head_max",
        "_head_min",
        "_last_t_ms",
        "_muted_frames",
        "_nonfinite_frames",
        "_onsets",
        "_peak",
        "_samples",
        "_steps",
        "_sum_squares",
        "_words",
        "_words_dropped_muted",
        "_words_estimated",
        "_zero_frames",
    )

    def __init__(self) -> None:
        self._first_t_ms: float | None = None
        self._last_t_ms: float | None = None
        self._frames = 0
        self._muted_frames = 0
        self._zero_frames = 0
        self._nonfinite_frames = 0
        self._sum_squares = 0.0
        self._samples = 0
        self._peak = 0.0
        self._frame_db_max: float | None = None
        self._frame_db_min: float | None = None
        self._words = 0
        self._words_estimated = 0
        self._words_dropped_muted = 0
        self._steps = 0
        self._onsets = 0
        self._head_min: float | None = None
        self._head_max: float | None = None

    def _frame_seen(self, t_ms: float) -> None:
        if self._first_t_ms is None:
            self._first_t_ms = t_ms
        self._last_t_ms = t_ms
        self._frames += 1

    def live_frame(self, samples: npt.NDArray[np.floating[Any]], t_ms: float) -> None:
        """One frame the mic gate passes live. ``samples`` is read, never kept or modified."""
        self._frame_seen(t_ms)
        block = np.asarray(samples)
        sum_squares = float(np.square(block, dtype=np.float64).sum()) if block.size else 0.0
        if not math.isfinite(sum_squares):
            self._nonfinite_frames += 1
            return
        if sum_squares == 0.0:
            self._zero_frames += 1
            return
        self._sum_squares += sum_squares
        self._samples += int(block.size)
        self._peak = max(self._peak, float(np.max(np.abs(block))))
        frame_db = rms_dbfs(block)
        if self._frame_db_max is None or frame_db > self._frame_db_max:
            self._frame_db_max = frame_db
        if self._frame_db_min is None or frame_db < self._frame_db_min:
            self._frame_db_min = frame_db

    def muted_frame(self, t_ms: float) -> None:
        """One frame the mute feeds as zeros (B3: its captured samples are never handed over)."""
        self._frame_seen(t_ms)
        self._muted_frames += 1

    def word(self, *, estimated: bool, dropped_muted: bool) -> None:
        """One ``SttWordEvent`` at ``_on_stt_event`` entry (its text is never handed over)."""
        self._words += 1
        if estimated:
            self._words_estimated += 1
        if dropped_muted:
            self._words_dropped_muted += 1

    def stt_step(self, pause_heads: Sequence[float]) -> None:
        """One ``SttStepEvent``'s ``pause_heads``; a vector too short for ``PAUSE_HEAD_INDEX`` counts as a step only."""
        self._steps += 1
        if len(pause_heads) <= PAUSE_HEAD_INDEX:
            return
        head = float(pause_heads[PAUSE_HEAD_INDEX])
        if not math.isfinite(head):
            return
        if self._head_min is None or head < self._head_min:
            self._head_min = head
        if self._head_max is None or head > self._head_max:
            self._head_max = head

    def onset(self) -> None:
        """One ``("onset", ...)`` endpointer decision."""
        self._onsets += 1

    def summary(self) -> dict[str, Any]:
        """The B1 measurement fields (the row minus ``what``, ``reason`` and ``stt``)."""
        measured = self._samples > 0
        return {
            "from_frame_t_ms": self._first_t_ms,
            "to_frame_t_ms": self._last_t_ms,
            "frames": self._frames,
            "muted_frames": self._muted_frames,
            "zero_frames": self._zero_frames,
            "nonfinite_frames": self._nonfinite_frames,
            "rms_dbfs": _floored_db(self._sum_squares / self._samples) if measured else None,
            "peak_dbfs": _floored_db(self._peak * self._peak) if measured else None,
            "max_frame_rms_dbfs": self._frame_db_max,
            "min_frame_rms_dbfs": self._frame_db_min,
            "words_received": self._words,
            "words_estimated": self._words_estimated,
            "words_dropped_muted": self._words_dropped_muted,
            "steps_received": self._steps,
            "onsets": self._onsets,
            "pause_head_min": self._head_min,
            "pause_head_max": self._head_max,
        }
