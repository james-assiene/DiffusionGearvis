"""The ``ASTRA_CD <json>`` record emitter and the per-process error counter (T-0067 criterion 8).

One line per committed canvas per constrained request, plus exactly one error record per failing request.
The field list is ``record_schema.RECORD_FIELDS`` (the numpy-free mirror of T-0064's ``CdCanvasRecord``;
the container has no ``local_astra``). Every field is written, null where it does not apply.

The logger is a child of ``vllm`` (``vllm.astra_cd``) so that, inside the engine, the line goes out
through vLLM's own configured handler (the ``vllm`` logger has one, the root logger usually does not);
outside vLLM it propagates to the root logger as any other record.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from typing import Any

from .record_schema import END_PROJECTION_FIELDS, RECORD_FIELDS

__all__ = ["LOGGER_NAME", "PREFIX", "emit", "error_count", "make_record"]

LOGGER_NAME = "vllm.astra_cd"
PREFIX = "ASTRA_CD "

_logger = logging.getLogger(LOGGER_NAME)
_lock = threading.Lock()
_errors = 0


def error_count() -> int:
    """Error records emitted by this process (criterion 7)."""
    with _lock:
        return _errors


def make_record(
    *,
    record_tag: str | None,
    arm: str | None,
    canvas_idx: int,
    committed_ids: Sequence[int],
    state_set_after: Sequence[int],
    in_final: bool,
    steps: int,
    bucket: int | None,
    error: str | None,
    shadow_argmax_ids: Sequence[int] | None = None,
    argmax_in_language: bool | None = None,
    map_equals_argmax: bool | None = None,
    stop_ids: Sequence[int] = (),
    mean_accepted_per_step: float | None = None,
) -> dict[str, Any]:
    """One canvas record. ``mean_accepted_per_step`` (T-0068 criterion 7, per-step arms only): the mean over the
    canvas's denoise steps of the accepted positions in ``[0, L_eff)``; null for ``end_projection``."""
    ids = [int(x) for x in committed_ids]
    stop_pos = next((i for i, t in enumerate(ids) if t in stop_ids), None)
    rec: dict[str, Any] = {
        "record_tag": record_tag,
        "arm": arm,
        "canvas_idx": int(canvas_idx),
        "committed_ids": ids,
        "state_set_after": [int(s) for s in state_set_after],
        "in_final": bool(in_final),
        "steps": max(0, int(steps)),
        "bucket": bucket,
        "stop_pos": stop_pos,
        "error": error,
        "mean_accepted_per_step": (
            None
            if mean_accepted_per_step is None or arm == "end_projection"
            else float(mean_accepted_per_step)
        ),
        "shadow_argmax_ids": None,
        "argmax_in_language": None,
        "map_equals_argmax": None,
    }
    if arm == "end_projection":
        rec["shadow_argmax_ids"] = (
            None if shadow_argmax_ids is None else [int(x) for x in shadow_argmax_ids]
        )
        rec["argmax_in_language"] = argmax_in_language
        rec["map_equals_argmax"] = map_equals_argmax
    if arm != "end_projection":
        for name in END_PROJECTION_FIELDS:
            rec[name] = None
    assert tuple(rec) == RECORD_FIELDS, "record fields drifted from record_schema"
    return rec


def emit(record: dict[str, Any]) -> None:
    """Log one ``ASTRA_CD <json>`` line; an error record also bumps the per-process counter."""
    global _errors
    if record.get("error") is not None:
        with _lock:
            _errors += 1
        _logger.warning("%s%s", PREFIX, json.dumps(record, separators=(",", ":")))
    else:
        _logger.info("%s%s", PREFIX, json.dumps(record, separators=(",", ":")))
