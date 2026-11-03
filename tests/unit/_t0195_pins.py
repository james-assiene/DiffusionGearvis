"""T-0195 group B census amendment: the one declared difference full-session-log differentials may discount
(docs/tasks/T-0195.md, "Group B census amendment", orchestrator ruling 2026-10-29).

Group B adds two log-only ``note`` records to the session log: ``mic.level`` (B1, every 10 s of capture time and at
each mute switch) and ``mic.unmute_check`` (B4). Differentials written before B compare a session's whole log with an
older commit's, and any rig running longer than 10 s of capture time now holds a ``mic.level`` row. Following the
``_t0197_pins.admit_hud_line`` precedent, :func:`without_diagnostic_notes` returns a capture with exactly those
records removed from every list and tuple, after checking each one:

- a record is removed only if it is a dict with ``kind == "note"`` whose ``data`` is a dict with ``what`` in
  :data:`DIAGNOSTIC_WHATS`; ``mic.gate``, ``mic.mute``, ``stt.estimated_word`` and every other note are compared as
  before;
- its ``data`` must have exactly B1's (``mic.level``) or B4's (``mic.unmute_check``) key set;
- no value may carry text or PCM: ``what`` and ``reason`` are the strings B1 names, ``stt`` is ``None`` or a flat
  mapping of numbers, every other value is ``None``, a bool or a number.

Anything else raises :class:`DiagnosticPinError`, so the move admits these two notes and nothing more. The base side
needs no normalisation: BASE writes neither note, so there the filter is a no-op. ``_t0177_pins.without_vision_frames``
chains this, which carries it to every row that routes through ``_t0177_pins.observed`` or
``_t0162c_pins.observed``.

Regime: CPU, pure, no I/O.
"""

from __future__ import annotations

from typing import Any

LEVEL = "mic.level"
CHECK = "mic.unmute_check"
DIAGNOSTIC_WHATS = frozenset({LEVEL, CHECK})

#: B1, verbatim
LEVEL_KEYS = frozenset(
    {
        "what",
        "reason",
        "from_frame_t_ms",
        "to_frame_t_ms",
        "frames",
        "muted_frames",
        "zero_frames",
        "nonfinite_frames",
        "rms_dbfs",
        "peak_dbfs",
        "max_frame_rms_dbfs",
        "min_frame_rms_dbfs",
        "words_received",
        "words_estimated",
        "words_dropped_muted",
        "steps_received",
        "onsets",
        "pause_head_min",
        "pause_head_max",
        "stt",
    }
)
#: B4, verbatim
CHECK_KEYS = frozenset(
    {
        "what",
        "from_frame_t_ms",
        "to_frame_t_ms",
        "cut_short",
        "words_received",
        "words_estimated",
        "steps_received",
        "onsets",
        "max_frame_rms_dbfs",
        "stt",
    }
)
_REASONS = frozenset({"period", "gate"})


class DiagnosticPinError(AssertionError):
    """A ``mic.level``/``mic.unmute_check`` note that is not B1's/B4's shape: not the declared difference."""


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _check(data: dict[str, Any]) -> None:
    what = data["what"]
    want = LEVEL_KEYS if what == LEVEL else CHECK_KEYS
    keys = set(data)
    if keys != want:
        raise DiagnosticPinError(
            f"T-0195 {'B1' if what == LEVEL else 'B4'}: {what} keys differ; extra {sorted(keys - want)}, "
            f"missing {sorted(want - keys)}"
        )
    for key, value in data.items():
        if key == "what":
            continue
        if key == "reason":
            if value not in _REASONS:
                raise DiagnosticPinError(f"T-0195 B1: reason {value!r} is not 'period' or 'gate'")
        elif key == "stt":
            if value is not None and not (
                isinstance(value, dict)
                and all(isinstance(k, str) and _number(v) for k, v in value.items())
            ):
                raise DiagnosticPinError(
                    f"T-0195 B1/B6: stt is null or a flat mapping of numbers; got {value!r}"
                )
        elif not (value is None or isinstance(value, bool) or _number(value)):
            raise DiagnosticPinError(
                f"T-0195 B6: {what}.{key} = {value!r} is not a number, bool or null"
            )


def is_diagnostic_note(item: Any) -> bool:
    """``item`` is a group B diagnostic note (shape checked; a wrong shape raises)."""
    if not isinstance(item, dict) or item.get("kind") != "note":
        return False
    data = item.get("data")
    if not isinstance(data, dict) or data.get("what") not in DIAGNOSTIC_WHATS:
        return False
    _check(data)
    return True


def without_diagnostic_notes(tree: Any) -> Any:
    """``tree`` with every ``mic.level``/``mic.unmute_check`` note record removed from every list and tuple (shape
    checked first); a new tree, the input is not mutated."""

    def records(items: Any) -> list[Any]:
        out: list[Any] = []
        removed = 0
        for item in items:
            if is_diagnostic_note(item):
                removed += 1
                continue
            out.append(_recount(walk(item), removed))
        return out

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return records(node)
        if isinstance(node, tuple):
            return tuple(records(node))
        return node

    return walk(tree)


def _recount(item: Any, removed: int) -> Any:
    """The session log's closing ``log.stopped`` record counts every record written, so it counts the removed notes
    too: its ``records`` is reduced by the diagnostic notes removed before it in the same capture. The arithmetic
    consequence of the removal, never anything else: a ``records`` that would go negative raises."""
    if not removed or not isinstance(item, dict) or item.get("kind") != "log.stopped":
        return item
    data = item.get("data")
    if not isinstance(data, dict) or not _number(data.get("records")):
        return item
    if data["records"] < removed:
        raise DiagnosticPinError(
            f"T-0195 B: log.stopped counts {data['records']} records, fewer than {removed} removed"
        )
    return {**item, "data": {**data, "records": data["records"] - removed}}
