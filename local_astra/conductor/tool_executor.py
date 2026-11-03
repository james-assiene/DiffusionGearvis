"""The dry-run tool executor (T-0091 criterion 7; T-0080d settled default 1).

``DryRunToolExecutor`` describes what a v1 call would do and does none of it: no socket, no process, no file, no
input device (``tests/unit/test_tool_executor_dry_run.py``). ``google_search`` is never executed and names no URL
(H2). The 0-999 coordinates of ``click_at``/``type_at`` are mapped to the pixels of ``inv.frame``::

    x_px = round(x * (width - 1) / 999)      (and the same for y against the height)

computed in integers (round half up; with the odd divisor 999 a tie cannot occur). A click or a type without a frame
is refused with ``"no frame"``. The status is always ``dry_run`` or ``refused``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from local_astra.contracts.toolcall import ToolInvocation, ToolOutcome

__all__ = ["DryRunToolExecutor", "to_pixels"]

_COORD_MAX = 999


def to_pixels(v: int, size: int) -> int:
    """``round(v * (size - 1) / 999)`` for ``0 <= v <= 999`` and ``size >= 1``, exactly."""
    return (2 * v * (size - 1) + _COORD_MAX) // (2 * _COORD_MAX)


def _coord(args: Mapping[str, Any], key: str) -> int | None:
    v = args.get(key)
    if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= _COORD_MAX:
        return None
    return v


def _str(args: Mapping[str, Any], key: str) -> str | None:
    v = args.get(key)
    return v if isinstance(v, str) else None


class DryRunToolExecutor:
    """A ``ToolExecutor`` (``contracts/protocols.py``) in ``dry_run`` mode."""

    mode = "dry_run"

    async def execute(self, inv: ToolInvocation) -> ToolOutcome:
        return self._describe(inv)

    def _describe(self, inv: ToolInvocation) -> ToolOutcome:
        args = inv.arguments
        if inv.name in ("click_at", "type_at"):
            if inv.frame is None:
                return self._refused(inv, "no frame")
            x, y = _coord(args, "x"), _coord(args, "y")
            if x is None or y is None:
                return self._refused(inv, "bad coordinates")
            x_px, y_px = to_pixels(x, inv.frame.width), to_pixels(y, inv.frame.height)
            resolved: dict[str, Any] = {"x_px": x_px, "y_px": y_px}
            if inv.name == "click_at":
                return self._dry(inv, f"would click at ({x_px}, {y_px})", resolved)
            text = _str(args, "text")
            enter = args.get("press_enter")
            if text is None or not isinstance(enter, bool):
                return self._refused(inv, "bad arguments")
            action = f"would type {text!r} at ({x_px}, {y_px})" + (
                " and press Enter" if enter else ""
            )
            return self._dry(inv, action, resolved | {"text": text, "press_enter": enter})
        if inv.name == "google_search":
            query = _str(args, "query")
            if query is None:
                return self._refused(inv, "bad arguments")
            return self._dry(inv, "would search for: " + query, {"query": query})
        if inv.name == "scroll":
            direction = _str(args, "direction")
            if direction is None:
                return self._refused(inv, "bad arguments")
            return self._dry(inv, "would scroll " + direction, {"direction": direction})
        if inv.name == "press_key":
            combo = _str(args, "combo")
            if combo is None:
                return self._refused(inv, "bad arguments")
            return self._dry(inv, "would press " + combo, {"combo": combo})
        return self._refused(inv, "unknown tool")

    @staticmethod
    def _dry(inv: ToolInvocation, action: str, resolved: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(
            call_id=inv.call_id, status="dry_run", action=action, reason=None, resolved=resolved
        )

    @staticmethod
    def _refused(inv: ToolInvocation, reason: str) -> ToolOutcome:
        return ToolOutcome(
            call_id=inv.call_id,
            status="refused",
            action=f"refused {inv.name}: {reason}",
            reason=reason,
            resolved={},
        )
