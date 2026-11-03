"""T-0162 group C: the one declared difference the older tool-call pins may discount (docs/tasks/T-0162.md, Group C
amendment 2026-10-27, "Pre-authorised pin moves").

Group C adds exactly two keys to ``astra.tool.call`` -- ``step`` and ``result`` (C-1) -- and changes nothing else that
a session publishes, logs or sends (C-4). The goldens and differentials written before C compare the HUD messages of
a session with an older commit's; this module lets them keep that meaning:

- :func:`without_tool_result_fields` returns a capture with ``step`` and ``result`` removed from every dict whose
  ``type`` is ``"astra.tool.call"`` -- and from no other dict, so a ``step`` on ``astra.canvas.step`` or a ``result``
  key anywhere else is compared as before. Before it removes them it checks they are C-1's shapes (``step`` an int
  >= 1 or ``None``; ``result`` ``None`` or a dict of exactly ``outcome``/``hosts``/``hosts_more`` and optionally
  ``results``, the outcome one of the six, at most 3 hosts), and that a ``result`` sits only on a final status; anything
  else raises :class:`ToolResultPinError`, so the move admits these two keys and nothing more.
- :func:`observed` is T-0155's ``_pre_t0155_persona.observed`` (the persona sentence) followed by the strip, for the
  rows that already normalise their observed side that way. The base side is never normalised.
- :data:`TOOL_CALL_ADDED` is the two field names, for the codec's field-set pin.

Every other difference still fails the row (the mutation checks are in the test-author's report).

Regime: CPU, pure, no I/O.
"""

from __future__ import annotations

from typing import Any

from tests.unit import _pre_t0155_persona as _persona
from tests.unit._t0177_pins import observed as _t0177_observed

TOOL_CALL_TYPE = "astra.tool.call"
#: C-1: the two fields group C adds to ``AstraToolCall``
TOOL_CALL_ADDED = frozenset({"step", "result"})
_OUTCOMES = frozenset({"executed", "dry_run", "refused", "failed", "cancelled", "timeout"})
_FINAL = frozenset({"dry_run", "executed", "refused", "cancelled"})
_RESULT_KEYS = frozenset({"outcome", "hosts", "hosts_more"})
_RESULT_OPTIONAL = frozenset({"results"})


class ToolResultPinError(AssertionError):
    """A ``step`` or ``result`` on ``astra.tool.call`` that is not C-1's shape: not the declared difference."""


def _check(message: dict[str, Any]) -> None:
    step = message.get("step")
    if step is not None and not (
        isinstance(step, int) and not isinstance(step, bool) and step >= 1
    ):
        raise ToolResultPinError(
            f"T-0162 C-1: step is an int >= 1 or null; got {step!r} in {message!r}"
        )
    result = message.get("result")
    if result is None:
        return
    if not isinstance(result, dict):
        raise ToolResultPinError(f"T-0162 C-1: result is an object; got {result!r}")
    keys = set(result)
    if not (_RESULT_KEYS <= keys <= _RESULT_KEYS | _RESULT_OPTIONAL):
        raise ToolResultPinError(f"T-0162 C-1: result keys {sorted(keys)} are not C-1's")
    if result["outcome"] not in _OUTCOMES:
        raise ToolResultPinError(f"T-0162 C-1: outcome {result['outcome']!r} is not one of the six")
    hosts = result["hosts"]
    if not isinstance(hosts, list) or len(hosts) > 3 or not all(isinstance(h, str) for h in hosts):
        raise ToolResultPinError(f"T-0162 C-1: hosts {hosts!r} is not a list of at most 3 strings")
    if message.get("status") not in _FINAL:
        raise ToolResultPinError(
            f"T-0162 C-1: a result on the non-final status {message.get('status')!r}"
        )


def without_tool_result_fields(tree: Any) -> Any:
    """``tree`` with ``step``/``result`` removed from every ``astra.tool.call`` dict (checked first); a new tree, the
    input is not mutated."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if node.get("type") == TOOL_CALL_TYPE:
                _check(node)
                return {k: walk(v) for k, v in node.items() if k not in TOOL_CALL_ADDED}
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, tuple):
            return tuple(walk(v) for v in node)
        return node

    return walk(tree)


def observed(tree: Any, *, persona: str | None = None) -> Any:
    """T-0155 O-8's ``observed`` (the persona sentence), then T-0162 C's two tool-call keys, then T-0177 A-P1's
    ``extras.vision_source == "browser"`` (``tests.unit._t0177_pins``; ``"shared"`` is never stripped), then T-0177
    B-P1's ``astra.vision.frame`` messages and their ``wire.out`` records (shape checked)."""
    return _t0177_observed(without_tool_result_fields(_persona.observed(tree, persona=persona)))
