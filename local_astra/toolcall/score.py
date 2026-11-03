"""The call scorer (T-0080a criterion 3): parse a call, validate it, compare it with ground truth.

The same code scores the eval (T-0080b) and will check calls in the conductor (T-0080d), so it is pure
Python plus ``jsonschema`` (criterion 6: no ``local_astra.decoding``, torch, numpy or outlines).

- ``parse_call(text)`` -> ``{"name": str, "arguments": dict}``, or a ``ParseFailure`` whose ``reason``
  names the failure class (each class has its own reason).
- ``schema_valid(call, tools)`` -> the arguments validate against the named tool's declaration.
- ``exact(call, gt)`` -> per tool, against a ground-truth case ``{"name", ...fields}``:
  ``google_search`` every keyword in the query (NFC, case-folded, whitespace runs collapsed);
  ``click_at`` the point inside ``bbox = [x0, y0, x1, y1]`` (0-999, inclusive); ``type_at`` the bbox,
  ``text`` equal after strip and ``press_enter`` equal; ``scroll`` the direction; ``press_key`` the
  ``combo`` string (D1: one key or shortcut from a fixed list). A call for a different tool is never exact. Malformed arguments score ``False``, never raise.
- ``query_problem(query)`` (T-0086) -> ``None`` for a plausible search query, else the first failing reason. A
  ``google_search`` call is exact only when its keywords are present **and** ``query_problem`` is ``None``. On the
  normalised query (NFC, whitespace runs collapsed, stripped, case-folded), in order: ``"empty"``; ``"too long"``
  (more than 32 space-separated words); ``"control character"`` (Unicode category Cc or Cf, checked after
  normalisation, so tabs and newlines are whitespace); ``"markup"`` (any of ``{ } [ ] < > \\ |`` or a backtick);
  ``"tool syntax"`` (``call:`` or an underscore v1 tool name: google_search, click_at, type_at, press_key -- not
  ``scroll``, an ordinary word); ``"narration"`` (a first-person agent clause -- i will, i'll, i should, i must,
  i need to, i have to, i am going to, i'm going to, let me -- then search, google, look up, use, call or query and
  a word end, anywhere, not anchored on the left; U+2019 counts as an apostrophe).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jsonschema

from local_astra.toolcall.tools import ToolSet

__all__ = ["ParseFailure", "exact", "parse_call", "query_problem", "schema_valid"]


@dataclass(frozen=True)
class ParseFailure:
    reason: str


_WS = re.compile(r"\s+")


def parse_call(text: str) -> dict[str, Any] | ParseFailure:
    """Parse ``{"name": ..., "arguments": {...}}`` (surrounding JSON whitespace allowed)."""
    if not text.strip():
        return ParseFailure("empty")
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError):
        return ParseFailure("not JSON")
    if not isinstance(obj, dict):
        return ParseFailure("not a JSON object")
    if "name" not in obj:
        return ParseFailure("no name")
    if not isinstance(obj["name"], str):
        return ParseFailure("name is not a string")
    if "arguments" not in obj:
        return ParseFailure("no arguments")
    if not isinstance(obj["arguments"], dict):
        return ParseFailure("arguments is not an object")
    return {"name": obj["name"], "arguments": obj["arguments"]}


def schema_valid(call: Mapping[str, Any], tools: ToolSet) -> bool:
    """``call["arguments"]`` validates against the declaration of ``call["name"]`` in ``tools``."""
    name = call.get("name")
    args = call.get("arguments")
    decl = tools.get(name) if isinstance(name, str) else None
    if decl is None or not isinstance(args, dict):
        return False
    schema = decl["parameters"]
    validator = jsonschema.validators.validator_for(schema)(schema)
    return bool(validator.is_valid(args))


def _norm_text(s: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", s)).strip().casefold()


_MAX_QUERY_WORDS = 32
_MARKUP = frozenset("{}[]<>\\|`")
_TOOL_SYNTAX = ("call:", "google_search", "click_at", "type_at", "press_key")
_NARRATION = re.compile(
    r"(?:i will|i'll|i should|i must|i need to|i have to|i am going to|i'm going to|let me)"
    r" (?:search|google|look up|use|call|query)\b"
)


def query_problem(query: str) -> str | None:
    """Why ``query`` is not a plausible search query, or ``None`` (T-0086; the full rule is in the module docstring).

    On the normalised query, the first failing reason of: "empty", "too long" (more than 32 words), "control
    character", "markup", "tool syntax", "narration". Pure and total.
    """
    n = _norm_text(query)
    if not n:
        return "empty"
    if len(n.split(" ")) > _MAX_QUERY_WORDS:
        return "too long"
    if any(unicodedata.category(ch) in ("Cc", "Cf") for ch in n):
        return "control character"
    if any(ch in _MARKUP for ch in n):
        return "markup"
    if any(t in n for t in _TOOL_SYNTAX):
        return "tool syntax"
    if _NARRATION.search(n.replace("’", "'")):
        return "narration"
    return None


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _in_bbox(args: Mapping[str, Any], bbox: Any) -> bool:
    x, y = args.get("x"), args.get("y")
    if not (_is_int(x) and _is_int(y)):
        return False
    if not (isinstance(bbox, Sequence) and len(bbox) == 4):
        return False
    x0, y0, x1, y1 = bbox
    return bool(x0 <= x <= x1 and y0 <= y <= y1)


def _keywords_present(args: Mapping[str, Any], keywords: Any) -> bool:
    query = args.get("query")
    if not isinstance(query, str) or not isinstance(keywords, Sequence):
        return False
    q = _norm_text(query)
    return all(isinstance(k, str) and _norm_text(k) in q for k in keywords)


def _type_at(args: Mapping[str, Any], gt: Mapping[str, Any]) -> bool:
    text, enter = args.get("text"), args.get("press_enter")
    return (
        _in_bbox(args, gt.get("bbox"))
        and isinstance(text, str)
        and text.strip() == str(gt.get("text", "")).strip()
        and isinstance(enter, bool)
        and enter is gt.get("press_enter")
    )


def exact(call: Mapping[str, Any], gt: Mapping[str, Any]) -> bool:
    """Whether ``call`` does what the ground-truth case asks (rules in the module docstring)."""
    name = call.get("name")
    args = call.get("arguments")
    if name != gt.get("name") or not isinstance(args, Mapping):
        return False
    if name == "google_search":
        query = args.get("query")
        return (
            _keywords_present(args, gt.get("keywords"))
            and isinstance(query, str)
            and query_problem(query) is None
        )
    if name == "click_at":
        return _in_bbox(args, gt.get("bbox"))
    if name == "type_at":
        return _type_at(args, gt)
    if name == "scroll":
        direction = args.get("direction")
        return isinstance(direction, str) and direction == gt.get("direction")
    if name == "press_key":
        combo = args.get("combo")
        return isinstance(combo, str) and combo == gt.get("combo")
    return False
