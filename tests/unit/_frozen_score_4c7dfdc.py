"""The google_search-and-all ``exact`` of ``local_astra/toolcall/score.py`` at 4c7dfdc, frozen as an oracle (T-0086).

Not a test module. T-0086 criterion 4 ("tightening only") compares the new ``score.exact`` with this copy, and the
re-score rows use it as "the old rule". The bodies below are a verbatim copy of 4c7dfdc's ``_WS``, ``_norm_text``,
``_is_int``, ``_in_bbox``, ``_keywords_present``, ``_type_at`` and ``exact`` (blob unchanged since 3ee4040, T-0080a);
``tests/unit/test_toolcall_score_tightening.py::test_the_frozen_oracle_is_4c7dfdcs_exact`` proves it against
``git show 4c7dfdc:local_astra/toolcall/score.py`` (AST equality, function by function).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

FROZEN_COMMIT = "4c7dfdc"
FROZEN_NAMES = ("_norm_text", "_is_int", "_in_bbox", "_keywords_present", "_type_at", "exact")

_WS = re.compile(r"\s+")


def _norm_text(s: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", s)).strip().casefold()


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
        return _keywords_present(args, gt.get("keywords"))
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
