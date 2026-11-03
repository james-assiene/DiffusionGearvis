"""T-0140 A1 (group A1): the optional tool file ``tools_v1_overview.json`` and the untouched v1 file
(docs/tasks/T-0140.md, ruling R-p).

**The surface these rows pin:** ``tools.TOOLS_V1_OVERVIEW_PATH`` (exported in ``tools.__all__``) names
``local_astra/toolcall/tools_v1_overview.json``, which validates through ``tools.load(path)``: v1's five
declarations deep-equal and in v1's order, then ``page_overview`` with an empty strict schema and A1's exact
description. ``tools.load()`` with no argument is still v1 (sha256 ``bbfc53cc…``).

The overview file does not exist at BASE, so its path is looked up inside each row (never at import time): a missing
constant or file is a named ``AssertionError``.

``test_overview_sha_pinned``: the task has the test-author write the file's sha literal "at activation", but at
activation the file does not exist yet and A1 fixes its content, not its bytes (whitespace, key order). The literal
was pinned in two steps (O-A1-1): ``None`` until the implementer's file existed, then its ``sha256sum`` at 5f6c856.

BASE (group A1): 0bc60ff (the task header's ca730f5 is stale). Regime: CPU, pure file reads, n = 1.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from local_astra.toolcall import tools

#: ``tools_v1.json`` sha256 (A1, A12); the full literal, never derived.
V1_SHA256 = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
V1_PATH = Path(__file__).resolve().parents[2] / "local_astra" / "toolcall" / "tools_v1.json"
#: A1's description, verbatim from the task.
DESCRIPTION = (
    "Shows one picture of the whole web page, scaled down to fit the screen, so you can see what lies beyond the "
    "visible part. The picture is for looking only: click_at and type_at coordinates always refer to the browser "
    "screen."
)
PARAMETERS = {"type": "object", "properties": {}, "additionalProperties": False}
V1_NAMES = ("google_search", "click_at", "type_at", "scroll", "press_key")
#: The overview file's sha256, pinned by the test-author once the file exists (O-A1-1; verified with ``sha256sum``
#: of the file at 5f6c856).
OVERVIEW_SHA256: str | None = "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc"


def overview_path() -> Path:
    p = getattr(tools, "TOOLS_V1_OVERVIEW_PATH", None)
    assert p is not None, "T-0140 A1: tools.TOOLS_V1_OVERVIEW_PATH does not exist yet"
    p = Path(p)
    assert p.is_file(), f"T-0140 A1: {p} does not exist yet"
    return p


def overview_set() -> Any:
    return tools.load(overview_path())


# ---------------------------------------------------------------------------------------------------- green at BASE


def test_v1_untouched() -> None:
    """A1/A12 (reviewer mutation 6, v1's sha moved): ``tools_v1.json``'s bytes hash to the full literal, and the
    default ``tools.load()`` is that file at that version (green at BASE and after)."""
    assert hashlib.sha256(V1_PATH.read_bytes()).hexdigest() == V1_SHA256
    v1 = tools.load()
    assert v1.version == V1_SHA256 and Path(v1.path).resolve() == V1_PATH
    assert v1.names == V1_NAMES, v1.names


# ---------------------------------------------------------------------------------------------------- needs A1


def test_overview_path_constant() -> None:
    """A1: the constant names ``tools_v1_overview.json`` beside ``tools_v1.json`` and is exported."""
    assert overview_path().resolve() == V1_PATH.with_name("tools_v1_overview.json")
    assert "TOOLS_V1_OVERVIEW_PATH" in tools.__all__, tools.__all__


def test_overview_file_extends_v1() -> None:
    """A1: the file validates through ``tools.load(path)``; its first five declarations deep-equal v1's, in order;
    the sixth is exactly ``{"name": "page_overview", "description": <A1's text>, "parameters": {"type": "object",
    "properties": {}, "additionalProperties": false}}``; nothing follows it."""
    ov = overview_set()
    v1 = tools.load()
    assert len(ov) == 6, ov.names
    assert ov.names == (*V1_NAMES, "page_overview"), ov.names
    for i, (a, b) in enumerate(zip(ov.declarations[:5], v1.declarations, strict=True)):
        assert dict(a) == dict(b), f"declaration {i} ({b['name']}) is not v1's"
    sixth = dict(ov.declarations[5])
    assert sixth == {
        "name": "page_overview",
        "description": DESCRIPTION,
        "parameters": PARAMETERS,
    }, sixth
    assert ov.parameters("page_overview") == PARAMETERS
    assert ov.version != V1_SHA256


def test_overview_sha_pinned() -> None:
    """A1 (and group B's arm B): the file's sha256 is its pinned literal; ``tools.load`` versions it by those bytes.
    The literal is written by the test-author once the file exists (module docstring)."""
    p = overview_path()
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    assert OVERVIEW_SHA256 is not None, (
        f"T-0140 A1: the overview file's sha256 is not pinned yet; the test-author pins {got} (never the "
        "implementer)"
    )
    assert got == OVERVIEW_SHA256, got
    assert tools.load(p).version == OVERVIEW_SHA256


def test_description_states_facts() -> None:
    """A1 (reviewer aid): the description is at most 300 characters and gives no "when to use" instruction: none of
    "when", "should", "always use", "if you" (case-insensitive, whole words)."""
    d = str(overview_set().get("page_overview")["description"])  # type: ignore[index]
    assert len(d) <= 300, len(d)
    for phrase in ("when", "should", "always use", "if you"):
        assert not re.search(rf"\b{phrase}\b", d, re.IGNORECASE), (phrase, d)
