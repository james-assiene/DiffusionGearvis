"""T-0202 group A (A1, A2, A6): the optional tool files ``tools_v1_wait.json`` and ``tools_v1_overview_wait.json``
and the untouched v1/overview files (docs/tasks/T-0202.md; the T-0140 ruling R-p pattern, modelled on
``tests/unit/test_t0140_tools.py``).

**The surface these rows pin:**
- ``tools.TOOLS_V1_WAIT_PATH`` (exported in ``tools.__all__``) names ``local_astra/toolcall/tools_v1_wait.json``:
  v1's five declarations deep-equal and in v1's order, then ``wait`` (A1).
- ``tools.TOOLS_V1_OVERVIEW_WAIT_PATH`` (exported) names ``tools_v1_overview_wait.json``: ``tools_v1_overview.json``'s
  six declarations deep-equal and in order, then the same ``wait`` declaration (A2).
- ``wait`` is exactly ``{"name": "wait", "description": <H-T202-1's text>, "parameters": {"type": "object",
  "properties": {}, "additionalProperties": false}}``. The description is typed from H-T202-1 and also pinned by the
  sha256 of that typed text (computed by hand from the task's sentence, never read from an implementation).

The new files do not exist at BASE, so their paths are looked up inside each row (never at import time): a missing
constant or file is a named ``AssertionError``.

Green-at-BASE guards (no marker): ``test_v1_and_overview_files_unchanged`` and ``test_nothing_loads_wait_files_yet``
(A6). ``test_nothing_loads_wait_files_yet`` is scoped to group A: group B's B8 makes ``cli.py`` load the new sets, and
B's test-author narrows the row to that one loader then (stated in the A report, not guessed here).

BASE (group A): 3a755e2b. Regime: CPU, pure file reads, n = 1.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall import tools

REPO = Path(__file__).resolve().parents[2]
TOOLCALL = REPO / "local_astra" / "toolcall"
V1_PATH = TOOLCALL / "tools_v1.json"
OVERVIEW_PATH = TOOLCALL / "tools_v1_overview.json"
NATIVE_T4_PATH = TOOLCALL / "payloads" / "native_T4.json"

#: A6: full sha256 literals at BASE 3a755e2b (``sha256sum``); never derived at run time.
V1_SHA256 = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
OVERVIEW_SHA256 = "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc"
NATIVE_T4_SHA256 = "591d6c28be3352138136342ff0cd27bc6d18b34f60778fb8bbca8c9255d13601"

V1_NAMES = ("google_search", "click_at", "type_at", "scroll", "press_key")
OVERVIEW_NAMES = (*V1_NAMES, "page_overview")

#: H-T202-1's description, verbatim from the task.
WAIT_DESCRIPTION = "Wait about one second without touching the page, then take a new screenshot of the browser screen."
#: sha256 of ``WAIT_DESCRIPTION`` as typed above (UTF-8), computed by hand at BASE.
WAIT_DESCRIPTION_SHA256 = "1e5eea64e19b6166f030e51312c24440f648811f489efb8c6e026d12a9ff0e35"
WAIT_PARAMETERS = {"type": "object", "properties": {}, "additionalProperties": False}
WAIT_DECLARATION = {"name": "wait", "description": WAIT_DESCRIPTION, "parameters": WAIT_PARAMETERS}

#: A6 census: the names group A introduces for the two files; nothing in ``local_astra/`` but ``tools.py`` names them.
NEW_FILE_NAMES = (
    "tools_v1_wait",
    "tools_v1_overview_wait",
    "TOOLS_V1_WAIT_PATH",
    "TOOLS_V1_OVERVIEW_WAIT_PATH",
)


def _path_const(name: str, filename: str) -> Path:
    p = getattr(tools, name, None)
    assert p is not None, f"T-0202 A1/A2: tools.{name} does not exist yet"
    p = Path(p)
    assert p.resolve() == (TOOLCALL / filename).resolve(), (name, p)
    assert p.is_file(), f"T-0202 A1/A2: {p} does not exist yet"
    return p


def wait_path() -> Path:
    return _path_const("TOOLS_V1_WAIT_PATH", "tools_v1_wait.json")


def overview_wait_path() -> Path:
    return _path_const("TOOLS_V1_OVERVIEW_WAIT_PATH", "tools_v1_overview_wait.json")


def _raw(p: Path) -> list[dict[str, Any]]:
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(raw, list), type(raw)
    return raw


# ---------------------------------------------------------------------------------------------------- green at BASE


def test_v1_and_overview_files_unchanged() -> None:
    """A6 (negative; green at BASE and after): ``tools_v1.json``, ``tools_v1_overview.json`` and
    ``payloads/native_T4.json`` hash to their BASE literals, and ``tools.load()`` with no argument is still v1 at
    ``bbfc53cc…`` with v1's five names (never a wait set)."""
    assert hashlib.sha256(V1_PATH.read_bytes()).hexdigest() == V1_SHA256
    assert hashlib.sha256(OVERVIEW_PATH.read_bytes()).hexdigest() == OVERVIEW_SHA256
    assert hashlib.sha256(NATIVE_T4_PATH.read_bytes()).hexdigest() == NATIVE_T4_SHA256
    v1 = tools.load()
    assert v1.version == V1_SHA256, v1.version
    assert Path(v1.path).resolve() == V1_PATH.resolve(), v1.path
    assert v1.names == V1_NAMES, v1.names
    assert "wait" not in v1.names


def test_nothing_loads_wait_files_yet() -> None:
    """A6 (negative), narrowed by T-0202 B's test-author (B8 makes ``cli.load_tool_kit`` load the wait sets): a grep
    census of every text file under ``local_astra/`` finds none of the new file names or path constants outside
    ``toolcall/tools.py`` (where A1/A2 define them), the two new JSON files themselves and ``cli.py`` (the one
    loader B8 allows). Power: the same scan sees ``TOOLS_V1_OVERVIEW_PATH`` in ``cli.py`` (T-0140's loader), so
    another loader of the new names would be seen."""
    root = REPO / "local_astra"
    allowed = {
        (TOOLCALL / "tools.py").resolve(),
        (TOOLCALL / "tools_v1_wait.json").resolve(),
        (TOOLCALL / "tools_v1_overview_wait.json").resolve(),
        (REPO / "local_astra" / "cli.py").resolve(),  # T-0202 B8: the only reader
    }
    pattern = re.compile("|".join(re.escape(n) for n in NEW_FILE_NAMES))
    power = re.compile(r"\bTOOLS_V1_OVERVIEW_PATH\b")
    hits: list[str] = []
    power_hits: list[str] = []
    scanned = 0
    for f in sorted(root.rglob("*")):
        if (
            not f.is_file()
            or "__pycache__" in f.parts
            or f.suffix in {".pyc", ".png", ".jpg", ".wav", ".bin"}
        ):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        rel = f.relative_to(REPO).as_posix()
        if power.search(text) and f.resolve() != (TOOLCALL / "tools.py").resolve():
            power_hits.append(rel)
        if f.resolve() in allowed:
            continue
        for m in pattern.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            hits.append(f"{rel}:{line}: {m.group(0)}")
    assert scanned > 50, scanned
    assert "local_astra/cli.py" in power_hits, power_hits
    assert not hits, (
        "T-0202 A6/B8: something in local_astra/ other than cli.py loads the wait tool files:\n"
        + "\n".join(hits)
    )


# ---------------------------------------------------------------------------------------------------- needs A1/A2


def test_paths_exported() -> None:
    """A1/A2: both constants name their files beside ``tools_v1.json`` and are exported in ``tools.__all__``; v1's
    and T-0140's constants stay exported."""
    assert wait_path().resolve() == V1_PATH.with_name("tools_v1_wait.json").resolve()
    assert (
        overview_wait_path().resolve() == V1_PATH.with_name("tools_v1_overview_wait.json").resolve()
    )
    for name in (
        "TOOLS_V1_WAIT_PATH",
        "TOOLS_V1_OVERVIEW_WAIT_PATH",
        "TOOLS_V1_PATH",
        "TOOLS_V1_OVERVIEW_PATH",
    ):
        assert name in tools.__all__, (name, tools.__all__)


def test_wait_file_extends_v1() -> None:
    """A1: ``tools.load(TOOLS_V1_WAIT_PATH)`` validates; its first five declarations deep-equal v1's (read from the
    raw v1 file, not through the loader), in v1's order; the sixth is exactly the ``wait`` declaration; nothing
    follows; the description hashes to H-T202-1's sha; the set is versioned by its own bytes."""
    p = wait_path()
    ws = tools.load(p)
    v1_raw = _raw(V1_PATH)
    assert len(ws) == 6, ws.names
    assert ws.names == (*V1_NAMES, "wait"), ws.names
    for i, (a, b) in enumerate(zip(ws.declarations[:5], v1_raw, strict=True)):
        assert dict(a) == b, f"declaration {i} ({b['name']}) is not v1's"
    sixth = dict(ws.declarations[5])
    assert sixth == WAIT_DECLARATION, sixth
    assert (
        hashlib.sha256(str(sixth["description"]).encode("utf-8")).hexdigest()
        == WAIT_DESCRIPTION_SHA256
    )
    assert ws.parameters("wait") == WAIT_PARAMETERS
    assert ws.version == hashlib.sha256(p.read_bytes()).hexdigest()
    assert ws.version not in (V1_SHA256, OVERVIEW_SHA256)
    # The raw file agrees with the loader (no declaration added or reordered by the loader).
    assert _raw(p) == [*v1_raw, WAIT_DECLARATION]


def test_overview_wait_file_extends_overview() -> None:
    """A2: ``tools.load(TOOLS_V1_OVERVIEW_WAIT_PATH)`` validates; its first six declarations deep-equal
    ``tools_v1_overview.json``'s (raw file), in order; the seventh is the same ``wait`` declaration as A1's;
    nothing follows; it is versioned by its own bytes, distinct from the three other sets."""
    p = overview_wait_path()
    ows = tools.load(p)
    ov_raw = _raw(OVERVIEW_PATH)
    assert [d["name"] for d in ov_raw] == list(OVERVIEW_NAMES)
    assert len(ows) == 7, ows.names
    assert ows.names == (*OVERVIEW_NAMES, "wait"), ows.names
    for i, (a, b) in enumerate(zip(ows.declarations[:6], ov_raw, strict=True)):
        assert dict(a) == b, f"declaration {i} ({b['name']}) is not the overview file's"
    assert dict(ows.declarations[6]) == WAIT_DECLARATION, dict(ows.declarations[6])
    assert _raw(p) == [*ov_raw, WAIT_DECLARATION]
    assert ows.parameters("wait") == WAIT_PARAMETERS
    assert ows.version == hashlib.sha256(p.read_bytes()).hexdigest()
    ws_version = tools.load(wait_path()).version
    assert len({ows.version, ws_version, V1_SHA256, OVERVIEW_SHA256}) == 4


def test_wait_declaration_identical_in_both_sets() -> None:
    """A1/A2 (adversarial; the mutant gives the overview+wait set a differently worded ``wait``): the two ``wait``
    declarations are equal to each other and to H-T202-1's, and their schema admits only the empty object."""
    import jsonschema

    a = dict(tools.load(wait_path()).get("wait") or {})
    b = dict(tools.load(overview_wait_path()).get("wait") or {})
    assert a == b == WAIT_DECLARATION, (a, b)
    jsonschema.validate({}, a["parameters"])
    for bad in ({"seconds": 2}, {"x": 1}, {"": None}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, a["parameters"])
