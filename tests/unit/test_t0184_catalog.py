"""T-0184 group A: the v2 A2UI catalog ``gearvis:cards/v2`` (docs/tasks/T-0184.md A1-A5).

**The interface these rows pin** (listed in the test-author's report for the orchestrator to accept):

- The catalog file is ``local_astra/ui/schemas/cards_v2/catalog.json``, read as package data of
  ``local_astra.ui.schemas.cards_v2`` (A1 names the path; the owned list makes that directory a package).
- ``local_astra.ui.a2ui_schema.CARDS_V2_CATALOG_ID == "gearvis:cards/v2"``, re-exported by ``local_astra.ui``.
- "A v2 loader": the task names no function. The row accepts any public, argument-free callable of ``a2ui_schema``
  whose name contains ``v2`` (case-insensitive) and returns a mapping JSON-equal to the catalog file. Every other row
  reads the file itself and builds ``A2UIValidator(catalog=<file>)``, so no row depends on the loader's name.

**Readings (reported):**

- "Each v2 kind has ``unevaluatedProperties: false``" is read at the top level of ``components[<kind>]``, as the basic
  kinds have it. The nested objects (tile, column, row, bar, step, toned cell) are also pinned closed: the A reviewer
  found that opening one survived (F1), and the coordinator ruled them closed (2026-10-28).
- "All strings are the model's own. There are no URLs, path bindings or function calls" is read as: every string role
  of a v2 kind is a JSON string, and a ``{"path": …}`` binding or ``{"call": …}`` in that role is invalid.
- A toned Comparison cell requires both ``text`` and ``tone`` (A2 writes ``{text, tone}``; the reviewer's F2 found the
  ``required`` unpinned, and the coordinator ruled both required, 2026-10-28). The type of ``Stat.value`` and of a tile's ``value`` is not pinned beyond "a string validates".
- "The only ``maxItems`` is on ``Actions``" is tested by behaviour: long lists of bars, tiles, steps, columns and rows
  validate.
- Whether a v2 kind may carry ``accessibility`` or ``weight`` (the basic kinds' common props) is not pinned; the extra
  prop rows use a name in no schema (``colour``).
- A5 "the same verdicts on the T-0164 census": every canvas of ``genui_census_20261027.jsonl`` through
  ``A2UIValidator().validate_messages``, with ``final=True`` and ``final=False``, pinned as literals computed at BASE
  (aeac453f) plus a sha256 of the full detail (schema errors, problems, warnings, reachable ids). That is the one
  place a value comes from current behaviour; A5 asks for exactly "the same as today".

Oracles: the vendored basic catalog (``schemas/a2ui_v0_9_1/catalog.json``, sha256 in the fixture MANIFEST); A2's table,
retyped here as literals; hand-written components and the hand-written exemplars in
``tests/fixtures/a2ui/cards_v2_exemplars.jsonl`` (MANIFEST row).

Activated for ``/implement T-0184`` group A (pending at 9ae763da): until the catalog file exists, the loader turns its
absence into a named ``AssertionError``. The A5 rows are green on main by design.

Regime: CPU only, no service, no socket; n = 1, no seed.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
from collections.abc import Callable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from local_astra.ui import a2ui_schema
from local_astra.ui.a2ui_schema import A2UIValidator
from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "a2ui"
EXEMPLARS = FIXTURES / "cards_v2_exemplars.jsonl"
CENSUS = FIXTURES / "genui_census_20261027.jsonl"
BASIC_CATALOG_FILE = REPO / "local_astra" / "ui" / "schemas" / "a2ui_v0_9_1" / "catalog.json"
#: tests/fixtures/a2ui/MANIFEST.md, "SHA-256 at vendoring time".
BASIC_CATALOG_SHA256 = "8cc94d0a482e67048f9fc989964ca5da56fe42f531d919315a508989fb22e13e"
COMMON_TYPES_PREFIX = "https://a2ui.org/specification/v0_9/common_types.json#"

V2_ID = "gearvis:cards/v2"
V2_KINDS = ("Stat", "StatTiles", "Comparison", "Bars", "Steps", "Source", "Actions")
#: The basic catalog's eighteen kinds (vendored a2ui-agent-sdk 0.6.0), retyped.
BASIC_KINDS = (
    "Text",
    "Image",
    "Icon",
    "Video",
    "AudioPlayer",
    "Row",
    "Column",
    "List",
    "Card",
    "Tabs",
    "Modal",
    "Divider",
    "Button",
    "TextField",
    "CheckBox",
    "ChoicePicker",
    "Slider",
    "DateTimeInput",
)
BASIC_IDS = frozenset(
    {
        "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json",
        "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json",
    }
)
ICONS = ("sun", "partlyCloudy", "cloud", "rain", "storm", "snow", "fog", "wind", "night")


# --- helpers ----------------------------------------------------------------------------------------------------


def _v2_catalog() -> dict[str, Any]:
    try:
        ref = resources.files("local_astra.ui.schemas.cards_v2").joinpath("catalog.json")
    except ModuleNotFoundError as e:
        raise AssertionError(
            f"A1: package local_astra.ui.schemas.cards_v2 is missing ({e})"
        ) from None
    assert ref.is_file(), "A1: local_astra/ui/schemas/cards_v2/catalog.json is missing"
    data = json.loads(ref.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "A1: the catalog is not a JSON object"
    return data


def _v2() -> A2UIValidator:
    return A2UIValidator(catalog=_v2_catalog())


def _basic_catalog() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(BASIC_CATALOG_FILE.read_text(encoding="utf-8"))
    return data


def _exemplars() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in EXEMPLARS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _components(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        c for m in messages if "updateComponents" in m for c in m["updateComponents"]["components"]
    ]


def _surface(component: Mapping[str, Any], sid: str = "v2-adv") -> list[dict[str, Any]]:
    root = {**component, "id": "root"}
    return [
        {"version": "v0.9.1", "createSurface": {"surfaceId": sid, "catalogId": V2_ID}},
        {"version": "v0.9.1", "updateComponents": {"surfaceId": sid, "components": [root]}},
    ]


def _refs(node: Any) -> list[str]:
    if isinstance(node, Mapping):
        out = [node["$ref"]] if isinstance(node.get("$ref"), str) else []
        for v in node.values():
            out.extend(_refs(v))
        return out
    if isinstance(node, list):
        return [r for v in node for r in _refs(v)]
    return []


def _resolves(document: Any, ref: str) -> bool:
    """Does a local ``$ref`` point at something in ``document``? (RFC 6901, by hand.)"""
    node = document
    for token in ref.lstrip("#").split("/")[1:]:
        key = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, Mapping) or key not in node:
            return False
        node = node[key]
    return True


#: One valid component per v2 kind, hand-written from A2's table, every optional prop present.
BASE: dict[str, dict[str, Any]] = {
    "Stat": {
        "id": "c",
        "component": "Stat",
        "label": "PARIS · NOW",
        "value": "[17°]",
        "detail": "Partly cloudy",
        "icon": "partlyCloudy",
    },
    "StatTiles": {
        "id": "c",
        "component": "StatTiles",
        "tiles": [
            {
                "title": "Friday",
                "value": "[19°]",
                "secondary": "[11°]",
                "percent": 20,
                "highlight": True,
            }
        ],
    },
    "Comparison": {
        "id": "c",
        "component": "Comparison",
        "columns": [
            {"title": "Deluxe hardcover", "badge": "Recommended"},
            {"title": "Regular volumes"},
        ],
        "highlight": 0,
        "rows": [{"label": "Best for", "cells": ["Shelves", {"text": "Gifts", "tone": "ok"}]}],
    },
    "Bars": {
        "id": "c",
        "component": "Bars",
        "title": "Highest mountains",
        "unit": "m",
        "bars": [{"label": "Everest", "value": 8849}, {"label": "Mont Blanc", "value": 4806}],
    },
    "Steps": {
        "id": "c",
        "component": "Steps",
        "steps": [{"title": "Whisk", "detail": "Three eggs."}, {"title": "Fold"}],
    },
    "Source": {"id": "c", "component": "Source", "text": "en.wikipedia.org"},
    "Actions": {"id": "c", "component": "Actions", "labels": ["Yes", "No"]},
}


def _assert_base_valid(v: A2UIValidator, kind: str) -> dict[str, Any]:
    base = copy.deepcopy(BASE[kind])
    assert v.component_errors(base) == [], f"A2 positive control: the valid {kind} must validate"
    return base


def _assert_rejected(v: A2UIValidator, mutant: Mapping[str, Any], why: str) -> None:
    errs = v.component_errors(mutant)
    assert errs, f"{why}: the v2 validator accepted {json.dumps(mutant, ensure_ascii=False)}"
    unknown = [e for e in errs if "unknown component kind" in e.message]
    assert not unknown, (
        f"{why}: rejected only as an unknown kind ({unknown}); the kind must be in the catalog"
    )
    report = v.validate_messages(_surface(mutant), final=True)
    assert not report.ok, f"{why}: a surface holding the mutant validated"


# --- A1: the catalog file ---------------------------------------------------------------------------------------


def test_v2_catalog_file_names_its_id_and_is_self_contained() -> None:
    """A1: catalogId ``gearvis:cards/v2``; every ``$ref`` is local (and resolves inside the file) or points into
    ``common_types.json``; no other document is referenced (a cross-file ref silently degrades salvage and is an
    egress path, fixer_schema docstring)."""
    cat = _v2_catalog()
    assert cat.get("catalogId") == V2_ID
    refs = _refs(cat)
    assert refs, "the catalog holds no $ref at all; the basic copies alone carry dozens"
    foreign = sorted(
        {r for r in refs if not r.startswith("#") and not r.startswith(COMMON_TYPES_PREFIX)}
    )
    assert foreign == [], f"A1: refs into another document: {foreign}"
    dangling = sorted({r for r in refs if r.startswith("#") and not _resolves(cat, r)})
    assert dangling == [], f"A1: local refs that resolve to nothing: {dangling}"


def test_v2_any_component_is_one_of_basic_plus_seven() -> None:
    """A1: ``$defs/anyComponent`` is a ``oneOf`` over every basic component plus the seven v2 kinds, and
    ``components`` holds exactly those 25 kinds."""
    cat = _v2_catalog()
    one_of = cat["$defs"]["anyComponent"]["oneOf"]
    got = sorted(entry["$ref"] for entry in one_of)
    want = sorted(f"#/components/{k}" for k in (*BASIC_KINDS, *V2_KINDS))
    assert got == want
    assert sorted(cat["components"]) == sorted((*BASIC_KINDS, *V2_KINDS))


@pytest.mark.parametrize("kind", V2_KINDS)
def test_v2_kind_is_closed(kind: str) -> None:
    """A1: each v2 kind has ``unevaluatedProperties: false`` (top level of its schema, as the basic kinds)."""
    schema = _v2_catalog()["components"][kind]
    assert schema.get("unevaluatedProperties") is False, (
        f"{kind}: unevaluatedProperties must be false"
    )


def test_basic_copies_equal_the_vendored_basic_catalog() -> None:
    """A1 drift row: every basic definition copied into the v2 file is JSON-equal to the vendored basic catalog:
    each of the eighteen ``components``, every ``$defs`` entry except ``anyComponent`` (``CatalogComponentCommon``,
    ``theme`` (the message schema's ``catalog.json#/$defs/theme``), ``anyFunction``), and ``functions``. The vendored
    file itself is checked against its vendoring sha256, so the oracle cannot drift with the copy."""
    raw = BASIC_CATALOG_FILE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASIC_CATALOG_SHA256, (
        "the vendored basic catalog changed"
    )
    basic = json.loads(raw)
    cat = _v2_catalog()
    for kind in BASIC_KINDS:
        assert cat["components"].get(kind) == basic["components"][kind], (
            f"{kind} drifted from the basic catalog"
        )
    for name, definition in basic["$defs"].items():
        if name == "anyComponent":
            continue
        assert cat["$defs"].get(name) == definition, f"$defs/{name} drifted from the basic catalog"
    assert cat.get("functions") == basic["functions"], "functions drifted from the basic catalog"
    basic_entries = [
        e for e in cat["$defs"]["anyComponent"]["oneOf"] if e["$ref"].split("/")[-1] in BASIC_KINDS
    ]
    assert basic_entries == basic["$defs"]["anyComponent"]["oneOf"], (
        "the basic anyComponent entries must be copied verbatim, in their order"
    )


def test_fixer_schema_derives_from_v2_catalog() -> None:
    """A1: ``fixer_schema.components_array_schema`` derives a self-contained salvage schema from the v2 catalog: no
    ``unevaluatedProperties``, every ``$ref`` local and resolving, the seven kinds reachable from ``anyComponent``;
    and json_repair's salvage mode, guided by it, returns every exemplar's components unchanged with an empty log
    (a lost allOf branch or a dangling ref would strip or drop valid props)."""
    import json_repair

    from local_astra.ui.fixer_schema import components_array_schema

    v = _v2()
    schema = components_array_schema(v.catalog, v.common_types)
    text = json.dumps(schema)
    assert "unevaluatedProperties" not in text
    refs = sorted(set(re.findall(r'"\$ref": "([^"]+)"', text)))
    assert [r for r in refs if not r.startswith("#")] == []
    assert [r for r in refs if not _resolves(schema, r)] == []
    reachable = {e["$ref"] for e in schema["$defs"]["anyComponent"]["oneOf"]}
    assert {f"#/components/{k}" for k in V2_KINDS} <= reachable
    for row in _exemplars():
        comps = _components(row["messages"])
        salvaged, log = json_repair.loads(
            json.dumps(comps), schema=schema, schema_repair_mode="salvage", logging=True
        )
        assert salvaged == comps, f"{row['key']}: salvage changed a valid surface"
        assert log == [], f"{row['key']}: salvage logged repairs on a valid surface: {log}"


# --- A3: the validator ------------------------------------------------------------------------------------------


def test_a2ui_schema_exposes_the_v2_id_and_a_loader() -> None:
    """A3: ``a2ui_schema.CARDS_V2_CATALOG_ID`` (re-exported by ``local_astra.ui``) and a v2 loader returning the
    catalog file's contents."""
    assert getattr(a2ui_schema, "CARDS_V2_CATALOG_ID", None) == V2_ID, (
        "A3: a2ui_schema.CARDS_V2_CATALOG_ID"
    )
    import local_astra.ui as ui

    assert getattr(ui, "CARDS_V2_CATALOG_ID", None) == V2_ID, (
        "A3: local_astra.ui must export CARDS_V2_CATALOG_ID"
    )
    want = _v2_catalog()
    candidates: list[tuple[str, Callable[[], Any]]] = [
        (name, fn)
        for name, fn in vars(a2ui_schema).items()
        if "v2" in name.lower()
        and not name.startswith("_")
        and callable(fn)
        and not inspect.isclass(fn)
        and all(
            p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            for p in inspect.signature(fn).parameters.values()
        )
    ]
    loaders = [name for name, fn in candidates if fn() == want]
    assert loaders, (
        "A3: no public argument-free callable in a2ui_schema with 'v2' in its name returns the catalog; "
        f"candidates: {[n for n, _ in candidates]}"
    )


def test_exemplar_fixture_covers_every_kind_and_the_three_boards() -> None:
    """Fixture integrity (not a criterion row; green on main): every v2 kind occurs, every surface is created under
    the v2 id, and the three mockup boards carry the approved content (task: Main, Compare, Chart)."""
    rows = _exemplars()
    keys = [r["key"] for r in rows]
    assert len(keys) == len(set(keys))
    kinds = {c["component"] for r in rows for c in _components(r["messages"])}
    assert set(V2_KINDS) <= kinds
    for r in rows:
        creates = [m["createSurface"] for m in r["messages"] if "createSurface" in m]
        assert [c["catalogId"] for c in creates] == [V2_ID], r["key"]
    boards = {
        r["board"]: {c["id"]: c for c in _components(r["messages"])} for r in rows if r["board"]
    }
    assert sorted(boards) == ["chart", "compare", "main"]

    main = boards["main"]
    stat = next(c for c in main.values() if c["component"] == "Stat")
    assert (stat["label"], stat["value"], stat["detail"], stat["icon"]) == (
        "PARIS · NOW",
        "[17°]",
        "Partly cloudy",
        "partlyCloudy",
    )
    tiles = next(c for c in main.values() if c["component"] == "StatTiles")["tiles"]
    assert [t["title"] for t in tiles] == ["Friday", "Saturday", "Sunday"]
    assert [t.get("highlight", False) for t in tiles] == [False, True, False]
    assert all("percent" in t for t in tiles)
    assert any(c["component"] == "Source" for c in main.values())

    compare = boards["compare"]
    cmp_ = next(c for c in compare.values() if c["component"] == "Comparison")
    assert [c["title"] for c in cmp_["columns"]] == ["Deluxe hardcover", "Regular volumes"]
    assert cmp_["columns"][0]["badge"] == "Recommended" and "badge" not in cmp_["columns"][1]
    assert cmp_["highlight"] == 0
    assert [r["label"] for r in cmp_["rows"]] == ["Volumes", "Price per book", "Format", "Best for"]
    assert all(isinstance(cell, dict) and "tone" in cell for cell in cmp_["rows"][3]["cells"])
    assert any(c["component"] == "Actions" for c in compare.values())

    chart = boards["chart"]
    bars = next(c for c in chart.values() if c["component"] == "Bars")
    assert [(b["label"], b["value"]) for b in bars["bars"]] == [
        ("Everest", 8849),
        ("K2", 8611),
        ("Kangchenjunga", 8586),
        ("Mont Blanc", 4806),
    ]
    assert bars["unit"] == "m"
    assert [c["text"] for c in chart.values() if c["component"] == "Source"] == ["en.wikipedia.org"]


def test_v2_catalog_validates_every_exemplar() -> None:
    """A3: ``A2UIValidator(catalog=<v2>)`` validates every exemplar with zero errors, zero problems and zero
    topology warnings (every exemplar is a whole surface, no orphan)."""
    v = _v2()
    rows = _exemplars()
    assert len(rows) >= len(V2_KINDS)
    bad = {}
    for row in rows:
        rep = v.validate_messages(row["messages"], final=True)
        warnings = rep.topology.warnings if rep.topology else ()
        if not rep.ok or rep.schema_errors or rep.problems or warnings:
            bad[row["key"]] = (
                [str(e) for e in rep.schema_errors],
                list(rep.problems),
                list(warnings),
            )
        for c in _components(row["messages"]):
            assert v.component_errors(c) == [], (row["key"], c["id"])
    assert bad == {}


def test_v2_catalog_component_kinds_are_basic_plus_seven() -> None:
    """A3: ``component_kinds`` is the eighteen basic kinds plus the seven, and the v2 id is a known catalog id."""
    v = _v2()
    assert v.component_kinds == frozenset((*BASIC_KINDS, *V2_KINDS))
    assert V2_ID in v.known_catalog_ids


def test_basic_kinds_validate_on_a_v2_surface() -> None:
    """A1/A3: the basic copies are live on a v2 surface: the T-0164 census components validate under the v2
    catalog exactly as under the basic one (component by component)."""
    v2, basic = _v2(), A2UIValidator()
    rows = [
        json.loads(line)
        for line in CENSUS.read_text(encoding="utf-8").splitlines()[1:]
        if line.strip()
    ]
    n = 0
    for row in rows:
        for canvas in row["canvases"]:
            for c in _components(canvas):
                assert v2.component_errors(c) == basic.component_errors(c), (
                    row["key"],
                    c.get("id"),
                )
                n += 1
    assert n > 100


# --- A2: the seven kinds, by the table ---------------------------------------------------------------------------


def test_v2_accepts_every_optional_prop_and_the_bounds() -> None:
    """A2 positives: each kind with every optional prop; each icon; ``percent`` 0 and 100; a Bars value 0, a
    fraction and 1e12; ``highlight`` beyond the columns (an int ≥ 0 is all A2 asks); 1 and 3 Actions labels."""
    v = _v2()
    for kind in V2_KINDS:
        _assert_base_valid(v, kind)
    for icon in ICONS:
        assert v.component_errors({**BASE["Stat"], "icon": icon}) == [], icon
    for percent in (0, 100, 37.5):
        tiles = [{"title": "t", "value": "v", "percent": percent}]
        assert v.component_errors({**BASE["StatTiles"], "tiles": tiles}) == [], percent
    for value in (0, 0.5, 10**12):
        assert (
            v.component_errors({**BASE["Bars"], "bars": [{"label": "x", "value": value}]}) == []
        ), value
    assert v.component_errors({**BASE["Comparison"], "highlight": 9}) == []
    for labels in (["Only"], ["One", "Two", "Three"]):
        assert v.component_errors({**BASE["Actions"], "labels": labels}) == [], labels
    minimal = {
        "Stat": {"label": "L", "value": "V"},
        "StatTiles": {"tiles": [{"title": "T", "value": "V"}]},
        "Comparison": {
            "columns": [{"title": "A"}, {"title": "B"}],
            "rows": [{"label": "L", "cells": ["x", "y"]}],
        },
        "Bars": {"bars": [{"label": "L", "value": 1}]},
        "Steps": {"steps": [{"title": "T"}]},
        "Source": {"text": "S"},
        "Actions": {"labels": ["A"]},
    }
    for kind, props in minimal.items():
        assert v.component_errors({"id": "c", "component": kind, **props}) == [], (
            f"{kind} with required props only"
        )


def test_only_actions_has_a_max_items() -> None:
    """A2: "The only ``maxItems`` is on ``Actions``": long lists elsewhere validate."""
    v = _v2()
    long_lists = {
        "StatTiles": {"tiles": [{"title": f"T{i}", "value": str(i)} for i in range(12)]},
        "Comparison": {
            "columns": [{"title": f"C{i}"} for i in range(8)],
            "rows": [{"label": f"R{j}", "cells": [str(i) for i in range(8)]} for j in range(30)],
        },
        "Bars": {"bars": [{"label": f"B{i}", "value": i} for i in range(40)]},
        "Steps": {"steps": [{"title": f"S{i}"} for i in range(25)]},
    }
    for kind, props in long_lists.items():
        assert v.component_errors({"id": "c", "component": kind, **props}) == [], kind


#: (kind, path to the required key, as a tuple of keys/indices into BASE[kind])
REQUIRED = [
    ("Stat", ("label",)),
    ("Stat", ("value",)),
    ("StatTiles", ("tiles",)),
    ("StatTiles", ("tiles", 0, "title")),
    ("StatTiles", ("tiles", 0, "value")),
    ("Comparison", ("columns",)),
    ("Comparison", ("rows",)),
    ("Comparison", ("columns", 1, "title")),
    ("Comparison", ("rows", 0, "label")),
    ("Comparison", ("rows", 0, "cells")),
    ("Comparison", ("rows", 0, "cells", 1, "text")),
    ("Comparison", ("rows", 0, "cells", 1, "tone")),
    ("Bars", ("bars",)),
    ("Bars", ("bars", 0, "label")),
    ("Bars", ("bars", 0, "value")),
    ("Steps", ("steps",)),
    ("Steps", ("steps", 1, "title")),
    ("Source", ("text",)),
    ("Actions", ("labels",)),
]


@pytest.mark.parametrize(
    ("kind", "path"), REQUIRED, ids=["-".join(map(str, (k, *p))) for k, p in REQUIRED]
)
def test_v2_rejects_missing_required(kind: str, path: tuple[Any, ...]) -> None:
    """A2: every key in the Required column is required."""
    v = _v2()
    mutant = _assert_base_valid(v, kind)
    node: Any = mutant
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    _assert_rejected(v, mutant, f"A2: {kind} without {'/'.join(map(str, path))}")


#: (case id, kind, mutation) for A2's minimums and types beyond A4's list.
A2_SHAPES: list[tuple[str, str, Callable[[dict[str, Any]], None]]] = [
    ("stattiles-empty", "StatTiles", lambda c: c.update(tiles=[])),
    ("comparison-one-column", "Comparison", lambda c: c.update(columns=[{"title": "Only"}])),
    ("comparison-no-rows", "Comparison", lambda c: c.update(rows=[])),
    ("comparison-highlight-negative", "Comparison", lambda c: c.update(highlight=-1)),
    ("comparison-highlight-fraction", "Comparison", lambda c: c.update(highlight=0.5)),
    ("comparison-highlight-string", "Comparison", lambda c: c.update(highlight="0")),
    ("bars-empty", "Bars", lambda c: c.update(bars=[])),
    ("steps-empty", "Steps", lambda c: c.update(steps=[])),
    ("actions-empty", "Actions", lambda c: c.update(labels=[])),
    ("percent-negative", "StatTiles", lambda c: c["tiles"][0].update(percent=-1)),
    ("percent-string", "StatTiles", lambda c: c["tiles"][0].update(percent="50")),
    ("highlight-tile-string", "StatTiles", lambda c: c["tiles"][0].update(highlight="yes")),
]


@pytest.mark.parametrize(("case", "kind", "mutate"), A2_SHAPES, ids=[c for c, _, _ in A2_SHAPES])
def test_v2_rejects_shape(case: str, kind: str, mutate: Callable[[dict[str, Any]], None]) -> None:
    """A2: ``≥ 1`` / ``≥ 2`` minimums, ``highlight`` an int ≥ 0, ``percent`` a number 0-100, a tile's
    ``highlight`` a bool."""
    v = _v2()
    mutant = _assert_base_valid(v, kind)
    mutate(mutant)
    _assert_rejected(v, mutant, f"A2 {case}")


#: Every string role of the seven kinds, as (case id, kind, setter of that role).
STRING_ROLES: list[tuple[str, str, Callable[[dict[str, Any], Any], None]]] = [
    ("stat-label", "Stat", lambda c, x: c.update(label=x)),
    ("stat-value", "Stat", lambda c, x: c.update(value=x)),
    ("stat-detail", "Stat", lambda c, x: c.update(detail=x)),
    ("tile-title", "StatTiles", lambda c, x: c["tiles"][0].update(title=x)),
    ("tile-value", "StatTiles", lambda c, x: c["tiles"][0].update(value=x)),
    ("tile-secondary", "StatTiles", lambda c, x: c["tiles"][0].update(secondary=x)),
    ("column-title", "Comparison", lambda c, x: c["columns"][0].update(title=x)),
    ("column-badge", "Comparison", lambda c, x: c["columns"][0].update(badge=x)),
    ("row-label", "Comparison", lambda c, x: c["rows"][0].update(label=x)),
    ("cell", "Comparison", lambda c, x: c["rows"][0]["cells"].__setitem__(0, x)),
    ("cell-text", "Comparison", lambda c, x: c["rows"][0]["cells"][1].update(text=x)),
    ("bars-title", "Bars", lambda c, x: c.update(title=x)),
    ("bars-unit", "Bars", lambda c, x: c.update(unit=x)),
    ("bar-label", "Bars", lambda c, x: c["bars"][0].update(label=x)),
    ("step-title", "Steps", lambda c, x: c["steps"][0].update(title=x)),
    ("step-detail", "Steps", lambda c, x: c["steps"][0].update(detail=x)),
    ("source-text", "Source", lambda c, x: c.update(text=x)),
    ("actions-label", "Actions", lambda c, x: c["labels"].__setitem__(0, x)),
]
BINDINGS = {
    "path": {"path": "/city"},
    # a FunctionCall the basic catalog's DynamicString accepts (formatString returns a string), so a role typed as
    # DynamicString would let it through; an ill-formed call would be rejected for the wrong reason
    "call": {"call": "formatString", "args": {"value": "${/city}"}, "returnType": "string"},
    "number": 828,
}


@pytest.mark.parametrize("binding", sorted(BINDINGS))
@pytest.mark.parametrize(("role", "kind", "put"), STRING_ROLES, ids=[r for r, _, _ in STRING_ROLES])
def test_v2_string_roles_are_plain_strings(
    role: str, kind: str, put: Callable[[dict[str, Any], Any], None], binding: str
) -> None:
    """A2: "All strings are the model's own. There are no URLs, path bindings or function calls": every string
    role rejects a ``{"path"}`` binding, a ``{"call"}`` and a bare number."""
    v = _v2()
    mutant = _assert_base_valid(v, kind)
    put(mutant, copy.deepcopy(BINDINGS[binding]))
    _assert_rejected(v, mutant, f"A2: {kind} {role} given a {binding}")


def test_v2_bars_value_is_a_literal_number() -> None:
    """A2: a Bars value is a number, never a data binding (DynamicNumber would accept ``{"path"}``; the basic
    catalog has no number-returning function, so a call row would be vacuous)."""
    v = _v2()
    mutant = _assert_base_valid(v, "Bars")
    mutant["bars"][0]["value"] = copy.deepcopy(BINDINGS["path"])
    _assert_rejected(v, mutant, "A2: Bars value given a path binding")


# --- A4: adversarial --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", V2_KINDS)
def test_v2_rejects_extra_prop(kind: str) -> None:
    """A4: an extra prop on each v2 kind."""
    v = _v2()
    mutant = _assert_base_valid(v, kind)
    mutant["colour"] = "red"
    _assert_rejected(v, mutant, f"A4: {kind} with an extra prop")


#: (case id, kind, path to one nested object in BASE[kind]) — every nested object of the seven kinds (review F1).
NESTED = [
    ("tile", "StatTiles", ("tiles", 0)),
    ("column", "Comparison", ("columns", 0)),
    ("row", "Comparison", ("rows", 0)),
    ("toned-cell", "Comparison", ("rows", 0, "cells", 1)),
    ("bar", "Bars", ("bars", 0)),
    ("step", "Steps", ("steps", 0)),
]


@pytest.mark.parametrize(("case", "kind", "path"), NESTED, ids=[c for c, _, _ in NESTED])
def test_v2_rejects_extra_prop_in_nested_object(
    case: str, kind: str, path: tuple[Any, ...]
) -> None:
    """A4 extended (review F1): an extra prop inside each nested object (tile, column, row, bar, step, toned cell)."""
    v = _v2()
    mutant = _assert_base_valid(v, kind)
    node: Any = mutant
    for key in path:
        node = node[key]
    assert isinstance(node, dict), f"{case}: the path must reach an object"
    node["colour"] = "red"
    _assert_rejected(v, mutant, f"A4: {kind} {case} with an extra prop")


def test_v2_rejects_bars_value_string() -> None:
    """A4: ``Bars.value`` given as the string ``"828"``."""
    v = _v2()
    mutant = _assert_base_valid(v, "Bars")
    mutant["bars"][0]["value"] = "828"
    _assert_rejected(v, mutant, "A4: Bars value as a string")


@pytest.mark.parametrize("value", [-1, -0.5])
def test_v2_rejects_negative_value(value: float) -> None:
    """A4: a negative Bars value."""
    v = _v2()
    mutant = _assert_base_valid(v, "Bars")
    mutant["bars"][1]["value"] = value
    _assert_rejected(v, mutant, f"A4: Bars value {value}")


@pytest.mark.parametrize("percent", [140, 100.5])
def test_v2_rejects_percent_over_100(percent: float) -> None:
    """A4: ``percent: 140`` (and just over the bound)."""
    v = _v2()
    mutant = _assert_base_valid(v, "StatTiles")
    mutant["tiles"][0]["percent"] = percent
    _assert_rejected(v, mutant, f"A4: percent {percent}")


def test_v2_rejects_tone_error() -> None:
    """A4: a Comparison cell with ``tone: "error"`` (only ok and warn)."""
    v = _v2()
    mutant = _assert_base_valid(v, "Comparison")
    for tone in ("ok", "warn"):
        mutant["rows"][0]["cells"][1]["tone"] = tone
        assert v.component_errors(mutant) == [], f"positive control: tone {tone}"
    mutant["rows"][0]["cells"][1]["tone"] = "error"
    _assert_rejected(v, mutant, "A4: tone error")


@pytest.mark.parametrize("icon", ["tornado", "Sun", "partly_cloudy", ""])
def test_v2_rejects_unknown_icon(icon: str) -> None:
    """A4: an unknown ``icon`` (case and spelling matter)."""
    v = _v2()
    mutant = _assert_base_valid(v, "Stat")
    mutant["icon"] = icon
    _assert_rejected(v, mutant, f"A4: icon {icon!r}")


def test_v2_rejects_actions_four_labels() -> None:
    """A4: ``Actions.labels`` with 4 entries."""
    v = _v2()
    mutant = _assert_base_valid(v, "Actions")
    mutant["labels"] = ["One", "Two", "Three", "Four"]
    _assert_rejected(v, mutant, "A4: four Actions labels")


@pytest.mark.parametrize("labels", [[""], ["Yes", ""]])
def test_v2_rejects_actions_empty_label(labels: list[str]) -> None:
    """A4: ``Actions.labels`` containing ``""``."""
    v = _v2()
    mutant = _assert_base_valid(v, "Actions")
    mutant["labels"] = labels
    _assert_rejected(v, mutant, f"A4: Actions labels {labels}")


@pytest.mark.parametrize("prop", ["child", "children"])
@pytest.mark.parametrize("kind", V2_KINDS)
def test_v2_rejects_child_reference(kind: str, prop: str) -> None:
    """A4/A2: the v2 kinds are leaves; one carrying ``child`` or ``children`` is invalid, even when the referenced
    component exists on the surface."""
    v = _v2()
    mutant = _assert_base_valid(v, kind)
    mutant[prop] = "t" if prop == "child" else ["t"]
    _assert_rejected(v, mutant, f"A4: {kind} with {prop}")
    sid = "v2-leaf"
    msgs = [
        {"version": "v0.9.1", "createSurface": {"surfaceId": sid, "catalogId": V2_ID}},
        {
            "version": "v0.9.1",
            "updateComponents": {
                "surfaceId": sid,
                "components": [
                    {**mutant, "id": "root"},
                    {"id": "t", "component": "Text", "text": "hi"},
                ],
            },
        },
    ]
    assert not v.validate_messages(msgs, final=True).ok


# --- A5: negative (green on main by design) ---------------------------------------------------------------------

#: Every canvas of the T-0164 census through ``A2UIValidator().validate_messages``, at BASE aeac453f:
#: (key, canvas index, ok with final=True, problems with final=True, ok with final=False). No canvas had a schema
#: error under either mode.
CENSUS_VERDICTS = [
    ("desk-t3", 0, True, [], True),
    ("desk-t3", 1, True, [], True),
    ("desk-t5", 0, True, [], True),
    ("desk-t5", 1, True, [], True),
    ("desk-t6", 0, True, [], True),
    ("desk-t6", 1, True, [], True),
    ("desk-t8", 0, True, [], True),
    ("desk-t8", 1, True, [], True),
    ("desk-t9", 0, True, [], True),
    ("desk-t9", 1, True, [], True),
    ("desk-t10", 0, True, [], True),
    ("desk-t10", 1, False, ["surface 'turn-t10': 'r1' references missing component 't1'"], True),
    ("desk-t10", 2, True, [], True),
    ("trial-omelette", 0, True, [], True),
    ("trial-omelette", 1, True, [], True),
    ("trial-flight", 0, True, [], True),
    ("trial-flight", 1, True, [], True),
    ("quiz", 0, True, [], True),
    ("census-01", 0, True, [], True),
    ("census-01", 1, True, [], True),
    ("census-02", 0, True, [], True),
    ("census-02", 1, True, [], True),
    ("census-03", 0, True, [], True),
    ("census-04", 0, True, [], True),
    ("census-04", 1, True, [], True),
]
#: sha256 of the canonical JSON of the full detail (both modes: ok, schema errors, problems, topology warnings,
#: reachable ids) at BASE aeac453f; see ``_census_detail``.
CENSUS_DETAIL_SHA256 = "58374b73832079438097c5acfa343175ab9cffa9c9e05be8baff17b87db4a9c2"


def _census_detail(v: A2UIValidator) -> tuple[list[tuple[Any, ...]], list[Any]]:
    rows = [
        json.loads(line)
        for line in CENSUS.read_text(encoding="utf-8").splitlines()[1:]
        if line.strip()
    ]
    verdicts: list[tuple[Any, ...]] = []
    detail: list[Any] = []
    for row in rows:
        for i, canvas in enumerate(row["canvases"]):
            reps = [v.validate_messages(canvas, final=final) for final in (True, False)]
            verdicts.append((row["key"], i, reps[0].ok, list(reps[0].problems), reps[1].ok))
            assert reps[0].schema_errors == () and reps[1].schema_errors == (), (row["key"], i)
            detail.append(
                [
                    row["key"],
                    i,
                    [
                        [
                            r.ok,
                            [str(e) for e in r.schema_errors],
                            list(r.problems),
                            list(r.topology.warnings) if r.topology else None,
                            list(r.topology.reachable) if r.topology else None,
                        ]
                        for r in reps
                    ],
                ]
            )
    return verdicts, detail


def test_default_validator_unchanged_on_census() -> None:
    """A5: ``A2UIValidator()`` is today's: the vendored basic catalog (by content and by its vendoring sha256), the
    eighteen kinds, only the two basic ids, and the same verdicts on every canvas of the T-0164 census."""
    v = A2UIValidator()
    assert hashlib.sha256(BASIC_CATALOG_FILE.read_bytes()).hexdigest() == BASIC_CATALOG_SHA256
    assert v.catalog == _basic_catalog()
    assert v.component_kinds == frozenset(BASIC_KINDS)
    assert v.known_catalog_ids == BASIC_IDS
    assert a2ui_schema.KNOWN_CATALOG_IDS == BASIC_IDS
    verdicts, detail = _census_detail(v)
    assert verdicts == CENSUS_VERDICTS
    digest = hashlib.sha256(json.dumps(detail, sort_keys=True).encode()).hexdigest()
    assert digest == CENSUS_DETAIL_SHA256


def test_default_validator_rejects_v2_surfaces() -> None:
    """A5 adversarial: the default validator never learns the v2 catalog. Every exemplar under ``A2UIValidator()``
    fails, naming the unknown catalog id; a v2 kind is an unknown kind there (no global registration)."""
    v = A2UIValidator()
    for row in _exemplars():
        rep = v.validate_messages(row["messages"], final=True)
        assert not rep.ok, row["key"]
        assert any(f"unknown catalogId {V2_ID!r}" in p for p in rep.problems), (
            row["key"],
            rep.problems,
        )
    for kind in V2_KINDS:
        errs = v.component_errors(BASE[kind])
        assert [e.message for e in errs] == [f"unknown component kind {kind!r}"], kind


#: T-0184 group A's BASE: main at aeac453f, the branch point of ``T-0184-a-catalog``. A literal.
BASE_A = "aeac453f3f0d7beccd5ac98641b0c6cbbf446af5"
#: The implementation tip: 2f587843 (T-0184 A: v2 catalog, CARDS_V2_CATALOG_ID, load_cards_v2_catalog). Was BASE_A.
#: Whoever merges group A bumps it to the merged branch tip if later implementation commits land (tests.md).
TIP_A = "2f587843df47e6ec6f650c42b9f420be09af2170"
#: Power ranges known to change the guarded paths: 52be36b0 (T-0178: session.py, turn_machine.py) and 78db2a87
#: (T-0165 WIP: hud/src/main.js, hud/src/panels.js), each against its parent.
POWER_CONDUCTOR = (
    "5f072f41d26c7195963d04bd5d6b0486e899291e",
    "52be36b07a2429d24207e3e0d0daac67627156de",
)
POWER_HUD = ("3deb54aa09d78a41a5cdf07d3dc796c3fa1168d1", "78db2a87a5f027f46b743cd2ccc5f8979d72a0e4")
GUARDED = ("local_astra/conductor", "hud")


def test_group_a_changes_nothing_under_conductor_or_hud() -> None:
    """A5: over ``BASE_A..TIP_A`` (two literals, never the working tree) nothing under ``local_astra/conductor/`` or
    ``hud/`` changes. Power: the same predicate sees a conductor change and a HUD change in known ranges."""
    gr.require_commits(BASE_A, TIP_A, *POWER_CONDUCTOR, *POWER_HUD)
    assert gr.is_ancestor(BASE_A, TIP_A), f"{BASE_A[:9]} is not an ancestor of {TIP_A[:9]}"
    moved = gr.changed(BASE_A, TIP_A, *GUARDED)
    assert moved == [], f"T-0184 A5: group A changed {moved}"
    assert "local_astra/conductor/session.py" in gr.changed(*POWER_CONDUCTOR, *GUARDED)
    assert "hud/src/main.js" in gr.changed(*POWER_HUD, *GUARDED)
