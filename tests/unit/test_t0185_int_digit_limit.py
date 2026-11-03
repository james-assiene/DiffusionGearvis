"""T-0185: the UI gate never raises on an integer literal past Python's int-string limit (docs/tasks/T-0185.md).

CPython refuses to convert an integer string of more than ``sys.get_int_max_str_digits()`` digits (4300 by default;
the sign does not count). At BASE (main 7646b8b6) a canvas carrying such a literal makes ``UiGate.gate`` raise
``ValueError`` on the strict parse and on the strict re-parse after the bracket repair, and json_repair's path quietly
returns the digits as a ``str``. These rows pin the criteria:

- **the literal** is ``"7" * D`` (optionally ``-``-signed), spliced as text into compact JSONL; the int is never built in
  this process (``str`` of it would raise here). Every string the canvases carry besides the literal is free of the
  digit 7, so "a string coming from the literal" is a string of 7s.
- **the control** is the same canvas with a 310-digit literal of the same sign (``7.7e309``, past the largest double),
  which BASE parses and T-0184 C10 refuses.
- **the verdict** is T-0184's ``_verdict``: every ``GateResult`` field, the report reduced to ``ok``.

The acceptance rows (activated 2026-10-28; they carried ``pending(task="T-0185")`` until then) are red at BASE: the
gate raises (turned into an ``AssertionError`` naming the exception by ``_call``), or a verdict differs from the
control's. The differential (criterion 10a), the too-wide
guard (criterion 9), the frozen-oracle pin, the source row (criterion 7) and the premise rows are green at BASE and
unmarked. The oracle is ``tests/unit/_frozen_ui_gate_t0185.py``: BASE's ``ui_gate.py`` verbatim plus BASE's
``parse_concatenated_json`` verbatim, which rebinds that module's global (implementation note 6).

Readings the test-author made where the task is open (named in the report):

1. *Criterion 4, first bullet, on the bracket path in a string-typed slot* (``Text.text`` and the component ``id``): at
   BASE the control's salvage turns the 310-digit number into the text ``"777…"`` (310 characters), so no verdict for
   the 4301-digit literal can *equal* the control's. These rows compare the literal's verdict with the control's
   after replacing every string equal to the control's digits by the literal's digits as written (``_substituted``),
   which is criterion 4's second bullet ("any emitted string coming from the literal is exactly the literal's
   characters as written"). Everywhere else the substitution is a no-op (asserted as a premise), so the comparison is
   plain equality.
2. *Criterion 5, second bullet* ("no emitted message anywhere carries a ``str`` of 4301 or more digits that the canvas
   wrote as a bare number") conflicts with criterion 4's second bullet in a string role, where the control is emitted
   as text and the literal must follow it with "exactly the literal's characters as written". These rows apply
   criterion 5's second bullet to the criterion-5 placements (number roles and the data model) and to every
   non-string slot of criterion 2's json_repair canvases; the string-role cut rows follow criterion 4.
3. *Criterion 7, source half* (the amendment's ruling (c)): the row forbids changing the limit, not reading it. Code
   under ``local_astra/`` may not name ``set_int_max_str_digits`` or ``PYTHONINTMAXSTRDIGITS``, nor carry a string
   naming ``int_max_str_digits`` other than as ``get_int_max_str_digits`` (the ``-X int_max_str_digits`` option);
   ``sys.get_int_max_str_digits()`` reads are allowed anywhere. The row reads code only (``ast``), so prose in a
   docstring cannot trip it.
4. *Criterion 2(b), "cut inside the literal"* (ruling (d)): a cut inside a 4301-digit literal leaves at most 4300
   digits, which is not past the limit. The inside cut is taken in the 10^5-digit literal, after 5 000 digits; a text
   cut right after a literal cannot be told from a text cut inside one, so the control of both cut shapes is the
   control canvas cut right after its 310 digits.

Budget (the amendment, criterion 12's 5 s): criterion 3 cuts within 10 of the literal's edges plus every 256th
offset; criteria 4, 8 and 11 run both signs only for the number-role placement of each catalog and the other
placements unsigned.

Regime: CPU only, no service, no socket; Python 3.11.16, json_repair 0.63.4, default int-string limit 4300 (rows skip
naming the limit otherwise); deterministic, n = 1 per call.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import traceback
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.ui_gate import GateResult, UiGate
from local_astra.ui import A2UIValidator
from local_astra.ui import parse_concatenated_json as live_parse_concatenated_json
from local_astra.ui.a2ui_schema import SurfaceReport, load_cards_v2_catalog
from local_astra.ui.bracket_repair import repair_structure
from tests import _git_range as gr
from tests import _home_jail
from tests.unit import _frozen_ui_gate_t0185 as frozen

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "a2ui"
CENSUS = FIXTURES / "genui_census_20261027.jsonl"
Q12 = FIXTURES / "q12_trials_sample.jsonl"
FROZEN_FILE = Path(__file__).with_name("_frozen_ui_gate_t0185.py")

#: T-0185's branch base: main at 7646b8b6 (T-0184 group B merged, C merged at 5fe1e9ff). A literal, never a ref.
BASE = "7646b8b6409a30fcc3794c2453f4b77e080cbdb6"
#: sha256 of ``git show 7646b8b6:local_astra/conductor/ui_gate.py`` (blob faff5376), measured 2026-10-28.
BASE_GATE_SHA256 = "129b0b5e435d82084629b75c3854c480964a29e9afc874c4589a7542d66ca0c0"
#: sha256 of BASE's ``parse_concatenated_json`` source (a2ui_schema.py lines 77-90 at 7646b8b6, blob ab7db0de).
BASE_PARSE_SHA256 = "f310f4719a0fade716ec9d10aec71b9a104e3938bab25d78357ab812debae6c7"
GATE_MARKER = (
    "# --- verbatim copy of 7646b8b6:local_astra/conductor/ui_gate.py below this line ---\n"
)
PARSE_MARKER = (
    "# --- verbatim copy of 7646b8b6:local_astra/ui/a2ui_schema.py parse_concatenated_json"
    " below this line ---\n"
)

V2_ID = "gearvis:cards/v2"
BASIC_V091 = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
BASIC_V09 = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"
CATALOGS = {"basic": BASIC_V091, "v2": V2_ID}

#: The regime's int-string limit (CPython's default). Every D below is chosen against it.
DEFAULT_LIMIT = 4300
OVER = DEFAULT_LIMIT + 1  # 4301: the smallest literal past the default limit
HUGE = 10**5  # criterion 1's second digit count
CONTROL = 310  # 7.7e309 > sys.float_info.max: BASE parses it, T-0184 C10 refuses it
INSIDE = (
    5_000  # criterion 2(b)'s inside cut: 5 000 digits of the 10^5-digit literal (past the limit)
)
assert CONTROL < DEFAULT_LIMIT < OVER < INSIDE < HUGE

PH = "__T0185_LITERAL__"
PH_JSON = json.dumps(PH)

GATE_FIELDS = tuple(f.name for f in dataclasses.fields(frozen.GateResult))


#: Rows that read source or the interpreter, not the gate's output: they run under any limit, so a process-wide
#: change (an import-time ``sys.set_int_max_str_digits(0)``) is reported by them rather than hidden behind the
#: regime check (reviewer B, round 1).
_REGIME_FREE = {
    "test_interpreter_limit_is_the_default",
    "test_no_process_wide_limit_change",
    "test_no_process_wide_limit_change_sees_a_set_call",
}


def _limit_message(limit: int) -> str:
    return (
        f"criterion 7 / regime: the in-process int-string limit is {limit}, not CPython's default 4300; something "
        "changed it process-wide (an import-time sys.set_int_max_str_digits, PYTHONINTMAXSTRDIGITS or "
        "-X int_max_str_digits in this run)"
    )


@pytest.fixture(autouse=True)
def _default_limit(request: pytest.FixtureRequest) -> None:
    """Every gate row assumes the default limit. Another limit fails loudly (never skips: a skip would hide an
    import-time ``set_int_max_str_digits`` behind 200 skipped rows)."""
    if request.function.__name__ in _REGIME_FREE:
        return
    limit = sys.get_int_max_str_digits()
    assert limit == DEFAULT_LIMIT, _limit_message(limit)


def test_interpreter_limit_is_the_default() -> None:
    """Criterion 7, the process half (regime-free): after importing the gate, ``A2UIValidator`` and the frozen
    oracle, this interpreter's int-string limit is still CPython's default 4300."""
    limit = sys.get_int_max_str_digits()
    assert limit == DEFAULT_LIMIT, _limit_message(limit)


# --- helpers (T-0184's shapes, copied) ----------------------------------------------------------------------------


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _compact(messages: list[Any]) -> str:
    return "\n".join(
        json.dumps(m, ensure_ascii=False, separators=(",", ":"), allow_nan=True) for m in messages
    )


def _canvas(catalog_id: str, components: list[Any], sid: str = "s") -> str:
    return _compact(
        [
            {"version": "v0.9.1", "createSurface": {"surfaceId": sid, "catalogId": catalog_id}},
            {"version": "v0.9.1", "updateComponents": {"surfaceId": sid, "components": components}},
        ]
    )


def _with(x: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": "root", "component": "Column", "children": ["t", "x"]},
        {"id": "t", "component": "Text", "text": "Kept"},
        x,
    ]


def _with_data(catalog_id: str, value: Any, path: Any = "/") -> str:
    return _compact(
        [
            {"version": "v0.9.1", "createSurface": {"surfaceId": "s", "catalogId": catalog_id}},
            {
                "version": "v0.9.1",
                "updateComponents": {
                    "surfaceId": "s",
                    "components": [
                        {"id": "root", "component": "Column", "children": ["t"]},
                        {"id": "t", "component": "Text", "text": "Kept"},
                    ],
                },
            },
            {
                "version": "v0.9.1",
                "updateDataModel": {"surfaceId": "s", "path": path, "value": value},
            },
        ]
    )


def _safe_repr(value: Any, limit: int = 400) -> str:
    """``repr`` that survives an int past the int-string limit (it would raise ``ValueError`` here)."""
    try:
        text = repr(value)
    except ValueError:
        return "<unprintable: holds an int past the int-string limit>"
    return text if len(text) <= limit else f"{text[:limit]}… ({len(text)} characters)"


def _snapshot(result: Any) -> dict[str, Any]:
    """Every BASE ``GateResult`` field of ``result``, as plain data (the surface dumped, the report as a dict)."""
    out: dict[str, Any] = {}
    for name in GATE_FIELDS:
        assert hasattr(result, name), f"GateResult lost BASE's field {name!r}"
        value = getattr(result, name)
        if name == "surface":
            value = None if value is None else value.model_dump(mode="python")
        elif name == "report":
            value = dataclasses.asdict(value)
        out[name] = value
    return out


def _verdict(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Everything but the wording of the schema errors and problems."""
    out = {k: v for k, v in snapshot.items() if k != "report"}
    out["ok"] = snapshot["report"]["ok"]
    return out


def _typed(node: Any) -> Any:
    """``node`` with every leaf and container tagged by its type, so ``1`` and ``1.0`` (or ``True``) differ
    (reviewer B, round 1: ``==`` alone cannot see a value whose type changed)."""
    if isinstance(node, Mapping):
        return (type(node).__name__, {k: _typed(v) for k, v in node.items()})
    if isinstance(node, list | tuple):
        return (type(node).__name__, [_typed(v) for v in node])
    return (type(node).__name__, node)


def _first_difference(a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    for name in (*GATE_FIELDS, "ok"):
        if (name in a or name in b) and a.get(name) != b.get(name):
            return (
                f"field {name!r}:\n  tip:     {_safe_repr(a.get(name))}\n"
                f"  other:   {_safe_repr(b.get(name))}"
            )
    return "(no field differs)"


def _leaves(node: Any, key: str | None = None) -> Iterator[tuple[str | None, Any]]:
    if isinstance(node, Mapping):
        for k, v in node.items():
            yield from _leaves(v, str(k))
    elif isinstance(node, list | tuple):
        for v in node:
            yield from _leaves(v, key)
    else:
        yield key, node


def _emitted(result: Any) -> dict[str, dict[str, Any]]:
    """Components the result's surface emits, by id (the last definition wins)."""
    out: dict[str, dict[str, Any]] = {}
    if result.surface is None:
        return out
    for message in result.surface.messages:
        body = message.get("updateComponents") if isinstance(message, Mapping) else None
        if isinstance(body, Mapping):
            for c in body.get("components", []):
                cid = c.get("id") if isinstance(c, Mapping) else None
                out[cid if isinstance(cid, str) else _safe_repr(cid)] = c
    return out


def _beyond_double(messages: Any) -> list[str]:
    """Numbers no double can hold (NaN/±Infinity, integers past the largest double), described without printing."""
    out = []
    for key, v in _leaves(list(messages)):
        if isinstance(v, bool):
            continue
        if isinstance(v, float) and not math.isfinite(v):
            out.append(f"{key}: {v!r}")
        elif isinstance(v, int) and abs(v) > sys.float_info.max:
            out.append(f"{key}: an int of about {v.bit_length() * 0.30103:.0f} digits")
    return out


_RESERVED = re.compile(r"^__gate[._]")


def _ungrounded(messages: Any, raw: str) -> list[str]:
    """String leaves of ``messages`` that are not the model's (T-0108 grounding): not in ``raw`` literally or
    JSON-escaped. Exempt: ``surfaceId`` (the gate rewrites it) and the reserved ``__gate`` namespace."""
    bad = []
    for key, value in _leaves(list(messages)):
        if not isinstance(value, str) or key == "surfaceId" or _RESERVED.match(value):
            continue
        if value in raw or json.dumps(value)[1:-1] in raw:
            continue
        bad.append(_safe_repr(value, 80))
    return bad


_SEVENS = re.compile(r"-?7+")
_DIGITS = re.compile(r"-?[0-9]+")


def _sevens(messages: Any) -> list[str]:
    """Emitted strings made of the literal's characters (a run of 7s, optionally signed)."""
    return [v for _, v in _leaves(list(messages)) if isinstance(v, str) and _SEVENS.fullmatch(v)]


def _long_digit_strings(messages: Any, limit: int = DEFAULT_LIMIT) -> list[str]:
    """Emitted ``str`` leaves that are a bare integer of more than ``limit`` digits (criterion 5), described."""
    return [
        f"{key}: a str of {len(v.lstrip('-'))} digits"
        for key, v in _leaves(list(messages))
        if isinstance(v, str) and _DIGITS.fullmatch(v) and len(v.lstrip("-")) > limit
    ]


def _digits(n: int, negative: bool = False) -> str:
    """The literal's text: ``n`` sevens, optionally signed. Never an ``int``."""
    return ("-" if negative else "") + "7" * n


def _splice(template: str, literal: str) -> str:
    assert template.count(PH_JSON) == 1, "premise: the template holds the placeholder once"
    return template.replace(PH_JSON, literal)


def _call(raw: str, *, final: bool, gate: Any = None, turn: str = "t", idx: int = 0) -> Any:
    """One gate call. An escaping exception becomes an ``AssertionError`` naming it (criteria 1-3), and the
    interpreter's int-string limit must be the same after the call as before (criterion 7)."""
    before = sys.get_int_max_str_digits()
    g = UiGate() if gate is None else gate
    try:
        result = g.gate(raw, turn_id=turn, canvas_idx=idx, final=final)
    except (
        Exception
    ) as err:  # the criterion is "never raises": name the exception and where, not a traceback
        # (``from None``: formatting the chained traceback of every pending row cost ~30 ms per row at BASE)
        frames = traceback.extract_tb(err.__traceback__)
        inner = [f for f in frames if "local_astra" in f.filename] or frames
        where = f"{Path(inner[-1].filename).name}:{inner[-1].lineno}" if inner else "?"
        raise AssertionError(
            f"the gate raised {type(err).__name__} at {where}: {str(err)[:160]} "
            f"(canvas of {len(raw)} characters)"
        ) from None
    after = sys.get_int_max_str_digits()
    assert after == before, f"criterion 7: the int-string limit changed from {before} to {after}"
    assert isinstance(result, GateResult), f"the gate returned {type(result).__name__}"
    return result


# --- the canvases ---------------------------------------------------------------------------------------------------

#: Criterion 1's placements. The number role is the catalog's own (basic ``Slider.value``, v2 ``Bars.value``).
PLACEMENTS = (
    "number",
    "text",
    "unknown",
    "id",
    "child",
    "create-extra",
    "sid",
    "data-bare",
    "data-nested",
)
CASES = [(cat, p) for cat in CATALOGS for p in PLACEMENTS]
CASE_IDS = [f"{cat}-{p}" for cat, p in CASES]
STRING_SLOTS = ("text", "id", "child")  # placements whose slot the catalog types as a string


def _template(catalog: str, placement: str) -> str:
    """The canvas of ``placement`` on ``catalog``, with ``PH_JSON`` where the literal goes."""
    cid = CATALOGS[catalog]
    if placement == "number":
        x: dict[str, Any] = (
            {"id": "x", "component": "Slider", "label": "Volume", "value": PH, "min": 0, "max": 10}
            if catalog == "basic"
            else {"id": "x", "component": "Bars", "bars": [{"label": "K2", "value": PH}]}
        )
        return _canvas(cid, _with(x))
    if placement == "text":
        return _canvas(cid, _with({"id": "x", "component": "Text", "text": PH}))
    if placement == "unknown":
        return _canvas(cid, _with({"id": "x", "component": "Text", "text": "Also", "zz": PH}))
    if placement == "id":
        return _canvas(
            cid,
            [
                {"id": "root", "component": "Column", "children": ["t"]},
                {"id": "t", "component": "Text", "text": "Kept"},
                {"id": PH, "component": "Text", "text": "Orphan"},
            ],
        )
    if placement == "child":
        return _canvas(
            cid,
            [
                {"id": "root", "component": "Column", "children": ["t", PH]},
                {"id": "t", "component": "Text", "text": "Kept"},
            ],
        )
    if placement == "create-extra":
        tree = [
            {"id": "root", "component": "Column", "children": ["t"]},
            {"id": "t", "component": "Text", "text": "Kept"},
        ]
        return _compact(
            [
                {
                    "version": "v0.9.1",
                    "createSurface": {"surfaceId": "s", "catalogId": cid, "extra": PH},
                },
                {"version": "v0.9.1", "updateComponents": {"surfaceId": "s", "components": tree}},
            ]
        )
    if placement == "sid":
        tree = [
            {"id": "root", "component": "Column", "children": ["t"]},
            {"id": "t", "component": "Text", "text": "Kept"},
        ]
        return _compact(
            [
                {"version": "v0.9.1", "createSurface": {"surfaceId": "s", "catalogId": cid}},
                {"version": "v0.9.1", "updateComponents": {"surfaceId": PH, "components": tree}},
            ]
        )
    if placement == "data-bare":
        return _with_data(cid, PH)
    if placement == "data-nested":
        return _with_data(cid, {"level": PH})
    raise AssertionError(f"unknown placement {placement}")


def _stray(template: str, placement: str) -> str:
    """Criterion 2(a): a stray ``]`` before the literal, right after a ``"surfaceId":"s"``: in ``updateComponents``
    before ``components`` (the task's shape), or in ``createSurface`` where the literal is in ``createSurface`` or
    is the ``updateComponents`` surfaceId (the only place before the literal)."""
    if placement in ("create-extra", "sid"):
        anchor = '"createSurface":{"surfaceId":"s"'
    else:
        anchor = '"updateComponents":{"surfaceId":"s"'
    i = template.index(anchor) + len(anchor)
    assert template.index(PH_JSON) > i, "premise: the stray ] is before the literal"
    return template[:i] + "]" + template[i:]


def _bare(template: str) -> str:
    """Criterion 2(b): the key right before the literal written bare (unquoted), which no strict parse accepts."""
    at = template.index(PH_JSON)
    key = list(re.finditer(r'"(\w+)":', template[:at]))[-1]
    return template[: key.start()] + key.group(1) + ":" + template[key.end() :]


def _cut_after(template: str, literal: str) -> str:
    """The canvas cut right after the literal (the decoder reaches the number before the cut)."""
    at = template.index(PH_JSON)
    return template[:at] + literal


#: Criterion 2(b)'s three json_repair shapes: (name, canvas text from a template and a sign, digits as written)
SHAPES = ("bare-key", "cut-after", "cut-inside")


def _shape_canvas(template: str, shape: str, negative: bool) -> tuple[str, str]:
    """``(canvas, the literal as written in it)`` for a json_repair shape."""
    if shape == "bare-key":
        lit = _digits(OVER, negative)
        return _splice(_bare(template), lit), lit
    if shape == "cut-after":
        lit = _digits(OVER, negative)
        return _cut_after(template, lit), lit
    if shape == "cut-inside":
        lit = _digits(INSIDE, negative)  # 5 000 digits of the 10^5-digit literal, then the cut
        return _cut_after(template, lit), lit
    raise AssertionError(shape)


def _control_shape_canvas(template: str, shape: str, negative: bool) -> str:
    lit = _digits(CONTROL, negative)
    if shape == "bare-key":
        return _splice(_bare(template), lit)
    return _cut_after(template, lit)  # a text cut inside a literal reads as one cut right after it


def _substituted(node: Any, old: str, new: str) -> Any:
    """``node`` with every string equal to ``old`` replaced by ``new`` (reading 1 of the module docstring)."""
    if isinstance(node, str):
        return new if node == old else node
    if isinstance(node, Mapping):
        return {k: _substituted(v, old, new) for k, v in node.items()}
    if isinstance(node, list):
        return [_substituted(v, old, new) for v in node]
    if isinstance(node, tuple):
        return tuple(_substituted(v, old, new) for v in node)
    return node


def _holds(node: Any, text: str) -> bool:
    return any(isinstance(v, str) and v == text for _, v in _leaves(node))


# --- criterion 1: the strict path never raises -----------------------------------------------------------------------


@pytest.mark.parametrize(("catalog", "placement"), CASES, ids=CASE_IDS)
def test_strict_path_never_raises(catalog: str, placement: str) -> None:
    """Criterion 1 (and criterion 7 around every call): a literal of D in {4301, 10^5} digits, unsigned and signed,
    with ``final`` True and False, returns a ``GateResult`` in every placement, on both catalogs."""
    template = _template(catalog, placement)
    for n in (OVER, HUGE):
        for negative in (False, True):
            raw = _splice(template, _digits(n, negative))
            for final in (True, False):
                _call(raw, final=final)


# --- criterion 2: the two repair paths never raise -------------------------------------------------------------------


@pytest.mark.parametrize(("catalog", "placement"), CASES, ids=CASE_IDS)
def test_bracket_reparse_premise(catalog: str, placement: str) -> None:
    """Premise of criterion 2(a) (unmarked, green at BASE): the stray ``]`` makes BASE's strict parse fail *at the
    bracket* (a ``JSONDecodeError``, before the literal), the structural pass makes at least one edit and keeps the
    literal, and on the control the structural pass alone makes the text parse. So the literal reaches the strict
    re-parse after the bracket repair (ui_gate.py:1032 at BASE), the path a strict-parse-only guard misses."""
    template = _stray(_template(catalog, placement), placement)
    lit = _digits(OVER)
    raw = _splice(template, lit)
    with pytest.raises(json.JSONDecodeError):
        frozen.parse_concatenated_json(raw)
    text, edits = repair_structure(raw)
    assert edits > 0 and lit in text, (edits, lit in text)
    control_text, control_edits = repair_structure(_splice(template, _digits(CONTROL)))
    assert control_edits > 0
    assert frozen.parse_concatenated_json(control_text), (
        "premise: the structural pass makes the control parse"
    )


@pytest.mark.parametrize(("catalog", "placement"), CASES, ids=CASE_IDS)
def test_bracket_reparse_path_never_raises(catalog: str, placement: str) -> None:
    """Criterion 2(a): the canvas with a stray ``]`` before the literal (4301 digits; unsigned with ``final=True``,
    signed with ``final=False``: the budget, criterion 12) returns a ``GateResult``. BASE raises at the re-parse
    after the bracket repair."""
    template = _stray(_template(catalog, placement), placement)
    for negative, final in ((False, True), (True, False)):
        _call(_splice(template, _digits(OVER, negative)), final=final)


@pytest.mark.parametrize(("catalog", "placement"), CASES, ids=CASE_IDS)
def test_json_repair_path_never_raises(catalog: str, placement: str) -> None:
    """Criterion 2(b): a bare key before the literal (json_repair's path), the canvas cut right after the literal,
    and the canvas cut inside the 10^5-digit literal after 5 000 digits, each unsigned, plus the bare key signed;
    ``final=False`` (the progressive call). Each returns a ``GateResult``. BASE raises on both cuts (its strict parse
    reaches the number before the end of the text)."""
    template = _template(catalog, placement)
    for shape, negative in (*((s, False) for s in SHAPES), ("bare-key", True)):
        raw, _ = _shape_canvas(template, shape, negative)
        _call(raw, final=False)


# --- criterion 3: cut canvases never raise and stay grounded -------------------------------------------------------


def _cut_offsets(text: str, start: int, end: int) -> list[int]:
    """Criterion 3's cut lengths (as amended for the budget): every offset within 10 of the literal's start and end,
    every 256th elsewhere, and the whole canvas."""
    near = {*range(start - 10, start + 11), *range(end - 10, end + 11)}
    every = {*range(0, len(text) + 1, 256), len(text)}
    return sorted(k for k in near | every if 1 <= k <= len(text))


def test_cuts_never_raise_and_stay_grounded() -> None:
    """Criterion 3 (cut set as amended): the full two-message basic canvas with a 4301-digit literal in
    ``Slider.value``, cut at every offset within 10 of the literal's start and end and every 256th offset elsewhere;
    each cut through a fresh gate
    with ``final=False`` returns, and every emitted string is a substring of the cut (or of its JSON-escaped
    rendering; ``surfaceId`` and the reserved namespace exempt, as in T-0108 criterion 6)."""
    lit = _digits(OVER)
    text = _splice(_template("basic", "number"), lit)
    start = text.index(lit)
    end = start + len(lit)
    offsets = _cut_offsets(text, start, end)
    edges = {start - 10, start, start + 1, start + 10, end - 10, end - 1, end, end + 10, len(text)}
    assert edges <= set(offsets) and len(offsets) > 50, "premise: the cut set"
    for k in offsets:
        cut = text[:k]
        result = _call(cut, final=False)
        if result.surface is not None:
            bad = _ungrounded(result.surface.messages, cut)
            assert not bad, f"criterion 3, cut {k}: ungrounded {bad}"


# --- criterion 4: refused like a beyond-float integer ----------------------------------------------------------------


def _expect_by_hand(placement: str, result: Any, path: str) -> None:
    """The consequences criterion 4 names, derived by hand (not from the control): a number role or an unknown
    property makes ``x`` invalid, so ``x`` is emitted nowhere, is lost and resampled, the card is not complete, and
    the rest (``root``, ``t``) is emitted; a literal outside the components fails its message, so no surface."""
    if placement in ("number", "unknown") or (placement == "text" and path == "strict"):
        assert result.surface is not None, (
            f"the rest of the card was lost: {result.report.problems[:3]}"
        )
        emitted = _emitted(result)
        assert "x" not in emitted and {"root", "t"} <= set(emitted), sorted(emitted)
        assert result.lost == ("x",) and "x" in result.resample and not result.complete
    elif placement in ("create-extra", "sid", "data-bare", "data-nested"):
        assert result.surface is None, "a message carrying the literal reached the renderer"


def _compare_with_control(
    raw: str, control_raw: str, lit: str, ctrl: str, *, final: bool, path: str
) -> Any:
    result = _call(raw, final=final)
    control = UiGate().gate(control_raw, turn_id="t", canvas_idx=0, final=final)
    tip = _verdict(_snapshot(result))
    want = _verdict(_snapshot(control))
    if path == "strict":
        assert not _holds(want, ctrl), (
            "premise: on the strict path the control's verdict carries no digit string"
        )
    expected = _substituted(want, ctrl, lit)
    assert tip == expected, (
        f"criterion 4 ({path}, final={final}): {_first_difference(tip, expected)}"
    )
    return result


@pytest.mark.parametrize("path", ["strict", "bracket"])
@pytest.mark.parametrize(("catalog", "placement"), CASES, ids=CASE_IDS)
def test_refused_like_a_beyond_float_integer(catalog: str, placement: str, path: str) -> None:
    """Criterion 4, number roles and the envelope, adversarial: for every criterion-1 placement (strict path) and
    every criterion-2(a) placement (the bracket path), 4301 digits, the verdict equals the control's (reading 1 of the
    module docstring for a string-typed slot on the bracket path). The number role runs unsigned (``final=True``) and
    signed (``final=False``); the other placements run unsigned only (the amendment's budget), ``final=True`` on the
    strict path and ``final=False`` on the bracket path. A fix that wraps ``gate()`` in ``except ValueError`` and
    returns "no message recovered" loses the rest of the surface, and the by-hand checks catch it."""
    template = _template(catalog, placement)
    if path == "bracket":
        template = _stray(template, placement)
    runs = ((False, True), (True, False)) if placement == "number" else ((False, path == "strict"),)
    for negative, final in runs:
        lit, ctrl = _digits(OVER, negative), _digits(CONTROL, negative)
        result = _compare_with_control(
            _splice(template, lit), _splice(template, ctrl), lit, ctrl, final=final, path=path
        )
        _expect_by_hand(placement, result, path)


@pytest.mark.parametrize("catalog", list(CATALOGS))
def test_premise_base_salvage_turns_a_number_in_a_string_role_into_text(catalog: str) -> None:
    """Premise of criterion 4's second bullet (unmarked): at BASE, the control in ``Text.text`` at a cut right after
    it goes through salvage and is emitted as the text of its digits, complete."""
    ctrl = _digits(CONTROL)
    result = frozen.UiGate().gate(
        _cut_after(_template(catalog, "text"), ctrl), turn_id="t", canvas_idx=0, final=False
    )
    assert result.surface is not None and result.complete
    assert _emitted(result)["x"]["text"] == ctrl


@pytest.mark.parametrize("negative", [False, True], ids=["unsigned", "signed"])
@pytest.mark.parametrize("catalog", list(CATALOGS))
def test_string_role_at_a_cut_follows_the_control(catalog: str, negative: bool) -> None:
    """Criterion 4, second bullet: ``Text.text`` cut right after the 4301-digit literal and inside the 10^5-digit
    one (5 000 digits written). The verdict agrees with the control's (cut right after its 310 digits) on: surface
    present, the emitted ids, ``lost``, ``resample``, ``complete``, ``root``, ``unreachable`` and the three repair
    counts; any emitted string made of the literal's characters is exactly the literal as written in the cut."""
    template = _template(catalog, "text")
    control = UiGate().gate(
        _control_shape_canvas(template, "cut-after", negative),
        turn_id="t",
        canvas_idx=0,
        final=False,
    )
    ctrl = _digits(CONTROL, negative)
    fields = ("resample", "lost", "complete", "root", "unreachable", "repaired", "structural_repairs",
              "component_repairs")  # fmt: skip
    for shape in ("cut-after", "cut-inside"):
        raw, lit = _shape_canvas(template, shape, negative)
        result = _call(raw, final=False)
        where = f"{shape}, {len(lit)} characters"
        assert (result.surface is None) == (control.surface is None), (
            f"{where}: surface presence differs"
        )
        assert set(_emitted(result)) == set(_substituted(list(_emitted(control)), ctrl, lit)), where
        for name in fields:
            got, want = getattr(result, name), _substituted(getattr(control, name), ctrl, lit)
            assert got == want, f"{where}: {name}: {_safe_repr(got)} != {_safe_repr(want)}"
        if result.surface is not None:
            odd = [len(s) for s in _sevens(result.surface.messages) if s != lit]
            assert not odd, (
                f"{where}: emitted strings of the literal's characters of lengths {odd}, not {len(lit)}"
            )


# --- criterion 5: no quiet conversion to a string on the json_repair path ---------------------------------------------


@pytest.mark.parametrize("catalog", list(CATALOGS))
def test_premise_base_turns_the_data_model_literal_into_a_string(catalog: str) -> None:
    """Premise of criterion 5 (unmarked, green at BASE): the frozen BASE gate on the nested data-model canvas with a
    bare key before the 4301-digit literal returns a surface, ``complete=True``, whose ``updateDataModel.value.level``
    is a ``str`` of 4301 digits: json_repair caught the conversion's ``ValueError`` and returned the digits."""
    raw, lit = _shape_canvas(_template(catalog, "data-nested"), "bare-key", False)
    result = frozen.UiGate().gate(raw, turn_id="t", canvas_idx=0, final=False)
    assert result.surface is not None and result.complete
    values = [
        m["updateDataModel"]["value"]["level"]
        for m in result.surface.messages
        if isinstance(m, Mapping) and "updateDataModel" in m
    ]
    assert values == [lit] and isinstance(values[0], str)


CRIT5 = [
    (cat, p, s) for cat in CATALOGS for p in ("data-nested", "data-bare", "number") for s in SHAPES
]


@pytest.mark.parametrize(
    ("catalog", "placement", "shape"), CRIT5, ids=[f"{c}-{p}-{s}" for c, p, s in CRIT5]
)
def test_json_repair_does_not_turn_the_literal_into_a_string(
    catalog: str, placement: str, shape: str
) -> None:
    """Criterion 5, adversarial: the criterion-2(b) canvases with the literal in the data model (nested, bare) or a
    number role give the control's verdict (the control canvas in the same shape), unsigned and signed, and no
    emitted message carries a ``str`` of 4301 or more digits. At BASE the nested data-model case gives a surface
    with ``complete=True``. A fix that guards only the strict parse leaves this red."""
    template = _template(catalog, placement)
    for negative in (False, True):
        raw, _ = _shape_canvas(template, shape, negative)
        result = _call(raw, final=False)
        control = UiGate().gate(
            _control_shape_canvas(template, shape, negative), turn_id="t", canvas_idx=0, final=False
        )
        tip, want = _verdict(_snapshot(result)), _verdict(_snapshot(control))
        assert tip == want, f"criterion 5 (negative={negative}): {_first_difference(tip, want)}"
        if result.surface is not None:
            assert not (long := _long_digit_strings(result.surface.messages)), long


# --- criterion 6: the behaviour follows the interpreter's limit ----------------------------------------------------------

_CHILD = r"""
import dataclasses, json, sys
import local_astra
from local_astra.conductor.ui_gate import UiGate
request = json.load(sys.stdin)
verdicts = []
for raw in request["raws"]:
    result = UiGate().gate(raw, turn_id="t", canvas_idx=0, final=True)
    snap = {}
    for f in dataclasses.fields(result):
        value = getattr(result, f.name)
        if f.name == "surface":
            snap["surface"] = None if value is None else value.model_dump(mode="json")
        elif f.name == "report":
            snap["ok"] = value.ok
        else:
            snap[f.name] = value
    verdicts.append(snap)
json.dump({"limit": sys.get_int_max_str_digits(), "origin": local_astra.__file__, "verdicts": verdicts}, sys.stdout)
"""

#: criterion 6's canvases: a number role on each catalog and the nested data model
CRIT6 = [("basic", "number"), ("v2", "number"), ("basic", "data-nested")]


def _plain_verdict(result: Any) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(_verdict(_snapshot(result))))
    return out


def _child_verdicts(limit: str, raws: list[str]) -> list[dict[str, Any]]:
    """The verdicts of a fresh interpreter run with ``PYTHONINTMAXSTRDIGITS=limit`` (a T-0186-clean env: no git
    selectors; the session's jailed ``HOME`` inherited)."""
    env = {k: v for k, v in os.environ.items() if k not in _home_jail.GIT_SELECTORS}
    env["PYTHONINTMAXSTRDIGITS"] = limit
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        input=json.dumps({"raws": raws}),
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"the child gate failed (limit {limit}): {proc.stderr[-600:]}"
    out = json.loads(proc.stdout)
    assert out["limit"] == int(limit), (
        f"premise: the child ran at limit {out['limit']}, not {limit}"
    )
    assert Path(out["origin"]).resolve().is_relative_to(REPO), f"the child imported {out['origin']}"
    verdicts: list[dict[str, Any]] = out["verdicts"]
    return verdicts


def test_limit_follows_the_interpreter_at_640() -> None:
    """Criterion 6, adversarial (a hard-coded 4300 is red here): with ``PYTHONINTMAXSTRDIGITS=640`` (the minimum), a
    child gate on a 641-digit literal returns, and its verdict equals the control's (computed in this process). Ruling
    (g): the json_repair path too (a bare key before the literal in the nested data model and in the basic number
    role), so a run finder hard-coded to 4300 is red here as well as a hard-coded strict parse."""
    raws, controls = [], []
    for catalog, placement in CRIT6:
        template = _template(catalog, placement)
        for negative in (False, True):
            raws.append(_splice(template, _digits(641, negative)))
            control = UiGate().gate(
                _splice(template, _digits(CONTROL, negative)), turn_id="t", canvas_idx=0, final=True
            )
            controls.append(_plain_verdict(control))
    for catalog, placement in (("basic", "data-nested"), ("basic", "number")):
        bare = _bare(_template(catalog, placement))
        raws.append(_splice(bare, _digits(641)))
        control = UiGate().gate(
            _splice(bare, _digits(CONTROL)), turn_id="t", canvas_idx=0, final=True
        )
        controls.append(_plain_verdict(control))
    for k, (got, want) in enumerate(zip(_child_verdicts("640", raws), controls, strict=True)):
        assert got == want, f"case {k}: {_first_difference(got, want)}"


def test_limit_follows_the_interpreter_when_unlimited() -> None:
    """Criterion 6: with ``PYTHONINTMAXSTRDIGITS=0`` (unlimited), a child gate on a 4301-digit literal gives the
    same verdict as this process's gate at the default limit."""
    raws, here = [], []
    for catalog, placement in CRIT6:
        template = _template(catalog, placement)
        for negative in (False, True):
            raw = _splice(template, _digits(OVER, negative))
            raws.append(raw)
            here.append(_plain_verdict(_call(raw, final=True)))
    for k, (got, want) in enumerate(zip(_child_verdicts("0", raws), here, strict=True)):
        assert got == want, f"case {k}: {_first_difference(got, want)}"


# --- criterion 7: no process-wide change (the source half) ----------------------------------------------------------

#: Identifiers that change the limit (ruling (c)): the setter and the environment variable. Reads
#: (``sys.get_int_max_str_digits()``, ``sys.flags.int_max_str_digits``) are allowed anywhere.
_SETTER_NAMES = ("set_int_max_str_digits", "PYTHONINTMAXSTRDIGITS")
#: A string naming the option other than as the getter: ``-X int_max_str_digits=…``, ``"set_int_max_str_digits"``,
#: ``"PYTHONINTMAXSTRDIGITS"``.
_SETTER_STRING = re.compile(r"(?<!get_)int_max_str_digits|PYTHONINTMAXSTRDIGITS")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            for stmt in body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    out.add(id(stmt.value))
    return out


def _limit_mentions(path: Path) -> list[str]:
    """Code (never prose: comments are not in the AST and bare string statements are skipped) that changes the
    limit or names the option to change it (ruling (c)); reads of the limit are not hits."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    prose = _docstring_nodes(tree)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and any(n in node.attr for n in _SETTER_NAMES):
            hits.append(f"attribute {node.attr}")
        elif isinstance(node, ast.Name) and any(n in node.id for n in _SETTER_NAMES):
            hits.append(f"name {node.id}")
        elif isinstance(node, ast.alias) and any(n in node.name for n in _SETTER_NAMES):
            hits.append(f"import {node.name}")
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in prose
            and _SETTER_STRING.search(node.value)
        ):
            hits.append(f"string {node.value[:60]!r}")
    return hits


def test_no_process_wide_limit_change() -> None:
    """Criterion 7, source half (ruling (c); green at BASE, unmarked): no code under ``local_astra/`` names
    ``set_int_max_str_digits`` or ``PYTHONINTMAXSTRDIGITS`` or carries the ``-X int_max_str_digits`` option.
    ``sys.get_int_max_str_digits()`` reads are allowed anywhere (BASE has one in ``toolcall/native.py``). Read with
    ``ast``: comments and docstrings cannot trip it."""
    found: dict[tuple[str, str], int] = {}
    scanned = 0
    for path in sorted((REPO / "local_astra").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "int_max_str_digits" not in text and "PYTHONINTMAXSTRDIGITS" not in text:
            continue  # neither word anywhere in the file (the cheap pass; the AST pass is the verdict)
        scanned += 1
        for hit in _limit_mentions(path):
            key = (path.relative_to(REPO).as_posix(), hit)
            found[key] = found.get(key, 0) + 1
    assert scanned >= 1, "premise: BASE's read in toolcall/native.py reaches the AST pass"
    assert found == {}, f"criterion 7: code changing the int-string limit: {found}"


def test_no_process_wide_limit_change_sees_a_set_call() -> None:
    """Power row for the source guard, both directions: the scan finds a ``set_int_max_str_digits`` call, an aliased
    import of it, a string naming the environment variable, an ``os.environ`` write of it, and the ``-X`` option in
    a subprocess argument list; it ignores ``sys.get_int_max_str_digits()`` and ``sys.flags.int_max_str_digits``
    reads, a string naming the getter, and the same words in a docstring and a comment."""
    import tempfile

    src = (
        '"""Mentions sys.set_int_max_str_digits(0) and PYTHONINTMAXSTRDIGITS in prose."""\n'
        "import os, sys\n"
        "from sys import set_int_max_str_digits as s  # set_int_max_str_digits in a comment\n"
        "sys.set_int_max_str_digits(0)\n"
        'ENV = "PYTHONINTMAXSTRDIGITS"\n'
        'os.environ["PYTHONINTMAXSTRDIGITS"] = "0"\n'
        'ARGS = [sys.executable, "-X", "int_max_str_digits=0"]\n'
        "LIMIT = sys.get_int_max_str_digits()\n"
        "FLAG = sys.flags.int_max_str_digits\n"
        'GETTER = getattr(sys, "get_int_max_str_digits")\n'
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.py"
        path.write_text(src, encoding="utf-8")
        hits = sorted(_limit_mentions(path))
    assert hits == sorted(
        [
            "attribute set_int_max_str_digits",
            "import set_int_max_str_digits",
            "string 'PYTHONINTMAXSTRDIGITS'",
            "string 'PYTHONINTMAXSTRDIGITS'",
            "string 'int_max_str_digits=0'",
        ]
    ), hits


# --- criterion 8: the result is safe to log -------------------------------------------------------------------------


def _assert_loggable(result: Any, digits: str, where: str, *, may_print: bool = False) -> None:
    try:
        json.dumps(_snapshot(result))
    except (ValueError, TypeError) as err:
        raise AssertionError(f"{where}: the GateResult snapshot is not JSON-safe: {err}") from err
    if result.surface is not None:
        assert not (bad := _beyond_double(result.surface.messages)), f"{where}: {bad}"
    if may_print:
        return  # ruling (e): the literal follows the control as written, a string id the problems may name
    run = digits.lstrip("-")
    for text in [*result.report.problems, *(str(e) for e in result.report.schema_errors)]:
        assert run not in text, f"{where}: a problem or schema error prints the literal's digits"


def _control_prints_its_digits(control_raw: str, negative: bool = False) -> bool:
    """Rulings (e) and (e')'s condition, read from BASE: on this control canvas (the path's own canvas with the
    310-digit control in place of the literal) the frozen gate turns the control into a string and a problem or
    schema-error string prints its digits."""
    ctrl = _digits(CONTROL, negative).lstrip("-")
    base = frozen.UiGate().gate(control_raw, turn_id="t", canvas_idx=0, final=False)
    texts = [*base.report.problems, *(str(e) for e in base.report.schema_errors)]
    return any(ctrl in t for t in texts)


#: Paths of criterion 8's calls: (name, the canvas from a template and a placement)
_PATHS = ("strict", "bracket", *SHAPES)


def _path_control(template: str, placement: str, path: str, negative: bool = False) -> str:
    ctrl = _digits(CONTROL, negative)
    if path == "strict":
        return _splice(template, ctrl)
    if path == "bracket":
        return _splice(_stray(template, placement), ctrl)
    return _control_shape_canvas(template, path, negative)


@pytest.mark.parametrize(("catalog", "placement"), CASES, ids=CASE_IDS)
def test_result_is_loggable(catalog: str, placement: str) -> None:
    """Criterion 8 over the adversarial canvases of each placement (strict: 4301 and 10^5 digits, both signs for the
    number role and unsigned elsewhere, the amendment's budget; bracket path; the three json_repair shapes):
    ``json.dumps`` of the snapshot succeeds at the default limit, no emitted number is beyond the float range or
    non-finite, and no problem or schema-error string carries the literal's digits. Rulings (e) and (e'): on ``id``
    and ``children``, on every path where BASE's control (the same path's canvas) becomes a string that BASE's
    problems print, the literal follows the control as written and only the problem-string bullet is lifted.
    ``surfaceId`` stays strict."""
    template = _template(catalog, placement)
    calls: list[tuple[str, str, str, bool]] = []

    def may_print(path: str, negative: bool = False) -> bool:
        return placement in ("id", "child") and _control_prints_its_digits(
            _path_control(template, placement, path, negative), negative
        )

    signs = (False, True) if placement == "number" else (False,)
    for n in (OVER, HUGE):
        for negative in signs:
            lit = _digits(n, negative)
            calls.append(
                (
                    f"strict {n} negative={negative}",
                    _splice(template, lit),
                    lit,
                    may_print("strict", negative),
                )
            )
    lit = _digits(OVER)
    calls.append(("bracket", _splice(_stray(template, placement), lit), lit, may_print("bracket")))
    for shape in SHAPES:
        raw, lit = _shape_canvas(template, shape, False)
        calls.append((shape, raw, lit, may_print(shape)))
    for where, raw, lit, may_print in calls:
        _assert_loggable(_call(raw, final=False), lit, where, may_print=may_print)


def test_ruling_e_condition_holds_exactly_where_base_prints_the_control() -> None:
    """Premise of rulings (e) and (e') (green at BASE): BASE's control prints its digits, read from the frozen gate
    on each path's own canvas, for ``id`` on the two cut shapes, ``children`` on the bracket path and all three
    json_repair shapes, and the ``updateComponents`` ``surfaceId`` on every path (in a schema error), on both
    catalogs, and nowhere else among criterion 1's placements and criterion 8's paths (so the condition cannot widen
    silently). The rulings name ``id`` and ``children`` only, so ``test_result_is_loggable`` lifts the
    problem-string bullet for those two and keeps it for ``surfaceId``."""
    got = {
        (cat, p, s)
        for cat in CATALOGS
        for p in PLACEMENTS
        for s in _PATHS
        if _control_prints_its_digits(_path_control(_template(cat, p), p, s))
    }
    want = {
        (cat, p, s)
        for cat in CATALOGS
        for p, s in (
            ("id", "cut-after"),
            ("id", "cut-inside"),
            ("child", "bracket"),
            ("child", "bare-key"),
            ("child", "cut-after"),
            ("child", "cut-inside"),
            *(("sid", s) for s in _PATHS),
        )
    }
    assert got == want, sorted(got ^ want)


# --- criterion 9: the too-wide guard (green at BASE, unmarked) ---------------------------------------------------------


def _both(raw: str, final: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    tip = _snapshot(UiGate().gate(raw, turn_id="w", canvas_idx=0, final=final))
    base = _snapshot(frozen.UiGate().gate(raw, turn_id="w", canvas_idx=0, final=final))
    return tip, base


TOO_WIDE: dict[str, tuple[str, str, str]] = {}
for _cat in CATALOGS:
    for _p in ("number", "text", "data-nested", "data-bare", "create-extra"):
        for _neg in (False, True):
            TOO_WIDE[f"{_cat}-{_p}-4300{'-signed' if _neg else ''}"] = (
                _cat,
                _p,
                _digits(DEFAULT_LIMIT, _neg),
            )
    for _p in ("number", "data-nested"):
        TOO_WIDE[f"{_cat}-{_p}-fraction"] = (_cat, _p, _digits(OVER) + ".5")
        TOO_WIDE[f"{_cat}-{_p}-exponent-signed"] = (_cat, _p, _digits(OVER, True) + "e5")
    TOO_WIDE[f"{_cat}-number-1e308"] = (_cat, "number", "1e308")
    TOO_WIDE[f"{_cat}-number-10**308"] = (_cat, "number", "1" + "0" * 308)
    TOO_WIDE[f"{_cat}-number-float-max"] = (_cat, "number", repr(sys.float_info.max))
    TOO_WIDE[f"{_cat}-data-nested-1,000"] = (_cat, "data-nested", "1,000")


@pytest.mark.parametrize("case", sorted(TOO_WIDE))
def test_too_wide_guard_matches_base(case: str) -> None:
    """Criterion 9: inputs that are not an integer literal past the limit give a ``GateResult`` field-equal to the
    frozen BASE gate (every field, wording included): a 4300-digit literal (at the limit) signed and unsigned in a
    component and in the envelope; 4301 digits with a fraction or an exponent (a float: ``inf``, refused by C10);
    ``1e308``, ``10**308`` and the largest double; json_repair's ``1,000`` in an object (a string, as at BASE)."""
    catalog, placement, literal = TOO_WIDE[case]
    raw = _splice(_template(catalog, placement), literal)
    tip, base = _both(raw)
    assert tip == base, f"criterion 9 {case}: {_first_difference(tip, base)}"


def test_too_wide_premise_json_repair_keeps_1000_as_a_string() -> None:
    """Premise of the ``1,000`` rows: at BASE that canvas goes through json_repair, which returns ``"1,000"``."""
    raw = _splice(_template("basic", "data-nested"), "1,000")
    result = frozen.UiGate().gate(raw, turn_id="w", canvas_idx=0, final=True)
    assert result.surface is not None
    assert "1,000" in [v for _, v in _leaves(list(result.surface.messages))]


@pytest.mark.parametrize("n", [OVER, HUGE], ids=["4301", "1e5"])
@pytest.mark.parametrize("catalog", list(CATALOGS))
def test_too_wide_guard_a_digit_run_inside_a_string_renders_verbatim(catalog: str, n: int) -> None:
    """Criterion 9: a run of 4301 or more digits inside a JSON string (``"text": "777…"``) is text, not a literal:
    it renders verbatim, complete, and the whole result equals BASE's."""
    run = _digits(n)
    raw = _splice(_template(catalog, "text"), json.dumps(run))
    tip, base = _both(raw)
    assert tip == base, f"criterion 9: {_first_difference(tip, base)}"
    result = UiGate().gate(raw, turn_id="w", canvas_idx=0, final=True)
    assert result.complete and _emitted(result)["x"]["text"] == run


#: Ruling (g): a 4301-digit run right after an escape inside a JSON string belongs to the string, never to a number.
#: ``\u`` takes the run's first four digits as its code point; ``7`` and ``—`` (an em dash, reviewer A's
#: probe) are complete escapes followed by the run; ``\x37`` is not a JSON escape, so only json_repair reads it.
#: ``\ua``, ``\uab`` and ``\u00a`` start the run 1, 2 or 3 hex digits into the escape, so the escape's remaining
#: digits come from the run (reviewer A, round 2: one row per arm of the ``\u`` lookbehind).
ESCAPES = {
    "u": "\\u",
    "u0037": "\\u0037",
    "u2014": "\\u2014",
    "x37": "\\x37",
    "ua": "\\ua",
    "uab": "\\uab",
    "u00a": "\\u00a",
}


@pytest.mark.parametrize("escape", sorted(ESCAPES))
@pytest.mark.parametrize("catalog", list(CATALOGS))
def test_too_wide_guard_digits_after_an_escape_stay_in_the_string(
    catalog: str, escape: str
) -> None:
    """Criterion 9, ruling (g): ``"text": "<escape>777…"`` (4301 sevens) with a bare key before it (json_repair's
    path) gives a ``GateResult`` field-equal to the frozen BASE gate (every field, wording included). A run finder
    that reads digits after an escape as a number loses ``x`` where BASE keeps it."""
    raw = _splice(_bare(_template(catalog, "text")), f'"{ESCAPES[escape]}{_digits(OVER)}"')
    tip, base = _both(raw)
    assert tip == base, f"criterion 9, escape {escape}: {_first_difference(tip, base)}"


@pytest.mark.parametrize("placement", ["data-nested", "number"])
def test_too_wide_guard_a_leading_fraction_is_a_float(placement: str) -> None:
    """Criterion 9, ruling (g): ``0.777…`` (4301 sevens after the point) on the json_repair path (a bare key) is a
    float, not an over-limit integer: field-equal to the frozen BASE gate. Pins the ``.`` in the run finder's
    lookbehind: without it the digits after the point read as an integer literal."""
    raw = _splice(_bare(_template("basic", placement)), "0." + _digits(OVER))
    tip, base = _both(raw)
    assert tip == base, f"criterion 9, 0.<4301 digits>: {_first_difference(tip, base)}"


def test_too_wide_guard_a_negative_exponent_is_a_float() -> None:
    """Criterion 9 (reviewer B, round 1): ``777…e-4000`` (4301 sevens) in ``updateDataModel.value.level`` on the
    json_repair path (a bare key) is a float (about 7.78e300), not an over-limit integer: field-equal to the frozen
    BASE gate. Pins the run finder's lookahead: without ``(?![eE][0-9-])`` the digits before ``e-`` read as an
    integer literal and the value changes."""
    raw = _splice(_bare(_template("basic", "data-nested")), _digits(OVER) + "e-4000")
    tip, base = _both(raw)
    assert tip == base, f"criterion 9, <4301 digits>e-4000: {_first_difference(tip, base)}"
    level = [
        m["updateDataModel"]["value"]["level"]
        for m in (base["surface"] or {}).get("messages", [])
        if isinstance(m, Mapping) and "updateDataModel" in m
    ]
    assert len(level) == 1 and isinstance(level[0], float) and 7.7e300 < level[0] < 7.8e300, (
        f"premise: BASE reads the value as a float near 7.78e300, got {_safe_repr(level)}"
    )


@pytest.mark.parametrize("negative", [False, True], ids=["unsigned", "signed"])
def test_an_underscore_joined_run_is_one_literal(negative: bool) -> None:
    """Criterion 5 on json_repair's ``_`` (reviewer B, round 1): ``7_777…`` (``7_`` then 4301 sevens) after a bare
    key in the nested data model is one over-limit literal to json_repair (its ``parse_number`` drops ``_``), so the
    verdict equals the control's (``7_`` then 310 sevens, the same canvas) and no ``str`` of 4301+ digits is
    emitted. A run finder that stops at ``_`` misses the literal or splits it."""
    bare = _bare(_template("basic", "data-nested"))
    sign = "-" if negative else ""
    result = _call(_splice(bare, sign + "7_" + _digits(OVER)), final=True)
    control = UiGate().gate(
        _splice(bare, sign + "7_" + _digits(CONTROL)), turn_id="t", canvas_idx=0, final=True
    )
    tip, want = _verdict(_snapshot(result)), _verdict(_snapshot(control))
    assert tip == want, f"7_<4301 digits>: {_first_difference(tip, want)}"
    if result.surface is not None:
        assert not (long := _long_digit_strings(result.surface.messages)), long


#: The first stand-in shape a run finder might pick: ``9`` followed by 309 zeros (``9e309``, past the largest double).
STAND_IN_SHAPED = "9" + "0" * 309


def test_a_stand_in_shaped_number_in_the_canvas_is_left_alone() -> None:
    """Ruling (g), collision: a canvas holding an over-limit literal (the nested data model, a bare key: json_repair's
    path) and the number ``9`` followed by 309 zeros written in ``t``'s text (``"Kept 9000…"``). Its verdict equals
    the control's (the same canvas with the 310-digit control in place of the literal, where no stand-in is needed),
    so ``t`` keeps the model's own text, never the literal's digits. (Written as a bare number in ``Text.text``
    instead, ``9e309`` gives no surface with or without the literal, so that shape cannot tell a stand-in collision
    apart; measured at 9c97d176.)"""
    bare = _bare(_template("basic", "data-nested")).replace(
        '"Kept"', json.dumps("Kept " + STAND_IN_SHAPED)
    )
    assert bare.count(STAND_IN_SHAPED) == 1, (
        "premise: the stand-in-shaped number is in the canvas once"
    )
    result = _call(_splice(bare, _digits(OVER)), final=True)
    control = UiGate().gate(_splice(bare, _digits(CONTROL)), turn_id="t", canvas_idx=0, final=True)
    tip, want = _verdict(_snapshot(result)), _verdict(_snapshot(control))
    assert tip == want, f"collision: {_first_difference(tip, want)}"
    assert not _sevens(result.surface.messages if result.surface else []), (
        "the literal's digits reached the surface"
    )


# --- the frozen oracle is BASE, verbatim (both parts) ----------------------------------------------------------------


def _base_function_source(module_text: str, name: str) -> str:
    tree = ast.parse(module_text)
    lines = module_text.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            assert node.end_lineno is not None
            return "".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{name} is not a top-level function of the BASE module")


def test_frozen_gate_is_base_verbatim() -> None:
    """The oracle is what it claims (implementation note 6). Part one, below its marker up to the second marker, is
    BASE's ``ui_gate.py`` followed by the two blank lines ``ruff format`` puts before a top-level comment; part two is
    BASE's ``parse_concatenated_json``. Both match a sha256 literal, the frozen module's global
    ``parse_concatenated_json`` is its own (not the live one), and, when 7646b8b6 is in the object store, both parts
    equal ``git show`` of BASE (skips naming the commit in an archive copy)."""
    text = FROZEN_FILE.read_text(encoding="utf-8")
    assert text.count(GATE_MARKER) == 1 and text.count(PARSE_MARKER) == 1, (
        "a marker line is missing"
    )
    gate_part, parse_part = text.split(GATE_MARKER, 1)[1].split(PARSE_MARKER, 1)
    assert gate_part.endswith("\n\n\n"), (
        "part one must end with ui_gate.py's newline and two blank lines"
    )
    gate_body = gate_part[:-2]
    assert hashlib.sha256(gate_body.encode("utf-8")).hexdigest() == BASE_GATE_SHA256, (
        "tests/unit/_frozen_ui_gate_t0185.py part one was edited: it must stay BASE's ui_gate.py verbatim"
    )
    assert hashlib.sha256(parse_part.encode("utf-8")).hexdigest() == BASE_PARSE_SHA256, (
        "part two must stay BASE's parse_concatenated_json verbatim"
    )
    assert frozen.parse_concatenated_json is not live_parse_concatenated_json
    assert frozen.parse_concatenated_json.__module__ == frozen.__name__
    gr.require_commits(BASE)
    shown = gr._git("show", f"{BASE}:local_astra/conductor/ui_gate.py")
    assert shown.returncode == 0, shown.stderr
    assert shown.stdout == gate_body
    schema = gr._git("show", f"{BASE}:local_astra/ui/a2ui_schema.py")
    assert schema.returncode == 0, schema.stderr
    assert _base_function_source(schema.stdout, "parse_concatenated_json") == parse_part


# --- criterion 10(a): the census and the Q12 sample give BASE's results (green at BASE, unmarked) -----------------------


def _run_both(
    calls: list[tuple[str, str, int, bool]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """``(tip, base)`` snapshots of one fresh live gate and one fresh frozen gate over the same calls, in order."""
    tip, base = UiGate(), frozen.UiGate()
    return [
        (
            _snapshot(tip.gate(raw, turn_id=turn, canvas_idx=idx, final=final)),
            _snapshot(base.gate(raw, turn_id=turn, canvas_idx=idx, final=final)),
        )
        for raw, turn, idx, final in calls
    ]


def _census_texts(catalog_id: str) -> list[tuple[dict[str, Any], list[str]]]:
    rows = [row for row in _jsonl(CENSUS) if "header" not in row]
    assert len(rows) == 13, "premise: the census holds 13 surfaces"
    return [
        (
            row,
            [
                _compact(canvas).replace(BASIC_V091, catalog_id).replace(BASIC_V09, catalog_id)
                for canvas in row["canvases"]
            ],
        )
        for row in rows
    ]


@pytest.mark.parametrize("catalog_id", [BASIC_V091, BASIC_V09], ids=["v0_9_1", "v0_9"])
def test_census_gives_base_results(catalog_id: str) -> None:
    """Criterion 10(a) on the T-0164 census, in T-0184 C1's harness shape: one fresh live gate and one fresh frozen
    gate per surface; the ``v0_9_1`` spelling replays every canvas with ``final=False``, the ``v0_9`` spelling each
    surface's last canvas with ``final=True``. Every ``GateResult`` is field-equal."""
    surfaces = 0
    for row, texts in _census_texts(catalog_id):
        turn = row["surface_id"].removeprefix("turn-")
        if catalog_id == BASIC_V091:
            calls = [(t, turn, i, False) for i, t in enumerate(texts)]
        else:
            calls = [(texts[-1], turn, len(texts) - 1, True)]
        for k, (tip, base) in enumerate(_run_both(calls)):
            assert tip == base, (
                f"{row['key']} call {k} ({catalog_id}): {_first_difference(tip, base)}"
            )
            assert _typed(tip) == _typed(base), f"{row['key']} call {k}: a value's type differs"
            surfaces += base["surface"] is not None
    assert surfaces > 0, "premise: the census yields surfaces at BASE"


def _q12_trials() -> list[dict[str, Any]]:
    rows = _jsonl(Q12)
    header, trials = rows[0]["header"], rows[1:]
    assert len(trials) == header["trials"] == 17 and header["ui_commits"] == 30
    return trials


def test_q12_sample_gives_base_results() -> None:
    """Criterion 10(a) on the stored Q12 sample (17 trials, 30 ui commits): every ui commit in order through a fresh
    live gate and a fresh frozen gate per trial (``final=False``, turn id = request id, the commit's canvas_idx).
    Field-equal throughout; the MANIFEST's counts (20 surfaces, 14 complete) are the premise."""
    shown = complete = 0
    for trial in _q12_trials():
        rid = trial["request_id"]
        calls = [(c["text"], rid, int(c["canvas_idx"] or 0), False) for c in trial["ui_commits"]]
        for k, (tip, base) in enumerate(_run_both(calls)):
            assert tip == base, f"{rid} call {k}: {_first_difference(tip, base)}"
            assert _typed(tip) == _typed(base), f"{rid} call {k}: a value's type differs"
            shown += base["surface"] is not None
            complete += bool(base["complete"])
    assert (shown, complete) == (20, 14), f"premise (MANIFEST): got {shown}, {complete}"


# --- criterion 11: A2UIValidator.validate_text -------------------------------------------------------------------------


_VALIDATORS: dict[str, A2UIValidator] = {}


def _validator(catalog: str) -> A2UIValidator:
    if catalog not in _VALIDATORS:
        _VALIDATORS[catalog] = (
            A2UIValidator()
            if catalog == "basic"
            else A2UIValidator(catalog=load_cards_v2_catalog())
        )
    return _VALIDATORS[catalog]


def _base_validate_text(
    validator: A2UIValidator, text: str, *, final: bool = True
) -> SurfaceReport:
    """BASE's ``A2UIValidator.validate_text`` (a2ui_schema.py:533-538 at 7646b8b6), restated over the frozen BASE
    ``parse_concatenated_json`` because the live one may change under T-0185: a ``JSONDecodeError`` gives
    ``SurfaceReport(ok=False, problems=(f"invalid json: {e.msg} at {e.pos}",))``, else ``validate_messages``."""
    try:
        msgs = frozen.parse_concatenated_json(text)
    except json.JSONDecodeError as e:
        return SurfaceReport(ok=False, problems=(f"invalid json: {e.msg} at {e.pos}",))
    return validator.validate_messages(msgs, final=final)


@pytest.mark.parametrize(("catalog", "placement"), CASES, ids=CASE_IDS)
def test_validate_text_never_raises(catalog: str, placement: str) -> None:
    """Criterion 11, the over-limit half: criterion 1's canvases (4301 and 10^5 digits; unsigned and signed for the
    number role, unsigned elsewhere, the amendment's budget) through ``validate_text`` of the basic and the v2
    validator never raise, leave the int-string limit alone, and return a ``SurfaceReport`` whose verdict (``ok``,
    the T-0184 ``_verdict`` reduction) equals BASE's ``validate_text`` on the 310-digit control of the same sign
    (ruling (f): C10 parity, not a refusal; BASE accepts the control in a number role and the data model)."""
    template = _template(catalog, placement)
    for n in (OVER, HUGE):
        for negative in (False, True) if placement == "number" else (False,):
            raw = _splice(template, _digits(n, negative))
            control_raw = _splice(template, _digits(CONTROL, negative))
            for which in CATALOGS:
                before = sys.get_int_max_str_digits()
                try:
                    report = _validator(which).validate_text(raw)
                except Exception as err:
                    raise AssertionError(
                        f"validate_text ({which}) raised {type(err).__name__}: {str(err)[:120]}"
                    ) from None
                assert sys.get_int_max_str_digits() == before
                assert isinstance(report, SurfaceReport), type(report).__name__
                want = _base_validate_text(_validator(which), control_raw).ok
                assert report.ok == want, (
                    f"criterion 11 ({which} validator, {n} digits, negative={negative}): ok={report.ok}, "
                    f"the control's ok={want}"
                )


def test_validate_text_gives_base_results_on_the_differential_fixtures() -> None:
    """Criterion 11, the BASE-equality half (green at BASE, unmarked): every census canvas (both basic spellings) and
    every Q12 ui commit through ``validate_text`` of the basic validator equals BASE's, field for field."""
    texts = [t for cid in (BASIC_V091, BASIC_V09) for _, ts in _census_texts(cid) for t in ts]
    texts += [c["text"] for trial in _q12_trials() for c in trial["ui_commits"]]
    assert len(texts) > 30
    invalid_json = 0
    v = _validator(
        "basic"
    )  # the fixtures' own catalog (the v2 validator dropped for the budget, criterion 12)
    for k, text in enumerate(texts):
        got = dataclasses.asdict(v.validate_text(text))
        want = dataclasses.asdict(_base_validate_text(v, text))
        assert got == want, f"text {k}: {_safe_repr(got)} != {_safe_repr(want)}"
        assert _typed(got) == _typed(want), f"text {k}: a value's type differs"
        invalid_json += any(p.startswith("invalid json") for p in want["problems"])
    assert invalid_json > 0, "premise: some fixture texts take the invalid-json branch"
