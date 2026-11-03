"""The structural (bracket) repair: a pure, linear, string-aware pass. T-0108 criteria 2-6.

``local_astra/ui/bracket_repair.py`` is a new module (imported by path, not exported from
``local_astra/ui/__init__.py``). The task file says only "a pure function ``text -> (text,
edits)``"; it does not name the function, so this module pins the name :data:`REPAIR_NAME`
(reported to the reviewer as an ambiguity the test-author resolved). The module does not exist
at the branch base and no stub may be written by the test-author, so :func:`repair` turns
exactly that one ``ModuleNotFoundError`` into ``NotImplementedError`` -- the pending outcome the
conftest tolerates. Any *other* import problem, and a module that exists without the name,
surface as real errors.

Oracles, never the implementation:

- the golden texts of criterion 4 are hand-written edits of the real canvases in
  ``tests/fixtures/a2ui/t0108_canvases.jsonl`` (each edit is a literal ``(before, after)`` pair
  whose occurrence count is asserted, so the expected text is reviewable without running
  anything);
- N1 is a character filter, N2 a naive string-lexeme scanner written here, N3 the set of strings
  the generator itself put into the document, N4 the function applied to its own output;
- the generated corruptions are the observed operators (task *Goal*), applied at structural
  positions the generator knows, so "undone" means "the document the generator started from".

Regime: CPU only, no service. Deterministic except the CPU-time row of criterion 5 (T-0129):
``time.thread_time``, gc off, sizes 4096/8192/16384 interleaved, minimum of 5, on this box,
serial or under the Stop hook's ``-n 12``; not a bench B-row. The work row counts traced line
events of the repair module (exact, load-independent). Both replace a ``perf_counter`` median of
5, which was red in 177 of 360 trials under load 8-32 (T-0129 *Regime*). Hypothesis
``max_examples`` <= 200, ``deadline=None``.
"""

from __future__ import annotations

import functools
import gc
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from local_astra.conductor.ui_gate import UiGate
from local_astra.conductor.ui_instructions import UI_EXAMPLE
from local_astra.ui import parse_concatenated_json
from tests.unit._work_count import WorkCapExceeded, line_events

REPO = Path(__file__).resolve().parents[2]
CANVASES = REPO / "tests" / "fixtures" / "a2ui" / "t0108_canvases.jsonl"
QUIZ = REPO / "tests" / "fixtures" / "a2ui" / "quiz_example.jsonl"

#: T-0108 *Owned files*: the module path is the task's; the function name is this module's choice
#: (the task file does not name it).
REPAIR_MODULE = "local_astra.ui.bracket_repair"
REPAIR_NAME = "repair_structure"

CATALOG = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
HEADER = '{"version":"v0.9.1","createSurface":{"surfaceId":"s","catalogId":"' + CATALOG + '"}}\n'
UC_OPEN = '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":['
UC_CLOSE = "]}}"

#: criterion 3 (N1): the structural alphabet A, and JSON whitespace
ALPHABET_A = frozenset('{}[],"')
JSON_WS = frozenset(" \t\n\r")


# --- access to the function under test ---------------------------------------------------------


def repair_fn() -> Callable[[str], tuple[str, int]]:
    try:
        module = importlib.import_module(REPAIR_MODULE)
    except ModuleNotFoundError as missing:
        if missing.name == REPAIR_MODULE:
            raise NotImplementedError(
                f"T-0108: {REPAIR_MODULE} does not exist yet (criterion 1)"
            ) from missing
        raise
    fn: Callable[[str], tuple[str, int]] = getattr(module, REPAIR_NAME)
    return fn


def repair(text: str) -> tuple[str, int]:
    result = repair_fn()(text)
    assert isinstance(result, tuple) and len(result) == 2, (
        f"criterion 1: the repair is `text -> (text, edits)`; it returned {type(result).__name__}"
    )
    out, edits = result
    assert isinstance(out, str) and isinstance(edits, int) and not isinstance(edits, bool), (
        f"criterion 1: (str, int); got ({type(out).__name__}, {type(edits).__name__})"
    )
    assert edits >= 0, f"an edit count is never negative; got {edits}"
    return out, edits


# --- independent oracles -----------------------------------------------------------------------


def content(text: str) -> str:
    """N1: ``text`` with every character of A and every JSON whitespace character removed."""
    return "".join(c for c in text if c not in ALPHABET_A and c not in JSON_WS)


def string_lexemes(text: str) -> list[str]:
    """Every string lexeme, as a scanner reads them that treats each unescaped ``"`` outside a
    string as an opener (an unterminated last string counts, to the end of the text)."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != '"':
            i += 1
            continue
        j = i + 1
        while j < n and text[j] != '"':
            j += 2 if text[j] == "\\" else 1
        out.append(text[i : j + 1])
        i = j + 1
    return out


def without_ws_outside_strings(text: str) -> str:
    """``text`` with JSON whitespace outside string lexemes removed (string lexemes verbatim)."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
            continue
        if ch not in JSON_WS:
            out.append(ch)
        i += 1
    return "".join(out)


def assert_n1_n2(raw: str, out: str, where: str) -> None:
    assert content(raw) == content(out), (
        f'criterion 3 (N1), {where}: removing A = {{ }} [ ] , " and JSON whitespace from raw and '
        f"from the repair must leave identical sequences; raw keeps {content(raw)[:120]!r}, the "
        f"repair keeps {content(out)[:120]!r}"
    )
    pos = 0
    for lexeme in string_lexemes(out):
        found = raw.find(lexeme, pos)
        assert found >= 0, (
            f"criterion 3 (N2), {where}: every string lexeme of the repair occurs verbatim in raw, "
            f"in order; {lexeme[:80]!r} does not (from offset {pos})"
        )
        pos = found + 1


def decoded_strings(value: Any) -> Iterator[str]:
    """Every string leaf and every key, recursively."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from decoded_strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from decoded_strings(item)


# --- fixtures ----------------------------------------------------------------------------------


def fixture_rows() -> list[dict[str, Any]]:
    return [json.loads(x) for x in CANVASES.read_text(encoding="utf-8").splitlines() if x.strip()]


def row(request_id: str, ui_index: int) -> dict[str, Any]:
    hits = [
        r for r in fixture_rows() if r["request_id"] == request_id and r["ui_index"] == ui_index
    ]
    assert len(hits) == 1, f"fixture: exactly one row for {request_id}#{ui_index}; {len(hits)}"
    return hits[0]


def edited(text: str, edits: list[tuple[str, str, int]]) -> str:
    """``text`` with each hand-written ``(before, after)`` applied; ``count`` occurrences each."""
    for before, after, count in edits:
        assert text.count(before) == count, (
            f"golden setup: {before!r} occurs {text.count(before)} times, the edit was written for "
            f"{count}"
        )
        text = text.replace(before, after)
    return text


# --- criterion 4: the hand-written golden texts -------------------------------------------------

#: Hand-written from the rules (task *Implementation notes*): delete the duplicated ``{"``; turn
#: ``"x" "{"`` into ``"x"},{``; drop a mismatched closer when the expected one follows, otherwise
#: insert the expected closer(s); drop a closer with no opener; delete a stray ``"`` after a
#: closer. ``sites`` counts the separate places in raw that must change: a lower bound on
#: ``edits`` under any counting unit (characters or operations). At end of text nothing is closed.
GOLDEN: dict[tuple[str, int], dict[str, Any]] = {
    # duplicated opener `{"{"id":`
    ("t0103-r6-omelette_steps-A0", 1): {
        "edits": [('},{"{"id":"h"', '},{"id":"h"', 1)],
        "sites": 1,
    },
    ("t0103-r36-flight_booking-C3", 1): {
        "edits": [
            ('{"{"id":"passengers"', '{"id":"passengers"', 1),
            ('{"{"id":"btn"', '{"id":"btn"', 1),
        ],
        "sites": 2,
    },
    # missing separator `"From:" "{"id"` (with duplicated openers, wrong closers; k0 is cut)
    ("t0103-r0-flight_booking-A0", 0): {
        "edits": [
            ('},{"{"id":"h"', '},{"id":"h"', 1),
            (
                '{"{"id":"t1","component":"Text","text":"From:" "{"id":"f1"',
                '{"id":"t1","component":"Text","text":"From:"},{"id":"f1"',
                1,
            ),
            ('"City or airport"]},{"{"id":"r2"', '"City or airport"},{"id":"r2"', 1),
            (
                '{"{"id":"t2","component":"Text","text":"To:" "{"id":"f2"',
                '{"id":"t2","component":"Text","text":"To:"},{"id":"f2"',
                1,
            ),
        ],
        "sites": 7,
        "modulo_whitespace": True,
    },
    ("t0103-r0-flight_booking-A0", 1): {
        "edits": [
            ('},{"{"id":"h"', '},{"id":"h"', 1),
            (
                '{"{"id":"t1","component":"Text","text":"From:" "{"id":"f1"',
                '{"id":"t1","component":"Text","text":"From:"},{"id":"f1"',
                1,
            ),
            ('"City or airport"]},{"{"id":"r2"', '"City or airport"},{"id":"r2"', 1),
            (
                '{"{"id":"t2","component":"Text","text":"To:" "{"id":"f2"',
                '{"id":"t2","component":"Text","text":"To:"},{"id":"f2"',
                1,
            ),
            ('"City or airport"]},{"{"id":"r3"', '"City or airport"},{"id":"r3"', 1),
            (
                '{"{"id":"t3","component":"Text","text":"Date:" "{"id":"d1",'
                '"component":"DateTimeInput"]},{"{"id":"btn"',
                '{"id":"t3","component":"Text","text":"Date:"},{"id":"d1",'
                '"component":"DateTimeInput"},{"id":"btn"',
                1,
            ),
            ('"Search Flights"}]}}}', '"Search Flights"}]}}', 1),
        ],
        "sites": 14,
        "modulo_whitespace": True,
    },
    # wrong closer, "drop a spurious closer when the expected one follows"
    ("t0103-r23-flight_booking-C3", 1): {
        "edits": [('"placeholder":"To"]},', '"placeholder":"To"},', 1)],
        "sites": 1,
    },
    ("t0103-r9-flight_booking-A0", 1): {
        "edits": [('"To (City or Airport)"]},', '"To (City or Airport)"},', 1)],
        "sites": 1,
    },
    # wrong closer, "insert the expected closer" (then a closer with no opener is dropped)
    ("t0103-r36-flight_booking_with_memory-C3", 1): {
        "edits": [('"Search Flights"}}}}', '"Search Flights"}]}}', 1)],
        "sites": 2,
    },
    ("t0103-r37-omelette_steps-C3", 1): {
        "edits": [('"}]}}}}', '"}]}]}}', 1)],
        "sites": 2,
    },
    ("t0098c-r2-omelette_steps-C4", 1): {
        "edits": [('"}]}}}}', '"}]}]}}', 1)],
        "sites": 2,
    },
    # stray quote after a closer `]"]`
    ("t0103-r14-omelette_steps-A0", 1): {
        "edits": [('["h","steps"]"]},', '["h","steps"]},', 1)],
        "sites": 1,
    },
    ("t0103-r17-flight_booking_with_memory-A0", 1): {
        "edits": [('["Economy","Business"]"]},', '["Economy","Business"]},', 1)],
        "sites": 1,
    },
}

REPAIRED_CLASSES = {
    "duplicated_opener",
    "missing_separator",
    "wrong_closer_drop",
    "wrong_closer_insert",
    "stray_quote",
}


def golden_expected(key: tuple[str, int]) -> tuple[dict[str, Any], str]:
    r = row(*key)
    return r, edited(r["text"], GOLDEN[key]["edits"])


def test_golden_table_covers_criterion_13_and_its_texts_are_sound() -> None:
    """Guard (passes today): the hand-written goldens are well-formed and mean what they claim.

    Criterion 13 (i) asks for >= 2 canvases per repaired class; every golden of a canvas that is
    not cut short parses strictly and, through today's gate, yields a surface -- so a golden that
    is itself broken cannot make criterion 4's row pass or fail for the wrong reason.
    """
    rows = fixture_rows()
    per_class: dict[str, int] = {}
    for r in rows:
        per_class[r["class"]] = per_class.get(r["class"], 0) + 1
    for cls in REPAIRED_CLASSES:
        assert per_class.get(cls, 0) >= 2, f"criterion 13 (i): >= 2 canvases of {cls}; {per_class}"
    for cls in ("duplicate_id", "missing_component_key", "inline_children", "header_cut"):
        assert per_class.get(cls, 0) >= 1, f"criterion 13 (i): >= 1 canvas of {cls}; {per_class}"
    repaired_rows = {(r["request_id"], r["ui_index"]) for r in rows if r["repaired_class"]}
    assert repaired_rows == set(GOLDEN), (
        f"every repaired-class canvas has a golden; missing {sorted(repaired_rows - set(GOLDEN))}"
    )
    for key in GOLDEN:
        r, expected = golden_expected(key)
        assert content(expected) == content(r["text"]), f"golden {key} changes content (N1)"
        if key == ("t0103-r0-flight_booking-A0", 0):
            continue  # cut inside a string: the golden stays cut, json_repair closes it
        parse_concatenated_json(expected)
        result = UiGate().gate(expected, turn_id="golden", canvas_idx=1, final=False)
        assert result.surface is not None, f"golden {key} must yield a surface; {result.report}"


@pytest.mark.parametrize("key", sorted(GOLDEN), ids=lambda k: f"{k[0]}#{k[1]}")
def test_observed_classes_golden(key: tuple[str, int]) -> None:
    """Criterion 4: each repaired class gives the hand-written text, then yields a surface."""
    r, expected = golden_expected(key)
    spec = GOLDEN[key]
    out, edits = repair(r["text"])
    if spec.get("modulo_whitespace"):
        # the missing separator `"From:" "{"id"` has a space beside the edit, and where the
        # inserted `},` goes relative to it is not fixed by the task; N1 allows either
        assert without_ws_outside_strings(out) == without_ws_outside_strings(expected), (
            f"criterion 4 ({r['class']}), {key}: the repair differs from the hand-written text "
            f"outside whitespace.\n got: {out!r}\nwant: {expected!r}"
        )
    else:
        assert out == expected, (
            f"criterion 4 ({r['class']}), {key}: the repair differs from the hand-written text."
            f"\n got: {out!r}\nwant: {expected!r}"
        )
    assert edits >= spec["sites"], (
        f"criterion 1: edits counts structural edits; {spec['sites']} places change, edits={edits}"
    )
    result = UiGate().gate(
        r["text"], turn_id=r["request_id"], canvas_idx=r["canvas_idx"], final=False
    )
    assert result.surface is not None, (
        f"criterion 4 ({r['class']}), {key}: after the repair the canvas yields a surface; the "
        f"gate said {result.report.problems or result.report.schema_errors[:3]}"
    )


# --- criterion 2: valid input is untouched ------------------------------------------------------

#: Valid concatenations that look like the corruptions: the repair must not read them as such.
TRICKY_VALID = (
    "",
    "   \n",
    '"a" "{"',  # two top-level strings: the missing-separator shape, valid
    '"From:" "{"\n"id"',
    '["a"]"]"',  # an array then the string "]": the stray-quote shape, valid
    '{"{":"{"}',  # an object whose key is "{": the duplicated-opener shape, valid
    '{"{":{"{":"id"}}',
    '{"a":"]"}{"b":"}"}',
    '[{"label":"1","value":"1"},{"label":"2"}]',
    '{"t":"a\\\\"}',  # escape edge: a, backslash
    '{"t":"a\\\\\\"b"}',  # escape edge: a, backslash, quote, b
    "{} {}\x0c[] []",  # str.isspace() separators parse_concatenated_json skips
    "1 2 true null",
    "12",
)

text_pool = st.sampled_from(
    [
        "{",
        "}",
        "[",
        "]",
        '"',
        "\\",
        '{"{"id":',
        '\\"',
        "héllo wörld",
        "日本語のテキスト",
        'a]},{"id":"x',
        '}"]',
        'x" "{"id',
        "From:",
        "City or airport",
        "",
        " ",
    ]
)
texts = st.one_of(
    text_pool, st.text(alphabet=st.sampled_from(list('ab{}[]",:\\ é日\n')), max_size=8)
)


@st.composite
def a2ui_docs(draw: st.DrawFn) -> dict[str, Any]:
    """An A2UI canvas: createSurface, then a root Column over 1-5 Text components."""
    n = draw(st.integers(1, 5))
    ids = draw(st.lists(texts.filter(lambda s: s != "root"), min_size=n, max_size=n, unique=True))
    values = draw(st.lists(texts, min_size=n, max_size=n))
    components: list[dict[str, Any]] = [{"id": "root", "component": "Column", "children": ids}]
    components += [
        {"id": i, "component": "Text", "text": v} for i, v in zip(ids, values, strict=True)
    ]
    return {"components": components, "ascii": draw(st.booleans())}


def serialise(component: dict[str, Any], ascii_only: bool) -> str:
    return json.dumps(component, ensure_ascii=ascii_only, separators=(",", ":"))


def canvas_text(doc: dict[str, Any]) -> str:
    parts = [serialise(c, doc["ascii"]) for c in doc["components"]]
    return HEADER + UC_OPEN + ",".join(parts) + UC_CLOSE


json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-9, 9) | texts,
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(texts, inner, max_size=3),
    max_leaves=8,
)
separators = st.sampled_from(["", " ", "\n", "\t", "\r\n", "\x0c", " ", " ", "\n\n  "])


@st.composite
def valid_texts(draw: st.DrawFn) -> str:
    if draw(st.booleans()):
        doc = draw(a2ui_docs())
        values: list[Any] = [
            json.loads(HEADER),
            json.loads(
                UC_OPEN + ",".join(serialise(c, True) for c in doc["components"]) + UC_CLOSE
            ),
        ]
    else:
        values = draw(st.lists(json_values, min_size=1, max_size=3))
    indent = draw(st.sampled_from([None, 0, 1, 2]))
    seps = draw(st.sampled_from([(",", ":"), (", ", ": "), (",", ": ")]))
    ascii_only = draw(st.booleans())
    out = draw(separators)
    for value in values:
        out += json.dumps(value, indent=indent, separators=seps, ensure_ascii=ascii_only)
        out += draw(separators)
    return out


@pytest.mark.parametrize("text", TRICKY_VALID, ids=range(len(TRICKY_VALID)))
def test_tricky_valid_text_is_byte_identical(text: str) -> None:
    """Criterion 2 (negative; T-0010 A5): strict-valid text that *looks* like a corruption."""
    parse_concatenated_json(text)  # precondition: valid
    assert repair(text) == (text, 0), (
        f"criterion 2: text parse_concatenated_json accepts comes back byte-identical with 0 "
        f"edits; {text!r} -> {repair(text)!r}"
    )


def test_valid_json_is_byte_identical_on_the_examples(quiz_example_text: str) -> None:
    """Criterion 2: ``UI_EXAMPLE``, the quiz fixture and every real strict-valid fixture canvas."""
    samples = {"UI_EXAMPLE": UI_EXAMPLE, "quiz_example.jsonl": quiz_example_text}
    for r in fixture_rows():
        try:
            parse_concatenated_json(r["text"])
        except json.JSONDecodeError:
            continue
        samples[f"{r['request_id']}#{r['ui_index']}"] = r["text"]
    assert len(samples) >= 4, f"setup: the fixture holds strict-valid canvases; {sorted(samples)}"
    for name, text in samples.items():
        assert repair(text) == (text, 0), f"criterion 2: {name} must come back byte-identical"


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(valid_texts())
def test_valid_json_is_byte_identical(text: str) -> None:
    """Criterion 2 over generated valid text: A2UI canvases and arbitrary concatenated JSON with
    every separator ``parse_concatenated_json`` skips (``str.isspace``, not only JSON whitespace)."""
    try:
        parse_concatenated_json(text)
    except json.JSONDecodeError:
        assume(False)
    assert repair(text) == (text, 0), f"criterion 2: {text!r} must come back byte-identical"


# --- criterion 3: N1-N4 over documents generated from S and corrupted ---------------------------

#: the observed corruption operators (task *Goal* and the fixture), keyed by where they apply
BOUNDARY_OPS = ("duplicated_opener", "missing_separator", "spurious_closer", "stray_quote")


@st.composite
def corrupted_docs(draw: st.DrawFn) -> tuple[str, str, set[str]]:
    """``(original, corrupted, S)``: a canvas, the same canvas after 1-3 observed corruptions, and
    every string (key or value) the original holds.

    Boundary ``j`` sits between component ``j-1`` and component ``j``:

    - ``duplicated_opener``: ``,{"id"`` -> ``,{"{"id"`` (any boundary);
    - ``missing_separator``: ``"x"},{"id"`` -> ``"x" "{"id"`` (after a Text, whose last value is a string);
    - ``spurious_closer``: ``"x"},{`` -> ``"x"]},{`` (after a Text; "drop when the expected follows");
    - ``stray_quote``: ``]},{`` -> ``]"]},{`` (after the Column, whose last value is a list);
    - at the end, ``"}]}}`` -> ``"}}}}`` (the components ``]`` written as ``}``: "insert the expected").
    """
    doc = draw(a2ui_docs())
    parts = [serialise(c, doc["ascii"]) for c in doc["components"]]
    original = HEADER + UC_OPEN + ",".join(parts) + UC_CLOSE
    ops: list[str | None] = []
    for j in range(1, len(parts)):
        prev_is_text = doc["components"][j - 1]["component"] == "Text"
        allowed = (
            ["duplicated_opener", "missing_separator", "spurious_closer"]
            if prev_is_text
            else [
                "duplicated_opener",
                "stray_quote",
            ]
        )
        ops.append(draw(st.one_of(st.none(), st.sampled_from(allowed))))
    end_insert = draw(st.booleans())
    n_ops = sum(op is not None for op in ops) + int(end_insert)
    assume(1 <= n_ops <= 3)

    out = parts[0]
    for j in range(1, len(parts)):
        op, nxt = ops[j - 1], parts[j]
        if op == "duplicated_opener":
            out += ',{"' + nxt
        elif op == "missing_separator":
            out = out[:-1] + ' "' + nxt
        elif op == "spurious_closer":
            out = out[:-1] + "]}," + nxt
        elif op == "stray_quote":
            out = out[:-1] + '"]},' + nxt
        else:
            out += "," + nxt
    corrupted = HEADER + UC_OPEN + out + ("}}}" if end_insert else UC_CLOSE)
    strings = set(decoded_strings([json.loads(HEADER), *doc["components"]]))
    strings |= {"version", "v0.9.1", "updateComponents", "surfaceId", "s", "components"}
    return original, corrupted, strings


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(corrupted_docs())
def test_alphabet_and_lexemes(case: tuple[str, str, set[str]]) -> None:
    """Criterion 3, N1-N4 (N5 is the gate's, in ``test_ui_gate_structural.py``)."""
    _, raw, strings = case
    try:
        parse_concatenated_json(raw)
    except json.JSONDecodeError:
        pass
    else:
        assume(False)  # an operator that happened to leave valid JSON is criterion 2's business
    out, _ = repair(raw)
    assert_n1_n2(raw, out, "generated document")
    assert repair(out) == (out, 0), f"criterion 3 (N4): repair(out) == (out, 0); out={out!r}"
    fixer, _ = UiGate().repair(raw)
    leaked = set(decoded_strings(fixer)) - strings
    assert not leaked, (
        f"criterion 3 (N3): every decoded string leaf and key of the gate's output is in S; "
        f"{sorted(leaked)[:5]} are not (raw={raw!r})"
    )
    result = UiGate().gate(raw, turn_id="n3", canvas_idx=0, final=False)
    if result.surface is not None:
        emitted = set(decoded_strings(list(result.surface.messages))) - strings - {"turn-n3"}
        assert not emitted, f"criterion 3 (N3): the surface carries {sorted(emitted)[:5]}"


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(corrupted_docs())
def test_generated_corruptions_are_undone(case: tuple[str, str, set[str]]) -> None:
    """Criterion 4 generalised: each observed operator, alone or two or three together on a
    canvas of hostile strings, is undone exactly (whitespace outside strings aside: the missing
    separator leaves a space beside the edit). Without this row a repair that does nothing
    satisfies N1, N2 and N4."""
    original, raw, _ = case
    out, edits = repair(raw)
    assert without_ws_outside_strings(out) == without_ws_outside_strings(original), (
        f"the repair must restore the document the corruption started from.\n raw: {raw!r}\n got: "
        f"{out!r}\nwant: {original!r}"
    )
    assert edits >= 1, f"criterion 1: a corrupted canvas was repaired with edits={edits}"


# --- adversarial rows (a)-(f) ------------------------------------------------------------------


def adversarial(components: str) -> str:
    return HEADER + UC_OPEN + components + UC_CLOSE


ROOT = '{"id":"root","component":"Column","children":["a","b"]},'


def test_a_brace_quote_inside_a_string_is_not_deduped() -> None:
    """(a) ``{"`` inside string values, and an object whose key is ``{``, beside a real
    duplicated opener: only the real one goes."""
    raw = adversarial(
        ROOT
        + r'{"{"id":"a","component":"Text","text":"say {\"{\"id\": now"},'
        + r'{"{":"v","id":"b","component":"Text","text":"{\"{\""}'
    )
    want = adversarial(
        ROOT
        + r'{"id":"a","component":"Text","text":"say {\"{\"id\": now"},'
        + r'{"{":"v","id":"b","component":"Text","text":"{\"{\""}'
    )
    assert repair(raw)[0] == want


def test_closers_inside_a_string_beside_a_real_wrong_closer() -> None:
    """(b) ``]`` and ``}`` inside strings, with a real spurious ``]`` right after one of them."""
    raw = adversarial(
        ROOT
        + '{"id":"a","component":"Text","text":"x]}y"]},'
        + '{"id":"b","component":"Text","text":"}]"}'
    )
    want = adversarial(
        ROOT
        + '{"id":"a","component":"Text","text":"x]}y"},{"id":"b","component":"Text","text":"}]"}'
    )
    out, edits = repair(raw)
    assert out == want and edits >= 1, f"(b): got {out!r}"


NESTED = (
    '{"id":"root","component":"Column","children":["p","t"]},'
    '{"id":"p","component":"ChoicePicker","options":[{"label":"One","value":"1"},'
    '{"label":"Two","value":"2"}],"value":{"path":"/a"}},'
    '{"id":"t","component":"Tabs","tabs":[{"title":"A","child":"x"},{"title":"B","child":"y"}]},'
    '{"id":"x","component":"Text","text":"ex"},'
    '{"id":"y","component":"Text","text":"why"}'
)


def test_legitimately_nested_lists_of_objects_get_zero_edits() -> None:
    """(c) ``[{…}]`` (ChoicePicker options, ``Tabs.tabs``) is not a wrong closer: zero edits when
    it is the whole canvas, and untouched when a real corruption elsewhere makes the repair run."""
    valid = adversarial(NESTED)
    parse_concatenated_json(valid)
    assert repair(valid) == (valid, 0)
    raw = adversarial(NESTED.replace('{"id":"y"', '{"{"id":"y"'))
    out, edits = repair(raw)
    assert out == valid and edits >= 1, f"(c): only the duplicated opener goes; got {out!r}"


def test_escape_edges() -> None:
    """(d) ``"a\\\\"`` (a, backslash) and ``"a\\\\\\"b"`` (a, backslash, quote, b), each followed
    by a real spurious closer: a scanner that misreads the escapes stays "inside" the string and
    misses it, or ends the string early and edits inside it."""
    raw = adversarial(
        ROOT
        + r'{"id":"a","component":"Text","text":"a\\"]},'
        + r'{"id":"b","component":"Text","text":"a\\\"b"]}'
    )
    want = adversarial(
        ROOT
        + r'{"id":"a","component":"Text","text":"a\\"},'
        + r'{"id":"b","component":"Text","text":"a\\\"b"}'
    )
    out, _ = repair(raw)
    assert out == want, f"(d): got {out!r}"
    parse_concatenated_json(out)


def test_a_body_cut_inside_the_header_gets_zero_edits() -> None:
    """(e) partial-canvas rule: at end of text close nothing (json_repair closes truncations)."""
    cuts = [r["text"] for r in fixture_rows() if r["class"] == "header_cut"]
    cuts += [HEADER[:k] for k in (1, 2, 9, 15, 40, len(HEADER) - 3)]
    cuts += [HEADER + UC_OPEN[:k] for k in (20, 45, len(UC_OPEN) - 1)]
    for text in cuts:
        assert repair(text) == (text, 0), f"(e): a canvas cut in its header is left as is: {text!r}"


@pytest.mark.parametrize(
    "tail",
    [
        '{"id":"a","component":"Text","text":"x"]',
        '{"id":"a","component":"Text","text":"x"}}',
        '{"id":"a","component":"Column","children":["b"}',
        '{"id":"a","component":"Text","text":"x"}]]]]',
        '{"id":"a","component":"Text","text":"x"]"',
        '{"id":"a","component":"Text","text":"x" "',
    ],
)
def test_a_mismatched_closer_at_the_very_end(tail: str) -> None:
    """(f) no exception, and N1/N2 hold, when the mismatch is the last character."""
    raw = HEADER + UC_OPEN + tail
    out, _ = repair(raw)
    assert_n1_n2(raw, out, f"tail {tail!r}")


def test_the_repair_keeps_no_state_between_calls() -> None:
    """Criterion 6: the same text gives the same answer whatever was repaired before it."""
    texts_ = [r["text"] for r in fixture_rows()]
    first = [repair(t) for t in texts_]
    again = [repair(t) for t in reversed(texts_)][::-1]
    assert first == again, "criterion 6: the repair keeps no state between calls"


def test_the_real_fixture_canvases_keep_n1_n2_n4() -> None:
    """Criterion 3 on every real canvas of the fixture, repaired classes or not."""
    for r in fixture_rows():
        out, _ = repair(r["text"])
        assert_n1_n2(r["text"], out, f"{r['request_id']}#{r['ui_index']}")
        assert repair(out) == (out, 0), f"N4 on {r['request_id']}#{r['ui_index']}"


# --- criterion 5: linear, iterative, bounded ---------------------------------------------------


def _family(unit: str, n: int) -> str:
    return (unit * (n // len(unit) + 1))[:n]


FAMILIES: dict[str, Callable[[int], str]] = {
    "]*n": lambda n: "]" * n,
    "[*n": lambda n: "[" * n,
    '{"{"*n': lambda n: _family('{"{"', n),
    '"*n': lambda n: '"' * n,
    "ws run + mismatched ]": lambda n: "{" + " " * (n - 2) + "]",
    "ws run + mismatched }": lambda n: "[" + "\n" * (n - 2) + "}",
    '"x" "{"id":*k': lambda n: _family('"x" "{"id":', n),
    '{"a":*k': lambda n: _family('{"a":', n),
    # Reviewer A: a mismatched closer every 3 chars, each needing a look at what follows. A
    # lookahead that scans the whole rest of the text instead of the next non-space is quadratic.
    "[ + {]}*k": lambda n: "[" + _family("{]}", n - 1),
    # …and the same with a whitespace run between, which the lookahead must skip
    "[ + {]    }*k": lambda n: "[" + _family("{]    }", n - 1),
    # T-0129 (adversarial, aimed at the recursion mutants the old row missed): one mismatched
    # closer followed by a long whitespace run before the expected closer, so the lookahead
    # itself must skip n-4 spaces (the two ws families above never reach it: their closer has
    # no opener of its kind)...
    "[{] + ws run + }": lambda n: "[{]" + " " * (n - 4) + "}",
    # ...and one mismatched closer that must insert n-2 closers before it matches.
    "[ + {*n + ]": lambda n: "[" + "{" * (n - 2) + "]",
}

#: T-0129 criterion 1: work sizes and the bar. Linear gives 2 per doubling, n log n about 2.17
#: at 4096, quadratic 4; the count is exact, so the bar can sit just above n log n.
WORK_SIZES = (1024, 2048, 4096)
WORK_RATIO = 2.25
#: T-0129 criterion 2: CPU sizes (interleaved), repetitions, and the bars. 10 ms is T-0108's
#: number; 8 is the geometric middle of linear (4) and quadratic (16) for the 4x step.
CPU_SIZES = (4096, 8192, 16384)
CPU_RUNS = 5
CPU_BOUND_8192_S = 0.010
CPU_RATIO_4X = 8.0
HASH_SEEDS = ("0", "1", "12345")


def _repair_module_file() -> Path:
    """The repair module's own file, from the imported module (a mutation copy counts itself)."""
    repair_fn()  # the pending NotImplementedError when the module is absent
    path = importlib.import_module(REPAIR_MODULE).__file__
    assert path is not None
    return Path(path)


def _work_counts(fn: Callable[[str], Any], root: Path, texts: list[str]) -> list[int]:
    """Traced line events of ``root`` for ``fn`` over ``texts`` (doubling sizes). Each run after
    the first stops once it passes ``WORK_RATIO`` x the previous count + 1."""
    counts: list[int] = []
    for text in texts:
        cap = None if not counts else int(WORK_RATIO * counts[-1]) + 1
        try:
            counts.append(line_events(functools.partial(fn, text), root, cap=cap))
        except WorkCapExceeded as exc:
            raise AssertionError(
                f"criterion 5 (work, T-0129): at {len(text)} chars the repair passed "
                f"{exc.cap} traced line events = {WORK_RATIO} x L({len(text) // 2}) + 1 "
                f"(counts so far {counts}); the scan is not linear"
            ) from None
    return counts


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_no_recursion_to_depth_8192(family: str) -> None:
    """Criterion 5 (T-0129 criterion 3): no exception at any depth up to 8192 (a recursive
    scanner dies at Python's default limit of 1000)."""
    build = FAMILIES[family]
    for n in (1000, 1001, 4096, 8192):
        text = build(n)
        assert len(text) == n
        try:
            repair(text)
        except RecursionError as err:
            raise AssertionError(f"criterion 5: {family} at {n} chars recursed: {err}") from err


def test_the_two_new_families_reach_the_lookahead_and_the_insert() -> None:
    """Premise of T-0129's two adversarial families, by hand from T-0108's rule 4: in
    ``[{]`` + spaces + ``}`` the ``]`` is dropped because the expected ``}`` follows after the
    whitespace (the lookahead skips the run); in ``[`` + ``{`` x 10 + ``]`` nothing expected
    follows, so ten ``}`` are inserted before the ``]`` matches."""
    out, edits = repair(FAMILIES["[{] + ws run + }"](12))
    assert (out, edits >= 1) == ("[{" + " " * 8 + "}", True), f"lookahead family: {out!r}"
    out, edits = repair(FAMILIES["[ + {*n + ]"](12))
    assert (out, edits >= 1) == ("[" + "{" * 10 + "}" * 10 + "]", True), f"insert family: {out!r}"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_linear_work(family: str) -> None:
    """Criterion 5 (T-0129 criterion 1), work: the traced line events L(n) of the repair module at
    n = 1024, 2048, 4096 satisfy L(2n)/L(n) <= 2.25 at each doubling. Exact and load-independent;
    blind to work inside C (the CPU row covers that)."""
    build = FAMILIES[family]
    texts = [build(n) for n in WORK_SIZES]
    assert [len(t) for t in texts] == list(WORK_SIZES)
    counts = _work_counts(repair_fn(), _repair_module_file(), texts)
    assert counts[0] > 0, (
        f"criterion 5 (work): no line of {_repair_module_file()} was traced for {family}; the "
        "count would be vacuous"
    )
    for k in range(1, len(WORK_SIZES)):
        n, small, large = WORK_SIZES[k - 1], counts[k - 1], counts[k]
        assert large <= WORK_RATIO * small, (
            f"criterion 5 (work, T-0129): {family}: L({2 * n})/L({n}) = {large}/{small} = "
            f"{large / small:.3f} > {WORK_RATIO}; the scan is not linear"
        )


_SEED_CHILD = """
import importlib, json, sys
from tests.unit._work_count import WorkCapExceeded, line_events
mod = importlib.import_module(sys.argv[1])
fn = getattr(mod, sys.argv[2])
out = {}
for family, texts in json.load(sys.stdin).items():
    counts = []
    for text in texts:
        cap = None if not counts else int(float(sys.argv[3]) * counts[-1]) + 1
        try:
            counts.append(line_events(lambda: fn(text), mod.__file__, cap=cap))
        except WorkCapExceeded:
            counts.append(None)
            break
    out[family] = counts
print(json.dumps({"file": mod.__file__, "counts": out}))
"""


def test_linear_work_does_not_depend_on_the_hash_seed() -> None:
    """Criterion 5 (T-0129 criterion 1): the work counts of every family at 1024/2048/4096 are
    identical under ``PYTHONHASHSEED`` 0, 1 and 12345 (three child processes of this tree), and
    equal to this process's own counts, so the work ratio cannot flicker with set or dict order."""
    fn, root = repair_fn(), _repair_module_file()
    texts = {family: [build(n) for n in WORK_SIZES] for family, build in FAMILIES.items()}
    here = {family: _work_counts(fn, root, t) for family, t in texts.items()}
    repo = str(REPO)
    procs = {
        seed: subprocess.Popen(
            [sys.executable, "-c", _SEED_CHILD, REPAIR_MODULE, REPAIR_NAME, str(WORK_RATIO)],
            cwd=repo,
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": repo},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for seed in HASH_SEEDS
    }
    payload = json.dumps(texts)
    results: dict[str, Any] = {}
    for seed, proc in procs.items():
        stdout, stderr = proc.communicate(payload, timeout=120)
        assert proc.returncode == 0, f"PYTHONHASHSEED={seed}: the child failed:\n{stderr[-2000:]}"
        results[seed] = json.loads(stdout)
    for seed, result in results.items():
        assert Path(result["file"]).resolve() == root.resolve(), (
            f"PYTHONHASHSEED={seed}: the child imported {result['file']}, not {root}"
        )
        assert result["counts"] == here, (
            f"criterion 5 (work, T-0129): the counts under PYTHONHASHSEED={seed} differ from this "
            f"process's: {result['counts']} vs {here}"
        )


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_linear_cpu_time(family: str) -> None:
    """Criterion 5 (T-0129 criterion 2), CPU: ``time.thread_time`` with gc off, 4096/8192/16384
    interleaved in each of 5 repetitions, the minimum per size: t(8192) <= 10 ms (T-0108's bound,
    now CPU time) and t(16384)/t(4096) <= 8 (linear 4, quadratic 16). Catches C-level blow-ups
    the work row cannot see (a regex or slice over the rest of the text)."""
    fn = repair_fn()
    build = FAMILIES[family]
    texts = {n: build(n) for n in CPU_SIZES}
    best = dict.fromkeys(CPU_SIZES, float("inf"))
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for _ in range(CPU_RUNS):
            for n in CPU_SIZES:
                start = time.thread_time()
                fn(texts[n])
                best[n] = min(best[n], time.thread_time() - start)
    finally:
        if enabled:
            gc.enable()
    t4, t8, t16 = (best[n] for n in CPU_SIZES)
    assert t8 <= CPU_BOUND_8192_S, (
        f"criterion 5 (CPU, T-0129): {family} at 8192 chars took {t8 * 1e3:.2f} ms of thread CPU "
        f"(min of {CPU_RUNS}); bound {CPU_BOUND_8192_S * 1e3:.0f} ms"
    )
    ratio = t16 / max(t4, 1e-9)
    assert ratio <= CPU_RATIO_4X, (
        f"criterion 5 (CPU, T-0129): {family}: t(16384)/t(4096) = {ratio:.2f} > {CPU_RATIO_4X} "
        f"({t4 * 1e3:.3f} ms -> {t16 * 1e3:.3f} ms, thread CPU, min of {CPU_RUNS}); not linear"
    )
