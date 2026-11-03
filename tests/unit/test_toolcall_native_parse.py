"""T-0084b: the native Gemma 4 call parser, the native draft name and the ``tools`` request field.

``local_astra/toolcall/native.py`` (pure; stdlib plus ``toolcall.score``/``.tools``):

- ``parse_call(text) -> dict | score.ParseFailure`` (criterion 1, QN4) on the text after ``<|tool_call>``;
- ``draft_name(draft_call_text) -> str | None`` (criterion 3);
- ``tools_field(tools) -> list[dict]`` (criterion 4);
- ``TOOL_CALL_OPEN``, ``TOOL_CALL_CLOSE``, ``STRING_DELIM`` (criterion 5; the tokenizer half is the cd row).

Every expectation is hand-written from the task file and from the pinned chat template's rendering
(``format_argument``: dictsorted bare keys, strings in ``<|"|>`` unescaped, ints as digits, booleans as
``true``/``false``, no whitespace). The Hypothesis round trip below renders with a tiny independent renderer
of that format (plus inserted whitespace, which QN4 allows); it never calls the parser to build expectations.
The template-rendered oracle, the vLLM ``_parse_gemma4_args`` agreement and the tokenizer pieces are
``tests/cd/unit/test_toolcall_native_oracle.py`` (criterion 2 and the second half of 5), added once T-0084a's
``tests/cd/oracle/`` helpers are on main.

The module is imported inside each row (``_native()``), so while it does not exist every row fails with an
``AssertionError`` naming it, not a collection error. The criterion 6 rows (byte identity and ``route()``) are
green by design: they guard the diff; the byte-identity row runs over T-0084b's closed range (T-0094).

Hypothesis rows: seeded (``@seed``), no example database, n stated per row.
"""

from __future__ import annotations

import copy
import importlib
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

from tests import _git_range as gr
from tests.unit._toolcall_support import TOOLS_V1

MODULE = "local_astra.toolcall.native"

#: Criterion 1: the ten distinct failure reasons, a closed set.
REASONS = frozenset(
    {
        "empty",
        "no call prefix",
        "bad name",
        "no close",
        "text after call",
        "multiple calls",
        "duplicate key",
        "bare value",
        "unterminated string",
        "malformed arguments",
    }
)
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
Q = '<|"|>'  # the string delimiter, id 52
CLOSE = "<tool_call|>"  # id 49
OPEN = "<|tool_call>"  # id 48
V1_ORDER = ["google_search", "click_at", "type_at", "scroll", "press_key"]


def _native() -> ModuleType:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == MODULE:
            raise AssertionError(
                f"T-0084b not implemented: `{MODULE}` does not exist ({exc})"
            ) from exc
        raise


def _score() -> ModuleType:
    return importlib.import_module("local_astra.toolcall.score")


def _tools() -> ModuleType:
    return importlib.import_module("local_astra.toolcall.tools")


def _parse(text: str) -> Any:
    return _native().parse_call(text)


def _ok(text: str) -> dict[str, Any]:
    got = _parse(text)
    assert isinstance(got, dict), f"{text!r} was refused: {got!r}"
    return got


def _reason(text: str) -> str:
    got = _parse(text)
    assert isinstance(got, _score().ParseFailure), f"{text!r} was accepted: {got!r}"
    return str(got.reason)


def _same(a: Any, b: Any) -> bool:
    """Equality that also requires equal types at every node (``True != 1``, ``1.0 != 1`` here)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


# =========================================================================== criterion 1: accepted shapes

#: The five v1 tools, rendered by hand exactly as the pinned template renders an assistant ``tool_calls``
#: entry (dictsorted keys, no whitespace), without the leading ``<|tool_call>`` (the router strips it).
CANONICAL = [
    (
        f"call:google_search{{query:{Q}weather in Paris{Q}}}{CLOSE}",
        {"name": "google_search", "arguments": {"query": "weather in Paris"}},
    ),
    ("call:click_at{x:260,y:115}" + CLOSE, {"name": "click_at", "arguments": {"x": 260, "y": 115}}),
    (
        f"call:type_at{{press_enter:true,text:{Q}hello{Q},x:500,y:42}}{CLOSE}",
        {
            "name": "type_at",
            "arguments": {"press_enter": True, "text": "hello", "x": 500, "y": 42},
        },
    ),
    (
        f"call:scroll{{direction:{Q}down{Q}}}{CLOSE}",
        {"name": "scroll", "arguments": {"direction": "down"}},
    ),
    (
        f"call:press_key{{combo:{Q}Ctrl+C{Q}}}{CLOSE}",
        {"name": "press_key", "arguments": {"combo": "Ctrl+C"}},
    ),
]


@pytest.mark.parametrize(("text", "want"), CANONICAL, ids=[w["name"] for _, w in CANONICAL])
def test_canonical_template_rendering_of_each_v1_tool_parses_to_the_call(
    text: str, want: dict[str, Any]
) -> None:
    """Too-strict guard: rejecting the canonical rendering (``call:click_at{x:260,y:115}<tool_call|>`` among
    them) fails this row. Types are preserved (``260`` is an int, ``true`` a bool)."""
    got = _ok(text)
    assert _same(got, want), got


@pytest.mark.parametrize(("text", "want"), CANONICAL, ids=[w["name"] for _, w in CANONICAL])
def test_the_unchanged_scorer_judges_a_native_parse(text: str, want: dict[str, Any]) -> None:
    """The goal: ``score.schema_valid`` scores the native parse unchanged (each canonical call is valid v1)."""
    got = _ok(text)
    assert _score().schema_valid(got, _tools().load()) is True, got


def test_exact_scores_a_native_click_and_type() -> None:
    sc = _score()
    click = _ok("call:click_at{x:260,y:115}" + CLOSE)
    assert sc.exact(click, {"name": "click_at", "bbox": [250, 100, 270, 130]}) is True
    assert sc.exact(click, {"name": "click_at", "bbox": [0, 0, 100, 100]}) is False
    typed = _ok(f"call:type_at{{press_enter:true,text:{Q}hello{Q},x:500,y:42}}{CLOSE}")
    gt = {"name": "type_at", "bbox": [490, 30, 510, 50], "text": "hello", "press_enter": True}
    assert sc.exact(typed, gt) is True


def test_empty_arguments_and_a_name_outside_v1_parse() -> None:
    """``ARGS`` may be empty, and the parser does not judge the name against v1 (the scorer does)."""
    assert _same(_ok("call:f{}" + CLOSE), {"name": "f", "arguments": {}})
    assert _same(
        _ok(f"call:open_app{{app:{Q}x{Q}}}{CLOSE}"), {"name": "open_app", "arguments": {"app": "x"}}
    )
    assert _same(_ok("call:_A9{_k1:1}" + CLOSE), {"name": "_A9", "arguments": {"_k1": 1}})


@pytest.mark.parametrize(
    ("lit", "want"),
    [
        ("0", 0),
        ("-7", -7),
        ("999", 999),
        ("-0", 0),
        ("12345678901234567890", 12345678901234567890),
        ("1.5", 1.5),
        ("-0.25", -0.25),
        ("1.0", 1.0),
        ("1e3", 1000.0),
        ("2E-2", 0.02),
        ("-1.5e+2", -150.0),
        ("true", True),
        ("false", False),
        ("null", None),
        (f"{Q}{Q}", ""),
        (f"{Q}up{Q}", "up"),
        ("[]", []),
        ("{}", {}),
        (f"[1,{Q}a{Q},true,null,2.5]", [1, "a", True, None, 2.5]),
        (f"{{b:1,c:{Q}z{Q},d:[false]}}", {"b": 1, "c": "z", "d": [False]}),
        ("[[1],[[2]],{e:[]}]", [[1], [[2]], {"e": []}]),
    ],
)
def test_typed_values(lit: str, want: Any) -> None:
    """A JSON number is an int without fraction or exponent, otherwise a float; literals, strings, objects and
    arrays as listed in criterion 1."""
    got = _ok(f"call:f{{v:{lit}}}{CLOSE}")
    assert _same(got, {"name": "f", "arguments": {"v": want}}), (lit, got)


def test_deep_nesting_parses() -> None:
    depth = 40
    want: Any = 1
    for _ in range(depth):
        want = [want]
    got = _ok("call:f{v:" + "[" * depth + "1" + "]" * depth + "}" + CLOSE)
    assert _same(got["arguments"]["v"], want)


@pytest.mark.parametrize(
    "text",
    [
        "  \n call:click_at{x:260,y:115}<tool_call|> \n\t",
        "call:click_at{ x : 260 , y : 115 }<tool_call|>",
        "call:click_at{\n  x: 260,\n  y: 115\n}<tool_call|>",
        "call:click_at{y:115,x:260}<tool_call|>",
        "\tcall:click_at{ y:115 ,x : 260}<tool_call|>\n",
        "call:click_at {x:260,y:115}<tool_call|>",  # B4: between NAME and `{`
        "call:click_at{x:260,y:115} <tool_call|>",  # B4: between the final `}` and the close
        "call:click_at \n{ x:260,y:115 }\n\t<tool_call|>",
    ],
)
def test_whitespace_leniency_and_free_key_order(text: str) -> None:
    """QN4 and B4: whitespace around ``{ } , :`` (including after NAME and before ``<tool_call|>``) and around
    the call, keys in any order."""
    assert _same(_ok(text), {"name": "click_at", "arguments": {"x": 260, "y": 115}})


def test_whitespace_inside_arrays_and_nested_objects() -> None:
    got = _ok(f"call:f{{ a : [ 1 , {Q}b{Q} , [ ] ] , o : {{ k : true , j : {{ }} }} }}{CLOSE}")
    assert _same(got, {"name": "f", "arguments": {"a": [1, "b", []], "o": {"k": True, "j": {}}}})


# --------------------------------------------------------------------------- strings verbatim

VERBATIM = [
    'he said "hi"',
    "a}b",
    "a,b:c",
    "{x:1}",
    "line one\nline two",
    "café ☕ 東京 — ü",
    "  padded  ",
    "back\\nslash",  # a backslash and an n: no JSON unescaping
    '\\"',
    "call:scroll{direction:up}",
    "[1,2]",
    "true",
]


@pytest.mark.parametrize("s", VERBATIM)
def test_string_contents_are_kept_verbatim(s: str) -> None:
    """``"``, ``}``, ``,``, ``:``, ``{``, a newline, unicode, padding and backslashes stay as written."""
    got = _ok(f"call:type_at{{press_enter:false,text:{Q}{s}{Q},x:1,y:2}}{CLOSE}")
    assert _same(
        got,
        {"name": "type_at", "arguments": {"press_enter": False, "text": s, "x": 1, "y": 2}},
    ), got


def test_adversarial_close_tag_inside_a_string_stays_in_the_string() -> None:
    """The task's adversarial case: ``}<tool_call|>`` inside ``<|"|>…<|"|>`` is string content, not the end of
    the call (a parser that finds the close before scanning strings returns ``text`` = ``"a"`` or refuses)."""
    text = f"call:type_at{{text:{Q}a}}{CLOSE}b{Q},press_enter:true,x:3,y:4}}{CLOSE}"
    got = _ok(text)
    assert _same(
        got,
        {
            "name": "type_at",
            "arguments": {"text": "a}<tool_call|>b", "press_enter": True, "x": 3, "y": 4},
        },
    ), got


def test_open_tag_and_a_second_call_inside_a_string_are_not_multiple_calls() -> None:
    inner = f"{CLOSE}{OPEN}call:scroll{{direction:up}}{CLOSE}"
    got = _ok(f"call:type_at{{text:{Q}{inner}{Q},press_enter:false,x:0,y:0}}{CLOSE}")
    assert got["arguments"]["text"] == inner


# =========================================================================== criterion 1: failure reasons

FAILURES = [
    ("", "empty"),
    ("   \n\t", "empty"),
    ("click_at{x:1,y:2}" + CLOSE, "no call prefix"),
    ('{"name": "click_at", "arguments": {"x": 1, "y": 2}}', "no call prefix"),
    ("Sure. call:click_at{x:1,y:2}" + CLOSE, "no call prefix"),
    ("CALL:click_at{x:1,y:2}" + CLOSE, "no call prefix"),
    ("call:9lives{}" + CLOSE, "bad name"),
    ("call:{x:1,y:2}" + CLOSE, "bad name"),
    (f"call:{Q}click_at{Q}{{x:1,y:2}}{CLOSE}", "bad name"),
    ("call:click-at{x:1}" + CLOSE, "bad name"),  # B2
    ("call: click_at{x:1,y:2}" + CLOSE, "bad name"),  # B4: no whitespace after `call:`
    ("call:click_at{x:1,y:2}", "no close"),
    ("call:click_at{x:1,y:2}\n  ", "no close"),
    ("call:click_at{x:1,", "no close"),  # B2: truncated arguments, no complete close anywhere
    ("call:click_at{x:1,y:2}<tool_call", "no close"),  # B2: a partial close tag
    ("call:click_at{x:1,y:2}" + CLOSE + "Done", "text after call"),
    ("call:click_at{x:1,y:2}" + CLOSE + " I clicked it.", "text after call"),
    (
        f"call:click_at{{x:1,y:2}}{CLOSE}{OPEN}call:scroll{{direction:{Q}down{Q}}}{CLOSE}",
        "multiple calls",
    ),
    (f"call:click_at{{x:1,y:2}}{CLOSE}\n{OPEN}call:click_at{{x:3,y:4}}{CLOSE}", "multiple calls"),
    ("call:click_at{x:1,x:2}" + CLOSE, "duplicate key"),
    ("call:click_at{x:1,y:2,x:1}" + CLOSE, "duplicate key"),
    ("call:f{a:{b:1,b:2}}" + CLOSE, "duplicate key"),
    ("call:scroll{direction:up}" + CLOSE, "bare value"),
    (f"call:type_at{{press_enter:True,text:{Q}a{Q},x:1,y:2}}{CLOSE}", "bare value"),
    ("call:f{a:None}" + CLOSE, "bare value"),
    ("call:f{a:[up]}" + CLOSE, "bare value"),
    (f"call:type_at{{text:{Q}hello}}{CLOSE}", "unterminated string"),
    (f"call:google_search{{query:{Q}weather", "unterminated string"),
    ("call:click_at{x:1,,y:2}" + CLOSE, "malformed arguments"),
    ("call:click_at{x:1,}" + CLOSE, "malformed arguments"),
    ("call:click_at{:1}" + CLOSE, "malformed arguments"),
    ("call:f{a:[1,2}" + CLOSE, "malformed arguments"),
    (f"call:f{{{Q}a{Q}:1}}{CLOSE}", "malformed arguments"),
    ('call:f{"a":1}' + CLOSE, "malformed arguments"),
    ("call:click_at{x:1,y:2}}" + CLOSE, "malformed arguments"),  # B2: an extra `}` before the close
]


@pytest.mark.parametrize(("text", "reason"), FAILURES, ids=[f"{r}:{t[:40]!r}" for t, r in FAILURES])
def test_each_failure_class_has_its_own_reason(text: str, reason: str) -> None:
    """Too-lenient guards among these: no close, ``…<tool_call|>Done``, a bare word ``direction:up`` and a
    duplicate key must each be refused, with the named reason."""
    assert _reason(text) == reason


def test_every_reason_is_exercised_by_the_table() -> None:
    """The table above covers all ten reasons (a self-check of this module, green by construction)."""
    assert {r for _, r in FAILURES} == REASONS


#: Ruling B1 (tightened at review, 2026-10-09): an invalid bare token is ``bare value``; a structural break (an
#: empty value, a missing ``,``) is ``malformed arguments``. These rows used to accept either reason.
B1_EXACT = [
    ("call:click_at{x:1 y:2}" + CLOSE, "malformed arguments"),  # missing `,`
    ("call:f{a:}" + CLOSE, "malformed arguments"),  # empty value
    ("call:f{a:01}" + CLOSE, "bare value"),
    ("call:f{a:+1}" + CLOSE, "bare value"),
    ("call:f{a:1.}" + CLOSE, "bare value"),
    ("call:f{a:.5}" + CLOSE, "bare value"),
    ('call:scroll{direction:"up"}' + CLOSE, "bare value"),
    ("call:scroll{direction:'up'}" + CLOSE, "bare value"),
    ("call:f{a:NaN}" + CLOSE, "bare value"),
    ("call:f{a:Infinity}" + CLOSE, "bare value"),
]


@pytest.mark.parametrize(("text", "reason"), B1_EXACT, ids=[f"{r}:{t[:30]!r}" for t, r in B1_EXACT])
def test_non_json_bare_values_are_refused_with_the_b1_reason(text: str, reason: str) -> None:
    """QN4: bare values are JSON literals only (a JSON string in ``"`` is not one of the listed VALUE forms)."""
    assert _reason(text) == reason


#: Review rows (mutants M3-M7, M10, M17, M2). Each reason follows from criterion 1 and rulings B1/B2/B7.
REVIEW_ROWS = [
    # strict containers: a closer must match its opener, and no trailing `,` in an array
    ("call:f{a:[1}]" + CLOSE, "malformed arguments"),  # M4
    ("call:f{]" + CLOSE, "malformed arguments"),  # M5
    ("call:f{a:[}]}" + CLOSE, "malformed arguments"),  # M5: `[}`
    ("call:f{a:[1,]}" + CLOSE, "malformed arguments"),  # M3
    # B2: the missing close outranks a bad name
    ("call:click-at{x:1", "no close"),  # M10
    # the bare-token boundary: a special piece ends a token, a plain `<` does not
    (
        f"call:f{{a:1{Q}x{Q}}}{CLOSE}",
        "malformed arguments",
    ),  # M6/M7: literal, then a string without `,`
    (
        "call:f{a:1" + CLOSE,
        "malformed arguments",
    ),  # M6/M7: literal, then the close (object unclosed)
    ("call:f{a:1" + OPEN + "}" + CLOSE, "malformed arguments"),  # literal, then the open tag
    ("call:f{a:1<2}" + CLOSE, "bare value"),  # M6/M7: `1<2` is one bare token, not a literal
    ("call:scroll{direction:up<|x}" + CLOSE, "bare value"),
    # a close that exists only inside a string is no close at all
    (f"call:type_at{{text:{Q}a}}{CLOSE}{Q}", "no close"),  # M17
    (f"call:f{{a:{Q}{CLOSE}{Q}}}", "no close"),  # M17: arguments complete, close only in the string
]


@pytest.mark.parametrize(
    ("text", "reason"), REVIEW_ROWS, ids=[f"{r}:{t[:30]!r}" for t, r in REVIEW_ROWS]
)
def test_review_rows_container_grammar_token_boundary_and_close_precedence(
    text: str, reason: str
) -> None:
    assert _reason(text) == reason


@pytest.mark.parametrize("sign", ["", "-"])
def test_an_integer_literal_past_the_int_digit_limit_is_a_bare_value_not_an_exception(
    sign: str,
) -> None:
    """M2: 4301 digits is past CPython's default ``int`` string limit (4300), where ``int()`` raises; the parser
    must refuse it as ``bare value`` instead (review ruling), and never raise."""
    assert _reason("call:f{a:" + sign + "7" * 4301 + "}" + CLOSE) == "bare value"


@pytest.mark.parametrize(
    "text",
    [
        "\ud800",
        "\x00",
        "call:",
        "call:f",
        "call:f{",
        "call:f{a",
        "call:f{a:",
        f"call:f{{a:{Q}",
        f"call:f{{a:{Q}{Q}",
        "call:f{a:[",
        "call:f{v:" + "[" * 100_000,
        "call:f{v:" + "{a:" * 50_000,
        CLOSE,
        OPEN,
        Q,
        "}" * 10,
    ],
)
def test_never_raises_on_hostile_strings(text: str) -> None:
    got = _parse(text)
    assert isinstance(got, _score().ParseFailure), got
    assert got.reason in REASONS, got.reason


_PIECES = st.sampled_from(
    [
        "call:",
        "f",
        "x",
        "_y",
        "9",
        "{",
        "}",
        "[",
        "]",
        ",",
        ":",
        Q,
        CLOSE,
        OPEN,
        "true",
        "null",
        "1",
        "-2.5e1",
        " ",
        "\n",
        "up",
        '"',
    ]
)
_FUZZ = st.one_of(
    st.lists(_PIECES, max_size=30).map("".join),
    st.lists(_PIECES, max_size=30).map(lambda ps: "call:f{" + "".join(ps)),
    st.text(max_size=60),
)


@seed(84_002)
@settings(
    max_examples=400,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(_FUZZ)
def test_total_and_reasons_are_a_closed_set(text: str) -> None:
    """Never raises on any ``str``; a refusal names one of the ten reasons; an acceptance is a
    ``{"name", "arguments"}`` dict with a NAME-shaped name. n = 400, seed 84002."""
    got = _parse(text)
    if isinstance(got, dict):
        assert set(got) == {"name", "arguments"}, got
        assert NAME_RE.match(got["name"]), got
        assert isinstance(got["arguments"], dict), got
    else:
        assert isinstance(got, _score().ParseFailure), got
        assert got.reason in REASONS, got.reason


# --------------------------------------------------------------------------- round trip, both directions

_KEY = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,6}", fullmatch=True)
_STR = st.text(max_size=12).filter(lambda s: Q not in s)
_SCALAR = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**12), max_value=10**12),
    st.floats(allow_nan=False, allow_infinity=False),
    _STR,
)
_VALUE = st.recursive(
    _SCALAR,
    lambda inner: st.one_of(
        st.lists(inner, max_size=3),
        st.dictionaries(_KEY, inner, max_size=3),
    ),
    max_leaves=8,
)
_WS = st.sampled_from(["", "", " ", "\n", "  ", "\t"])


def _render(v: Any, ws: Any) -> str:
    """The template's ``format_argument(v, escape_keys=False)``, with optional whitespace (``ws()``) around
    structural characters. Written from the template, independent of the parser."""
    if isinstance(v, str):
        return Q + v + Q
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ws() + ("," + ws()).join(_render(x, ws) + ws() for x in v) + "]"
    assert isinstance(v, dict)
    pairs = (
        k + ws() + ":" + ws() + _render(x, ws) + ws() for k, x in sorted(v.items(), reverse=True)
    )
    return "{" + ws() + ("," + ws()).join(pairs) + "}"


@seed(84_003)
@settings(max_examples=300, deadline=None, database=None)
@given(name=_KEY, args=st.dictionaries(_KEY, _VALUE, max_size=4), data=st.data())
def test_rendered_calls_round_trip_and_break_when_the_close_or_tail_is_wrong(
    name: str, args: dict[str, Any], data: st.DataObject
) -> None:
    """Both directions on generated calls (n = 300, seed 84003): a rendering with random whitespace (QN4) and
    reverse key order parses to exactly the arguments, types preserved; the same text without its final
    ``<tool_call|>`` is ``no close``, and with ``x`` appended is ``text after call``."""

    def ws() -> str:
        return str(data.draw(_WS))

    body = _render(args, ws)
    text = ws() + "call:" + name + ws() + body + ws() + CLOSE + ws()
    got = _ok(text)
    assert _same(got, {"name": name, "arguments": args}), (text, got)
    assert _reason(text.rstrip() + "x") == "text after call", text
    assert _reason(text.rstrip()[: -len(CLOSE)]) == "no close", text


# =========================================================================== criterion 3: draft_name


@pytest.mark.parametrize(
    ("draft", "want"),
    [
        ("call:click_at{x:1,y:2}" + CLOSE, "click_at"),
        ("call:type_at{", "type_at"),
        ("call:type_at{tex", "type_at"),
        (f"call:google_search{{query:{Q}we", "google_search"),
        ("\n  call:scroll{", "scroll"),
        ("call:press_key{combo:up}" + CLOSE + "junk", "press_key"),
        ("call:type_at", None),
        ("call:type_", None),
        ("call:", None),
        ("", None),
        ("call:open_app{}" + CLOSE, None),
        ("call:click{x:1}" + CLOSE, None),
        ("call:click_atx{x:1}" + CLOSE, None),
        ("call: scroll{", None),
        ("call:scroll {", "scroll"),  # B4: draft_name is \A\s*call:NAME\s*\{ (was None before B4)
        ("call:click_at \n\t{x:1", "click_at"),
        ("call:scroll  ", None),
        ("ok call:click_at{x:1,y:2}" + CLOSE, None),
        ('{"name": "click_at", "arguments": {}}', None),
        (f"call:{Q}scroll{Q}{{", None),
    ],
)
def test_draft_name(draft: str, want: str | None) -> None:
    """The v1 name in ``\\A\\s*call:NAME\\s*\\{`` (B4); anything else is None. Both directions: a complete or truncated
    v1 draft names its tool (too strict fails), a non-v1 name, a prefix of a v1 name or no ``{`` gives None (too
    lenient fails)."""
    assert _native().draft_name(draft) == want


@seed(84_004)
@settings(max_examples=200, deadline=None, database=None)
@given(_FUZZ)
def test_draft_name_is_total(text: str) -> None:
    """n = 200, seed 84004: never raises; returns None or a v1 name."""
    got = _native().draft_name(text)
    assert got is None or got in V1_ORDER, got


# =========================================================================== criterion 4: tools_field


def test_tools_field_shape_and_file_order() -> None:
    """The request field is ``{"type": "function", "function": decl}`` per declaration, in file order; the
    declarations are compared with the file's own JSON (an independent read of ``tools_v1.json``)."""
    field = _native().tools_field(_tools().load())
    raw = json.loads(TOOLS_V1.read_bytes())
    assert isinstance(field, list)
    assert [e["function"]["name"] for e in field] == V1_ORDER
    assert field == [{"type": "function", "function": d} for d in raw]
    assert all(set(e) == {"type", "function"} for e in field)


def test_tools_field_follows_the_given_set_not_v1(tmp_path: Path) -> None:
    """``tools_field`` reads the set it is given (a two-tool file in reverse order), not the shipped v1."""
    raw = json.loads(TOOLS_V1.read_bytes())
    two = [raw[3], raw[1]]
    p = tmp_path / "two.json"
    p.write_text(json.dumps(two))
    field = _native().tools_field(_tools().load(p))
    assert field == [{"type": "function", "function": d} for d in two]


def test_tools_field_is_a_deep_copy() -> None:
    """Mutating the result at any depth leaves the ToolSet and the next result untouched."""
    nt = _native()
    ts = _tools().load()
    before = copy.deepcopy([dict(d) for d in ts.declarations])
    field = nt.tools_field(ts)
    for e, d in zip(field, ts.declarations, strict=True):
        assert e["function"] is not d
        assert e["function"]["parameters"] is not d["parameters"]
        assert e["function"]["parameters"]["properties"] is not d["parameters"]["properties"]
    field[1]["function"]["parameters"]["properties"]["x"]["maximum"] = 5
    field[1]["function"]["parameters"]["required"].append("z")
    field[0]["function"]["name"] = "hijacked"
    field[2]["type"] = "other"
    assert [dict(d) for d in ts.declarations] == before
    assert nt.tools_field(ts) == [{"type": "function", "function": d} for d in before]


# =========================================================================== criterion 5: constants


def test_constants_are_the_template_pieces_and_the_router_tag() -> None:
    nt = _native()
    assert nt.TOOL_CALL_OPEN == "<|tool_call>"
    assert nt.TOOL_CALL_CLOSE == "<tool_call|>"
    assert nt.STRING_DELIM == '<|"|>'
    router = importlib.import_module("local_astra.toolcall.router")
    assert router.TAGS["native"] == nt.TOOL_CALL_OPEN


# =========================================================================== criterion 6: negative (diff guards)

FROZEN = (
    "local_astra/toolcall/router.py",
    "local_astra/toolcall/score.py",
    "local_astra/toolcall/tools.py",
    "local_astra/toolcall/tools_v1.json",
)


#: T-0084b's merge commit 9737368; BASE = ``git merge-base 9737368^1 9737368^2``, TIP = ``9737368^2`` (T-0094
#: criterion 2). Not the declared fork point 1e06a24: T-0084b merged main mid-branch, which brought T-0082a's router.
T0084B_MERGE = "9737368"
T0084B_BASE = "4a4513a"
T0084B_TIP = "a1ef4dd"
#: the fork point docs/tasks/T-0084b.md declared; wrong as a range base (see the wrong-base row)
T0084B_DECLARED_FORK = "1e06a24"


def test_router_score_tools_and_the_tool_file_are_unchanged_over_t0084b_s_range() -> None:
    """Criterion 6 over T-0084b's closed range (T-0094): the four ``FROZEN`` files do not differ between BASE and TIP
    (byte identity of a file is "absent from ``git diff --name-only``")."""
    gr.prove_range(T0084B_BASE, T0084B_TIP, T0084B_MERGE)
    changed = gr.changed(T0084B_BASE, T0084B_TIP, *FROZEN)
    assert changed == [], f"T-0084b criterion 6: changed in {T0084B_BASE}..{T0084B_TIP}: {changed}"


def test_the_t0084b_range_guard_has_power() -> None:
    """T-0094 criterion 4 (adversarial): over ``BASE..6a8600b`` the same predicate sees the router (T-0085)."""
    gr.require_commits(T0084B_BASE, gr.LATER)
    assert "local_astra/toolcall/router.py" in gr.changed(T0084B_BASE, gr.LATER, *FROZEN)


def test_the_declared_fork_point_is_the_wrong_base() -> None:
    """T-0094 criterion 6 (adversarial): over the declared ``1e06a24..a1ef4dd`` the predicate reports the router --
    T-0082a's change that reached T-0084b through its mid-branch merge of main, not T-0084b's own edit. That is why
    the base is the merge-time merge-base 4a4513a."""
    gr.require_commits(T0084B_DECLARED_FORK, T0084B_TIP)
    assert gr.changed(T0084B_DECLARED_FORK, T0084B_TIP, *FROZEN) == [
        "local_astra/toolcall/router.py"
    ]


@pytest.mark.parametrize(
    ("answer", "want"),
    [
        (
            "Clicking.<|tool_call>call:click_at{x:1,y:2}<tool_call|>",
            ("toolcall", "native", "Clicking.", "call:click_at{x:1,y:2}<tool_call|>", None),
        ),
        ("<|tool_call>call:type_at{", ("toolcall", "native", "", "call:type_at{", None)),
        (
            'Ok <cc>{"name":"scroll"}<|tool_call>call:scroll{',
            ("toolcall", "cc", "Ok ", '{"name":"scroll"}<|tool_call>call:scroll{', "scroll"),
        ),
    ],
)
def test_route_is_unchanged_for_native_answers(answer: str, want: tuple[Any, ...]) -> None:
    """Criterion 6 (green by design): the router still leaves ``draft_name`` None for native drafts; the native
    name comes from ``native.draft_name`` only (umbrella: "The router stays untouched")."""
    r = importlib.import_module("local_astra.toolcall.router").route(answer)
    assert (r.kind, r.tag, r.spoken_prefix, r.draft_call_text, r.draft_name) == want
