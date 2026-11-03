"""T-0080a criteria 1 and 3: the v1 tool schemas, ``tools.load()``, and the call scorer.

Criterion 1 has no module of its own in the task's "Tests to write"; its rows live here because the
scorer is validated against the same file (K-note for the reviewer).

Expected values are hand-written from the criteria and Q1 (human decision 2026-10-08):
``google_search(query)``, ``click_at(x, y)`` (0-999, normalised to the attached frame),
``type_at(x, y, text, press_enter)``, ``scroll(direction: up|down|left|right)``,
``press_key(combo)`` -- one string from a fixed list (D1, superseding Q1's ``keys`` array). Declaration-level facts are checked by validating hand-built
instances with ``jsonschema`` directly against the shipped declaration (an oracle independent of
``score.schema_valid``).

K-notes (proposals where the task is silent):

- K5 ``tools.load(path=None)`` returns a ``ToolSet`` with ``.version`` (sha256 hex of the file's bytes);
  an invalid file raises ``ValueError``.
- K6 ``parse_call`` reasons are free-form non-empty strings, but distinct failure classes (not JSON, not an
  object, no ``name``, no ``arguments``, ``arguments`` not an object) carry distinct reasons.
- K7 ``exact(call, gt)`` takes the ground-truth shape of ``_toolcall_support.gt`` and returns a ``bool``.
- K8 ``type_at.press_enter`` required or optional is not stated in Q1; not asserted. Keyword whitespace
  handling: surrounding whitespace/newlines/tabs in the query do not matter, and internal runs collapse
  to one space (reviewer ruling, 1e500ef).
- K9 Whether the shipped declarations set ``additionalProperties: false`` is not stated; the extra-key row
  builds its own file with it set (T-0078's lesson: the BFCL scorer admits extra keys, so schema-invalid
  rows elsewhere use out-of-enum, out-of-range or wrong-type values).
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from tests.unit._toolcall_support import (
    D1_NAMED_COMBOS,
    TOOLS_V1,
    V1_NAMES,
    arrow_combos,
    call,
    call_parts,
    combo_enum,
    decl_by_name,
    declarations,
    failure_reason,
    gt,
    params_of,
    raw_tools,
    score_mod,
    shipped_params,
    tools_mod,
    toolset,
    write_mutated,
)


def _valid_against_declaration(name: str, arguments: dict[str, Any]) -> bool:
    schema = shipped_params(name)
    cls = jsonschema.validators.validator_for(schema)
    return bool(cls(schema).is_valid(arguments))


# =========================================================================== criterion 1: schemas


def test_tools_v1_declares_exactly_the_q1_set() -> None:
    names = [str(d["name"]) for d in declarations(raw_tools())]
    assert sorted(names) == sorted(V1_NAMES)
    assert len(names) == len(set(names))


def test_no_ref_anywhere_in_tools_v1() -> None:
    assert "$ref" not in json.dumps(raw_tools())


@pytest.mark.parametrize(
    ("name", "required"),
    [
        ("google_search", {"query"}),
        ("click_at", {"x", "y"}),
        ("type_at", {"x", "y", "text"}),
        ("scroll", {"direction"}),
        ("press_key", {"combo"}),
    ],
)
def test_q1_arguments_are_required(name: str, required: set[str]) -> None:
    p = shipped_params(name)
    assert required <= set(p.get("required", [])), p
    assert set(p.get("required", [])) <= set(p["properties"])


def test_google_search_query_is_a_string() -> None:
    assert _valid_against_declaration("google_search", {"query": "weather in Paris"})
    assert not _valid_against_declaration("google_search", {"query": 7})
    assert not _valid_against_declaration("google_search", {})


@pytest.mark.parametrize(
    ("x", "y", "ok"),
    [
        (0, 0, True),
        (999, 999, True),
        (500, 12, True),
        (1000, 5, False),
        (5, 1000, False),
        (-1, 5, False),
        (5, -1, False),
        (5.5, 5, False),
        ("5", 5, False),
    ],
)
def test_click_at_coordinates_are_integers_0_to_999(x: Any, y: Any, ok: bool) -> None:
    assert _valid_against_declaration("click_at", {"x": x, "y": y}) is ok


def test_type_at_argument_types() -> None:
    good = {"x": 10, "y": 20, "text": "hello", "press_enter": True}
    assert _valid_against_declaration("type_at", good)
    assert not _valid_against_declaration("type_at", {**good, "x": 1000})
    assert not _valid_against_declaration("type_at", {**good, "text": 3})
    assert not _valid_against_declaration("type_at", {**good, "press_enter": "yes"})


@pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
def test_scroll_accepts_the_four_directions(direction: str) -> None:
    assert _valid_against_declaration("scroll", {"direction": direction})


@pytest.mark.parametrize("direction", ["forward", "UP", "", "upward"])
def test_scroll_rejects_other_directions(direction: str) -> None:
    assert not _valid_against_declaration("scroll", {"direction": direction})


def test_press_key_combo_enum_holds_the_d1_list() -> None:
    enum = combo_enum()
    assert len(enum) == len(set(enum))
    missing = D1_NAMED_COMBOS - set(enum)
    assert not missing, f"D1 combos missing from press_key.combo: {sorted(missing)}"
    arrows = arrow_combos(enum)
    assert all(len(v) == 1 for v in arrows.values()), f"D1: one member per arrow, got {arrows}"
    assert 20 <= len(enum) <= 32, f"D1: ~20 keys and shortcuts, got {len(enum)}"


def test_press_key_is_one_combo_string() -> None:
    enum = combo_enum()
    for combo in enum:
        assert _valid_against_declaration("press_key", {"combo": combo}), combo
    assert not _valid_against_declaration("press_key", {"combo": "Hyper+F13"})  # out of enum
    assert not _valid_against_declaration("press_key", {"combo": [enum[0]]})  # wrong type
    assert not _valid_against_declaration("press_key", {"keys": [enum[0]]})  # the pre-D1 shape
    assert not _valid_against_declaration("press_key", {})


def test_load_exposes_the_sha256_of_the_file_bytes_as_version() -> None:
    ts = toolset()
    assert ts.version == hashlib.sha256(TOOLS_V1.read_bytes()).hexdigest()


def test_load_accepts_a_byte_copy_and_versions_it_by_its_own_bytes(tmp_path: Path) -> None:
    """Guard width: an unmodified copy loads, and a reformatted copy gets a different version."""
    raw_tools()  # fails as an assertion while the file is absent
    same = tmp_path / "same.json"
    same.write_bytes(TOOLS_V1.read_bytes())
    assert tools_mod().load(same).version == hashlib.sha256(same.read_bytes()).hexdigest()
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(raw_tools(), indent=7) + "\n")
    v = tools_mod().load(pretty).version
    assert v == hashlib.sha256(pretty.read_bytes()).hexdigest()
    assert v != toolset().version


def _duplicate_a_name(raw: Any) -> None:
    decls = declarations(raw)
    decls[1]["name"] = decls[0]["name"]


def _required_not_in_properties(raw: Any) -> None:
    decls = declarations(raw)
    params = (
        decls[0]["parameters"] if "parameters" in decls[0] else decls[0]["function"]["parameters"]
    )
    params.setdefault("required", []).append("no_such_property")


def _add_a_ref(raw: Any) -> None:
    decls = declarations(raw)
    params = (
        decls[0]["parameters"] if "parameters" in decls[0] else decls[0]["function"]["parameters"]
    )
    first = next(iter(params["properties"]))
    params["properties"][first] = {"$ref": "#/definitions/whatever"}


@pytest.mark.parametrize(
    "mutate",
    [_duplicate_a_name, _required_not_in_properties, _add_a_ref],
    ids=["duplicate-name", "required-not-in-properties", "ref"],
)
def test_load_rejects_an_invalid_file(tmp_path: Path, mutate: Any) -> None:
    bad = write_mutated(tmp_path, mutate)
    with pytest.raises(ValueError):
        tools_mod().load(bad)


# =========================================================================== criterion 3: parse_call


def test_parse_call_success() -> None:
    res = score_mod().parse_call('{"name": "scroll", "arguments": {"direction": "up"}}')
    assert call_parts(res) == ("scroll", {"direction": "up"})


def test_parse_call_tolerates_json_whitespace_around_the_object() -> None:
    res = score_mod().parse_call('\n  {"name": "scroll",\n "arguments": {"direction": "left"}}  \n')
    assert call_parts(res) == ("scroll", {"direction": "left"})


_FAILURES = {
    "empty": "",
    "not_json": "search the web for the weather",
    "truncated": '{"name": "google_search", "arguments": {"query": "x"',
    "not_object": '["google_search", {"query": "x"}]',
    "no_name": '{"arguments": {"query": "x"}}',
    "name_not_str": '{"name": 5, "arguments": {"query": "x"}}',
    "no_arguments": '{"name": "google_search"}',
    "arguments_not_object": '{"name": "google_search", "arguments": "query=x"}',
}


@pytest.mark.parametrize("case", sorted(_FAILURES))
def test_parse_call_failure_has_a_reason(case: str) -> None:
    res = score_mod().parse_call(_FAILURES[case])
    assert call_parts(res) is None, f"{case}: parsed as a call: {res!r}"
    reason = failure_reason(res)
    assert reason, f"{case}: failure without a reason: {res!r}"


def test_parse_call_reasons_distinguish_failure_classes() -> None:
    s = score_mod()
    reasons = {
        c: failure_reason(s.parse_call(_FAILURES[c]))
        for c in ("not_json", "not_object", "no_name", "no_arguments", "arguments_not_object")
    }
    assert len(set(reasons.values())) == len(reasons), reasons


# =========================================================================== criterion 3: schema_valid


@pytest.mark.parametrize(
    "c",
    [
        call("google_search", query="weather in Paris"),
        call("click_at", x=0, y=999),
        call("type_at", x=10, y=20, text="hello", press_enter=False),
        call("scroll", direction="right"),
    ],
    ids=lambda c: c["name"],
)
def test_schema_valid_accepts_valid_calls(c: dict[str, Any]) -> None:
    assert score_mod().schema_valid(c, toolset()) is True


def test_schema_valid_accepts_a_valid_press_key() -> None:
    assert score_mod().schema_valid(call("press_key", combo="Ctrl+C"), toolset()) is True


@pytest.mark.parametrize(
    "c",
    [
        call("scroll", direction="forward"),  # out of enum (the T-0078 pattern)
        call("press_key", combo="Hyper+F13"),  # out of enum (D1)
        call("press_key", combo=["Ctrl+C"]),  # wrong type (D1: a string, not a list)
        call("press_key", keys=["Ctrl", "C"]),  # the pre-D1 shape: `combo` missing
        call("click_at", x=1000, y=5),  # out of range
        call("click_at", x="12", y=5),  # wrong type
        call("type_at", x=10, y=20, text="hi", press_enter="yes"),  # wrong type
        call("google_search"),  # missing required
        call("navigate", url="https://example.com"),  # not a v1 tool (Q1: no navigate)
    ],
    ids=[
        "scroll-out-of-enum",
        "press_key-out-of-enum",
        "press_key-combo-wrong-type",
        "press_key-pre-d1-keys",
        "click_at-x-out-of-range",
        "click_at-x-wrong-type",
        "type_at-press_enter-wrong-type",
        "google_search-missing-query",
        "unknown-tool",
    ],
)
def test_schema_valid_rejects_invalid_calls(c: dict[str, Any]) -> None:
    assert score_mod().schema_valid(c, toolset()) is False


def test_schema_valid_rejects_a_valid_arguments_object_under_the_wrong_name() -> None:
    """Adversarial: ``{x, y}`` is a valid click_at, not a valid scroll."""
    assert score_mod().schema_valid(call("scroll", x=5, y=5), toolset()) is False


def _close_additional_properties(raw: Any) -> None:
    for d in declarations(raw):
        params = d["parameters"] if "parameters" in d else d["function"]["parameters"]
        params["additionalProperties"] = False


def test_extra_arguments_fail_when_additional_properties_is_false(tmp_path: Path) -> None:
    closed = tools_mod().load(write_mutated(tmp_path, _close_additional_properties))
    s = score_mod()
    assert s.schema_valid(call("scroll", direction="up"), closed) is True
    assert s.schema_valid(call("scroll", direction="up", speed=3), closed) is False


def test_shipped_declarations_are_valid_json_schema() -> None:
    for name, d in decl_by_name(raw_tools()).items():
        schema = params_of(d)
        jsonschema.validators.validator_for(schema).check_schema(schema)
        assert schema.get("type") == "object", name


# =========================================================================== criterion 3: exact


def exact(c: dict[str, Any], g: dict[str, Any]) -> bool:
    res = score_mod().exact(c, g)
    assert isinstance(res, bool), f"exact() must return a bool, got {res!r}"
    return res


@pytest.mark.parametrize(
    ("query", "ok"),
    [
        ("weather in Paris tomorrow", True),
        ("WEATHER IN PARIS", True),  # case-insensitive
        ("  paris\tweather\n", True),  # surrounding whitespace, tabs, newlines
        ("weather in London", False),  # a required keyword missing
        ("paris", False),
    ],
)
def test_google_search_keywords(query: str, ok: bool) -> None:
    g = gt("google_search", keywords=["weather", "Paris"])
    assert exact(call("google_search", query=query), g) is ok


def test_google_search_keyword_internal_whitespace_collapses() -> None:
    """K8 (reviewer ruling): runs of whitespace inside the query match a single space in a keyword."""
    g = gt("google_search", keywords=["new york"])
    assert exact(call("google_search", query="new\t  york weather"), g) is True
    # Guard width: collapsing is not deleting -- "newyork" is not "new york".
    assert exact(call("google_search", query="newyork weather"), g) is False


def test_google_search_keyword_match_is_nfc() -> None:
    g = gt("google_search", keywords=[unicodedata.normalize("NFC", "Café")])
    nfd_query = unicodedata.normalize("NFD", "best café near me")
    assert exact(call("google_search", query=nfd_query), g) is True
    # Guard width: NFC is not accent-stripping.
    assert exact(call("google_search", query="best cafe near me"), g) is False


def test_google_search_name_must_match() -> None:
    g = gt("google_search", keywords=["weather"])
    assert exact(call("type_at", x=1, y=1, text="weather", press_enter=True), g) is False


_BBOX = [100, 200, 300, 400]


@pytest.mark.parametrize(
    ("x", "y", "ok"),
    [
        (100, 200, True),  # inclusive corner
        (300, 400, True),  # inclusive corner
        (100, 400, True),
        (200, 300, True),
        (99, 300, False),
        (301, 300, False),
        (200, 199, False),
        (200, 401, False),
    ],
)
def test_click_at_bbox_inclusive(x: int, y: int, ok: bool) -> None:
    assert exact(call("click_at", x=x, y=y), gt("click_at", bbox=_BBOX)) is ok


@pytest.mark.parametrize(
    ("x", "y", "text", "enter", "ok"),
    [
        (150, 250, "hello world", True, True),
        (100, 400, "  hello world \n", True, True),  # equal after strip; inclusive edge
        (150, 250, "Hello world", True, False),  # strip, not case-fold
        (150, 250, "hello world", False, False),  # press_enter differs
        (99, 250, "hello world", True, False),  # outside the bbox
    ],
)
def test_type_at(x: int, y: int, text: str, enter: bool, ok: bool) -> None:
    g = gt("type_at", bbox=_BBOX, text="hello world", press_enter=True)
    assert exact(call("type_at", x=x, y=y, text=text, press_enter=enter), g) is ok


@pytest.mark.parametrize(("direction", "ok"), [("down", True), ("up", False)])
def test_scroll_direction(direction: str, ok: bool) -> None:
    assert exact(call("scroll", direction=direction), gt("scroll", direction="down")) is ok


@pytest.mark.parametrize(
    ("combo", "ok"),
    [
        ("Ctrl+C", True),
        ("Ctrl+V", False),
        ("ctrl+c", False),  # D1: the combo *string* is compared, no case-folding
        ("C", False),
    ],
)
def test_press_key_compares_the_combo_string(combo: str, ok: bool) -> None:
    assert exact(call("press_key", combo=combo), gt("press_key", combo="Ctrl+C")) is ok


@pytest.mark.parametrize(
    ("c", "g"),
    [
        # Adversarial: a call valid for a *different* tool whose arguments would satisfy the case.
        (call("click_at", x=150, y=250), gt("type_at", bbox=_BBOX, text="hi", press_enter=False)),
        (
            call("type_at", x=150, y=250, text="hi", press_enter=False),
            gt("click_at", bbox=_BBOX),
        ),
        (call("scroll", direction="down"), gt("press_key", combo="Down")),
    ],
    ids=["click-for-type", "type-for-click", "scroll-for-press_key"],
)
def test_a_call_for_a_different_tool_is_not_exact(c: dict[str, Any], g: dict[str, Any]) -> None:
    assert exact(c, g) is False


def test_parse_call_stays_strict_on_a_closing_tag() -> None:
    """T-0082a criterion 7 (negative): the strip lives in the router only; the scorer still refuses a call followed
    by ``</cc>`` with the reason ``"not JSON"``, and the same call without the tag still parses (so the refusal is
    the tag's, not the call's). Byte-identity of ``score.py`` is pinned in ``tests/cd/unit/test_twopass_rescore.py``."""
    s = score_mod()
    text = '{"name": "scroll", "arguments": {"direction": "up"}}'
    assert s.parse_call(text + "</cc>") == s.ParseFailure("not JSON")
    assert s.parse_call(text + "</cc>\n") == s.ParseFailure("not JSON")
    assert call_parts(s.parse_call(text)) == ("scroll", {"direction": "up"})


# =========================================================================== T-0086: the plausible-query rule

#: T-0086 criterion 2: the six reasons, in the documented order (the first failing one is returned).
T86_REASONS = ("empty", "too long", "control character", "markup", "tool syntax", "narration")
#: The T-0084f review's query, verbatim (docs/tasks/T-0086.md, Goal).
ZERMATT = (
    " best hiking trails near Zermatt. I should use the `google_search` tool for this.1callcall:"
    "google_search{query:\\[best hiking trails near Zermatt"
)


def query_problem(query: str) -> str | None:
    """``score.query_problem`` (T-0086 criterion 2); a clear assertion while it is missing."""
    fn = getattr(score_mod(), "query_problem", None)
    assert callable(fn), (
        "T-0086 not implemented: local_astra.toolcall.score.query_problem is missing"
    )
    res = fn(query)
    assert res is None or res in T86_REASONS, (
        f"query_problem({query!r}) -> {res!r}: not a documented reason"
    )
    return res


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def _search_exact(query: str, keywords: list[str]) -> bool:
    return exact(call("google_search", query=query), gt("google_search", keywords=keywords))


def test_t0086_existing_whitespace_row_is_still_exact() -> None:
    """Criterion 2 (guard width): tabs and newlines are whitespace after normalisation, so the T-0080a row
    ``"  paris\\tweather\\n"`` has no problem and stays exact (the control-character check runs after normalisation)."""
    assert query_problem("  paris\tweather\n") is None
    assert _search_exact("  paris\tweather\n", ["weather", "Paris"]) is True


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("", "empty"),
        ("   \t\n ", "empty"),
        (_words(33), "too long"),
        ("weather\x00lisbon", "control character"),
        ("weather lisbon\x07", "control character"),
        ("weather​lisbon", "control character"),  # ZERO WIDTH SPACE, Cf
        ("﻿weather lisbon", "control character"),  # BOM, Cf
        ("weather {lisbon}", "markup"),
        ("call:weather lisbon", "tool syntax"),
        ("weather lisbon. I will search for it", "narration"),
    ],
    ids=[
        "empty",
        "whitespace-only",
        "33-words",
        "nul",
        "bel",
        "zwsp-cf",
        "bom-cf",
        "braces",
        "call-colon",
        "i-will-search",
    ],
)
def test_t0086_one_row_per_reason(query: str, reason: str) -> None:
    """Criterion 2: each reason, hand-picked so that no earlier reason applies."""
    assert query_problem(query) == reason


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        (_words(32), None),
        (_words(33), "too long"),
        (
            "  " + "\t ".join(f"w{i}" for i in range(32)) + "\n",
            None,
        ),  # whitespace runs collapse first
        (
            " ".join(f"w{i}" for i in range(31)) + " w31　w32",
            "too long",
        ),  # IDEOGRAPHIC SPACE splits
    ],
    ids=["32-words", "33-words", "32-words-in-whitespace-runs", "33-with-ideographic-space"],
)
def test_t0086_the_32_word_boundary(query: str, reason: str | None) -> None:
    """Criterion 2: "more than 32 space-separated words" on the normalised text -- 32 passes, 33 fails, counted
    after whitespace runs collapse (both directions)."""
    assert query_problem(query) == reason


@pytest.mark.parametrize(
    "sep",
    ["\t", "\n", "\r\n", "\x0b", "\x0c", "\x1c", "\x1f", "\x85", " ", " "],
    ids=["tab", "lf", "crlf", "vt", "ff", "fs", "us", "nel", "nbsp", "line-sep"],
)
def test_t0086_whitespace_control_characters_are_whitespace(sep: str) -> None:
    """Criterion 2 (too wide): Cc characters that ``str`` counts as whitespace (tab, LF, CR, VT, FF, the
    separators U+001C-U+001F, NEL U+0085) become a space during normalisation, so "control character" -- checked
    after normalisation -- never fires on them. A check on the raw query would reject each of these rows."""
    q = f"weather{sep}lisbon{sep}"
    assert query_problem(q) is None
    assert _search_exact(q, ["weather", "lisbon"]) is True


@pytest.mark.parametrize("ch", list("{}[]<>\\|`"), ids=lambda c: f"U+{ord(c):04X}")
def test_t0086_every_markup_character(ch: str) -> None:
    """Criterion 2: each of ``{ } [ ] < > \\ | ` `` is markup (human decision 2: the strict set for v1)."""
    assert query_problem(f"weather {ch} lisbon") == "markup"
    assert query_problem(f"weather{ch}lisbon") == "markup"


@pytest.mark.parametrize(
    "token",
    [
        "call:",
        "CALL:",
        "google_search",
        "Google_Search",
        "click_at",
        "type_at",
        "press_key",
    ],
)
def test_t0086_tool_syntax(token: str) -> None:
    """Criterion 2 as amended by ruling S5: ``call:`` or an underscore v1 tool name (not ``scroll``), case-folded."""
    assert query_problem(f"weather lisbon {token} now") == "tool syntax"
    assert query_problem(f"{token}weather lisbon") == "tool syntax"


@pytest.mark.parametrize(
    "query",
    [
        "call me maybe lyrics",
        "google search tips",
        "type at speed test",
        "click here",
        "infinite scroll css",  # ruling S5: scroll is an ordinary word
        "scroll lock key",
        "scroll down weather lisbon",
    ],
)
def test_t0086_tool_syntax_is_not_too_wide(query: str) -> None:
    """Criterion 2 (too wide): ``call`` without the colon and tool names written with a space are ordinary words."""
    assert query_problem(query) is None


_CLAUSES = (
    "i will",
    "i'll",
    "i should",
    "i must",
    "i need to",
    "i have to",
    "i am going to",
    "i'm going to",
    "let me",
)
_VERBS = ("search", "google", "look up", "use", "call", "query")


@pytest.mark.parametrize("verb", _VERBS)
@pytest.mark.parametrize("clause", _CLAUSES)
def test_t0086_every_narration_clause_and_verb(clause: str, verb: str) -> None:
    """Criterion 2: every listed first-person agent clause followed by every search-act verb (then a word end) is
    narration, anywhere in the query and in any case."""
    assert query_problem(f"weather lisbon. {clause} {verb} for it") == "narration"
    assert query_problem(f"{clause.upper()} {verb.upper()}") == "narration"


@pytest.mark.parametrize(
    "query",
    [
        "I’ll search for X",
        "I’m going to google it",
        "weather lisbon. I  will\t search it",  # whitespace runs inside the clause collapse
        " information. looksI will search for the weather forecast in Lisbon for tomorrow.",  # not left-anchored
    ],
    ids=["u2019-ill", "u2019-im", "collapsed-whitespace", "glued-on-the-left"],
)
def test_t0086_narration_variants(query: str) -> None:
    """Criterion 2: U+2019 counts as ``'``; the clause is matched on normalised text; it is not anchored to a word
    start on the left."""
    assert query_problem(query) == "narration"


@pytest.mark.parametrize(
    "query",
    [
        "i will survive lyrics",
        "how do i replace a bike chain",
        "I need a hero lyrics",
        "i will searchlight repair",  # the verb needs a word end
        "let me go lyrics",
        "i'll be there for you lyrics",
        "i must have been blind lyrics",
    ],
)
def test_t0086_user_style_first_person_queries_are_not_narration(query: str) -> None:
    """Criterion 2 (too wide): first-person queries a user types are not narration -- the clause must be followed
    by a search-act verb and a word end."""
    assert query_problem(query) is None


@pytest.mark.parametrize(
    ("query", "keywords"),
    [
        ("is the Louvre open on Tuesdays?", ["louvre"]),
        ('"bohemian rhapsody" lyrics', ["bohemian rhapsody"]),
        ("site:github.com htmx", ["htmx"]),
        ("c++ tutorial", ["c++"]),
        ("easyJet's cabin baggage", ["easyjet", "baggage"]),
        ("Zürich weather", ["zürich", "weather"]),
        ("東京 天気", ["東京"]),
        ("i will survive lyrics", ["survive"]),
        ("how do I replace a bike chain", ["replace", "chain"]),
        ("I need a hero lyrics", ["hero"]),
    ],
)
def test_t0086_plausible_queries_stay_exact(query: str, keywords: list[str]) -> None:
    """Criterion 2 (too wide): the task's allowed rows have no problem and score exact with their keywords."""
    assert query_problem(query) is None
    assert _search_exact(query, keywords) is True


@pytest.mark.parametrize(
    ("query", "keywords", "reason"),
    [
        (ZERMATT, ["hiking", "zermatt"], "markup"),
        (
            " information. looksI will search for the weather forecast in Lisbon for tomorrow.",
            ["weather", "lisbon"],
            "narration",
        ),
        ("I will search for the htmx GitHub repository.", ["htmx"], "narration"),
        ("I’ll search for X", ["x"], "narration"),
        ("weather​ lisbon", ["weather", "lisbon"], "control character"),
        ("weather lisbon\x00", ["weather", "lisbon"], "control character"),
        ("weather lisbon </cc>", ["weather", "lisbon"], "markup"),
        ("use google_search for weather lisbon", ["weather", "lisbon"], "tool syntax"),
        ("call:weather lisbon", ["weather", "lisbon"], "tool syntax"),
    ],
    ids=[
        "zermatt",
        "looks-i-will",
        "htmx",
        "u2019",
        "zwsp",
        "nul",
        "cc-close",
        "use-google_search",
        "call-colon",
    ],
)
def test_t0086_adversarial_queries_are_not_exact_with_keywords_present(
    query: str, keywords: list[str], reason: str
) -> None:
    """Criterion 1 (the T-0084f failure mode): every keyword is present -- the frozen 4c7dfdc rule scores the call
    exact -- yet the new rule does not, for the stated reason."""
    from tests.unit import _frozen_score_4c7dfdc as old

    c, g = call("google_search", query=query), gt("google_search", keywords=keywords)
    assert old.exact(c, g) is True, "fixture: the keywords are not all present"
    assert query_problem(query) == reason
    assert exact(c, g) is False


def test_t0086_a_bare_nul_is_a_control_character() -> None:
    """Criterion 2: ``"\\x00"`` is not whitespace, so it survives normalisation and is a control character."""
    assert query_problem("\x00") == "control character"


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("weather [lisbon] i will search", "markup"),  # markup + narration (the task's row)
        (_words(33) + " \x00", "too long"),  # too long + control
        ("\x00 " + _words(40), "too long"),
        ("weather\x00 {lisbon}", "control character"),  # control + markup
        ("call:{weather}", "markup"),  # markup + tool syntax
        ("i will search google_search", "tool syntax"),  # tool syntax + narration
        ("let me use call: now", "tool syntax"),
    ],
)
def test_t0086_reason_order(query: str, reason: str) -> None:
    """Criterion 2: the first failing reason in the documented order is returned."""
    assert query_problem(query) == reason


def test_t0086_keywords_still_required() -> None:
    """Criterion 1: a plausible query still needs every keyword (the existing rule holds, unchanged)."""
    assert query_problem("weather in london") is None
    assert _search_exact("weather in london", ["weather", "paris"]) is False


@pytest.mark.parametrize(
    "bad", [None, 5, ["weather paris"], {"q": "weather paris"}, b"weather paris"]
)
def test_t0086_malformed_query_scores_false_and_never_raises(bad: Any) -> None:
    """Criterion 1: malformed arguments score ``False`` and never raise."""
    assert (
        exact(
            {"name": "google_search", "arguments": {"query": bad}},
            gt("google_search", keywords=["weather"]),
        )
        is False
    )
    assert (
        exact(
            {"name": "google_search", "arguments": None}, gt("google_search", keywords=["weather"])
        )
        is False
    )


def test_t0086_query_problem_is_public_and_documented() -> None:
    """Criteria 2 and 5: ``query_problem`` is exported and its docstring (or the module's) states the rule: every
    reason and the 32-word limit."""
    s = score_mod()
    assert "query_problem" in getattr(s, "__all__", ()), "query_problem is not in score.__all__"
    doc = (getattr(getattr(s, "query_problem", None), "__doc__", None) or "") + (s.__doc__ or "")
    missing = [r for r in (*T86_REASONS, "32") if r not in doc]
    assert missing == [], f"the docstring does not state: {missing}"
