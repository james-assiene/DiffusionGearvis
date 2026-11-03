"""T-0080a criteria 4 and 5: tool-call grammars with the thought variants T1/T2/T4, by host replay.

``decoding/toolcall.py::compile_toolcall(tools, *, scope, thought, whitespace="json")`` over the v1 tools
(``local_astra.toolcall.tools.load()``), for every scope (``"union"`` and each tool) and every thought
variant. Token sequences are hand-written text tokenised by the pinned HF ``tokenizer.json`` (the oracle in
``tests/cd/_compile_support.py``) with the single-id markers written as ids:
``<|channel>`` 100, ``<channel|>`` 101, ``\\n`` 107, ``thought`` 45518, stops ``{1, 106, 50}``, ``<|turn>``
105, PAD 0. Acceptance is decided by the host replay ``eval/cd/replay.accepts`` (criterion 5) and checked
against the independent set-of-states ``Walker`` of ``_compile_support``; the two must agree on every row.

Every automaton is ``FA · STOP · Σ*`` (T-0063), so an accepted sequence ends with a stop id and a stop
inside the call is a rejection.

The languages, from the task and the umbrella's "Thought handling":

- T1 = ``100 · Σ_name{1,8} · 107 · FREE* · 101 · ["<cc>"] · CALL``, FREE = every id except
  ``{PAD} ∪ stops ∪ {100, 101, 105}``;
- T2 = ``100 · Σ_name{1,8} · 107 · 101 · ["<cc>"] · CALL`` (the existing ``prepend_channel_header``);
- T4 = ``CALL`` (the prompt already ends with ``<cc>``, so neither a header nor the tag is admitted);
- CALL = ``{"name": <const>, "arguments": <the tool's parameters schema>}`` in JSON whitespace, key
  order as declared.

Regime: CPU; tokenizer.json sha256 cc8d3a0c...; model rev 2ea8372...; JSON whitespace; v1 tools file as
shipped (its sha256 is printed with the size report).

K-notes: K10 ``compile_toolcall`` takes the ``ToolSet`` returned by ``tools.load()``; K11
``report_sizes()`` takes no required argument and prints one line per (scope, thought) holding the scope,
the thought, |S| and the bucket; K12 an unknown scope or thought raises ``ValueError``/``KeyError``;
K13 whitespace between ``<cc>`` and ``{`` is not specified and not asserted.

Human decisions after the size report (2026-10-08):

- D1 ``press_key(combo)``: one string from a fixed enum (``_toolcall_support.D1_NAMED_COMBOS`` plus the
  four arrows). Rows: every shipped combo is accepted; an out-of-enum string, a list and the pre-D1
  ``keys`` shape are rejected.
- D2 ``scope="name"``: a name-only grammar that picks the tool when the draft names none. Its exact
  language is left open by D2 (``{"name": <v1 name>`` ..., or just the name enum), so the rows are
  form-agnostic: among the candidate renderings in ``NAME_FORMS`` there must be one form accepted for
  *every* v1 name, and no form may be accepted for a name outside v1. Thought variants: all three
  (decision of the test-author, K14): T1/T2 keep their header and optional ``<cc>`` because pass 2's
  prompt is pass 1's either way; T4 has neither, as for the per-tool grammars.
- Live-path gate: each tool and ``"name"`` under T1/T2/T4 has ``|S| <= 128`` (``report.LIVE_MAX_STATES``,
  "<= 128 states", so the bucket is at most 256). ``union`` is not on the live path; it is still
  compiled, encoded under the cap and reported.
"""

from __future__ import annotations

import functools
import json
from typing import Any

import pytest

from tests.cd._compile_support import encode, walker
from tests.unit._toolcall_support import combo_enum, need, shipped_params

PAD, EOS, TOOL_RESPONSE, OPEN, CLOSE, TURN_OPEN, TURN_CLOSE = 0, 1, 50, 100, 101, 105, 106
NL, THOUGHT = 107, 45518
TOOLS = ("google_search", "click_at", "type_at", "scroll", "press_key")
SCOPES = ("union", *TOOLS)
THOUGHTS = ("T1", "T2", "T4")
#: D2: the live path's scopes (each tool, and the name picker); ``union`` is reporting/eval only.
LIVE_SCOPES = (*TOOLS, "name")
ALL_SCOPES = (*SCOPES, "name")
#: docs/07 §0 and ``report.LIVE_MAX_STATES``: the project build gate for the live path.
LIVE_MAX_STATES = 128
#: ``contracts/cd.py::MAX_AUTOMATON_CHARS`` (T-0062 decision 4): 1 MiB of payload characters.
CAP_CHARS = 1 << 20
LADDER = (16, 32, 64, 128, 256, 512, 1024, 2048)


# --------------------------------------------------------------------------- code under test


def toolcall() -> Any:
    return need("local_astra.decoding.toolcall")


@functools.cache
def toolset() -> Any:
    return need("local_astra.toolcall.tools").load()


@functools.cache
def compiled(scope: str, thought: str) -> Any:
    return toolcall().compile_toolcall(toolset(), scope=scope, thought=thought)


def accepts(scope: str, thought: str, ids: list[int]) -> bool:
    from local_astra.eval.cd import replay

    a = compiled(scope, thought)
    got = replay.accepts(a, ids)
    oracle = walker(a).accepts(ids)
    assert got == oracle, f"host replay ({got}) and the walker ({oracle}) disagree on {ids}"
    return got


# --------------------------------------------------------------------------- hand-written calls


def args_in_declared_order(name: str, values: dict[str, Any]) -> dict[str, Any]:
    order = list(shipped_params(name)["properties"])
    missing = set(values) - set(order)
    assert not missing, f"{name}: rows use {sorted(missing)} which the declaration lacks"
    return {k: values[k] for k in order if k in values}


def call_text(name: str, values: dict[str, Any]) -> str:
    """``{"name": ..., "arguments": {...}}`` with ``json.dumps``'s separators (JSON whitespace)."""
    args = args_in_declared_order(name, values) if name in TOOLS else values
    return json.dumps({"name": name, "arguments": args}, ensure_ascii=False)


def valid_values(name: str) -> dict[str, Any]:
    if name == "google_search":
        return {"query": "weather in Paris"}
    if name == "click_at":
        return {"x": 999, "y": 0}
    if name == "type_at":
        return {"x": 10, "y": 20, "text": "hello", "press_enter": True}
    if name == "scroll":
        return {"direction": "down"}
    if name == "press_key":
        return {"combo": "Ctrl+C"}
    raise AssertionError(name)


def in_scope(scope: str) -> str:
    """A tool the scope admits (union: click_at, which carries the range rows)."""
    return "click_at" if scope == "union" else scope


def out_of_scope(scope: str) -> tuple[str, dict[str, Any]]:
    if scope == "union":
        return "navigate", {"url": "https://example.com"}  # Q1: no navigate in v1
    other = "scroll" if scope != "scroll" else "google_search"
    return other, valid_values(other)


REAL_THOUGHT = "The user wants the weather.\nI will search for it.\n\nThen answer briefly."
EMPTY_HEADER = [OPEN, THOUGHT, NL, CLOSE]
#: ``<cc>`` as four single-character pieces ``<``, ``c``, ``c``, ``>``: another tokenisation of the literal.
CC_SPLIT = [236820, 236755, 236755, 236813]


def t1_header() -> list[int]:
    """Tokenised lazily: the tokenizer is read inside a row, never at collection."""
    return [OPEN, THOUGHT, NL, *encode(REAL_THOUGHT), CLOSE]


def _cc() -> list[int]:
    return list(encode("<cc>"))


def head(thought: str, *, cc: bool = True) -> list[int]:
    if thought == "T1":
        return t1_header() + (_cc() if cc else [])
    if thought == "T2":
        return EMPTY_HEADER + (_cc() if cc else [])
    return []


def seq(thought: str, text: str, *, cc: bool = True, stop: int = EOS) -> list[int]:
    return [*head(thought, cc=cc), *encode(text), stop]


# =========================================================================== criterion 5: accepted


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("scope", SCOPES)
def test_accepts_a_valid_call_in_scope(scope: str, thought: str) -> None:
    tool = in_scope(scope)
    assert accepts(scope, thought, seq(thought, call_text(tool, valid_values(tool))))


@pytest.mark.parametrize("tool", TOOLS)
def test_union_accepts_every_v1_tool(tool: str) -> None:
    for thought in THOUGHTS:
        assert accepts("union", thought, seq(thought, call_text(tool, valid_values(tool)))), thought


@pytest.mark.parametrize("thought", ["T1", "T2"])
def test_cc_is_optional_after_the_header(thought: str) -> None:
    text = call_text("scroll", valid_values("scroll"))
    assert accepts("scroll", thought, seq(thought, text, cc=False))
    assert accepts("scroll", thought, seq(thought, text, cc=True))


@pytest.mark.parametrize("thought", ["T1", "T2"])
def test_cc_admits_every_tokenisation(thought: str) -> None:
    ids = [
        *head(thought, cc=False),
        *CC_SPLIT,
        *encode(call_text("scroll", {"direction": "up"})),
        EOS,
    ]
    assert accepts("scroll", thought, ids)


@pytest.mark.parametrize("stop", [EOS, TURN_CLOSE, TOOL_RESPONSE])
def test_every_stop_id_ends_the_call(stop: int) -> None:
    text = call_text("google_search", valid_values("google_search"))
    assert accepts("google_search", "T2", seq("T2", text, stop=stop))


def test_t1_accepts_a_real_thought_with_newlines() -> None:
    assert NL in encode(REAL_THOUGHT)
    text = call_text("google_search", valid_values("google_search"))
    assert accepts("google_search", "T1", seq("T1", text))


def test_t1_free_region_admits_json_and_tag_text_inside_the_thought() -> None:
    """Guard width: FREE is every id but the reserved ones, so tool-ish text in the thought is fine."""
    thought = 'Maybe <cc>{"name": "click_at", "arguments": {"x": 1}} -- no, search instead.'
    ids = [OPEN, THOUGHT, NL, *encode(thought), CLOSE, *_cc()]
    ids += [*encode(call_text("google_search", valid_values("google_search"))), EOS]
    assert accepts("google_search", "T1", ids)
    assert accepts("union", "T1", ids)


def test_t1_accepts_an_empty_thought() -> None:
    text = call_text("click_at", {"x": 0, "y": 999})
    assert accepts("click_at", "T1", [*EMPTY_HEADER, *_cc(), *encode(text), EOS])


def test_t4_accepts_the_call_at_position_zero() -> None:
    ids = [*encode(call_text("click_at", {"x": 0, "y": 0})), EOS]
    assert ids[0] != OPEN
    assert accepts("click_at", "T4", ids)


#: Every bounded coordinate (Q1: 0-999, normalised to the attached frame), as (tool, argument).
COORDS = [("click_at", "x"), ("click_at", "y"), ("type_at", "x"), ("type_at", "y")]


def with_coord(tool: str, arg: str, value: int) -> dict[str, Any]:
    """The tool's valid call with one coordinate replaced (the other stays at a valid 5)."""
    values = {**valid_values(tool), "x": 5, "y": 5}
    values[arg] = value
    return values


@pytest.mark.parametrize(("tool", "arg"), COORDS)
@pytest.mark.parametrize("v", [0, 9, 10, 99, 100, 500, 998, 999])
def test_coordinates_accept_the_whole_range(tool: str, arg: str, v: int) -> None:
    assert accepts(tool, "T4", [*encode(call_text(tool, with_coord(tool, arg, v))), EOS])


# =========================================================================== criterion 5: rejected


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("scope", SCOPES)
def test_rejects_a_call_outside_the_scope(scope: str, thought: str) -> None:
    name, values = out_of_scope(scope)
    assert not accepts(scope, thought, seq(thought, call_text(name, values)))


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("scope", SCOPES)
def test_rejects_a_missing_required_argument(scope: str, thought: str) -> None:
    tool = in_scope(scope)
    values = valid_values(tool)
    required = list(shipped_params(tool).get("required", []))
    assert required, f"{tool} declares no required argument"
    assert required[0] in values, f"{tool}: the row's valid call lacks required {required[0]!r}"
    values.pop(required[0])
    assert not accepts(scope, thought, seq(thought, call_text(tool, values)))


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("in_union", [False, True], ids=["own-scope", "union"])
@pytest.mark.parametrize(("tool", "arg"), COORDS)
@pytest.mark.parametrize("v", [1000, 1500, 9999, -1])
def test_rejects_an_out_of_range_coordinate(
    tool: str, arg: str, in_union: bool, thought: str, v: int
) -> None:
    """Each of the four bounds separately (a grammar bounding only click_at.x must fail here)."""
    scope = "union" if in_union else tool
    assert not accepts(scope, thought, seq(thought, call_text(tool, with_coord(tool, arg, v))))


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("stop", [EOS, TURN_CLOSE])
def test_rejects_a_stop_id_inside_a_string(scope: str, thought: str, stop: int) -> None:
    tool = in_scope(scope)
    text = call_text(tool, valid_values(tool))
    cut = text.index(tool) + 3  # inside the name string
    ids = [*head(thought), *encode(text[:cut]), stop, *encode(text[cut:]), EOS]
    assert not accepts(scope, thought, ids)


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("stop", [EOS, TURN_CLOSE])
def test_rejects_a_stop_id_inside_the_query(thought: str, stop: int) -> None:
    pre = '{"name": "google_search", "arguments": {"query": "weather'
    post = ' in Paris"}}'
    assert accepts("google_search", thought, seq(thought, pre + post))  # the unbroken call is fine
    ids = [*head(thought), *encode(pre), stop, *encode(post), EOS]
    assert not accepts("google_search", thought, ids)


def test_t1_rejects_a_second_channel_open_in_the_thought() -> None:
    ids = [OPEN, THOUGHT, NL, *encode("first idea"), OPEN, *encode("second"), CLOSE, *_cc()]
    ids += [*encode(call_text("scroll", {"direction": "up"})), EOS]
    assert not accepts("scroll", "T1", ids)


def test_t1_free_region_excludes_the_channel_close() -> None:
    """``100 thought 107 a 101 b 101 <cc> CALL``: FREE must not contain 101, so the thought ends at the
    first 101 and ``b 101`` is not a call. Companion: the same thought without ``b 101`` is accepted."""
    call = [*_cc(), *encode(call_text("scroll", {"direction": "up"})), EOS]
    good = [OPEN, THOUGHT, NL, *encode("a"), CLOSE, *call]
    bad = [OPEN, THOUGHT, NL, *encode("a"), CLOSE, *encode("b"), CLOSE, *call]
    assert accepts("scroll", "T1", good)
    assert not accepts("scroll", "T1", bad)
    assert not accepts("union", "T1", bad)


def test_t1_rejects_a_second_thought_block() -> None:
    ids = [*t1_header(), OPEN, THOUGHT, NL, CLOSE, *_cc()]
    ids += [*encode(call_text("scroll", {"direction": "up"})), EOS]
    assert not accepts("scroll", "T1", ids)


@pytest.mark.parametrize("marker", [CLOSE, OPEN])
def test_t1_rejects_a_channel_marker_inside_the_call(marker: int) -> None:
    pre = '{"name": "google_search", "arguments": {"query": "weather'
    ids = [*t1_header(), *_cc(), *encode(pre), marker, *encode(' in Paris"}}'), EOS]
    assert not accepts("google_search", "T1", ids)


@pytest.mark.parametrize("reserved", [PAD, TURN_OPEN, EOS, TURN_CLOSE, TOOL_RESPONSE])
def test_t1_free_region_excludes_reserved_ids(reserved: int) -> None:
    ids = [OPEN, THOUGHT, NL, *encode("let me"), reserved, *encode("think"), CLOSE, *_cc()]
    ids += [*encode(call_text("scroll", {"direction": "up"})), EOS]
    assert not accepts("scroll", "T1", ids)


@pytest.mark.parametrize("scope", ["union", "scroll"])
def test_t2_rejects_any_thought_text(scope: str) -> None:
    text = call_text("scroll", {"direction": "up"})
    for body in ("hmm", "The user wants to scroll.\nOk.", " "):
        ids = [OPEN, THOUGHT, NL, *encode(body), CLOSE, *_cc(), *encode(text), EOS]
        assert not accepts(scope, "T2", ids), body


def test_t2_requires_the_header() -> None:
    text = call_text("scroll", {"direction": "up"})
    assert not accepts("scroll", "T2", [*encode(text), EOS])


@pytest.mark.parametrize("scope", ["union", "click_at"])
def test_t4_rejects_a_leading_header(scope: str) -> None:
    text = call_text("click_at", {"x": 1, "y": 2})
    assert not accepts(scope, "T4", [*EMPTY_HEADER, *encode(text), EOS])
    assert not accepts(scope, "T4", [*t1_header(), *encode(text), EOS])


def test_t4_rejects_a_leading_cc() -> None:
    text = call_text("click_at", {"x": 1, "y": 2})
    assert not accepts("click_at", "T4", [*_cc(), *encode(text), EOS])


def test_rejects_scroll_out_of_enum() -> None:
    for thought in THOUGHTS:
        assert not accepts(
            "scroll", thought, seq(thought, call_text("scroll", {"direction": "forward"}))
        )


@pytest.mark.parametrize("thought", THOUGHTS)
def test_press_key_accepts_every_shipped_combo(thought: str) -> None:
    enum = combo_enum()
    assert "Ctrl+C" in enum
    rejected = [
        c
        for c in enum
        if not accepts("press_key", thought, seq(thought, call_text("press_key", {"combo": c})))
    ]
    assert not rejected, f"{thought}: shipped combos the grammar rejects: {rejected}"


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize(
    "values",
    [
        {"combo": "Hyper+F13"},  # out of enum
        {"combo": "Ctrl"},  # a strict prefix of enum members
        {"combo": "Ctrl+C "},  # an enum member plus a trailing space
        {"combo": ["Ctrl+C"]},  # a list, not a string
        {"keys": ["Ctrl", "C"]},  # the pre-D1 shape
    ],
    ids=["out-of-enum", "prefix", "trailing-space", "list", "pre-d1-keys"],
)
def test_press_key_rejects_anything_but_one_combo(thought: str, values: dict[str, Any]) -> None:
    combo = values.get("combo")
    assert not (isinstance(combo, str) and combo in combo_enum()), "row premise: not a member"
    text = json.dumps({"name": "press_key", "arguments": values})
    assert not accepts("press_key", thought, seq(thought, text))


# --------------------------------------------------------------------------- D2: scope="name"

#: Candidate renderings of the name picker's output, ``"{}"`` = the tool name.
NAME_FORMS = {
    "object": '{{"name": "{}"}}',
    "open-object": '{{"name": "{}"',
    "open-object-comma": '{{"name": "{}", ',
    "string": '"{}"',
    "bare": "{}",
}
NOT_V1 = ("navigate", "click", "google_searc", "press_keys", "Click_at", "scroll_up", "go_back")


def name_seq(thought: str, form: str, name: str, *, stop: int = EOS) -> list[int]:
    return seq(thought, NAME_FORMS[form].format(name), stop=stop)


def name_forms_accepted(thought: str, name: str) -> set[str]:
    return {f for f in NAME_FORMS if accepts("name", thought, name_seq(thought, f, name))}


@functools.cache
def name_form(thought: str) -> str:
    """The rendering the name picker accepts for every v1 name (fails the row if there is none)."""
    common = set(NAME_FORMS)
    per_name = {}
    for tool in TOOLS:
        per_name[tool] = name_forms_accepted(thought, tool)
        common &= per_name[tool]
    assert common, (
        f"{thought}: no rendering of NAME_FORMS is accepted for every v1 name: {per_name}"
    )
    return sorted(common)[0]


@pytest.mark.parametrize("thought", THOUGHTS)
def test_name_scope_accepts_every_v1_name(thought: str) -> None:
    form = name_form(thought)
    for tool in TOOLS:
        assert accepts("name", thought, name_seq(thought, form, tool)), (form, tool)


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("other", NOT_V1)
def test_name_scope_rejects_names_outside_v1(thought: str, other: str) -> None:
    name_form(thought)  # the language must exist before its rejections mean anything
    assert name_forms_accepted(thought, other) == set(), other


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("stop", [EOS, TURN_CLOSE])
def test_name_scope_rejects_a_stop_inside_the_name(thought: str, stop: int) -> None:
    form = name_form(thought)
    text = NAME_FORMS[form].format("click_at")
    cut = text.index("click_at") + 3
    ids = [*head(thought), *encode(text[:cut]), stop, *encode(text[cut:]), EOS]
    assert not accepts("name", thought, ids)


def test_name_scope_thought_variants_keep_their_headers() -> None:
    t4 = name_form("T4")
    text = NAME_FORMS[t4].format("scroll")
    assert not accepts("name", "T4", [*EMPTY_HEADER, *encode(text), EOS])  # T4: no header
    t2 = name_form("T2")
    text2 = NAME_FORMS[t2].format("scroll")
    assert not accepts("name", "T2", [*encode(text2), EOS])  # T2: header required
    assert not accepts(
        "name", "T2", [OPEN, THOUGHT, NL, *encode("hmm"), CLOSE, *encode(text2), EOS]
    )
    t1 = name_form("T1")
    assert accepts(
        "name", "T1", [*t1_header(), *_cc(), *encode(NAME_FORMS[t1].format("scroll")), EOS]
    )


def test_rejects_an_unterminated_call() -> None:
    text = call_text("scroll", {"direction": "up"})
    assert not accepts("scroll", "T4", [*encode(text)])  # no stop id
    assert not accepts("scroll", "T4", [*encode(text[:-1]), EOS])  # missing the closing brace


# =========================================================================== criterion 4: sizes


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("scope", ALL_SCOPES)
def test_every_automaton_encodes_under_the_cap(scope: str, thought: str) -> None:
    from local_astra.decoding import artifact

    a = compiled(scope, thought)
    payload = artifact.encode_automaton(a)
    assert isinstance(payload, str) and len(payload) <= CAP_CHARS, (scope, thought, len(payload))
    assert int(a.n_states_bucket) in LADDER or int(a.n_states_bucket) > LADDER[-1]
    assert int(a.n_states) + 1 <= int(a.n_states_bucket)
    assert bool(a.is_dfa), f"{scope}/{thought} is not a DFA (eq. 8 would take its slow path)"
    print(
        f"T-0080a size: scope={scope} thought={thought} |S|={a.n_states} bucket={a.n_states_bucket}"
    )


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("scope", LIVE_SCOPES)
def test_live_path_automata_have_at_most_128_states(scope: str, thought: str) -> None:
    """D2's gate (supersedes the original "single-tool buckets <= 512" row)."""
    a = compiled(scope, thought)
    assert int(a.n_states) <= LIVE_MAX_STATES, (scope, thought, a.n_states, a.n_states_bucket)
    assert int(a.n_states_bucket) <= 256
    assert bool(a.is_dfa), (scope, thought)


def test_report_sizes_prints_every_scope_and_thought(capsys: pytest.CaptureFixture[str]) -> None:
    toolcall().report_sizes()
    out = capsys.readouterr().out
    lines = out.splitlines()
    missing = []
    for scope in ALL_SCOPES:
        for thought in THOUGHTS:
            a = compiled(scope, thought)
            want = (scope, thought, str(int(a.n_states)), str(int(a.n_states_bucket)))
            if not any(all(w in line for w in want) for line in lines):
                missing.append(want)
    assert not missing, f"report_sizes() omitted {missing}:\n{out}"
    with capsys.disabled():
        print("\n" + out)


@pytest.mark.parametrize(
    ("scope", "thought"), [("navigate", "T4"), ("union", "T3"), ("click_at", "t1")]
)
def test_unknown_scope_or_thought_is_refused(scope: str, thought: str) -> None:
    with pytest.raises((ValueError, KeyError)):
        toolcall().compile_toolcall(toolset(), scope=scope, thought=thought)


# =========================================================================== T-0084a: call_format
#
# T-0084a criterion 1 (API) and criterion 8 (negative). ``compile_toolcall(..., call_format="json")`` is
# the default and must keep every JSON payload byte-identical: the sha256 of
# ``artifact.encode_automaton(...)`` (the payload string, utf-8) for every (scope, thought) was measured
# by the T-0084a test-author at ``main`` = 1e06a24 *before* the implementation (CPU; tools_v1 sha256
# bbfc53cc...; tokenizer cc8d3a0c...; model rev 2ea8372; outlines-core 0.2.14; whitespace="json").

JSON_PAYLOAD_SHA256_AT_1E06A24 = {
    ("name", "T1"): "752c124f71a2672a60c40b802cbcf13a9193157d28144119cdb230a3f0da4c96",
    ("name", "T2"): "2af81b849a3ef7ca6dfaad3be0fbb88f99300aad1f4fad385aa55f5ed5ca74e0",
    ("name", "T4"): "85b37891a41b7694b989c477c9a45c3d63427d3d141fe38df33ac559cf5944a6",
    ("google_search", "T1"): "66502f77115f548e51507dece66d758fc560f162fec4e7b80b9fa0598775a5cd",
    ("google_search", "T2"): "df87a122f33ffe348c083f3964e4c9c6ff5e98f005be425fe6994196af54205b",
    ("google_search", "T4"): "1502f509d79d49f61dc3fd9fa5aecf46c4955b4d4ec8b94330121aff5ea28329",
    ("click_at", "T1"): "678fdbc8b7ed5aad3ce38910bfa2238663816ca03b73bbcc44316274319c0edd",
    ("click_at", "T2"): "34913caf3ec6c3903809c622c095d56f26f442eb4a31aaa9371aa85d1d8457e2",
    ("click_at", "T4"): "e073926aad55d4011e4fe14cce36604d17b837a3765c311fd01c0d0dab0b9336",
    ("type_at", "T1"): "3e96ae81911256417a8e533ef0835c0e724356451fdd3ccd7068f03798f783e3",
    ("type_at", "T2"): "120632bb61c5500f983939e33bb08e182b2d8cd78c7790bb1902e2a940e7f434",
    ("type_at", "T4"): "0d98baef814ddfc1088f0be9ecfac275405b5e0812fe578bbc3ba14bed81c1f6",
    ("scroll", "T1"): "b29b63be88f905aec46558dfa7af5c6d2a62cf8d7b154e4415d5327407d86e9f",
    ("scroll", "T2"): "9f19709c37127768af6148be1a637bc25bef9eec80b93558a776467bd2a58790",
    ("scroll", "T4"): "97e2d08ef50018a64b2d2351753d81e92ecb11dc32a39dc2fa7be1bcfd41ee2f",
    ("press_key", "T1"): "19fe941c1cb68fa04fd4c0ab86734a5a8590a64ca734de24d0b16c51131461a1",
    ("press_key", "T2"): "c7e0f469d03b7af4b71bfddc4abbbc5d98d68f701e95a620eee649b89de266c9",
    ("press_key", "T4"): "918ab0566487dba6beceb3d54aa564719855bd6b69ad813897cc284c3636251d",
    ("union", "T1"): "74f8a86bbf63d6ed7c5285537fab4fa8e1b365348008c6de0a52e3ca533b83b5",
    ("union", "T2"): "32b7db13c7138050a1d64bb0ed00d675e38554526c2c0ccbb80c91b925fd2d00",
    ("union", "T4"): "4c7330894263774ca8bf45365b7055e722475a3e4e1b91f6c38fde7d469e7e3f",
}
#: sha256 of ``pickle.dumps(vocab.build_vocabulary())`` (default arguments) at 1e06a24 (stable across
#: processes: checked in separate interpreters by the test-author).
VOCABULARY_PICKLE_SHA256_AT_1E06A24 = (
    "ba3695c5fc02fe2e2e5f427dff900c840a6aa79ee3848e1d122f96854900b665"
)


def _payload_sha(a: Any) -> str:
    import hashlib

    from local_astra.decoding import artifact

    return hashlib.sha256(artifact.encode_automaton(a).encode("utf-8")).hexdigest()


def test_the_json_pin_table_covers_every_scope_and_thought() -> None:
    assert set(JSON_PAYLOAD_SHA256_AT_1E06A24) == {(s, t) for s in ALL_SCOPES for t in THOUGHTS}


@pytest.mark.parametrize(("scope", "thought"), sorted(JSON_PAYLOAD_SHA256_AT_1E06A24))
def test_json_payloads_are_byte_identical_to_main(scope: str, thought: str) -> None:
    """Criterion 8: explicit ``call_format="json"`` gives the payload pinned at 1e06a24."""
    a = toolcall().compile_toolcall(toolset(), scope=scope, thought=thought, call_format="json")
    assert _payload_sha(a) == JSON_PAYLOAD_SHA256_AT_1E06A24[(scope, thought)], (scope, thought)


@pytest.mark.parametrize(
    ("scope", "thought"), [("scroll", "T4"), ("name", "T2"), ("press_key", "T1")]
)
def test_the_default_call_format_is_json(scope: str, thought: str) -> None:
    """Criterion 1: ``call_format`` defaults to ``"json"`` (no keyword = the pinned JSON payload)."""
    a = toolcall().compile_toolcall(toolset(), scope=scope, thought=thought)
    assert _payload_sha(a) == JSON_PAYLOAD_SHA256_AT_1E06A24[(scope, thought)]
    b = toolcall().compile_toolcall(toolset(), scope=scope, thought=thought, call_format="json")
    assert _payload_sha(b) == _payload_sha(a)


def test_the_grammar_vocabulary_default_is_unchanged() -> None:
    """Criterion 8: ``build_vocabulary()`` with default arguments is the vocabulary of 1e06a24, where the
    markers 48/49/52 still appear as their literal bytes (a native-only mapping must not leak here)."""
    import hashlib
    import pickle

    from local_astra.decoding import vocab

    voc = vocab.build_vocabulary()
    assert hashlib.sha256(pickle.dumps(voc)).hexdigest() == VOCABULARY_PICKLE_SHA256_AT_1E06A24
    for literal, tid in ((b"<|tool_call>", 48), (b"<tool_call|>", 49), (b'<|"|>', 52)):
        assert tid in (voc.get(literal) or []), (literal, tid)


@pytest.mark.parametrize("bad", ["JSON", "Native", "gemma4", "", "native ", "cc", None, 0])
def test_call_format_must_be_json_or_native(bad: Any) -> None:
    """Criterion 1: anything but ``"json"``/``"native"`` raises ``ValueError``; both valid values do not."""
    with pytest.raises(ValueError):
        toolcall().compile_toolcall(toolset(), scope="scroll", thought="T4", call_format=bad)
    for ok in ("json", "native"):
        toolcall().compile_toolcall(toolset(), scope="scroll", thought="T4", call_format=ok)


@pytest.mark.parametrize("first", ["native", "json"])
def test_the_cache_key_separates_the_formats(first: str) -> None:
    """Criterion 1: whichever format is compiled first, the other is not served from its cache entry,
    and the automata's ``schema_hash`` (the stamp the plugin keys on) differ."""
    tc = toolcall()
    other = "json" if first == "native" else "native"
    for scope, thought in (("click_at", "T2"), ("name", "T4")):
        a = tc.compile_toolcall(toolset(), scope=scope, thought=thought, call_format=first)
        b = tc.compile_toolcall(toolset(), scope=scope, thought=thought, call_format=other)
        j, n = (b, a) if first == "native" else (a, b)
        assert _payload_sha(j) == JSON_PAYLOAD_SHA256_AT_1E06A24[(scope, thought)]
        assert _payload_sha(n) != _payload_sha(j), (scope, thought)
        assert n.schema_hash != j.schema_hash, (scope, thought)
        again = tc.compile_toolcall(toolset(), scope=scope, thought=thought, call_format="native")
        assert _payload_sha(again) == _payload_sha(n)


def test_report_sizes_rows_cover_both_formats(capsys: pytest.CaptureFixture[str]) -> None:
    """Criterion 1/5: ``report_sizes()`` prints and returns a row per (call_format, scope, thought), each
    row naming its ``call_format``, with the |S| of the automaton that format compiles."""
    tc = toolcall()
    rows = tc.report_sizes()
    out = capsys.readouterr().out
    keyed = {(r["call_format"], r["scope"], r["thought"]): r for r in rows}
    want = {(f, s, t) for f in ("json", "native") for s in ALL_SCOPES for t in THOUGHTS}
    assert want <= set(keyed), sorted(want - set(keyed))
    for f, s, t in sorted(want):
        a = tc.compile_toolcall(toolset(), scope=s, thought=t, call_format=f)
        assert keyed[(f, s, t)]["n_states"] == int(a.n_states), (f, s, t)
    lines = out.splitlines()
    for f, s, t in sorted(want):
        n = str(keyed[(f, s, t)]["n_states"])
        assert any(f in ln and s in ln and t in ln and n in ln for ln in lines), (f, s, t)
    natives = [ln for ln in lines if "native" in ln]
    assert len(natives) >= len(ALL_SCOPES) * len(THOUGHTS), out
    with capsys.disabled():
        print("\n" + out)
