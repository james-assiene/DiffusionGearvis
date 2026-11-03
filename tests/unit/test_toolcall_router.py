"""T-0080a criterion 2: the pure, total router over the committed answer channel.

``router.route(answer_text, *, thought_open=False) -> Route`` (``thought_open`` is ``llm_client``'s own
flag, implementation notes). Expected values are hand-written from the criterion:

- only the answer channel is scanned; a thought (``<|channel>...<channel|>``) never routes, and text with
  an unclosed thought (``<|channel>`` with no later ``<channel|>``, or ``thought_open=True``) routes
  ``none``;
- the first tag wins; ``<cc>`` -> ``toolcall``, ``<genui>`` -> ``genui`` (no draft parsing), the native
  ``<|tool_call>`` -> ``toolcall`` with ``tag="native"``;
- ``spoken_prefix`` is the text before the first tag, ``draft_call_text`` the text after it;
- never raises.

K-notes (proposals where the task is silent; the rows below pin them):

- K1 ``tag`` is the tag's *name* (``"cc"``, ``"genui"``, ``"native"``), ``None`` for ``kind="none"``
  (criterion 2 names ``tag="native"``, so the others follow the umbrella's tag table keys).
- K2 For ``kind="none"`` on plain text, ``spoken_prefix`` is the whole text, ``draft_call_text`` is
  ``""`` and ``draft_name`` is ``None``.
- K3 A *closed* thought before the answer is not part of ``spoken_prefix`` (CLAUDE.md: never voice
  thinking); the "scanned suffix" of the Hypothesis property is the text after the closing ``<channel|>``.
- K4 ``draft_name`` is read from a JSON draft ``{"name": ...}``, also when the draft is truncated after
  the name (criterion 2: "if it parses enough to name one").

T-0082a (human decision H-T80c, 2026-10-09) **amends T-0080a criterion 2** deliberately: ``draft_call_text`` is
no longer always "the text after the tag". When the tag is ``cc`` and the text after ``<cc>`` matches
``}\\s*</cc>\\s*\\Z``, the router removes that one terminal ``</cc>`` (plus the whitespace after it) into a new
field ``Route.draft_close``; otherwise ``draft_close == ""``. The reconstruction property therefore becomes
``spoken_prefix + TAGS[tag] + draft_call_text + draft_close == scanned`` -- a spec change, not a weakening.
The T-0082a rows below are hand-written from docs/tasks/T-0082a.md criteria 1-6; ``draft_close`` is read through
``_close`` so a router without the field fails each row with an assertion, not an ``AttributeError``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.unit._toolcall_support import (
    CLICK01_DRAFT,
    CLICK01_NATIVE_TEXT,
    CLICK01_SPOKEN,
    TAG_LITERAL,
    combo_enum,
    router_mod,
    score_mod,
)

OPEN, CLOSE = "<|channel>", "<channel|>"
SEARCH = '{"name": "google_search", "arguments": {"query": "weather in Paris"}}'
CLICK = '{"name": "click_at", "arguments": {"x": 120, "y": 640}}'


def route(text: str, **kw: Any) -> Any:
    return router_mod().route(text, **kw)


def _fields(r: Any) -> tuple[Any, ...]:
    return (r.kind, r.tag, r.spoken_prefix, r.draft_call_text, r.draft_name)


def _close(r: Any) -> Any:
    """``Route.draft_close`` (T-0082a criterion 1). Its absence is an assertion, not an error."""
    if not hasattr(r, "draft_close"):
        raise AssertionError("T-0082a not implemented: Route has no `draft_close` field")
    return r.draft_close


def _strip_fields(r: Any) -> tuple[Any, ...]:
    return (r.kind, r.tag, r.spoken_prefix, r.draft_call_text, _close(r), r.draft_name)


# --------------------------------------------------------------------------- criterion 2 rows


def test_plain_answer_routes_none_and_is_all_spoken() -> None:
    r = route("It is sunny in Paris today.")
    assert (r.kind, r.tag, r.draft_name) == ("none", None, None)
    assert r.spoken_prefix == "It is sunny in Paris today."
    assert r.draft_call_text == ""


def test_cc_routes_toolcall_with_prefix_draft_and_name() -> None:
    r = route("Let me look that up. <cc>" + SEARCH)
    assert _fields(r) == ("toolcall", "cc", "Let me look that up. ", SEARCH, "google_search")


def test_cc_at_position_zero_has_empty_spoken_prefix() -> None:
    r = route("<cc>" + CLICK)
    assert _fields(r) == ("toolcall", "cc", "", CLICK, "click_at")


def test_genui_routes_genui_without_draft_parsing() -> None:
    body = '{"name": "click_at", "arguments": {}}'
    r = route("Here is a card. <genui>" + body)
    assert (r.kind, r.tag, r.spoken_prefix, r.draft_call_text) == (
        "genui",
        "genui",
        "Here is a card. ",
        body,
    )
    # No draft parsing for genui: the body happens to look like a call and must not be named.
    assert r.draft_name is None


def test_native_tool_call_marker_routes_toolcall_logged_native() -> None:
    draft = 'call:google_search{query:<|"|>weather<|"|>}<tool_call|>'
    r = route("Searching. <|tool_call>" + draft)
    assert (r.kind, r.tag, r.spoken_prefix, r.draft_call_text) == (
        "toolcall",
        "native",
        "Searching. ",
        draft,
    )


def test_closed_thought_before_the_answer_is_not_scanned_and_not_spoken() -> None:
    text = OPEN + "thought\nThe user wants the weather.\n" + CLOSE + "One moment. <cc>" + SEARCH
    r = route(text)
    assert _fields(r) == ("toolcall", "cc", "One moment. ", SEARCH, "google_search")


def test_unclosed_thought_routes_none() -> None:
    r = route("Sure. <cc>" + SEARCH + " " + OPEN + "thought\nwait, maybe")
    assert (r.kind, r.tag, r.draft_name) == ("none", None, None)
    # K3 extension (reviewer ruling): nothing is voiced while a thought is open.
    assert (r.spoken_prefix, r.draft_call_text) == ("", "")


def test_thought_open_flag_routes_none() -> None:
    r = route("Let me look that up. <cc>" + SEARCH, thought_open=True)
    assert (r.kind, r.tag, r.draft_name) == ("none", None, None)
    assert (r.spoken_prefix, r.draft_call_text) == ("", "")


def test_thought_open_false_is_not_too_wide() -> None:
    """Guard width: the same text with the flag explicitly off still routes (the flag alone decides)."""
    r = route("Let me look that up. <cc>" + SEARCH, thought_open=False)
    assert r.kind == "toolcall"


def test_draft_name_none_when_the_draft_names_nothing() -> None:
    r = route("Okay <cc>not a call at all")
    assert (r.kind, r.tag, r.draft_call_text, r.draft_name) == (
        "toolcall",
        "cc",
        "not a call at all",
        None,
    )


def test_draft_name_from_a_truncated_draft() -> None:
    r = route('Clicking. <cc>{"name": "click_at", "arguments": {"x": 5')
    assert r.kind == "toolcall"
    assert r.draft_name == "click_at"


# --------------------------------------------------------------------------- adversarial rows


def test_cc_inside_a_closed_thought_never_routes_or_speaks() -> None:
    text = OPEN + "thought\nmaybe <cc>" + SEARCH + "\nno, just answer.\n" + CLOSE + "It is 4."
    r = route(text)
    assert (r.kind, r.tag, r.draft_name) == ("none", None, None)
    assert r.spoken_prefix == "It is 4."
    assert "<cc>" not in r.spoken_prefix and "maybe" not in r.spoken_prefix


def test_cc_in_thought_and_in_answer_routes_on_the_answer_tag() -> None:
    text = OPEN + "thought\nI could <cc>" + CLICK + CLOSE + "Searching. <cc>" + SEARCH
    r = route(text)
    assert _fields(r) == ("toolcall", "cc", "Searching. ", SEARCH, "google_search")


def test_unclosed_thought_containing_a_call_routes_none() -> None:
    r = route(OPEN + "thought\nI will <cc>" + SEARCH)
    assert (r.kind, r.tag, r.draft_name) == ("none", None, None)
    assert "<cc>" not in r.spoken_prefix
    assert (r.spoken_prefix, r.draft_call_text) == ("", "")


def test_two_tags_first_wins() -> None:
    r = route("A <cc>" + CLICK + " B <cc>" + SEARCH)
    assert (r.kind, r.tag, r.spoken_prefix) == ("toolcall", "cc", "A ")
    assert r.draft_call_text == CLICK + " B <cc>" + SEARCH
    assert r.draft_name != "google_search"


def test_cc_split_across_canvas_boundaries() -> None:
    canvases = [
        "Let me check. <",
        "c",
        "c>",
        '{"name": "scroll", ',
        '"arguments": {"direction": "down"}}',
    ]
    r = route("".join(canvases))
    assert _fields(r) == (
        "toolcall",
        "cc",
        "Let me check. ",
        '{"name": "scroll", "arguments": {"direction": "down"}}',
        "scroll",
    )


def test_genui_before_cc_routes_genui() -> None:
    r = route("Look: <genui>{card} then <cc>" + SEARCH)
    assert (r.kind, r.tag, r.spoken_prefix) == ("genui", "genui", "Look: ")
    assert r.draft_call_text == "{card} then <cc>" + SEARCH
    assert r.draft_name is None


def test_cc_before_genui_routes_toolcall() -> None:
    r = route("Look: <cc>" + SEARCH + " <genui>{card}")
    assert (r.kind, r.tag, r.spoken_prefix) == ("toolcall", "cc", "Look: ")


def test_empty_draft_after_the_tag() -> None:
    r = route("Searching now.<cc>")
    assert _fields(r) == ("toolcall", "cc", "Searching now.", "", None)


def test_unicode_around_the_tag() -> None:
    prefix = "Café ☕ — 東京の天気を調べます。"
    draft = '{"name": "google_search", "arguments": {"query": "naïve 日本 🌧"}}'
    r = route(prefix + "<cc>" + draft)
    assert _fields(r) == ("toolcall", "cc", prefix, draft, "google_search")
    # NFD text is not normalised away by the router (it splits, it does not rewrite).
    nfd = unicodedata.normalize("NFD", prefix)
    r2 = route(nfd + "<cc>" + draft)
    assert r2.spoken_prefix == nfd


def test_partial_tag_at_the_end_routes_none() -> None:
    r = route("Let me check <c")
    assert (r.kind, r.tag) == ("none", None)


# --------------------------------------------------------------------------- Hypothesis property

_ANSWER_FRAGMENTS = [
    "<cc>",
    "<genui>",
    "<|tool_call>",
    "<",
    ">",
    "c",
    "cc",
    "<c",
    "c>",
    "|",
    "{",
    "}",
    '"name"',
    ": ",
    '"click_at"',
    "genui",
    "tool_call",
    " ",
    "\n",
    "é",
    "日本",
    "🙂",
    "ok",
    # T-0082a: the closing tag and its pieces
    "</cc>",
    "}</cc>",
    "</",
    "/",
]
_answers = st.lists(
    st.one_of(st.sampled_from(_ANSWER_FRAGMENTS), st.text(max_size=3)), max_size=12
).map("".join)
#: Thought bodies never contain a channel delimiter (the model's own single ids 100/101).
_thoughts = _answers.filter(lambda s: OPEN not in s and CLOSE not in s)
_anything = st.lists(
    st.one_of(st.sampled_from([*_ANSWER_FRAGMENTS, OPEN, CLOSE, "<|", "|>"]), st.text(max_size=4)),
    max_size=16,
).map("".join)

_SETTINGS = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@_SETTINGS
@given(text=_anything, thought_open=st.booleans())
def test_route_never_raises_and_returns_a_closed_kind(text: str, thought_open: bool) -> None:
    r = route(text, thought_open=thought_open)
    assert r.kind in {"none", "toolcall", "genui"}
    assert r.tag in {None, *TAG_LITERAL}
    assert isinstance(r.spoken_prefix, str) and isinstance(r.draft_call_text, str)
    assert r.draft_name is None or isinstance(r.draft_name, str)
    assert isinstance(_close(r), str)
    if r.kind == "none":
        assert _close(r) == ""  # T-0082a criterion 2
    if thought_open:
        assert (r.kind, r.spoken_prefix, r.draft_call_text) == ("none", "", "")


def _expected_close(tag: str, after: str) -> str:
    """The suffix criterion 1 removes from ``after`` (the text after the tag), stated without a regex: the last
    ``</cc>``, if only whitespace follows it and the text before it, right-stripped of whitespace, ends in ``}``;
    only for ``cc``. (``str.isspace``/``str.rstrip`` and ``re``'s ``\\s`` use the same Unicode predicate.)"""
    if tag != "cc" or "</cc>" not in after:
        return ""
    i = after.rindex("</cc>")
    rest = after[i + len("</cc>") :]
    if (rest == "" or rest.isspace()) and after[:i].rstrip().endswith("}"):
        return after[i:]
    return ""


@_SETTINGS
@given(answer=_thoughts)
def test_route_reconstructs_the_scanned_answer(answer: str) -> None:
    """T-0082a criterion 2 (amends T-0080a criterion 2): ``spoken_prefix + tag + draft_call_text + draft_close``
    is the scanned text, the tag is the *first* one, ``draft_close`` is exactly the terminal ``}``-preceded
    ``</cc>`` (+ trailing whitespace) of a ``cc`` draft and empty otherwise; ``kind == "none"`` has no close; a
    draft that already parses is never changed (criterion 4)."""
    r = route(answer)
    hits = [answer.find(lit) for lit in TAG_LITERAL.values() if lit in answer]
    if not hits:
        assert (r.kind, r.tag) == ("none", None)
        assert r.spoken_prefix == answer
        assert _close(r) == ""
        return
    assert r.tag in TAG_LITERAL
    assert r.spoken_prefix + TAG_LITERAL[r.tag] + r.draft_call_text + _close(r) == answer
    assert len(r.spoken_prefix) == min(hits)
    assert r.kind == ("genui" if r.tag == "genui" else "toolcall")
    after = answer[len(r.spoken_prefix) + len(TAG_LITERAL[r.tag]) :]
    assert _close(r) == _expected_close(r.tag, after), (after, _close(r))
    if not isinstance(score_mod().parse_call(after), score_mod().ParseFailure):
        assert r.draft_call_text == after


@_SETTINGS
@given(thought=_thoughts, answer=_thoughts)
def test_a_closed_thought_does_not_change_the_route(thought: str, answer: str) -> None:
    alone = route(answer)
    with_thought = route(OPEN + "thought\n" + thought + CLOSE + answer)
    assert _strip_fields(with_thought) == _strip_fields(alone)


@_SETTINGS
@given(thought=_thoughts, answer=_thoughts)
def test_an_unclosed_thought_always_routes_none(thought: str, answer: str) -> None:
    r = route(answer + OPEN + "thought\n" + thought)
    assert (r.kind, r.tag, r.draft_name) == ("none", None, None)
    assert (r.spoken_prefix, r.draft_call_text, _close(r)) == ("", "", "")


# =========================================================================== T-0082a: the terminal </cc> strip

CLICK_260 = '{"name": "click_at", "arguments": {"x": 260, "y": 115}}'
SCROLL_UP = '{"name": "scroll", "arguments": {"direction": "up"}}'


def test_terminal_close_after_brace_is_stripped() -> None:
    """Criterion 1: ``<cc>`` + call + ``</cc>`` -> the call is the draft, ``</cc>`` is ``draft_close``; the draft
    names the tool (criterion 5)."""
    r = route("Clicking it. <cc>" + CLICK_260 + "</cc>")
    assert _strip_fields(r) == ("toolcall", "cc", "Clicking it. ", CLICK_260, "</cc>", "click_at")


def test_whitespace_after_close_goes_to_draft_close() -> None:
    """Criterion 1: the whitespace after ``</cc>`` (newline, spaces, tab, CR, a no-break space -- ``\\s`` as in
    ``str``) is part of the removed suffix."""
    for ws in ("\n", "  \n", "\t", "\r\n", " ", "\n\n "):
        r = route("On it. <cc>" + SCROLL_UP + "</cc>" + ws)
        assert (r.draft_call_text, _close(r)) == (SCROLL_UP, "</cc>" + ws), repr(ws)


def test_whitespace_before_close_stays_in_draft() -> None:
    """Criterion 1 (limits): whitespace between ``}`` and ``</cc>`` stays in the draft."""
    for ws in (" ", "\n", " \n\t"):
        r = route("<cc>" + SCROLL_UP + ws + "</cc>")
        assert (r.draft_call_text, _close(r)) == (SCROLL_UP + ws, "</cc>"), repr(ws)
        assert r.draft_name == "scroll"


def test_text_after_close_is_not_stripped() -> None:
    """Criterion 3 (too wide): ``</cc>`` followed by any non-whitespace stays in the draft, whole."""
    for tail in ("</cc> Done.", "</cc>.", "</cc>\nOK", "</cc></genui>", "</cc>}"):
        r = route("Sure. <cc>" + SCROLL_UP + tail)
        assert (r.kind, r.tag, r.draft_call_text, _close(r)) == (
            "toolcall",
            "cc",
            SCROLL_UP + tail,
            "",
        ), tail


def test_two_closes_only_the_last_is_stripped() -> None:
    """Criterion 1 (limits: at most one tag) and 3: with two ``}``-preceded closes only the last is removed;
    the earlier one stays in the draft verbatim. (The literal ``X}</cc></cc>`` example is ruling A1's row below.)"""
    first = '{"name": "scroll", "arguments": {"direction": "up"}}</cc>'
    r = route("<cc>" + first + SCROLL_UP + "</cc>\n")
    assert (r.draft_call_text, _close(r)) == (first + SCROLL_UP, "</cc>\n")
    r2 = route("<cc>" + SCROLL_UP + "}</cc>}</cc>")
    assert (r2.draft_call_text, _close(r2)) == (SCROLL_UP + "}</cc>}", "</cc>")


def test_literal_double_close_example_strips_nothing() -> None:
    """Ruling A1 (criterion 3's ``X}</cc></cc>`` example; criterion 1's regex governs): the last ``</cc>`` follows
    ``>``, not ``}``, so nothing is stripped -- ``draft_close == ""`` and the draft is the whole text after
    ``<cc>``, both closes included (it stays invalid; constrained decoding repairs it)."""
    draft = SCROLL_UP + "</cc></cc>"
    r = route("Scrolling. <cc>" + draft)
    assert _strip_fields(r) == ("toolcall", "cc", "Scrolling. ", draft, "", "scroll")


def test_native_and_genui_never_strip() -> None:
    """Criterion 3: the strip is for the ``cc`` tag only; a native or genui body ending ``}</cc>`` is unchanged."""
    for lit, tag, kind in (("<|tool_call>", "native", "toolcall"), ("<genui>", "genui", "genui")):
        r = route("Here. " + lit + SCROLL_UP + "</cc>\n")
        assert (r.kind, r.tag, r.spoken_prefix, r.draft_call_text, _close(r)) == (
            kind,
            tag,
            "Here. ",
            SCROLL_UP + "</cc>\n",
            "",
        ), tag


def test_truncated_string_ending_in_close_is_not_stripped() -> None:
    """Criterion 3 (too wide): ``</cc>`` not preceded by ``}`` -- a draft cut inside a string -- stays. The
    draft still names its tool from the (unchanged) prefix."""
    for draft in (
        '{"name": "type_at", "arguments": {"x": 1, "y": 2, "text": "a</cc>',
        '{"name": "google_search", "arguments": {"query": "zelda"</cc>',
        '{"name": "scroll", "arguments": {"direction": "up"} </cc> x',
        "</cc>",
    ):
        r = route("Typing. <cc>" + draft)
        assert (r.draft_call_text, _close(r)) == (draft, ""), draft


def test_close_inside_a_string_of_a_parseable_draft_is_untouched() -> None:
    """Criterion 3/4 (too wide): ``}</cc>`` inside a JSON string of a parseable call is call content, never
    removed -- also when whitespace follows the call."""
    for draft in (
        '{"name": "google_search", "arguments": {"query": "}</cc>"}}',
        '{"name": "type_at", "arguments": {"x": 3, "y": 4, "text": "a}</cc> ", "press_enter": false}}\n',
        '{"name": "google_search", "arguments": {"query": "x}</cc>\\n"}} ',
    ):
        assert not isinstance(score_mod().parse_call(draft), score_mod().ParseFailure), draft
        r = route("Ok. <cc>" + draft)
        assert (r.draft_call_text, _close(r)) == (draft, ""), draft


def test_close_after_nested_braces_and_newline_is_stripped() -> None:
    """Criterion 1 (too narrow): ``}}\\n</cc>\\n`` -- nested closing braces, a newline before the tag (kept in the
    draft) and one after it (removed)."""
    draft = '{"name": "click_at", "arguments": {"x": 260, "y": 115}}\n'
    r = route("Clicking. <cc>" + draft + "</cc>\n")
    assert _strip_fields(r) == ("toolcall", "cc", "Clicking. ", draft, "</cc>\n", "click_at")


def test_strip_runs_on_the_scanned_text_after_closed_thoughts() -> None:
    """Criterion 1: the strip sees the answer channel only. A closed thought *after* ``</cc>`` is removed first,
    so the close is terminal and stripped; a ``}</cc>`` that ends inside a closed thought is not in the scanned
    text at all."""
    r = route("Ok. <cc>" + SCROLL_UP + "</cc>" + OPEN + "thought\nthat was it" + CLOSE)
    assert (r.draft_call_text, _close(r)) == (SCROLL_UP, "</cc>")
    r2 = route("Ok. <cc>" + SCROLL_UP + OPEN + "thought\n{}</cc>" + CLOSE)
    assert (r2.draft_call_text, _close(r2)) == (SCROLL_UP, "")


def test_draft_name_is_read_from_the_stripped_draft() -> None:
    """Criterion 5: with ``name`` after ``arguments`` the prefix reader cannot name the tool; only parsing the
    stripped draft can. Unstripped (text after the close) the same call names nothing."""
    draft = '{"arguments": {"direction": "down"}, "name": "scroll"}'
    r = route("<cc>" + draft + "</cc>")
    assert (r.draft_call_text, _close(r), r.draft_name) == (draft, "</cc>", "scroll")
    r2 = route("<cc>" + draft + "</cc> ok")
    assert r2.draft_name is None


def test_none_route_with_malformed_open_and_close_is_unchanged() -> None:
    """Criterion 6 (negative; pins *current* behaviour, not the product target -- T-0083 fixes it): the search-22
    shape, a malformed ``<cc {`` open closed by ``</cc>``, routes ``none`` and is spoken whole, ``</cc>``
    included; nothing is stripped."""
    text = 'Searching for it. <cc {"name": "google_search", "arguments": {"query": "zelda"}}</cc>'
    r = route(text)
    assert _strip_fields(r) == ("none", None, text, "", "", None)


# --------------------------------------------------------------------------- T-0082a Hypothesis properties

_WS = ["", " ", "\n", "\t", "\r", "  \n", " ", "　", "\x0b", "\x0c"]
_JSON_WS = ["", " ", "\n", "\t", "\r\n"]
_query = st.lists(
    st.one_of(
        st.sampled_from(["</cc>", "}</cc>", "}", "{", '"', "\\", "\n", " ", "日本", "<cc>"]),
        st.text(max_size=3),
    ),
    max_size=6,
).map("".join)
_xy = st.integers(min_value=0, max_value=999)


@st.composite
def _v1_call(draw: Any) -> dict[str, Any]:
    name = draw(st.sampled_from(["google_search", "click_at", "type_at", "scroll", "press_key"]))
    if name == "google_search":
        args: dict[str, Any] = {"query": draw(_query)}
    elif name == "click_at":
        args = {"x": draw(_xy), "y": draw(_xy)}
    elif name == "type_at":
        args = {
            "x": draw(_xy),
            "y": draw(_xy),
            "text": draw(_query),
            "press_enter": draw(st.booleans()),
        }
    elif name == "scroll":
        args = {"direction": draw(st.sampled_from(["up", "down", "left", "right"]))}
    else:
        args = {"combo": draw(st.sampled_from(combo_enum()))}
    return {"name": name, "arguments": args}


@st.composite
def _call_text(draw: Any) -> str:
    """A v1 call as ``json.dumps`` writes it under random JSON whitespace (indent, separators, a leading blank)."""
    obj = draw(_v1_call())
    indent = draw(st.sampled_from([None, 0, 1, 2, "\t"]))
    seps = draw(st.sampled_from([(", ", ": "), (",", ":"), (",\n", " : ")]))
    ascii_ = draw(st.booleans())
    return draw(st.sampled_from(_JSON_WS)) + json.dumps(
        obj, indent=indent, separators=seps, ensure_ascii=ascii_
    )


_spoken = st.sampled_from(["", "Sure. ", "On it — 東京. ", "Let me check </c c>. "])


@_SETTINGS
@given(spoken=_spoken, call=_call_text(), before=st.sampled_from(_WS), after=st.sampled_from(_WS))
def test_strip_never_changes_a_parseable_draft(
    spoken: str, call: str, before: str, after: str
) -> None:
    """Criteria 1 and 4: for a generated v1 call (strings may hold ``}</cc>``, ``<cc>``, quotes, newlines),
    (a) the call alone, or followed by trailing JSON whitespace, parses and is the draft unchanged with no close;
    (b) appending whitespace + ``</cc>`` + whitespace gives back exactly the call + the whitespace before the
    tag as the draft, and ``</cc>`` + the whitespace after it as ``draft_close``."""
    assert "<cc>" not in spoken
    raw = call + (before if before in _JSON_WS else "")
    assert not isinstance(score_mod().parse_call(raw), score_mod().ParseFailure), raw
    r = route(spoken + "<cc>" + raw)
    assert (r.kind, r.tag, r.spoken_prefix, r.draft_call_text, _close(r)) == (
        "toolcall",
        "cc",
        spoken,
        raw,
        "",
    )
    r2 = route(spoken + "<cc>" + call + before + "</cc>" + after)
    assert (r2.kind, r2.tag, r2.spoken_prefix) == ("toolcall", "cc", spoken)
    assert (r2.draft_call_text, _close(r2)) == (call + before, "</cc>" + after)
    assert r2.draft_name == json.loads(call)["name"]


_CLOSE_RE = re.compile(r"</cc>")


@_SETTINGS
@given(spoken=_spoken, draft=_answers, thought=_thoughts)
def test_toolcall_spoken_prefix_never_contains_close(spoken: str, draft: str, thought: str) -> None:
    """Criterion 6: whatever the draft holds (closes, braces, more tags), a ``toolcall`` route never voices
    ``</cc>`` -- the strip moves text out of the draft, never into ``spoken_prefix``. With a closed thought in
    front as well. (Answers whose spoken part itself holds ``</cc>`` before the first tag are out of scope: see the
    report -- first-tag-wins puts that text in ``spoken_prefix`` by T-0080a criterion 2.)"""
    assume_ok = "<cc>" not in spoken and not _CLOSE_RE.search(spoken)
    assert assume_ok
    for text in (
        spoken + "<cc>" + draft,
        OPEN + "thought\n" + thought + CLOSE + spoken + "<cc>" + draft,
    ):
        r = route(text)
        assert r.kind == "toolcall"
        assert "</cc>" not in r.spoken_prefix, (text, r.spoken_prefix)
        assert r.spoken_prefix == spoken


# =========================================================================== T-0085: the spoken prefix is template-clean
#
# docs/tasks/T-0085.md criteria 1 and 2 (amending T-0080a criterion 2 and T-0082a criterion 2: what ``spoken_prefix``
# is). Expected values are hand-written from the criteria. ``_clean`` is the criterion's own definition -- "the text
# before the tag with only channel markers removed" -- read as: delete ``<channel|>`` / ``<|channel>`` until none is
# left (a deletion can splice a new marker, ``<|chan<channel|>nel>``; the two markers cannot overlap, so the order
# of deletions does not matter). ``router.template_content`` is read through ``_template_content`` so its absence is
# an assertion, not an ``AttributeError``; its own oracle row (against the pinned template) is
# ``tests/cd/unit/test_template_content_oracle.py``.

MARKERS = (OPEN, CLOSE)


def _template_content(text: str) -> str:
    fn = getattr(router_mod(), "template_content", None)
    if not callable(fn):
        raise AssertionError("T-0085 not implemented: router.template_content is missing")
    out = fn(text)
    assert isinstance(out, str), type(out)
    return out


def _clean(text: str) -> str:
    """Remove the leftmost channel marker, one at a time, until none is left (hand oracle, criterion 1)."""
    while True:
        hits = [(text.find(m), m) for m in MARKERS if m in text]
        if not hits:
            return text
        i, m = min(hits)
        text = text[:i] + text[i + len(m) :]


def _no_marker(s: str) -> bool:
    return OPEN not in s and CLOSE not in s


def test_t0085_the_click01_text_verbatim() -> None:
    """Criterion 2: the T-0084e stop's native pass 1 routes ``native`` with the stray close removed from the spoken
    prefix and the draft unchanged (at 9aee66d the prefix was ``"I will click the Sign in button.<channel|>"``)."""
    r = route(CLICK01_NATIVE_TEXT)
    assert (r.kind, r.tag) == ("toolcall", "native")
    assert r.spoken_prefix == CLICK01_SPOKEN, r.spoken_prefix
    assert (r.draft_call_text, _close(r), r.draft_name) == (CLICK01_DRAFT, "", None)


def test_t0085_click01_continuation_agrees_with_the_template() -> None:
    """Criterion 2: for the click-01 route, ``template_content(spoken_prefix + "<|tool_call>")`` is the stripped
    continuation itself, ``"I will click the Sign in button.<|tool_call>"``."""
    r = route(CLICK01_NATIVE_TEXT)
    c = r.spoken_prefix + TAG_LITERAL["native"]
    assert c == "I will click the Sign in button.<|tool_call>"
    assert _template_content(c) == c.strip()


def test_t0085_a_stray_close_in_a_none_route_is_not_spoken() -> None:
    """Criterion 1: a ``none`` route speaks the whole scanned text with the stray close removed."""
    r = route(OPEN + "thought\nNo tool.\n" + CLOSE + "It is 12 degrees.<channel|> Anything else?")
    assert (r.kind, r.tag, r.draft_call_text, _close(r)) == ("none", None, "", "")
    assert r.spoken_prefix == "It is 12 degrees. Anything else?"
    r2 = route("Done.<channel|>")
    assert (r2.kind, r2.spoken_prefix) == ("none", "Done.")


def test_t0085_a_stray_close_before_a_cc_tag_is_not_spoken() -> None:
    """Criterion 1, the cc twin of click-01 (no stored cc answer has it; T-0080c/T-0082c hold 0 such markers)."""
    r = route("Let me look that up.<channel|> <cc>" + SEARCH)
    assert _fields(r) == ("toolcall", "cc", "Let me look that up. ", SEARCH, "google_search")


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("<|tool_<channel|>call>", "<|tool_call>"),
        (
            "<|tool_<channel|>call>call:click_at{x:1,y:2}<tool_call|>",
            "<|tool_call>call:click_at{x:1,y:2}<tool_call|>",
        ),
        ("<c<channel|>c>" + CLICK, "<cc>" + CLICK),
        ("<gen<channel|>ui>{}", "<genui>{}"),
    ],
)
def test_t0085_marker_removal_never_creates_a_tag(text: str, spoken: str) -> None:
    """Criterion 1: the tag is searched before the removal, so a tag spliced by removing a marker never routes; the
    route is ``none`` and (only channel markers removed) the spliced text is what is left. Whether it may be voiced
    is the speech gate's business (T-0080d), not the router's (human decision, router scope)."""
    r = route(text)
    assert (r.kind, r.tag, r.draft_call_text, r.draft_name) == ("none", None, "", None)
    assert r.spoken_prefix == spoken


def test_t0085_a_spliced_tag_before_a_real_tag_stays_spoken() -> None:
    """Criterion 1: the first tag is the first one in the raw scanned text -- the spliced ``<|tool_call>`` before a
    real ``<cc>`` is not a tag; ``<cc>`` routes and the splice is spoken."""
    r = route("<|tool_<channel|>call> then <cc>" + CLICK)
    assert _fields(r) == ("toolcall", "cc", "<|tool_call> then ", CLICK, "click_at")


@pytest.mark.parametrize(
    ("text", "spoken", "draft"),
    [
        # a removal splices an open: <|chan + <channel|> + nel> -> <|channel> -> removed too
        (
            "I will<|chan<channel|>nel> click.<|tool_call>call:click_at{x:1,y:2}<tool_call|>",
            "I will click.",
            "call:click_at{x:1,y:2}<tool_call|>",
        ),
        # nested splice: the outer pieces form a close once the inner open is gone
        ("A<cha<|chan<channel|>nel>nnel|>B <cc>" + CLICK, "AB ", CLICK),
        # the thought removal splices an open in the scanned text: <|chan + (closed thought) + nel>
        ("<|chan<|channel>thought\nx<channel|>nel>Hi <cc>" + CLICK, "Hi ", CLICK),
    ],
)
def test_t0085_a_spliced_channel_open_is_not_spoken(text: str, spoken: str, draft: str) -> None:
    """Criterion 1: no channel marker in ``spoken_prefix`` even when the removal (or the closed-thought strip)
    creates one; the draft is untouched."""
    r = route(text)
    assert r.kind == "toolcall"
    assert r.spoken_prefix == spoken, r.spoken_prefix
    assert r.draft_call_text == draft


@pytest.mark.parametrize(
    ("text", "tag", "spoken", "draft"),
    [
        (
            "Searching.<|tool_call>call:google_search{query:<channel|>}<tool_call|>",
            "native",
            "Searching.",
            "call:google_search{query:<channel|>}<tool_call|>",
        ),
        (
            'Ok <cc>{"name": "google_search", "arguments": {"query": "<channel|><|chan"}}',
            "cc",
            "Ok ",
            '{"name": "google_search", "arguments": {"query": "<channel|><|chan"}}',
        ),
        ("Here. <genui>{<channel|>}", "genui", "Here. ", "{<channel|>}"),
    ],
)
def test_t0085_a_close_inside_the_draft_stays_in_the_draft(
    text: str, tag: str, spoken: str, draft: str
) -> None:
    """Criterion 1: drafts are untouched -- a marker after the tag stays in the draft, and ``TAGS[tag] + draft +
    draft_close`` is still the scanned text from the tag onward."""
    r = route(text)
    assert (r.tag, r.spoken_prefix, r.draft_call_text, _close(r)) == (tag, spoken, draft, "")
    assert (
        TAG_LITERAL[tag] + r.draft_call_text + _close(r)
        == text[len(text) - len(TAG_LITERAL[tag] + draft) :]
    )


def test_t0085_clean_text_is_not_changed() -> None:
    """Criterion 1, the other direction (not too wide): text with marker *pieces* but no marker -- ``<channel``,
    ``|>``, ``<|chan``, ``nel>``, ``<channel |>`` -- is spoken as it is."""
    spoken = "Pipes |> and <channel or <|chan, nel> and <channel |> are text. "
    r = route(spoken + "<cc>" + CLICK)
    assert _fields(r) == ("toolcall", "cc", spoken, CLICK, "click_at")
    r2 = route(spoken)
    assert (r2.kind, r2.spoken_prefix) == ("none", spoken)


# --------------------------------------------------------------------------- T-0085 Hypothesis properties

_T85_FRAGMENTS = [
    *_ANSWER_FRAGMENTS,
    CLOSE,
    "<channel",
    "|>",
    "<|chan",
    "nel>",
    "<|",
    "chan",
    "<tool_",
    "call>",
    "\t",
    "  ",
    "　",
]
#: answers whose raw text holds no ``<|channel>`` (so nothing is a thought: the scanned text is the raw text) but
#: stray closes and every piece that can splice a marker
_t85_answers = (
    st.lists(st.one_of(st.sampled_from(_T85_FRAGMENTS), st.text(max_size=3)), max_size=14)
    .map("".join)
    .filter(lambda s: OPEN not in s)
)
_T85_SETTINGS = settings(
    max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


@_T85_SETTINGS
@given(answer=_t85_answers)
def test_t0085_route_on_stray_markers_is_clean_and_reconstructs(answer: str) -> None:
    """Criteria 1 and 2 over fragments plus ``<channel``, ``|>``, ``<|chan``, ``nel>`` and stray closes:

    - no channel marker in ``spoken_prefix``;
    - the tag is the first tag literal of the *raw* text (before any removal);
    - ``spoken_prefix`` is the raw text before it (or all of it) with only channel markers removed;
    - ``TAGS[tag] + draft_call_text + draft_close`` is exactly the raw text from the tag onward (drafts untouched);
    - for a ``toolcall`` route and both its tag and the other toolcall tag, ``template_content(spoken_prefix + TAG)
      == (spoken_prefix + TAG).strip()``."""
    r = route(answer)
    assert _no_marker(r.spoken_prefix), (answer, r.spoken_prefix)
    hits = sorted((answer.find(lit), tag) for tag, lit in TAG_LITERAL.items() if lit in answer)
    if not hits:
        assert (r.kind, r.tag, r.draft_call_text, _close(r)) == ("none", None, "", "")
        assert r.spoken_prefix == _clean(answer)
        return
    i, tag = hits[0]
    assert r.tag == tag, (answer, r.tag)
    assert r.spoken_prefix == _clean(answer[:i]), (answer, r.spoken_prefix)
    assert TAG_LITERAL[tag] + r.draft_call_text + _close(r) == answer[i:]
    if r.kind == "toolcall":
        for t in ("cc", "native"):
            c = r.spoken_prefix + TAG_LITERAL[t]
            assert _template_content(c) == c.strip(), (answer, c)


def _scanned(text: str) -> str | None:
    """T-0080a criterion 2's answer channel, by hand: delete each ``<|channel>...<channel|>`` (the first close after
    the open); ``None`` when an open is left unclosed."""
    out = ""
    while OPEN in text:
        i = text.index(OPEN)
        j = text.find(CLOSE, i + len(OPEN))
        if j < 0:
            return None
        out += text[:i]
        text = text[j + len(CLOSE) :]
    return out + text


_t85_anything = st.lists(
    st.one_of(st.sampled_from([*_T85_FRAGMENTS, OPEN, "thought\n"]), st.text(max_size=3)),
    max_size=16,
).map("".join)


@_T85_SETTINGS
@given(text=_t85_anything)
def test_t0085_route_with_thoughts_and_splices_never_voices_a_marker(text: str) -> None:
    """Criterion 1 with thoughts in play (closed, unclosed, and opens spliced by the thought strip): ``route`` never
    raises, never voices a channel marker, finds the tag on the scanned text before any removal, and leaves the
    draft as the scanned text after the tag."""
    r = route(text)
    assert _no_marker(r.spoken_prefix), (text, r.spoken_prefix)
    scanned = _scanned(text)
    if scanned is None:
        assert (r.kind, r.spoken_prefix, r.draft_call_text) == ("none", "", "")
        return
    hits = sorted((scanned.find(lit), tag) for tag, lit in TAG_LITERAL.items() if lit in scanned)
    if not hits:
        assert (r.kind, r.spoken_prefix) == ("none", _clean(scanned))
        return
    i, tag = hits[0]
    assert (r.tag, r.spoken_prefix) == (tag, _clean(scanned[:i])), text
    assert TAG_LITERAL[tag] + r.draft_call_text + _close(r) == scanned[i:]
