"""T-0113 criteria 10, 12 and 14: the loop's host-side rendering (T-0111a §1, §4; H-T111a-10 form G).

**Goldens.** Every literal below was rendered by the pinned template (sha ``36e3a42e...``) through
``tests/cd/oracle/gemma4_template.py`` on 2026-10-17 (jinja2 sandbox, ``trim_blocks``/``lstrip_blocks``), from an
assistant message with ``tool_calls`` plus the legacy ``tool_responses`` field (template lines 243-269, the only
path that renders a *mapping* response), and cut after ``<|turn>model\\n``. The oracle needs jinja2, which is in the
``cd`` group only, so the live oracle rows are ``tests/cd/unit/test_toolcall_loop_oracle.py`` (``ASTRA_CD_TESTS=1``);
these unit rows pin the same strings for the fast suite. Two template facts the goldens carry and a naive renderer misses:
``dictsort`` is case-insensitive (``a:1,B:...``), and a Python ``True`` is ``true``, never ``1``.

**Form D** (criterion 10) is read as the oracle's rendering of the same steps as structured messages (one
assistant message per step with ``tool_calls`` and ``tool_responses``, no content): the second message continues the
same model turn, so with empty step texts form G's continuation must equal that model turn, after the step-1 prefix.

**Bounds** (criterion 12) are measured on ``render_response(...)``'s output, the whole ``<|tool_response>...
<tool_response|>`` block (ruling 3); a cut snippet may end in "…" (ruling 5). The 8-step regime is in its row's
docstring (ruling 7). Deliberately untested (ruling 6): the order of sanitising and bounding, a bound on the query
length, and ``step_prompt``'s ``prefix_len``. (``loop.py`` importing pydantic directly is not allowed: the imports
row admits stdlib, contracts, ``special_tokens`` and ``router`` only, as criterion 13 reads.)

API (proposed by the test-author, accepted 2026-10-17, ruling 2):
``loop.search_payload(SearchResults | None) -> dict``; ``loop.action_payload(action) -> dict``;
``loop.status_payload(status, reason) -> dict`` for ``refused``/``failed``/``timeout``;
``loop.LoopStep(text=, name=, arguments=, response=)``; ``loop.step_prompt(prompt, steps) -> Prompt``.
``render_call`` renders what it is given (validation is ``live``'s), so a non-v1 name is fine here.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any

import pytest

from local_astra.contracts.llm import Message, Prompt, TextPart
from local_astra.contracts.perception import AssistFlags
from tests import _git_range as gr
from tests.fakes.search import FakeSearch

LOOP_MODULE = "local_astra.toolcall.loop"
D = '<|"|>'
RESP_CLOSE = "<tool_response|>"


def _loop() -> ModuleType:
    try:
        return importlib.import_module(LOOP_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == LOOP_MODULE:
            raise NotImplementedError(f"T-0113: {LOOP_MODULE} does not exist yet") from exc
        raise


def _results(rows: Sequence[Mapping[str, str]], query: str) -> Any:
    from local_astra.contracts import toolcall as ct

    if not hasattr(ct, "SearchResults"):
        raise NotImplementedError("T-0113: contracts.toolcall.SearchResults does not exist yet")
    return asyncio.run(FakeSearch(default=rows).search(query))


def _s(text: str) -> str:
    return D + text + D


def golden_search(query: str, rows: Sequence[Mapping[str, str]]) -> str:
    """A hand-written instance of the template's mapping form for a search response (keys in dictsort order).
    Checked against an oracle literal by ``test_the_hand_written_search_golden_matches_the_oracle``."""
    items = ",".join(
        "{snippet:" + _s(r["snippet"]) + ",title:" + _s(r["title"]) + ",url:" + _s(r["url"]) + "}"
        for r in rows
    )
    return (
        "<|tool_response>response:google_search{query:"
        + _s(query)
        + ",results:["
        + items
        + "]}"
        + RESP_CLOSE
    )


# --------------------------------------------------------------------------- oracle goldens

PARIS_ROW = {
    "title": "Paris weather",
    "url": "https://example.org/paris",
    "snippet": "14C, light rain",
}

CALLS = [
    (
        "google_search",
        {"query": "weather in Paris"},
        '<|tool_call>call:google_search{query:<|"|>weather in Paris<|"|>}<tool_call|>',
    ),
    ("click_at", {"x": 412, "y": 300}, "<|tool_call>call:click_at{x:412,y:300}<tool_call|>"),
    (
        "type_at",
        {"x": 5, "y": 999, "text": "hello world", "press_enter": True},
        '<|tool_call>call:type_at{press_enter:true,text:<|"|>hello world<|"|>,x:5,y:999}<tool_call|>',
    ),
    (
        "scroll",
        {"direction": "down"},
        '<|tool_call>call:scroll{direction:<|"|>down<|"|>}<tool_call|>',
    ),
    (
        "press_key",
        {"combo": "Ctrl+C"},
        '<|tool_call>call:press_key{combo:<|"|>Ctrl+C<|"|>}<tool_call|>',
    ),
    # adversarial: nested lists and mappings, booleans next to ints, a zero
    (
        "nest",
        {"a": [1, [2, "x"], {"k": False}], "b": True, "c": 0},
        '<|tool_call>call:nest{a:[1,[2,<|"|>x<|"|>],{k:false}],b:true,c:0}<tool_call|>',
    ),
    # adversarial: dictsort is case-insensitive at every depth (Python's sorted() would put B first)
    (
        "nest",
        {"a": 1, "B": {"d": -3, "C": [True]}},
        "<|tool_call>call:nest{a:1,B:{C:[true],d:-3}}<tool_call|>",
    ),
]

RESPONSES = [
    (
        "click_at",
        {"action": "would click at (412, 300)", "screen_changed": False, "status": "dry_run"},
        '<|tool_response>response:click_at{action:<|"|>would click at (412, 300)<|"|>,screen_changed:false,'
        'status:<|"|>dry_run<|"|>}<tool_response|>',
    ),
    (
        "google_search",
        "1. Paris 14C",
        '<|tool_response>response:google_search{value:<|"|>1. Paris 14C<|"|>}<tool_response|>',
    ),
    (
        "google_search",
        {"n": 3, "xs": [1, True, [2]], "m": {"b": 1, "a": "z"}},
        '<|tool_response>response:google_search{m:{a:<|"|>z<|"|>,b:1},n:3,xs:[1,true,[2]]}<tool_response|>',
    ),
    (
        "scroll",
        {"n": -7, "deep": [[1, [False]], {"B": True, "a": []}]},
        "<|tool_response>response:scroll{deep:[[1,[false]],{a:[],B:true}],n:-7}<tool_response|>",
    ),
]

SEARCH_GOLDEN = (
    '<|tool_response>response:google_search{query:<|"|>weather in Paris<|"|>,results:[{snippet:<|"|>14C, light '
    'rain<|"|>,title:<|"|>Paris weather<|"|>,url:<|"|>https://example.org/paris<|"|>}]}<tool_response|>'
)
NO_BACKEND_GOLDEN = (
    '<|tool_response>response:google_search{note:<|"|>search is not connected in this session<|"|>,results:[],'
    'status:<|"|>dry_run<|"|>}<tool_response|>'
)


def test_the_hand_written_search_golden_matches_the_oracle() -> None:
    """Self-check (green today): the builder used by the bound rows reproduces an oracle literal."""
    assert golden_search("weather in Paris", [PARIS_ROW]) == SEARCH_GOLDEN


# --------------------------------------------------------------------------- criterion 10: parity


@pytest.mark.parametrize(
    "name, arguments, want", CALLS, ids=[f"{c[0]}-{i}" for i, c in enumerate(CALLS)]
)
def test_render_call_equals_the_pinned_template(
    name: str, arguments: dict[str, Any], want: str
) -> None:
    assert _loop().render_call(name, arguments) == want


@pytest.mark.parametrize(
    "name, payload, want", RESPONSES, ids=["mapping", "string", "nested", "deep"]
)
def test_render_response_equals_the_pinned_template(name: str, payload: Any, want: str) -> None:
    assert _loop().render_response(name, payload) == want


def test_search_with_a_backend_renders_query_and_results() -> None:
    loop = _loop()
    payload = loop.search_payload(_results([PARIS_ROW], "weather in Paris"))
    assert loop.render_response("google_search", payload) == SEARCH_GOLDEN


def test_search_without_a_backend_says_it_is_not_connected() -> None:
    """H-T111a-9."""
    loop = _loop()
    assert loop.render_response("google_search", loop.search_payload(None)) == NO_BACKEND_GOLDEN


@pytest.mark.parametrize(
    "name, action",
    [
        ("click_at", "would click at (412, 300)"),
        ("type_at", "would type 'hi' at (5, 999)"),
        ("scroll", "would scroll down"),
        ("press_key", "would press Ctrl+C"),
    ],
)
def test_a_dry_run_action_reports_no_screen_change(name: str, action: str) -> None:
    loop = _loop()
    want = (
        f"<|tool_response>response:{name}{{action:{_s(action)},screen_changed:false,"
        f"status:{_s('dry_run')}}}{RESP_CLOSE}"
    )
    assert loop.render_response(name, loop.action_payload(action)) == want


@pytest.mark.parametrize(
    "name, status, reason, want",
    [
        (
            "type_at",
            "refused",
            "not allowed",
            '<|tool_response>response:type_at{reason:<|"|>not allowed<|"|>,status:<|"|>refused<|"|>}<tool_response|>',
        ),
        (
            "scroll",
            "failed",
            "boom",
            '<|tool_response>response:scroll{reason:<|"|>boom<|"|>,status:<|"|>failed<|"|>}<tool_response|>',
        ),
        (
            "press_key",
            "timeout",
            "no result within 15 s",
            '<|tool_response>response:press_key{reason:<|"|>no result within 15 s<|"|>,'
            'status:<|"|>timeout<|"|>}<tool_response|>',
        ),
    ],
    ids=["refused", "failed", "timeout"],
)
def test_refused_failed_and_timeout_carry_a_reason(
    name: str, status: str, reason: str, want: str
) -> None:
    loop = _loop()
    assert loop.render_response(name, loop.status_payload(status, reason)) == want


# --------------------------------------------------------------------------- criterion 12: bounds


def _row(i: int, snippet: str = "s") -> dict[str, str]:
    return {"title": f"Result {i}", "url": f"https://example.org/{i}", "snippet": snippet}


def _search(rows: Sequence[Mapping[str, str]], query: str = "paris") -> str:
    loop = _loop()
    rendered: str = loop.render_response(
        "google_search", loop.search_payload(_results(rows, query))
    )
    return rendered


def test_a_sixth_result_is_dropped() -> None:
    rows = [_row(i) for i in range(1, 7)]
    assert _search(rows) == golden_search("paris", rows[:5])


def test_a_snippet_of_exactly_300_chars_is_kept_verbatim() -> None:
    snippet = ("abcdefghij" * 30)[:300]
    rows = [_row(1, snippet)]
    assert _search(rows) == golden_search("paris", rows)


def test_a_snippet_of_301_chars_is_cut_to_at_most_300() -> None:
    snippet = ("abcdefghij" * 31)[:301]
    got = _search([_row(1, snippet)])
    assert got != golden_search("paris", [_row(1, snippet)])
    body = got.split("{snippet:" + D, 1)[1].split(D, 1)[0]
    assert 290 <= len(body) <= 300, len(body)  # too-strict twin: not cut to nothing
    assert snippet.startswith(body.rstrip("…")), body[-20:]


def _rows_rendering_to(total: int) -> list[dict[str, str]]:
    """Five results whose unbounded search block is exactly ``total`` chars, every snippet ≤ 300."""
    rows = [_row(i, "") for i in range(1, 6)]
    room = total - len(golden_search("paris", rows))
    assert 0 < room <= 5 * 300
    for i, row in enumerate(rows):
        n = min(300, room - sum(len(r["snippet"]) for r in rows))
        row["snippet"] = (f"snippet {i} " + "x" * 300)[:n]
    assert len(golden_search("paris", rows)) == total
    return rows


def test_the_1500_and_1501_fixtures_are_exact() -> None:
    """Self-check (green today): the bound rows' inputs are what their names say."""
    for total in (1500, 1501):
        rows = _rows_rendering_to(total)
        assert len(rows) == 5 and all(len(r["snippet"]) <= 300 for r in rows)


def test_a_response_of_exactly_1500_chars_is_kept_verbatim() -> None:
    rows = _rows_rendering_to(1500)
    assert _search(rows) == golden_search("paris", rows)


def test_a_response_of_1501_chars_is_bounded_to_1500() -> None:
    rows = _rows_rendering_to(1501)
    got = _search(rows)
    assert len(got) <= 1500
    assert len(got) >= 1000, f"too strict: a 1-char overflow cost {1501 - len(got)} chars"
    assert got.startswith(
        "<|tool_response>response:google_search{query:" + _s("paris") + ",results:["
    )
    assert got.endswith("]}" + RESP_CLOSE)


def test_every_response_kind_is_bounded() -> None:
    """A long reason or action text (a typed string can be long) is bounded like a search."""
    loop = _loop()
    long = "y" * 2000
    assert len(loop.render_response("type_at", loop.action_payload(long))) <= 1500
    assert len(loop.render_response("type_at", loop.status_payload("refused", long))) <= 1500
    assert len(loop.render_response("type_at", long)) <= 1500


# --------------------------------------------------------------------------- criterion 14: step_prompt

TOOLS = (
    {"type": "function", "function": {"name": "google_search", "parameters": {"type": "object"}}},
)
ASSISTS = AssistFlags(page_text=True)


def _prompt1(*, thinking: bool = True) -> Prompt:
    return Prompt(
        messages=(
            Message(role="system", content=(TextPart(text="You are Astra."),)),
            Message(role="user", content=(TextPart(text="What is the weather in Paris?"),)),
        ),
        prefix_len=2,
        thinking=thinking,
        assists=ASSISTS,
        tools=TOOLS,
    )


def _steps(loop: ModuleType, specs: Sequence[tuple[str, str, Mapping[str, Any], Any]]) -> list[Any]:
    return [loop.LoopStep(text=t, name=n, arguments=dict(a), response=r) for t, n, a, r in specs]


FORM_D_TWO_STEPS = (
    '<|tool_call>call:google_search{query:<|"|>weather in Paris<|"|>}<tool_call|><|tool_response>response:'
    'google_search{query:<|"|>weather in Paris<|"|>,results:[{snippet:<|"|>14C, light rain<|"|>,title:<|"|>Paris '
    'weather<|"|>,url:<|"|>https://example.org/paris<|"|>}]}<tool_response|><|tool_call>call:click_at{x:412,y:300}'
    '<tool_call|><|tool_response>response:click_at{action:<|"|>would click at (412, 300)<|"|>,screen_changed:false,'
    'status:<|"|>dry_run<|"|>}<tool_response|>'
)
SEARCH_RESPONSE = {
    "query": "weather in Paris",
    "results": [{k: PARIS_ROW[k] for k in sorted(PARIS_ROW)}],
}
CLICK_RESPONSE = {
    "action": "would click at (412, 300)",
    "screen_changed": False,
    "status": "dry_run",
}


@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("n_steps", [1, 2, 3])
def test_step_prompt_keeps_step_one_and_continues_the_final_assistant_message(
    thinking: bool, n_steps: int
) -> None:
    loop = _loop()
    p1 = _prompt1(thinking=thinking)
    specs = [
        (f"Step {k}.", "scroll", {"direction": "down"}, CLICK_RESPONSE) for k in range(n_steps)
    ]
    p = loop.step_prompt(p1, _steps(loop, specs))
    assert p.messages[: len(p1.messages)] == p1.messages
    assert len(p.messages) == len(p1.messages) + 1
    last = p.messages[-1]
    assert (
        last.role == "assistant"
        and len(last.content) == 1
        and isinstance(last.content[0], TextPart)
    )
    assert (p.tools, p.thinking, p.assists) == (TOOLS, thinking, ASSISTS)
    assert p.chat_template_kwargs() == p1.chat_template_kwargs()
    assert p.continue_final_message is True
    assert last.content[0].text.endswith(RESP_CLOSE)
    assert last.content[0].text.count(RESP_CLOSE) == n_steps


def test_two_steps_equal_the_oracle_form_d_after_the_step_one_prefix() -> None:
    loop = _loop()
    specs = [
        ("Let me look that up. ", "google_search", {"query": "weather in Paris"}, SEARCH_RESPONSE),
        ("", "click_at", {"x": 412, "y": 300}, CLICK_RESPONSE),
    ]
    text = loop.step_prompt(_prompt1(), _steps(loop, specs)).messages[-1].content[0].text
    assert text == "Let me look that up. " + FORM_D_TWO_STEPS


def test_step_text_sits_between_the_previous_response_and_its_call() -> None:
    loop = _loop()
    specs = [
        ("Let me look that up. ", "google_search", {"query": "weather in Paris"}, SEARCH_RESPONSE),
        ("Opening the first result.", "click_at", {"x": 412, "y": 300}, CLICK_RESPONSE),
    ]
    text = loop.step_prompt(_prompt1(), _steps(loop, specs)).messages[-1].content[0].text
    cut = FORM_D_TWO_STEPS.index("<|tool_call>call:click_at")
    want = (
        "Let me look that up. "
        + FORM_D_TWO_STEPS[:cut]
        + "Opening the first result."
        + FORM_D_TWO_STEPS[cut:]
    )
    assert text == want


def test_an_eight_step_continuation_stays_under_14000_chars() -> None:
    """Regime: 8 searches, each backend answer 5 results of 120-char titles, 100-char urls and 600-char snippets
    (every one over its bound), 60-char queries, 80-char step texts, a 120-char step-1 prefix."""
    loop = _loop()
    rows = [
        {
            "title": f"t{i}" + "T" * 118,
            "url": f"https://example.org/{i}/" + "u" * 80,
            "snippet": "S" * 600,
        }
        for i in range(5)
    ]
    specs = []
    for k in range(8):
        query = f"query {k} " + "q" * 52
        payload = loop.search_payload(_results(rows, query))
        specs.append(
            ("P" * 120 if k == 0 else "w" * 80, "google_search", {"query": query}, payload)
        )
    text = loop.step_prompt(_prompt1(), _steps(loop, specs)).messages[-1].content[0].text
    assert text.count(RESP_CLOSE) == 8
    assert len(text) <= 14_000, len(text)


# --------------------------------------------------------------------------- criterion 12 (negative): no diff

#: T-0113's closed range (T-0094 "pin both ends"; .claude/rules/tests.md). BASE is the task-file commit the branch
#: started from (4a64c3e, "T-0111a2: task file"); TIP is the verified branch head (d1945ce, the last commit before
#: this conversion, which edits tests only). The only commit between BASE and the branch's merge-base with main
#: (2c31168) touches docs/tasks only, so the range covers every T-0113 change to the guarded paths.
T0113_BASE = "4a64c3e4da7329430d89c47a162bafe0597f0eeb"
T0113_TIP = "d1945ce19dc492f65247ccd974b48f2e9aba0892"
#: The power pair: T-0102's commit 9b283f1 and its parent b81f5e6, which edits ``live.py``.
POWER_BASE = "b81f5e6efa91b984aafcff121abf3dd178f937de"
POWER_TIP = "9b283f18797c6d6336f82b1a61691902d019f45a"
GUARDED = (
    "local_astra/toolcall/router.py",
    "local_astra/toolcall/native.py",
    "local_astra/toolcall/score.py",
    "local_astra/toolcall/tools.py",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/toolcall/live.py",
    "local_astra/conductor/tool_executor.py",
)


def test_the_existing_toolcall_modules_are_unchanged() -> None:
    """Criterion 12 (negative) over T-0113's closed range BASE..TIP (two literal commits, never the working tree):
    the loop is a new module, never an edit of the call path. Skips, naming the SHA, if either commit is absent."""
    gr.require_commits(T0113_BASE, T0113_TIP)
    assert gr.is_ancestor(T0113_BASE, T0113_TIP), f"{T0113_BASE} is not an ancestor of {T0113_TIP}"
    assert gr.changed(T0113_BASE, T0113_TIP, *GUARDED) == []


def test_the_no_diff_guard_has_power() -> None:
    """Adversarial twin: the same predicate and path set over T-0102's commit (b81f5e6..9b283f1) sees its
    ``live.py`` edit; a predicate that reports nothing fails here."""
    gr.require_commits(POWER_BASE, POWER_TIP)
    assert gr.changed(POWER_BASE, POWER_TIP, *GUARDED) == ["local_astra/toolcall/live.py"]


def test_the_closed_range_is_t0113_s_own_change() -> None:
    """Too-wide twin: BASE..TIP is the T-0113 change itself (``loop.py`` is added in it), not an empty or reversed
    range that would make the guard vacuous."""
    gr.require_commits(T0113_BASE, T0113_TIP)
    assert gr.changed(T0113_BASE, T0113_TIP, "local_astra/toolcall/loop.py", diff_filter="A") == [
        "local_astra/toolcall/loop.py"
    ]
