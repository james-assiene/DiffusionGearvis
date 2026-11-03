"""T-0113 criterion 10 against the live oracle (``ASTRA_CD_TESTS=1``; the oracle needs jinja2, in the ``cd`` group).

The pinned template (``tests/cd/oracle/gemma4_template.py``, sha ``36e3a42e...``) renders, and ``loop`` must equal it:

- ``render_call(name, arguments)`` == the template's call text, for every v1 tool and for general argument trees
  (nested lists and mappings, booleans, ints, mixed-case keys: ``dictsort`` is case-insensitive);
- ``render_response(name, payload)`` == the ``format_tool_response_block`` the template renders for the same
  payload, mapping form (legacy ``tool_responses``, lines 263-269) and string form (``{value:...}``, line 170);
- ``step_prompt`` with two steps == form D: the template's rendering of the same steps as structured messages (one
  assistant message per step with ``tool_calls`` and ``tool_responses``, no content), after the step-1 prefix;
- end to end, with an empty step-1 text: the template's rendering of the step prompt (step-1 messages plus the
  continued assistant message) cut after the continuation is byte-identical to its rendering of form D. That is
  what vLLM sees for form G (T-0087 q9: the prompt is cut after the final message's content).

Strings are drawn without ``<``, ``>``, ``|`` and ``"``, so no special literal can occur and the sanitiser is the
identity (injection is ``tests/unit/test_toolcall_loop_injection.py``); trees are small, so no bound is reached.

Regime: CPU; model rev 2ea8372; template sha 36e3a42e...; Hypothesis seeds 113010/113011/113012, no example
database, 100 examples per row.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any

import pytest
from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

from local_astra.contracts.llm import Message, Prompt, TextPart
from tests.cd.oracle import gemma4_template as tpl

LOOP_MODULE = "local_astra.toolcall.loop"
SETTINGS = settings(
    max_examples=100, deadline=None, database=None, suppress_health_check=[HealthCheck.too_slow]
)
USER = {"role": "user", "content": "What is the weather in Paris?"}


def _loop() -> ModuleType:
    try:
        return importlib.import_module(LOOP_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == LOOP_MODULE:
            raise NotImplementedError(f"T-0113: {LOOP_MODULE} does not exist yet") from exc
        raise


def _assistant(name: str, arguments: Mapping[str, Any], response: Any | None) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c0",
                "type": "function",
                "function": {"name": name, "arguments": dict(arguments)},
            }
        ],
    }
    if response is not None:
        msg["tool_responses"] = [{"name": name, "response": response}]
    return msg


def oracle_model_turn(steps: Sequence[tuple[str, Mapping[str, Any], Any]]) -> str:
    """Form D: the template's model turn for structured steps, after ``<|turn>model\\n``."""
    text = tpl.render([USER, *(_assistant(n, a, r) for n, a, r in steps)])
    return text[text.rindex(tpl.MODEL_TURN) + len(tpl.MODEL_TURN) :]


def oracle_response(name: str, arguments: Mapping[str, Any], payload: Any) -> str:
    turn = oracle_model_turn([(name, arguments, payload)])
    call = turn.index("<tool_call|>") + len("<tool_call|>")
    return turn[call:]


def test_the_oracle_response_cut_is_the_response_block() -> None:
    """Self-check (no ``loop`` needed): the cut starts at ``<|tool_response>`` and ends at ``<tool_response|>``."""
    got = oracle_response("scroll", {"direction": "up"}, {"status": "ok"})
    assert got == '<|tool_response>response:scroll{status:<|"|>ok<|"|>}<tool_response|>'


# --------------------------------------------------------------------------- strategies

TEXT = st.text(
    alphabet=st.characters(
        blacklist_characters='<>|"',
        blacklist_categories=("Cs",),
        min_codepoint=32,
        max_codepoint=0x2FF,
    ),
    max_size=12,
)
KEY = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,5}", fullmatch=True)
LEAF = st.one_of(TEXT, st.booleans(), st.integers(-(10**6), 10**6))
TREE = st.recursive(
    LEAF,
    lambda inner: st.one_of(st.lists(inner, max_size=3), st.dictionaries(KEY, inner, max_size=3)),
    max_leaves=8,
)
ARGS = st.dictionaries(KEY, TREE, max_size=4)
NAME = st.sampled_from(["google_search", "click_at", "type_at", "scroll", "press_key"])

V1_CALLS = [
    ("google_search", {"query": "weather in Paris"}),
    ("click_at", {"x": 412, "y": 300}),
    ("type_at", {"x": 0, "y": 999, "text": "hello, world: 1", "press_enter": False}),
    ("scroll", {"direction": "left"}),
    ("press_key", {"combo": "Alt+Tab"}),
]


def _prompt1() -> Prompt:
    return Prompt(
        messages=(Message(role="user", content=(TextPart(text=USER["content"]),)),),
        prefix_len=1,
        thinking=True,
        tools=None,
    )


def _continuation(steps: Sequence[tuple[str, str, Mapping[str, Any], Any]]) -> str:
    loop = _loop()
    p = loop.step_prompt(
        _prompt1(),
        [loop.LoopStep(text=t, name=n, arguments=dict(a), response=r) for t, n, a, r in steps],
    )
    text: str = p.messages[-1].content[0].text
    return text


# --------------------------------------------------------------------------- render_call


@pytest.mark.parametrize("name, arguments", V1_CALLS, ids=[c[0] for c in V1_CALLS])
def test_render_call_equals_the_oracle_for_every_v1_tool(
    name: str, arguments: dict[str, Any]
) -> None:
    assert _loop().render_call(name, arguments) == tpl.call_text(name, arguments)


@seed(113010)
@SETTINGS
@given(name=NAME, arguments=ARGS)
def test_render_call_equals_the_oracle_on_argument_trees(
    name: str, arguments: dict[str, Any]
) -> None:
    assert _loop().render_call(name, arguments) == tpl.call_text(name, arguments)


# --------------------------------------------------------------------------- render_response


@seed(113011)
@SETTINGS
@given(name=NAME, payload=st.one_of(ARGS, TEXT))
def test_render_response_equals_the_oracle_in_both_forms(name: str, payload: Any) -> None:
    """Mapping payloads take the legacy ``tool_responses`` path; a string takes ``{value:...}``."""
    assert _loop().render_response(name, payload) == oracle_response(name, {"q": "x"}, payload)


def test_the_builders_render_like_the_oracle() -> None:
    loop = _loop()
    cases = [
        ("google_search", loop.search_payload(None)),
        ("click_at", loop.action_payload("would click at (412, 300)")),
        ("type_at", loop.status_payload("refused", "not allowed")),
        ("scroll", loop.status_payload("failed", "boom")),
        ("press_key", loop.status_payload("timeout", "no result within 15 s")),
    ]
    for name, payload in cases:
        assert loop.render_response(name, payload) == oracle_response(name, {"q": "x"}, payload), (
            name
        )


# --------------------------------------------------------------------------- step_prompt: form D


@pytest.mark.parametrize("prefix", ["", "Let me look that up. "])
def test_two_steps_equal_form_d_after_the_step_one_prefix(prefix: str) -> None:
    steps = [
        (
            "google_search",
            {"query": "weather in Paris"},
            {"query": "weather in Paris", "results": []},
        ),
        (
            "click_at",
            {"x": 412, "y": 300},
            {"action": "would click", "screen_changed": False, "status": "dry_run"},
        ),
    ]
    text = _continuation([(prefix, *steps[0]), ("", *steps[1])])
    assert text == prefix + oracle_model_turn(steps)


@seed(113012)
@SETTINGS
@given(
    steps=st.lists(st.tuples(NAME, ARGS, st.one_of(ARGS, TEXT)), min_size=2, max_size=2),
)
def test_any_two_steps_equal_form_d(steps: list[tuple[str, dict[str, Any], Any]]) -> None:
    text = _continuation([("", n, a, r) for n, a, r in steps])
    assert text == oracle_model_turn(steps)


def test_the_rendered_step_prompt_is_form_d_up_to_the_cut() -> None:
    """End to end: the template renders the continued assistant message (``strip_thinking`` of its content) right
    after ``<|turn>model\\n``; cut after the continuation, that is byte-identical to the template's rendering of the
    structured form D (thinking on in both)."""
    steps = [
        (
            "google_search",
            {"query": "weather in Paris"},
            {"query": "weather in Paris", "results": []},
        ),
        ("scroll", {"direction": "down"}, "scrolled"),
    ]
    text = _continuation([("", *s) for s in steps])
    g = tpl.render([USER, {"role": "assistant", "content": text}], enable_thinking=True)
    d = tpl.render([USER, *(_assistant(n, a, r) for n, a, r in steps)], enable_thinking=True)
    end = g.index(text) + len(text)
    assert g[:end] == d
