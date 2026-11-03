"""T-0084a: native Gemma 4 tool-call grammars (``call_format="native"``), by host replay and oracles.

``compile_toolcall(tools, *, scope, thought, whitespace="json", call_format="native")`` over the v1 tools.
Every expected language comes from one of three places, never from the implementation:

- **the pinned chat template** (``tests/cd/oracle/gemma4_template.py``, sha ``36e3a42e...``): its
  ``tool_calls`` path renders ``<|tool_call>call:NAME{k:v,...}<tool_call|>`` with dictsorted keys,
  ``format_argument`` values (bare integers, ``true``/``false``, strings in ``<|"|>``), and, after a call
  with no content, ``<|tool_response>`` (stop 50);
- **the vendored vLLM v0.29.0 argument parser** (``tests/cd/oracle/vllm_gemma4_args.py``, upstream sha
  ``478c04e6...``), applied to the decoded random walks of the automata (the language-side oracle);
- **hand construction** from criteria 2-4 and 7 (the rejected sets, the BODY rule, the variants).

Token ids come from the pinned HF ``tokenizer.json`` (``tests/cd/_compile_support.encode``): 48
``<|tool_call>``, 49 ``<tool_call|>``, 52 ``<|"|>``, 46/47 ``<|tool>``/``<tool|>``, 51 ``<tool_response|>``,
98 ``<|think|>``, 100/101 the channel markers, 107 ``\\n``, 45518 ``thought``, stops ``{1, 106, 50}``.
Acceptance is decided by ``eval/cd/replay.accepts`` and must agree with the independent set-of-states
``Walker``. Every automaton is ``FA . STOP . Sigma*``, so an accepted sequence ends with a stop id.

Variants (criterion 4): ``T4`` = CALL (the pass-2 prompt ends with 48); ``T2`` = the empty-thought header
. 48 . CALL; ``T1`` = ``100 . name . 107 . FREE* . 101 . 48 . CALL`` with FREE = the JSON T1 free set minus
{48, 49} (QN5); CALL = ``call:`` NAME ``{`` ARGS ``}`` . 49. ``scope="name"`` = [header] [48, not in T4]
``call:`` NAME ``{`` + STOP.

Regime: CPU; tokenizer.json sha256 cc8d3a0c...; model rev 2ea8372; template sha 36e3a42e...; tools_v1
bbfc53cc...; outlines-core 0.2.14; Hypothesis seed 20261009 (``@seed``, no example database), 200
examples per tool; random walks seeded ``random.Random(84_000 + i)``, 500 per tool.
"""

from __future__ import annotations

import functools
import itertools
import json
import random
import re
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, seed, settings
from hypothesis import strategies as st

from tests import _git_range as gr
from tests.cd._compile_support import VOCAB_SIZE, encode, hf_tokenizer, walker
from tests.cd.oracle import gemma4_template as tpl
from tests.cd.oracle import vllm_gemma4_args as vparse
from tests.unit._toolcall_support import combo_enum, need, shipped_params

PAD, EOS, TOOL_RESPONSE, OPEN, CLOSE, TURN_OPEN, TURN_CLOSE = 0, 1, 50, 100, 101, 105, 106
NL, THOUGHT = 107, 45518
TC_OPEN, TC_CLOSE, DELIM = 48, 49, 52
DECL_OPEN, DECL_CLOSE, TOOL_RESPONSE_CLOSE, THINK = 46, 47, 51, 98
STOPS = (EOS, TURN_CLOSE, TOOL_RESPONSE)
EMPTY_HEADER = [OPEN, THOUGHT, NL, CLOSE]
#: ``<``, ``|``, ``"``, ``|``, ``>`` as single-character pieces, and as byte-fallback pieces ``<0xXX>``.
DELIM_CHARS = [236820, 236909, 236775, 236909, 236813]
DELIM_BYTES = [298, 362, 272, 362, 300]

TOOLS = ("google_search", "click_at", "type_at", "scroll", "press_key")
THOUGHTS = ("T1", "T2", "T4")
LIVE_SCOPES = (*TOOLS, "name")
LIVE_MAX_STATES = 128
CAP_CHARS = 1 << 20
Q = '<|"|>'
END = "<tool_call|>"
#: Added-token surfaces the HF tokenizer parses out of plain text (they are not ordinary text).
SEED = 20261009

REAL_THOUGHT = "The user wants the weather.\nI will search for it.\n\nThen answer briefly."


# --------------------------------------------------------------------------- code under test


def toolcall() -> Any:
    return need("local_astra.decoding.toolcall")


@functools.cache
def toolset() -> Any:
    return need("local_astra.toolcall.tools").load()


@functools.cache
def native(scope: str, thought: str) -> Any:
    return toolcall().compile_toolcall(
        toolset(), scope=scope, thought=thought, call_format="native"
    )


def accepts_with(a: Any, ids: Sequence[int]) -> bool:
    from local_astra.eval.cd import replay

    got = replay.accepts(a, list(ids))
    oracle = walker(a).accepts(list(ids))
    assert got == oracle, f"host replay ({got}) and the walker ({oracle}) disagree on {list(ids)}"
    return bool(got)


def accepts(scope: str, thought: str, ids: Sequence[int]) -> bool:
    return accepts_with(native(scope, thought), ids)


# --------------------------------------------------------------------------- sequences


def nat(text: str) -> list[int]:
    """HF tokenisation; ``<|"|>`` and ``<tool_call|>`` written in ``text`` become ids 52 and 49."""
    return list(encode(text))


def thought_header(thought: str) -> list[int]:
    """What precedes 48 in each variant (T4: nothing, the prompt already ends with 48)."""
    if thought == "T1":
        return [OPEN, THOUGHT, NL, *nat(REAL_THOUGHT), CLOSE, TC_OPEN]
    if thought == "T2":
        return [*EMPTY_HEADER, TC_OPEN]
    return []


def seq(thought: str, after48: Sequence[int], *, stop: int = TOOL_RESPONSE) -> list[int]:
    return [*thought_header(thought), *after48, stop]


def char_pieces(text: str) -> list[int]:
    """One token per character (each character's own HF encoding): another tokenisation of ``text``."""
    out: list[int] = []
    for ch in text:
        out.extend(encode(ch))
    return out


def byte_pieces(text: str) -> list[int]:
    """One byte-fallback piece ``<0xXX>`` per UTF-8 byte of ``text``."""
    tok = hf_tokenizer()
    return [int(tok.token_to_id(f"<0x{b:02X}>")) for b in text.encode("utf-8")]


def decode(ids: Sequence[int]) -> str:
    return str(hf_tokenizer().decode(list(ids), skip_special_tokens=False))


def template_call_ids(name: str, args: Mapping[str, Any]) -> list[int]:
    """The template's call, tokenised: ``48 ... 49``."""
    ids = nat(tpl.call_text(name, args))
    assert ids[0] == TC_OPEN and ids[-1] == TC_CLOSE, ids
    return ids


def valid_args(name: str) -> dict[str, Any]:
    return {
        "google_search": {"query": "weather in Paris"},
        "click_at": {"x": 999, "y": 0},
        "type_at": {"x": 10, "y": 20, "text": "hello", "press_enter": True},
        "scroll": {"direction": "down"},
        "press_key": {"combo": "Ctrl+C"},
    }[name]


def dictsort(keys: Sequence[str]) -> list[str]:
    """jinja's ``dictsort`` default: case-insensitive by key."""
    return sorted(keys, key=str.lower)


# =========================================================================== criterion 5: sizes


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("scope", LIVE_SCOPES)
def test_live_native_automata_have_at_most_128_states_and_are_dfas(
    scope: str, thought: str
) -> None:
    from local_astra.decoding import artifact

    a = native(scope, thought)
    assert int(a.n_states) <= LIVE_MAX_STATES, (scope, thought, int(a.n_states))
    assert bool(a.is_dfa), f"native {scope}/{thought} is not a DFA"
    assert len(artifact.encode_automaton(a)) <= CAP_CHARS
    print(
        f"T-0084a native size: scope={scope} thought={thought} |S|={a.n_states} "
        f"bucket={a.n_states_bucket}"
    )


@pytest.mark.parametrize("thought", THOUGHTS)
def test_native_union_compiles_and_is_reported(thought: str) -> None:
    """Union stays eval-only (criterion 4): compiled and encoded under the cap, size printed, no gate."""
    from local_astra.decoding import artifact

    a = native("union", thought)
    assert len(artifact.encode_automaton(a)) <= CAP_CHARS
    for tool in TOOLS:
        assert accepts(
            "union", thought, seq(thought, template_call_ids(tool, valid_args(tool))[1:])
        )
    print(f"T-0084a native size: scope=union thought={thought} |S|={a.n_states}")


# =========================================================================== criterion 6: template oracle

_ADDED = (
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "<mask>",
)


def _plain(s: str) -> bool:
    """Text the tokenizer treats as text and round-trips (no added-token surface, no ``<|``/``|>``)."""
    if "<|" in s or "|>" in s or any(a in s for a in _ADDED):
        return False
    return decode(encode(s)) == s


_TEXT = st.text(alphabet=st.characters(exclude_categories=("Cs",)), min_size=1, max_size=24).filter(
    _plain
)


def args_strategy(name: str) -> st.SearchStrategy[dict[str, Any]]:
    props = shipped_params(name)["properties"]
    fields: dict[str, st.SearchStrategy[Any]] = {}
    for key, schema in props.items():
        if schema.get("type") == "integer":
            fields[key] = st.integers(int(schema["minimum"]), int(schema["maximum"]))
        elif schema.get("type") == "boolean":
            fields[key] = st.booleans()
        elif "enum" in schema:
            fields[key] = st.sampled_from(list(schema["enum"]))
        else:
            assert schema.get("type") == "string" and schema.get("minLength") == 1, (name, key)
            fields[key] = _TEXT
    return st.fixed_dictionaries(fields)


_THOUGHT_TEXT = st.lists(
    st.text(
        alphabet=st.characters(exclude_categories=("Cs", "Zl", "Zp", "Cc")), min_size=1, max_size=30
    ).filter(lambda s: s == s.strip() and _plain(s)),
    min_size=2,
    max_size=4,
).map("\n".join)


def check_template_call(name: str, args: Mapping[str, Any], reasoning: str) -> None:
    call = template_call_ids(name, args)
    # T4: the call after 48, then the stop the template writes next (50).
    assert accepts(name, "T4", [*call[1:], TOOL_RESPONSE]), ("T4", name, args)
    # T2: the empty header (as the generation prompt renders it) . 48 . call . 50.
    header = nat(tpl.empty_thought_header())
    assert header == EMPTY_HEADER, header
    assert accepts(name, "T2", [*header, *call, TOOL_RESPONSE]), ("T2", name, args)
    # T1: the template's whole model turn, thought included, ending with <|tool_response> (50).
    turn = nat(tpl.model_turn(name, args, reasoning))
    assert turn[:3] == [OPEN, THOUGHT, NL] and turn[-1] == TOOL_RESPONSE, turn[:4]
    assert NL in turn[3 : turn.index(CLOSE)], "the thought must be multi-line"
    assert accepts(name, "T1", turn), ("T1", name, args, reasoning)


@pytest.mark.parametrize("name", TOOLS)
@seed(SEED)
@settings(
    max_examples=200,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)
@given(data=st.data())
def test_every_template_rendering_is_accepted(name: str, data: st.DataObject) -> None:
    """Criterion 6 (n = 200 per tool, seed 20261009): every schema-valid v1 call, rendered by the pinned
    template's ``tool_calls`` path, is accepted under T4, T2 and T1."""
    args = data.draw(args_strategy(name), label="args")
    reasoning = data.draw(_THOUGHT_TEXT, label="reasoning")
    assume(NL in nat(reasoning))
    check_template_call(name, args, reasoning)


EDGE_STRINGS = [
    "<",
    "|",
    ">",
    '"',
    "}",
    "{",
    ",",
    ":",
    "\\",
    "\n",
    "a\nb",
    "日本語 café",
    " x",
    "x ",
    "  ",
    "a<b|c>d",
    "|<",
    ">|",
    "<>",
    "'",
    "[1,2]",
    "key:value,k2:v2",
    "call:x{}",
    "true",
    "0",
    "x" * 80,
]


@pytest.mark.parametrize("text", EDGE_STRINGS, ids=repr)
def test_template_renderings_with_edge_strings_are_accepted(text: str) -> None:
    assert _plain(text), "row premise"
    check_template_call("google_search", {"query": text}, REAL_THOUGHT)
    check_template_call(
        "type_at", {"x": 0, "y": 999, "text": text, "press_enter": False}, REAL_THOUGHT
    )


@pytest.mark.parametrize("v", [0, 1, 9, 10, 99, 100, 500, 998, 999])
def test_template_renderings_with_edge_coordinates_are_accepted(v: int) -> None:
    check_template_call("click_at", {"x": v, "y": 999 - v}, REAL_THOUGHT)
    check_template_call("type_at", {"x": v, "y": v, "text": "a", "press_enter": True}, REAL_THOUGHT)


@pytest.mark.parametrize("name,key", [("scroll", "direction"), ("press_key", "combo")])
def test_every_enum_member_rendered_is_accepted(name: str, key: str) -> None:
    for member in shipped_params(name)["properties"][key]["enum"]:
        check_template_call(name, {key: member}, REAL_THOUGHT)
    assert len(combo_enum()) >= 20


def test_template_key_order_is_dictsort_for_v1() -> None:
    """The oracle's key order (jinja dictsort, case-insensitive) for the v1 tools, from the template."""
    for name in TOOLS:
        props = list(shipped_params(name)["properties"])
        text = tpl.call_text(name, valid_args(name))
        order = re.findall(r"[{,]([A-Za-z_]+):", text)
        assert order == dictsort(props), (name, text)
    assert dictsort(list(shipped_params("type_at")["properties"])) == [
        "press_enter",
        "text",
        "x",
        "y",
    ]


# =========================================================================== criterion 7: rejections

#: (row id, rejected text after 48, accepted control after 48). Every rejected row is paired with the
#: closest accepted call, so a grammar that rejects everything fails the row too.
CASES: list[tuple[str, str, str]] = [
    ("ws-after-comma", "call:click_at{x:5, y:7}" + END, "call:click_at{x:5,y:7}" + END),
    ("ws-after-colon", "call:click_at{x: 5,y:7}" + END, "call:click_at{x:5,y:7}" + END),
    ("ws-before-brace", "call:click_at {x:5,y:7}" + END, "call:click_at{x:5,y:7}" + END),
    ("ws-after-call", "call: click_at{x:5,y:7}" + END, "call:click_at{x:5,y:7}" + END),
    ("ws-after-open", "call:click_at{ x:5,y:7}" + END, "call:click_at{x:5,y:7}" + END),
    ("ws-before-close", "call:click_at{x:5,y:7} " + END, "call:click_at{x:5,y:7}" + END),
    ("newline-in-call", "call:click_at{x:5,\ny:7}" + END, "call:click_at{x:5,y:7}" + END),
    (
        "ws-around-string",
        f"call:scroll{{direction: {Q}up{Q}}}" + END,
        f"call:scroll{{direction:{Q}up{Q}}}" + END,
    ),
    ("keys-swapped", "call:click_at{y:7,x:5}" + END, "call:click_at{x:5,y:7}" + END),
    (
        "keys-declared-order",
        f"call:type_at{{x:1,y:2,text:{Q}hi{Q},press_enter:true}}" + END,
        f"call:type_at{{press_enter:true,text:{Q}hi{Q},x:1,y:2}}" + END,
    ),
    ("missing-key", "call:click_at{x:5}" + END, "call:click_at{x:5,y:7}" + END),
    ("extra-key", "call:click_at{x:5,y:7,z:1}" + END, "call:click_at{x:5,y:7}" + END),
    ("duplicate-key", "call:click_at{x:5,x:5,y:7}" + END, "call:click_at{x:5,y:7}" + END),
    ("x-1000", "call:click_at{x:1000,y:7}" + END, "call:click_at{x:999,y:7}" + END),
    ("x-minus-1", "call:click_at{x:-1,y:7}" + END, "call:click_at{x:0,y:7}" + END),
    ("x-007", "call:click_at{x:007,y:7}" + END, "call:click_at{x:7,y:7}" + END),
    ("x-00", "call:click_at{x:00,y:7}" + END, "call:click_at{x:0,y:7}" + END),
    ("x-plus", "call:click_at{x:+5,y:7}" + END, "call:click_at{x:5,y:7}" + END),
    ("x-float", "call:click_at{x:5.0,y:7}" + END, "call:click_at{x:5,y:7}" + END),
    ("y-1000", "call:click_at{x:5,y:1000}" + END, "call:click_at{x:5,y:999}" + END),
    ("x-quoted", f"call:click_at{{x:{Q}5{Q},y:7}}" + END, "call:click_at{x:5,y:7}" + END),
    ("x-empty", "call:click_at{x:,y:7}" + END, "call:click_at{x:5,y:7}" + END),
    (
        "bool-True",
        f"call:type_at{{press_enter:True,text:{Q}hi{Q},x:1,y:2}}" + END,
        f"call:type_at{{press_enter:true,text:{Q}hi{Q},x:1,y:2}}" + END,
    ),
    (
        "bool-quoted",
        f"call:type_at{{press_enter:{Q}true{Q},text:{Q}hi{Q},x:1,y:2}}" + END,
        f"call:type_at{{press_enter:false,text:{Q}hi{Q},x:1,y:2}}" + END,
    ),
    (
        "bool-1",
        f"call:type_at{{press_enter:1,text:{Q}hi{Q},x:1,y:2}}" + END,
        f"call:type_at{{press_enter:false,text:{Q}hi{Q},x:1,y:2}}" + END,
    ),
    (
        "enum-out-of-list",
        f"call:scroll{{direction:{Q}forward{Q}}}" + END,
        f"call:scroll{{direction:{Q}up{Q}}}" + END,
    ),
    (
        "enum-case",
        f"call:scroll{{direction:{Q}Up{Q}}}" + END,
        f"call:scroll{{direction:{Q}up{Q}}}" + END,
    ),
    (
        "enum-prefix",
        f"call:press_key{{combo:{Q}Ctrl{Q}}}" + END,
        f"call:press_key{{combo:{Q}Ctrl+C{Q}}}" + END,
    ),
    ("enum-bare", "call:scroll{direction:up}" + END, f"call:scroll{{direction:{Q}up{Q}}}" + END),
    (
        "empty-string-minlength-1",
        f"call:google_search{{query:{Q}{Q}}}" + END,
        f"call:google_search{{query:{Q}a{Q}}}" + END,
    ),
    (
        "string-unquoted",
        "call:google_search{query:weather}" + END,
        f"call:google_search{{query:{Q}weather{Q}}}" + END,
    ),
    (
        "body-has-<|",
        f"call:google_search{{query:{Q}a<|b{Q}}}" + END,
        f"call:google_search{{query:{Q}a<b{Q}}}" + END,
    ),
    (
        "body-has-|>",
        f"call:google_search{{query:{Q}a|>b{Q}}}" + END,
        f"call:google_search{{query:{Q}a|b{Q}}}" + END,
    ),
    (
        "body-is-<|",
        f"call:google_search{{query:{Q}<|{Q}}}" + END,
        f"call:google_search{{query:{Q}<{Q}}}" + END,
    ),
    (
        "body-is-|>",
        f"call:google_search{{query:{Q}|>{Q}}}" + END,
        f"call:google_search{{query:{Q}>{Q}}}" + END,
    ),
    (
        "type_at-body-has-<|",
        f"call:type_at{{press_enter:true,text:{Q}x<|y{Q},x:1,y:2}}" + END,
        f"call:type_at{{press_enter:true,text:{Q}x<y{Q},x:1,y:2}}" + END,
    ),
    (
        "missing-49",
        f"call:scroll{{direction:{Q}up{Q}}}",
        f"call:scroll{{direction:{Q}up{Q}}}" + END,
    ),
    (
        "missing-brace",
        f"call:scroll{{direction:{Q}up{Q}" + END,
        f"call:scroll{{direction:{Q}up{Q}}}" + END,
    ),
    (
        "text-after-49",
        f"call:scroll{{direction:{Q}up{Q}}}" + END + " ok",
        f"call:scroll{{direction:{Q}up{Q}}}" + END,
    ),
    (
        "space-after-49",
        f"call:scroll{{direction:{Q}up{Q}}}" + END + " ",
        f"call:scroll{{direction:{Q}up{Q}}}" + END,
    ),
    (
        "two-calls",
        "call:click_at{x:5,y:7}" + END + "<|tool_call>call:click_at{x:5,y:7}" + END,
        "call:click_at{x:5,y:7}" + END,
    ),
    ("no-call-prefix", "click_at{x:5,y:7}" + END, "call:click_at{x:5,y:7}" + END),
    ("wrong-name-case", "call:Click_at{x:5,y:7}" + END, "call:click_at{x:5,y:7}" + END),
    (
        "json-shape",
        '{"name": "click_at", "arguments": {"x": 5, "y": 7}}',
        "call:click_at{x:5,y:7}" + END,
    ),
    (
        "json-quotes-not-delim",
        'call:google_search{query:"weather"}' + END,
        f"call:google_search{{query:{Q}weather{Q}}}" + END,
    ),
]


def case_scope(text: str) -> str:
    m = re.search(r"call:([a-z_]+)\{", text)
    return m.group(1) if m and m.group(1) in TOOLS else "click_at"


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize(("bad", "good"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_rejected_calls_beside_their_accepted_control(thought: str, bad: str, good: str) -> None:
    scope = case_scope(good)
    assert accepts(scope, thought, seq(thought, nat(good))), ("control", good)
    assert not accepts(scope, thought, seq(thought, nat(bad))), ("rejected", bad)


@pytest.mark.parametrize("thought", THOUGHTS)
def test_a_call_to_another_tool_is_outside_the_scope(thought: str) -> None:
    other = template_call_ids("scroll", {"direction": "up"})[1:]
    assert accepts("scroll", thought, seq(thought, other))
    assert not accepts("click_at", thought, seq(thought, other))


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("stop", STOPS)
def test_a_stop_id_inside_a_body_is_rejected(thought: str, stop: int) -> None:
    pre, post = nat(f"call:google_search{{query:{Q}wea"), nat(f"ther{Q}}}" + END)
    assert accepts("google_search", thought, seq(thought, [*pre, *post]))
    assert not accepts("google_search", thought, seq(thought, [*pre, stop, *post]))


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("stop", STOPS)
def test_a_stop_id_inside_the_call_structure_is_rejected(thought: str, stop: int) -> None:
    pre, post = nat("call:click_at{x:5,"), nat("y:7}" + END)
    assert not accepts("click_at", thought, seq(thought, [*pre, stop, *post]))


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("stop", STOPS)
def test_every_stop_ends_the_call(thought: str, stop: int) -> None:
    """Stops are out of band (``FA . STOP . Sigma*``): any of the three ends a complete call."""
    call = template_call_ids("click_at", {"x": 1, "y": 2})[1:]
    assert accepts("click_at", thought, seq(thought, call, stop=stop))
    assert not accepts("click_at", thought, [*thought_header(thought), *call])  # no stop: not done


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize(
    "marker",
    [
        TC_OPEN,
        TC_CLOSE,
        DELIM,
        DECL_OPEN,
        DECL_CLOSE,
        TOOL_RESPONSE_CLOSE,
        THINK,
        OPEN,
        CLOSE,
        TURN_OPEN,
    ],
)
def test_a_marker_id_inside_a_body_is_rejected(thought: str, marker: int) -> None:
    """Adversarial: 49 inside a ``google_search`` body (and every other marker whose surface carries
    ``<|`` or ``|>``) must not be admitted as body text."""
    surface = decode([marker])
    assert "<|" in surface or "|>" in surface, (marker, surface)
    pre, post = nat(f"call:google_search{{query:{Q}a"), nat(f"b{Q}}}" + END)
    assert accepts("google_search", thought, seq(thought, [*pre, *post]))
    assert not accepts("google_search", thought, seq(thought, [*pre, marker, *post]))


# --------------------------------------------------------------------------- criterion 3: single ids


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("spelling", [DELIM_CHARS, DELIM_BYTES], ids=["chars", "byte-fallback"])
@pytest.mark.parametrize("where", ["open", "close", "both"])
def test_the_string_delimiter_is_only_id_52(thought: str, spelling: list[int], where: str) -> None:
    """Adversarial: the delimiter spelled as five byte-level tokens is not a delimiter."""
    assert decode(spelling) == Q, 'row premise: the five tokens spell <|"|>'
    head, body, tail = nat("call:google_search{query:"), nat("weather"), nat("}" + END)
    good = [*head, DELIM, *body, DELIM, *tail]
    assert accepts("google_search", thought, seq(thought, good))
    o = spelling if where in ("open", "both") else [DELIM]
    c = spelling if where in ("close", "both") else [DELIM]
    assert not accepts("google_search", thought, seq(thought, [*head, *o, *body, *c, *tail]))


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("spelling", [DELIM_CHARS, DELIM_BYTES], ids=["chars", "byte-fallback"])
def test_a_body_spelling_the_delimiter_is_rejected(thought: str, spelling: list[int]) -> None:
    """Adversarial (google_search): ``a`` + the five delimiter pieces + ``b`` inside a real string."""
    head, tail = nat(f"call:google_search{{query:{Q}a"), nat(f"b{Q}}}" + END)
    assert accepts("google_search", thought, seq(thought, [*head, *tail]))
    assert not accepts("google_search", thought, seq(thought, [*head, *spelling, *tail]))


def _spelled(text: str) -> list[int]:
    ids = char_pieces(text)
    assert decode(ids) == text and len(ids) == len(text), (text, ids)
    return ids


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("pieces", ["chars", "byte-fallback"])
def test_the_call_close_is_only_id_49(thought: str, pieces: str) -> None:
    spelled = _spelled(END) if pieces == "chars" else byte_pieces(END)
    assert decode(spelled) == END
    body = nat("call:click_at{x:5,y:7}")
    assert accepts("click_at", thought, seq(thought, [*body, TC_CLOSE]))
    assert not accepts("click_at", thought, seq(thought, [*body, *spelled]))


@pytest.mark.parametrize("thought", ["T1", "T2"])
@pytest.mark.parametrize("pieces", ["chars", "byte-fallback"])
def test_the_call_open_is_only_id_48(thought: str, pieces: str) -> None:
    spelled = _spelled("<|tool_call>") if pieces == "chars" else byte_pieces("<|tool_call>")
    assert decode(spelled) == "<|tool_call>"
    call = nat("call:click_at{x:5,y:7}" + END)
    head = thought_header(thought)
    assert head[-1] == TC_OPEN
    assert accepts("click_at", thought, [*head, *call, EOS])
    assert not accepts("click_at", thought, [*head[:-1], *spelled, *call, EOS])


def test_t4_rejects_a_spelled_out_call_open() -> None:
    call = nat("call:click_at{x:5,y:7}" + END)
    assert accepts("click_at", "T4", [*call, EOS])
    assert not accepts("click_at", "T4", [*_spelled("<|tool_call>"), *call, EOS])


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("pieces", ["chars", "byte-fallback", "mixed"])
def test_plain_text_is_admitted_in_every_tokenisation(thought: str, pieces: str) -> None:
    """Too narrow: ``call:``, the name, keys, values and bodies, re-tokenised one character (or one byte)
    per token, with only 49/52 as single ids."""

    def tok(text: str, i: int) -> list[int]:
        if pieces == "chars" or (pieces == "mixed" and i % 2 == 0):
            return char_pieces(text)
        return byte_pieces(text)

    parts: list[int] = []
    for i, chunk in enumerate(
        [
            "call:type_at{press_enter:false,text:",
            DELIM,
            "Héllo, wörld",
            DELIM,
            ",x:120,y:7}",
            TC_CLOSE,
        ]
    ):
        parts.extend([chunk] if isinstance(chunk, int) else tok(chunk, i))
    assert (
        decode(parts) == f"call:type_at{{press_enter:false,text:{Q}Héllo, wörld{Q},x:120,y:7}}{END}"
    )
    assert accepts("type_at", thought, seq(thought, parts))


def _call_region(a: Any, thought: str) -> tuple[Any, set[int]]:
    """The walker and the states of the call region: T4's start state, or (T1/T2) the target of the 48
    edge leaving the state the header's 101 enters. ACC (the stop's Sigma* tail) is excluded."""
    w = walker(a)
    acc = w.acc()
    if thought == "T4":
        starts = set(w.start)
    else:
        b = w.body_start()
        starts = {d for d, c in w.out.get(b, ()) if w.member(c, TC_OPEN)}
        others = [c for d, c in w.out.get(b, ()) if not w.member(c, TC_OPEN)]
        assert not others, "criterion 4: nothing but 48 may follow 101"
    assert len(starts) == 1, starts
    seen, todo = set(starts), list(starts)
    while todo:
        s = todo.pop()
        for d, _ in w.out.get(s, ()):
            if d != acc and d not in seen:
                seen.add(d)
                todo.append(d)
    return w, seen


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("scope", [*TOOLS, "name"])
def test_the_markers_are_admitted_only_as_their_own_single_id(scope: str, thought: str) -> None:
    """Criterion 3, structurally: from the call's start on, every transition that admits 48, 49 or 52
    admits that id alone. A class holding another token too means some other piece (a byte-fallback
    spelling of a placeholder, a multi-character piece) also opens, closes or delimits."""
    w, region = _call_region(native(scope, thought), thought)
    wide = []
    for s in sorted(region):
        for d, c in w.out.get(s, ()):
            for marker in (TC_OPEN, TC_CLOSE, DELIM):
                if w.member(c, marker) and w.class_size(c) != 1:
                    wide.append((s, d, marker, w.class_size(c)))
    assert not wide, wide[:10]


# --------------------------------------------------------------------------- BODY: both directions

TOO_NARROW_BODIES = [
    '"',
    "}",
    ",",
    ":",
    "{",
    "\\",
    "\n",
    "日本語",
    "café",
    "<",
    "|",
    ">",
    "|<",
    ">|",
    "<>",
    "a<b|c>d",
    "<<",
    "||",
    ">>",
    " ",
    "\t",
    "😀",
]


@pytest.mark.parametrize("thought", THOUGHTS)
@pytest.mark.parametrize("body", TOO_NARROW_BODIES, ids=repr)
def test_bodies_with_json_and_delimiter_lookalike_characters_are_accepted(
    thought: str, body: str
) -> None:
    for ids in (nat(body), char_pieces(body), byte_pieces(body)):
        assert decode(ids) == body
        call = [*nat("call:google_search{query:"), DELIM, *ids, DELIM, *nat("}" + END)]
        assert accepts("google_search", thought, seq(thought, call)), (body, ids)


@functools.cache
def _body_alphabet_strings() -> tuple[str, ...]:
    return tuple("".join(t) for n in range(0, 5) for t in itertools.product("<|>a", repeat=n))


def test_body_rule_matches_the_predicate_exhaustively_to_length_4() -> None:
    """BODY = no ``<|`` and no ``|>``, non-empty under ``minLength: 1``: every string over ``< | > a`` up
    to length 4 (341 strings), tokenised by HF and one character per token, against the predicate."""
    wrong = []
    for s in _body_alphabet_strings():
        want = bool(s) and "<|" not in s and "|>" not in s
        for ids in {tuple(nat(s)), tuple(char_pieces(s))}:
            call = [*nat("call:google_search{query:"), DELIM, *ids, DELIM, *nat("}" + END)]
            got = accepts("google_search", "T4", [*call, EOS])
            if got != want:
                wrong.append((s, ids, got))
    n_ok = sum(1 for s in _body_alphabet_strings() if s and "<|" not in s and "|>" not in s)
    assert 0 < n_ok < len(_body_alphabet_strings())
    assert not wrong, wrong[:10]


@seed(SEED)
@settings(max_examples=300, deadline=None, database=None)
@given(s=st.text(alphabet='<|>"a {}\n:,', min_size=0, max_size=12))
def test_body_rule_matches_the_predicate_on_random_strings(s: str) -> None:
    """n = 300, seed 20261009: random strings over delimiter-like characters, HF tokenisation."""
    want = bool(s) and "<|" not in s and "|>" not in s
    ids = nat(s)
    assume(decode(ids) == s)
    call = [*nat("call:google_search{query:"), DELIM, *ids, DELIM, *nat("}" + END)]
    assert accepts("google_search", "T4", [*call, EOS]) == want, (s, ids)


# =========================================================================== criterion 4: variants


def test_t4_rejects_a_leading_48() -> None:
    call = template_call_ids("scroll", {"direction": "up"})
    assert accepts("scroll", "T4", [*call[1:], EOS])
    assert not accepts("scroll", "T4", [*call, EOS])


@pytest.mark.parametrize("thought", ["T1", "T2"])
def test_t4_rejects_any_header(thought: str) -> None:
    call = template_call_ids("scroll", {"direction": "up"})
    assert not accepts("scroll", "T4", [*thought_header(thought), *call[1:], EOS])


@pytest.mark.parametrize("body", ["hmm", "The user wants to scroll.\nOk.", " "])
def test_t2_rejects_thought_text(body: str) -> None:
    call = template_call_ids("scroll", {"direction": "up"})
    assert accepts("scroll", "T2", [*EMPTY_HEADER, *call, EOS])
    assert not accepts("scroll", "T2", [OPEN, THOUGHT, NL, *nat(body), CLOSE, *call, EOS])


@pytest.mark.parametrize("thought", ["T1", "T2"])
def test_t1_t2_require_the_48_and_the_header(thought: str) -> None:
    call = template_call_ids("scroll", {"direction": "up"})
    head = thought_header(thought)
    assert accepts("scroll", thought, [*head, *call[1:], EOS])
    assert not accepts("scroll", thought, [*head[:-1], *call[1:], EOS]), "no 48"
    assert not accepts("scroll", thought, [*head, TC_OPEN, *call[1:], EOS]), "two 48s"
    assert not accepts("scroll", thought, [*call, EOS]), "no header"


@pytest.mark.parametrize("thought", ["T1", "T2"])
@pytest.mark.parametrize(
    "between", [" ", "\n", "Sure.", "Sure, scrolling now. ", "<cc>", "\n\n"], ids=repr
)
def test_nothing_between_the_header_and_48(thought: str, between: str) -> None:
    """Criterion 4: no whitespace and no spoken text between 101 and 48."""
    head = thought_header(thought)
    assert head[-2:] == [CLOSE, TC_OPEN]
    call = template_call_ids("scroll", {"direction": "up"})[1:]
    assert accepts("scroll", thought, [*head, *call, EOS])
    bad = [*head[:-1], *nat(between), TC_OPEN, *call, EOS]
    assert not accepts("scroll", thought, bad)


@pytest.mark.parametrize("marker", [TC_OPEN, TC_CLOSE])
def test_t1_thought_excludes_48_and_49(marker: int) -> None:
    """QN5: the native T1 free region is the JSON one minus {48, 49}."""
    call = template_call_ids("scroll", {"direction": "up"})
    good = [OPEN, THOUGHT, NL, *nat("let me"), *nat(" think"), CLOSE, *call, EOS]
    bad = [OPEN, THOUGHT, NL, *nat("let me"), marker, *nat(" think"), CLOSE, *call, EOS]
    assert accepts("scroll", "T1", good)
    assert not accepts("scroll", "T1", bad)
    pick = [*nat("call:scroll{"), EOS]
    assert accepts("name", "T1", [*good[: good.index(CLOSE) + 1], TC_OPEN, *pick])
    assert not accepts("name", "T1", [*bad[: bad.index(CLOSE) + 1], TC_OPEN, *pick])


@pytest.mark.parametrize(
    "inside",
    [
        [DECL_OPEN],
        [DECL_CLOSE],
        [DELIM],
        [TOOL_RESPONSE_CLOSE],
        [THINK],
        "<cc>",
        "call:click_at{x:1,y:2}",
        "<|tool_call>call:x{}<tool_call|>",
        '<|"|>',
    ],
    ids=["46", "47", "52", "51", "98", "cc", "call-text", "spelled-tool-call", "spelled-delim"],
)
def test_t1_thought_still_admits_everything_else(inside: Any) -> None:
    """Guard width: only 48/49 leave the JSON T1 free set; markers 46/47/51/52/98 and tool-call text
    spelled in ordinary pieces stay admissible inside the thought."""
    if isinstance(inside, str):
        ids = char_pieces(inside) if "<|" in inside or "|>" in inside else nat(inside)
        assert TC_OPEN not in ids and TC_CLOSE not in ids
    else:
        ids = list(inside)
    call = template_call_ids("scroll", {"direction": "up"})
    seqn = [OPEN, THOUGHT, NL, *nat("so"), *ids, *nat(" ok"), CLOSE, *call, EOS]
    assert accepts("scroll", "T1", seqn)


def test_qn5_is_native_only_json_t1_still_admits_48_in_the_thought() -> None:
    """The JSON T1 free set is unchanged (criterion 8): 48 and 49 are still thought text there."""
    a = toolcall().compile_toolcall(toolset(), scope="scroll", thought="T1", call_format="json")
    body = json.dumps({"name": "scroll", "arguments": {"direction": "up"}})
    for marker in (TC_OPEN, TC_CLOSE):
        ids = [OPEN, THOUGHT, NL, *nat("a"), marker, *nat("b"), CLOSE, *nat(body), EOS]
        assert accepts_with(a, ids), marker


def test_t1_accepts_a_template_thought_with_a_blank_line() -> None:
    turn = nat(tpl.model_turn("google_search", {"query": "weather"}, REAL_THOUGHT))
    assert accepts("google_search", "T1", turn)


# --------------------------------------------------------------------------- scope="name"

NOT_V1 = (
    "navigate",
    "click",
    "google_searc",
    "press_keys",
    "Click_at",
    "scroll_up",
    "name",
    "union",
)


def name_seq(thought: str, name: str, *, stop: int = EOS) -> list[int]:
    return [*thought_header(thought), *nat(f"call:{name}{{"), stop]


@pytest.mark.parametrize("thought", THOUGHTS)
def test_name_scope_accepts_exactly_the_v1_prefixes(thought: str) -> None:
    for tool in TOOLS:
        assert accepts("name", thought, name_seq(thought, tool)), tool
        assert accepts(
            "name", thought, [*thought_header(thought), *char_pieces(f"call:{tool}{{"), EOS]
        ), tool
    for other in NOT_V1:
        assert not accepts("name", thought, name_seq(thought, other)), other


@pytest.mark.parametrize("thought", THOUGHTS)
def test_name_scope_stops_right_after_the_brace(thought: str) -> None:
    head = thought_header(thought)
    assert not accepts("name", thought, [*head, *nat("call:scroll"), EOS]), "no brace"
    assert not accepts("name", thought, [*head, *nat("call:scroll{direction"), EOS]), (
        "past the brace"
    )
    assert not accepts("name", thought, [*head, *nat("call:scroll{}"), EOS]), "a closed call"
    assert not accepts("name", thought, [*head, *nat('{"name": "scroll"'), EOS]), "JSON form"
    assert not accepts("name", thought, [*head, *nat("call:scroll {"), EOS]), "whitespace"


def test_name_scope_variant_headers() -> None:
    pre = nat("call:scroll{")
    assert not accepts("name", "T4", [TC_OPEN, *pre, EOS]), "T4: no 48"
    assert not accepts("name", "T4", [*EMPTY_HEADER, TC_OPEN, *pre, EOS]), "T4: no header"
    assert not accepts("name", "T2", [*EMPTY_HEADER, *pre, EOS]), "T2: 48 required"
    assert not accepts("name", "T2", [OPEN, THOUGHT, NL, *nat("hm"), CLOSE, TC_OPEN, *pre, EOS])
    assert not accepts("name", "T1", [OPEN, THOUGHT, NL, *nat("hm"), CLOSE, *pre, EOS])


# =========================================================================== random walks


class Sampler:
    """Random accepted paths of one automaton (distances to acceptance computed once)."""

    def __init__(self, a: Any) -> None:
        self.a = a
        self.w = walker(a)
        rev: dict[int, list[int]] = {}
        for src, dst, _ in self.w.edges:
            rev.setdefault(dst, []).append(src)
        self.dist = {f: 0 for f in self.w.finals}
        dq = deque(self.w.finals)
        while dq:
            s = dq.popleft()
            for p in rev.get(s, ()):
                if p not in self.dist:
                    self.dist[p] = self.dist[s] + 1
                    dq.append(p)
        self.positive = {
            c: tuple(sorted(self.w.stored[c])) for c in range(self.w.n_classes) if not self.w.neg[c]
        }

    def token(self, cls: int, rng: random.Random) -> int:
        if not self.w.neg[cls]:
            members = self.positive[cls]
            assert members, f"empty class {cls}"
            return rng.choice(members)
        while True:
            t = rng.randrange(VOCAB_SIZE)
            if self.w.member(cls, t):
                return t

    def path(self, rng: random.Random, *, soft_max: int = 40) -> list[int]:
        """Start to acceptance (ends with the stop that enters it); uniform over live edges, and after
        ``soft_max`` tokens only over edges that approach acceptance."""
        assert len(self.w.start) == 1, "a DFA has one start state"
        (state,) = tuple(self.w.start)
        ids: list[int] = []
        while state not in self.w.finals:
            live = [(d, c) for d, c in self.w.out.get(state, ()) if d in self.dist]
            if len(ids) >= soft_max:
                live = [(d, c) for d, c in live if self.dist[d] < self.dist[state]]
            assert live, f"dead end at state {state} after {ids}"
            dst, cls = rng.choice(live)
            ids.append(self.token(cls, rng))
            state = dst
        return ids


def _typed(schema: Mapping[str, Any], raw: Any) -> Any:
    t = schema.get("type")
    if t == "integer":
        assert isinstance(raw, str) and re.fullmatch(r"0|[1-9][0-9]*", raw), raw
        v = int(raw)
        assert schema["minimum"] <= v <= schema["maximum"], v
        return v
    if t == "boolean":
        assert raw in ("true", "false"), raw
        return raw == "true"
    assert isinstance(raw, str), raw
    assert Q not in raw and "<|" not in raw and "|>" not in raw, raw
    if "enum" in schema:
        assert raw in schema["enum"], raw
    if schema.get("minLength", 0) >= 1:
        assert raw, "empty string under minLength 1"
    return raw


@pytest.mark.parametrize("name", TOOLS)
def test_random_walks_parse_with_the_vllm_parser(name: str) -> None:
    """Language-side oracle (n = 500 walks per tool over native T4, seeds 84000+i): every accepted path,
    decoded, is ``call:NAME{ARGS}<tool_call|>`` whose ARGS the vendored vLLM parser reads as exactly the
    tool's properties in dictsort order, strings free of the delimiter, bare values range-valid integer
    or boolean literals -- and the parsed call re-renders through the template to the same text."""
    a = native(name, "T4")
    sampler = Sampler(a)
    props = shipped_params(name)["properties"]
    n_strings = sum(1 for s in props.values() if s.get("type") == "string")
    head, tail = f"call:{name}{{", "}" + END
    for i in range(500):
        rng = random.Random(84_000 + i)
        ids = sampler.path(rng)
        assert ids[-1] in STOPS, ids
        assert accepts_with(a, ids)
        text = decode(ids[:-1])
        assert text.startswith(head) and text.endswith(tail), (i, text)
        assert text.count(Q) == 2 * n_strings, (i, text)
        parsed = vparse._parse_gemma4_args(text[len(head) : -len(tail)])
        assert list(parsed) == dictsort(list(props)), (i, text, parsed)
        typed = {k: _typed(props[k], v) for k, v in parsed.items()}
        assert tpl.call_text(name, typed) == "<|tool_call>" + text, (i, text)


@pytest.mark.parametrize("thought", THOUGHTS)
def test_lock_step_the_picker_prefix_is_a_prefix_of_every_tool_language(thought: str) -> None:
    """Criterion 4 (n = 200 picker walks per thought, seeds 84500+i): every accepted path of the name
    picker, minus its stop, keeps the named tool's automaton (same thought) alive token for token, and
    completes there with the rest of a valid template call."""
    sampler = Sampler(native("name", thought))
    seen = set()
    for i in range(200):
        rng = random.Random(84_500 + i)
        ids = sampler.path(rng)
        assert ids[-1] in STOPS
        prefix = ids[:-1]
        last48 = len(prefix) - 1 - prefix[::-1].index(TC_OPEN) if TC_OPEN in prefix else -1
        text = decode(prefix[last48 + 1 :])
        m = re.fullmatch(r"call:([a-z_]+)\{", text)
        assert m and m.group(1) in TOOLS, (i, text)
        tool = m.group(1)
        seen.add(tool)
        tw = walker(native(tool, thought))
        for k in range(1, len(prefix) + 1):
            assert tw.run(prefix[:k]), (i, tool, k, prefix)
        full = tpl.call_text(tool, valid_args(tool))
        rest = full[len(f"<|tool_call>call:{tool}{{") :]
        assert accepts(tool, thought, [*prefix, *nat(rest), EOS]), (i, tool)
    assert seen == set(TOOLS), seen


# =========================================================================== criterion 2: schema shapes


def _decl(
    name: str, props: Mapping[str, Any], required: Sequence[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"probe {name}",
        "parameters": {
            "type": "object",
            "properties": dict(props),
            "required": list(props) if required is None else list(required),
            "additionalProperties": False,
        },
    }


PROBES = [
    _decl(
        "probe_ints",
        {
            "a": {"type": "integer"},
            "b": {"type": "integer", "minimum": 0, "maximum": 9},
            "c": {"type": "integer", "minimum": 0, "maximum": 99},
        },
    ),
    _decl(
        "probe_strings",
        {
            "s": {"type": "string"},
            "e": {"type": "string", "enum": ["x y", "a,b:c", "<b>"]},
        },
    ),
    _decl(
        "probe_case",
        {
            "Beta": {"type": "boolean"},
            "alpha": {"type": "integer", "minimum": 0, "maximum": 9},
        },
    ),
]


@pytest.fixture(scope="module")
def probes(tmp_path_factory: pytest.TempPathFactory) -> Any:
    path = tmp_path_factory.mktemp("t0084a") / "probes.json"
    path.write_text(json.dumps(PROBES))
    return need("local_astra.toolcall.tools").load(path)


def probe_accepts(ts: Any, name: str, after48: str) -> bool:
    a = toolcall().compile_toolcall(ts, scope=name, thought="T4", call_format="native")
    return accepts_with(a, [*nat(after48), EOS])


def _tpl_after48(name: str, args: Mapping[str, Any]) -> str:
    return tpl.call_text(name, args)[len("<|tool_call>") :]


@pytest.mark.parametrize("a", [0, 7, -1, -42, 123456789012, -987654321])
def test_unbounded_integer_is_any_canonical_integer(probes: Any, a: int) -> None:
    assert probe_accepts(
        probes, "probe_ints", _tpl_after48("probe_ints", {"a": a, "b": 3, "c": 42})
    )


@pytest.mark.parametrize("raw", ["007", "--1", "+1", "1.5", "-", "", "1e3", "-01", "00"])
def test_unbounded_integer_rejects_non_canonical_spellings(probes: Any, raw: str) -> None:
    assert probe_accepts(probes, "probe_ints", "call:probe_ints{a:1,b:3,c:42}" + END)
    assert not probe_accepts(probes, "probe_ints", f"call:probe_ints{{a:{raw},b:3,c:42}}" + END)


def test_bounded_integers_follow_their_k(probes: Any) -> None:
    """[0, 9] is ``0|[1-9]`` and [0, 99] is ``0|[1-9][0-9]?``: both directions at each edge."""
    for b in range(10):
        assert probe_accepts(
            probes, "probe_ints", _tpl_after48("probe_ints", {"a": 0, "b": b, "c": 0})
        )
    for c in (0, 9, 10, 99):
        assert probe_accepts(
            probes, "probe_ints", _tpl_after48("probe_ints", {"a": 0, "b": 0, "c": c})
        )
    for b in ("10", "-1", "01"):
        assert not probe_accepts(probes, "probe_ints", f"call:probe_ints{{a:0,b:{b},c:0}}" + END), b
    for c in ("100", "00", "09", "-1"):
        assert not probe_accepts(probes, "probe_ints", f"call:probe_ints{{a:0,b:0,c:{c}}}" + END), c


def test_string_without_min_length_admits_the_empty_body(probes: Any) -> None:
    assert probe_accepts(
        probes, "probe_strings", _tpl_after48("probe_strings", {"s": "", "e": "x y"})
    )


def test_string_enum_members_are_literals_between_delimiters(probes: Any) -> None:
    for member in ("x y", "a,b:c", "<b>"):
        assert probe_accepts(
            probes, "probe_strings", _tpl_after48("probe_strings", {"s": "q", "e": member})
        ), member
    for bad in ("x", "x  y", "a,b", "<b", "X Y"):
        assert not probe_accepts(
            probes, "probe_strings", f"call:probe_strings{{e:{Q}{bad}{Q},s:{Q}q{Q}}}" + END
        ), bad


def test_an_enum_with_the_empty_string_admits_its_rendering(tmp_path: Path) -> None:
    """Review F3: ``""`` as an enum member renders as two adjacent delimiters and is accepted; the
    other member still is, and a non-member is not."""
    path = tmp_path / "empty_enum.json"
    path.write_text(
        json.dumps([_decl("probe_empty_enum", {"e": {"type": "string", "enum": ["", "on"]}})])
    )
    ts = need("local_astra.toolcall.tools").load(path)
    empty = _tpl_after48("probe_empty_enum", {"e": ""})
    assert empty == f"call:probe_empty_enum{{e:{Q}{Q}}}" + END, empty
    assert probe_accepts(ts, "probe_empty_enum", empty)
    assert probe_accepts(ts, "probe_empty_enum", _tpl_after48("probe_empty_enum", {"e": "on"}))
    assert not probe_accepts(ts, "probe_empty_enum", f"call:probe_empty_enum{{e:{Q}o{Q}}}" + END)


def test_key_order_is_case_insensitive_dictsort(probes: Any) -> None:
    """The template's order for keys ``Beta``/``alpha`` is ``alpha, Beta`` (jinja dictsort ignores case);
    Python's ``sorted`` would give ``Beta, alpha``."""
    text = _tpl_after48("probe_case", {"Beta": True, "alpha": 3})
    assert text.index("alpha:") < text.index("Beta:"), text
    assert probe_accepts(probes, "probe_case", text)
    assert not probe_accepts(probes, "probe_case", "call:probe_case{Beta:true,alpha:3}" + END)


UNSUPPORTED = {
    "optional": _decl("probe_opt", {"p": {"type": "string"}, "q": {"type": "string"}}, ["p"]),
    "number": _decl("probe_num", {"n": {"type": "number"}}),
    "array": _decl("probe_arr", {"l": {"type": "array", "items": {"type": "string"}}}),
    "object": _decl(
        "probe_obj",
        {"o": {"type": "object", "properties": {"k": {"type": "string"}}, "required": ["k"]}},
    ),
    "null": _decl("probe_null", {"z": {"type": "null"}}),
    # Review F1/F2: a keyword the native grammar cannot enforce, and minLength beyond 1, fail loud.
    "max_length": _decl("probe_maxlen", {"m": {"type": "string", "maxLength": 5}}),
    "min_length_2": _decl("probe_minlen2", {"m": {"type": "string", "minLength": 2}}),
}


@pytest.mark.parametrize("kind", sorted(UNSUPPORTED))
def test_unsupported_shapes_raise_under_native(kind: str, tmp_path: Path) -> None:
    decl = UNSUPPORTED[kind]
    path = tmp_path / f"{kind}.json"
    path.write_text(json.dumps([decl]))
    ts = need("local_astra.toolcall.tools").load(path)
    with pytest.raises(ValueError):
        toolcall().compile_toolcall(ts, scope=decl["name"], thought="T4", call_format="native")


@pytest.mark.parametrize("kind", ["optional", "number"])
def test_the_refusal_is_native_only(kind: str, tmp_path: Path) -> None:
    """Companion: the same declarations still compile under the JSON format (unchanged behaviour)."""
    decl = UNSUPPORTED[kind]
    path = tmp_path / f"{kind}.json"
    path.write_text(json.dumps([decl]))
    ts = need("local_astra.toolcall.tools").load(path)
    toolcall().compile_toolcall(ts, scope=decl["name"], thought="T4", call_format="json")


# =========================================================================== criterion 8: scope

#: Paths T-0084a (and the T-0084 umbrella, criterion 2) must not change over its closed range.
UNCHANGED = (
    "local_astra/toolcall",
    "local_astra/eval",
    "local_astra/conductor",
    "local_astra/contracts",
    "local_astra/bench/thresholds.py",
    "plugin",
)


#: T-0084a's merge commit edc3ff0; BASE = ``git merge-base edc3ff0^1 edc3ff0^2``, TIP = ``edc3ff0^2`` (T-0094
#: criterion 2). Over this closed range no sibling-task exemption is needed: T-0084b/c, T-0085 and T-0086 came later.
T0084A_MERGE = "edc3ff0"
T0084A_BASE = "1e06a24"
T0084A_TIP = "f0a0561"
#: the paths criterion 8 diffs
SCOPE_PATHS = ("local_astra", "plugin")
ALLOWED_CHANGES = frozenset({"local_astra/decoding/toolcall.py"})


def _scope_violations(
    changed: set[str] | list[str], allowed: frozenset[str] = ALLOWED_CHANGES
) -> list[str]:
    """Changed paths outside what T-0084a may touch."""
    return sorted(p for p in changed if p.startswith(UNCHANGED) or p not in allowed)


def test_the_scope_predicate_allows_only_decoding_toolcall() -> None:
    """Only ``decoding/toolcall.py`` is allowed: the files later tasks own (T-0084b's parser, T-0084c's driver, the
    router, the scorer), a prefix sibling, a subdirectory of that name, backup copies and other modules are all
    reported (T-0094: the old exemption list is retired with the closed range)."""
    others = {
        "local_astra/toolcall/native.py",
        "local_astra/eval/cd/twopass.py",
        "local_astra/toolcall/router.py",
        "local_astra/toolcall/score.py",
        "local_astra/toolcall/native_extra.py",
        "local_astra/toolcall/native/x.py",
        "local_astra/toolcall/native.py.bak",
        "local_astra/toolcall/router.py.bak",
        "local_astra/toolcall/score_extra.py",
        "local_astra/toolcall/score.py.bak",
        "local_astra/toolcall/tools.py",
        "local_astra/eval/cd/runner.py",
        "local_astra/eval/cd/twopass_native.py",
        "local_astra/eval/cd/twopass.py.bak",
        "local_astra/eval/cd/replay.py",
        "local_astra/decoding/toolcall.py.bak",
        "plugin/astra_vllm_plugin/astra_vllm_plugin/cd/automaton.py",
    }
    assert _scope_violations(set(ALLOWED_CHANGES)) == []
    assert _scope_violations({*ALLOWED_CHANGES, *others}) == sorted(others)


def test_t0084a_changes_nothing_outside_decoding_toolcall() -> None:
    """Criterion 8 / umbrella criterion 2 over T-0084a's closed range (T-0094): no file under ``UNCHANGED`` differs
    between BASE and TIP and the only changed file under ``local_astra/`` or ``plugin/`` is ``decoding/toolcall.py``.
    The untracked-file half of the old guard is retired: it only meant something while the branch was open."""
    gr.prove_range(T0084A_BASE, T0084A_TIP, T0084A_MERGE)
    changed = gr.changed(T0084A_BASE, T0084A_TIP, *SCOPE_PATHS)
    assert _scope_violations(changed) == [], changed


def test_t0084a_s_range_changes_exactly_its_owned_file() -> None:
    """T-0094 criterion 5: the range reports ``decoding/toolcall.py`` as the only change and it is allowed; with
    ``ALLOWED_CHANGES`` emptied the predicate reports it (the guard is not merely too strict to notice)."""
    gr.require_commits(T0084A_BASE, T0084A_TIP)
    changed = gr.changed(T0084A_BASE, T0084A_TIP, *SCOPE_PATHS)
    assert changed == ["local_astra/decoding/toolcall.py"], changed
    assert _scope_violations(changed) == []
    assert _scope_violations(changed, allowed=frozenset()) == ["local_astra/decoding/toolcall.py"]


def test_the_t0084a_range_guard_has_power() -> None:
    """T-0094 criterion 4 (adversarial): over ``BASE..6a8600b`` the predicate reports the router (T-0082a, T-0085)."""
    gr.require_commits(T0084A_BASE, gr.LATER)
    got = _scope_violations(gr.changed(T0084A_BASE, gr.LATER, *SCOPE_PATHS))
    assert "local_astra/toolcall/router.py" in got, got


# =========================================================================== the test-side compile cache


def test_the_twopass_compile_cache_separates_the_formats() -> None:
    """Verification finding (2026-10-10): ``tests/cd/_twopass_support._install_toolcall_cache`` replaces
    ``compile_toolcall`` for the whole process. Once installed, ``call_format`` must reach the real function and
    key the cache: json and native differ, and each equals an uncached compile, on the first call and on a hit."""
    from local_astra.decoding import artifact
    from tests.cd import _twopass_support as tps

    tc = tps._install_toolcall_cache()
    wrapper, orig = tc.compile_toolcall, tc.compile_toolcall.__wrapped__
    assert getattr(wrapper, "_t0080b_cache", False) and orig is not wrapper
    payload: dict[str, str] = {}
    for fmt in ("json", "native", "json", "native"):  # second pass: memo hits
        got = artifact.encode_automaton(
            wrapper(toolset(), scope="click_at", thought="T2", call_format=fmt)
        )
        want = artifact.encode_automaton(
            orig(toolset(), scope="click_at", thought="T2", call_format=fmt)
        )
        assert got == want, fmt
        payload.setdefault(fmt, got)
        assert payload[fmt] == got, fmt
    assert payload["json"] != payload["native"]
    default = artifact.encode_automaton(wrapper(toolset(), scope="click_at", thought="T2"))
    assert default == payload["json"], "the default call_format is json through the cache too"
    with pytest.raises(ValueError):
        wrapper(toolset(), scope="click_at", thought="T2", call_format="gemma4")
