"""T-0084b criteria 2 and 5 against independent oracles (``make cd-test``).

- **Criterion 2:** for every schema-valid v1 call, ``native.parse_call`` of what the pinned chat template renders
  (``tests/cd/oracle/gemma4_template.py``, sha ``36e3a42e...``) equals the call, type-strict. The same holds for
  general argument trees (nested objects and arrays, floats), which the template renders with the same macro.
- **Criterion 2:** the arguments agree with the vendored vLLM v0.29.0 ``_parse_gemma4_args``
  (``tests/cd/oracle/vllm_gemma4_args.py``) once scalars are rendered to the raw text vLLM keeps.
- **Criterion 5:** the constants are the tokenizer pieces of ids 48, 49 and 52 (pinned ``tokenizer.json``).

The call is cut out of the template's model turn from its first ``<|tool_call>`` to its **last** ``<tool_call|>``
(``rindex``: a string argument may itself contain ``<tool_call|>``; the turn ends with ``<|tool_response>``, so
the last close is the real one). Strings containing ``<|"|>`` are not representable and are filtered out.

Regime: CPU; model rev 2ea8372; template sha 36e3a42e...; tokenizer.json sha cc8d3a0c...; tools_v1 as shipped.
Hypothesis: seeds 84010/84011/84012, no example database, n stated per row.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

from tests.cd._compile_support import encode, hf_tokenizer
from tests.cd.oracle import gemma4_template as tpl
from tests.cd.oracle import vllm_gemma4_args as vparse
from tests.unit._toolcall_support import shipped_params

MODULE = "local_astra.toolcall.native"
V1 = ("google_search", "click_at", "type_at", "scroll", "press_key")
_COMMON = dict(
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


def _native() -> ModuleType:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == MODULE:
            raise AssertionError(f"T-0084b not implemented: `{MODULE}` does not exist") from exc
        raise


def _same(a: Any, b: Any) -> bool:
    """Equality with equal types at every node (``True != 1``, ``1.0 != 1``)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def _draft(name: str, args: dict[str, Any]) -> str:
    """The template's call, minus the leading ``<|tool_call>`` (what the router hands the parser)."""
    turn = tpl.model_turn(name, args)
    start = turn.index(tpl.TOOL_CALL_OPEN) + len(tpl.TOOL_CALL_OPEN)
    end = turn.rindex(tpl.TOOL_CALL_CLOSE) + len(tpl.TOOL_CALL_CLOSE)
    assert turn[end:] == tpl.TOOL_RESPONSE, turn[end:]
    return turn[start:end]


def _args_text(draft: str, name: str) -> str:
    """``ARGS`` of ``call:NAME{ARGS}<tool_call|>`` as the template wrote it (the text vLLM parses)."""
    head = f"call:{name}{{"
    tail = "}" + tpl.TOOL_CALL_CLOSE
    assert draft.startswith(head) and draft.endswith(tail), draft
    return draft[len(head) : -len(tail)]


def _raw(v: Any) -> Any:
    """Scalars as vLLM keeps them: the bare text the template wrote (strings stay strings)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)  # jinja's ``{{ argument }}`` is ``str()``
    if isinstance(v, dict):
        return {k: _raw(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_raw(x) for x in v]
    return v


_STR = st.text(max_size=24).filter(lambda s: tpl.STRING_DELIM not in s)
#: Strings that stress the scanner: structural characters, the special pieces, and a newline.
_HOSTILE = st.lists(
    st.sampled_from(
        [
            "a",
            " ",
            "}",
            "{",
            ",",
            ":",
            "[",
            "]",
            "\n",
            '"',
            "é",
            "<tool_call|>",
            "<|tool_call>",
            "<",
        ]
    ),
    min_size=1,
    max_size=10,
).map("".join)


def _v1_args(name: str) -> st.SearchStrategy[dict[str, Any]]:
    fields: dict[str, st.SearchStrategy[Any]] = {}
    for key, schema in shipped_params(name)["properties"].items():
        if schema.get("type") == "integer":
            fields[key] = st.integers(int(schema["minimum"]), int(schema["maximum"]))
        elif schema.get("type") == "boolean":
            fields[key] = st.booleans()
        elif "enum" in schema:
            fields[key] = st.sampled_from(list(schema["enum"]))
        else:
            assert schema.get("type") == "string" and schema.get("minLength") == 1, (name, key)
            fields[key] = st.one_of(_STR, _HOSTILE).filter(lambda s: len(s) >= 1)
    return st.fixed_dictionaries(fields)


_V1_CALL = st.sampled_from(V1).flatmap(lambda n: st.tuples(st.just(n), _v1_args(n)))

_KEY = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,6}", fullmatch=True)
#: No ``None``: jinja renders it as ``None``, which is not a native literal (the template never emits a null).
_SCALAR = st.one_of(
    st.booleans(),
    st.integers(-(10**9), 10**9),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    _STR,
    _HOSTILE,
)
_TREE = st.recursive(
    _SCALAR,
    lambda inner: st.one_of(st.lists(inner, max_size=3), st.dictionaries(_KEY, inner, max_size=3)),
    max_leaves=8,
)
_GENERAL_CALL = st.tuples(_KEY, st.dictionaries(_KEY, _TREE, max_size=4))


# =========================================================================== criterion 2


@seed(84010)
@settings(max_examples=600, **_COMMON)
@given(_V1_CALL)
def test_template_rendered_v1_calls_parse_back_exactly(call: tuple[str, dict[str, Any]]) -> None:
    """n = 600, seed 84010: every schema-valid v1 call round-trips through the template, types preserved."""
    name, args = call
    draft = _draft(name, args)
    got = _native().parse_call(draft)
    assert _same(got, {"name": name, "arguments": args}), (draft, got)


@seed(84011)
@settings(max_examples=400, **_COMMON)
@given(_GENERAL_CALL)
def test_template_rendered_general_trees_parse_back_exactly(
    call: tuple[str, dict[str, Any]],
) -> None:
    """n = 400, seed 84011: nested objects and arrays, floats, ints and booleans as the template renders them."""
    name, args = call
    draft = _draft(name, args)
    got = _native().parse_call(draft)
    assert _same(got, {"name": name, "arguments": args}), (draft, got)


@seed(84012)
@settings(max_examples=600, **_COMMON)
@given(st.one_of(_V1_CALL, _GENERAL_CALL))
def test_arguments_agree_with_vllm_once_scalars_are_raw_text(
    call: tuple[str, dict[str, Any]],
) -> None:
    """n = 600, seed 84012: ``_parse_gemma4_args`` keeps bare scalars as text; rendering the parser's typed
    values to that text gives vLLM's dict exactly."""
    name, args = call
    draft = _draft(name, args)
    got = _native().parse_call(draft)
    assert isinstance(got, dict), (draft, got)
    want = vparse._parse_gemma4_args(_args_text(draft, name))
    assert _same(_raw(got["arguments"]), want), (draft, got, want)


def test_the_adversarial_string_round_trips_through_the_template() -> None:
    args = {"press_enter": True, "text": "a}<tool_call|>b", "x": 3, "y": 4}
    draft = _draft("type_at", args)
    assert (
        draft == 'call:type_at{press_enter:true,text:<|"|>a}<tool_call|>b<|"|>,x:3,y:4}<tool_call|>'
    )
    assert _same(_native().parse_call(draft), {"name": "type_at", "arguments": args})


# =========================================================================== criterion 5


def test_constants_are_the_tokenizer_pieces_of_ids_48_49_52() -> None:
    nt = _native()
    tok = hf_tokenizer()
    assert tok.id_to_token(48) == nt.TOOL_CALL_OPEN
    assert tok.id_to_token(49) == nt.TOOL_CALL_CLOSE
    assert tok.id_to_token(52) == nt.STRING_DELIM
    assert encode(nt.TOOL_CALL_OPEN) == (48,)
    assert encode(nt.TOOL_CALL_CLOSE) == (49,)
    assert encode(nt.STRING_DELIM) == (52,)
