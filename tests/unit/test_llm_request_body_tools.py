"""T-0088 criteria 1 and 2: the contracts (``Segment`` "tool", ``Prompt.tools``, ``Prompt.continue_final_message``,
``Commit.stop_id``) and the tools-on request body.

Oracles:
- the arm-(b) field set and values are T-0087's *recorded* request (``tests/fixtures/llm/t0087_streams.jsonl``),
  whose ``tools`` sha equals ``native.tools_field(tools.load())``;
- the tools-off bodies are golden bytes captured from main's ``VllmClient`` with ``local_astra/`` byte-identical to
  the base ``9669258`` (``request_bodies_9669258.json``), before any T-0088 code existed.

The golden rows and the ``LlmOptions`` row are the criterion's *negative* half: green by design before and after.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx
import pytest

from local_astra.conductor.llm_client import VllmClient
from local_astra.contracts.llm import ImagePart, LlmOptions, Message, TextPart
from local_astra.toolcall import native, tools
from tests.fakes import vllm_sse as V
from tests.unit import _llm_id_support as S

ARM_B_FIELDS = {"tools", "tool_choice", "skip_special_tokens", "return_token_ids"}
CONTINUATION_FIELDS = {"continue_final_message", "add_generation_prompt"}


def _body(prompt: Any, options: LlmOptions | None = None) -> dict[str, Any]:
    return VllmClient().request_body(prompt, "r1", options)


def _tools_off(prompt: Any) -> Any:
    return S.make_prompt(messages=prompt.messages, prefix_len=prompt.prefix_len)


# --------------------------------------------------------------------------- criterion 1: contracts


def test_segment_admits_tool_and_the_old_segments_still_construct() -> None:
    c = S.make_commit(segment="tool", text="<|tool_call>call:x{}<tool_call|>")
    assert c.segment == "tool"
    for seg in ("thought", "answer", "ui"):
        assert S.make_commit(segment=seg).segment == seg


@pytest.mark.parametrize("stop", [50, 106, 1, None])
def test_a_tool_commit_carries_a_stop_id_from_stop_ids_or_none(stop: int | None) -> None:
    c = S.make_commit(segment="tool", stop_id=stop)
    assert S.stop_id(c) == stop


def test_stop_id_defaults_to_none_on_every_commit() -> None:
    c = S.make_commit(segment="answer")
    assert S.stop_id(c) is None


@pytest.mark.parametrize("segment", ["answer", "thought", "ui"])
def test_stop_id_is_refused_on_a_non_tool_commit(segment: str) -> None:
    S.make_commit(
        segment="tool", stop_id=50
    )  # the field exists (else this row would pass for the wrong reason)
    with pytest.raises(ValueError):
        S.make_commit(segment=segment, stop_id=50)


@pytest.mark.parametrize("stop", [0, 2, 48, 49, 101, 7])
def test_stop_id_outside_stop_ids_is_refused(stop: int) -> None:
    S.make_commit(segment="tool", stop_id=50)
    with pytest.raises(ValueError):
        S.make_commit(segment="tool", stop_id=stop)


@pytest.mark.parametrize("stop", [True, "50", "1"], ids=["bool-True", "str-50", "str-1"])
def test_stop_id_is_strict_no_bool_and_no_string_e14(stop: Any) -> None:
    """Ruling E14: ``Commit.stop_id`` is ``Field(default=None, strict=True)``. ``True == 1`` and ``"50"`` both name a
    stop id under lax coercion, so only strictness refuses them."""
    assert S.stop_id(S.make_commit(segment="tool", stop_id=50)) == 50
    with pytest.raises(ValueError):
        S.make_commit(segment="tool", stop_id=stop)


def test_prompt_tools_defaults_to_none_and_continue_final_message_to_false() -> None:
    p = S.make_prompt()
    assert S.need(p, "tools", "criterion 1") is None
    assert S.need(p, "continue_final_message", "criterion 1") is False


def test_prefix_carries_the_tools() -> None:
    """q8: the tools must be in the prefill, so the prefill prompt carries them."""
    p = S.tools_prompt(messages=(S.SYSTEM, S.USER), prefix_len=1)
    assert p.prefix().tools == S.TOOLS


def test_shares_prefix_with_is_false_when_tools_differ_and_true_when_equal() -> None:
    with_tools = S.tools_prompt(messages=(S.SYSTEM, S.USER), prefix_len=1)
    other_tools = S.tools_prompt(messages=(S.SYSTEM, S.USER), prefix_len=1, tools=S.TOOLS[:2])
    without = S.make_prompt(messages=(S.SYSTEM, S.USER), prefix_len=1)
    assert with_tools.prefix().shares_prefix_with(with_tools), "same tools: the D9 rule still holds"
    assert not with_tools.prefix().shares_prefix_with(other_tools)
    assert not with_tools.prefix().shares_prefix_with(without)
    assert not without.prefix().shares_prefix_with(with_tools)
    assert without.prefix().shares_prefix_with(without)


def _assistant(*parts: Any) -> Message:
    return Message(role="assistant", content=tuple(parts))


def test_continue_final_message_accepts_one_assistant_text_part() -> None:
    p = S.make_prompt(
        messages=(S.SYSTEM, S.USER, _assistant(TextPart(text="I will.<|tool_call>"))),
        prefix_len=1,
        continue_final_message=True,
    )
    assert p.continue_final_message is True


@pytest.mark.parametrize(
    "last",
    [
        pytest.param(S.USER, id="last-is-user"),
        pytest.param(_assistant(TextPart(text="a"), TextPart(text="b")), id="two-text-parts"),
        pytest.param(_assistant(ImagePart(url="data:image/jpeg;base64,AA==")), id="image-part"),
        pytest.param(_assistant(), id="no-part"),
    ],
)
def test_continue_final_message_refuses_anything_but_one_assistant_text_part(last: Message) -> None:
    S.make_prompt(continue_final_message=False)  # the field exists
    msgs = (S.SYSTEM, S.USER, last) if last is not S.USER else (S.SYSTEM, S.USER)
    with pytest.raises(ValueError):
        S.make_prompt(messages=msgs, prefix_len=1, continue_final_message=True)
    # the same messages without the flag stay valid: the refusal is the flag's, not the messages'
    S.make_prompt(messages=msgs, prefix_len=1)


# --------------------------------------------------------------------------- criterion 2: the body


def _recorded_request() -> dict[str, Any]:
    reqs = [r["request"] for r in S.recorded() if r["arm"] == "b"]
    assert len(reqs) == 12
    return dict(reqs[0])


def test_the_tools_body_is_todays_body_plus_exactly_the_arm_b_fields() -> None:
    p = S.tools_prompt()
    body = _body(p)
    off = _body(_tools_off(p))
    assert set(body) == set(off) | ARM_B_FIELDS, (
        f"exactly arm (b) is added: {sorted(set(body) - set(off))}"
    )
    for key, value in off.items():
        assert body[key] == value, f"{key} differs from the tools-off body"
    assert body["tool_choice"] == "none"
    assert body["skip_special_tokens"] is False
    assert body["return_token_ids"] is True


def test_the_tool_fields_equal_t0087s_recorded_arm_b_request() -> None:
    """The recorded probe request is the oracle: same values, and ``tools`` hashes to the recorded sha."""
    rec = _recorded_request()
    body = _body(S.tools_prompt())
    for key in ("tool_choice", "skip_special_tokens", "return_token_ids", "stream"):
        assert body[key] == rec[key], key
    tools_json = json.dumps(body["tools"], separators=(",", ":"), ensure_ascii=False)
    assert {
        "len": len(body["tools"]),
        "sha256": hashlib.sha256(tools_json.encode()).hexdigest(),
    } == rec["tools"]


def test_tools_go_out_in_native_tools_field_order_and_in_the_prompts_order() -> None:
    assert _body(S.tools_prompt())["tools"] == native.tools_field(tools.load())
    rev = tuple(reversed(S.TOOLS))
    assert _body(S.tools_prompt(tools=rev))["tools"] == list(rev)


def test_options_and_xargs_still_merge_with_tools() -> None:
    body = VllmClient().request_body(
        S.tools_prompt(), "r9", LlmOptions(top_k=40), {"astra_tap": "1"}
    )
    assert body["top_k"] == 40
    assert body["vllm_xargs"] == {"astra_tap": "1", "astra_request_id": "r9"}
    assert set(body) >= ARM_B_FIELDS


def test_neither_output_format_field_enters_llm_options() -> None:
    """Negative (green by design): both follow from ``Prompt.tools``, never from ``LlmOptions``."""
    assert not ({"skip_special_tokens", "return_token_ids"} & set(LlmOptions.model_fields))
    with pytest.raises(ValueError):
        LlmOptions(skip_special_tokens=False)  # type: ignore[call-arg]


def test_the_continuation_body_sends_string_content_and_the_two_flags() -> None:
    p = S.continuation_prompt("I will click it.<|tool_call>")
    body = _body(p)
    assert body["continue_final_message"] is True
    assert body["add_generation_prompt"] is False
    assert body["messages"][-1] == {
        "role": "assistant",
        "content": "I will click it.<|tool_call>",
    }, "q9: the final assistant content is a plain string"
    off = _body(S.make_prompt(messages=p.messages[:-1], prefix_len=1))
    assert body["messages"][:-1] == off["messages"], "earlier messages keep the list-of-parts form"
    assert set(body) >= ARM_B_FIELDS, (
        "a continuation prompt with tools still carries the tool fields"
    )


def test_the_continuation_fields_appear_only_with_the_flag() -> None:
    msgs = (S.SYSTEM, S.USER, _assistant(TextPart(text="Sure.")))
    body = _body(S.tools_prompt(messages=msgs, prefix_len=1))
    assert not (CONTINUATION_FIELDS & set(body))
    assert body["messages"][-1]["content"] == [{"type": "text", "text": "Sure."}]


async def test_prefill_and_generate_carry_the_same_tool_fields_on_the_wire() -> None:
    """B11 / q8: the prefill is ``request_body(prefix, ..., max_tokens=1)`` and carries the tool fields too."""
    p = S.tools_prompt()
    sent: list[dict[str, Any]] = []
    lines = S.stream_lines([[*S.thought("t"), *S.prose("Ok."), S.TURN_CLOSE]])

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read()))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=V.sse_text(lines)
        )

    cl = S.client(httpx.MockTransport(handler))
    await cl.prefill(p.prefix(), "p1")
    async for _ in cl.generate(p, "g1"):
        pass
    await cl.aclose()
    prefill, gen = sent
    for key in ARM_B_FIELDS:
        assert key in prefill and prefill[key] == gen[key], key
    assert prefill == VllmClient().request_body(p.prefix(), "p1", LlmOptions(max_tokens=1))


# --------------------------------------------------------------------------- negative: tools off is main


def _golden() -> list[dict[str, Any]]:
    data = json.loads(S.GOLDEN.read_text())
    assert data["base_sha"] == S.BASE_SHA
    return list(data["rows"])


@pytest.mark.parametrize("name", [n for n, *_ in S.golden_battery()])
def test_tools_off_request_body_is_byte_identical_to_the_base(name: str) -> None:
    row = {r["name"]: r for r in _golden()}[name]
    _, prompt, options, xargs = {n: (n, p, o, x) for n, p, o, x in S.golden_battery()}[name]
    body = VllmClient().request_body(prompt, f"b-{name}", options, xargs)
    assert json.dumps(body) == row["request_body"]


@pytest.mark.parametrize("name", [n for n, *_ in S.golden_battery()])
async def test_tools_off_wire_bytes_of_generate_and_prefill_are_identical_to_the_base(
    name: str,
) -> None:
    row = {r["name"]: r for r in _golden()}[name]
    _, prompt, options, xargs = {n: (n, p, o, x) for n, p, o, x in S.golden_battery()}[name]
    sent: list[bytes] = []
    sse = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.read())
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)

    cl = VllmClient(transport=httpx.MockTransport(handler))
    async for _ in cl.generate(prompt, f"g-{name}", options, xargs=xargs):
        pass
    await cl.prefill(prompt.prefix(), f"p-{name}")
    await cl.aclose()
    assert sent[0].decode() == row["generate_wire"]
    assert sent[1].decode() == row["prefill_wire"]
