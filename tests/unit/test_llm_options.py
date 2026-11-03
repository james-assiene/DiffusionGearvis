"""Sampling whitelist and prompt/prefix contracts (docs/10 §9, docs/01 D9)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_astra.contracts.llm import (
    SAMPLING_WHITELIST,
    VLLM_XARGS_KEY,
    ImagePart,
    LlmOptions,
    Message,
    Prompt,
    TextPart,
)


@pytest.mark.parametrize(
    "forbidden",
    [
        "temperature",
        "seed",
        "min_p",
        "logit_bias",
        "response_format",
        "presence_penalty",
        "n",
        "guided_json",
    ],
)
def test_forbidden_sampling_fields_are_rejected_at_construction(forbidden):
    """These return HTTP 400 from vLLM's diffusion path; the contract refuses them earlier."""
    with pytest.raises(ValidationError):
        LlmOptions(**{forbidden: 1})


def test_whitelist_is_exactly_the_four_vllm_diffusion_fields():
    assert {"top_k", "top_p", "max_tokens", "stop"} == SAMPLING_WHITELIST
    assert set(LlmOptions.model_fields) == SAMPLING_WHITELIST


def test_request_fields_only_include_set_values():
    assert LlmOptions().to_request_fields() == {}
    o = LlmOptions(top_k=40, stop=("</ui>",))
    assert o.to_request_fields() == {"top_k": 40, "stop": ["</ui>"]}


@pytest.mark.parametrize(
    "bad",
    [
        {"top_k": 0},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"max_tokens": 0},
        {"stop": ("",)},
        {"stop": ("a", "b", "c", "d", "e")},
    ],
)
def test_out_of_range_values_are_rejected(bad):
    with pytest.raises(ValidationError):
        LlmOptions(**bad)


def test_extra_body_wraps_plugin_args_under_vllm_xargs():
    assert LlmOptions.extra_body(None) == {}
    assert LlmOptions.extra_body({}) == {}
    body = LlmOptions.extra_body(
        {"astra_tap": True, "astra_request_id": "t7", "constraint": "length_bounded:64"}
    )
    assert set(body) == {VLLM_XARGS_KEY}
    assert body[VLLM_XARGS_KEY]["astra_request_id"] == "t7"


def test_extra_body_rejects_nested_objects():
    """vLLM forwards only simple types through extra_args."""
    with pytest.raises(TypeError):
        LlmOptions.extra_body({"nested": {"a": 1}})


def _msg(role, text):
    return Message(role=role, content=(TextPart(text=text),))


def test_prompt_prefix_is_the_leading_messages_up_to_and_including_the_frame():
    frame = Message(
        role="user", content=(ImagePart(url="data:image/jpeg;base64,AAAA", phash="0" * 16),)
    )
    msgs = (
        _msg("system", "sys"),
        _msg("user", "hi"),
        _msg("assistant", "hello"),
        frame,
        _msg("user", "[tutor state]\nproblem: x"),
        _msg("user", "what is this?"),
    )
    p = Prompt(messages=msgs, prefix_len=4)
    pre = p.prefix()
    assert pre.messages == msgs[:4]
    assert pre.prefix_len == 4 and pre.thinking is True
    assert pre.messages[-1] is frame  # frame is the last cached element (D9)


def test_prompt_prefix_len_cannot_exceed_messages():
    with pytest.raises(ValidationError):
        Prompt(messages=(_msg("system", "s"),), prefix_len=2)


def test_message_text_concatenates_text_parts_only():
    m = Message(
        role="user", content=(TextPart(text="a"), ImagePart(url="data:x"), TextPart(text="b"))
    )
    assert m.text() == "ab"
