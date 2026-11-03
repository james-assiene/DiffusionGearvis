"""The pinned Gemma 4 chat template as an oracle for the native tool-call format (T-0084a).

Renders ``chat_template.jinja`` of ``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` at the revision pinned in
``services/versions.lock`` the way ``transformers`` does (``ImmutableSandboxedEnvironment`` with
``trim_blocks=True, lstrip_blocks=True`` and the ``loopcontrols`` extension), reading the file from the
local HF cache only and refusing any other bytes (sha256 ``TEMPLATE_SHA256``).

Nothing here imports ``local_astra``: the native grammar is checked against what the template renders,
never against the implementation's idea of the format.
"""

from __future__ import annotations

import functools
import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MODEL_ID = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
MODEL_REVISION = "2ea837236295d617ac27f8c17d61228081932c40"  # services/versions.lock
#: sha256 of the pinned ``chat_template.jinja`` (docs/tasks/T-0084.md, "template sha 36e3a42e...").
TEMPLATE_SHA256 = "36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0"

TOOL_CALL_OPEN = "<|tool_call>"
TOOL_CALL_CLOSE = "<tool_call|>"
TOOL_RESPONSE = "<|tool_response>"
STRING_DELIM = '<|"|>'
MODEL_TURN = "<|turn>model\n"


def _hf_hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def template_path() -> Path:
    return (
        _hf_hub_cache()
        / f"models--{MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / MODEL_REVISION
        / "chat_template.jinja"
    )


@functools.cache
def template_source() -> str:
    path = template_path()
    if not path.is_file():
        raise AssertionError(f"the pinned chat template is not in the local HF cache: {path}")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != TEMPLATE_SHA256:
        raise AssertionError(f"{path} has sha256 {digest}, not the pinned {TEMPLATE_SHA256}")
    return data.decode("utf-8")


@functools.cache
def _compiled() -> Any:
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
    )

    def raise_exception(message: str) -> None:
        raise AssertionError(f"template raised: {message}")

    env.globals["raise_exception"] = raise_exception
    return env.from_string(template_source())


def render(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool | None = None,
) -> str:
    """``apply_chat_template(..., tokenize=False)`` with ``bos_token="<bos>"``."""
    kwargs: dict[str, Any] = {
        "messages": list(messages),
        "bos_token": "<bos>",
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        kwargs["tools"] = list(tools)
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    out: str = _compiled().render(**kwargs)
    return out


def _messages(name: str, arguments: Mapping[str, Any], reasoning: str | None) -> list[Any]:
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": name, "arguments": dict(arguments)},
            }
        ],
    }
    if reasoning is not None:
        assistant["reasoning"] = reasoning
    return [{"role": "user", "content": "Do it."}, assistant]


def model_turn(name: str, arguments: Mapping[str, Any], reasoning: str | None = None) -> str:
    """The assistant's model turn after ``<|turn>model\\n``, as the template's ``tool_calls`` path renders
    it: ``[<|channel>thought\\n{reasoning}\\n<channel|>]<|tool_call>call:NAME{...}<tool_call|><|tool_response>``
    (no content and no tool message, so the turn ends with ``<|tool_response>``, stop id 50)."""
    text = render(_messages(name, arguments, reasoning))
    cut = text.rindex(MODEL_TURN) + len(MODEL_TURN)
    turn = text[cut:]
    assert turn.endswith(TOOL_RESPONSE), f"the turn does not end with {TOOL_RESPONSE!r}: {turn!r}"
    return turn


def call_text(name: str, arguments: Mapping[str, Any]) -> str:
    """The call substring ``<|tool_call>call:NAME{...}<tool_call|>`` the template renders."""
    turn = model_turn(name, arguments)
    start = turn.index(TOOL_CALL_OPEN)
    end = turn.index(TOOL_CALL_CLOSE, start) + len(TOOL_CALL_CLOSE)
    return turn[start:end]


def empty_thought_header() -> str:
    """The generation prompt's empty thought (``enable_thinking`` off): ``<|channel>thought\\n<channel|>``."""
    text = render([{"role": "user", "content": "Do it."}], add_generation_prompt=True)
    cut = text.rindex(MODEL_TURN) + len(MODEL_TURN)
    return text[cut:]
