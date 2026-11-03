"""T-0172 A2: ``announced_call.reask_prompt(sent, said)`` and its prefix property (docs/tasks/T-0172.md).

**Shape** (A2, pure, runs in the fast suite): the re-ask prompt is ``sent``'s messages, then ``Message(assistant,
(TextPart(said),))``, then ``Message(user, (TextPart(REASK_NUDGE),))``, with ``continue_final_message=False`` and
every other ``Prompt`` field equal to ``sent``'s. For a step >= 2 ``sent`` (a continuation, ``loop.step_prompt``) the
final assistant message is replaced by one whose text is the continuation's text followed by ``said``.

**Prefix property** (A2, oracle): ``sent`` and the re-ask prompt are turned into chat-completions bodies by the real
``VllmClient.request_body`` (what goes on the wire, not the code under test) and rendered by the pinned Gemma 4
template through ``tests/cd/oracle/gemma4_template.py`` the way vLLM renders them: image parts become the template's
``{"type": "image"}`` placeholder, a non-continuation gets ``add_generation_prompt=True``, and a continuation is
rendered without it and cut right after its final message's stripped content (``transformers``'
``continue_final_message``). The rendered re-ask must start with the rendered ``sent`` byte for byte, at step 1 (a
system turn, an image user turn, tools) and at step 3 (``step_prompt`` over two executed steps), and the system turn
must be byte-identical with ``<|think|>`` first. The oracle needs jinja2 (the ``cd`` dependency group) and the pinned
template in the local HF cache; without them those rows skip, naming what is missing, after the module check (so at
BASE they still fail on ``NotImplementedError``). Run them with::

    uv run --group cd pytest tests/contract/test_t0172_reask_prefix.py -q -rxX

No ``backend`` parametrisation: nothing here talks to a service; the "real" side is the pinned template.

Adversarial: ``test_nudge_never_in_system`` (the nudge or ``said`` leaking into the system message, the cached
prefix, or appearing twice) and ``test_sent_is_not_mutated``.

Regime: CPU, pure; the template is ``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` at rev 2ea8372 (sha 36e3a42e...).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import ModuleType
from typing import Any

import pytest

from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart
from local_astra.contracts.perception import AssistFlags

MODULE = "local_astra.toolcall.announced_call"

SAID = "I will search for the current weather in Paris again."
SYSTEM = "You are Gearvis, a helpful assistant. Use the tools when an action is needed."
#: a 1x1 JPEG as a data URL (the client refuses non-data URLs)
JPEG = (
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP////////////////////////////////////////////////"
    "//////////////////////////////////////////wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAAA//EABQQAQAAAAAAAAAAAAAAAA"
    "AAAAD/2gAIAQEAAD8AN//Z"
)
TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Search the web.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The query."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_at",
            "description": "Click at a point on the screen.",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
            },
        },
    },
)


def _mod() -> ModuleType:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == MODULE:
            raise NotImplementedError(f"T-0172 A2: {MODULE} does not exist yet") from exc
        raise


def step1(*, thinking: bool = True, tools: tuple[dict[str, Any], ...] | None = TOOLS) -> Prompt:
    """A live step-1 prompt: system turn, then a user turn with the frame and the request."""
    return Prompt(
        messages=(
            Message(role="system", content=(TextPart(text=SYSTEM),)),
            Message(
                role="user",
                content=(
                    ImagePart(url=JPEG, phash="ffff0000ffff0000"),
                    TextPart(text="Screen now. What is the weather in Paris today?"),
                ),
            ),
        ),
        prefix_len=2,
        thinking=thinking,
        assists=AssistFlags(),
        tools=tools,
    )


def step3(base: Prompt | None = None) -> Prompt:
    """The step-3 request: ``loop.step_prompt`` over two executed steps (a search, then a click)."""
    from local_astra.toolcall import loop

    steps = [
        loop.LoopStep(
            text="I'll look that up. ",
            name="google_search",
            arguments={"query": "weather in Paris"},
            response={"query": "weather in Paris", "results": []},
        ),
        loop.LoopStep(
            text="",
            name="click_at",
            arguments={"x": 412, "y": 300},
            response=loop.action_payload("would click at (412, 300)"),
        ),
    ]
    return loop.step_prompt(base or step1(), steps)


def nudge_message(m: ModuleType) -> Message:
    return Message(role="user", content=(TextPart(text=m.REASK_NUDGE),))


def other_fields(p: Prompt) -> dict[str, Any]:
    return {
        f: getattr(p, f)
        for f in Prompt.model_fields
        if f not in ("messages", "continue_final_message")
    }


# --------------------------------------------------------------------------------------------- shape


def test_shape_step1() -> None:
    m = _mod()
    sent = step1()
    got = m.reask_prompt(sent, SAID)
    assert isinstance(got, Prompt)
    assert got.messages == (
        *sent.messages,
        Message(role="assistant", content=(TextPart(text=SAID),)),
        nudge_message(m),
    ), got.messages
    assert got.continue_final_message is False
    assert other_fields(got) == other_fields(sent)


def test_shape_step_k() -> None:
    m = _mod()
    sent = step3()
    assert sent.continue_final_message is True, "premise: a step >= 2 request is a continuation"
    continuation = sent.messages[-1].text()
    got = m.reask_prompt(sent, SAID)
    assert got.messages == (
        *sent.messages[:-1],
        Message(role="assistant", content=(TextPart(text=continuation + SAID),)),
        nudge_message(m),
    ), got.messages[-2:]
    assert got.continue_final_message is False
    assert other_fields(got) == other_fields(sent)


def test_nudge_constant() -> None:
    """A2: ``REASK_NUDGE`` is one non-empty constant (its wording is the implementer's)."""
    m = _mod()
    assert isinstance(m.REASK_NUDGE, str) and m.REASK_NUDGE.strip(), m.REASK_NUDGE
    a = m.reask_prompt(step1(), SAID).messages[-1].text()
    b = m.reask_prompt(step3(), "Let me check the weather.").messages[-1].text()
    assert a == b == m.REASK_NUDGE
    assert SAID not in m.REASK_NUDGE


@pytest.mark.parametrize(
    "sent",
    [
        step1(thinking=True),
        step1(thinking=False),
        step1(tools=None),
        Prompt(
            messages=(Message(role="user", content=(TextPart(text="Find the form."),)),),
            prefix_len=0,
            thinking=True,
            assists=AssistFlags(page_text=True, pdf_text=True),
            tools=TOOLS[:1],
        ),
    ],
    ids=["thinking", "no-thinking", "no-tools", "assists-no-system"],
)
@pytest.mark.parametrize("continued", [False, True], ids=["step1", "step3"])
def test_fields_unchanged(sent: Prompt, continued: bool) -> None:
    """A2: every field but ``messages`` and ``continue_final_message`` equals ``sent``'s, the thinking flag included;
    the first ``len(sent.messages) - 1`` messages are ``sent``'s, untouched."""
    m = _mod()
    if continued:
        sent = step3(sent)
    got = m.reask_prompt(sent, SAID)
    assert other_fields(got) == other_fields(sent), (other_fields(got), other_fields(sent))
    assert got.thinking is sent.thinking
    assert got.continue_final_message is False
    assert got.messages[: len(sent.messages) - 1] == sent.messages[:-1]
    assert len(got.messages) == len(sent.messages) + (1 if continued else 2)


def test_sent_is_not_mutated() -> None:
    m = _mod()
    for sent in (step1(), step3()):
        before = sent.model_dump()
        m.reask_prompt(sent, SAID)
        assert sent.model_dump() == before


@pytest.mark.parametrize("continued", [False, True], ids=["step1", "step3"])
def test_nudge_never_in_system(continued: bool) -> None:
    """Adversarial: the system message (the start of the cached prefix) is ``sent``'s, unchanged; the nudge is in
    exactly one message, the last, a user turn; ``said`` appears only in the final assistant message."""
    m = _mod()
    sent = step3() if continued else step1()
    got = m.reask_prompt(sent, SAID)
    assert got.messages[0] == sent.messages[0] and got.messages[0].role == "system"
    holders = [i for i, msg in enumerate(got.messages) if m.REASK_NUDGE in msg.text()]
    assert holders == [len(got.messages) - 1], holders
    assert got.messages[-1].role == "user"
    said_in = [i for i, msg in enumerate(got.messages) if SAID in msg.text()]
    assert said_in == [len(got.messages) - 2] and got.messages[-2].role == "assistant", said_in
    assert all(msg.role != "system" for msg in got.messages[1:])


# --------------------------------------------------------------------------------------------- oracle


def _oracle() -> Any:
    pytest.importorskip("jinja2", reason="the template oracle needs jinja2: uv run --group cd ...")
    from tests.cd.oracle import gemma4_template as tpl

    if not tpl.template_path().is_file():
        pytest.skip(f"the pinned chat template is not in the local HF cache: {tpl.template_path()}")
    return tpl


def _oracle_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    out = []
    for part in content:
        if part.get("type") == "image_url":
            out.append({"type": "image"})
        else:
            out.append(dict(part))
    return out


def render(tpl: Any, prompt: Prompt) -> str:
    """The text vLLM renders for ``prompt`` (module docstring)."""
    from local_astra.conductor.llm_client import VllmClient

    body: Mapping[str, Any] = VllmClient().request_body(prompt, "t0172-oracle")
    messages = [
        {"role": msg["role"], "content": _oracle_content(msg["content"])}
        for msg in body["messages"]
    ]
    cont = bool(body.get("continue_final_message", False))
    text: str = tpl.render(
        messages,
        tools=body.get("tools"),
        add_generation_prompt=not cont,
        enable_thinking=body["chat_template_kwargs"].get("enable_thinking"),
    )
    if cont:
        final = messages[-1]["content"]
        assert isinstance(final, str) and final.strip() in text
        text = text[: text.rindex(final.strip()) + len(final.strip())]
    return text


def system_turn(text: str) -> str:
    end = text.index("<turn|>")
    return text[: end + len("<turn|>")]


def check_prefix(tpl: Any, sent: Prompt, got: Prompt) -> None:
    a = render(tpl, sent)
    b = render(tpl, got)
    assert b.startswith(a), (
        "A2 prefix property: the re-ask's rendering does not start with the step's rendering "
        f"(first difference at {next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), len(a))}):\n"
        f"sent tail: {a[-200:]!r}\nreask at: {b[len(a) - 200 : len(a) + 100]!r}"
    )
    assert system_turn(a) == system_turn(b)
    assert system_turn(b).startswith("<bos><|turn>system\n<|think|>"), system_turn(b)[:60]
    assert SAID in b[len(a) :], "the announcement is after the prefix"


def test_prefix_step1() -> None:
    m = _mod()
    tpl = _oracle()
    sent = step1()
    got = m.reask_prompt(sent, SAID)
    check_prefix(tpl, sent, got)
    tail = render(tpl, got)[len(render(tpl, sent)) :]
    assert tail.startswith(SAID), (
        f"step 1: the model turn the step opened continues with {SAID!r}: {tail[:120]!r}"
    )


def test_prefix_step_k() -> None:
    m = _mod()
    tpl = _oracle()
    sent = step3()
    got = m.reask_prompt(sent, SAID)
    check_prefix(tpl, sent, got)
    tail = render(tpl, got)[len(render(tpl, sent)) :]
    assert tail.startswith(SAID), (
        f"step 3: the continuation continues with {SAID!r}: {tail[:120]!r}"
    )
    assert m.REASK_NUDGE in tail
