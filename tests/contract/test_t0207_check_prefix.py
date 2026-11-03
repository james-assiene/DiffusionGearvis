"""T-0207 A1.3: the prefix property of ``completion_check.check_prompt`` through the pinned Gemma 4 template.

``sent`` and the check prompt are turned into chat-completions bodies by the real ``VllmClient.request_body`` and
rendered by ``tests/cd/oracle/gemma4_template.py`` the way vLLM renders them (``render`` and ``system_turn`` are
T-0172's, imported from ``tests/contract/test_t0172_reask_prefix.py``: image parts become ``{"type": "image"}``,
which the template renders as ``<|image|>``; a continuation is rendered without the generation prompt and cut after
its final message's stripped content). For every step >= 2 shape the session can send (step 2, step k, the report
step with ``REPORT_FACT``, a T-0201 reminder step) and for both instructions:

- the rendered ``sent`` is a byte prefix of the rendered check prompt;
- the system turn is byte-identical and starts ``<bos><|turn>system\\n<|think|>``;
- there is exactly one more ``<|image|>`` than in ``sent``, and it lies inside the final user turn (after the
  last ``<|turn>user\\n``), first in it, followed by the instruction;
- the tail after the prefix is ``said``, then the model turn's end, then that user turn, then a fresh model turn.

The oracle needs jinja2 (the ``cd`` group) and the pinned template in the local HF cache; without them those rows
skip, naming what is missing, after the module check (so before the module exists they fail on
``NotImplementedError``). Run with::

    uv run --group cd pytest tests/contract/test_t0207_check_prefix.py -q -rs

No ``backend`` parametrisation: nothing here talks to a service; the "real" side is the pinned template.

Adversarial: ``test_instruction_never_in_system_and_one_image_more_only_in_final_user_turn``. If a prefix row
fails, the task says stop and re-plan (A1.3).

Regime: CPU, pure; template ``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` rev 2ea8372 (sha 36e3a42e...); n = 1.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any

import pytest

from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart
from local_astra.contracts.perception import AssistFlags, Frame, Source
from local_astra.toolcall import loop
from tests.contract.test_t0172_reask_prefix import JPEG, SYSTEM, TOOLS, _oracle, render, system_turn
from tests.unit.test_t0207_check_prompt import H_CHECK, H_CHECK_REPORT, REMINDER, _mod

SAID = "I have purchased The Legend of Zelda: Tears of the Kingdom for you."
#: a different 1x1 JPEG body (any bytes do: the template sees only the placeholder)
FRESH = Frame(
    source=Source.WINDOW,
    ts_ms=9,
    width=1280,
    height=800,
    jpeg=base64.b64decode(JPEG.split(",", 1)[1]),
    phash="fedcba9876543210",
)
IMAGE = "<|image|>"
USER_TURN = "<|turn>user\n"


def step1(*, thinking: bool = True) -> Prompt:
    """A live step-1 prompt: system turn, a user turn with the frame, the screen label and the request."""
    return Prompt(
        messages=(
            Message(role="system", content=(TextPart(text=SYSTEM),)),
            Message(
                role="user",
                content=(
                    ImagePart(url=JPEG, phash="ffff0000ffff0000"),
                    TextPart(
                        text="[This screenshot is the browser screen now, after your last action.]"
                    ),
                    TextPart(text="Buy the synthetic game in the store."),
                ),
            ),
        ),
        prefix_len=2,
        thinking=thinking,
        assists=AssistFlags(),
        tools=TOOLS,
    )


def executed(n: int) -> list[loop.LoopStep]:
    out = [
        loop.LoopStep(
            text="I'll look that up. ",
            name="google_search",
            arguments={"query": "synthetic game store"},
            response={"query": "synthetic game store", "results": []},
        )
    ]
    for i in range(n - 1):
        out.append(
            loop.LoopStep(
                text="I will click the proceed to purchase button." if i % 2 else "",
                name="click_at",
                arguments={"x": 825, "y": 235},
                response=loop.screenshot_payload(
                    loop.action_payload("would click at (825, 235)"), True
                ),
            )
        )
    return out


def step(k: int) -> Prompt:
    return loop.step_prompt(step1(), executed(k - 1))


CASES: dict[str, tuple[Callable[[], Prompt], bool]] = {
    "step2": (lambda: step(2), False),
    "step6": (lambda: step(6), False),
    "report": (lambda: loop.with_report_fact(step(5)), True),
    "reminder": (lambda: loop.with_screen_reminder(step(3), REMINDER), False),
    # the instruction choice is the caller's: both texts keep the prefix at any step
    "step4-report-wording": (lambda: step(4), True),
}


def rendered(tpl: Any, name: str) -> tuple[Prompt, Prompt, str, str, bool]:
    m = _mod()
    make, report = CASES[name]
    sent = make()
    assert sent.continue_final_message is True, "premise: a step >= 2 request is a continuation"
    got = m.check_prompt(sent, SAID, FRESH, report=report)
    return sent, got, render(tpl, sent), render(tpl, got), report


@pytest.mark.parametrize("name", list(CASES))
def test_prefix_holds(name: str) -> None:
    """A1.3: the rendered step is a byte prefix of the rendered check; the system turn is identical, ``<|think|>``
    first; the tail is ``said`` then a closed model turn, a user turn and an open model turn."""
    _mod()
    tpl = _oracle()
    _sent, _got, a, b, report = rendered(tpl, name)
    assert b.startswith(a), (
        "A1.3 prefix property (stop and re-plan): the check's rendering does not start with the step's "
        f"(first difference at {next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), len(a))}):\n"
        f"sent tail: {a[-200:]!r}\ncheck at: {b[len(a) - 200 : len(a) + 100]!r}"
    )
    assert system_turn(a) == system_turn(b)
    assert system_turn(b).startswith("<bos><|turn>system\n<|think|>"), system_turn(b)[:60]
    tail = b[len(a) :]
    assert tail.startswith(SAID), tail[:160]
    instr = H_CHECK_REPORT if report else H_CHECK
    rest = tail[len(SAID) :]
    want_rest = "<turn|>\n" + USER_TURN + IMAGE + instr + "<turn|>\n<|turn>model\n"
    assert rest == want_rest, f"the tail after said: {rest[:400]!r}"


def test_prefix_without_thinking_keeps_system_turn() -> None:
    """A1.2/A1.3: the thinking flag is ``sent``'s, so a no-thinking step renders the same system turn in both."""
    m = _mod()
    tpl = _oracle()
    sent = loop.step_prompt(step1(thinking=False), executed(2))
    got = m.check_prompt(sent, SAID, FRESH, report=False)
    a, b = render(tpl, sent), render(tpl, got)
    assert b.startswith(a)
    assert system_turn(a) == system_turn(b)
    assert "<|think|>" not in system_turn(b)


@pytest.mark.parametrize("name", list(CASES))
def test_instruction_never_in_system_and_one_image_more_only_in_final_user_turn(name: str) -> None:
    """Adversarial: neither instruction is in the system turn or the cached prefix; the chosen one appears exactly
    once, inside the final user turn; the image count grows by exactly one, and the new one is in that turn."""
    _mod()
    tpl = _oracle()
    _sent, _got, a, b, report = rendered(tpl, name)
    instr = H_CHECK_REPORT if report else H_CHECK
    other = H_CHECK if report else H_CHECK_REPORT
    for text in (H_CHECK, H_CHECK_REPORT):
        assert text not in system_turn(b)
        assert text not in a
    assert b.count(instr) == 1 and other not in b
    assert b.count(IMAGE) == a.count(IMAGE) + 1, (a.count(IMAGE), b.count(IMAGE))
    final_user = b[b.rindex(USER_TURN) :]
    assert b.rindex(USER_TURN) >= len(a), "the final user turn is after the prefix"
    assert final_user.count(IMAGE) == 1 and instr in final_user
    assert final_user.index(IMAGE) < final_user.index(instr), (
        "the image comes before the instruction"
    )
    assert b[: b.rindex(USER_TURN)].count(IMAGE) == a.count(IMAGE)
