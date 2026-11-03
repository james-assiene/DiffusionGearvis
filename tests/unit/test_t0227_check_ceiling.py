"""T-0227 R-3: the completion check never builds a request over the engine's image limit (docs/tasks/T-0227.md R-3).

The bug (review focus 1 at db948dd2, already on main): ``completion_check.check_prompt`` appends the fresh screen to
``announced_call.reask_prompt(sent, said)``, which keeps every image of the step request, so a step already at 4
images becomes a 5-image check request (HTTP 400 at ``--limit-mm-per-prompt '{"image": 4}'``). It is reachable at the
default cap 3 (3 current screens + 1 history image) and at cap 4 (4 current screens), and at cap 2 with 2 history
images.

The ruling, as these rows read it:

- ``check_prompt(sent, said, frame, *, report, max_images=...)`` (the new keyword; the session passes
  ``self.prompts.budget.max_images``);
- the fresh image is always kept;
- while the total would exceed ``max_images``, the oldest earlier ``ImagePart`` (first in message order) is replaced
  **in place** by ``TextPart("(an earlier screen, not shown)")``, oldest first (reading R3-1: one text part per
  replaced image, at that image's position);
- a check request already within the limit is byte-identical to today's (``reask_prompt`` with the final user message
  replaced by ``(fresh image, instruction)``, the T-0207 A1.2 shape).

Every expected prompt is typed by hand from the step shapes below (phash labels ``h1``, ``h2`` for history images,
``c1``..``c4`` for the turn's own screens), never computed by a second copy of the rule.

Regime: CPU, pure; no measurement.
"""

from __future__ import annotations

import base64
import inspect
from typing import Any

import pytest

from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart
from local_astra.contracts.perception import AssistFlags, Frame, Source
from local_astra.toolcall import announced_call
from local_astra.toolcall import completion_check as cc

NOT_SHOWN = "(an earlier screen, not shown)"
TOOLS: tuple[dict[str, Any], ...] = (
    {"type": "function", "function": {"name": "click_at", "parameters": {"type": "object"}}},
)
SAID = "The settings menu is open now."
MAX_IMAGES = 4
FRESH = Frame(
    source=Source.WINDOW,
    ts_ms=9,
    width=1280,
    height=800,
    jpeg=b"\xff\xd8\xff\xe0t0227-fresh",
    phash="fe" * 8,
)


def img(tag: str) -> ImagePart:
    return ImagePart(
        url="data:image/jpeg;base64," + base64.b64encode(tag.encode()).decode(), phash=tag
    )


def fresh_part() -> ImagePart:
    return ImagePart(
        url="data:image/jpeg;base64," + base64.b64encode(FRESH.jpeg).decode(), phash=FRESH.phash
    )


def step(history: int, current: int) -> Prompt:
    """A step-1 request: system, ``history`` earlier turns each with one image, then the turn's message with
    ``current`` screens and the question."""
    msgs: list[Message] = [Message(role="system", content=(TextPart(text="sys"),))]
    for k in range(1, history + 1):
        msgs.append(
            Message(role="user", content=(img(f"h{k}"), TextPart(text=f"earlier question {k}")))
        )
        msgs.append(Message(role="assistant", content=(TextPart(text=f"earlier answer {k}"),)))
    own = tuple(img(f"c{k}") for k in range(1, current + 1))
    msgs.append(Message(role="user", content=(*own, TextPart(text="open the settings menu"))))
    return Prompt(
        messages=tuple(msgs),
        prefix_len=len(msgs) - 1,
        thinking=True,
        assists=AssistFlags(),
        tools=TOOLS,
    )


def shape(prompt: Prompt) -> list[tuple[str, list[str]]]:
    """Each message as (role, [image phash | "T:" + text]); the fresh image reads "FRESH"."""
    out = []
    for m in prompt.messages:
        parts = []
        for p in m.content:
            if isinstance(p, ImagePart):
                parts.append("FRESH" if p.phash == FRESH.phash else str(p.phash))
            else:
                parts.append("T:" + p.text)
        out.append((m.role, parts))
    return out


def images(prompt: Prompt) -> int:
    return sum(1 for m in prompt.messages for p in m.content if isinstance(p, ImagePart))


def call(sent: Prompt, *, max_images: Any = MAX_IMAGES, report: bool = False) -> Prompt:
    params = inspect.signature(cc.check_prompt).parameters
    assert "max_images" in params and params["max_images"].kind is inspect.Parameter.KEYWORD_ONLY, (
        f"T-0227 R-3: check_prompt(..., *, max_images=...) does not exist yet; it takes {list(params)}"
    )
    return cc.check_prompt(sent, SAID, FRESH, report=report, max_images=max_images)


def todays(sent: Prompt, *, report: bool = False) -> Prompt:
    """Today's check request (T-0207 A1.2): ``reask_prompt(sent, said)`` with the final user message replaced by
    the fresh image and the instruction."""
    reasked = announced_call.reask_prompt(sent, SAID)
    text = cc.CHECK_INSTRUCTION_REPORT if report else cc.CHECK_INSTRUCTION
    last = Message(role="user", content=(fresh_part(), TextPart(text=text)))
    return reasked.model_copy(update={"messages": (*reasked.messages[:-1], last)})


INSTR = "T:" + cc.CHECK_INSTRUCTION


#: cap -> (history images, current screens): the step at the ceiling for each cap (4 images in total)
AT_CEILING = {2: (2, 2), 3: (1, 3), 4: (0, 4)}
#: the hand-typed check request for each: the oldest image becomes the marker, the fresh one is kept
EXPECTED: dict[int, list[tuple[str, list[str]]]] = {
    2: [
        ("system", ["T:sys"]),
        ("user", [f"T:{NOT_SHOWN}", "T:earlier question 1"]),
        ("assistant", ["T:earlier answer 1"]),
        ("user", ["h2", "T:earlier question 2"]),
        ("assistant", ["T:earlier answer 2"]),
        ("user", ["c1", "c2", "T:open the settings menu"]),
        ("assistant", ["T:" + SAID]),
        ("user", ["FRESH", INSTR]),
    ],
    3: [
        ("system", ["T:sys"]),
        ("user", [f"T:{NOT_SHOWN}", "T:earlier question 1"]),
        ("assistant", ["T:earlier answer 1"]),
        ("user", ["c1", "c2", "c3", "T:open the settings menu"]),
        ("assistant", ["T:" + SAID]),
        ("user", ["FRESH", INSTR]),
    ],
    4: [
        ("system", ["T:sys"]),
        ("user", [f"T:{NOT_SHOWN}", "c2", "c3", "c4", "T:open the settings menu"]),
        ("assistant", ["T:" + SAID]),
        ("user", ["FRESH", INSTR]),
    ],
}


def test_the_bug_premise() -> None:
    """Premise (green at BASE, the bug itself): today's construction of a check over a 4-image step has 5 images."""
    for history, current in AT_CEILING.values():
        sent = step(history, current)
        assert images(sent) == MAX_IMAGES
        assert images(todays(sent)) == MAX_IMAGES + 1


@pytest.mark.parametrize("cap", sorted(AT_CEILING))
def test_a_check_over_a_four_image_step_stays_at_the_limit(cap: int) -> None:
    """R-3 at caps 2, 3 and 4: the check request over a step already at 4 images carries 4, keeps the fresh image
    last-but-text, and replaces exactly the oldest earlier image by the marker text."""
    sent = step(*AT_CEILING[cap])
    got = call(sent)
    assert images(got) == MAX_IMAGES, f"R-3: cap {cap} check request carries {images(got)} images"
    assert shape(got) == EXPECTED[cap], shape(got)
    others = {f: getattr(got, f) for f in Prompt.model_fields if f != "messages"}
    assert others == {f: getattr(todays(sent), f) for f in Prompt.model_fields if f != "messages"}


@pytest.mark.parametrize(
    ("history", "current", "limit", "expected_images"),
    [(1, 3, 2, ["c3"]), (2, 2, 1, []), (0, 4, 3, ["c3", "c4"])],
    ids=["limit-2", "limit-1-only-fresh", "limit-3"],
)
def test_oldest_first_and_the_fresh_image_always_kept(
    history, current, limit, expected_images
) -> None:
    """R-3 (adversarial): a lower ``max_images`` replaces the oldest images first, one marker each, until the total is
    within it; the fresh image survives even at ``max_images=1``."""
    got = call(step(history, current), max_images=limit)
    kept = [p.phash for m in got.messages for p in m.content if isinstance(p, ImagePart)]
    assert kept == [*expected_images, FRESH.phash], kept
    markers = sum(
        1
        for m in got.messages
        for p in m.content
        if isinstance(p, TextPart) and p.text == NOT_SHOWN
    )
    assert markers == history + current - len(expected_images), markers


@pytest.mark.parametrize(("history", "current"), [(0, 1), (1, 2), (0, 3), (2, 1)])
@pytest.mark.parametrize("report", [False, True])
def test_a_check_within_the_limit_is_byte_identical_to_todays(history, current, report) -> None:
    """R-3 negative: when step images + the fresh one fit within 4, the check request is today's, byte for byte."""
    sent = step(history, current)
    assert images(sent) + 1 <= MAX_IMAGES, "premise"
    assert (
        call(sent, report=report).model_dump_json() == todays(sent, report=report).model_dump_json()
    )


def test_a_step_two_request_is_capped_too() -> None:
    """R-3 on a step >= 2 shape (``continue_final_message``): the oldest image is replaced and the result never
    continues the final message."""
    base = step(0, 4)
    sent = base.model_copy(
        update={
            "messages": (
                *base.messages,
                Message(role="assistant", content=(TextPart(text="I clicked."),)),
            ),
            "continue_final_message": True,
        }
    )
    got = call(sent)
    assert images(got) == MAX_IMAGES and got.continue_final_message is False
    assert shape(got)[1] == (
        "user",
        [f"T:{NOT_SHOWN}", "c2", "c3", "c4", "T:open the settings menu"],
    )
    assert shape(got)[-2:] == [("assistant", ["T:I clicked." + SAID]), ("user", ["FRESH", INSTR])]
