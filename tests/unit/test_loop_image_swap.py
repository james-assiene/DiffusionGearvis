"""T-0111g B8: the pure newest-image swap and the ``screenshot`` payload builder in ``toolcall/loop.py``.

Pinned names (``tests/unit/_live_browser_support.py`` docstring): ``loop.swap_newest_image(prompt, frame) -> Prompt``
and ``loop.screenshot_payload(payload, updated) -> dict``. Expected prompts are built by hand from the step-1
prompt (one part replaced), and the replacement part's data URL comes from the real ``PromptBuilder`` (the oracle
B8 names), never from ``loop``.

Readings (reported): "the newest ``ImagePart`` of the prompt" is read as the newest image of the **final user
message** (the current turn), per the task's "replacing the newest image of the user turn in place"; a prompt whose
current turn carries no image (no frame, or a frame riding on the newest history image) comes back equal, so a
history image is never swapped. ``test_no_image_prompt_unchanged_or_refused`` pins "unchanged".

Regime: CPU, pure, hand-built prompts; no measurement.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.prompt_builder import PromptBuilder
from local_astra.contracts.llm import ImagePart, Message, Prompt
from local_astra.contracts.turn import HistoryTurn
from local_astra.toolcall import loop
from tests.unit._live_browser_support import builder_url, data_url, live_frame

REPO = Path(__file__).resolve().parents[2]


def swap() -> Any:
    fn = getattr(loop, "swap_newest_image", None)
    assert callable(fn), "T-0111g B8: toolcall/loop.py has no swap_newest_image(prompt, frame)"
    return fn


def screenshot_payload() -> Any:
    fn = getattr(loop, "screenshot_payload", None)
    assert callable(fn), "T-0111g B8: toolcall/loop.py has no screenshot_payload(payload, updated)"
    return fn


HIST = live_frame("history")
C1, C2, C3 = live_frame("c1"), live_frame("c2"), live_frame("c3")
NEW = live_frame("post-action")


def _prompt(frames: tuple[Any, ...], *, history: bool = True) -> Prompt:
    turns = (
        [
            HistoryTurn(
                user_text="what was there",
                assistant_spoken="a page.",
                frame_phash=HIST.phash,
                frame_caption="a page",
            )
        ]
        if history
        else []
    )
    b = PromptBuilder("system", tools=({"type": "function", "function": {"name": "click_at"}},))
    prefix = b.prefix(
        turns, resolve=lambda p: HIST.jpeg if p == HIST.phash else None, frames=frames
    )
    return b.full(prefix, "click the blue mug", None, ())


def _replace_last_current(prompt: Prompt, part: ImagePart) -> Prompt:
    """By hand: the final message with its last ImagePart replaced; nothing else touched."""
    last = prompt.messages[-1]
    idx = max(i for i, p in enumerate(last.content) if isinstance(p, ImagePart))
    content = list(last.content)
    content[idx] = part
    messages = (*prompt.messages[:-1], Message(role=last.role, content=tuple(content)))
    return prompt.model_copy(update={"messages": messages})


def _urls(prompt: Prompt) -> list[str]:
    return [p.url for m in prompt.messages for p in m.content if isinstance(p, ImagePart)]


def test_swaps_only_newest_image() -> None:
    """B8 (mutation 1, "swap every image"): with a history image and three current-turn frames, only the last
    current frame is replaced; the history image, the other frames, the labels, the transcript and every field are
    unchanged."""
    base = _prompt((C1, C2, C3))
    assert _urls(base) == [
        data_url(HIST.jpeg),
        data_url(C1.jpeg),
        data_url(C2.jpeg),
        data_url(C3.jpeg),
    ], "premise: one history image then three current frames"
    frozen = copy.deepcopy(base)
    got = swap()(base, NEW)
    want = _replace_last_current(base, ImagePart(url=data_url(NEW.jpeg), phash=NEW.phash))
    assert got == want, f"T-0111g B8: only the newest image changes; images {_urls(got)}"
    assert base == frozen, "T-0111g B8: the swap is pure (its input prompt is unchanged)"
    assert (got.tools, got.thinking, got.prefix_len, got.continue_final_message) == (
        base.tools,
        base.thinking,
        base.prefix_len,
        base.continue_final_message,
    )


def test_swap_twice_replaces_the_same_slot() -> None:
    """B8: step k+1 is ``swap(loop_base, F_k)``; swapping the result again replaces the same slot, never a second
    one (so a three-step turn carries one post-action image, not a growing list)."""
    base = _prompt((C1,))
    once = swap()(base, NEW)
    twice = swap()(once, C3)
    assert _urls(twice) == [data_url(HIST.jpeg), data_url(C3.jpeg)], _urls(twice)


@pytest.mark.parametrize("history", [True, False], ids=["history_image_only", "no_image_at_all"])
def test_no_image_prompt_unchanged_or_refused(history: bool) -> None:
    """B8, the documented behaviour (reading: unchanged): a current turn with no image comes back equal, and a
    history image is never swapped in its place."""
    base = _prompt((), history=history)
    assert _urls(base) == ([data_url(HIST.jpeg)] if history else []), "premise"
    got = swap()(base, NEW)
    assert got == base, f"T-0111g B8: no current image, nothing swapped; images {_urls(got)}"


def test_data_url_matches_prompt_builder() -> None:
    """B8 "Encoding" (mutation 18): the swapped part is byte-identical to what the real builder makes for the same
    frame as a current frame. Bytes chosen so base64 pads (length not a multiple of 3)."""
    odd = live_frame("odd").model_copy(update={"jpeg": b"\xff\xd8\x00\x01\xfe"})
    got = swap()(_prompt((C1,)), odd)
    newest = [p for p in got.messages[-1].content if isinstance(p, ImagePart)][-1]
    assert newest.url == builder_url(odd), (
        "T-0111g B8: the data URL differs from the prompt builder's"
    )
    assert newest.phash == odd.phash


def test_pure_and_imports_contracts_only() -> None:
    """B8: the swap and the builder live in ``loop.py``, which still imports only the stdlib,
    ``local_astra.contracts`` and ``toolcall.special_tokens`` (T-0113 criterion 13), and opens nothing."""
    swap()
    screenshot_payload()
    tree = ast.parse((REPO / "local_astra" / "toolcall" / "loop.py").read_text())
    bad: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for n in names:
            if n.startswith("local_astra") and not (
                n.startswith("local_astra.contracts") or n == "local_astra.toolcall.special_tokens"
            ):
                bad.append(n)
            if n.split(".")[0] in ("socket", "subprocess", "asyncio", "time", "os", "httpx"):
                bad.append(n)
    assert not bad, f"T-0111g B8: loop.py imports {bad}"
    a = swap()(_prompt((C1,)), NEW)
    b = swap()(_prompt((C1,)), NEW)
    assert a == b


def show() -> Any:
    fn = getattr(loop, "show_current_image", None)
    assert callable(fn), (
        "T-0111g B8 (accepted reading): toolcall/loop.py has show_current_image(prompt, frame)"
    )
    return fn


def test_show_current_image_ride_on_places_frame_first() -> None:
    """Review A2 (kills X1, X2): a current turn whose frame rode on the newest history image (its own message holds
    no image) gets ``frame`` as the first part of its message, as the prompt builder places a current frame. The
    oracle is the builder's own prompt for the same turn with ``frame`` as the current frame; the history image is
    untouched and the input is not mutated."""
    base = _prompt(())
    assert _urls(base) == [data_url(HIST.jpeg)], "premise: the history image only"
    frozen = copy.deepcopy(base)
    got = show()(base, NEW)
    want = _prompt((NEW,))
    assert [type(p).__name__ for p in want.messages[-1].content] == ["ImagePart", "TextPart"], (
        "premise: the builder puts a current frame first in the current message"
    )
    assert got == want, (
        "T-0111g review A2: the frame goes first in the current user message, the history image untouched; "
        f"images {_urls(got)}; final parts {[type(p).__name__ for p in got.messages[-1].content]}"
    )
    assert got.messages[:-1] == base.messages[:-1], "every earlier message is unchanged"
    assert base == frozen, "pure: the input prompt is unchanged"


def test_show_current_image_with_current_image_swaps_and_without_any_is_unchanged() -> None:
    """Review A2, the other two branches: with a current image it is the swap (the newest current image replaced,
    built by hand); with no image anywhere the prompt comes back equal (a blind turn stays blind)."""
    base = _prompt((C1, C2))
    want = _replace_last_current(base, ImagePart(url=data_url(NEW.jpeg), phash=NEW.phash))
    assert show()(base, NEW) == want, _urls(show()(base, NEW))
    blind = _prompt((), history=False)
    assert _urls(blind) == [], "premise: no image at all"
    assert show()(blind, NEW) == blind, "a prompt with no image comes back unchanged"


def test_screenshot_payload_values() -> None:
    """B8: ``screenshot`` is ``"updated"`` when the image was replaced, else ``"unchanged"``; the payload's own keys
    are kept and the input is not mutated. Written out by hand."""
    build = screenshot_payload()
    executed = {"action": "clicked at (640, 200)", "status": "executed"}
    before = dict(executed)
    assert build(executed, True) == {
        "action": "clicked at (640, 200)",
        "screenshot": "updated",
        "status": "executed",
    }
    assert executed == before, "the input payload is not mutated"
    refused = loop.status_payload("refused", "the browser window was closed")
    assert build(refused, False) == {
        "reason": "the browser window was closed",
        "screenshot": "unchanged",
        "status": "refused",
    }
    rendered = loop.render_response("click_at", build(executed, True))
    assert rendered == (
        '<|tool_response>response:click_at{action:<|"|>clicked at (640, 200)<|"|>,'
        'screenshot:<|"|>updated<|"|>,status:<|"|>executed<|"|>}<tool_response|>'
    ), rendered
