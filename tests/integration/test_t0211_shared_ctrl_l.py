"""T-0211 group B, B6: the browser premise of S_SHARED (docs/tasks/T-0211.md, "Tests to write", integration).

S_SHARED promises the model that from the user's shared screen it can "press Ctrl+L to open your browser's start page
and continue there". These rows prove the promise holds at BASE, through a real ``Session`` and the real web-mode
executor, before any guidance text changes. If ``test_shared_step1_ctrl_l_switches_to_browser`` is red at BASE,
S_SHARED must not ship (B6).

**Gate.** T-0111c's autouse ``_browser_layer`` (imported through the T-0208 B / T-0205 browser modules):
``integration`` + ``browser``, runs only with ``ASTRA_BROWSER_TESTS=1``; run by path::

    ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0211_shared_ctrl_l.py -m browser -q

**The rig** is T-0208 B's handoff session rig, imported (``session_on``: the scenario server on loopback, the web-mode
executor with the deny-all resolver and connector opened at H1's start page, ``vision_rig`` with the production
ingress, loop 20, the screen label on). H1's committed shared image is posted as an ``astra.frame`` (screen source),
``astra.vision.select {source: "shared"}`` goes over the wire, then H1's request as ``UserText``. Step 1 is scripted;
step 2 is a no-call answer, so the turn ends after it.

**What is pinned** (B6, readings stated):

- ``test_shared_step1_ctrl_l_switches_to_browser``: step 1 ``press_key Ctrl+L`` is ``executed`` with the action
  ``"opened the start page"`` (T-0191's ``START_PAGE_ACTION``, typed); the page is then ``<origin>/web.html``; step 2's
  current image (its last user message holds exactly one image) is a fresh browser frame (one the executor returned
  after the step-1 action: ``Watch.after``) and not the stored shared bytes; the text part right after it is a browser
  label (reading: it opens ``[This screenshot is the browser screen now``, the opening shared by ``LABEL_NOW``,
  ``LABEL_UPDATED`` and ``LABEL_UNCHANGED``, typed from T-0134) and is not ``SHARED_LABEL_SCREEN``; no step-2 body
  (generate, prefill, resend) carries the shared bytes' base64. Premise: step 1's own body carries the shared image and
  ``SHARED_LABEL_SCREEN`` (so the switch is a switch).
- ``test_shared_step1_click_still_refused`` (T-0177 B2 unchanged, the too-wide companion: Ctrl+L must not open a
  path for coordinates read off the picture): a step-1 ``click_at`` is ``refused`` with ``SHARED_FRAME_REASON``,
  nothing reaches the executor at step 1, and the page is still the start page.

Both rows are active at BASE (main 175a29b0): they pin existing behaviour that S_SHARED relies on.

Regime: Playwright and Chromium as pinned for ``make browser``, headless, 1280x800 at DSF 1, web mode on loopback,
deny-all resolver and connector, ``ScriptedClock``, loop 20, screen label on, FakeLlm one script per step; n = 1.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.perception import Source
from local_astra.contracts.turn import TurnState
from tests.fakes.llm import is_caption_prompt
from tests.integration import test_t0208_handoff_session as h
from tests.integration.test_computer_live_session import click, drive
from tests.integration.test_t0208_handoff_session import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
)
from tests.unit._live_browser_support import b64
from tests.unit._tool_turn_support import answer, tool
from tests.unit.test_session_live import _wire, last_user

pytestmark = [
    pytest.mark.integration,
    *([pytest.mark.browser] if _browser_marker_registered() else []),
]

HID = "H1"
Q = '<|"|>'
PRESS_CTRL_L = f"<|tool_call>call:press_key{{combo:{Q}Ctrl+L{Q}}}<tool_call|>"
#: T-0191 / executor ``START_PAGE_ACTION``, typed from T-0191's "Confirmed in source"
START_PAGE_ACTION = "opened the start page"
START_PAGE_PATH = "/web.html"
#: T-0134 A3: the opening every browser label shares (LABEL_NOW, LABEL_UPDATED, LABEL_UNCHANGED), typed
BROWSER_LABEL_OPENING = "[This screenshot is the browser screen now"
SAY = "I will open my own browser to find that part. "
DONE = "I opened my browser's start page. "


@dataclass
class SharedTurn:
    stored: bytes
    step1_bodies: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    later_bodies: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    results: list[tuple[str, str]] = field(default_factory=list)
    outcomes: list[Any] = field(default_factory=list)
    cards: dict[str, Any] = field(default_factory=dict)
    after_step1: list[bytes] = field(default_factory=list)
    url_after_step1: str = ""
    start_url: str = ""
    origin: str = ""
    step2_body: dict[str, Any] | None = None


async def shared_turn(tmp_path: Path, step1: str) -> SharedTurn:
    """One shared H1 turn: post and select the shared image, ``UserText``, step 1 ``step1``, step 2 a no-call
    answer."""
    data = h.image_bytes(HID)
    async with h.session_on(tmp_path, HID) as (rig, watch, ex, _net, origin):
        page = ex.driver.page
        start_url = page.url
        await h.post(rig, h.frame_message(HID, data))
        stored_frame = await rig.perception.get_frame(Source.SCREEN)
        assert stored_frame is not None and stored_frame.jpeg == data, (
            "premise: the session's screen source holds the posted shared bytes"
        )
        await h.select(rig, "shared")
        assert rig.session.vision_source == "shared", "premise: the select reached the session"
        rig.llm.script = [answer(SAY), tool(step1)]
        rig.llm.hold_steps()
        rig.submit(E.UserText(t_ms=rig.clock.now_ms, text=h.row(HID).request))
        turn = SharedTurn(stored=data, start_url=start_url, origin=origin)
        await h.wait_result(rig, watch, 1)
        turn.url_after_step1 = page.url
        turn.after_step1 = [f.jpeg for f in watch.after if f is not None]
        await h.give_step(rig, watch, 1, [answer(DONE)])
        rig.llm.release_steps()
        await drive(
            rig,
            watch,
            lambda: watch.inflight == 0 and rig.state.state is TurnState.IDLE and bool(rig.lines()),
            "T-0211 B6: the turn never ended",
            limit_s=120,
        )
        await rig.advance(2000)
        turn.step2_body = _wire(h.tsl.step_request(rig, 2))
        for rid, prompt, _opts in rig.llm.generates:
            if is_caption_prompt(prompt) or "t1-gen" not in rid:
                continue
            (turn.later_bodies if "t1-gen-s" in rid else turn.step1_bodies).append(
                (rid, _wire(prompt))
            )
        for rid, prefix in rig.llm.prefills:
            if "t1-gen" not in rid:
                continue
            (turn.later_bodies if "t1-gen-s" in rid else turn.step1_bodies).append(
                (f"prefill:{rid}", _wire(prefix))
            )
        turn.results = [(e.call_id, e.status) for e in h.results_of(rig)]
        turn.outcomes = list(watch.outcomes)
        turn.cards = {
            cid: rig.tool_calls(cid)[-1] for cid, _ in turn.results if rig.tool_calls(cid)
        }
    return turn


def current_image_and_label(body: dict[str, Any]) -> tuple[bytes, Any]:
    parts = last_user(body)["content"]
    at = [i for i, p in enumerate(parts) if p.get("type") == "image_url"]
    assert len(at) == 1, f"the step's last user message holds {len(at)} images, not one"
    url = parts[at[0]]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,"), url[:40]
    nxt = parts[at[0] + 1] if at[0] + 1 < len(parts) else None
    return base64.b64decode(url.split(",", 1)[1]), nxt


def assert_step1_is_shared(turn: SharedTurn) -> None:
    """Premise: step 1 showed the shared image with ``SHARED_LABEL_SCREEN`` (so a browser image at step 2 is a
    switch)."""
    gens = [(rid, b) for rid, b in turn.step1_bodies if not rid.startswith("prefill:")]
    assert gens, "premise: the step-1 generate was read"
    for rid, body in gens:
        img, label = current_image_and_label(body)
        assert img == turn.stored, f"premise: {rid}'s image is the shared bytes"
        assert label == {"type": "text", "text": h.SHARED_LABEL_SCREEN}, (
            f"premise: {rid} labels the shared screen; {label!r}"
        )


async def test_shared_step1_ctrl_l_switches_to_browser(tmp_path: Path) -> None:
    """B6 (active at BASE; S_SHARED's premise): from a shared step 1, ``press_key Ctrl+L`` is executed with
    ``"opened the start page"``, the page is ``/web.html``, and step 2's current image is a fresh browser frame of that
    page under a browser label, with no shared byte in any step-2 body."""
    turn = await shared_turn(tmp_path, PRESS_CTRL_L)
    assert_step1_is_shared(turn)
    assert turn.start_url != turn.origin + START_PAGE_PATH, (
        f"premise: the browser did not start on the start page ({turn.start_url})"
    )
    assert turn.results and turn.results[0] == ("t1-call", "executed"), (
        f"T-0211 B6: step 1's Ctrl+L is executed; results {turn.results}"
    )
    assert turn.outcomes and turn.outcomes[0].action == START_PAGE_ACTION, (
        f"T-0211 B6: step 1's action is {START_PAGE_ACTION!r}; "
        f"{[getattr(o, 'action', o) for o in turn.outcomes[:1]]}"
    )
    assert turn.url_after_step1 == turn.origin + START_PAGE_PATH, (
        f"T-0211 B6: Ctrl+L opened the start page; the page is {turn.url_after_step1}"
    )
    s2 = turn.step2_body
    assert s2 is not None, "premise: step 2 was requested"
    img, label = current_image_and_label(s2)
    assert img != turn.stored, "T-0211 B6: step 2's current image is the shared screen"
    assert turn.after_step1 and img == turn.after_step1[0], (
        "T-0211 B6: step 2's current image is the browser frame the executor returned after Ctrl+L"
    )
    assert isinstance(label, dict) and label.get("type") == "text", label
    text = label.get("text", "")
    assert text != h.SHARED_LABEL_SCREEN and text.startswith(BROWSER_LABEL_OPENING), (
        f"T-0211 B6: step 2's image carries a browser label; {text!r}"
    )
    assert turn.later_bodies, "premise: step-2 bodies were read"
    shared = b64(turn.stored)
    for rid, body in turn.later_bodies:
        assert shared not in json.dumps(body), f"T-0211 B6: {rid} carries the shared bytes"


async def test_shared_step1_click_still_refused(tmp_path: Path) -> None:
    """T-0177 B2 unchanged (too-wide companion): a step-1 ``click_at`` read off the shared image is refused with
    ``SHARED_FRAME_REASON``; nothing reaches the executor at step 1 and the page does not move."""
    turn = await shared_turn(tmp_path, click(500, 500))
    assert_step1_is_shared(turn)
    assert turn.results and turn.results[0] == ("t1-call", "refused"), turn.results[:1]
    card = turn.cards.get("t1-call")
    assert card is not None and card.reason == h.SHARED_FRAME_REASON, card
    assert turn.outcomes == [], f"the refused click reached the executor: {turn.outcomes}"
    assert turn.url_after_step1 == turn.start_url, (turn.start_url, turn.url_after_step1)
