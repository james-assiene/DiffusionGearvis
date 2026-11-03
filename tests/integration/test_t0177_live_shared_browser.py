"""T-0177 group B, the live e2e (docs/tasks/T-0177.md "Tests to write", group B; ``make browser-live``): a real
``Session``, the real headless agent browser on the loopback start pages, a fake LLM, and HUD frames posted as
``astra.frame`` through the real ``BrowserFrameSource`` (``vision_rig(..., ingress=True)``, the production shape of
``cli``), with the choice switched to ``shared`` over the wire.

Rows (one per bullet of the task):

1. B7/A6: step 1 carries the posted shared bytes and ``SHARED_LABEL_SCREEN``, and no byte of the agent browser; the
   ``astra.vision.frame`` published for step 1 carries exactly the bytes the body sent (B7's key row, live).
2. B1: the model scrolls; the scroll executes on the real page, and step 2's newest image is the post-scroll browser
   frame, with the shared bytes and the shared label nowhere in step 2.
3. B2: a ``click_at`` read off the shared image is refused with ``SHARED_FRAME_REASON``; nothing is dispatched and no
   click event reaches the page (a capture-phase listener the test installs counts them: an oracle independent of the
   session and of the executor).

**Gate.** As ``tests/integration/test_computer_live_session.py``: ``integration`` + ``browser``, runs only with
``ASTRA_BROWSER_TESTS=1`` (its autouse gate imported). **The rig.** That module's ``Watch`` over a real
``BrowserToolExecutor(headless=True, end_when_closed=False, allowed_origins={start-page origin})`` on a start-page
server at an ephemeral port on 127.0.0.1, passed as the browser of ``tests.unit.test_t0177_shared_frame.vision_rig``
(the live tool rig, a socketless ``HudServer`` and one greeted page). Its ``drive`` advances virtual time only while no
browser call is in flight.

Expected values: the shared image's URL is ``builder_url`` of the posted ``Frame`` (the real ``PromptBuilder``'s
encoding of those bytes); the browser image's URL is ``data_url`` of the JPEG the real executor returned
(``Watch.after``/``Watch.frames``); ``SHARED_LABEL_SCREEN`` and ``SHARED_FRAME_REASON`` are typed from the task.

Regime: Playwright and Chromium as pinned for ``make browser``, headless, 1280x800 at DSF 1, start pages on 127.0.0.1
at an ephemeral port, FakeLlm with one script per step, ScriptedClock, fake STT/TTS/audio, n = 1 per row.
"""

from __future__ import annotations

import base64
import contextlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.turn import TurnState
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
    computer,
)
from tests.integration.test_computer_live_session import (
    SCROLL,
    SEARCH_BOX,
    Watch,
    ask,
    center,
    click,
    drive,
    per_turn_checks,
)
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import b64, builder_url, data_url
from tests.unit._tool_turn_support import answer, tool
from tests.unit.test_session_live import _wire, last_user
from tests.unit.test_start_pages import new_server
from tests.unit.test_t0177_shared_frame import (
    SHARED_LABEL_SCREEN,
    post,
    posted,
    select,
    vision_rig,
)

pytestmark = [pytest.mark.integration] + (
    [pytest.mark.browser] if _browser_marker_registered() else []
)

FRAME = "astra.vision.frame"
#: B2, typed from the task
SHARED_FRAME_REASON = (
    "the image you read those coordinates from is the user's shared screen, not your browser; click_at and type_at "
    "act only on your browser"
)


@contextlib.asynccontextmanager
async def shared_live(
    tmp_path: Path, start: str = "/", *, tag: str
) -> AsyncIterator[tuple[Any, Watch, Any]]:
    """The real browser on the start pages under ``vision_rig(..., ingress=True)``; one shared screen frame posted
    as ``astra.frame`` through the page, then ``select shared``. Yields ``(rig, watch, shared Frame)``."""
    srv = new_server()
    srv.start()
    ex = computer().BrowserToolExecutor(
        now_ms=lambda: 0,
        headless=True,
        end_when_closed=False,
        allowed_origins=frozenset({srv.origin}),
    )
    watch = Watch(ex)
    try:
        await ex.open(srv.origin + start)
        async with vision_rig(tmp_path, watch, ingress=True) as rig:
            ex._now_ms = lambda: rig.clock.now_ms
            message, frame = posted(tag)
            await post(rig, message)
            await select(rig, "shared")
            yield rig, watch, frame
    finally:
        with contextlib.suppress(Exception):
            await ex.close()
        srv.stop()


async def turn_end(rig: Any, watch: Watch) -> None:
    await drive(
        rig, watch, lambda: rig.state.state is TurnState.IDLE and bool(rig.lines()), "turn end"
    )


def current_images(body: dict[str, Any]) -> list[str]:
    """The image URLs of the body's final user message, in order (the request's current image(s))."""
    return [
        p["image_url"]["url"] for p in last_user(body)["content"] if p.get("type") == "image_url"
    ]


def step_body(rig: Any, k: int) -> dict[str, Any]:
    return _wire(tsl.step_request(rig, k))


# --------------------------------------------------------------------------------------------- rows


async def test_step1_carries_the_shared_bytes_and_none_of_the_browsers(tmp_path: Path) -> None:
    """B7/A6 live: the step-1 request's current image is the posted shared frame, labelled ``SHARED_LABEL_SCREEN``;
    the browser's own current screenshot (taken here straight from the executor, so it is a real candidate) is in
    no request body, and the session asked the agent browser for no frame for that snapshot."""
    async with shared_live(tmp_path, tag="e2e-step1") as (rig, watch, shared):
        own = (
            await watch._inner.frame()
        )  # not through Watch: the session's frame() calls are counted
        rig.llm.script = [answer("I see your shared screen. ")]
        await ask(rig, watch, "what is on my screen")
        await turn_end(rig, watch)
        body = _wire(tsl.sent(rig, "t1-gen"))
        bodies = json.dumps(rig.llm.bodies)
        session_frames = list(watch.frames)
        published = [m for m in rig.published.messages if getattr(m, "type", None) == FRAME]
    assert current_images(body) == [builder_url(shared)], "A6/B7: step 1 carries the shared bytes"
    # B7 key row, live: the panel's frame for step 1 is exactly what the body sent (decoded from the body's own URL)
    sent = [base64.b64decode(u.split(",", 1)[1]) for u in current_images(body)]
    step1 = [m for m in published if m.turn_id == "t1" and m.step == 1]
    assert step1, f"B7: step 1 publishes one astra.vision.frame; published {len(published)}"
    assert (step1[-1].source, step1[-1].blind) == ("shared", False), step1[-1]
    assert [base64.b64decode(i.jpeg_b64) for i in step1[-1].images] == sent == [shared.jpeg], (
        "B7: the published bytes are exactly the bytes the step-1 body sent"
    )
    assert {"type": "text", "text": SHARED_LABEL_SCREEN} in last_user(body)["content"], (
        "A7: the shared label follows the shared image"
    )
    assert b64(own.jpeg) not in bodies, "A6 adversarial: the agent browser's bytes reach no body"
    assert session_frames == [], (
        f"A6: the shared step 1 never calls the live browser's frame(); {len(session_frames)} call(s)"
    )


async def test_a_scroll_swaps_in_the_browser_frame_at_step_2(tmp_path: Path) -> None:
    """B1 live: under ``shared`` the model's step-1 ``scroll`` executes on the real deals page (it scrolls), and step
    2's current image is exactly the post-scroll JPEG the executor returned: the shared image is replaced, never
    kept beside it, and the shared label is gone."""
    async with shared_live(tmp_path, "/demo/deals.html", tag="e2e-scroll") as (rig, watch, shared):
        ex = watch._inner
        assert await ex.driver.page.evaluate("window.scrollY") == 0, (
            "premise: the page starts at the top"
        )
        rig.llm.script = [answer("I will look further down. "), tool(SCROLL)]
        rig.llm.script_steps([answer("The deals continue below. ")])
        await ask(rig, watch, "what deals are further down")
        await turn_end(rig, watch)
        scrolled = await ex.driver.page.evaluate("window.scrollY")
        gen = _wire(tsl.sent(rig, "t1-gen"))
        s2 = step_body(rig, 2)
        outcomes = list(watch.outcomes)
        after = list(watch.after)
        per_turn_checks(rig, "t1")
    assert current_images(gen) == [builder_url(shared)], "premise: step 1 is the shared frame"
    assert [o.status for o in outcomes] == ["executed"], f"B1: the scroll executes; {outcomes}"
    assert scrolled > 0, "B1: the scroll reached the real page"
    assert current_images(s2) == [data_url(after[0].jpeg)], (
        "B1: step 2 shows the post-scroll browser frame, and only it"
    )
    assert b64(shared.jpeg) not in json.dumps(s2), (
        "B1: the shared image is replaced, never kept beside it"
    )
    assert SHARED_LABEL_SCREEN not in json.dumps(s2), (
        "B1: the shared label goes with the shared image"
    )


async def test_a_click_read_off_the_shared_image_is_refused_and_never_reaches_the_page(
    tmp_path: Path,
) -> None:
    """B2 live ("not caught" direction, the failure mode): the model clicks at the search box's coordinates while its
    current image is the shared screen. The call is refused with ``SHARED_FRAME_REASON`` (not the freshness reason),
    the executor is never asked to execute it, and the page records no click event at all. Step 2 then shows a fresh
    frame of the agent browser (B2: the failed/timeout path), not the shared image."""
    async with shared_live(tmp_path, tag="e2e-click") as (rig, watch, _shared):
        ex = watch._inner
        x, y = await center(ex, SEARCH_BOX)
        await ex.driver.page.evaluate(
            "window.__t0177Clicks = 0;"
            " document.addEventListener('click', () => { window.__t0177Clicks += 1; }, true);"
            " document.addEventListener('mousedown', () => { window.__t0177Clicks += 1; }, true);"
        )
        rig.llm.script = [answer("I will click the search box. "), tool(click(x, y))]
        rig.llm.script_steps([answer("That was your screen, not my browser. ")])
        await ask(rig, watch, "click the search box")
        await turn_end(rig, watch)
        clicks = await ex.driver.page.evaluate("window.__t0177Clicks")
        cards = rig.tool_calls("t1-call")
        s2 = step_body(rig, 2)
        session_frames = list(watch.frames)
        invocations = list(watch.invocations)
    assert invocations == [], f"B2: nothing is dispatched; {invocations}"
    assert clicks == 0, f"B2: no click reaches the page; the page counted {clicks}"
    assert cards and (cards[-1].status, cards[-1].reason) == ("refused", SHARED_FRAME_REASON), (
        f"B2: refused with SHARED_FRAME_REASON; {cards[-1] if cards else None}"
    )
    assert session_frames, "B2: a fresh browser frame is taken for step 2"
    assert current_images(s2) == [data_url(session_frames[-1][1].jpeg)], (
        "B2: step 2 shows the fresh browser frame"
    )
