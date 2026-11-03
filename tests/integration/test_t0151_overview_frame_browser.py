"""T-0151 A1-7 (group A1, browser): the overview frame through the real executor and Chromium
(docs/tasks/T-0151.md, "Tests to write"; the planner's measured scope).

**Gate.** T-0140's browser module's gate, imported (which is T-0111c's autouse ``_browser_layer``): ``integration`` +
``browser``, runs only with ``ASTRA_BROWSER_TESTS=1`` (``make browser``; the implementer adds this file to that
target, A1's Makefile scope).

**The surface** (pinned in ``tests/unit/test_t0151_executor.py``): ``overview_frame_of(key)`` after an executed
``page_overview``; a ``click_at`` on the returned frame's geometry, in the call's overview units, is translated to the
page, scrolled into view once and clicked; echo ``"model"`` (the live default) reports the call's own units with a
``scrolled down by N px and `` prefix after a scroll.

**Oracles, independent of the executor.** deals.html (a start page, not hostile) gives its own
``document.documentElement.scrollHeight`` (1220 px, measured by the planner on Chromium 156) and each button's
``getBoundingClientRect`` plus ``scrollY`` (its page box); the hand-mapped page point ``(to_pixels(x, 1280),
to_pixels(y, 1220))`` must fall inside the intended button's page box, and ``window.__state.cart`` (``cart.js``) says
what was really added. By hand: ``to_pixels(160, 1280) = 205``, ``to_pixels(933, 1220) = 1138``, ``to_pixels(935,
1220) = 1141``, ``to_pixels(308, 1220) = 376``; from ``scrollY`` 0 the scroll is ``clamp(P_y - 400, 0, 420) = 420``.
``t0151_two_screens.html`` is 1600 px in fixed CSS px with four buttons at known page boxes (its CSS):
``to_pixels(552, 1600) = 884`` (``#mid``, 860..908; scroll ``484``, unclamped) and ``to_pixels(952, 1600) = 1524``
(``#low``, 1500..1548; scroll ``clamp(1124, 0, 800) = 800``); a viewport-height mapping of (160, 552) would land on
``#decoy`` (page y 420..468) without a scroll. keys.html is 4000 px (clipped at 4.0 screens); ``?wide=1`` makes the
two-screen fixture 3840 px wide.

BASE_A1 = ab33831. Activated for group A1 (rulings O-A1 in the task file).

Regime: Playwright 1.64.0, Chromium 156 headless, 1280x800 at DSF 1, loopback only, a fresh browser per row (and per
deals click), n = 1.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_astra.conductor.tool_executor import to_pixels
from local_astra.contracts.perception import Frame
from tests.integration.test_t0140_overview_browser import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
    execute,
    executor,
    inv,
    run,
    server,
)
from tests.unit.test_start_pages import new_server

pytestmark = [pytest.mark.integration] + (
    [pytest.mark.browser] if _browser_marker_registered() else []
)

W, H = 1280, 800
BOXES = """Array.from(document.querySelectorAll('button')).map(b => {
  const r = b.getBoundingClientRect();
  return {id: b.id, item: b.dataset.item || null, left: r.left + scrollX, top: r.top + scrollY,
          right: r.right + scrollX, bottom: r.bottom + scrollY};
})"""


def overview_frame_of(ex: Any, key: Any) -> Any:
    fn = getattr(ex, "overview_frame_of", None)
    assert callable(fn), (
        "T-0151 A1-1: BrowserToolExecutor.overview_frame_of(key) does not exist yet"
    )
    return fn(key)


async def registered(ex: Any) -> Frame:
    """The live screen frame first (the session's live snapshot), then the overview, registered."""
    await run(ex.frame())
    out = await execute(ex, inv("page_overview"))
    assert out.status == "executed", out
    key = out.resolved.get("overview")
    assert isinstance(key, str), f"premise: the page is long enough for an overview: {out}"
    f = overview_frame_of(ex, key)
    assert isinstance(f, Frame), f"overview_frame_of({key!r}) gave {f!r}"
    return f


def box_of(boxes: list[dict[str, Any]], **match: Any) -> dict[str, Any]:
    hits = [b for b in boxes if all(b.get(k) == v for k, v in match.items())]
    assert len(hits) == 1, (match, boxes)
    return hits[0]


def inside(box: dict[str, Any], x: int, y: int) -> bool:
    return bool(box["left"] <= x < box["right"] and box["top"] <= y < box["bottom"])


@pytest.mark.parametrize("y", [933, 935])
async def test_deals_tea_towel_via_overview(y: int) -> None:
    """A1-7 (B2 P9's oracle): on a fresh deals.html, ``click_at(160, y)`` on the overview frame (838x798) is
    ``executed`` with exactly ``scrolled down by 420 px and clicked at (160, y)``, leaves ``scrollY`` at 420, and the
    cart is exactly ``["cotton tea towel"]``. The hand-mapped page point is inside the tea towel button's page box."""
    srv = new_server()
    srv.start()
    try:
        async with executor(allowed_origins=frozenset({srv.origin}), echo="model") as ex:
            await run(ex.open(srv.origin + "/demo/deals.html"))
            page = ex.driver.page
            height = await page.evaluate("document.documentElement.scrollHeight")
            assert height == 1220, (
                f"premise: deals.html is 1220 px at 1280x800 (planner, Chromium 156): {height}"
            )
            boxes = await page.evaluate(BOXES)
            px, py = to_pixels(160, W), to_pixels(y, 1220)
            assert inside(box_of(boxes, item="cotton tea towel"), px, py), (px, py, boxes)
            ov = await registered(ex)
            assert (ov.width, ov.height) == (838, 798), (ov.width, ov.height)
            out = await execute(ex, inv("click_at", ov, x=160, y=y))
            assert out.status == "executed", out
            assert out.action == f"scrolled down by 420 px and clicked at (160, {y})", out.action
            assert await page.evaluate("window.scrollY") == 420
            assert await page.evaluate("window.__state.cart") == ["cotton tea towel"]
            r = out.resolved
            assert (r.get("x_px"), r.get("y_px"), r.get("page_y_px"), r.get("scrolled_px")) == (
                px,
                py - 420,
                py,
                420,
            ), r
    finally:
        srv.stop()


async def test_deals_teapot_no_scroll() -> None:
    """A1-7: ``click_at(160, 308)`` (the teapot, page y 376, in view) is ``executed`` with BASE's ``clicked at (160,
    308)`` and no scroll prefix; ``scrollY`` stays 0 and the cart is ``["ceramic teapot"]``."""
    srv = new_server()
    srv.start()
    try:
        async with executor(allowed_origins=frozenset({srv.origin}), echo="model") as ex:
            await run(ex.open(srv.origin + "/demo/deals.html"))
            page = ex.driver.page
            boxes = await page.evaluate(BOXES)
            assert inside(
                box_of(boxes, item="ceramic teapot"), to_pixels(160, W), to_pixels(308, 1220)
            )
            ov = await registered(ex)
            out = await execute(ex, inv("click_at", ov, x=160, y=308))
            assert (out.status, out.action) == ("executed", "clicked at (160, 308)"), out
            assert await page.evaluate("window.scrollY") == 0
            assert await page.evaluate("window.__state.cart") == ["ceramic teapot"]
    finally:
        srv.stop()


TWO_SCREEN_ROWS = [
    pytest.param(552, 884, 484, "mid", id="mid-unclamped"),
    pytest.param(952, 1524, 800, "low", id="low-clamped"),
]


@pytest.mark.parametrize(("y", "page_y", "scrolled", "button"), TWO_SCREEN_ROWS)
async def test_two_screen_fixture_unclamped(
    server: str,  # noqa: F811  (the fixture)
    y: int,
    page_y: int,
    scrolled: int,
    button: str,
) -> None:
    """A1-4/A1-7 on the 1600 px fixture (2.0 screens, overview 640x800): a mid-page target (page y 884) scrolls to
    ``clamp(884 - 400, 0, 800) = 484`` and a bottom one (1524) to the clamp 800; each hits exactly the intended button
    (``window.__clicks``), never ``#decoy``, with the exact action string."""
    async with executor(echo="model") as ex:
        await run(ex.open(f"{server}/t0151_two_screens.html"))
        page = ex.driver.page
        assert await page.evaluate("document.documentElement.scrollHeight") == 1600, (
            "premise: 1600 px"
        )
        boxes = await page.evaluate(BOXES)
        assert to_pixels(y, 1600) == page_y
        assert inside(box_of(boxes, id=button), to_pixels(160, W), page_y), boxes
        ov = await registered(ex)
        assert (ov.width, ov.height) == (640, 800), (ov.width, ov.height)
        out = await execute(ex, inv("click_at", ov, x=160, y=y))
        assert out.status == "executed", out
        assert out.action == f"scrolled down by {scrolled} px and clicked at (160, {y})", out.action
        assert await page.evaluate("window.scrollY") == scrolled
        assert await page.evaluate("window.__clicks") == [button]


async def test_wide_and_clipped_not_registered(server: str) -> None:  # noqa: F811  (the fixture)
    """A1-1 in Chromium: keys.html (4000 px, 5.0 screens) gives a clipped overview, and the two-screen fixture made
    3840 px wide (``?wide=1``) gives a picture wider than the viewport; ``overview_frame_of`` is ``None`` for both and
    the picture is dropped. ``scroll_wide.html``, which "Tests to write" names for the wide case, is one screen tall
    (``100vh``), so it gets no overview at all (the fits outcome, no key): that is asserted as the premise for why the
    wide row uses the fixture's wide variant instead."""
    async with executor() as ex:
        await run(ex.open(f"{server}/keys.html"))
        await run(ex.frame())
        out = await execute(ex, inv("page_overview"))
        assert out.status == "executed" and out.resolved.get("clipped") is True, out
        key = out.resolved["overview"]
        assert overview_frame_of(ex, key) is None, "a clipped picture was registered"
        assert ex.overview_of(key) is None, "the refused picture was not dropped"
    async with executor() as ex:
        await run(ex.open(f"{server}/t0151_two_screens.html?wide=1"))
        page = ex.driver.page
        assert await page.evaluate("document.documentElement.scrollWidth") == 3840, (
            "premise: 3840 px wide"
        )
        await run(ex.frame())
        out = await execute(ex, inv("page_overview"))
        assert out.status == "executed", out
        key = out.resolved.get("overview")
        assert isinstance(key, str), out
        assert overview_frame_of(ex, key) is None, (
            "a picture wider than the viewport was registered"
        )
    async with executor() as ex:
        await run(ex.open(f"{server}/scroll_wide.html"))
        await run(ex.frame())
        out = await execute(ex, inv("page_overview"))
        assert out.status == "executed" and out.resolved.get("overview") is None, (
            f"premise: scroll_wide.html is one screen tall and gets no overview: {out}"
        )
