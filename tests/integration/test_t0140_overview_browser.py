"""T-0140 A3/A4 (group A1, browser): ``page_overview`` through the real executor and Chromium
(docs/tasks/T-0140.md, "Tests to write").

**Gate.** As T-0111c's module (its autouse gate imported): ``integration`` + ``browser``, runs only with
``ASTRA_BROWSER_TESTS=1`` (``make browser``; the implementer adds this file to that target, A1's Makefile scope).

**The surface** (pinned in ``tests/unit/test_t0140_executor.py``): ``execute(page_overview)`` with A3's action
strings, ``resolved["overview"]`` and ``BrowserToolExecutor.overview_of(id)``.

Expected values: deals.html (a start page) is not hostile, so its own ``document.documentElement.scrollHeight`` and
``window.scrollY`` are an independent oracle (``s = scrollHeight / 800``; 1220 px, 1.5 screens, measured 2026-10-23
on Chromium 156); the hostile fixture's true height is 2800 px by hand from its CSS (3.5 screens, not clipped), while
the page tells its own script 100; keys.html is 4000 px (5.0 screens: clipped to the top 4.0); color.html is one
screen (fits). The page-side listener is installed by the row (``page.evaluate``, test-side only) and counts
``pointer*``, ``wheel`` and ``scroll`` events in the capture phase; Chromium fires a ``resize`` during a
beyond-viewport capture (probe 2026-10-23), which is not input and is not counted.

Activated for T-0140 group A1 (rulings O-A1-1..8). BASE (group A1): 0bc60ff.

Regime: Playwright 1.64.0, Chromium 156 headless, 1280x800 at DSF 1, loopback only, n = 1.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
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
FITS = "the whole page fits on the browser screen; no overview was taken"
LISTEN = """(() => {
  window.__t0140 = [];
  const kinds = ["pointerdown", "pointerup", "pointermove", "pointerover", "pointerenter", "pointerout",
                 "pointerleave", "pointercancel", "wheel", "scroll"];
  for (const k of kinds) {
    window.addEventListener(k, (e) => window.__t0140.push(e.type), true);
    document.addEventListener(k, (e) => window.__t0140.push("document:" + e.type), true);
  }
})()"""
ACTION_RE = re.compile(
    r"^showed the whole page \(about (\d+\.\d) screens tall(, top 4\.0 screens shown)?\) as a picture before the "
    r"screen$"
)


def overview_frame(ex: Any, out: Any) -> Any:
    fn = getattr(ex, "overview_of", None)
    assert callable(fn), "T-0140 A3: BrowserToolExecutor.overview_of does not exist yet"
    key = out.resolved.get("overview")
    assert key is not None, out
    f = fn(key)
    assert f is not None, "overview_of gave nothing on the first read"
    assert fn(key) is None, "overview_of gave the frame twice"
    return f


async def test_deals_overview_no_input_no_scroll() -> None:
    """A3/A4, B2's P6 facts on deals.html at 1280x800: the overview is taken (``s >= 1.3``, the action's number
    is ``scrollHeight / 800`` to one decimal, not clipped), its frame is within 1280x800 with the page's aspect,
    ``scrollY`` is unchanged, and the page saw no ``pointer*``/``wheel``/``scroll`` event."""
    srv = new_server()
    srv.start()
    try:
        async with executor(allowed_origins=frozenset({srv.origin})) as ex:
            await run(ex.open(srv.origin + "/demo/deals.html"))
            await run(ex.frame())
            page = ex.driver.page
            height = float(await page.evaluate("document.documentElement.scrollHeight"))
            await page.evaluate(LISTEN)
            y0 = await page.evaluate("window.scrollY")
            out = await execute(ex, inv("page_overview"))
            assert out.status == "executed", out
            m = ACTION_RE.match(out.action)
            assert m is not None, out.action
            assert m.group(1) == f"{height / H:.1f}" and m.group(2) is None, (out.action, height)
            assert height / H >= 1.3, height
            f = overview_frame(ex, out)
            assert f.width <= W and f.height <= H, (f.width, f.height)
            assert abs(f.width / f.height - W / height) <= 0.02, (f.width, f.height, height)
            assert await page.evaluate("window.scrollY") == y0, "the overview scrolled the page"
            events = await page.evaluate("window.__t0140")
            assert events == [], f"the overview dispatched input or scrolled: {events}"
    finally:
        srv.stop()


async def test_scrolled_page_stays_put(server: str) -> None:  # noqa: F811  (the fixture)
    """A3 "does not scroll": on keys.html (4000 px in fixed CSS px) after one ``scroll down`` (600 px), the overview
    leaves ``scrollY`` at 600 and reports 5.0 screens, clipped.

    Not scroll.html: its body is ``300vh``, and Chromium's beyond-viewport capture re-lays it out, which moved a
    page scrolled to 600 back to 2 (probe 2026-10-23); see ``test_vh_page_moved_is_failed`` and the A1 report."""
    async with executor() as ex:
        await run(ex.open(f"{server}/keys.html"))
        await run(ex.frame())
        out = await execute(ex, inv("scroll", None, direction="down"))
        assert out.status == "executed", out
        page = ex.driver.page
        y = await page.evaluate("window.scrollY")
        assert y == 600, f"premise: one scroll down is 600 px; {y}"
        out = await execute(ex, inv("page_overview"))
        assert out.action == (
            "showed the whole page (about 5.0 screens tall, top 4.0 screens shown) as a picture before the screen"
        ), out.action
        assert await page.evaluate("window.scrollY") == 600


async def test_vh_page_moved_is_failed(server: str) -> None:  # noqa: F811  (the fixture)
    """A3 "Metrics" in a real browser: scroll.html's body is ``300vh``; scrolled to 600, a full-page capture re-lays
    it out and its CDP ``scroll_y`` comes back changed (probe 2026-10-23: 600 -> 2), so the outcome is ``failed``
    (``the page moved during the overview``) with no image. (That the page *did* move is a spec finding reported
    with A1; this row pins only that the executor does not present such a picture as taken.)"""
    async with executor() as ex:
        await run(ex.open(f"{server}/scroll.html"))
        await run(ex.frame())
        out = await execute(ex, inv("scroll", None, direction="down"))
        assert out.status == "executed", out
        out = await execute(ex, inv("page_overview"))
        assert (out.status, out.reason) == ("failed", "the page moved during the overview"), out
        assert out.resolved.get("overview") is None, out.resolved


async def test_hostile_page_true_height(server: str) -> None:  # noqa: F811  (the fixture)
    """A4 (adversarial; the T-0134 A4a fixture): a page that tells its script it is 100 px tall still gets its true
    2800 px (3.5 screens), not clipped, a frame of about 366x800."""
    async with executor() as ex:
        await run(ex.open(f"{server}/t0134_hostile_extent.html"))
        page = ex.driver.page
        assert await page.evaluate("document.documentElement.scrollHeight") == 100, (
            "premise: the page lies"
        )
        out = await execute(ex, inv("page_overview"))
        assert out.action == (
            "showed the whole page (about 3.5 screens tall) as a picture before the screen"
        ), out.action
        f = overview_frame(ex, out)
        assert f.height <= H and abs(f.width - 1280 * 800 / 2800) <= 3, (f.width, f.height)


async def test_tall_page_clipped(server: str) -> None:  # noqa: F811  (the fixture)
    """A3 clip at four screens through Chromium: keys.html (4000 px) reads 5.0 screens, ``top 4.0 screens shown``,
    and its frame is the top 1280x3200 scaled to 320x800 (within 2 px)."""
    async with executor() as ex:
        await run(ex.open(f"{server}/keys.html"))
        out = await execute(ex, inv("page_overview"))
        assert out.action == (
            "showed the whole page (about 5.0 screens tall, top 4.0 screens shown) as a picture before the screen"
        ), out.action
        f = overview_frame(ex, out)
        assert abs(f.width - 320) <= 2 and abs(f.height - 800) <= 2, (f.width, f.height)


async def test_fitting_page_fits(server: str) -> None:  # noqa: F811  (the fixture)
    """A3 "Page fits": color.html is one screen, so the outcome is the fits string and no image."""
    async with executor() as ex:
        await run(ex.open(f"{server}/color.html"))
        out = await execute(ex, inv("page_overview"))
        assert (out.status, out.action) == ("executed", FITS), out
        assert out.resolved.get("overview") is None, out.resolved
