"""T-0134 A2/A4/A4a (browser): the page extent and the echo through the real executor and Chromium
(docs/tasks/T-0134.md, "Tests to write").

**Gate.** As T-0111c's module (its autouse gate imported): ``integration`` + ``browser``, runs only with
``ASTRA_BROWSER_TESTS=1`` (``make browser``; the implementer adds this file to that target). A missing ``playwright``
or Chromium is then a failure naming P1.

**The surface** (pinned in ``tests/unit/test_t0134_extent.py``): ``BrowserToolExecutor.extent_of(phash) ->
{"above_px", "below_px", "viewport_px"} | None`` and ``BrowserToolExecutor(..., echo="model")``.

Expected values: deals.html (a start page, ``local_astra/pages``) is not hostile, so the page's own
``document.documentElement.scrollHeight`` and ``scrollY`` are an independent oracle for it (within 1 CSS px); the
hostile fixture's true geometry is by hand from its CSS (2800 px tall: 0 above and 2000 below at scroll 0, 600 above
and 1400 below after one ``scroll down`` of 600 px).

All rows ``pending(task="T-0134")``.

Regime: Playwright 1.64.0, Chromium 156 headless, 1280x800 at DSF 1, loopback only, n = 1.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
    computer,
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


def extent(ex: Any, frame: Any) -> Any:
    fn = getattr(ex, "extent_of", None)
    assert callable(fn), "T-0134 A4: BrowserToolExecutor.extent_of does not exist yet"
    return fn(frame.phash)


def close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= 1.0


async def test_deals_extent_at_scroll_0_and_after_scroll_down() -> None:
    """A4/B2's P5 facts on deals.html at 1280x800: at scroll 0 the extent is 0 above, ``scrollHeight - 800`` below
    (at least 0.3 screens: the tea towel is below the fold), viewport 800; after ``scroll down`` above is the page's
    ``scrollY`` (> 0) and above + viewport + below is the page height."""
    srv = new_server()
    srv.start()
    try:
        async with executor(allowed_origins=frozenset({srv.origin})) as ex:
            await run(ex.open(srv.origin + "/demo/deals.html"))
            f0 = await run(ex.frame())
            height = float(await ex.driver.page.evaluate("document.documentElement.scrollHeight"))
            e0 = extent(ex, f0)
            assert e0 is not None, "the extent of a settled page is known"
            assert e0["viewport_px"] == 800 and close(e0["above_px"], 0), e0
            assert close(e0["below_px"], height - 800) and e0["below_px"] >= 0.3 * 800, (e0, height)
            out = await execute(ex, inv("scroll", None, direction="down"))
            assert out.status == "executed", out
            f1 = ex.latest_frame
            y = float(await ex.driver.page.evaluate("window.scrollY"))
            e1 = extent(ex, f1)
            assert e1 is not None and y > 0 and close(e1["above_px"], y), (e1, y)
            assert close(e1["above_px"] + e1["below_px"] + e1["viewport_px"], height), (e1, height)
            assert extent(ex, f0) == e0, "the onset frame keeps its own extent"
    finally:
        srv.stop()


async def test_hostile_page_cannot_spoof_extent(server: str) -> None:  # noqa: F811  (the fixture)
    """A4a (adversarial): a page that redefines ``scrollY``, ``innerHeight`` and ``scrollingElement`` (and the root
    element's metrics) still gets its true extent: 0/2000 at scroll 0, 600/1400 after ``scroll down``."""
    async with executor() as ex:
        await run(ex.open(f"{server}/t0134_hostile_extent.html"))
        assert await ex.driver.page.evaluate("window.scrollY") == 4242, "premise: the page lies"
        f0 = await run(ex.frame())
        e0 = extent(ex, f0)
        assert e0 is not None and close(e0["above_px"], 0) and close(e0["below_px"], 2000), e0
        assert e0["viewport_px"] == 800, e0
        out = await execute(ex, inv("scroll", None, direction="down"))
        assert out.status == "executed", out
        e1 = extent(ex, ex.latest_frame)
        assert e1 is not None and close(e1["above_px"], 600) and close(e1["below_px"], 1400), e1


async def test_echo_through_the_real_executor() -> None:
    """A2 through Chromium: with ``echo="model"`` a click at (128, 390) on deals.html reads ``clicked at (128,
    390)`` and ``resolved`` holds the pixels ``to_pixels``: (164, 312) by hand."""
    cls = computer().BrowserToolExecutor
    assert "echo" in inspect.signature(cls.__init__).parameters, "T-0134 A1: no `echo` keyword yet"
    srv = new_server()
    srv.start()
    try:
        async with executor(allowed_origins=frozenset({srv.origin}), echo="model") as ex:
            await run(ex.open(srv.origin + "/demo/deals.html"))
            f0 = await run(ex.frame())
            out = await execute(ex, inv("click_at", f0, x=128, y=390))
            assert out.status == "executed", out
            assert out.action == "clicked at (128, 390)", out.action
            assert (out.resolved["x_px"], out.resolved["y_px"]) == (164, 312), out.resolved
    finally:
        srv.stop()
