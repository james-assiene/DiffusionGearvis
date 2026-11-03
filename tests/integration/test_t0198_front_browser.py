"""T-0198 criterion 8 in a real Chromium: the executor's fronting policy calls Playwright's ``Page.bring_to_front``
once per page key on the then-active page, and the next action and capture still succeed (docs/tasks/T-0198.md).

**Gate.** ``integration`` + ``browser``; runs only with ``ASTRA_BROWSER_TESTS=1`` (T-0111c's autouse gate,
imported). ``ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0198_front_browser.py -q -m browser``.

**The rig.** T-0111c's loopback fixture server (``popup.html``: the whole viewport is a ``target=_blank`` link to the
next page, page n in ``PALETTE[n]``). A real ``PlaywrightDriver(headless=True)`` is injected into
``BrowserToolExecutor(driver=..., headless=False)``, so the policy runs while Chromium stays headless.
``Page.bring_to_front`` is spied at the class (each call records the page's ``window.__n`` URL and then runs the
real method). The optional headed row runs only with ``DISPLAY`` and ``ASTRA_BROWSER_HEADED=1``.

Regime: Playwright 1.64.0, Chromium 156.0.8078.4 headless shell, 1280x800 at DSF 1, loopback, n = 1, no network.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import http.server
import os
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from playwright.async_api import Page

from local_astra.computer import executor as executor_mod
from local_astra.computer import runtime as runtime_mod
from local_astra.contracts.toolcall import ToolInvocation
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    CLOCK_MS,
    FIXTURES,
    PALETTE,
    _browser_layer,
    _Quiet,
    execute,
    geom,
    near,
    rgb_at,
    run,
)

pytestmark = [pytest.mark.integration, pytest.mark.browser]

_N = iter(range(1_000_000))


@pytest.fixture(scope="module")
def origin() -> Iterator[str]:
    """T-0111c's fixture directory on an ephemeral loopback port (as its ``server`` fixture)."""
    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Quiet, directory=str(FIXTURES))
    )
    t = threading.Thread(target=srv.serve_forever, name="t0198-fixture-server", daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(5)


def call(name: str, frame: Any = None, *, turn: str, **arguments: Any) -> ToolInvocation:
    return ToolInvocation(
        call_id=f"b{next(_N)}",
        turn_id=turn,
        name=name,
        arguments=arguments,
        source="draft",
        frame=geom(frame) if frame is not None else None,
    )


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    real = Page.bring_to_front

    async def wrapped(self: Page) -> None:
        seen.append(self.url)
        await real(self)

    monkeypatch.setattr(Page, "bring_to_front", wrapped)
    return seen


async def until_page(ex: Any, n: int) -> None:
    """Own frames (which never front) until the active page shows ``PALETTE[n]`` (a popup opens asynchronously)."""
    for _ in range(50):
        f = await run(ex.frame())
        if near(rgb_at(f, 640, 400), PALETTE[n]):
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"premise: popup page {n} never became the active page")


@contextlib.asynccontextmanager
async def rig(origin: str, *, headless: bool, driver_headless: bool = True) -> AsyncIterator[Any]:
    d = runtime_mod.PlaywrightDriver(
        viewport=(1280, 800),
        on_egress=lambda e: None,
        headless=driver_headless,
    )
    ex = executor_mod.BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS,
        viewport=(1280, 800),
        driver=d,
        headless=headless,
    )
    try:
        yield ex
    finally:
        with contextlib.suppress(Exception):
            await run(d.close(), 20)


async def test_two_turns_front_twice(origin: str, spy: list[str]) -> None:
    """C1/C8: two scrolls in t1 and one in t2: two real ``Page.bring_to_front`` calls, both on the open page; every
    action executed and the next frame captured."""
    async with rig(origin, headless=False) as ex:
        await run(ex.open(f"{origin}/scroll.html"))
        await run(ex.frame())
        for turn in ("t1", "t1", "t2"):
            out = await execute(ex, call("scroll", turn=turn, direction="down"))
            assert out.status == "executed", out
        assert len(spy) == 2 and all(u.endswith("/scroll.html") for u in spy), spy
        await run(ex.frame())


async def test_window_open_click_fronts_the_popup(origin: str, spy: list[str]) -> None:
    """C2/C8: a click on ``popup.html?n=0`` opens ``?n=1``: one fronting on the opener before the click, one on the
    popup; the after-frame and the next frame are the popup's colour; the next click executes."""
    async with rig(origin, headless=False) as ex:
        await run(ex.open(f"{origin}/popup.html?n=0"))
        f = await run(ex.frame())
        out = await execute(ex, call("click_at", f, turn="t1", x=500, y=500))
        assert out.status == "executed", out
        assert spy[:1] and spy[0].endswith("n=0"), spy
        # the popup is adopted by this click's settle or, if it opened later, by the next ensure(); either way it is
        # fronted once before the next dispatch, and frame() itself never fronts
        await until_page(ex, 1)
        out = await execute(ex, call("scroll", turn="t1", direction="down"))
        assert out.status == "executed", out
        assert [u.rsplit("=", 1)[-1] for u in spy] == ["0", "1"], (
            f"T-0198 C2/C8: one fronting per page, the popup's on the popup; {spy}"
        )
        f = await run(ex.frame())
        out = await execute(ex, call("click_at", f, turn="t1", x=500, y=500))
        assert out.status == "executed", out
        await until_page(ex, 2)
        out = await execute(ex, call("scroll", turn="t1", direction="down"))
        assert out.status == "executed", out
        assert [u.rsplit("=", 1)[-1] for u in spy] == ["0", "1", "2"], spy


async def test_headless_executor_never_fronts(origin: str, spy: list[str]) -> None:
    """C3: the same click sequence with ``headless=True``: zero calls. The headed run fronts (premise)."""
    async with rig(origin, headless=False) as ex:
        await run(ex.open(f"{origin}/popup.html?n=0"))
        f = await run(ex.frame())
        await execute(ex, call("click_at", f, turn="t1", x=500, y=500))
    assert spy, "premise: the headed run fronted"
    spy.clear()
    async with rig(origin, headless=True) as ex:
        await run(ex.open(f"{origin}/popup.html?n=0"))
        f = await run(ex.frame())
        out = await execute(ex, call("click_at", f, turn="t1", x=500, y=500))
        assert out.status == "executed", out
        await execute(ex, call("scroll", turn="t2", direction="down"))
    assert spy == [], spy


async def test_front_does_not_count_as_navigation(origin: str, spy: list[str]) -> None:
    """Implementation notes (T-0174 B1-1b): a real ``bring_to_front`` moves no navigation bookkeeping:
    ``navigated_since(mark)`` reads the same after it as before."""
    async with rig(origin, headless=False) as ex:
        await run(ex.open(f"{origin}/scroll.html"))
        d = ex.driver
        mark = d.navigation_mark()
        before = d.navigated_since(mark)
        out = await execute(ex, call("scroll", turn="t1", direction="down"))
        assert out.status == "executed" and len(spy) == 1, (out, spy)
        assert d.navigated_since(mark) == before, (before, d.navigated_since(mark))
        assert out.resolved.get("navigated") in (None, False), out.resolved


@pytest.mark.skipif(
    not (os.environ.get("DISPLAY") and os.environ.get("ASTRA_BROWSER_HEADED") == "1"),
    reason="headed row: needs DISPLAY and ASTRA_BROWSER_HEADED=1",
)
async def test_headed_chromium_fronts(origin: str, spy: list[str]) -> None:
    """C8 optional: a headed Chromium; one fronting per turn, actions still succeed."""
    async with rig(origin, headless=False, driver_headless=False) as ex:
        await run(ex.open(f"{origin}/scroll.html"))
        for turn in ("t1", "t2"):
            out = await execute(ex, call("scroll", turn=turn, direction="down"))
            assert out.status == "executed", out
    assert len(spy) == 2, spy
