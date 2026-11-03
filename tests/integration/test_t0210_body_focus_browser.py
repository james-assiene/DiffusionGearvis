"""T-0210 A6 end to end: a click on a canvas that cannot take focus leaves the focus on ``body``, and a ``type_at`` there
is refused in web mode with nothing reaching the page (docs/tasks/T-0210.md, A6 table; "Tests to write").

**The rig.** ``StartPageServer(static_dir=tests/fixtures/t0210)`` on loopback is the start origin of a web-mode
``BrowserToolExecutor`` (real ``PlaywrightDriver``, real egress proxy, deny-all resolver and connector: the fixture is
loopback, so neither may be called). Each row opens ``/canvas.html`` afresh with the text input focused
(``tests/fixtures/t0210/MANIFEST.md``). Coordinates are computed test-side from element boxes (targeting only).
``window.__keys`` lists every ``keydown`` that reached the top ``document``.

**Gate.** ``integration`` + ``browser``; runs only with ``ASTRA_BROWSER_TESTS=1`` (``make browser`` once the orchestrator
adds the module at merge).

**State.** A6a is the gap: at BASE the ``type_at`` executes and every character lands in ``__keys``; it carried
``pending(task="T-0210")`` until activation for ``/implement T-0210`` (A, tip 3b12ae5c). A6b-e hold at BASE and are active (a pending marker would XPASS); A6d also pins the premise
that a designMode body reads ``contenteditable: "true"`` through the driver's frame descent, which is what keeps it
typed after A.

**Regime.** Playwright 1.64.0, Chromium headless 1280x800, DSF 1, sandbox on, web mode on loopback, deny-all resolver
and connector, a fresh server and executor per row, n = 1.
"""

from __future__ import annotations

import contextlib
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.conductor.tool_executor import to_pixels
from local_astra.contracts.toolcall import ToolOutcome
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
    execute,
    executor,
    inv,
    run,
)
from tests.unit.test_start_pages import new_server

pytestmark = [
    pytest.mark.integration,
    *([pytest.mark.browser] if _browser_marker_registered() else []),
]

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "t0210"
W, H = 1280, 800
TEXT = "Crate"
SEED = "seed-value"
FOCUS_UNKNOWN = "could not check the field before typing; nothing was typed"


class DenyNet:
    """The web-mode proxy's seams: every resolve and connect raises, each recorded (T-0205 A7's shape)."""

    def __init__(self) -> None:
        self.resolves: list[str] = []
        self.connects: list[tuple[str, int]] = []

    async def resolve(self, host: str, port: int, *a: Any, **kw: Any) -> list[Any]:
        self.resolves.append(host)
        raise socket.gaierror(socket.EAI_NONAME, "T-0210 rig: no network")

    async def connect(self, ip: str, port: int) -> Any:
        self.connects.append((ip, port))
        raise ConnectionRefusedError("T-0210 rig: no network")


@contextlib.asynccontextmanager
async def opened() -> AsyncIterator[Any]:
    """A fresh fixture server and web-mode executor at ``/canvas.html``, the input focused, the iframe in designMode,
    ``__keys`` empty; on exit, zero resolver and connector calls."""
    srv = new_server(FIXTURES)
    srv.start()
    net = DenyNet()
    try:
        async with executor(
            web=True,
            start_origin=srv.origin,
            allowed_origins=frozenset({srv.origin}),
            proxy_resolve=net.resolve,
            proxy_connect=net.connect,
        ) as ex:
            await run(ex.open(srv.origin + "/canvas.html"))
            await run(ex.frame())
            page = ex.driver.page
            await page.wait_for_function("window.__richReady === true", timeout=10_000)
            assert await page.evaluate("document.activeElement.id") == "field", (
                "premise: each case starts with the input focused"
            )
            assert await page.evaluate("window.__keys") == [], "premise: no key yet"
            yield ex
    finally:
        srv.stop()
    assert (net.resolves, net.connects) == ([], []), f"the proxy seams were called: {net.__dict__}"


def unit(lo: float, hi: float, size: int) -> int:
    """The 0-999 coordinate whose pixel is nearest the centre of ``[lo, hi)`` and inside it."""
    centre = (lo + hi) / 2
    cands = [c for c in range(1000) if lo + 1 <= to_pixels(c, size) <= hi - 2]
    assert cands, (lo, hi, size)
    return min(cands, key=lambda c: abs(to_pixels(c, size) - centre))


async def at(ex: Any, selector: str) -> tuple[int, int]:
    box = await ex.driver.page.locator(selector).bounding_box(timeout=10_000)
    assert box is not None, selector
    return (
        unit(box["x"], box["x"] + box["width"], W),
        unit(box["y"], box["y"] + box["height"], H),
    )


async def type_at(ex: Any, selector: str, text: str = TEXT) -> ToolOutcome:
    x, y = await at(ex, selector)
    f = await run(ex.frame())
    return await execute(ex, inv("type_at", f, x=x, y=y, text=text, press_enter=False))


async def keys(ex: Any) -> list[str]:
    return list(await ex.driver.page.evaluate("window.__keys"))


async def field_value(ex: Any) -> str:
    return str(await ex.driver.page.locator("#field").input_value(timeout=10_000))


# =====================================================================================================
# A6
# =====================================================================================================


async def test_canvas_click_type_refused_no_keys() -> None:
    """A6a, adversarial (the named failure: typing over a WebGL viewport fires the page's single-letter shortcuts):
    ``type_at`` on the canvas without ``tabindex``: refused with ``FOCUS_UNKNOWN_REASON`` and ``focus_check ==
    "body"``, ``typed: 0``; ``__keys == []``; the input keeps its value. Premise: the click left the focus on ``body``."""
    async with opened() as ex:
        out = await type_at(ex, "#plain")
        assert await ex.driver.page.evaluate("document.activeElement.tagName") == "BODY", (
            "premise: a click on a canvas that cannot take focus leaves body focused"
        )
        got = await keys(ex)
        assert got == [], f"T-0210 A6a: keys reached the page's document listener: {got} ({out})"
        assert (out.status, out.reason) == ("failed", FOCUS_UNKNOWN), out
        assert out.action == f"failed type_at: {FOCUS_UNKNOWN}", out.action
        assert out.resolved.get("focus_check") == "body", out.resolved
        assert out.resolved.get("typed") == 0 and "text" not in out.resolved, out.resolved
        assert await field_value(ex) == SEED


async def test_plain_div_type_refused_no_keys() -> None:
    """A1/A6a variant (implementation notes: "a click on a canvas, a bare background or a non-focusable ``div`` all
    leave ``body`` and are refused alike"): ``type_at`` on the plain ``div`` is refused with ``"body"``, no keys."""
    async with opened() as ex:
        out = await type_at(ex, "#div")
        assert await keys(ex) == [], out
        assert (out.status, out.resolved.get("focus_check")) == ("failed", "body"), out
        assert await field_value(ex) == SEED


async def test_tabindex_canvas_unreadable() -> None:
    """A6b: the ``tabindex`` canvas takes the focus and reads ``unreadable``: refused as BASE, no keys. Holds at
    BASE."""
    async with opened() as ex:
        out = await type_at(ex, "#tab")
        assert await ex.driver.page.evaluate("document.activeElement.id") == "tab", "premise"
        assert (out.status, out.reason) == ("failed", executor_mod.FOCUS_UNKNOWN_REASON), out
        assert out.resolved.get("focus_check") == "unreadable", out.resolved
        assert await keys(ex) == []
        assert await field_value(ex) == SEED


async def test_input_typed() -> None:
    """A6c: ``type_at`` on the input executes; its text replaces the seed value (T-0189). Holds at BASE."""
    async with opened() as ex:
        out = await type_at(ex, "#field")
        assert out.status == "executed", out
        assert out.resolved.get("replaced") is True, out.resolved
        assert await field_value(ex) == TEXT


async def test_designmode_iframe_typed() -> None:
    """A6d (the guard too wide: a rich-text iframe body is a field): ``type_at`` in the designMode iframe executes and
    the text lands in the iframe's document; the driver's focus read (frame descent) reports ``body`` with
    ``contenteditable: "true"``, the A2 exemption. Holds at BASE."""
    async with opened() as ex:
        out = await type_at(ex, "#rich")
        assert out.status == "executed", out
        attrs = await run(ex.driver.focused_field())
        assert attrs is not None and (attrs.get("tag"), attrs.get("contenteditable")) == (
            "body",
            "true",
        ), f"premise (A2): a designMode body reads as contenteditable through the descent: {attrs}"
        rich = ex.driver.page.frame_locator("#rich").locator("body")
        assert TEXT in await rich.inner_text(timeout=10_000)
        assert await field_value(ex) == SEED


async def test_press_key_reaches_document() -> None:
    """A6e (A4, H-T210-2): ``click_at`` on the canvas executes (A5), then ``press_key Delete`` executes and the page's
    document listener sees exactly ``["Delete"]``. Holds at BASE."""
    async with opened() as ex:
        x, y = await at(ex, "#plain")
        f = await run(ex.frame())
        out = await execute(ex, inv("click_at", f, x=x, y=y))
        assert out.status == "executed", out
        assert await ex.driver.page.evaluate("document.activeElement.tagName") == "BODY", "premise"
        assert await keys(ex) == []
        await run(ex.frame())
        out = await execute(ex, inv("press_key", None, combo="Delete"))
        assert (out.status, out.action) == ("executed", "pressed Delete"), out
        assert await keys(ex) == ["Delete"]
