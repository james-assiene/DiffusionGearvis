"""T-0188 group A end to end: Restart (``reset_page``) and web mode's Ctrl+T/Ctrl+L on a page that refuses to unload
land on the start page with one page open, in a real headless Chromium (docs/tasks/T-0188.md, criteria 1, 4, 5).

**Gate.** ``integration`` + ``browser``; runs only with ``ASTRA_BROWSER_TESTS=1`` (T-0111c's autouse gate,
imported). ``ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0188_beforeunload_browser.py -q -m browser``.

**The rig.** One loopback ``http.server`` on 127.0.0.1 at an ephemeral port serves ``t0188_beforeunload.html``
(``tests/fixtures/browser/``, MANIFEST.md: a ``beforeunload`` handler, armed unless ``?arm=0``) and two stand-in
start pages, ``/start.html`` (loopback mode: the first page ``open()`` is given, T-0162) and ``/web.html`` (web mode's
start page, ``<start origin>/web.html``). Loopback rows run a default ``BrowserToolExecutor`` with the server's origin
as the only allowed origin and ``end_when_closed=True`` (so a replacement that closed the last page before opening
the new one would end the browser, criterion 5). The web row runs it in web mode with the same origin as start origin
and an injected proxy resolver and connector that refuse everything (nothing can leave the box). The user gesture
Chromium needs before it asks about leaving is an executed ``type_at`` on ``#field``.

**The bug, reproduced** (the planner's probe, Playwright 1.64.0): the driver dismisses the beforeunload dialog, which
cancels the navigation, and ``page.goto`` raises ``Error: net::ERR_ABORTED``. Today ``reset_page`` lets that
``Error`` out and the browser stays on the old page; Ctrl+T is ``failed: Error during press_key``.

**Readings** (as the unit module's, ``tests/unit/test_t0188_reset_page.py``): ``reset_page()`` returns
``{replaced, reason}``; the real ``PlaywrightDriver`` gains the replacement (``replace_page``), which these rows reach
only through the executor. "Exactly one page" after a fallback covers every other page too (H-T188-3, provisional):
the row with a second page open checks both are closed.

Regime: Playwright 1.64.0, Chromium 156.0.8078.4 headless (versions.lock ``[browser]``), 1280x800 at DSF 1, one
browser per row, n = 1, no network.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import socket
import threading
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.conductor.tool_executor import to_pixels
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    execute,
    inv,
    run,
)

pytestmark = [pytest.mark.integration, pytest.mark.browser]

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "browser" / "t0188_beforeunload.html"
CLOCK_MS = 1_700_000_018_800
W, H = 1280, 800
#: ``#field``'s centre in CSS px (the fixture: x 100..499, y 100..139)
FIELD = (300, 120)
START_HTML = b"<!doctype html><title>T-0188 start</title><body style='background:#dfe'>T-0188 start page</body>"
WEB_HTML = b"<!doctype html><title>T-0188 web</title><body style='background:#edf'>T-0188 web start page</body>"
#: an off-loopback documentation address (RFC 5737) the fence must refuse and log
OFF_BOX = "198.51.100.7"


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        body = {
            "/start.html": START_HTML,
            "/web.html": WEB_HTML,
            "/t0188_beforeunload.html": FIXTURE.read_bytes(),
        }.get(path)
        if body is None:
            self.send_response(404)
            self.send_header("content-length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def origin() -> Iterator[str]:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, name="t0188-fixture-server", daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(5)


class DenyNet:
    """Web mode's proxy seams: every resolve fails and every connect is refused."""

    async def resolve(self, host: str, port: int, *a: Any, **kw: Any) -> list[Any]:
        raise socket.gaierror(socket.EAI_NONAME, "T-0188 rig: no network")

    async def connect(self, ip: str, port: int) -> Any:
        raise ConnectionRefusedError("T-0188 rig: no network")


@contextlib.asynccontextmanager
async def browser(
    origin: str, *, web: bool = False, user_data_dir: Path | None = None
) -> AsyncIterator[Any]:
    net = DenyNet()
    kw: dict[str, Any] = (
        {
            "web": True,
            "start_origin": origin,
            "proxy_resolve": net.resolve,
            "proxy_connect": net.connect,
        }
        if web
        else {}
    )
    if user_data_dir is not None:
        kw["user_data_dir"] = user_data_dir
    ex = executor_mod.BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS,
        viewport=(W, H),
        headless=True,
        allowed_origins=frozenset({origin}),
        end_when_closed=True,
        **kw,
    )
    try:
        yield ex
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ex.close(), 20)


def units(px: int, size: int) -> int:
    c = round(px * 999 / (size - 1))
    assert abs(to_pixels(c, size) - px) <= 1, (px, size, c)
    return c


async def gesture(ex: Any) -> None:
    """The user gesture (an executed ``type_at`` on ``#field``) Chromium needs before it asks about leaving."""
    f = await run(ex.frame())
    out = await execute(
        ex,
        inv(
            "type_at", f, x=units(FIELD[0], W), y=units(FIELD[1], H), text="doc", press_enter=False
        ),
    )
    assert out.status == "executed", f"premise: the gesture is executed; {out}"
    assert await ex.driver.page.evaluate("document.getElementById('field').value") == "doc", (
        "premise: typed"
    )


def outcome(result: object) -> dict[str, Any]:
    if isinstance(result, Mapping) and "replaced" in result and "reason" in result:
        return {"replaced": result["replaced"], "reason": result["reason"]}
    if hasattr(result, "replaced") and hasattr(result, "reason"):
        return {"replaced": result.replaced, "reason": result.reason}
    raise AssertionError(
        f"T-0188 C3: reset_page must return {{replaced, reason}}; it returned {type(result).__name__}"
    )


async def reset(ex: Any) -> Any:
    try:
        return await asyncio.wait_for(ex.reset_page(), 30)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(
            f"T-0188 C1: reset_page raised {type(exc).__name__}; the browser stays on {ex.driver.page.url}"
        ) from None


def open_pages(ex: Any) -> list[Any]:
    return [p for p in ex.driver.page.context.pages if not p.is_closed()]


# =====================================================================================================
# criterion 1: Restart replaces a page that refuses to unload
# =====================================================================================================


async def test_reset_replaces_a_page_that_refuses_to_unload(origin: str) -> None:
    """C1-C3, C5: after a gesture on the armed page, Restart lands on the start page in a fresh page, with exactly one
    page open, ``page_serial`` moved, the old page closed (its beforeunload not run), class ``beforeunload``; the
    browser has not ended (the new page opened first), and a frame of the fresh page is clickable (it is the active
    page, not re-adopted as a popup)."""
    start = f"{origin}/start.html"
    async with browser(origin) as ex:
        await run(ex.open(start))
        await run(ex.open(f"{origin}/t0188_beforeunload.html"))
        await gesture(ex)
        old, serial = ex.driver.page, ex.driver.page_serial
        # C5 / implementation notes: the new page opens before the old one closes, so a reader of the end facts
        # *during* the replacement (the session's frame source, the HUD) never sees zero pages; ``closed`` latches
        samples: list[tuple[int, bool]] = []
        done = asyncio.Event()

        async def sample() -> None:
            while not done.is_set():
                samples.append((ex.driver.page_count, ex.closed))
                await asyncio.sleep(0)

        sampler = asyncio.create_task(sample())
        try:
            got = outcome(await reset(ex))
        finally:
            done.set()
            await sampler
        assert len(samples) > 3, (
            f"premise: the sampler ran during the replacement ({len(samples)} samples)"
        )
        assert min(n for n, _ in samples) >= 1 and not any(c for _, c in samples), (
            f"T-0188 C5: during the replacement the browser had 0 pages or ended: "
            f"min page_count {min(n for n, _ in samples)}, closed seen {any(c for _, c in samples)}"
        )
        assert got == {"replaced": True, "reason": "beforeunload"}, f"T-0188 C2/C3: {got}"
        assert ex.driver.page.url == start, (
            f"T-0188 C1: Restart left the browser on {ex.driver.page.url}"
        )
        assert old.is_closed(), "T-0188 C1: the page that refused to unload is still open"
        assert len(open_pages(ex)) == 1 and ex.driver.page_count == 1, (
            f"T-0188 C1: {len(open_pages(ex))} pages open"
        )
        assert ex.driver.page_serial > serial, (
            "T-0188 C1: page_serial must move with the active page"
        )
        assert ex.closed is False, "T-0188 C5: replacing the page ended the browser"
        f = await run(ex.frame())
        out = await execute(ex, inv("click_at", f, x=500, y=500))
        assert out.status == "executed", (
            f"T-0188 C1: the fresh page's own frame is not clickable: {out}"
        )
        assert ex.driver.page.url == start


async def test_reset_closes_every_other_page(origin: str) -> None:
    """C1 / H-T188-3 (provisional): the armed page is a popup in front of the start page; after Restart exactly one
    page is open, the start page, and both earlier pages are closed."""
    start = f"{origin}/start.html"
    async with browser(origin) as ex:
        await run(ex.open(start))
        opener = ex.driver.page
        popup = await opener.context.new_page()
        await run(popup.goto(f"{origin}/t0188_beforeunload.html", wait_until="domcontentloaded"))
        await gesture(ex)  # its frame() adopts the popup as the active page
        assert ex.driver.page is popup, "premise: the armed popup is the active page"
        assert len(open_pages(ex)) == 2, "premise: two pages open"
        got = outcome(await reset(ex))
        assert got == {"replaced": True, "reason": "beforeunload"}, got
        assert ex.driver.page.url == start
        assert opener.is_closed() and popup.is_closed(), (
            f"T-0188 C1/H-T188-3: closed opener={opener.is_closed()} popup={popup.is_closed()}"
        )
        assert len(open_pages(ex)) == 1, f"T-0188 C1: {len(open_pages(ex))} pages open"


async def test_fresh_page_keeps_dialogs_and_the_fence(origin: str) -> None:
    """C5 / implementation notes (``_adopt``): the fresh page still has its dialogs dismissed and reported (an
    ``alert`` returns, an event is recorded), and an off-loopback fetch from it is refused and logged."""
    start = f"{origin}/start.html"
    async with browser(origin) as ex:
        await run(ex.open(start))
        await run(ex.open(f"{origin}/t0188_beforeunload.html"))
        await gesture(ex)
        assert outcome(await reset(ex))["replaced"] is True
        page = ex.driver.page
        ex.driver.take_events()
        try:
            await asyncio.wait_for(page.evaluate("() => { alert('t0188'); return 1 }"), 10)
        except TimeoutError:
            raise AssertionError(
                "T-0188 C5: an alert on the fresh page was never dismissed"
            ) from None
        kinds = [(e.get("kind"), e.get("type")) for e in ex.driver.take_events()]
        assert ("dialog", "alert") in kinds, (
            f"T-0188 C5: the fresh page's dialog was not reported: {kinds}"
        )
        before = len(ex.egress_log)
        await page.evaluate(
            f"() => fetch('http://{OFF_BOX}/t0188-secret').then(() => 'ok', () => 'blocked')"
        )
        hosts = [r.get("host") for r in ex.egress_log[before:]]
        assert OFF_BOX in hosts, f"T-0188 C5: the fence on the fresh page logged {hosts}"


#: the four permissions the driver sets to ``denied`` (runtime docstring, "geolocation, notifications, camera and
#: microphone"), as the page itself sees them
PERMISSIONS = ("geolocation", "notifications", "camera", "microphone")
QUERY_PERMISSIONS = (
    "async (names) => Promise.all(names.map((name) => navigator.permissions.query({name})"
    ".then((s) => s.state, (e) => 'error: ' + e.name)))"
)


async def permission_states(ex: Any) -> dict[str, str]:
    states = await run(ex.driver.page.evaluate(QUERY_PERMISSIONS, list(PERMISSIONS)))
    return dict(zip(PERMISSIONS, states, strict=True))


async def test_permissions_stay_denied_after_a_replacement_in_a_persistent_context(
    origin: str, tmp_path: Path
) -> None:
    """A-R1: in a persistent context (``user_data_dir``) the permission overrides live on a page's CDP session, which
    closing the old page detaches. After Restart replaces a page that refuses to unload, the fresh page still sees
    geolocation, notifications, camera and microphone ``denied``, and so does the next document it navigates to."""
    denied = dict.fromkeys(PERMISSIONS, "denied")
    start = f"{origin}/start.html"
    async with browser(origin, user_data_dir=tmp_path / "profile") as ex:
        await run(ex.open(start))
        await run(ex.open(f"{origin}/t0188_beforeunload.html"))
        assert await permission_states(ex) == denied, (
            f"premise: denied before the replacement; {await permission_states(ex)}"
        )
        await gesture(ex)
        old = ex.driver.page
        got = outcome(await reset(ex))
        assert got == {"replaced": True, "reason": "beforeunload"}, f"premise: replaced; {got}"
        assert old.is_closed() and ex.driver.page.url == start, "premise: the fresh page is active"
        after = await permission_states(ex)
        assert after == denied, (
            f"T-0188 A-R1: after the replacement in a persistent context the fresh page sees {after}"
        )
        await run(ex.open(f"{origin}/t0188_beforeunload.html?arm=0"))
        assert ex.driver.page.url.endswith("arm=0"), "premise: navigated further"
        later = await permission_states(ex)
        assert later == denied, f"T-0188 A-R1: after a further navigation the page sees {later}"


async def test_failed_replacement_leaves_everything_as_it_was(origin: str) -> None:
    """A-R3: the real driver's ``replace_page`` with a URL the fence refuses (an off-loopback documentation address)
    raises, and nothing moved: the same page is active and open at the same URL, ``page_serial`` and ``page_count``
    are unchanged, no page is waiting to be adopted as new, and a later ``ensure``/``settle`` does not switch pages
    (the half-made page is neither active nor left open for adoption)."""
    page_url = f"{origin}/t0188_beforeunload.html?arm=0"
    refused = f"http://{OFF_BOX}/t0188-replace"
    async with browser(origin) as ex:
        await run(ex.open(f"{origin}/start.html"))
        await run(ex.open(page_url))
        d = ex.driver
        page, serial, count = d.page, d.page_serial, d.page_count
        before_pages = open_pages(ex)
        assert count == 1 and before_pages == [page], "premise: one page open"
        egress_before = len(ex.egress_log)
        raised: BaseException | None = None
        try:
            await run(d.replace_page(refused))
        except Exception as exc:
            raised = exc
        assert raised is not None, "T-0188 A-R3: replace_page to a refused URL did not raise"
        assert OFF_BOX in [r.get("host") for r in ex.egress_log[egress_before:]], (
            "premise: the fence refused (and logged) the replacement's navigation"
        )
        await asyncio.sleep(0.3)  # let a stray page's events land
        assert d.page is page and not page.is_closed() and page.url == page_url, (
            f"T-0188 A-R3: the active page changed (same={d.page is page}, closed={page.is_closed()}, url={d.page.url})"
        )
        assert d.page_serial == serial, f"T-0188 A-R3: page_serial {serial} -> {d.page_serial}"
        assert d.page_count == count and open_pages(ex) == before_pages, (
            f"T-0188 A-R3: page_count {count} -> {d.page_count}; open {[p.url for p in open_pages(ex)]}"
        )
        assert not [p for p in d._new_pages if not p.is_closed()], (
            f"T-0188 A-R3: a page is tracked as new: {[p.url for p in d._new_pages]}"
        )
        await run(d.ensure())
        await run(d.settle())
        assert d.page is page and d.page_serial == serial and page.url == page_url, (
            f"T-0188 A-R3: ensure/settle switched pages (same={d.page is page}, serial {serial} -> {d.page_serial})"
        )
        assert d.page_count == 1, f"T-0188 A-R3: {d.page_count} pages after ensure/settle"


# =====================================================================================================
# criterion 4: Ctrl+T / Ctrl+L in web mode
# =====================================================================================================


@pytest.mark.parametrize("combo", ["Ctrl+T", "Ctrl+L"])
async def test_start_page_combo_on_a_page_that_refuses_to_unload(origin: str, combo: str) -> None:
    """C4 / H-T188-2 (provisional): in web mode, Ctrl+T/Ctrl+L on the armed page ends ``executed`` with ``opened the
    start page`` on a fresh page at ``<start origin>/web.html``, one page open, the old one closed, the dismissed
    beforeunload reported."""
    web = f"{origin}/web.html"
    async with browser(origin, web=True) as ex:
        await run(ex.open(web))
        await run(ex.open(f"{origin}/t0188_beforeunload.html"))
        await gesture(ex)
        old = ex.driver.page
        await run(ex.frame())
        out = await execute(ex, inv("press_key", None, combo=combo))
        assert out.status == "executed", (
            f"T-0188 C4: {combo} on a page that refuses to unload: {out.status} {out.reason!r}"
        )
        assert out.action == "opened the start page", out.action
        assert ex.driver.page.url == web, (
            f"T-0188 C4: {combo} left the browser on {ex.driver.page.url}"
        )
        assert old.is_closed() and len(open_pages(ex)) == 1, (
            f"T-0188 C4: old closed={old.is_closed()}, {len(open_pages(ex))} pages open"
        )
        assert {"dialog": "beforeunload"} in out.resolved.get("dismissed", []), out.resolved


# =====================================================================================================
# criterion 5: the control (negative; green before the implementation)
# =====================================================================================================


async def test_reset_without_a_handler_takes_the_plain_path(origin: str) -> None:
    """C5 / H-T188-1: the same page without the handler (``?arm=0``): Restart navigates the *same* page to the start
    page; the serial does not move, nothing is closed, and a returned outcome says nothing was replaced."""
    start = f"{origin}/start.html"
    async with browser(origin) as ex:
        await run(ex.open(start))
        await run(ex.open(f"{origin}/t0188_beforeunload.html?arm=0"))
        await gesture(ex)
        assert await ex.driver.page.evaluate("window.__armed") is False, (
            "premise: the control page is unarmed"
        )
        page, serial = ex.driver.page, ex.driver.page_serial
        result = await reset(ex)
        assert ex.driver.page is page and not page.is_closed(), (
            "T-0188 C5: a page that unloads was replaced"
        )
        assert ex.driver.page_serial == serial
        assert ex.driver.page.url == start
        assert len(open_pages(ex)) == 1
        if result is not None:
            assert outcome(result) == {"replaced": False, "reason": None}, result
