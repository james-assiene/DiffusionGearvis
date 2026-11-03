"""T-0191 group A, A6 (browser): ``press_key Ctrl+L`` then ``type_at`` on the start page's address box leaves an
arbitrary website, in a real headless Chromium in web mode (docs/tasks/T-0191.md).

**Gate.** ``integration`` + ``browser``; runs only with ``ASTRA_BROWSER_TESTS=1`` (T-0111c's autouse gate, imported).
Web mode needs Chromium's sandbox (T-0160 H0). Not in a ``make`` target (T-0191 owns no ``Makefile`` line); run by
path::

    ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0191_navigate_browser.py -q -m browser

**The rig** (T-0160 B-7 / T-0174 B2-3, no network). The real ``StartPageServer`` on 127.0.0.1 (the real
``/web.html``: ``#address``, Go, its ``target()`` rules); the real ``BrowserToolExecutor(web=True,
start_origin=<pages origin>, allowed_origins={origin})`` with the egress proxy's resolver and connector injected:
``shop.example`` and ``nav.example`` resolve to public-looking addresses and every connect is piped to one loopback TLS
server (a self-signed certificate for both names, made per run in the pytest temp dir) that serves
``tests/fixtures/t0191/`` by ``Host`` (``MANIFEST.md`` there); anything else fails to resolve. ``Browser.new_context``
gets ``ignore_https_errors=True`` (T-0160's one test-only relaxation, imported from the T-0174 module). Calls go
straight to the executor (no session): the criterion is about what the executor does with the two calls S teaches.

**Warm-up** (T-0174 review F1): before the measured calls the harness loads ``https://nav.example/`` once, so the
measured navigation needs no first resolve, connect or handshake and ``navigated`` is not left absent by a pending
request under load. The rows count the fixture server's hits *after* the warm-up.

**Rows.** ``test_ctrl_l_then_address_navigates`` (A6's main path); ``test_ctrl_l_from_beforeunload_page`` (the shop
refuses to unload, after a gesture typed into its own search box; T-0188 A); ``test_address_with_scheme``
(``"https://nav.example"``); the adversarial premise ``test_typing_into_site_search_does_not_leave`` (the reported
bug: the address typed into the shop's own autofocused search box searches the shop and never reaches nav.example);
``test_no_web_ctrl_l_refused`` (outside web mode ``Ctrl+L`` is refused ``not available in the browser`` and nothing
moves; this is why v3 is web mode only).

**Pending or active.** These rows exercise the executor and ``web.html`` as they are at BASE (f1caef6a): T-0191 does
not change either (A7 freezes them). So every row holds at BASE and is active; a pending marker here would be a
strict XPASS. They are A6's evidence that the sentence S teaches works, not a test of S.

Regime: Playwright 1.64.0, Chromium headless at 1280x800, DSF 1, sandbox on (H0), web mode through the egress proxy
with injected resolver/connector, ``StartPageServer`` and the fixture TLS server on 127.0.0.1 at ephemeral ports,
n = 1 per row, one browser per row.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import http.server
import socket
import ssl
import subprocess
import threading
import time
import urllib.parse
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
    _Quiet,
    execute,
    inv,
    run,
)
from tests.integration.test_computer_live_session import center
from tests.integration.test_t0174_page_web_browser import relaxed_tls  # noqa: F401  (fixture)

pytestmark = [pytest.mark.integration] + (
    [pytest.mark.browser] if _browser_marker_registered() else []
)

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "t0191"
W, H = 1280, 800
SHOP = "shop.example"
NAV = "nav.example"
IPS = {SHOP: "93.184.215.31", NAV: "93.184.215.32"}
SHOP_URL = f"https://{SHOP}/"
SHOP_UNLOAD_URL = f"https://{SHOP}/unload.html"
NAV_URL = f"https://{NAV}/"
#: (host, path) -> fixture file (MANIFEST.md)
ROUTES = {
    (SHOP, "/"): "shop.html",
    (SHOP, "/unload.html"): "shop_unload.html",
    (SHOP, "/search.html"): "search.html",
    (NAV, "/"): "nav.html",
}
ADDRESS = "document.getElementById('address')"
SHOP_BOX = "document.getElementById('q')"
#: executor.py's action string for web mode's Ctrl+T/Ctrl+L, typed from the task ("Confirmed in source")
START_PAGE_ACTION = "opened the start page"
NOT_AVAILABLE = "not available in the browser"


# =====================================================================================================
# the rig
# =====================================================================================================


class _ShopTLS(http.server.BaseHTTPRequestHandler):
    """``ROUTES`` by ``Host``; 404 for anything else. HTTP/1.1 with ``Content-Length`` (keep-alive), so the warm-up's
    tunnel can carry the measured navigation."""

    protocol_version = "HTTP/1.1"
    hits: list[tuple[str, str]]

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        u = urllib.parse.urlsplit(self.path)
        self.hits.append((host, u.path))
        name = ROUTES.get((host, u.path))
        if name is None:
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, (FIX / name).read_bytes(), "text/html; charset=utf-8")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class _QuietTLSServer(http.server.ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        pass


class Site:
    def __init__(self, tls_port: int, hits: list[tuple[str, str]], start: str) -> None:
        self.tls_port, self.hits, self.start = tls_port, hits, start


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Site]:
    d = tmp_path_factory.mktemp("t0191-tls")
    key, crt = d / "key.pem", d / "crt.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(crt),
         "-days", "2", "-subj", f"/CN={SHOP}", "-addext", f"subjectAltName=DNS:{SHOP},DNS:{NAV}"],
        check=True, capture_output=True,
    )  # fmt: skip
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    hits: list[tuple[str, str]] = []
    handler = type("_Bound", (_ShopTLS,), {"hits": hits})
    srv = _QuietTLSServer(("127.0.0.1", 0), handler)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True, do_handshake_on_connect=False)
    t = threading.Thread(target=srv.serve_forever, name="t0191-tls", daemon=True)
    t.start()
    from local_astra.pages.server import StartPageServer

    pages = StartPageServer()
    pages.start()
    try:
        yield Site(srv.server_address[1], hits, pages.origin)
    finally:
        pages.stop()
        srv.shutdown()
        srv.server_close()


class Net:
    """The proxy's injected seams: the two fixture hosts only, piped to the loopback TLS server."""

    def __init__(self, tls_port: int) -> None:
        self.tls_port = tls_port
        self.resolves: list[str] = []
        self.connects: list[tuple[str, int]] = []

    async def resolve(self, host: str, port: int, *a: Any, **kw: Any) -> list[Any]:
        self.resolves.append(host)
        ip = IPS.get(host.lower().rstrip("."))
        if ip is None:
            raise socket.gaierror(socket.EAI_NONAME, "not a fixture host")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    async def connect(
        self, ip: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        self.connects.append((ip, port))
        return await asyncio.open_connection("127.0.0.1", self.tls_port)


@contextlib.asynccontextmanager
async def web_executor(site: Site) -> AsyncIterator[tuple[Any, Net]]:
    """A web-mode executor opened on ``<origin>/web.html``."""
    from local_astra.computer.executor import BrowserToolExecutor

    net = Net(site.tls_port)
    ex = BrowserToolExecutor(
        now_ms=lambda: 0,
        viewport=(W, H),
        headless=True,
        web=True,
        start_origin=site.start,
        allowed_origins=frozenset({site.start}),
        proxy_resolve=net.resolve,
        proxy_connect=net.connect,
    )
    try:
        await run(ex.open(site.start + "/web.html"))
        yield ex, net
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ex.close(), 30)


async def goto(ex: Any, url: str) -> None:
    """Harness navigation (through the proxy, like any page load)."""
    await run(ex.frame())
    await ex.driver.page.goto(url, wait_until="load", timeout=15_000)
    assert ex.driver.page.url == url, f"premise: the harness reached {url}; at {ex.driver.page.url}"


async def warm_then_shop(ex: Any, site: Site, shop_url: str) -> int:
    """Warm the nav.example path (review F1), then land on the shop; returns the hit count to measure from."""
    await goto(ex, NAV_URL)
    assert (NAV, "/") in site.hits, "premise: the warm-up reached nav.example"
    await goto(ex, shop_url)
    focused = await ex.driver.page.evaluate("document.activeElement && document.activeElement.id")
    assert focused == "q", f"premise: the shop's own search box has focus (autofocus); {focused!r}"
    return len(site.hits)


async def wait_url(ex: Any, url: str, limit_s: float = 10.0) -> None:
    deadline = time.monotonic() + limit_s
    while ex.driver.page.url != url and time.monotonic() < deadline:
        await asyncio.sleep(0.05)


def open_pages(ex: Any) -> list[Any]:
    return [p for p in ex.driver.page.context.pages if not p.is_closed()]


async def ctrl_l(ex: Any, site: Site) -> Any:
    await run(ex.frame())
    out = await execute(ex, inv("press_key", None, combo="Ctrl+L"))
    assert out.status == "executed", f"T-0191 A6: Ctrl+L in web mode: {out.status} {out.reason!r}"
    assert out.action == START_PAGE_ACTION, f"T-0191 A6: Ctrl+L's action is {out.action!r}"
    assert ex.driver.page.url == site.start + "/web.html", (
        f"T-0191 A6: after Ctrl+L the active page is {ex.driver.page.url}"
    )
    assert len(open_pages(ex)) == 1, f"T-0191 A6: {len(open_pages(ex))} pages open (no new tab)"
    return out


async def type_address(ex: Any, site: Site, text: str, mark: int) -> Any:
    """``type_at`` on ``#address`` with ``text`` and Enter; the page lands on nav.example (the fixture server saw the
    request after ``mark``) and ``navigated`` is ``True``."""
    f = await run(ex.frame())
    focused = await ex.driver.page.evaluate("document.activeElement && document.activeElement.id")
    assert focused == "address", f"premise: web.html's address box has focus; {focused!r}"
    x, y = await center(ex, ADDRESS)
    out = await execute(ex, inv("type_at", f, x=x, y=y, text=text, press_enter=True))
    assert out.status == "executed", f"T-0191 A6: type_at on #address: {out.status} {out.reason!r}"
    await wait_url(ex, NAV_URL)
    assert ex.driver.page.url == NAV_URL, f"T-0191 A6: the address led to {ex.driver.page.url}"
    assert (NAV, "/") in site.hits[mark:], (
        f"T-0191 A6: the nav.example server saw no request after the warm-up; {site.hits[mark:]}"
    )
    assert out.resolved.get("navigated") is True, f"T-0191 A6: resolved {out.resolved}"
    return out


# =====================================================================================================
# A6
# =====================================================================================================


@pytest.mark.usefixtures("relaxed_tls")
async def test_ctrl_l_then_address_navigates(site: Site) -> None:
    """A6: on ``https://shop.example/`` (its own search box focused), ``press_key Ctrl+L`` is ``executed`` with
    ``"opened the start page"`` and the active page is ``<start origin>/web.html``; then ``type_at`` on ``#address``
    with ``"nav.example"`` and Enter is ``executed``, nav.example's server sees the request, the active URL is
    ``https://nav.example/`` and ``resolved["navigated"] is True``."""
    async with web_executor(site) as (ex, net):
        mark = await warm_then_shop(ex, site, SHOP_URL)
        await ctrl_l(ex, site)
        await type_address(ex, site, NAV, mark)
        assert (IPS[NAV], 443) in net.connects, net.connects


@pytest.mark.usefixtures("relaxed_tls")
async def test_ctrl_l_from_beforeunload_page(site: Site) -> None:
    """A6 (T-0188 A): the same from a shop page with a ``beforeunload`` handler, armed by a gesture (an executed
    ``type_at`` into the shop's own search box, without Enter): Ctrl+L still lands on ``/web.html`` with one page
    open, and the address then navigates."""
    async with web_executor(site) as (ex, _net):
        mark = await warm_then_shop(ex, site, SHOP_UNLOAD_URL)
        assert await ex.driver.page.evaluate("window.__armed") is True, (
            "premise: the handler is armed"
        )
        f = await run(ex.frame())
        x, y = await center(ex, SHOP_BOX)
        out = await execute(ex, inv("type_at", f, x=x, y=y, text="photo editor", press_enter=False))
        assert out.status == "executed", f"premise: the gesture is executed; {out}"
        old = ex.driver.page
        pressed = await ctrl_l(ex, site)
        assert {"dialog": "beforeunload"} in pressed.resolved.get("dismissed", []), (
            f"premise: the page asked about leaving and the dialog was dismissed (T-0188 A); {pressed.resolved}"
        )
        assert old.is_closed(), "premise (T-0188 A): the page that refused to unload was replaced"
        await type_address(ex, site, NAV, mark)


@pytest.mark.usefixtures("relaxed_tls")
async def test_address_with_scheme(site: Site) -> None:
    """A6: ``"https://nav.example"`` typed into the address box reaches the same page as the bare host."""
    async with web_executor(site) as (ex, _net):
        mark = await warm_then_shop(ex, site, SHOP_URL)
        await ctrl_l(ex, site)
        await type_address(ex, site, f"https://{NAV}", mark)


@pytest.mark.usefixtures("relaxed_tls")
async def test_typing_into_site_search_does_not_leave(site: Site) -> None:
    """A6 adversarial premise (the reported bug, reproduced): ``type_at`` on the shop's own search box with
    ``"nav.example"`` and Enter searches the shop (``/search.html?q=nav.example`` on shop.example) and never reaches
    nav.example. This is the behaviour v3's sentence exists to stop; if it ever left the site the guidance would not
    be needed."""
    async with web_executor(site) as (ex, _net):
        mark = await warm_then_shop(ex, site, SHOP_URL)
        f = await run(ex.frame())
        x, y = await center(ex, SHOP_BOX)
        out = await execute(ex, inv("type_at", f, x=x, y=y, text=NAV, press_enter=True))
        assert out.status == "executed", out
        want = f"https://{SHOP}/search.html?q={NAV}"
        await wait_url(ex, want)
        assert ex.driver.page.url == want, f"premise: the shop searched; at {ex.driver.page.url}"
        assert await ex.driver.page.evaluate("window.__query") == NAV
        assert (NAV, "/") not in site.hits[mark:], (
            f"the shop's search box reached nav.example: {site.hits[mark:]}"
        )
        assert urllib.parse.urlsplit(ex.driver.page.url).hostname == SHOP


# =====================================================================================================
# outside web mode: Ctrl+L is refused (why v3 is web mode only)
# =====================================================================================================


@pytest.fixture(scope="module")
def loopback_origin() -> Iterator[str]:
    handler = functools.partial(_Quiet, directory=str(FIX))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, name="t0191-loopback", daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(5)


async def test_no_web_ctrl_l_refused(loopback_origin: str) -> None:
    """A4's premise: outside web mode (live ``--no-web``) ``press_key Ctrl+L`` is ``refused`` with ``"not available in
    the browser"``, and the same document is still showing on the same URL with one page open."""
    from local_astra.computer.executor import BrowserToolExecutor

    ex = BrowserToolExecutor(
        now_ms=lambda: 0,
        viewport=(W, H),
        headless=True,
        allowed_origins=frozenset({loopback_origin}),
    )
    try:
        url = f"{loopback_origin}/shop.html"
        await run(ex.open(url))
        await run(ex.frame())
        before = await ex.driver.page.evaluate("window.__doc")
        out = await execute(ex, inv("press_key", None, combo="Ctrl+L"))
        assert out.status == "refused" and out.reason == NOT_AVAILABLE, (
            f"premise (A4): Ctrl+L outside web mode: {out.status} {out.reason!r}"
        )
        assert ex.driver.page.url == url and await ex.driver.page.evaluate("window.__doc") == before
        assert len(open_pages(ex)) == 1
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ex.close(), 30)
