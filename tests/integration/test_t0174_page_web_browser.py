"""T-0174 group B2, B2-3 (browser): W1-1 and W2-0 through the real executor, a real ``Session`` and a fake LLM; the
executor's ``navigated`` and the session's ``page`` key over ``/web.html`` served by ``StartPageServer``
(docs/tasks/T-0174.md).

**Gate.** ``integration`` + ``browser``; runs only with ``ASTRA_BROWSER_TESTS=1`` (T-0111c's autouse gate, imported).
Web mode needs Chromium's sandbox (T-0160 H0). Not in a ``make`` target: B2 owns no ``Makefile`` line (reported);
run ``ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0174_page_web_browser.py -q -m browser``.

**The rig** (T-0160 B-7 style, no network). The real ``StartPageServer`` on 127.0.0.1; the real
``BrowserToolExecutor(web=True, start_origin=<pages origin>, allowed_origins={origin})`` with the egress proxy's
resolver and connector injected: ``nav.example`` resolves to a public-looking address and every connect is piped to a
loopback TLS server (a self-signed certificate made per run in the pytest temp dir) that serves the T-0174 B1 nav
fixtures (``tests/fixtures/t0174/nav/``: hash link, ``pushState`` button, iframe) for that host; anything else fails
to resolve. ``Browser.new_context`` gets ``ignore_https_errors=True`` (the per-run certificate; T-0160's one
test-only relaxation, the fence is untouched). The executor sits behind ``test_computer_live_session.Watch`` (delegates
every name, records outcomes); the session is ``feedback_rig``'s (``tool_rig``, loop on, live kit over the watch)
built with ``Session(..., page_fact=True)``. Turn: step 1 is one scripted call whose coordinates come from the live
DOM; step 2 answers. The ``page`` value is read from the step-2 request's rendered tool response (the literal
``action, page, screenshot, status`` order, typed by hand).

**Rows** (B2-3): Go on an empty field (W1-1: ``web.html`` refuses it, nothing loads) gives ``navigated: False`` and
``page: "no new page loaded"``; ``type_at`` "nav.example" with Enter gives ``True`` and ``"new page loaded"``; on
``https://nav.example/`` (reached by a harness ``goto`` before the turn) a hash link, a ``pushState`` button and a
link inside the iframe each give ``False`` / ``"no new page loaded"``. Each ``False`` row proves its premise from the
page (the URL or the iframe's URL changed; the same document, ``window.__doc``, is still showing; ``web.html``
showed its refusal). ``test_default_no_page_key`` (no keyword; holds at BASE now that B1 is merged) is active.

Group B2. Every other row was written ahead as pending and activated for ``/implement T-0174`` B2; at BASE each fails on a named ``AssertionError``
(``Session(..., page_fact=...)`` does not exist yet).

Regime: Playwright 1.64.0, Chromium headless at 1280x800, DSF 1, sandbox on (H0), web mode through the egress proxy
with injected resolver/connector, ``StartPageServer`` on 127.0.0.1 at an ephemeral port, FakeLlm (one call, then an
answer), ``ScriptedClock`` driven only while no browser call is in flight, n = 1 per row, one browser per row.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import inspect
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

from local_astra.contracts.turn import TurnState
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
)
from tests.integration.test_computer_live_session import (
    Watch,
    ask,
    center,
    click,
    coord,
    drive,
    release_step,
    type_at,
)
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import feedback_rig, live_kit, session_mod
from tests.unit._tool_turn_support import answer, tool

pytestmark = [pytest.mark.integration] + (
    [pytest.mark.browser] if _browser_marker_registered() else []
)

REPO = Path(__file__).resolve().parents[2]
NAV = REPO / "tests" / "fixtures" / "t0174" / "nav"
W, H = 1280, 800
Q = '<|"|>'
HOST = "nav.example"
HOST_IP = "93.184.215.20"
NAV_URL = f"https://{HOST}/"
NEW = "new page loaded"
NO_NEW = "no new page loaded"
#: nav/index.html cell centres (CSS px, by hand from its stylesheet; MANIFEST.md): the hash link, the pushState
#: button, the iframe (x 650..1269, y 410..589; its page is one full-size link that navigates the iframe only)
HASH = (480, 100)
PUSH = (800, 100)
IFRAME = (960, 500)
ADDRESS = "document.getElementById('address')"
GO = "document.getElementById('go')"


# =====================================================================================================
# the rig
# =====================================================================================================


class _NavTLS(http.server.BaseHTTPRequestHandler):
    """``tests/fixtures/t0174/nav/<name>.html`` for ``Host: nav.example``; 404 for anything else. HTTP/1.1 with
    ``Content-Length`` (keep-alive), so a tunnel a warm-up opened can carry the measured navigation (review F1)."""

    protocol_version = "HTTP/1.1"
    hits: list[tuple[str, str]]

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        path = urllib.parse.urlsplit(self.path).path
        self.hits.append((host, path))
        name = "index.html" if path in ("", "/") else path.lstrip("/")
        f = NAV / name
        if host != HOST or "/" in name or not f.is_file() or f.suffix != ".html":
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, f.read_bytes(), "text/html; charset=utf-8")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class _QuietTLSServer(http.server.ThreadingHTTPServer):
    """Chromium probes the self-signed certificate before ``ignore_https_errors`` applies; those aborted handshakes
    are not row failures (T-0160's note)."""

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass


class Site:
    def __init__(self, tls_port: int, hits: list[tuple[str, str]], start: str) -> None:
        self.tls_port, self.hits, self.start = tls_port, hits, start


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Site]:
    d = tmp_path_factory.mktemp("t0174-b2-tls")
    key, crt = d / "key.pem", d / "crt.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(crt),
         "-days", "2", "-subj", f"/CN={HOST}", "-addext", f"subjectAltName=DNS:{HOST}"],
        check=True, capture_output=True,
    )  # fmt: skip
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    hits: list[tuple[str, str]] = []
    handler = type("_Bound", (_NavTLS,), {"hits": hits})
    srv = _QuietTLSServer(("127.0.0.1", 0), handler)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True, do_handshake_on_connect=False)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
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
    """The proxy's injected seams: ``nav.example`` only, piped to the loopback TLS server."""

    def __init__(self, tls_port: int) -> None:
        self.tls_port = tls_port
        self.resolves: list[str] = []
        self.connects: list[tuple[str, int]] = []

    async def resolve(self, host: str, port: int, *a: Any, **kw: Any) -> list[Any]:
        self.resolves.append(host)
        if host.lower().rstrip(".") != HOST:
            raise socket.gaierror(socket.EAI_NONAME, "not a fixture host")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (HOST_IP, port))]

    async def connect(
        self, ip: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        self.connects.append((ip, port))
        return await asyncio.open_connection("127.0.0.1", self.tls_port)


@pytest.fixture
def relaxed_tls(monkeypatch: pytest.MonkeyPatch) -> None:
    """T-0160's one test-only relaxation: the per-run certificate is accepted (certificate checking only)."""
    from playwright.async_api import Browser

    orig = Browser.new_context

    async def new_context(self: Any, **kw: Any) -> Any:
        kw["ignore_https_errors"] = True
        return await orig(self, **kw)

    monkeypatch.setattr(Browser, "new_context", new_context)


def require_page_fact() -> None:
    params = inspect.signature(session_mod().Session.__init__).parameters
    assert "page_fact" in params, "T-0174 B2-1: Session(..., page_fact=...) does not exist yet"


@contextlib.asynccontextmanager
async def web_session(
    tmp_path: Path, site: Site, *, page_fact: bool | None = True
) -> AsyncIterator[tuple[Any, Watch, Net]]:
    """A web-mode executor on ``<origin>/web.html`` behind ``Watch``, and a loop-on live session over it built with
    ``Session(..., page_fact=<page_fact>)`` (``None``: no keyword)."""
    from local_astra.computer.executor import BrowserToolExecutor

    if page_fact is not None:
        require_page_fact()
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
    watch = Watch(ex)
    kw: dict[str, Any] = {} if page_fact is None else {"page_fact": page_fact}
    try:
        await asyncio.wait_for(ex.open(site.start + "/web.html"), 30)
        async with feedback_rig(tmp_path, watch, session_kwargs=kw, kit=live_kit(watch)) as rig:
            ex._now_ms = lambda: rig.clock.now_ms
            yield rig, watch, net
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ex.close(), 30)


async def goto_nav(ex: Any) -> None:
    """Harness navigation to ``https://nav.example/`` before the turn (through the proxy, like any page load)."""
    await asyncio.wait_for(ex.frame(), 30)
    await ex.driver.page.goto(NAV_URL, wait_until="load", timeout=15_000)
    assert ex.driver.page.url == NAV_URL, (
        f"premise: the harness reached {NAV_URL}; at {ex.driver.page.url}"
    )


async def one_call_turn(rig: Any, watch: Watch, call: str) -> None:
    """Step 1 is ``call``; step 2 answers; the turn runs to its end."""
    rig.llm.script = [answer("I will do that. "), tool(call)]
    rig.llm.hold_steps()
    await ask(rig, watch, "open it")
    await drive(rig, watch, lambda: len(watch.outcomes) >= 1 and watch.inflight == 0, "step 1")
    await release_step(rig, watch, 1, [answer("Done. ")])
    rig.llm.release_steps()
    await drive(
        rig, watch, lambda: rig.state.state is TurnState.IDLE and bool(rig.lines()), "turn end"
    )


def only_outcome(watch: Watch, name: str) -> Any:
    assert len(watch.outcomes) == 1, f"premise: one call executed; {watch.outcomes}"
    out = watch.outcomes[0]
    assert (watch.invocations[0].name, out.status) == (name, "executed"), out
    return out


def rendered(name: str, action: str, page: str, screenshot: str) -> str:
    """B2-2's rendered response, typed by hand (dictsort: ``action``, ``page``, ``screenshot``, ``status``)."""
    return (
        f"<|tool_response>response:{name}{{action:{Q}{action}{Q},page:{Q}{page}{Q},"
        f"screenshot:{Q}{screenshot}{Q},status:{Q}executed{Q}}}<tool_response|>"
    )


def assert_page(rig: Any, name: str, out: Any, page: str) -> None:
    """The step-2 request carries step 1's response with ``page`` (either B8 ``screenshot`` value; the executor's
    action string as it reported it)."""
    tail = tsl.tail(tsl.step_request(rig, 2))
    options = [rendered(name, out.action, page, s) for s in ("updated", "unchanged")]
    assert sum(tail.count(o) for o in options) == 1, (
        f"T-0174 B2-3: step 1's response carries page {page!r}; want one of {options}; got ...{tail[-600:]}"
    )


async def doc_token(ex: Any) -> str:
    return str(await ex.driver.page.evaluate("window.__doc"))


# =====================================================================================================
# B2-3
# =====================================================================================================


async def test_go_on_empty_field(tmp_path: Path, site: Site, relaxed_tls: None) -> None:
    """B2-3 (W1-1): Go clicked with the address field empty: ``web.html`` refuses it (its error line shows text,
    the URL is still ``/web.html``); ``navigated`` is ``False`` and the step-2 response says ``"no new page
    loaded"``."""
    async with web_session(tmp_path, site) as (rig, watch, net):
        ex = watch._inner
        assert await ex.driver.page.evaluate(f"{ADDRESS}.value") == "", (
            "premise: the field is empty"
        )
        x, y = await center(ex, GO)
        await one_call_turn(rig, watch, click(x, y))
        out = only_outcome(watch, "click_at")
        assert ex.driver.page.url == site.start + "/web.html", (
            f"premise: nothing loaded; {ex.driver.page.url}"
        )
        err = await ex.driver.page.evaluate("document.getElementById('error').textContent")
        assert err.strip(), "premise: web.html refused the empty field"
        assert out.resolved.get("navigated") is False, f"T-0174 B1-1: {out.resolved}"
        assert_page(rig, "click_at", out, NO_NEW)
        assert net.connects == [], f"premise: nothing left the box; {net.connects}"


async def test_enter_valid_host(tmp_path: Path, site: Site, relaxed_tls: None) -> None:
    """B2-3 (W2-0): ``type_at`` "nav.example" with Enter on the address field: ``web.html`` assigns
    ``https://nav.example/``, which loads through the proxy (the fixture server saw it); ``navigated`` is ``True``
    and the step-2 response says ``"new page loaded"``.

    Review F1 (2 of 50 failed under 40-48 CPU spinners: the first proxy connect, TLS handshake and certificate path
    to nav.example were still pending at settle, so B1-1b correctly left ``navigated`` absent). Before the turn the
    harness warms that path: ``goto`` ``https://nav.example/`` (premise: the fixture server saw it), then back to
    ``/web.html``, so the measured navigation needs no first resolve, connect or handshake (the server keeps the
    connection alive). Absent is never accepted: the row still requires ``navigated is True``."""
    async with web_session(tmp_path, site) as (rig, watch, net):
        ex = watch._inner
        warm_hits = len(site.hits)
        await goto_nav(ex)
        assert len(site.hits) > warm_hits and net.connects, (
            "premise: the warm-up reached nav.example"
        )
        await ex.driver.page.goto(site.start + "/web.html", wait_until="load", timeout=15_000)
        assert ex.driver.page.url == site.start + "/web.html", (
            f"premise: back on web.html; {ex.driver.page.url}"
        )
        x, y = await center(ex, ADDRESS)
        await one_call_turn(rig, watch, type_at(x, y, HOST, True))
        out = only_outcome(watch, "type_at")
        deadline = time.monotonic() + 10
        while ex.driver.page.url != NAV_URL and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert ex.driver.page.url == NAV_URL, f"premise: the page loaded; at {ex.driver.page.url}"
        assert (HOST, "/") in site.hits and (HOST_IP, 443) in net.connects, (
            site.hits,
            net.connects,
        )
        assert out.resolved.get("navigated") is True, f"T-0174 B1-1: {out.resolved}"
        assert_page(rig, "type_at", out, NEW)


#: case -> (the cell's CSS centre, the premise: what changed without a new main-frame document)
SAME_DOC = {
    "hash": (HASH, "url", NAV_URL + "#section"),
    "pushState": (PUSH, "url", NAV_URL + "?pushed=1"),
    "iframe": (IFRAME, "frame", f"https://{HOST}/frame_target.html"),
}


@pytest.mark.parametrize("case", list(SAME_DOC))
async def test_same_document_is_no_new_page(
    tmp_path: Path, site: Site, relaxed_tls: None, case: str
) -> None:
    """B2-3: on ``https://nav.example/`` a hash link, a ``pushState`` button and a link inside the iframe each give
    ``navigated: False`` and ``"no new page loaded"``. Premise: the URL (or the iframe's URL) changed while the same
    main-frame document (``window.__doc``) is still showing, so neither a URL comparison nor a bare
    ``framenavigated`` passes."""
    (cx, cy), kind, want_url = SAME_DOC[case]
    async with web_session(tmp_path, site) as (rig, watch, _net):
        ex = watch._inner
        await goto_nav(ex)
        before = await doc_token(ex)
        await one_call_turn(rig, watch, click(coord(cx, W), coord(cy, H)))
        out = only_outcome(watch, "click_at")
        page = ex.driver.page
        if kind == "url":
            assert page.url == want_url, f"premise ({case}): the URL changed; {page.url}"
        else:
            deadline = time.monotonic() + 10
            urls: list[str] = []
            while time.monotonic() < deadline:
                urls = [f.url for f in page.frames if f is not page.main_frame]
                if want_url in urls:
                    break
                await asyncio.sleep(0.05)
            assert want_url in urls, f"premise ({case}): the iframe navigated; {urls}"
            assert page.url == NAV_URL, page.url
        assert await doc_token(ex) == before, f"premise ({case}): the same document is showing"
        assert out.resolved.get("navigated") is False, f"T-0174 B1-1 ({case}): {out.resolved}"
        assert_page(rig, "click_at", out, NO_NEW)


async def test_default_no_page_key(tmp_path: Path, site: Site, relaxed_tls: None) -> None:
    """B2-4 (active): without the keyword the same W1-1 turn's step-2 response has no ``page`` key, although the
    executor reported ``navigated`` (B1)."""
    async with web_session(tmp_path, site, page_fact=None) as (rig, watch, _net):
        ex = watch._inner
        x, y = await center(ex, GO)
        await one_call_turn(rig, watch, click(x, y))
        out = only_outcome(watch, "click_at")
        assert out.resolved.get("navigated") is False, f"premise (B1): {out.resolved}"
        tail = tsl.tail(tsl.step_request(rig, 2))
        assert "<|tool_response>response:click_at{" in tail, f"premise: ...{tail[-400:]}"
        assert "page:" not in tail.split("<|tool_response>")[-1] and "page loaded" not in tail, (
            f"T-0174 B2-4: default is BASE; got ...{tail[-400:]}"
        )
