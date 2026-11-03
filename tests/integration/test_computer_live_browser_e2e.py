"""T-0111g group A against a real headless Chromium: the end-on-close rule (A2), the origin allow-list (A3) and the
start pages (A4, A5) (docs/tasks/T-0111g.md).

**Gate.** As ``tests/integration/test_computer_browser_e2e.py``, whose gate, fixtures and helpers are imported, not
copied: the module is ``integration`` + ``browser`` and runs only with ``ASTRA_BROWSER_TESTS=1``; with it, a missing
``playwright`` or Chromium is a failure naming P1. ``make browser`` does not run this module (its target is pinned
by ``test_computer_imports.py``); ``make browser-live`` (group B) will.

**The API these rows use** is pinned in ``tests/unit/test_computer_live_options.py`` (``end_when_closed``,
``allowed_origins``, ``closed``, ``executor.BrowserClosedError``) and ``tests/unit/test_start_pages.py``
(``pages.server.StartPageServer`` with ``start()``, ``origin``, ``stop()``), plus T-0111c's ``ex.driver.page``.

**Page shapes these rows read** (A5 names behaviour, not markup; these are the least-committal readings, reported):

- an "Add to cart" control is any ``button``/``a``/``input[type=button|submit]`` whose text is exactly "Add to cart"
  (case-insensitive); its *card* is its highest ancestor below ``body`` holding no other such control;
- a deals card names exactly one price (a currency sign and a number, or a number with two decimals);
- the visible cart count is a digit on a line of ``body.innerText`` that says "cart" once "Add to cart" is removed;
- the contact form is two visible inputs (one ``type=email``) and one ``textarea``, and a "Send" control;
  ``__state.sent`` is ``{"name", "email", "message"}``;
- "over two viewports" is a document height in (800, 1600]; "below the fold" is a card whose top is at or below
  800 px with the page unscrolled; cart names are lowercase, as C-E1's ``["blue mug"]``.

**Regime.** Playwright 1.64.0, Chromium 156.0.8078.4 headless, 1280x800 at DSF 1, the start-page server on 127.0.0.1
at an ephemeral port, n = 1, one executor (one browser) per row.

``test_attempt_vectors_are_live_under_the_t0111c_rule`` is the one row green at BASE, by design: it proves the six
attempt vectors of the allow-list row both reach a loopback canary under T-0111c's rule (``allowed_origins=None``,
which A3 keeps) and are seen by the route (logged when off-loopback), so the allow-list row cannot pass vacuously.
Every other row was ``pending(task="T-0111g")`` until group A's implementation.

**Group A review rows** (Rulings, "on the group A reviews of c4618da"): A1 (redirects inside the allow-list, green at
c4618da, kills Mr and Mredir_nolog), A2 (SharedWorker), A3 (the last page closed inside ``ensure()``, fault-injected
on Playwright's ``Page`` class) and B-F1 (plural and any-word search). They use ``FixtureSite``, a stdlib server in
this module, for the 302/307 hops, the worker script and a held page.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import signal
import socket
import threading
import time
import urllib.parse
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from local_astra.conductor.tool_executor import DryRunToolExecutor
from local_astra.contracts.perception import Frame
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    CANARY_IP,
    CLOCK_MS,
    Canary,
    _browser_layer,
    _browser_marker_registered,
    chromium_mains,
    computer,
    execute,
    inv,
    mean_abs_diff,
    near,
    poll,
    rgb_at,
    run,
    server,
)
from tests.unit.test_computer_live_options import CLOSED, closed_error, closed_of
from tests.unit.test_start_pages import new_server

pytestmark = [pytest.mark.integration] + (
    [pytest.mark.browser] if _browser_marker_registered() else []
)

W, H = 1280, 800
#: A3's named service ports (vLLM, the plugin tap, the HUD and STT/TTS defaults, Tauri's dev port).
SERVICE_PORTS = (8000, 8001, 8080, 8081, 1420)
VECTORS = ("img", "fetch", "beacon", "iframe", "websocket")
SECRET = "Q7Z9"
PAGES = ("/", "/search.html?q=mug", "/demo/shop.html", "/demo/contact.html", "/demo/deals.html")


@contextlib.asynccontextmanager
async def live_executor(**kw: Any) -> AsyncIterator[Any]:
    """A T-0111c executor with T-0111g's options; a missing option is a named AssertionError, not a TypeError."""
    cls = computer().BrowserToolExecutor
    kw.setdefault("now_ms", lambda: CLOCK_MS)
    try:
        ex = cls(**kw)
    except TypeError as exc:
        raise AssertionError(
            f"T-0111g: BrowserToolExecutor does not take {sorted(kw)} yet: {exc}"
        ) from exc
    try:
        yield ex
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ex.close(), 20)


@contextlib.contextmanager
def start_pages() -> Iterator[Any]:
    srv = new_server()
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def service_port_canaries() -> tuple[dict[int, Canary], tuple[int, ...]]:
    """A canary on each A3 service port that is free, and the ports that are busy (Rulings, "on cd9272a"). A busy
    port is a real service (8080 is the human's STT container at the desk), which a row must not require stopping:
    its connections cannot be counted, so it is checked by the route log only. The canary binds with
    ``SO_REUSEADDR`` (a TIME_WAIT left by an earlier run is not "in use"; a listener is)."""
    out: dict[int, Canary] = {}
    busy: list[int] = []
    for port in SERVICE_PORTS:
        try:
            out[port] = PortCanary(port)
        except OSError:
            busy.append(port)
    return out, tuple(busy)


class PortCanary(Canary):
    """T-0111c's ``Canary`` (its ``close``) on a given 127.0.0.1 port instead of port 0. Every accepted connection
    counts, as in T-0111c's; the first bytes of each are kept in ``heads`` so a failure says whether the browser
    (a ``/t0111g-`` path) or some other process on this box (a health probe of a service port) connected."""

    def __init__(self, port: int) -> None:
        self.sock = socket.socket()
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("127.0.0.1", port))
            self.sock.listen(64)
        except OSError:
            self.sock.close()
            raise
        self.sock.settimeout(0.1)
        self.port, self.count = port, 0
        self.heads: list[bytes] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name=f"t0111g-canary-{port}", daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                c, _ = self.sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.count += 1
            head = b""
            with contextlib.suppress(OSError):
                c.settimeout(0.3)
                head = c.recv(160)
            self.heads.append(head)
            with contextlib.suppress(OSError):
                c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            c.close()


#: One attempt of one vector against ``authority`` (``host:port``); every URL carries a secret query.
ATTEMPT_JS = """async ([authority, kind]) => {
  const http = `http://${authority}/t0111g-${kind}?secret=Q7Z9`;
  const ws = `ws://${authority}/t0111g-ws?secret=Q7Z9`;
  const settle = (p) => Promise.race([
    p.then((ok) => ({ ok }), (e) => ({ ok: false, error: String((e && e.name) || e) })),
    new Promise((r) => setTimeout(() => r({ ok: false, error: "timeout" }), 3000)),
  ]);
  if (kind === "img") return settle(new Promise((res, rej) => {
    const i = new Image(); i.onload = () => res(true); i.onerror = () => rej(new Error("img")); i.src = http; }));
  if (kind === "fetch") return settle(fetch(http, { mode: "no-cors", cache: "no-store" }).then(() => true));
  if (kind === "beacon") {
    try { return { ok: null, queued: navigator.sendBeacon(http, "t0111g") }; }
    catch (e) { return { ok: false, error: String(e.name) }; } }
  if (kind === "iframe") return settle(new Promise((res) => {
    const f = document.createElement("iframe"); f.onload = () => res(null); f.src = http;
    document.body.appendChild(f); }));
  if (kind === "websocket") return settle(new Promise((res, rej) => {
    let s; try { s = new WebSocket(ws); } catch (e) { rej(e); return; }
    s.onopen = () => { s.close(); res(true); }; s.onerror = () => rej(new Error("ws")); }));
  throw new Error("unknown vector " + kind);
}"""


def _scheme(vector: str) -> str:
    return "ws" if vector == "websocket" else "http"


#: ``ATTEMPT_JS``'s own settle (3000 ms), plus margin for a loaded box (T-0218 B).
LOG_WAIT_S = 3.0 + 3.0
#: how long a top-level navigation may take to settle, and how long the route log must stay unchanged to count
#: as quiet (T-0218 B: condition waits instead of fixed sleeps).
NAV_WAIT_S = 10.0
LOG_QUIET_S = 1.0


async def _log_since(ex: Any, n0: int, timeout: float = LOG_WAIT_S) -> list[dict[str, str]]:
    log = await poll(lambda: asyncio.sleep(0, list(ex.egress_log)), lambda v: len(v) > n0, timeout)
    return list(log[n0:])


#: a marker on the document the navigation starts from; gone once another document (landed or Chromium's error
#: page) has committed and loaded.
_NAV_MARK_JS = "() => { window.__t0218_nav_from = 1; return 1; }"
_NAV_DONE_JS = "() => window.__t0218_nav_from !== 1 && document.readyState === 'complete'"


async def _nav_settled(page: Any) -> bool:
    try:
        return bool(await page.evaluate(_NAV_DONE_JS))
    except Exception:  # the context is destroyed mid-navigation: not settled yet
        return False


async def _top_level(ex: Any, authority: str) -> None:
    """A top-level navigation to ``authority``, returned once it settled: the document it started from is gone and
    the next one (the landed page or the error page of a refused request) is loaded, bounded by
    :data:`NAV_WAIT_S`. Never raises on the bound: the callers assert what happened."""
    page = ex.driver.page
    with contextlib.suppress(Exception):
        await page.evaluate(_NAV_MARK_JS)
    with contextlib.suppress(Exception):
        await page.evaluate(
            "u => { location.href = u; }", f"http://{authority}/t0111g-nav?secret={SECRET}"
        )
    await poll(lambda: _nav_settled(page), bool, NAV_WAIT_S)


async def _quiet_log(ex: Any) -> list[dict[str, str]]:
    """The route log once it has not grown for :data:`LOG_QUIET_S` (bounded by :data:`NAV_WAIT_S`): a late
    record (a queued beacon, an iframe's retry) is in it before the record shapes are read."""
    deadline = time.monotonic() + NAV_WAIT_S
    last = list(ex.egress_log)
    quiet_since = time.monotonic()
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        now = list(ex.egress_log)
        if len(now) != len(last):
            last, quiet_since = now, time.monotonic()
        elif time.monotonic() - quiet_since >= LOG_QUIET_S:
            break
    return list(ex.egress_log)


# --------------------------------------------------------------------------------------------- A3 premise (green)


async def test_attempt_vectors_are_live_under_the_t0111c_rule(server: str) -> None:  # noqa: F811
    """Self-check (green at BASE): under T-0111c's rule, which A3's ``None`` keeps, each of the allow-list row's
    vectors (img, fetch, sendBeacon, iframe, WebSocket, top-level navigation) reaches a canary on a free
    127.0.0.1 port at least once, and each aimed off-loopback (127.0.1.1) is seen by the route and logged with its
    scheme. Without this, a vector that never fires would make the allow-list row pass vacuously."""
    async with live_executor() as ex:
        await run(ex.open(f"{server}/grid.html"))
        canary = Canary("127.0.0.1")
        reached: dict[str, int] = {}
        try:
            for vector in VECTORS:
                c0 = canary.count
                await run(ex.driver.page.evaluate(ATTEMPT_JS, [f"127.0.0.1:{canary.port}", vector]))
                await poll(lambda: asyncio.sleep(0, canary.count), lambda n: n > c0, 3.0)  # noqa: B023
                reached[vector] = canary.count - c0
            c0 = canary.count
            await _top_level(ex, f"127.0.0.1:{canary.port}")
            reached["toplevel"] = canary.count - c0
        finally:
            canary.close()
        assert all(n >= 1 for n in reached.values()), (
            f"a vector never connected unfenced: {reached}"
        )

        await run(ex.open(f"{server}/grid.html"))
        off = Canary(CANARY_IP)
        logged: dict[str, list[dict[str, str]]] = {}
        try:
            for vector in VECTORS:
                n0 = len(ex.egress_log)
                await run(ex.driver.page.evaluate(ATTEMPT_JS, [f"{CANARY_IP}:{off.port}", vector]))
                logged[vector] = await _log_since(ex, n0)
            n0 = len(ex.egress_log)
            await _top_level(ex, f"{CANARY_IP}:{off.port}")
            logged["toplevel"] = await _log_since(ex, n0)
        finally:
            off.close()
        for vector, recs in logged.items():
            want = {"host": CANARY_IP, "scheme": "ws" if vector == "websocket" else "http"}
            assert want in recs, f"the route did not see {vector}: {recs}"
        assert off.count == 0, off.count


# --------------------------------------------------------------------------------------------- A2


async def _ended(ex: Any, origin: str, mains_before: set[int]) -> None:
    """A2's post-conditions on a real browser: every action refused with the reason, ``closed``, ``frame()``
    raising the named exception, ``open()`` opening nothing, search the dry run, and no new Chromium process."""
    err = closed_error()
    f = ex.latest_frame
    assert isinstance(f, Frame)
    for i in (
        inv("click_at", f, x=500, y=500),
        inv("type_at", f, x=500, y=500, text="mug", press_enter=True),
        inv("scroll", direction="down"),
        inv("press_key", combo="Tab"),
    ):
        out = await execute(ex, i)
        assert (out.status, out.reason) == ("refused", CLOSED), (i.name, out)
    assert closed_of(ex) is True
    with pytest.raises(err):
        await run(ex.frame())
    with contextlib.suppress(err):
        await run(ex.open(f"{origin}/"))
    for i in (inv("google_search", query="blue mug"), inv("google_search")):
        assert await execute(ex, i) == await DryRunToolExecutor().execute(i)
    await asyncio.sleep(0.5)
    after = {p for p, _, _ in chromium_mains()}
    assert after <= mains_before, (
        f"a Chromium process started after the end: {after - mains_before}"
    )


async def test_close_all_pages_ends_it() -> None:
    """A2 (reviewer mutation 12), headless: every page closed through Playwright's API (the stand-in for the human
    closing the window; Q1). The browser stays connected with no page; the executor ends it for good, opens no
    page and launches nothing."""
    with start_pages() as srv:
        async with live_executor(
            end_when_closed=True, allowed_origins=frozenset({srv.origin})
        ) as ex:
            await run(ex.open(f"{srv.origin}/"))
            await run(ex.frame())
            assert closed_of(ex) is False
            ctx = ex.driver.page.context
            mains = {p for p, _, _ in chromium_mains()}
            assert len(mains) == 1, mains
            for p in list(ctx.pages):
                await run(p.close())
            assert ctx.pages == []
            await asyncio.sleep(0.3)
            await _ended(ex, srv.origin, mains)
            assert ctx.pages == [], f"a page was opened after the end: {ctx.pages}"


async def test_kill_with_end_when_closed_ends_it() -> None:
    """A2, headless: the browser is SIGKILLed (only the main process this row started). The next calls are refused
    with ``the browser window was closed`` instead of T-0111c's relaunch (``failed: browser restarted``), and the
    pid set of Chromium mains never gains a member."""
    with start_pages() as srv:
        async with live_executor(
            end_when_closed=True, allowed_origins=frozenset({srv.origin})
        ) as ex:
            await run(ex.open(f"{srv.origin}/"))
            await run(ex.frame())
            mains = chromium_mains()
            assert len(mains) == 1, mains
            os.kill(mains[0][0], signal.SIGKILL)
            await asyncio.sleep(1.0)
            await _ended(ex, srv.origin, {mains[0][0]})
            assert chromium_mains() == [], "relaunched after the end"


async def test_self_closed_popup_is_not_an_end(server: str) -> None:  # noqa: F811
    """A2's too-wide side on a real browser, with ``end_when_closed=True``: T-0111c's ``self_close_popup.html`` popup
    closes itself while the opener stays; ``closed`` stays False, the next frame shows the opener, and a click on it
    executes (the S9 row's flow, in end mode)."""
    async with live_executor(end_when_closed=True) as ex:
        await run(ex.open(f"{server}/self_close_popup.html"))
        f = await run(ex.frame())
        opener = ex.driver.page
        out = await execute(ex, inv("click_at", f, x=500, y=500))
        assert out.status == "executed", out
        popup = ex.driver.page
        assert popup is not opener, "the premise failed: no popup became active"
        await run(popup.wait_for_function("window.__ready === true"))
        await run(ex.frame())
        await opener.evaluate("new BroadcastChannel('t0111c-self-close').postMessage('close')")
        closed = await poll(lambda: asyncio.sleep(0, popup.is_closed()), bool, timeout=5.0)
        assert closed, "the premise failed: the popup did not close itself"
        await asyncio.sleep(0.2)
        nxt = await run(ex.frame())
        assert closed_of(ex) is False, "a popup's own close ended the browser"
        assert ex.driver.page is opener and near(rgb_at(nxt, 640, 400), (255, 0, 0))
        out = await execute(ex, inv("click_at", nxt, x=500, y=500))
        assert out.status == "executed", out


# --------------------------------------------------------------------------------------------- A3


async def test_service_ports_unreachable_with_allow_list() -> None:
    """A3 adversarial (reviewer mutation 13). With the allow-list set to the start-page origin, from the start page:
    img, fetch, sendBeacon, iframe, WebSocket and a top-level navigation to 127.0.0.1 on 8000, 8001, 8080, 8081,
    1420 and a free port each fail and each adds a ``{host, scheme}`` record (with its scheme) to ``egress_log``;
    ``open()`` of the canary is refused; the canary on the free port, and one on each service port, accept zero
    connections; no record holds a port, path or the query secret. The start-page origin loads (the page and a
    same-origin fetch).

    Busy service ports (Rulings, "on cd9272a"): a free service port gets a canary (zero connections, and every
    attempt logged); a busy one (a real service, e.g. the human's STT on 8080) is checked by the route log only:
    every attempt is logged as ``{127.0.0.1, http|ws}`` and none succeeds, and a miss there names the port as
    "checked by the route log only". The extra free port always has a canary, so at least one port is
    canary-checked whatever is running. The premise row (``test_attempt_vectors_are_live_under_the_t0111c_rule``)
    proves the vectors fire."""
    with start_pages() as srv:
        canary = Canary("127.0.0.1")
        services: dict[int, Canary] = {}
        try:
            services, busy = service_port_canaries()

            def tag(port: int) -> str:
                return " (checked by the route log only: busy)" if port in busy else ""

            async with live_executor(allowed_origins=frozenset({srv.origin})) as ex:
                await run(ex.open(f"{srv.origin}/"))
                page = ex.driver.page
                assert page.url.startswith(srv.origin), page.url
                same = await run(
                    page.evaluate(
                        "() => fetch('/', {cache: 'no-store'}).then(r => r.ok, () => false)"
                    )
                )
                assert same is True, "a same-origin fetch on the start page failed"
                assert list(ex.egress_log) == [], ex.egress_log
                problems: list[str] = []
                for port in (*SERVICE_PORTS, canary.port):
                    authority = f"127.0.0.1:{port}"
                    for vector in VECTORS:
                        n0 = len(ex.egress_log)
                        res = await run(ex.driver.page.evaluate(ATTEMPT_JS, [authority, vector]))
                        new = await _log_since(ex, n0)
                        if {"host": "127.0.0.1", "scheme": _scheme(vector)} not in new:
                            problems.append(f"{authority} {vector}: not logged ({new}){tag(port)}")
                        if vector in ("img", "fetch", "websocket") and res.get("ok") is not False:
                            problems.append(f"{authority} {vector}: succeeded ({res}){tag(port)}")
                    n0 = len(ex.egress_log)
                    await _top_level(ex, authority)
                    new = await _log_since(ex, n0)
                    if {"host": "127.0.0.1", "scheme": "http"} not in new:
                        problems.append(f"{authority} top-level: not logged ({new}){tag(port)}")
                    landed = urllib.parse.urlsplit(ex.driver.page.url)
                    if landed.port == port:
                        problems.append(
                            f"{authority} top-level: the page is now {landed.netloc}{tag(port)}"
                        )
                    await run(ex.open(f"{srv.origin}/"))
                with pytest.raises(ValueError):
                    await run(
                        ex.open(f"http://127.0.0.1:{canary.port}/t0111g-open?secret={SECRET}")
                    )
                log = await _quiet_log(ex)
            for rec in log:
                if set(rec) != {"host", "scheme"} or SECRET in str(rec) or ":" in rec["host"]:
                    problems.append(f"a record keeps more than host and scheme: {rec}")
            counts = {"free": canary.count} | {str(p): c.count for p, c in services.items()}
            route_only = ", ".join(str(p) for p in busy) or "none"
            heads = {p: c.heads for p, c in services.items() if c.count}
            assert all(n == 0 for n in counts.values()), (
                f"connections reached a canary: {counts}; service-port request heads: {heads}; "
                f"ports checked by the route log only (busy): {route_only}"
            )
            assert not problems, "\n".join(
                [*problems, f"ports checked by the route log only (busy): {route_only}"]
            )
        finally:
            canary.close()
            for c in services.values():
                c.close()


# --------------------------------------------------------------------------------------------- group A review rows

#: The fixture site's pages: a blank page with a link, a same-origin child, and a SharedWorker script that fetches
#: its ``to`` query and reports back.
_SITE_PAGE = b'<!doctype html><html><body style="margin:0;background:#fff"><a id="nav" href="#">nav</a></body></html>'
_SITE_WORKER = b"""onconnect = (e) => {
  const port = e.ports[0];
  const to = new URL(self.location.href).searchParams.get("to");
  fetch(to, { mode: "no-cors", cache: "no-store" }).then(() => port.postMessage("reached"),
    (err) => port.postMessage("failed " + String(err && err.name)));
};
"""


class FixtureSite:
    """A small stdlib server on 127.0.0.1 for the review rows (``tests/integration/test_computer_browser_e2e.py``'s
    ``server`` has no 307 and no slow page): ``/page.html`` and ``/child.html`` (``_SITE_PAGE``); ``/r/<code>?to=<url>``
    answers ``<code>`` (302 or 307) with ``Location: <url>`` and counts the hit; ``/sw.js`` is ``_SITE_WORKER``;
    ``/slow.html`` holds its answer until ``release`` is set (bounded at 15 s). Logs nothing."""

    def __init__(self) -> None:
        import http.server

        site = self
        self.hits: dict[int, int] = {302: 0, 307: 0}
        self.release = threading.Event()
        self.slow_requests = 0

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _send(self, code: int, body: bytes, ctype: str, **headers: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                u = urllib.parse.urlsplit(self.path)
                m = re.fullmatch(r"/r/(302|307)", u.path)
                if m:
                    code = int(m.group(1))
                    site.hits[code] += 1
                    to = urllib.parse.parse_qs(u.query)["to"][0]
                    self._send(code, b"", "text/plain", Location=to)
                elif u.path in ("/page.html", "/child.html"):
                    self._send(200, _SITE_PAGE, "text/html; charset=utf-8")
                elif u.path == "/sw.js":
                    self._send(200, _SITE_WORKER, "text/javascript")
                elif u.path == "/slow.html":
                    site.slow_requests += 1
                    site.release.wait(15)
                    with contextlib.suppress(OSError):
                        self._send(200, _SITE_PAGE, "text/html; charset=utf-8")
                else:
                    self._send(404, b"no", "text/plain")

        self._srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._srv.daemon_threads = True
        self.origin = f"http://127.0.0.1:{self._srv.server_address[1]}"
        self._t = threading.Thread(
            target=self._srv.serve_forever, name="t0111g-fixture-site", daemon=True
        )
        self._t.start()

    def close(self) -> None:
        self.release.set()
        self._srv.shutdown()
        self._srv.server_close()
        self._t.join(5)


@contextlib.contextmanager
def fixture_site() -> Iterator[FixtureSite]:
    site = FixtureSite()
    try:
        yield site
    finally:
        site.close()


async def _redirect_attempt(ex: Any, kind: str, hop: str) -> None:
    page = ex.driver.page
    with contextlib.suppress(Exception):
        if kind == "fetch":
            await run(
                page.evaluate(
                    "u => fetch(u, {mode: 'no-cors', cache: 'no-store'}).catch(() => 0)", hop
                )
            )
        elif kind == "img":
            await run(page.evaluate("u => { const i = new Image(); i.src = u; }", hop))
        elif kind == "iframe":
            await run(
                page.evaluate(
                    "u => { const f = document.createElement('iframe'); f.src = u; document.body.appendChild(f); }",
                    hop,
                )
            )
        else:
            await run(page.evaluate("u => { location.href = u; }", hop))


async def test_redirects_inside_the_allow_list_reach_nothing() -> None:
    """Group A review ruling, row A1 (kills mutants Mr, the set-mode route following redirects, and Mredir_nolog,
    the redirect's target not logged). The allowed origin is a fixture server that answers 302 and 307 to a canary
    on another 127.0.0.1 port, for fetch, img, iframe and top-level navigation. Each of the eight attempts reaches
    the redirect (the server counts the hop), adds a ``{"host": "127.0.0.1", "scheme": "http"}`` record to
    ``egress_log``, and the canary accepts zero connections; no record keeps the port, path or secret, and the
    top-level attempt does not land on the canary. (T-0111c's redirect row covers a 302 off loopback; this is a
    redirect to loopback that only the allow-list can stop: the resolver rule admits 127.0.0.1.)"""
    with fixture_site() as site:
        async with live_executor(allowed_origins=frozenset({site.origin})) as ex:
            await run(ex.open(f"{site.origin}/page.html"))
            assert list(ex.egress_log) == [], ex.egress_log
            counts: dict[str, int] = {}
            problems: list[str] = []
            for code in (302, 307):
                for kind in ("fetch", "img", "iframe", "toplevel"):
                    canary = Canary("127.0.0.1")
                    try:
                        target = f"http://127.0.0.1:{canary.port}/t0111g-secret-path/{kind}?secret={SECRET}"
                        hop = f"{site.origin}/r/{code}?to={urllib.parse.quote(target, safe='')}"
                        h0, n0 = site.hits[code], len(ex.egress_log)
                        await _redirect_attempt(ex, kind, hop)
                        new = await _log_since(ex, n0)
                        await asyncio.sleep(0.5)
                        counts[f"{code} {kind}"] = canary.count
                        if site.hits[code] == h0:
                            problems.append(f"{code} {kind}: the redirect was never requested")
                        if {"host": "127.0.0.1", "scheme": "http"} not in new:
                            problems.append(
                                f"{code} {kind}: the redirect target was not logged ({new})"
                            )
                        landed = urllib.parse.urlsplit(ex.driver.page.url)
                        if landed.port == canary.port:
                            problems.append(f"{code} {kind}: the page is now {landed.netloc}")
                    finally:
                        canary.close()
                    await run(ex.open(f"{site.origin}/page.html"))
            log = list(ex.egress_log)
    for rec in log:
        if set(rec) != {"host", "scheme"} or SECRET in str(rec) or ":" in rec["host"]:
            problems.append(f"a record keeps more than host and scheme: {rec}")
    assert all(n == 0 for n in counts.values()), f"a redirect reached the canary: {counts}"
    assert not problems, "\n".join(problems)


_SW_TYPES_JS = """async (childUrl) => {
  const load = (f) => new Promise((r) => { f.onload = () => r(null); setTimeout(() => r(null), 3000); });
  const same = document.createElement("iframe"); same.src = childUrl;
  const p = load(same); document.body.appendChild(same); await p;
  const blank = document.createElement("iframe"); document.body.appendChild(blank);
  return { top: typeof window.SharedWorker, same_origin_child: typeof same.contentWindow.SharedWorker,
           about_blank_child: typeof blank.contentWindow.SharedWorker,
           in_window: "SharedWorker" in window };
}"""
_SW_ATTEMPT_JS = """(url) => new Promise((resolve) => {
  let w;
  try { w = new SharedWorker(url); } catch (e) { resolve("not made: " + String(e && e.name)); return; }
  w.port.onmessage = (m) => resolve(String(m.data));
  w.onerror = () => resolve("worker error");
  w.port.start();
  setTimeout(() => resolve("timeout"), 4000);
})"""


async def test_shared_worker_absent_and_reaches_nothing() -> None:
    """Group A review ruling, row A2 (fix A2: SharedWorker removed by the init script, beside WebRTC). Under the
    allow-list, ``typeof SharedWorker`` is ``"undefined"`` in the top frame, a same-origin child and a script-made
    about:blank child, and a page's attempt to reach another 127.0.0.1 port through a SharedWorker (a same-origin
    worker script that fetches the canary) reaches nothing: the canary accepts zero connections. A context route
    does not see a shared worker's requests, so without the removal the fetch goes straight to the port."""
    with fixture_site() as site:
        canary = Canary("127.0.0.1")
        try:
            async with live_executor(allowed_origins=frozenset({site.origin})) as ex:
                await run(ex.open(f"{site.origin}/page.html"))
                page = ex.driver.page
                types = await run(page.evaluate(_SW_TYPES_JS, f"{site.origin}/child.html"))
                to = f"http://127.0.0.1:{canary.port}/t0111g-shared-worker?secret={SECRET}"
                attempt = await run(
                    page.evaluate(
                        _SW_ATTEMPT_JS, f"{site.origin}/sw.js?to={urllib.parse.quote(to, safe='')}"
                    )
                )
                await asyncio.sleep(1.0)
                reached = canary.count
        finally:
            canary.close()
    present = {k: v for k, v in types.items() if v not in ("undefined", False)}
    assert reached == 0, (
        f"{reached} connection(s) reached the canary through a SharedWorker (attempt: {attempt})"
    )
    assert present == {}, f"SharedWorker is still there: {types}"


class _CloseAllInside:
    """Fault injection for row A3 on a real browser: the next call (once armed) of a Playwright ``Page`` method,
    made by the driver inside ``ensure()``, closes every page of the context, as a human closing the window at
    that instant. ``"close"`` is the cap close of the oldest page while adopting popups (the target closes, then
    the rest); ``"wait_for_load_state"`` is the adopted page's load wait (all pages close, then the real wait runs
    on the closed page: Playwright rejects only a close that comes *during* the wait, so it returns at once when
    the state already fired, as for the blank popup still loading ``/slow.html``, or expires and is suppressed;
    either way ``ensure()`` goes on to its reopen path, which is the case the fix guards). The patch is on
    the Playwright class, the library's public API; ``fired`` is the premise, so a runtime that stops making the
    call fails the row loudly instead of passing it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, method: str) -> None:
        from playwright.async_api import Page

        self.armed = False
        self.fired = False
        original = getattr(Page, method)
        original_close = Page.close
        hook = self

        async def wrapper(page: Any, *a: Any, **kw: Any) -> Any:
            if not hook.armed:
                return await original(page, *a, **kw)
            hook.armed = False
            ctx = page.context
            if method == "close":
                await original_close(page, *a, **kw)
            for p in list(ctx.pages):
                if not p.is_closed():
                    await original_close(p)
            hook.fired = True
            if method == "close":
                return None
            return await original(page, *a, **kw)

        monkeypatch.setattr(Page, method, wrapper)


@pytest.mark.parametrize("stage", ["adopting", "load-wait"])
async def test_last_page_closed_inside_ensure_is_not_reopened(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Group A review ruling, row A3, the runtime's half (fix A3: ``ensure()`` must not reopen a page in end mode).
    With ``end_when_closed=True`` and the allow-list, the last page closes while ``ensure()`` is inside
    ``frame()``: while adopting four popups over the cap (the cap close of the oldest) or while waiting for an
    adopted popup's load (``/slow.html`` holds its answer). ``frame()`` raises ``BrowserClosedError``; the context
    has no open page afterwards (none reopened); ``closed`` is True and A2's post-conditions hold (every action
    refused with the reason, no new Chromium process). The fault is injected with ``_CloseAllInside``."""
    err = closed_error()
    with fixture_site() as site:
        async with live_executor(
            end_when_closed=True, allowed_origins=frozenset({site.origin})
        ) as ex:
            await run(ex.open(f"{site.origin}/page.html"))
            await run(ex.frame())
            opener = ex.driver.page
            ctx = opener.context
            mains = {p for p, _, _ in chromium_mains()}
            assert len(mains) == 1, mains
            hook = _CloseAllInside(
                monkeypatch, "close" if stage == "adopting" else "wait_for_load_state"
            )
            if stage == "adopting":
                await run(
                    opener.evaluate(
                        "o => { for (let i = 0; i < 4; i++) window.open(o + '/page.html?n=' + i, '_blank'); }",
                        site.origin,
                    )
                )
                want = 5
            else:
                # opened blank, then sent to the held page: Playwright reports a popup opened straight at a
                # held URL only once its first response arrives (seen on this regime), so it would never be
                # adopted mid-load
                await run(
                    opener.evaluate(
                        "u => { const w = window.open('', '_blank'); w.location.href = u; }",
                        f"{site.origin}/slow.html",
                    )
                )
                want = 2
            n = await poll(lambda: asyncio.sleep(0, len(ctx.pages)), lambda v: v == want, 5.0)
            assert n == want, f"the premise failed: {n} pages open, not {want}"
            if stage == "adopting":
                await asyncio.sleep(0.5)  # the popups load
            else:
                assert await poll(lambda: asyncio.sleep(0, site.slow_requests), bool, 5.0), (
                    "the premise failed: the slow popup was never requested"
                )
            assert closed_of(ex) is False
            hook.armed = True
            try:
                got: Any = await run(ex.frame())
            except err:
                got = "closed"
            except Exception as exc:
                got = exc
            site.release.set()
            assert hook.fired, "the premise failed: ensure() made no call the fault rides on"
            await asyncio.sleep(0.3)
            open_pages = [p.url for p in ctx.pages if not p.is_closed()]
            assert got == "closed", (
                f"frame() with the last page closed inside ensure() gave {got!r}; open pages: {open_pages}"
            )
            assert open_pages == [], f"a page was reopened in end mode: {open_pages}"
            await _ended(ex, site.origin, mains)
            assert [p for p in ctx.pages if not p.is_closed()] == [], (
                "a page was opened after the end"
            )


async def test_search_tolerates_plurals_and_any_word() -> None:
    """Group A review ruling, row B-F1 (fix F1: a trailing "s" is stripped, and a query matches when any word
    hits). On the start pages, "mugs" and "blue mugs" each list Blue mug linking to the shop, and so does "blue
    kettle" (one word hits, the other is in no entry). ``__state.results`` names Blue mug too."""
    with start_pages() as srv:
        o = srv.origin
        async with live_executor(allowed_origins=frozenset({o})) as ex:
            bad: list[str] = []
            for q in ("mugs", "blue mugs", "blue kettle"):
                await run(ex.open(f"{o}/search.html?q={urllib.parse.quote_plus(q)}"))
                page = ex.driver.page
                links = await page.evaluate(
                    """() => [...document.querySelectorAll("a[href]")].map((a) => ({ text: a.innerText, href: a.href }))"""
                )
                blue = [r for r in links if "blue mug" in r["text"].casefold()]
                if not (
                    len(blue) == 1
                    and urllib.parse.urlsplit(blue[0]["href"]).path == "/demo/shop.html"
                    and blue[0]["href"].startswith(o)
                ):
                    bad.append(f"{q!r}: Blue mug -> shop not listed once: {links}")
                results = (await _state(page)).get("results")
                if not (isinstance(results, list) and "Blue mug" in results):
                    bad.append(f"{q!r}: __state.results is {results!r}")
    assert not bad, "\n".join(bad)


#: Round-3 search rows (Rulings, "on the group A re-reviews at d2e27fd"): query -> the entry listed first, or
#: ``None`` for no results. The first three are the stop-word rows; "s" is F1c (a term empty after stripping the
#: plural "s" matches nothing); "blue mugs" is F1d (the entry hit by both words ranks first). "the" and "find the
#: page" are the implementer line's "a query that is all stop words gives no results".
SEARCH_FIRST: tuple[tuple[str, str | None], ...] = (
    ("the contact page", "Contact us"),
    ("the teapot", "Kitchen deals"),
    ("contact the shop", "Contact us"),
    ("s", None),
    ("blue mugs", "Blue mug"),
    ("the", None),
    ("find the page", None),
)

_DEMO_LINKS_JS = """() => [...document.querySelectorAll("a[href]")]
  .filter((a) => new URL(a.href).pathname.startsWith("/demo/"))
  .map((a) => ({ text: a.innerText, href: a.href }))"""


@pytest.mark.parametrize(("q", "want"), SEARCH_FIRST, ids=[q for q, _ in SEARCH_FIRST])
async def test_search_ignores_stop_words_and_ranks(q: str, want: str | None) -> None:
    """Round-3 search rows (Rulings, "on the group A re-reviews at d2e27fd"), one case per query of
    ``SEARCH_FIRST``. The first demo link on the results page starts with the expected entry's title and
    ``__state.results[0]`` is that title; a ``None`` case lists no demo link and ``__state.results == []``.
    "First" is read twice because two entries link to the deals page and "Enamel mug on offer" mentions "kitchen
    deals" in its text, so the link's text must *start* with the title."""
    with start_pages() as srv:
        o = srv.origin
        async with live_executor(allowed_origins=frozenset({o})) as ex:
            await run(ex.open(f"{o}/search.html?q={urllib.parse.quote_plus(q)}"))
            page = ex.driver.page
            links = await page.evaluate(_DEMO_LINKS_JS)
            results = (await _state(page)).get("results")
    if want is None:
        assert links == [] and results == [], (
            f"{q!r}: expected no results, got {results!r} / {links}"
        )
        return
    first = links[0]["text"].strip().casefold() if links else ""
    assert isinstance(results, list) and results[:1] == [want], (
        f"{q!r}: __state.results is {results!r}, not {want!r} first"
    )
    assert first.startswith(want.casefold()), (
        f"{q!r}: the first demo link is {first[:40]!r}, not {want!r}"
    )


# --------------------------------------------------------------------------------------------- A4, A5

_CARDS_JS = """() => {
  const isAdd = (el) => ["BUTTON", "A", "INPUT"].includes(el.tagName)
    && (el.tagName !== "INPUT" || ["button", "submit"].includes(el.type))
    && ((el.innerText || el.value || "").trim().toLowerCase() === "add to cart");
  const all = [...document.querySelectorAll("button, a, input")].filter(isAdd);
  return all.map((b, i) => {
    b.setAttribute("data-t0111g-add", String(i));
    let card = b;
    while (card.parentElement && card.parentElement !== document.body
           && [...card.parentElement.querySelectorAll("button, a, input")].filter(isAdd).length === 1) {
      card = card.parentElement;
    }
    const r = card.getBoundingClientRect();
    return { i, text: card.innerText, top: r.top + window.scrollY };
  });
}"""
_CART_LINES_JS = """() => document.body.innerText.replace(/add to cart/gi, "").split("\\n")
  .filter((l) => /cart/i.test(l)).join(" | ")"""
_STATE_JS = "() => JSON.parse(JSON.stringify(window.__state))"
_PRICE = re.compile(r"[$€£]\s?\d+(?:[.,]\d{1,2})?|\d+[.,]\d{2}")


def _price(text: str) -> float:
    found = _PRICE.findall(text)
    assert len(found) == 1, f"a deals card names {len(found)} prices: {text!r}"
    return float(re.sub(r"[^\d.,]", "", found[0]).replace(",", "."))


async def _state(page: Any) -> dict[str, Any]:
    s = await page.evaluate(_STATE_JS)
    assert isinstance(s, dict), f"window.__state is not an object: {s!r}"
    return s


async def _cart_count_shows(page: Any, n: int) -> bool:
    return re.search(rf"\b{n}\b", await page.evaluate(_CART_LINES_JS)) is not None


async def test_start_pages_render_and_state() -> None:
    """A4/A5 in the browser, under the allow-list (A3's "the start-page origin loads"). Each page renders the same
    pixels twice at 1280x800 and exposes ``window.__state`` as an object. ``/`` is one centred search box; typing
    "mug" and Enter opens ``/search.html?q=mug``, whose results all match and link to demo pages, "Blue mug" to the
    shop; a query matching nothing lists no demo link. The shop has three mugs with a visible cart count; adding
    blue then green gives ``["blue mug", "green mug"]``. Contact's Send shows "Thanks, Ada Lovelace" with no request
    and ``__state.sent`` holds the three values. Deals has eight priced items over two viewports, the cheapest
    below the fold; adding it puts exactly it in the cart. Nothing is blocked (``egress_log`` stays empty)."""
    with start_pages() as srv:
        o = srv.origin
        async with live_executor(allowed_origins=frozenset({o})) as ex:
            for path in PAGES:
                await run(ex.open(f"{o}{path}"))
                f1 = await run(ex.frame())
                await run(ex.open(f"{o}{path}"))
                f2 = await run(ex.frame())
                assert (f1.width, f1.height) == (W, H), (path, f1.width, f1.height)
                assert f1.phash == f2.phash and mean_abs_diff(f1, f2) < 1.0, (
                    f"{path} is not deterministic: {mean_abs_diff(f1, f2):.2f}"
                )
                await _state(ex.driver.page)

            # / and search
            await run(ex.open(f"{o}/"))
            page = ex.driver.page
            boxes = await page.evaluate(
                """() => [...document.querySelectorAll("input, textarea")]
                     .filter((e) => e.type !== "hidden" && e.offsetParent !== null)
                     .map((e) => { const r = e.getBoundingClientRect();
                                   return { type: e.type, cx: r.left + r.width / 2, cy: r.top + r.height / 2 }; })"""
            )
            assert len(boxes) == 1 and boxes[0]["type"] in ("text", "search"), boxes
            assert abs(boxes[0]["cx"] - W / 2) <= 8 and 0 < boxes[0]["cy"] < H, boxes
            await page.click("input:not([type=hidden])")
            await page.keyboard.type("mug")
            await page.keyboard.press("Enter")
            await run(page.wait_for_url(re.compile(r".*/search\.html\?.*")))
            u = urllib.parse.urlsplit(page.url)
            assert u.path == "/search.html" and urllib.parse.parse_qs(u.query).get("q") == [
                "mug"
            ], page.url
            results = await page.evaluate(
                """() => [...document.querySelectorAll("a[href]")].map((a) => ({ text: a.innerText, href: a.href }))"""
            )
            demo = [
                r for r in results if urllib.parse.urlsplit(r["href"]).path.startswith("/demo/")
            ]
            assert demo, f"no result links to a demo page: {results}"
            for r in demo:
                assert r["href"].startswith(o), r
                assert "mug" in r["text"].casefold(), f"a result does not match 'mug': {r}"
                assert urllib.parse.urlsplit(r["href"]).path in (
                    "/demo/shop.html",
                    "/demo/contact.html",
                    "/demo/deals.html",
                ), r
            blue = [r for r in demo if "blue mug" in r["text"].casefold()]
            assert (
                len(blue) == 1 and urllib.parse.urlsplit(blue[0]["href"]).path == "/demo/shop.html"
            ), demo
            await run(ex.open(f"{o}/search.html?q=zzqx-no-such-entry"))
            none = await ex.driver.page.evaluate(
                """() => [...document.querySelectorAll("a[href]")].map((a) => a.href)
                     .filter((h) => new URL(h).pathname.startsWith("/demo/"))"""
            )
            assert none == [], f"a query matching nothing lists {none}"

            # the Blue mug result opens the shop
            await run(ex.open(f"{o}/search.html?q=mug"))
            page = ex.driver.page
            await page.click("a[href*='/demo/shop.html']:has-text('lue mug')")
            await run(page.wait_for_url(re.compile(r".*/demo/shop\.html.*")))
            assert (await _state(page)).get("cart") == [], await _state(page)
            cards = await page.evaluate(_CARDS_JS)
            assert len(cards) == 3, cards
            colours = [
                [c for c in ("red", "blue", "green") if c in card["text"].casefold()]
                for card in cards
            ]
            assert sorted(c for cs in colours for c in cs) == ["blue", "green", "red"], colours
            assert await _cart_count_shows(page, 0), await page.evaluate(_CART_LINES_JS)
            for colour, want in (("blue", ["blue mug"]), ("green", ["blue mug", "green mug"])):
                (i,) = [
                    card["i"] for card, cs in zip(cards, colours, strict=True) if cs == [colour]
                ]
                await page.click(f"[data-t0111g-add='{i}']")
                await asyncio.sleep(0.2)
                assert (await _state(page))["cart"] == want, await _state(page)
                assert await _cart_count_shows(page, len(want)), await page.evaluate(_CART_LINES_JS)

            # contact
            await run(ex.open(f"{o}/demo/contact.html"))
            page = ex.driver.page
            fields = await page.evaluate(
                """() => [...document.querySelectorAll("input, textarea")]
                     .filter((e) => !["hidden", "submit", "button"].includes(e.type) && e.offsetParent !== null)
                     .map((e, i) => { e.setAttribute("data-t0111g-field", String(i));
                                      return { i, tag: e.tagName, type: e.type }; })"""
            )
            email = [x["i"] for x in fields if x["tag"] == "INPUT" and x["type"] == "email"]
            name = [x["i"] for x in fields if x["tag"] == "INPUT" and x["type"] != "email"]
            message = [x["i"] for x in fields if x["tag"] == "TEXTAREA"]
            assert (len(email), len(name), len(message)) == (1, 1, 1), fields
            sent = {
                "name": "Ada Lovelace",
                "email": "ada@example.com",
                "message": "Hello from Astra",
            }
            for idx, value in (
                (name[0], sent["name"]),
                (email[0], sent["email"]),
                (message[0], sent["message"]),
            ):
                await page.fill(f"[data-t0111g-field='{idx}']", value)
            requests: list[str] = []
            page.on("request", lambda r: requests.append(r.url))
            await page.click(":is(button, input[type=submit], input[type=button]):text-is('Send')")
            await asyncio.sleep(0.5)
            assert requests == [], (
                f"Send made requests: {[urllib.parse.urlsplit(r).path for r in requests]}"
            )
            assert "Thanks, Ada Lovelace" in await page.evaluate("() => document.body.innerText")
            assert (await _state(page)).get("sent") == sent, await _state(page)

            # deals
            await run(ex.open(f"{o}/demo/deals.html"))
            page = ex.driver.page
            assert (await _state(page)).get("cart") == []
            height = await page.evaluate("() => document.scrollingElement.scrollHeight")
            assert H < height <= 2 * H, (
                f"the deals page is {height} px tall, not over two viewports"
            )
            cards = await page.evaluate(_CARDS_JS)
            assert len(cards) == 8, len(cards)
            prices = [_price(c["text"]) for c in cards]
            cheapest = min(range(8), key=lambda k: prices[k])
            assert prices.count(prices[cheapest]) == 1, prices
            assert cards[cheapest]["top"] >= H, (
                f"the cheapest item is not below the fold: top {cards[cheapest]['top']}"
            )
            await page.click(f"[data-t0111g-add='{cards[cheapest]['i']}']")
            await asyncio.sleep(0.2)
            cart = (await _state(page))["cart"]
            assert len(cart) == 1 and cart[0].casefold() in cards[cheapest]["text"].casefold(), (
                cart,
                cards[cheapest],
            )
            assert list(ex.egress_log) == [], (
                f"a start page attempted a blocked request: {ex.egress_log}"
            )
