"""T-0203 group A (browser): the real ``PlaywrightDriver.follow_navigation`` against real Chromium on loopback
(docs/tasks/T-0203.md, A1-A4, "Tests to write").

**Gate.** As T-0111c's module (its autouse gate imported): ``integration`` + ``browser``, runs only with
``ASTRA_BROWSER_TESTS=1``. A missing ``playwright`` or Chromium is then a failure naming P1 of T-0111c.

**Fixtures.** ``tests/fixtures/t0203/`` (``MANIFEST.md`` there), served by this module's own loopback
``http.server`` on 127.0.0.1. ``search.html`` is the whole viewport in rgb(30, 60, 200) with a white search box whose
centre is (640, 130) (by hand from its CSS); Enter submits its form natively to ``/results.html`` (whole viewport
rgb(230, 120, 20)). The page copies its own ``delay``/``key``/``mode`` query parameters into the form. The server
answers ``/results.html`` after ``delay`` ms (default 600), or holds it until gate ``key`` is released by the row
(never by a timer), then answers per ``mode``: the page; ``download`` (an attachment: Chromium sends
``requestfailed`` ERR_ABORTED then ``download``, no commit); ``close`` (the connection closes with no response:
``requestfailed`` ERR_EMPTY_RESPONSE, then Chromium commits its own error page); ``204``. ``/redirect?to=PATH`` is a
302 (for group B's rows); ``/redirect`` without ``to`` is a 302 to ``/results.html`` with the same query (B8.3).
``mode=stream`` (group B, R0) answers ``/results.html`` with one streamed response: the header band
(``stream_head.html``) after ``ttfb`` ms or once gate ``key`` opens, the results block (``stream_tail.html``)
``delay`` ms later (``DelayHandler._stream``).

**The premise every row checks** (the T-0203 root cause): after ``key_up("Enter")`` and ``settle()``, the results
request is in flight: ``navigated_since(mark)`` is ``None`` and the page URL is still the search page. Reading R1
(below) says this holds on loopback for a native submit.

Activated for group A (the pending markers are off).

Readings measured on loopback with this module's server and fixtures while writing it (n = 5 each, 2026-10-29,
c49b666f): R1, the native submit's ``request`` event 34-35 ms after the first input, ``settle()`` returned at
49-50 ms with ``navigated_since`` ``None`` (5/5). R2, the 600 ms row's main-frame order was ``request`` (34 ms),
``framenavigated`` (638), ``requestfinished`` (639), ``domcontentloaded`` (639-641), 5/5: ``requestfinished`` never
came before ``framenavigated``.

Regime: Playwright 1.64.0, Chromium headless (versions.lock ``[browser]``), 1280x800 at DSF 1, not web mode,
loopback only, n = 1 per row, one browser per row.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import http.server
import io
import threading
import time
import urllib.parse
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from local_astra.computer import runtime as runtime_mod
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
    _Quiet,
    run,
)

pytestmark = [pytest.mark.integration] + (
    [pytest.mark.browser] if _browser_marker_registered() else []
)

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "t0203"
W, H = 1280, 800

#: search.html's box centre and a point of the page well away from it (CSS px, by hand from its stylesheet)
BOX = (640, 130)
SAMPLE = (640, 600)
#: fixture colours, chosen by hand (search.html, results.html)
SEARCH_RGB = (30, 60, 200)
RESULTS_RGB = (230, 120, 20)
#: JPEG tolerance per channel
TOL = 16
#: the default server delay of ``/results.html`` (ms)
DELAY_MS = 600


# --------------------------------------------------------------------------------------------- server


HOLD_MAX_S = 30.0
_GATES: dict[str, threading.Event] = {}
_GATES_LOCK = threading.Lock()


def gate(key: str) -> threading.Event:
    with _GATES_LOCK:
        return _GATES.setdefault(key, threading.Event())


def release_all() -> None:
    with _GATES_LOCK:
        for g in _GATES.values():
            g.set()


class DelayHandler(_Quiet):
    """``_Quiet`` over ``tests/fixtures/t0203`` plus the delayed ``/results.html`` and ``/redirect`` (module
    docstring)."""

    def do_GET(self) -> None:
        u = urllib.parse.urlsplit(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/redirect":
            self.send_response(302)
            # T-0203 B8.3: without ``to``, the 302 goes to ``/results.html`` with this request's own query (the search
            # form submits to ``/redirect`` with its fields; a GET form drops the action's own query)
            self.send_header("Location", q["to"][0] if "to" in q else f"/results.html?{u.query}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if u.path == "/results.html" and q.get("mode", [""])[0] == "stream":
            self._stream(q)
            return
        if u.path == "/results.html":
            key = q.get("key", [""])[0]
            if key:
                gate(key).wait(HOLD_MAX_S)
            else:
                time.sleep(int(q.get("delay", [str(DELAY_MS)])[0]) / 1000)
            mode = q.get("mode", [""])[0]
            if mode == "close":
                self.close_connection = True
                with contextlib.suppress(OSError):
                    self.connection.shutdown(2)
                return
            if mode == "download":
                body = b"t0203 download\n"
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", "attachment; filename=t0203.bin")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if mode == "204":
                self.send_response(204)
                self.end_headers()
                return
        super().do_GET()

    def _stream(self, q: dict[str, list[str]]) -> None:
        """T-0203 B8 (R0, orchestrator 2026-10-29): ``mode=stream`` answers ``/results.html`` with one streamed
        response: after ``ttfb`` ms (default 0), or once gate ``key`` opens (a held response: nothing is sent before),
        the status line, the headers and ``stream_head.html`` (a header band, no results) at once; then, ``delay`` ms
        later (default 600), ``stream_tail.html`` (the results block) and the end of the body (the connection closes:
        no ``Content-Length``)."""
        key = q.get("key", [""])[0]
        if key:
            gate(key).wait(HOLD_MAX_S)
        time.sleep(int(q.get("ttfb", ["0"])[0]) / 1000)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        with contextlib.suppress(OSError):
            self.wfile.write((FIX / "stream_head.html").read_bytes())
            self.wfile.flush()
            time.sleep(int(q.get("delay", [str(DELAY_MS)])[0]) / 1000)
            self.wfile.write((FIX / "stream_tail.html").read_bytes())
            self.wfile.flush()


@pytest.fixture(scope="module")
def delay_server() -> Iterator[str]:
    handler = functools.partial(DelayHandler, directory=str(FIX))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, name="t0203-delay-server", daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        release_all()
        srv.shutdown()
        srv.server_close()
        t.join(5)


# --------------------------------------------------------------------------------------------- rig


@contextlib.asynccontextmanager
async def browser() -> AsyncIterator[Any]:
    d = runtime_mod.PlaywrightDriver(viewport=(W, H), on_egress=lambda e: None, headless=True)
    try:
        await run(d.ensure())
        yield d
    finally:
        with contextlib.suppress(Exception):
            await run(d.close(), 20)


def follow_fn(d: Any) -> Callable[[object, float], Any]:
    fn = getattr(d, "follow_navigation", None)
    assert fn is not None, "T-0203 A1: PlaywrightDriver.follow_navigation does not exist yet"
    return fn  # type: ignore[no-any-return]


async def submit(d: Any, server: str, query: str = "") -> tuple[object, float]:
    """Open ``search.html?<query>``, then click the box, type, press Enter and settle, as the executor's
    ``type_at`` + ``press_enter`` does through the seam. Checks the premise and returns ``(mark, t_enter)``."""
    await run(d.goto(f"{server}/search.html{'?' + query if query else ''}"))
    await run(d.settle())
    mark = d.navigation_mark()
    await run(d.mouse_move(*BOX))
    await run(d.mouse_down())
    await run(d.mouse_up())
    await run(d.type_text("kettle"))
    t_enter = time.monotonic()
    await run(d.key_down("Enter"))
    await run(d.key_up("Enter"))
    await run(d.settle())
    assert d.navigated_since(mark) is None, (
        "premise (R1): the results request is in flight when settle() returns"
    )
    assert "/search.html" in d.page.url, f"premise: still the search page: {d.page.url}"
    return mark, t_enter


async def pixel(d: Any, point: tuple[int, int]) -> tuple[int, int, int]:
    from PIL import Image

    shot = await run(d.screenshot())
    px: Any = Image.open(io.BytesIO(shot)).convert("RGB").getpixel(point)
    return int(px[0]), int(px[1]), int(px[2])


def near(rgb: tuple[int, int, int], want: tuple[int, int, int]) -> bool:
    return all(abs(a - b) <= TOL for a, b in zip(rgb, want, strict=True))


# --------------------------------------------------------------------------------------------- rows


async def test_driver_follows_delayed_submit(delay_server: str) -> None:
    """A2: a native form submit answered after 600 ms. The follow returns ``"loaded"``; the page is the results page,
    ``navigated_since`` is ``True``, and the screenshot taken right after shows the results colour, not the search
    page's. Enter to return within 0.6 s + 1.5 s."""
    async with browser() as d:
        mark, t_enter = await submit(d, delay_server)
        out = await run(follow_fn(d)(mark, 3.0))
        took = time.monotonic() - t_enter
        assert out == "loaded", out
        assert d.navigated_since(mark) is True
        assert "/results.html" in d.page.url, d.page.url
        assert took <= DELAY_MS / 1000 + 1.5, f"Enter to loaded {took:.3f} s"
        rgb = await pixel(d, SAMPLE)
        assert near(rgb, RESULTS_RGB), (
            f"after the follow the screen is {rgb}, not results {RESULTS_RGB}"
        )


async def test_driver_follow_failed(delay_server: str) -> None:
    """A3: the held results response becomes a download when released (Chromium: ``requestfailed`` ERR_ABORTED, then
    ``download``, no commit). The follow, running while it is held, returns ``"ended"`` within 0.3 s of the
    release; ``navigated_since`` is then ``False``; the search page is still showing."""
    key = uuid.uuid4().hex
    try:
        async with browser() as d:
            mark, _ = await submit(d, delay_server, f"mode=download&key={key}")
            task = asyncio.ensure_future(follow_fn(d)(mark, 3.0))
            await asyncio.sleep(0.2)
            assert not task.done(), (
                f"the follow returned {task.result()!r} while the response was held"
            )
            t_release = time.monotonic()
            gate(key).set()
            out = await run(task)
            took = time.monotonic() - t_release
            assert out == "ended", out
            assert took <= 0.3, f"release to 'ended' {took:.3f} s"
            assert d.navigated_since(mark) is False
            assert "/search.html" in d.page.url, d.page.url
    finally:
        gate(key).set()


async def test_driver_follow_failed_connection(delay_server: str) -> None:
    """A3: the held results request fails at the network (the server closes the connection: ``requestfailed``
    ERR_EMPTY_RESPONSE). The follow returns ``"ended"`` within 0.3 s of the release. ``navigated_since`` is not
    asserted: Chromium then commits its own error page, which T-0174 counts as a new document (the report's
    finding on A3/B8.5)."""
    key = uuid.uuid4().hex
    try:
        async with browser() as d:
            mark, _ = await submit(d, delay_server, f"mode=close&key={key}")
            task = asyncio.ensure_future(follow_fn(d)(mark, 3.0))
            await asyncio.sleep(0.2)
            assert not task.done(), (
                f"the follow returned {task.result()!r} while the response was held"
            )
            t_release = time.monotonic()
            gate(key).set()
            out = await run(task)
            took = time.monotonic() - t_release
            assert out == "ended", out
            assert took <= 0.3, f"release to 'ended' {took:.3f} s"
    finally:
        gate(key).set()


async def test_driver_follow_held_timeout(delay_server: str) -> None:
    """A4: the results response is held past the budget (0.5 s): ``"timeout"`` within 0.5 s + 0.1 s, not before
    0.5 s; ``navigated_since`` is still ``None`` and the search page still shows. Released afterwards, the page then
    loads (the follow left nothing behind that blocks it)."""
    key = uuid.uuid4().hex
    try:
        async with browser() as d:
            mark, _ = await submit(d, delay_server, f"key={key}")
            t0 = time.monotonic()
            out = await run(follow_fn(d)(mark, 0.5))
            took = time.monotonic() - t0
            assert out == "timeout", out
            assert 0.5 - 0.02 <= took <= 0.5 + 0.1, f"a 0.5 s budget took {took:.3f} s"
            assert d.navigated_since(mark) is None
            assert "/search.html" in d.page.url, d.page.url
            gate(key).set()
            deadline = time.monotonic() + 5.0
            while "/results.html" not in d.page.url and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            assert "/results.html" in d.page.url, d.page.url
            assert d.navigated_since(mark) is True
    finally:
        gate(key).set()
