"""T-0174 B1-1/B1-2 (browser): ``resolved["navigated"]`` through the real executor and Chromium (docs/tasks/T-0174.md,
"Tests to write", B1).

**Gate.** As T-0111c's module (its autouse gate imported): ``integration`` + ``browser``, runs only with
``ASTRA_BROWSER_TESTS=1`` (``make browser``; the implementer appends this file to that target). A missing
``playwright`` or Chromium is then a failure naming P1 of T-0111c.

**Fixtures.** ``tests/fixtures/t0174/nav/`` (``MANIFEST.md`` there), served by this module's own loopback
``http.server`` on 127.0.0.1. ``index.html`` is a fixed 4x4 grid of 300x180 cells whose centres are x 160/480/800/1120,
y 100/300/500/700 (by hand from its CSS). Every page sets ``window.__doc`` to a random token at load, so "the same
document is still showing" is a premise the rows can check from the page.

**The meaning pinned** (B1-1, H-T174-1): for an ``executed`` ``click_at``/``type_at``/``press_key``,
``navigated`` is ``True`` iff the active page after the settle shows a different main-frame document than at
dispatch: a cross-document main-frame commit after dispatch began and before the settle returned (a link, a form
that assigns ``location``, ``location.reload()``, ``history.back()`` to the previous document), or a switch to a
newly opened page. ``False`` for a fragment change, ``pushState``/``replaceState`` (and ``history.back()`` across a
pushed entry), an iframe navigation, a form that refuses its input, a popup that opens after the settle returned, and
a commit that landed before the dispatch began. Every ``False`` row first proves its premise: the URL changed
(hash, pushState, replaceState), so neither a URL comparison nor Playwright's bare ``framenavigated`` is a passing
implementation; or a commit really happened elsewhere (iframe, stale).

**The capped popup** (reported as an ambiguity): with the runtime's cap (the newest new page becomes active, the
oldest pages close), a popup closed by the cap without ever becoming active only exists when one action opens more
than ``MAX_PAGES`` pages, and then the newest of them *is* the active page, so B1-1's main clause gives ``True``.
``test_five_popups_cap`` pins that, with ``closed_pages`` reported; the cap-only ``False`` case is pinned at the seam in
``tests/unit/test_t0174_navigated_executor.py`` (a ``page_closed`` event and no commit).

**Nothing the page can observe** (B1-2): ``test_no_new_page_script`` wraps Playwright's main-world script surface
(``evaluate``, init scripts, script tags, bindings, ``wait_for_function``) and the CDP sessions' ``send`` for the
whole life of a browser, from the launch on. Every script that reaches a page must be one BASE already ran
(sha256 of ``runtime._RAF2``, ``_FOCUSED_JS``, ``_IS_FOCUSED_JS``, ``_NO_WEBRTC`` at 0dfa2f4c, literals below), and no
CDP call may run script (``Runtime.evaluate`` and kin). The page also watches its own DOM and global names.

Activated for group B1 (the pending markers are off); the BASE-key row, the scroll row and the no-script row hold at BASE and
are active.

T-0203 B7 (2026-10-29) supersedes B1-1b (c): ``test_navigation_pending_at_read_is_absent`` is re-pinned (the hold
outlasts the follow) and ``test_navigation_pending_at_read_followed`` is its sibling (released inside the budget:
``True``).

Regime: Playwright 1.64.0, Chromium 156 headless (versions.lock ``[browser]``), 1280x800 at DSF 1, not web mode,
loopback only, n = 1, one browser per row.
"""

from __future__ import annotations

import asyncio
import functools
import gc
import hashlib
import html
import http.server
import threading
import urllib.parse
import uuid
import weakref
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
    _Quiet,
    coord,
    execute,
    executor,
    inv,
    poll,
    run,
)

pytestmark = [pytest.mark.integration] + (
    [pytest.mark.browser] if _browser_marker_registered() else []
)

REPO = Path(__file__).resolve().parents[2]
NAV = REPO / "tests" / "fixtures" / "t0174" / "nav"
W, H = 1280, 800

#: index.html cell centres (CSS px, by hand from its stylesheet)
LINK = (160, 100)
HASH = (480, 100)
PUSH = (800, 100)
REPLACE = (1120, 100)
RELOAD = (160, 300)
BLANK = (480, 300)
POPUP = (800, 300)
FIVE = (1120, 300)
LATE = (160, 500)
BACK = (480, 500)
IFRAME = (960, 500)
FIELD = (160, 700)
GO = (480, 700)
#: target.html: the back button's centre, and an inert corner
TARGET_BACK = (640, 400)
TARGET_INERT = (100, 700)

#: BASE ``resolved`` keys of an executed action outside web mode (executor.py at 0dfa2f4c: ``_plan``, ``_report``)
BASE_KEYS = frozenset(
    {"x_px", "y_px", "text", "press_enter", "combo", "direction", "dx_px", "dy_px", "frame_after"}
    | {"blocked", "dismissed", "closed_pages"}
)

#: sha256 of the scripts BASE runs in a page (runtime.py at 0dfa2f4c): ``_RAF2`` (settle), ``_FOCUSED_JS`` and
#: ``_IS_FOCUSED_JS`` (focused_field), ``_NO_WEBRTC`` (the one init script outside web mode)
BASE_SCRIPTS = frozenset(
    {
        "6e40611ecc91fdc46adea63e760b73aa8eca60db9027da8dc4e6c4c73852a9cd",
        "b5cbd31f402531743e704db68ca5422b6b2451c3892fff251dc371a757cdb8bf",
        "2c33b88f179b90f6d1d73da96325fc6b30976e51d290b99f75837a1b4327b338",
        "d8d605fa3836014c4e276ccf5c8b7fee0d2774757457485b7d47ffaf4098055f",
    }
)
#: CDP methods that run script in a page or prepare it to (none may be sent by the driver)
SCRIPT_CDP = frozenset(
    {
        "Runtime.evaluate",
        "Runtime.callFunctionOn",
        "Runtime.compileScript",
        "Runtime.runScript",
        "Runtime.addBinding",
        "Page.addScriptToEvaluateOnNewDocument",
        "Page.createIsolatedWorld",
    }
)


# --------------------------------------------------------------------------------------------- rig


#: B1-1b's slow server: a held response waits on its gate, released by the row (never by a timer), so the timing
#: does not depend on load. ``HOLD_MAX_S`` only frees a server thread whose row died without releasing.
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


#: ``/held_body.html?key=K``: the head and the visible body arrive at once (the main frame commits), then a
#: parser-blocking ``<script src="/held.js?key=K">`` holds DOMContentLoaded until gate K opens
HELD_BODY = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>T-0174 B1-1b held body</title>
<style>html, body {{ margin: 0; overflow: hidden; background: rgb(0, 160, 0); }}
#top {{ height: 800px; font: 28px sans-serif; }}</style></head>
<body><div id="top">held body</div>
<script>window.__doc = Math.random().toString(36).slice(2);</script>
<script src="/held.js?key={key}"></script>
<p>tail</p>
</body></html>
"""
#: ``/launch.html?to=URL``: the whole viewport is one plain link to ``to``
LAUNCH = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>T-0174 B1-1b launch</title>
<style>html, body {{ margin: 0; overflow: hidden; }}
#go {{ position: fixed; inset: 0; display: block; background: rgb(200, 200, 0); }}</style></head>
<body><a id="go" href="{to}">go</a>
<script>window.__doc = Math.random().toString(36).slice(2);</script>
</body></html>
"""


class _NavHandler(_Quiet):
    """``_Quiet`` over ``tests/fixtures/t0174/nav`` plus the held responses (B1-1b): ``/held.js`` and
    ``/held_head.html`` send nothing (no status line, no header) until their gate opens; ``/held_body.html`` and
    ``/launch.html`` are generated."""

    def _send(self, body: str, ctype: str) -> None:
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        u = urllib.parse.urlsplit(self.path)
        q = urllib.parse.parse_qs(u.query)
        key = q.get("key", [""])[0]
        if u.path == "/held.js":
            gate(key).wait(HOLD_MAX_S)
            self._send("window.__heldRan = true;", "text/javascript")
        elif u.path == "/held_head.html":
            gate(key).wait(HOLD_MAX_S)
            self._send(
                "<!doctype html><html><head><meta charset='utf-8'><title>held head</title></head>"
                "<body><script>window.__doc = 'held-head';</script></body></html>",
                "text/html; charset=utf-8",
            )
        elif u.path == "/held_body.html":
            self._send(HELD_BODY.format(key=html.escape(key)), "text/html; charset=utf-8")
        elif u.path == "/launch.html":
            self._send(LAUNCH.format(to=html.escape(q["to"][0])), "text/html; charset=utf-8")
        else:
            super().do_GET()


@pytest.fixture(scope="module")
def nav_server() -> Iterator[str]:
    handler = functools.partial(_NavHandler, directory=str(NAV))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, name="t0174-nav-server", daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        release_all()
        srv.shutdown()
        srv.server_close()
        t.join(5)


def at(frame: Any, name: str, point: tuple[int, int], **kw: Any) -> Any:
    return inv(name, frame, x=coord(point[0], W), y=coord(point[1], H), **kw)


def navigated(out: Any) -> bool:
    """B1-1: an executed click/type/press carries a ``bool`` ``navigated``; a named failure while it does not."""
    assert out.status == "executed", out
    assert "navigated" in out.resolved, (
        f"T-0174 B1-1: an executed {out.action!r} carries no resolved['navigated'] yet: {out.resolved}"
    )
    value = out.resolved["navigated"]
    assert isinstance(value, bool), f"B1-1: navigated must be a bool, got {value!r}"
    return value


async def doc(ex: Any) -> str:
    return str(await ex.driver.page.evaluate("window.__doc"))


async def opened(ex: Any, url: str) -> Any:
    await run(ex.open(url))
    return await run(ex.frame())


# --------------------------------------------------------------------------------------------- True rows


async def test_plain_link_is_navigated(nav_server: str) -> None:
    """B1-1 / W1-1's positive half: a click on a plain link loads ``target.html``: ``navigated`` is ``True``."""
    async with executor() as ex:
        f = await opened(ex, f"{nav_server}/index.html")
        out = await execute(ex, at(f, "click_at", LINK))
        assert ex.driver.page.url.endswith("/target.html?from=link"), "premise: the link loaded"
        assert navigated(out) is True, out.resolved


async def test_type_enter_on_valid_input_is_navigated(nav_server: str) -> None:
    """B1-1 (``type_at``): text plus Enter in the form submits, the page assigns ``location``: ``True``."""
    async with executor() as ex:
        f = await opened(ex, f"{nav_server}/index.html")
        out = await execute(ex, at(f, "type_at", FIELD, text="kettle", press_enter=True))
        assert ex.driver.page.url.endswith("/target.html?q=kettle"), ex.driver.page.url
        assert navigated(out) is True, out.resolved


async def test_type_without_enter_then_press_key_enter(nav_server: str) -> None:
    """B1-1 (``type_at`` without Enter is ``False``: nothing loads; ``press_key`` ``Enter`` then submits: ``True``)."""
    async with executor() as ex:
        f = await opened(ex, f"{nav_server}/index.html")
        before = await doc(ex)
        out = await execute(ex, at(f, "type_at", FIELD, text="kettle", press_enter=False))
        assert await doc(ex) == before, "premise: typing loaded nothing"
        assert navigated(out) is False, out.resolved
        out = await execute(ex, inv("press_key", None, combo="Enter"))
        assert ex.driver.page.url.endswith("/target.html?q=kettle"), ex.driver.page.url
        assert navigated(out) is True, out.resolved


async def test_reload_is_navigated(nav_server: str) -> None:
    """B1-1 (adversarial against a URL comparison): ``location.reload()`` keeps the URL and loads a new document:
    ``True``."""
    async with executor() as ex:
        f = await opened(ex, f"{nav_server}/index.html")
        url, before = ex.driver.page.url, await doc(ex)
        out = await execute(ex, at(f, "click_at", RELOAD))
        assert ex.driver.page.url == url and await doc(ex) != before, (
            "premise: same URL, new document"
        )
        assert navigated(out) is True, out.resolved


async def test_back_to_previous_document_is_navigated(nav_server: str) -> None:
    """B1-1 (back/forward): link to ``target.html`` (``True``), then its ``history.back()`` button returns to
    ``index.html``, a different document from the one at dispatch: ``True``."""
    async with executor() as ex:
        f = await opened(ex, f"{nav_server}/index.html")
        out = await execute(ex, at(f, "click_at", LINK))
        assert navigated(out) is True, out.resolved
        f = await run(ex.frame())
        out = await execute(ex, at(f, "click_at", TARGET_BACK))
        assert ex.driver.page.url.endswith("/index.html"), ex.driver.page.url
        assert navigated(out) is True, out.resolved


#: Under load Playwright can announce a popup after the action's settle returned; the runtime then adopts it at the
#: next ``ensure()`` (BASE, T-0111c's late popup). A row that needs the popup adopted *within* the action retries
#: up to this many fresh browsers, and on every attempt still checks the spec: when the active page did not change
#: (and nothing committed on it) the answer must be ``False``.
POPUP_TRIES = 4


def assert_not_adopted(ex: Any, opener: Any, out: Any) -> None:
    """An attempt whose popup arrived after the settle: the opener is still active, so B1-1 says ``False``."""
    assert ex.driver.page is opener
    assert navigated(out) is False, (
        f"the popup was not adopted, the opener is unchanged: {out.resolved}"
    )


async def test_popup_is_navigated(nav_server: str) -> None:
    """H-T174-1: a ``target=_blank`` link opens a page that the settle makes active: ``True`` (retried under load,
    ``POPUP_TRIES``)."""
    for _ in range(POPUP_TRIES):
        async with executor() as ex:
            f = await opened(ex, f"{nav_server}/index.html")
            opener = ex.driver.page
            out = await execute(ex, at(f, "click_at", POPUP))
            if ex.driver.page is opener:
                assert_not_adopted(ex, opener, out)
                continue
            assert ex.driver.page.url.endswith("/target.html?from=popup"), ex.driver.page.url
            assert navigated(out) is True, out.resolved
            return
    pytest.fail(f"premise: the popup was never adopted within the action in {POPUP_TRIES} tries")


async def test_five_popups_cap(nav_server: str) -> None:
    """B1-1 with the ``MAX_PAGES`` cap (module docstring): one click opens five pages; the cap closes the opener and
    the first popup, which never became active (``closed_pages`` reported); the fifth is active, so ``navigated`` is
    ``True`` by H-T174-1. Retried under load (``POPUP_TRIES``) until all five arrived within the action; any attempt
    where a popup became active is ``True``, one where none did is ``False``."""
    for _ in range(POPUP_TRIES):
        async with executor() as ex:
            f = await opened(ex, f"{nav_server}/index.html")
            opener = ex.driver.page
            out = await execute(ex, at(f, "click_at", FIVE))
            if ex.driver.page is opener:
                assert_not_adopted(ex, opener, out)
                continue
            assert navigated(out) is True, f"a popup became active: {out.resolved}"
            if not ex.driver.page.url.endswith("&i=5"):
                continue  # only some popups arrived within the action: retry for the cap premise
            assert out.resolved.get("closed_pages", 0) >= 1, (
                f"premise: the cap closed pages: {out.resolved}"
            )
            open_urls = [p.url for p in ex.driver.page.context.pages]
            assert not any(u.endswith("&i=1") for u in open_urls), (
                f"premise: popup 1 was capped: {open_urls}"
            )
            return
    pytest.fail(
        f"premise: the five popups never all arrived within the action in {POPUP_TRIES} tries"
    )


# --------------------------------------------------------------------------------------------- False rows


async def test_form_refuses_empty_input_not_navigated(nav_server: str) -> None:
    """W1-1's case: Go on an empty field; the page refuses it (like ``web.html``) and nothing loads: ``False``."""
    async with executor() as ex:
        f = await opened(ex, f"{nav_server}/index.html")
        before = await doc(ex)
        out = await execute(ex, at(f, "click_at", GO))
        assert await ex.driver.page.evaluate("window.__refused") == 1, "premise: the form refused"
        assert await doc(ex) == before, "premise: the same document"
        assert navigated(out) is False, out.resolved


SAME_DOCUMENT = [
    pytest.param(HASH, "hashes", "#section", id="hash-link"),
    pytest.param(PUSH, "pushes", "?pushed=1", id="pushState"),
    pytest.param(REPLACE, "replaces", "?replaced=1", id="replaceState"),
]


@pytest.mark.parametrize(("point", "counter", "suffix"), SAME_DOCUMENT)
async def test_same_document_change_not_navigated(
    nav_server: str, point: tuple[int, int], counter: str, suffix: str
) -> None:
    """B1-1 (adversarial, the named failure mode: Playwright's ``framenavigated`` fires for these too): a fragment
    link, ``pushState`` and ``replaceState`` change the URL of the same document: ``False``. Premises: the URL did
    change, and the page's own counter saw the change in this document."""
    async with executor() as ex:
        f = await opened(ex, f"{nav_server}/index.html")
        url, before = ex.driver.page.url, await doc(ex)
        out = await execute(ex, at(f, "click_at", point))
        seen = await poll(
            lambda: ex.driver.page.evaluate(f"window.__nav.{counter}"), lambda v: v == 1
        )
        assert seen == 1, f"premise: the page saw its {counter} change ({seen})"
        assert ex.driver.page.url != url and ex.driver.page.url.endswith(suffix), ex.driver.page.url
        assert await doc(ex) == before, "premise: the same document"
        assert navigated(out) is False, out.resolved


async def test_back_over_pushed_entry_not_navigated(nav_server: str) -> None:
    """B1-1: ``pushState`` (``False``), then ``history.back()`` returns to the entry before it, in the same document
    (a ``popstate``): ``False``."""
    async with executor() as ex:
        f = await opened(ex, f"{nav_server}/index.html")
        before = await doc(ex)
        out = await execute(ex, at(f, "click_at", PUSH))
        assert navigated(out) is False, out.resolved
        f = await run(ex.frame())
        out = await execute(ex, at(f, "click_at", BACK))
        pops = await poll(lambda: ex.driver.page.evaluate("window.__nav.pops"), lambda v: v == 1)
        assert pops == 1 and await doc(ex) == before, "premise: a popstate in the same document"
        assert ex.driver.page.url.endswith("/index.html"), ex.driver.page.url
        assert navigated(out) is False, out.resolved


async def test_iframe_navigation_not_navigated(nav_server: str) -> None:
    """B1-1: a link inside an iframe navigates the iframe only (a subframe commit): ``False``."""
    async with executor() as ex:
        f = await opened(ex, f"{nav_server}/index.html")
        before = await doc(ex)
        out = await execute(ex, at(f, "click_at", IFRAME))
        inner = await poll(
            lambda: asyncio.sleep(0, [fr.url for fr in ex.driver.page.frames[1:]]),
            lambda urls: any(u.endswith("/frame_target.html") for u in urls),
        )
        assert any(u.endswith("/frame_target.html") for u in inner), (
            f"premise: the iframe moved: {inner}"
        )
        assert await doc(ex) == before, "premise: the main frame kept its document"
        assert navigated(out) is False, out.resolved


async def test_blank_click_not_navigated(nav_server: str) -> None:
    """B1-1: a click on an inert area: ``False`` (the key is present, not absent: the capability is there)."""
    async with executor() as ex:
        f = await opened(ex, f"{nav_server}/index.html")
        out = await execute(ex, at(f, "click_at", BLANK))
        assert navigated(out) is False, out.resolved


async def test_late_popup_not_navigated(nav_server: str) -> None:
    """B1-1: a popup the page opens after the click's settle returned (the click arms it; the row fires the page's
    own ``__openLate()`` once ``execute`` has returned, so no timer races the load) never became active within the
    action: ``False``. The next action is refused (the active page changed) and carries no key."""
    async with executor() as ex:
        f = await opened(ex, f"{nav_server}/index.html")
        opener = ex.driver.page
        out = await execute(ex, at(f, "click_at", LATE))
        assert len(opener.context.pages) == 1, "premise: no popup within the action"
        assert navigated(out) is False, out.resolved
        assert await opener.evaluate("window.__openLate()") is True, "premise: the page opened it"
        pages = await poll(
            lambda: asyncio.sleep(0, list(opener.context.pages)), lambda ps: len(ps) == 2, 5.0
        )
        assert len(pages) == 2, "premise: the late popup opened"
        nxt = await execute(ex, at(ex.latest_frame, "click_at", BLANK))
        assert nxt.status == "refused" and "navigated" not in nxt.resolved, nxt


async def test_commit_before_dispatch_not_navigated(nav_server: str) -> None:
    """B1-1 (adversarial: stale): ``stale.html`` replaces itself with ``target.html`` 300 ms after load, with no
    input. The row waits for that commit, takes a frame and clicks an inert corner of ``target.html``: the commit
    landed before the dispatch began, so ``False``."""
    async with executor() as ex:
        await run(ex.open(f"{nav_server}/stale.html"))
        url = await poll(
            lambda: asyncio.sleep(0, ex.driver.page.url), lambda u: "from=stale" in u, 5.0
        )
        assert "from=stale" in url, f"premise: the page navigated by itself: {url}"
        await run(ex.driver.page.wait_for_load_state("domcontentloaded"))
        await asyncio.sleep(0.2)
        f = await run(ex.frame())
        out = await execute(ex, at(f, "click_at", TARGET_INERT))
        assert navigated(out) is False, out.resolved


# --------------------------------------------------------------------------------------------- B1-1b: slow commits

#: the launch page's link fills the viewport; the held body's ``#top`` is inert
CENTRE = (640, 400)


def launch_url(server: str, to: str) -> str:
    return f"{server}/launch.html?to={urllib.parse.quote(to, safe='')}"


async def ready_state(ex: Any) -> str:
    return str(await ex.driver.page.evaluate("document.readyState"))


async def onto_held_body(ex: Any, server: str, key: str) -> Any:
    """Launch page, click its link to ``/held_body.html?key=K``: the new document commits at once and its
    DOMContentLoaded is held by gate K (the parser waits on ``/held.js``). Returns the outcome."""
    f = await opened(ex, launch_url(server, f"/held_body.html?key={key}"))
    out = await execute(ex, at(f, "click_at", CENTRE))
    assert "/held_body.html?key=" in ex.driver.page.url, f"premise: committed: {ex.driver.page.url}"
    assert await ready_state(ex) == "loading", "premise: DOMContentLoaded had not fired at return"
    return out


async def test_commit_with_body_pending_is_navigated(nav_server: str) -> None:
    """B1-1b (a): the link's document commits but its DOMContentLoaded is held past the settle's 2 s bound (a
    parser-blocking script the server holds until the row releases it). The active page shows a new main-frame
    document: ``True``. A DOMContentLoaded counter reads ``False`` here."""
    key = uuid.uuid4().hex
    try:
        async with executor() as ex:
            out = await onto_held_body(ex, nav_server, key)
            assert navigated(out) is True, out.resolved
    finally:
        gate(key).set()


async def test_late_dcl_of_an_earlier_commit_not_navigated(nav_server: str) -> None:
    """B1-1b (b): on the held page of (a), an inert click whose own dispatch releases the held script, so the earlier
    document's DOMContentLoaded lands during this click's settle. No new document was committed by this action:
    ``False``. A DOMContentLoaded counter reads ``True`` here (the fact moves to the wrong action)."""
    key = uuid.uuid4().hex
    try:
        async with executor() as ex:
            await onto_held_body(ex, nav_server, key)
            before = await doc(ex)
            f = await run(ex.frame())
            driver = ex.driver
            real_up = driver.mouse_up

            async def up_then_release() -> None:
                await real_up()
                gate(key).set()  # after the mark (taken before the first input), before the settle

            driver.mouse_up = up_then_release
            try:
                out = await execute(ex, at(f, "click_at", CENTRE))
            finally:
                del driver.mouse_up
            assert await ready_state(ex) != "loading", (
                "premise: DOMContentLoaded fired during the action"
            )
            assert await ex.driver.page.evaluate("window.__heldRan === true"), (
                "premise: the script ran"
            )
            assert await doc(ex) == before, "premise: the same document"
            assert navigated(out) is False, out.resolved
    finally:
        gate(key).set()


async def test_navigation_pending_at_read_is_absent(
    nav_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1-1b (c), re-pinned by T-0203 B7 (which supersedes it): a navigation pending at the after-settle read is now
    followed, so the key is absent only when the navigation is *still* pending after the follow. The link's response
    is held (no status line) and released only ``NAV_FOLLOW_S`` (patched to 0.3 s) + 0.5 s after the settle returned,
    past the follow's budget: the key is absent, never a guessed ``False``; the action is executed with BASE's keys.

    The row's instance-level settle wrapper schedules the release after the real settle returned (the post-action
    screenshot then waits for the commit: Chromium gives no frame while the navigation is pending, probed
    2026-10-27). The premise is recorded inside the wrapper: the old document was still showing."""
    monkeypatch.setattr(executor_mod, "NAV_FOLLOW_S", 0.3, raising=False)
    key = uuid.uuid4().hex
    try:
        async with executor() as ex:
            f = await opened(ex, launch_url(nav_server, f"/held_head.html?key={key}"))
            driver = ex.driver
            real_settle = driver.settle
            at_return: list[str] = []
            loop = asyncio.get_running_loop()

            async def settle_then_release_late() -> None:
                await real_settle()
                at_return.append(driver.page.url)
                loop.call_later(0.3 + 0.5, gate(key).set)

            driver.settle = settle_then_release_late
            try:
                out = await execute(ex, at(f, "click_at", CENTRE))
            finally:
                del driver.settle
            assert len(at_return) == 1 and "/launch.html" in at_return[0], (
                f"premise: the response had not committed when the settle returned: {at_return}"
            )
            assert out.status == "executed", out
            await poll(
                lambda: asyncio.sleep(0, ex.driver.page.url), lambda u: "/held_head.html" in u, 5.0
            )
            assert "/held_head.html" in ex.driver.page.url, "premise: the held response came"
            assert "navigated" not in out.resolved, (
                f"T-0203 B7: a navigation still pending after the follow must leave the key absent: {out.resolved}"
            )
            assert set(out.resolved) == {"x_px", "y_px", "frame_after"}, out.resolved
    finally:
        gate(key).set()


async def test_navigation_pending_at_read_followed(nav_server: str) -> None:
    """T-0203 B7's sibling: the same held link, released right after the settle returned (well inside the default
    ``NAV_FOLLOW_S``): the executor follows the navigation and the response says ``navigated: True``. At BASE (no
    follow) the key is absent here: this is the T-0174 B1-1b (c) answer T-0203 supersedes."""
    key = uuid.uuid4().hex
    try:
        async with executor() as ex:
            f = await opened(ex, launch_url(nav_server, f"/held_head.html?key={key}"))
            driver = ex.driver
            real_settle = driver.settle
            at_return: list[str] = []

            async def settle_then_respond() -> None:
                await real_settle()
                at_return.append(driver.page.url)
                gate(key).set()

            driver.settle = settle_then_respond
            try:
                out = await execute(ex, at(f, "click_at", CENTRE))
            finally:
                del driver.settle
            assert len(at_return) == 1 and "/launch.html" in at_return[0], (
                f"premise: the response had not committed when the settle returned: {at_return}"
            )
            assert "/held_head.html" in ex.driver.page.url, ex.driver.page.url
            assert navigated(out) is True, out.resolved
    finally:
        gate(key).set()


# --------------------------------------------------------------------------------------------- B1-1b: other pages


async def test_background_page_commit_not_navigated(nav_server: str) -> None:
    """B1-1b ("a non-active page's commit never counts"): ``opener.html`` opens ``steer.html`` with ``window.open``
    (``True``: the popup becomes active). A click on the popup makes it navigate its *opener*, a background page,
    to ``target.html``. The settle is lengthened (test-side, on the driver instance) until that commit has landed,
    so it is observed before the read. The active page kept its document: ``False``. The first click is retried
    under load (``POPUP_TRIES``) until the popup is adopted within it."""
    for _ in range(POPUP_TRIES):
        async with executor() as ex:
            f = await opened(ex, f"{nav_server}/opener.html")
            opener = ex.driver.page
            out = await execute(ex, at(f, "click_at", CENTRE))
            if ex.driver.page is opener:
                assert_not_adopted(ex, opener, out)
                continue
            await steer_the_opener(ex, opener, out)
            return
    pytest.fail(f"premise: the popup was never adopted within the action in {POPUP_TRIES} tries")


async def steer_the_opener(ex: Any, opener: Any, out: Any) -> None:
    """``test_background_page_commit_not_navigated`` once the popup is active."""
    assert ex.driver.page.url.endswith("/steer.html"), "premise: the popup is active"
    assert navigated(out) is True, out.resolved
    popup = ex.driver.page
    before = await doc(ex)
    f = await run(ex.frame())
    driver = ex.driver
    real_settle = driver.settle

    async def settle_after_opener_commits() -> None:
        await real_settle()
        # inside the executor's 8 s action bound
        await opener.wait_for_url("**/target.html?from=steered", timeout=4_000)
        await opener.wait_for_load_state("domcontentloaded", timeout=2_000)

    driver.settle = settle_after_opener_commits
    try:
        out = await execute(ex, at(f, "click_at", CENTRE))
    finally:
        del driver.settle
    assert opener.url.endswith("/target.html?from=steered"), f"premise: {opener.url}"
    assert ex.driver.page is popup and await doc(ex) == before, (
        "premise: the popup kept its document"
    )
    assert navigated(out) is False, out.resolved


def _closed_page_paths(root: Any, closed: list[Any]) -> list[str]:
    """Where, in the state reachable from ``root``'s attributes, a closed page is still held: through dicts (keys and
    values), sequences, sets, weak containers, partials and closures, and objects of ``local_astra`` classes
    (``__dict__`` and ``__slots__``). Playwright's own objects are not entered: they are not the driver's state."""
    hits: list[str] = []
    seen: set[int] = set()
    stack: list[tuple[str, Any]] = [(f"driver.{k}", v) for k, v in vars(root).items()]
    while stack:
        path, obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if any(obj is p for p in closed):
            hits.append(path)
            continue
        if isinstance(obj, (weakref.WeakKeyDictionary, weakref.WeakValueDictionary)):
            stack += [(f"{path}[key]", k) for k in list(obj.keys())]
            stack += [(f"{path}[value]", v) for v in list(obj.values())]
        elif isinstance(obj, weakref.WeakSet):
            stack += [(f"{path}[item]", v) for v in list(obj)]
        elif isinstance(obj, dict):
            stack += [(f"{path}[key]", k) for k in obj]
            stack += [(f"{path}[{k!r}]", v) for k, v in obj.items()]
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack += [(f"{path}[{i}]", v) for i, v in enumerate(obj)]
        elif isinstance(obj, functools.partial):
            stack += [(f"{path}.args", obj.args), (f"{path}.keywords", obj.keywords)]
        elif callable(obj) and getattr(obj, "__closure__", None):
            stack += [(f"{path}.<cell>", c.cell_contents) for c in obj.__closure__ if c is not None]
        elif type(obj).__module__.startswith("local_astra"):
            if hasattr(obj, "__dict__"):
                stack += [(f"{path}.{k}", v) for k, v in vars(obj).items()]
            for s in getattr(type(obj), "__slots__", ()):
                if hasattr(obj, s):
                    stack.append((f"{path}.{s}", getattr(obj, s)))
    return hits


async def test_closed_pages_leave_no_state(nav_server: str) -> None:
    """B1-1b ("a closed page's state is dropped"): five popups from one click (the cap closes the opener and popup
    1, which had navigated state), then the row closes popup 2 itself and a frame is taken. No closed page is
    reachable from the driver's state. (BASE's own lists drop closed pages too, so this pins the new state.)"""
    async with executor() as ex:
        f = await opened(ex, f"{nav_server}/index.html")
        opener = ex.driver.page
        born: list[Any] = []
        opener.context.on(
            "page", lambda p: born.append(p)
        )  # test-side, Playwright-level: every popup
        out = await execute(ex, at(f, "click_at", FIVE))
        assert out.status == "executed", out
        # under load some popups arrive after the settle: wait for all five, then a frame adopts them (the cap
        # closes the oldest there at the latest)
        await poll(lambda: asyncio.sleep(0, len(born)), lambda n: n == 5, 10.0)
        assert len(born) == 5, f"premise: five popups opened ({len(born)})"
        await run(ex.frame())
        for p in born:
            if not p.is_closed():
                await run(p.wait_for_load_state("domcontentloaded"))
        assert opener.is_closed(), "premise: the cap closed the opener"
        live = [p for p in born if not p.is_closed()]
        second = next((p for p in live if p.url.endswith("&i=2")), None)
        assert second is not None and second is not ex.driver.page, (
            f"premise: {[p.url for p in live]}"
        )
        await run(second.close())
        await run(ex.frame())
        closed = [p for p in [opener, *born] if p.is_closed()]
        assert len(closed) == 3 and opener in closed and second in closed, (
            f"premise: the opener, popup 1 and popup 2 are closed: {len(closed)}"
        )
        gc.collect()
        held = _closed_page_paths(ex.driver, closed)
        assert held == [], f"B1-1b: the driver still holds closed pages at {held}"


# --------------------------------------------------------------------------------------------- B1-2, active at BASE


async def test_scroll_has_no_key(nav_server: str) -> None:
    """B1-1: ``scroll`` is not one of the three tools: no key, whatever the page does."""
    async with executor() as ex:
        await opened(ex, f"{nav_server}/index.html")
        out = await execute(ex, inv("scroll", None, direction="down"))
        assert out.status == "executed" and "navigated" not in out.resolved, out


async def test_base_keys_and_actions_unchanged(nav_server: str) -> None:
    """B1-2: every other ``resolved`` key and every action string is BASE's: a click, a type with Enter and a key
    press, each with exactly BASE's keys (plus at most ``navigated``) and BASE's values (pixels by hand: the
    cell centres)."""
    async with executor() as ex:
        f = await opened(ex, f"{nav_server}/index.html")
        out = await execute(ex, at(f, "click_at", BLANK))
        assert out.action == "clicked at (480, 300)", out.action
        assert set(out.resolved) - {"navigated"} == {"x_px", "y_px", "frame_after"}, out.resolved
        assert (out.resolved["x_px"], out.resolved["y_px"]) == BLANK
        assert out.resolved["frame_after"] == ex.latest_frame.phash
        f = await run(ex.frame())
        out = await execute(ex, at(f, "type_at", FIELD, text="kettle", press_enter=True))
        assert out.action == "clicked at (160, 700) and typed 6 characters and pressed Enter", (
            out.action
        )
        # T-0189 criterion 1 (criterion 7 re-read): the field is a single-line input, so the type replaces its text
        # and ``resolved`` gains exactly ``replaced: True``; every other key is BASE's.
        assert set(out.resolved) - {"navigated", "replaced"} == {
            "x_px",
            "y_px",
            "text",
            "press_enter",
            "frame_after",
        }, out.resolved
        assert out.resolved.get("replaced") is True, (
            f"T-0189 criterion 1: a type_at into the text field carries replaced: True; got {out.resolved}"
        )
        assert (out.resolved["text"], out.resolved["press_enter"]) == ("kettle", True)
        out = await execute(ex, inv("press_key", None, combo="Tab"))
        assert out.action == "pressed Tab", out.action
        assert set(out.resolved) - {"navigated"} == {"combo", "frame_after"}, out.resolved
        assert set(out.resolved) <= BASE_KEYS | {"navigated"}, out.resolved


class _Recorder:
    """Every main-world script and every CDP method the process sends while installed (the test's own reads are
    made with ``paused``)."""

    def __init__(self) -> None:
        self.scripts: list[tuple[str, str]] = []
        self.cdp: list[str] = []
        self.paused = False


def _install(monkeypatch: pytest.MonkeyPatch, rec: _Recorder) -> None:
    from playwright.async_api import (
        BrowserContext,
        CDPSession,
        ElementHandle,
        Frame,
        JSHandle,
        Locator,
        Page,
    )

    def wrap(cls: type, name: str) -> None:
        orig = getattr(cls, name)

        async def recorded(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not rec.paused:
                script = args[0] if args else kwargs.get("expression", kwargs.get("script", ""))
                if not isinstance(script, str):
                    script = repr(script)
                rec.scripts.append((f"{cls.__name__}.{name}", script))
            return await orig(self, *args, **kwargs)

        monkeypatch.setattr(cls, name, recorded)

    for cls, names in (
        (
            Page,
            (
                "evaluate",
                "evaluate_handle",
                "add_init_script",
                "add_script_tag",
                "expose_binding",
                "expose_function",
                "wait_for_function",
                "eval_on_selector",
                "eval_on_selector_all",
            ),
        ),
        (
            Frame,
            (
                "evaluate",
                "evaluate_handle",
                "add_script_tag",
                "wait_for_function",
                "eval_on_selector",
                "eval_on_selector_all",
            ),
        ),
        (BrowserContext, ("add_init_script", "expose_binding", "expose_function")),
        (
            ElementHandle,
            ("evaluate", "evaluate_handle", "eval_on_selector", "eval_on_selector_all"),
        ),
        (JSHandle, ("evaluate", "evaluate_handle")),
        (Locator, ("evaluate", "evaluate_all", "evaluate_handle")),
    ):
        for name in names:
            wrap(cls, name)

    send = CDPSession.send

    async def sent(self: Any, method: str, *args: Any, **kwargs: Any) -> Any:
        if not rec.paused:
            rec.cdp.append(method)
        return await send(self, method, *args, **kwargs)

    monkeypatch.setattr(CDPSession, "send", sent)


async def peek(rec: _Recorder, page: Any, expression: str) -> Any:
    rec.paused = True
    try:
        return await page.evaluate(expression)
    finally:
        rec.paused = False


async def test_no_new_page_script(nav_server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """B1-2: nothing runs in the page to decide navigation. From the launch on, across a blank click, a hash link, a
    ``pushState``, a type with Enter and a link: every script handed to a page is one BASE runs (``BASE_SCRIPTS``),
    no CDP call runs script (``SCRIPT_CDP``), and the page saw no node added and no new global name while it stayed
    the same document."""
    rec = _Recorder()
    _install(monkeypatch, rec)
    async with executor() as ex:
        f = await opened(ex, f"{nav_server}/index.html")
        page = ex.driver.page
        names0 = set(await peek(rec, page, "Object.getOwnPropertyNames(window)"))
        # the parser's own trailing text node may land after the observer starts: compare, do not assume 0
        added0 = await peek(rec, page, "window.__added")
        for point in (BLANK, HASH, PUSH):
            out = await execute(ex, at(f, "click_at", point))
            assert out.status == "executed", out
            f = await run(ex.frame())
        added = await peek(rec, page, "window.__added")
        assert added == added0, f"the page saw {added - added0} node(s) added to its DOM"
        names1 = set(await peek(rec, page, "Object.getOwnPropertyNames(window)"))
        assert names1 == names0, f"new or removed globals: {sorted(names1 ^ names0)}"
        out = await execute(ex, at(f, "type_at", FIELD, text="kettle", press_enter=True))
        assert out.status == "executed", out
        f = await run(ex.frame())
        out = await execute(ex, at(f, "click_at", TARGET_INERT))
        assert out.status == "executed", out
    foreign = sorted(
        {
            (where, script[:80])
            for where, script in rec.scripts
            if hashlib.sha256(script.encode()).hexdigest() not in BASE_SCRIPTS
        }
    )
    assert foreign == [], f"B1-2: scripts BASE never ran reached a page: {foreign}"
    assert rec.scripts, "premise: the recorder saw BASE's settle script"
    ran = sorted(set(rec.cdp) & SCRIPT_CDP)
    assert ran == [], f"B1-2: CDP calls that run script: {ran}"
