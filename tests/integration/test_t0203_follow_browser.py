"""T-0203 group B (browser): the executor follows a navigation still in flight after the settle, through the real
``BrowserToolExecutor`` and Chromium on loopback (docs/tasks/T-0203.md, B8 and B6; "Tests to write").

**Gate.** As T-0111c's module (its autouse gate imported): ``integration`` + ``browser``, runs only with
``ASTRA_BROWSER_TESTS=1``.

**Server and fixtures.** Group A's loopback server (``DelayHandler``/``server`` of
``tests/integration/test_t0203_follow_driver_browser.py``) over ``tests/fixtures/t0203/`` (``MANIFEST.md`` there).
Per the orchestrator's ruling on R0, B8.1-B8.4's results are **streamed**: ``mode=stream`` sends the header band
(``stream_head.html``: a 200 px band rgb(20, 20, 20) over a rgb(245, 245, 245) page) at once, after ``ttfb`` ms or once
the row opens gate ``key``, and the results block (``stream_tail.html``: rgb(230, 120, 20) from y = 200 to 800)
``DELAY_MS`` = 1500 ms later in the same response (600 ms in the task's B8; lengthened by the orchestrator 2026-10-29
because at load 7-9 BASE's capture of the ttfb-0 page sometimes already showed the results, so the BASE premise did
not hold; still inside ``NAV_FOLLOW_S`` = 3 s, and the wall-time bound moves with it: ``DELAY_MS`` + 1.5 s). So a
capture taken at the commit shows the header without the results, and the
rows fail at BASE for the right reason (reading R0 below). Expected pixels are these hand-chosen colours.

**BASE** is the executor at ``BASE_SHA`` (this task's group-B fork: Merge T-0202-b + T-0203-a; it never calls
``follow_navigation``), executed from ``git show`` as T-0200's differential does, over the same runtime. The rows
that state "the row first shows at BASE" run it first, in its own browser, and assert the failure there.

**The spy** (B8.6) wraps Playwright's ``Page.wait_for_load_state``, ``Page.evaluate`` and ``Page.screenshot`` at
class level and records each call with its arguments; ``PlaywrightDriver.follow_navigation`` and ``navigated_since``
are wrapped at class level to record calls, results and answers (the executor looks the capability up on the class).

**Readings (R0, measured 2026-10-29 on this box with these fixtures, BASE 174214db, n = 5 each).** Enter on
``search.html?mode=stream&delay=600``: BASE's ``navigated`` is ``True`` (the header committed before the read) and
its after-frame shows the header band and the empty page where the results go, (245, 245, 245), in 82-99 ms. With
``ttfb=150``: the key is absent (the request is pending at the read), the same header-only frame, 211-217 ms; this is
the field case (21 of 75 Enter calls had no key). A **click** on ``link.html`` to the same streamed page (ttfb
150): the same header-only frame at BASE. A form submitted to ``http://192.0.2.1/x`` (aborted by the fence route):
Chromium commits its error page before the read, BASE reads ``True``, and the white error page costs T-0200's 3 s
blank re-capture (2.99-3.01 s, n = 3).

Regime: Playwright 1.64.0, Chromium headless (versions.lock ``[browser]``), 1280x800 at DSF 1, not web mode, no
``allowed_origins`` (the default loopback fence: an origin set would make the route buffer the streamed response),
loopback only, one browser per run; B8.1 n = 5 (ttfb 150) + 2 (ttfb 0), the others n = 1.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import http.server
import io
import sys
import threading
import time
import types
import urllib.parse
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.computer import runtime as runtime_mod
from tests import _git_range as gr
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
    coord,
    inv,
    run,
)
from tests.integration.test_t0203_follow_driver_browser import (
    BOX,
    FIX,
    DelayHandler,
    H,
    W,
    gate,
    release_all,
)

pytestmark = [pytest.mark.integration] + (
    [pytest.mark.browser] if _browser_marker_registered() else []
)

#: T-0203 B's BASE: HEAD of T-0203-b-follow-exec before the B tests (Merge T-0202-b + T-0203-a). A literal.
BASE_SHA = "174214db3dd35864cb2797fad0ee324d92e756d5"
CLOCK_MS = 1_700_000_020_300

#: stream_head.html / stream_tail.html colours and geometry (CSS px, by hand from their stylesheets)
HEADER_RGB = (20, 20, 20)
PAGE_RGB = (245, 245, 245)
RESULTS_RGB = (230, 120, 20)
LINK_RGB = (200, 200, 0)
NOOP_RGB = (0, 150, 60)
HEADER_AT = (640, 100)
RESULTS_AT = (640, 600)
#: noop_enter.html: an inert point of the page (outside the box at (340..940, 100..160))
INERT = (640, 600)
TOL = 16
#: the streamed results arrive this long after the header (ms); the second chunk's server delay
DELAY_MS = 1500

NOTE_WHAT = "live frame: followed navigation"
NOTE_KEYS = {"what", "path", "result", "late", "waited_ms"}


# --------------------------------------------------------------------------------------------- BASE


_BASE: types.ModuleType | None = None


def base_module() -> types.ModuleType:
    """BASE's executor (no follow), executed from ``git show`` over the current runtime."""
    global _BASE
    if _BASE is None:
        gr.require_commits(BASE_SHA)
        src = gr._git("show", f"{BASE_SHA}:local_astra/computer/executor.py").stdout
        assert "def _capture" in src and "follow_navigation" not in src, (
            "premise: BASE's executor does not follow"
        )
        name = "_t0203b_base_executor_browser"
        mod = types.ModuleType(name)
        mod.__file__ = f"<{BASE_SHA[:8]}>/local_astra/computer/executor.py"
        sys.modules[name] = mod
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        _BASE = mod
    return _BASE


# --------------------------------------------------------------------------------------------- spies


class Spy:
    """Class-level recorders (module docstring)."""

    def __init__(self) -> None:
        self.page: list[tuple[str, str]] = []
        self.follows: list[tuple[float, str]] = []
        self.follow_started: list[float] = []
        self.answers: list[object] = []
        self.after_follow: list[Any] = []

    def reset(self) -> None:
        self.page.clear()
        self.follows.clear()
        self.follow_started.clear()
        self.answers.clear()


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> Iterator[Spy]:
    from playwright.async_api import Page

    s = Spy()

    def wrap_page(name: str) -> None:
        real = getattr(Page, name)

        async def recorded(self: Any, *args: Any, **kwargs: Any) -> Any:
            s.page.append((name, repr((args, sorted(kwargs.items())))))
            return await real(self, *args, **kwargs)

        monkeypatch.setattr(Page, name, recorded)

    for n in ("wait_for_load_state", "evaluate", "screenshot"):
        wrap_page(n)

    cls = runtime_mod.PlaywrightDriver
    real_follow = getattr(cls, "follow_navigation", None)
    assert real_follow is not None, "premise: group A's follow_navigation exists"

    async def follow(self: Any, mark: object, timeout_s: float) -> Any:
        s.follow_started.append(timeout_s)
        out = await real_follow(self, mark, timeout_s)
        s.follows.append((timeout_s, out))
        for fn in s.after_follow:
            fn()
        return out

    monkeypatch.setattr(cls, "follow_navigation", follow)
    real_since = cls.navigated_since

    def since(self: Any, mark: object) -> Any:
        out = real_since(self, mark)
        s.answers.append(out)
        return out

    monkeypatch.setattr(cls, "navigated_since", since)
    yield s


# --------------------------------------------------------------------------------------------- rig


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    """Group A's ``DelayHandler`` over ``tests/fixtures/t0203`` on 127.0.0.1 (this module's own instance)."""
    handler = functools.partial(DelayHandler, directory=str(FIX))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, name="t0203b-server", daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        release_all()
        srv.shutdown()
        srv.server_close()
        t.join(5)


@contextlib.asynccontextmanager
async def executor(module: Any = executor_mod) -> AsyncIterator[tuple[Any, list[dict[str, Any]]]]:
    ex = module.BrowserToolExecutor(now_ms=lambda: CLOCK_MS, viewport=(W, H), headless=True)
    notes: list[dict[str, Any]] = []
    ex.on_note = notes.append
    try:
        yield ex, notes
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ex.close(), 20)


async def opened(ex: Any, url: str) -> Any:
    await run(ex.open(url))
    return await run(ex.frame())


def pixel(jpeg: bytes, point: tuple[int, int]) -> tuple[int, int, int]:
    from PIL import Image

    px: Any = Image.open(io.BytesIO(jpeg)).convert("RGB").getpixel(point)
    return int(px[0]), int(px[1]), int(px[2])


def near(rgb: tuple[int, int, int], want: tuple[int, int, int]) -> bool:
    return all(abs(a - b) <= TOL for a, b in zip(rgb, want, strict=True))


def search_url(server: str, **q: Any) -> str:
    return f"{server}/search.html?{urllib.parse.urlencode(q)}"


def link_url(server: str, to: str) -> str:
    return f"{server}/link.html?to={urllib.parse.quote(to, safe='')}"


async def type_enter(ex: Any, frame: Any) -> tuple[Any, float]:
    i = inv(
        "type_at",
        frame,
        x=coord(BOX[0], W),
        y=coord(BOX[1], H),
        text="kettle",
        press_enter=True,
    )
    t0 = time.monotonic()
    out = await run(ex.execute(i))
    return out, time.monotonic() - t0


async def click(ex: Any, frame: Any, point: tuple[int, int] = (640, 400)) -> tuple[Any, float]:
    t0 = time.monotonic()
    out = await run(ex.execute(inv("click_at", frame, x=coord(point[0], W), y=coord(point[1], H))))
    return out, time.monotonic() - t0


def follow_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [n for n in notes if n.get("what") == NOTE_WHAT]
    for n in out:
        assert set(n) == NOTE_KEYS, f"B5: the note has exactly {sorted(NOTE_KEYS)}: {n}"
        assert n["path"] == "action", n
        assert n["result"] in ("loaded", "ended", "timeout"), n
        assert type(n["late"]) is bool and type(n["waited_ms"]) is int, n
    return out


def shows(ex: Any) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    jpeg = ex.latest_frame.jpeg
    return pixel(jpeg, HEADER_AT), pixel(jpeg, RESULTS_AT)


def gate_at_settle(ex: Any, key: str, when: str) -> tuple[list[str], list[float]]:
    """Make the order of the settle, the commit and the read deterministic (orchestrator 2026-10-29: at load 7-19
    BASE's own settle sometimes saw an un-gated commit and waited for the whole document, so the BASE premise was a
    race). The streamed response is held on gate ``key`` (nothing sent) until the action's settle has returned; the
    instance-level wrapper then opens it. ``when="pending"``: it returns at once, so the read sees the request in flight
    (no await in between: T-0174's B1-1b (c) rig). ``when="committed"``: it waits (polling the page URL, at most 5 s)
    until the header has committed, so the read sees a new document still loading. Returns the URLs seen at the
    settle's return (the premise: the old page was still showing) and the release time."""
    driver = ex.driver
    real_settle = driver.settle
    at_return: list[str] = []
    released: list[float] = []

    async def settle_then_release() -> None:
        await real_settle()
        at_return.append(driver.page.url)
        released.append(time.monotonic())
        gate(key).set()
        if when == "committed":
            end = time.monotonic() + 5.0
            while "/results.html" not in driver.page.url and time.monotonic() < end:
                await asyncio.sleep(0.01)

    driver.settle = settle_then_release
    return at_return, released


async def gated_action(
    module: Any, url_of: Any, action: str, when: str
) -> tuple[Any, list[dict[str, Any]], Any, float, Any]:
    """Open ``url_of(key)`` in a fresh executor of ``module``, gate the response at the settle (``gate_at_settle``),
    act; ``(executor, notes, outcome, seconds, context manager)``; the caller closes the context manager."""
    key = uuid.uuid4().hex
    cm = executor(module)
    ex, notes = await cm.__aenter__()
    try:
        f = await opened(ex, url_of(key))
        at_return, released = gate_at_settle(ex, key, when)
        try:
            out, took = await (type_enter(ex, f) if action == "type" else click(ex, f))
            # the action's own settle may wait its bound for the response the rig holds until it returns (a click
            # always, an Enter under load: the request can start inside the settle); that wait is BASE's settle, not
            # the follow, so the B8 wall time is counted from the release
            took = time.monotonic() - released[0] if released else took
        finally:
            del ex.driver.settle
            gate(key).set()
        assert len(at_return) == 1 and "/results.html" not in at_return[0], (
            f"premise: the results had not committed when the settle returned: {at_return}"
        )
        if when == "committed":
            assert "/results.html" in ex.driver.page.url, (
                "premise: the header committed before the read"
            )
    except BaseException:
        await cm.__aexit__(None, None, None)
        raise
    return ex, notes, out, took, cm


async def base_shows_header_only(url_of: Any, action: str, when: str) -> None:
    """The BASE half of B8.1/B8.2/B8.3 (R0): BASE's after-frame shows the header band without the results."""
    ex, _notes, out, _took, cm = await gated_action(base_module(), url_of, action, when)
    try:
        assert out.status == "executed", out
        hdr, res = shows(ex)
        assert near(hdr, HEADER_RGB) and near(res, PAGE_RGB), (
            f"premise (R0): BASE's after-frame is the streamed page's header without its results "
            f"(header {hdr}, results region {res}); the row would not fail at BASE for the right reason"
        )
    finally:
        await cm.__aexit__(None, None, None)


async def followed(spy: Spy, url_of: Any, action: str, when: str) -> Any:
    """The current executor's half: B8.1's three assertions and B5's note."""
    spy.reset()
    ex, notes, out, took, cm = await gated_action(executor_mod, url_of, action, when)
    try:
        await check_followed(ex, notes, out, took)
        return str(ex.driver.page.url)
    finally:
        await cm.__aexit__(None, None, None)


async def check_followed(ex: Any, notes: list[dict[str, Any]], out: Any, took: float) -> None:
    """B8.1's three assertions, plus B5's note."""
    assert out.status == "executed", out
    hdr, res = shows(ex)
    assert near(res, RESULTS_RGB), (
        f"T-0203 B8: the after-frame's results region is {res}, not the results colour {RESULTS_RGB} "
        f"(header {hdr}); the screenshot did not wait for the new page"
    )
    assert out.resolved.get("navigated") is True, out.resolved
    assert took <= DELAY_MS / 1000 + 1.5, (
        f"B8: the call took {took:.3f} s (bound {DELAY_MS / 1000 + 1.5:.1f} s); notes {notes}"
    )
    fn = follow_notes(notes)
    assert [n["result"] for n in fn] == ["loaded"], f"B5: one 'loaded' follow note: {notes}"
    assert fn[0]["late"] is False, fn


# --------------------------------------------------------------------------------------------- B8.1


@pytest.mark.parametrize(
    "when",
    [pytest.param("pending", id=f"pending-{k}") for k in range(5)]
    + [pytest.param("committed", id=f"committed-{k}") for k in range(2)],
)
async def test_b8_1_enter_streamed_results_followed(server: str, spy: Spy, when: str) -> None:
    """B8.1: Enter in the search box submits natively to the streamed results (the header once the settle returned,
    the results ``DELAY_MS`` later). First, at BASE, the after-frame is the header without the results (R0). Then the
    after-frame's results region is the results colour, ``navigated`` is ``True``, and the call returns at most
    ``DELAY_MS`` + 1.5 s after the release (``gated_action``). ``pending``: the request is in flight at the read (the field's absent key); ``committed``:
    the header committed before the read (``True``, document still loading)."""

    def url_of(key: str) -> str:
        return search_url(server, mode="stream", delay=DELAY_MS, key=key)

    await base_shows_header_only(url_of, "type", when)
    await followed(spy, url_of, "type", when)
    assert len(spy.follows) == 1, f"B1: one follow (no late re-check after True): {spy.follows}"


# --------------------------------------------------------------------------------------------- B8.2


async def test_b8_2_click_link_to_streamed_results(server: str, spy: Spy) -> None:
    """B8.2: a click on a link to the same streamed page (pending at the read). First, at BASE, the after-frame is the
    header without the results (R0); then the same three assertions as B8.1."""

    def url_of(key: str) -> str:
        return link_url(server, f"/results.html?mode=stream&delay={DELAY_MS}&key={key}")

    await base_shows_header_only(url_of, "click", "pending")
    await followed(spy, url_of, "click", "pending")


# --------------------------------------------------------------------------------------------- B8.3


async def test_b8_3_redirect_to_streamed_results(server: str, spy: Spy) -> None:
    """B8.3: Enter submits to ``/redirect``, a 302 to the streamed results (held on the second hop until the settle
    returned). BASE first shows the header without results; then the follow reaches the results: the results colour,
    ``True``, on the redirected page."""

    def url_of(key: str) -> str:
        return search_url(server, to="/redirect", mode="stream", delay=DELAY_MS, key=key)

    await base_shows_header_only(url_of, "type", "pending")
    page_url = await followed(spy, url_of, "type", "pending")
    assert "/results.html" in page_url, f"premise: redirected: {page_url}"


# --------------------------------------------------------------------------------------------- B8.4


async def test_b8_4_held_past_cap_times_out(
    server: str, spy: Spy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B8.4: the results response is held (nothing sent) past ``NAV_FOLLOW_S`` (patched to 0.5 s): ``executed``
    within 0.5 s + 1.5 s, no ``navigated`` key, one note with ``result: "timeout"``.

    Chromium gives no screenshot while the main-frame navigation is pending (T-0174's probe, 2026-10-27; R0), so the
    row releases the hold as the follow returns (synchronously: the executor re-reads with no await in between) (or 3 s after the call started, whichever is first, so a
    BASE that never follows is not stuck until ``ACTION_TIMEOUT_S``); the streamed header then commits and the capture
    completes."""
    monkeypatch.setattr(executor_mod, "NAV_FOLLOW_S", 0.5, raising=False)
    key = uuid.uuid4().hex
    loop = asyncio.get_running_loop()
    spy.after_follow.append(gate(key).set)
    try:
        async with executor() as (ex, notes):
            f = await opened(ex, search_url(server, mode="stream", delay=DELAY_MS, key=key))
            backstop = loop.call_later(3.0, gate(key).set)
            out, took = await type_enter(ex, f)
            backstop.cancel()
            assert out.status == "executed", out
            assert took <= 0.5 + 1.5, f"B8.4: {took:.3f} s (bound 2.0 s)"
            assert "navigated" not in out.resolved, (
                f"B7/B8.4: a navigation still pending after the follow leaves the key absent: {out.resolved}"
            )
            fn = follow_notes(notes)
            assert [n["result"] for n in fn] == ["timeout"], f"B8.4/B5: one 'timeout' note: {notes}"
            assert spy.follows and spy.follows[0][1] == "timeout", spy.follows
            assert spy.follows[0][0] <= 0.5 + 1e-9, (
                f"B1: the follow's budget is NAV_FOLLOW_S: {spy.follows}"
            )
    finally:
        gate(key).set()


# --------------------------------------------------------------------------------------------- B8.5


@pytest.mark.parametrize("action", ["type", "click"])
async def test_b8_5_aborted_navigation(
    server: str, spy: Spy, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    """B8.5, re-pinned by the orchestrator's ruling (2): a navigation to ``http://192.0.2.1/x`` (outside the
    loopback fence, aborted by the route; no packet leaves) makes Chromium commit its error page. The follow's
    result is whatever it observed, ``"ended"`` or ``"loaded"`` (never ``"timeout"``), within 0.3 s; the response
    carries T-0174's read (the driver's last ``navigated_since`` answer, here ``True``); the call is not slower than
    BASE's by more than 0.3 s. The blank re-capture budget is patched to 0.3 s in both executors (the error page is
    white, T-0200)."""
    target = "http://192.0.2.1/x"
    url = search_url(server, to=target) if action == "type" else link_url(server, target)

    async def once(module: Any) -> tuple[Any, float, list[dict[str, Any]], Any]:
        monkeypatch.setattr(module, "BLANK_RECAPTURE_ACTION_S", 0.3)
        async with executor(module) as (ex, notes):
            f = await opened(ex, url)
            out, took = await (type_enter(ex, f) if action == "type" else click(ex, f))
            assert {"host": "192.0.2.1", "scheme": "http"} in ex.egress_log, (
                f"premise: the route aborted the target: {ex.egress_log}"
            )
            return out, took, notes, ex.driver.page.url

    _, base_took, _, _ = await once(base_module())
    spy.reset()
    out, took, notes, page_url = await once(executor_mod)
    assert out.status == "executed", out
    assert page_url.startswith("chrome-error://"), f"premise (ruling 2): the error page: {page_url}"
    fn = follow_notes(notes)
    for n in fn:
        assert n["result"] in ("ended", "loaded"), f"ruling (2): never 'timeout': {n}"
        assert n["waited_ms"] <= 300, f"B8.5: the follow ended within 0.3 s: {n}"
    for _budget, result in spy.follows:
        assert result in ("none", "ended", "loaded"), spy.follows
    bools = [a for a in spy.answers if isinstance(a, bool)]
    assert spy.answers and spy.answers[-1] is not None, f"premise: T-0174 can tell: {spy.answers}"
    assert out.resolved.get("navigated") is bools[-1], (
        f"ruling (2): the response carries T-0174's read {bools[-1]!r}: {out.resolved}"
    )
    assert took <= base_took + 0.3, f"B8.5: {took:.3f} s against BASE's {base_took:.3f} s"


# --------------------------------------------------------------------------------------------- B8.6 / B6


async def _negative_trace(
    module: Any, spy: Spy, action: str, server: str
) -> tuple[list[Any], list[Any], Any]:
    spy.reset()
    async with executor(module) as (ex, notes):
        f = await opened(ex, f"{server}/noop_enter.html")
        spy.reset()
        if action == "type":
            out, _ = await type_enter(ex, f)
        else:
            out, _ = await click(ex, f, INERT)
        assert "/noop_enter.html" in ex.driver.page.url, "premise: nothing navigated"
        assert near(pixel(ex.latest_frame.jpeg, INERT), NOOP_RGB), "premise: the same page"
        rec = (list(spy.page), [out.status, out.action, out.reason, sorted(out.resolved)])
        return rec[0], [rec[1], list(notes)], list(spy.follow_started)


@pytest.mark.parametrize("action", ["type", "click"])
async def test_b8_6_no_navigation_base_calls(server: str, spy: Spy, action: str) -> None:
    """B8.6 / B6 (negative): Enter in a box whose ``keydown`` calls ``preventDefault``, and a click on an inert area
    of the same page. The Playwright-level spy records exactly BASE's ``wait_for_load_state``/``evaluate``/
    ``screenshot`` calls (T-0204 B10: plus one ``screenshot`` with a ``clip``, the pre-click guard's crop; T-0204 C10: plus
    one more capture, the visual settle's second shot), the outcome
    and its keys are BASE's, there is no follow note, and ``follow_navigation`` is never awaited (B2's extra read for a twice-``False`` Enter is synchronous, not a Playwright call)."""
    base_calls, base_out, base_follows = await _negative_trace(base_module(), spy, action, server)
    calls, got_out, follows = await _negative_trace(executor_mod, spy, action, server)
    assert base_follows == [], "premise: BASE never follows"
    assert follows == [], f"B6/B8.6: follow_navigation was awaited: {follows}"
    assert got_out == base_out, (
        f"B6: the outcome or a note differs from BASE: {got_out} vs {base_out}"
    )
    # T-0204 B10 supersedes "exact BASE Playwright calls" for the real driver's click_at/type_at: BASE plus one
    # ``screenshot`` with a ``clip`` (the pre-click guard's region capture)
    clipped = [i for i, c in enumerate(calls) if c[0] == "screenshot" and "'clip'" in c[1]]
    assert len(clipped) == 1, (
        f"T-0204 B10: exactly one clipped screenshot (the guard's crop): {calls}"
    )
    rest = [c for i, c in enumerate(calls) if i != clipped[0]]
    # T-0204 C10 supersedes it again: BASE plus one more ``screenshot`` per kept action capture on a static page (the
    # visual settle's second shot, right after BASE's capture; more only on a changing page)
    last = max(i for i, c in enumerate(base_calls) if c[0] == "screenshot")
    want = [*base_calls[: last + 1], base_calls[last], *base_calls[last + 1 :]]
    assert rest == want, (
        f"B6/B8.6 + T-0204 B10/C10: Playwright calls differ from BASE {BASE_SHA[:8]} plus one clipped "
        f"screenshot and one more capture:\n got  {calls}\n base {base_calls}"
    )
    assert any(c[0] == "screenshot" for c in rest), "premise: the spy saw the capture"
