"""T-0203 group A: ``PlaywrightDriver.follow_navigation(mark, timeout_s) -> FollowResult`` on the real driver class
without a browser (docs/tasks/T-0203.md, A1-A6, "Tests to write").

**The rig.** A ``PlaywrightDriver`` is built, never launched. A ``FakePage`` is adopted through the driver's own
``_adopt`` (so the driver's real ``request``/``requestfinished``/``requestfailed``/``framenavigated``/
``domcontentloaded``/``download``/``close`` listeners are the ones registered) and made the active page. The rows
fire Playwright's page events by calling the registered handlers, in the order real Chromium emits them on loopback
(readings R2, ``tests/integration/test_t0203_follow_driver_browser.py``: ``request``, ``framenavigated``,
``requestfinished``, ``domcontentloaded``). The fake models one fact of a real page: ``wait_for_load_state(
"domcontentloaded")`` returns at once while the *current* document has reached it (the old document, before a
commit: the T-0203 root cause) and waits otherwise; ``evaluate`` is scripted per call. Every other page attribute
used as a call is recorded in ``page.unexpected``.

**What is pinned** (from the criteria, never from an implementation):

- A1: the method and ``FollowResult = Literal["none", "loaded", "ended", "timeout"]`` exported; ``"none"`` at once
  (< 5 ms, zero page calls) when nothing started after the mark, for a foreign mark, another generation, or no active
  page. A commit counted *before* the mark is not followed even while its DOMContentLoaded is outstanding.
- A2: no page call before the commit (a ``wait_for_load_state`` on the old document returns at once: the bug); after
  it ``wait_for_load_state("domcontentloaded", timeout <= the budget)`` then ``evaluate(_RAF2)`` (the only script);
  a redirect hop and a document replaced during the wait are waited for again; woken by the listeners, never by
  ``asyncio.sleep`` polling (``runtime.asyncio`` is a spy that records every positive sleep).
- A3: ``"ended"`` within 20 ms of a ``requestfailed``, a ``download`` or the page's ``close``; ``navigated_since`` is
  ``False`` after the failure.
- A4: ``timeout_s + 0.1`` s at most; ``"timeout"`` leaves ``navigated_since`` at ``None``; a Playwright error while
  time remains is ``"ended"``, never raised; ``CancelledError`` propagates.
- A5 (adversarial): a main-frame request started *before* the mark and never finished costs nothing.
- A6 (negative, green at BASE by design): ``settle()``'s page calls, the listeners' effects on ``commits``,
  ``requests`` and ``awaiting_dcl``, and ``navigated_since``'s answers on the same event sequences.

Rows red at BASE carried ``pending(task="T-0203")`` until the activation commit (activated for group A).

Regime: CPU, no browser, no socket; real asyncio time, budgets of 0.2-3.0 s.
"""

from __future__ import annotations

import asyncio
import time
import typing
from collections import deque
from collections.abc import Callable
from typing import Any

import pytest
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from local_astra.computer import runtime as runtime_mod

#: ``runtime._RAF2`` at BASE c49b666f, as a literal (the only script the follow may evaluate, A6)
RAF2 = "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))"
#: Playwright's message when a navigation destroys the context an ``evaluate`` ran in (``runtime._navigated_away``)
NAV_AWAY = "Execution context was destroyed, most likely because of a navigation"
CLOSED = "Target page, context or browser has been closed"

#: A1: "at once"; A3: "within one event-loop turn" read as 20 ms (the task's own figure for the failed row)
AT_ONCE_S = 0.005
WAKE_S = 0.020
#: A4: the bound past ``timeout_s``
SLACK_S = 0.1

ORIGIN = "http://127.0.0.1:8123"


# --------------------------------------------------------------------------------------------- the fake page


class FakeFrame:
    def __init__(self, url: str) -> None:
        self.url = url


class FakeRequest:
    def __init__(self, url: str, frame: FakeFrame, navigation: bool = True) -> None:
        self.url, self.frame, self._navigation = url, frame, navigation

    def is_navigation_request(self) -> bool:
        return self._navigation


class FakeDownload:
    async def cancel(self) -> None:
        return None


#: an ``evaluate`` behaviour: a callable run when the call arrives, before it resolves
Behaviour = Callable[["FakePage"], Any]


class FakePage:
    """See the module docstring. ``log`` interleaves the events the row fired and the calls the driver made."""

    def __init__(self, url: str, log: list[tuple[str, ...]] | None = None) -> None:
        self.main_frame = FakeFrame(url)
        self.handlers: dict[str, list[Callable[..., Any]]] = {}
        self.log: list[tuple[str, ...]] = [] if log is None else log
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.unexpected: list[str] = []
        self.closed = False
        #: the current document has reached DOMContentLoaded
        self.loaded = asyncio.Event()
        self.loaded.set()
        self.evaluate_plan: deque[str] = deque()
        self.on_evaluate: dict[int, Behaviour] = {}
        self.load_state_error: BaseException | None = None
        self.load_state_error_late = False
        self.tasks: list[asyncio.Task[Any]] = []

    # -- what the driver may use

    @property
    def url(self) -> str:
        return self.main_frame.url

    def on(self, event: str, handler: Callable[..., Any]) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def is_closed(self) -> bool:
        return self.closed

    async def wait_for_load_state(self, state: str = "load", timeout: float | None = None) -> None:
        self.calls.append(("wait_for_load_state", (state,), {"timeout": timeout}))
        self.log.append(("call", "wait_for_load_state", state))
        if self.load_state_error is not None:
            if self.load_state_error_late and timeout is not None:
                # a Playwright error that arrives only after the time the follow gave the call has passed
                await asyncio.sleep(timeout / 1000 + 0.03)
            raise self.load_state_error
        if self.loaded.is_set():
            return
        try:
            await asyncio.wait_for(self.loaded.wait(), None if timeout is None else timeout / 1000)
        except TimeoutError:
            raise PlaywrightTimeoutError(f"Timeout {timeout}ms exceeded.") from None

    async def evaluate(self, expression: str, arg: Any = None) -> Any:
        n = len([c for c in self.calls if c[0] == "evaluate"])
        self.calls.append(("evaluate", (expression,), {}))
        self.log.append(("call", "evaluate", expression))
        if n in self.on_evaluate:
            self.on_evaluate[n](self)
        plan = self.evaluate_plan.popleft() if self.evaluate_plan else "ok"
        if plan == "ok":
            await asyncio.sleep(0.002)
            return True
        if plan == "hang":
            await asyncio.Event().wait()
        if plan == "away":
            raise PlaywrightError(NAV_AWAY)
        if plan == "closed":
            raise PlaywrightError(CLOSED)
        raise AssertionError(f"unknown evaluate plan {plan!r}")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        self.unexpected.append(name)

        def refuse(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError(f"T-0203: the follow used page.{name}")

        return refuse

    # -- what the row fires

    def emit(self, event: str, arg: Any) -> None:
        self.log.append(("event", event))
        for handler in list(self.handlers.get(event, [])):
            out = handler(arg)
            if asyncio.iscoroutine(out):
                # Playwright runs a coroutine listener as its own task
                self.tasks.append(asyncio.ensure_future(out))

    def request(
        self, path: str, navigation: bool = True, frame: FakeFrame | None = None
    ) -> FakeRequest:
        req = FakeRequest(ORIGIN + path, frame or self.main_frame, navigation)
        self.emit("request", req)
        return req

    def finished(self, req: FakeRequest) -> None:
        self.emit("requestfinished", req)

    def failed(self, req: FakeRequest) -> None:
        self.emit("requestfailed", req)

    def commit(self, path: str) -> None:
        """A main-frame cross-document commit: a new document that has not reached DOMContentLoaded."""
        self.main_frame.url = ORIGIN + path
        self.loaded.clear()
        self.emit("framenavigated", self.main_frame)

    def dcl(self) -> None:
        self.loaded.set()
        self.emit("domcontentloaded", self)

    def download(self) -> None:
        self.emit("download", FakeDownload())

    def close(self) -> None:
        self.closed = True
        self.emit("close", self)

    # -- reading

    def page_calls(self) -> list[str]:
        return [c[0] for c in self.calls] + self.unexpected

    def scripts(self) -> list[str]:
        return [c[1][0] for c in self.calls if c[0] == "evaluate"]


# --------------------------------------------------------------------------------------------- rig


def driver() -> Any:
    return runtime_mod.PlaywrightDriver(viewport=(1280, 800), on_egress=lambda e: None)


def rig(path: str = "/search.html") -> tuple[Any, FakePage]:
    """A driver "after launch" (generation 1) whose active page is a loaded ``FakePage`` at ``path``."""
    d = driver()
    page = FakePage(ORIGIN + path)
    d._adopt(page)
    d._page = page
    d.generation = 1
    return d, page


def follow_fn(d: Any) -> Callable[[object, float], Any]:
    fn = getattr(d, "follow_navigation", None)
    assert fn is not None, "T-0203 A1: PlaywrightDriver.follow_navigation does not exist yet"
    return typing.cast(Callable[[object, float], Any], fn)


def start(d: Any, mark: object, timeout_s: float) -> asyncio.Task[Any]:
    return asyncio.ensure_future(follow_fn(d)(mark, timeout_s))


async def timed(d: Any, mark: object, timeout_s: float) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = await follow_fn(d)(mark, timeout_s)
    return out, time.perf_counter() - t0


async def result_within(task: asyncio.Task[Any], seconds: float, what: str) -> Any:
    """The follow's result, which must arrive within ``seconds`` of now."""
    t0 = time.perf_counter()
    done, _ = await asyncio.wait({task}, timeout=seconds)
    waited = time.perf_counter() - t0
    if not done:
        await asyncio.wait({task}, timeout=5.0)
        raise AssertionError(
            f"T-0203: the follow had not returned {seconds * 1000:.0f} ms after {what} "
            f"(then: {task.result() if task.done() and not task.exception() else task})"
        )
    assert waited <= seconds
    return task.result()


async def still_running(task: asyncio.Task[Any], seconds: float, why: str) -> None:
    await asyncio.sleep(seconds)
    assert not task.done(), (
        f"T-0203: the follow returned {task.result() if not task.exception() else task.exception()!r} "
        f"while {why}"
    )


class _AsyncioSpy:
    """``runtime.asyncio`` replaced: everything is asyncio's, but every ``sleep(d > 0)`` is recorded (A2: woken by
    the listeners, not by polling)."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []
        #: every ``wait()`` on an ``asyncio.Event`` the runtime created, and every ``wait_for`` it started (a wait
        #: loop's iterations, whichever primitive it uses)
        self.waits = 0
        spy = self

        class CountingEvent(asyncio.Event):
            async def wait(self) -> Any:
                spy.waits += 1
                return await super().wait()

        self.Event = CountingEvent

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)

    async def sleep(self, delay: float, result: Any = None) -> Any:
        if delay > 0:
            self.sleeps.append(delay)
        return await asyncio.sleep(delay, result)

    async def wait_for(self, aw: Any, timeout: float | None) -> Any:
        self.waits += 1
        return await asyncio.wait_for(aw, timeout)


@pytest.fixture
def no_poll(monkeypatch: pytest.MonkeyPatch) -> _AsyncioSpy:
    spy = _AsyncioSpy()
    monkeypatch.setattr(runtime_mod, "asyncio", spy)
    return spy


def first_call_index(page: FakePage) -> int:
    return next((i for i, e in enumerate(page.log) if e[0] == "call"), len(page.log))


def index_of(page: FakePage, entry: tuple[str, ...], start_at: int = 0) -> int:
    for i in range(start_at, len(page.log)):
        if page.log[i] == entry:
            return i
    raise AssertionError(f"{entry} not in {page.log[start_at:]}")


# --------------------------------------------------------------------------------------------- A1: the surface


def test_follow_result_exported() -> None:
    """A1: ``FollowResult`` is exported, ``Literal["none", "loaded", "ended", "timeout"]``; the method is async."""
    assert "FollowResult" in runtime_mod.__all__, (
        "T-0203 A1: FollowResult is not in runtime.__all__"
    )
    fr = getattr(runtime_mod, "FollowResult", None)
    assert typing.get_origin(fr) is typing.Literal, fr
    assert set(typing.get_args(fr)) == {"none", "loaded", "ended", "timeout"}, typing.get_args(fr)
    fn = getattr(runtime_mod.PlaywrightDriver, "follow_navigation", None)
    assert fn is not None and asyncio.iscoroutinefunction(fn), "A1: an async method"


# --------------------------------------------------------------------------------------------- A1: "none" at once


async def test_follow_none_without_pending() -> None:
    """A1: nothing started after the mark (and the page fully loaded): ``"none"`` in < 5 ms, zero page calls."""
    d, page = rig()
    mark = d.navigation_mark()
    out, took = await timed(d, mark, 3.0)
    assert out == "none", out
    assert took < AT_ONCE_S, f"A1: 'none' took {took * 1000:.1f} ms"
    assert page.page_calls() == [], page.page_calls()


async def test_follow_none_after_completed_navigation() -> None:
    """A1: a navigation that started, committed and reached DOMContentLoaded *before* the follow is called is done:
    ``"none"`` at once (only the outstanding part is followed)."""
    d, page = rig()
    mark = d.navigation_mark()
    req = page.request("/results.html?q=a")
    page.commit("/results.html?q=a")
    page.finished(req)
    page.dcl()
    assert d.navigated_since(mark) is True, "premise: committed"
    out, took = await timed(d, mark, 3.0)
    assert out == "none", out
    assert took < AT_ONCE_S, f"A1: 'none' took {took * 1000:.1f} ms"
    assert page.page_calls() == [], page.page_calls()


async def test_follow_none_without_active_page() -> None:
    """A1: no active page: ``"none"`` at once (never the ``page`` property's ``RuntimeError``)."""
    d, page = rig()
    page.request("/results.html?q=a")
    d._page = None
    mark = d.navigation_mark()
    out, took = await timed(d, mark, 3.0)
    assert out == "none", out
    assert took < AT_ONCE_S
    assert page.page_calls() == []


async def test_foreign_mark_none() -> None:
    """A1: a mark this driver did not issue (any object; another driver's mark), while this driver's page has a
    request in flight that started after its own mark: ``"none"`` at once, zero page calls."""
    d, page = rig()
    other, _other_page = rig()
    other_mark = other.navigation_mark()
    page.request("/results.html?q=a")
    for mark in (object(), None, "mark", other_mark):
        out, took = await timed(d, mark, 3.0)
        assert out == "none", (mark, out)
        assert took < AT_ONCE_S, (mark, took)
    assert page.page_calls() == []


async def test_other_generation_none() -> None:
    """A1: a mark from an earlier generation (the browser relaunched since): ``"none"`` at once."""
    d, page = rig()
    mark = d.navigation_mark()
    d.generation += 1
    page.request("/results.html?q=a")
    out, took = await timed(d, mark, 3.0)
    assert out == "none", out
    assert took < AT_ONCE_S
    assert page.page_calls() == []


async def test_commit_before_mark_awaiting_dcl_is_none() -> None:
    """A1 (the commit half is "commits after ``mark``"): a document committed *before* the mark whose
    DOMContentLoaded is still outstanding is not this action's navigation: ``"none"`` at once (T-0174 B1-1b (b)'s
    case: an earlier commit's late DOMContentLoaded belongs to no later action)."""
    d, page = rig()
    page.request("/held.html")
    page.commit("/held.html")
    mark = d.navigation_mark()
    out, took = await timed(d, mark, 3.0)
    assert out == "none", out
    assert took < AT_ONCE_S
    assert page.page_calls() == []


# --------------------------------------------------------------------------------------------- A2: commit, DCL, RAF2


async def test_follow_commit_then_dcl_raf2(no_poll: _AsyncioSpy) -> None:
    """A2: a main-frame request after the mark, pending. Nothing is called on the page before the commit (the old
    document's ``wait_for_load_state`` returns at once: the bug); after the commit, ``wait_for_load_state(
    "domcontentloaded", timeout <= the budget)``; after the document's DOMContentLoaded, ``evaluate(_RAF2)`` and
    ``"loaded"``, within 20 ms of that event, with no polling sleep. ``navigated_since`` is then ``True``."""
    d, page = rig()
    mark = d.navigation_mark()
    req = page.request("/results.html?q=kettle")
    task = start(d, mark, 3.0)
    await still_running(task, 0.05, "the request was pending (no commit)")
    assert page.page_calls() == [], f"A2: page calls before the commit: {page.page_calls()}"
    at_commit = len(page.log)
    page.commit("/results.html?q=kettle")
    page.finished(req)  # R2's loopback order: framenavigated, requestfinished, domcontentloaded
    await still_running(task, 0.05, "the committed document had not reached DOMContentLoaded")
    assert first_call_index(page) >= at_commit, page.log
    waits = [c for c in page.calls if c[0] == "wait_for_load_state"]
    assert waits and waits[0][1] == ("domcontentloaded",), page.calls
    timeout_ms = waits[0][2]["timeout"]
    assert timeout_ms is not None and 0 < timeout_ms <= 3000, (
        f"A2: bounded by the time left: {timeout_ms}"
    )
    at_dcl = len(page.log)
    page.dcl()
    out = await result_within(task, WAKE_S, "DOMContentLoaded (+ the fake's 2 ms frame)")
    assert out == "loaded", out
    assert page.scripts() == [RAF2], f"A2/A6: the only script is _RAF2: {page.scripts()}"
    index_of(page, ("call", "evaluate", RAF2), at_dcl)
    assert page.unexpected == [], page.unexpected
    assert no_poll.sleeps == [], f"A2: polling sleeps {no_poll.sleeps}"
    assert d.navigated_since(mark) is True


async def test_follow_awaiting_dcl_only(no_poll: _AsyncioSpy) -> None:
    """A1/A2: the commit came after the mark, before the follow, and its DOMContentLoaded has not (``navigated_since``
    already ``True``): the follow waits for that document's DOMContentLoaded, then ``_RAF2``: ``"loaded"``."""
    d, page = rig()
    mark = d.navigation_mark()
    req = page.request("/results.html?q=kettle")
    page.commit("/results.html?q=kettle")
    page.finished(req)
    assert d.navigated_since(mark) is True, "premise: committed"
    task = start(d, mark, 3.0)
    await still_running(task, 0.05, "the committed document had not reached DOMContentLoaded")
    assert page.scripts() == [], "A2: _RAF2 before DOMContentLoaded"
    at_dcl = len(page.log)
    page.dcl()
    out = await result_within(task, WAKE_S, "DOMContentLoaded")
    assert out == "loaded", out
    index_of(page, ("call", "evaluate", RAF2), at_dcl)
    assert page.scripts() == [RAF2]
    assert no_poll.sleeps == []


async def test_follow_redirect_hop(no_poll: _AsyncioSpy) -> None:
    """A2: a redirect: request A finishes (the 3xx) with no commit, request B starts; the follow keeps waiting
    through the hop and returns ``"loaded"`` only after B's commit and DOMContentLoaded."""
    d, page = rig()
    mark = d.navigation_mark()
    ra = page.request("/redirect?to=results")
    task = start(d, mark, 3.0)
    await asyncio.sleep(0.02)
    page.finished(ra)
    rb = page.request("/results.html?q=kettle")
    await still_running(task, 0.05, "the redirect's second hop was pending")
    page.commit("/results.html?q=kettle")
    page.finished(rb)
    await still_running(task, 0.03, "B had not reached DOMContentLoaded")
    page.dcl()
    out = await result_within(task, WAKE_S, "DOMContentLoaded")
    assert out == "loaded", out
    assert page.scripts() == [RAF2]
    assert no_poll.sleeps == []


async def test_follow_navigated_away_retry(no_poll: _AsyncioSpy) -> None:
    """A2 (``settle()``'s ``_navigated_away`` rule): during the follow's ``_RAF2`` a navigation replaces the document
    (request C, commit C) and ``evaluate`` raises "Execution context was destroyed"; time remains, so C is waited
    for: its DOMContentLoaded, then ``_RAF2`` again: ``"loaded"``."""
    d, page = rig()
    mark = d.navigation_mark()
    req = page.request("/results.html?q=kettle")
    task = start(d, mark, 3.0)
    await asyncio.sleep(0.02)

    def replaced(p: FakePage) -> None:
        rc = p.request("/results.html?q=kettle&page=again")
        p.commit("/results.html?q=kettle&page=again")
        p.finished(rc)

    page.on_evaluate[0] = replaced
    page.evaluate_plan.extend(["away", "ok"])
    page.commit("/results.html?q=kettle")
    page.finished(req)
    page.dcl()
    await still_running(task, 0.05, "the replacing document had not reached DOMContentLoaded")
    assert page.scripts() == [RAF2], page.scripts()
    at_dcl = len(page.log)
    page.dcl()
    out = await result_within(task, WAKE_S, "the replacing document's DOMContentLoaded")
    assert out == "loaded", out
    assert page.scripts() == [RAF2, RAF2], page.scripts()
    index_of(page, ("call", "evaluate", RAF2), at_dcl)
    assert no_poll.sleeps == []


# --------------------------------------------------------------------------------------------- A3: ended


async def test_follow_failed_ends_fast(no_poll: _AsyncioSpy) -> None:
    """A3: the followed request fails: ``"ended"`` within 20 ms of ``requestfailed``; ``navigated_since`` is then
    ``False``, not ``None``."""
    d, page = rig()
    mark = d.navigation_mark()
    req = page.request("/results.html?q=kettle")
    task = start(d, mark, 3.0)
    await still_running(task, 0.05, "the request was pending")
    page.failed(req)
    out = await result_within(task, WAKE_S, "requestfailed")
    assert out == "ended", out
    assert d.navigated_since(mark) is False
    assert page.scripts() == []
    assert no_poll.sleeps == []


async def test_follow_download_ends(no_poll: _AsyncioSpy) -> None:
    """A3: a download starts on the page while the request is pending: ``"ended"`` within 20 ms of the ``download``
    event (before any ``requestfailed``; on loopback Chromium sends ``requestfailed`` ERR_ABORTED about 1 ms
    *before* the download, R2's probe, so the event is fired alone here to pin the download path itself). After the
    request's ``requestfailed``, ``navigated_since`` is ``False``."""
    d, page = rig()
    mark = d.navigation_mark()
    req = page.request("/file.bin")
    task = start(d, mark, 3.0)
    await still_running(task, 0.05, "the request was pending")
    page.download()
    out = await result_within(task, WAKE_S, "download")
    assert out == "ended", out
    page.failed(req)
    assert d.navigated_since(mark) is False
    assert no_poll.sleeps == []


async def test_follow_page_close_ends(no_poll: _AsyncioSpy) -> None:
    """A3: the page closes while its request is pending: ``"ended"`` within 20 ms of ``close``, no page call."""
    d, page = rig()
    mark = d.navigation_mark()
    page.request("/results.html?q=kettle")
    task = start(d, mark, 3.0)
    await still_running(task, 0.05, "the request was pending")
    page.close()
    out = await result_within(task, WAKE_S, "close")
    assert out == "ended", out
    assert page.page_calls() == []
    assert no_poll.sleeps == []


# --------------------------------------------------------------------------------------------- A4: bounded


async def test_follow_timeout_bounded() -> None:
    """A4: a request that never answers: ``"timeout"`` after the budget (not before it: it was following), within
    ``timeout_s + 0.1`` s; ``navigated_since`` is still ``None`` (never a guessed ``False``)."""
    d, page = rig()
    mark = d.navigation_mark()
    page.request("/results.html?q=kettle")
    out, took = await timed(d, mark, 0.3)
    assert out == "timeout", out
    assert 0.3 - 0.02 <= took <= 0.3 + SLACK_S, f"A4: took {took:.3f} s for a 0.3 s budget"
    assert d.navigated_since(mark) is None
    assert page.page_calls() == []
    assert not d._nav_waiters, "A4: a finished follow left its waiter registered"


async def test_follow_quiet_wait_does_not_spin(no_poll: _AsyncioSpy) -> None:
    """A2 (woken, not polling; adversarial against a wake-up that is never re-armed): the follow is woken once by
    a redirect hop (request A finishes, request B starts and stays pending), then nothing happens for 200 ms. In
    that quiet window it waits: at most 2 wait starts (one pending wait, one spare), never a loop at full CPU."""
    d, page = rig()
    mark = d.navigation_mark()
    ra = page.request("/redirect?to=results")
    task = start(d, mark, 3.0)
    await asyncio.sleep(0.02)
    page.finished(ra)
    page.request("/results.html?q=kettle")
    await asyncio.sleep(0.02)  # the wake-up from the hop has been taken
    try:
        before = no_poll.waits
        cpu0 = time.thread_time()
        await still_running(task, 0.2, "B was pending")
        cpu = time.thread_time() - cpu0
        quiet = no_poll.waits - before
        assert quiet <= 2, f"A2: {quiet} wait starts in 200 ms with no event (a spinning wait loop)"
        assert cpu < 0.1, f"A2: {cpu:.3f} s of CPU in 200 ms with no event (a spinning wait loop)"
        assert no_poll.sleeps == []
    finally:
        # end the follow (B commits and loads); a spinning loop may swallow a cancel, so never rely on one here
        page.commit("/results.html?q=kettle")
        page.dcl()
        await asyncio.wait({task}, timeout=1.0)
        if not task.done():
            task.cancel()


async def test_follow_exception_after_deadline_is_timeout() -> None:
    """A4: a Playwright error that arrives after the time is up (the DOMContentLoaded wait raises a non-timeout
    error only once the time it was given has passed) ends the follow with ``"timeout"``, never ``"ended"``,
    within ``timeout_s + 0.1`` s; nothing propagates and no waiter is left."""
    d, page = rig()
    mark = d.navigation_mark()
    req = page.request("/results.html?q=kettle")
    page.load_state_error = PlaywrightError(CLOSED)
    page.load_state_error_late = True
    t0 = time.perf_counter()
    task = start(d, mark, 0.3)
    await asyncio.sleep(0.02)
    page.commit("/results.html?q=kettle")
    page.finished(req)
    out = await result_within(task, 0.3 + SLACK_S - (time.perf_counter() - t0), "the start")
    assert [c[0] for c in page.calls] == ["wait_for_load_state"], page.calls
    assert out == "timeout", f"A4: an error after the deadline must be 'timeout', got {out!r}"
    assert not d._nav_waiters


@pytest.mark.parametrize("stuck", ["dcl", "raf2"])
async def test_follow_timeout_bounded_after_commit(stuck: str) -> None:
    """A4: committed, then stuck: DOMContentLoaded never comes (the fake's ``wait_for_load_state`` honours the
    timeout it is given), or ``_RAF2`` never resolves (``evaluate`` takes no timeout, so the follow must bound it
    itself). Within ``timeout_s + 0.1`` s; neither ``"none"`` nor ``"ended"`` (the result between ``"loaded"`` and
    ``"timeout"`` is not pinned: see the report's ambiguity on A2/A4)."""
    d, page = rig()
    mark = d.navigation_mark()
    req = page.request("/results.html?q=kettle")
    page.commit("/results.html?q=kettle")
    page.finished(req)
    if stuck == "raf2":
        page.evaluate_plan.append("hang")
    t0 = time.perf_counter()
    task = start(d, mark, 0.3)
    if stuck == "raf2":
        await asyncio.sleep(0.02)
        page.dcl()  # while the follow waits: the document loads, then no animation frame ever comes
    out = await result_within(task, 0.3 + SLACK_S - (time.perf_counter() - t0), "the start")
    took = time.perf_counter() - t0
    assert took <= 0.3 + SLACK_S, f"A4: took {took:.3f} s for a 0.3 s budget"
    assert out in ("loaded", "timeout"), out
    if stuck == "raf2":
        assert page.scripts() == [RAF2], page.scripts()
    assert d.navigated_since(mark) is True
    assert not d._nav_waiters


@pytest.mark.parametrize("where", ["wait_for_load_state", "evaluate"])
async def test_follow_exception_swallowed(where: str) -> None:
    """A4: a Playwright error inside the follow while time remains (the target closed) ends it with ``"ended"``;
    nothing propagates."""
    d, page = rig()
    mark = d.navigation_mark()
    req = page.request("/results.html?q=kettle")
    task = start(d, mark, 3.0)
    await asyncio.sleep(0.02)
    if where == "wait_for_load_state":
        page.load_state_error = PlaywrightError(CLOSED)
    else:
        page.evaluate_plan.append("closed")
    page.commit("/results.html?q=kettle")
    page.finished(req)
    page.dcl()
    out = await result_within(task, 0.5, "the error")
    assert out == "ended", out
    assert not d._nav_waiters


@pytest.mark.parametrize("phase", ["before_commit", "in_wait_for_load_state", "in_raf2"])
async def test_follow_cancel_propagates(phase: str) -> None:
    """A4: ``CancelledError`` propagates, whichever await the follow is in."""
    d, page = rig()
    mark = d.navigation_mark()
    req = page.request("/results.html?q=kettle")
    if phase != "before_commit":
        page.commit("/results.html?q=kettle")
        page.finished(req)
    if phase == "in_raf2":
        page.evaluate_plan.append("hang")
    task = start(d, mark, 3.0)
    if phase == "in_raf2":
        await asyncio.sleep(0.02)
        page.dcl()  # while the follow waits
    await still_running(task, 0.05, f"cancel phase {phase}")
    if phase == "in_wait_for_load_state":
        assert [c[0] for c in page.calls] == ["wait_for_load_state"], page.calls
    if phase == "in_raf2":
        assert page.scripts() == [RAF2], page.calls
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1.0)
    assert not d._nav_waiters, "A4: a cancelled follow left its waiter registered"


# --------------------------------------------------------------------------------------------- A5: stale entries


async def test_stale_request_before_mark_costs_nothing() -> None:
    """A5 (adversarial: following any pending entry would make every later action on this page pay the whole cap):
    a main-frame request that started before the mark and never finished. The follow returns ``"none"`` at once,
    every time, with no page call; ``navigated_since`` keeps T-0174's answer, ``None``."""
    d, page = rig()
    page.request("/beacon-like-navigation")
    for _ in range(3):  # three later actions on the same page
        mark = d.navigation_mark()
        out, took = await timed(d, mark, 3.0)
        assert out == "none", out
        assert took < AT_ONCE_S, f"A5: a stale entry cost {took * 1000:.1f} ms"
        assert d.navigated_since(mark) is None, "A5: T-0174's answer is kept"
    assert page.page_calls() == []


async def test_stale_plus_fresh_follows_only_fresh(no_poll: _AsyncioSpy) -> None:
    """A5: a stale pending entry from before the mark and a fresh request after it: the fresh one is followed to
    ``"loaded"`` within 20 ms of its DOMContentLoaded; the stale one, still pending, is not waited for."""
    d, page = rig()
    page.request("/stale")
    mark = d.navigation_mark()
    req = page.request("/results.html?q=kettle")
    task = start(d, mark, 3.0)
    await still_running(task, 0.03, "the fresh request was pending")
    page.commit("/results.html?q=kettle")
    page.finished(req)
    page.dcl()
    out = await result_within(task, WAKE_S, "the fresh document's DOMContentLoaded")
    assert out == "loaded", out
    assert no_poll.sleeps == []


async def test_finished_before_commit_waits_for_commit(no_poll: _AsyncioSpy) -> None:
    """Adversarial (A1/A2): the request's response finished before the commit arrived (``navigated_since`` is then
    T-0174's ``False``, the R2 hazard). It started after the mark, so it is followed: the follow does not return on
    the finished flag, it waits for the commit and DOMContentLoaded: ``"loaded"``."""
    d, page = rig()
    mark = d.navigation_mark()
    req = page.request("/results.html?q=kettle")
    page.finished(req)
    assert d.navigated_since(mark) is False, (
        "premise: T-0174 reads a finished, uncommitted request as False"
    )
    task = start(d, mark, 3.0)
    await still_running(task, 0.05, "the finished request had not committed")
    assert page.page_calls() == []
    page.commit("/results.html?q=kettle")
    page.dcl()
    out = await result_within(task, WAKE_S, "the commit and DOMContentLoaded")
    assert out == "loaded", out
    assert no_poll.sleeps == []


async def test_new_active_page_all_requests_count(no_poll: _AsyncioSpy) -> None:
    """A1: the active page became active after the mark (a popup the settle switched to): all of its requests count,
    including one that started before the mark. It is followed on that page, never on the opener."""
    d, opener = rig()
    popup = FakePage(ORIGIN + "/blank")
    d._adopt(popup)
    preq = popup.request("/results.html?q=popup")  # before the mark
    mark = d.navigation_mark()
    d._page = popup
    task = start(d, mark, 3.0)
    await still_running(task, 0.05, "the popup's request was pending")
    popup.commit("/results.html?q=popup")
    popup.finished(preq)
    popup.dcl()
    out = await result_within(task, WAKE_S, "the popup's DOMContentLoaded")
    assert out == "loaded", out
    assert opener.page_calls() == [], opener.page_calls()
    assert popup.scripts() == [RAF2]
    assert no_poll.sleeps == []


async def test_subframe_and_non_navigation_requests_not_followed() -> None:
    """A1 ("requests of the active page's main frame"): after the mark, an iframe's navigation request and a
    main-frame subresource request, both pending: ``"none"`` at once."""
    d, page = rig()
    mark = d.navigation_mark()
    page.request("/frame.html", frame=FakeFrame(ORIGIN + "/frame.html"))
    page.request("/app.js", navigation=False)
    out, took = await timed(d, mark, 3.0)
    assert out == "none", out
    assert took < AT_ONCE_S
    assert page.page_calls() == []


# --------------------------------------------------------------------------------------------- A6: negatives (BASE)


def nav_state(d: Any, page: FakePage) -> tuple[int, bool, dict[str, bool]]:
    s = d._navigation[page]
    return s.commits, s.awaiting_dcl, dict(s.requests)


def test_listener_effects_unchanged() -> None:
    """A6 (negative, green at BASE): the listeners' effects on ``commits``, ``requests`` and ``awaiting_dcl`` and
    ``navigated_since``'s answers, step by step, computed by hand from T-0174 B1-1b's rules."""
    d, page = rig()
    a = f"{ORIGIN}/a.html"
    mark = d.navigation_mark()
    assert nav_state(d, page) == (0, False, {}) and d.navigated_since(mark) is False
    ra = page.request("/a.html#frag")
    assert nav_state(d, page) == (0, False, {a: False}) and d.navigated_since(mark) is None
    page.request("/sub.js", navigation=False)
    page.request("/f.html", frame=FakeFrame(ORIGIN + "/f.html"))
    assert nav_state(d, page) == (0, False, {a: False})
    page.finished(ra)
    assert nav_state(d, page) == (0, False, {a: True}) and d.navigated_since(mark) is False
    page.commit("/a.html")
    assert nav_state(d, page) == (1, True, {}) and d.navigated_since(mark) is True
    page.dcl()
    assert nav_state(d, page) == (1, False, {})
    mark2 = d.navigation_mark()
    rb = page.request("/b.html")
    page.failed(rb)
    assert nav_state(d, page) == (1, False, {}) and d.navigated_since(mark2) is False
    page.dcl()  # a document no request matched (an error page, a document.open)
    assert nav_state(d, page) == (2, False, {}) and d.navigated_since(mark2) is True


async def test_settle_page_calls_unchanged() -> None:
    """A6 (negative, green at BASE): ``settle()`` on a loaded page is one ``wait_for_load_state("domcontentloaded",
    timeout=2000)`` and one ``evaluate(_RAF2)``, nothing else, also with a request pending (the root cause is
    *kept* in settle: the follow is a separate call)."""
    d, page = rig()
    page.request("/results.html?q=kettle")
    await d.settle()
    assert page.calls == [
        ("wait_for_load_state", ("domcontentloaded",), {"timeout": 2000}),
        ("evaluate", (RAF2,), {}),
    ], page.calls
    assert page.unexpected == []
