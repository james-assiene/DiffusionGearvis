"""T-0111g group A, criteria A1-A3 against a fake driver: the executor's ``headless``, ``end_when_closed`` and
``allowed_origins`` options (docs/tasks/T-0111g.md).

**The API these rows pin** (the task fixes behaviour; these are the names a row needs):

=====================================================================  ===============================================
``BrowserToolExecutor(*, ..., headless: bool = True,``                 keyword-only, beside T-0111c's arguments
``end_when_closed: bool = False,``
``allowed_origins: frozenset[str] | None = None)``
``PlaywrightDriver(*, ..., headless=..., allowed_origins=...)``         the executor passes both through when it
                                                                       builds the runtime (``driver=None``)
``ex.closed -> bool``                                                  A2's state; ``False`` until the browser ends
``executor.BrowserClosedError``                                        what ``frame()`` raises once ``closed``
                                                                       (in ``executor.__all__``)
``runtime.url_allowed(url, allowed_origins) -> bool``                  the route/WebSocket predicate: ``None`` is
                                                                       ``fence.is_loopback_url``; a set admits a URL
                                                                       whose ``scheme://host:port`` is in it *and*
                                                                       that is loopback (the allow-list composes
                                                                       with the fence, H-T111c-E)
=====================================================================  ===============================================

**The driver seam, extended for A2** (``tests/unit/test_computer_executor.py`` docstring holds the rest). The rule
"the browser ended" is the executor's, so a fake can exercise it; the driver only reports two facts:

=======================  ================================================================================
``connected: bool``      the current generation's browser is running (``False`` before the first launch
                         and after a disconnect or a kill)
``page_count: int``      pages open in the current browser (closed ones not counted; 0 when not connected)
=======================  ================================================================================

With ``end_when_closed=True``, once the browser has launched (``generation >= 1``), the executor checks both
*before* any ``ensure()``: ``not connected`` or ``page_count == 0`` ends it for good. With the default ``False`` the
executor reads neither name: the T-0111c ``FakeDriver`` raises a named ``AssertionError`` on both, so every T-0111c
row pins that half. The executor's own page-cap close never closes the last page, and ``ex.close()`` is not a
row's concern.

**My reading of the open points** (reported, not guessed silently): ``open(url)`` is a navigation, so with a set it
refuses an origin outside the set with ``ValueError``, as it refuses a non-loopback URL today; after the end
``open()`` may raise ``BrowserClosedError`` but never launches, opens a page or navigates.

Every row was ``pending(task="T-0111g")`` before group A's implementation, failing with a named ``AssertionError``
(never a ``TypeError``). The group A review row A3 (Rulings, "on the group A reviews of c4618da") is
``test_last_page_closed_inside_ensure_ends_it``, driven by ``ClosableDriver.arm_close_in_ensure``.
"""

from __future__ import annotations

import contextlib
import importlib
from typing import Any

import pytest

from local_astra.computer import fence
from local_astra.conductor.tool_executor import DryRunToolExecutor
from local_astra.contracts.perception import Frame
from local_astra.contracts.toolcall import ToolInvocation, ToolOutcome
from tests.unit.test_computer_executor import CLOCK_MS, FakeBrowserGone, FakeDriver, inv

CLOSED = "the browser window was closed"
ORIGIN = "http://127.0.0.1:43123"
_MISSING = object()


def executor_mod() -> Any:
    return importlib.import_module("local_astra.computer.executor")


def runtime_mod() -> Any:
    return importlib.import_module("local_astra.computer.runtime")


class ClosableDriver(FakeDriver):
    """T-0111c's fake plus A2's two facts and the ways a browser goes away that are not the executor's doing.

    ``close_all_pages()`` is a human closing every window (or Playwright's ``page.close()`` on each): the browser
    stays connected with no page, and the next ``ensure()`` would open a new one, as ``PlaywrightDriver.ensure``
    does today (``reopened`` counts those). ``kill()`` is the browser dying: the next ``ensure()`` relaunches
    (``launches`` counts those). ``close_popup_itself()`` is a popup calling ``window.close()`` while its opener
    stays open."""

    def __init__(self) -> None:
        super().__init__()
        self.pages = 0
        self.reopened = 0
        self._active_closed = False
        self._close_in_ensure: str | None = None
        self.closed_in_ensure = False
        self._ensure_error: BaseException | None = None
        self.ensure_errors = 0

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise AssertionError(
            f"the executor used driver.{name}, which is in neither the T-0111c driver seam nor T-0111g's "
            "connected/page_count (tests/unit/test_computer_live_options.py docstring)"
        )

    @property
    def connected(self) -> bool:
        return self.alive

    @property
    def page_count(self) -> int:
        return self.pages if self.alive else 0

    # -- test controls
    def open_popup(self, name: str = "popup") -> None:
        super().open_popup(name)
        self.pages += 1

    def close_all_pages(self) -> None:
        assert self.alive, "the premise failed: no browser to close"
        self.pages = 0

    def close_popup_itself(self) -> None:
        assert self.active != "opener" and self.pages >= 2, (self.active, self.pages)
        self.pages -= 1
        self._active_closed = True

    def arm_close_in_ensure(self, then: str) -> None:
        """The hook of the group A review row A3. The next ``ensure()`` of a live browser first does its normal
        work (adopts a pending popup: ``page_serial`` +1), then every page closes before it returns: a human
        closing the window while ``PlaywrightDriver.ensure`` waits for the adopted page's load or closes the oldest
        page over the cap. It then either returns with no page open (``then="return"``: a runtime whose reopen
        path refuses by leaving none) or raises ``FakeBrowserGone`` (``then="raise"``: Playwright's close error out
        of the load wait, or a runtime refusing by raising). It never reopens a page: whether the real runtime's
        reopen path refuses is the e2e row's (``test_computer_live_browser_e2e.py``); this hook pins the
        executor's half, which must notice the end whichever way ``ensure()`` comes back."""
        assert then in ("return", "raise"), then
        self._close_in_ensure = then

    def arm_ensure_error(self, exc: BaseException) -> None:
        """The hook of the round-3 rows (a) and (b) (Rulings, "on the group A re-reviews at d2e27fd"). The next
        ``ensure()`` of a live browser raises ``exc`` and changes nothing: the browser stays connected with its pages
        open, so the error is not an end. That is a transient driver failure, e.g. a load wait that timed out."""
        self._ensure_error = exc

    # -- the seam
    async def ensure(self) -> None:
        if self.alive and self._ensure_error is not None:
            exc, self._ensure_error = self._ensure_error, None
            self.calls.append(("ensure", ()))
            self.ensure_errors += 1
            raise exc
        if self.alive and self._close_in_ensure is not None:
            then, self._close_in_ensure = self._close_in_ensure, None
            self.calls.append(("ensure", ()))
            if self._popup is not None:
                self.active, self._popup = self._popup, None
                self.page_serial += 1
                self.scene += 1
            self.pages, self.closed_in_ensure = 0, True
            if then == "raise":
                raise FakeBrowserGone("Target page, context or browser has been closed")
            return
        if self.alive and self.pages == 0:
            self.calls.append(("ensure", ()))
            self.pages, self.reopened = 1, self.reopened + 1
            self.page_serial += 1
            self.active, self.scene = "reopened", self.scene + 1
            return
        if self.alive and self._active_closed:
            self.calls.append(("ensure", ()))
            self._active_closed = False
            self.page_serial += 1
            self.active, self.scene = "opener", self.scene + 1
            return
        launches = self.launches
        await super().ensure()
        if self.launches != launches:
            self.pages = 1


def build(fake: FakeDriver | None = None, **kw: Any) -> tuple[Any, Any]:
    fake = fake if fake is not None else ClosableDriver()
    try:
        ex = executor_mod().BrowserToolExecutor(
            now_ms=lambda: CLOCK_MS, viewport=fake.viewport, driver=fake, **kw
        )
    except TypeError as exc:
        raise AssertionError(
            f"T-0111g: BrowserToolExecutor does not take {sorted(kw)} yet: {exc}"
        ) from exc
    return ex, fake


def closed_of(ex: Any) -> bool:
    v = getattr(ex, "closed", _MISSING)
    assert v is not _MISSING, "T-0111g A2: BrowserToolExecutor has no `closed`"
    assert isinstance(v, bool), f"`closed` is {type(v).__name__}, not bool"
    return v


def closed_error() -> type[BaseException]:
    mod = executor_mod()
    err = getattr(mod, "BrowserClosedError", None)
    assert isinstance(err, type) and issubclass(err, Exception), (
        "T-0111g A2: executor.BrowserClosedError (the named exception frame() raises once closed) does not exist"
    )
    assert "BrowserClosedError" in mod.__all__, mod.__all__
    return err


def every_action(f: Frame) -> list[ToolInvocation]:
    return [
        inv("click_at", f, x=500, y=500),
        inv("type_at", f, x=500, y=500, text="abc", press_enter=True),
        inv("scroll", direction="down"),
        inv("press_key", combo="Tab"),
    ]


# --------------------------------------------------------------------------------------------- A1


class _Spy:
    """Records the keyword arguments of every ``PlaywrightDriver`` the executor builds (the real class underneath,
    so construction stays as cheap as it is: nothing launches until ``ensure()``)."""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = runtime_mod().PlaywrightDriver
        seen = self.seen

        class SpyDriver(real):  # type: ignore[misc, valid-type]
            def __init__(self, **kw: Any) -> None:
                seen.append(dict(kw))
                super().__init__(**kw)

        monkeypatch.setattr(executor_mod(), "PlaywrightDriver", SpyDriver)


def _construct(**kw: Any) -> Any:
    try:
        return executor_mod().BrowserToolExecutor(now_ms=lambda: CLOCK_MS, **kw)
    except TypeError as exc:
        raise AssertionError(
            f"T-0111g: BrowserToolExecutor does not take {sorted(kw)} yet: {exc}"
        ) from exc


def test_headless_passed_to_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """A1: ``headless`` reaches ``PlaywrightDriver`` as given; the default stays headless (T-0111c's rows and
    ``make browser`` construct without it). Construction launches nothing."""
    spy = _Spy()
    spy.install(monkeypatch)
    _construct(headless=False)
    _construct(headless=True)
    _construct()
    assert len(spy.seen) == 3, spy.seen
    assert spy.seen[0].get("headless", _MISSING) is False, spy.seen[0]
    assert spy.seen[1].get("headless", True) is True, spy.seen[1]
    assert spy.seen[2].get("headless", True) is True, (
        "the default must stay headless (T-0111c and make browser are unchanged)",
        spy.seen[2],
    )


# --------------------------------------------------------------------------------------------- A2


async def _kill_mid_action(ex: Any, fake: ClosableDriver, f: Frame) -> None:
    import asyncio

    fake.arm_hang("mouse_down")
    task = asyncio.ensure_future(ex.execute(inv("click_at", f, x=500, y=500)))
    assert fake.hanging is not None
    await asyncio.wait_for(fake.hanging.wait(), 5)
    fake.kill()
    out = await asyncio.wait_for(task, 5)
    assert out.status == "failed" and "FakeBrowserGone" in (out.reason or ""), (
        "the action in flight when the browser died settles as T-0111c's failed outcome",
        out,
    )


@pytest.mark.parametrize("how", ["all-pages-closed", "killed", "killed-mid-action"])
async def test_end_when_closed_refuses_and_never_relaunches(how: str) -> None:
    """A2 (reviewer mutation 12): with ``end_when_closed=True`` the browser ends for good when every page is closed
    by someone other than the executor, or when it disconnects (a kill, idle or mid-action). After that: ``closed``
    is True; no ``ensure()`` relaunches or opens a page (``launches`` and ``reopened`` stay put); every non-search
    call is ``refused`` with exactly ``the browser window was closed`` and dispatches nothing; ``frame()`` raises
    ``BrowserClosedError``; ``open()`` navigates nothing; ``google_search`` is still the dry run. Before the end the
    executor works as T-0111c's (the first ``frame()`` launches once)."""
    err = closed_error()
    ex, fake = build(end_when_closed=True)
    assert closed_of(ex) is False, "closed before anything happened"
    f = await ex.frame()
    assert isinstance(f, Frame) and fake.launches == 1 and fake.pages == 1
    assert closed_of(ex) is False
    out = await ex.execute(inv("scroll", direction="down"))
    assert out.status == "executed", out
    f = ex.latest_frame

    if how == "all-pages-closed":
        fake.close_all_pages()
    elif how == "killed":
        fake.kill()
    else:
        await _kill_mid_action(ex, fake, f)
    end_mark, presses = (
        len(fake.calls),
        len(fake.presses),
    )  # the in-flight press, if any, came before the end

    for _round in range(2):  # it stays ended
        for i in every_action(f):
            mark = len(fake.calls)
            out = await ex.execute(i)
            assert isinstance(out, ToolOutcome)
            assert (out.status, out.reason) == ("refused", CLOSED), (how, i.name, out)
            assert fake.dispatched(mark) == [], (how, i.name, fake.dispatched(mark))
        assert closed_of(ex) is True, how
        mark = len(fake.calls)
        with pytest.raises(err):
            await ex.frame()
        assert [c for c in fake.calls[mark:] if c[0] == "screenshot"] == [], (
            "frame() captured after the end"
        )
        with contextlib.suppress(err):
            await ex.open(f"{ORIGIN}/")
        for i in (inv("google_search", query="blue mug"), inv("google_search")):
            assert await ex.execute(i) == await DryRunToolExecutor().execute(i), i.arguments
    assert fake.launches == 1, f"{how}: relaunched after the end ({fake.launches} launches)"
    assert fake.reopened == 0, f"{how}: opened a new page after the end"
    assert fake.dispatched(end_mark) == [], (
        f"{how}: something was dispatched after the end (open() included): {fake.dispatched(end_mark)}"
    )
    assert fake.typed == [] and len(fake.presses) == presses, (fake.presses, fake.typed)


async def test_default_still_relaunches_once() -> None:
    """A2's default half (T-0111c criterion 10 unchanged): with ``end_when_closed=False`` given explicitly, closing
    every page is not an end (the next ``frame()`` opens a page and works), and a kill relaunches exactly once with
    ``failed: browser restarted``, after which calls run again. ``closed`` stays False throughout."""
    ex, fake = build(end_when_closed=False)
    await ex.frame()
    fake.close_all_pages()
    f = await ex.frame()
    assert isinstance(f, Frame) and fake.reopened == 1 and fake.launches == 1, (
        fake.reopened,
        fake.launches,
    )
    assert closed_of(ex) is False
    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert out.status == "executed", out

    fake.kill()
    out = await ex.execute(inv("press_key", combo="Tab"))
    assert (out.status, out.reason) == ("failed", "browser restarted"), out
    assert fake.launches == 2
    out = await ex.execute(inv("press_key", combo="Tab"))
    assert out.status == "executed", out
    assert fake.launches == 2, "relaunched more than once"
    assert closed_of(ex) is False

    # the T-0111c fake (which has neither connected nor page_count) still drives a default executor
    ex, _base = build(FakeDriver(), end_when_closed=False)
    f = await ex.frame()
    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert out.status == "executed", out


async def test_popup_self_close_is_not_an_end() -> None:
    """A2's too-wide side: with ``end_when_closed=True``, a popup that closes itself while its opener stays open is
    not an end. The next ``frame()`` works (the opener is active again), ``closed`` stays False, nothing relaunches,
    and a click on the new frame executes."""
    ex, fake = build(end_when_closed=True)
    await ex.frame()
    fake.open_popup()
    popup = await ex.frame()
    assert fake.active == "popup" and fake.pages == 2, "the premise failed: the popup is not active"
    fake.close_popup_itself()
    out = await ex.execute(inv("click_at", popup, x=500, y=500))
    assert out.status == "refused" and out.reason != CLOSED, (
        "a popup's own close ended the browser",
        out,
    )
    f = await ex.frame()
    assert isinstance(f, Frame) and fake.active == "opener"
    assert closed_of(ex) is False
    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert out.status == "executed", out
    assert fake.launches == 1 and fake.reopened == 0


@pytest.mark.parametrize("then", ["return", "raise"])
@pytest.mark.parametrize("popup", [True, False], ids=["adopting", "no-popup"])
@pytest.mark.parametrize("entry", ["execute", "frame"])
async def test_last_page_closed_inside_ensure_ends_it(entry: str, popup: bool, then: str) -> None:
    """Group A review ruling, row A3, the executor's half (fix A3: "the reopen path refuses so the executor then
    reports closed"). With ``end_when_closed=True``, every page closes *inside* an ``ensure()`` (after the
    executor's pre-``ensure()`` check passed), while the driver adopts a popup or with none pending, and
    ``ensure()`` comes back with no page open, returning or raising. The call in flight already ends it:
    ``execute`` is ``refused`` with exactly ``the browser window was closed`` and dispatches nothing (no capture
    either), ``frame()`` raises ``BrowserClosedError`` and captures nothing; ``closed`` is True; later calls are
    refused the same way; nothing relaunches and no page is reopened.

    My reading, reported: the in-flight call is refused with A2's reason (not ``failed`` with the close error, not
    the page-changed refusal), because the end came before any dispatch."""
    err = closed_error()
    ex, fake = build(end_when_closed=True)
    f = await ex.frame()
    assert fake.launches == 1 and fake.pages == 1
    if popup:
        fake.open_popup()
    fake.arm_close_in_ensure(then)
    mark = len(fake.calls)
    if entry == "execute":
        try:
            out: Any = await ex.execute(inv("click_at", f, x=500, y=500))
        except Exception as exc:  # the executor must not let the driver's error out of execute
            out = exc
        assert fake.closed_in_ensure, "the premise failed: the hook did not run"
        assert isinstance(out, ToolOutcome) and (out.status, out.reason) == ("refused", CLOSED), (
            f"the call in flight when the last page closed inside ensure() was {out!r}"
        )
    else:
        try:
            got: Any = await ex.frame()
        except err:
            got = "closed"
        except Exception as exc:
            got = exc
        assert fake.closed_in_ensure, "the premise failed: the hook did not run"
        assert got == "closed", f"frame() with the last page closed inside ensure() gave {got!r}"
    assert fake.dispatched(mark) == [], fake.dispatched(mark)
    assert [c for c in fake.calls[mark:] if c[0] == "screenshot"] == [], "captured after the end"
    assert closed_of(ex) is True
    for i in every_action(f):
        out = await ex.execute(i)
        assert (out.status, out.reason) == ("refused", CLOSED), (i.name, out)
    with pytest.raises(err):
        await ex.frame()
    assert fake.dispatched(mark) == []
    assert (fake.launches, fake.reopened) == (1, 0), (fake.launches, fake.reopened)


_ENSURE_BOOM = "synthetic ensure failure (t0111g round 3)"


@pytest.mark.parametrize("entry", ["execute", "frame"])
@pytest.mark.parametrize("end_when_closed", [True, False], ids=["end-mode", "default-mode"])
async def test_ensure_error_with_browser_alive_is_not_an_end(
    end_when_closed: bool, entry: str
) -> None:
    """Round-3 rows (a) (end mode) and (b) (default mode), Rulings "on the group A re-reviews at d2e27fd". The
    browser is alive (connected, one page open) and ``ensure()`` raises a plain ``RuntimeError``. That is not an end:
    ``execute`` returns ``failed`` with a reason naming ``RuntimeError`` and dispatches nothing; ``frame()`` raises
    that ``RuntimeError`` (not ``BrowserClosedError``, which subclasses it, so the type is compared exactly);
    ``closed`` stays False. The browser then works on (A2: before an end the executor is T-0111c's): the next
    ``frame()`` captures and a click on it executes, with no relaunch and no reopened page.

    Kills Mx1 (``except Exception: return True`` in ``_ensure_ended``), which reports every ``ensure()`` error as
    an end."""
    err = closed_error()
    ex, fake = build(end_when_closed=end_when_closed)
    f = await ex.frame()
    assert fake.launches == 1 and fake.pages == 1
    fake.arm_ensure_error(RuntimeError(_ENSURE_BOOM))
    mark = len(fake.calls)
    if entry == "execute":
        try:
            out: Any = await ex.execute(inv("click_at", f, x=500, y=500))
        except Exception as exc:  # the executor must not let the driver's error out of execute
            out = exc
        assert fake.ensure_errors == 1, "the premise failed: the armed ensure() did not run"
        assert isinstance(out, ToolOutcome), f"execute let {out!r} out"
        assert out.status == "failed" and out.reason != CLOSED, (
            "a RuntimeError from ensure() with the browser alive was treated as the end",
            out,
        )
        assert "RuntimeError" in (out.reason or ""), out
        assert err.__name__ not in (out.reason or ""), out
    else:
        with pytest.raises(RuntimeError) as info:
            await ex.frame()
        assert fake.ensure_errors == 1, "the premise failed: the armed ensure() did not run"
        assert type(info.value) is RuntimeError and not isinstance(info.value, err), (
            f"frame() raised {type(info.value).__name__}, not the driver's RuntimeError",
            info.value,
        )
        assert str(info.value) == _ENSURE_BOOM, info.value
    assert fake.dispatched(mark) == [], fake.dispatched(mark)
    assert [c for c in fake.calls[mark:] if c[0] == "screenshot"] == [], "captured after the error"
    assert closed_of(ex) is False, "closed after a transient ensure() error"

    f = await ex.frame()
    assert isinstance(f, Frame)
    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert out.status == "executed", out
    assert closed_of(ex) is False
    assert (fake.launches, fake.reopened) == (1, 0), (fake.launches, fake.reopened)


@pytest.mark.parametrize("entry", ["execute", "frame"])
async def test_cancel_during_hanging_ensure_after_the_end_propagates(entry: str) -> None:
    """Round-3 row (c), Rulings "on the group A re-reviews at d2e27fd". In end mode an ``ensure()`` hangs (after its
    effect: a load wait), every page closes while it hangs, and then the task is cancelled (a barge-in). The
    cancellation is not swallowed: the task ends cancelled (awaiting it raises ``CancelledError``), with no
    outcome and nothing raised in its place. The end is still noticed afterwards: the next call is refused with
    A2's reason, ``closed`` is True, and nothing relaunched, reopened or was dispatched.

    Kills Mx9 (``except BaseException`` in ``_ensure_ended``), which turns the cancellation into the end's refusal
    (or ``BrowserClosedError`` from ``frame()``) because the end facts say ended."""
    import asyncio

    ex, fake = build(end_when_closed=True)
    f = await ex.frame()
    assert fake.launches == 1 and fake.pages == 1
    fake.arm_hang("ensure")
    mark = len(fake.calls)
    coro = ex.execute(inv("click_at", f, x=500, y=500)) if entry == "execute" else ex.frame()
    task = asyncio.ensure_future(coro)
    assert fake.hanging is not None
    await asyncio.wait_for(fake.hanging.wait(), 5)
    fake.close_all_pages()
    assert (fake.connected, fake.page_count) == (True, 0), (
        "the premise failed: the browser has not ended"
    )
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=5)
    assert task in done, "the cancelled task did not finish"
    if not task.cancelled():
        exc = task.exception()
        got = exc if exc is not None else task.result()
        raise AssertionError(
            f"cancelling {entry} during a hanging ensure() after the end gave {got!r}, not CancelledError"
        )
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.dispatched(mark) == [], fake.dispatched(mark)

    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert (out.status, out.reason) == ("refused", CLOSED), out
    assert closed_of(ex) is True
    assert (fake.launches, fake.reopened) == (1, 0), (fake.launches, fake.reopened)
    assert fake.dispatched(mark) == [], fake.dispatched(mark)


# --------------------------------------------------------------------------------------------- A3


def _url_allowed() -> Any:
    fn = getattr(runtime_mod(), "url_allowed", None)
    assert callable(fn), "T-0111g A3: runtime.url_allowed(url, allowed_origins) does not exist"
    return fn


NONE_ORACLE = [
    "http://127.0.0.1:8000/v1/completions",
    "http://127.0.0.1:43123/",
    "http://localhost:1420/",
    "ws://127.0.0.1:8081/ws",
    "http://[::1]:8080/",
    "http://example.com/",
    "http://127.0.1.1:8000/",
    "http://user@127.0.0.1:43123/",
    "file:///etc/passwd",
    "data:text/html,x",
]


def test_allowed_origins_route_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A3 (reviewer mutation 13). ``None`` is T-0111c's rule exactly (oracle: ``fence.is_loopback_url``). A set admits
    exactly its ``scheme://host:port`` origins: the same host on another port, ``localhost`` for ``127.0.0.1`` (and
    back), ``https`` for ``http``, ``ws`` for ``http``, the default port, IPv6 loopback and credentials are refused;
    a non-loopback origin is refused even when listed (the allow-list composes with the fence). The executor
    passes the set to ``PlaywrightDriver`` as ``allowed_origins``."""
    allowed = _url_allowed()
    for url in NONE_ORACLE:
        assert allowed(url, None) is fence.is_loopback_url(url), url

    only = frozenset({ORIGIN})
    for url in (
        f"{ORIGIN}/",
        f"{ORIGIN}",
        f"{ORIGIN}/demo/shop.html?q=mug#cart",
        f"{ORIGIN}/search.html?q=x",
    ):
        assert allowed(url, only) is True, url
    for url in (
        "http://127.0.0.1:43124/",
        "http://127.0.0.1:8000/v1/completions",
        "http://127.0.0.1:1420/",
        "http://127.0.0.1/",
        "http://localhost:43123/",
        "https://127.0.0.1:43123/",
        "ws://127.0.0.1:43123/",
        "http://[::1]:43123/",
        "http://user@127.0.0.1:43123/",
        "http://127.0.1.1:43123/",
        "file:///etc/passwd",
        "data:text/html,x",
    ):
        assert allowed(url, only) is False, url

    local = frozenset({"http://localhost:43123"})
    assert allowed("http://localhost:43123/x", local) is True
    assert allowed("http://127.0.0.1:43123/x", local) is False
    assert allowed("http://example.com:80/", frozenset({"http://example.com:80"})) is False, (
        "a listed non-loopback origin passed: the allow-list must compose with fence.is_loopback_url"
    )

    spy = _Spy()
    spy.install(monkeypatch)
    _construct(allowed_origins=only)
    _construct()
    assert spy.seen[0].get("allowed_origins") == only, spy.seen[0]
    assert spy.seen[1].get("allowed_origins") is None, spy.seen[1]


async def test_open_outside_the_allow_list_refused() -> None:
    """A3, ``open()`` as a navigation (my reading, reported): with a set, ``open()`` of the listed origin navigates;
    another port of the same host, or ``localhost`` for ``127.0.0.1``, raises ``ValueError`` and navigates nothing.
    Without a set T-0111c's rule holds (any loopback port opens)."""
    ex, fake = build(allowed_origins=frozenset({ORIGIN}))
    await ex.open(f"{ORIGIN}/demo/shop.html")
    assert [c for c in fake.dispatched() if c[0] == "goto"] == [
        ("goto", (f"{ORIGIN}/demo/shop.html",))
    ]
    for url in ("http://127.0.0.1:43124/", "http://localhost:43123/", "http://127.0.0.1:8000/"):
        mark = len(fake.calls)
        with pytest.raises(ValueError):
            await ex.open(url)
        assert fake.dispatched(mark) == [], (url, fake.dispatched(mark))

    ex, fake = build(allowed_origins=None)
    await ex.open("http://127.0.0.1:43124/")
    assert [c[1][0] for c in fake.dispatched() if c[0] == "goto"] == ["http://127.0.0.1:43124/"]
