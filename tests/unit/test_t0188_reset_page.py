"""T-0188 group A, criteria 1-5 (CPU, fake driver): when Restart's navigation to the start page fails, the executor
replaces the page with a fresh one at the start page and classifies why the navigation failed from event kinds and
exception types only; Ctrl+T/Ctrl+L in web mode get the same fallback (docs/tasks/T-0188.md).

**The seam this module adds** (implementation notes, "a driver method that replaces the active page with a fresh one
at a URL"; the name is this module's reading, reported): ``async replace_page(url: str) -> None`` on the driver. The
real driver opens a new page in the same context, navigates it to ``url``, makes it the active page
(``page_serial`` +1) and closes every other page with ``run_before_unload=False``; the real-browser half of that
contract is ``tests/integration/test_t0188_beforeunload_browser.py``. ``ResetDriver`` below is T-0111c's
``FakeDriver`` (every other public name is still its named-``AssertionError`` sentinel) plus ``replace_page``: while
``block`` is armed every ``goto`` records the scripted events (a dismissed beforeunload, as
``runtime._on_dialog`` records it: ``{"kind": "dialog", "type": "beforeunload"}``) and raises the scripted exception,
as a page whose beforeunload is dismissed does on every navigation; ``replace_page`` makes a fresh page (``active =
"fresh"``) that no longer blocks, or raises its own scripted exception.

**Readings (reported):**

- C3 "returns or exposes": ``reset_page()`` *returns* the outcome, a mapping (or an object with attributes)
  ``replaced``/``reason``. On the unrecovered path the raised exception carries the class as ``exc.reason``.
- C2 "a beforeunload dialog event was recorded during the attempt" wins over the exception type for a base ``Error``
  (the adversarial row of the task). Which class wins when a beforeunload event and a ``TimeoutError`` coincide is
  not stated: no row pins it. The unrecovered raise carries the first navigation's class as ``reason`` and the
  replacement's as ``fallback_reason`` (review ruling A-R4, pinned by the different-classes row), and it chains no
  driver exception: ``__cause__`` and ``__context__`` are both ``None`` (A-R2).
- C5 "byte-identical calls" on a successful ``goto`` excludes draining the driver's events before the attempt, so an
  undrained event recorded *before* the attempt cannot be told from one recorded during it; no row pins that.
- C4 in the fake: the start-page plan's outcome keeps reporting the dismissed dialog (``resolved["dismissed"]``, the
  executor's existing report contract, T-0111c criterion 13).

Exception types are Playwright's own (``playwright.async_api.Error``/``TimeoutError``, ``TargetClosedError`` from
``playwright._impl._errors``: 1.64.0 does not re-export it) and the builtin ``TimeoutError`` (``asyncio.TimeoutError``
is the same class on Python 3.11). Expected classes are by hand from criterion 2's table.

Rows green before the implementation by design (negative, C5): the successful-goto rows, the no-op rows and the
click-failure row. Every other row was written pending and activated for ``/implement T-0188`` (group A).

Regime: CPU over fakes; no browser, no socket, no measurement.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest
from playwright._impl._errors import TargetClosedError
from playwright.async_api import Error as PwError
from playwright.async_api import TimeoutError as PwTimeoutError

from local_astra.computer import executor as executor_mod
from tests.unit.test_computer_executor import CLOCK_MS, FakeDriver, inv, own_frame

ORIGIN = "http://127.0.0.1:45678"
WEB_URL = "http://127.0.0.1:45678/web.html"
#: the first page ``open()`` is given in loopback mode: Restart's target there (T-0162)
LOOP_START = "http://127.0.0.1:45678/index.html"
APP = "http://127.0.0.1:45678/app.html"
CLASSES = frozenset({"beforeunload", "timeout", "target_closed", "other"})
BU = {"kind": "dialog", "type": "beforeunload"}
#: the adversarial message (task, "Tests to write"): a URL and the word "timeout", on the base ``Error`` type
SECRET_HOST = "www.photopea.com"
LEAKY = (
    'Page.goto: net::ERR_ABORTED; Timeout 6000ms exceeded while navigating to "https://www.photopea.com/'
    '#doc-7f3a?token=zz9" timeout'
)
#: fragments of LEAKY that must appear nowhere in an outcome or a raised exception (the class "timeout" is a word
#: the outcome may legitimately carry, so it is not one of them)
LEAK_MARKS = ("photopea", "ERR_ABORTED", "6000ms", "token=zz9", "doc-7f3a", "Page.goto")


class LookalikeTimeout(Exception):
    """Named like a timeout, typed like nothing the criterion lists."""


LookalikeTimeout.__name__ = "TimeoutError"
LookalikeTimeout.__qualname__ = "TimeoutError"


class ResetDriver(FakeDriver):
    """``FakeDriver`` plus T-0188's ``replace_page`` (module docstring)."""

    def __init__(self) -> None:
        super().__init__()
        self.block: tuple[BaseException, list[dict[str, Any]]] | None = None
        self.replace_error: BaseException | None = None

    def arm_block(self, exc: BaseException, events: list[dict[str, Any]] | None = None) -> None:
        self.block = (exc, list(events or []))

    async def goto(self, url: str) -> None:
        await super().goto(url)
        if self.block is not None:
            exc, events = self.block
            self.events.extend(dict(e) for e in events)
            raise exc

    async def replace_page(self, url: str) -> None:
        self._enter("replace_page", url)
        if self.replace_error is not None:
            raise self.replace_error
        self.block = None
        self.page_serial += 1
        self.active = "fresh"
        self.scene += 1


def executor(d: FakeDriver, *, web: bool, **kw: Any) -> Any:
    """As ``cli._start_live`` builds it (start origin = allowed origin), over ``d``."""
    return executor_mod.BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS,
        viewport=d.viewport,
        driver=d,
        allowed_origins=frozenset({ORIGIN}),
        web=web,
        start_origin=ORIGIN,
        **kw,
    )


async def started(d: FakeDriver, *, web: bool, **kw: Any) -> tuple[Any, str]:
    """A launched browser that opened its start page, then went elsewhere on the origin (the model navigated)."""
    ex = executor(d, web=web, **kw)
    start = WEB_URL if web else LOOP_START
    await ex.open(start)
    await ex.open(APP)
    return ex, start


def names(d: FakeDriver, since: int) -> list[tuple[str, tuple[Any, ...]]]:
    return list(d.calls[since:])


def outcome(result: object) -> dict[str, Any]:
    """C3: ``{replaced, reason}`` read off the returned mapping or object; anything else is a named failure."""
    if isinstance(result, Mapping):
        if "replaced" not in result or "reason" not in result:
            raise AssertionError(
                f"T-0188 C3: reset_page returned a mapping without replaced/reason: {sorted(result)}"
            )
        return {"replaced": result["replaced"], "reason": result["reason"]}
    if hasattr(result, "replaced") and hasattr(result, "reason"):
        return {"replaced": result.replaced, "reason": result.reason}
    raise AssertionError(
        f"T-0188 C3: reset_page must return {{replaced, reason}}; it returned {type(result).__name__}"
    )


async def reset(ex: Any) -> Any:
    """``reset_page()``; any exception becomes a named assertion (the type name only)."""
    try:
        return await asyncio.wait_for(ex.reset_page(), 5)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(
            f"T-0188 C1: reset_page raised {type(exc).__name__} instead of replacing the page"
        ) from None


def unchained(exc: BaseException, where: str) -> None:
    """A-R2: the raised exception reaches no driver exception at all, neither as ``__cause__`` nor as the implicit
    ``__context__`` (``raise ... from None`` only hides the context from a printed traceback; anything that walks
    ``__context__``, a logger or ``traceback.TracebackException``, still reaches the driver's message)."""
    assert exc.__cause__ is None, (
        f"T-0188 A-R2: {where} chains a __cause__ ({type(exc.__cause__).__name__})"
    )
    assert exc.__context__ is None, (
        f"T-0188 A-R2: {where} keeps the driver exception as __context__ "
        f"({type(exc.__context__).__name__}); classify inside the except, raise after it"
    )


def no_leak(where: str, *texts: str) -> None:
    for text in texts:
        hit = [m for m in LEAK_MARKS if m.lower() in text.lower()]
        assert not hit, (
            f"T-0188 C2/C3: the exception message leaked into {where}: {hit} in {text[:300]!r}"
        )


# =====================================================================================================
# criteria 1-3: the fallback and the reason class (Restart)
# =====================================================================================================

CLASS_ROWS = [
    pytest.param(
        lambda: PwError("net::ERR_ABORTED"), [BU], "beforeunload", id="error+beforeunload"
    ),
    pytest.param(
        lambda: PwTimeoutError("Timeout 6000ms exceeded."), [], "timeout", id="playwright-timeout"
    ),
    pytest.param(lambda: TimeoutError(), [], "timeout", id="asyncio-timeout"),
    pytest.param(lambda: TargetClosedError(), [], "target_closed", id="target-closed"),
    pytest.param(lambda: PwError("net::ERR_FAILED"), [], "other", id="base-error"),
    pytest.param(lambda: RuntimeError("boom"), [], "other", id="runtime-error"),
    # adversarial: only a *beforeunload* dialog counts, not any dialog dismissed during the attempt
    pytest.param(
        lambda: PwError("net::ERR_ABORTED"),
        [{"kind": "dialog", "type": "alert"}],
        "other",
        id="alert-dialog",
    ),
    # adversarial: the type decides, not its name
    pytest.param(
        lambda: LookalikeTimeout("timeout"), [], "other", id="lookalike-named-TimeoutError"
    ),
]


@pytest.mark.parametrize("web", [pytest.param(True, id="web"), pytest.param(False, id="loopback")])
@pytest.mark.parametrize(("make_exc", "events", "want"), CLASS_ROWS)
async def test_failed_reset_replaces_and_classifies(
    make_exc: Any, events: Any, want: str, web: bool
) -> None:
    """C1-C3: the start-page ``goto`` raises (after recording ``events``): the executor replaces the page with a fresh
    one at the *same* start URL, exactly once, after the failed ``goto``, and returns ``{replaced: True, reason:
    <class>}`` with the class from criterion 2's table."""
    d = ResetDriver()
    ex, start = await started(d, web=web)
    d.arm_block(make_exc(), events)
    mark = len(d.calls)
    got = outcome(await reset(ex))
    calls = names(d, mark)
    gotos = [i for i, c in enumerate(calls) if c[0] == "goto"]
    replaces = [i for i, c in enumerate(calls) if c[0] == "replace_page"]
    assert [c[1] for c in calls if c[0] == "replace_page"] == [(start,)], (
        f"T-0188 C1: one replace_page({start}) after the failed navigation; calls {calls}"
    )
    assert gotos and calls[gotos[0]][1] == (start,) and gotos[0] < replaces[0], (
        f"T-0188 C1/H-T188-1: the start page is tried first, the page replaced only when that fails; calls {calls}"
    )
    assert got == {"replaced": True, "reason": want}, f"T-0188 C2/C3: {got}, want reason {want!r}"
    assert d.active == "fresh", "premise: the fake's fresh page is active"


@pytest.mark.parametrize(
    ("events", "want"),
    [
        pytest.param([], "other", id="no-event"),
        pytest.param([BU], "beforeunload", id="beforeunload-event"),
    ],
)
async def test_adversarial_message_says_timeout_but_type_is_base_error(
    events: Any, want: str
) -> None:
    """C2 (the task's adversarial row): the exception is Playwright's base ``Error`` and its message carries a URL and
    the word "timeout". The class is ``other`` (``beforeunload`` with the event), never ``timeout``, and no fragment
    of the message appears in the outcome."""
    d = ResetDriver()
    ex, _ = await started(d, web=True)
    d.arm_block(PwError(LEAKY), events)
    result = await reset(ex)
    got = outcome(result)
    assert got["reason"] != "timeout", f"T-0188 C2: the message decided the class: {got}"
    assert got == {"replaced": True, "reason": want}, got
    no_leak("the outcome", repr(result), str(result), repr(got))


@pytest.mark.parametrize(
    "make_exc",
    [
        pytest.param(lambda: PwTimeoutError(LEAKY), id="timeout"),
        pytest.param(lambda: TargetClosedError(LEAKY), id="target_closed"),
    ],
)
async def test_unrecovered_reset_raises_the_class_not_the_message(make_exc: Any) -> None:
    """C3: the navigation and the fallback both fail with the same type (messages carrying the URL): ``reset_page``
    raises an exception carrying the class as ``reason``; no fragment of either message is in its text, its
    ``repr``, its ``args`` or a chained ``__cause__``."""
    d = ResetDriver()
    ex, start = await started(d, web=True)
    first = make_exc()
    want = "timeout" if isinstance(first, PwTimeoutError) else "target_closed"
    d.arm_block(first)
    d.replace_error = make_exc()
    mark = len(d.calls)
    with pytest.raises(Exception) as info:
        await asyncio.wait_for(ex.reset_page(), 5)
    exc = info.value
    assert ("replace_page", (start,)) in d.calls[mark:], (
        f"premise: the fallback was tried; {d.calls[mark:]}"
    )
    assert exc is not first and exc is not d.replace_error, (
        f"T-0188 C3: the driver's exception ({type(exc).__name__}) escaped with its message"
    )
    assert getattr(exc, "reason", None) == want, (
        f"T-0188 C3: the raised {type(exc).__name__} must carry reason={want!r}; it has {getattr(exc, 'reason', None)!r}"
    )
    no_leak("the raised exception", str(exc), repr(exc), repr(exc.args))
    if exc.__cause__ is not None:
        no_leak("the raised exception's __cause__", str(exc.__cause__), repr(exc.__cause__))
    unchained(exc, f"the raised {type(exc).__name__}")


async def test_unrecovered_reset_mixed_types_still_a_class() -> None:
    """C3: the navigation fails on a beforeunload, the fallback with a base ``Error`` carrying the URL: the raised
    exception's ``reason`` is one of the four classes (which one is not stated; module docstring) and the message is
    nowhere in it."""
    d = ResetDriver()
    ex, _ = await started(d, web=True)
    d.arm_block(PwError(LEAKY), [BU])
    d.replace_error = PwError(LEAKY)
    with pytest.raises(Exception) as info:
        await asyncio.wait_for(ex.reset_page(), 5)
    exc = info.value
    assert getattr(exc, "reason", None) in CLASSES, (
        f"T-0188 C3: the raised {type(exc).__name__} carries no class (reason={getattr(exc, 'reason', None)!r})"
    )
    no_leak("the raised exception", str(exc), repr(exc), repr(exc.args))
    unchained(exc, f"the raised {type(exc).__name__}")


@pytest.mark.parametrize(
    ("make_first", "events", "make_second", "want_reason", "want_fallback"),
    [
        pytest.param(
            lambda: PwError(LEAKY),
            [BU],
            lambda: PwTimeoutError(LEAKY),
            "beforeunload",
            "timeout",
            id="beforeunload-then-timeout",
        ),
        pytest.param(
            lambda: TargetClosedError(LEAKY),
            [],
            lambda: RuntimeError(LEAKY),
            "target_closed",
            "other",
            id="target_closed-then-other",
        ),
        pytest.param(
            lambda: TimeoutError(),
            [],
            lambda: TargetClosedError(LEAKY),
            "timeout",
            "target_closed",
            id="timeout-then-target_closed",
        ),
    ],
)
async def test_unrecovered_reset_reason_is_the_first_navigations_class(
    make_first: Any, events: Any, make_second: Any, want_reason: str, want_fallback: str
) -> None:
    """A-R4: the navigation and the replacement fail with *different* classes (expected by hand from criterion 2's
    table): the raised ``ResetPageError.reason`` is the first navigation's class and ``fallback_reason`` the
    replacement's, never swapped; still no message and no chained driver exception."""
    d = ResetDriver()
    ex, _ = await started(d, web=True)
    d.arm_block(make_first(), events)
    d.replace_error = make_second()
    with pytest.raises(Exception) as info:
        await asyncio.wait_for(ex.reset_page(), 5)
    exc = info.value
    assert isinstance(exc, executor_mod.ResetPageError), (
        f"T-0188 A-R4: an unrecovered reset raises ResetPageError, not {type(exc).__name__}"
    )
    got = (getattr(exc, "reason", None), getattr(exc, "fallback_reason", None))
    assert got == (want_reason, want_fallback), (
        f"T-0188 A-R4: (reason, fallback_reason) = {got}, want {(want_reason, want_fallback)} "
        "(the first navigation's class, then the replacement's)"
    )
    no_leak("the raised exception", str(exc), repr(exc), repr(exc.args))
    unchained(exc, "the raised ResetPageError")


# =====================================================================================================
# criterion 4: Ctrl+T / Ctrl+L in web mode
# =====================================================================================================


@pytest.mark.parametrize("combo", ["Ctrl+T", "Ctrl+L"])
async def test_start_page_combo_falls_back_to_a_fresh_page(combo: str) -> None:
    """C4: a beforeunload-blocked start-page navigation ends ``executed`` with ``opened the start page``, the page
    replaced at ``/web.html``; the post-action frame is the fresh page's, and the dismissed dialog is still
    reported."""
    d = ResetDriver()
    ex, _ = await started(d, web=True)
    f = await own_frame(ex)
    assert f.phash
    d.arm_block(PwError(LEAKY), [BU])
    mark = len(d.calls)
    out = await asyncio.wait_for(ex.execute(inv("press_key", None, combo=combo)), 5)
    calls = names(d, mark)
    assert out.status == "executed", (
        f"T-0188 C4: {combo} on a beforeunload page: {out.status} {out.reason!r}"
    )
    assert out.action == executor_mod.START_PAGE_ACTION == "opened the start page", out.action
    assert [c[1] for c in calls if c[0] == "replace_page"] == [(WEB_URL,)], (
        f"T-0188 C4: calls {calls}"
    )
    assert d.shots[-1] == "fresh", (
        f"T-0188 C4: the frame after is the fresh page's; shots {d.shots}"
    )
    assert {"dialog": "beforeunload"} in out.resolved.get("dismissed", []), (
        f"T-0188 C5: the dismissed beforeunload is still reported; resolved {out.resolved}"
    )
    no_leak("the Ctrl+T outcome", out.model_dump_json())


async def test_start_page_combo_unrecovered_fails_without_the_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C4/C3: the navigation and the fallback both fail: ``failed``, and no fragment of either message (which carry
    the URL) is in the action, the reason or ``resolved``. A-R2: the exception the executor turned into ``failed``
    (seen at its ``_failed`` helper) is not the driver's and chains no driver exception."""
    d = ResetDriver()
    ex, _ = await started(d, web=True)
    await own_frame(ex)
    d.arm_block(PwError(LEAKY), [BU])
    d.replace_error = PwError(LEAKY)
    seen: list[Exception] = []
    real_failed = executor_mod._failed

    def spy(i: Any, exc: Exception) -> Any:
        seen.append(exc)
        return real_failed(i, exc)

    monkeypatch.setattr(executor_mod, "_failed", spy)
    mark = len(d.calls)
    out = await asyncio.wait_for(ex.execute(inv("press_key", None, combo="Ctrl+T")), 5)
    assert ("replace_page", (WEB_URL,)) in d.calls[mark:], (
        f"T-0188 C4: the fallback was not tried; calls {d.calls[mark:]}"
    )
    assert out.status == "failed", out
    no_leak("the failed Ctrl+T outcome", out.model_dump_json())
    assert len(seen) == 1, f"premise: the failure went through _failed once; {seen!r}"
    exc = seen[0]
    assert exc is not d.replace_error and not isinstance(exc, PwError), (
        f"T-0188 C3/C4: the driver's {type(exc).__name__} reached the outcome path with its message"
    )
    no_leak("the Ctrl+T exception", str(exc), repr(exc), repr(exc.args))
    unchained(exc, f"the Ctrl+T {type(exc).__name__}")


# =====================================================================================================
# criterion 5: negative (green before the implementation by design)
# =====================================================================================================


@pytest.mark.parametrize("web", [pytest.param(True, id="web"), pytest.param(False, id="loopback")])
async def test_successful_reset_is_unchanged(web: bool) -> None:
    """C5 / H-T188-1: a ``goto`` that succeeds: the calls are T-0162's exactly (``ensure``, ``goto(start)``,
    ``settle``), the page and its serial are kept, nothing is replaced."""
    d = ResetDriver()
    ex, start = await started(d, web=web)
    serial, active = d.page_serial, d.active
    mark = len(d.calls)
    result = await reset(ex)
    assert names(d, mark) == [("ensure", ()), ("goto", (start,)), ("settle", ())], names(d, mark)
    assert (d.page_serial, d.active) == (serial, active)
    if result is not None:  # C3 "returns or exposes": a returned outcome says nothing was replaced
        assert outcome(result) == {"replaced": False, "reason": None}, result


async def test_successful_reset_with_a_dialog_is_unchanged() -> None:
    """C5 / H-T188-1 (guard too wide): the navigation succeeds although a beforeunload dialog was recorded (the
    page's handler let it go): the page is kept. The fallback is keyed on the navigation failing, not on the event."""
    d = ResetDriver()
    ex, start = await started(d, web=True)
    d.events.append(dict(BU))
    serial = d.page_serial
    mark = len(d.calls)
    result = await reset(ex)
    assert names(d, mark) == [("ensure", ()), ("goto", (start,)), ("settle", ())], names(d, mark)
    assert d.page_serial == serial
    if result is not None:
        assert outcome(result) == {"replaced": False, "reason": None}, result


@pytest.mark.parametrize("combo", ["Ctrl+T", "Ctrl+L"])
async def test_successful_start_page_combo_is_unchanged(combo: str) -> None:
    """C5: a web-mode Ctrl+T/Ctrl+L whose ``goto`` succeeds: T-0169's calls (``ensure``, ``goto``, ``settle``,
    ``screenshot``, ``take_events``), no replacement, ``executed``."""
    d = ResetDriver()
    ex, _ = await started(d, web=True)
    await own_frame(ex)
    serial = d.page_serial
    mark = len(d.calls)
    out = await ex.execute(inv("press_key", None, combo=combo))
    assert out.status == "executed" and out.action == "opened the start page", out
    assert names(d, mark) == [
        ("ensure", ()),
        ("goto", (WEB_URL,)),
        ("settle", ()),
        ("screenshot", ()),
        ("take_events", ()),
    ], names(d, mark)
    assert d.page_serial == serial


@pytest.mark.parametrize("web", [pytest.param(True, id="web"), pytest.param(False, id="loopback")])
async def test_reset_without_a_browser_is_a_noop(web: bool) -> None:
    """C5 / T-0162: never launched: no driver call, no raise."""
    d = ResetDriver()
    ex = executor(d, web=web)
    await ex.reset_page()
    assert d.calls == []


async def test_reset_with_nothing_opened_is_a_noop() -> None:
    """C5 / T-0162: loopback mode, launched by a frame but no page ever ``open()``ed: no driver call."""
    d = ResetDriver()
    ex = executor(d, web=False)
    await own_frame(ex)
    d.arm_block(PwError("net::ERR_ABORTED"), [BU])
    mark = len(d.calls)
    await ex.reset_page()
    assert names(d, mark) == []


class EndingDriver(ResetDriver):
    """``ResetDriver`` with T-0111g's end facts (``connected``, ``page_count``)."""

    def __init__(self) -> None:
        super().__init__()
        self.pages = 1

    @property
    def connected(self) -> bool:
        return self.alive

    @property
    def page_count(self) -> int:
        return self.pages if self.alive else 0


async def test_reset_after_the_browser_ended_is_a_noop() -> None:
    """C5 / T-0162: under ``end_when_closed``, a browser whose last page closed has ended: Restart touches nothing
    (no relaunch, no replacement)."""
    d = EndingDriver()
    ex, _ = await started(d, web=True, end_when_closed=True)
    d.pages = 0
    d.arm_block(PwError("net::ERR_ABORTED"), [BU])
    mark = len(d.calls)
    await ex.reset_page()
    assert ex.closed is True
    assert names(d, mark) == []


async def test_failed_click_does_not_replace_the_page() -> None:
    """C5 (guard too wide): the fallback belongs to the start-page navigation only. A click that fails while a
    beforeunload was dismissed is ``failed`` as before, and no page is replaced."""
    d = ResetDriver()
    ex, _ = await started(d, web=True)
    f = await own_frame(ex)
    d.events.append(dict(BU))
    d.arm_raise(2)  # mouse_move succeeds, mouse_down raises
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert out.status == "failed", out
    assert not [c for c in d.calls[mark:] if c[0] == "replace_page"], d.calls[mark:]
