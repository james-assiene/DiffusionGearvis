"""T-0198 criteria 1-3 and 5-7 against fake drivers: the executor brings the headed agent browser's active page to
the front once per page key, just before the first dispatched input, and never otherwise (docs/tasks/T-0198.md).

**The seam these rows pin** (the task's definitions). ``BrowserToolExecutor(..., driver=d, headless=False)`` keeps
its ``headless`` keyword and applies the policy to an injected driver too; ``ex.raise_window`` (default ``True``) is
the session's per-turn switch. **Fronting** is one call to the driver's optional ``async bring_to_front() -> None``,
looked up on the driver's *class* (as ``replace_page``/``focused_field``): every fake here is
``test_computer_executor.FakeDriver``, whose instance ``__getattr__`` turns any other public name into a named
``AssertionError``, so an instance lookup of the capability on a driver without it fails loudly. A **page key** is
``(inv.turn_id, driver.generation, driver.page_serial)``.

``FrontMixin`` adds the capability. It records each call in ``driver.calls`` as ``("bring_to_front", (turn,
generation, page_serial, active))`` (``turn`` is the turn the row is executing, ``driver.turn``, set by ``act``) and
behaves per ``front_mode``: ``"ok"``; ``"raise"`` (``front_error``, any ``Exception``); ``"hang"`` (blocks until
cancelled, then counts ``front_cancelled``). ``SettleMixin`` makes the next ``settle()`` adopt a popup
(``settle_popup``: the active page changes, ``page_serial`` +1), as the real driver's settle adopts a ``window.open``
popup. A late popup is ``FakeDriver.open_popup`` (adopted by the next ``ensure()``, T-0187); T-0188's replacement
is ``test_t0188_reset_page.ResetDriver``.

Expected values are by hand from the criteria: which call fronts, which key, and where in the driver's call list
(after the action's ``ensure()``, before its first input; after the settle that changed the page, before the
after-frame's ``screenshot``). Outcomes are compared as ``(status, action, reason, resolved)`` under a fixed phash
salt (``secrets.randbits`` patched, as T-0192's rows do).

**Readings (reported):**

- C1 "the first ``execute()`` of a turn that passes every refusal check": the front comes after that ``execute()``'s
  ``ensure()`` (the implementation notes: "only after ``_ensure_ended()`` and every refusal").
- C6 "runs past ``FRONT_TIMEOUT_S``": the wait is in line and bounded, so the hung call is cancelled by the time
  ``execute()`` returns (no fronting task outlives its action); "the wait counts inside ``ACTION_TIMEOUT_S``" is
  pinned by elapsed time only (with the action bound patched below ``FRONT_TIMEOUT_S`` the call returns at the
  action bound); the outcome in that row is not pinned (the action timed out).
- C4's ``raise_window`` default is ``True`` (the task's own browser row builds ``headless=False`` and expects fronts
  without setting it).
- The overview-frame path (``_act_on_overview``, T-0151) is one dispatch site the notes name; it gets one row.

Regime: CPU over fakes; no browser, no socket, no measurement beyond two wall-clock bounds (1 s fronting timeout).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest import mock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.computer import executor as executor_mod
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.toolcall import FrameGeometry, ToolInvocation, ToolOutcome
from tests.unit.test_computer_executor import CLOCK_MS, INPUT, FakeDriver, own_frame
from tests.unit.test_computer_live_options import ClosableDriver
from tests.unit.test_t0151_executor import PageDriver
from tests.unit.test_t0188_reset_page import ResetDriver
from tests.unit.test_t0192_checkout_executor import CHECKOUT, SHOP, UrlsDriver

FRONT = "bring_to_front"
SALT = 0x0198_0198_0198_0198
ORIGIN = "http://127.0.0.1:45698"
WEB_URL = f"{ORIGIN}/web.html"
APP = f"{ORIGIN}/app.html"
#: a Luhn-valid test card number (fence.card_like)
CARD = "4111 1111 1111 1111"


class FrontBoom(RuntimeError):
    """The fake's fronting failure."""


class OtherBoom(Exception):
    """A fronting failure that is no ``RuntimeError``."""


class FrontMixin:
    """T-0198's optional ``bring_to_front`` (module docstring)."""

    calls: list[tuple[str, tuple[Any, ...]]]
    generation: int
    page_serial: int
    active: str
    #: the turn the row is executing (set by ``act``); recorded with each fronting
    turn: str = "-"
    front_mode: str = "ok"
    front_error: BaseException = FrontBoom("synthetic fronting failure")
    front_cancelled: int = 0

    async def bring_to_front(self) -> None:
        self.calls.append((FRONT, (self.turn, self.generation, self.page_serial, self.active)))
        if self.front_mode == "raise":
            raise self.front_error
        if self.front_mode == "hang":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.front_cancelled += 1
                raise


class SettleMixin:
    """The next ``settle()`` adopts ``settle_popup`` (a popup the action opened)."""

    calls: list[tuple[str, tuple[Any, ...]]]
    page_serial: int
    active: str
    scene: int
    settle_popup: str | None = None

    async def settle(self) -> None:
        await super().settle()  # type: ignore[misc]
        if self.settle_popup is not None:
            self.active, self.settle_popup = self.settle_popup, None
            self.page_serial += 1
            self.scene += 1


class PlainDriver(SettleMixin, FakeDriver):
    """No capability."""


class FrontDriver(FrontMixin, SettleMixin, FakeDriver):
    pass


class FrontClosable(FrontMixin, ClosableDriver):
    pass


class FrontReset(FrontMixin, ResetDriver):
    pass


class FrontPage(FrontMixin, PageDriver):
    pass


class FrontUrls(FrontMixin, UrlsDriver):
    pass


def const(name: str) -> Any:
    value = getattr(executor_mod, name, None)
    assert value is not None, f"T-0198 C6: executor.{name} does not exist yet"
    return value


def build(d: FakeDriver, *, headless: bool = False, web: bool = False, **kw: Any) -> Any:
    if web:
        kw.setdefault("start_origin", ORIGIN)
        kw.setdefault("allowed_origins", frozenset({ORIGIN}))
    return executor_mod.BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS,
        viewport=d.viewport,
        driver=d,
        headless=headless,
        web=web,
        **kw,
    )


_N = iter(range(1_000_000))


def call(
    name: str, frame: Frame | FrameGeometry | None = None, *, turn: str = "t1", **arguments: Any
) -> ToolInvocation:
    g = (
        FrameGeometry(
            phash=frame.phash, source=frame.source, width=frame.width, height=frame.height
        )
        if isinstance(frame, Frame)
        else frame
    )
    return ToolInvocation(
        call_id=f"f{next(_N)}",
        turn_id=turn,
        name=name,
        arguments=arguments,
        source="draft",
        frame=g,
    )


async def framed(ex: Any) -> Frame:
    """``own_frame`` bounded (a ``frame()`` that waited on a hung fronting would otherwise hang the row)."""
    try:
        return await asyncio.wait_for(own_frame(ex), 10)
    except TimeoutError:
        raise AssertionError("T-0198 C5: frame() hung (did it front?)") from None


async def act(ex: Any, d: Any, i: ToolInvocation) -> ToolOutcome:
    d.turn = i.turn_id
    out = await asyncio.wait_for(ex.execute(i), 10)
    assert isinstance(out, ToolOutcome), type(out)
    return out


async def scroll(ex: Any, d: Any, turn: str = "t1") -> ToolOutcome:
    return await act(ex, d, call("scroll", turn=turn, direction="down"))


async def click(ex: Any, d: Any, turn: str = "t1", frame: Frame | None = None) -> ToolOutcome:
    f = frame if frame is not None else ex.latest_frame
    if f is None:
        f = await framed(ex)
    return await act(ex, d, call("click_at", f, turn=turn, x=500, y=500))


def fronts(d: Any, since: int = 0) -> list[tuple[Any, ...]]:
    return [c[1] for c in d.calls[since:] if c[0] == FRONT]


def front_at(d: Any, since: int = 0) -> list[int]:
    return [i for i, c in enumerate(d.calls) if i >= since and c[0] == FRONT]


def first(d: Any, names: set[str] | frozenset[str], since: int) -> int:
    hits = [i for i, c in enumerate(d.calls) if i >= since and c[0] in names]
    assert hits, f"premise: no {sorted(names)} call after {since}: {d.calls[since:]}"
    return hits[0]


def non_front(d: Any) -> list[tuple[str, tuple[Any, ...]]]:
    return [c for c in d.calls if c[0] != FRONT]


def dump(out: ToolOutcome, latest: Frame | None) -> str:
    resolved = dict(out.resolved)
    if latest is not None and resolved.get("frame_after") == latest.phash:
        resolved["frame_after"] = "<latest>"
    return json.dumps([out.status, out.action, out.reason, resolved], sort_keys=True)


@pytest.fixture
def fixed_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor_mod.secrets, "randbits", lambda k: SALT)


ACTIONS = [
    pytest.param("click_at", {"x": 500, "y": 500}, id="click_at"),
    pytest.param(
        "type_at", {"x": 500, "y": 500, "text": "hello", "press_enter": True}, id="type_at"
    ),
    pytest.param("scroll", {"direction": "down"}, id="scroll"),
    pytest.param("press_key", {"combo": "Ctrl+A"}, id="press_key"),
]


# =====================================================================================================
# C1: once per turn, before the first input
# =====================================================================================================


@pytest.mark.parametrize(("tool", "args"), ACTIONS)
async def test_first_dispatched_action_of_turn_fronts_once_before_input(
    tool: str, args: dict[str, Any]
) -> None:
    """C1: headed, the capability on the class, ``raise_window`` left at its default: the turn's first dispatched
    action calls ``bring_to_front`` exactly once, after its own ``ensure()`` and before its first input call, on the
    active page (generation 1, page serial 0, the opener)."""
    d = FrontDriver()
    ex = build(d)
    assert getattr(ex, "raise_window", None) is True, (
        "T-0198 C4: BrowserToolExecutor.raise_window exists and defaults to True"
    )
    f = await framed(ex)
    mark = len(d.calls)
    out = await act(ex, d, call(tool, f if tool in ("click_at", "type_at") else None, **args))
    assert out.status == "executed", out
    assert fronts(d, mark) == [("t1", 1, 0, "opener")], (
        f"T-0198 C1: one fronting of the active page; calls {d.calls[mark:]}"
    )
    i_front = front_at(d, mark)[0]
    i_ensure = first(d, {"ensure"}, mark)
    i_input = first(d, INPUT, mark)
    assert i_ensure < i_front < i_input, (
        f"T-0198 C1: ensure() < bring_to_front < first input; calls {d.calls[mark:]}"
    )


async def test_same_turn_same_page_no_second_front() -> None:
    """C1: three more actions in t1 on the same page front nothing; t2's first fronts once, its second nothing; t3's
    first once more. Keys by hand: (t1, 1, 0), (t2, 1, 0), (t3, 1, 0)."""
    d = FrontDriver()
    ex = build(d)
    await framed(ex)
    for turn, n in (("t1", 4), ("t2", 2), ("t3", 1)):
        for k in range(n):
            out = await (click(ex, d, turn) if k % 2 else scroll(ex, d, turn))
            assert out.status == "executed", (turn, k, out)
    assert fronts(d) == [("t1", 1, 0, "opener"), ("t2", 1, 0, "opener"), ("t3", 1, 0, "opener")], (
        f"T-0198 C1: once per turn on one page; fronts {fronts(d)}"
    )
    for i in front_at(d):
        nxt = first(d, INPUT, i)
        assert all(c[0] != "screenshot" for c in d.calls[i:nxt]), (
            f"T-0198 C1: each fronting is followed by an input, not a capture; {d.calls[i : nxt + 1]}"
        )


async def test_overview_frame_click_fronts_before_dispatch() -> None:
    """C1 on the T-0151 dispatch site (``_act_on_overview``): a click on a registered overview frame is the turn's
    first dispatched action: one fronting, after the capture of the overview, before its first input."""
    from tests.unit.test_t0151_executor import register

    d = FrontPage(content_height=1220.0)
    ex = build(d, echo="model")
    await framed(ex)
    d.turn = "t1"
    ov = await register(ex)
    assert fronts(d) == [], f"T-0198 C5: page_overview never fronts; {fronts(d)}"
    mark = len(d.calls)
    out = await act(ex, d, call("click_at", ov, x=160, y=308))
    assert out.status == "executed", out
    assert fronts(d, mark) == [("t1", 1, 0, "opener")], d.calls[mark:]
    assert front_at(d, mark)[0] < first(d, INPUT, mark), d.calls[mark:]


# =====================================================================================================
# C2: on a page change
# =====================================================================================================


@pytest.mark.parametrize(
    "later", [pytest.param(False, id="first-action"), pytest.param(True, id="later-action")]
)
async def test_popup_in_settle_fronts_after_settle_before_capture(later: bool) -> None:
    """C2: the action's ``settle()`` adopts a popup: one fronting of the popup (t1, 1, 1), after that settle and before
    the after-frame's screenshot, which is the popup's. The next t1 action on the popup fronts nothing."""
    d = FrontDriver()
    ex = build(d)
    f = await framed(ex)
    want = [("t1", 1, 0, "opener")]
    if later:
        assert (await scroll(ex, d)).status == "executed"
        f = ex.latest_frame
    mark = len(d.calls)
    d.settle_popup = "popup"
    out = await click(ex, d, frame=f)
    assert out.status == "executed", out
    got = fronts(d)
    assert got == [*want, ("t1", 1, 1, "popup")], f"T-0198 C2: fronts {got}; calls {d.calls[mark:]}"
    i_new = front_at(d, mark)[-1]
    settles = [i for i in range(mark, i_new) if d.calls[i][0] == "settle"]
    shots = [i for i, c in enumerate(d.calls) if i > i_new and c[0] == "screenshot"]
    assert settles and shots, f"T-0198 C2: settle < front(popup) < screenshot; {d.calls[mark:]}"
    assert not any(c[0] == "screenshot" for c in d.calls[settles[-1] : i_new]), d.calls[mark:]
    assert d.shots[-1] == "popup", "premise: the after-frame is the popup's"
    assert (await click(ex, d)).status == "executed"
    assert len(fronts(d)) == len(want) + 1, f"T-0198 C2: the popup's key fronted once; {fronts(d)}"


@pytest.mark.parametrize("found_by", ["refused-execute", "frame"])
async def test_late_popup_in_ensure_fronts_before_dispatch(found_by: str) -> None:
    """C2 (T-0187): a popup opens after t1's first action settled. Found by the next ``execute()``'s ``ensure()``
    (refused with ``PAGE_CHANGED_REASON``, nothing fronted) or by ``frame()`` (nothing fronted): the next dispatched
    action fronts the popup (t1, 1, 1) once, after its ensure, before its first input."""
    d = FrontDriver()
    ex = build(d)
    await framed(ex)
    assert (await scroll(ex, d)).status == "executed"
    d.open_popup()
    if found_by == "refused-execute":
        out = await scroll(ex, d)
        assert (out.status, out.reason) == ("refused", executor_mod.PAGE_CHANGED_REASON), out
    f = await framed(ex)
    assert d.active == "popup", "premise: the popup is adopted"
    assert fronts(d) == [("t1", 1, 0, "opener")], (
        f"T-0198 C5: a refusal and frame() front nothing; {fronts(d)}"
    )
    mark = len(d.calls)
    assert (await click(ex, d, frame=f)).status == "executed"
    assert fronts(d, mark) == [("t1", 1, 1, "popup")], d.calls[mark:]
    assert first(d, {"ensure"}, mark) < front_at(d, mark)[0] < first(d, INPUT, mark), d.calls[mark:]


@pytest.mark.parametrize("combo", ["Ctrl+T", "Ctrl+L"])
@pytest.mark.parametrize(
    "later", [pytest.param(False, id="first-action"), pytest.param(True, id="later-action")]
)
async def test_replace_page_during_ctrl_t_fronts_new_page(combo: str, later: bool) -> None:
    """C2 (T-0188): web mode, the start-page navigation is refused by the page, the executor replaces the page
    (``page_serial`` +1, ``active == "fresh"``): the fresh page is fronted once, after the action's settle, before its
    after-frame screenshot. As the turn's first action the opener is fronted first, before the navigation."""
    d = FrontReset()
    ex = build(d, web=True)
    await ex.open(WEB_URL)
    await ex.open(APP)
    if later:
        assert (await scroll(ex, d)).status == "executed"
    d.arm_block(RuntimeError("net::ERR_ABORTED"))
    mark = len(d.calls)
    out = await act(ex, d, call("press_key", combo=combo))
    assert out.status == "executed", out
    assert d.active == "fresh" and d.page_serial == 1, "premise: the page was replaced"
    assert fronts(d) == [("t1", 1, 0, "opener"), ("t1", 1, 1, "fresh")], (
        f"T-0198 C1/C2: fronts {fronts(d)}; calls {d.calls[mark:]}"
    )
    i_new = front_at(d, mark)[-1]
    i_replace = first(d, {"replace_page"}, mark)
    settles = [i for i in range(i_replace, i_new) if d.calls[i][0] == "settle"]
    assert settles, (
        f"T-0198 C2: the new page is fronted after the action's settle; {d.calls[mark:]}"
    )
    assert any(c[0] == "screenshot" for c in d.calls[i_new:]), d.calls[mark:]
    if not later:
        assert front_at(d, mark)[0] < first(d, {"goto"}, mark), (
            f"T-0198 C1: the opener is fronted before the navigation; {d.calls[mark:]}"
        )


# =====================================================================================================
# C3, C4: headless, raise_window
# =====================================================================================================


async def _scenario(ex: Any, d: Any) -> list[str]:
    """Turns, a popup in a settle, a late popup refused then taken, a refusal, a restart: the outcomes."""
    outs: list[str] = []
    await framed(ex)
    for o in (
        await click(ex, d, "t1"),
        await scroll(ex, d, "t1"),
    ):
        outs.append(dump(o, ex.latest_frame))
    d.settle_popup = "popup-a"
    outs.append(dump(await click(ex, d, "t1"), ex.latest_frame))
    outs.append(dump(await scroll(ex, d, "t2"), ex.latest_frame))
    d.open_popup("popup-b")
    outs.append(dump(await scroll(ex, d, "t2"), ex.latest_frame))
    outs.append(dump(await act(ex, d, call("open_the_pod_bay_doors", turn="t2")), ex.latest_frame))
    f = await framed(ex)
    outs.append(dump(await click(ex, d, "t2", frame=f), ex.latest_frame))
    d.kill()
    outs.append(dump(await scroll(ex, d, "t3"), ex.latest_frame))
    await framed(ex)
    outs.append(dump(await click(ex, d, "t3"), ex.latest_frame))
    outs.append(dump(await scroll(ex, d, "t4"), ex.latest_frame))
    return outs


@pytest.mark.parametrize("raise_window", [None, True])
async def test_headless_never_fronts(fixed_salt: None, raise_window: bool | None) -> None:
    """C3 adversarial: ``headless=True`` (the default) with the capability on the class: zero fronting over turns,
    popups, refusals and a restart, ``raise_window`` left alone or set ``True``. The headed run of the same scenario
    fronts (the premise that the scenario reaches fronting at all)."""
    headed = FrontDriver()
    await _scenario(build(headed), headed)
    assert len(fronts(headed)) >= 4, f"premise: the headed scenario fronts; {fronts(headed)}"
    d = FrontDriver()
    ex = executor_mod.BrowserToolExecutor(now_ms=lambda: CLOCK_MS, viewport=d.viewport, driver=d)
    if raise_window is not None:
        ex.raise_window = raise_window
    await _scenario(ex, d)
    assert fronts(d) == [], f"T-0198 C3: headless fronted {fronts(d)}"


async def test_raise_window_false_never_fronts_then_resumes() -> None:
    """C4: with ``raise_window`` ``False`` a turn with actions, a popup in a settle and a late popup fronts nothing; set
    back to ``True``, the next turn's first action fronts once (t2 on the page then active)."""
    d = FrontDriver()
    ex = build(d)
    ex.raise_window = False
    await framed(ex)
    assert (await click(ex, d)).status == "executed"
    d.settle_popup = "popup"
    assert (await scroll(ex, d)).status == "executed"
    d.open_popup("late")
    await framed(ex)
    assert (await click(ex, d)).status == "executed"
    assert fronts(d) == [], f"T-0198 C4: raise_window False fronted {fronts(d)}"
    ex.raise_window = True
    assert (await scroll(ex, d, "t2")).status == "executed"
    assert (await scroll(ex, d, "t2")).status == "executed"
    assert fronts(d) == [("t2", 1, 2, "late")], (
        f"T-0198 C4: a later browser turn fronts; {fronts(d)}"
    )


# =====================================================================================================
# C5: never on a refusal, a search, an overview, frame/open/reset
# =====================================================================================================


def _guard_denies(ids: set[str]) -> Any:
    return lambda inv: "the guard says no" if inv.call_id in ids else None


async def _refusal(case: str) -> tuple[Any, Any, ToolOutcome, str, str | None]:
    """Build the case and run its refused call: ``(ex, d, outcome, want_status, want_reason)``; ``want_reason`` None
    means "not executed" with any reason (premise only)."""
    if case == "closed":
        d: Any = FrontClosable()
        ex = build(d, end_when_closed=True)
        await framed(ex)
        assert (await scroll(ex, d, "t0")).status == "executed"
        d.close_all_pages()
        return ex, d, await scroll(ex, d), "refused", executor_mod.CLOSED_REASON
    if case == "checkout":
        d = FrontUrls()
        ex = build(d, web=True)
        await framed(ex)
        d.visit(CHECKOUT)
        return ex, d, await scroll(ex, d), "refused", executor_mod.CHECKOUT_REASON
    if case == "paste":
        d = FrontDriver()
        ex = build(d, web=True)
        await framed(ex)
        out = await act(ex, d, call("press_key", combo="Ctrl+V"))
        return ex, d, out, "refused", executor_mod.PASTE_REASON
    deny: set[str] = set()
    d = FrontDriver()
    ex = build(d, guard=_guard_denies(deny))
    f = await framed(ex)
    if case == "guard":
        i = call("click_at", f, x=500, y=500)
        deny.add(i.call_id)
        return ex, d, await act(ex, d, i), "refused", "the guard says no"
    if case == "unknown-tool":
        return ex, d, await act(ex, d, call("open_the_pod_bay_doors")), "refused", None
    if case == "no-frame":
        return ex, d, await act(ex, d, call("click_at", None, x=5, y=5)), "refused", None
    if case == "foreign-frame":
        g = FrameGeometry(phash="0123456789abcdef", source=Source.WINDOW, width=1280, height=800)
        return ex, d, await act(ex, d, call("click_at", g, x=5, y=5)), "refused", None
    if case == "bad-arguments":
        return ex, d, await act(ex, d, call("scroll", direction="sideways")), "refused", None
    if case == "card-text":
        i = call("type_at", f, x=500, y=500, text=CARD, press_enter=False)
        return ex, d, await act(ex, d, i), "refused", executor_mod.CARD_TEXT_REASON
    if case == "browser-restarted":
        d.kill()
        return ex, d, await scroll(ex, d), "failed", None
    if case == "page-changed-serial-moved":
        d.open_popup()
        return ex, d, await scroll(ex, d), "refused", executor_mod.PAGE_CHANGED_REASON
    if case == "page-changed-stale-frame":
        d.open_popup()
        await framed(ex)
        return ex, d, await click(ex, d, frame=f), "refused", executor_mod.PAGE_CHANGED_REASON
    raise AssertionError(case)


REFUSALS = [
    "unknown-tool",
    "no-frame",
    "foreign-frame",
    "bad-arguments",
    "card-text",
    "paste",
    "guard",
    "closed",
    "browser-restarted",
    "page-changed-serial-moved",
    "page-changed-stale-frame",
    "checkout",
]


@pytest.mark.parametrize("case", REFUSALS)
async def test_refusals_never_front(case: str) -> None:
    """C5 adversarial: a call refused (or failed before any dispatch) fronts nothing and dispatches nothing, and does
    not use up the turn's fronting: the next dispatched action of the same turn fronts once (except after the browser
    ended)."""
    ex, d, out, status, reason = await _refusal(case)
    assert out.status == status, f"premise ({case}): {out}"
    if reason is not None:
        assert out.reason == reason, f"premise ({case}): {out}"
    if case == "browser-restarted":
        assert out.reason == "browser restarted", out
    assert [k for k in fronts(d) if k[0] != "t0"] == [], (
        f"T-0198 C5 ({case}): a refused call fronted {fronts(d)}"
    )
    if case == "closed":
        assert fronts(d) == [("t0", 1, 0, "opener")], (
            f"premise: the headed rig fronts an executed action (t0); {fronts(d)}"
        )
        return
    if case == "checkout":
        d.visit(f"{SHOP}/")
    f = await framed(ex)
    mark = len(d.calls)
    after = await click(ex, d, frame=f)
    assert after.status == "executed", f"premise ({case}): {after}"
    assert len(fronts(d, mark)) == 1, (
        f"T-0198 C1/C5 ({case}): the refusal did not mark the key; the next action fronts once: {fronts(d)}"
    )


async def test_search_overview_frame_open_reset_never_front() -> None:
    """C5: ``google_search``, ``page_overview`` (an executed whole-page capture), ``frame()``, ``open()`` and
    ``reset_page()``, over three turns, front nothing; the first dispatched action after them fronts once."""
    d = FrontPage(content_height=2400.0)
    ex = build(d, allowed_origins=frozenset({ORIGIN}))
    await ex.open(f"{ORIGIN}/index.html")
    for turn in ("t1", "t2", "t3"):
        await framed(ex)
        out = await act(ex, d, call("google_search", turn=turn, query="teapots"))
        assert out.status != "executed", f"premise: a search is not a browser action; {out}"
        ov = await act(ex, d, call("page_overview", turn=turn))
        assert ov.status == "executed", f"premise: the overview was taken; {ov}"
        await ex.open(APP)
        await ex.reset_page()
        await framed(ex)
    assert fronts(d) == [], f"T-0198 C5: fronted {fronts(d)}"
    assert (await scroll(ex, d, "t3")).status == "executed"
    assert fronts(d) == [("t3", 1, 0, "opener")], fronts(d)


# =====================================================================================================
# C6: best effort, bounded
# =====================================================================================================


async def _pair(front: Any, plain: Any, steps: Any) -> tuple[list[str], list[str]]:
    a, b = build(front), build(plain)
    return await steps(a, front), await steps(b, plain)


async def _three(ex: Any, d: Any) -> list[str]:
    f = await framed(ex)
    outs = [dump(await click(ex, d, "t1", frame=f), ex.latest_frame)]
    outs.append(dump(await scroll(ex, d, "t1"), ex.latest_frame))
    outs.append(dump(await click(ex, d, "t2"), ex.latest_frame))
    return outs


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(FrontBoom("x"), id="RuntimeError"),
        pytest.param(OtherBoom("x"), id="Exception"),
        pytest.param(TimeoutError(), id="TimeoutError"),
        pytest.param(ValueError("x"), id="ValueError"),
    ],
)
async def test_front_raises_outcome_unchanged(fixed_salt: None, error: Exception) -> None:
    """C6: ``bring_to_front`` raises: every outcome (status, action, reason, resolved) and every non-front driver call
    equals a driver without the capability; t1 tried once, t2 once."""
    front, plain = FrontDriver(), PlainDriver()
    front.front_mode, front.front_error = "raise", error
    got, want = await _pair(front, plain, _three)
    assert got == want, f"T-0198 C6: a fronting error changed an outcome:\n{got}\n{want}"
    assert non_front(front) == plain.calls, "T-0198 C6/C7: the driver calls differ beyond fronting"
    assert [k[0] for k in fronts(front)] == ["t1", "t2"], fronts(front)


async def test_front_hangs_bounded(fixed_salt: None) -> None:
    """C6, H-T198-4: ``FRONT_TIMEOUT_S`` is 1.0 and below ``ACTION_TIMEOUT_S``. A ``bring_to_front`` that never returns
    is cut at it: the action returns within 3 s (1 s + slack for a loaded box), the hung call was cancelled by then,
    and the outcomes and non-front calls equal a driver without the capability. The key is not tried again (t1's
    second action returns at once)."""
    bound = const("FRONT_TIMEOUT_S")
    assert bound == 1.0 and bound < executor_mod.ACTION_TIMEOUT_S, bound
    front, plain = FrontDriver(), PlainDriver()
    front.front_mode = "hang"
    a, b = build(front), build(plain)
    fa, fb = await framed(a), await framed(b)
    t0 = time.monotonic()
    got = dump(await click(a, front, frame=fa), a.latest_frame)
    took = time.monotonic() - t0
    assert took < 3.0, f"T-0198 C6: a hung fronting held the action {took:.2f} s"
    assert front.front_cancelled == 1, (
        "T-0198 C6: the hung fronting was not cut when the action returned"
    )
    assert got == dump(await click(b, plain, frame=fb), b.latest_frame), got
    t0 = time.monotonic()
    got2 = dump(await scroll(a, front), a.latest_frame)
    assert time.monotonic() - t0 < 0.5, "T-0198 C6: the failed key was tried again"
    assert got2 == dump(await scroll(b, plain), b.latest_frame)
    assert non_front(front) == plain.calls
    assert len(fronts(front)) == 1, fronts(front)


async def test_front_wait_counts_inside_action_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """C6 "the wait counts inside ``ACTION_TIMEOUT_S``": with the action bound patched to 0.3 s (below
    ``FRONT_TIMEOUT_S``), a hung fronting ends the call at the action bound, well before 1 s."""
    const("FRONT_TIMEOUT_S")
    monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", 0.3)
    d = FrontDriver()
    d.front_mode = "hang"
    ex = build(d)
    f = await framed(ex)
    t0 = time.monotonic()
    await click(ex, d, frame=f)
    took = time.monotonic() - t0
    assert fronts(d), "premise: the fronting was called"
    assert took < 0.85, f"T-0198 C6: the fronting wait ran outside ACTION_TIMEOUT_S ({took:.2f} s)"


async def test_front_cancelled_propagates() -> None:
    """C6: the task running ``execute()`` is cancelled while ``bring_to_front`` hangs: ``CancelledError`` reaches the
    caller (not swallowed into an outcome)."""
    d = FrontDriver()
    d.front_mode = "hang"
    ex = build(d)
    f = await framed(ex)
    d.turn = "t1"
    task = asyncio.ensure_future(ex.execute(call("click_at", f, x=500, y=500)))
    for _ in range(200):
        if fronts(d) or task.done():
            break
        await asyncio.sleep(0.005)
    assert fronts(d) and not task.done(), f"T-0198 C1: the fronting was never called; {d.calls}"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert d.dispatched() == [], "premise: nothing was dispatched before the cancel"


@pytest.mark.parametrize("mode", ["raise", "hang"])
async def test_failed_key_not_retried(mode: str) -> None:
    """C6 + C2: a page key whose fronting failed is not tried again in its turn, even once fronting works; a new
    page key (the next turn, a popup) is tried once each."""
    d = FrontDriver()
    d.front_mode = mode
    ex = build(d)
    await framed(ex)
    assert (await click(ex, d)).status == "executed"
    d.front_mode = "ok"
    for _ in range(3):
        assert (await scroll(ex, d)).status == "executed"
    assert fronts(d) == [("t1", 1, 0, "opener")], f"T-0198 C6: retried a failed key; {fronts(d)}"
    d.settle_popup = "popup"
    assert (await scroll(ex, d)).status == "executed"
    assert (await scroll(ex, d, "t2")).status == "executed"
    assert (await scroll(ex, d, "t2")).status == "executed"
    assert fronts(d) == [("t1", 1, 0, "opener"), ("t1", 1, 1, "popup"), ("t2", 1, 1, "popup")], (
        fronts(d)
    )


async def test_driver_without_capability_unchanged(fixed_salt: None) -> None:
    """C6/C7: a driver class without ``bring_to_front`` (its instance ``__getattr__`` raises on any unknown name):
    headed and headless runs of the same scenario give the same outcomes and the same driver calls."""
    const("FRONT_TIMEOUT_S")
    a, b = PlainDriver(), PlainDriver()
    got = await _scenario(build(a), a)
    want = await _scenario(build(b, headless=True), b)
    assert got == want
    assert a.calls == b.calls


# =====================================================================================================
# C7: headed and headless are the same apart from fronting (differential, Hypothesis)
# =====================================================================================================


async def test_outcomes_identical_headed_vs_headless(fixed_salt: None) -> None:
    """C7: the scripted scenario (turns, popups, refusals, a restart) gives identical outcomes and identical
    non-front driver calls headed and headless; the headed run fronted, the headless one did not."""
    a, b = FrontDriver(), FrontDriver()
    got = await _scenario(build(a), a)
    want = await _scenario(build(b, headless=True), b)
    assert fronts(a), "premise: the headed run fronted"
    assert fronts(b) == []
    assert got == want
    assert non_front(a) == non_front(b)


OPS = st.lists(
    st.one_of(
        st.tuples(st.sampled_from(["scroll", "click", "bad"]), st.booleans()),
        st.tuples(st.sampled_from(["frame", "settle_popup", "late_popup", "kill"]), st.just(False)),
    ),
    min_size=1,
    max_size=12,
)


async def _run_ops(ex: Any, d: Any, ops: list[tuple[str, bool]]) -> tuple[list[str], list[str]]:
    """Run ``ops``; return the outcomes and, per op, a violation message list (headed oracle)."""
    outs: list[str] = []
    bad: list[str] = []
    turn_n = 1
    popups = 0
    for name, next_turn in ops:
        if next_turn:
            turn_n += 1
        turn = f"t{turn_n}"
        mark = len(d.calls)
        if name == "frame":
            await framed(ex)
        elif name == "settle_popup":
            popups += 1
            d.settle_popup = f"s{popups}"
        elif name == "late_popup":
            popups += 1
            d.open_popup(f"l{popups}")
        elif name == "kill":
            d.kill()
        else:
            if name == "scroll":
                o = await scroll(ex, d, turn)
            elif name == "click":
                if ex.latest_frame is None:
                    await framed(ex)
                o = await click(ex, d, turn)
            else:
                o = await act(ex, d, call("open_the_pod_bay_doors", turn=turn))
            outs.append(dump(o, ex.latest_frame))
            new = fronts(d, mark)
            if o.status != "executed" and new:
                bad.append(f"{name} {o.status} fronted {new}")
            if o.status == "executed":
                key = (turn, d.generation, d.page_serial, d.active)
                if key not in fronts(d):
                    bad.append(f"{name} executed on {key} never fronted")
        if name in ("frame", "settle_popup", "late_popup", "kill") and fronts(d, mark):
            bad.append(f"{name} fronted {fronts(d, mark)}")
    keys = [k[:3] for k in fronts(d)]
    if len(keys) != len(set(keys)):
        bad.append(f"a page key fronted twice: {keys}")
    return outs, bad


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ops=OPS)
def test_property_front_once_per_page_key(ops: list[tuple[str, bool]]) -> None:
    """C1/C2/C3/C5/C7 as a property over random op sequences (small viewport for speed): headed, every executed
    action's page key (turn, generation, serial) is fronted, no key twice, nothing fronted by a refused call,
    ``frame()`` or a page event; headless fronts nothing; outcomes and non-front calls are identical."""

    async def both() -> None:
        a, b = FrontDriver((320, 200)), FrontDriver((320, 200))
        with mock.patch.object(executor_mod.secrets, "randbits", lambda k: SALT):
            got, bad = await _run_ops(build(a), a, ops)
            want, _ = await _run_ops(build(b, headless=True), b, ops)
        assert bad == [], f"T-0198: {bad}; ops {ops}; calls {a.calls}"
        assert fronts(b) == [], f"T-0198 C3: headless fronted {fronts(b)}"
        assert got == want, f"T-0198 C7: outcomes differ; ops {ops}"
        assert non_front(a) == non_front(b), f"T-0198 C7: driver calls differ; ops {ops}"

    asyncio.run(both())
