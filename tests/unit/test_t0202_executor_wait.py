"""T-0202 group B, B1-B6 (executor side): the ``wait`` tool on ``BrowserToolExecutor`` over the recording fake
drivers (docs/tasks/T-0202.md).

**The surface these rows pin** (the task's names; readings stated where it is silent):

- Module constants of ``local_astra.computer.executor``, exported in ``__all__`` and read at call time (B5):
  ``WAIT_S = 1.0`` (H-T202-2), ``WAIT_ACTION = "waited 1 s"`` (B1), ``MAX_CONSECUTIVE_WAITS = 3``,
  ``MAX_WAITS_PER_TURN = 6`` (H-T202-3), ``WAIT_RUN_REASON`` and ``WAIT_TURN_REASON`` (H-T202-5, typed here from the
  task, never imported).
- ``execute(wait)`` takes ``scroll``'s path (B1): the same refusals in the same order, then ``asyncio.sleep(WAIT_S)``
  where ``scroll`` dispatches ``scroll_by``, then everything ``scroll`` does after its dispatch (settle, fronting, the
  after-settle checkout read, ``_capture``, ``take_events``). The oracle is a **differential against scroll** on
  identical fakes: wait's driver-call sequence equals scroll's with the one ``scroll_by`` replaced by the sleep.
- The outcome: ``executed``, ``action == WAIT_ACTION``, ``resolved`` exactly ``frame_after`` (+ ``hosts``/
  ``hosts_more`` in web mode, + ``checkout`` when the after-settle read names one); never ``navigated`` (B2).
- Counters per ``inv.turn_id`` (B3/B4): a wait counts once its sleep has begun; the fourth wait in a row is refused
  with ``WAIT_RUN_REASON`` and the seventh counted wait of a turn with ``WAIT_TURN_REASON``, with no driver call, no
  sleep and no capture; only an executed ``click_at``/``type_at``/``scroll``/``press_key``/``page_overview``/
  overview-frame action resets the run.

**Observing the sleep.** ``asyncio.sleep`` is wrapped (``spy``): every delay is recorded, and when a driver is
attached the wrapper also appends ``("sleep", (delay,))`` to that driver's ``calls``, so the sleep's place in the
call sequence is visible. Most rows patch ``WAIT_S`` to ``FAST`` (a value no other sleep uses), so "the wait's sleep"
is the recorded delay equal to ``FAST``; the rows that pin the real duration run it (1 s each).

**Readings (reported):**

- R1 (B1 "action ``WAIT_ACTION``"): the action string is the module constant itself. The rows that patch ``WAIT_S``
  still expect ``WAIT_ACTION`` (``"waited 1 s"``): the constant is built once from ``WAIT_S = 1.0`` (at import),
  not re-formatted per call from the patched value.
- R2 (B3 "before ``ensure()``"): a run/turn-limit refusal makes no driver call at all and never calls the guard's
  verdict a second time is not pinned; the rows pin only "no driver call, no sleep, no capture".
- R3 (B2 "refused ``bad arguments`` before anything is called"): no driver call and no guard call.
- R4 (B5 "with the real constants, one ``execute(wait)`` takes at most ``ACTION_TIMEOUT_S``"): pinned on a painted
  page (the blank-forever page under the real constants would cost about 4 s of wall time per run; its bound is
  pinned with ``ACTION_TIMEOUT_S`` patched to 2.5 s, as the criterion says).

Regime: CPU over fakes, real asyncio time (``WAIT_S`` patched to 12.3 ms except in the duration rows), headless
unless a row says headed; n = 1, deterministic. BASE 5821b628 (main after Merge T-0202-a).
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
from collections.abc import Iterator
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.contracts.toolcall import ToolOutcome
from tests import _git_range as gr
from tests.unit.test_computer_executor import INPUT, FakeDriver
from tests.unit.test_t0151_executor import register
from tests.unit.test_t0192_checkout_executor import CHECKOUT, SHOP
from tests.unit.test_t0198_front_executor import FRONT
from tests.unit.test_t0200_blank_recapture import (
    WHITE,
    AllDriver,
    AllField,
    ShotDriver,
    act,
    build,
    call,
    frame_of,
)

#: B's BASE: main after Merge T-0202-a (a literal, never a moving ref)
BASE_SHA = "5821b628"

#: typed from the task (H-T202-2/3/5, B1); never imported
WAIT_S = 1.0
WAIT_ACTION = "waited 1 s"
MAX_CONSECUTIVE_WAITS = 3
MAX_WAITS_PER_TURN = 6
WAIT_RUN_REASON = "you have already waited three times in a row; act on the screen or answer"
WAIT_TURN_REASON = "you have already waited six times in this request; act on the screen or answer"
CHECKOUT_REASON = "this is a checkout page; the user finishes the purchase themselves"
PAGE_CHANGED_REASON = "the active page changed; take a new frame"
CLOSED_REASON = "the browser window was closed"
#: the patched wait duration: no other sleep in these rigs has this value
FAST = 0.0123
CONSTANTS = {
    "WAIT_S": WAIT_S,
    "WAIT_ACTION": WAIT_ACTION,
    "MAX_CONSECUTIVE_WAITS": MAX_CONSECUTIVE_WAITS,
    "MAX_WAITS_PER_TURN": MAX_WAITS_PER_TURN,
    "WAIT_RUN_REASON": WAIT_RUN_REASON,
    "WAIT_TURN_REASON": WAIT_TURN_REASON,
}


# ============================================================================================ seams


def need(name: str) -> Any:
    value = getattr(executor_mod, name, None)
    assert value is not None, f"T-0202 B1: executor.{name} does not exist yet"
    return value


class Spy:
    """Every ``asyncio.sleep`` delay; with a driver attached, the sleep also lands in ``driver.calls``."""

    def __init__(self) -> None:
        self.delays: list[float] = []
        self.driver: Any = None
        self.started = asyncio.Event()

    def attach(self, d: Any) -> Any:
        self.driver = d
        return d

    def waits(self) -> list[float]:
        return [x for x in self.delays if x == FAST]


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> Iterator[Spy]:
    s = Spy()
    real = asyncio.sleep

    async def rec(delay: float, result: Any = None) -> Any:
        s.delays.append(float(delay))
        if s.driver is not None:
            s.driver.calls.append(("sleep", (float(delay),)))
        if float(delay) == FAST or float(delay) >= 0.5:
            s.started.set()
        return await real(delay, result)

    def blocking(*a: Any) -> None:
        raise AssertionError("T-0202: a blocking time.sleep on the event loop")

    monkeypatch.setattr(asyncio, "sleep", rec)
    monkeypatch.setattr(time, "sleep", blocking)
    yield s


@pytest.fixture
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """``WAIT_S`` patched to ``FAST`` (B5: read at call time). Set even where it does not exist yet (BASE), so a row
    fails in its body on a named assertion, never as a setup error."""
    monkeypatch.setattr(executor_mod, "WAIT_S", FAST, raising=False)


@pytest.fixture(autouse=True)
def zero_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor_mod.secrets, "randbits", lambda k: 0)


def wait(turn: str = "t1", **arguments: Any) -> Any:
    return call("wait", turn=turn, **arguments)


def names(d: Any, since: int = 0) -> list[str]:
    return [c[0] for c in d.calls[since:]]


def screenshots(d: Any, since: int = 0) -> int:
    return names(d, since).count("screenshot")


async def ready(d: Any, **kw: Any) -> tuple[Any, Any]:
    """An executor over ``d`` with one own frame taken (the browser launched)."""
    ex = build(d, **kw)
    f = await frame_of(ex)
    return ex, f


def check_executed_wait(out: ToolOutcome) -> None:
    assert out.status == "executed", (
        f"T-0202 B1: wait is executed; got {out.status} {out.reason!r} ({out.action})"
    )
    assert out.action == WAIT_ACTION == need("WAIT_ACTION"), (
        f"T-0202 B1: the action is WAIT_ACTION ({WAIT_ACTION!r}); got {out.action!r}"
    )
    assert out.reason is None, out


def check_refused(out: ToolOutcome, reason: str) -> None:
    assert out.status == "refused" and out.reason == reason, (
        f"T-0202: refused with {reason!r}; got {out.status} {out.reason!r} ({out.action})"
    )
    assert out.resolved == {}, out


async def waits(ex: Any, d: Any, n: int, turn: str = "t1") -> list[ToolOutcome]:
    return [await act(ex, d, wait(turn)) for _ in range(n)]


# ============================================================================================ constants


def test_constants_pinned() -> None:
    """B1/B3/B4/B5, H-T202-2/3/5: the six constants exist at module level with the task's values and are exported in
    ``__all__``."""
    for name, want in CONSTANTS.items():
        got = getattr(executor_mod, name, None)
        assert got == want and type(got) is type(want), (
            f"T-0202: executor.{name} == {want!r}; got {got!r}"
        )
        assert name in executor_mod.__all__, f"T-0202: {name} is exported in executor.__all__"
    assert "wait" not in executor_mod._NAVIGATION_TOOLS, "T-0202 B2: wait is not a navigation tool"


# ============================================================================================ B1


async def test_wait_call_sequence(spy: Spy, fast: None) -> None:
    """B1: on the plain fake (headless, no capability) a wait is ``ensure``, the sleep of ``WAIT_S``, ``settle``,
    one ``screenshot``, ``take_events``: by hand, scroll's ``ensure, scroll_by, settle, screenshot, take_events``
    with the dispatch replaced by the sleep."""
    d = spy.attach(FakeDriver())
    ex, _ = await ready(d)
    since = len(d.calls)
    out = await act(ex, d, wait())
    check_executed_wait(out)
    assert d.calls[since:] == [
        ("ensure", ()),
        ("sleep", (FAST,)),
        ("settle", ()),
        ("screenshot", ()),
        ("take_events", ()),
    ], f"T-0202 B1: the wait's call sequence; got {names(d, since)}"


async def _differential(kind: str, spy: Spy, *, web: bool, headed: bool) -> tuple[list[Any], Any]:
    """One fresh ``AllDriver`` (fronting, navigation mark, checkout read, metrics) executor, one own frame, then one
    ``kind`` call; the calls after the frame, with a scroll's ``scroll_by`` replaced by the wait's sleep."""
    d = spy.attach(AllDriver())
    kw: dict[str, Any] = {"web": web}
    if headed:
        kw["headless"] = False
    ex, _ = await ready(d, **kw)
    since = len(d.calls)
    i = call("scroll", direction="down") if kind == "scroll" else wait()
    out = await act(ex, d, i)
    assert out.status == "executed", (kind, out)
    seq = [("sleep", (FAST,)) if c[0] == "scroll_by" else c for c in d.calls[since:]]
    return seq, out


@pytest.mark.parametrize(
    "web,headed", [(False, False), (True, False), (True, True)], ids=["plain", "web", "web-headed"]
)
async def test_wait_sequence_is_scrolls(spy: Spy, fast: None, web: bool, headed: bool) -> None:
    """B1: over every optional capability (fronting when headed, the checkout read in web mode, layout metrics),
    wait's driver-call sequence is scroll's with ``scroll_by`` replaced by the sleep: the same checks before it, the
    same settle, fronting, after-settle checkout read and capture after it."""
    scroll_seq, _ = await _differential("scroll", spy, web=web, headed=headed)
    wait_seq, out = await _differential("wait", spy, web=web, headed=headed)
    check_executed_wait(out)
    assert ("sleep", (FAST,)) in scroll_seq, "premise: the scroll dispatched once"
    assert wait_seq == scroll_seq, (
        f"T-0202 B1: wait takes scroll's path\n scroll {[c[0] for c in scroll_seq]}\n wait   "
        f"{[c[0] for c in wait_seq]}"
    )
    if headed:
        assert any(c[0] == FRONT for c in wait_seq), "premise: a headed wait fronts the page"
    if web:
        assert [c[0] for c in wait_seq].count("main_frame_urls") == 2, (
            "premise: the checkout read before and after the settle"
        )


async def test_wait_outcome_and_resolved_keys(spy: Spy) -> None:
    """B1 (the real ``WAIT_S``): one sleep of 1.0 s; ``executed`` with ``WAIT_ACTION``; ``resolved`` is exactly
    ``{frame_after}`` outside web mode and ``{frame_after, hosts}`` in web mode, ``frame_after`` the phash of the new
    ``latest_frame`` (a capture taken after the sleep); ``hosts_more`` past ``MAX_HOSTS``; ``checkout`` when the
    page reached a checkout during the wait."""
    d = spy.attach(ShotDriver())
    ex, f0 = await ready(d)
    t0 = time.monotonic()
    out = await act(ex, d, wait())
    took = time.monotonic() - t0
    check_executed_wait(out)
    assert spy.delays.count(1.0) == 1 and took >= 0.95, (spy.delays, took)
    assert set(out.resolved) == {"frame_after"}, out.resolved
    assert ex.latest_frame is not f0 and out.resolved["frame_after"] == ex.latest_frame.phash
    i_sleep = [i for i, c in enumerate(d.calls) if c == ("sleep", (1.0,))]
    i_shot = [i for i, c in enumerate(d.calls) if c[0] == "screenshot"]
    assert len(i_sleep) == 1 and i_shot[-1] > i_sleep[0], (
        "B1: the after-frame is captured after the sleep"
    )

    monkey = pytest.MonkeyPatch()
    monkey.setattr(executor_mod, "WAIT_S", FAST)
    try:
        w = spy.attach(AllDriver())
        ex, _ = await ready(w, web=True)
        w.events.append({"kind": "host", "host": "a.example"})
        out = await act(ex, w, wait())
        check_executed_wait(out)
        assert out.resolved == {
            "frame_after": ex.latest_frame.phash,
            "hosts": ["a.example"],
        }, f"T-0202 B1: web mode resolved; got {out.resolved}"
        mh = need("MAX_HOSTS") if hasattr(executor_mod, "MAX_HOSTS") else 32
        w.events.extend({"kind": "host", "host": f"h{k}.example"} for k in range(mh + 1))
        out = await act(ex, w, wait())
        check_executed_wait(out)
        assert len(out.resolved["hosts"]) == mh and out.resolved.get("hosts_more") == 1, (
            out.resolved
        )
        w.visit_on("settle", CHECKOUT)
        out = await act(ex, w, wait())
        check_executed_wait(out)
        assert out.resolved.get("checkout") is True, (
            f"T-0202 B1: the after-settle checkout read (T-0192 A4); got {out.resolved}"
        )
    finally:
        monkey.undo()


# ============================================================================================ B2


async def test_wait_no_input_no_navigation_read(spy: Spy, fast: None) -> None:
    """B2 (H-T202-9): over the navigation-capable fake, with a main-frame commit between two waits (so a read would
    say ``True``): no input call, no ``navigation_mark``/``navigated_since`` call, and no ``navigated`` key. Power: a
    ``press_key`` on the same driver does read it."""
    d = spy.attach(AllDriver())
    ex, _ = await ready(d, web=True)
    since = len(d.calls)
    outs = [await act(ex, d, wait())]
    d.commit_now()
    outs.append(await act(ex, d, wait()))
    for out in outs:
        check_executed_wait(out)
        assert "navigated" not in out.resolved, out.resolved
        assert "page" not in out.resolved, out.resolved
    got = names(d, since)
    assert not [n for n in got if n in INPUT], f"T-0202 B2: a wait dispatched input: {got}"
    assert "navigation_mark" not in got and "navigated_since" not in got, (
        f"T-0202 B2: a wait read the navigation fact: {got}"
    )
    out = await act(ex, d, call("press_key", combo="Escape"))
    assert "navigated" in out.resolved, "premise: the fake answers navigation reads"


@pytest.mark.parametrize(
    "arguments", [{"seconds": 2}, {"seconds": 1.0}, {"x": 1}, {"": None}, {"wait": True}]
)
async def test_wait_with_argument_refused_before_ensure(
    spy: Spy, fast: None, arguments: dict[str, Any]
) -> None:
    """B2: a wait with any argument is refused ``bad arguments``; no driver call, no guard call, no sleep; it does
    not count (three argument-free waits after it are all executed)."""
    need("WAIT_S")
    seen: list[Any] = []

    def guard(i: Any) -> None:
        seen.append(i.name)
        return None

    d = spy.attach(FakeDriver())
    ex, _ = await ready(d, guard=guard)
    since = len(d.calls)
    out = await act(ex, d, wait(**arguments))
    check_refused(out, "bad arguments")
    assert d.calls[since:] == [] and seen == [] and spy.waits() == [], (
        names(d, since),
        seen,
        spy.delays,
    )
    for out in await waits(ex, d, MAX_CONSECUTIVE_WAITS):
        check_executed_wait(out)


# ============================================================================================ B3


async def test_fourth_consecutive_wait_refused_no_sleep(spy: Spy, fast: None) -> None:
    """B3: three waits in a row are executed (three sleeps); the fourth is refused ``WAIT_RUN_REASON`` with no driver
    call (no ``ensure``, no fronting, no capture) and no sleep; ``latest_frame`` is unchanged by it."""
    d = spy.attach(AllDriver())
    ex, _ = await ready(d, web=True, headless=False)
    for out in await waits(ex, d, 3):
        check_executed_wait(out)
    assert len(spy.waits()) == 3
    latest = ex.latest_frame
    since = len(d.calls)
    out = await act(ex, d, wait())
    check_refused(out, WAIT_RUN_REASON)
    assert d.calls[since:] == [] and len(spy.waits()) == 3, names(d, since)
    assert ex.latest_frame is latest


async def test_run_reset_only_by_executed_action(spy: Spy, fast: None) -> None:
    """B3 (adversarial; mutant: any call resets the run): wait, a refused click (a frame from no browser), wait, a
    failed scroll (the driver raises in ``scroll_by``), wait, wait: the fourth wait is refused ``WAIT_RUN_REASON``.
    Power: with an executed scroll in that place instead, the same fourth wait is executed."""
    d = spy.attach(FakeDriver())
    ex, f = await ready(d)
    check_executed_wait(await act(ex, d, wait()))
    stranger = f.model_copy(update={"phash": "0123456789abcdef"})
    r = await act(ex, d, call("click_at", stranger, x=500, y=500))
    assert r.status == "refused", r
    check_executed_wait(await act(ex, d, wait()))
    d.arm_raise(1)  # the scroll's first seam call after ensure: scroll_by
    s = await act(ex, d, call("scroll", direction="down"))
    assert s.status == "failed", s
    check_executed_wait(await act(ex, d, wait()))
    check_refused(await act(ex, d, wait()), WAIT_RUN_REASON)

    d2 = spy.attach(FakeDriver())
    ex2, _ = await ready(d2)
    check_executed_wait(await act(ex2, d2, wait()))
    check_executed_wait(await act(ex2, d2, wait()))
    s = await act(ex2, d2, call("scroll", direction="down"))
    assert s.status == "executed", s
    check_executed_wait(await act(ex2, d2, wait()))
    check_executed_wait(await act(ex2, d2, wait()))


RESETS = ["click_at", "type_at", "scroll", "press_key", "page_overview", "overview_frame"]


@pytest.mark.parametrize("kind", RESETS)
async def test_executed_action_resets_run(spy: Spy, fast: None, kind: str) -> None:
    """B3: each of the six executed actions, after three waits, resets the run: the next three waits are executed
    and the fourth after them is refused. The page is a long page (the overview is taken and registered before the
    waits; the waits do not change its pixels, so the overview frame stays current)."""
    d = spy.attach(AllDriver())
    ex, f = await ready(d)
    ov = await register(ex) if kind == "overview_frame" else None
    for out in await waits(ex, d, 3):
        check_executed_wait(out)
    f = ex.latest_frame
    i = {
        "click_at": lambda: call("click_at", f, x=500, y=500),
        "type_at": lambda: call("type_at", f, x=500, y=500, text="milk", press_enter=False),
        "scroll": lambda: call("scroll", direction="down"),
        "press_key": lambda: call("press_key", combo="Escape"),
        "page_overview": lambda: call("page_overview"),
        "overview_frame": lambda: call("click_at", ov, x=500, y=900),
    }[kind]()
    out = await act(ex, d, i)
    assert out.status == "executed", f"premise: the {kind} is executed; {out}"
    for out in await waits(ex, d, 3):
        check_executed_wait(out)
    check_refused(await act(ex, d, wait()), WAIT_RUN_REASON)


async def test_search_does_not_reset_run(spy: Spy, fast: None) -> None:
    """B3: three waits, a ``google_search`` (the dry run), a fourth wait: refused ``WAIT_RUN_REASON``."""
    d = spy.attach(FakeDriver())
    ex, _ = await ready(d)
    for out in await waits(ex, d, 3):
        check_executed_wait(out)
    s = await act(ex, d, call("google_search", query="milk prices"))
    assert s.status == "dry_run", s
    check_refused(await act(ex, d, wait()), WAIT_RUN_REASON)


async def test_new_turn_resets(spy: Spy, fast: None) -> None:
    """B3/B4: a new ``turn_id`` resets both counts: three waits in t1 (the fourth refused), then three in t2 executed
    and its fourth refused; six counted waits in t3 (resets between), then t4's first wait is executed."""
    d = spy.attach(FakeDriver())
    ex, _ = await ready(d)
    for out in await waits(ex, d, 3, "t1"):
        check_executed_wait(out)
    check_refused(await act(ex, d, wait("t1")), WAIT_RUN_REASON)
    for out in await waits(ex, d, 3, "t2"):
        check_executed_wait(out)
    check_refused(await act(ex, d, wait("t2")), WAIT_RUN_REASON)
    for _k in range(2):
        for out in await waits(ex, d, 3, "t3"):
            check_executed_wait(out)
        s = await act(ex, d, call("scroll", turn="t3", direction="down"))
        assert s.status == "executed", s
    check_refused(await act(ex, d, wait("t3")), WAIT_TURN_REASON)
    check_executed_wait(await act(ex, d, wait("t4")))


async def test_failed_wait_after_sleep_counts(spy: Spy, fast: None) -> None:
    """B3 ("a wait counts once its sleep has begun"): a wait whose ``settle`` raises after the sleep is ``failed``
    and still counts: two executed waits after it, and the next is refused ``WAIT_RUN_REASON``."""
    d = spy.attach(FakeDriver())
    ex, _ = await ready(d)
    d.arm_raise(1)  # the first seam call after ensure: settle (the sleep is not a seam call)
    out = await act(ex, d, wait())
    assert out.status == "failed" and len(spy.waits()) == 1, (out, spy.delays)
    for out in await waits(ex, d, 2):
        check_executed_wait(out)
    check_refused(await act(ex, d, wait()), WAIT_RUN_REASON)


# ============================================================================================ B4


async def test_turn_cap_refuses_seventh(spy: Spy, fast: None) -> None:
    """B4: waits 1-3, an executed click, waits 4-6, an executed click, wait 7: refused ``WAIT_TURN_REASON`` with no
    driver call and no sleep; six wait sleeps in the turn in all (B5: at most ``MAX_WAITS_PER_TURN x WAIT_S``)."""
    d = spy.attach(FakeDriver())
    ex, _ = await ready(d)
    for _ in range(2):
        for out in await waits(ex, d, 3):
            check_executed_wait(out)
        c = await act(ex, d, call("click_at", ex.latest_frame, x=500, y=500))
        assert c.status == "executed", c
    since = len(d.calls)
    check_refused(await act(ex, d, wait()), WAIT_TURN_REASON)
    assert d.calls[since:] == [], names(d, since)
    for _k in range(3):
        c = await act(ex, d, call("press_key", combo="Escape"))
        assert c.status == "executed", c
        check_refused(await act(ex, d, wait()), WAIT_TURN_REASON)
    assert len(spy.waits()) == MAX_WAITS_PER_TURN, spy.delays
    assert sum(spy.waits()) <= MAX_WAITS_PER_TURN * FAST + 1e-9


async def test_limits_read_at_call_time(
    spy: Spy, fast: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B5 ("the constants are module-level and read at call time"): with ``MAX_CONSECUTIVE_WAITS`` patched to 1 the
    second wait in a row is refused; with ``MAX_WAITS_PER_TURN`` patched to 2 the third counted wait of a turn is
    refused; the sleep is the patched ``WAIT_S``."""
    need("MAX_CONSECUTIVE_WAITS")
    need("MAX_WAITS_PER_TURN")
    d = spy.attach(FakeDriver())
    ex, _ = await ready(d)
    monkeypatch.setattr(executor_mod, "MAX_CONSECUTIVE_WAITS", 1)
    check_executed_wait(await act(ex, d, wait("a")))
    check_refused(await act(ex, d, wait("a")), WAIT_RUN_REASON)
    monkeypatch.setattr(executor_mod, "MAX_CONSECUTIVE_WAITS", 3)
    monkeypatch.setattr(executor_mod, "MAX_WAITS_PER_TURN", 2)
    for out in await waits(ex, d, 2, "b"):
        check_executed_wait(out)
    check_refused(await act(ex, d, wait("b")), WAIT_TURN_REASON)
    assert spy.waits() == [FAST, FAST, FAST], spy.delays


# ============================================================================================ B5


async def test_blank_forever_wait_executed_within_timeout(
    spy: Spy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B5: a page that stays white forever (T-0200's fake), ``ACTION_TIMEOUT_S`` patched to 2.5 s and the real
    ``WAIT_S``: the wait is ``executed`` (the blank re-capture stops at its deadline), never a timeout failure, and
    returns within 2.5 s (+ 0.3 s of scheduling slack)."""
    need("WAIT_S")
    monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", 2.5)
    d = spy.attach(ShotDriver())
    ex, _ = await ready(d)
    d.forever = WHITE
    since = len(d.calls)
    t0 = time.monotonic()
    out = await act(ex, d, wait())
    took = time.monotonic() - t0
    check_executed_wait(out)
    assert took <= 2.5 + 0.3, f"T-0202 B5: the wait took {took:.2f} s > ACTION_TIMEOUT_S"
    assert screenshots(d, since) >= 2, "premise: the blank after-frame was re-taken (T-0200)"
    assert ex.latest_frame.jpeg == WHITE


async def test_real_constants_one_wait_bounded(spy: Spy) -> None:
    """B5 (reading R4): with the real constants on a painted page one wait sleeps exactly ``WAIT_S`` once and takes
    between ``WAIT_S`` and ``ACTION_TIMEOUT_S``."""
    d = spy.attach(FakeDriver())
    ex, _ = await ready(d)
    t0 = time.monotonic()
    out = await act(ex, d, wait())
    took = time.monotonic() - t0
    check_executed_wait(out)
    assert spy.delays == [WAIT_S], spy.delays
    assert WAIT_S - 0.05 <= took <= executor_mod.ACTION_TIMEOUT_S, took


# ============================================================================================ B6


async def test_cancel_during_sleep(spy: Spy, monkeypatch: pytest.MonkeyPatch) -> None:
    """B6: a wait cancelled during its sleep raises ``CancelledError``; nothing is captured after the sleep began and
    ``latest_frame`` is the same object; the cancelled wait counts (two more waits are executed, the next is refused
    ``WAIT_RUN_REASON``)."""
    need("WAIT_S")
    monkeypatch.setattr(executor_mod, "WAIT_S", 5.0)
    d = spy.attach(FakeDriver())
    ex, _ = await ready(d)
    latest = ex.latest_frame
    since = len(d.calls)
    task = asyncio.ensure_future(ex.execute(wait()))
    await asyncio.wait_for(spy.started.wait(), 3.0)
    assert ("sleep", (5.0,)) in d.calls[since:], "premise: the wait's sleep began"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 3.0)
    assert screenshots(d, since) == 0 and "settle" not in names(d, since), names(d, since)
    assert ex.latest_frame is latest
    monkeypatch.setattr(executor_mod, "WAIT_S", FAST)
    for out in await waits(ex, d, 2):
        check_executed_wait(out)
    check_refused(await act(ex, d, wait()), WAIT_RUN_REASON)


# ============================================================================================ refusals


async def test_wait_refused_at_checkout(spy: Spy, fast: None) -> None:
    """B1 (T-0192 A3): in web mode on a checkout page a wait is refused ``CHECKOUT_REASON`` with no sleep and no
    capture, and does not count: after the page leaves the checkout, three waits are executed before the fourth is
    refused ``WAIT_RUN_REASON``."""
    d = spy.attach(AllDriver())
    ex, _ = await ready(d, web=True)
    d.visit(CHECKOUT)
    for _ in range(3):
        since = len(d.calls)
        check_refused(await act(ex, d, wait()), CHECKOUT_REASON)
        assert screenshots(d, since) == 0 and "settle" not in names(d, since)
    assert spy.waits() == [], spy.delays
    d.push_state(f"{SHOP}/")
    for out in await waits(ex, d, 3):
        check_executed_wait(out)
    check_refused(await act(ex, d, wait()), WAIT_RUN_REASON)


def _scenario(name: str) -> tuple[dict[str, Any], Any]:
    """Executor keywords and a pre-call setup for one refusal (or failure) case on a fresh ``AllDriver``."""

    def guard(i: Any) -> str:
        return "not on this page"

    setups: dict[str, tuple[dict[str, Any], Any]] = {
        "guard": ({"guard": guard}, lambda d: None),
        "closed": ({"end_when_closed": True}, lambda d: d.close_all_pages()),
        "guard+closed": ({"guard": guard, "end_when_closed": True}, lambda d: d.close_all_pages()),
        "restart": ({}, lambda d: d.kill()),
        "popup": ({}, lambda d: d.open_popup()),
        "checkout": ({"web": True}, lambda d: d.visit(CHECKOUT)),
        "restart+popup": ({}, lambda d: (d.kill(), d.open_popup())),
        "popup+checkout": ({"web": True}, lambda d: (d.open_popup(), d.visit(CHECKOUT))),
        "guard+checkout": ({"guard": guard, "web": True}, lambda d: d.visit(CHECKOUT)),
        "ensure-timeout": ({}, lambda d: d.arm_hang("ensure")),
    }
    return setups[name]


SCENARIOS = [
    "guard",
    "closed",
    "guard+closed",
    "restart",
    "popup",
    "checkout",
    "restart+popup",
    "popup+checkout",
    "guard+checkout",
    "ensure-timeout",
]


@pytest.mark.parametrize("scenario", SCENARIOS)
async def test_refusal_order_matches_scroll(
    spy: Spy, fast: None, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    """B1 (adversarial; mutant: the wait checks in another order, or skips one): in each case, alone and two at a
    time, a wait gets the status and reason a scroll gets (the tool name in a failure reason aside) and makes the
    same driver calls, and it never sleeps."""
    monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", 0.4)
    got: dict[str, Any] = {}
    for kind in ("scroll", "wait"):
        kw, setup = _scenario(scenario)
        d = spy.attach(AllDriver())
        ex, _ = await ready(d, **kw)
        setup(d)
        since = len(d.calls)
        i = call("scroll", direction="down") if kind == "scroll" else wait()
        out = await act(ex, d, i)
        reason = (
            (out.reason or "")
            .replace(" during scroll", " during <tool>")
            .replace(" during wait", " during <tool>")
        )
        got[kind] = (out.status, reason, [c for c in d.calls[since:] if c[0] != "sleep"])
    s_status, s_reason, s_calls = got["scroll"]
    assert s_status != "executed", f"premise: {scenario} stops a scroll; {got['scroll']}"
    assert got["wait"][:2] == (s_status, s_reason), (
        f"T-0202 B1 ({scenario}): wait {got['wait'][:2]} != scroll {(s_status, s_reason)}"
    )
    assert got["wait"][2] == s_calls, (
        f"T-0202 B1 ({scenario}): driver calls differ\n scroll {[c[0] for c in s_calls]}\n wait   "
        f"{[c[0] for c in got['wait'][2]]}"
    )
    assert spy.waits() == [], f"T-0202 B1 ({scenario}): a stopped wait slept; {spy.delays}"


# ============================================================================================ B9/B10 (unit)

_BASE: types.ModuleType | None = None


def base_module() -> types.ModuleType:
    """BASE 5821b628's executor (no wait), executed from ``git show`` as T-0200's differential does."""
    global _BASE
    if _BASE is None:
        gr.require_commits(BASE_SHA)
        src = gr._git("show", f"{BASE_SHA}:local_astra/computer/executor.py").stdout
        assert "def _capture" in src and "WAIT_S" not in src, "premise: BASE has no wait"
        name = "_t0202_base_executor"
        mod = types.ModuleType(name)
        mod.__file__ = f"<{BASE_SHA}>/local_astra/computer/executor.py"
        sys.modules[name] = mod
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        _BASE = mod
    return _BASE


async def _trace(module: Any, spy: Spy) -> list[Any]:
    """A fixed script of non-wait calls (click, type with Enter, scroll, press_key, page_overview, a search, a refused
    click) on a headed web ``AllField``: every driver call with every sleep, and each outcome."""
    d = spy.attach(AllField())
    ex = build(d, module=module, web=True, headless=False)
    f = await frame_of(ex)
    trace: list[Any] = []
    for i in (
        call("click_at", f, x=500, y=500),
        call("type_at", None, x=200, y=200, text="milk", press_enter=True),
        call("scroll", direction="down"),
        call("press_key", combo="Escape"),
        call("page_overview"),
        call("google_search", query="milk"),
        call("click_at", f.model_copy(update={"phash": "0123456789abcdef"}), x=1, y=1),
    ):
        if i.name == "type_at":
            g = ex.latest_frame
            i = call("type_at", g, x=200, y=200, text="milk", press_enter=True)
        out = await act(ex, d, i)
        trace.append((i.name, out.status, out.action, out.reason, sorted(out.resolved)))
    # a navigation mark is an opaque object per driver: compared by its repr (index and commit count)
    return [*trace, *(repr(c) for c in d.calls)]


async def test_other_tools_byte_identical_to_base(spy: Spy) -> None:
    """B10/B9 (negative; "no sleep anywhere outside wait"): the same script of every other tool over BASE's executor
    and this one gives the same driver calls, the same sleeps (none on these painted pages) and the same outcomes."""
    base = await _trace(base_module(), spy)
    base_sleeps = list(spy.delays)
    spy.delays.clear()
    tip = await _trace(executor_mod, spy)
    assert spy.delays == base_sleeps == [], (spy.delays, base_sleeps)
    assert tip == base, "T-0202 B10: a non-wait tool changed its driver calls or outcome"
