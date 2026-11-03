"""T-0204 group A: the driver signals on the real ``PlaywrightDriver`` class without a browser
(docs/tasks/T-0204.md, A1-A4, "Tests to write" A).

**The rig.** A ``PlaywrightDriver`` is built, never launched. Its ``_context`` is a ``FakeContext`` whose
``new_cdp_session(page)`` hands out ``FakeSession``s that record every ``send``/``on``; a ``FakePage`` is adopted
through the driver's own ``_adopt`` and made the active page (generation 1). The rows fire Chromium's
``PerformanceTimeline.timelineEventAdded`` payloads, in the shape reading R1 recorded on the pinned Chromium 156
(``{"event": {"type": "layout-shift", "layoutShiftDetails": {"sources": [{"previousRect", "currentRect",
"nodeId"}]}}}``, rects ``{x, y, width, height}`` in viewport CSS px), to every session of that page that subscribed to
the event and was sent ``PerformanceTimeline.enable`` (whatever the send's outcome, so a failed arm is not rescued by
the fake). Every page, context and session attribute the rows do not model is recorded and raises.

**What is pinned** (from the criteria, never from an implementation):

- A1: ``screenshot_region(x, y, w, h)`` is one ``page.screenshot`` (``type="jpeg"``, ``quality=JPEG_QUALITY``,
  ``scale="css"``, not full page) whose ``clip`` is the rectangle clamped to the 1280x800 viewport (by hand below);
  ``ValueError`` with no page call when nothing of it is inside; ``RuntimeError`` with no active page; no other page
  call (no input, no scroll, no script).
- A2: ``shift_mark()``/``shifts_since(mark)``: the rects (previous and current of each source) of every entry after the
  mark on the same page, as ``(x, y, w, h)``; ``()`` when armed and quiet; ``None`` when unarmed, after a failed arm, for
  a foreign mark (``object()``, a ``navigation_mark``, another driver's), another generation or page, or no active
  page; ``MAX_SHIFT_ENTRIES = 256`` with an overflow since the mark read as one rect covering the viewport; never
  raises (malformed payloads included).
- A3: ``arm_shift_recorder()``: ``True``; idempotent per page with exactly one ``PerformanceTimeline.enable``
  (``{"eventTypes": ["layout-shift"]}``) per page; ``False`` on any failure, within ``SHIFT_ARM_TIMEOUT_S = 0.5`` (+ slack)
  when a CDP call hangs; ``CancelledError`` propagates; a relaunch drops everything.
- A4 (negative): ``PerformanceTimeline.enable`` is the only new CDP method, ``nodeId`` is never resolved (no ``DOM.*``),
  and ``layout_metrics()``'s answer is unchanged.

**Ambiguities reported, not guessed:** zero-area rects (Chromium reports ``{0, 0, 0, 0}`` for a source that moved fully
into or out of the viewport, R1) may be kept or dropped: the rows compare the non-zero rects only. The viewport the
clip is clamped to is the driver's ``viewport`` and the page's ``viewport_size``, which the rig keeps equal.

Rows red at BASE carried ``pending(task="T-0204")`` until the activation commit (activated for group A).

Regime: CPU, no browser, no socket; real asyncio time; arm budgets 0.5 s (the shipped constant).
"""

from __future__ import annotations

import asyncio
import time
import typing
from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest
from playwright.async_api import Error as PlaywrightError

from local_astra.computer import runtime as runtime_mod

W, H = 1280, 800
ORIGIN = "http://127.0.0.1:8124"
EVENT = "PerformanceTimeline.timelineEventAdded"
ENABLE = "PerformanceTimeline.enable"
#: A3: the arm's bound, and the slack a hung call may add on a loaded box
ARM_TIMEOUT_S = 0.5
SLACK_S = 0.15
JPEG = b"\xff\xd8\xff\xe0t0204-region\xff\xd9"

Rect = tuple[float, float, float, float]

#: what ``Page.getLayoutMetrics`` answers in the fake, and ``layout_metrics()``'s answer at BASE (by hand, T-0134 A4)
METRICS = {
    "cssLayoutViewport": {"pageX": 0, "pageY": 120, "clientWidth": W, "clientHeight": H},
    "cssContentSize": {"x": 0, "y": 0, "width": W, "height": 3000},
}
METRICS_ANSWER = {"scroll_y": 120.0, "viewport_height": 800.0, "content_height": 3000.0}


# --------------------------------------------------------------------------------------------- fakes


class _Strict:
    """Every attribute a row does not model is recorded in ``unexpected`` and raises when called."""

    unexpected: list[str]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        self.unexpected.append(name)

        def refuse(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError(f"T-0204 A: unexpected use of {type(self).__name__}.{name}")

        return refuse


class FakeSession(_Strict):
    def __init__(self, ctx: FakeContext, page: Any) -> None:
        self.unexpected = []
        self.ctx, self.page = ctx, page
        self.sent: list[tuple[str, Any]] = []
        self.handlers: dict[str, list[Callable[..., Any]]] = {}
        self.detached = False

    async def send(self, method: str, params: Any = None) -> Any:
        self.sent.append((method, params))
        plan = self.ctx.send_plan.get(method, "ok")
        if plan == "hang":
            await asyncio.Event().wait()
        if isinstance(plan, BaseException):
            raise plan
        if method == "Page.getLayoutMetrics":
            return METRICS
        return {}

    def on(self, event: str, handler: Callable[..., Any]) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def remove_listener(self, event: str, handler: Callable[..., Any]) -> None:
        if handler in self.handlers.get(event, []):
            self.handlers[event].remove(handler)

    async def detach(self) -> None:
        # ``detached`` is set when the call arrives, before its outcome (A-R3 "best-effort")
        self.detached = True
        plan = self.ctx.detach_plan
        if plan == "hang":
            await asyncio.Event().wait()
        if isinstance(plan, BaseException):
            raise plan


class FakeContext(_Strict):
    def __init__(self) -> None:
        self.unexpected = []
        self.sessions: list[FakeSession] = []
        #: per CDP method: "ok", "hang" or an exception to raise
        self.send_plan: dict[str, Any] = {}
        #: ``new_cdp_session``: "ok", "hang" or an exception to raise
        self.attach_plan: Any = "ok"
        #: ``FakeSession.detach``: "ok", "hang" or an exception to raise
        self.detach_plan: Any = "ok"
        self.closed = False

    async def new_cdp_session(self, page: Any) -> FakeSession:
        if self.attach_plan == "hang":
            await asyncio.Event().wait()
        if isinstance(self.attach_plan, BaseException):
            raise self.attach_plan
        s = FakeSession(self, page)
        self.sessions.append(s)
        return s

    async def close(self) -> None:
        self.closed = True

    def methods(self) -> list[str]:
        return [m for s in self.sessions for m, _ in s.sent]

    def enables(self, page: Any | None = None) -> list[Any]:
        return [
            p
            for s in self.sessions
            if page is None or s.page is page
            for m, p in s.sent
            if m == ENABLE
        ]


class FakeFrame:
    def __init__(self, url: str) -> None:
        self.url = url


class FakePage(_Strict):
    def __init__(self, ctx: FakeContext, path: str = "/tiles.html") -> None:
        self.unexpected = []
        self._ctx = ctx
        self.main_frame = FakeFrame(ORIGIN + path)
        self.handlers: dict[str, list[Callable[..., Any]]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.viewport_size = {"width": W, "height": H}

    @property
    def url(self) -> str:
        return self.main_frame.url

    @property
    def context(self) -> FakeContext:
        return self._ctx

    def on(self, event: str, handler: Callable[..., Any]) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def is_closed(self) -> bool:
        return False

    async def screenshot(self, **kwargs: Any) -> bytes:
        self.calls.append(("screenshot", kwargs))
        return JPEG

    def page_calls(self) -> list[str]:
        return [c[0] for c in self.calls] + self.unexpected


# --------------------------------------------------------------------------------------------- rig


def rig() -> tuple[Any, FakeContext, FakePage]:
    """A driver "after launch" (generation 1) whose active page is a ``FakePage`` on a ``FakeContext``."""
    d = runtime_mod.PlaywrightDriver(viewport=(W, H), on_egress=lambda e: None)
    ctx = FakeContext()
    page = FakePage(ctx)
    d._context = ctx
    d._adopt(page)
    d._page = page
    d.generation = 1
    return d, ctx, page


def other_page(d: Any, ctx: FakeContext, path: str = "/other.html") -> FakePage:
    page = FakePage(ctx, path)
    d._adopt(page)
    return page


def _fn(d: Any, name: str) -> Callable[..., Any]:
    fn = getattr(d, name, None)
    assert fn is not None, f"T-0204 A: PlaywrightDriver.{name} does not exist yet"
    return typing.cast(Callable[..., Any], fn)


async def arm(d: Any) -> Any:
    return await _fn(d, "arm_shift_recorder")()


def mark(d: Any) -> object:
    return _fn(d, "shift_mark")()


def since(d: Any, m: object) -> Any:
    return _fn(d, "shifts_since")(m)


def rect(x: float, y: float, w: float, h: float) -> dict[str, float]:
    return {"x": x, "y": y, "width": w, "height": h}


def source(prev: Rect, cur: Rect, node_id: int = 7) -> dict[str, Any]:
    return {"previousRect": rect(*prev), "currentRect": rect(*cur), "nodeId": node_id}


def payload(*sources: dict[str, Any], recent: bool = False) -> dict[str, Any]:
    """One ``timelineEventAdded`` params object in R1's recorded shape."""
    return {
        "event": {
            "frameId": "F0",
            "type": "layout-shift",
            "name": "",
            "time": 1793314128.8913,
            "layoutShiftDetails": {
                "value": 0.05,
                "hadRecentInput": recent,
                "lastInputTime": 0,
                "sources": list(sources),
            },
        }
    }


async def fire(ctx: FakeContext, page: FakePage, params: dict[str, Any]) -> int:
    """Deliver ``params`` as Chromium would: to every session of ``page`` that subscribed and was sent the enable.
    Returns how many handlers received it."""
    n = 0
    for s in ctx.sessions:
        if s.page is not page or not any(m == ENABLE for m, _ in s.sent):
            continue
        for handler in list(s.handlers.get(EVENT, [])):
            out = handler(params)
            if asyncio.iscoroutine(out):
                await out
            n += 1
    await asyncio.sleep(0)
    return n


async def shift(ctx: FakeContext, page: FakePage, *pairs: tuple[Rect, Rect]) -> None:
    n = await fire(ctx, page, payload(*(source(p, c, 10 + i) for i, (p, c) in enumerate(pairs))))
    assert n >= 1, (
        "T-0204 A2: the recorder never subscribed to PerformanceTimeline.timelineEventAdded"
    )


def nonzero(rects: Any) -> Counter[Rect]:
    assert isinstance(rects, tuple), f"shifts_since must return a tuple here, got {rects!r}"
    out: Counter[Rect] = Counter()
    for r in rects:
        assert isinstance(r, tuple) and len(r) == 4, r
        assert all(isinstance(v, float | int) for v in r), r
        if r[2] > 0 and r[3] > 0:
            out[(float(r[0]), float(r[1]), float(r[2]), float(r[3]))] += 1
    return out


def covers_viewport(r: Rect) -> bool:
    x, y, w, h = r
    return x <= 0 and y <= 0 and x + w >= W and y + h >= H


# --------------------------------------------------------------------------------------------- A2


A: Rect = (40.0, 20.0, 600.0, 200.0)
A2: Rect = (40.0, 170.0, 600.0, 200.0)
B: Rect = (40.0, 240.0, 600.0, 200.0)
B2: Rect = (40.0, 390.0, 600.0, 200.0)
C: Rect = (700.0, 500.0, 300.0, 100.0)
C2: Rect = (700.0, 620.0, 300.0, 100.0)
E: Rect = (10.0, 10.0, 50.0, 50.0)
E2: Rect = (10.0, 300.0, 50.0, 50.0)


async def test_shifts_since_returns_rects_after_mark() -> None:
    """A2: armed and quiet reads ``()`` (not ``None``); after two entries (one with two sources, one with
    ``hadRecentInput``), the read is exactly their previous and current rects; an entry before the mark is absent; a
    second read gives the same answer; a later mark reads ``()`` again."""
    d, ctx, page = rig()
    assert await arm(d) is True
    await shift(ctx, page, (E, E2))  # before the mark
    m = mark(d)
    assert since(d, m) == (), "armed and quiet must read as an empty tuple"
    await shift(ctx, page, (A, A2), (B, B2))
    await fire(ctx, page, payload(source(C, C2), recent=True))
    want = Counter({A: 1, A2: 1, B: 1, B2: 1, C: 1, C2: 1})
    got = since(d, m)
    assert nonzero(got) == want, f"after the mark: {got}"
    assert nonzero(since(d, m)) == want, "a second read of the same mark changed"
    m2 = mark(d)
    assert since(d, m2) == ()
    await shift(ctx, page, (E, E2))
    assert nonzero(since(d, m2)) == Counter({E: 1, E2: 1})
    assert nonzero(since(d, m)) == want + Counter({E: 1, E2: 1})


async def test_zero_area_rects_do_not_hide_the_others() -> None:
    """A2 with R1's real shape: a source that moved in from outside the viewport has a ``{0,0,0,0}`` previous rect.
    The non-zero rects of the entry are all returned (whether the zero one is kept is not pinned)."""
    d, ctx, page = rig()
    assert await arm(d) is True
    m = mark(d)
    await shift(ctx, page, ((0.0, 0.0, 0.0, 0.0), (40.0, 0.0, 600.0, 150.0)), (A, A2))
    got = since(d, m)
    assert nonzero(got) == Counter({(40.0, 0.0, 600.0, 150.0): 1, A: 1, A2: 1}), got
    for r in got:
        assert r in {A, A2, (40.0, 0.0, 600.0, 150.0), (0.0, 0.0, 0.0, 0.0)}, r


async def test_foreign_or_other_page_mark_none() -> None:
    """A2: ``None`` for ``object()``, ``None``, a ``navigation_mark`` of the same driver, another driver's shift mark,
    a mark from another generation, a mark whose page is not the active page, and with no active page. Entries that
    arrive on page A while page B is active are A's: back on A the mark reads them."""
    d, ctx, page = rig()
    assert await arm(d) is True
    m = mark(d)
    d2, ctx2, page2 = rig()
    assert await arm(d2) is True
    m_other_driver = mark(d2)
    await shift(ctx, page, (A, A2))
    await shift(ctx2, page2, (A, A2))
    for foreign in (object(), None, d.navigation_mark(), m_other_driver):
        assert since(d, foreign) is None, f"foreign mark {foreign!r} was read"
    assert nonzero(since(d, m)) == Counter({A: 1, A2: 1}), "premise: the own mark reads"

    # another page
    b = other_page(d, ctx)
    d._page = b
    assert since(d, m) is None, "a mark of page A read while page B is active"
    mb = mark(d)
    assert since(d, mb) is None, "page B is not armed"
    assert await arm(d) is True
    mb = mark(d)
    await shift(ctx, page, (B, B2))  # on A while B is active
    assert since(d, mb) == (), "an entry of page A was read on page B"
    await shift(ctx, b, (C, C2))  # on B, armed and active
    assert nonzero(since(d, mb)) == Counter({C: 1, C2: 1})
    assert since(d, m) is None, "a mark of page A read on page B (armed) as if it were B's"
    d._page = page
    assert nonzero(since(d, m)) == Counter({A: 1, A2: 1, B: 1, B2: 1})

    # no active page
    d._page = None
    assert since(d, m) is None

    # another generation, same page object
    d._page = page
    d.generation = 2
    assert since(d, m) is None, "a mark of generation 1 read in generation 2"
    # armed again in generation 2 (however the driver does it on this artificial same page object) and shifting:
    # still None for the old mark
    await arm(d)
    await fire(ctx, page, payload(source(E, E2)))
    assert since(d, m) is None, "a mark of generation 1 read in generation 2 once re-armed"


async def test_overflow_reads_as_whole_viewport() -> None:
    """Adversarial (A2 bounded memory): ``MAX_SHIFT_ENTRIES`` is 256. 257 entries after a mark, each a 2x2 shift in
    the bottom-right corner, must read as one rect covering the whole viewport, never as the corner (a dropped entry
    may have been anywhere); more entries keep it so. A mark taken after the flood reads only what follows it; 256
    entries after a mark with 10 before it drop nothing since the mark and are all returned."""
    assert getattr(runtime_mod, "MAX_SHIFT_ENTRIES", None) == 256, "A2: MAX_SHIFT_ENTRIES = 256"
    d, ctx, page = rig()
    assert await arm(d) is True
    m0 = mark(d)
    tiny, tiny2 = (1270.0, 790.0, 2.0, 2.0), (1271.0, 791.0, 2.0, 2.0)
    for _ in range(257):
        await shift(ctx, page, (tiny, tiny2))
    got = since(d, m0)
    assert isinstance(got, tuple) and len(got) == 1, f"overflow read as {len(got or ())} rect(s)"
    assert covers_viewport(got[0]), f"overflow rect {got[0]} does not cover the {W}x{H} viewport"
    m1 = mark(d)
    for _ in range(3):
        await shift(ctx, page, (A, A2))
    assert nonzero(since(d, m1)) == Counter({A: 3, A2: 3})
    got = since(d, m0)
    assert len(got) == 1 and covers_viewport(got[0]), "the old mark recovered from its overflow"

    d, ctx, page = rig()
    assert await arm(d) is True
    for _ in range(10):
        await shift(ctx, page, (E, E2))
    m = mark(d)
    for _ in range(256):
        await shift(ctx, page, (tiny, tiny2))
    assert nonzero(since(d, m)) == Counter({tiny: 256, tiny2: 256}), (
        "256 entries since the mark dropped nothing since it; they must all be returned"
    )


async def test_unarmed_none() -> None:
    """A2: never armed, the read is ``None`` before and after anything happens; no CDP session is created by
    ``shift_mark``/``shifts_since``."""
    d, ctx, page = rig()
    m = mark(d)
    assert since(d, m) is None
    assert await fire(ctx, page, payload(source(A, A2))) == 0
    assert since(d, mark(d)) is None
    assert ctx.sessions == [] and page.page_calls() == []


async def test_malformed_event_never_raises() -> None:
    """A2 "never raises": payloads missing keys, with a non-list ``sources``, a rect missing a field or holding text,
    or another event type, reach the handler without an exception escaping it, and the read still answers. A good
    entry after them is still recorded."""
    d, ctx, page = rig()
    assert await arm(d) is True
    m = mark(d)
    bad: list[dict[str, Any]] = [
        {},
        {"event": {}},
        {"event": {"type": "layout-shift"}},
        {"event": {"type": "layout-shift", "layoutShiftDetails": {"sources": None}}},
        {
            "event": {
                "type": "layout-shift",
                "layoutShiftDetails": {"sources": [{"currentRect": rect(*A)}]},
            }
        },
        {
            "event": {
                "type": "layout-shift",
                "layoutShiftDetails": {"sources": [{"previousRect": {"x": 1}}]},
            }
        },
        {
            "event": {
                "type": "layout-shift",
                "layoutShiftDetails": {"sources": [{"previousRect": rect("a", 0, 1, 1)}]},  # type: ignore[arg-type]
            }
        },
        {"event": {"type": "largest-contentful-paint", "lcpDetails": {}}},
    ]
    for p in bad:
        await fire(ctx, page, p)
        got = since(d, m)
        assert got is None or isinstance(got, tuple), got
    await shift(ctx, page, (B, B2))
    got = since(d, m)
    assert got is not None and nonzero(got)[B] >= 1 and nonzero(got)[B2] >= 1, got


async def test_relaunch_drops_everything() -> None:
    """A3: after a relaunch (``_teardown``, a new context, a new page, generation 2) a mark of generation 1 reads
    ``None``, the new page is unarmed (``None``) until armed, an event late on the old session raises nothing, and
    arming the new page sends its own enable."""
    d, ctx, page = rig()
    assert await arm(d) is True
    m = mark(d)
    await d._teardown()
    ctx2 = FakeContext()
    page2 = FakePage(ctx2)
    d._context = ctx2
    d._adopt(page2)
    d._page = page2
    d.generation = 2
    assert since(d, m) is None
    assert since(d, mark(d)) is None, "the new page reads as armed after a relaunch"
    await fire(ctx, page, payload(source(A, A2)))
    assert since(d, m) is None
    assert await arm(d) is True
    assert len(ctx2.enables(page2)) == 1
    m2 = mark(d)
    await shift(ctx2, page2, (B, B2))
    assert nonzero(since(d, m2)) == Counter({B: 1, B2: 1})


# --------------------------------------------------------------------------------------------- A-R1, A-R2

#: five distinct non-zero sources (Chromium attributes at most 5 per entry: reviewer's probe, ruling A-R1)
FIVE: list[tuple[Rect, Rect]] = [
    ((0.0, 700.0 - 60.0 * i, 900.0, 60.0), (0.0, 800.0 - 60.0 * i, 900.0, 60.0)) for i in range(5)
]


@pytest.mark.parametrize(
    "pairs",
    [FIVE, [*FIVE[:4], ((0.0, 0.0, 0.0, 0.0), (40.0, 0.0, 600.0, 150.0))]],
    ids=["five-distinct", "five-with-a-zero-rect"],
)
async def test_five_sources_read_as_whole_viewport(pairs: list[tuple[Rect, Rect]]) -> None:
    """A-R1: an entry with 5 sources (Chromium's cap: more nodes may have moved than it names) reads as exactly one
    rect covering the viewport, never as the five sources alone. Read with an entry of 4 sources before it in another
    mark's window, the 4-source entry still reads as its own 8 rects (no covering rect)."""
    d, ctx, page = rig()
    assert await arm(d) is True
    m4 = mark(d)
    await shift(ctx, page, *FIVE[:4])
    got4 = since(d, m4)
    want4 = Counter([r for p in FIVE[:4] for r in p])
    assert nonzero(got4) == want4, f"4 sources: {got4}"
    assert not any(covers_viewport(r) for r in got4), (
        "4 sources (under the cap) read as the whole viewport"
    )
    m5 = mark(d)
    await shift(ctx, page, *pairs)
    got5 = since(d, m5)
    assert isinstance(got5, tuple) and len(got5) == 1 and covers_viewport(got5[0]), (
        f"A-R1: an entry at the 5-source cap read as {got5}, not one whole-viewport rect"
    )
    both = since(d, m4)
    assert any(covers_viewport(r) for r in both), both
    assert nonzero(tuple(r for r in both if not covers_viewport(r))) == want4, both


def _ls(details: Any) -> dict[str, Any]:
    """A ``layout-shift`` event whose ``layoutShiftDetails`` is ``details`` (omitted when ``...``)."""
    event: dict[str, Any] = {"frameId": "F0", "type": "layout-shift", "name": "", "time": 1.0}
    if details is not ...:
        event["layoutShiftDetails"] = details
    return {"event": event}


def _src(prev: Any, cur: Any) -> dict[str, Any]:
    return {"previousRect": prev, "currentRect": cur, "nodeId": 3}


GOOD = rect(*A)
INF = float("inf")
MALFORMED: list[tuple[str, dict[str, Any]]] = [
    ("details-missing", _ls(...)),
    ("details-none", _ls(None)),
    ("details-text", _ls("x")),
    ("details-list", _ls([1, 2])),
    ("sources-missing", _ls({"value": 0.1})),
    ("sources-none", _ls({"sources": None})),
    ("sources-dict", _ls({"sources": {"0": _src(GOOD, GOOD)}})),
    ("sources-text", _ls({"sources": "abc"})),
    ("source-none", _ls({"sources": [None]})),
    ("source-text", _ls({"sources": ["x"]})),
    ("source-no-current", _ls({"sources": [{"previousRect": GOOD, "nodeId": 1}]})),
    ("source-no-previous", _ls({"sources": [{"currentRect": GOOD, "nodeId": 1}]})),
    ("good-then-bad-source", _ls({"sources": [_src(GOOD, GOOD), "x"]})),
    ("rect-none", _ls({"sources": [_src(None, GOOD)]})),
    ("rect-list", _ls({"sources": [_src([40, 20, 600, 200], GOOD)]})),
    ("rect-no-width", _ls({"sources": [_src({"x": 1, "y": 2, "height": 3}, GOOD)]})),
    (
        "rect-text-value",
        _ls({"sources": [_src(GOOD, {"x": "40", "y": 0, "width": 1, "height": 1})]}),
    ),
    (
        "rect-bool-value",
        _ls({"sources": [_src(GOOD, {"x": True, "y": 0, "width": 1, "height": 1})]}),
    ),
    ("good-then-bad-rect", _ls({"sources": [_src(GOOD, GOOD), _src(GOOD, {"x": 1})]})),
    ("x-inf", _ls({"sources": [_src({"x": INF, "y": 0, "width": 10, "height": 10}, GOOD)]})),
    (
        "height-neg-inf",
        _ls({"sources": [_src(GOOD, {"x": 0, "y": 0, "width": 10, "height": -INF})]}),
    ),
    (
        "y-nan",
        _ls({"sources": [_src(GOOD, {"x": 0, "y": float("nan"), "width": 10, "height": 10})]}),
    ),
]


@pytest.mark.parametrize("params", [p for _, p in MALFORMED], ids=[n for n, _ in MALFORMED])
async def test_malformed_layout_shift_reads_whole_viewport(params: dict[str, Any]) -> None:
    """A-R2: a ``layout-shift`` event whose geometry cannot be read (unparseable details, a bad source, a bad rect, a
    non-finite coordinate, non-list sources) may have moved anything: the read since a mark before it holds a rect
    covering the viewport, never ``()`` and never only the readable part."""
    d, ctx, page = rig()
    assert await arm(d) is True
    m = mark(d)
    n = await fire(ctx, page, params)
    assert n >= 1, "premise: the recorder subscribed"
    got = since(d, m)
    assert isinstance(got, tuple) and any(covers_viewport(r) for r in got), (
        f"A-R2: a malformed layout-shift entry read as {got}"
    )


async def test_empty_sources_reads_as_no_rect() -> None:
    """A-R2: a well-formed entry with ``sources: []`` reads as no rect (``()``), not as the whole viewport; a good entry
    after it reads as its own rects only."""
    d, ctx, page = rig()
    assert await arm(d) is True
    m = mark(d)
    assert await fire(ctx, page, _ls({"value": 0.0, "hadRecentInput": False, "sources": []})) >= 1
    assert since(d, m) == (), f"sources: [] read as {since(d, m)}"
    await shift(ctx, page, (B, B2))
    assert since(d, m) is not None and nonzero(since(d, m)) == Counter({B: 1, B2: 1})
    assert not any(covers_viewport(r) for r in since(d, m))


# --------------------------------------------------------------------------------------------- A-R3

AT_ONCE_S = 0.005
DETACH_S = 0.2

ARM_FAILS: list[tuple[str, Callable[[FakeContext], None], bool]] = [
    (
        "enable-raises",
        lambda c: c.send_plan.update({ENABLE: PlaywrightError("Target closed")}),
        True,
    ),
    ("enable-hangs", lambda c: c.send_plan.update({ENABLE: "hang"}), True),
    ("attach-raises", lambda c: setattr(c, "attach_plan", PlaywrightError("no target")), False),
]


@pytest.mark.parametrize(
    "how,attached", [(f[1], f[2]) for f in ARM_FAILS], ids=[f[0] for f in ARM_FAILS]
)
async def test_failed_arm_remembered_per_page(
    how: Callable[[FakeContext], None], attached: bool
) -> None:
    """A-R3: after a failed arm the page is remembered as failed: with CDP healthy again, a second arm on that page
    returns ``False`` in under 5 ms with no new session and no CDP call; a mark reads ``None`` at once, even when the
    page then shifts; a session the failed arm attached has had ``detach`` called within 0.2 s of the arm returning.
    Another page still arms and records."""
    d, ctx, page = rig()
    how(ctx)
    assert await asyncio.wait_for(arm(d), 5.0) is False
    t_failed = time.perf_counter()
    if attached:
        assert ctx.sessions, "premise: the failed arm attached a session"
        while (
            not all(s.detached for s in ctx.sessions) and time.perf_counter() - t_failed < DETACH_S
        ):
            await asyncio.sleep(0.005)
        assert all(s.detached for s in ctx.sessions), (
            f"A-R3: the failed arm's session was not detached within {DETACH_S} s"
        )
    ctx.send_plan.clear()
    ctx.attach_plan = "ok"
    sessions, sent = len(ctx.sessions), len(ctx.methods())
    t0 = time.perf_counter()
    again = await arm(d)
    took = time.perf_counter() - t0
    assert again is False, "A-R3: a page whose arm failed was armed again (no retry on that page)"
    assert took < AT_ONCE_S, f"A-R3: the remembered failure took {took * 1000:.1f} ms"
    assert len(ctx.sessions) == sessions and len(ctx.methods()) == sent, "A-R3: a retry reached CDP"
    m = mark(d)
    assert since(d, m) is None
    for s in ctx.sessions:  # even if a late enable started events on the old session
        for handler in list(s.handlers.get(EVENT, [])):
            handler(payload(source(A, A2)))
    assert since(d, m) is None

    b = other_page(d, ctx)
    d._page = b
    assert await arm(d) is True, "A-R3: a failure on one page blocked another page"
    mb = mark(d)
    await shift(ctx, b, (C, C2))
    assert nonzero(since(d, mb)) == Counter({C: 1, C2: 1})
    d._page = page
    assert await arm(d) is False, "A-R3: the failed page was retried after another page armed"


@pytest.mark.parametrize("detach", ["hang", "raise"])
async def test_failed_arm_detach_is_bounded(detach: str) -> None:
    """A-R3: the failed session's ``detach`` is best-effort and bounded: when it hangs or raises, the arm still
    returns ``False`` without an exception, within ``SHIFT_ARM_TIMEOUT_S`` + 0.2 s (+ slack) for a hung enable and
    0.2 s (+ slack) for a refused one."""
    for plan, bound in (
        (PlaywrightError("Target closed"), DETACH_S + SLACK_S),
        ("hang", ARM_TIMEOUT_S + DETACH_S + SLACK_S),
    ):
        d, ctx, _ = rig()
        ctx.send_plan[ENABLE] = plan
        ctx.detach_plan = "hang" if detach == "hang" else PlaywrightError("detach failed")
        t0 = time.perf_counter()
        assert await asyncio.wait_for(arm(d), 5.0) is False
        took = time.perf_counter() - t0
        assert took <= bound, f"enable {plan!r}, detach {detach}: the arm took {took:.3f} s"
        await asyncio.sleep(DETACH_S)
        assert all(s.detached for s in ctx.sessions) and ctx.sessions, (
            "A-R3: detach was never called"
        )


# --------------------------------------------------------------------------------------------- A3


async def test_arm_idempotent_one_round_trip() -> None:
    """A3: three arms on one page: ``True`` each, exactly one ``PerformanceTimeline.enable`` with
    ``{"eventTypes": ["layout-shift"]}`` and at most one new CDP session; a re-arm does not reset what was recorded.
    A second page gets its own single enable; going back to the first sends nothing."""
    assert getattr(runtime_mod, "SHIFT_ARM_TIMEOUT_S", None) == ARM_TIMEOUT_S, (
        "A3: SHIFT_ARM_TIMEOUT_S = 0.5"
    )
    d, ctx, page = rig()
    for _ in range(3):
        assert await arm(d) is True
    assert ctx.enables(page) == [{"eventTypes": ["layout-shift"]}], ctx.methods()
    assert len(ctx.sessions) <= 1
    assert [m for m in ctx.methods() if m != ENABLE] == []
    m = mark(d)
    await shift(ctx, page, (A, A2))
    assert await arm(d) is True
    assert nonzero(since(d, m)) == Counter({A: 1, A2: 1}), "a re-arm reset the record"
    assert len(ctx.enables()) == 1

    b = other_page(d, ctx)
    d._page = b
    assert await arm(d) is True
    assert len(ctx.enables(b)) == 1 and len(ctx.enables()) == 2
    d._page = page
    assert await arm(d) is True
    assert len(ctx.enables()) == 2, "going back to an armed page sent another enable"
    assert page.page_calls() == [] and b.page_calls() == []


FAILURES: list[tuple[str, Callable[[Any, FakeContext], None]]] = [
    (
        "enable raises PlaywrightError",
        lambda d, c: c.send_plan.update({ENABLE: PlaywrightError("Target closed")}),
    ),
    ("enable raises RuntimeError", lambda d, c: c.send_plan.update({ENABLE: RuntimeError("boom")})),
    ("enable raises KeyError", lambda d, c: c.send_plan.update({ENABLE: KeyError("x")})),
    (
        "attach raises PlaywrightError",
        lambda d, c: setattr(c, "attach_plan", PlaywrightError("no target")),
    ),
    ("enable hangs", lambda d, c: c.send_plan.update({ENABLE: "hang"})),
    ("attach hangs", lambda d, c: setattr(c, "attach_plan", "hang")),
    ("no active page", lambda d, c: setattr(d, "_page", None)),
]


@pytest.mark.parametrize("how", [f[1] for f in FAILURES], ids=[f[0] for f in FAILURES])
async def test_arm_failure_false_never_raises(how: Callable[[Any, FakeContext], None]) -> None:
    """A3: every failure gives ``False`` without an exception, within ``SHIFT_ARM_TIMEOUT_S`` + slack when a CDP call
    hangs; then (a page being active) a mark reads ``None`` even if the page shifts: a failed arm is never quiet."""
    d, ctx, page = rig()
    how(d, ctx)
    t0 = time.perf_counter()
    ok = await asyncio.wait_for(arm(d), 5.0)
    took = time.perf_counter() - t0
    assert ok is False
    assert took <= ARM_TIMEOUT_S + SLACK_S, f"a failing arm took {took:.3f} s"
    if d._page is None:
        return
    m = mark(d)
    await fire(ctx, page, payload(source(A, A2)))
    assert since(d, m) is None, "a failed arm reads as armed"


@pytest.mark.parametrize("where", ["enable", "attach"])
async def test_arm_cancel_propagates(where: str) -> None:
    """A3: a cancel while the arm waits on CDP (well inside its 0.5 s budget) propagates as ``CancelledError``."""
    d, ctx, _ = rig()
    if where == "enable":
        ctx.send_plan[ENABLE] = "hang"
    else:
        ctx.attach_plan = "hang"
    fn = _fn(d, "arm_shift_recorder")
    task = asyncio.ensure_future(fn())
    await asyncio.sleep(0.05)
    assert not task.done(), "premise: the arm is waiting on the hung CDP call"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2.0)


# --------------------------------------------------------------------------------------------- A1


CLAMPS: list[tuple[tuple[float, float, float, float], dict[str, float]]] = [
    ((100, 200, 128, 128), {"x": 100, "y": 200, "width": 128, "height": 128}),
    ((100.5, 10.25, 64, 64), {"x": 100.5, "y": 10.25, "width": 64, "height": 64}),
    ((1200, 750, 200, 200), {"x": 1200, "y": 750, "width": 80, "height": 50}),
    ((-50, -20, 128, 128), {"x": 0, "y": 0, "width": 78, "height": 108}),
    ((-10, -10, 2000, 2000), {"x": 0, "y": 0, "width": 1280, "height": 800}),
    ((1276, 796, 128, 128), {"x": 1276, "y": 796, "width": 4, "height": 4}),
]
OFFSCREEN = [
    (1300, 10, 50, 50),
    (1280, 0, 10, 10),
    (-200, 0, 100, 100),
    (10, 800, 50, 50),
    (10, -60, 50, 50),
    (10, 10, 0, 50),
    (10, 10, 50, 0),
]


async def test_region_clamped_and_offscreen_valueerror() -> None:
    """A1: each region is exactly one ``page.screenshot`` with ``type="jpeg"``, ``quality=JPEG_QUALITY``,
    ``scale="css"``, not full page, and ``clip`` = the rectangle clamped to the 1280x800 viewport (by hand); its bytes
    are returned. A rectangle with nothing inside the viewport raises ``ValueError`` with no page call; no active
    page raises ``RuntimeError``. No other page call is ever made (no input, scroll or script)."""
    d, ctx, page = rig()
    fn = _fn(d, "screenshot_region")
    for (x, y, w, h), clip in CLAMPS:
        page.calls.clear()
        out = await fn(x, y, w, h)
        assert out == JPEG
        assert [c[0] for c in page.calls] == ["screenshot"], page.calls
        kw = page.calls[0][1]
        assert kw.get("type") == "jpeg" and kw.get("quality") == runtime_mod.JPEG_QUALITY, kw
        assert kw.get("scale") == "css", kw
        assert kw.get("full_page", False) is False, kw
        got = kw.get("clip")
        assert isinstance(got, dict) and set(got) == {"x", "y", "width", "height"}, got
        for k, v in clip.items():
            assert got[k] == pytest.approx(v), f"region {(x, y, w, h)}: clip {got}, want {clip}"
    for x, y, w, h in OFFSCREEN:
        page.calls.clear()
        with pytest.raises(ValueError):
            await fn(x, y, w, h)
        assert page.calls == [], f"an off-screen region {(x, y, w, h)} called the page"
    assert page.unexpected == [], page.unexpected
    assert ctx.sessions == [], "a region screenshot opened a CDP session"
    d._page = None
    with pytest.raises(RuntimeError):
        await fn(100, 100, 64, 64)


# --------------------------------------------------------------------------------------------- A4


async def test_node_id_never_resolved() -> None:
    """A2: ``nodeId`` is ignored: across an arm, entries carrying node ids and reads, the only CDP method sent is the
    enable (no ``DOM.*``, nothing on the page), and two sources with the same geometry but different node ids read the
    same as the geometry alone."""
    d, ctx, page = rig()
    assert await arm(d) is True
    m = mark(d)
    await fire(ctx, page, payload(source(A, A2, 4), source(B, B2, 99999)))
    since(d, m)
    await fire(ctx, page, payload(source(C, C2, 5)))
    got = since(d, m)
    assert nonzero(got) == Counter({A: 1, A2: 1, B: 1, B2: 1, C: 1, C2: 1})
    assert ctx.methods() == [ENABLE], ctx.methods()
    assert not any(m_.startswith("DOM.") for m_ in ctx.methods())
    assert page.page_calls() == []
    assert all(s.unexpected == [] for s in ctx.sessions) and ctx.unexpected == []


async def test_only_new_cdp_method_is_performance_timeline_enable() -> None:
    """A4 (negative): a sequence using every new method beside ``layout_metrics()`` sends only
    ``Page.getLayoutMetrics`` (BASE) and ``PerformanceTimeline.enable`` (new, ``eventTypes: ["layout-shift"]``), and
    ``layout_metrics()`` answers BASE's mapping (by hand from the fake's metrics) before and after."""
    d, ctx, page = rig()
    assert await d.layout_metrics() == METRICS_ANSWER
    assert await arm(d) is True
    m = mark(d)
    await shift(ctx, page, (A, A2))
    assert nonzero(since(d, m)) == Counter({A: 1, A2: 1})
    await _fn(d, "screenshot_region")(600, 300, 128, 128)
    assert await d.layout_metrics() == METRICS_ANSWER
    assert await arm(d) is True
    assert set(ctx.methods()) == {"Page.getLayoutMetrics", ENABLE}, ctx.methods()
    assert ctx.enables() == [{"eventTypes": ["layout-shift"]}]
    assert [c[0] for c in page.calls] == ["screenshot"] and page.unexpected == []
    assert ctx.unexpected == [] and all(s.unexpected == [] for s in ctx.sessions)
    # owned-files A: ``__all__`` gains the two new constants and keeps BASE's names
    assert set(runtime_mod.__all__) >= {
        "MAX_PAGES",
        "BrowserEndedError",
        "FollowResult",
        "PlaywrightDriver",
        "failure_type_class",
        "url_allowed",
        "MAX_SHIFT_ENTRIES",
        "SHIFT_ARM_TIMEOUT_S",
    }
