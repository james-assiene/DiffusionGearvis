"""T-0151 group A1 (A1-1..A1-6, A1-8): the overview frame in ``BrowserToolExecutor``, over a fake driver with
scripted layout metrics, a whole-page picture and a scroll model (docs/tasks/T-0151.md).

**The surface these rows pin** (A1's names):

- ``BrowserToolExecutor.overview_frame_of(key) -> Frame | None``: reads the picture named by an executed
  ``page_overview``'s ``resolved["overview"]`` from the same once-only store as ``overview_of`` and registers it as an
  *overview frame* only when the capture was not clipped, its CSS width is the viewport's, the driver's generation and
  ``page_serial`` are the capture's, and ``latest_frame`` is still the own frame that was latest at the capture;
- a ``click_at``/``type_at`` on a registered overview frame maps ``P = (to_pixels(x, page_w), to_pixels(y, page_h))``
  (page size = the captured CSS size), refuses ``OVERVIEW_STALE`` / ``OVERVIEW_PAGE_CHANGED`` before dispatch, scrolls
  once by ``dy = clamp(P_y - H/2, 0, C - H) - S`` when ``P_y`` is outside ``[S, S + H)``, settles, checks two equal
  metrics reads with ``S' <= P_y < S' + H`` and an unchanged ``content_height`` (else ``failed``,
  ``OVERVIEW_NOT_REACHED``, nothing clicked, ``frame_after`` captured), then clicks at ``(P_x, P_y - S')``;
- the action string is BASE's, prefixed ``scrolled down by N px and `` / ``scrolled up by N px and `` after a scroll;
  echo ``"model"`` reports the call's own overview units, ``"pixels"`` the viewport point dispatched (R-d).

**The fake** is T-0140's ``OverviewDriver`` pattern, copied (never imported, as the task asks), over T-0111c's
``FakeDriver`` and T-0111g's ``ClosableDriver``: any driver name not deliberately faked raises a named
``AssertionError`` (A1-6 pixel-first: no ``page``, ``evaluate``, title, URL or DOM is reachable). ``scroll_by`` moves
``scroll_y`` within ``[0, content - viewport]``; ``scroll_land`` makes a scroll land elsewhere (short), ``drift`` keeps
the page moving one px per metrics read after a scroll, ``grow_on_scroll`` changes ``content_height`` with the scroll,
``metrics_fail`` makes every read raise and ``metrics_viewport`` overrides the reported viewport height. Every input
call changes the fake's screenshot (``scene``), so a capture after an action is a different own frame.

**Expected values by hand** (viewport 1280x800, DSF 1; ``to_pixels(v, n) = (2 v (n - 1) + 999) // 1998``):

- deals-like page, ``content_height`` 1220: the picture 1280x1220 is scaled by ``min(1, 800/1220)`` to 839.3x800 and
  even-floored to 838x798 (T-0140's ``_overview_frame``); ``to_pixels(160, 1280) = 205``; ``to_pixels(935, 1220) =
  2280529 // 1998 = 1141``; ``to_pixels(933, 1220) = 1138``; ``to_pixels(308, 1220) = 751903 // 1998 = 376``. From
  ``S = 0``: 1141 is not in ``[0, 800)``, ``dy = clamp(741, 0, 420) - 0 = 420``, click at ``1141 - 420 = 721``. The
  viewport mutant: ``to_pixels(935, 800) = 748``, in view, no scroll. The overview-width mutant: ``to_pixels(160, 838)
  = 134``.
- 2.0-screen page, 1600: picture 640x800; ``to_pixels(552, 1600) = 884`` -> ``dy = 484``, click y 400;
  ``to_pixels(990, 1600) = 1585`` -> ``dy = 800`` from 0 (click 785), ``200`` from ``S = 600``;
  ``to_pixels(50, 1600) = 80`` from ``S = 600`` -> ``dy = 0 - 600 = -600``, click 80; boundaries
  ``to_pixels(499, 1600) = 799`` (in view at 0) and ``to_pixels(500, 1600) = 800`` (not: ``dy = 400``, click 400);
  ``to_pixels(375, 1600) = 600`` (in view at ``S = 600``, click y 0) and ``to_pixels(374, 1600) = 599`` (``dy = 199 -
  600 = -401``, click 400).
- The task's "Tests to write" says the teapot click lands at y 375; by hand it is 376 (``751903 // 1998``), which is
  what these rows assert (flagged in the report).

BASE_A1 = ab33831 (main when the A1 branch was cut). Activated for group A1 (rulings O-A1 in the task file; the
pending markers are off); ``test_screen_frames_unchanged`` and the frozen-path row are green at BASE_A1. New
symbols (``overview_frame_of``, ``OVERVIEW_STALE``, ``OVERVIEW_PAGE_CHANGED``, ``OVERVIEW_NOT_REACHED``) are looked up
inside the rows, so collection never errors.

Regime: CPU over a fake driver, no browser, 1280x800 at DSF 1, n = 1, deterministic.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

import pytest
from PIL import Image, ImageDraw

from local_astra.computer import executor as executor_mod
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.toolcall import FrameGeometry
from tests import _git_range as gr
from tests.unit.test_computer_executor import FakeDriver, inv, own_frame
from tests.unit.test_computer_live_options import ClosableDriver

#: The commit group A1 branched from (main at "T-0151: H-T151-1..5 approved (James)"). A literal, never a moving ref.
BASE_A1 = "ab33831"
#: The implementation tip for the frozen-path row; starts at BASE_A1 (no implementation yet) and is bumped by the
#: test-author or the orchestrator to the last implementation commit before review (the implementer never edits tests).
TIP_A1 = "cad8538"

CLOCK_MS = 1_700_000_001_234

STALE = "the screen changed since the whole-page picture; take a new frame"
PAGE_CHANGED = "the page changed since the whole-page picture; take a new frame"
NOT_REACHED = "the page did not scroll to the point; nothing was clicked"
NOT_OWN = "frame not from this browser"
ADOPTED = "the active page changed; take a new frame"
CLOSED = "the browser window was closed"
RESTART_FRAME = "frame from before a browser restart"
#: A1-6: the only driver names an overview click may touch (the fake raises on any other public name anyway)
SEAM = frozenset(
    {
        "ensure",
        "layout_metrics",
        "scroll_by",
        "settle",
        "mouse_move",
        "mouse_down",
        "mouse_up",
        "type_text",
        "key_down",
        "key_up",
        "screenshot",
        "take_events",
    }
)


# =====================================================================================================
# the fake (T-0140's OverviewDriver pattern, copied and extended with a scroll model)
# =====================================================================================================


def page_jpeg(width: int, height: int, variant: int = 0) -> bytes:
    """A whole-page picture unlike any viewport screenshot of the fake (those are grey gradients): a red top quarter,
    a blue bottom quarter, a green middle with a black block whose place is ``variant`` (12 places), so successive
    captures can have different pixels and so different phashes."""
    img = Image.new("RGB", (width, height), (40, 200, 40))
    img.paste((220, 30, 30), (0, 0, width, max(1, height // 4)))
    img.paste((30, 30, 220), (0, height - max(1, height // 4), width, height))
    col, row = variant % 4, (variant // 4) % 3
    bw, bh = width // 4, height // 6
    top = height // 4 + row * bh
    ImageDraw.Draw(img).rectangle([col * bw, top, (col + 1) * bw - 1, top + bh - 1], fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class MetricsBoom(RuntimeError):
    """The fake's failed layout-metrics read."""


class PageDriver(ClosableDriver):
    """T-0111c's fake (with T-0111g's two end facts) plus T-0134's ``layout_metrics`` and T-0140's
    ``full_page_screenshot`` over a page ``content_height`` CSS px tall, scrolled by ``scroll_by`` (module docstring
    for the scripted faults)."""

    def __init__(
        self, viewport: tuple[int, int] = (1280, 800), content_height: float = 1220.0
    ) -> None:
        super().__init__()
        self.viewport = viewport
        self.content_height = float(content_height)
        self.scroll_y = 0.0
        self.image_width: int | None = None
        self.variant = 0
        self.full_page_calls: list[int] = []
        #: faults
        self.scroll_land: float | None = None
        self.drift = False
        self._drifting = False
        self.grow_on_scroll = 0.0
        self.metrics_fail = False
        self.metrics_viewport: float | None = None
        #: O-A1-3 (F1, the real-page fault): offsets the page snaps to, one per ``settle()`` after a scroll; between
        #: two back-to-back reads it stands still (a scroll-snap page still animating)
        self.settle_moves: list[float] = []
        #: O-A1-3: the viewport height the page reports from the first ``settle()`` after a scroll on
        self.viewport_on_settle: float | None = None
        self._scrolled = False

    async def settle(self) -> None:
        await super().settle()
        if not self._scrolled:
            return
        if self.settle_moves:
            self.scroll_y = self.settle_moves.pop(0)
        if self.viewport_on_settle is not None:
            self.metrics_viewport, self.viewport_on_settle = self.viewport_on_settle, None

    async def layout_metrics(self) -> dict[str, float]:
        self._enter("layout_metrics")
        if self.metrics_fail:
            raise MetricsBoom("synthetic metrics failure")
        value = {
            "scroll_y": self.scroll_y,
            "viewport_height": (
                float(self.viewport[1]) if self.metrics_viewport is None else self.metrics_viewport
            ),
            "content_height": self.content_height,
        }
        if self._drifting:  # a page still moving after the wheel: every read sees another offset
            self.scroll_y -= 1.0
        return value

    async def full_page_screenshot(self, max_height_css: int) -> bytes:
        self._enter("full_page_screenshot", max_height_css)
        self.full_page_calls.append(max_height_css)
        await self._maybe_hang("full_page_screenshot")
        w = self.image_width if self.image_width is not None else self.viewport[0]
        h = int(min(self.content_height, float(max_height_css)))
        return page_jpeg(w, h, self.variant)

    async def scroll_by(self, dx: float, dy: float) -> None:
        await super().scroll_by(dx, dy)
        top = max(0.0, self.content_height - self.viewport[1])
        landed = max(0.0, min(top, self.scroll_y + dy))
        self.scroll_y = landed if self.scroll_land is None else self.scroll_land
        self.content_height += self.grow_on_scroll
        if self.drift:
            self._drifting = True
        self._scrolled = True


def build(driver: FakeDriver, **kw: Any) -> Any:
    return executor_mod.BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS, viewport=driver.viewport, driver=driver, **kw
    )


def overview_frame_of(ex: Any, key: Any) -> Any:
    fn = getattr(ex, "overview_frame_of", None)
    assert callable(fn), (
        "T-0151 A1-1: BrowserToolExecutor.overview_frame_of(key) does not exist yet"
    )
    return fn(key)


def const(name: str) -> Any:
    value = getattr(executor_mod, name, None)
    assert value is not None, f"T-0151 A1-3/A1-4: executor.{name} does not exist yet"
    return value


async def capture(ex: Any) -> str:
    """One executed ``page_overview`` with an image; its key."""
    out = await ex.execute(inv("page_overview"))
    assert out.status == "executed", out
    key = out.resolved.get("overview")
    assert isinstance(key, str), f"premise: the overview names its image: {out}"
    return key


async def register(ex: Any) -> Frame:
    key = await capture(ex)
    f = overview_frame_of(ex, key)
    assert isinstance(f, Frame), f"T-0151 A1-1: overview_frame_of({key!r}) gave {f!r}"
    return f


async def long_page(
    content: float = 1220.0, *, echo: str = "model", scroll_first: bool = False, **kw: Any
) -> tuple[Any, PageDriver, Frame]:
    """An executor over a ``content``-px page: one own frame (optionally one ``scroll down`` of 600 px first), then
    the overview captured and registered."""
    d = PageDriver(content_height=content)
    ex = build(d, echo=echo, **kw)
    await own_frame(ex)
    if scroll_first:
        out = await ex.execute(inv("scroll", None, direction="down"))
        assert out.status == "executed" and d.scroll_y == 600.0, (out, d.scroll_y)
    ov = await register(ex)
    return ex, d, ov


def names(d: FakeDriver, since: int = 0) -> list[str]:
    return [c[0] for c in d.calls[since:]]


def inputs(d: FakeDriver, since: int = 0) -> list[tuple[str, tuple[Any, ...]]]:
    return d.dispatched(since)


def mouse_moves(d: FakeDriver, since: int = 0) -> list[tuple[int, int]]:
    return [(round(c[1][0]), round(c[1][1])) for c in d.calls[since:] if c[0] == "mouse_move"]


def scrolls(d: FakeDriver, since: int = 0) -> list[tuple[int, int]]:
    return [(round(c[1][0]), round(c[1][1])) for c in d.calls[since:] if c[0] == "scroll_by"]


# =====================================================================================================
# A1-1 registration
# =====================================================================================================


async def test_register_once() -> None:
    """A1-1: ``overview_frame_of(key)`` gives the frame once (838x798 by hand, ``WINDOW``, a phash unlike the
    screen's); a second read, ``overview_of`` after it, an unknown key and a non-string key give ``None``;
    ``latest_frame`` and ``extent_of`` are unchanged and the overview has no extent ("kept apart", R-c). In the other
    order, ``overview_of`` first takes the picture and ``overview_frame_of`` then gives ``None`` (one store)."""
    d = PageDriver(content_height=1220.0)
    ex = build(d)
    f0 = await own_frame(ex)
    extent0 = ex.extent_of(f0.phash)
    assert extent0 == {"above_px": 0.0, "below_px": 420.0, "viewport_px": 800.0}, extent0
    key = await capture(ex)
    ov = overview_frame_of(ex, key)
    assert isinstance(ov, Frame), ov
    assert (ov.width, ov.height) == (838, 798), (ov.width, ov.height)
    assert ov.source == Source.WINDOW and ov.phash is not None and ov.phash != f0.phash
    assert overview_frame_of(ex, key) is None, "the overview frame was given twice"
    assert ex.overview_of(key) is None, "overview_of read a picture overview_frame_of already took"
    assert overview_frame_of(ex, "no-such-overview") is None
    assert overview_frame_of(ex, 7) is None
    assert ex.latest_frame == f0, "registering the overview changed latest_frame"
    assert ex.extent_of(f0.phash) == extent0
    assert ex.extent_of(ov.phash) is None, (
        "the overview frame has an extent: it went through the own registry"
    )
    key2 = await capture(ex)
    assert isinstance(ex.overview_of(key2), Frame)
    assert overview_frame_of(ex, key2) is None, (
        "overview_of and overview_frame_of do not share one store"
    )


REGISTRATION_ROWS = [
    # case, content height, expected registered?
    pytest.param("control", 1220.0, True, id="control"),
    pytest.param("control", 3200.0, True, id="four-screens-not-clipped"),
    pytest.param("control", 4000.0, False, id="clipped-5-screens"),
    pytest.param("wide", 1220.0, False, id="wider-than-viewport"),
    pytest.param("stale", 1220.0, False, id="another-frame-after"),
    pytest.param("generation", 1220.0, False, id="generation-changed"),
    pytest.param("page_serial", 1220.0, False, id="page-serial-changed"),
]


@pytest.mark.parametrize(("case", "content", "registered"), REGISTRATION_ROWS)
async def test_not_registered_when_clipped_wide_or_stale(
    case: str, content: float, registered: bool
) -> None:
    """A1-1's four conditions, each alone, and the too-wide side (a 1.5- and an exactly 4.0-screen page register):
    a 5-screen (clipped) capture, a picture 2560 CSS px wide, another ``frame()`` after the capture (new pixels), a
    browser restart and an adopted popup each give ``None``, and the picture is dropped (``overview_of`` is ``None``
    too)."""
    d = PageDriver(content_height=content)
    if case == "wide":
        d.image_width = 2560
    ex = build(d)
    f0 = await own_frame(ex)
    key = await capture(ex)
    if case == "stale":
        d.scene += 1
        f1 = await own_frame(ex)
        assert f1.phash != f0.phash, "premise: the second frame has other pixels"
    elif case == "generation":
        d.kill()
        out = await ex.execute(inv("press_key", combo="Tab"))
        assert (out.status, out.reason) == ("failed", "browser restarted"), out
        assert ex.latest_frame == f0, "premise: only the generation changed"
    elif case == "page_serial":
        d.open_popup()
        out = await ex.execute(inv("press_key", combo="Tab"))
        assert (out.status, out.reason) == ("refused", ADOPTED) and d.page_serial == 1, out
        assert ex.latest_frame == f0, "premise: only the page serial changed"
    got = overview_frame_of(ex, key)
    if registered:
        assert isinstance(got, Frame), f"{case}: not registered (the guard is too wide): {got!r}"
    else:
        assert got is None, f"{case}: registered: {got!r}"
        assert ex.overview_of(key) is None, f"{case}: the refused picture was not dropped"


async def test_not_registered_without_own_frame() -> None:
    """A1-1 (ruling O-A1, coordinator 2026-10-24): with no own frame at capture (``latest_frame`` is ``None``: the
    overview is the first thing the executor captured), ``overview_frame_of`` returns ``None``, registers nothing and
    drops the picture; a click on its geometry is refused with nothing dispatched. The too-wide side: the same page
    with one ``frame()`` first registers (``test_register_once``)."""
    d = PageDriver(content_height=1220.0)
    ex = build(d)
    out = await ex.execute(inv("page_overview"))
    assert out.status == "executed", out
    assert ex.latest_frame is None, "premise: no own frame was captured"
    key = out.resolved.get("overview")
    assert isinstance(key, str), out
    assert overview_frame_of(ex, key) is None, "registered an overview with no own frame at capture"
    assert ex.overview_of(key) is None, "the refused picture was not dropped"
    assert ex.latest_frame is None
    d.variant = 1
    key = await capture(ex)
    ov = ex.overview_of(key)
    assert isinstance(ov, Frame)
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert out.status == "refused", out
    assert inputs(d, mark) == [], inputs(d, mark)


async def test_overview_geometry_must_match() -> None:
    """A1-2 "frame geometry (phash, ``WINDOW``, width and height)" (reviewer B at 9bb4ab5; mutants: the size check
    deleted, a height-only check, the source check deleted): a registered overview's phash (838x798) sent with (a) the
    screen's size 1280x800, (b) a width one off (839x798, the height right) or (c) a non-``WINDOW`` source
    (``SCREEN``, the size right) is not that overview frame and not an own frame: each is ``refused`` with ``frame
    not from this browser``, nothing dispatched, no scroll. The too-wide side: the exact geometry then executes (the
    refusals left the overview fresh)."""
    ex, d, ov = await long_page(1220.0)
    assert (ov.width, ov.height) == (838, 798), "premise"
    wrong = {
        "screen-size": FrameGeometry(phash=ov.phash, source=Source.WINDOW, width=1280, height=800),
        "width-off-by-one": FrameGeometry(
            phash=ov.phash, source=Source.WINDOW, width=839, height=798
        ),
        "not-window": FrameGeometry(phash=ov.phash, source=Source.SCREEN, width=838, height=798),
    }
    for case, g in wrong.items():
        mark = len(d.calls)
        out = await ex.execute(inv("click_at", g, x=160, y=935))
        assert (out.status, out.reason) == ("refused", NOT_OWN), (case, out)
        assert inputs(d, mark) == [], (case, inputs(d, mark))
        assert scrolls(d, mark) == [], (case, scrolls(d, mark))
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert out.status == "executed", (
        f"the exact overview geometry is refused (the guard is too wide): {out}"
    )


async def test_registry_capped_and_apart() -> None:
    """A1-1 "kept apart" (reviewer mutation 8, the overview put in ``_own``): nine overviews with different pixels are
    registered in turn (the latest own frame unchanged throughout); the registry keeps the newest eight, like
    ``_MAX_OVERVIEWS``, so a click on the first is refused with nothing dispatched (an overview that leaked into the
    own-frame registry would execute as a screen frame instead), and a click on the ninth executes."""
    d = PageDriver(content_height=1220.0)
    ex = build(d)
    f0 = await own_frame(ex)
    frames: list[Frame] = []
    for k in range(9):
        d.variant = k
        frames.append(await register(ex))
    assert len({f.phash for f in frames}) == 9, "premise: nine different pictures"
    assert ex.latest_frame == f0
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", frames[0], x=160, y=308))
    assert out.status == "refused", f"the ninth registration did not forget the first: {out}"
    assert inputs(d, mark) == [], inputs(d, mark)
    out = await ex.execute(inv("click_at", frames[8], x=160, y=308))
    assert out.status == "executed", f"the newest overview frame is not clickable: {out}"


async def test_overview_frame_salted() -> None:
    """A1-1 (T-0140 A3's salt): the registered frame's phash is not the raw DCT hash of its pixels, and the same
    pixels after a browser restart give another phash."""
    d = PageDriver(content_height=1220.0)
    ex = build(d)
    await own_frame(ex)
    ov1 = await register(ex)
    with Image.open(io.BytesIO(ov1.jpeg)) as img:
        raw = executor_mod.phash(img)
    assert ov1.phash != raw, "the overview frame's phash is unsalted"
    d.kill()
    out = await ex.execute(inv("press_key", combo="Tab"))
    assert (out.status, out.reason) == ("failed", "browser restarted"), out
    await own_frame(ex)
    ov2 = await register(ex)
    with Image.open(io.BytesIO(ov2.jpeg)) as img:
        assert executor_mod.phash(img) == raw, "premise: the same pixels"
    assert ov2.phash != ov1.phash, "no per-generation salt"


# =====================================================================================================
# A1-2 / A1-4 mapping, scroll, act
# =====================================================================================================

MAPPING_ROWS = [
    # content, (x, y), page point, scroll deltas, viewport point
    pytest.param(1220.0, (160, 935), (205, 1141), [(0, 420)], (205, 721), id="deals-1220"),
    pytest.param(1600.0, (160, 552), (205, 884), [(0, 484)], (205, 400), id="two-screens-1600"),
]


@pytest.mark.parametrize(("content", "xy", "page", "deltas", "point"), MAPPING_ROWS)
async def test_mapping_page_coordinates(
    content: float,
    xy: tuple[int, int],
    page: tuple[int, int],
    deltas: list[tuple[int, int]],
    point: tuple[int, int],
) -> None:
    """A1-2/A1-4 by hand (module docstring): the call is planned against the page, scrolled once, settled, checked
    by two metrics reads, then clicked at the viewport point; ``resolved`` carries ``x_px``/``y_px`` (the viewport
    point), ``page_y_px``, ``scrolled_px`` and ``frame_after`` (the post-action capture, now ``latest_frame``). A1-6:
    the click touched only the driver seam and CDP metrics (no full-page capture, nothing else)."""
    ex, d, ov = await long_page(content)
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=xy[0], y=xy[1]))
    assert out.status == "executed", out
    assert scrolls(d, mark) == deltas, scrolls(d, mark)
    assert mouse_moves(d, mark) == [point], mouse_moves(d, mark)
    assert d.presses[-1] == point
    assert d.scroll_y == float(deltas[0][1])
    r = out.resolved
    assert (r.get("x_px"), r.get("y_px"), r.get("page_y_px")) == (point[0], point[1], page[1]), r
    assert r.get("scrolled_px") == deltas[0][1], r
    latest = ex.latest_frame
    assert latest is not None and r.get("frame_after") == latest.phash, (r, latest)
    seq = names(d, mark)
    assert set(seq) <= SEAM, f"A1-6: the overview click used {set(seq) - SEAM}"
    i_scroll, i_move = seq.index("scroll_by"), seq.index("mouse_move")
    assert any(n == "layout_metrics" for n in seq[:i_scroll]), (
        "A1-3: no metrics read before dispatch"
    )
    between = seq[i_scroll + 1 : i_move]
    assert "settle" in between, f"A1-4: no settle() between the scroll and the click: {between}"
    after_settle = between[between.index("settle") + 1 :]
    assert after_settle.count("layout_metrics") >= 2, (
        f"A1-4: the moved-page check is two metrics reads after the settle: {between}"
    )
    tail = seq[seq.index("mouse_up") + 1 :]
    assert "settle" in tail and "screenshot" in tail[tail.index("settle") :], (
        f"A1-4: settle() and the capture follow the click: {tail}"
    )


async def test_mapped_against_page_not_viewport() -> None:
    """A1-2 (adversarial; reviewer mutation 3: ``to_pixels(y, viewport_h)`` reads overview y 935 as viewport y 748,
    in view, and clicks there without a scroll; reviewer mutation 4: no scroll, clicking at page y 1141 off the
    screen). The dispatched pointer is at x 205 (``to_pixels(160, 1280)``, the page width, not the 838 px overview's
    134) and y 721, after exactly one ``scroll_by(0, 420)``."""
    ex, d, ov = await long_page(1220.0)
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert out.status == "executed", out
    moves = mouse_moves(d, mark)
    assert moves != [(205, 748)], "mapped against the viewport height (mutation 3)"
    assert moves != [(134, 721)] and moves[0][0] == 205, (
        f"x mapped against the overview width: {moves}"
    )
    assert scrolls(d, mark) == [(0, 420)], (
        f"no scroll before the click (mutation 4): {scrolls(d, mark)}"
    )
    assert moves == [(205, 721)], moves


IN_VIEW_ROWS = [
    # content, scroll first, (x, y), scroll deltas, viewport point, page y
    pytest.param(1220.0, False, (160, 308), [], (205, 376), 376, id="teapot-in-view"),
    pytest.param(1600.0, False, (160, 499), [], (205, 799), 799, id="last-row-in-view"),
    pytest.param(1600.0, False, (160, 500), [(0, 400)], (205, 400), 800, id="first-row-below"),
    pytest.param(1600.0, True, (160, 375), [], (205, 0), 600, id="first-row-in-view-scrolled"),
    pytest.param(1600.0, True, (160, 374), [(0, -401)], (205, 400), 599, id="last-row-above"),
]


@pytest.mark.parametrize(
    ("content", "scroll_first", "xy", "deltas", "point", "page_y"), IN_VIEW_ROWS
)
async def test_in_view_no_scroll(
    content: float,
    scroll_first: bool,
    xy: tuple[int, int],
    deltas: list[tuple[int, int]],
    point: tuple[int, int],
    page_y: int,
) -> None:
    """A1-4 "Target in view" (``S <= P_y < S + H``), both edges from ``S = 0`` and ``S = 600``: the teapot at
    overview (160, 308) is page y 376, in view: no ``scroll_by``, BASE's string (``clicked at (160, 308)``), a click
    at (205, 376), ``scrolled_px`` 0. Page y 799 is in view and 800 is not (``dy = 400``); scrolled to 600, page y 600
    is in view (viewport y 0) and 599 is not (``dy = -401``)."""
    ex, d, ov = await long_page(content, scroll_first=scroll_first)
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=xy[0], y=xy[1]))
    assert out.status == "executed", out
    assert scrolls(d, mark) == deltas, scrolls(d, mark)
    assert mouse_moves(d, mark) == [point], mouse_moves(d, mark)
    assert out.resolved.get("page_y_px") == page_y, out.resolved
    if not deltas:
        assert out.action == f"clicked at ({xy[0]}, {xy[1]})", out.action
        assert out.resolved.get("scrolled_px") == 0, out.resolved
    else:
        assert out.resolved.get("scrolled_px") == deltas[0][1], out.resolved


CLAMP_ROWS = [
    # scroll first, (x, y), scroll deltas, viewport point, action
    pytest.param(
        False,
        (160, 990),
        [(0, 800)],
        (205, 785),
        "scrolled down by 800 px and clicked at (160, 990)",
        id="bottom-from-0",
    ),
    pytest.param(
        True,
        (160, 990),
        [(0, 200)],
        (205, 785),
        "scrolled down by 200 px and clicked at (160, 990)",
        id="bottom-from-600",
    ),
    pytest.param(
        True,
        (160, 50),
        [(0, -600)],
        (205, 80),
        "scrolled up by 600 px and clicked at (160, 50)",
        id="top-from-600",
    ),
]


@pytest.mark.parametrize(("scroll_first", "xy", "deltas", "point", "action"), CLAMP_ROWS)
async def test_scroll_clamped(
    scroll_first: bool,
    xy: tuple[int, int],
    deltas: list[tuple[int, int]],
    point: tuple[int, int],
    action: str,
) -> None:
    """A1-4 clamp (reviewer mutation 5, the target not clamped; the fake's page clamps the scroll itself, so the
    mutant still lands and clicks right and only the dispatched ``dy`` tells): on the 1600 px page a target near the
    bottom gives ``dy = C - H - S`` (800 from 0, 200 from 600; unclamped 1185 and 585), and one near the top from
    ``S = 600`` gives ``dy = -600`` (unclamped -920) and the ``scrolled up by`` prefix."""
    ex, d, ov = await long_page(1600.0, scroll_first=scroll_first)
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=xy[0], y=xy[1]))
    assert out.status == "executed", out
    assert scrolls(d, mark) == deltas, scrolls(d, mark)
    assert mouse_moves(d, mark) == [point], mouse_moves(d, mark)
    assert out.action == action, out.action
    assert out.resolved.get("scrolled_px") == deltas[0][1], out.resolved


async def test_scroll_from_current_position() -> None:
    """A1-4 ``S`` is the *current* ``scroll_y`` (the guard's too-wide side of A1-3: a scroll alone is no page change).
    The page scrolled itself to 200 after the capture, with no new own frame: the click still executes, with ``dy =
    420 - 200 = 220`` and ``scrolled down by 220 px``, at viewport y 721."""
    ex, d, ov = await long_page(1220.0)
    d.scroll_y = 200.0
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert out.status == "executed", out
    assert scrolls(d, mark) == [(0, 220)], scrolls(d, mark)
    assert mouse_moves(d, mark) == [(205, 721)], mouse_moves(d, mark)
    assert out.action == "scrolled down by 220 px and clicked at (160, 935)", out.action


NOT_REACHED_ROWS = [
    pytest.param("short", id="lands-short"),
    pytest.param("drift", id="reads-differ"),
    pytest.param("grow", id="content-changed"),
]


@pytest.mark.parametrize("fault", NOT_REACHED_ROWS)
@pytest.mark.parametrize("tool", ["click_at", "type_at"])
async def test_not_reached_fails(fault: str, tool: str) -> None:
    """A1-4 moved-page check (adversarial; reviewer mutations 6, the check removed, and 7, its content-height part
    removed; the named failure mode is clicking anyway): a scroll that lands at 100 (page y 1141 is not in ``[100,
    900)``), a page still moving (the two reads after the settle differ; 1141 stays in view, so only the equality
    test fails), and a content height that changed with the scroll (1220 -> 1520; the scroll landed at 420, in view,
    the reads equal). Each gives ``failed`` (``OVERVIEW_NOT_REACHED``) after exactly one ``scroll_by``: no button
    press, no typing, and the post-action frame captured (``frame_after`` is the new ``latest_frame``)."""
    assert const("OVERVIEW_NOT_REACHED") == NOT_REACHED
    ex, d, ov = await long_page(1220.0)
    if fault == "short":
        d.scroll_land = 100.0
    elif fault == "drift":
        d.drift = True
    else:
        d.grow_on_scroll = 300.0
    before = ex.latest_frame
    args: dict[str, Any] = {"x": 160, "y": 935}
    if tool == "type_at":
        args |= {"text": "mug", "press_enter": True}
    mark = len(d.calls)
    out = await ex.execute(inv(tool, ov, **args))
    assert (out.status, out.reason) == ("failed", NOT_REACHED), out
    assert out.action == f"failed {tool}: {NOT_REACHED}", out.action
    dispatched = [c[0] for c in inputs(d, mark)]
    assert dispatched.count("scroll_by") == 1, dispatched
    assert not {"mouse_down", "mouse_up", "type_text", "key_down"} & set(dispatched), (
        f"something was clicked or typed although the point was not reached: {dispatched}"
    )
    seq = names(d, mark)
    assert "screenshot" in seq[seq.index("scroll_by") :], (
        "A1-4: the post-action frame was not captured"
    )
    latest = ex.latest_frame
    assert latest is not None and latest != before, "premise: the scroll changed the screen"
    assert out.resolved.get("frame_after") == latest.phash, out.resolved


def assert_not_reached(ex: Any, d: PageDriver, out: Any, mark: int, before: Any) -> None:
    """``failed`` ``OVERVIEW_NOT_REACHED`` after exactly one scroll: nothing pressed or typed, the frame captured."""
    assert (out.status, out.reason) == ("failed", NOT_REACHED), out
    dispatched = [c[0] for c in inputs(d, mark)]
    assert dispatched.count("scroll_by") == 1, dispatched
    assert not {"mouse_down", "mouse_up", "type_text", "key_down"} & set(dispatched), (
        f"something was clicked or typed although the page had not arrived: {dispatched}"
    )
    seq = names(d, mark)
    assert "screenshot" in seq[seq.index("scroll_by") :], (
        "A1-4: the post-action frame was not captured"
    )
    latest = ex.latest_frame
    assert latest is not None and latest != before, "premise: the scroll changed the screen"
    assert out.resolved.get("frame_after") == latest.phash, out.resolved


@pytest.mark.parametrize("tool", ["click_at", "type_at"])
async def test_snap_page_still_moving_fails(tool: str) -> None:
    """O-A1-3 (F1, the real-page fault; reviewer A at 0a7ca21): a scroll-snap page that is still animating moves
    once per ``settle()`` and stands still between back-to-back reads. On the 1600 px page, overview (160, 625) is
    page y 1000 (``to_pixels(625, 1600) = 1999749 // 1998 = 1000``), the planned ``dy`` is ``clamp(600, 0, 800) - 0
    = 600``; the wheel lands at 450, the next settle snaps it to 501 and the one after to 600. Page y 1000 is in view
    at 501, so two reads taken back to back after one settle agree (501, 501) and the shipped check clicks at viewport
    y 499; two reads separated by a ``settle()`` see 501 then 600 (or 450 then 501), which differ. Expected:
    ``failed`` ``OVERVIEW_NOT_REACHED``, no button press, and the ``frame_after`` capture. Red at 0a7ca21."""
    ex, d, ov = await long_page(1600.0)
    d.scroll_land = 450.0
    d.settle_moves = [501.0, 600.0]
    before = ex.latest_frame
    args: dict[str, Any] = {"x": 160, "y": 625}
    if tool == "type_at":
        args |= {"text": "mug", "press_enter": True}
    mark = len(d.calls)
    out = await ex.execute(inv(tool, ov, **args))
    assert scrolls(d, mark) == [(0, 600)], f"premise: the planned scroll: {scrolls(d, mark)}"
    assert_not_reached(ex, d, out, mark, before)


async def test_viewport_changed_during_scroll_fails() -> None:
    """O-A1-3 (``viewport_height`` must equal the pre-scroll ``H``): on the 1220 px page the click at overview (160,
    935) scrolls 420 as planned, and during the post-scroll settle the viewport height becomes 600 (a resized
    window). Every read after the settle agrees with the next (600 tall, at 420, the content unchanged, page y 1141
    still inside ``[420, 420 + 800)``), so only the viewport comparison sees it: ``failed`` ``OVERVIEW_NOT_REACHED``,
    nothing clicked, the frame captured. Red at 0a7ca21."""
    ex, d, ov = await long_page(1220.0)
    d.viewport_on_settle = 600.0
    before = ex.latest_frame
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert scrolls(d, mark) == [(0, 420)], scrolls(d, mark)
    assert_not_reached(ex, d, out, mark, before)


@pytest.mark.parametrize("echo", ["model", "pixels"])
async def test_landed_in_view_not_at_target(echo: str) -> None:
    """A1-4/A1-5 with ``S'`` measured, not planned (F2; kills ``s_now = S + dy`` and a prefix from the planned
    ``dy``): on the 1220 px page the click at overview (160, 935) plans ``dy = 420``, but the page lands at 400 and
    stays there. Page y 1141 is in view at 400 (``[400, 1200)``), so the click goes ahead at viewport y ``1141 - 400 =
    741`` (not 721), ``scrolled_px`` is 400 and the prefix ``scrolled down by 400 px`` (``N = |S' - S|``)."""
    ex, d, ov = await long_page(1220.0, echo=echo)
    d.scroll_land = 400.0
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert out.status == "executed", out
    assert scrolls(d, mark) == [(0, 420)], f"premise: the planned scroll: {scrolls(d, mark)}"
    assert mouse_moves(d, mark) == [(205, 741)], mouse_moves(d, mark)
    shown = "(160, 935)" if echo == "model" else "(205, 741)"
    assert out.action == f"scrolled down by 400 px and clicked at {shown}", out.action
    r = out.resolved
    assert (r.get("x_px"), r.get("y_px"), r.get("page_y_px"), r.get("scrolled_px")) == (
        205,
        741,
        1141,
        400,
    ), r


# =====================================================================================================
# A1-3 refusals before dispatch
# =====================================================================================================


@pytest.mark.parametrize("fault", ["content", "viewport", "read-fails"])
async def test_page_changed_refused(fault: str) -> None:
    """A1-3 "Page changed size" (reviewer mutation 7, the pre-dispatch half): after registration the page grew
    (1220 -> 1300), the viewport height read changed (800 -> 700), or the metrics read fails; each click is
    ``refused`` (``OVERVIEW_PAGE_CHANGED``) and dispatches nothing. The too-wide side is ``test_scroll_from_current_
    position`` (a scroll alone) and the mapping rows (nothing changed)."""
    assert const("OVERVIEW_PAGE_CHANGED") == PAGE_CHANGED
    ex, d, ov = await long_page(1220.0)
    if fault == "content":
        d.content_height = 1300.0
    elif fault == "viewport":
        d.metrics_viewport = 700.0
    else:
        d.metrics_fail = True
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert (out.status, out.reason) == ("refused", PAGE_CHANGED), out
    assert inputs(d, mark) == [], inputs(d, mark)


@pytest.mark.parametrize("tool", ["click_at", "type_at"])
async def test_stale_overview_refused(tool: str) -> None:
    """A1-3 "Stale picture" (adversarial; reviewer mutation 9, the latest-frame check dropped): an overview
    registered, then one ``frame()`` capture with new pixels, then a call on the overview is ``refused``
    (``OVERVIEW_STALE``) and dispatches nothing, no scroll included. R-c "valid for one action": after an executed
    overview click (whose capture is the new latest frame), a second call on the same overview is stale too."""
    assert const("OVERVIEW_STALE") == STALE
    args: dict[str, Any] = {"x": 160, "y": 935}
    if tool == "type_at":
        args |= {"text": "mug", "press_enter": False}
    ex, d, ov = await long_page(1220.0)
    d.scene += 1
    await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(inv(tool, ov, **args))
    assert (out.status, out.reason) == ("refused", STALE), out
    assert inputs(d, mark) == [], inputs(d, mark)

    ex, d, ov = await long_page(1220.0)
    f0 = ex.latest_frame
    # the fake's screenshot cycles through four scenes and a scrolled click is four inputs: start from another
    # scene so the click's capture has new pixels (a new own frame), as a real click on a page has
    d.scene = 1
    out = await ex.execute(inv(tool, ov, **args))
    assert out.status == "executed", (
        f"premise (the guard's too-wide side): a fresh overview executes: {out}"
    )
    assert f0 is not None and out.resolved["frame_after"] != f0.phash, (
        "premise: the click's capture is new"
    )
    mark = len(d.calls)
    out = await ex.execute(inv(tool, ov, **args))
    assert (out.status, out.reason) == ("refused", STALE), (
        f"the overview frame served two actions: {out}"
    )
    assert inputs(d, mark) == []


async def test_existing_rules_apply() -> None:
    """A1-2/A1-3 "Existing rules" on an overview frame, each with BASE's reason and nothing dispatched: the guard's
    verdict, ``bad coordinates``, ``bad arguments``, a control character, an adopted popup (``the active page
    changed``), a frame taken on another page than the active one (same reason), a dead browser (``failed: browser
    restarted``) and, under ``end_when_closed``, every page closed (``the browser window was closed``). After a
    restart the old overview stays refused (the reason is not pinned: A1 does not say whether an overview of an
    earlier generation reads as ``frame from before a browser restart`` or as unknown)."""
    seen: list[str] = []

    def guard(i: Any) -> str | None:
        seen.append(i.name)
        return "not on this page" if i.arguments.get("y") == 111 else None

    ex, d, ov = await long_page(1220.0, guard=guard)
    cases: list[tuple[dict[str, Any], str, str, str]] = [
        ({"x": 160, "y": 111}, "click_at", "refused", "not on this page"),
        ({"x": 1000, "y": 935}, "click_at", "refused", "bad coordinates"),
        (
            {"x": 160, "y": 935, "text": "", "press_enter": False},
            "type_at",
            "refused",
            "bad arguments",
        ),
        (
            {"x": 160, "y": 935, "text": "a\x01b", "press_enter": False},
            "type_at",
            "refused",
            "bad arguments: control character",
        ),
    ]
    for args, tool, status, reason in cases:
        mark = len(d.calls)
        out = await ex.execute(inv(tool, ov, **args))
        assert (out.status, out.reason) == (status, reason), (args, out)
        assert inputs(d, mark) == [], (args, inputs(d, mark))
    assert seen, "premise: the guard was consulted"

    ex, d, ov = await long_page(1220.0)
    d.open_popup()
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert (out.status, out.reason) == ("refused", ADOPTED), out
    assert inputs(d, mark) == []
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert (out.status, out.reason) == ("refused", ADOPTED), f"frame-page rule: {out}"
    assert inputs(d, mark) == []

    ex, d, ov = await long_page(1220.0)
    d.kill()
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert (out.status, out.reason) == ("failed", "browser restarted"), out
    assert inputs(d, mark) == []
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert out.status == "refused" and out.reason in (RESTART_FRAME, NOT_OWN), out
    assert inputs(d, mark) == []

    ex, d, ov = await long_page(1220.0, end_when_closed=True)
    d.close_all_pages()
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert (out.status, out.reason) == ("refused", CLOSED), out
    assert inputs(d, mark) == []


async def test_overview_action_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Implementation notes ("the A1-4 scroll and click inside the same ``ACTION_TIMEOUT_S`` block"): with the bound
    patched to 0.3 s, a ``scroll_by`` that hangs settles as ``failed`` naming the timeout within the bound plus
    slack, and no button is pressed."""
    small = 0.3
    monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", small)
    ex, d, ov = await long_page(1220.0)
    d.arm_hang("scroll_by")
    mark = len(d.calls)
    try:
        out = await asyncio.wait_for(ex.execute(inv("click_at", ov, x=160, y=935)), small + 1.5)
    except TimeoutError:
        raise AssertionError("a hung overview scroll was not bounded by ACTION_TIMEOUT_S") from None
    assert d.hanging is not None and d.hanging.is_set(), "premise: the scroll hung"
    assert out.status == "failed" and out.reason and "Timeout" in out.reason, out
    assert "mouse_down" not in [c[0] for c in inputs(d, mark)]


# =====================================================================================================
# A1-5 action strings
# =====================================================================================================

ECHO_ROWS = [
    # tool, (x, y), text/enter, model string, pixels string
    pytest.param(
        "click_at", (160, 308), None, "clicked at (160, 308)", "clicked at (205, 376)", id="click"
    ),
    pytest.param(
        "click_at",
        (160, 935),
        None,
        "scrolled down by 420 px and clicked at (160, 935)",
        "scrolled down by 420 px and clicked at (205, 721)",
        id="click-scrolled",
    ),
    pytest.param(
        "type_at",
        (160, 308),
        ("mug", False),
        "clicked at (160, 308) and typed 3 characters",
        "clicked at (205, 376) and typed 3 characters",
        id="type",
    ),
    pytest.param(
        "type_at",
        (160, 308),
        ("mugs", True),
        "clicked at (160, 308) and typed 4 characters and pressed Enter",
        "clicked at (205, 376) and typed 4 characters and pressed Enter",
        id="type-enter",
    ),
    pytest.param(
        "type_at",
        (160, 935),
        ("mug", False),
        "scrolled down by 420 px and clicked at (160, 935) and typed 3 characters",
        "scrolled down by 420 px and clicked at (205, 721) and typed 3 characters",
        id="type-scrolled",
    ),
    pytest.param(
        "type_at",
        (160, 935),
        ("mug", True),
        "scrolled down by 420 px and clicked at (160, 935) and typed 3 characters and pressed Enter",
        "scrolled down by 420 px and clicked at (205, 721) and typed 3 characters and pressed Enter",
        id="type-enter-scrolled",
    ),
]


@pytest.mark.parametrize("echo", ["model", "pixels"])
@pytest.mark.parametrize(("tool", "xy", "typed", "model", "pixels"), ECHO_ROWS)
async def test_action_strings_echo(
    echo: str,
    tool: str,
    xy: tuple[int, int],
    typed: tuple[str, bool] | None,
    model: str,
    pixels: str,
) -> None:
    """A1-5 exactly, for both echoes (reviewer mutation 13, the echo in page units, would say (205, 1141) or (205,
    376) under ``model``, and (205, 1141) under ``pixels``): without a scroll BASE's strings, in the call's own
    overview units (``model``) or the viewport pixels dispatched (``pixels``); with one, the ``scrolled down by 420 px
    and `` prefix. ``resolved`` has ``x_px``/``y_px`` (viewport), ``page_y_px``, ``scrolled_px``, ``frame_after``
    and, for ``type_at``, BASE's ``text``/``press_enter``."""
    ex, _, ov = await long_page(1220.0, echo=echo)
    args: dict[str, Any] = {"x": xy[0], "y": xy[1]}
    if typed is not None:
        args |= {"text": typed[0], "press_enter": typed[1]}
    out = await ex.execute(inv(tool, ov, **args))
    assert out.status == "executed", out
    assert out.action == (model if echo == "model" else pixels), out.action
    scrolled = xy[1] == 935
    r = out.resolved
    want = {
        "x_px": 205,
        "y_px": 721 if scrolled else 376,
        "page_y_px": 1141 if scrolled else 376,
        "scrolled_px": 420 if scrolled else 0,
    }
    assert {k: r.get(k) for k in want} == want, r
    latest = ex.latest_frame
    assert latest is not None and r.get("frame_after") == latest.phash
    if typed is not None:
        assert (r.get("text"), r.get("press_enter")) == typed, r


@pytest.mark.parametrize("scrolled", [False, True])
async def test_type_at_on_overview(scrolled: bool) -> None:
    """A1-4 for ``type_at``: the text and Enter are dispatched after the click at the viewport point, in BASE's
    order (move, down, up, type, Enter down/up), after the one scroll when there is one; then a settle and the
    capture."""
    ex, d, ov = await long_page(1220.0)
    y, point = (935, (205, 721)) if scrolled else (308, (205, 376))
    mark = len(d.calls)
    out = await ex.execute(inv("type_at", ov, x=160, y=y, text="mug", press_enter=True))
    assert out.status == "executed", out
    want: list[tuple[str, tuple[Any, ...]]] = [("scroll_by", (0, 420))] if scrolled else []
    want += [
        ("mouse_move", point),
        ("mouse_down", ()),
        ("mouse_up", ()),
        ("type_text", ("mug",)),
        ("key_down", ("Enter",)),
        ("key_up", ("Enter",)),
    ]
    got = [
        (n, tuple(round(a) if isinstance(a, float) else a for a in args))
        for n, args in inputs(d, mark)
    ]
    assert got == want, got
    seq = names(d, mark)
    tail = seq[len(seq) - seq[::-1].index("key_up") :]
    assert "settle" in tail and "screenshot" in tail[tail.index("settle") :], tail


# =====================================================================================================
# A1-8 screen frames unchanged
# =====================================================================================================


def screen_corpus() -> list[tuple[str, dict[str, Any], bool, str, str, dict[str, Any]]]:
    """(tool, arguments, on the screen frame?, echo, expected action, expected resolved minus ``frame_after``) by hand
    from T-0111c/T-0134: ``to_pixels(500, 1280) = 640``, ``to_pixels(500, 800) = 400``, a scroll is 3/4 of 800."""
    return [
        (
            "click_at",
            {"x": 500, "y": 500},
            True,
            "pixels",
            "clicked at (640, 400)",
            {"x_px": 640, "y_px": 400},
        ),
        (
            "click_at",
            {"x": 500, "y": 500},
            True,
            "model",
            "clicked at (500, 500)",
            {"x_px": 640, "y_px": 400},
        ),
        (
            "type_at",
            {"x": 500, "y": 500, "text": "mug", "press_enter": True},
            True,
            "pixels",
            "clicked at (640, 400) and typed 3 characters and pressed Enter",
            {"x_px": 640, "y_px": 400, "text": "mug", "press_enter": True},
        ),
        (
            "scroll",
            {"direction": "down"},
            False,
            "pixels",
            "scrolled down by 600 px",
            {"direction": "down", "dx_px": 0, "dy_px": 600},
        ),
        ("press_key", {"combo": "Tab"}, False, "model", "pressed Tab", {"combo": "Tab"}),
    ]


@pytest.mark.parametrize("driver", ["metrics", "plain"])
async def test_screen_frames_unchanged(driver: str) -> None:
    """A1-8 (green at BASE_A1): on a 1220 px page (a driver with layout metrics and a full-page capture) and on
    T-0111c's plain fake, a corpus of screen-frame calls gives BASE's outcomes, by hand: exact action strings, exact
    ``resolved`` (no ``page_y_px``/``scrolled_px``), the viewport mapping and no ``scroll_by`` before a click;
    ``bad coordinates`` is refused; and a click on a T-0140 overview read through ``overview_of`` is still refused
    with ``frame not from this browser`` (``overview_of`` registers nothing)."""
    for tool, args, framed, echo, action, resolved in screen_corpus():
        d: FakeDriver = (
            PageDriver(content_height=1220.0) if driver == "metrics" else ClosableDriver()
        )
        ex = build(d, echo=echo)
        f0 = await own_frame(ex)
        mark = len(d.calls)
        out = await ex.execute(inv(tool, f0 if framed else None, **args))
        assert (out.status, out.action, out.reason) == ("executed", action, None), out
        latest = ex.latest_frame
        assert latest is not None
        assert out.resolved == resolved | {"frame_after": latest.phash}, out.resolved
        if framed:
            assert mouse_moves(d, mark) == [(640, 400)], mouse_moves(d, mark)
            assert scrolls(d, mark) == []
        out = await ex.execute(inv("click_at", f0, x=1000, y=5))
        assert (out.status, out.reason) == ("refused", "bad coordinates"), out

    d2 = PageDriver(content_height=1220.0)
    ex = build(d2)
    await own_frame(ex)
    key = await capture(ex)
    ov = ex.overview_of(key)
    assert isinstance(ov, Frame)
    for g in (ov, FrameGeometry(phash=ov.phash, source=Source.WINDOW, width=1280, height=800)):
        mark = len(d2.calls)
        out = await ex.execute(inv("click_at", g, x=160, y=935))
        assert (out.status, out.reason) == ("refused", NOT_OWN), out
        assert inputs(d2, mark) == []


async def test_screen_frame_after_registration() -> None:
    """A1-8 with an overview frame registered: a click on the screen frame (the latest own frame) is BASE's, by
    hand: ``clicked at (500, 500)`` (model echo), ``resolved`` exactly ``{x_px: 640, y_px: 400, frame_after}``, the
    pointer at (640, 400) and no scroll. The overview frame is then stale (the click's capture is the latest frame)."""
    ex, d, ov = await long_page(1220.0)
    f0 = ex.latest_frame
    assert f0 is not None
    mark = len(d.calls)
    out = await ex.execute(inv("click_at", f0, x=500, y=500))
    assert (out.status, out.action) == ("executed", "clicked at (500, 500)"), out
    latest = ex.latest_frame
    assert latest is not None
    assert out.resolved == {"x_px": 640, "y_px": 400, "frame_after": latest.phash}, out.resolved
    assert mouse_moves(d, mark) == [(640, 400)] and scrolls(d, mark) == []
    out = await ex.execute(inv("click_at", ov, x=160, y=935))
    assert (out.status, out.reason) == ("refused", STALE), out


# =====================================================================================================
# A1-8 frozen paths
# =====================================================================================================

FROZEN = (
    "local_astra/computer/runtime.py",
    "local_astra/computer/fence.py",
    "local_astra/toolcall",
    "local_astra/conductor",
    "local_astra/contracts",
    "tests/fakes",
    "local_astra/pages",
    "services",
)


def test_frozen_paths_over_closed_range() -> None:
    """A1-8: the frozen paths are byte-identical over ``BASE_A1..TIP_A1`` (two literals, never the working tree).
    Green at BASE_A1 (an empty range until ``TIP_A1`` is bumped to the implementation's last commit)."""
    gr.require_commits(BASE_A1, TIP_A1)
    assert gr.is_ancestor(BASE_A1, TIP_A1), (BASE_A1, TIP_A1)
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_A1, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), (
            f"premise: {p} exists at {BASE_A1}"
        )
    changed = gr.changed(BASE_A1, TIP_A1, *FROZEN)
    assert changed == [], f"T-0151 A1-8: frozen paths differ in {BASE_A1}..{TIP_A1}: {changed}"
