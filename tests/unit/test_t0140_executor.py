"""T-0140 A3/A4 (group A1): ``BrowserToolExecutor`` executing ``page_overview`` over a fake driver, and the driver
method ``runtime.PlaywrightDriver.full_page_screenshot`` over a fake page (docs/tasks/T-0140.md).

**The surface these rows pin** (A3's names; where A3 is silent the choice is stated here and in the report):

- the driver method ``async full_page_screenshot(max_height_css: int) -> bytes`` (a JPEG at CSS scale of the whole
  page, clipped to the top ``max_height_css`` CSS px), looked up on the driver's *class*, as T-0134 looks up
  ``layout_metrics``; a class without it, or without ``layout_metrics``, gives ``refused`` (``page overview is not
  available``) with nothing dispatched;
- ``execute(page_overview)``: ``executed`` with ``the whole page fits on the browser screen; no overview was taken``
  when ``content_height <= 1.05 * viewport_height`` (no image), else ``executed`` with ``showed the whole page (about
  {s:.1f} screens tall{C}) as a picture before the screen``, ``s = content_height / viewport_height``, ``{C} = ",
  top 4.0 screens shown"`` when ``s > MAX_OVERVIEW_SCREENS = 4``; ``failed`` (``the page moved during the
  overview``) when the metric reads around the capture differ;
- ``resolved["overview"]`` (an opaque id) and ``BrowserToolExecutor.overview_of(id) -> Frame | None`` (the scaled
  frame once, then ``None``). ``resolved["screens"]`` (``s``) and ``resolved["clipped"]`` (bool) are the
  test-author's reading of "Tests to write" (``gives `clipped`, `screens` 9.3``); flagged in the report;
- the overview frame fits within the viewport's CSS size, keeps the aspect, is never upscaled; it is never an own
  frame (not clickable, not ``latest_frame``, no ``extent_of``).

Expected values by hand (viewport 1280x800 unless stated): a 2000 px page is 2.5 screens and its 1280x2000 image is
scaled by min(1280/1280, 800/2000) = 0.4 to 512x800; a 7440 px page is 9.3 screens, clipped to 4 x 800 = 3200 CSS
px, and 1280x3200 is scaled by 0.25 to 320x800; viewport 1000x600 with a 1500 px page: 2.5 screens, 1000x1500 x 0.4
= 400x600; a 2560x1000 image (a page wider than the screen) x min(0.5, 0.8) = 1280x500. Fit boundary: 840 = 1.05 x
800 fits; 841 (1.05125 screens) does not and reads ``about 1.1 screens tall``.

BASE (group A1): 0bc60ff (the task header's ca730f5 is stale; main had moved when the branch was cut). Activated for
group A1 (rulings O-A1-1..8).

Regime: CPU over a fake driver, no browser, 1280x800 at DSF 1, n = 1, deterministic.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import re
from typing import Any

import pytest
from PIL import Image

from local_astra.computer import executor as executor_mod
from local_astra.computer import runtime as runtime_mod
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.toolcall import FrameGeometry, ToolOutcome
from tests.unit.test_computer_executor import FakeDriver, inv, own_frame
from tests.unit.test_computer_live_options import ClosableDriver

FITS = "the whole page fits on the browser screen; no overview was taken"
MOVED = "the page moved during the overview"
UNAVAILABLE = "page overview is not available"
ADOPTED = "the active page changed; take a new frame"
CLOSED = "the browser window was closed"
NOT_OWN = "frame not from this browser"
CLOCK_MS = 1_700_000_001_234


def full_page_jpeg(width: int, height: int) -> bytes:
    """A whole-page picture unlike any viewport screenshot of the fake: a red top quarter, a blue bottom quarter and
    a green middle (the fake's viewport frames are grey gradients), so a row can tell the overview is the page."""
    img = Image.new("RGB", (width, height), (40, 200, 40))
    img.paste((220, 30, 30), (0, 0, width, max(1, height // 4)))
    img.paste((30, 30, 220), (0, height - max(1, height // 4), width, height))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class OverviewDriver(ClosableDriver):
    """T-0111c's fake (with T-0111g's two end facts) plus T-0134's ``layout_metrics`` and T-0140's
    ``full_page_screenshot``: a page ``content_height`` CSS px tall over the viewport, scrolled by ``scroll_by``.

    ``full_page_screenshot(max_height_css)`` returns a JPEG ``image_width`` (default the viewport width) by
    ``min(content_height, max_height_css)`` (DSF 1: CSS px are pixels). ``image_size`` overrides the size outright
    (a synthetic small image, for the never-upscale row). ``move_during_capture`` changes the page *while* the
    capture runs (``{"content_height": +d}`` or ``{"scroll_y": +d}``), so any metrics read after it differs from
    any read before it, however many reads the executor makes."""

    def __init__(
        self, viewport: tuple[int, int] = (1280, 800), content_height: float = 2000.0
    ) -> None:
        super().__init__()
        self.viewport = viewport
        self.content_height = float(content_height)
        self.scroll_y = 0.0
        self.image_width: int | None = None
        self.image_size: tuple[int, int] | None = None
        self.move_during_capture: dict[str, float] = {}
        self.full_page_calls: list[int] = []

    async def layout_metrics(self) -> dict[str, float]:
        self._enter("layout_metrics")
        return {
            "scroll_y": self.scroll_y,
            "viewport_height": float(self.viewport[1]),
            "content_height": self.content_height,
        }

    async def full_page_screenshot(self, max_height_css: int) -> bytes:
        self._enter("full_page_screenshot", max_height_css)
        self.full_page_calls.append(max_height_css)
        await self._maybe_hang("full_page_screenshot")
        if self.image_size is not None:
            w, h = self.image_size
        else:
            w = self.image_width if self.image_width is not None else self.viewport[0]
            h = int(min(self.content_height, float(max_height_css)))
        for key, delta in self.move_during_capture.items():
            setattr(self, key, getattr(self, key) + delta)
        return full_page_jpeg(w, h)

    async def scroll_by(self, dx: float, dy: float) -> None:
        await super().scroll_by(dx, dy)
        top = max(0.0, self.content_height - self.viewport[1])
        self.scroll_y = max(0.0, min(top, self.scroll_y + dy))


class NoFullPageDriver(ClosableDriver):
    """A driver class with ``layout_metrics`` but no ``full_page_screenshot`` (A3 "Unavailable")."""

    async def layout_metrics(self) -> dict[str, float]:
        self._enter("layout_metrics")
        return {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 2000.0}


class NoMetricsDriver(ClosableDriver):
    """A driver class with ``full_page_screenshot`` but no ``layout_metrics`` (A3 "Unavailable")."""

    async def full_page_screenshot(self, max_height_css: int) -> bytes:
        self._enter("full_page_screenshot", max_height_css)
        return full_page_jpeg(1280, 2000)


def build(driver: FakeDriver, **kw: Any) -> Any:
    return executor_mod.BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS, viewport=driver.viewport, driver=driver, **kw
    )


def overview_of(ex: Any, key: Any) -> Any:
    fn = getattr(ex, "overview_of", None)
    assert callable(fn), "T-0140 A3: BrowserToolExecutor.overview_of(id) does not exist yet"
    return fn(key)


def overview_call() -> Any:
    return inv("page_overview")


def pixel(frame: Frame, fx: float, fy: float) -> tuple[int, int, int]:
    img = Image.open(io.BytesIO(frame.jpeg)).convert("RGB")
    rgb = img.getpixel((int(fx * (img.width - 1)), int(fy * (img.height - 1))))
    assert isinstance(rgb, tuple)
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def calls_named(d: FakeDriver, name: str, since: int = 0) -> list[int]:
    return d.index(name, since)


async def shown(ex: Any, d: OverviewDriver) -> tuple[ToolOutcome, Frame]:
    """One executed overview with an image: the outcome and the frame read back once."""
    out = await ex.execute(overview_call())
    assert out.status == "executed", out
    key = out.resolved.get("overview")
    assert key is not None, (
        f"T-0140 A3: an executed overview names its image in resolved['overview']: {out}"
    )
    f = overview_of(ex, key)
    assert isinstance(f, Frame), f"overview_of({key!r}) gave {f!r}"
    return out, f


# =====================================================================================================
# the executed overview: action, id, frame once, scaling
# =====================================================================================================

SCALE_ROWS = [
    # viewport, content height, image override (w, h) or None, expected action, expected frame size
    pytest.param(
        (1280, 800),
        2000.0,
        None,
        "showed the whole page (about 2.5 screens tall) as a picture before the screen",
        (512, 800),
        id="1280x800-2.5",
    ),
    pytest.param(
        (1000, 600),
        1500.0,
        None,
        "showed the whole page (about 2.5 screens tall) as a picture before the screen",
        (400, 600),
        id="1000x600-2.5",
    ),
    pytest.param(
        (1280, 800),
        1040.0,
        (2560, 1000),
        "showed the whole page (about 1.3 screens tall) as a picture before the screen",
        (1280, 500),
        id="wider-than-screen",
    ),
    pytest.param(
        (1280, 800),
        841.0,
        None,
        "showed the whole page (about 1.1 screens tall) as a picture before the screen",
        (1216, 800),
        id="just-over-the-fit-line",
    ),
]


@pytest.mark.parametrize("echo", ["pixels", "model"])
@pytest.mark.parametrize(("viewport", "content", "image", "action", "size"), SCALE_ROWS)
async def test_overview_action_and_frame(
    viewport: tuple[int, int],
    content: float,
    image: tuple[int, int] | None,
    action: str,
    size: tuple[int, int],
    echo: str,
) -> None:
    """A3: the exact action string (under either echo: it carries no coordinates), ``resolved["overview"]``,
    ``overview_of`` giving the frame once (the second read and an unknown id are ``None``), and the scaled size by
    hand (module docstring), within the viewport, aspect kept. The frame is the whole-page picture (red top, blue
    bottom), not a viewport screenshot. The metrics were read before and after the capture; nothing was dispatched.

    841 px: 1280x841 x min(1, 800/841) = 1217.6x800, which the screen_max policy's even rounding makes 1216 (or
    1218 by rounding up; either is accepted, within 2 px)."""
    d = OverviewDriver(viewport, content)
    if image is not None:
        d.image_size = image
    ex = build(d, echo=echo)
    await own_frame(ex)
    mark = len(d.calls)
    out, f = await shown(ex, d)
    assert out.action == action, out.action
    assert out.reason is None
    vw, vh = viewport
    assert f.width <= vw and f.height <= vh, (f.width, f.height, viewport)
    assert abs(f.width - size[0]) <= 2 and abs(f.height - size[1]) <= 2, (f.width, f.height, size)
    with Image.open(io.BytesIO(f.jpeg)) as img:
        assert img.size == (f.width, f.height), "the frame's bytes are its stated size"
    src_w, src_h = image if image is not None else (vw, int(content))
    assert abs(f.width / f.height - src_w / src_h) <= 0.01, "aspect not kept"
    top, bottom = pixel(f, 0.5, 0.02), pixel(f, 0.5, 0.98)
    assert top[0] > 150 and top[2] < 100, f"the top of the overview is not the page's top: {top}"
    assert bottom[2] > 150 and bottom[0] < 100, f"the bottom is not the page's bottom: {bottom}"
    assert overview_of(ex, out.resolved["overview"]) is None, "the overview is returned once"
    assert overview_of(ex, "no-such-overview") is None
    assert d.dispatched(mark) == [], d.dispatched(mark)
    shots = calls_named(d, "full_page_screenshot", mark)
    reads = calls_named(d, "layout_metrics", mark)
    assert len(shots) == 1, shots
    assert any(i < shots[0] for i in reads) and any(i > shots[0] for i in reads), (
        "A3: layout_metrics is read before and after the capture",
        d.calls[mark:],
    )


async def test_overview_resolved_screens_and_clipped() -> None:
    """A3 ("Tests to write": ``clipped``, ``screens``): an unclipped 2.5-screen page reports ``screens == 2.5`` and
    ``clipped is False`` beside its id."""
    d = OverviewDriver((1280, 800), 2000.0)
    ex = build(d)
    await own_frame(ex)
    out, _ = await shown(ex, d)
    assert out.resolved.get("screens") == pytest.approx(2.5), out.resolved
    assert out.resolved.get("clipped") is False, out.resolved


async def test_overview_never_upscaled() -> None:
    """A3 "never upscaled": a (synthetic) 600x700 picture is already within 1280x800, so the frame is 600x700
    (scaling it up to fit would give 686x800)."""
    d = OverviewDriver((1280, 800), 900.0)
    d.image_size = (600, 700)
    ex = build(d)
    await own_frame(ex)
    _, f = await shown(ex, d)
    assert (f.width, f.height) == (600, 700), (f.width, f.height)


# =====================================================================================================
# fits, clip, moved
# =====================================================================================================


@pytest.mark.parametrize("content", [800.0, 840.0], ids=["one-screen", "1.05-screens"])
async def test_fits_no_image(content: float) -> None:
    """A3 "Page fits": ``content_height <= 1.05 x 800`` (800 and exactly 840) gives ``executed`` with the fits
    string, no ``resolved["overview"]``, and no capture taken (the string says no overview was taken). The other
    side of the line is ``SCALE_ROWS``' 841 px row, which takes one."""
    d = OverviewDriver((1280, 800), content)
    ex = build(d)
    await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(overview_call())
    assert (out.status, out.action) == ("executed", FITS), out
    assert out.resolved.get("overview") is None, out.resolved
    assert d.full_page_calls == [], (
        f"an overview was taken of a page that fits: {d.full_page_calls}"
    )
    assert d.dispatched(mark) == []


async def test_clip_at_four_screens() -> None:
    """A3 clip (H-T140-2: ``MAX_OVERVIEW_SCREENS = 4``): a 9.3-screen page (7440 px) reads ``about 9.3 screens
    tall, top 4.0 screens shown``, ``screens`` 9.3 and ``clipped``; the capture asks for 4 x 800 = 3200 CSS px; the
    1280x3200 picture is scaled to 320x800. The constant is the executor's ``MAX_OVERVIEW_SCREENS``."""
    assert getattr(executor_mod, "MAX_OVERVIEW_SCREENS", None) == 4, (
        "T-0140 A3: executor.MAX_OVERVIEW_SCREENS = 4 does not exist yet"
    )
    d = OverviewDriver((1280, 800), 7440.0)
    ex = build(d)
    await own_frame(ex)
    out, f = await shown(ex, d)
    assert out.action == (
        "showed the whole page (about 9.3 screens tall, top 4.0 screens shown) as a picture before the screen"
    ), out.action
    assert out.resolved.get("screens") == pytest.approx(9.3), out.resolved
    assert out.resolved.get("clipped") is True, out.resolved
    assert d.full_page_calls == [3200], d.full_page_calls
    assert (f.width, f.height) == (320, 800), (f.width, f.height)


async def test_clip_boundary_at_exactly_four_screens() -> None:
    """A3 (the clip line's other side): exactly 4.0 screens (3200 px) is not clipped (``{C}`` only when clipped)."""
    d = OverviewDriver((1280, 800), 3200.0)
    ex = build(d)
    await own_frame(ex)
    out, f = await shown(ex, d)
    assert out.action == (
        "showed the whole page (about 4.0 screens tall) as a picture before the screen"
    ), out.action
    assert out.resolved.get("clipped") is False, out.resolved
    assert (f.width, f.height) == (320, 800), (f.width, f.height)


@pytest.mark.parametrize(
    "move", [{"content_height": 500.0}, {"scroll_y": 120.0}], ids=["grew", "scrolled"]
)
async def test_moved_page_fails(move: dict[str, float]) -> None:
    """A3 "Metrics": the page grows, or scrolls, while it is captured, so the reads before and after differ:
    ``failed`` (``the page moved during the overview``) with no ``resolved["overview"]`` (no image is offered), and
    nothing dispatched. Once the page is still again, the next overview is shown."""
    d = OverviewDriver((1280, 800), 2000.0)
    d.move_during_capture = dict(move)
    ex = build(d)
    await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(overview_call())
    assert d.full_page_calls, "premise: the capture ran"
    assert (out.status, out.reason) == ("failed", MOVED), out
    assert out.resolved.get("overview") is None, out.resolved
    assert d.dispatched(mark) == []
    d.move_during_capture = {}
    out, _ = await shown(ex, d)
    assert out.status == "executed", "a still page is shown again (the guard's too-wide side)"


async def test_stable_page_does_not_fail() -> None:
    """A3 "Metrics" (the guard's too-wide side): a scrolled but still page (scroll_y 600 before and after) is not
    "moved": it is shown, and from the top (red at the top of the picture)."""
    d = OverviewDriver((1280, 800), 2400.0)
    ex = build(d)
    await own_frame(ex)
    out = await ex.execute(inv("scroll", None, direction="down"))
    assert out.status == "executed" and d.scroll_y == 600.0, (out, d.scroll_y)
    out, f = await shown(ex, d)
    assert out.action == (
        "showed the whole page (about 3.0 screens tall) as a picture before the screen"
    ), out.action
    assert d.scroll_y == 600.0
    assert pixel(f, 0.5, 0.02)[0] > 150


# =====================================================================================================
# not an own frame
# =====================================================================================================


async def test_overview_never_own_frame() -> None:
    """A3 "Not an own frame" (adversarial; reviewer mutation 3, the overview registered in the own registry): a
    ``click_at``/``type_at`` whose frame identity is the overview's phash is ``refused`` like a foreign frame and
    dispatches nothing, with the overview's own geometry and with the screen's; ``latest_frame`` is still the
    screen frame (never the overview); ``extent_of`` of the overview is ``None`` and the screen frame's extent is
    unchanged. The phash is 16 hex and differs from every own frame's."""
    d = OverviewDriver((1280, 800), 2000.0)
    ex = build(d)
    f0 = await own_frame(ex)
    extent0 = ex.extent_of(f0.phash)
    assert extent0 == {"above_px": 0.0, "below_px": 1200.0, "viewport_px": 800.0}, extent0
    _, ov = await shown(ex, d)
    assert ov.phash is not None and re.fullmatch(r"[0-9a-f]{16}", ov.phash), ov.phash
    assert ov.phash != f0.phash
    latest = ex.latest_frame
    assert latest is not None and latest.phash != ov.phash, "the overview became latest_frame"
    assert (latest.width, latest.height) == (1280, 800), (latest.width, latest.height)
    assert latest == f0, "an overview changed latest_frame"
    assert ex.extent_of(ov.phash) is None, (
        "the overview has an extent: it was registered as an own frame"
    )
    assert ex.extent_of(f0.phash) == extent0
    for w, h in ((ov.width, ov.height), (1280, 800)):
        g = FrameGeometry(phash=ov.phash, source=Source.WINDOW, width=w, height=h)
        for call in (
            inv("click_at", g, x=500, y=500),
            inv("type_at", g, x=500, y=500, text="mug", press_enter=False),
        ):
            mark = len(d.calls)
            out = await ex.execute(call)
            assert (out.status, out.reason) == ("refused", NOT_OWN), (w, h, call.name, out)
            assert d.dispatched(mark) == [], d.dispatched(mark)
    out = await ex.execute(inv("click_at", f0, x=500, y=500))
    assert out.status == "executed", (
        f"the screen frame is still clickable (the guard's too-wide side): {out}"
    )


async def test_overview_phash_is_salted() -> None:
    """A3 "salted the same way": the overview's phash is not the raw (unsalted) DCT hash of its pixels, and it
    changes across a browser restart for the same pixels (a per-generation salt)."""
    d = OverviewDriver((1280, 800), 2000.0)
    ex = build(d)
    await own_frame(ex)
    _, ov1 = await shown(ex, d)
    with Image.open(io.BytesIO(ov1.jpeg)) as img:
        raw = executor_mod.phash(img)
    assert ov1.phash != raw, "the overview's phash is unsalted"
    d.kill()
    out = await ex.execute(inv("press_key", combo="Tab"))
    assert (out.status, out.reason) == ("failed", "browser restarted"), out
    await own_frame(ex)
    _, ov2 = await shown(ex, d)
    with Image.open(io.BytesIO(ov2.jpeg)) as img:
        assert executor_mod.phash(img) == raw, "premise: the same pixels"
    assert ov2.phash != ov1.phash, (
        "the same pixels across a restart share a phash: no per-generation salt"
    )


# =====================================================================================================
# unavailable, and the fence rules every non-search action follows
# =====================================================================================================


@pytest.mark.parametrize(
    "cls", [NoFullPageDriver, NoMetricsDriver], ids=["no-full-page", "no-metrics"]
)
async def test_unavailable_driver_refused(cls: type[ClosableDriver]) -> None:
    """A3 "Unavailable" (the T-0134 O-2 pattern): a driver class without ``full_page_screenshot`` or without
    ``layout_metrics`` gives ``refused`` (``page overview is not available``), dispatches nothing and captures no
    full page. The executor never touches the missing name (the fake's sentinel would raise)."""
    d = cls()
    ex = build(d)
    await own_frame(ex)
    mark = len(d.calls)
    out = await ex.execute(overview_call())
    assert (out.status, out.reason) == ("refused", UNAVAILABLE), out
    assert d.dispatched(mark) == []
    assert calls_named(d, "full_page_screenshot", mark) == []


async def test_fence_rules_apply_page_serial() -> None:
    """A3 fence: a popup adopted by ``execute()``'s ``ensure()`` refuses the overview with BASE's reason, and
    nothing is captured or dispatched."""
    d = OverviewDriver((1280, 800), 2000.0)
    ex = build(d)
    await own_frame(ex)
    d.open_popup()
    mark = len(d.calls)
    out = await ex.execute(overview_call())
    assert d.page_serial == 1, "premise: ensure() adopted the popup"
    assert (out.status, out.reason) == ("refused", ADOPTED), out
    assert calls_named(d, "full_page_screenshot", mark) == [] and d.dispatched(mark) == []
    out, _ = await shown(ex, d)
    assert out.status == "executed", "the next overview runs (the guard's too-wide side)"


async def test_fence_rules_apply_restart() -> None:
    """A3 fence: after the browser died, the overview call relaunches once and is ``failed: browser restarted``
    without capturing; the next one runs."""
    d = OverviewDriver((1280, 800), 2000.0)
    ex = build(d)
    await own_frame(ex)
    d.kill()
    mark = len(d.calls)
    out = await ex.execute(overview_call())
    assert (out.status, out.reason) == ("failed", "browser restarted"), out
    assert calls_named(d, "full_page_screenshot", mark) == []
    assert d.launches == 2
    out, _ = await shown(ex, d)
    assert out.status == "executed"


async def test_fence_rules_apply_end_when_closed() -> None:
    """A3 fence: under ``end_when_closed`` with every page closed, the overview is ``refused`` with ``the browser
    window was closed`` and nothing is captured, relaunched or reopened. Green at BASE (the closed check precedes
    every non-search call); it stays a row so an overview path that skips the check reddens it."""
    d = OverviewDriver((1280, 800), 2000.0)
    ex = build(d, end_when_closed=True)
    await own_frame(ex)
    d.close_all_pages()
    mark = len(d.calls)
    out = await ex.execute(overview_call())
    assert (out.status, out.reason) == ("refused", CLOSED), out
    assert calls_named(d, "full_page_screenshot", mark) == []
    assert d.launches == 1 and d.reopened == 0


async def test_fence_rules_apply_action_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A3 fence: the capture runs under ``ACTION_TIMEOUT_S`` (patched to 0.3 s): a capture that hangs settles as
    ``failed`` naming the timeout within the bound plus slack, and no image is kept."""
    small = 0.3
    monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", small)
    d = OverviewDriver((1280, 800), 2000.0)
    ex = build(d)
    await own_frame(ex)
    d.arm_hang("full_page_screenshot")
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    try:
        out = await asyncio.wait_for(ex.execute(overview_call()), small + 1.5)
    except TimeoutError:
        raise AssertionError(
            "a hung full-page capture was not bounded by ACTION_TIMEOUT_S"
        ) from None
    assert d.hanging is not None and d.hanging.is_set(), "premise: the capture hung"
    assert loop.time() - t0 >= small * 0.9
    assert out.status == "failed" and out.reason and "Timeout" in out.reason, out
    assert out.resolved.get("overview") is None


# =====================================================================================================
# the driver method
# =====================================================================================================


class _Sentinel:
    """Closes by a raising sentinel: any name not deliberately faked is a named failure (tests.md)."""

    _what = "object"

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise AssertionError(
            f"T-0140 A3/A4: full_page_screenshot used {self._what}.{name}; it may only take a screenshot (and read "
            "CDP layout metrics): no input, no scroll, no page text, title, URL or DOM"
        )


class FakeCdp(_Sentinel):
    _what = "cdp_session"

    def __init__(self, content_height: float) -> None:
        self.content_height = content_height
        self.sent: list[str] = []

    async def send(self, method: str, params: Any = None) -> Any:
        self.sent.append(method)
        assert method == "Page.getLayoutMetrics", (
            f"CDP {method}: only the layout metrics may be read"
        )
        return {
            "cssLayoutViewport": {"pageY": 0, "clientHeight": 800, "clientWidth": 1280},
            "cssContentSize": {"x": 0, "y": 0, "width": 1280, "height": self.content_height},
            "cssVisualViewport": {"pageY": 0, "clientHeight": 800, "clientWidth": 1280},
        }

    async def detach(self) -> None:
        pass


class FakePage(_Sentinel):
    _what = "page"

    def __init__(self) -> None:
        self.shots: list[dict[str, Any]] = []

    async def screenshot(self, **kw: Any) -> bytes:
        self.shots.append(dict(kw))
        return full_page_jpeg(1280, 3200)

    def is_closed(self) -> bool:
        return False

    @property
    def viewport_size(self) -> dict[str, int]:
        return {"width": 1280, "height": 800}


class FakeContext(_Sentinel):
    _what = "context"

    def __init__(self, cdp: FakeCdp) -> None:
        self.cdp = cdp

    async def new_cdp_session(self, page: Any) -> FakeCdp:
        return self.cdp


async def test_driver_full_page_screenshot_args() -> None:
    """A3 "Driver": ``PlaywrightDriver.full_page_screenshot(max_height_css)`` is a coroutine taking
    ``max_height_css``; on a 7440 px page asked for 3200 it takes one JPEG screenshot at CSS scale with
    ``full_page=True`` clipped to the top 3200 CSS px (``clip`` from y 0, height 3200), and touches nothing else of
    the page (no mouse, keyboard, wheel, evaluate, content, title or URL: the fake page's sentinel)."""
    cls = runtime_mod.PlaywrightDriver
    fn = getattr(cls, "full_page_screenshot", None)
    assert fn is not None and inspect.iscoroutinefunction(fn), (
        "T-0140 A3: PlaywrightDriver.full_page_screenshot does not exist yet"
    )
    assert "max_height_css" in inspect.signature(fn).parameters
    d = cls(viewport=(1280, 800), on_egress=lambda r: None)
    page = FakePage()
    d._page = page  # type: ignore[assignment]
    d._context = FakeContext(FakeCdp(7440.0))  # type: ignore[assignment]
    d._alive = True
    out = await d.full_page_screenshot(3200)
    assert isinstance(out, bytes) and out[:2] == b"\xff\xd8", "a JPEG"
    assert len(page.shots) == 1, page.shots
    kw = page.shots[0]
    assert kw.get("type") == "jpeg" and kw.get("scale") == "css" and kw.get("full_page") is True, kw
    clip = kw.get("clip")
    assert isinstance(clip, dict), f"no clip: {kw}"
    assert clip.get("x") == 0 and clip.get("y") == 0 and clip.get("height") == 3200, clip
