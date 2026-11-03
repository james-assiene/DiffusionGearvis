"""T-0204 group A (browser): the real ``PlaywrightDriver``'s ``screenshot_region`` and layout-shift recorder against
real Chromium on loopback (docs/tasks/T-0204.md, A1-A4, "Tests to write" A).

**Gate.** As T-0111c's module (its autouse gate imported): ``integration`` + ``browser``, runs only with
``ASTRA_BROWSER_TESTS=1``. A missing ``playwright`` or Chromium is then a failure naming P1 of T-0111c.

**Fixture.** ``tests/fixtures/t0204/tiles.html`` (geometry and colours by hand in ``MANIFEST.md`` there), served by this
module's own ``http.server`` on 127.0.0.1. The rows drive the page's test hooks (``window.t0204.*``) through
``page.evaluate``; the driver itself never evaluates them. Expected geometry: tile 7 at x 40-240, y 250-400 in the
document, rgb(240, 120, 20); after a 200 px scroll and a 300 px insert above the grid, the grid's visible part moves
from (40, 0, 1200, 800) to (40, 200, 1200, 600) in viewport CSS px.

**Readings** (2026-10-29, BASE 7da07925, Playwright 1.64.0, Chromium 156.0.8078.4, 1280x800, loopback, load 4.8 (not
< 2: other agents were running), scratchpad probes driving Playwright directly, not this module):

- R1, headless and headed (DISPLAY :1): ``PerformanceTimeline.timelineEventAdded`` with ``eventTypes:
  ["layout-shift"]`` on a page CDP session delivers one event for an insert after a wheel scroll, with
  ``layoutShiftDetails.sources[].previousRect/currentRect`` as ``{x, y, width, height}`` in **viewport** CSS px,
  clipped to the viewport (a source moving fully in or out reads ``{0, 0, 0, 0}``); the same CSS values at device
  scale factor 2. A CSS ``transform`` move gives no event. After a same-origin and after a cross-origin
  (127.0.0.1 -> localhost) cross-document navigation, events keep arriving on the same session without a second
  enable. Enable round trip 0.4 ms headless, 1.3 ms headed. Latency from ``evaluate`` return to the event, n = 10:
  headless median 15.7 ms (first 4.9 ms, then 15.5-16.9: one frame), headed median 15.6 ms (2.5-17.5).
- R2, n = 20 interleaved, after 3 warm-ups, 128x128 clip at (576, 336): headless full ``screenshot()`` median 38.5 ms,
  p90 38.9 ms; region median 28.2 ms, p90 29.1 ms. Headed: full median 39.6 ms, p90 46.5 ms (one 2.99 s outlier);
  region median 27.4 ms, p90 29.7 ms (one 0.98 s outlier). A region costs about 70 % of a full frame: the capture
  waits for a frame either way.

Rows red at BASE carried ``pending(task="T-0204")`` until the activation commit (activated for group A).

Regime: Playwright 1.64.0, Chromium headless (versions.lock ``[browser]``), 1280x800 at DSF 1, not web mode, loopback
only, n = 1 per row, one browser per row.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import http.server
import io
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from local_astra.computer import runtime as runtime_mod
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
    _Quiet,
    run,
)
from tests.integration.test_t0174_navigated_browser import (
    BASE_SCRIPTS,
    SCRIPT_CDP,
    _install,
    _Recorder,
    peek,
)

pytestmark = [pytest.mark.integration] + (
    [pytest.mark.browser] if _browser_marker_registered() else []
)

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "t0204"
W, H = 1280, 800
#: tile 7's colour (tiles.html, by hand)
TILE7_RGB = (240, 120, 20)
#: JPEG tolerance per channel, and A1's luma bound
TOL = 16
LUMA_MAX = 3.0
#: "within +-4 px of the fixture's geometry" (Tests to write A)
PX = 4.0
#: how long a row waits for a shift entry to arrive (R1: about 16 ms)
ARRIVE_S = 1.0

#: CDP methods BASE's driver sends on a plain launch and ``layout_metrics()`` (runtime.py at 7da07925: ``_deny_permissions``
#: and T-0134's metrics session)
BASE_CDP = frozenset({"Target.getTargetInfo", "Browser.setPermission", "Page.getLayoutMetrics"})
#: ``runtime.CHROMIUM_ARGS`` at 7da07925, as a literal (A4: the launch arguments are unchanged)
BASE_ARGS = [
    "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE localhost , EXCLUDE 127.0.0.1",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-sync",
    "--disable-domain-reliability",
    "--disable-client-side-phishing-detection",
    "--no-pings",
    "--dns-prefetch-disable",
    "--disable-smooth-scrolling",
    "--deny-permission-prompts",
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
]
#: sha256 of ``runtime._NO_WEBRTC``, the one init script outside web mode (one of T-0174's ``BASE_SCRIPTS``)
NO_WEBRTC_SHA = "d8d605fa3836014c4e276ccf5c8b7fee0d2774757457485b7d47ffaf4098055f"

Rect = tuple[float, float, float, float]


# --------------------------------------------------------------------------------------------- server


@pytest.fixture(scope="module")
def tiles_server() -> Iterator[int]:
    """Serves ``tests/fixtures/t0204`` on 127.0.0.1; yields the port (rows build 127.0.0.1 and localhost URLs)."""
    handler = functools.partial(_Quiet, directory=str(FIX))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, name="t0204-server", daemon=True)
    t.start()
    try:
        yield int(srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(5)


# --------------------------------------------------------------------------------------------- rig


@contextlib.asynccontextmanager
async def browser() -> AsyncIterator[Any]:
    d = runtime_mod.PlaywrightDriver(viewport=(W, H), on_egress=lambda e: None, headless=True)
    try:
        await run(d.ensure())
        yield d
    finally:
        with contextlib.suppress(Exception):
            await run(d.close(), 20)


def _fn(d: Any, name: str) -> Callable[..., Any]:
    fn = getattr(d, name, None)
    assert fn is not None, f"T-0204 A: PlaywrightDriver.{name} does not exist yet"
    return fn  # type: ignore[no-any-return]


async def opened(d: Any, port: int, query: str = "", host: str = "127.0.0.1") -> None:
    await run(d.goto(f"http://{host}:{port}/tiles.html{'?' + query if query else ''}"))
    await run(d.settle())


async def arm(d: Any) -> None:
    ok = await run(_fn(d, "arm_shift_recorder")())
    assert ok is True, f"A3: arming on a loopback page gave {ok!r}"


async def hook(d: Any, expression: str, rec: _Recorder | None = None) -> Any:
    """The test's own call of a fixture hook (never the driver's)."""
    if rec is not None:
        return await peek(rec, d.page, expression)
    return await run(d.page.evaluate(expression))


async def arrived(d: Any, mark: object) -> tuple[Rect, ...]:
    """``shifts_since(mark)`` once it is non-empty (then 150 ms later, so a late second entry is included)."""
    since = _fn(d, "shifts_since")
    deadline = time.monotonic() + ARRIVE_S
    got = since(mark)
    while not got and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
        got = since(mark)
    assert got, f"A2: no layout-shift entry within {ARRIVE_S} s (read {got!r})"
    await asyncio.sleep(0.15)
    got = since(mark)
    assert isinstance(got, tuple), got
    return got


def nonzero(rects: tuple[Rect, ...]) -> list[Rect]:
    return [r for r in rects if r[2] > 0 and r[3] > 0]


def near(r: Rect, want: Rect) -> bool:
    return all(abs(a - b) <= PX for a, b in zip(r, want, strict=True))


def has(rects: tuple[Rect, ...], want: Rect) -> bool:
    return any(near(r, want) for r in rects)


def in_viewport(r: Rect) -> bool:
    x, y, w, h = r
    return x >= -PX and y >= -PX and x + w <= W + PX and y + h <= H + PX


def luma(jpeg: bytes) -> Any:
    from PIL import Image

    return Image.open(io.BytesIO(jpeg)).convert("L")


def mean_abs(a: Any, b: Any) -> float:
    from PIL import ImageChops, ImageStat

    assert a.size == b.size, (a.size, b.size)
    return float(ImageStat.Stat(ImageChops.difference(a, b)).mean[0])


# --------------------------------------------------------------------------------------------- A1


#: (requested region, the clamped box (x0, y0, x1, y1) by hand)
REGIONS: list[tuple[Rect, tuple[int, int, int, int]]] = [
    (
        (44, 54, 128, 128),
        (44, 54, 172, 182),
    ),  # inside tile 7 after the 200 px scroll (viewport y 50-200)
    ((580, 300, 128, 128), (580, 300, 708, 428)),
    ((1200, 750, 200, 200), (1200, 750, 1280, 800)),
    ((-50, -20, 128, 128), (0, 0, 78, 108)),
]


async def test_region_matches_full_screenshot_region(tiles_server: int) -> None:
    """A1: on the static tiles page scrolled by 200 px (so viewport and document coordinates differ), each region's
    JPEG has the clamped size and matches the same box of ``screenshot()`` within a mean absolute luma difference of 3;
    the first region's corner pixel is tile 7's colour (viewport (50, 60) = document (50, 260)). The reads dispatch no
    input and do not scroll (``window.__input`` and ``layout_metrics()``'s ``scroll_y`` unchanged)."""
    from PIL import Image

    async with browser() as d:
        await opened(d, tiles_server)
        await run(d.scroll_by(0, 200))
        await run(d.settle())
        await asyncio.sleep(0.2)
        assert (await run(d.layout_metrics()))["scroll_y"] == 200.0, "premise: scrolled by 200"
        inputs = await hook(d, "window.__input")
        region = _fn(d, "screenshot_region")
        full = luma(await run(d.screenshot()))
        assert full.size == (W, H)
        for (x, y, w, h), box in REGIONS:
            shot = await run(region(x, y, w, h))
            assert isinstance(shot, bytes) and shot[:2] == b"\xff\xd8", "A1: a JPEG"
            got = luma(shot)
            assert got.size == (box[2] - box[0], box[3] - box[1]), (
                f"region {(x, y, w, h)}: {got.size}, clamped box {box}"
            )
            diff = mean_abs(got, full.crop(box))
            assert diff <= LUMA_MAX, (
                f"region {(x, y, w, h)} differs from the full frame: {diff:.2f}"
            )
        corner: Any = Image.open(io.BytesIO(await run(region(44, 54, 128, 128)))).convert("RGB")
        rgb = corner.getpixel((6, 6))
        assert all(abs(a - b) <= TOL for a, b in zip(rgb, TILE7_RGB, strict=True)), (
            f"A1: the region's corner is {rgb}, not tile 7's {TILE7_RGB} (document vs viewport coordinates?)"
        )
        assert await hook(d, "window.__input") == inputs, "A1: a region screenshot dispatched input"
        assert (await run(d.layout_metrics()))["scroll_y"] == 200.0, (
            "A1: a region screenshot scrolled"
        )
        with pytest.raises(ValueError):
            await run(region(1300, 10, 50, 50))


# --------------------------------------------------------------------------------------------- A2/A3


async def test_lazy_insert_reports_shift_rect(tiles_server: int) -> None:
    """A2, R1: ``tiles.html?lazy=300`` inserts 300 px above the grid 50 ms after the first scroll. Armed, marked, then
    scrolled by 200 px: the read holds the grid's visible part before (40, 0, 1200, 800) and after
    (40, 200, 1200, 600), each within 4 px (document coordinates would put it at y 400), and every non-zero rect lies
    in the viewport. A mark taken once it settled reads ``()``."""
    async with browser() as d:
        await opened(d, tiles_server, "lazy=300")
        await arm(d)
        m = _fn(d, "shift_mark")()
        assert _fn(d, "shifts_since")(m) == (), "armed and quiet before the scroll"
        await run(d.scroll_by(0, 200))
        got = await arrived(d, m)
        rects = nonzero(got)
        assert has(got, (40, 200, 1200, 600)), (
            f"no current rect of the grid near (40, 200, 1200, 600): {rects}"
        )
        assert has(got, (40, 0, 1200, 800)), (
            f"no previous rect of the grid near (40, 0, 1200, 800): {rects}"
        )
        assert all(in_viewport(r) for r in rects), f"a rect outside the viewport frame: {rects}"
        assert await hook(d, "document.getElementById('above') !== null"), (
            "premise: the insert happened"
        )
        await asyncio.sleep(0.2)
        m2 = _fn(d, "shift_mark")()
        await asyncio.sleep(0.3)
        assert _fn(d, "shifts_since")(m2) == (), "a settled static page reported shifts"


async def test_transform_move_reports_no_shift(tiles_server: int) -> None:
    """A2 / threat model: a CSS ``transform`` moving the grid 300 px gives no layout-shift entry, so the read stays
    ``()`` (not ``None``) for 0.4 s (about 24 frames). Then, as a control on the same mark's recorder, a real insert
    above the grid is reported: the empty read was not a dead recorder."""
    async with browser() as d:
        await opened(d, tiles_server)
        await arm(d)
        m = _fn(d, "shift_mark")()
        await hook(d, "window.t0204.transformGrid(300)")
        await asyncio.sleep(0.4)
        assert _fn(d, "shifts_since")(m) == (), "a transform move was reported as a layout shift"
        m2 = _fn(d, "shift_mark")()
        await hook(d, "window.t0204.insertAbove(300)")
        got = await arrived(d, m2)
        assert nonzero(got), got


async def test_shift_after_cross_document_navigation(tiles_server: int) -> None:
    """A3, R1: armed once on ``tiles.html``; after a same-origin cross-document navigation and then a cross-origin one
    (127.0.0.1 -> localhost), with no second arm, an insert of 300 px above the grid at scroll 0 is still reported:
    the grid's visible part from (40, 100, 1200, 700) to (40, 400, 1200, 400), within 4 px."""
    async with browser() as d:
        await opened(d, tiles_server)
        await arm(d)
        for query, host in (("second=1", "127.0.0.1"), ("third=1", "localhost")):
            await opened(d, tiles_server, query, host)
            assert f"{host}:{tiles_server}" in d.page.url and query in d.page.url, d.page.url
            m = _fn(d, "shift_mark")()
            await hook(d, "window.t0204.insertAbove(300)")
            got = await arrived(d, m)
            assert has(got, (40, 400, 1200, 400)) and has(got, (40, 100, 1200, 700)), (
                f"after navigating to {host} ({query}): {nonzero(got)}"
            )


def covers(rects: tuple[Rect, ...], point: tuple[float, float]) -> bool:
    px, py = point
    return any(x <= px <= x + w and y <= py <= y + h for x, y, w, h in rects)


async def test_capped_sources_cover_every_moved_block(tiles_server: int) -> None:
    """A-R1, the reviewer's probe: ``blocks.html`` stacks 12 sibling 60 px blocks under a 10 px spacer (block *i* at
    y ``10 + 60 i``, x 0 to ``300 + 20 i``, by hand); 100 px are inserted above them, so all 12 move. Chromium names at
    most 5 sources per entry, so the read must still cover block 2 at its old place (150, 160) and its new place
    (150, 260): an entry at the cap reads as the whole viewport."""
    async with browser() as d:
        await run(d.goto(f"http://127.0.0.1:{tiles_server}/blocks.html"))
        await run(d.settle())
        await arm(d)
        m = _fn(d, "shift_mark")()
        await hook(d, "window.t0204.insertTop(100)")
        got = await arrived(d, m)
        for point in ((150.0, 160.0), (150.0, 260.0)):
            assert covers(got, point), (
                f"A-R1: block 2 moved but no rect covers {point}: {nonzero(got)} (Chromium's 5-source cap)"
            )


async def test_no_new_page_script_with_recorder(
    tiles_server: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A4 (negative): T-0174's recorder over the whole life of a browser (from the launch on) while the new methods
    run: arm, region screenshots, marks and reads across an insert and a cross-document navigation, plus BASE's
    ``layout_metrics`` and ``settle``. Every script handed to a page is one BASE runs; no CDP call runs script or
    touches ``DOM.*``; the CDP methods sent are BASE's plus exactly ``PerformanceTimeline.enable``; the only init
    script is ``_NO_WEBRTC``, added once; Chromium is launched with BASE's arguments."""
    from playwright.async_api import BrowserType

    rec = _Recorder()
    _install(monkeypatch, rec)
    launches: list[dict[str, Any]] = []
    launch = BrowserType.launch

    async def recorded_launch(self: Any, *args: Any, **kwargs: Any) -> Any:
        launches.append(dict(kwargs))
        return await launch(self, *args, **kwargs)

    monkeypatch.setattr(BrowserType, "launch", recorded_launch)
    async with browser() as d:
        await opened(d, tiles_server)
        await arm(d)
        await run(_fn(d, "screenshot_region")(580, 300, 128, 128))
        m = _fn(d, "shift_mark")()
        await hook(d, "window.t0204.insertAbove(300)", rec)
        assert nonzero(await arrived(d, m))
        await run(d.layout_metrics())
        await opened(d, tiles_server, "second=1")
        await arm(d)
        m = _fn(d, "shift_mark")()
        await hook(d, "window.t0204.insertAbove(120)", rec)
        assert nonzero(await arrived(d, m))
        await run(_fn(d, "screenshot_region")(0, 0, 128, 128))
        await run(d.settle())
    foreign = sorted(
        {
            (where, script[:80])
            for where, script in rec.scripts
            if hashlib.sha256(script.encode()).hexdigest() not in BASE_SCRIPTS
        }
    )
    assert foreign == [], f"A4: scripts BASE never ran reached a page: {foreign}"
    assert rec.scripts, "premise: the recorder saw BASE's settle script"
    methods = set(rec.cdp)
    assert methods & SCRIPT_CDP == set(), (
        f"A4: CDP calls that run script: {sorted(methods & SCRIPT_CDP)}"
    )
    assert not any(m_.startswith("DOM.") for m_ in methods), (
        f"A2: a node was resolved: {sorted(methods)}"
    )
    assert methods - BASE_CDP == {"PerformanceTimeline.enable"}, (
        f"A4: new CDP methods {sorted(methods - BASE_CDP)}; only PerformanceTimeline.enable is allowed"
    )
    inits = [s for where, s in rec.scripts if where.endswith("add_init_script")]
    assert [hashlib.sha256(s.encode()).hexdigest() for s in inits] == [NO_WEBRTC_SHA], (
        f"A4: init scripts changed: {[s[:60] for s in inits]}"
    )
    assert launches == [{"headless": True, "args": BASE_ARGS}], (
        f"A4: launch arguments changed: {launches}"
    )
