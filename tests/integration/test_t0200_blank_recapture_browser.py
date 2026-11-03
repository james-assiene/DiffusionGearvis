"""T-0200 criterion 10 in real Chromium on loopback: a late-painting page, a WAF-shaped token reload and a white page
through the real ``PlaywrightDriver`` and ``BrowserToolExecutor`` (docs/tasks/T-0200.md).

**Gate.** ``integration`` + ``browser``; runs only with ``ASTRA_BROWSER_TESTS=1`` (T-0111c's autouse gate,
imported). ``ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0200_blank_recapture_browser.py -q -m
browser``.

**The rig.** T-0111c's fixture directory on an ephemeral loopback port, a headless ``PlaywrightDriver`` at 1280x800
injected into the executor, ``PlaywrightDriver.screenshot`` spied (each call counted), ``on_note`` recording. The
fixtures are ``t0200_late_paint.html``, ``t0200_token_reload.html`` and ``t0200_white.html`` (MANIFEST.md);
``color.html`` (T-0111c) is the normal page.

**Oracles.** Pixels read independently of ``blank_frame``: the painted block's colour at the viewport centre
(``rgb_at``/``near``, T-0111c's helpers), and a white frame's full-resolution luma extrema.

**Readings (reported).** C10 "a page painting nothing for 1.5 s then content gives a non-blank ``frame()`` after
``open()``": the page's timer starts before ``open()`` returns (its settle waits for ``domcontentloaded`` and two
frames), so that row's page is ``?ms=1000`` (within ``BLANK_RECAPTURE_FRAME_S`` with a margin for screenshot time);
at 1500 the paint would land at the very end of the 1.5 s budget and the row would be a coin toss. The click row
uses the default 1500 ms (the action budget is 3 s).

Regime: Playwright's pinned Chromium, headless, loopback, 1280x800, n = 1 per row.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import http.server
import io
import threading
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.computer import runtime as runtime_mod
from local_astra.contracts.toolcall import ToolInvocation
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    CLOCK_MS,
    FIXTURES,
    _browser_layer,
    _Quiet,
    execute,
    geom,
    near,
    rgb_at,
    run,
)

pytestmark = [pytest.mark.integration, pytest.mark.browser]

BLUE = (0, 90, 200)
GREEN = (0, 150, 60)
RED = (255, 0, 0)
NOTE_KEYS = {"what", "path", "captures", "blank", "waited_ms"}

_N = iter(range(1_000_000))


@pytest.fixture(scope="module")
def origin() -> Iterator[str]:
    """T-0111c's fixture directory on an ephemeral loopback port (as its ``server`` fixture)."""
    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Quiet, directory=str(FIXTURES))
    )
    t = threading.Thread(target=srv.serve_forever, name="t0200-fixture-server", daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(5)


def call(name: str, frame: Any = None, **arguments: Any) -> ToolInvocation:
    return ToolInvocation(
        call_id=f"w{next(_N)}",
        turn_id="t1",
        name=name,
        arguments=arguments,
        source="draft",
        frame=geom(frame) if frame is not None else None,
    )


@pytest.fixture
def shots(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The monotonic time of every ``PlaywrightDriver.screenshot`` call."""
    seen: list[float] = []
    real = runtime_mod.PlaywrightDriver.screenshot

    async def wrapped(self: Any) -> bytes:
        seen.append(time.monotonic())
        out: bytes = await real(self)
        return out

    monkeypatch.setattr(runtime_mod.PlaywrightDriver, "screenshot", wrapped)
    return seen


@contextlib.asynccontextmanager
async def rig(notes: list[dict[str, Any]]) -> AsyncIterator[Any]:
    d = runtime_mod.PlaywrightDriver(viewport=(1280, 800), on_egress=lambda e: None, headless=True)
    ex = executor_mod.BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS, viewport=(1280, 800), driver=d, headless=True
    )
    assert hasattr(executor_mod, "blank_frame") and ex.on_note is None, (
        "T-0200: no blank re-capture yet"
    )
    ex.on_note = notes.append
    try:
        yield ex
    finally:
        with contextlib.suppress(Exception):
            await run(d.close(), 20)


def luma_range(f: Any) -> tuple[int, int]:
    from PIL import Image

    lo, hi = Image.open(io.BytesIO(f.jpeg)).convert("L").getextrema()
    return int(lo), int(hi)


def one_note(notes: list[dict[str, Any]], path: str, blank: bool) -> dict[str, Any]:
    # T-0204 C10: the visual settle adds its own note kind ("live frame: waited for a stable screen"); T-0200's rows
    # count T-0200's notes only
    notes = [n for n in notes if n.get("what") == "live frame: blank capture retaken"]
    assert len(notes) == 1, f"T-0200 C9/C10: one note for the re-taken capture; {notes}"
    n = notes[0]
    assert (
        set(n) == NOTE_KEYS and n["path"] == path and n["blank"] is blank and n["captures"] >= 2
    ), n
    return n


async def test_late_paint_frame_after_open(origin: str, shots: list[float]) -> None:
    """C10: ``open()`` a page that paints after 1 s: the next ``frame()`` shows the blue block (re-taken, one
    ``frame`` note)."""
    notes: list[dict[str, Any]] = []
    async with rig(notes) as ex:
        await run(ex.open(f"{origin}/t0200_late_paint.html?ms=1000"))
        n0 = len(shots)
        f = await run(ex.frame())
        assert near(rgb_at(f, 640, 400), BLUE), (
            f"T-0200 C10: frame() is still blank: {rgb_at(f, 640, 400)}"
        )
        assert len(shots) - n0 >= 2
        one_note(notes, "frame", False)


async def test_late_paint_click_after_frame(origin: str, shots: list[float]) -> None:
    """C10: a click on a painted link navigating to a page that paints after 1.5 s: ``executed``, the after-frame
    shows the blue block, one ``action`` note."""
    notes: list[dict[str, Any]] = []
    async with rig(notes) as ex:
        await run(ex.open(f"{origin}/t0200_late_paint.html?ms=0"))
        await asyncio.sleep(0.2)
        f = await run(ex.frame())
        assert near(rgb_at(f, 640, 400), BLUE) and notes == [], "premise: the start page is painted"
        out = await execute(ex, call("click_at", f, x=500, y=500))
        assert out.status == "executed", out
        after = ex.latest_frame
        assert out.resolved["frame_after"] == after.phash
        assert near(rgb_at(after, 640, 400), BLUE), (
            f"T-0200 C10: the after-frame is blank: {luma_range(after)}"
        )
        one_note(notes, "action", False)


async def test_token_reload_after_frame(origin: str, shots: list[float]) -> None:
    """C10: a click on a link to a WAF-shaped page (white, stores a token, reloads after 800 ms, then paints): the
    after-frame shows the green block."""
    notes: list[dict[str, Any]] = []
    async with rig(notes) as ex:
        await run(ex.open(f"{origin}/t0200_token_reload.html?start=1"))
        f = await run(ex.frame())
        assert near(rgb_at(f, 640, 400), GREEN) and notes == [], (
            "premise: the start page is painted"
        )
        out = await execute(ex, call("click_at", f, x=500, y=500))
        assert out.status == "executed", out
        after = ex.latest_frame
        assert near(rgb_at(after, 640, 400), GREEN), (
            f"T-0200 C10: the after-frame is blank: {luma_range(after)}"
        )
        one_note(notes, "action", False)


async def test_white_page_frame_bounded(origin: str, shots: list[float]) -> None:
    """C10/C6: a white page gives a white frame within ``BLANK_RECAPTURE_FRAME_S + 0.5`` s, after at most 16
    screenshots, with one ``frame`` note saying it is still blank."""
    notes: list[dict[str, Any]] = []
    async with rig(notes) as ex:
        await run(ex.open(f"{origin}/t0200_white.html"))
        n0 = len(shots)
        t0 = time.monotonic()
        f = await run(ex.frame())
        took = time.monotonic() - t0
        lo, hi = luma_range(f)
        assert lo >= 245 and hi >= 250, (lo, hi)
        assert 2 <= len(shots) - n0 <= 16, len(shots) - n0
        assert took < 1.5 + 0.5, f"T-0200 C10: frame() took {took:.2f} s on a white page"
        one_note(notes, "frame", True)


async def test_normal_page_one_screenshot(
    origin: str, shots: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """C10/C5: a painted page (``color.html``): ``frame()`` and a click's after-frame are one screenshot each, no
    note. T-0204 B10 supersedes "one screenshot" for the real driver's ``click_at``: BASE plus one ``screenshot``
    with a ``clip`` (``screenshot_region``, the pre-click guard's crop); ``frame()`` takes none. T-0204 C10: the click's
    after-frame is two captures on this static page (the visual settle's second shot); ``frame()`` stays one."""
    regions: list[float] = []
    real_region = getattr(runtime_mod.PlaywrightDriver, "screenshot_region", None)
    assert real_region is not None, "premise: T-0204 A's screenshot_region"

    async def region(self: Any, *a: Any, **k: Any) -> bytes:
        regions.append(time.monotonic())
        out: bytes = await real_region(self, *a, **k)
        return out

    monkeypatch.setattr(runtime_mod.PlaywrightDriver, "screenshot_region", region)
    notes: list[dict[str, Any]] = []
    async with rig(notes) as ex:
        await run(ex.open(f"{origin}/color.html"))
        n0 = len(shots)
        f = await run(ex.frame())
        assert near(rgb_at(f, 640, 400), RED) and len(shots) - n0 == 1, len(shots) - n0
        assert regions == [], f"T-0204 B10: frame() takes no region screenshot: {len(regions)}"
        n1 = len(shots)
        out = await execute(ex, call("click_at", f, x=500, y=500))
        # T-0204 C10: BASE plus one more ``screenshot`` per kept action capture (the visual settle's second shot on a
        # static page); frame() is untouched
        assert out.status == "executed" and len(shots) - n1 == 2, (out, len(shots) - n1)
        assert len(regions) == 1, (
            f"T-0204 B10: the click adds exactly one region screenshot: {len(regions)}"
        )
        assert notes == [], notes  # T-0204 C9: a 2-shot settle writes no note
