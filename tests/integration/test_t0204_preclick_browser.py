"""T-0204 group B (browser): the pre-click guard through the real ``BrowserToolExecutor`` and ``PlaywrightDriver`` in
real Chromium on loopback (docs/tasks/T-0204.md, B8 and B9; "Tests to write" B).

**Gate.** As T-0111c's module (its autouse gate imported): ``integration`` + ``browser``, runs only with
``ASTRA_BROWSER_TESTS=1``.

**Fixtures** (``tests/fixtures/t0204``, geometry and colours by hand in its ``MANIFEST.md``), served by this module's
own ``http.server`` on 127.0.0.1. ``tiles.html``: tile 7 at x 40-240, y 250-400 (centre (140, 325)); its click
handler pushes the tile's number to ``window.__clicked``, which only the test reads. ``animated.html``: a spinner 200 px
right of (640, 400) (``?spinner=1``) or a 300x250 noise block repainted every 100 ms under (640, 400) (``?block=1``).
``caret.html``: a text field whose empty caret sits near (346, 400).

**Think time.** Every row waits ``THINK_S`` = 0.25 s between the frame and the call (the model thinks for seconds; a
quarter second lets a layout-shift entry arrive (R1: ~16 ms) and the noise block repaint twice).

**BASE** is the executor at ``BASE_SHA`` (no guard), run from ``git show`` over the current runtime; rows that say
"first shows at BASE" run it first, in its own browser.

**Readings** (reported, not guessed silently):

- B9.1 says "a 300 px block ... at tile 7's coordinates ... The row first shows at BASE that a wrong tile is clicked".
  With ``tiles.html`` a 300 px insert puts the inserted block itself under (140, 325), so BASE clicks no tile (the
  wrong *target*, not a wrong tile); a 150 px insert puts tile 1 there. Both are run: 300 px (BASE: tile 7 not
  clicked) and 150 px (BASE: tile 1 clicked).
- B9.4's far-below insert: at scroll 0 the grid's bottom is off-screen, so an insert there moves nothing visible and
  gives no shift entry. The row scrolls down 600 px first, so the insert moves the visible tail (a real shift entry,
  checked through A's ``shifts_since``) far below the clicked tile 26.
- B9.6 "arming forced to fail": ``PlaywrightDriver.arm_shift_recorder`` is patched at class level to answer ``False``
  (``shifts_since`` then answers ``None``).
- B8 is measured as each guarded click's wall time minus the median of interleaved unguarded clicks
  (``PRECLICK_GUARD = False``) on the same static page, n = 20 each.

Regime: Playwright and Chromium per ``services/versions.lock`` ``[browser]``, headless, 1280x800 at DSF 1, not web
mode, loopback only, one browser per row; n = 1 per B9 row (caret: 3 delays), n = 20 + 20 for B8.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import http.server
import statistics
import sys
import threading
import time
import types
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.computer import runtime as runtime_mod
from local_astra.conductor.tool_executor import to_pixels
from local_astra.contracts.toolcall import FrameGeometry, ToolInvocation, ToolOutcome
from tests import _git_range as gr
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
    _Quiet,
    run,
)

pytestmark = [pytest.mark.integration] + (
    [pytest.mark.browser] if _browser_marker_registered() else []
)

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "t0204"
W, H = 1280, 800
#: group B's BASE (group A's branch head, before any B code). A literal.
BASE_SHA = "e715c33994a8321cc0b9a1b18d8677825ea91379"
CLOCK_MS = 1_700_000_020_404
FRESHNESS = "the screen changed since this frame; take a new frame"
NOTE_WHAT = "live click: page moved check"
NOTE_KEYS = {"what", "result", "shift", "changed_share", "ms"}
THINK_S = 0.25
#: tile centres in document CSS px (tiles.html, by hand): tile n at col (n-1)%6, row (n-1)//6
TILE7 = (140, 325)


def tile_centre(n: int, scroll_y: int = 0) -> tuple[int, int]:
    col, row = (n - 1) % 6, (n - 1) // 6
    return 40 + 200 * col + 100, 100 + 150 * row + 75 - scroll_y


def units_for(px: int, size: int) -> int:
    """The 0-999 unit whose pixel is closest to ``px`` (brute force over ``to_pixels``)."""
    return min(range(1000), key=lambda u: abs(to_pixels(u, size) - px))


# --------------------------------------------------------------------------------------------- rig


@pytest.fixture(scope="module")
def origin() -> Iterator[str]:
    handler = functools.partial(_Quiet, directory=str(FIX))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, name="t0204b-server", daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(5)


_BASE: types.ModuleType | None = None


def base_module() -> types.ModuleType:
    global _BASE
    if _BASE is None:
        gr.require_commits(BASE_SHA)
        src = gr._git("show", f"{BASE_SHA}:local_astra/computer/executor.py").stdout
        assert "screenshot_region" not in src, "premise: BASE has no pre-click guard"
        mod = types.ModuleType("_t0204b_base_executor_browser")
        mod.__file__ = f"<{BASE_SHA[:8]}>/local_astra/computer/executor.py"
        sys.modules[mod.__name__] = mod
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        _BASE = mod
    return _BASE


@contextlib.asynccontextmanager
async def executor(module: Any = executor_mod) -> AsyncIterator[tuple[Any, list[dict[str, Any]]]]:
    ex = module.BrowserToolExecutor(now_ms=lambda: CLOCK_MS, viewport=(W, H), headless=True)
    notes: list[dict[str, Any]] = []
    ex.on_note = notes.append
    try:
        yield ex, notes
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ex.close(), 20)


_N = iter(range(1_000_000))


def call(name: str, frame: Any = None, *, turn: str = "t1", **arguments: Any) -> ToolInvocation:
    g = (
        FrameGeometry(
            phash=frame.phash, source=frame.source, width=frame.width, height=frame.height
        )
        if frame is not None
        else None
    )
    return ToolInvocation(
        call_id=f"p{next(_N)}",
        turn_id=turn,
        name=name,
        arguments=arguments,
        source="draft",
        frame=g,
    )


async def click(ex: Any, frame: Any, point: tuple[int, int], *, turn: str = "t1") -> ToolOutcome:
    out = await run(
        ex.execute(
            call("click_at", frame, turn=turn, x=units_for(point[0], W), y=units_for(point[1], H))
        )
    )
    assert isinstance(out, ToolOutcome)
    return out


async def evaluate(ex: Any, js: str) -> Any:
    return await run(ex.driver.page.evaluate(js))


def guard_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [n for n in notes if n.get("what") == NOTE_WHAT]
    for n in out:
        assert set(n) == NOTE_KEYS, f"T-0204 B6: {n}"
    return out


def moved(out: ToolOutcome) -> None:
    reason = getattr(executor_mod, "MOVED_REASON", None)
    assert reason == FRESHNESS, "T-0204 B3: executor.MOVED_REASON is the session's freshness text"
    assert (out.status, out.reason) == ("refused", FRESHNESS), out


async def opened(ex: Any, url: str) -> Any:
    await run(ex.open(url))
    return await run(ex.frame())


# --------------------------------------------------------------------------------------------- B9.1-B9.3


@pytest.mark.parametrize(("px", "base_clicked"), [(300, []), (150, [1])], ids=["300px", "150px"])
async def test_b9_1_lazy_insert_above_refused(
    origin: str, px: int, base_clicked: list[int]
) -> None:
    """B9.1: the page inserts a block above the grid after the frame was captured; a click at tile 7's coordinates in
    that frame is refused with ``MOVED_REASON`` and no tile records a click. First at BASE: the click lands elsewhere
    (the inserted block at 300 px, tile 1 at 150 px; module docstring)."""
    async with executor(base_module()) as (ex, _):
        f = await opened(ex, f"{origin}/tiles.html")
        await evaluate(ex, f"window.t0204.insertAbove({px})")
        await asyncio.sleep(THINK_S)
        out = await click(ex, f, TILE7)
        assert out.status == "executed", out
        got = await evaluate(ex, "window.__clicked")
        assert got == base_clicked and 7 not in got, f"premise: BASE misclicks: {got}"
    async with executor() as (ex, notes):
        f = await opened(ex, f"{origin}/tiles.html")
        await evaluate(ex, f"window.t0204.insertAbove({px})")
        await asyncio.sleep(THINK_S)
        out = await click(ex, f, TILE7)
        moved(out)
        assert await evaluate(ex, "window.__clicked") == [], "B3: nothing was clicked"
        assert ex.latest_frame is f, "B3: latest_frame unchanged"
        assert [n["result"] for n in guard_notes(notes)] == ["refused"], notes


async def test_b9_2_transform_move_refused_by_pixels(origin: str) -> None:
    """B9.2: the grid moves 150 px down by a CSS ``transform`` (no layout-shift entry): refused by pixels, the note
    says ``shift`` ``False`` and a changed share of at least 0.10. At BASE the click lands on tile 1."""
    async with executor(base_module()) as (ex, _):
        f = await opened(ex, f"{origin}/tiles.html")
        await evaluate(ex, "window.t0204.transformGrid(150)")
        await asyncio.sleep(THINK_S)
        assert (await click(ex, f, TILE7)).status == "executed"
        assert await evaluate(ex, "window.__clicked") == [1], "premise: BASE clicks tile 1"
    async with executor() as (ex, notes):
        f = await opened(ex, f"{origin}/tiles.html")
        await evaluate(ex, "window.t0204.transformGrid(150)")
        await asyncio.sleep(THINK_S)
        moved(await click(ex, f, TILE7))
        assert await evaluate(ex, "window.__clicked") == []
        (n,) = guard_notes(notes)
        assert n["result"] == "refused" and n["shift"] is False, n
        assert isinstance(n["changed_share"], float) and n["changed_share"] >= 0.10, n


async def test_b9_3_static_page_executes(origin: str) -> None:
    """B9.3: a static page: executed, tile 7 recorded, no note."""
    async with executor() as (ex, notes):
        f = await opened(ex, f"{origin}/tiles.html")
        await asyncio.sleep(THINK_S)
        out = await click(ex, f, TILE7)
        assert out.status == "executed", out
        assert await evaluate(ex, "window.__clicked") == [7]
        assert guard_notes(notes) == [], notes
        assert getattr(executor_mod, "PRECLICK_GUARD", None) is True, "T-0204 B: the guard is on"


# --------------------------------------------------------------------------------------------- B9.4 false positives


@pytest.mark.parametrize("query", ["", "?drawn=1"], ids=["chromium-caret", "drawn-caret"])
async def test_b9_4_caret_blink_not_refused(origin: str, query: str) -> None:
    """B9.4 (adversarial): a focused input with a blinking caret at the click point; ``type_at`` after three think
    times (spanning the 0.53 s blink phase) is executed each time with no guard note. Chromium's own caret is hidden
    by Playwright's screenshots (``caret="hide"``; MANIFEST reading), so ``?drawn=1`` adds a page-drawn caret that
    does blink in the captures (premise checked: region captures of it differ)."""
    async with executor() as (ex, notes):
        await run(ex.open(f"{origin}/caret.html{query}"))
        await evaluate(ex, "document.getElementById('q').focus()")
        if query:
            seen = set()
            for _ in range(8):
                seen.add(await run(ex.driver.screenshot_region(336, 384, 24, 32)))
                await asyncio.sleep(0.15)
            assert len(seen) >= 2, "premise: the drawn caret blinks in the captures"
        point = (346, 400)
        for delay in (0.27, 0.53, 0.8):
            f = await run(ex.frame())
            await asyncio.sleep(delay)
            out = await run(
                ex.execute(
                    call(
                        "type_at",
                        f,
                        x=units_for(point[0], W),
                        y=units_for(point[1], H),
                        text="v",
                        press_enter=False,
                    )
                )
            )
            assert out.status == "executed", (delay, out)
        assert guard_notes(notes) == [], notes
        assert getattr(executor_mod, "PRECLICK_GUARD", None) is True, "T-0204 B: the guard is on"


async def test_b9_4_spinner_far_away_not_refused(origin: str) -> None:
    """B9.4 (adversarial): a spinner animating 200 px right of the point: executed, no note."""
    assert getattr(executor_mod, "PRECLICK_GUARD", None) is True, "T-0204 B: the guard is on"
    async with executor() as (ex, notes):
        f = await opened(ex, f"{origin}/animated.html?spinner=1")
        await asyncio.sleep(THINK_S)
        out = await click(ex, f, (640, 400))
        assert out.status == "executed", out
        assert await evaluate(ex, "window.__clicks") == 1
        assert guard_notes(notes) == [], notes


async def test_b9_4_insert_far_below_not_refused(origin: str) -> None:
    """B9.4 (adversarial): after a 600 px scroll, a 100 px block is inserted below the grid (viewport y 400 down);
    the tail moves (a shift entry, premise read through A's ``shifts_since``) far below tile 26 (viewport centre
    (340, 175)): executed, tile 26 recorded, no note."""
    async with executor() as (ex, notes):
        await opened(ex, f"{origin}/tiles.html")
        out = await run(ex.execute(call("scroll", None, direction="down")))
        assert out.status == "executed" and out.resolved.get("dy_px") == 600, out
        f = ex.latest_frame
        assert await run(ex.driver.arm_shift_recorder()), "premise: the recorder arms"
        mark = ex.driver.shift_mark()
        await evaluate(ex, "window.t0204.insertBelowGrid(100)")
        await asyncio.sleep(THINK_S)
        rects = ex.driver.shifts_since(mark)
        assert rects and all(r[1] >= 300 for r in rects if r[2] * r[3] > 0), (
            f"premise: a shift entry far below the point: {rects}"
        )
        assert getattr(executor_mod, "PRECLICK_GUARD", None) is True, "T-0204 B: the guard is on"
        out = await click(ex, f, tile_centre(26, 600))
        assert out.status == "executed", out
        assert await evaluate(ex, "window.__clicked") == [26]
        assert guard_notes(notes) == [], notes


# --------------------------------------------------------------------------------------------- B9.5 the cap


async def test_b9_5_animated_region_cap(origin: str) -> None:
    """B9.5: the noise block under the point changes every 100 ms; each call is made on a fresh frame after the think
    time (as the session does after a refusal). Refused twice, then the third call is executed with the
    ``allowed_after_cap`` note."""
    async with executor() as (ex, notes):
        await run(ex.open(f"{origin}/animated.html?block=1"))
        outs = []
        for _ in range(3):
            f = await run(ex.frame())
            await asyncio.sleep(THINK_S)
            outs.append(await click(ex, f, (640, 400)))
        moved(outs[0])
        moved(outs[1])
        assert outs[2].status == "executed", outs[2]
        assert await evaluate(ex, "window.__clicks") == 1
        assert [n["result"] for n in guard_notes(notes)] == [
            "refused",
            "refused",
            "allowed_after_cap",
        ], notes


# --------------------------------------------------------------------------------------------- B9.6 CDP unavailable


async def test_b9_6_cdp_unavailable_pixels_decide(
    origin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B9.6: arming forced to fail: B9.1 (300 px) is still refused, by pixels (note ``shift`` ``None``), and B9.3 is
    still executed on tile 7."""
    assert getattr(runtime_mod.PlaywrightDriver, "arm_shift_recorder", None) is not None

    async def no_arm(self: Any) -> bool:
        return False

    monkeypatch.setattr(runtime_mod.PlaywrightDriver, "arm_shift_recorder", no_arm)
    async with executor() as (ex, notes):
        f = await opened(ex, f"{origin}/tiles.html")
        await evaluate(ex, "window.t0204.insertAbove(300)")
        await asyncio.sleep(THINK_S)
        moved(await click(ex, f, TILE7))
        (n,) = guard_notes(notes)
        assert n["shift"] is None and n["changed_share"] >= 0.10, n
    async with executor() as (ex, notes):
        f = await opened(ex, f"{origin}/tiles.html")
        await asyncio.sleep(THINK_S)
        assert (await click(ex, f, TILE7)).status == "executed"
        assert await evaluate(ex, "window.__clicked") == [7]
        assert guard_notes(notes) == [], notes


# --------------------------------------------------------------------------------------------- B7 / B8


async def test_guard_off_real_driver_base_calls(
    origin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B7 (negative): with ``PRECLICK_GUARD = False`` the real driver's guard methods are never called and the
    Playwright ``screenshot`` calls of ``frame()`` + a click are BASE's (none with a ``clip``)."""
    assert getattr(executor_mod, "PRECLICK_GUARD", None) is True, "T-0204 B: PRECLICK_GUARD"
    from playwright.async_api import Page

    seen: list[str] = []
    real_shot = Page.screenshot

    async def shot(self: Any, *a: Any, **k: Any) -> Any:
        seen.append("clip" if "clip" in k else "screenshot")
        return await real_shot(self, *a, **k)

    monkeypatch.setattr(Page, "screenshot", shot)
    for name in ("arm_shift_recorder", "screenshot_region", "shift_mark", "shifts_since"):
        real = getattr(runtime_mod.PlaywrightDriver, name)
        if asyncio.iscoroutinefunction(real):

            async def wrapped(self: Any, *a: Any, _r: Any = real, _n: str = name, **k: Any) -> Any:
                seen.append(_n)
                return await _r(self, *a, **k)

        else:

            def wrapped(self: Any, *a: Any, _r: Any = real, _n: str = name, **k: Any) -> Any:  # type: ignore[misc]
                seen.append(_n)
                return _r(self, *a, **k)

        monkeypatch.setattr(runtime_mod.PlaywrightDriver, name, wrapped)

    async def trace(module: Any) -> list[str]:
        seen.clear()
        async with executor(module) as (ex, _):
            f = await opened(ex, f"{origin}/tiles.html")
            assert (await click(ex, f, TILE7)).status == "executed"
        return list(seen)

    monkeypatch.setattr(executor_mod, "PRECLICK_GUARD", False)
    # T-0204 C10: the visual settle (group C) adds a capture per action on a capability driver; this row pins the
    # guard switch alone, so the settle is off too (C10: STABLE_SCREEN = False is BASE)
    monkeypatch.setattr(executor_mod, "STABLE_SCREEN", False, raising=False)
    tip = await trace(executor_mod)
    base = await trace(base_module())
    assert tip == base == ["screenshot", "screenshot"], f"B7: TIP {tip} vs BASE {base}"


async def test_b8_guard_cost_n20(origin: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """B8 (and R3 at n = 20): on a static ``tiles.html`` the guard's added wall time per click (a guarded click minus
    the median unguarded click, interleaved, each on a fresh frame) has a median <= 60 ms (B-R3) and a p90 <= 100 ms; no
    click is refused or noted."""
    assert getattr(executor_mod, "PRECLICK_GUARD", None) is True, "T-0204 B: PRECLICK_GUARD"
    on: list[float] = []
    off: list[float] = []
    async with executor() as (ex, notes):
        await run(ex.open(f"{origin}/tiles.html"))
        for i in range(43):
            guarded = i % 2 == 0
            monkeypatch.setattr(executor_mod, "PRECLICK_GUARD", guarded)
            f = await run(ex.frame())
            t0 = time.perf_counter()
            out = await click(ex, f, (640, 50), turn=f"t{i}")
            took = time.perf_counter() - t0
            assert out.status == "executed", out
            if i >= 3:  # three warm-ups
                (on if guarded else off).append(took)
        assert guard_notes(notes) == [], notes
    added = sorted(t - statistics.median(off) for t in on[:20])
    med, p90 = statistics.median(added), added[int(0.9 * len(added)) - 1]
    assert med <= 0.060 and p90 <= 0.100, (
        f"B8 (bar per ruling B-R3: median <= 60 ms, p90 <= 100 ms): added median {med * 1000:.1f} ms, "
        f"p90 {p90 * 1000:.1f} ms (n = {len(added)})"
    )


# --------------------------------------------------------------------------------------------- B-R4


REFLOWS = {
    # six tiles far right of tile 7 (x 840-1240) gain 6 px of top padding: six sources, read as the whole viewport
    # (A-R1); probe 2026-10-30: ((0, 0, 1280, 800),), the phash unchanged
    "far-six-sources": (
        "[5, 6, 11, 12, 17, 18].forEach(n => "
        "document.querySelector('[data-n=\"' + n + '\"]').style.paddingTop = '6px')"
    ),
    # tile 7's number reflows 6 px down: a shift at the point; probe: ((40, 298, 200, 53), (40, 304, 200, 53))
    "tile7-6px": "document.querySelector('[data-n=\"7\"]').style.paddingTop = '6px'",
}


@pytest.mark.parametrize("reflow", sorted(REFLOWS))
async def test_b_r4_same_phash_recapture_after_seen_change(
    origin: str, reflow: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ruling B-R4 (adversarial: a false refusal). After the frame, the page reflows (``REFLOWS``) without changing
    the phash. A click on tile 7 carrying the pre-change frame is refused. The fresh frame the session would show next
    has the same phash and replaces the reference; a click carrying it executes and tile 7 records it.

    T-0234 B2 (named re-pin): both reflows are shift-only at the crop (probe at 0516a4f8: ``changed_share`` 0.0 and
    0.051), so with a crop the first click now executes (``test_t0234_reflow_shift_alone_executes``). Here the
    region screenshot is made to fail for the first click (class level, as B9.6 patches arming), so the shift decides
    by BASE's rule; it is restored before the recapture."""
    real_region = runtime_mod.PlaywrightDriver.screenshot_region

    async def no_region(self: Any, x: float, y: float, w: float, h: float) -> bytes:
        raise RuntimeError("T-0234 B2: no crop")

    async with executor() as (ex, notes):
        f = await opened(ex, f"{origin}/tiles.html")
        await evaluate(ex, REFLOWS[reflow])
        await asyncio.sleep(THINK_S)
        monkeypatch.setattr(runtime_mod.PlaywrightDriver, "screenshot_region", no_region)
        moved(await click(ex, f, TILE7))
        monkeypatch.setattr(runtime_mod.PlaywrightDriver, "screenshot_region", real_region)
        assert await evaluate(ex, "window.__clicked") == [], "B3: the refused click clicked nothing"
        g = await run(ex.frame())
        assert g.phash == f.phash, "premise: the reflow leaves the phash alone"
        await asyncio.sleep(THINK_S)
        out = await click(ex, g, TILE7)
        assert out.status == "executed", (
            f"B-R4: the newest same-phash capture is the reference: {out}"
        )
        assert await evaluate(ex, "window.__clicked") == [7]
        assert [n["result"] for n in guard_notes(notes)] == ["refused"], notes


#: T-0234 R-2 verdicts: far-six-sources leaves the crop untouched (share 0.0, probe at 0516a4f8); tile7-6px pads tile 7
#: itself, so blocks of the point's 3x3 change (whole-crop share 0.051 < 0.10, but R-2's 3x3 rule moves it)
REFLOW_VERDICT = {"far-six-sources": "executed", "tile7-6px": "refused"}


@pytest.mark.parametrize("reflow", sorted(REFLOWS))
async def test_t0234_reflow_shift_alone_executes(origin: str, reflow: str) -> None:
    """T-0234 B1 as amended by R-2. Both B-R4 reflows report a layout shift near tile 7 and the crop is taken.
    ``far-six-sources``: nothing in the crop changes (share 0.0): pixels decide, the click carrying the pre-change
    frame executes, tile 7 records it, and the note is ``executed_despite_shift``. ``tile7-6px`` (re-pinned under
    R-2): tile 7's own number moves 6 px, so blocks of the point's 3x3 change while the whole-crop share stays under
    10 % (0.051); with a shift near the point that is moved: refused with the moved reason, nothing clicked, one
    ``refused`` note with ``shift`` ``True``."""
    async with executor() as (ex, notes):
        f = await opened(ex, f"{origin}/tiles.html")
        await evaluate(ex, REFLOWS[reflow])
        await asyncio.sleep(THINK_S)
        out = await click(ex, f, TILE7)
        got = [n for n in notes if n.get("what") == NOTE_WHAT]
        if REFLOW_VERDICT[reflow] == "executed":
            assert out.status == "executed", (
                f"T-0234 B1: a shift alone with an unchanged crop executes: {out}"
            )
            assert await evaluate(ex, "window.__clicked") == [7]
            assert [(n["result"], n["shift"]) for n in got] == [("executed_despite_shift", True)], (
                got
            )
        else:
            moved(out)
            assert await evaluate(ex, "window.__clicked") == [], (
                "B3: the refused click clicked nothing"
            )
            assert [(n["result"], n["shift"]) for n in got] == [("refused", True)], got
            assert got[0]["changed_share"] < 0.10, (
                f"R-2: refused by the 3x3 rule, not the share: {got}"
            )
