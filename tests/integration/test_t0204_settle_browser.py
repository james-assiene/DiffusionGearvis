"""T-0204 group C (browser): the visual settle before an action's after-frame, through the real
``BrowserToolExecutor`` and ``PlaywrightDriver`` in real Chromium on loopback (docs/tasks/T-0204.md, C3-C7; "Tests to
write" C).

**Gate.** As T-0111c's module (its autouse gate imported): ``integration`` + ``browser``, runs only with
``ASTRA_BROWSER_TESTS=1``.

**Fixtures** (``tests/fixtures/t0204``, geometry and colours by hand in its ``MANIFEST.md``), served by this module's
own ``http.server`` on 127.0.0.1: ``tiles.html`` (static), ``lazy_scroll.html`` (C4: five 120 px inserts at the
viewport top, every 80 ms after the first scroll; final colour at (640, 60) rgb(150, 50, 170)), ``animated.html?
steps=1`` (C5: a 300x250 block stepping its grey every 100 ms) and ``noise.html`` (C6: full-viewport noise every
50 ms). C7 uses T-0203's ``search.html?js=120`` (a ``location.assign`` 120 ms after Enter) over T-0203's
``DelayHandler`` server (``tests/fixtures/t0203``; results page rgb(230, 120, 20), search page rgb(30, 60, 200)).

**Counting shots.** ``playwright.async_api.Page.screenshot`` is wrapped at class level: a call with a ``clip`` is
the pre-click guard's crop (B), one without is a capture.

**BASE premise.** C's switch is read at call time, so "BASE" rows run this executor with ``STABLE_SCREEN = False``
(the C10 negative: that is BASE's capture). Rows that state a premise run it first.

**Readings** (reported, not guessed silently):

- C3's "added wall time" is a settled click minus the median of interleaved unsettled clicks (``STABLE_SCREEN =
  False``) on the same static page, each on a fresh frame, as B8.
- C4/C5/C6 "wall time" is the executed call's wall time (``execute()`` start to return).
- C5 uses ``?steps=1``, not B's ``?block=1``: B's per-pixel noise averages out in a 20x20 cell, so between two shots a
  random ~1/4 of its cells differ by more than 6 and a different random subset each pair; C2's animation rule (a cell
  that also differed in the previous pair) does not cover that, and the settle would run to the cap.
  ``test_reading_per_pixel_noise_block`` records what happens (not gating).

Regime: Playwright and Chromium per ``services/versions.lock`` ``[browser]``, headless, 1280x800 at DSF 1, not web
mode, loopback only; n = 20 + 20 for C3, n = 5 for C4, C5 and C7 (each run its own browser), n = 1 for C6.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import http.server
import io
import statistics
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.conductor.tool_executor import to_pixels
from local_astra.contracts.toolcall import FrameGeometry, ToolInvocation, ToolOutcome
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
    _Quiet,
    run,
)
from tests.integration.test_t0203_follow_driver_browser import DelayHandler, release_all

pytestmark = [pytest.mark.integration] + (
    [pytest.mark.browser] if _browser_marker_registered() else []
)

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "t0204"
FIX0203 = REPO / "tests" / "fixtures" / "t0203"
W, H = 1280, 800
CLOCK_MS = 1_700_000_020_404
NOTE_WHAT = "live frame: waited for a stable screen"
NOTE_KEYS = {"what", "path", "shots", "stable", "waited_ms"}
STABLE_CAP_S = 1.0
#: lazy_scroll.html's final colour at the viewport top after the 600 px scroll, and the colour BASE shows there
FINAL_RGB = (150, 50, 170)
SEARCH_RGB = (30, 60, 200)
RESULTS_RGB = (230, 120, 20)
#: the search box centre on search.html
BOX = (640, 130)
TOL = 24


def units_for(px: int, size: int) -> int:
    """The 0-999 unit whose pixel is closest to ``px`` (brute force over ``to_pixels``)."""
    return min(range(1000), key=lambda u: abs(to_pixels(u, size) - px))


# --------------------------------------------------------------------------------------------- rig


def _serve(handler: Any, name: str) -> Iterator[str]:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, name=name, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(5)


@pytest.fixture(scope="module")
def origin() -> Iterator[str]:
    yield from _serve(functools.partial(_Quiet, directory=str(FIX)), "t0204c-server")


@pytest.fixture(scope="module")
def server0203() -> Iterator[str]:
    try:
        yield from _serve(functools.partial(DelayHandler, directory=str(FIX0203)), "t0204c-0203")
    finally:
        release_all()


@pytest.fixture
def shots(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, str]]:
    """Every Playwright ``Page.screenshot`` call: (monotonic start, ``"clip"`` or ``"capture"``)."""
    from playwright.async_api import Page

    seen: list[tuple[float, str]] = []
    real = Page.screenshot

    async def shot(self: Any, *a: Any, **k: Any) -> Any:
        seen.append((time.monotonic(), "clip" if "clip" in k else "capture"))
        return await real(self, *a, **k)

    monkeypatch.setattr(Page, "screenshot", shot)
    return seen


def captures(seen: list[tuple[float, str]], since: int) -> int:
    return sum(1 for _, kind in seen[since:] if kind == "capture")


@contextlib.asynccontextmanager
async def executor() -> AsyncIterator[tuple[Any, list[dict[str, Any]]]]:
    ex = executor_mod.BrowserToolExecutor(now_ms=lambda: CLOCK_MS, viewport=(W, H), headless=True)
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
        call_id=f"s{next(_N)}",
        turn_id=turn,
        name=name,
        arguments=arguments,
        source="draft",
        frame=g,
    )


async def timed(ex: Any, inv: ToolInvocation) -> tuple[ToolOutcome, float]:
    t0 = time.perf_counter()
    out = await run(ex.execute(inv))
    assert isinstance(out, ToolOutcome)
    return out, time.perf_counter() - t0


def pixel(jpeg: bytes, point: tuple[int, int]) -> tuple[int, int, int]:
    from PIL import Image

    px: Any = Image.open(io.BytesIO(jpeg)).convert("RGB").getpixel(point)
    return int(px[0]), int(px[1]), int(px[2])


def near(rgb: tuple[int, int, int], want: tuple[int, int, int]) -> bool:
    return all(abs(a - b) <= TOL for a, b in zip(rgb, want, strict=True))


def settle_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [n for n in notes if n.get("what") == NOTE_WHAT]
    for n in out:
        assert set(n) == NOTE_KEYS and n["path"] == "action", f"T-0204 C9: {n}"
        assert (
            type(n["shots"]) is int and type(n["stable"]) is bool and type(n["waited_ms"]) is int
        ), n
    return out


def switch(monkeypatch: pytest.MonkeyPatch, on: bool) -> None:
    assert not on or hasattr(executor_mod, "STABLE_SCREEN"), (
        "T-0204 C: executor.STABLE_SCREEN does not exist yet"
    )
    monkeypatch.setattr(executor_mod, "STABLE_SCREEN", on, raising=False)


# --------------------------------------------------------------------------------------------- C3


async def test_c3_static_page_two_shots_and_cost_n20(
    origin: str, shots: list[tuple[float, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """C3 (ruling a; R4 static): on a static ``tiles.html`` every executed click takes exactly 2 captures (and one
    guard crop), no settle note; the added wall time (settled minus the median unsettled click, interleaved, n = 20
    each after 3 warm-ups) has a median <= 0.20 s and a p90 <= 0.25 s. ``press_key`` and ``wait`` also take 2."""
    assert getattr(executor_mod, "STABLE_SCREEN", None) is True, "T-0204 C: STABLE_SCREEN"
    monkeypatch.setattr(executor_mod, "WAIT_S", 0.05)
    on: list[float] = []
    off: list[float] = []
    async with executor() as (ex, notes):
        await run(ex.open(f"{origin}/tiles.html"))
        for i in range(43):
            settled = i % 2 == 0
            switch(monkeypatch, settled)
            f = await run(ex.frame())
            n0 = len(shots)
            out, took = await timed(
                ex, call("click_at", f, turn=f"t{i}", x=units_for(640, W), y=units_for(50, H))
            )
            assert out.status == "executed", out
            got = captures(shots, n0)
            assert got == (2 if settled else 1), (
                f"T-0204 C3: a static page's click takes {'2' if settled else '1 (off)'} captures, got {got}"
            )
            if i >= 3:
                (on if settled else off).append(took)
        assert settle_notes(notes) == [], notes
        switch(monkeypatch, True)
        for inv in (call("press_key", combo="Escape"), call("wait")):
            n0 = len(shots)
            out, _ = await timed(ex, inv)
            assert out.status == "executed" and captures(shots, n0) == 2, (out, captures(shots, n0))
    added = sorted(t - statistics.median(off) for t in on[:20])
    med, p90 = statistics.median(added), added[int(0.9 * len(added)) - 1]
    print(
        f"R4 static: added median {med * 1000:.1f} ms, p90 {p90 * 1000:.1f} ms (n = {len(added)})"
    )
    assert med <= 0.20 and p90 <= 0.25, (
        f"T-0204 C3: added median {med * 1000:.1f} ms (<= 200), p90 {p90 * 1000:.1f} ms (<= 250), n = {len(added)}"
    )


# --------------------------------------------------------------------------------------------- C4


async def _lazy_scroll(
    origin: str, shots: list[tuple[float, str]], n: int = 5
) -> tuple[Any, list[Any], int, float, Any]:
    async with executor() as (ex, notes):
        await run(ex.open(f"{origin}/lazy_scroll.html?n={n}"))
        await run(ex.frame())
        n0 = len(shots)
        out, took = await timed(ex, call("scroll", direction="down"))
        assert out.status == "executed", out
        rgb = pixel(ex.latest_frame.jpeg, (640, 60))
        inserted = await run(ex.driver.page.evaluate("window.__inserted"))
        await asyncio.sleep(0.08 * n + 0.2)
        done = await run(ex.driver.page.evaluate("window.__inserted"))
        assert done == n, f"premise: the scroll started the {n} inserts ({done})"
        return rgb, list(notes), captures(shots, n0), took, inserted


async def test_c4_premise_settle_off_shows_a_mid_change_frame(
    origin: str, shots: list[tuple[float, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """C4 premise (settle off = BASE's capture): the scroll's after-frame is one capture taken while the inserts are
    still arriving, so (640, 60) is not the final layout's colour."""
    switch(monkeypatch, False)
    rgb, _, n, _, _ = await _lazy_scroll(origin, shots)
    assert n == 1, n
    assert not near(rgb, FINAL_RGB), (
        f"premise: BASE's after-frame already shows the final layout {rgb}"
    )


@pytest.mark.parametrize("trial", range(5))
async def test_c4_lazy_insert_waited_out(
    origin: str, shots: list[tuple[float, str]], trial: int
) -> None:
    """C4 (n = 5): after a scroll the page inserts a 120 px block at the viewport top every 80 ms for 400 ms: the
    after-frame shows the final layout's colour at (640, 60), the settle took >= 3 captures (its note says so), and
    the call took <= ``STABLE_CAP_S`` + 0.4 s (ruling C-R4: under C-R1 the inserts hold the 2-3 pair, so the settle runs
    to its 4th shot near the cap; ``STABLE_CAP_S`` read at call time)."""
    assert getattr(executor_mod, "STABLE_SCREEN", None) is True, "T-0204 C: STABLE_SCREEN"
    rgb, notes, n, took, inserted = await _lazy_scroll(origin, shots)
    assert near(rgb, FINAL_RGB), (
        f"T-0204 C4: the after-frame is not the final layout: {rgb} at (640, 60), want {FINAL_RGB} "
        f"({inserted} inserts at read time, {n} captures)"
    )
    assert n >= 3, f"T-0204 C4: the settle took {n} captures"
    (note,) = settle_notes(notes)
    assert note["shots"] >= 3 and note["shots"] == n, (note, n)
    bound = executor_mod.STABLE_CAP_S + 0.4
    assert took <= bound, (
        f"T-0204 C4 (C-R4): the scroll took {took:.3f} s, bound STABLE_CAP_S + 0.4 = {bound:.2f} s"
    )


async def test_c_r1_long_lazy_insert_run_waited_out(
    origin: str, shots: list[tuple[float, str]]
) -> None:
    """Ruling C-R1 (adversarial: the spec gap): ``lazy_scroll.html?n=10`` inserts for 800 ms, past the 3rd shot. The
    inserts change the same cells every pair (so the cell rule alone calls the 3rd shot stable), but each is a layout
    shift, so the settle keeps going: the after-frame shows the final layout's colour, within ``STABLE_CAP_S`` +
    0.3 s."""
    assert getattr(executor_mod, "STABLE_SCREEN", None) is True, "T-0204 C: STABLE_SCREEN"
    rgb, notes, n, took, inserted = await _lazy_scroll(origin, shots, n=10)
    assert near(rgb, FINAL_RGB), (
        f"T-0204 C-R1: the after-frame is mid-shift: {rgb} at (640, 60), want {FINAL_RGB} "
        f"({inserted} inserts at read time, {n} captures, notes {settle_notes(notes)})"
    )
    assert n >= 4, f"T-0204 C-R1: the shifts held the settle past the 3rd shot: {n} captures"
    assert took <= STABLE_CAP_S + 0.3 + 0.1, f"T-0204 C-R1: the scroll took {took:.3f} s"


@pytest.mark.parametrize("trial", range(5))
async def test_c_r1_transform_carousel_settles(
    origin: str, shots: list[tuple[float, str]], trial: int
) -> None:
    """Ruling C-R1 (n = 5; the shift check is not too wide): a carousel moved by CSS ``transform`` every 100 ms gives no
    layout-shift entry, so it is ignored as animation: stable within 4 captures, in <= 0.8 s."""
    assert getattr(executor_mod, "STABLE_SCREEN", None) is True, "T-0204 C: STABLE_SCREEN"
    async with executor() as (ex, notes):
        await run(ex.open(f"{origin}/animated.html?carousel=1"))
        await run(ex.frame())
        n0 = len(shots)
        out, took = await timed(ex, call("press_key", combo="Escape"))
        assert out.status == "executed", out
        n = captures(shots, n0)
        assert 2 <= n <= 4, f"T-0204 C-R1: {n} captures on a transform carousel"
        assert all(x["stable"] is True for x in settle_notes(notes)), notes
        assert took <= 0.8, f"T-0204 C-R1: {took:.3f} s"


# --------------------------------------------------------------------------------------------- C5


@pytest.mark.parametrize("trial", range(5))
async def test_c5_animation_does_not_hold(
    origin: str, shots: list[tuple[float, str]], trial: int
) -> None:
    """C5 (n = 5): a page static but for a 300x250 block changing every 100 ms: an executed ``press_key`` is stable
    within 4 captures (a note, when there is one, says ``stable: true``) in <= 0.8 s."""
    assert getattr(executor_mod, "STABLE_SCREEN", None) is True, "T-0204 C: STABLE_SCREEN"
    async with executor() as (ex, notes):
        await run(ex.open(f"{origin}/animated.html?steps=1"))
        await run(ex.frame())
        n0 = len(shots)
        out, took = await timed(ex, call("press_key", combo="Escape"))
        assert out.status == "executed", out
        n = captures(shots, n0)
        assert 2 <= n <= 4, (
            f"T-0204 C5: {n} captures (a settle runs, the animation does not hold it)"
        )
        assert all(x["stable"] is True for x in settle_notes(notes)), notes
        assert took <= 0.8, f"T-0204 C5: {took:.3f} s"


async def test_reading_per_pixel_noise_block(origin: str, shots: list[tuple[float, str]]) -> None:
    """Reading (not gating; module docstring): B's ``animated.html?block=1`` (per-pixel noise in the 300x250 block).
    Prints the captures, the note and the wall time; asserts only that the action executed."""
    assert getattr(executor_mod, "STABLE_SCREEN", None) is True, "T-0204 C: STABLE_SCREEN"
    async with executor() as (ex, notes):
        await run(ex.open(f"{origin}/animated.html?block=1"))
        await run(ex.frame())
        n0 = len(shots)
        out, took = await timed(ex, call("press_key", combo="Escape"))
        assert out.status == "executed", out
        print(
            f"reading block=1: {captures(shots, n0)} captures, {took:.3f} s, notes {settle_notes(notes)}"
        )


# --------------------------------------------------------------------------------------------- C6


@pytest.mark.parametrize("timeout", [None, 1.5], ids=["default", "action-timeout-1.5"])
async def test_c6_never_stable_bounded(
    origin: str,
    shots: list[tuple[float, str]],
    monkeypatch: pytest.MonkeyPatch,
    timeout: float | None,
) -> None:
    """C6 (adversarial): a full-viewport noise canvas gives ``executed`` within ``STABLE_CAP_S`` + 0.3 s and one note
    ``stable: false``; with ``ACTION_TIMEOUT_S`` patched to 1.5 s it still ends ``executed`` (within 1.5 s)."""
    assert getattr(executor_mod, "STABLE_SCREEN", None) is True, "T-0204 C: STABLE_SCREEN"
    if timeout is not None:
        monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", timeout)
    async with executor() as (ex, notes):
        await run(ex.open(f"{origin}/noise.html"))
        await run(ex.frame())
        n0 = len(shots)
        out, took = await timed(ex, call("press_key", combo="Escape"))
        assert out.status == "executed", out
        n = captures(shots, n0)
        print(f"C6 {timeout}: {n} captures, {took:.3f} s, {settle_notes(notes)}")
        assert n >= 2, f"T-0204 C6: the settle ran: {n} captures"
        bound = STABLE_CAP_S + 0.3 if timeout is None else timeout
        assert took <= bound, f"T-0204 C6: {took:.3f} s on a never-stable page (bound {bound} s)"
        (note,) = settle_notes(notes)
        assert note["stable"] is False and note["shots"] == n, (note, n)


# --------------------------------------------------------------------------------------------- C7


async def _js_submit(server: str) -> tuple[Any, tuple[int, int, int], str]:
    async with executor() as (ex, _):
        await run(ex.open(f"{server}/search.html?js=120&delay=0"))
        f = await run(ex.frame())
        assert near(pixel(f.jpeg, (100, 400)), SEARCH_RGB), "premise: the search page"
        out, _ = await timed(
            ex,
            call(
                "type_at",
                f,
                x=units_for(BOX[0], W),
                y=units_for(BOX[1], H),
                text="kettle",
                press_enter=True,
            ),
        )
        assert out.status == "executed", out
        return (
            out.resolved.get("navigated"),
            pixel(ex.latest_frame.jpeg, (100, 400)),
            ex.driver.page.url,
        )


async def test_c7_premise_settle_off_misses_120ms_js_submit(
    server0203: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C7 premise (n = 5, settle off: BASE with T-0203 B): Enter, then a ``location.assign`` 120 ms later; T-0203 B2's
    late read comes too early, so the after-frame is the search page and ``navigated`` is not ``True``."""
    switch(monkeypatch, False)
    missed = 0
    for _ in range(5):
        navigated, rgb, _ = await _js_submit(server0203)
        missed += navigated is not True and near(rgb, SEARCH_RGB)
    print(f"C7 premise: settle off missed {missed}/5")
    assert missed == 5, f"premise (T-0203 R1: missed 5/5): settle off missed only {missed}/5"


@pytest.mark.parametrize("trial", range(5))
async def test_c7_js_submit_120ms_now_caught(server0203: str, trial: int) -> None:
    """C7 (adversarial, T-0203 R1's miss; n = 5): with the settle, the late read comes after the 120 ms JS submit
    started: the after-frame shows the results page and ``navigated`` is ``True``."""
    assert getattr(executor_mod, "STABLE_SCREEN", None) is True, "T-0204 C: STABLE_SCREEN"
    navigated, rgb, url = await _js_submit(server0203)
    assert navigated is True, f"T-0204 C7: navigated = {navigated!r} ({url})"
    assert near(rgb, RESULTS_RGB), (
        f"T-0204 C7: the after-frame is not the results page: {rgb} ({url})"
    )
