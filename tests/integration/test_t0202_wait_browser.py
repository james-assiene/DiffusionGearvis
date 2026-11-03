"""T-0202 B10 in real Chromium on loopback: ``wait`` through the real ``PlaywrightDriver`` and ``BrowserToolExecutor``
(docs/tasks/T-0202.md).

**Gate.** ``integration`` + ``browser``; runs only with ``ASTRA_BROWSER_TESTS=1`` (T-0111c's autouse gate, imported).
``ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0202_wait_browser.py -q -m browser``.

**Fixtures** (MANIFEST.md): ``t0202_slow_results.html`` (Enter paints a header at once and a blue results block
800 ms later), ``t0202_never_loads.html`` (never changes); T-0111c's ``color.html`` is the normal page.

**Oracles.** Pixels read independently of the executor (``rgb_at``/``near``, T-0111c's helpers) at the results
block's centre (640, 450); the phash difference is between the two outcomes' ``frame_after``. The normal-page row is
a differential against BASE 5821b628's executor (loaded from git) over the same page, recording every driver method
the executor calls and every ``asyncio.sleep`` it awaits.

**Reading (reported).** "The four calls take at most 3 x ``ACTION_TIMEOUT_S`` of wall time": measured from the first
wait's start to the fourth's return.

Regime: Playwright's pinned Chromium, headless, loopback, 1280x800, real ``WAIT_S`` (1 s), n = 1 per row.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import http.server
import sys
import threading
import time
import types
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from local_astra.computer import executor as executor_mod
from local_astra.computer import runtime as runtime_mod
from local_astra.contracts.toolcall import ToolInvocation
from tests import _git_range as gr
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

BASE_SHA = "5821b628"
BLUE = (0, 90, 200)
WHITE = (255, 255, 255)
WAIT_RUN_REASON = "you have already waited three times in a row; act on the screen or answer"
#: the Driver seam's methods (``executor.Driver``) and the optional capabilities the executor looks up
SEAM = (
    "ensure",
    "screenshot",
    "settle",
    "mouse_move",
    "mouse_down",
    "mouse_up",
    "key_down",
    "key_up",
    "type_text",
    "scroll_by",
    "goto",
    "take_events",
    "layout_metrics",
    "focused_field",
    "navigation_mark",
    "navigated_since",
    "main_frame_urls",
    "bring_to_front",
    # T-0204 B10: the pre-click guard's driver calls are traced too
    "screenshot_region",
    "arm_shift_recorder",
    "shift_mark",
    "shifts_since",
)
#: T-0204 B10: the guard's synchronous reads and the capture's arming (not Playwright screenshots)
GUARD_READS = frozenset({"arm_shift_recorder", "shift_mark", "shifts_since"})

_N = iter(range(1_000_000))


@pytest.fixture(scope="module")
def origin() -> Iterator[str]:
    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Quiet, directory=str(FIXTURES))
    )
    t = threading.Thread(target=srv.serve_forever, name="t0202-fixture-server", daemon=True)
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


@contextlib.asynccontextmanager
async def rig(module: Any = executor_mod) -> AsyncIterator[Any]:
    d = runtime_mod.PlaywrightDriver(viewport=(1280, 800), on_egress=lambda e: None, headless=True)
    ex = module.BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS, viewport=(1280, 800), driver=d, headless=True
    )
    try:
        yield ex
    finally:
        with contextlib.suppress(Exception):
            await run(d.close(), 20)


async def test_slow_results_wait_shows_them(origin: str) -> None:
    """B10: type a search and press Enter: ``type_at``'s after-frame has the header but not the results (white at
    (640, 450)); then ``wait``: ``executed``, its after-frame shows the blue results block and its phash differs."""
    async with rig() as ex:
        await run(ex.open(f"{origin}/t0202_slow_results.html"))
        f = await run(ex.frame())
        typed = await execute(ex, call("type_at", f, x=500, y=75, text="milk", press_enter=True))
        assert typed.status == "executed", typed
        before = ex.latest_frame
        assert near(rgb_at(before, 640, 450), WHITE), (
            f"premise: the results are not painted yet after Enter: {rgb_at(before, 640, 450)}"
        )
        out = await execute(ex, call("wait"))
        assert out.status == "executed" and out.action == "waited 1 s", out
        after = ex.latest_frame
        assert out.resolved["frame_after"] == after.phash != typed.resolved["frame_after"], (
            out.resolved,
            typed.resolved,
        )
        assert near(rgb_at(after, 640, 450), BLUE), (
            f"T-0202 B10: the wait's after-frame shows no results: {rgb_at(after, 640, 450)}"
        )


async def test_never_loads_run_limit(origin: str) -> None:
    """B10: on a page that never changes, waits 1-3 are ``executed`` and wait 4 is refused ``WAIT_RUN_REASON``; the
    four calls take at most 3 x ``ACTION_TIMEOUT_S`` of wall time."""
    async with rig() as ex:
        await run(ex.open(f"{origin}/t0202_never_loads.html"))
        await run(ex.frame())
        t0 = time.monotonic()
        outs = [await execute(ex, call("wait")) for _ in range(4)]
        took = time.monotonic() - t0
        assert [o.status for o in outs] == ["executed"] * 3 + ["refused"], outs
        assert outs[3].reason == WAIT_RUN_REASON, outs[3]
        assert took <= 3 * executor_mod.ACTION_TIMEOUT_S, (
            f"T-0202 B10: four calls took {took:.1f} s"
        )


_BASE: types.ModuleType | None = None


def base_module() -> types.ModuleType:
    global _BASE
    if _BASE is None:
        gr.require_commits(BASE_SHA)
        src = gr._git("show", f"{BASE_SHA}:local_astra/computer/executor.py").stdout
        assert "WAIT_S" not in src, "premise: BASE has no wait"
        mod = types.ModuleType("_t0202_base_executor_browser")
        mod.__file__ = f"<{BASE_SHA}>/local_astra/computer/executor.py"
        sys.modules[mod.__name__] = mod
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        _BASE = mod
    return _BASE


async def _click_trace(origin: str, module: Any, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The driver methods and ``asyncio.sleep`` delays one ``click_at`` on ``color.html`` makes, in order."""
    seen: list[str] = []
    cls = runtime_mod.PlaywrightDriver
    for name in SEAM:
        real = getattr(cls, name, None)
        if real is None:
            continue
        if asyncio.iscoroutinefunction(real):

            async def wrapped(self: Any, *a: Any, _r: Any = real, _n: str = name, **k: Any) -> Any:
                seen.append(_n)
                return await _r(self, *a, **k)

        else:

            def wrapped(self: Any, *a: Any, _r: Any = real, _n: str = name, **k: Any) -> Any:  # type: ignore[misc]
                seen.append(_n)
                return _r(self, *a, **k)

        monkeypatch.setattr(cls, name, wrapped)
    real_sleep = asyncio.sleep

    async def sleep(delay: float, result: Any = None) -> Any:
        # only the executor's own sleeps (Playwright's internals sleep too, nondeterministically)
        if sys._getframe(1).f_globals.get("__name__") == module.__name__:
            seen.append(f"sleep({delay})")
        return await real_sleep(delay, result)

    async with rig(module) as ex:
        await run(ex.open(f"{origin}/color.html"))
        f = await run(ex.frame())
        start = len(seen)
        monkeypatch.setattr(asyncio, "sleep", sleep)
        try:
            out = await execute(ex, call("click_at", f, x=500, y=500))
        finally:
            monkeypatch.setattr(asyncio, "sleep", real_sleep)
        assert out.status == "executed", out
        trace = [s for s in seen[start:] if s in SEAM or s.startswith("sleep(")]
    monkeypatch.undo()
    return trace


async def test_normal_click_sequence_as_base(origin: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """B10 (James: no fixed wait anywhere): a normal page's ``click_at`` calls the same driver methods in the same
    order as BASE's executor, and the executor awaits no ``asyncio.sleep`` that BASE does not. T-0204 B10 supersedes
    "the same driver call sequence" for the real driver: BASE plus one ``screenshot_region`` (a ``screenshot`` with a
    ``clip``) before the first ``mouse_move``, once the guard's synchronous reads and the capture's arming are set
    aside; still no sleep. T-0204 C10: plus the visual settle's second capture after BASE's, and at most its one
    sleep of <= 0.1 s."""
    base = await _click_trace(origin, base_module(), monkeypatch)
    tip = await _click_trace(origin, executor_mod, monkeypatch)
    assert not any(s in GUARD_READS or s == "screenshot_region" for s in base), base
    assert "mouse_move" in base, base
    move = base.index("mouse_move")
    want = [*base[:move], "screenshot_region", *base[move:]]
    # T-0204 C10 supersedes it again (James's ruling (a) relaxes "no fixed wait" for the visual settle only): BASE plus
    # one more capture (the settle's second shot, with its two layout-metrics reads) right after BASE's capture, and
    # at most one executor sleep, of at most ``STABLE_SCHEDULE_S[0]`` = 0.1 s, before it
    last = max(i for i, s in enumerate(want) if s == "screenshot")
    assert want[last - 1] == want[last + 1] == "layout_metrics", (
        f"premise: BASE's capture block: {want}"
    )
    want = [*want[: last + 2], *want[last - 1 : last + 2], *want[last + 2 :]]
    sleeps = [s for s in tip if s.startswith("sleep(")]
    assert len(sleeps) <= 1 and all(0 < float(s[6:-1]) <= 0.1 + 1e-9 for s in sleeps), (
        f"T-0204 C10: the click's only executor sleep is the settle's first offset: {sleeps}"
    )
    got = [s for s in tip if s not in GUARD_READS and not s.startswith("sleep(")]
    assert got == want, f"T-0202 B10 + T-0204 B10/C10: click_at changed\n WANT {want}\n TIP  {got}"
