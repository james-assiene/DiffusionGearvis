"""T-0131 B (browser): ReGround fixes a miss on the real shop page (docs/tasks/T-0131.md, "Tests to write → Group B").

**Gate.** As T-0111g's live-session module (its autouse gate imported): ``integration`` + ``browser``, runs only with
``ASTRA_BROWSER_TESTS=1``; a missing ``playwright`` or Chromium is then a failure naming P1. Not in ``make
browser-live`` yet (the task: the verifier names this module by explicit command).

**The rig.** A real ``BrowserToolExecutor`` (headless, 1280x800) on the start-page server's ``/demo/shop.html``
behind T-0111g's ``Watch`` (delegates every name), driven by a real ``Session(..., reground=True)`` over the fakes
(``_live_browser_support.reground_rig``: the pass-2 fake LLM recognises the crop as the one image no browser frame
has).

**The row.** Pass 1 clicks 20 px *below the bottom edge* of the blue mug's "Add to cart" (the reading of "20 px
below": a point off the button, so executing pass 1 misses); pass 2 names the button's centre in the crop's own
0-999 coordinates, the crop being ``reground.crop_rect`` of pass 1's pixel point (the pinned pure helper). Passes
when ``__state.cart == ["blue mug"]``; under the mutation "execute pass 1" the cart stays empty (the row's premise
checks pass 1's point is outside the button's box).

``pending(task="T-0131")``: at BASE the session has no ``reground`` keyword (a named ``AssertionError`` from the
rig).

Regime: Playwright 1.64.0, Chromium 156 headless, 1280x800 at DSF 1, start pages on 127.0.0.1 (ephemeral port),
fake LLM, ScriptedClock, fake STT/TTS/audio, n = 1.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.tool_executor import to_pixels
from local_astra.contracts.turn import TurnState
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    _browser_layer,
    _browser_marker_registered,
    computer,
)
from tests.integration.test_computer_live_session import (
    ADD_BLUE,
    Watch,
    ask,
    click,
    coord,
    drive,
    release_step,
    state,
)
from tests.unit._live_browser_support import reground_rig
from tests.unit._tool_turn_support import answer, tool
from tests.unit.test_start_pages import new_server

pytestmark = [pytest.mark.integration] + (
    [pytest.mark.browser] if _browser_marker_registered() else []
)

W, H = 1280, 800


class FramesWatch(Watch):
    """``Watch`` plus ``frames_returned``: every own frame the session was given (its ``frame()`` captures and the
    post-action frames), which is how the pass-2 fake tells the crop apart."""

    @property
    def frames_returned(self) -> list[Any]:
        return [f for _, f in self.frames] + [a for a in self.after if a is not None]


@contextlib.asynccontextmanager
async def reground_session(tmp_path: Path, start: str) -> AsyncIterator[tuple[Any, FramesWatch]]:
    srv = new_server()
    srv.start()
    ex = computer().BrowserToolExecutor(
        now_ms=lambda: 0,
        headless=True,
        end_when_closed=False,
        allowed_origins=frozenset({srv.origin}),
    )
    watch = FramesWatch(ex)
    try:
        await ex.open(srv.origin + start)
        async with reground_rig(tmp_path, watch) as rig:  # type: ignore[arg-type]
            ex._now_ms = lambda: rig.clock.now_ms
            yield rig, watch
    finally:
        with contextlib.suppress(Exception):
            await ex.close()
        srv.stop()


async def box(ex: Any, js: str) -> tuple[float, float, float, float]:
    got = await ex.driver.page.evaluate(
        f"(() => {{ const e = ({js}); if (!e) return null; const r = e.getBoundingClientRect();"
        " return [r.left, r.top, r.width, r.height]; })()"
    )
    assert got is not None, f"no element for {js}"
    return float(got[0]), float(got[1]), float(got[2]), float(got[3])


def crop_rect() -> Any:
    try:
        mod = importlib.import_module("local_astra.toolcall.reground")
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "T-0131 B2: local_astra/toolcall/reground.py does not exist yet"
        ) from exc
    return mod.crop_rect


async def test_reground_turns_a_miss_into_the_blue_mug(tmp_path: Path) -> None:
    """B (browser): pass 1 lands 20 px below the blue mug's "Add to cart", pass 2 names the button's centre in the
    crop; the regrounded click executes and the cart holds the blue mug."""
    async with reground_session(tmp_path, "/demo/shop.html") as (rig, watch):
        ex = watch._inner
        left, top, width, height = await box(ex, ADD_BLUE)
        cx, cy = left + width / 2, top + height / 2
        x1, y1 = coord(cx, W), coord(top + height + 20, H)
        px1, py1 = to_pixels(x1, W), to_pixels(y1, H)
        assert not (left <= px1 <= left + width and top <= py1 <= top + height), (
            "premise: pass 1's point is off the button (executing it misses)"
        )
        x0, y0, cw, ch = crop_rect()(W, H, px1, py1)
        assert x0 <= cx <= x0 + cw and y0 <= cy <= y0 + ch, "premise: the button is inside the crop"
        x2 = max(0, min(999, round((cx - x0) * 999 / (cw - 1))))
        y2 = max(0, min(999, round((cy - y0) * 999 / (ch - 1))))
        rig.llm.script = [answer("I will add the blue mug. "), tool(click(x1, y1))]
        rig.llm.script_pass2([answer(""), tool(click(x2, y2))])
        rig.llm.hold_steps()
        await ask(rig, watch, "add the blue mug to the cart")
        await drive(rig, watch, lambda: len(watch.outcomes) >= 1 and watch.inflight == 0, "step 1")
        await release_step(rig, watch, 1, [answer("The blue mug is in the cart. ")])
        rig.llm.release_steps()
        await drive(
            rig, watch, lambda: rig.state.state is TurnState.IDLE and bool(rig.lines()), "turn end"
        )
        assert [o.status for o in watch.outcomes] == ["executed"], watch.outcomes
        assert await state(ex, "cart") == ["blue mug"], (
            "T-0131 B: the regrounded click added the blue mug"
        )
        assert len(rig.llm.pass2_prompts) == 1, "one pass-2 request"
