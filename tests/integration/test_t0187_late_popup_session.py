"""T-0187 criterion 3 with a real browser: the late popup through a real ``Session`` (docs/tasks/T-0187.md).

**Gate.** As ``tests/integration/test_computer_browser_e2e.py`` (its autouse gate imported): ``integration`` +
``browser``, runs only with ``ASTRA_BROWSER_TESTS=1``; a missing ``playwright`` or Chromium is then a failure naming
P1 of T-0111c. The orchestrator adds the module to ``make browser-live`` at merge.

**The rig.** ``tests/unit/_tool_turn_support.tool_rig`` with the loop on and a live kit whose executor is a real
``BrowserToolExecutor(headless=True, end_when_closed=False, allowed_origins={fixture origin})`` behind
``test_computer_live_session.Watch`` (delegates every name, records outcomes and the frames the session took).
``tests/fixtures/browser/`` is served on 127.0.0.1 at an ephemeral port. ``late_popup.html`` is one red full-viewport
target whose click opens ``popup.html?n=2`` (blue, ``PALETTE[2]``) 300 ms later, after the executor's settle has
returned, as in James's t16 (photopea.com). Step 2's request is held
(``FakeLlm.hold_steps``), so the popup opens on wall time between step 1's result and step 2's execute; steps 2-4
are then released together with their scripts queued.

**The row.** Step 1 clicks the red page (executed; the post-action frame is the opener's). The popup opens. Step 2
clicks at the same point, carrying the opener's frame: the executor's ``ensure()`` adopts the popup and refuses it
with ``the active page changed; take a new frame``. After the fix, step 3 shows a fresh frame of the popup, and its
click, carrying that frame, is executed on the popup; the popup's recorder saw it. Before the fix, step 3 kept the
opener's frame and its click was refused again (``plan.frame_page != d.page_serial``).

Expected values: the colours are the fixtures' (``PALETTE`` in ``test_computer_browser_e2e``); (500, 500) in model
units is pixel (640, 400) on 1280x800 by ``to_pixels``' formula ``round(v * (size - 1) / 999)``, by hand; the reason
text and the note are typed from criteria 2 and 7.

Regime: Playwright 1.64.0, Chromium 156.0.8078.4 headless, 1280x800 at DSF 1, fixture server on 127.0.0.1 at an
ephemeral port, FakeLlm with one script per step, ScriptedClock, fake STT/TTS/audio, n = 1.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import http.server
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.turn import TurnState
from tests.integration.test_computer_browser_e2e import (  # noqa: F401  (_browser_layer is the autouse gate)
    FIXTURES,
    PALETTE,
    _browser_layer,
    _browser_marker_registered,
    _Quiet,
    computer,
    near,
    of_type,
    poll,
    rgb_at,
)
from tests.integration.test_computer_live_session import Watch, ask, click, drive
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import data_url, live_kit
from tests.unit._tool_turn_support import answer, tool, tool_rig

pytestmark = [
    pytest.mark.integration,
] + ([pytest.mark.browser] if _browser_marker_registered() else [])

#: criterion 2: the executor's page-changed reason, typed
PAGE_CHANGED = "the active page changed; take a new frame"
#: criterion 7's note
REFRESHED = "live frame: refreshed after a page-change refusal"
RED = (255, 0, 0)
#: (500, 500) in model units on 1280x800 (module docstring)
PX, PY = 640, 400
CLICK = click(500, 500)
POINTER = ("pointerdown", "pointerup", "click")


@contextlib.asynccontextmanager
async def popup_session(tmp_path: Path) -> AsyncIterator[tuple[Any, Watch, str]]:
    handler = functools.partial(_Quiet, directory=str(FIXTURES))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, name="t0187-fixture-server", daemon=True)
    t.start()
    origin = f"http://127.0.0.1:{srv.server_address[1]}"
    ex = computer().BrowserToolExecutor(
        now_ms=lambda: 0,
        headless=True,
        end_when_closed=False,
        allowed_origins=frozenset({origin}),
    )
    watch = Watch(ex)
    try:
        await asyncio.wait_for(ex.open(origin + "/late_popup.html"), 30)
        async with tool_rig(tmp_path, kit=live_kit(watch), policy=tsl.LOOP) as rig:
            ex._now_ms = lambda: rig.clock.now_ms
            yield rig, watch, origin
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ex.close(), 20)
        srv.shutdown()
        srv.server_close()
        t.join(5)


def current_image_urls(rig: Any, k: int) -> list[str]:
    """The image URLs of step request ``k``'s final user message (its current image(s))."""
    msgs = tsl.step_request(rig, k).messages
    last = [m for m in msgs if m.role == "user"][-1]
    return [p.url for p in last.content if getattr(p, "type", "") == "image_url"]


async def test_page_change_refusal_then_click_on_the_fresh_frame_executes_on_the_popup(
    tmp_path: Path,
) -> None:
    """Criterion 3 (and criteria 1 and 7 end to end; module docstring for the steps)."""
    async with popup_session(tmp_path) as (rig, watch, origin):
        ex = watch._inner
        opener = ex.driver.page
        rig.llm.script = [answer("I will click the red page. "), tool(CLICK)]
        rig.llm.hold_steps()
        await ask(rig, watch, "click the red page, then click what opens")
        await drive(
            rig, watch, lambda: len(watch.outcomes) >= 1 and watch.inflight == 0, "step 1 never ran"
        )
        assert watch.outcomes[0].status == "executed", watch.outcomes[0]
        opener_after = watch.after[0]
        assert opener_after is not None and near(rgb_at(opener_after, PX, PY), RED), (
            "premise: step 1's post-action frame is the red opener (the popup came before the settle returned and "
            "the row tests nothing)"
        )
        pages = await poll(
            lambda: asyncio.sleep(0, list(opener.context.pages)),
            lambda ps: len(ps) == 2,
            timeout=5.0,
        )
        assert len(pages) == 2, f"premise: the late popup never opened: {pages}"
        popup = next(p for p in pages if p is not opener)
        await asyncio.wait_for(popup.wait_for_load_state("domcontentloaded"), 10)
        await asyncio.sleep(0.3)
        assert ex.driver.page is opener, (
            "premise: the popup is not adopted before step 2's execute (late, as in t16)"
        )
        await opener.evaluate("window.__log.length = 0")

        # Steps 2, 3 and 4 get their scripts at once and the flow stays open: re-holding after step 2 races the
        # step-3 request (it can be issued, and consume an empty script, before a re-hold lands).
        rig.llm.step_scripts = [
            [answer(""), tool(CLICK)],
            [answer(""), tool(CLICK)],
            [answer("Done. ")],
        ]
        rig.llm.release_steps()
        await drive(
            rig, watch, lambda: rig.state.state is TurnState.IDLE and bool(rig.lines()), "turn end"
        )
        assert rig.llm.step_overruns == [], (
            f"rig: every step request had its script; overruns {rig.llm.step_overruns}"
        )
        assert len(watch.outcomes) >= 2, f"premise: step 2 reached the executor; {watch.outcomes}"
        second = watch.outcomes[1]
        assert (second.status, second.reason) == ("refused", PAGE_CHANGED), (
            f"premise: step 2's click, read off the opener's frame, is refused with the page-changed reason; {second}"
        )
        assert current_image_urls(rig, 2) == [data_url(opener_after.jpeg)], (
            "premise: step 2 showed the opener's post-action frame"
        )
        assert tsl.results_for(rig, "t1-call-s3"), (
            f"premise: step 3's call settled in the session; results "
            f"{[(e.call_id, e.status) for e in rig.machine.events(tsl.E.ToolResult)]}"
        )
        statuses = [(o.status, o.reason) for o in watch.outcomes]
        step3 = [e.status for e in tsl.results_for(rig, "t1-call-s3")]
        cards = [(m.status, m.reason) for m in rig.tool_calls("t1-call-s3")]
        assert step3 == ["executed"] and len(statuses) == 3 and statuses[2] == ("executed", None), (
            f"criterion 3: the click after the refusal is executed (before the fix it is refused again); step 3 "
            f"{step3}, its cards {cards}, executor outcomes {statuses}"
        )
        # every frame() after the onset: step 2 and step 4 read the executor's post-action frames (latest_frame)
        fresh = [f for _gen, f in watch.frames[1:]]
        assert len(fresh) == 1, (
            f"criterion 1: one fresh frame was taken after the onset (after the refusal); frames "
            f"{len(watch.frames)}, outcomes {statuses}"
        )
        popup_frame = fresh[0]
        assert near(rgb_at(popup_frame, PX, PY), PALETTE[2]), (
            f"criterion 1: the fresh frame is of the active page, the blue popup; {rgb_at(popup_frame, PX, PY)}"
        )
        assert current_image_urls(rig, 3) == [data_url(popup_frame.jpeg)], (
            "criterion 1: step 3 shows exactly the popup's fresh frame (the opener's frame is replaced)"
        )
        third = watch.invocations[2]
        assert third.frame is not None and third.frame.phash == popup_frame.phash, (
            "criterion 3: step 3's click carries the fresh frame"
        )
        popup_log = of_type(await popup.evaluate("window.__log"), *POINTER)
        clicks = [e for e in popup_log if e.get("type") == "click"]
        assert len(clicks) == 1 and (clicks[0].get("x"), clicks[0].get("y")) == (PX, PY), (
            f"criterion 3: the popup's recorder saw the click at ({PX}, {PY}); {popup_log}"
        )
        opener_log = of_type(await opener.evaluate("window.__log"), *POINTER)
        assert opener_log == [], f"the refused click dispatched nothing to the opener: {opener_log}"
    notes = [d for d in rig.kind("note") if d.get("what") == REFRESHED]
    assert notes == [{"what": REFRESHED, "reason_kind": "PAGE_CHANGED_REASON", "tool_step": 2}], (
        f"criterion 7: one note for the refresh after step 2; notes {rig.kind('note')}"
    )
    assert origin not in json.dumps(notes) and "http" not in json.dumps(notes), (
        f"criterion 7: no URL in the note; {notes}"
    )
