"""T-0205 group A, criterion A10: a real ``Session`` in live mode drives the web-mode executor on the scenario pages
through D4's golden path continued to checkout, and hands the turn back (docs/tasks/T-0205.md, "Tests to write",
``tests/integration/test_t0205_scenarios_session.py``).

**Gate.** As the browser module (T-0111c's autouse ``_browser_layer``): ``integration`` + ``browser``, runs only with
``ASTRA_BROWSER_TESTS=1``; ``make browser-demo`` (A9) runs it.

**The rig.** The executor is the browser module's: ``BrowserToolExecutor(web=True, start_origin=<scenario origin>,
allowed_origins={origin}, ...)`` with a deny-all resolver and connector (A7), on ``StartPageServer(static_dir=
SCENARIO_DIR)``. The session is T-0111g's (``tool_rig`` with a live kit over ``Watch``, ``drive``, ``ask``,
``release_step`` from ``tests/integration/test_computer_live_session.py``) with the web budget, 20 steps
(``max_tool_steps_web``, T-0178). The scripted fake LLM holds each step; before releasing step k+1 the test computes
its call from the current page's ``data-scn`` boxes (test-side), so the script is the test's and the page is the
browser's. The ``data-scn`` ids are the browser module's.

**Expected, by hand from A10 / T-0192 B2-B4.** Six executed actions (type the model into ``q``, Search, the BES870
result, Add on the Steam Wand Assembly, the cart link, Checkout; plus any scroll the page geometry needs); the
Checkout click's outcome carries ``resolved["checkout"] is True``; its ``ToolResult`` carries
``hand_back="checkout"``; one ``at_checkout`` clip plays; no step request follows the Checkout click (step requests ==
actions - 1); the turn line's ``tool_loop_end`` is ``"handed_back"``; ``judge("D4", __state, voiced)`` succeeds with
the voiced text the session actually spoke.

BASE_A = c97e5b30.

**Group B (BASE_B = a4ba3cae).** ``test_d9_hand_back_through_session``: the same check for D9 (A10 [B]); ten actions
(type "SFO", its suggestion, type "Seattle", its suggestion, Depart, 6 November, Return, 8 November, Search, Select
on ``select-ua-0600``) and the arriving "Continue to United" click, then the same hand-back expectations, with
``/flights/united/checkout.html`` and ``$212`` voiced. ``test_d7_request_quote_refused`` (the D7 refusal row; T-0206 A
has not merged before B's activation, so it skips naming T-0206 while ``executor.SUBMIT_REASON`` is absent, and
T-0206 A's own rows also cover it): after D7's golden fill in the A7 web-mode rig, the ``click_at`` on "Request quote"
is ``refused`` with ``SUBMIT_REASON``, ``submissions == []`` and the judge succeeds.

Regime: Playwright 1.64.0, Chromium headless 1280x800, DSF 1, sandbox on, web mode on loopback, deny-all seams,
``ScriptedClock`` for the session, loop 20, n = 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.turn import TurnState
from tests.integration.test_computer_live_session import (
    Watch,
    ask,
    click,
    drive,
    release_step,
    type_at,
)
from tests.integration.test_t0205_scenarios_browser import (  # noqa: F401  (_browser_layer is the autouse gate)
    BOX_JS,
    SAID,
    STATE_JS,
    DenyNet,
    H,
    W,
    _browser_layer,
    _browser_marker_registered,
    _d7_fill,
    _submit_reason_or_skip,
    judge,
    opened,
    scn_dir,
    unit,
    vdict,
    web_executor,
)
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import live_kit
from tests.unit._tool_turn_support import answer, tool, tool_rig
from tests.unit.test_start_pages import new_server

pytestmark = [
    pytest.mark.integration,
    *([pytest.mark.browser] if _browser_marker_registered() else []),
]

S1 = "The Steam Wand Assembly for the Barista Express is $39.95; I will put it in the cart. "
#: the adversarial step after the arrival: it must never be requested or voiced
DEAD = "I will place the order now. "
SCROLL = '<|tool_call>call:scroll{direction:<|"|>down<|"|>}<tool_call|>'


async def _box(ex: Any, sid: str) -> dict[str, Any]:
    b = await ex.driver.page.evaluate(BOX_JS, sid)
    assert b is not None, f'no data-scn="{sid}" on {ex.driver.page.url}'
    return dict(b)


async def _call_for(ex: Any, sid: str, text: str | None = None) -> str:
    """The next step's call: a scroll when ``sid`` is not in the viewport yet, else a click (or a type) at its box."""
    b = await _box(ex, sid)
    if b["bottom"] > H:
        return SCROLL
    x, y = unit(b["left"], b["right"], W), unit(b["top"], b["bottom"], H)
    return type_at(x, y, text, False) if text is not None else click(x, y)


async def test_d4_hand_back_through_session(tmp_path: Path) -> None:
    """A10: D4's golden path continued to checkout through a real ``Session``: ends ``handed_back`` with the
    ``at_checkout`` clip, no step after the Checkout click, and the judge succeeds."""
    srv = new_server(scn_dir())
    srv.start()
    net = DenyNet()
    try:
        async with web_executor(srv.origin, net) as ex:
            await ex.open(srv.origin + "/breville/")
            watch = Watch(ex)
            async with tool_rig(tmp_path, kit=live_kit(watch), policy=tsl.loop_policy(20)) as s:
                ex._now_ms = lambda: s.clock.now_ms
                s.llm.script = [answer(S1), tool(await _call_for(ex, "q", "Barista Express"))]
                s.llm.hold_steps()
                await ask(
                    s,
                    watch,
                    "my Breville Barista Express steam wand broke; find the part and buy it",
                )
                plan = [
                    "search",
                    "result-bes870",
                    "add-steam-wand-assembly",
                    "nav-cart",
                    "checkout",
                ]
                n = 1
                while plan:
                    await drive(
                        s,
                        watch,
                        lambda n=n: len(watch.outcomes) >= n and watch.inflight == 0,
                        f"step {n}",
                    )
                    call = await _call_for(ex, plan[0])
                    if call != SCROLL:
                        plan.pop(0)
                    if not plan:
                        # the Checkout click: release it and never wait for a further step request
                        s.llm.step_scripts = [[answer(""), tool(call)], [answer(DEAD)]]
                        s.llm.release_steps()
                        break
                    await release_step(s, watch, n, [answer(""), tool(call)])
                    n += 1
                await drive(
                    s,
                    watch,
                    lambda: (
                        len(watch.outcomes) >= n + 1
                        and watch.inflight == 0
                        and s.state.state is TurnState.IDLE
                    ),
                    "the turn never ended",
                    limit_s=120,
                )
                await s.advance(2000)
                outs = watch.outcomes
                assert [o.status for o in outs] == ["executed"] * len(outs), outs
                assert ex.driver.page.url == srv.origin + "/breville/checkout.html", (
                    ex.driver.page.url
                )
                last = outs[-1]
                assert last.resolved.get("checkout") is True, (
                    f"T-0205 A10 (T-0192 A4): the Checkout click carries no checkout: {last}"
                )
                results = list(s.machine.events(E.ToolResult))
                assert getattr(results[-1], "hand_back", None) == "checkout", results[-1]
                kinds = [getattr(u, "kind", None) for u in s.audio.clip_plays()]
                assert kinds.count("at_checkout") == 1, f"T-0205 A10: the at_checkout clip; {kinds}"
                assert len(tsl.steps_of(s)) == len(outs) - 1, (
                    f"T-0205 A10: a step was requested after the hand-back; {len(tsl.steps_of(s))} step "
                    f"requests for {len(outs)} actions"
                )
                assert DEAD not in s.spoken_sources(), (
                    "the drafted 'place the order' step was voiced"
                )
                st = await ex.driver.page.evaluate(STATE_JS)
            end = s.record("t1").extras.get("tool_loop_end")
            assert end == "handed_back", f"T-0205 A10: tool_loop_end is {end!r}"
            voiced = " ".join(h.assistant_spoken or "" for h in s.state.history)
            assert "39.95" in voiced, f"premise: the session voiced S1; {voiced!r}"
            v = judge("D4", st, voiced)
            assert v.success is True, (st, vdict(v))
    finally:
        srv.stop()
    assert (net.resolves, net.connects) == ([], []), (net.resolves, net.connects)


S1_D9 = "The cheapest nonstop round trip is United at 6:00 for $212; I will take it to United's checkout. "
D9_PLAN: list[tuple[str, str | None]] = [
    ("suggest-sfo", None),
    ("destination", "Seattle"),
    ("suggest-sea", None),
    ("depart", None),
    ("day-2026-11-06", None),
    ("return", None),
    ("day-2026-11-08", None),
    ("search", None),
    ("select-ua-0600", None),
    ("continue", None),
]


async def test_d9_hand_back_through_session(tmp_path: Path) -> None:
    """A10 [B]: D9's golden path continued to checkout ("Continue to United") through a real ``Session``: every action
    executed, the arriving click carries ``resolved["checkout"]``, its ``ToolResult`` carries ``hand_back="checkout"``,
    one ``at_checkout`` clip, no step after it, ``tool_loop_end == "handed_back"``, and the judge succeeds with the
    voiced text."""
    srv = new_server(scn_dir())
    srv.start()
    net = DenyNet()
    try:
        async with web_executor(srv.origin, net) as ex:
            await ex.open(srv.origin + "/flights/")
            watch = Watch(ex)
            async with tool_rig(tmp_path, kit=live_kit(watch), policy=tsl.loop_policy(20)) as s:
                ex._now_ms = lambda: s.clock.now_ms
                s.llm.script = [answer(S1_D9), tool(await _call_for(ex, "origin", "SFO"))]
                s.llm.hold_steps()
                await ask(
                    s,
                    watch,
                    "find the cheapest nonstop round trip from SFO to Seattle, leaving Friday 6 November and back "
                    "Sunday 8 November, and take it to the airline's checkout",
                )
                plan = list(D9_PLAN)
                n = 1
                while plan:
                    await drive(
                        s,
                        watch,
                        lambda n=n: len(watch.outcomes) >= n and watch.inflight == 0,
                        f"step {n}",
                    )
                    call = await _call_for(ex, *plan[0])
                    if call != SCROLL:
                        plan.pop(0)
                    if not plan:
                        s.llm.step_scripts = [[answer(""), tool(call)], [answer(DEAD)]]
                        s.llm.release_steps()
                        break
                    await release_step(s, watch, n, [answer(""), tool(call)])
                    n += 1
                await drive(
                    s,
                    watch,
                    lambda: (
                        len(watch.outcomes) >= n + 1
                        and watch.inflight == 0
                        and s.state.state is TurnState.IDLE
                    ),
                    "the turn never ended",
                    limit_s=120,
                )
                await s.advance(2000)
                outs = watch.outcomes
                assert [o.status for o in outs] == ["executed"] * len(outs), outs
                assert len(outs) >= 11, [o.action for o in outs]
                assert ex.driver.page.url == srv.origin + "/flights/united/checkout.html", (
                    ex.driver.page.url
                )
                assert outs[-1].resolved.get("checkout") is True, outs[-1]
                results = list(s.machine.events(E.ToolResult))
                assert getattr(results[-1], "hand_back", None) == "checkout", results[-1]
                kinds = [getattr(u, "kind", None) for u in s.audio.clip_plays()]
                assert kinds.count("at_checkout") == 1, f"T-0205 A10: the at_checkout clip; {kinds}"
                assert len(tsl.steps_of(s)) == len(outs) - 1, (
                    f"T-0205 A10: a step was requested after the hand-back; {len(tsl.steps_of(s))} step "
                    f"requests for {len(outs)} actions"
                )
                assert DEAD not in s.spoken_sources(), (
                    "the drafted 'place the order' step was voiced"
                )
                st = await ex.driver.page.evaluate(STATE_JS)
            end = s.record("t1").extras.get("tool_loop_end")
            assert end == "handed_back", f"T-0205 A10: tool_loop_end is {end!r}"
            voiced = " ".join(h.assistant_spoken or "" for h in s.state.history)
            assert "212" in voiced, f"premise: the session voiced S1_D9; {voiced!r}"
            v = judge("D9", st, voiced)
            assert v.success is True, (st, vdict(v))
    finally:
        srv.stop()
    assert (net.resolves, net.connects) == ([], []), (net.resolves, net.connects)


async def test_d7_request_quote_refused() -> None:
    """The D7 refusal row (T-0206 A3; skips while ``executor.SUBMIT_REASON`` is absent): after D7's golden fill in the
    A7 web-mode rig, the ``click_at`` on "Request quote" is ``refused`` with ``SUBMIT_REASON``; ``submissions == []``,
    the form is still there, and the judge succeeds (the stop is the scenario's success)."""
    reason = _submit_reason_or_skip()
    async with opened("D7") as rig:
        await _d7_fill(rig)
        rig.executed()
        out = await rig.click("request-quote")
        assert (out.status, out.reason) == ("refused", reason), out
        st = await rig.dstate()
        assert st.get("submissions") == [], st
        assert (await rig.box("request-quote"))["shown"], "the form went"
        v = judge("D7", st, SAID["D7"][0])
        assert (v.success, v.submit_pressed) == (True, False), vdict(v)
