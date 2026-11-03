"""T-0192 group B end to end (B-RV4): a real ``Session`` over the session fakes, driving the real web-mode
``BrowserToolExecutor`` (real headless Chromium, sandbox on, the real ``EgressProxy`` with T-0160's injected resolver
and connector, loopback fixtures posing as ``shop.example``), reaches a checkout and hands the turn back
(docs/tasks/T-0192.md B2, B4, "Group B review rulings" B-RV4).

**Gate.** ``integration`` + ``browser``; runs only with ``ASTRA_BROWSER_TESTS=1``, exactly like the group A module
(``ASTRA_BROWSER_TESTS=1 uv run pytest tests/integration/test_t0192_hand_back_browser.py -q -m browser``). The rig is
T-0160's (``tests/integration/test_computer_web_browser.py``: ``rig``, ``launches``, the autouse ``_browser_layer``
gate, ``Net``, ``web_executor``, ``center``); the session side is T-0111g's (``Watch``, ``drive``, ``ask`` from
``tests/integration/test_computer_live_session.py``; ``tool_rig`` with the loop on and a live kit over the executor).

**The two paths** (fixture ``t0192_cart.html``, ``tests/fixtures/browser/web/MANIFEST.md``):

- ``#proceed``: ``/checkout/entry?proceedToCheckout=1`` -> 302 -> ``/signin.html?openid.return_to=...`` (the demo
  shape: the checkout is only a redirect hop; A4 reports the arrival from the hop).
- ``#buynow``: ``/checkout/entry2`` -> 302 -> ``/checkout/<token>/pay`` (lands on a checkout page).

Each must hand back (expected by hand from B1/B2/B4): the click's ``ToolResult`` carries ``hand_back="checkout"``;
``at_checkout`` is among the played clips (the rig may also play the ``ack`` filler); the model is never asked for a
further step (the scripted "place the order" step is never requested or voiced); the turn line's ``tool_loop_end``
is ``"handed_back"``; history holds the played step-1 text and never the clip's text.

**Regime.** Playwright 1.64.0, Chromium headless 1280x800, DSF 1, sandbox on, no real network, ``ScriptedClock`` for
the session (virtual time), loop on, n = 1 per row, one browser per row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.turn import TurnState
from tests.integration import test_computer_web_browser as web_rig
from tests.integration.test_computer_live_session import Watch, ask, click, drive
from tests.integration.test_computer_web_browser import Net, Rig, center, web_executor
from tests.integration.test_t0192_checkout_browser import CHECKOUT, SIGNIN_PREFIX, on_cart
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import live_kit
from tests.unit._tool_turn_support import answer, tool, tool_rig

#: T-0160's fixtures, re-bound here so pytest registers them in this module (``_browser_layer`` is the autouse gate)
_browser_layer = web_rig._browser_layer
rig = web_rig.rig
launches = web_rig.launches

pytestmark = [pytest.mark.integration, pytest.mark.browser]

#: B1, by hand from the task file
AT_CHECKOUT_TEXT = (
    "I've reached the checkout, so I'll stop here. Over to you to finish the purchase."
)
S1 = "I will go to the checkout. "
#: the adversarial next step: it must never be requested or voiced
DEAD = "I will place the order now. "

#: selector -> (the hop the fixture server must have served, where the page ends up)
PATHS = {
    "#proceed": ("/checkout/entry", SIGNIN_PREFIX),
    "#buynow": ("/checkout/entry2", CHECKOUT),
}


@pytest.mark.parametrize("sel", sorted(PATHS), ids=lambda s: s.lstrip("#"))
async def test_brv4_real_session_hands_back_at_checkout(
    rig: Rig, launches: list[dict[str, Any]], tmp_path: Path, sel: str
) -> None:
    """B-RV4: one turn, "buy the blue mug": step 1 voices S1 and clicks ``sel`` on the cart; the real executor
    reports the checkout (A4); the session submits ``hand_back="checkout"`` (B4) and the machine ends the turn with
    the ``at_checkout`` clip and no further step (B2)."""
    hop, landing = PATHS[sel]
    net = Net(rig.tls_port)
    async with web_executor(rig, net) as ex:
        await on_cart(ex)
        x, y = await center(ex, sel)
        watch = Watch(ex)
        async with tool_rig(tmp_path, kit=live_kit(watch), policy=tsl.LOOP) as s:
            ex._now_ms = lambda: s.clock.now_ms
            s.llm.script = [answer(S1), tool(click(x, y))]
            s.llm.step_scripts = [[answer(DEAD)]]
            await ask(s, watch, "buy the blue mug")
            await drive(
                s,
                watch,
                lambda: (
                    len(watch.outcomes) >= 1
                    and watch.inflight == 0
                    and s.state.state is TurnState.IDLE
                ),
                "the turn never ended",
                limit_s=60,
            )
            await s.advance(2000)
            # premises: the real browser went through the hop and landed where the fixture sends it
            assert ("shop.example", hop) in rig.hits, f"premise: {hop} was served; {rig.hits}"
            assert ex.driver.page.url.startswith(landing), (
                f"premise: landed on {landing}; {ex.driver.page.url}"
            )
            assert [i.name for i in watch.invocations] == ["click_at"], watch.invocations
            out = watch.outcomes[0]
            assert out.status == "executed" and out.resolved.get("checkout") is True, (
                f"premise (A4): the click reached a checkout; {out}"
            )
            results = list(s.machine.events(E.ToolResult))
            assert [(r.call_id, r.status, getattr(r, "hand_back", None)) for r in results] == [
                (out.call_id, "executed", "checkout")
            ], f"T-0192 B4: the result is submitted with hand_back='checkout'; {results}"
            kinds = [getattr(u, "kind", None) for u in s.audio.clip_plays()]
            assert "at_checkout" in kinds, f"T-0192 B2: the at_checkout clip played; {kinds}"
            assert set(kinds) <= {"at_checkout", "ack"} and kinds.count("at_checkout") == 1, (
                f"T-0192 B2/L10: one ending clip, at_checkout (ack is the rig's filler); {kinds}"
            )
            assert not s.llm.step_prompts, (
                f"T-0192 B2/L3: the model was asked for a further step; {len(s.llm.step_prompts)}"
            )
            assert DEAD not in s.spoken_sources(), "the drafted 'place the order' step was voiced"
        # the session log is read when the rig closes
        rows = [r for r in s.kind("tool.result") if r.get("call_id") == out.call_id]
        assert len(rows) == 1 and rows[0].get("hand_back") == "checkout", (
            f"T-0192 B4: the tool.result row gains hand_back='checkout'; {rows}"
        )
        end = s.record("t1").extras.get("tool_loop_end")
        assert end == "handed_back", f"T-0192 B4: tool_loop_end; got {end!r}"
        history = [(h.user_text, h.assistant_spoken) for h in s.state.history]
        assert any(S1.strip() in (spoken or "") for _, spoken in history), (
            f"T-0192 B2: history holds the played S1; {history}"
        )
        assert all(
            AT_CHECKOUT_TEXT not in (spoken or "") and DEAD.strip() not in (spoken or "")
            for _, spoken in history
        ), f"H8/L6: no clip text (and no unvoiced step) in history; {history}"
