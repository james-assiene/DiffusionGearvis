"""T-0166 P1: the T-0165 reviewer's live-before-hello probe, made deterministic (T-0162 D-3).

``docs/tasks/T-0166.md`` P1. After one Restart, a page capturing with ``?source=canvas`` that is
reloaded must hear the replayed ``astra.session.cleared`` as its **first message ever**, so D-3's
exception keeps its capture running. On main a live ``astra.turn.state`` (20 Hz while THINKING)
can reach the new socket before its hello reply; the page then takes it as its first message, and
the replayed cleared that follows is a reset it "missed", which stops the capture (about 3/25
reloads under load, per the reviewer).

**Why it is deterministic here.** Besides the session's own 20 Hz state (the LLM is held, so the
turn sits in THINKING) and a 20 Hz ticker that publishes the session's current turn state in case
the session publishes less often, ``server.open`` is wrapped so that one ``astra.turn.state`` is
published *the instant every new connection is accepted*, before its reader can see a hello. On
main that message is written ahead of the reply on every reload; under the fix it is held until the
reply has gone. So the row is red on main 10/10, not 3/25, and green under the fix at any load.

**Seams.** ``Page.addScriptToEvaluateOnNewDocument`` installs ``INIT_JS`` before any page script:
it wraps ``window.WebSocket`` (a ``Proxy``) and records, per socket, the first message received.
``Live``/``Page`` come from ``test_hud_rehearsal.py`` through T-0165's ``started``/``tab`` helpers
(``test_hud_startup.py``), and ``restart``/``state``/``cleared_on_wire`` from
``test_hud_restart.py``; all imported, none edited.

Regime: headless Chrome as found by ``requires_chrome`` (``Browser.getVersion`` in the failure
message); the page from ``HudStaticServer(HUD_DIR, 0)``, fresh profile, DSF 1; real
``SessionClock``; n = 10 reloads per run, every reload must pass; 20 Hz ``astra.turn.state``;
``os.getloadavg()`` and the per-reload reload-to-hello gap (``time.monotonic``, from just before
``Page.reload`` to the server seeing the hello) in the failure message. No seed.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from typing import Any

import pytest

from local_astra.contracts import events as E
from tests.integration.test_hud_headless import requires_chrome
from tests.integration.test_hud_restart import cleared_on_wire, restart, state
from tests.integration.test_hud_startup import hold_an_open_turn, release, started, tab

pytestmark = pytest.mark.integration

RELOADS = 10
QUESTION = "how many moons does the probe orbit"
#: real seconds after ``connected`` in which a wrong stopCapture would have landed
SETTLE_S = 0.8

INIT_JS = """
(() => {
  const rec = (window.__t0166 = { sockets: [] });
  const Real = window.WebSocket;
  window.WebSocket = new Proxy(Real, {
    construct(target, args, newTarget) {
      const ws = Reflect.construct(target, args, newTarget);
      const entry = { first: null, n: 0 };
      rec.sockets.push(entry);
      ws.addEventListener('message', (e) => {
        entry.n += 1;
        if (entry.first !== null) return;
        try {
          const m = JSON.parse(e.data);
          entry.first = { type: m.type, replay: m.replay === true };
        } catch (err) {
          entry.first = { type: 'unparsed', replay: false };
        }
      });
      return ws;
    },
  });
})();
"""

FIRST_RECORDED_JS = (
    "Boolean(window.__t0166) && window.__t0166.sockets.length > 0"
    " && window.__t0166.sockets[0].first !== null"
)
RESTART_ENABLED_JS = (
    "(() => { const b = document.querySelector('[data-restart-button]');"
    " return Boolean(b) && !b.disabled; })()"
)


def publish_on_every_open(live: Any) -> None:
    """One live ``astra.turn.state`` the instant each connection is accepted (see the docstring)."""
    real_open = live.server.open

    def open_then_publish() -> Any:
        connection = real_open()
        live.server.publish(live.session.turn_state_message())
        return connection

    live.server.open = open_then_publish


async def twenty_hz(live: Any) -> None:
    while True:
        live.server.publish(live.session.turn_state_message())
        await asyncio.sleep(0.05)


@requires_chrome
async def test_reloads_after_a_restart_under_twenty_hz_state_keep_the_capture(tmp_path):
    """P1: 10 reloads of a capturing page after one Restart, with the session held in THINKING.
    Every reload: the first message is ``astra.session.cleared`` with ``replay: true``, the capture
    is still running after ``connected``, and no ``ClientDisconnected`` was submitted."""
    rows: list[dict[str, Any]] = []
    version: dict[str, Any] = {}
    async with started(tmp_path) as (live, url):
        asking: asyncio.Task[None] | None = None
        ticker: asyncio.Future[None] | None = None
        try:
            async with tab(url, init_js=INIT_JS) as (chrome, page):
                version = await asyncio.to_thread(chrome._cmd, "Browser.getVersion")
                await page.connected()
                await page.until(RESTART_ENABLED_JS, "the Restart button is mounted and enabled")
                await restart(page, live)
                await live.until(
                    lambda: len(cleared_on_wire(live)) == 1, "the server published the cleared"
                )
                asking, _turn = await hold_an_open_turn(live, QUESTION)
                publish_on_every_open(live)
                ticker = asyncio.ensure_future(twenty_hz(live))

                for n in range(RELOADS):
                    hellos = live.hellos
                    departed = len(live.machine.events(E.ClientDisconnected))
                    t0 = time.monotonic()
                    await asyncio.to_thread(chrome.reload)
                    await live.until(lambda h=hellos: live.hellos > h, f"reload {n}: the hello")
                    gap_ms = (time.monotonic() - t0) * 1000
                    await page.until(FIRST_RECORDED_JS, f"reload {n}: a first message recorded")
                    await page.connected()
                    await asyncio.sleep(SETTLE_S)
                    rec = await page.evaluate("window.__t0166")
                    shown = await state(page)
                    rows.append(
                        {
                            "reload": n,
                            "gap_ms": round(gap_ms),
                            "sockets": len(rec["sockets"]),
                            "first": rec["sockets"][0]["first"],
                            "banner": shown["banner_visible"],
                            "mute_disabled": shown["mute_disabled"],
                            "capture_size": shown["capture_size"],
                            "departed": len(live.machine.events(E.ClientDisconnected)) - departed,
                            "machine": live.session.state.state.value,
                        }
                    )
        finally:
            if ticker is not None:
                ticker.cancel()
                with contextlib.suppress(BaseException):
                    await ticker
            if asking is not None:
                await release(live, asking)

    regime = (
        f"Chrome {version.get('product')}, loadavg {os.getloadavg()}, "
        f"gaps (ms) {[r['gap_ms'] for r in rows]}"
    )
    assert len(rows) == RELOADS, f"premise: {RELOADS} reloads ran; {rows}; {regime}"
    wrong_first = [
        r for r in rows if r["first"] != {"type": "astra.session.cleared", "replay": True}
    ]
    assert not wrong_first, (
        "P1 / H2 / T-0162 D-3: after a Restart, a reloaded page's first message ever is the "
        "replayed astra.session.cleared; a live message reached it ahead of its hello reply in "
        f"{len(wrong_first)}/{RELOADS} reloads: {wrong_first}; {regime}"
    )
    stopped = [r for r in rows if not r["banner"] or r["mute_disabled"] or r["capture_size"] == ""]
    assert not stopped, (
        "P1 / T-0162 D-3: the reloaded page keeps its ?source=canvas capture after connecting; "
        f"stopped in {len(stopped)}/{RELOADS}: {stopped}; {regime}"
    )
    left = [r for r in rows if r["departed"]]
    assert not left, f"P1 / G1: a reload is not a departure; {left}; {regime}"
