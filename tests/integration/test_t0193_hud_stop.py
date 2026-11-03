"""The HUD's "Interrupt Gearvis" button in headless Chrome (docs/tasks/T-0193.md group B: B-1..B-7).

*Tests to write*, B (Chrome over CDP, ``HudStub``, the T-0181 C rig shape), plus the orchestrator's additions for
group B (2026-10-29): the label is exactly "Interrupt Gearvis" and the title exact (H-T193-4); the stale and
replayed turn-state rule; ``astra.session.cleared``; Escape on the body, in the ask box, in Settings and in the
Restart confirm sends no stop (and still closes Settings and the confirm); no header overflow at 1100 and 1440 px
in both themes; no request leaves loopback. Reviewer A's note on group A: after a reconnect the button takes its
``turn_id`` only from the new server's hello reply.

Regime: the box's Chrome headless over CDP, one module fixture that drives the page through every phase and records
probes (each test asserts on one), the stub Conductor ``tests/fakes/hud_server.HudStub`` (no Session: every
``astra.turn.state`` is built from the real ``AstraTurnState`` model and pushed; the stub never echoes a stop).
Clicks are real mouse events (``Input.dispatchMouseEvent``), keys go through the browser's key pipeline
(``Input.dispatchKeyEvent``). A "Share" is a real click on ``#share-button`` with ``getDisplayMedia`` replaced by
T-0177 C's init script. No measured number beyond contrast ratios and pixel boxes.

**The DOM contract this module pins** (B-1 names the id; H-T193-4 the copy and the place, written out by hand):

=====================================  ==================================================================
``header #stop-button``                a ``<button>``; hidden before the first ``astra.turn.state``; then
                                       shown, reading "Interrupt Gearvis", titled as H-T193-4, enabled iff
                                       the last echo is THINKING, SPEAKING or PAUSED; right after
                                       ``#turn-chip``
``#settings-popover``                  Settings (T-0162 B-6): ``hidden`` while closed
``[data-restart-confirm]``             the Restart confirm (T-0162 D-1): ``hidden`` while closed
=====================================  ==================================================================

Readings (stated, not guessed silently; the node half, ``tests/hud/stop.test.mjs``, lists R1-R4 in full):
- R1 "stale": a state for a turn this page has already taken that is not the last echo's turn moves nothing,
  live or replayed (``hud/src/state.js``'s chip rule).
- R2 after a click the next accepted state re-arms the button if it is THINKING/SPEAKING/PAUSED, also for the
  same turn.
- R3 a live ``astra.session.cleared`` disarms the button until a new turn's state.
- R4 a dropped socket disarms it; the reconnected page's first ``astra.turn.state`` re-arms it even when a
  restarted server reuses a turn id the page saw before.
- "Escape in the ask box" runs in IDLE: the ask box is enabled only in IDLE (T-0039), so it cannot hold the focus
  while the button is enabled. The page still holds an echoed ``turn_id`` there, so a key handler that sends a
  stop without asking the model is caught; the body, Settings, the Restart confirm and the button itself are
  pressed with the button enabled.
- "Every button state" (B-6) is measured as enabled (SPEAKING) and disabled (INTERRUPTED, the widest chip),
  each in the widest header (a capture running, so Share reads "Sharing screen"; the mic muted, so its badge
  shows). Hidden is BASE's header, which ``test_t0181_hud_mic``'s overflow rows already measure.
- "Contrast as the existing header buttons" is read as T-0056's text floor (4.5:1, 3:1 for large text) through
  ``test_hud_chrome``'s ``measure_js``/``_contrast_report``, enabled and disabled, both themes — the floor every
  header button is held to.
- B-RV1 (review ruling): the widest header above plus all four assists on (``#assist-header-badge``), in SPEAKING and
  ENDPOINTING, at 1100, 1250, 1280, 1300, 1320, 1366 and 1440 px, both themes; "the Settings gear fully inside the
  header" is ``#settings-button``'s bounding box within the header's (0.5 px slack), all four edges.
- "No header overflow" is ``test_t0181_hud_mic``'s reading: ``header.scrollWidth <= header.clientWidth`` and every
  rendered header descendant inside the header's edges (0.5 px slack), and the page does not scroll sideways.
"""

from __future__ import annotations

import json
import string
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import pytest

from local_astra.contracts.perception import AssistFlags
from local_astra.contracts.protocol import AstraAssistState, AstraSessionCleared, AstraTurnState
from local_astra.contracts.turn import TurnState
from local_astra.hud_static import HudStaticServer
from tests.fakes.hud_server import HudStub, mic_state
from tests.integration.test_hud_chrome import _contrast_report, choose_theme_js, measure_js
from tests.integration.test_hud_headless import CHROME, HUD, requires_chrome
from tests.integration.test_hud_redesign import _panels_line, _Recorder
from tests.integration.test_t0177_hud_vision import INIT_JS, _click

pytestmark = pytest.mark.integration

# --- the expected copy, by hand from H-T193-4 ------------------------------------------------------------

STOP_TEXT = "Interrupt Gearvis"
STOP_TITLE = "Interrupt Gearvis: it stops talking and stops what it is doing in the browser"

#: The page's own client->server types at BASE 611068bb that this script can provoke (hello, frames, the mic
#: mute). B-5: the only new one is astra.turn.stop.
BASE_SENT = {"astra.hello", "astra.frame", "astra.mic.mute"}

STOP = "#stop-button"

PROBE_JS = """
(() => {
  const vis = (el) => !!el && el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true});
  const norm = (s) => String(s || '').replace(/\\s+/g, ' ').trim();
  const b = document.getElementById('stop-button');
  const chip = document.getElementById('turn-chip');
  const pop = document.getElementById('settings-popover');
  const conf = document.querySelector('[data-restart-confirm]');
  const muteBtn = document.getElementById('mute-button');
  let next = chip ? chip.nextElementSibling : null;
  return {
    stop: !!b, stop_tag: b ? b.tagName.toLowerCase() : null, stop_in_header: !!(b && b.closest('header')),
    stop_visible: vis(b), stop_text: b ? norm(b.innerText || b.textContent) : null,
    stop_title: b ? b.getAttribute('title') : null, stop_disabled: b ? b.disabled : null,
    stop_keys: b ? b.getAttribute('aria-keyshortcuts') : null,
    after_chip: next ? (next.id || next.tagName.toLowerCase()) : null,
    chip: chip ? norm(chip.textContent) : null,
    settings_open: !!pop && !pop.hidden, confirm_open: !!conf && !conf.hidden,
    capturing: !!muteBtn && !muteBtn.disabled,
    active: document.activeElement ? (document.activeElement.id || document.activeElement.tagName.toLowerCase()) : null,
    theme: document.documentElement.getAttribute('data-theme'),
  };
})()
"""

#: B-6: the header's horizontal overflow and every rendered descendant outside its edges (T-0181 B4's probe).
OVERFLOW_JS = """
(() => {
  const h = document.querySelector('header');
  const hr = h.getBoundingClientRect();
  const out = [];
  for (const el of h.querySelectorAll('*')) {
    if (!el.checkVisibility || !el.checkVisibility()) continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    if (r.left < hr.left - 0.5 || r.right > hr.right + 0.5) {
      out.push((el.id ? '#' + el.id : el.tagName.toLowerCase() + '.' + el.className) + ' [' + r.left.toFixed(1) + ', ' + r.right.toFixed(1) + ']');
    }
  }
  const vis = (sel) => { const e = document.querySelector(sel); return !!e && e.checkVisibility(); };
  const b = document.getElementById('stop-button');
  return {scroll: h.scrollWidth, client: h.clientWidth, header: [hr.left, hr.right], outside: out,
          doc_scroll: document.documentElement.scrollWidth, inner: window.innerWidth,
          stop_visible: vis('#stop-button'), stop_disabled: b ? b.disabled : null,
          stop_text: ((b || {}).textContent || '').trim(),
          badge_visible: vis('header [data-mic-badge]'),
          share_text: ((document.getElementById('share-button') || {}).textContent || '').trim(),
          chip: ((document.getElementById('turn-chip') || {}).textContent || '').trim()};
})()
"""

#: B-RV1 (review ruling, docs/tasks/T-0193.md "Group B review rulings"): the widths the assists-on rows run at.
RV1_WIDTHS = (1100, 1250, 1280, 1300, 1320, 1366, 1440)

#: B-RV1: OVERFLOW_JS's measures plus the assist badge and the Settings gear's box against the header's.
RV1_JS = (
    "(() => { const o = "
    + OVERFLOW_JS.strip()
    + """;
  const h = document.querySelector('header').getBoundingClientRect();
  const a = document.getElementById('assist-header-badge');
  const g = document.getElementById('settings-button');
  const gr = g ? g.getBoundingClientRect() : null;
  o.assist_visible = !!a && a.checkVisibility();
  o.assist_text = a ? (a.textContent || '').trim() : null;
  o.gear = gr ? [gr.left, gr.right, gr.top, gr.bottom] : null;
  o.gear_visible = !!g && g.checkVisibility({checkOpacity: true, checkVisibilityCSS: true});
  o.header_box = [h.left, h.right, h.top, h.bottom];
  o.chip_state = (document.getElementById('turn-chip') || {getAttribute: () => null}).getAttribute('data-turn-state');
  o.width = window.innerWidth;
  return o; })()"""
)

CONTRAST = [("stop button", STOP, None)]

CAPTURING_JS = (
    "!document.getElementById('mute-button').disabled && "
    "/^\\d+x\\d+$/.test((document.getElementById('capture-size').textContent || '').trim())"
)
STOP_VISIBLE_JS = (
    "(() => { const b = document.getElementById('stop-button');"
    " return !!b && b.checkVisibility({checkOpacity: true, checkVisibilityCSS: true}); })()"
)


def _chip_is(state: str) -> str:
    return (
        "(() => { const c = document.getElementById('turn-chip');"
        f" return !!c && c.getAttribute('data-turn-state') === {json.dumps(state)}; }})()"
    )


def ts(state: str, turn_id: str, *, replay: bool = False) -> AstraTurnState:
    """``astra.turn.state`` through the real model (the shape ``Session.turn_state_message`` publishes)."""
    thinking = 150 if state == "THINKING" else None
    return AstraTurnState(
        state=TurnState(state), turn_id=turn_id, thinking_ms=thinking, replay=replay
    )


@dataclass
class Run:
    error: str | None = None
    probes: dict[str, Any] = field(default_factory=dict)
    stops: dict[str, list[str]] = field(default_factory=dict)
    mutes: dict[str, int] = field(default_factory=dict)
    contrast: dict[str, Any] = field(default_factory=dict)
    overflow: dict[str, Any] = field(default_factory=dict)
    rv1: dict[str, Any] = field(default_factory=dict)
    keys: dict[str, Any] = field(default_factory=dict)
    all_types: list[str] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _mounted(chrome: Any, timeout_s: float = 25.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and _panels_line(chrome.console[chrome._mark :]) is None:
        chrome.drain(0.1)
    return _panels_line(chrome.console[chrome._mark :]) is not None


def _dblclick(chrome: Any, selector: str) -> bool:
    """A real double click: two press/release pairs at one point, clickCount 1 then 2."""
    got = chrome.evaluate(
        "(() => { const e = document.querySelector(" + json.dumps(selector) + ");"
        " if (!e) return null; const r = e.getBoundingClientRect();"
        " return [r.left + r.width / 2, r.top + r.height / 2]; })()"
    )
    if not isinstance(got, list):
        return False
    x, y = float(got[0]), float(got[1])
    chrome._cmd("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
    for count in (1, 2):
        for kind in ("mousePressed", "mouseReleased"):
            chrome._cmd(
                "Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=count
            )
    return True


def _drive(result: Run, chrome: Any, stub: HudStub, port: int) -> None:
    ws = urllib.parse.quote(stub.url, safe="")
    base = f"http://127.0.0.1:{port}/"
    probes = result.probes

    def stops() -> list[str]:
        return [str(m.turn_id) for m in stub.turn_stops()]

    def raw_stop_count() -> int:
        return stub.received_types().count("astra.turn.stop")

    def probe(name: str) -> dict[str, Any]:
        chrome.evaluate(
            "new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))"
        )
        got = chrome.evaluate(PROBE_JS)
        probes[name] = got if isinstance(got, dict) else {"__probe__": got}
        return probes[name]

    def step(name: str, action: Any, *, expect: int = 0, settle: float = 0.8) -> None:
        """Run ``action``; record the stops (decoded turn ids) and the mic mutes it caused."""
        before, raw_before = len(stops()), raw_stop_count()
        mutes_before = len(stub.mic_mutes())
        action()
        if expect:
            stub.wait_for(lambda s: len(s.turn_stops()) >= before + expect, 6)
        chrome.drain(settle)
        result.stops[name] = stops()[before:]
        # A malformed stop is not in turn_stops(); the raw count catches it.
        result.stops[name + "__raw"] = ["x"] * (raw_stop_count() - raw_before)
        result.mutes[name] = len(stub.mic_mutes()) - mutes_before

    def push(msg: Any) -> None:
        stub.push(msg)
        chrome.drain(0.25)

    def blur() -> None:
        chrome.evaluate(
            "document.activeElement && document.activeElement.blur && document.activeElement.blur(); 1"
        )

    def focus(selector: str) -> bool:
        return bool(
            chrome.evaluate(
                "(() => { const e = document.querySelector(" + json.dumps(selector) + ");"
                " if (!e) return false; e.focus(); return document.activeElement === e; })()"
            )
        )

    def escape() -> None:
        chrome.key("Escape", code=27)
        chrome.drain(0.15)

    def state(msg: AstraTurnState) -> None:
        push(msg)
        chrome.wait_until(_chip_is(msg.state.value), 5)

    chrome.add_init_script(INIT_JS)
    chrome.viewport(1440, 900)
    chrome.evaluate(choose_theme_js("dark"))
    hellos = stub.received_types().count("astra.hello")
    chrome.navigate(f"{base}?ws={ws}")
    if not _mounted(chrome):
        raise RuntimeError("the page never logged 'panels mounted:'")
    if not stub.wait_for(lambda s: s.received_types().count("astra.hello") > hellos, 15):
        raise RuntimeError(f"no astra.hello; the stub saw {stub.received_types()!r}")
    stub.wait_for_client(10)
    chrome.wait_until(
        "(() => { const r = document.getElementById('restart-button'); return !!r && !r.disabled; })()",
        10,
    )
    chrome.evaluate(choose_theme_js("dark"))
    probe("load")

    # B-1: other messages before any turn state show nothing; Escape and a sweep before a state send nothing.
    push(mic_state(False))
    push(
        json.dumps(
            {"type": "astra.transcript", "turn_id": "t1", "text": "how many", "final": False}
        )
    )
    blur()
    step("keys_before_state", lambda: (escape(), chrome.key("s", text="s")), settle=0.5)
    probe("before_state")

    # B-1: the first state, LISTENING: shown, disabled; a real click sends nothing.
    state(ts("LISTENING", "t1"))
    chrome.wait_until(STOP_VISIBLE_JS, 5)
    probe("listening")
    step("click_listening", lambda: _click(chrome, STOP), settle=0.8)
    state(ts("ENDPOINTING", "t1"))
    probe("endpointing")
    step("click_endpointing", lambda: _click(chrome, STOP), settle=0.5)
    state(ts("IDLE", "t1"))
    probe("idle")
    step("click_idle", lambda: _click(chrome, STOP), settle=0.8)

    # B-1/B-2: THINKING and SPEAKING arm it; a real click in SPEAKING sends exactly one stop for t2.
    state(ts("THINKING", "t2"))
    probe("thinking")
    state(ts("SPEAKING", "t2"))
    probe("speaking")
    step("click_speaking", lambda: _click(chrome, STOP), expect=1, settle=1.0)
    probe("after_click")
    # B-2 adversarial: the echo is withheld; more clicks and a double click send nothing.
    step(
        "clicks_withheld",
        lambda: (_click(chrome, STOP), _click(chrome, STOP), _dblclick(chrome, STOP)),
        settle=1.0,
    )
    probe("after_clicks_withheld")

    # B-2/R2: the next echo (PAUSED, same turn) re-arms it; a real double click sends exactly one stop.
    state(ts("PAUSED", "t2"))
    probe("paused")
    step("dblclick_paused", lambda: _dblclick(chrome, STOP), expect=1, settle=1.0)
    probe("after_dblclick")
    state(ts("INTERRUPTED", "t2"))
    probe("interrupted")
    step("click_interrupted", lambda: _click(chrome, STOP), settle=0.5)
    state(ts("IDLE", "t2"))
    probe("idle_t2")

    # B-3: a stale state for the earlier turn t1 (live, then replayed) re-enables nothing. The chip does not
    # move either (state.js's rule), so the wait is a fixed drain.
    push(ts("SPEAKING", "t1"))
    chrome.drain(0.4)
    probe("stale_live")
    push(ts("THINKING", "t1", replay=True))
    chrome.drain(0.4)
    probe("stale_replay")
    step("click_stale", lambda: _click(chrome, STOP), settle=0.8)

    # B-4: Escape (and every letter and digit) with the button armed, on the body, on the button itself, in
    # Settings and in the Restart confirm: no stop, and Settings and the confirm still close.
    state(ts("THINKING", "t3"))
    state(ts("SPEAKING", "t3"))
    probe("armed_t3")
    keys = result.keys

    def esc_body() -> None:
        blur()
        keys["body_focused"] = bool(chrome.evaluate("document.activeElement === document.body"))
        escape()

    step("esc_body", esc_body, settle=0.6)

    def esc_button() -> None:
        keys["button_focused"] = focus(STOP)
        escape()

    step("esc_button", esc_button, settle=0.6)
    blur()

    def esc_settings() -> None:
        _click(chrome, "#settings-button")
        chrome.wait_until("!document.getElementById('settings-popover').hidden", 5)
        keys["settings_opened"] = probe("settings_open")["settings_open"]
        escape()

    step("esc_settings", esc_settings, settle=0.6)
    probe("settings_after_escape")

    def esc_restart() -> None:
        _click(chrome, "#restart-button")
        chrome.wait_until("!document.querySelector('[data-restart-confirm]').hidden", 5)
        keys["confirm_opened"] = probe("confirm_open")["confirm_open"]
        escape()

    step("esc_restart", esc_restart, settle=0.6)
    probe("confirm_after_escape")

    def sweep() -> None:
        blur()
        for ch in string.ascii_lowercase + string.digits:
            chrome.key(ch, text=ch)
        chrome.key("Backspace", code=8)
        chrome.key("Delete", code=46)
        chrome.drain(0.2)

    step("sweep_body", sweep, settle=0.8)
    probe("after_keys")
    # Anti-vacuity: after all those keys the button still works by click, for t3.
    step("click_after_keys", lambda: _click(chrome, STOP), expect=1, settle=0.8)

    # B-4: Escape in the ask box (IDLE: the only state the box is enabled in), with "stop" typed into it.
    state(ts("IDLE", "t3"))
    chrome.wait_until(
        "(() => { const i = document.querySelector('[data-ask-input]'); return !!i && !i.disabled; })()",
        5,
    )

    def esc_ask() -> None:
        keys["ask_focused"] = focus("[data-ask-input]")
        chrome.type_text("stop")
        escape()
        keys["ask_value"] = chrome.evaluate(
            "(document.querySelector('[data-ask-input]') || {}).value"
        )

    step("esc_ask", esc_ask, settle=0.8)
    chrome.evaluate(
        "(() => { const i = document.querySelector('[data-ask-input]'); if (i) { i.value = '';"
        " i.dispatchEvent(new Event('input', {bubbles: true})); } })()"
    )
    blur()

    # B-7: the mic and the stop share nothing. A capture runs; a stop click leaves it running and mutes
    # nothing; a mic click sends one mute and no stop; Stop sharing still stops by click and sends no stop.
    _click(chrome, "#share-button")
    chrome.wait_until(CAPTURING_JS, 10)
    state(ts("SPEAKING", "t4"))
    step("click_while_sharing", lambda: _click(chrome, STOP), expect=1, settle=0.8)
    probe("sharing_after_stop")
    step(
        "mic_click",
        lambda: (_click(chrome, "#mic-button"), stub.wait_for_mic_mutes(1, 5)),
        settle=0.8,
    )

    # B-6: the widest header (sharing, the mic muted with its badge), 1100 and 1440 px, both themes, the button
    # enabled (SPEAKING) and disabled (INTERRUPTED); contrast enabled and disabled.
    push(mic_state(True))
    chrome.wait_until(
        "(() => { const b = document.querySelector('header [data-mic-badge]'); return !!b && b.checkVisibility(); })()",
        5,
    )
    for label, msg in (("enabled", ts("SPEAKING", "t5")), ("disabled", ts("INTERRUPTED", "t5"))):
        state(msg)
        for width in (1100, 1440):
            chrome.viewport(width, 900)
            for theme in ("dark", "light"):
                chrome.evaluate(choose_theme_js(theme))
                time.sleep(0.3)
                result.overflow[f"{width}_{theme}_{label}"] = chrome.evaluate(OVERFLOW_JS) or {}
                if width == 1440:
                    result.contrast[f"{theme}_{label}"] = (
                        chrome.evaluate(measure_js(CONTRAST)) or {}
                    )

    # B-RV1: the same widest header plus all four assists on (#assist-header-badge), SPEAKING and ENDPOINTING,
    # seven widths, both themes. Fresh turn ids, so no chip rule can call these states stale.
    all_on = AssistFlags(page_text=True, pdf_text=True, cells=True, at_spi=True)
    push(AstraAssistState(flags=all_on, badge=all_on.badge()))
    chrome.wait_until(
        "(() => { const b = document.getElementById('assist-header-badge'); return !!b && b.checkVisibility(); })()",
        5,
    )
    for st, tid in (("SPEAKING", "t5s"), ("ENDPOINTING", "t5e")):
        state(ts(st, tid))
        for width in RV1_WIDTHS:
            chrome.viewport(width, 900)
            for theme in ("dark", "light"):
                chrome.evaluate(choose_theme_js(theme))
                chrome.evaluate(
                    "new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))"
                )
                time.sleep(0.05)
                result.rv1[f"{st}_{width}_{theme}"] = chrome.evaluate(RV1_JS) or {}
    off = AssistFlags()
    push(AstraAssistState(flags=off, badge=off.badge()))
    chrome.viewport(1440, 900)
    chrome.evaluate(choose_theme_js("dark"))
    _click(chrome, "#mute-button")
    chrome.wait_until("document.getElementById('mute-button').disabled", 5)

    # R3: a live astra.session.cleared disarms the button; a late state for the ended turn is stale.
    state(ts("SPEAKING", "t6"))
    probe("armed_t6")
    push(AstraSessionCleared(resets=1, ended_turn_id="t6", memory_forgotten=0))
    chrome.drain(0.6)
    probe("after_cleared")
    step("click_after_cleared", lambda: _click(chrome, STOP), settle=0.6)
    state(ts("THINKING", "t7"))
    probe("armed_t7")

    # R4 (reviewer A): a dropped socket disarms it; only the new hello's state re-arms it, even for a reused id.
    state(ts("SPEAKING", "t8"))
    probe("armed_t8")
    hellos = stub.received_types().count("astra.hello")
    stub.drop_client()
    stub.wait_for(lambda s: s.received_types().count("astra.hello") > hellos, 20)
    stub.wait_for_client(10)
    chrome.drain(0.5)
    probe("reconnected_no_state")
    step("click_reconnected_no_state", lambda: _click(chrome, STOP), settle=0.8)
    push(ts("SPEAKING", "t1"))  # a restarted server's first turn reuses an id this page saw before
    chrome.drain(0.6)
    probe("reconnected")
    step("click_reconnected", lambda: _click(chrome, STOP), expect=1, settle=0.8)

    result.all_types = stub.received_types()


@pytest.fixture(scope="module")
def run() -> Any:
    if CHROME is None:  # pragma: no cover - requires_chrome is the real guard
        yield Run(error="no chrome")
        return
    result = Run()
    started = time.monotonic()
    stub = HudStub([]).start()
    server = HudStaticServer(HUD, 0)
    try:
        with _Recorder(CHROME) as chrome:
            chrome._cmd("Runtime.enable")
            chrome._cmd("Network.enable")
            try:
                _drive(result, chrome, stub, server.port)
            finally:
                chrome.drain(0.3)
                result.requests = list(chrome.requests)
                result.exceptions = list(chrome.exceptions)
    except Exception as exc:  # a broken harness is data, not an error inside a test
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        server.close()
        stub.stop()
    result.seconds = time.monotonic() - started
    print(f"[T-0193 B] headless session wall time {result.seconds:.1f} s")
    yield result


def _p(run: Run, name: str) -> dict[str, Any]:
    assert run.error is None, f"the T-0193 B headless session failed: {run.error}"
    got = run.probes.get(name)
    assert isinstance(got, dict) and "__probe__" not in got, (
        f"probe {name!r} missing or broken: {got!r}"
    )
    return got


def _shown(p: dict[str, Any], *, disabled: bool, why: str) -> None:
    assert p["stop"] and p["stop_tag"] == "button" and p["stop_in_header"], (
        f"{why}: a <button id=stop-button> in the header: {p}"
    )
    assert p["stop_visible"], (
        f"{why}: B-1: the button is rendered once an astra.turn.state arrived: {p}"
    )
    assert p["stop_text"] == STOP_TEXT, (
        f"{why}: H-T193-4 label {p['stop_text']!r}, want {STOP_TEXT!r}"
    )
    assert p["stop_title"] == STOP_TITLE, f"{why}: H-T193-4 title {p['stop_title']!r}"
    assert p["stop_disabled"] is disabled, (
        f"{why}: B-1: disabled should be {disabled}, the page has {p['stop_disabled']}"
    )


def _sent(run: Run, name: str) -> list[str]:
    assert run.error is None, run.error
    assert name in run.stops, f"step {name!r} was not recorded"
    assert run.stops[name] == [] or len(run.stops[name + "__raw"]) == len(run.stops[name]), (
        f"{name}: a malformed astra.turn.stop went out: {run.stops[name + '__raw']}"
    )
    return run.stops[name]


def _none_sent(run: Run, name: str, why: str) -> None:
    assert _sent(run, name) == [] and run.stops[name + "__raw"] == [], (
        f"{why}: sent astra.turn.stop {run.stops[name]} (raw {len(run.stops[name + '__raw'])})"
    )


# --- B-1: visibility --------------------------------------------------------------------------------


@requires_chrome
def test_b1_hidden_before_the_first_turn_state(run):
    """B-1: ``#stop-button`` exists in the header and is not rendered before an ``astra.turn.state``, whatever else
    arrives (a page with no button fails at the "exists" line, not by passing the "not shown" one)."""
    for name in ("load", "before_state"):
        p = _p(run, name)
        assert p["stop"] and p["stop_tag"] == "button" and p["stop_in_header"], (
            f"B-1 ({name}): a <button id=stop-button> in the header: {p}"
        )
        assert not p["stop_visible"], (
            f"B-1 ({name}): the button shows before any astra.turn.state: {p}"
        )
    _none_sent(run, "keys_before_state", "B-4: Escape and a letter before a state")


@requires_chrome
def test_b1_h_t193_4_right_after_the_turn_chip_with_the_exact_label_and_title(run):
    p = _p(run, "speaking")
    assert p["after_chip"] == "stop-button", (
        f"H-T193-4: the button sits right after #turn-chip; the chip's next sibling is {p['after_chip']!r}"
    )
    _shown(p, disabled=False, why="SPEAKING")
    assert p["stop_keys"] is None, f"B-4: no aria-keyshortcuts ({p['stop_keys']!r})"


@requires_chrome
@pytest.mark.parametrize(
    "probe_name, disabled",
    [
        ("listening", True),
        ("endpointing", True),
        ("idle", True),
        ("thinking", False),
        ("speaking", False),
        ("paused", False),
        ("interrupted", True),
        ("idle_t2", True),
    ],
)
def test_b1_enabled_iff_thinking_speaking_or_paused(run, probe_name, disabled):
    _shown(_p(run, probe_name), disabled=disabled, why=probe_name)


@requires_chrome
def test_b1_a_click_in_idle_listening_endpointing_or_interrupted_sends_nothing(run):
    # Premise: the clicks landed on a shown, disabled button (a page with no button sends nothing vacuously).
    for probe_name in ("listening", "endpointing", "idle", "interrupted"):
        _shown(_p(run, probe_name), disabled=True, why=f"premise ({probe_name})")
    for name in ("click_listening", "click_endpointing", "click_idle", "click_interrupted"):
        _none_sent(run, name, f"B-1: a real click on the disabled button ({name})")


# --- B-2: the click ---------------------------------------------------------------------------------


@requires_chrome
def test_b2_a_click_in_speaking_sends_one_stop_with_the_echoed_turn_id(run):
    _shown(_p(run, "speaking"), disabled=False, why="premise")
    assert _sent(run, "click_speaking") == ["t2"], (
        f"B-2: one real click in SPEAKING t2 sends exactly one astra.turn.stop for t2; got {run.stops['click_speaking']}"
    )
    _shown(
        _p(run, "after_click"),
        disabled=True,
        why="B-2: after the click, until the next astra.turn.state",
    )


@requires_chrome
def test_b2_with_the_echo_withheld_further_clicks_send_nothing(run):
    assert _sent(run, "click_speaking") == ["t2"], "premise: the first click sent its stop"
    _none_sent(
        run,
        "clicks_withheld",
        "B-2 adversarial: two clicks and a double click with the echo withheld",
    )
    _shown(_p(run, "after_clicks_withheld"), disabled=True, why="B-2: still no echo")


@requires_chrome
def test_b2_the_next_echo_re_arms_it_and_a_double_click_sends_once(run):
    _shown(_p(run, "paused"), disabled=False, why="R2: the PAUSED echo re-arms the button")
    assert _sent(run, "dblclick_paused") == ["t2"], (
        f"B-2: a real double click sends exactly one stop; got {run.stops['dblclick_paused']}"
    )
    _shown(_p(run, "after_dblclick"), disabled=True, why="after the double click")


# --- B-3: only the echo moves it -------------------------------------------------------------------------


@requires_chrome
def test_b3_a_stale_or_replayed_state_for_an_earlier_turn_does_not_re_enable_it(run):
    _shown(_p(run, "idle_t2"), disabled=True, why="premise: IDLE t2")
    _shown(_p(run, "stale_live"), disabled=True, why="B-3: a late live SPEAKING for t1")
    _shown(_p(run, "stale_replay"), disabled=True, why="B-3: a replayed THINKING for t1")
    _none_sent(run, "click_stale", "B-3: a click after the stale states")


@requires_chrome
def test_r3_a_live_session_cleared_disarms_it_until_a_new_turn(run):
    _shown(_p(run, "armed_t6"), disabled=False, why="premise: SPEAKING t6")
    _shown(
        _p(run, "after_cleared"),
        disabled=True,
        why="R3: after a live astra.session.cleared ending t6",
    )
    _none_sent(run, "click_after_cleared", "R3: a click after the reset")
    _shown(_p(run, "armed_t7"), disabled=False, why="R3: the next turn's THINKING re-arms it")


@requires_chrome
def test_r4_after_a_reconnect_only_the_new_hellos_state_arms_it(run):
    """Reviewer A's note on group A: the button takes its ``turn_id`` only from the new server's hello reply. The
    old socket's echo was SPEAKING t8; after the drop and the new hello, a click before any new state sends
    nothing, and the restarted server's SPEAKING t1 (an id the page saw on the old server) arms it for t1."""
    _shown(_p(run, "armed_t8"), disabled=False, why="premise: SPEAKING t8")
    _shown(_p(run, "reconnected_no_state"), disabled=True, why="R4: reconnected, no state yet")
    _none_sent(
        run, "click_reconnected_no_state", "R4: a click before the new server's state (no stale t8)"
    )
    _shown(_p(run, "reconnected"), disabled=False, why="R4: the new server's SPEAKING t1")
    assert _sent(run, "click_reconnected") == ["t1"], (
        f"R4: the stop names the new server's turn; got {run.stops['click_reconnected']}"
    )


# --- B-4: no shortcut ------------------------------------------------------------------------------------


@requires_chrome
def test_b4_escape_on_the_body_the_button_settings_and_the_confirm_sends_no_stop(run):
    _shown(
        _p(run, "armed_t3"),
        disabled=False,
        why="premise: the button is armed while the keys are pressed",
    )
    k = run.keys
    assert k.get("body_focused") and k.get("button_focused"), (
        f"premise: the focus targets held: {k}"
    )
    assert k.get("settings_opened") and k.get("confirm_opened"), (
        f"premise: Settings and the Restart confirm opened: {k}"
    )
    for name in ("esc_body", "esc_button", "esc_settings", "esc_restart"):
        _none_sent(run, name, f"B-4: Escape ({name})")
    _none_sent(run, "sweep_body", "B-4: a letter, digit, Backspace or Delete on the body")
    _shown(_p(run, "after_keys"), disabled=False, why="B-4: no key disarmed the button")
    assert _sent(run, "click_after_keys") == ["t3"], (
        f"anti-vacuity: the armed button still sends by click after the keys; got {run.stops['click_after_keys']}"
    )


@requires_chrome
def test_b4_escape_still_closes_settings_and_the_restart_confirm(run):
    _shown(
        _p(run, "armed_t3"),
        disabled=False,
        why="premise: the button was armed while Escape was pressed",
    )
    assert _p(run, "settings_open")["settings_open"], "premise: Settings opened"
    assert not _p(run, "settings_after_escape")["settings_open"], (
        "B-4: Escape still closes Settings"
    )
    assert _p(run, "confirm_open")["confirm_open"], "premise: the Restart confirm opened"
    assert not _p(run, "confirm_after_escape")["confirm_open"], (
        "B-4: Escape still closes the Restart confirm"
    )
    assert "astra.session.reset" not in run.all_types, (
        "Escape in the confirm never confirms a reset"
    )


@requires_chrome
def test_b4_escape_in_the_ask_box_sends_no_stop(run):
    k = run.keys
    assert k.get("ask_focused"), f"premise: the ask box held the focus: {k}"
    assert k.get("ask_value") == "stop", f"premise: the keys landed in the ask box: {k}"
    # Premise: the page held an echoed turn id when Escape was pressed (the click just before sent one for t3).
    assert _sent(run, "click_after_keys") == ["t3"], (
        f"premise: the button had a turn to stop: {run.stops}"
    )
    _none_sent(run, "esc_ask", "B-4: 'stop' typed and Escape in the ask box")


# --- B-5: egress ---------------------------------------------------------------------------------------


@requires_chrome
def test_b5_the_only_new_message_the_page_sends_is_turn_stop(run):
    assert run.error is None, run.error
    sent = set(run.all_types)
    assert "astra.turn.stop" in sent, (
        f"premise: the page sent a stop (anti-vacuity): {sorted(sent)}"
    )
    assert sent - BASE_SENT == {"astra.turn.stop"}, (
        f"B-5: new message types {sorted(sent - BASE_SENT)}"
    )


@requires_chrome
def test_b5_no_request_leaves_loopback(run):
    assert run.error is None, run.error
    assert run.requests, "the CDP network log is empty, so this row would be vacuous"
    assert "astra.turn.stop" in run.all_types, (
        "anti-vacuity: the page sent a stop during the logged session"
    )
    assert any(u.rstrip("/").endswith("/src/stop.js") for u in run.requests), (
        f"anti-vacuity: the page loaded hud/src/stop.js, so its requests are in the log: {run.requests[:5]}…"
    )
    off = [
        u
        for u in run.requests
        if u.split("://", 1)[0] in {"http", "https", "ws", "wss"}
        and not any(u.split("://", 1)[1].startswith(h) for h in ("127.0.0.1", "localhost", "[::1]"))
    ]
    assert off == [], f"B-5: zero non-loopback requests; got {off}"


# --- B-6: layout and contrast ----------------------------------------------------------------------------


@requires_chrome
@pytest.mark.parametrize("width", [1100, 1440])
@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("label", ["enabled", "disabled"])
def test_b6_no_horizontal_overflow_in_the_header(run, width, theme, label):
    assert run.error is None, run.error
    o = run.overflow.get(f"{width}_{theme}_{label}") or {}
    assert o, f"premise: the header was measured at {width} px, {theme}, {label}"
    assert o.get("stop_visible") and o.get("stop_text") == STOP_TEXT, (
        f"premise: the button shows: {o}"
    )
    assert o.get("stop_disabled") is (label == "disabled"), f"premise: the button is {label}: {o}"
    assert o.get("badge_visible") and str(o.get("share_text", "")).startswith("Sharing"), (
        f"premise: the widest header (mic badge, 'Sharing …'): {o}"
    )
    assert o["scroll"] <= o["client"], (
        f"B-6: the header overflows ({o['scroll']} > {o['client']}) at {width} px, {theme}, {label}"
    )
    assert o["outside"] == [], (
        f"B-6: header elements outside the header at {width} px, {theme}, {label}: {o['outside']}"
    )
    assert o["doc_scroll"] <= o["inner"], f"B-6: the page scrolls sideways at {width} px: {o}"


@requires_chrome
@pytest.mark.parametrize("width", RV1_WIDTHS)
@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("turn", ["SPEAKING", "ENDPOINTING"])
def test_b_rv1_assists_on_header_fits_and_the_settings_gear_is_inside(run, width, theme, turn):
    """B-RV1 (review ruling, 2026-10-29): with all four assists on (``#assist-header-badge`` shown), a capture
    running ("Sharing …") and "Mic muted" shown, in SPEAKING (button enabled) and ENDPOINTING (disabled), the header
    does not overflow and the Settings gear lies wholly inside the header's box. Reviewer B measured overflow at
    dcd3b965 between ~1250 and ~1390 px (1300 px: 1326 > 1244 SPEAKING, 1346 > 1244 ENDPOINTING)."""
    assert run.error is None, run.error
    o = run.rv1.get(f"{turn}_{width}_{theme}") or {}
    assert o, f"premise: the header was measured at {width} px, {theme}, {turn}"
    where = f"{width} px, {theme}, {turn}"
    assert o.get("width") == width, f"premise: the viewport is {width} px: {o.get('width')}"
    assert o.get("chip_state") == turn, f"premise: the chip shows {turn}: {o.get('chip_state')}"
    assert o.get("stop_visible") and o.get("stop_disabled") is (turn == "ENDPOINTING"), (
        f"premise: the button shows, {'disabled' if turn == 'ENDPOINTING' else 'enabled'} ({where}): {o}"
    )
    assert o.get("assist_visible") and all(
        k in str(o.get("assist_text")) for k in ("dom", "pdf", "cells", "a11y")
    ), f"premise: the assist badge shows all four assists ({where}): {o}"
    assert o.get("badge_visible") and str(o.get("share_text", "")).startswith("Sharing"), (
        f"premise: the widest header (mic badge, 'Sharing …') ({where}): {o}"
    )
    assert o["scroll"] <= o["client"], (
        f"B-RV1: the header overflows ({o['scroll']} > {o['client']}) at {where}"
    )
    assert o["outside"] == [], (
        f"B-RV1: header elements outside the header at {where}: {o['outside']}"
    )
    assert o["doc_scroll"] <= o["inner"], f"B-RV1: the page scrolls sideways at {where}: {o}"
    gear, box = o.get("gear"), o.get("header_box")
    assert gear and box and o.get("gear_visible"), (
        f"premise: the Settings gear is rendered ({where}): {o}"
    )
    gl, gr, gt, gb = gear
    hl, hr, ht, hb = box
    assert gr > gl and gb > gt, f"premise: the gear has a box ({where}): {gear}"
    assert gl >= hl - 0.5 and gr <= hr + 0.5 and gt >= ht - 0.5 and gb <= hb + 0.5, (
        f"B-RV1: the Settings gear {gear} is not fully inside the header {box} at {where}"
    )


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("label", ["enabled", "disabled"])
def test_b6_the_button_text_clears_the_header_buttons_contrast_floor(run, theme, label):
    assert run.error is None, run.error
    measured = run.contrast.get(f"{theme}_{label}") or {}
    missing, vacuous, failures, _ = _contrast_report(measured)
    assert not missing and not vacuous, (
        f"B-6 ({theme}, {label}): the button was not measured: {missing} {vacuous}"
    )
    assert failures == [], (
        f"B-6 ({theme}, {label}): the button text is below T-0056's floor: {failures}"
    )


# --- B-7: the mic and Stop sharing are unchanged ------------------------------------------------------------


@requires_chrome
def test_b7_stop_and_mic_and_stop_sharing_do_not_share_anything(run):
    assert _sent(run, "click_while_sharing") == ["t4"], (
        f"premise: the stop click while sharing sent its stop; got {run.stops['click_while_sharing']}"
    )
    assert run.mutes.get("click_while_sharing") == 0, "B-7: the stop click sent no mic mute"
    assert _p(run, "sharing_after_stop")["capturing"], (
        "B-7: the stop click did not stop the capture"
    )
    assert run.mutes.get("mic_click") == 1, f"B-7: a mic click still sends one mute: {run.mutes}"
    _none_sent(run, "mic_click", "B-7: a mic click")
    for name, n in run.mutes.items():
        if name not in ("mic_click",):
            assert n == 0, f"B-7: step {name} sent {n} mic mutes"


@requires_chrome
def test_the_page_threw_nothing(run):
    assert run.error is None, run.error
    _shown(_p(run, "speaking"), disabled=False, why="premise: the session drove the button")
    assert run.exceptions == [], f"the page threw: {run.exceptions[:3]}"


@requires_chrome
def test_the_session_fits_the_module_budget(run):
    assert run.error is None, run.error
    _shown(_p(run, "reconnected"), disabled=False, why="premise: the session ran to its last phase")
    assert run.seconds < 120.0, f"the T-0193 B headless session took {run.seconds:.1f} s"
