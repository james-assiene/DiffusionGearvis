"""The HUD microphone control in headless Chrome (docs/tasks/T-0181.md group B: B1-B5, with B0 folded in).

*Tests to write*, Group B (Chrome via ``HudStub``, 1100/1440 px, both themes), as amended by the "Group A0
replan" knock-on edits and B3 (H-T181-4, James 2026-10-28: no keyboard shortcut for the mic, and the ``m``
shortcut for Stop sharing removed):

- nothing is shown before a state;
- a click changes nothing until the echo (adversarial: the stub withholds the echo);
- the badge and ``aria-pressed`` follow the echo;
- ``m`` anywhere sends nothing (the body with nothing focused, the ask box, the mic button, Stop sharing, the
  theme select, the vision combo; and every other letter and digit on the body) and stops no capture;
- "how many" typed in the ask box sends no mute;
- unavailable: disabled with the reason, and neither a click nor ``m`` sends anything;
- Stop sharing by click still works, and ``m`` does not trigger it;
- no header overflow at 1100 and 1440 px, both themes;
- no request off loopback (B5), and the only new message the page sends is ``astra.mic.mute``;
- (Group B gains ``hud/src/panels/transcript.js``) an empty final transcript line shows no bubble.

Regime: the box's Chrome headless over CDP, the T-0177 C rig shape (one module fixture drives the page
through every phase and records probes; each test asserts on one), the stub Conductor
``tests/fakes/hud_server.HudStub`` (no Session: every ``astra.mic.state`` is built by the fake's
``mic_state`` from A1's real model and pushed; the stub never echoes on its own). A "Share" is a real
mouse click on ``#share-button`` with ``getDisplayMedia`` replaced by T-0177 C's init script (a real
``canvas.captureStream``, so the capture is not synthetic). Clicks are real mouse events
(``Input.dispatchMouseEvent``), keys go through the browser's key pipeline (``Input.dispatchKeyEvent``).
No measured number beyond contrast ratios and pixel boxes.

**The DOM contract this module pins** (B1/B2 name the id and the badge hook; the copy is B1/B2 as amended
by B3, written out by hand):

=====================================  ==================================================================
``header #mic-button``                 the control; not rendered (hidden) before an ``astra.mic.state``
``header [data-mic-badge]``            "Mic muted" while the echo is muted; not rendered otherwise
``#mute-button``                       "Stop sharing" (B0-1); no ``aria-keyshortcuts`` (B3)
``[data-thread] [data-transcript]``    the user bubble of a turn (T-0154 B1)
=====================================  ==================================================================

Readings (stated, not guessed silently):
- "Uses the error/alert token" is read as: the badge's computed ``color``, ``background-color`` or border
  colour equals ``--bad`` as the page resolves it in that theme (an in-page probe element styled
  ``color: var(--bad)``); "high contrast" as T-0056's text floor (4.5:1, 3:1 for large text) through
  ``test_hud_chrome``'s ``measure_js``/``_contrast_report``.
- "No horizontal overflow in the header" is read as: ``header.scrollWidth <= header.clientWidth`` and every
  rendered header descendant's box lies within the header's left and right edges (0.5 px slack), measured
  in the widest state (muted, so the badge shows, and sharing, so Share reads "Sharing screen").
- Space and Enter on a focused button are that button's own activation (a click), not a shortcut; the
  key sweep therefore types only letters and digits.
- The ``m`` rows run with a capture running, so "``m`` stops nothing" is observable (Stop sharing stays
  enabled, the stub keeps receiving frames).
"""

from __future__ import annotations

import json
import string
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import pytest

from local_astra.hud_static import HudStaticServer
from tests.fakes.hud_server import HudStub, mic_state, vision_state
from tests.integration.test_hud_chrome import _contrast_report, choose_theme_js, measure_js
from tests.integration.test_hud_headless import CHROME, HUD, requires_chrome
from tests.integration.test_hud_redesign import _panels_line, _Recorder
from tests.integration.test_t0177_hud_vision import INIT_JS, _click

pytestmark = pytest.mark.integration

# --- the expected copy, by hand from B1/B2 as amended by B3 (no "(m)") --------------------------------

LIVE_TEXT = "Mute mic"
LIVE_TITLE = (
    "Mute the microphone: Gearvis hears nothing until you unmute; its own speech still plays and "
    "typing still works"
)
MUTED_TEXT = "Unmute mic"
MUTED_TITLE = "Unmute the microphone"
NO_MIC = "No mic"
BADGE_TEXT = "Mic muted"
BADGE_TITLE = "Gearvis can't hear you. Click Unmute mic to let it listen again."
STOP_TEXT = "Stop sharing"
STOP_TITLE = "Stop sharing: end the screen or camera capture"
#: A3's ``--silent`` reason, verbatim: the title of a disabled "No mic".
REASON = "this session is silent: no audio device is open (--silent)"

#: The page's own client->server types at BASE 2f955738 that this script can provoke (hello, frames, the
#: T-0177 vision select on a Share in live). B5: the only new one is astra.mic.mute.
BASE_SENT = {"astra.hello", "astra.frame", "astra.vision.select"}

MIC = "#mic-button"
BADGE = "header [data-mic-badge]"

PROBE_JS = """
(() => {
  const vis = (el) => !!el && el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true});
  const box = (el) => { if (!el) return null; const r = el.getBoundingClientRect();
                        return {left: r.left, right: r.right, top: r.top, bottom: r.bottom, w: r.width, h: r.height}; };
  const norm = (s) => String(s || '').replace(/\\s+/g, ' ').trim();
  const mic = document.getElementById('mic-button');
  const badge = document.querySelector('header [data-mic-badge]');
  const stop = document.getElementById('mute-button');
  const share = document.getElementById('share-button');
  return {
    mic: !!mic, mic_in_header: !!(mic && mic.closest('header')), mic_tag: mic ? mic.tagName.toLowerCase() : null,
    mic_visible: vis(mic), mic_text: mic ? norm(mic.innerText || mic.textContent) : null,
    mic_title: mic ? mic.getAttribute('title') : null, mic_pressed: mic ? mic.getAttribute('aria-pressed') : null,
    mic_disabled: mic ? mic.disabled : null, mic_box: box(mic),
    mic_before_share: !!(mic && share && (mic.compareDocumentPosition(share) & Node.DOCUMENT_POSITION_FOLLOWING)),
    badge: !!badge, badge_visible: vis(badge), badge_text: badge ? norm(badge.textContent) : null,
    badge_title: badge ? badge.getAttribute('title') : null,
    stop_text: stop ? norm(stop.innerText || stop.textContent) : null, stop_title: stop ? stop.getAttribute('title') : null,
    stop_keys: stop ? stop.getAttribute('aria-keyshortcuts') : null, stop_disabled: stop ? stop.disabled : null,
    share_text: share ? norm(share.innerText || share.textContent) : null, share_box: box(share),
    capturing: !!stop && !stop.disabled,
    frames_sent: Number((document.getElementById('frames-sent') || {dataset: {}}).dataset.framesSent || 0),
    active: document.activeElement ? (document.activeElement.id || document.activeElement.tagName.toLowerCase()) : null,
    theme: document.documentElement.getAttribute('data-theme'),
  };
})()
"""

#: B2's token: the badge's colours against ``--bad`` as this page resolves it (an in-page probe element).
TOKEN_JS = """
(() => {
  const badge = document.querySelector('header [data-mic-badge]');
  const probe = document.createElement('span');
  probe.style.color = 'var(--bad)';
  document.body.append(probe);
  const bad = getComputedStyle(probe).color;
  const probe2 = document.createElement('span');
  probe2.style.color = 'var(--bad-surface)';
  document.body.append(probe2);
  const badSurface = getComputedStyle(probe2).color;
  probe.remove(); probe2.remove();
  if (!badge) return {bad, badSurface, badge: false};
  const s = getComputedStyle(badge);
  return {bad, badSurface, badge: true, color: s.color, background: s.backgroundColor,
          border: s.borderTopColor, border_w: parseFloat(s.borderTopWidth) || 0,
          raw_bad: getComputedStyle(document.documentElement).getPropertyValue('--bad').trim()};
})()
"""

#: B4: the header's horizontal overflow and every rendered descendant outside its edges.
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
  return {scroll: h.scrollWidth, client: h.clientWidth, header: [hr.left, hr.right], outside: out,
          doc_scroll: document.documentElement.scrollWidth, inner: window.innerWidth,
          badge_visible: (() => { const b = document.querySelector('header [data-mic-badge]');
                                  return !!b && b.checkVisibility(); })(),
          mic_text: ((document.getElementById('mic-button') || {}).textContent || '').trim(),
          share_text: ((document.getElementById('share-button') || {}).textContent || '').trim()};
})()
"""

#: The user bubbles of the thread, per turn id: present and visible.
BUBBLES_JS = """
(() => {
  const out = {};
  for (const a of document.querySelectorAll('[data-thread] [data-turn-id]')) {
    const b = a.querySelector('[data-transcript]');
    out[a.getAttribute('data-turn-id')] = b ? {visible: b.checkVisibility({checkOpacity: true, checkVisibilityCSS: true}),
                                              text: (b.textContent || '').trim(),
                                              box: [b.getBoundingClientRect().width, b.getBoundingClientRect().height]}
                                            : null;
  }
  return out;
})()
"""

MIC_VISIBLE_JS = (
    "(() => { const b = document.getElementById('mic-button');"
    " return !!b && b.checkVisibility({checkOpacity: true, checkVisibilityCSS: true}); })()"
)
CAPTURING_JS = (
    "!document.getElementById('mute-button').disabled && "
    "/^\\d+x\\d+$/.test((document.getElementById('capture-size').textContent || '').trim())"
)
STOPPED_JS = "document.getElementById('mute-button').disabled"

CONTRAST = [("mic badge", BADGE, None)]


def _mic_text_is(text: str) -> str:
    return (
        "(() => { const b = document.getElementById('mic-button');"
        f" return !!b && (b.textContent || '').replace(/\\s+/g, ' ').trim() === {json.dumps(text)}; }})()"
    )


@dataclass
class Run:
    error: str | None = None
    probes: dict[str, Any] = field(default_factory=dict)
    mutes: dict[str, list[bool]] = field(default_factory=dict)
    frames: dict[str, int] = field(default_factory=dict)
    token: dict[str, Any] = field(default_factory=dict)
    contrast: dict[str, Any] = field(default_factory=dict)
    overflow: dict[str, Any] = field(default_factory=dict)
    bubbles: dict[str, Any] = field(default_factory=dict)
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


def _drive(result: Run, chrome: Any, stub: HudStub, port: int) -> None:
    ws = urllib.parse.quote(stub.url, safe="")
    base = f"http://127.0.0.1:{port}/"
    probes = result.probes

    def mutes() -> list[bool]:
        return [bool(m.muted) for m in stub.mic_mutes()]

    def probe(name: str) -> dict[str, Any]:
        chrome.evaluate(
            "new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))"
        )
        got = chrome.evaluate(PROBE_JS)
        probes[name] = got if isinstance(got, dict) else {"__probe__": got}
        return probes[name]

    def step(name: str, action: Any, *, expect: int = 0, settle: float = 0.8) -> None:
        """Run ``action``; record the mutes it caused (wait for ``expect``, then ``settle`` more)."""
        before = len(mutes())
        action()
        if expect:
            stub.wait_for(lambda s: len(s.mic_mutes()) >= before + expect, 6)
        chrome.drain(settle)
        result.mutes[name] = mutes()[before:]

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

    def share() -> None:
        _click(chrome, "#share-button")
        chrome.wait_until(CAPTURING_JS, 10)

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
    chrome.evaluate(choose_theme_js("dark"))
    probe("idle")

    # B1: other messages before any mic state show nothing; `m` before a state sends nothing.
    push(vision_state("browser"))
    push(
        json.dumps(
            {"type": "astra.turn.state", "state": "LISTENING", "turn_id": "t0", "thinking_ms": 0}
        )
    )
    blur()
    step("m_before_state", lambda: chrome.key("m", text="m", code=77))
    probe("before_state")

    # B1: the first state: live.
    push(mic_state(False))
    chrome.wait_until(MIC_VISIBLE_JS, 5)
    probe("live")

    # B1 adversarial: the stub withholds the echo. Two clicks, both ask to mute; the control does not move.
    step("click1", lambda: _click(chrome, MIC), expect=1, settle=1.0)
    probe("after_click_no_echo")
    step("click2", lambda: _click(chrome, MIC), expect=1, settle=0.6)
    probe("after_click2_no_echo")

    # B1/B2: the muted echo.
    push(mic_state(True))
    chrome.wait_until(_mic_text_is(MUTED_TEXT), 5)
    probe("muted")
    for theme in ("dark", "light"):
        chrome.evaluate(choose_theme_js(theme))
        time.sleep(0.3)
        probe(f"muted_{theme}")
        result.token[theme] = chrome.evaluate(TOKEN_JS) or {}
        result.contrast[theme] = chrome.evaluate(measure_js(CONTRAST)) or {}
    chrome.evaluate(choose_theme_js("dark"))

    # B1: a click while muted asks to unmute; nothing moves until the echo.
    step("click_unmute", lambda: _click(chrome, MIC), expect=1, settle=0.8)
    probe("after_unmute_click_no_echo")
    push(mic_state(False))
    chrome.wait_until(_mic_text_is(LIVE_TEXT), 5)
    probe("unmuted")

    # B0/B3: a real capture runs, so "m stops nothing" is observable.
    share()
    probe("sharing")

    # B3: `m` everywhere, with the mic live.
    targets = [
        ("body", None),
        ("mic", MIC),
        ("stop", "#mute-button"),
        ("share", "#share-button"),
        ("theme", "header select[data-theme-control]"),
        ("combo", "main > .pane.thumb select[data-vision-source]"),
    ]
    focused: dict[str, bool] = {}
    frames0 = len(stub.frames())

    def press_m_everywhere() -> None:
        for name, sel in targets:
            if sel is None:
                blur()
                focused[name] = bool(chrome.evaluate("document.activeElement === document.body"))
            else:
                focused[name] = focus(sel)
            chrome.key("m", text="m", code=77)
            chrome.key("M", text="M", code=77)
            chrome.drain(0.1)

    step("m_everywhere_live", press_m_everywhere, settle=1.0)
    result.keys["focused"] = dict(focused)
    probe("after_m_live")

    # B3 adversarial: every other letter and digit on the body.
    def sweep() -> None:
        blur()
        for ch in string.ascii_lowercase + string.digits:
            chrome.key(ch, text=ch)
        chrome.drain(0.2)

    step("sweep_body", sweep, settle=1.0)
    probe("after_sweep")

    # "how many" typed in the ask box sends no mute (and stops no capture).
    push(
        json.dumps({"type": "astra.turn.state", "state": "IDLE", "turn_id": "t0", "thinking_ms": 0})
    )  # the ask box is enabled only in IDLE
    ask_focused = focus("[data-ask-input]")

    def type_question() -> None:
        chrome.type_text("how many mics")

    step("ask_typed", type_question, settle=1.0)
    result.keys["ask"] = {
        "focused": ask_focused,
        "value": chrome.evaluate("(document.querySelector('[data-ask-input]') || {}).value"),
    }
    chrome.evaluate(
        "(() => { const i = document.querySelector('[data-ask-input]'); if (i) { i.value = '';"
        " i.dispatchEvent(new Event('input', {bubbles: true})); } })()"
    )
    probe("after_ask")

    # B3: `m` on the body while muted: nothing is sent, the control stays muted.
    push(mic_state(True))
    chrome.wait_until(_mic_text_is(MUTED_TEXT), 5)
    step("m_everywhere_muted", press_m_everywhere, settle=1.0)
    probe("after_m_muted")
    stub.wait_for(lambda s: len(s.frames()) >= frames0 + 2, 6)
    result.frames["during_keys"] = len(stub.frames()) - frames0

    # B4: the widest header (muted with its badge, sharing), 1100 and 1440 px, both themes.
    for width in (1100, 1440):
        chrome.viewport(width, 900)
        for theme in ("dark", "light"):
            chrome.evaluate(choose_theme_js(theme))
            time.sleep(0.3)
            result.overflow[f"{width}_{theme}_muted"] = chrome.evaluate(OVERFLOW_JS) or {}
    chrome.viewport(1440, 900)
    chrome.evaluate(choose_theme_js("dark"))

    # B0: Stop sharing by click still stops the capture.
    step("stop_click", lambda: (_click(chrome, "#mute-button"), chrome.wait_until(STOPPED_JS, 5)))
    probe("stopped")

    # B1 (H-T181-5): no mic. Disabled with the reason; a click and `m` send nothing.
    push(mic_state(False, available=False, reason=REASON))
    chrome.wait_until(_mic_text_is(NO_MIC), 5)
    probe("gone")
    step("click_gone", lambda: _click(chrome, MIC), settle=1.0)
    blur()
    step("m_gone", lambda: (chrome.key("m", text="m", code=77), chrome.drain(0.1)), settle=0.8)
    probe("gone_after")
    for width in (1100, 1440):
        chrome.viewport(width, 900)
        for theme in ("dark", "light"):
            chrome.evaluate(choose_theme_js(theme))
            time.sleep(0.3)
            result.overflow[f"{width}_{theme}_gone"] = chrome.evaluate(OVERFLOW_JS) or {}
    chrome.viewport(1440, 900)
    chrome.evaluate(choose_theme_js("dark"))

    # A0-S4 / H-T181-10: an abandoned question's empty final shows no bubble; a worded final shows.
    push(
        json.dumps(
            {"type": "astra.transcript", "turn_id": "t9", "text": "what is the", "final": False}
        )
    )
    chrome.wait_until("!!document.querySelector('[data-thread] [data-turn-id=\"t9\"]')", 5)
    result.bubbles["partial"] = chrome.evaluate(BUBBLES_JS) or {}
    push(json.dumps({"type": "astra.transcript", "turn_id": "t9", "text": "", "final": True}))
    push(
        json.dumps(
            {
                "type": "astra.transcript",
                "turn_id": "t10",
                "text": "how many eighths",
                "final": True,
            }
        )
    )
    chrome.drain(0.5)
    result.bubbles["final"] = chrome.evaluate(BUBBLES_JS) or {}

    # B1: a reconnect's hello replays the state, and the control follows it.
    stub.drop_client()
    stub.wait_for(lambda s: s.received_types().count("astra.hello") >= hellos + 2, 20)
    stub.wait_for_client(10)
    push(mic_state(True))
    chrome.wait_until(_mic_text_is(MUTED_TEXT), 5)
    probe("replayed")

    # A9 / B-T1: a live reset (astra.session.cleared, no hello) keeps a muted control muted; the state is
    # re-sent only on hello, so nothing re-echoes it here.
    push('{"type":"astra.session.cleared","resets":1,"ended_turn_id":null,"memory_forgotten":0}')
    chrome.drain(1.0)
    probe("after_live_reset")

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
    print(f"[T-0181 B] headless session wall time {result.seconds:.1f} s")
    yield result


def _p(run: Run, name: str) -> dict[str, Any]:
    assert run.error is None, f"the T-0181 B headless session failed: {run.error}"
    got = run.probes.get(name)
    assert isinstance(got, dict) and "__probe__" not in got, (
        f"probe {name!r} missing or broken: {got!r}"
    )
    return got


def _live(p: dict[str, Any], why: str) -> None:
    assert p["mic_visible"], f"{why}: the mic control is shown (an astra.mic.state arrived): {p}"
    assert p["mic_text"] == LIVE_TEXT, f"{why}: B1 label {p['mic_text']!r}, want {LIVE_TEXT!r}"
    assert p["mic_pressed"] == "false", f"{why}: B1 aria-pressed {p['mic_pressed']!r}"
    assert p["mic_title"] == LIVE_TITLE, f"{why}: B1/B3 title {p['mic_title']!r}"
    assert p["mic_disabled"] is False, f"{why}: an available mic's control is enabled"
    assert not p["badge_visible"], f"{why}: B2: no badge while not muted ({p['badge_text']!r})"


def _muted(p: dict[str, Any], why: str) -> None:
    assert p["mic_visible"], f"{why}: the mic control is shown: {p}"
    assert p["mic_text"] == MUTED_TEXT, f"{why}: B1 label {p['mic_text']!r}, want {MUTED_TEXT!r}"
    assert p["mic_pressed"] == "true", f"{why}: B1 aria-pressed {p['mic_pressed']!r}"
    assert p["mic_title"] == MUTED_TITLE, f"{why}: B1/B3 title {p['mic_title']!r}"
    assert p["mic_disabled"] is False
    assert p["badge_visible"], f"{why}: B2: the 'Mic muted' badge shows while the echo is muted"
    assert p["badge_text"] == BADGE_TEXT, f"{why}: B2 badge text {p['badge_text']!r}"
    assert p["badge_title"] == BADGE_TITLE, f"{why}: B2/B3 badge title {p['badge_title']!r}"


# --- B1: nothing before a state ---------------------------------------------------------------------


@requires_chrome
def test_b1_the_control_is_declared_and_shown_only_after_a_state(run):
    """B1: ``#mic-button`` exists in the header (before Share) and is not rendered before an
    ``astra.mic.state``, whatever else arrives; then the first state shows it. (A page with no control at all
    fails here at the "exists" line, not by passing the "not shown" one.)"""
    for name in ("idle", "before_state"):
        p = _p(run, name)
        assert p["mic"] and p["mic_tag"] == "button" and p["mic_in_header"], (
            f"B1 ({name}): a <button id=mic-button> in the header: {p}"
        )
        assert not p["mic_visible"], (
            f"B1 ({name}): the control shows before any astra.mic.state: {p}"
        )
        assert not p["badge_visible"], f"B2 ({name}): no badge before a state"
    assert run.mutes.get("m_before_state") == [], (
        f"B3: `m` before a state sent {run.mutes.get('m_before_state')}"
    )
    _live(_p(run, "live"), "the first state")


@requires_chrome
def test_b1_the_control_sits_before_share(run):
    p = _p(run, "live")
    assert p["mic_before_share"], f"B1: #mic-button precedes #share-button in the header: {p}"
    mic, share = p["mic_box"], p["share_box"]
    assert mic and share and mic["right"] <= share["left"] + 0.5, (
        f"B1: 'in the header, before Share' — on screen the control is left of Share: mic {mic}, share {share}"
    )


# --- B1: the echo, never the click ---------------------------------------------------------------------


@requires_chrome
def test_b1_a_click_sends_not_echoed_and_changes_nothing_until_the_echo(run):
    """Adversarial (the stub withholds the echo): two clicks on a live control send muted:true twice, and the
    control stays live; a click on a muted control sends muted:false and the control stays muted until the
    live echo."""
    _live(_p(run, "live"), "premise")
    assert run.mutes.get("click1") == [True], (
        f"B1: a click on 'Mute mic' sends muted:true; got {run.mutes.get('click1')}"
    )
    _live(_p(run, "after_click_no_echo"), "B1: no echo yet, the click changed the control")
    assert run.mutes.get("click2") == [True], (
        f"B1: still no echo, so the second click inverts the echo again (muted:true); got {run.mutes.get('click2')} "
        "(a page that toggles locally sends false here)"
    )
    _live(_p(run, "after_click2_no_echo"), "B1: two clicks, no echo")
    assert run.mutes.get("click_unmute") == [False], (
        f"B1: a click on 'Unmute mic' sends muted:false; got {run.mutes.get('click_unmute')}"
    )
    _muted(_p(run, "after_unmute_click_no_echo"), "B1: the unmute click alone moved the control")
    _live(_p(run, "unmuted"), "the live echo")


@requires_chrome
def test_b1_b2_the_label_aria_pressed_and_badge_follow_the_echo(run):
    _muted(_p(run, "muted"), "the muted echo")
    for theme in ("dark", "light"):
        p = _p(run, f"muted_{theme}")
        assert p["theme"] == theme, f"premise: the {theme} theme is applied: {p['theme']}"
        _muted(p, f"muted, {theme}")
    _muted(_p(run, "replayed"), "a reconnect's replayed state")


@requires_chrome
def test_b1_a_live_reset_keeps_a_muted_control_muted(run):
    """A9 (H-T181-3) as read in B-T1: a mute survives a reset, and the state is re-sent only on hello, so a
    live ``astra.session.cleared`` (no hello) must leave the control reading muted with its badge shown;
    the page has no fresh echo to restore it from if it forgets."""
    _muted(_p(run, "replayed"), "premise: muted before the reset")
    _muted(_p(run, "after_live_reset"), "after a live astra.session.cleared")


# --- B2: unmistakable -----------------------------------------------------------------------------------


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_b2_the_badge_uses_the_alert_token_at_high_contrast(run, theme):
    assert run.error is None, run.error
    t = run.token.get(theme) or {}
    assert t.get("badge"), f"B2: no [data-mic-badge] in the header: {t}"
    assert t.get("raw_bad"), f"premise: the page defines --bad in {theme}: {t}"
    colours = {t.get("color"), t.get("background")} | (
        {t.get("border")} if (t.get("border_w") or 0) > 0 else set()
    )
    assert t["bad"] in colours, (
        f"B2 ({theme}): the badge uses the error/alert token: --bad resolves to {t['bad']}, the badge paints "
        f"color {t.get('color')}, background {t.get('background')}, border {t.get('border')} ({t.get('border_w')} px)"
    )
    measured = run.contrast.get(theme) or {}
    missing, vacuous, failures, _ = _contrast_report(measured)
    assert not missing and not vacuous, (
        f"B2 ({theme}): the badge was not measured: {missing} {vacuous}"
    )
    assert failures == [], f"B2 ({theme}): the badge text is below T-0056's floor: {failures}"


# --- B1 (H-T181-5): no mic --------------------------------------------------------------------------------


@requires_chrome
def test_b1_no_mic_is_disabled_with_the_reason_and_nothing_sends(run):
    p = _p(run, "gone")
    assert p["mic_visible"], f"B1: a session with no mic still shows the control: {p}"
    assert p["mic_text"] == NO_MIC, f"B1: label {p['mic_text']!r}"
    assert p["mic_disabled"] is True, "B1: disabled"
    assert p["mic_title"] == REASON, (
        f"B1 (H-T181-5): the server's reason verbatim as title: {p['mic_title']!r}"
    )
    assert not p["badge_visible"], "B2: an unavailable mic is never muted: no badge"
    assert run.mutes.get("click_gone") == [], (
        f"B1: a click on a disabled control sent {run.mutes.get('click_gone')}"
    )
    assert run.mutes.get("m_gone") == [], f"B3: `m` with no mic sent {run.mutes.get('m_gone')}"
    after = _p(run, "gone_after")
    assert after["mic_text"] == NO_MIC and after["mic_disabled"] is True, (
        f"nothing moved it: {after}"
    )


# --- B3: no shortcut (H-T181-4) ------------------------------------------------------------------------------


@requires_chrome
def test_b3_m_anywhere_sends_nothing_and_stops_nothing(run):
    """B3 adversarial: ``m`` and ``M`` on the body (nothing focused), the mic button, Stop sharing, Share, the
    theme select and the vision combo, with the mic live and then muted: no ``astra.mic.mute``, the capture
    still runs (Stop sharing enabled, frames keep arriving), and the control keeps the echoed state."""
    _live(_p(run, "unmuted"), "premise: the control is live before the keys")
    assert _p(run, "sharing")["capturing"], "premise: a capture runs, so a stop would be visible"
    focused = run.keys.get("focused") or {}
    assert focused and all(focused.values()), (
        f"premise: every target held the focus for its keys: {focused}"
    )
    for name in ("m_everywhere_live", "m_everywhere_muted"):
        assert run.mutes.get(name) == [], (
            f"B3: `m` sent astra.mic.mute ({name}): {run.mutes.get(name)}"
        )
    live, muted = _p(run, "after_m_live"), _p(run, "after_m_muted")
    _live(live, "after `m` everywhere (live)")
    _muted(muted, "after `m` everywhere (muted)")
    for name, p in (("live", live), ("muted", muted)):
        assert p["capturing"], f"B3: `m` stopped the capture ({name}); Stop sharing has no key: {p}"
    assert run.frames.get("during_keys", 0) >= 2, f"B3: frames kept going out: {run.frames}"


@requires_chrome
def test_b3_no_letter_or_digit_on_the_body_sends_a_mute_or_stops_the_capture(run):
    _live(_p(run, "unmuted"), "premise")
    assert run.mutes.get("sweep_body") == [], (
        f"B3: a key on the body sent {run.mutes.get('sweep_body')}"
    )
    p = _p(run, "after_sweep")
    assert p["capturing"], f"B3: a single key stopped the capture: {p}"
    _live(p, "after the sweep")


@requires_chrome
def test_b3_how_many_typed_in_the_ask_box_sends_no_mute(run):
    ask = run.keys.get("ask") or {}
    assert ask.get("focused"), f"premise: the ask box held the focus: {ask}"
    assert ask.get("value") == "how many mics", f"premise: the keys landed in the box: {ask}"
    _live(_p(run, "unmuted"), "premise: the mic control was live while typing")
    assert run.mutes.get("ask_typed") == [], f"B3: typing sent {run.mutes.get('ask_typed')}"
    assert _p(run, "after_ask")["capturing"], "typing stopped the capture"


# --- B0 / B3: Stop sharing --------------------------------------------------------------------------------------


@requires_chrome
def test_b0_stop_sharing_is_labelled_and_still_stops_by_click(run):
    s = _p(run, "sharing")
    assert s["capturing"], f"premise: a capture runs: {s}"
    assert s["stop_text"] == STOP_TEXT, f"B0-1: #mute-button reads {s['stop_text']!r}"
    assert s["stop_title"] == STOP_TITLE, f"B3: its title {s['stop_title']!r}"
    assert s["stop_keys"] is None, f"B3: no aria-keyshortcuts ({s['stop_keys']!r})"
    stopped = _p(run, "stopped")
    assert not stopped["capturing"] and stopped["stop_disabled"], (
        f"B0-1: a click on Stop sharing stops the capture: {stopped}"
    )
    assert run.mutes.get("stop_click") == [], "Stop sharing is not the mic"


# --- B4: layout ----------------------------------------------------------------------------------------------


@requires_chrome
@pytest.mark.parametrize("width", [1100, 1440])
@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("state", ["muted", "gone"])
def test_b4_no_horizontal_overflow_in_the_header(run, width, theme, state):
    assert run.error is None, run.error
    o = run.overflow.get(f"{width}_{theme}_{state}") or {}
    assert o, f"premise: the header was measured at {width} px, {theme}, {state}"
    if state == "muted":
        assert o.get("badge_visible") and o.get("mic_text") == MUTED_TEXT, (
            f"premise: the widest state (muted): {o}"
        )
        assert str(o.get("share_text", "")).startswith("Sharing"), (
            f"premise: Share reads 'Sharing …': {o}"
        )
    else:
        assert o.get("mic_text") == NO_MIC, f"premise: the no-mic state: {o}"
    assert o["scroll"] <= o["client"], (
        f"B4: the header overflows ({o['scroll']} > {o['client']}) at {width} px, {theme}"
    )
    assert o["outside"] == [], (
        f"B4: header elements outside the header at {width} px, {theme}: {o['outside']}"
    )
    assert o["doc_scroll"] <= o["inner"], f"B4: the page scrolls sideways at {width} px: {o}"


# --- the empty final bubble (A0-S4, H-T181-10) -----------------------------------------------------------------


@requires_chrome
def test_an_empty_final_line_renders_no_visible_bubble(run):
    assert run.error is None, run.error
    partial = (run.bubbles.get("partial") or {}).get("t9")
    assert partial and partial["visible"] and partial["text"] == "what is the", (
        f"premise: the partial bubble shows with its words: {run.bubbles.get('partial')}"
    )
    final = run.bubbles.get("final") or {}
    t9 = final.get("t9")
    assert t9 is None or not t9["visible"], (
        f"A0-S4/H-T181-10: the abandoned question's empty final leaves a visible bubble: {t9}"
    )
    assert t9 is None or "what is the" not in t9["text"], (
        f"none of the dropped words stays on the page: {t9}"
    )
    t10 = final.get("t10")
    assert t10 and t10["visible"] and t10["text"] == "how many eighths", (
        f"too wide: a worded final still shows: {t10}"
    )


# --- B5: negatives --------------------------------------------------------------------------------------------


@requires_chrome
def test_b5_the_only_new_message_the_page_sends_is_mic_mute(run):
    assert run.error is None, run.error
    sent = set(run.all_types)
    assert "astra.mic.mute" in sent, (
        f"premise: the page sent the mute (anti-vacuity): {sorted(sent)}"
    )
    assert sent - BASE_SENT == {"astra.mic.mute"}, (
        f"B5: new message types {sorted(sent - BASE_SENT)}"
    )


@requires_chrome
def test_b5_no_request_leaves_loopback(run):
    assert run.error is None, run.error
    assert run.requests, "the CDP network log is empty, so this row would be vacuous"
    off = [
        u
        for u in run.requests
        if u.split("://", 1)[0] in {"http", "https", "ws", "wss"}
        and not any(u.split("://", 1)[1].startswith(h) for h in ("127.0.0.1", "localhost", "[::1]"))
    ]
    assert off == [], f"B5: zero non-loopback requests; got {off}"


@requires_chrome
def test_the_page_threw_nothing(run):
    assert run.error is None, run.error
    assert run.exceptions == [], f"the page threw: {run.exceptions[:3]}"


@requires_chrome
def test_the_session_fits_the_module_budget(run):
    assert run.error is None, run.error
    assert run.seconds < 90.0, f"the T-0181 B headless session took {run.seconds:.1f} s"
