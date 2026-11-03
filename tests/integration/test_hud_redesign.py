"""T-0128 — the HUD redesign, part 1, in headless Chrome: the rail, the theme, the honest Thinking pane.

docs/tasks/T-0128.md *Tests to write* T3, T5 and T6. The textual half is
``tests/hud/shell.test.mjs``, ``tests/hud/styles.test.mjs`` and ``tests/hud/theme.test.mjs``; this
module asks what only a renderer can answer: where the rail panes are laid out, what an appended
pane computes to, what ``<html data-theme>`` and ``color-scheme`` say after a system flip, a stored
choice, a reload and a ``localStorage`` that throws, and whether the page still renders when the
theme module 404s.

**Driving.** The CDP helpers of ``tests/integration/test_hud_headless.py`` (``Chrome``, ``CHROME``,
``HUD``, ``requires_chrome``) are imported, never copied; this module only subclasses ``Chrome`` to
record console lines, uncaught exceptions and requests. The contrast oracle and the motion/chip
probes are imported from ``tests/integration/test_hud_chrome.py``. The page is served by the
production ``local_astra.hud_static.HudStaticServer`` on a loopback ephemeral port.

**What is pinned that the task leaves open** (the readings, so a reviewer can disagree with them):

- the rail panes are found as ``main > .pane.thumb``, ``main > .pane-thinking``,
  ``#assist-panel``, ``#log-panel`` and ``main > .pane-diagnostics`` (the task's spelling: classes);
- the Settings control is the one ``<select>`` in ``<header>`` whose option values are
  ``system``/``light``/``dark`` (criterion 8: "native", "exactly three options");
- "one column" below 1100 px means every pane shares one left edge and one width, and the five
  rail panes start below the last conversation pane;
- "within one frame" is checked as two ``requestAnimationFrame`` callbacks after the emulated
  media change (the change event fires in the rendering step that precedes the first);
- the Thinking pane's "animation-name is none" is asserted for the pane *and its descendants*: a
  pulse on a child is the same fake trajectory.

Regime (T-0128 *Regime*): Google Chrome 147.0.7727.101 headless, CDP, DPR 1, viewports 1000x800,
1280x800, 1440x900, 1920x1080; emulated scheme light/dark x data-theme light/dark; reduced motion
on and off; n = 1, deterministic, no seed. No GPU.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import importlib.util
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.ui_gate import GateResult, UiGate
from local_astra.contracts.base import Contract
from local_astra.contracts.codec import decode
from local_astra.contracts.protocol import AstraTurnState, AstraUiSurface
from local_astra.contracts.turn import TurnState
from local_astra.hud_static import HudStaticServer
from tests.fakes.hud_server import HudStub
from tests.integration.test_hud_chrome import (
    CHAIN_JS,
    CHIP_STATES_JS,
    DENOISE_EMPTY,
    FILL_JS,
    FIND_THEME_CONTROL_JS,
    MIN_PANE_WIDTH_PX,
    MOTION_JS,
    THEME_STATE_JS,
    TURN_STATES,
    _contrast_report,
    _durations,
    choose_theme_js,
    closed_loopback_port,
    composite,
    contrast_ratio,
    hexc,
    measure_js,
    painted,
    parse_css_color,
    ratio_of,
    relative_luminance,
    threshold_for,
)
from tests.integration.test_hud_headless import A2UI_ROOT, CHROME, HUD, Chrome, requires_chrome

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "hud" / "session_20261020_t3_t11.jsonl"
#: Recorded in tests/fixtures/hud/MANIFEST.md with the extraction command (T-0128 T5).
FIXTURE_SHA256 = "71ed804b2c0e0c889ed79f94c8fc510daca89646647836fe5019026f324f1e79"
#: Hand-counted from the extraction (docs/tasks/T-0128.md T5: the wire.out data.msg objects of
#: turns t3-t11 of logs/sessions/20261020-165855-1581291.jsonl, in order).
FIXTURE_LINES = 312
THEME_KEY = "astra.hud.theme"

WIDE = {"1280x800": (1280, 800), "1440x900": (1440, 900), "1920x1080": (1920, 1080)}
NARROW = (1000, 800)
RAIL_WIDTH_PX, RAIL_TOLERANCE_PX = 380.0, 20.0

#: Criterion 4's five rail panes, by the names the task gives them.
RAIL = {
    "pane thumb": "main > .pane.thumb",
    "pane-thinking": "main > .pane-thinking",
    # **Amended by T-0162 B-9** (named spec change): the Tools pane takes the Context pane's slot;
    # the assist panel moves into the Settings popover (found there by the checkbox rows below).
    "tools pane": "main > #tools-panel",
    "log pane": "main > #log-panel",
    "pane-diagnostics": "main > .pane-diagnostics",
}
#: **Amended by T-0154 B1** (docs/tasks/T-0154.md G2): the three conversation panes (spoken, you
#: said, shown) are one conversation pane, the direct child of ``<main>`` holding ``[data-thread]``
#: (the selector tests/integration/test_hud_thread_parity.py uses).
CONVERSATION = {"conversation": "main > :has([data-thread])"}
#: The declared rail panes an appended ``section.pane`` must look like (criterion 3, last bullet).
DECLARED_RAIL = ["pane thumb", "pane-thinking", "pane-diagnostics"]
#: Four declared panes (the conversation pane, What I see, Thinking, Diagnostics) plus the assists
#: and log panels. **Amended by T-0154 B1** (G2): was 8 (six declared panes, three of them the
#: retired conversation panes, plus the same two).
ALL_PANES = 6


# --- the fixture (T5) ------------------------------------------------------------------------------


def _turn_of(msg: dict[str, Any]) -> str | None:
    turn = msg.get("turn_id")
    if turn is None and str(msg.get("surface_id") or "").startswith("turn-"):
        turn = str(msg["surface_id"])[len("turn-") :]
    return turn


def fixture_lines() -> list[str]:
    return FIXTURE.read_text(encoding="utf-8").splitlines()


def test_the_t3_t11_fixture_is_what_the_manifest_says_and_every_line_decodes():
    """T-0128 T5: the committed fixture, decoded line by line with the production codec.

    Plain (no browser). The SHA-256 is the MANIFEST's; the turn set is H-T128-1's ruling — t3 to
    t11 only, because t1 holds an email address and t2 a password. A structural proxy for that
    ruling is asserted too (no ``type_at`` call, nothing shaped like an email address), since the
    secrets themselves can never be written into a test.
    """
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256, (
        "the fixture's SHA-256 is not the one tests/fixtures/hud/MANIFEST.md records; a fixture "
        "re-extracted without updating the MANIFEST is unattributable"
    )
    lines = fixture_lines()
    assert len(lines) == FIXTURE_LINES
    turns: list[str] = []
    for i, line in enumerate(lines, 1):
        decoded = decode(line)
        assert isinstance(decoded, Contract), (
            f"line {i}: local_astra.contracts.codec.decode returned {type(decoded).__name__}, not a "
            "contract — the fixture must replay as the real server's messages"
        )
        msg = json.loads(line)
        turn = _turn_of(msg)
        assert turn is not None, f"line {i}: a message with no turn id ({msg.get('type')})"
        if not turns or turns[-1] != turn:
            turns.append(turn)
        assert msg.get("type") != "astra.tool.call" or msg.get("name") != "type_at", (
            f"line {i}: a type_at call — t1/t2 were the typing turns (H-T128-1)"
        )
        assert not str(msg.get("type", "")).startswith("astra.canvas."), (
            "the task records zero astra.canvas.* among the session's wire.out records"
        )
    assert turns == [f"t{n}" for n in range(3, 12)], f"turns in fixture order: {turns}"
    email = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    assert len(email.findall(raw)) == 0, "H-T128-1: nothing shaped like an email address"


def _last_index_of(turn: str) -> int:
    idx = [i for i, line in enumerate(fixture_lines()) if _turn_of(json.loads(line)) == turn]
    return idx[-1]


# --- the card the current gate makes from t9 (ruling on f6b7d98, criteria 10 and 15 amended) -------
#
# The wire fixture's turn-t9 surfaces were rendered before T-0127's repair: the root Column has no
# children, so nothing in them is a card. The card rows use the surface the *current* gate makes
# from t9's raw ``ui`` commits in T-0127's committed fixture. The commits are read with
# ``scripts/replay_ui_gate.py``'s own ``ui_commits`` and gated exactly as that script and
# ``Session._gate_surface`` gate them (a fresh ``UiGate``, the logged ``turn_id`` and
# ``canvas_idx``, ``final=False``); then, as ``Session`` does at the turn's end, the last surface is
# published once more with ``final=True``.

UI_FIXTURE = REPO / "tests" / "fixtures" / "a2ui" / "desk_20261020_card_children.jsonl"
REPLAY_UI_GATE = REPO / "scripts" / "replay_ui_gate.py"
#: T-0127 records t9 complete (tests/integration/test_t0127_desk_replay.py: 5/6, only t3 is not).
CARD_TURN = "t9"
CARD_SURFACE_ID = f"turn-{CARD_TURN}"
#: Hand-counted in the wire fixture: three astra.ui.surface lines for turn-t9 (canvas 0, canvas 1,
#: canvas 1 final), all with an empty root Column. They are withheld from the replay, because the
#: page's bridge skips any canvas_idx at or below the last one applied to the mounted surface
#: (hud/src/ui/bridge.js), so the gated canvases pushed after them would never render.
STALE_CARD_SURFACES = 3
#: Floor on the gated card's Text components (the brief: the card rows are vacuous below 3).
MIN_CARD_TEXTS = 3


def _ui_commits(turn: str) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("_t0128_replay_ui_gate", REPLAY_UI_GATE)
    assert spec is not None and spec.loader is not None, f"cannot load {REPLAY_UI_GATE}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commits = module.ui_commits(UI_FIXTURE).get(turn) or []
    return list(commits)


def gated_card() -> list[GateResult]:
    """t9's ``ui`` commits, in order, through one fresh ``UiGate`` (one result per commit)."""
    gate = UiGate()
    return [
        gate.gate(
            c.get("text", ""),
            turn_id=CARD_TURN,
            canvas_idx=int(c.get("canvas_idx", 0)),
            final=False,
        )
        for c in _ui_commits(CARD_TURN)
    ]


def card_surfaces(results: list[GateResult]) -> list[AstraUiSurface]:
    """What the session publishes for the turn: every emitted surface, then the last as final."""
    out = [r.surface for r in results if r.surface is not None]
    if out:
        out.append(out[-1].model_copy(update={"final": True}))
    return out


def card_texts(surface: AstraUiSurface) -> list[str]:
    """The literal ``text`` of every Text component in the surface's updateComponents."""
    texts: list[str] = []
    for message in surface.messages:
        for comp in (message.get("updateComponents") or {}).get("components") or []:
            if comp.get("component") == "Text" and isinstance(comp.get("text"), str):
                texts.append(comp["text"].strip())
    return [t for t in texts if t]


def _is_stale_card_surface(line: str) -> bool:
    msg = json.loads(line)
    return msg.get("type") == "astra.ui.surface" and msg.get("surface_id") == CARD_SURFACE_ID


def test_the_gated_t9_card_is_complete_and_has_text_to_measure():
    """Premise of the t9 card rows (ruling on f6b7d98): the current gate makes a complete card.

    Plain (no browser). t9's raw ``ui`` commits from T-0127's fixture, gated as the session gates
    them, end complete (T-0127 criterion 4) on surface ``turn-t9`` with at least three Text
    components; and the wire fixture really holds the three stale turn-t9 surfaces this module
    withholds, each with an empty root Column (so the stale surface really is no card).
    """
    results = gated_card()
    assert results, f"no ui commits for {CARD_TURN} in {UI_FIXTURE.name}"
    last = results[-1]
    assert last.complete, (
        f"T-0127: {CARD_TURN}'s last gate call is not complete (lost {list(last.lost)}, "
        f"unreachable {list(last.unreachable)}); pick the completed card turn T-0127 records"
    )
    assert last.surface is not None and last.surface.surface_id == CARD_SURFACE_ID
    texts = card_texts(last.surface)
    assert len(texts) >= MIN_CARD_TEXTS, f"the gated card has {len(texts)} Text components: {texts}"
    stale = [json.loads(line) for line in fixture_lines() if _is_stale_card_surface(line)]
    assert len(stale) == STALE_CARD_SURFACES, f"{len(stale)} turn-t9 surfaces on the wire"
    for msg in stale:
        roots = [
            c
            for m in msg["messages"]
            for c in (m.get("updateComponents") or {}).get("components") or []
            if c.get("id") == "root"
        ]
        assert roots and all(r.get("children") == [] for r in roots), (
            "premise: the wire's turn-t9 surface is the pre-T-0127 empty Column; if it is not, the "
            f"fixture changed and the card rows no longer need the gated surface: {roots}"
        )


# --- a recording Chrome -------------------------------------------------------------------------------


class _Recorder(Chrome):
    """``Chrome`` plus console lines, uncaught exceptions and request URLs, as CDP reports them."""

    def __init__(self, binary: str) -> None:
        super().__init__(binary)
        self.console: list[str] = []
        self.exceptions: list[str] = []
        self.requests: list[str] = []
        self.init_scripts: list[str] = []
        self._mark = 0

    def _note(self, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        params = msg.get("params") or {}
        if method == "Runtime.consoleAPICalled":
            args = params.get("args") or []
            text = " ".join(str(a.get("value", a.get("description", ""))) for a in args)
            self.console.append(f"{params.get('type')}: {text}")
        elif method == "Runtime.exceptionThrown":
            details = params.get("exceptionDetails") or {}
            exc = details.get("exception") or {}
            self.exceptions.append(str(exc.get("description") or details.get("text")))
        elif method == "Network.requestWillBeSent":
            url = (params.get("request") or {}).get("url")
            if isinstance(url, str):
                self.requests.append(url)

    def _cmd(self, method: str, **params: Any) -> dict[str, Any]:
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv(timeout=30))
            self._note(msg)
            if msg.get("id") == self._id:
                assert "error" not in msg, f"{method}: {msg['error']}"
                return dict(msg["result"])

    def drain(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                self._note(json.loads(self.ws.recv(timeout=0.2)))
            except Exception:
                continue

    def viewport(self, width: int, height: int) -> None:
        self._cmd(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=1,
            mobile=False,
        )
        self.evaluate("document.documentElement.getBoundingClientRect().height")
        time.sleep(0.2)

    def media(self, features: list[dict[str, str]]) -> None:
        self._cmd("Emulation.setEmulatedMedia", features=features)

    def navigate(self, url: str) -> None:
        self._mark = len(self.console)
        super().navigate(url)

    def reload(self) -> None:
        self._mark = len(self.console)
        self._cmd("Page.reload")

    def add_init_script(self, source: str) -> str:
        ident = str(self._cmd("Page.addScriptToEvaluateOnNewDocument", source=source)["identifier"])
        self.init_scripts.append(ident)
        return ident

    def remove_init_script(self, ident: str) -> None:
        self._cmd("Page.removeScriptToEvaluateOnNewDocument", identifier=ident)
        self.init_scripts.remove(ident)

    def reset(self) -> None:
        """Between two sessions on the one shared browser: a blank page, no emulation, no init
        script, empty records. Each session serves the HUD from its own ephemeral port, so its
        origin (and its ``localStorage``) is fresh too."""
        for ident in list(self.init_scripts):
            self.remove_init_script(ident)
        self._cmd("Page.enable")
        self._cmd("Page.navigate", url="about:blank")
        self._cmd("Emulation.setEmulatedMedia", features=[])
        self._cmd("Emulation.clearDeviceMetricsOverride")
        self.drain(0.2)
        self.console.clear()
        self.exceptions.clear()
        self.requests.clear()
        self._mark = 0

    def wait_panes(self, n: int, timeout_s: float = 25.0) -> bool:
        """True once ``n`` panes are mounted. It stops early, with the count as it then stands, once
        ``hud/src/main.js`` has logged ``panels mounted:`` for this navigation: that line is
        written after ``mountPanels`` settles, so no further pane is coming. Without the early
        stop, a page that mounts fewer panes costs the full timeout at every navigation."""
        expr = f"document.querySelectorAll('main > .pane').length >= {n}"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.evaluate(expr):
                return True
            if _panels_line(self.console[self._mark :]) is not None:
                return bool(self.evaluate(expr))
            time.sleep(0.1)
        return bool(self.evaluate(expr))


def _scheme(value: str) -> list[dict[str, str]]:
    return [{"name": "prefers-color-scheme", "value": value}]


# --- page probes ---------------------------------------------------------------------------------------

LAYOUT_JS = """
(() => {
  window.scrollTo(0, 0);
  const RAIL = __RAIL__, CONV = __CONV__;
  const reachable = (el) => {
    for (let n = el.parentElement; n; n = n.parentElement) {
      const st = getComputedStyle(n);
      if (/(auto|scroll)/.test(st.overflowY) && n.scrollHeight > n.clientHeight + 1) return true;
    }
    const de = document.documentElement;
    const clipped = (e) => /(hidden|clip)/.test(getComputedStyle(e).overflowY);
    return de.scrollHeight > window.innerHeight + 1 && !clipped(de) && !clipped(document.body);
  };
  const snap = (el) => {
    if (!el) return {present: false};
    const s = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    return {present: true, hidden: !!el.closest('[hidden]'), display: s.display,
            visibility: s.visibility, x: r.x, y: r.y, w: r.width, h: r.height,
            right: r.right, bottom: r.bottom, reachable: reachable(el),
            heading: ((el.querySelector('h2') || {}).textContent || '').trim(),
            heading_h: el.querySelector('h2') ? el.querySelector('h2').getBoundingClientRect().height : null};
  };
  const pick = (map) => Object.fromEntries(Object.entries(map).map(([k, sel]) => [k, snap(document.querySelector(sel))]));
  const de = document.documentElement;
  return {inner_w: window.innerWidth, inner_h: window.innerHeight, client_w: de.clientWidth,
          scroll_w: de.scrollWidth, rail: pick(RAIL), conversation: pick(CONV),
          panes: [...document.querySelectorAll('main > .pane')].map((p) => {
            const r = p.getBoundingClientRect();
            return {label: (((p.querySelector('h2') || {}).textContent) || p.id || p.className).trim().slice(0, 32),
                    x: r.x, w: r.width};
          })};
})()
""".replace("__RAIL__", json.dumps(RAIL)).replace("__CONV__", json.dumps(CONVERSATION))

APPENDED_JS = """
(() => {
  const RAIL = __RAIL__;
  const look = (el) => {
    if (!el) return null;
    const s = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    return {padding: [s.paddingTop, s.paddingRight, s.paddingBottom, s.paddingLeft].join(' '),
            background: s.backgroundColor,
            radius: [s.borderTopLeftRadius, s.borderTopRightRadius, s.borderBottomRightRadius,
                     s.borderBottomLeftRadius].join(' '),
            x: Math.round(r.x), w: Math.round(r.width)};
  };
  const pane = document.createElement('section');
  pane.className = 'pane';
  pane.dataset.t0128Appended = 'true';
  const h2 = document.createElement('h2');
  h2.textContent = 'appended';
  pane.append(h2, Object.assign(document.createElement('p'), {textContent: 'a panel that mounted itself'}));
  document.querySelector('main').append(pane);
  const out = {appended: look(pane),
               declared: Object.fromEntries(Object.entries(RAIL).map(([k, s]) => [k, look(document.querySelector(s))]))};
  pane.remove();
  return out;
})()
""".replace("__RAIL__", json.dumps(RAIL))

THINKING_JS = """
(() => {
  const pane = document.querySelector('main > .pane-thinking');
  if (!pane) return {present: false};
  const root = pane.querySelector('[data-denoise-root]');
  const EMPTY = __EMPTY__;
  let line = null;
  for (const el of pane.querySelectorAll('*')) {
    if (el.closest('[data-denoise-root]')) continue;
    if ((el.textContent || '').replace(/\\s+/g, ' ').trim() === EMPTY) line = el;
  }
  const ls = line ? getComputedStyle(line) : null;
  const lr = line ? line.getBoundingClientRect() : null;
  const chip = document.querySelector('[data-turn-state]');
  const before = chip ? chip.getAttribute('data-turn-state') : null;
  const PSEUDOS = ['::before', '::after'];
  const label = (el) => el.tagName + (el.id ? '#' + el.id : '') +
    (typeof el.className === 'string' && el.className.trim()
      ? '.' + el.className.trim().split(/\\s+/).join('.') : '');
  const anim = {};
  for (const state of __STATES__) {
    if (chip) chip.setAttribute('data-turn-state', state);
    const moving = [], pseudo_content = [];
    for (const el of [pane, ...pane.querySelectorAll('*')]) {
      const own = getComputedStyle(el).animationName;
      if (own !== 'none') moving.push(label(el) + ': ' + own);
      // A shimmer on ::before/::after is the same fake trajectory as one on the element.
      for (const p of PSEUDOS) {
        const ps = getComputedStyle(el, p);
        if (ps.animationName !== 'none') moving.push(label(el) + p + ': ' + ps.animationName);
        if (el.closest('[data-denoise-root]') && ps.content !== 'none' && ps.content !== 'normal') {
          pseudo_content.push(label(el) + p + ' content ' + ps.content);
        }
      }
    }
    // Web Animations too (el.animate(), or a CSS animation on a pseudo-element), but not
    // transitions: a running CSSTransition is a state change easing, not a trajectory.
    const scripted = document.getAnimations()
      .filter((a) => !(typeof CSSTransition === 'function' && a instanceof CSSTransition))
      .filter((a) => a.effect && a.effect.target && pane.contains(a.effect.target))
      .map((a) => label(a.effect.target) + (a.effect.pseudoElement || '') + ': ' +
                  (a.animationName || a.id || a.constructor.name));
    anim[state] = {pane: getComputedStyle(pane).animationName, moving, pseudo_content, scripted};
  }
  if (chip) chip.setAttribute('data-turn-state', before);
  return {present: true,
          root_present: !!root,
          root_children: root ? root.children.length : null,
          root_child_tags: root ? [...root.children].map((c) => c.tagName) : [],
          line_present: !!line,
          line_visible: !!line && ls.display !== 'none' && ls.visibility !== 'hidden' &&
                        parseFloat(ls.opacity) > 0 && lr.width > 0 && lr.height > 0 &&
                        !line.closest('[hidden]'),
          meter_in_pane: !!pane.querySelector('#thinking-meter'),
          bar_in_pane: !!pane.querySelector('#thinking-bar'),
          animation: anim};
})()
""".replace("__EMPTY__", json.dumps(DENOISE_EMPTY)).replace("__STATES__", json.dumps(TURN_STATES))

HEADINGS_JS = """
(() => [...document.querySelectorAll('main .pane h2')].map((h) => ({
  text: (h.textContent || '').trim(),
  rendered: (h.innerText || '').trim(),
  transform: getComputedStyle(h).textTransform,
  first_letter: getComputedStyle(h, '::first-letter').textTransform,
})))()
"""

CONTROL_JS = (
    "(() => {"
    + FIND_THEME_CONTROL_JS
    + """
  const c = themeControl();
  if (!c) return {present: false};
  const named = (c.getAttribute('aria-label') || '').trim() ||
    (c.getAttribute('aria-labelledby') || '').trim() ||
    [...(c.labels || [])].map((l) => l.textContent.trim()).join(' ');
  return {present: true, value: c.value, in_header: !!c.closest('header'), name: named,
          options: [...c.options].map((o) => [o.value, o.textContent.trim()]),
          stored: (() => { try { return localStorage.getItem('__KEY__'); } catch (e) { return 'THROWS'; } })()};
})()""".replace("__KEY__", THEME_KEY)
)

#: T-0181 B3: a window-level capturing keydown listener records every key that reached the page (before
#: any page handler can stop it), the premise of "`m` does nothing".
KEYS_SEEN_JS = (
    "(() => { window.__t0181_keys = []; if (!window.__t0181_keys_on) { window.__t0181_keys_on = true;"
    " window.addEventListener('keydown', (e) => window.__t0181_keys.push(e.key), true); } return true; })()"
)

CAPTURE_JS = """
(() => ({mute_enabled: !document.getElementById('mute-button').disabled,
         banner: !document.getElementById('capture-banner').hidden,
         active: document.activeElement ? document.activeElement.tagName : null}))()
"""

FOCUS_RING_JS = (
    "(() => {"
    + CHAIN_JS
    + FIND_THEME_CONTROL_JS
    + """
  const c = themeControl();
  if (!c) return {present: false};
  const s = getComputedStyle(c);
  return {present: true, focused: document.activeElement === c, focus_visible: c.matches(':focus-visible'),
          outline_width: parseFloat(s.outlineWidth) || 0, outline_style: s.outlineStyle,
          outline_color: s.outlineColor, parent_chain: chainOf(c.parentElement || c)};
})()"""
)

ASSIST_BOX_JS = """
(() => {
  // T-0162 B-9: the switches live in the Settings popover; open it so they are laid out.
  { const g = document.querySelector('header button[aria-label="Settings"]'); const p = document.querySelector('[data-settings-popover] #assist-panel'); if (g && !(p && p.checkVisibility())) g.click(); }
  const boxes = [...document.querySelectorAll('[data-settings-popover] #assist-panel input[type=checkbox]')];
  return {count: boxes.length, all_checkbox_inputs: boxes.every((b) => b.type === 'checkbox'),
          checked_at_load: boxes.filter((b) => b.checked).length,
          rects: boxes.map((b) => { const r = b.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; })};
})()
"""

#: Every control on the page, activated once; after each, the rail is re-checked (criterion 4).
CLICK_ALL_JS = """
(async () => {
  const RAIL = __RAIL__;
  const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const railBroken = () => Object.entries(RAIL).flatMap(([name, sel]) => {
    const el = document.querySelector(sel);
    if (!el) return [name + ': gone'];
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    const bad = [];
    if (el.closest('[hidden]')) bad.push('hidden');
    if (s.display === 'none') bad.push('display none');
    if (s.visibility === 'hidden') bad.push('visibility hidden');
    if (r.width <= 0 || r.height <= 0) bad.push(r.width + 'x' + r.height);
    return bad.length ? [name + ': ' + bad.join(', ')] : [];
  });
  const controls = [...document.querySelectorAll(
    'button, input, select, textarea, summary, label, a[href], [role=button], [role=switch], ' +
    '[role=tab], [tabindex]:not([tabindex="-1"])')];
  const offenders = [], activated = [], skipped = [];
  const name = (el) => el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') +
    (el.getAttribute('aria-label') ? '[' + el.getAttribute('aria-label') + ']' : '') +
    (el.textContent ? ' "' + el.textContent.trim().slice(0, 24) + '"' : '');
  for (const el of controls) {
    if (!el.isConnected) continue;
    if (el.tagName === 'A' && !/^#/.test(el.getAttribute('href') || '')) { skipped.push(name(el)); continue; }
    if (el.tagName === 'SELECT') {
      for (const opt of [...el.options]) {
        el.value = opt.value;
        el.dispatchEvent(new Event('input', {bubbles: true}));
        el.dispatchEvent(new Event('change', {bubbles: true}));
        await frame();
        const broken = railBroken();
        activated.push(name(el) + '=' + opt.value);
        if (broken.length) offenders.push(name(el) + '=' + opt.value + ' -> ' + broken.join('; '));
      }
      continue;
    }
    el.click();
    await frame();
    activated.push(name(el));
    const broken = railBroken();
    if (broken.length) offenders.push(name(el) + ' -> ' + broken.join('; '));
  }
  return {activated, offenders, skipped, before_any: railBroken()};
})()
""".replace("__RAIL__", json.dumps(RAIL))

#: The shown card's text, every leaf with its own text node, through shadow roots.
CARD_TEXT_JS = (
    "(() => {"
    + CHAIN_JS
    + """
  // T-0154 B4: one <a2ui-surface> per surface id, so the t9 card is addressed by its wrapper
  // (was: the panel's one surface, `.a2ui-panel a2ui-surface`, or the first on the page).
  // T-0164 B2: re-keyed by name to [data-a2ui-root] once group B is activated (light DOM: the
  // shadow walk below then finds no shadow root and walks the host's own subtree only).
  const host = document.querySelector('[data-ui-surface="turn-t9"] ' + __A2UI_ROOT__);
  if (!host) return {host: false, items: []};
  const items = [];
  const walk = (root) => {
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) walk(el.shadowRoot);
      const own = [...el.childNodes].filter((n) => n.nodeType === 3).map((n) => n.textContent).join('').trim();
      if (!own) continue;
      const s = getComputedStyle(el), r = el.getBoundingClientRect();
      if (r.width <= 0 || r.height <= 0 || s.display === 'none' || s.visibility === 'hidden') continue;
      // T-0164 C2: each leaf is keyed by its role, the model's component id on the nearest
      // [data-a2-id] (class names are the implementation's to choose).
      const owner = el.closest('[data-a2-id]');
      const role = owner ? owner.getAttribute('data-a2-id') : null;
      items.push({selector: role ? '[data-a2-id="' + role + '"]' : el.tagName.toLowerCase(),
                  role, index: items.length, pseudo: null,
                  text: own.slice(0, 48), color: s.color, font_size: parseFloat(s.fontSize),
                  font_weight: parseInt(s.fontWeight, 10) || 400, opacity: parseFloat(s.opacity),
                  chain: chainOf(el)});
    }
  };
  walk(host);
  if (host.shadowRoot) walk(host.shadowRoot);
  return {host: true, items};
})()"""
).replace("__A2UI_ROOT__", json.dumps(A2UI_ROOT))


def _await_frame(chrome: Chrome) -> None:
    chrome.evaluate(
        "new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))"
    )


def _theme(chrome: Chrome) -> dict[str, Any]:
    return dict(chrome.evaluate(THEME_STATE_JS) or {})


def _screenshot(chrome: Chrome, rect: list[float]) -> str:
    x, y, w, h = rect
    shot = chrome._cmd(
        "Page.captureScreenshot",
        format="png",
        clip={"x": x - 2, "y": y - 2, "width": w + 4, "height": h + 4, "scale": 1},
    )
    return str(shot.get("data") or "")


# --- the shell session ---------------------------------------------------------------------------------


@dataclass
class Shell:
    error: str = ""
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0
    layout: dict[str, Any] = field(default_factory=dict)
    narrow: dict[str, Any] = field(default_factory=dict)
    appended: dict[str, Any] = field(default_factory=dict)
    thinking: dict[str, Any] = field(default_factory=dict)
    fill: dict[str, Any] = field(default_factory=dict)
    headings: list[Any] = field(default_factory=list)
    control_at_load: dict[str, Any] = field(default_factory=dict)
    theme: dict[str, Any] = field(default_factory=dict)
    focus: dict[str, Any] = field(default_factory=dict)
    assists: dict[str, Any] = field(default_factory=dict)
    reduced: dict[str, Any] = field(default_factory=dict)
    click_all: dict[str, Any] = field(default_factory=dict)
    baseline_console: list[str] = field(default_factory=list)
    baseline_exceptions: list[str] = field(default_factory=list)
    baseline_panels_line: str | None = None
    storage: dict[str, Any] = field(default_factory=dict)


def _panels_line(lines: list[str]) -> str | None:
    hits = [line for line in lines if "panels mounted:" in line]
    return hits[-1] if hits else None


STORAGE_FAILURES = {
    "localStorage is undefined": (
        "Object.defineProperty(window, 'localStorage', {configurable: true, get() { return undefined; }});"
    ),
    "reading localStorage throws": (
        "Object.defineProperty(window, 'localStorage', {configurable: true, "
        "get() { throw new DOMException('denied (test)', 'SecurityError'); }});"
    ),
    "getItem throws": (
        "Storage.prototype.getItem = function () { "
        "throw new DOMException('denied (test)', 'SecurityError'); };"
    ),
    "setItem throws": (
        "Storage.prototype.setItem = function () { "
        "throw new DOMException('full (test)', 'QuotaExceededError'); };"
        "Storage.prototype.removeItem = function () { "
        "throw new DOMException('full (test)', 'QuotaExceededError'); };"
    ),
}

#: **Moved by T-0154 B3** (G2): criterion 11's row used to seed a ``section.turn > p.utterance``
#: into ``#spoken-pane`` (retired, B1) on the shell session and flip the chip's attribute. It now
#: runs on the replay session and measures the utterances the page itself rendered from the
#: fixture, inside ``[data-thread]``, while real ``astra.turn.state`` messages move the page through
#: all six states (for the last fixture turn, t11, and for a new turn, t12). Per utterance, keyed by
#: ``data-utterance-id``: colour, opacity, visibility, display, filter, decoration, and those of
#: its article (a rule that dims the whole article dims the utterance too).
#:
#: **Widened by O-B3 (3), F2** (reviewer 2: four dimmers survived the article-only read —
#: ``[data-interrupted] .assistant-body``, ``.turn[data-interrupted] > .assistant``, a
#: ``body:has(.chip[data-turn-state=…])`` rule and an inline ``style.opacity`` written by the
#: page's own JS). The look now walks the *whole* ancestor chain from the utterance up to and
#: including ``[data-thread]``: the effective opacity is the product of every element's computed
#: opacity on that chain (opacity does not inherit, so only the product says what the eye sees),
#: every non-``none`` filter on the chain is listed with its depth, and so is every visibility
#: other than ``visible`` and every display of ``none``/``contents``. Nothing was removed.
UTTERANCE_LOOK_JS = """
(() => {
  const out = {};
  const thread = document.querySelector('[data-thread]');
  for (const u of document.querySelectorAll('[data-thread] [data-utterance-id]')) {
    const s = getComputedStyle(u);
    const a = u.closest('article');
    const as = a ? getComputedStyle(a) : null;
    const r = u.getBoundingClientRect();
    let alpha = 1;
    const filters = [], hidden = [], displays = [];
    let depth = 0, reached = false;
    for (let n = u; n; n = n.parentElement, depth += 1) {
      const ns = getComputedStyle(n);
      alpha *= parseFloat(ns.opacity);
      if (ns.filter && ns.filter !== 'none') filters.push(depth + ':' + ns.filter);
      if (ns.visibility !== 'visible') hidden.push(depth + ':' + ns.visibility);
      if (ns.display === 'none' || ns.display === 'contents') displays.push(depth + ':' + ns.display);
      if (n === thread) { reached = true; break; }
    }
    out[u.getAttribute('data-utterance-id')] = {
      text: (u.textContent || '').trim().slice(0, 48),
      alpha: alpha,
      look: [s.color, s.opacity, s.visibility, s.display, s.filter, s.textDecorationLine,
             as ? [as.opacity, as.visibility, as.display, as.filter].join(' ') : 'no article',
             r.width > 0 && r.height > 0 ? 'boxed' : 'no box',
             'chain opacity ' + alpha.toFixed(4),
             'chain filters [' + filters.join(', ') + ']',
             'chain visibility [' + hidden.join(', ') + ']',
             'chain display [' + displays.join(', ') + ']',
             reached ? 'in thread' : 'chain never reached [data-thread]'].join(' | '),
    };
  }
  return out;
})()
"""


# --- the one browser ----------------------------------------------------------------------------------
#
# T-0128 criterion 16 (the module in <= 60 s) and the orchestrator's budget ruling: the module's
# three sessions (shell, replay, missing theme module) share one headless Chrome, reset between
# them (blank page, no emulation, no init script, empty records; each session's own server port
# gives it a fresh origin and so a fresh localStorage).


@pytest.fixture(scope="module")
def browser() -> Any:
    """The module's one headless Chrome, or the reason it could not start (a string). Never
    raises: a session handed a string records it on its ``error`` and every row fails on it."""
    if CHROME is None:  # pragma: no cover - requires_chrome is the real guard
        yield "no chrome"
        return
    stack = contextlib.ExitStack()
    try:
        chrome = stack.enter_context(_Recorder(CHROME))
        chrome._cmd("Runtime.enable")
        chrome._cmd("Network.enable")
    except Exception as exc:
        stack.close()
        yield f"headless Chrome did not start: {type(exc).__name__}: {exc}"
        return
    with stack:
        yield chrome


@contextlib.contextmanager
def _session(browser: _Recorder | str) -> Iterator[_Recorder]:
    if isinstance(browser, str):
        raise RuntimeError(browser)
    browser.reset()
    yield browser


def _run_shell(result: Shell, browser: _Recorder | str) -> None:
    server = HudStaticServer(HUD, 0)
    try:
        ws = f"ws://127.0.0.1:{closed_loopback_port()}/v1/realtime"
        url = f"http://127.0.0.1:{server.port}/?source=canvas&ws=" + urllib.parse.quote(ws, safe="")
        with _session(browser) as chrome:
            chrome._cmd("Runtime.enable")
            chrome.media(_scheme("dark"))
            chrome.navigate(url)
            if not chrome.wait_panes(ALL_PANES):
                result.notes.append(
                    "only "
                    + str(chrome.evaluate("document.querySelectorAll('main > .pane').length"))
                    + f" of {ALL_PANES} panes mounted in 25 s"
                )
            chrome.drain(1.0)
            result.baseline_console = list(chrome.console)
            result.baseline_exceptions = list(chrome.exceptions)
            result.baseline_panels_line = _panels_line(chrome.console)
            result.control_at_load = dict(chrome.evaluate(CONTROL_JS) or {})
            result.assists["at_load"] = chrome.evaluate(ASSIST_BOX_JS) or {}

            # criterion 4: the rail at three wide viewports, one column at 1000 px
            for name, (w, h) in WIDE.items():
                chrome.viewport(w, h)
                result.layout[name] = chrome.evaluate(LAYOUT_JS) or {}
            chrome.viewport(*NARROW)
            result.narrow = chrome.evaluate(LAYOUT_JS) or {}
            chrome.viewport(*WIDE["1440x900"])

            # criterion 3: an appended pane; criterion 5: the Thinking pane with zero messages
            result.appended = chrome.evaluate(APPENDED_JS) or {}
            result.thinking = chrome.evaluate(THINKING_JS) or {}
            result.fill = chrome.evaluate(FILL_JS) or {}
            result.headings = list(chrome.evaluate(HEADINGS_JS) or [])

            # criterion 6: system scheme, live; the control; a stored choice; a reload
            th = result.theme
            th["system_dark"] = _theme(chrome)
            chrome.media(_scheme("light"))
            th["flip_to_light"] = _theme(chrome)  # two frames after the change, no reload
            chrome.media(_scheme("dark"))
            th["flip_to_dark"] = _theme(chrome)
            chrome.media(_scheme("light"))
            th["choose_dark_under_light"] = chrome.evaluate(choose_theme_js("dark")) or {}
            th["stored_after_dark"] = (chrome.evaluate(CONTROL_JS) or {}).get("stored")
            chrome.media(_scheme("dark"))
            _await_frame(chrome)
            chrome.media(_scheme("light"))
            th["dark_after_system_flips"] = _theme(chrome)
            chrome.media(_scheme("dark"))
            th["choose_light_under_dark"] = chrome.evaluate(choose_theme_js("light")) or {}
            chrome.reload()
            time.sleep(0.5)
            chrome.wait_panes(ALL_PANES)
            th["after_reload"] = _theme(chrome)
            th["control_after_reload"] = dict(chrome.evaluate(CONTROL_JS) or {})
            th["choose_system_under_dark"] = chrome.evaluate(choose_theme_js("system")) or {}
            th["control_after_system"] = dict(chrome.evaluate(CONTROL_JS) or {})
            chrome.media(_scheme("light"))
            th["system_follows_again"] = _theme(chrome)
            chrome.media(_scheme("dark"))
            _await_frame(chrome)

            # criterion 8: Tab reaches the control, it shows :focus-visible, `m` does not mute
            chrome.wait_until(
                "!document.getElementById('mute-button').disabled", 10
            )  # ?source=canvas starts the synthetic capture at load
            result.focus["capture_before"] = chrome.evaluate(CAPTURE_JS) or {}
            chrome.evaluate("document.activeElement && document.activeElement.blur()")
            tabs = 0
            reached = False
            for _ in range(40):
                tabs += 1
                chrome.key("Tab", code=9)
                if chrome.evaluate(
                    "(() => {"
                    + FIND_THEME_CONTROL_JS
                    + " return document.activeElement === themeControl(); })()"
                ):
                    reached = True
                    break
            result.focus["tabs"] = tabs
            result.focus["reached"] = reached
            ring: dict[str, Any] = {}
            for theme in ("dark", "light"):
                chrome.evaluate(choose_theme_js(theme))
                ring[theme] = chrome.evaluate(FOCUS_RING_JS) or {}
            chrome.evaluate(choose_theme_js("system"))
            result.focus["ring"] = ring
            result.focus["focused_before_m"] = (chrome.evaluate(FOCUS_RING_JS) or {}).get("focused")
            chrome.evaluate(KEYS_SEEN_JS)
            chrome.key("m", text="m", code=77)
            time.sleep(0.3)
            result.focus["capture_after_m_on_control"] = chrome.evaluate(CAPTURE_JS) or {}
            result.focus["keys_after_m_on_control"] = chrome.evaluate("window.__t0181_keys || []")
            chrome.evaluate("document.activeElement && document.activeElement.blur()")
            chrome.key("m", text="m", code=77)
            time.sleep(0.6)
            result.focus["capture_after_m_on_body"] = chrome.evaluate(CAPTURE_JS) or {}
            result.focus["keys_after_m_on_body"] = chrome.evaluate("window.__t0181_keys || []")

            # criterion 12: a checked and an unchecked assist differ in both themes
            for theme in ("dark", "light"):
                chrome.evaluate(choose_theme_js(theme))
                boxes = chrome.evaluate(ASSIST_BOX_JS) or {}
                if not boxes.get("rects"):
                    result.assists[theme] = {"error": "no assist checkbox"}
                    continue
                rect = boxes["rects"][0]
                chrome.evaluate(
                    "document.querySelector('[data-settings-popover] #assist-panel input[type=checkbox]').checked = false"
                )
                _await_frame(chrome)
                off = _screenshot(chrome, rect)
                chrome.evaluate(
                    "document.querySelector('[data-settings-popover] #assist-panel input[type=checkbox]').checked = true"
                )
                _await_frame(chrome)
                on = _screenshot(chrome, rect)
                chrome.evaluate(
                    "document.querySelector('[data-settings-popover] #assist-panel input[type=checkbox]').checked = false"
                )
                result.assists[theme] = {
                    "theme": _theme(chrome),
                    "off_len": len(off),
                    "on_len": len(on),
                    "differ": off != on,
                }
            chrome.evaluate(choose_theme_js("system"))

            # criterion 13: reduced motion, in both themes
            chrome.media(
                [
                    {"name": "prefers-reduced-motion", "value": "reduce"},
                    {"name": "prefers-color-scheme", "value": "dark"},
                ]
            )
            _await_frame(chrome)
            for theme in ("dark", "light"):
                chrome.evaluate(choose_theme_js(theme))
                time.sleep(0.2)
                result.reduced[theme] = {
                    "theme": _theme(chrome),
                    "motion": list(chrome.evaluate(MOTION_JS) or []),
                    "chips": chrome.evaluate(CHIP_STATES_JS) or {},
                }
            chrome.evaluate(choose_theme_js("system"))
            chrome.media(_scheme("dark"))

            # criterion 4: activate every control, re-check the rail after each
            result.click_all = chrome.evaluate(CLICK_ALL_JS) or {}
            chrome.evaluate(choose_theme_js("system"))

            # criterion 7: an init script makes localStorage unusable, four ways
            for kind, script in STORAGE_FAILURES.items():
                chrome.console.clear()
                chrome.exceptions.clear()
                added = chrome.add_init_script(script)
                chrome.media(_scheme("light"))
                chrome.navigate(url)
                time.sleep(0.5)
                mounted = chrome.wait_panes(ALL_PANES)
                chrome.drain(1.0)
                row: dict[str, Any] = {"mounted": mounted, "resolved": _theme(chrome)}
                row["choose_dark"] = chrome.evaluate(choose_theme_js("dark")) or {}
                row["choose_light"] = chrome.evaluate(choose_theme_js("light")) or {}
                row["choose_dark_again"] = chrome.evaluate(choose_theme_js("dark")) or {}
                chrome.drain(0.5)
                row["errors"] = chrome.evaluate(
                    "document.getElementById('error-strip').children.length"
                )
                row["console"] = list(chrome.console)
                row["exceptions"] = list(chrome.exceptions)
                row["panels_line"] = _panels_line(chrome.console)
                result.storage[kind] = row
                chrome.remove_init_script(added)
    finally:
        server.close()


@pytest.fixture(scope="module")
def shell(browser: Any) -> Any:
    """The shared Chrome, one origin, the phases above in order. Never raises: failures land on
    ``error``."""
    if CHROME is None:  # pragma: no cover - requires_chrome is the real guard
        yield Shell(error="no chrome")
        return
    result = Shell()
    started = time.monotonic()
    try:
        _run_shell(result, browser)
    except Exception as exc:  # a broken session must fail in a row, not here
        result.error = f"{type(exc).__name__}: {exc}"
    result.seconds = time.monotonic() - started
    yield result


def _ok(s: Any) -> None:
    assert not s.error, f"the headless session failed before any assertion ran: {s.error}"


# --- criterion 4: the rail -------------------------------------------------------------------------------


def _pane_rendered(name: str, snap: dict[str, Any]) -> list[str]:
    if not snap.get("present"):
        return [f"{name}: not on the page"]
    problems = []
    if snap["hidden"]:
        problems.append(f"{name}: inside a [hidden] subtree")
    if snap["display"] == "none" or snap["visibility"] == "hidden":
        problems.append(f"{name}: display {snap['display']}, visibility {snap['visibility']}")
    if snap["w"] <= 0 or snap["h"] <= 0:
        problems.append(f"{name}: a {snap['w']:.0f}x{snap['h']:.0f} box")
    return problems


#: A rail pane collapsed to a sliver is "rendered" by every box test and shows nothing. The floor
#: is the pane's own heading height (a pane must at least hold its title), and never under 32 px.
MIN_RAIL_PANE_H_PX = 32.0


def _rail_pane_tall_enough(name: str, snap: dict[str, Any]) -> list[str]:
    if not snap.get("present"):
        return []  # _pane_rendered already reports it
    heading_h = snap.get("heading_h")
    if heading_h is None:
        return [f"{name}: no <h2> to size the pane against"]
    floor = max(MIN_RAIL_PANE_H_PX, float(heading_h))
    if snap["h"] + 0.5 < floor:
        return [
            f"{name}: {snap['h']:.1f} px tall, shorter than its own heading ({heading_h:.1f} px) "
            f"or the {MIN_RAIL_PANE_H_PX:.0f} px floor: collapsed, not stacked"
        ]
    return []


@requires_chrome
@pytest.mark.parametrize("viewport", list(WIDE))
def test_the_five_rail_panes_sit_in_a_fixed_right_column(shell, viewport):
    """Criterion 4 (human choice 1): always rendered, 380 px +/- 20, right of the conversation."""
    _ok(shell)
    g = shell.layout[viewport]
    problems: list[str] = []
    rail = g["rail"]
    for name, snap in rail.items():
        problems += _pane_rendered(name, snap)
        problems += _rail_pane_tall_enough(name, snap)
    present = {k: v for k, v in rail.items() if v.get("present")}
    for name, snap in present.items():
        if abs(snap["w"] - RAIL_WIDTH_PX) > RAIL_TOLERANCE_PX:
            problems.append(f"{name}: {snap['w']:.0f} px wide, the rail is 380 +/- 20 px")
        if snap["x"] < -0.5 or snap["right"] > g["client_w"] + 0.5:
            problems.append(f"{name}: x {snap['x']:.0f}..{snap['right']:.0f} leaves the viewport")
        if snap["bottom"] > g["inner_h"] + 0.5 and not snap["reachable"]:
            problems.append(
                f"{name}: ends at {snap['bottom']:.0f} of {g['inner_h']} px with no scroll area "
                "that reaches it (clipped by overflow: hidden?)"
            )
    xs = [v["x"] for v in present.values()]
    if xs and max(xs) - min(xs) > 1.0:
        problems.append(f"the rail panes do not share one column: left edges {sorted(xs)}")
    conv_right = max(
        (v["right"] for v in g["conversation"].values() if v.get("present")), default=0
    )
    if xs and min(xs) < conv_right - 0.5:
        problems.append(
            f"the rail starts at x={min(xs):.0f}, left of the conversation's right edge "
            f"{conv_right:.0f}: it is not a right column"
        )
    if g["scroll_w"] > g["client_w"]:
        problems.append(f"horizontal scrollbar: scrollWidth {g['scroll_w']} > {g['client_w']}")
    for p in g["panes"]:
        if p["w"] + 1e-6 < MIN_PANE_WIDTH_PX:
            problems.append(f"pane {p['label']!r} is {p['w']:.0f} px wide (floor 280)")
    assert problems == [], f"T-0128 criterion 4 at {viewport}:\n  " + "\n  ".join(problems)


@requires_chrome
def test_below_1100_px_the_rail_panes_stack_under_the_conversation(shell):
    """Criterion 4: one column at 1000 px, the rail panes under the conversation, never hidden."""
    _ok(shell)
    g = shell.narrow
    problems: list[str] = []
    for name, snap in {**g["conversation"], **g["rail"]}.items():
        problems += _pane_rendered(name, snap)
    for name, snap in g["rail"].items():
        problems += _rail_pane_tall_enough(name, snap)
    panes = g["panes"]
    if len(panes) < ALL_PANES:
        problems.append(f"{len(panes)} panes in <main>, expected {ALL_PANES}")
    xs = [p["x"] for p in panes]
    ws = [p["w"] for p in panes]
    if panes and (max(xs) - min(xs) > 1.0 or max(ws) - min(ws) > 1.0):
        problems.append(f"not one column at 1000 px: left edges {xs}, widths {ws}")
    conv_bottom = max(
        (v["bottom"] for v in g["conversation"].values() if v.get("present")), default=0
    )
    for name, snap in g["rail"].items():
        if snap.get("present") and snap["y"] < conv_bottom - 0.5:
            problems.append(
                f"{name} starts at y={snap['y']:.0f}, above the conversation's end {conv_bottom:.0f}"
            )
    if g["scroll_w"] > g["client_w"]:
        problems.append(f"horizontal scrollbar at 1000 px: {g['scroll_w']} > {g['client_w']}")
    assert problems == [], "T-0128 criterion 4 at 1000x800:\n  " + "\n  ".join(problems)


@requires_chrome
def test_activating_every_control_never_hides_the_rail(shell):
    """Criterion 4, adversarial: no element's activation hides the rail or a rail pane.

    Every button, input, select option, summary, label and focusable element is activated once
    (``click()``, or each option of a select as a change event), and the five rail panes are
    re-checked after each. Links to another document are skipped and listed.
    """
    _ok(shell)
    ca = shell.click_all
    assert ca.get("before_any") == [], f"premise: the rail is whole before any click: {ca!r}"
    assert len(ca.get("activated") or []) >= 6, (
        f"only {len(ca.get('activated') or [])} activations — the page has at least Share, Mute, the "
        f"ask box, four assist checkboxes, the log checkbox and three theme options: {ca!r}"
    )
    assert ca.get("offenders") == [], (
        "T-0128 criterion 4 (human choice 1: the rail has no toggle): these activations hid a rail "
        "pane:\n  " + "\n  ".join(ca.get("offenders") or [])
    )


@requires_chrome
def test_an_appended_pane_lands_in_the_rail_and_looks_like_a_declared_one(shell):
    """Criterion 3, last bullet (T-0056 criterion 4, retained): `main > section.pane` is a rail pane."""
    _ok(shell)
    a = shell.appended
    assert a.get("appended"), f"the appended pane was not measured: {a!r}"
    appended = a["appended"]
    problems = []
    for name in DECLARED_RAIL:
        declared = a["declared"].get(name)
        if not declared:
            problems.append(f"{name}: not on the page")
            continue
        for key in ("padding", "background", "radius", "x", "w"):
            if appended[key] != declared[key]:
                problems.append(f"{key}: appended {appended[key]!r} vs {name} {declared[key]!r}")
    assert problems == [], (
        "T-0128 criterion 3: a section.pane appended to <main> (what hud/src/assists/index.js, "
        "hud/src/log/index.js and T-0047's listen panel do) must sit in the rail column with the "
        "same padding, background and radius as a declared rail pane (implementation note 1: "
        "`.pane` defaults to the rail column):\n  " + "\n  ".join(problems)
    )


# --- criterion 3: the rendered headings ---------------------------------------------------------------


@requires_chrome
def test_the_pane_headings_render_in_sentence_case(shell):
    """Criterion 3: "the pane headings become sentence case", as rendered.

    Either an HTML edit or an ``h2::first-letter { text-transform: uppercase }`` satisfies it;
    ``capitalize`` or ``uppercase`` on the heading does not (that is title or upper case).
    """
    _ok(shell)
    # **Amended by T-0154 B1** (G2): was ``>= ALL_PANES`` (every pane had a heading). The one
    # conversation pane has none (ruling O-B (1): the mockup has none), so the floor is the
    # rail's five headings.
    assert len(shell.headings) >= len(RAIL), f"headings measured: {shell.headings!r}"
    problems = []
    for h in shell.headings:
        rendered = h["rendered"] or h["text"]
        if h["transform"] in ("uppercase", "capitalize"):
            problems.append(f"{h['text']!r}: text-transform {h['transform']} is not sentence case")
            continue
        first_upper = rendered[:1].isupper() or (
            rendered[:1].islower() and h["first_letter"] == "uppercase"
        )
        rest = re.sub(r"\bI\b", "i", rendered[1:])
        if not first_upper or rest != rest.lower():
            problems.append(
                f"{h['text']!r} renders as {rendered!r} (::first-letter {h['first_letter']})"
            )
    assert problems == [], "T-0128 criterion 3, sentence case:\n  " + "\n  ".join(problems)


# --- criterion 5: the Thinking pane never fakes a trajectory -------------------------------------------


def _thinking_problems(t: dict[str, Any]) -> list[str]:
    if not t.get("present"):
        return ["main > .pane-thinking is not on the page"]
    problems = []
    if not t["root_present"]:
        problems.append("no [data-denoise-root] in the Thinking pane")
    elif t["root_children"] != 0:
        problems.append(f"[data-denoise-root] holds element children: {t['root_child_tags']}")
    if not t["line_present"]:
        problems.append(f"no element in the pane reads {DENOISE_EMPTY!r}")
    elif not t["line_visible"]:
        problems.append("the empty line is in the DOM but not visible")
    if not (t["meter_in_pane"] and t["bar_in_pane"]):
        problems.append("#thinking-meter and #thinking-bar are not both in the Thinking pane")
    animation = t.get("animation") or {}
    if set(animation) != set(TURN_STATES):
        problems.append(f"animation probed in {sorted(animation)}, not every turn state")
    for state, a in animation.items():
        if a["pane"] != "none" or a["moving"]:
            problems.append(
                f"{state}: pane animation-name {a['pane']}, animating (element or "
                f"::before/::after): {a['moving']}"
            )
        if a["scripted"]:
            problems.append(f"{state}: running animations on the pane: {a['scripted']}")
        if a["pseudo_content"]:
            problems.append(
                f"{state}: a ::before/::after inside [data-denoise-root] paints content (a "
                f"placeholder drawn in CSS is still a fake canvas): {a['pseudo_content']}"
            )
    return problems


@requires_chrome
def test_the_thinking_pane_is_empty_and_says_so_with_zero_messages(shell):
    """Criterion 5 (negative; D15, CLAUDE.md): no canvas data, so no trajectory and an honest line."""
    _ok(shell)
    problems = _thinking_problems(shell.thinking)
    paints = (shell.fill or {}).get("paints") or {}
    if len(set(paints.values())) != 3:
        problems.append(f"#thinking-bar does not track --fill (0/50/100 %): {paints!r}")
    assert problems == [], (
        "T-0128 criterion 5 with no astra.canvas.* message received:\n  "
        + "\n  ".join(problems)
        + "\n(The denoising view shows real per-step canvases or an empty state, never a "
        "placeholder animation; the plugin tap is not relayed to the page today — H-T128-4.)"
    )


# --- criterion 6: theme resolution in the browser --------------------------------------------------------


@requires_chrome
def test_the_theme_follows_an_emulated_system_flip_live(shell):
    """Criterion 6 (b)/(c): nothing stored, the system flips, data-theme and color-scheme follow."""
    _ok(shell)
    th = shell.theme
    for key, expected in (
        ("system_dark", "dark"),
        ("flip_to_light", "light"),
        ("flip_to_dark", "dark"),
    ):
        got = th.get(key) or {}
        assert got.get("theme") == expected and got.get("scheme") == expected, (
            f"criterion 6, {key}: <html data-theme> {got.get('theme')!r} and color-scheme "
            f"{got.get('scheme')!r}, expected {expected!r} for both, within one frame of the "
            "emulated change and with no reload"
        )
    dark = parse_css_color(str(th["system_dark"].get("body")))
    light = parse_css_color(str(th["flip_to_light"].get("body")))
    assert relative_luminance(light[:3]) > relative_luminance(dark[:3]), (
        "the light theme's page background is not lighter than the dark theme's: "
        f"{th['flip_to_light'].get('body')} vs {th['system_dark'].get('body')}"
    )


@requires_chrome
def test_a_stored_choice_beats_the_system_and_survives_a_reload(shell):
    """Criterion 6 (a), (d) and human choice 3, adversarially: the stored theme wins over the flip."""
    _ok(shell)
    th = shell.theme
    chosen = th.get("choose_dark_under_light") or {}
    assert chosen.get("ok") and chosen.get("theme") == "dark", (
        f"criterion 8: choosing Dark in the control under a light system gives {chosen!r}"
    )
    assert th.get("stored_after_dark") == "dark", (
        f"human choice 3: the choice persists under {THEME_KEY}; it reads "
        f"{th.get('stored_after_dark')!r}"
    )
    assert (th.get("dark_after_system_flips") or {}).get("theme") == "dark", (
        "criterion 6 (c), adversarial: with Dark stored, the system flipping dark -> light did "
        f"not leave the page dark: {th.get('dark_after_system_flips')!r}"
    )
    assert (th.get("after_reload") or {}).get("theme") == "light", (
        "persistence across a reload: Light was chosen under a dark system and the page reloaded "
        f"under the same dark system; it reads {th.get('after_reload')!r}"
    )
    assert (th.get("control_after_reload") or {}).get("value") == "light", (
        "criterion 8: the control reflects the current preference at load; after the reload it "
        f"shows {(th.get('control_after_reload') or {}).get('value')!r}"
    )
    system = th.get("choose_system_under_dark") or {}
    assert system.get("theme") == "dark", (
        f"criterion 6 (d): choosing System under a dark system re-resolves at once to dark: {system!r}"
    )
    assert (th.get("control_after_system") or {}).get("stored") in (None, "system"), (
        "criterion 6 (d): System removes the override; the key holds "
        f"{(th.get('control_after_system') or {}).get('stored')!r}"
    )
    assert (th.get("system_follows_again") or {}).get("theme") == "light", (
        "criterion 6 (c)/(d): after System, the page follows the system again"
    )


# --- criterion 7: localStorage that throws, in the browser ------------------------------------------------


@requires_chrome
@pytest.mark.parametrize("kind", list(STORAGE_FAILURES))
def test_the_page_survives_an_unusable_local_storage(shell, kind):
    """Criterion 7, with an init script: renders, resolves from the system, switches, stays quiet."""
    _ok(shell)
    row = shell.storage.get(kind)
    assert row, f"the {kind!r} navigation was not run"
    assert row["mounted"], f"criterion 7 ({kind}): the page did not render its {ALL_PANES} panes"
    assert row["resolved"].get("theme") == "light", (
        f"criterion 7 ({kind}): the system is emulated light, so the theme resolves light: "
        f"{row['resolved']!r}"
    )
    for key, expected in (
        ("choose_dark", "dark"),
        ("choose_light", "light"),
        ("choose_dark_again", "dark"),
    ):
        got = row[key]
        assert got.get("ok") and got.get("theme") == expected, (
            f"criterion 7 ({kind}): the control still switches the theme for the life of the page; "
            f"{key} gave {got!r}"
        )
    assert row["exceptions"] == [], f"criterion 7 ({kind}): uncaught exceptions {row['exceptions']}"
    assert row["errors"] == 0, f"criterion 7 ({kind}): #error-strip gained {row['errors']} line(s)"
    baseline = collections.Counter(shell.baseline_console)
    new = [line for line in row["console"] if line not in baseline]
    assert len(new) <= 1, (
        f"criterion 7 ({kind}): at most one console line per kind of failure; lines a normal load "
        f"does not write:\n  " + "\n  ".join(new)
    )
    assert row["panels_line"] == shell.baseline_panels_line, (
        f"criterion 7 ({kind}): mountPanels reports {row['panels_line']!r}; a normal load reports "
        f"{shell.baseline_panels_line!r}"
    )


# --- criterion 8: the control ---------------------------------------------------------------------------


@requires_chrome
def test_the_settings_control_is_native_labelled_reachable_and_does_not_eat_m(shell):
    """Criterion 8: three options, labelled, reflects the preference, Tab, :focus-visible, `m`."""
    _ok(shell)
    c = shell.control_at_load
    assert c.get("present"), "criterion 8: no single <select> in <header> with system/light/dark"
    assert c["in_header"], "the control is in the header"
    assert c["options"] == [["system", "System"], ["light", "Light"], ["dark", "Dark"]], c[
        "options"
    ]
    assert c["name"], "criterion 8: the control has an accessible label"
    assert c["value"] == "system", (
        f"nothing stored: the control reflects System at load, not {c['value']!r}"
    )
    f = shell.focus
    assert f.get("reached"), (
        f"criterion 8: Tab never reached the control in {f.get('tabs')} presses"
    )
    for theme, ring in (f.get("ring") or {}).items():
        assert ring.get("focused") and ring.get("focus_visible"), (
            f"criterion 8 ({theme}): the control is focused and matches :focus-visible: {ring!r}"
        )
        behind = painted(ring["parent_chain"])
        colour = parse_css_color(str(ring["outline_color"]))
        visible = ring["outline_width"] > 0 and ring["outline_style"] != "none"
        ratio = contrast_ratio(composite(colour, behind), behind) if visible else 0.0
        assert ratio >= 3.0, (
            f"criterion 10 ({theme}): the focus ring on the control is {ratio:.2f}:1 against "
            f"{hexc(behind)} ({ring['outline_width']}px {ring['outline_style']} {ring['outline_color']})"
        )
    before = f.get("capture_before") or {}
    assert before.get("mute_enabled") and before.get("banner"), (
        f"premise: capture is running: {before!r}"
    )
    assert f.get("focused_before_m"), "premise: the control holds the focus when `m` is pressed"
    after = f.get("capture_after_m_on_control") or {}
    assert after.get("mute_enabled") and after.get("banner"), (
        "criterion 8 (A19's fix): pressing m while the Settings control has focus stopped the "
        f"capture: {after!r}. hud/src/main.js mutes on a bare `m`; the control's key handler stops "
        "propagation (implementation note 3)."
    )
    assert "m" in (f.get("keys_after_m_on_control") or []), (
        f"premise: the m keydown on the control reached the page: {f.get('keys_after_m_on_control')}"
    )
    # T-0181 B3 (H-T181-4) re-scoped the old anti-vacuity line ("`m` with nothing focused must still
    # mute") to its opposite: the next row pins that `m` on the body does nothing.


@requires_chrome
def test_m_on_the_body_does_nothing(shell):
    """T-0181 B3 (H-T181-4, James 2026-10-28): the ``m`` shortcut is removed, so ``m`` with nothing
    focused, during a running capture, stops nothing. The key did reach the page (a window capture
    listener saw both ``m`` presses: the one on the Settings control and the one on the body)."""
    _ok(shell)
    f = shell.focus
    before = f.get("capture_before") or {}
    assert before.get("mute_enabled") and before.get("banner"), (
        f"premise: capture is running: {before!r}"
    )
    seen = f.get("keys_after_m_on_body") or []
    assert seen.count("m") == 2, f"premise: both m keydowns reached the page: {seen}"
    body = f.get("capture_after_m_on_body") or {}
    assert body.get("mute_enabled") and body.get("banner"), (
        f"B3: `m` with nothing focused stopped the capture (no keydown stops it): {body!r}"
    )


# --- criterion 11: no turn-state styling of the spoken text ---------------------------------------------


@requires_chrome
def test_the_spoken_text_looks_the_same_in_every_turn_state(replay):
    """Criterion 11 (H-T128-2), behaviourally: an utterance computes the same in all six states.

    **Moved by T-0154 B3** (G2): from the shell session's seeded ``p.utterance`` in the retired
    ``#spoken-pane`` to the replay session's real rendered utterances in ``[data-thread]``, with
    the states delivered as ``astra.turn.state`` messages for the utterances' own last turn (t11)
    and for a new one (t12). The assertion is unchanged in kind and stronger in scope: every
    rendered utterance, not one seeded paragraph, keeps one look across all twelve measurements.
    """
    _ok(replay)
    by_state = replay.utterance_by_state
    expected = {f"{turn}:{state.value}" for turn in ("t11", "t12") for state in TurnState}
    assert set(by_state) == expected, f"measured {sorted(by_state)}"
    ids = {uid for looks in by_state.values() for uid in looks}
    assert ids, (
        "premise (T-0154 B3): the replay rendered no [data-thread] [data-utterance-id] at all, so "
        "there was no spoken text to measure"
    )
    assert all(set(looks) == ids for looks in by_state.values()), (
        "B3: no utterance appears or disappears with the turn state: "
        + "; ".join(f"{k}: {sorted(v)}" for k, v in by_state.items())
    )
    # O-B3 (3), F2: the chain walk really reached [data-thread] (else the product covers less than
    # the ruling's chain and a dimmer between the break and the thread would go unseen).
    short = sorted(
        f"{k}/{uid}"
        for k, v in by_state.items()
        for uid, row in v.items()
        if "chain never reached [data-thread]" in row["look"]
    )
    assert short == [], f"F2: the ancestor walk stopped short of [data-thread] for {short}"
    changed = []
    for uid in sorted(ids):
        looks = {k: v[uid]["look"] for k, v in by_state.items()}
        if len(set(looks.values())) != 1:
            changed.append(f"{uid}: " + "; ".join(f"{k} -> {v}" for k, v in looks.items()))
    assert changed == [], (
        "criterion 11 / T-0154 B3: the spoken text's colour/opacity/visibility/display/filter/"
        "decoration (or its article's) changes with the turn state:\n  " + "\n  ".join(changed)
    )


# --- criterion 12: the assists ----------------------------------------------------------------------------


@requires_chrome
def test_assists_are_unchecked_checkboxes_whose_checked_state_shows_in_both_themes(shell):
    """Criterion 12: input[type=checkbox], unchecked at load, checked != unchecked in both themes.

    "Computed appearance" is read as pixels: a clipped screenshot of the first assist checkbox,
    unchecked and checked, must differ (a style-only comparison misses `appearance: none` with no
    :checked rule, which is exactly how a restyle loses the state).
    """
    _ok(shell)
    load = shell.assists.get("at_load") or {}
    assert load.get("count") == 4 and load.get("all_checkbox_inputs"), f"four checkboxes: {load!r}"
    assert load.get("checked_at_load") == 0, "pixel-first: every assist is unchecked at load"
    for theme in ("dark", "light"):
        row = shell.assists.get(theme) or {}
        assert not row.get("error"), row
        assert (row.get("theme") or {}).get("theme") == theme, (
            f"premise: data-theme {theme}: {row!r}"
        )
        assert row["off_len"] > 0 and row["on_len"] > 0, "the screenshots are not empty"
        assert row["differ"], (
            f"criterion 12 (data-theme={theme}): a checked and an unchecked assist checkbox render "
            "identically — the human cannot see which assists are on"
        )


# --- criterion 13: reduced motion, both themes -------------------------------------------------------------


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_reduced_motion_is_off_and_the_states_stay_distinct_in_both_themes(shell, theme):
    """Criterion 13: under reduce, every animation/transition <= 0.01 s; six distinct chip looks."""
    _ok(shell)
    row = shell.reduced.get(theme) or {}
    assert (row.get("theme") or {}).get("theme") == theme, f"premise: data-theme {theme}: {row!r}"
    offenders = []
    for m in row.get("motion") or []:
        animating = (
            m["animation_name"] != "none"
            and max(_durations(m["animation_duration"]) or [0.0]) > 0.01
        )
        if animating or max(_durations(m["transition_duration"]) or [0.0]) > 0.01:
            offenders.append(str(m))
    assert offenders == [], f"criterion 13 ({theme}):\n  " + "\n  ".join(offenders)
    chips = row.get("chips") or {}
    assert set(chips) == set(TURN_STATES)
    looks = {json.dumps(v, sort_keys=True) for v in chips.values()}
    assert len(looks) == len(TURN_STATES), (
        f"criterion 13 ({theme}): the six turn states need six distinct treatments without motion"
    )


# --- the replay session: the committed fixture through HudStub ----------------------------------------------


@dataclass
class Replay:
    error: str = ""
    seconds: float = 0.0
    thinking_after_t9: dict[str, Any] = field(default_factory=dict)
    card: dict[str, Any] = field(default_factory=dict)
    chip_text: dict[str, str] = field(default_factory=dict)
    requests: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    #: the gated card's Text components, and the stale turn-t9 wire surfaces left out
    card_texts: list[str] = field(default_factory=list)
    withheld: int = 0
    card_rendered: bool = False
    card_wait_s: float = 0.0
    #: T-0154 B3 / T-0128 criterion 11: "<turn>:<state>" -> UTTERANCE_LOOK_JS
    utterance_by_state: dict[str, Any] = field(default_factory=dict)


def _card_rendered_js(texts: list[str]) -> str:
    """True once every Text of the gated card is a measured leaf of the shown surface."""
    want = json.dumps([t[:48] for t in texts])
    return (
        f"(() => {{ const r = {CARD_TEXT_JS}; const got = new Set((r.items || []).map((i) => i.text));"
        f" return {want}.every((t) => got.has(t)); }})()"
    )


def _run_replay(result: Replay, browser: _Recorder | str) -> None:
    lines = fixture_lines()
    t9_end = _last_index_of(CARD_TURN)
    surfaces = card_surfaces(gated_card())
    result.card_texts = card_texts(surfaces[-1]) if surfaces else []
    stub = HudStub([]).start()
    server = HudStaticServer(HUD, 0)
    try:
        url = f"http://127.0.0.1:{server.port}/?source=canvas&ws=" + urllib.parse.quote(
            stub.url, safe=""
        )
        with _session(browser) as chrome:
            chrome._cmd("Runtime.enable")
            chrome._cmd("Network.enable")
            chrome.media(_scheme("dark"))
            chrome.navigate(url)
            stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20)
            stub.wait_for_client(10)
            chrome.wait_panes(ALL_PANES)
            chrome.viewport(*WIDE["1440x900"])
            for line in lines[: t9_end + 1]:
                if _is_stale_card_surface(line):
                    result.withheld += 1
                    continue
                stub.push(line)
                time.sleep(0.02)
            # the wire has reached the end of t9: the card the current gate makes, as published
            for surface in surfaces:
                stub.push(surface)
                time.sleep(0.02)
            waited = time.monotonic()
            result.card_rendered = bool(result.card_texts) and chrome.wait_until(
                _card_rendered_js(result.card_texts), 10
            )
            result.card_wait_s = time.monotonic() - waited
            _await_frame(chrome)
            result.thinking_after_t9 = chrome.evaluate(THINKING_JS) or {}
            for theme, opposite in (("dark", "light"), ("light", "dark")):
                chosen = chrome.evaluate(choose_theme_js(theme)) or {}
                chrome.media(_scheme(opposite))
                _await_frame(chrome)
                result.card[theme] = {
                    "chosen": chosen,
                    "theme": _theme(chrome),
                    "measured": chrome.evaluate(CARD_TEXT_JS) or {},
                }
            chrome.evaluate(choose_theme_js("system"))
            chrome.media(_scheme("dark"))
            for line in lines[t9_end + 1 :]:
                stub.push(line)
                time.sleep(0.02)
            time.sleep(0.5)
            chrome.media(
                [
                    {"name": "prefers-reduced-motion", "value": "reduce"},
                    {"name": "prefers-color-scheme", "value": "dark"},
                ]
            )
            # T-0154 B3 (criterion 11 moved here): the real rendered utterances under every
            # state, first for the last fixture turn (whose article holds utterances), then for
            # a new turn. The t12 pass is also the chip rows' measurement, as before.
            for turn in ("t11", "t12"):
                for state in TurnState:
                    stub.push(AstraTurnState(state=state, turn_id=turn))
                    chrome.wait_until(
                        "(document.querySelector('[data-turn-state]') || {}).textContent"
                        f" === {json.dumps(state.value)}",
                        3,
                    )
                    _await_frame(chrome)
                    result.utterance_by_state[f"{turn}:{state.value}"] = (
                        chrome.evaluate(UTTERANCE_LOOK_JS) or {}
                    )
                    if turn == "t12":
                        result.chip_text[state.value] = str(
                            chrome.evaluate(
                                "(document.querySelector('[data-turn-state]') || {}).textContent"
                            )
                        )
            chrome.drain(1.0)
            result.requests = list(chrome.requests)
            result.exceptions = list(chrome.exceptions)
    finally:
        server.close()
        stub.stop()


@pytest.fixture(scope="module")
def replay(browser: Any) -> Any:
    if CHROME is None:  # pragma: no cover
        yield Replay(error="no chrome")
        return
    result = Replay()
    started = time.monotonic()
    try:
        _run_replay(result, browser)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    result.seconds = time.monotonic() - started
    yield result


@requires_chrome
def test_after_replaying_t9_the_thinking_pane_still_shows_no_trajectory(replay):
    """Criterion 5 after t9 (the shopping-list card): the fixture holds no astra.canvas.* message."""
    _ok(replay)
    problems = _thinking_problems(replay.thinking_after_t9)
    assert problems == [], "T-0128 criterion 5 after replaying t3-t9:\n  " + "\n  ".join(problems)


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_t9_card_text_clears_its_floor_in_both_themes(replay, theme):
    """Criterion 10 (amended, ruling on f6b7d98): t9's gated card, >= 4.5:1 (3.0:1 large).

    The card is the surface the current gate makes from t9's raw ``ui`` commits, pushed through
    ``HudStub`` once the wire replay has reached the end of t9 (the wire's own stale turn-t9
    surfaces withheld). Every Text of that card must be a measured leaf, so a page still showing
    the empty pre-T-0127 Column, or a card that rendered only partly, fails here, not vacuously.
    """
    _ok(replay)
    assert replay.withheld == STALE_CARD_SURFACES, (
        f"premise: {replay.withheld} stale turn-t9 wire surfaces withheld, expected "
        f"{STALE_CARD_SURFACES}"
    )
    assert len(replay.card_texts) >= MIN_CARD_TEXTS, f"premise: gated texts {replay.card_texts}"
    row = replay.card.get(theme) or {}
    assert (row.get("chosen") or {}).get("ok"), f"the control could not choose {theme}: {row!r}"
    assert (row.get("theme") or {}).get("theme") == theme, f"premise: data-theme {theme}: {row!r}"
    items = (row.get("measured") or {}).get("items") or []
    measured = {str(m["text"]) for m in items}
    missing = [t for t in replay.card_texts if t[:48] not in measured]
    assert replay.card_rendered and missing == [], (
        f"the gated t9 card did not render on the page within 10 s "
        f"(waited {replay.card_wait_s:.1f} s): these Text components are not measured leaves of "
        f"the shown surface: {missing}; measured: {sorted(measured)}"
    )
    assert len(items) >= MIN_CARD_TEXTS, (
        f"t9's card rendered {len(items)} text elements; the row would be vacuous"
    )
    # T-0164 C2 (re-pointed at the new roles): the heading ``h`` and the nine list rows
    # ``i1``..``i9`` of the gated t9 card (hand-read component ids) are each a measured role.
    roles = {str(m.get("role")) for m in items}
    want_roles = {"h", *(f"i{i}" for i in range(1, 10))}
    assert want_roles <= roles, (
        f"T-0164 C2: these t9 roles are not measured leaves: {sorted(want_roles - roles)}"
    )
    failures = []
    for m in items:
        floor = threshold_for(float(m["font_size"]), int(m["font_weight"]))
        if ratio_of(m) + 1e-9 < floor:
            fg = parse_css_color(str(m["color"]))
            failures.append(
                f"{m['selector']} {m['text']!r}: {m['color']} ({hexc(fg[:3])}) at "
                f"{ratio_of(m):.2f}:1, needs {floor}:1"
            )
    assert failures == [], f"T-0128 criterion 10, t9 card, data-theme={theme}:\n  " + "\n  ".join(
        failures
    )


@requires_chrome
def test_the_chip_text_is_the_server_state_under_reduced_motion(replay):
    """Criterion 13: the chip's text equals the server state, all six, with motion off. A fence."""
    _ok(replay)
    assert replay.chip_text == {s.value: s.value for s in TurnState}, replay.chip_text


@requires_chrome
def test_the_replay_requests_nothing_off_loopback(replay):
    """Criterion 2: the fixture names a catalog at https://a2ui.org; nothing may fetch it. A fence."""
    _ok(replay)
    assert replay.requests, "the request log is empty, so this row would be vacuous"
    offenders = [
        u
        for u in replay.requests
        if urllib.parse.urlsplit(u).scheme in {"http", "https", "ws", "wss"}
        and urllib.parse.urlsplit(u).hostname not in {"127.0.0.1", "localhost", "::1"}
    ]
    assert offenders == [], f"criterion 2: non-loopback requests {offenders}"
    assert replay.exceptions == [], f"uncaught exceptions during the replay: {replay.exceptions}"


# --- adversarial: the theme module 404s ---------------------------------------------------------------------


@dataclass
class Missing:
    error: str = ""
    mounted: bool = False
    layout: dict[str, Any] = field(default_factory=dict)
    body: str = ""
    dark_body: str = ""
    exceptions: list[str] = field(default_factory=list)
    errors: Any = None
    panels_line: str | None = None
    theme_requested: bool = False


@pytest.fixture(scope="module")
def missing_theme(browser: Any) -> Any:
    """The page served from a copy of hud/ without src/theme.js: a real 404, not an interception."""
    if CHROME is None:  # pragma: no cover
        yield Missing(error="no chrome")
        return
    result = Missing()
    tmp = Path(tempfile.mkdtemp(prefix="astra-hud-no-theme-"))
    try:
        copy = tmp / "hud"
        shutil.copytree(HUD, copy, ignore=shutil.ignore_patterns("node_modules"))
        (copy / "src" / "theme.js").unlink(missing_ok=True)
        server = HudStaticServer(copy, 0)
        try:
            ws = f"ws://127.0.0.1:{closed_loopback_port()}/v1/realtime"
            url = f"http://127.0.0.1:{server.port}/?source=canvas&ws=" + urllib.parse.quote(
                ws, safe=""
            )
            with _session(browser) as chrome:
                chrome._cmd("Runtime.enable")
                chrome._cmd("Network.enable")
                chrome.media(_scheme("dark"))
                chrome.navigate(url)
                result.mounted = chrome.wait_panes(ALL_PANES)
                chrome.drain(1.0)
                chrome.viewport(*WIDE["1440x900"])
                result.layout = chrome.evaluate(LAYOUT_JS) or {}
                result.body = str(
                    chrome.evaluate("getComputedStyle(document.body).backgroundColor")
                )
                chrome.evaluate("document.documentElement.setAttribute('data-theme', 'dark')")
                _await_frame(chrome)
                result.dark_body = str(
                    chrome.evaluate("getComputedStyle(document.body).backgroundColor")
                )
                result.errors = chrome.evaluate(
                    "document.getElementById('error-strip').children.length"
                )
                result.exceptions = list(chrome.exceptions)
                result.panels_line = _panels_line(chrome.console)
                result.theme_requested = any(u.endswith("/src/theme.js") for u in chrome.requests)
        finally:
            server.close()
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    yield result


@requires_chrome
def test_the_page_still_renders_when_the_theme_module_404s(missing_theme, shell):
    """T3 adversarial: hud/src/theme.js 404s; the page renders, dark, with every pane and no throw."""
    _ok(missing_theme)
    m = missing_theme
    assert m.theme_requested, "premise: the page asked for src/theme.js and got a 404"
    assert m.mounted, f"the page did not mount its {ALL_PANES} panes without the theme module"
    problems = []
    for name, snap in {**m.layout.get("rail", {}), **m.layout.get("conversation", {})}.items():
        problems += _pane_rendered(name, snap)
    assert problems == [], "without the theme module:\n  " + "\n  ".join(problems)
    assert m.body == m.dark_body, (
        "without the theme module the page must still be the dark theme (:root declares the dark "
        f"tokens, criterion 9): body {m.body} vs {m.dark_body} under data-theme=dark"
    )
    assert m.exceptions == [], f"uncaught exceptions without the theme module: {m.exceptions}"
    assert m.errors == 0, f"#error-strip gained {m.errors} line(s)"
    _ok(shell)
    assert m.panels_line == shell.baseline_panels_line, (
        f"mountPanels without the theme module: {m.panels_line!r} vs {shell.baseline_panels_line!r}"
    )


# --- T6: the screenshot script ---------------------------------------------------------------------------------


def _playwright_chromium() -> str | None:
    """Why the screenshot smoke cannot run here, or None if Playwright and its Chromium are present."""
    if importlib.util.find_spec("playwright") is None:
        return "playwright is not installed (it comes from the `browser` extra)"
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            path = Path(p.chromium.executable_path)
    except Exception as exc:  # pragma: no cover - environment-dependent
        return f"playwright could not report its Chromium: {exc}"
    return None if path.exists() else f"Playwright's Chromium is not installed at {path}"


def _png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def test_the_screenshot_script_writes_eight_pngs_and_requests_only_loopback(tmp_path):
    """Criterion 15 / T6: 8 non-empty 1440x900 PNGs, requests.txt listing only 127.0.0.1.

    Skipped unless Playwright and its Chromium are installed. Runs the script the way criterion 15
    does (``python scripts/demo/hud_screens.py --out DIR`` in this interpreter, which has the
    ``browser`` extra when Playwright imports), and checks that nothing appears under artifacts/.
    """
    why = _playwright_chromium()
    if why:
        pytest.skip(why)
    script = REPO / "scripts" / "demo" / "hud_screens.py"
    assert script.is_file(), "T-0128 criterion 15 adds scripts/demo/hud_screens.py"
    artifacts = REPO / "artifacts"
    before = sorted(p.as_posix() for p in artifacts.rglob("*")) if artifacts.exists() else []
    out = tmp_path / "shots"
    proc = subprocess.run(
        [sys.executable, str(script), "--out", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"hud_screens.py failed:\n{proc.stdout}\n{proc.stderr}"
    pngs = sorted(out.glob("*.png"))
    assert len(pngs) == 8, f"criterion 15: 8 PNGs ({{dark, light}} x four states); got {pngs}"
    for png in pngs:
        data = png.read_bytes()
        assert len(data) > 0, f"{png.name} is empty"
        assert _png_size(data) == (1440, 900), f"{png.name} is {_png_size(data)}, not 1440x900"
    requests_txt = out / "requests.txt"
    assert requests_txt.is_file(), "criterion 15: the script writes requests.txt"
    hosts = []
    for line in requests_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        found = re.findall(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s]+", line)
        if found:
            for url in found:
                parts = urllib.parse.urlsplit(url)
                if parts.scheme in {"data", "blob", "about"}:
                    continue
                hosts.append(parts.hostname or "")
        else:
            hosts.append(line.split()[0].split(":")[0])
    assert hosts, "requests.txt lists no request at all — the page fetches at least its own modules"
    assert set(hosts) == {"127.0.0.1"}, (
        f"criterion 15: requests.txt lists only 127.0.0.1: {set(hosts)}"
    )
    after = sorted(p.as_posix() for p in artifacts.rglob("*")) if artifacts.exists() else []
    assert after == before, "criterion 15: the script writes nothing under artifacts/"


# =================================================================================================
# T-0154 group A: the header's Share label and banner (A4), the Context switches (A5), the What I
# see caption (A6). docs/tasks/T-0154.md, rulings O-A.
#
# One more session on the module's one Chrome (reset like the others). Three page loads on one
# origin, all against a live ``HudStub`` socket (frames are only encoded while the socket is open,
# so ``#capture-size`` — and with it the caption — only gets a size on an open socket):
#
#   1. no ``?source``: an init script replaces ``MediaDevices.prototype.getDisplayMedia`` and
#      ``getUserMedia`` with a 640x400 canvas ``captureStream`` (a *real* MediaStream handed to the
#      page's real providers, so ``capture.synthetic`` is false exactly as for a human's Share).
#      Share is clicked; measured idle, sharing (both themes) and after Mute.
#   2. ``?source=camera``: the same, through ``getUserMedia`` (O-A: "Sharing camera").
#   3. ``?source=canvas``: the page's own synthetic capture starts at load; Share is then clicked
#      once more (it restarts the synthetic capture). A MutationObserver installed by the same init
#      script before any page script runs records every distinct rendered Share label
#      (``innerText | aria-label | title``) from the first DOM node on.
#
# Readings pinned here (the task leaves them open; a reviewer may disagree):
#   - "the green treatment" = ``#share-button``'s computed ``color`` is the theme's green within
#     A2's colour tolerance (CIEDE2000 <= 2), and its ``background-color`` differs from the idle
#     button's. The green is not read from the page (the page names it ``--ok``, the mockup
#     ``--green``; a test keyed on either name dictates the token's name): dark is the mockup's
#     ``--green`` read from the committed tokens.css, light is T-0128's moved green (A2's named
#     deviation, ``T0128_DEVIATIONS`` of the parity module). Idle, synthetic and after-Mute the
#     button's text is not within that tolerance of the green.
#   - "reads as" = the button's ``innerText``, whitespace-normalised, equals the phrase exactly.
#   - "no second row under the header": with the synthetic banner up, ``<header>``'s height and
#     ``<main>``'s top equal (+/- 1 px) the idle page's at 1440x900, and the banner's box lies
#     inside the header's vertical extent.
#   - the caption hook is ``main > .pane.thumb [data-capture-caption]`` (the previous test-author's
#     pin, shared with test_hud_chrome.py's contrast row); "a source kind" is one of the words
#     screen / tab / window / display for a screen capture, camera for the camera, and
#     synthetic / canvas for the synthetic capture — never only "screen", although its frames carry
#     ``source: screen`` (ruling O-A2: the caption is honest about what the model sees).
#   - "computed appearance" of a switch = the input's own computed style together with its
#     ``::before``/``::after`` (the signature below) AND a clipped screenshot of the input's box;
#     both must differ for each pair. Measured under emulated reduced motion so a knob transition
#     cannot be caught mid-flight.
# =================================================================================================

#: The stub stream's size, chosen here: #capture-size must read exactly this while it runs.
T0154_STREAM_W, T0154_STREAM_H = 640, 400
T0154_STREAM_SIZE = f"{T0154_STREAM_W}x{T0154_STREAM_H}"
#: The banner's text, verbatim (T-0008 criterion 7).
SYNTHETIC_BANNER = "SYNTHETIC CAPTURE"
#: The source-kind words a caption may use, per capture (reading above).
CAPTION_KINDS = {
    "screen": re.compile(r"\b(screen|tab|window|display)\b", re.IGNORECASE),
    "camera": re.compile(r"\bcamera\b", re.IGNORECASE),
    # O-A2: a synthetic capture's caption says "synthetic" (or "canvas"), never only "screen".
    "synthetic": re.compile(r"\b(synthetic|canvas)\b", re.IGNORECASE),
}
#: O-A: the label for each real capture kind.
SHARING_LABEL = {"screen": "Sharing screen", "camera": "Sharing camera"}

T0154_INIT_JS = """
(() => {
  const rec = {labels: [], calls: []};
  Object.defineProperty(window, '__t0154', {value: rec, configurable: true});
  const fake = (kind) => {
    rec.calls.push(kind);
    const c = document.createElement('canvas');
    c.width = __W__; c.height = __H__;
    const g = c.getContext('2d');
    let n = 0;
    const paint = () => {
      g.fillStyle = 'rgb(' + ((n * 37) % 256) + ',90,140)';
      g.fillRect(0, 0, __W__, __H__);
      n += 1;
    };
    paint();
    setInterval(paint, 100);
    return Promise.resolve(c.captureStream(5));
  };
  if (window.MediaDevices) {
    MediaDevices.prototype.getDisplayMedia = function () { return fake('display'); };
    MediaDevices.prototype.getUserMedia = function () { return fake('camera'); };
  }
  const note = () => {
    const b = document.getElementById('share-button');
    if (!b) return;
    const entry = [(b.innerText || '').replace(/\\s+/g, ' ').trim(),
                   (b.getAttribute('aria-label') || '').trim(),
                   (b.getAttribute('title') || '').trim()].join(' | ');
    if (rec.labels[rec.labels.length - 1] !== entry) rec.labels.push(entry);
  };
  new MutationObserver(note).observe(document, {subtree: true, childList: true,
                                                characterData: true, attributes: true});
})();
""".replace("__W__", str(T0154_STREAM_W)).replace("__H__", str(T0154_STREAM_H))

T0154_SNAP_JS = """
(() => {
  window.scrollTo(0, 0);
  const b = document.getElementById('share-button');
  const s = getComputedStyle(b);
  const box = (el) => { const r = el.getBoundingClientRect(); return {top: r.top, bottom: r.bottom, h: r.height, w: r.width}; };
  const banner = document.getElementById('capture-banner');
  const cap = document.querySelector('main > .pane.thumb [data-capture-caption]');
  const size = document.getElementById('capture-size');
  const prev = document.getElementById('preview');
  const mute = document.getElementById('mute-button');
  return {
    label: (b.innerText || '').replace(/\\s+/g, ' ').trim(),
    color: s.color, background: s.backgroundColor,
    header: box(document.querySelector('header')),
    main_top: document.querySelector('main').getBoundingClientRect().top,
    banner: {hidden: banner.hidden || !!banner.closest('[hidden]'), display: getComputedStyle(banner).display,
             text: (banner.textContent || '').trim(), in_header: !!banner.closest('header'), ...box(banner)},
    caption: cap ? {present: true, text: (cap.textContent || '').replace(/\\s+/g, ' ').trim(), ...box(cap)}
                 : {present: false},
    size: size ? (size.textContent || '').trim() : null,
    mute_enabled: !mute.disabled,
    mute: {text: (mute.innerText || mute.textContent || '').replace(/\\s+/g, ' ').trim(),
           keys: mute.getAttribute('aria-keyshortcuts'), title: mute.getAttribute('title')},
    preview: {hidden: prev.hidden, fit: getComputedStyle(prev).objectFit, ...box(prev)},
    theme: document.documentElement.getAttribute('data-theme'),
  };
})()
"""

#: The computed signature of one switch: the input and its two pseudo-elements. ``cursor`` is left
#: out on purpose (a disabled cursor is not an appearance a human sees without hovering).
SWITCH_SIG_JS = """
const SIG_PROPS = ['appearance', 'width', 'height', 'background-color', 'background-image',
  'border-top-color', 'border-right-color', 'border-top-width', 'border-top-style', 'box-shadow',
  'opacity', 'filter', 'transform', 'translate', 'left', 'color', 'outline-color',
  'content', 'display'];
const sigOf = (el) => [null, '::before', '::after'].map((p) => {
  const s = getComputedStyle(el, p);
  return (p || 'input') + '{' + SIG_PROPS.map((k) => k + ':' + s.getPropertyValue(k)).join(';') + '}';
}).join(' ');
"""

SWITCH_JS = (
    "(() => {"
    + SWITCH_SIG_JS
    + """
  // T-0162 B-9: the switches live in the Settings popover; open it so they are laid out.
  { const g = document.querySelector('header button[aria-label="Settings"]'); const p = document.querySelector('[data-settings-popover] #assist-panel'); if (g && !(p && p.checkVisibility())) g.click(); }
  const boxes = [...document.querySelectorAll('[data-settings-popover] #assist-panel input[type=checkbox]')];
  const rect = (el) => { const r = el.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; };
  const enabled = boxes.find((b) => !b.disabled);
  const disabled = boxes.find((b) => b.disabled);
  return {
    count: boxes.length,
    boxes: boxes.map((b) => ({kind: (b.closest('[data-assist-kind]') || {}).getAttribute
                               ? b.closest('[data-assist-kind]').getAttribute('data-assist-kind') : null,
                             disabled: b.disabled, rect: rect(b)})),
    enabled: enabled ? {rect: rect(enabled), checked: enabled.checked, sig: sigOf(enabled)} : null,
    disabled: disabled ? {rect: rect(disabled), checked: disabled.checked, sig: sigOf(disabled)} : null,
  };
})()"""
)

SET_ENABLED_SWITCH_JS = """
((on) => {
  // T-0162 B-9: the switches live in the Settings popover; open it so they are laid out.
  { const g = document.querySelector('header button[aria-label="Settings"]'); const p = document.querySelector('[data-settings-popover] #assist-panel'); if (g && !(p && p.checkVisibility())) g.click(); }
  const b = [...document.querySelectorAll('[data-settings-popover] #assist-panel input[type=checkbox]')].find((x) => !x.disabled);
  if (b) b.checked = on;
  return !!b;
})(__ON__)
"""


def _clip(chrome: Chrome, rect: list[float]) -> str:
    """A screenshot of exactly the element's box (no margin: a neighbour's glyphs must not count)."""
    x, y, w, h = rect
    shot = chrome._cmd(
        "Page.captureScreenshot",
        format="png",
        clip={"x": x, "y": y, "width": max(w, 1), "height": max(h, 1), "scale": 1},
    )
    return str(shot.get("data") or "")


@dataclass
class Header154:
    error: str = ""
    seconds: float = 0.0
    #: kind ("screen" | "camera") -> phase -> measurement
    real: dict[str, Any] = field(default_factory=dict)
    synthetic: dict[str, Any] = field(default_factory=dict)
    switches: dict[str, Any] = field(default_factory=dict)
    exceptions: list[str] = field(default_factory=list)


def _snap(chrome: Chrome) -> dict[str, Any]:
    return dict(chrome.evaluate(T0154_SNAP_JS) or {})


def _mounted_this_navigation(chrome: Chrome, timeout_s: float = 25.0) -> bool:
    """At least ``ALL_PANES`` panes **and** this navigation's ``panels mounted:`` line (which
    hud/src/main.js logs after it has wired the header and settled ``mountPanels``). The pane
    count alone is satisfied by the static HTML before main.js runs (T-0154 B1: ALL_PANES 6)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and _panels_line(chrome.console[chrome._mark :]) is None:
        chrome.drain(0.1)  # reads CDP events, so console lines land
    return _panels_line(chrome.console[chrome._mark :]) is not None and chrome.wait_panes(ALL_PANES)


_SIZE_READY_JS = (
    "!document.getElementById('mute-button').disabled && "
    "/^\\d+x\\d+$/.test((document.getElementById('capture-size').textContent || '').trim())"
)


def _run_t0154(result: Header154, browser: _Recorder | str) -> None:
    stub = HudStub([]).start()
    server = HudStaticServer(HUD, 0)
    ws = urllib.parse.quote(stub.url, safe="")
    base = f"http://127.0.0.1:{server.port}/"
    try:
        with _session(browser) as chrome:
            chrome._cmd("Runtime.enable")
            chrome.media(_scheme("dark"))
            chrome.add_init_script(T0154_INIT_JS)
            chrome.viewport(*WIDE["1440x900"])

            # 1-2: a real screen capture and a real camera capture, through the page's providers
            for kind, query in (("screen", ""), ("camera", "source=camera&")):
                r: dict[str, Any] = {}
                # T-0154 B1 fix (G2, no row weakened): ALL_PANES fell from 8 to 6, and the
                # pre-B index.html already declares 6 static panes, so wait_panes(6) is true at
                # parse, before hud/src/main.js runs; and the stub's earlier "astra.hello" (from
                # the screen pass) satisfied a plain membership wait on the camera pass. Share
                # was then clicked before main.js had wired it. Wait for *this* navigation's
                # hello (the count grows) and for this navigation's "panels mounted:" line.
                hellos = stub.received_types().count("astra.hello")
                chrome.navigate(f"{base}?{query}ws={ws}")
                r["mounted"] = _mounted_this_navigation(chrome)
                chrome.evaluate(choose_theme_js("dark"))
                stub.wait_for(lambda s, n=hellos: s.received_types().count("astra.hello") > n, 10)
                r["idle"] = _snap(chrome)
                frames_before = len(stub.frames())
                chrome.evaluate("document.getElementById('share-button').click()")
                r["started"] = chrome.wait_until(_SIZE_READY_JS, 15)
                r["frames"] = [f.source.value for f in stub.frames()[frames_before:]]
                r["calls"] = list(chrome.evaluate("window.__t0154.calls") or [])
                for theme in ("dark", "light"):
                    r[f"chosen_{theme}"] = chrome.evaluate(choose_theme_js(theme)) or {}
                    _await_frame(chrome)
                    time.sleep(0.3)  # a colour transition on the button settles
                    r[theme] = _snap(chrome)
                    r[f"contrast_{theme}"] = (
                        chrome.evaluate(measure_js([("sharing label", "#share-button", None)]))
                        or {}
                    )
                chrome.evaluate(choose_theme_js("dark"))
                chrome.evaluate("document.getElementById('mute-button').click()")
                _await_frame(chrome)
                time.sleep(0.3)
                r["after_mute"] = _snap(chrome)
                r["labels"] = list(chrome.evaluate("window.__t0154.labels") or [])
                result.real[kind] = r

            # 3: the synthetic capture (?source=canvas starts it at load)
            syn: dict[str, Any] = {}
            chrome.navigate(f"{base}?source=canvas&ws={ws}")
            syn["mounted"] = _mounted_this_navigation(chrome)
            chrome.evaluate(choose_theme_js("dark"))
            syn["started"] = chrome.wait_until(_SIZE_READY_JS, 15)
            chrome.drain(1.5)
            time.sleep(0.3)
            syn["dark"] = _snap(chrome)
            chrome.evaluate(choose_theme_js("light"))
            _await_frame(chrome)
            time.sleep(0.3)
            syn["light"] = _snap(chrome)
            chrome.evaluate(choose_theme_js("dark"))
            chrome.evaluate("document.getElementById('share-button').click()")
            syn["restarted"] = chrome.wait_until(_SIZE_READY_JS, 15)
            chrome.drain(1.5)
            syn["after_click"] = _snap(chrome)
            syn["labels"] = list(chrome.evaluate("window.__t0154.labels") or [])
            syn["calls"] = list(chrome.evaluate("window.__t0154.calls") or [])
            result.synthetic = syn

            # A5: the switches, on the synthetic page (page_text is available over the synthetic
            # canvas, the other three kinds are disabled), under reduced motion, both themes
            chrome.media(
                [
                    {"name": "prefers-reduced-motion", "value": "reduce"},
                    {"name": "prefers-color-scheme", "value": "dark"},
                ]
            )
            for theme in ("dark", "light"):
                sw: dict[str, Any] = {"chosen": chrome.evaluate(choose_theme_js(theme)) or {}}
                chrome.evaluate(SET_ENABLED_SWITCH_JS.replace("__ON__", "false"))
                _await_frame(chrome)
                off = dict(chrome.evaluate(SWITCH_JS) or {})
                chrome.evaluate(SET_ENABLED_SWITCH_JS.replace("__ON__", "true"))
                _await_frame(chrome)
                on = dict(chrome.evaluate(SWITCH_JS) or {})
                sw["off"], sw["on"] = off, on
                if off.get("enabled") and on.get("enabled") and off.get("disabled"):
                    sw["png_on"] = _clip(chrome, on["enabled"]["rect"])
                    chrome.evaluate(SET_ENABLED_SWITCH_JS.replace("__ON__", "false"))
                    _await_frame(chrome)
                    sw["png_off"] = _clip(chrome, off["enabled"]["rect"])
                    sw["png_disabled"] = _clip(chrome, off["disabled"]["rect"])
                chrome.evaluate(SET_ENABLED_SWITCH_JS.replace("__ON__", "false"))
                result.switches[theme] = sw
            chrome.evaluate(choose_theme_js("system"))
            chrome.media(_scheme("dark"))
            result.exceptions = list(chrome.exceptions)
    finally:
        server.close()
        stub.stop()


@pytest.fixture(scope="module")
def header154(browser: Any) -> Any:
    """The T-0154 A4-A6 session. Never raises: failures land on ``error``."""
    if CHROME is None:  # pragma: no cover - requires_chrome is the real guard
        yield Header154(error="no chrome")
        return
    result = Header154()
    started = time.monotonic()
    try:
        _run_t0154(result, browser)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    result.seconds = time.monotonic() - started
    yield result


def _parity() -> Any:
    """tests/integration/test_hud_mockup_parity.py, imported at call time: it imports this module
    at its top, so a module-level import here is circular. Its CIEDE2000, its A2 tolerance and its
    named T-0128 deviations are reused, never copied."""
    from tests.integration import test_hud_mockup_parity

    return test_hud_mockup_parity


#: A2's colour tolerance (CIEDE2000), the parity module's ``COLOUR_TOL_DE``.
COLOUR_TOL_DE = 2.0


def _mockup_greens() -> dict[str, str]:
    """The mockup's ``--green`` per theme, read from the committed tokens.css (dark block first)."""
    tokens = (REPO / "tests" / "fixtures" / "hud" / "mockup_20261021" / "tokens.css").read_text()
    found = re.findall(r"--green:\s*(#[0-9a-fA-F]{6})", tokens)
    assert len(found) == 2, f"premise: tokens.css declares --green in its two blocks: {found}"
    return {"dark": found[0], "light": found[1]}


def green_for(theme: str) -> str:
    """A2's target: the mockup token, or T-0128's moved token where A2 names the deviation."""
    return _parity().T0128_DEVIATIONS.get(("green", theme)) or _mockup_greens()[theme]


def _is_green(colour: str, theme: str) -> bool:
    p = _parity()
    r, g, b, a = parse_css_color(colour)
    return a > 0.5 and p.delta_e((r, g, b), p.hex_rgb(green_for(theme))) <= COLOUR_TOL_DE


def test_t0154_the_green_oracle_is_the_mockup_token_and_the_t0128_deviation():
    """Plain (no browser): the oracle the Share rows use, hand-checked against tokens.css lines 17
    and 41 and hud/styles.css's T-0128 comment (#149a5a -> #107a47 for contrast)."""
    assert _parity().COLOUR_TOL_DE == COLOUR_TOL_DE, "the tolerance is A2's, kept in one place"
    assert _mockup_greens() == {"dark": "#5ee09a", "light": "#149a5a"}
    assert green_for("dark") == "#5ee09a" and green_for("light") == "#107a47"
    assert _is_green("rgb(94, 224, 154)", "dark") and _is_green("rgb(16, 122, 71)", "light")
    assert not _is_green("rgb(180, 186, 200)", "dark"), "the idle button's text is not green"
    assert not _is_green("rgba(94, 224, 154, 0)", "dark"), "a transparent colour is not green"


def _real_premise(h: Header154, kind: str) -> dict[str, Any]:
    assert not h.error, f"the T-0154 headless session failed before any assertion ran: {h.error}"
    r = h.real.get(kind) or {}
    assert r.get("mounted"), f"premise ({kind}): the page did not mount its {ALL_PANES} panes"
    want = "display" if kind == "screen" else "camera"
    assert r.get("calls") == [want], (
        f"premise ({kind}): Share must reach the page's own {want} provider exactly once; the stub "
        f"saw {r.get('calls')!r}"
    )
    assert r.get("started"), (
        f"premise ({kind}): the stubbed real capture never ran (Mute enabled and #capture-size a "
        f"WxH); idle {r.get('idle')!r}"
    )
    assert r["dark"]["size"] == T0154_STREAM_SIZE, (
        f"premise ({kind}): #capture-size reads {r['dark']['size']!r}; the stub stream is "
        f"{T0154_STREAM_SIZE}"
    )
    assert not r["dark"]["banner"]["text"] and r["dark"]["banner"]["hidden"], (
        f"premise ({kind}): a real stream is not synthetic, so no banner: {r['dark']['banner']!r}"
    )
    return r


@requires_chrome
@pytest.mark.parametrize("kind", ["screen", "camera"])
def test_t0154_share_reads_sharing_in_green_while_a_real_capture_runs(header154, kind):
    """A4 (O-A): ``#share-button`` reads "Sharing screen" ("Sharing camera" for ``?source=camera``)
    with the green treatment while a real capture runs, in both themes; it does not before Share is
    pressed or after Mute. Iff: idle and after-Mute it neither reads "Sharing" nor is green."""
    r = _real_premise(header154, kind)
    problems: list[str] = []
    idle, after = r["idle"], r["after_mute"]
    for phase, snap in (("idle", idle), ("after Mute", after)):
        if re.search(r"sharing", snap["label"], re.IGNORECASE):
            problems.append(f"{phase}: Share reads {snap['label']!r} with no capture running")
        if _is_green(snap["color"], snap["theme"] or "dark"):
            problems.append(f"{phase}: Share is green ({snap['color']}) with no capture running")
    for theme in ("dark", "light"):
        snap = r[theme]
        if snap["theme"] != theme:
            problems.append(f"premise: data-theme reads {snap['theme']!r}, not {theme!r}")
            continue
        if snap["label"] != SHARING_LABEL[kind]:
            problems.append(
                f"{theme}: Share reads {snap['label']!r} during a real {kind} capture, not "
                f"{SHARING_LABEL[kind]!r}"
            )
        if not _is_green(snap["color"], theme):
            problems.append(
                f"{theme}: Share's text is {snap['color']}, not the theme's green "
                f"{green_for(theme)} (dE00 <= {COLOUR_TOL_DE})"
            )
        if snap["background"] == idle["background"] and theme == "dark":
            problems.append(
                f"{theme}: Share's fill {snap['background']} is the idle button's: no sharing "
                "treatment behind the text"
            )
    assert problems == [], f"T-0154 A4, real {kind} capture:\n  " + "\n  ".join(problems)


@requires_chrome
def test_t0154_a_synthetic_capture_never_shows_sharing_not_even_for_one_mutation(header154):
    """A4, adversarial: during a synthetic capture (``?source=canvas``, started at load, then Share
    pressed once more) no rendered Share label — every one a MutationObserver saw from the first
    DOM node on — says "Sharing", and the button is never green.

    Anti-vacuity: the same observer, on the real-capture page, *did* record "Sharing screen". A
    page that never says "Sharing" anywhere passes the negative half trivially; it fails here."""
    h = header154
    real = _real_premise(h, "screen")
    assert any(e.split(" | ")[0] == SHARING_LABEL["screen"] for e in real["labels"]), (
        "anti-vacuity: the MutationObserver must see the label change on a real capture "
        f'("Sharing screen"); it recorded {real["labels"]!r}'
    )
    syn = h.synthetic
    assert syn.get("started") and syn.get("restarted"), f"premise: the synthetic capture ran: {syn}"
    assert syn.get("calls") == [], (
        f"premise: a synthetic capture never calls getDisplayMedia/getUserMedia: {syn.get('calls')}"
    )
    assert syn["labels"], "premise: the observer recorded the Share label at least once"
    sharing = [e for e in syn["labels"] if re.search(r"sharing", e, re.IGNORECASE)]
    assert sharing == [], (
        "A4: Share said 'Sharing' during a SYNTHETIC capture (the page cannot vouch for the stream; "
        f"T-0008 criterion 7). Every label the observer saw: {syn['labels']!r}"
    )
    for phase in ("dark", "light", "after_click"):
        snap = syn[phase]
        assert not _is_green(snap["color"], snap["theme"] or "dark"), (
            f"A4 ({phase}): Share is green ({snap['color']}) during a synthetic capture"
        )


@requires_chrome
def test_t0154_the_synthetic_banner_sits_in_the_header_row_and_adds_no_second_row(header154):
    """A4: during a synthetic capture ``#capture-banner`` reads ``SYNTHETIC CAPTURE`` verbatim,
    is laid out, sits inside ``<header>`` (DOM and geometry), and the header is no taller and
    ``<main>`` starts no lower than on the idle page at the same viewport."""
    h = header154
    assert not h.error, h.error
    idle = (h.real.get("screen") or {}).get("idle") or {}
    assert idle, "premise: the idle page was measured"
    problems: list[str] = []
    for phase in ("dark", "after_click"):
        snap = h.synthetic[phase]
        b = snap["banner"]
        if b["text"] != SYNTHETIC_BANNER or b["hidden"] or b["display"] == "none" or b["h"] <= 0:
            problems.append(f"{phase}: the banner is not up with {SYNTHETIC_BANNER!r}: {b}")
            continue
        if not b["in_header"]:
            problems.append(f"{phase}: #capture-banner is not inside <header> (a second row)")
        hd = snap["header"]
        if b["top"] < hd["top"] - 0.5 or b["bottom"] > hd["bottom"] + 0.5:
            problems.append(
                f"{phase}: the banner spans y {b['top']:.1f}..{b['bottom']:.1f}, outside the "
                f"header's {hd['top']:.1f}..{hd['bottom']:.1f}"
            )
        if abs(hd["h"] - idle["header"]["h"]) > 1.0:
            problems.append(
                f"{phase}: the header is {hd['h']:.1f} px tall with the banner, "
                f"{idle['header']['h']:.1f} px without"
            )
        if abs(snap["main_top"] - idle["main_top"]) > 1.0:
            problems.append(
                f"{phase}: <main> starts at y {snap['main_top']:.1f} with the banner, "
                f"{idle['main_top']:.1f} without: a second row under the header"
            )
    assert problems == [], "T-0154 A4, the banner:\n  " + "\n  ".join(problems)


@requires_chrome
@pytest.mark.parametrize("kind", ["screen", "camera", "synthetic"])
def test_t0154_the_caption_shows_the_capture_size_and_a_source_kind_iff_a_capture_runs(
    header154, kind
):
    """A6 (O-A), **amended by T-0177 C4**, which supersedes T-0154 A6 by name: the caption now
    describes the image the session sent (``<source name> · <W>x<H> · turn <id> step <k>``), and
    "with no frame, the caption is empty", while "``#capture-size``, the frame counters and the
    Share button keep reporting the capture". This session's stub sends no ``astra.vision.frame``,
    so in every phase (idle, a real or synthetic capture running in both themes, after Mute) the
    caption is present and empty, and while a capture runs ``#capture-size`` still reads its WxH.
    The caption with a frame (its words, the size of the image shown) is pinned by
    tests/integration/test_t0177_hud_vision.py."""
    h = header154
    assert not h.error, h.error
    problems: list[str] = []
    if kind == "synthetic":
        running = [("dark", h.synthetic["dark"]), ("after Share", h.synthetic["after_click"])]
        empty: list[tuple[str, dict[str, Any]]] = []
    else:
        r = _real_premise(h, kind)
        running = [("dark", r["dark"]), ("light", r["light"])]
        empty = [("idle", r["idle"]), ("after Mute", r["after_mute"])]
    for phase, snap in empty:
        cap = snap["caption"]
        if not cap["present"]:
            problems.append(f"{phase}: no main > .pane.thumb [data-capture-caption]")
        elif cap["text"] != "":
            problems.append(f"{phase}: the caption reads {cap['text']!r} with no capture running")
    for phase, snap in running:
        cap, size = snap["caption"], snap["size"]
        if not re.fullmatch(r"\d+x\d+", size or ""):
            problems.append(
                f"{phase}: #capture-size is {size!r}, not a WxH: C4 keeps it reporting the capture"
            )
        if not cap["present"]:
            problems.append(f"{phase}: no main > .pane.thumb [data-capture-caption]")
            continue
        if cap["text"] != "":
            problems.append(
                f"{phase}: the caption reads {cap['text']!r} with no astra.vision.frame (C4: with no "
                "frame the caption is empty; it no longer describes the local capture)"
            )
    assert problems == [], f"T-0177 C4, the caption ({kind}):\n  " + "\n  ".join(problems)


@requires_chrome
def test_t0154_the_frame_stays_letterboxed_while_a_capture_runs(header154):
    """A6 (kept): ``#preview`` computes ``object-fit: contain``: the model's view is never cropped.
    A fence: true before T-0154, and what a "fill the pane" restyle breaks.

    **Amended by T-0177 C3** (which supersedes T-0008's local thumbnail): ``#preview`` no longer
    shows the local capture, so this session (no ``astra.vision.frame``) has no frame on screen and
    the premise "the frame is shown" is dropped here; the object-fit fence stays. The letterbox of a
    shown session frame is pinned by tests/integration/test_t0177_hud_vision.py
    (``test_c4_the_caption_of_a_shared_frame_and_the_live_badge``, ``preview_fit``)."""
    h = header154
    assert not h.error, h.error
    snaps = [("synthetic", h.synthetic.get("dark") or {})]
    if (h.real.get("screen") or {}).get("started"):
        snaps.append(("screen", h.real["screen"]["dark"]))
    for name, snap in snaps:
        p = snap["preview"]
        assert p["fit"] == "contain", f"A6 ({name}): #preview object-fit is {p['fit']!r}"


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_t0154_sharing_screen_clears_its_contrast_floor(header154, theme):
    """A6/A7 (T-0128 criterion 10, list extended): "Sharing screen" on its green treatment clears
    4.5:1 (3.0:1 large) in both themes. Measured here because test_hud_chrome.py's capture is
    synthetic and never shows it."""
    r = _real_premise(header154, "screen")
    assert r[theme]["label"] == SHARING_LABEL["screen"], (
        f"premise: the button reads {SHARING_LABEL['screen']!r} while measured: {r[theme]['label']!r}"
    )
    missing, vacuous, failures, _excluded = _contrast_report(r[f"contrast_{theme}"])
    assert missing == [] and vacuous == [], f"{missing} {vacuous}"
    assert failures == [], f"T-0154 A7, data-theme={theme}:\n  " + "\n  ".join(failures)


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_t0154_the_context_checkboxes_render_as_switches_with_three_distinct_states(
    header154, theme
):
    """A5 (O-A): every assist input is still ``input[type=checkbox]`` and its own box is wider
    than tall (a switch); checked, unchecked and disabled differ pairwise in computed appearance
    (input + pseudo-elements) and in pixels, in both themes."""
    h = header154
    assert not h.error, h.error
    sw = h.switches.get(theme) or {}
    assert (sw.get("chosen") or {}).get("theme") == theme, f"premise: data-theme {theme}: {sw}"
    off, on = sw["off"], sw["on"]
    assert off.get("count") == 4, f"premise: four assist checkboxes: {off}"
    assert off.get("enabled") and off.get("disabled"), (
        f"premise: over the synthetic canvas one kind is available and the others are disabled: "
        f"{off.get('boxes')}"
    )
    problems: list[str] = []
    for b in off["boxes"]:
        _x, _y, w, ht = b["rect"]
        if not w > ht:
            problems.append(
                f"{b['kind']}: the input's box is {w:.1f}x{ht:.1f}, not wider than tall"
            )
    states = {
        "unchecked": (off["enabled"]["sig"], sw.get("png_off")),
        "checked": (on["enabled"]["sig"], sw.get("png_on")),
        "disabled": (off["disabled"]["sig"], sw.get("png_disabled")),
    }
    names = list(states)
    for i, a in enumerate(names):
        for b2 in names[i + 1 :]:
            (sa, pa), (sb, pb) = states[a], states[b2]
            if sa == sb:
                problems.append(f"{a} and {b2} compute the same appearance: {sa}")
            if pa is None or pb is None or pa == pb:
                problems.append(f"{a} and {b2} render the same pixels")
    assert problems == [], f"T-0154 A5, the switches (data-theme={theme}):\n  " + "\n  ".join(
        problems
    )
