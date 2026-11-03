"""T-0154 group B — the conversation thread measured against the approved mockup (P2; B1, B2, B4,
B7, B8, B9, B12), in the same headless Chrome as the mockup.

docs/tasks/T-0154.md B12 and *Tests to write*; ruling **O-P**: group B's parity rows live in this
module, which imports the measuring helpers of ``tests/integration/test_hud_mockup_parity.py``
(``Row``, ``evaluate_row``, ``measure_js``, the CIEDE2000 oracle, the probe pixel reader, the
module's one-Chrome fixture) and never copies or edits them. Every row that needs group B carries
``pending(task="T-0154")``; the mockup-side premise rows and the pure rows that already hold do not.

**What is compared with what.** As in P1: the committed mockup (``tests/fixtures/hud/
mockup_20261021/hud.html``, theme by ``?theme=``) against the HUD served by ``HudStaticServer``
with ``HudStub`` replaying ``session_20261020_t3_t11.jsonl`` to the end of t9, the t9 card gated as
``scripts/demo/hud_screens.py`` gates it, then one SPEAKING. 1440x900, DPR 1, reduced motion. The
reference value of every row is measured from the mockup in the same run.

**The P2 role table** (B12), mockup selector -> HUD selector:

====================  ============================  ===============================================
conversation pane     ``.panel.convo``              ``main > :has([data-thread])`` (B1: the one
                                                    direct child of <main> holding the thread)
thread                ``.thread``                   ``[data-thread]``
user bubble           ``.thread > .u``              ``[data-thread] [data-transcript]``
assistant row         ``.thread > .a``              ``[data-thread] [data-assistant]``
avatar                ``.a > .av``                  ``[data-assistant] [data-avatar]``
prose                 ``.said``                     ``[data-thread] [data-utterance-id]``
action chip           ``.action``                   ``[data-thread] [data-tool-call-id]``
DRY RUN tag           ``.action .tag``              ``[data-tool-call-id] [data-tool-status="dry_run"]``
card wrap             ``.card-wrap``                ``[data-thread] [data-ui-surface]``
composer              ``.composer``                 ``[data-ask-form]``
composer submit       ``.composer > .mic``          ``[data-ask-submit]``
====================  ============================  ===============================================

The hooks ``[data-assistant]``, ``[data-avatar]`` and the bubble's ``[data-transcript]`` are the
ones ``tests/hud/thread.test.mjs`` pins; every other hook is an existing contract.

Readings the task leaves open, stated so a reviewer can disagree with them:

- "assistant grid 30 px + 14 gap" is measured as geometry, not as declarations: the avatar's left
  edge sits at the row's left edge, it is 30 px wide, and the first utterance starts 14 px after the
  avatar's right edge. A flex row and a grid row render the same.
- "composer margin 0 22 22" is measured as the composer's inset from the conversation pane's border
  box (left, right, bottom), which also covers where the implementer puts the margin.
- "user bubble max-width 70%" is the computed ``max-width`` as a percentage of the bubble's
  containing block (a px value is converted with the parent's content width).
- "an amber DRY RUN tag" is the action card's status chip while the status is ``dry_run`` (the
  card's own copy, T-0090; B7 keeps it): its text colour is ``--amber`` (a named T-0128 deviation
  in light) and its fill the mockup's ``--amber-soft``. Its radius is compared as the *effective*
  radius ``min(r, h/2)``, so 999px and a 99px pill read the same.
- "the thread scrolls" (B9) is read as ``[data-thread]`` being the scroll container; the composer
  sits outside it, at the bottom of the conversation pane.
- the probes (B12 "ΔE probes on the bubble, the conversation interior and the composer
  interior") sit in each element's own padding, so no glyph is sampled: the bubble 8 px in from its
  left edge at mid-height (the bubble's left padding is 16), the conversation 12 px in from the
  pane's left edge at the bubble's mid-height (the thread's left padding is 34, and the bubble is
  right-aligned), the composer 8 px in from its left edge at mid-height (left padding 18). Each
  element is scrolled into view first (``scrollIntoView``; the mockup's own document is 1095 px tall
  at 900 and its composer starts below the fold). The element under the probe must be inside the
  role's element, or the row says what it hit.

Regime: Google Chrome (``google-chrome --version``) headless, CDP, DPR 1; 1440x900 for parity,
1100x800 / 1440x900 / 1920x1080 for "the document never scrolls"; data-theme dark and light; reduced
motion on; n = 1, deterministic, no seed. No GPU.
"""

from __future__ import annotations

import io
import json
import re
import time
import tokenize
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.protocol import AstraTurnState
from local_astra.contracts.turn import TurnState
from local_astra.hud_static import HudStaticServer
from tests.fakes.hud_server import HudStub
from tests.integration import test_hud_mockup_parity as _parity
from tests.integration.test_hud_chrome import choose_theme_js
from tests.integration.test_hud_headless import A2UI_ROOT, CHROME, HUD, requires_chrome
from tests.integration.test_hud_mockup_parity import (
    COLOUR_TOL_DE,
    FONT_SIZE_TOL_PX,
    LENGTH_TOL_PX,
    MOCKUP,
    PIXEL_TOL_DE,
    T0128_DEVIATIONS,
    THEMES,
    VIEW,
    Row,
    _media,
    _pixel,
    _serve_dir,
    _settle,
    _token_rgb,
    delta_e,
    evaluate_row,
    hex_rgb,
    measure_js,
)
from tests.integration.test_hud_redesign import (
    CARD_TURN,
    _is_stale_card_surface,
    _last_index_of,
    _Recorder,
    card_surfaces,
    fixture_lines,
    gated_card,
)

pytestmark = pytest.mark.integration

#: The module's one headless Chrome: the parity module's own fixture, re-bound by assignment (an
#: unused import would be stripped by ruff --fix), never copied. pytest registers it by its name.
browser = _parity.browser

REPO = Path(__file__).resolve().parents[2]
HUD_SCREENS = REPO / "scripts" / "demo" / "hud_screens.py"

#: B9, verbatim: "within 48 px of its bottom".
STICK_PX = 48
#: B9: "The document never scrolls at 1100 px and wider."
NO_DOC_SCROLL = ((1100, 800), (1440, 900), (1920, 1080))

M_BODY = "body"
M_CONVO, H_CONVO = ".panel.convo", "main > :has([data-thread])"
M_THREAD, H_THREAD = ".thread", "[data-thread]"
M_BUBBLE, H_BUBBLE = ".thread > .u", "[data-thread] [data-transcript]"
M_ROW, H_ROW = ".thread > .a", "[data-thread] [data-assistant]"
M_AVATAR, H_AVATAR = ".a > .av", "[data-assistant] [data-avatar]"
M_PROSE, H_PROSE = ".said", "[data-thread] [data-utterance-id]"
M_CHIP, H_CHIP = ".action", "[data-thread] [data-tool-call-id]"
M_TAG, H_TAG = ".action .tag", '[data-thread] [data-tool-call-id] [data-tool-status="dry_run"]'
M_WRAP, H_WRAP = ".card-wrap", "[data-thread] [data-ui-surface]"
M_COMPOSER, H_COMPOSER = ".composer", "[data-ask-form]"
M_SUBMIT, H_SUBMIT = ".composer > .mic", "[data-ask-submit]"

L, F, C = LENGTH_TOL_PX, FONT_SIZE_TOL_PX, COLOUR_TOL_DE

#: String-valued properties compared exactly (``evaluate_row`` compares numbers and colours).
EXACT = {"border_style"}

P2: list[Row] = [
    Row("conversation pane: radius 22", M_CONVO, H_CONVO, "radius", L),
    Row("conversation pane: 1 px border", M_CONVO, H_CONVO, "border_w", L),
    Row(
        "conversation pane: line border colour",
        M_CONVO,
        H_CONVO,
        "color:border",
        C,
        backdrop=M_CONVO,
    ),
    Row("conversation pane: shadow", M_CONVO, H_CONVO, "shadow", L, backdrop=M_BODY),
    Row("conversation pane: fill", M_CONVO, H_CONVO, "color:background", C, "surface"),
    Row("thread: padding top 26", M_THREAD, H_THREAD, "pad_t", L),
    Row("thread: padding right 34", M_THREAD, H_THREAD, "pad_r", L),
    Row("thread: padding left 34", M_THREAD, H_THREAD, "pad_l", L),
    Row("user bubble: max-width 70%", M_BUBBLE, H_BUBBLE, "max_width_pct", 0.5),
    Row("user bubble: radius top-left 18", M_BUBBLE, H_BUBBLE, "radius_tl", L),
    Row("user bubble: radius top-right 18", M_BUBBLE, H_BUBBLE, "radius_tr", L),
    Row("user bubble: radius bottom-right 6", M_BUBBLE, H_BUBBLE, "radius_br", L),
    Row("user bubble: radius bottom-left 18", M_BUBBLE, H_BUBBLE, "radius_bl", L),
    Row("user bubble: padding top 11", M_BUBBLE, H_BUBBLE, "pad_t", L),
    Row("user bubble: padding right 16", M_BUBBLE, H_BUBBLE, "pad_r", L),
    Row("user bubble: padding bottom 11", M_BUBBLE, H_BUBBLE, "pad_b", L),
    Row("user bubble: padding left 16", M_BUBBLE, H_BUBBLE, "pad_l", L),
    Row("user bubble: 15 px", M_BUBBLE, H_BUBBLE, "font_size", F),
    Row("user bubble: --bubble fill", M_BUBBLE, H_BUBBLE, "color:background", C, backdrop=M_CONVO),
    Row("user bubble: right-aligned", M_BUBBLE, H_BUBBLE, "geom_right_inset", L),
    Row("assistant row: max-width 860", M_ROW, H_ROW, "max_width_px", L),
    Row("assistant row: avatar at the row's left edge", M_ROW, H_ROW, "geom_avatar_left", L),
    Row("assistant row: 14 px from the avatar to the text", M_ROW, H_ROW, "geom_avatar_gap", L),
    Row("avatar: 30 wide", M_AVATAR, H_AVATAR, "w", L),
    Row("avatar: 30 high", M_AVATAR, H_AVATAR, "h", L),
    Row("avatar: radius 10", M_AVATAR, H_AVATAR, "radius", L),
    Row("prose: 16.5 px", M_PROSE, H_PROSE, "font_size", F),
    Row("prose: line-height 1.65", M_PROSE, H_PROSE, "line_height", 0.5),
    Row("prose: colour", M_PROSE, H_PROSE, "color:text", C, backdrop=M_CONVO),
    Row("action chip: dashed border", M_CHIP, H_CHIP, "border_style", 0),
    Row("action chip: 1 px border", M_CHIP, H_CHIP, "border_w", L),
    Row("action chip: --line-strong border", M_CHIP, H_CHIP, "color:border", C, backdrop=M_CONVO),
    Row("action chip: radius 12", M_CHIP, H_CHIP, "radius", L),
    Row("DRY RUN tag: amber text", M_TAG, H_TAG, "color:text", C, "amber", backdrop=M_CONVO),
    Row("DRY RUN tag: amber-soft fill", M_TAG, H_TAG, "color:background", C, backdrop=M_CONVO),
    Row("DRY RUN tag: pill", M_TAG, H_TAG, "eff_radius", L),
    Row("card wrap: max-width 520", M_WRAP, H_WRAP, "max_width_px", L),
    Row("composer: radius 16", M_COMPOSER, H_COMPOSER, "radius", L),
    Row("composer: 1 px border", M_COMPOSER, H_COMPOSER, "border_w", L),
    Row(
        "composer: line border colour",
        M_COMPOSER,
        H_COMPOSER,
        "color:border",
        C,
        backdrop=M_COMPOSER,
    ),
    Row(
        "composer: --surface-2 fill",
        M_COMPOSER,
        H_COMPOSER,
        "color:background",
        C,
        backdrop=M_CONVO,
    ),
    Row("composer: 22 from the pane's left", M_COMPOSER, H_COMPOSER, "geom_inset_left", L),
    Row("composer: 22 from the pane's right", M_COMPOSER, H_COMPOSER, "geom_inset_right", L),
    Row("composer: 22 from the pane's bottom", M_COMPOSER, H_COMPOSER, "geom_inset_bottom", L),
    Row("composer: 0 below the thread", M_COMPOSER, H_COMPOSER, "geom_below_thread", L),
    Row("composer submit: 38 wide", M_SUBMIT, H_SUBMIT, "w", L),
    Row("composer submit: 38 high", M_SUBMIT, H_SUBMIT, "h", L),
    Row("composer submit: radius 12", M_SUBMIT, H_SUBMIT, "radius", L),
    Row(
        "composer submit: accent fill",
        M_SUBMIT,
        H_SUBMIT,
        "color:background",
        C,
        "accent",
        backdrop=M_COMPOSER,
    ),
]

MOCK_SELECTORS = sorted({r.mock for r in P2} | {M_BODY})
HUD_SELECTORS = sorted({r.hud for r in P2} | {"body"})

#: Extra per-element measurements this module needs and ``MEASURE_JS`` does not take (it is
#: imported, never edited): the four radii, max-width, line-height, border style, the effective
#: pill radius, and geometry relative to the role's own container. Same element order as
#: ``measure_js`` (``querySelectorAll``), so the two lists merge by index.
EXTRA_JS = """
(() => {
  const SELS = __SELS__, ROLES = __ROLES__;
  const px = (v) => { const n = parseFloat(v); return Number.isFinite(n) ? n : null; };
  const contentW = (el) => { if (!el) return null; const s = getComputedStyle(el), r = el.getBoundingClientRect();
    return r.width - px(s.paddingLeft) - px(s.paddingRight) - px(s.borderLeftWidth) - px(s.borderRightWidth); };
  const extra = (el) => {
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    const mw = s.maxWidth;
    let pct = null, mwpx = null;
    if (mw.endsWith('%')) pct = px(mw);
    else if (mw.endsWith('px')) { mwpx = px(mw); const cw = contentW(el.parentElement); pct = cw ? 100 * mwpx / cw : null; }
    const out = {radius_tl: px(s.borderTopLeftRadius), radius_tr: px(s.borderTopRightRadius),
                 radius_br: px(s.borderBottomRightRadius), radius_bl: px(s.borderBottomLeftRadius),
                 max_width_px: mwpx, max_width_pct: pct, line_height: px(s.lineHeight),
                 border_style: s.borderTopStyle,
                 eff_radius: Math.min(px(s.borderTopLeftRadius) || 0, r.height / 2)};
    const pane = el.closest(ROLES.pane), thread = document.querySelector(ROLES.thread);
    if (pane) { const p = pane.getBoundingClientRect();
      out.geom_inset_left = r.left - p.left; out.geom_inset_right = p.right - r.right; out.geom_inset_bottom = p.bottom - r.bottom; }
    if (thread && !thread.contains(el) && pane && pane.contains(thread)) out.geom_below_thread = r.top - thread.getBoundingClientRect().bottom;
    if (thread && thread.contains(el)) {
      const tr = thread.getBoundingClientRect(), ts = getComputedStyle(thread);
      out.geom_right_inset = (tr.right - px(ts.borderRightWidth) - px(ts.paddingRight)) - r.right;
    }
    const av = el.querySelector(ROLES.avatar), said = el.querySelector(ROLES.prose);
    if (av) { const a = av.getBoundingClientRect(); out.geom_avatar_left = a.left - r.left;
      if (said) out.geom_avatar_gap = said.getBoundingClientRect().left - a.right; }
    return out;
  };
  const res = {};
  for (const sel of SELS) res[sel] = [...document.querySelectorAll(sel)].map(extra);
  return res;
})()
"""

MOCK_ROLES = {"pane": M_CONVO, "thread": M_THREAD, "avatar": ".av", "prose": M_PROSE}
HUD_ROLES = {
    "pane": H_CONVO,
    "thread": H_THREAD,
    "avatar": "[data-avatar]",
    "prose": "[data-utterance-id]",
}


def extra_js(selectors: list[str], roles: dict[str, str]) -> str:
    return EXTRA_JS.replace("__SELS__", json.dumps(selectors)).replace(
        "__ROLES__", json.dumps(roles)
    )


#: One probe point in an element's own padding, after scrolling the element into view. ``pick``
#: chooses which match: the mockup's first bubble, the HUD's last (the newest turn's).
PROBE_JS = """
(() => {
  const P = __P__;
  const all = [...document.querySelectorAll(P.sel)];
  const el = P.pick === 'last' ? all[all.length - 1] : all[0];
  if (!el) return {missing: P.sel};
  el.scrollIntoView({block: 'center', inline: 'nearest'});
  const r = el.getBoundingClientRect();
  let x, y;
  if (P.kind === 'pane') {
    const ref = [...document.querySelectorAll(P.ref)];
    const b = (P.pick === 'last' ? ref[ref.length - 1] : ref[0]);
    const br = b ? b.getBoundingClientRect() : r;
    x = r.left + 12; y = br.top + br.height / 2;
  } else { x = r.left + 8; y = r.top + r.height / 2; }
  x = Math.floor(x); y = Math.floor(y);
  const hit = document.elementFromPoint(x + 0.5, y + 0.5);
  return {x, y, inside: !!hit && el.contains(hit), hit: hit ? hit.tagName.toLowerCase() +
          (typeof hit.className === 'string' && hit.className ? '.' + hit.className.trim().split(/\\s+/).join('.') : '') : null,
          in_view: y >= 0 && y < innerHeight && x >= 0 && x < innerWidth};
})()
"""


def probe_js(sel: str, kind: str, pick: str, ref: str = "") -> str:
    return PROBE_JS.replace(
        "__P__", json.dumps({"sel": sel, "kind": kind, "pick": pick, "ref": ref})
    )


#: B12: "ΔE probes on the bubble, the conversation interior and the composer interior".
#: name -> (mockup selector, HUD selector, kind, T-0128 token the painted colour is built on)
PROBES = {
    "bubble interior": (M_BUBBLE, H_BUBBLE, "box", None),
    "conversation interior": (M_CONVO, H_CONVO, "pane", "surface"),
    "composer interior": (M_COMPOSER, H_COMPOSER, "box", None),
}

#: B1/B2/B4/B7/B8 facts read off the live page after the replay (before any B9 push).
STRUCTURE_JS = """
(() => {
  const main = document.querySelector('main');
  const holders = main ? [...main.children].filter((c) => c.querySelector('[data-thread]') || c.matches('[data-thread]')) : [];
  const pane = holders[0] || null, thread = document.querySelector('[data-thread]');
  const form = document.querySelector('[data-ask-form]');
  const arts = [...document.querySelectorAll('[data-thread] article[data-turn-id]')];
  const wrappers = [...document.querySelectorAll('[data-ui-surface]')].map((w) => {
    const art = w.closest('article[data-turn-id]');
    return {id: w.getAttribute('data-ui-surface'), inert: w.hasAttribute('inert') || !!w.closest('[inert]'),
            surfaces: w.querySelectorAll('a2ui-surface').length + (w.matches('a2ui-surface') ? 1 : 0),
            turn: art ? art.getAttribute('data-turn-id') : null};
  });
  const card = document.querySelector('[data-tool-call-id="t3-call"]');
  const retired = ['#spoken-pane', '#transcript-pane', '#shown-channel', '.pane-spoken', '.pane-said', '.pane-shown']
    .filter((s) => document.querySelector(s));
  return {
    holders: holders.length,
    pane_is_child_of_main: !!pane && pane.parentElement === main,
    thread_in_pane: !!pane && !!thread && pane.contains(thread),
    form_in_pane: !!pane && !!form && pane.contains(form),
    form_after_thread: !!thread && !!form && !!(thread.compareDocumentPosition(form) & Node.DOCUMENT_POSITION_FOLLOWING) && !thread.contains(form),
    form_bottom_gap: pane && form ? pane.getBoundingClientRect().bottom - form.getBoundingClientRect().bottom : null,
    form_method: form ? form.getAttribute('method') : null,
    form_has_input: !!form && !!form.querySelector('[data-ask-input]'),
    form_has_submit: !!form && !!form.querySelector('[data-ask-submit]'),
    retired,
    placeholder: (document.body.innerText || '').includes('no surface yet (T-0010)'),
    articles: arts.map((a) => a.getAttribute('data-turn-id')),
    surfaces: document.querySelectorAll('a2ui-surface').length,
    wrappers,
    t3_card_turn: card && card.closest('article[data-turn-id]') ? card.closest('article[data-turn-id]').getAttribute('data-turn-id') : null,
  };
})()
""".replace("a2ui-surface", A2UI_ROOT)  # T-0164 B2: the card element, re-keyed by name

SCROLL_STATE_JS = """
(() => { const t = document.querySelector('[data-thread]'); if (!t) return null;
  return {top: t.scrollTop, height: t.scrollHeight, client: t.clientHeight,
          from_bottom: t.scrollHeight - t.scrollTop - t.clientHeight,
          scrollable: getComputedStyle(t).overflowY}; })()
"""

DOC_SCROLL_JS = (
    "({h: document.scrollingElement.scrollHeight, w: document.scrollingElement.scrollWidth,"
    " ih: innerHeight, iw: innerWidth, y: scrollY})"
)

TWO_FRAMES_JS = (
    "new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))"
)


@dataclass
class Measured:
    styles: dict[str, Any] = field(default_factory=dict)
    pixels: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    probes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    error: str = ""
    seconds: float = 0.0
    mock: dict[str, Measured] = field(default_factory=dict)
    hud: dict[str, Measured] = field(default_factory=dict)
    hud_theme: dict[str, Any] = field(default_factory=dict)
    structure: dict[str, Any] = field(default_factory=dict)
    card_rendered: bool = False
    scroll: dict[str, Any] = field(default_factory=dict)
    doc_scroll: dict[str, Any] = field(default_factory=dict)
    requests: list[str] = field(default_factory=list)


def _measure(
    chrome: _Recorder, selectors: list[str], roles: dict[str, str], pick: str, side: int
) -> Measured:
    m = Measured()
    chrome.viewport(*VIEW)
    _settle(chrome)
    base = dict(chrome.evaluate(measure_js(selectors)) or {})
    extra = dict(chrome.evaluate(extra_js(selectors, roles)) or {})
    for sel in selectors:
        base[sel] = [
            {**a, **b} for a, b in zip(base.get(sel) or [], extra.get(sel) or [], strict=False)
        ]
    m.styles = base
    for name, sels in PROBES.items():
        sel, kind = sels[side], sels[2]
        ref = (M_BUBBLE, H_BUBBLE)[side]
        p = chrome.evaluate(probe_js(sel, kind, pick, ref)) or {}
        chrome.evaluate(TWO_FRAMES_JS)
        m.probes[name] = p
        if "x" in p and p.get("in_view"):
            m.pixels[name] = _pixel(chrome, int(p["x"]), int(p["y"]))
    return m


def _fixture_message(turn: str, kind: str) -> str:
    for line in fixture_lines():
        msg = json.loads(line)
        if msg.get("type") == kind and msg.get("turn_id") == turn:
            return line
    raise AssertionError(f"the fixture has no {kind} for {turn}")


def _scroll_rows(chrome: _Recorder, stub: HudStub, result: Session) -> None:
    """B9 in the order a human meets it: at the bottom after the replay; scrolled up by more than
    48 px, new content leaves scrollTop alone; within 48 px, new content keeps the thread at its end."""
    chrome.evaluate(TWO_FRAMES_JS)
    result.scroll["after_replay"] = chrome.evaluate(SCROLL_STATE_JS)
    up = chrome.evaluate(
        "(() => { const t = document.querySelector('[data-thread]'); if (!t) return null;"
        f" t.scrollTop = Math.max(0, t.scrollHeight - t.clientHeight - {STICK_PX + 152});"
        " return t.scrollTop; })()"
    )
    chrome.evaluate(TWO_FRAMES_JS)
    result.scroll["scrolled_up_to"] = up
    stub.push(_fixture_message("t10", "astra.transcript"))
    time.sleep(0.3)
    chrome.evaluate(TWO_FRAMES_JS)
    result.scroll["after_push_up"] = chrome.evaluate(SCROLL_STATE_JS)
    near = chrome.evaluate(
        "(() => { const t = document.querySelector('[data-thread]'); if (!t) return null;"
        f" t.scrollTop = Math.max(0, t.scrollHeight - t.clientHeight - {STICK_PX - 18});"
        " return t.scrollHeight - t.scrollTop - t.clientHeight; })()"
    )
    chrome.evaluate(TWO_FRAMES_JS)
    result.scroll["near_from_bottom"] = near
    stub.push(_fixture_message("t10", "astra.spoken"))
    time.sleep(0.3)
    chrome.evaluate(TWO_FRAMES_JS)
    result.scroll["after_push_near"] = chrome.evaluate(SCROLL_STATE_JS)


def _run(result: Session, chrome_or_why: _Recorder | str) -> None:
    if isinstance(chrome_or_why, str):
        raise RuntimeError(chrome_or_why)
    mock_server, mock_port = _serve_dir(MOCKUP)
    lines = fixture_lines()
    t9_end = _last_index_of(CARD_TURN)
    surfaces = card_surfaces(gated_card())
    stub = HudStub([]).start()
    hud_server = HudStaticServer(HUD, 0)
    try:
        chrome = chrome_or_why
        chrome.reset()
        chrome._cmd("Runtime.enable")
        chrome._cmd("Network.enable")
        for theme in THEMES:
            _media(chrome, theme)
            chrome.viewport(*VIEW)
            chrome.navigate(f"http://127.0.0.1:{mock_port}/hud.html?theme={theme}")
            chrome.wait_until("document.readyState === 'complete'", 15)
            result.mock[theme] = _measure(chrome, MOCK_SELECTORS, MOCK_ROLES, "first", 0)

        url = f"http://127.0.0.1:{hud_server.port}/?ws=" + urllib.parse.quote(stub.url, safe="")
        _media(chrome, "dark")
        chrome.viewport(*VIEW)
        chrome.navigate(url)
        stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20)
        stub.wait_for_client(10)
        chrome.wait_panes(8)
        for line in lines[: t9_end + 1]:
            if _is_stale_card_surface(line):
                continue
            stub.push(line)
            time.sleep(0.01)
        for surface in surfaces:
            stub.push(surface)
            time.sleep(0.02)
        stub.push(AstraTurnState(state=TurnState.SPEAKING, turn_id=CARD_TURN))
        result.card_rendered = bool(
            chrome.wait_until(
                "!!document.querySelector('[data-ui-surface=\"turn-t9\"] a2ui-surface, a2ui-surface')".replace(
                    "a2ui-surface", A2UI_ROOT
                ),
                10,
            )
        )
        chrome.drain(0.5)
        result.structure = dict(chrome.evaluate(STRUCTURE_JS) or {})
        _scroll_rows(chrome, stub, result)
        for theme in THEMES:
            result.hud_theme[theme] = chrome.evaluate(choose_theme_js(theme)) or {}
            _media(chrome, theme)
            result.hud[theme] = _measure(chrome, HUD_SELECTORS, HUD_ROLES, "last", 1)
        chrome.evaluate(choose_theme_js("system"))
        for w, h in NO_DOC_SCROLL:
            chrome.viewport(w, h)
            chrome.evaluate(TWO_FRAMES_JS)
            result.doc_scroll[f"{w}x{h}"] = chrome.evaluate(DOC_SCROLL_JS)
        chrome.viewport(*VIEW)
        chrome.drain(0.3)
        result.requests = list(chrome.requests)
    finally:
        hud_server.close()
        stub.stop()
        mock_server.shutdown()
        mock_server.server_close()


@pytest.fixture(scope="module")
def session(browser: Any) -> Any:
    result = Session()
    if CHROME is None:  # pragma: no cover - requires_chrome is the real guard
        result.error = "no chrome"
        yield result
        return
    started = time.monotonic()
    try:
        _run(result, browser)
    except Exception as exc:  # a broken session fails in a row, not here
        result.error = f"{type(exc).__name__}: {exc}"
    result.seconds = time.monotonic() - started
    yield result


def _ok(s: Session) -> None:
    assert not s.error, f"the thread parity session failed before any assertion ran: {s.error}"


def _evaluate(row: Row, mock: dict[str, Any], hud: dict[str, Any], theme: str) -> list[str]:
    if row.prop in EXACT:
        mels, hels = mock.get(row.mock) or [], hud.get(row.hud) or []
        if not mels:
            return [f"mockup selector {row.mock!r} matched nothing (the reference is missing)"]
        if not hels:
            return [f"HUD selector {row.hud!r} matched nothing"]
        want = mels[0].get(row.prop)
        return [
            f"{e.get('tag', row.hud)}: {row.prop} {e.get(row.prop)!r} vs the mockup's {want!r} (exact)"
            for e in hels
            if e.get(row.prop) != want
        ]
    return evaluate_row(row, mock, hud, theme)


# --- the premise: the mockup's thread is what its CSS says (no group B needed) --------------------


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_the_mockups_thread_renders_what_its_own_css_says(session, theme):
    """Anti-vacuity for every P2 row: the reference values, hand-read from the committed
    ``hud.html`` (B12's numbers), are what the mockup really computes in this Chrome; and its own
    probes land inside the role they name."""
    _ok(session)
    m = session.mock[theme]
    s = m.styles
    assert s.get("__theme__") == theme, s.get("__theme__")
    convo, thread, bubble = s[M_CONVO][0], s[M_THREAD][0], s[M_BUBBLE][0]
    row, avatar, prose, chip = s[M_ROW][0], s[M_AVATAR][0], s[M_PROSE][0], s[M_CHIP][0]
    wrap, composer, submit = s[M_WRAP][0], s[M_COMPOSER][0], s[M_SUBMIT][0]
    assert (convo["radius"], convo["border_w"]) == (22, 1) and convo["shadow"] != "none", convo
    assert (thread["pad_t"], thread["pad_r"], thread["pad_l"]) == (26, 34, 34), thread
    assert bubble["max_width_pct"] == 70, bubble
    assert (bubble["radius_tl"], bubble["radius_tr"], bubble["radius_br"], bubble["radius_bl"]) == (
        18,
        18,
        6,
        18,
    )
    assert (bubble["pad_t"], bubble["pad_r"], bubble["pad_l"], bubble["font_size"]) == (
        11,
        16,
        16,
        15,
    ), bubble
    assert abs(bubble["geom_right_inset"]) < 0.5, bubble
    assert (
        row["max_width_px"] == 860 and row["geom_avatar_left"] == 0 and row["geom_avatar_gap"] == 14
    ), row
    assert (avatar["w"], avatar["h"], avatar["radius"]) == (30, 30, 10), avatar
    assert prose["font_size"] == 16.5 and abs(prose["line_height"] - 16.5 * 1.65) < 0.01, prose
    assert (chip["border_style"], chip["border_w"], chip["radius"]) == ("dashed", 1, 12), chip
    assert wrap["max_width_px"] == 520, wrap
    assert (composer["radius"], composer["border_w"]) == (16, 1), composer
    # margin 0 22 22 inside a pane with a 1 px border: the border box sits 23 px in.
    assert (
        composer["geom_inset_left"],
        composer["geom_inset_right"],
        composer["geom_inset_bottom"],
    ) == (23, 23, 23)
    assert composer["geom_below_thread"] == 0, composer
    assert (submit["w"], submit["h"], submit["radius"]) == (38, 38, 12), submit
    for name, p in m.probes.items():
        assert p.get("inside") and p.get("in_view"), (
            f"the mockup's {name} probe missed its element: {p}"
        )


def test_the_p2_table_names_every_role_b12_lists():
    """Pure: the role table covers B12's list (so a dropped row is a visible diff, not a silent gap)."""
    roles = " | ".join(r.role for r in P2)
    for needle in (
        "conversation pane: radius 22",
        "shadow",
        "thread: padding top 26",
        "thread: padding left 34",
        "max-width 70%",
        "radius bottom-right 6",
        "padding top 11",
        "padding left 16",
        "user bubble: 15 px",
        "--bubble fill",
        "max-width 860",
        "14 px from the avatar",
        "avatar: 30 wide",
        "avatar: radius 10",
        "prose: 16.5 px",
        "line-height 1.65",
        "dashed border",
        "action chip: radius 12",
        "amber text",
        "card wrap: max-width 520",
        "composer: radius 16",
        "22 from the pane's left",
        "22 from the pane's bottom",
        "--surface-2 fill",
        "composer: 1 px border",
        "composer submit: 38 wide",
        "submit: radius 12",
        "accent fill",
    ):
        assert needle in roles, f"B12: the P2 table has no row for {needle!r}"
    assert set(PROBES) == {"bubble interior", "conversation interior", "composer interior"}


# --- B12: every P2 row, and the probes ----------------------------------------------------------------


@requires_chrome
def test_the_hud_session_reached_t9_with_the_card_in_the_thread(session):
    """Premise of the HUD side: the replay reached t9, the gated card mounted, and both themes were
    chosen through the Settings control."""
    _ok(session)
    assert session.card_rendered, "the t9 card never mounted an <a2ui-surface>"
    assert "t9" in (session.structure.get("articles") or []), session.structure
    for theme in THEMES:
        chosen = session.hud_theme.get(theme) or {}
        assert chosen.get("ok") and chosen.get("theme") == theme, chosen


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_b12_every_thread_role_matches_the_mockup(session, theme):
    """B12: the P2 role table, measured live from the mockup, both themes; every failing row listed."""
    _ok(session)
    failures: list[str] = []
    for row in P2:
        mock, hud = session.mock[theme].styles, session.hud[theme].styles
        for problem in _evaluate(row, mock, hud, theme):
            dev = f" [deviation: --{row.deviation}]" if row.deviation else ""
            failures.append(f"{row.role}{dev}: {problem}")
    assert failures == [], (
        f"T-0154 B12 (data-theme={theme}): {len(failures)} thread role(s) differ from the mockup:\n  "
        + "\n  ".join(failures)
    )


def _probe_tolerance(mock: Measured, token: str | None, theme: str) -> tuple[float, str]:
    """A3's rule, reused for B12: dE <= 3, a named T-0128 deviation up to its own token dE + 1."""
    if token is None or (token, theme) not in T0128_DEVIATIONS:
        return PIXEL_TOL_DE, ""
    mine = _token_rgb(mock.styles.get("__tokens__") or {}, token)
    target = hex_rgb(T0128_DEVIATIONS[(token, theme)])
    if mine is None:
        return PIXEL_TOL_DE, f" (mockup --{token} unreadable)"
    dev = delta_e(mine, target)
    return max(PIXEL_TOL_DE, dev + 1.0), f" (--{token} deviation dE {dev:.2f} + 1)"


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_b12_the_thread_probe_pixels_match_the_mockup(session, theme):
    """B12: dE00 on the bubble, the conversation interior and the composer interior, each probe in
    its element's own padding and inside the element it names."""
    _ok(session)
    m, h = session.mock[theme], session.hud[theme]
    failures = []
    for name, (_ms, _hs, _kind, token) in PROBES.items():
        hp = h.probes.get(name) or {}
        if not hp.get("inside") or name not in h.pixels:
            failures.append(f"{name}: the HUD probe did not land inside its element: {hp}")
            continue
        tol, why = _probe_tolerance(m, token, theme)
        de = delta_e(m.pixels[name], h.pixels[name])
        if de > tol:
            failures.append(
                f"{name}: HUD {tuple(round(c) for c in h.pixels[name])} vs mockup "
                f"{tuple(round(c) for c in m.pixels[name])}, dE00 {de:.2f} > {tol:.2f}{why}"
            )
    assert failures == [], f"T-0154 B12 probes (data-theme={theme}):\n  " + "\n  ".join(failures)


# --- B1, B2, B4, B7, B8 on the live page ---------------------------------------------------------------


@requires_chrome
def test_b1_one_conversation_pane_holds_the_thread_and_the_ask_form_at_its_bottom(session):
    """B1: one direct child of <main> holds [data-thread] and the ask form after it, at its bottom;
    the form keeps method=dialog and its two hooks; the retired ids and the placeholder are gone."""
    _ok(session)
    s = session.structure
    assert s.get("holders") == 1 and s.get("pane_is_child_of_main"), (
        f"B1: one conversation pane, a direct child of <main>: {s}"
    )
    assert s.get("thread_in_pane") and s.get("form_in_pane"), s
    assert s.get("form_after_thread"), "B1: the ask form sits after the thread, outside it"
    assert s.get("form_bottom_gap") is not None and 0 <= s["form_bottom_gap"] <= 40, (
        f"B1: the ask form is at the pane's bottom (gap {s.get('form_bottom_gap')} px)"
    )
    assert (
        s.get("form_method") == "dialog" and s.get("form_has_input") and s.get("form_has_submit")
    ), s
    assert s.get("retired") == [], f"B1: retired elements still on the page: {s.get('retired')}"
    assert not s.get("placeholder"), "B1: the 'no surface yet (T-0010)' placeholder is retired"


@requires_chrome
def test_b2_b4_b7_b8_the_replayed_thread_holds_one_article_per_turn_and_one_card_per_surface(
    session,
):
    """After the replay to t9 (fixture, hand-counted): articles t3–t9 in order; surfaces for t3, t4,
    t5, t6, t8 and the gated t9 (t7 has none), each in its own turn's article, one <a2ui-surface> per
    id; t3's action card in t3's article; every wrapper but turn-t9's inert."""
    _ok(session)
    s = session.structure
    assert s.get("articles") == ["t3", "t4", "t5", "t6", "t7", "t8", "t9"], (
        f"B2: {s.get('articles')}"
    )
    wrappers = s.get("wrappers") or []
    ids = [w["id"] for w in wrappers]
    want = ["turn-t3", "turn-t4", "turn-t5", "turn-t6", "turn-t8", "turn-t9"]
    assert sorted(ids) == sorted(want) and len(ids) == len(set(ids)), (
        f"B4: one wrapper per surface id: {ids}"
    )
    assert s.get("surfaces") == 6, (
        f"B4: one <a2ui-surface> per surface id (6), found {s.get('surfaces')}"
    )
    for w in wrappers:
        assert w["surfaces"] == 1, f"B4: {w['id']}'s wrapper holds one <a2ui-surface>: {w}"
        assert w["turn"] == w["id"].removeprefix("turn-"), (
            f"B4: {w['id']} renders in its own turn's article: {w}"
        )
        assert w["inert"] == (w["id"] != "turn-t9"), (
            f"B8/H-T154-4: only the newest card is live: {w}"
        )
    assert s.get("t3_card_turn") == "t3", "B7: t3's action card stays in t3's article as history"


# --- B9: scrolling ------------------------------------------------------------------------------------


@requires_chrome
def test_b9_the_thread_sticks_to_its_bottom_within_48_px_and_leaves_a_reader_alone_beyond(session):
    """B9, in the order a human meets it (both directions of the 48 px guard)."""
    _ok(session)
    sc = session.scroll
    after = sc.get("after_replay") or {}
    assert after, "B9: the page has no [data-thread]"
    assert after["height"] > after["client"] + 2 * STICK_PX + 152, (
        f"premise: the replayed thread overflows its pane (scrollHeight {after['height']}, "
        f"clientHeight {after['client']}); otherwise nothing here can scroll"
    )
    assert after["from_bottom"] <= 1, (
        f"B9: new content kept the thread at its bottom during the replay: {after}"
    )
    up = sc.get("after_push_up") or {}
    assert up.get("top") == sc.get("scrolled_up_to"), (
        f"B9: scrolled up by more than {STICK_PX} px, new content must not change scrollTop: "
        f"was {sc.get('scrolled_up_to')}, now {up}"
    )
    assert sc.get("near_from_bottom") is not None and sc["near_from_bottom"] <= STICK_PX, sc
    near = sc.get("after_push_near") or {}
    assert near.get("from_bottom") is not None and near["from_bottom"] <= 1, (
        f"B9: within {STICK_PX} px of the bottom, new content keeps the thread at its bottom: {near}"
    )


@requires_chrome
def test_b9_the_document_never_scrolls_at_1100_px_and_wider(session):
    """B9's last sentence. Holds before group B too (T-0128 criterion 4's panes scroll inside
    themselves), so it is an ordinary row: the thread must keep it, not earn it."""
    _ok(session)
    for view, d in session.doc_scroll.items():
        assert d and d["h"] <= d["ih"] and d["w"] <= d["iw"], (
            f"B9 at {view}: the document scrolls (scrollHeight {d and d['h']} > {d and d['ih']} or "
            f"scrollWidth {d and d['w']} > {d and d['iw']}); the thread scrolls inside its pane"
        )


# --- B9 under a stream, and its edges (O-B3 (3): F1, F3) -------------------------------------------
#
# Reviewer 2 (group B review at 6cdef97, F1): a reader cannot scroll away from the bottom with small
# wheel steps while the fixture's 20 Hz THINKING ``astra.turn.state`` ticks stream, because every
# message re-sticks the thread while the reader is still inside the 48 px band (probe: 20 steps of
# 12 px at ~60 Hz left the thread 0 px from its bottom). O-B3 (2) rules the fix: re-stick only on a
# message that changed the DOM, and stop sticking on the user's upward wheel/key/touch intent. The
# rows below read B9 as a human meets it, in one fresh page of the module's one Chrome:
#
# 1. F3, the ResizeObserver: replay to the end of t9 *without* t9's card, at the bottom; then the
#    gated card's surfaces and nothing else (no tick, no text). The renderer draws the card after its
#    message was handled, so only a re-stick on the article's growth keeps the thread at its end.
# 2. F3, STICK_PX: scrolled to exactly 47 px from the bottom, a DOM-changing message keeps it at the
#    bottom; at exactly 49 px it leaves scrollTop alone (B9's "within 48 px" — both sides, one px in).
# 3. F1: 20 wheel steps of -12 px at ~60 Hz (CDP ``Input.dispatchMouseEvent`` mouseWheel over the
#    thread), first with no stream (premise: the wheel really scrolls >= 200 px here), then under
#    20 Hz THINKING ticks for the newest turn: the reader must end >= 200 px from the bottom, and
#    0.5 s more ticks must not move scrollTop. Also three -100 px notches under ticks.
# 4. The other direction (a fix must not be too wide): the reader wheels back down to the bottom by
#    hand, and the next DOM-changing message keeps the thread at its bottom again.
#
# Regime: Google Chrome headless, CDP, DPR 1, 1440x900, dark, reduced motion off (the page's own
# default), fixture session_20261020_t3_t11.jsonl; ticks pushed every 50 ms from a test thread; n=1.

#: O-B3 (3), F1: "must leave the reader where they scrolled" — 20 x 12 px is 240 px of intent;
#: 200 px leaves room for wheel rounding while sitting far outside the 48 px band.
F1_MIN_FROM_BOTTOM = 200
F1_STEPS, F1_DELTA, F1_GAP_S = 20, -12, 0.016
TICK_S = 0.05

THREAD_CENTRE_JS = (
    "(() => { const t = document.querySelector('[data-thread]'); if (!t) return null;"
    " const r = t.getBoundingClientRect(); return {x: r.x + r.width / 2, y: r.y + r.height / 2}; })()"
)


def _set_from_bottom_js(px: int) -> str:
    return (
        "(() => { const t = document.querySelector('[data-thread]'); if (!t) return null;"
        f" t.scrollTop = t.scrollHeight - t.clientHeight - {px};"
        " return t.scrollHeight - t.scrollTop - t.clientHeight; })()"
    )


CARD_HEIGHT_JS = (
    "(() => { const w = document.querySelector('[data-ui-surface=\"turn-t9\"]');"
    " const s = w && w.querySelector('a2ui-surface');"
    " return s ? Math.round(w.getBoundingClientRect().height) : 0; })()"
).replace("a2ui-surface", A2UI_ROOT)


@dataclass
class B9Stream:
    error: str = ""
    seconds: float = 0.0
    rows: dict[str, Any] = field(default_factory=dict)


class _Ticker:
    """20 Hz THINKING ``astra.turn.state`` for one turn, pushed from a thread (the fixture's cadence)."""

    def __init__(self, stub: HudStub, turn: str) -> None:
        import threading

        self._stop = threading.Event()
        self.sent = 0

        def run() -> None:
            ms = 0
            while not self._stop.is_set():
                stub.push(AstraTurnState(state=TurnState.THINKING, turn_id=turn, thinking_ms=ms))
                self.sent += 1
                ms += int(TICK_S * 1000)
                time.sleep(TICK_S)

        self._thread = threading.Thread(target=run, daemon=True)

    def __enter__(self) -> _Ticker:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(5)


def _wheel(chrome: _Recorder, at: dict[str, float], dy: int, n: int, gap_s: float) -> None:
    for _ in range(n):
        chrome._cmd(
            "Input.dispatchMouseEvent", type="mouseWheel", x=at["x"], y=at["y"], deltaX=0, deltaY=dy
        )
        time.sleep(gap_s)


def _settled_scroll(chrome: _Recorder, quiet_s: float = 0.4) -> Any:
    chrome.drain(quiet_s)
    chrome.evaluate(TWO_FRAMES_JS)
    return chrome.evaluate(SCROLL_STATE_JS)


def _run_b9_stream(result: B9Stream, chrome_or_why: _Recorder | str) -> None:
    if isinstance(chrome_or_why, str):
        raise RuntimeError(chrome_or_why)
    chrome = chrome_or_why
    lines = fixture_lines()
    t9_end = _last_index_of(CARD_TURN)
    surfaces = card_surfaces(gated_card())
    rows = result.rows
    stub = HudStub([]).start()
    hud_server = HudStaticServer(HUD, 0)
    try:
        chrome.reset()
        _media(chrome, "dark")
        chrome.viewport(*VIEW)
        chrome.navigate(
            f"http://127.0.0.1:{hud_server.port}/?ws=" + urllib.parse.quote(stub.url, safe="")
        )
        stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20)
        stub.wait_for_client(10)
        chrome.wait_panes(6)
        for line in lines[: t9_end + 1]:
            if not _is_stale_card_surface(line):
                stub.push(line)
        rows["before_card"] = _settled_scroll(chrome, 1.0)
        rows["card_height_before"] = chrome.evaluate(CARD_HEIGHT_JS)

        # 1. F3: the card renders after its message; nothing else arrives. One message only: the
        #    gate's last canvas, final (a later duplicate of it would be a message, and every
        #    message re-sticks — that would hide a missing ResizeObserver).
        rows["card_messages"] = 1 if surfaces else 0
        if surfaces:
            stub.push(surfaces[-1])
        chrome.wait_until(f"({CARD_HEIGHT_JS}) > {2 * STICK_PX}", 10)
        chrome.drain(0.8)  # let the renderer finish growing the card
        rows["card_height_after"] = chrome.evaluate(CARD_HEIGHT_JS)
        rows["after_card"] = _settled_scroll(chrome)

        # 2. F3: STICK_PX, one px either side of 48.
        for px, msg in ((47, ("t10", "astra.transcript")), (49, ("t10", "astra.spoken"))):
            rows[f"set_{px}"] = chrome.evaluate(_set_from_bottom_js(px))
            chrome.evaluate(TWO_FRAMES_JS)
            rows[f"before_{px}"] = chrome.evaluate(SCROLL_STATE_JS)
            stub.push(_fixture_message(*msg))
            rows[f"after_{px}"] = _settled_scroll(chrome)

        # 3. F1: small wheel steps, quiet and under ticks; then notches under ticks.
        at = chrome.evaluate(THREAD_CENTRE_JS)
        rows["centre"] = at
        for arm, ticking, dy, n, gap in (
            ("quiet", False, F1_DELTA, F1_STEPS, F1_GAP_S),
            ("thinking", True, F1_DELTA, F1_STEPS, F1_GAP_S),
            ("notches_thinking", True, -100, 3, 0.08),
        ):
            chrome.evaluate(_set_from_bottom_js(0))
            rows[f"{arm}_start"] = _settled_scroll(chrome)
            if ticking:
                with _Ticker(stub, "t10") as ticker:
                    time.sleep(0.2)  # the stream is already running when the reader starts
                    _wheel(chrome, at, dy, n, gap)
                    rows[f"{arm}_end"] = _settled_scroll(chrome, 0.3)
                    time.sleep(0.5)
                    rows[f"{arm}_later"] = _settled_scroll(chrome, 0.1)
                    rows[f"{arm}_ticks"] = ticker.sent
            else:
                _wheel(chrome, at, dy, n, gap)
                rows[f"{arm}_end"] = _settled_scroll(chrome, 0.6)

        # 4. The other direction: back to the bottom by the reader's own wheel, then new content.
        _wheel(chrome, at, 120, 8, 0.03)
        rows["wheeled_back"] = _settled_scroll(chrome, 0.6)
        stub.push(_fixture_message("t11", "astra.transcript"))
        rows["after_return"] = _settled_scroll(chrome, 0.6)
    finally:
        hud_server.close()
        stub.stop()


@pytest.fixture(scope="module")
def b9_stream(browser: Any, session: Any) -> Any:
    """Runs after ``session`` (same Chrome, a fresh page): requesting it keeps the order fixed."""
    result = B9Stream()
    if CHROME is None:  # pragma: no cover
        result.error = "no chrome"
        yield result
        return
    started = time.monotonic()
    try:
        _run_b9_stream(result, browser)
    except Exception as exc:  # a broken session fails in a row, not here
        result.error = f"{type(exc).__name__}: {exc}"
    result.seconds = time.monotonic() - started
    yield result


def _b9(s: B9Stream) -> dict[str, Any]:
    assert not s.error, f"the B9 stream session failed before any assertion ran: {s.error}"
    return s.rows


@requires_chrome
def test_b9_f3_a_card_that_renders_after_its_message_keeps_the_thread_at_its_bottom(b9_stream):
    """O-B3 (3), F3: the ResizeObserver re-stick. At the bottom after the replay to t9 without its
    card; then one message, the gated card's last canvas. The card grows t9's article after that
    message was handled; the thread must still end at its bottom (no tick, no text and no second
    canvas arrives to re-stick it)."""
    r = _b9(b9_stream)
    assert r.get("card_messages") == 1, "premise: the card arrived as exactly one message"
    before, after = r.get("before_card") or {}, r.get("after_card") or {}
    assert before and before["from_bottom"] <= 1, (
        f"premise: at the bottom before the card: {before}"
    )
    assert before["height"] > before["client"] + 2 * STICK_PX, (
        f"premise: the thread overflows its pane before the card: {before}"
    )
    assert r.get("card_height_before") == 0, "premise: no t9 card before its surfaces arrive"
    assert (r.get("card_height_after") or 0) > 2 * STICK_PX, (
        f"premise: the card rendered taller than the 48 px band ({r.get('card_height_after')} px)"
    )
    assert after.get("height", 0) >= before["height"] + 2 * STICK_PX, (
        f"premise: the card grew the thread: {before} -> {after}"
    )
    assert after.get("from_bottom") is not None and after["from_bottom"] <= 1, (
        f"B9/F3: at the bottom when the card arrived, the thread must stay at its bottom after the "
        f"card renders: {after} (card {r.get('card_height_after')} px)"
    )


@requires_chrome
def test_b9_f3_stick_px_is_48_47_px_sticks_and_49_px_stays(b9_stream):
    """O-B3 (3), F3: B9's band, one px inside and one px outside. Scrolled to exactly 47 px from
    the bottom, a DOM-changing message (t10's transcript) keeps the thread at its bottom; at exactly
    49 px, the next one (t10's utterance) leaves scrollTop where the reader put it."""
    r = _b9(b9_stream)
    assert r.get("set_47") == 47 and r.get("set_49") == 49, (
        f"premise: scrolled to exactly 47 / 49 px from the bottom: {r.get('set_47')}, {r.get('set_49')}"
    )
    a47 = r.get("after_47") or {}
    assert a47.get("from_bottom") is not None and a47["from_bottom"] <= 1, (
        f"B9/F3: 47 px from the bottom is within 48 px — new content keeps the thread at its bottom: "
        f"before {r.get('before_47')}, after {a47}"
    )
    b49, a49 = r.get("before_49") or {}, r.get("after_49") or {}
    assert a49.get("height", 0) > b49.get("height", 0), (
        f"premise: t10's utterance grew the thread: {b49} -> {a49}"
    )
    assert a49.get("top") == b49.get("top"), (
        f"B9/F3: 49 px from the bottom is beyond 48 px — new content must not change scrollTop: "
        f"{b49} -> {a49}"
    )


@requires_chrome
def test_b9_f1_small_wheel_steps_under_thinking_ticks_leave_the_reader_where_they_scrolled(
    b9_stream,
):
    """O-B3 (3), F1 (adversarial, the failure mode reviewer 2 named): 20 wheel steps of 12 px at
    ~60 Hz while THINKING ticks stream at 20 Hz. The reader must end >= 200 px from the bottom, and
    0.5 s more ticks must not move scrollTop. The quiet arm is the premise: the same wheel, with no
    stream, does carry the reader >= 200 px here."""
    r = _b9(b9_stream)
    for arm in ("quiet", "thinking", "notches_thinking"):
        start = r.get(f"{arm}_start") or {}
        assert start and start["from_bottom"] <= 1, (
            f"premise ({arm}): starts at the bottom: {start}"
        )
    quiet = r.get("quiet_end") or {}
    assert quiet.get("from_bottom", 0) >= F1_MIN_FROM_BOTTOM, (
        f"premise: with no stream, {F1_STEPS} wheel steps of {-F1_DELTA} px carry the reader >= "
        f"{F1_MIN_FROM_BOTTOM} px from the bottom: {quiet}"
    )
    assert (r.get("thinking_ticks") or 0) >= 10, (
        f"premise: the stream ran (>= 10 THINKING ticks during the arm): {r.get('thinking_ticks')}"
    )
    end, later = r.get("thinking_end") or {}, r.get("thinking_later") or {}
    assert end.get("from_bottom", 0) >= F1_MIN_FROM_BOTTOM, (
        f"B9/F1: under 20 Hz THINKING ticks, {F1_STEPS} wheel steps of {-F1_DELTA} px must leave the "
        f"reader >= {F1_MIN_FROM_BOTTOM} px from the bottom (quiet arm: {quiet.get('from_bottom')} px); "
        f"got {end}. A turn state is not new content: it must not re-stick the thread."
    )
    assert later.get("top") == end.get("top"), (
        f"B9/F1: scrolled up, 0.5 s more ticks must not change scrollTop: {end} -> {later}"
    )
    notch, notch_later = r.get("notches_thinking_end") or {}, r.get("notches_thinking_later") or {}
    assert notch.get("from_bottom", 0) >= F1_MIN_FROM_BOTTOM and notch_later.get(
        "top"
    ) == notch.get("top"), (
        f"B9/F1: three 100 px notches under ticks leave the reader where they went: {notch} -> {notch_later}"
    )


@requires_chrome
def test_b9_f1_a_reader_who_wheels_back_to_the_bottom_is_stuck_to_it_again(b9_stream):
    """The other side of F1's fix (a guard needs a row for being too wide): stopping on the user's
    upward intent must not stop sticking for good. Wheeled back to the bottom by hand, the next
    DOM-changing message (t11's transcript) keeps the thread at its bottom (B9's first sentence)."""
    r = _b9(b9_stream)
    back, after = r.get("wheeled_back") or {}, r.get("after_return") or {}
    assert back and back["from_bottom"] <= 1, (
        f"premise: the reader's wheel reached the bottom: {back}"
    )
    assert after.get("height", 0) > back.get("height", 0), (
        f"premise: t11's transcript grew the thread: {back} -> {after}"
    )
    assert after.get("from_bottom") is not None and after["from_bottom"] <= 1, (
        f"B9: back at the bottom, new content keeps the thread at its bottom: {back} -> {after}"
    )


@requires_chrome
def test_the_b9_stream_session_stays_inside_its_budget(b9_stream):
    """G1: the stream session adds <= 30 s to the module (its sleeps are ~5 s of it)."""
    _b9(b9_stream)
    assert b9_stream.seconds <= 30, f"G1: the B9 stream session took {b9_stream.seconds:.1f} s"


@requires_chrome
def test_the_thread_parity_session_requests_nothing_off_loopback(session):
    _ok(session)
    assert session.requests, "premise: the request log recorded something"
    offenders = [
        u
        for u in session.requests
        if not re.match(r"^(https?|wss?)://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?/", u)
        and not u.startswith(("data:", "blob:", "about:"))
    ]
    assert offenders == [], f"T-0128 criterion 2: off-loopback requests: {offenders}"


@requires_chrome
def test_the_module_stays_inside_its_budget(session):
    """G1: the parity module runs in <= 60 s (this session is nearly all of it)."""
    _ok(session)
    assert session.seconds <= 60, f"G1: the thread parity session took {session.seconds:.1f} s"


# --- B12: the screenshot script stops scrolling panes by hand -----------------------------------------


def _code_of(path: Path) -> str:
    """The script's code and string literals with comments and docstrings removed (the JS the
    script runs lives in string literals, so strings stay; prose does not)."""
    out = []
    prev = tokenize.INDENT
    for tok in tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and prev in (
            tokenize.INDENT,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.DEDENT,
        ):
            prev = tok.type  # a bare string statement: a docstring
            continue
        out.append(tok.string)
        if tok.type not in (tokenize.NL, tokenize.COMMENT):
            prev = tok.type
    return " ".join(out)


def test_the_code_reader_keeps_js_strings_and_drops_docstrings(tmp_path):
    """Anti-vacuity for the row below: hand-built source with each case."""
    src = tmp_path / "s.py"
    src.write_text(
        '"""module docstring: p.scrollTop = 1"""\n'
        "# comment: p.scrollTop = 2\n"
        'JS = "p.scrollTop = p.scrollHeight"\n'
        "def f():\n"
        '    """doc: scrollIntoView"""\n'
        "    return 1\n",
        encoding="utf-8",
    )
    code = _code_of(src)
    assert "p.scrollTop = p.scrollHeight" in code
    assert "p.scrollTop = 1" not in code and "p.scrollTop = 2" not in code
    assert "scrollIntoView" not in code


def test_b12_the_screenshot_script_stops_scrolling_panes_by_hand():
    """B12: with the thread sticking to its bottom (B9), ``hud_screens.py`` no longer scrolls any
    pane itself — no ``scrollTop =``, ``scrollTo``, ``scrollBy`` or ``scrollIntoView`` in the JS it
    runs. (A source guard: it reads code and string literals, not comments or docstrings.)"""
    code = _code_of(HUD_SCREENS)
    hits = re.findall(r"scrollTop\s*=(?!=)|scrollTo\s*\(|scrollBy\s*\(|scrollIntoView", code)
    assert hits == [], f"B12: scripts/demo/hud_screens.py still scrolls panes by hand: {hits}"
