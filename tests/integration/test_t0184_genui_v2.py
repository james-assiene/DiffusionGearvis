"""T-0184 group B: the v2 cards in the real HUD page, measured in Playwright's Chromium.

docs/tasks/T-0184.md B3 (the look, by computed style and pixels, both themes), B5 (a real click,
decoded, equals a gate-made Button's frame), B8 (contrast against what is actually behind each
text and fill, with the mutant arm), B9 (the overflow matrix), B10 (the screenshot script) and B11
(loopback only, no ``@font-face``), plus H-T184-5 (one frame for every card, v1 included; this
module takes the frame rows that ``test_hud_card_parity.py`` gave up by name). The node rows (B1,
B2, B4-B7, B11's sink half) are in ``tests/hud/ui_cards_v2.test.mjs``.

**Regime.** Playwright 1.64.0, its pinned Chromium, one browser for the whole module (closed again
before any row runs, so B10's script run never overlaps it), DPR 1, reduced motion emulated and
the T-0056 neutraliser injected (``transition-property: none``, not a near-zero duration). The HUD
is the production ``HudStaticServer`` on loopback with a ``HudStub`` socket; surfaces are pushed as
the session pushes them (``AstraUiSurface``, ``final``). Nothing goes through the gate here except
B5's gate-made Button (group C owns the v2 gate). No GPU, n = 1, deterministic.

**What is pushed.** The 19 payloads of ``tests/fixtures/a2ui/cards_v2_exemplars.jsonl`` (group A),
the v1 card ``desk-t9`` of the T-0164 census (H-T184-5), and this module's own labelled synthetic
payloads (ids ``t184-*``): a tie for the largest bar, extra and missing comparison cells, and B9's
stress cases (12 bars with 60-character labels, a 5-column comparison, a 10,000-character word in
every role; the 4 tiles with 12-digit values are the exemplar ``stattiles-four-wide``).

**Theme states.** "system-dark"/"system-light": the Settings control on ``system`` and the colour
scheme emulated; "data-dark"/"data-light": the control on ``dark``/``light`` with the *opposite*
scheme emulated (B8: "``data-theme`` and the emulated system theme"). B3 is read in the two system
states, B8 in all four.

**How pixels are read.** A card is scrolled to the middle of its pane, its text is made
transparent by an injected rule (so the shot shows exactly what is behind every glyph: frame,
gradient, tint, pill), and the clip is decoded in a second, blank page through a canvas (no
request, nothing leaves the process). Colours are compared with CIEDE2000 and WCAG 2.1 ratios
(``test_hud_mockup_parity.delta_e``, ``test_hud_chrome.contrast_ratio``).

**Readings the task leaves open, stated so a reviewer can disagree:**

- *Locating parts.* Components by T-0164's ``data-a2-id``/``data-a2-kind``; everything inside a v2
  component through the model's own text nodes (a label's element is its text node's parent); a
  tile is the highest ancestor of its title inside the StatTiles element that holds no other
  tile's title. Bar and progress fills are the *innermost painted text-less* elements of a Bars or
  StatTiles component (painted: a background colour with alpha or a background image), one per
  bar, in document order. No class name is read.
- *Light frame* ("derived equivalents", H-T184-5): the same radius, border width, padding and
  shadow as dark; the frame must stand off the pane behind it (fill or border dE00 >= 1, the
  T-0164 rule); its colours are not pinned.
- *Large text* is WCAG 2.1's (>= 18 px, or >= 14 px at 700): 3:1, else 4.5:1. The task names the
  56 px value; a 22 px tile value is large too.
- *A fill's neighbour* is the pixel 2 px right of its end at mid-height (the gap or the track a
  bar or progress fill is read against); fills narrower than 2 px are not measured.
- *The secondary button's boundary* is the better of its border and its fill against the card
  pixel 3 px outside it; *the focus ring* is the outline drawn by ``focus({focusVisible: true})``
  on the newest card's first Actions button, against the pixel just outside it.
- *Highlight tint* (StatTiles, Comparison): the tinted backdrop differs from the plain one by
  dE00 >= 1 and is nearer ``--accent``; untinted columns of one row differ by dE00 <= 0.6.
- *"From 0 to the largest value"*: bar widths are read as proportional to the values
  (``w_i = w_max * v_i / v_max`` +/- 1.5 px), whatever the scale's end.

Activated for ``/implement T-0184`` group B (the rows carried ``pending(task="T-0184")`` from
375494ed until activation); the mock premise rows were guards throughout.

**Group B amendment after the implementation (2026-10-28), rows added here:**

- *B5a* (``test_b5a_*``): the newest card (``TAB_SID``, pushed last, so every other card is
  ``inert``) holds a 5-column Comparison that scrolls sideways and two Actions; Tab is pressed
  from the top of the page until focus wraps (at most ``TAB_MAX``), and every stop inside
  ``[data-a2ui-root]`` must be an Actions ``<button>``. Premises: the Comparison scrolls, and the
  walk reaches both Actions buttons. Red at f17952a2: the Comparison's scroller takes focus.
- *B7a* (``test_b7a_*``): the tick labels of every Bars card in ``TICK_KEYS`` (the B7 cases, a
  13-digit maximum ``t184-13digit``, and the other Bars cards) at every B9 view (ruling
  195866ca; at 420x900, ``NO_TICK_VIEWS``, no tick labels at all is also valid).
  Tick nodes are read from the payload's own order (B4: label, ``String(value)``, unit per bar;
  everything else in the Bars component but its title is a tick). No two tick glyph boxes
  intersect, and the ticks drawn pass ``tick_problem`` (0 first, equally spaced by a step from
  ``NICE`` times a power of ten, at most five, the end within one step past the largest value,
  each ``String(n)``). A tick hidden by CSS but left in the DOM still counts as drawn (B4 reads the
  DOM). Red at f17952a2 on 1e12 and on the 13-digit maximum.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import re
import time
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.ui_gate import UiGate
from local_astra.contracts.protocol import AstraUiSurface
from local_astra.hud_static import HudStaticServer
from tests.fakes.hud_server import HudStub
from tests.integration.test_hud_card_parity import payload_surface
from tests.integration.test_hud_chrome import (
    FIND_THEME_CONTROL_JS,
    choose_theme_js,
    composite,
    contrast_ratio,
    parse_css_color,
    threshold_for,
)
from tests.integration.test_hud_mockup_parity import (
    _playwright_chromium,
    _png_size,
    delta_e,
    hex_rgb,
    hexc,
)

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
HUD = REPO / "hud"
SCREENS = REPO / "scripts" / "demo" / "hud_screens.py"
EXEMPLARS = REPO / "tests" / "fixtures" / "a2ui" / "cards_v2_exemplars.jsonl"
MOCK_DIR = REPO / "tests" / "fixtures" / "hud" / "genui_v2_20261028"
FIXTURES_MANIFEST = REPO / "tests" / "fixtures" / "hud" / "MANIFEST.md"
MOCK_FILES = ("Main.dc.html", "Compare.dc.html", "Chart.dc.html")

V2_ID = "gearvis:cards/v2"
GATE_CATALOG_ID = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
V1_KEY = "desk-t9"
VIEW = (1440, 900)
#: B9's matrix; the document must not scroll at the first three (>= 1100 px).
#: B9d (reviewer B round 2): 540x900 and 580x900 put a Bars between 300 and 400 px wide, where a
#: drift between the renderer's and the sheet's narrow threshold (340 px) shows.
LAYOUT_VIEWS = (
    (1280, 800),
    (1440, 900),
    (1920, 1080),
    (1000, 800),
    (420, 900),
    (540, 900),
    (580, 900),
)
STATES = ("system-dark", "system-light", "data-dark", "data-light")
LOOK_STATES = ("system-dark", "system-light")
PANES = 6
SETTLE_S = 0.6

#: B3 frame values (the approved Main board's inline styles, MANIFEST genui_v2_20261028).
FRAME_RADIUS = 20.0
FRAME_PAD = (22.0, 24.0)
DARK_TOP, DARK_BOTTOM = hex_rgb("#1b2030"), hex_rgb("#171a22")
DARK_BORDER = (255.0, 255.0, 255.0, 0.08)
FRAME_DE = 3.0
TOKEN_DE = 2.0
BORDER_MIN_DE = 1.0
TINT_MIN_DE = 1.0
#: "the same backdrop": means over a text's line box (single pixels differ by ~1 unit under the
#: gradient's dithering, dE00 up to ~0.8)
SAME_DE = 0.8
LEN = 0.5

#: Cards whose every text and fill B8 measures (each fits the 900 px view): every v2 kind, the
#: three boards and the v1 card (H-T184-5).
CONTRAST_KEYS = (
    "board-main",
    "board-compare",
    "board-chart",
    "steps-omelette",
    "bars-titled",
    "stattiles-percent-bounds",
    "comparison-laptops",
    "actions-three",
    "source-only",
    "stat-minimal",
    "t184-tie",
    "t184-cells",
    "comparison-highlight-out-of-range",
    V1_KEY,
)

LONG_WORD = "W" * 10_000


def _exemplars() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in EXEMPLARS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row["key"])] = row
    return out


def _v2(sid: str, components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"version": "v0.9.1", "createSurface": {"surfaceId": sid, "catalogId": V2_ID}},
        {"version": "v0.9.1", "updateComponents": {"surfaceId": sid, "components": components}},
    ]


def _every_role(s: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "root",
            "component": "Column",
            "children": ["st", "tl", "cmp", "br", "sp", "src", "act"],
        },
        {"id": "st", "component": "Stat", "label": s, "value": s, "detail": s},
        {"id": "tl", "component": "StatTiles", "tiles": [{"title": s, "value": s, "secondary": s}]},
        {
            "id": "cmp",
            "component": "Comparison",
            "columns": [{"title": s, "badge": s}, {"title": s}],
            "rows": [{"label": s, "cells": [s, {"text": s, "tone": "warn"}]}],
        },
        {
            "id": "br",
            "component": "Bars",
            "title": s,
            "unit": s,
            "bars": [{"label": s, "value": 1}],
        },
        {"id": "sp", "component": "Steps", "steps": [{"title": s, "detail": s}]},
        {"id": "src", "component": "Source", "text": s},
        {"id": "act", "component": "Actions", "labels": [s]},
    ]


#: This module's synthetic payloads (labelled: written here, not model output).
SYNTHETIC: dict[str, list[dict[str, Any]]] = {
    "t184-tie": [
        {
            "id": "root",
            "component": "Bars",
            "unit": "pts",
            "bars": [
                {"label": "First", "value": 5},
                {"label": "Second", "value": 9},
                {"label": "Third", "value": 9},
                {"label": "Fourth", "value": 2},
            ],
        }
    ],
    "t184-cells": [
        {
            "id": "root",
            "component": "Comparison",
            "columns": [{"title": "Col A"}, {"title": "Col B"}],
            "rows": [
                {"label": "Row one", "cells": ["a1", "b1", "x1-extra"]},
                {"label": "Row two", "cells": ["a2"]},
            ],
        }
    ],
    "t184-bars12": [
        {
            "id": "root",
            "component": "Bars",
            "title": "Twelve long labels",
            "unit": "units",
            "bars": [
                {"label": f"Label {k:02d} " + "long " * 10 + "end", "value": (k + 1) * 37}
                for k in range(12)
            ],
        }
    ],
    "t184-cmp5": [
        {
            "id": "root",
            "component": "Comparison",
            "columns": [{"title": f"Option number {k}", "badge": "Pick"} for k in range(5)],
            "highlight": 2,
            "rows": [
                {"label": "Price per month", "cells": [f"{k * 7} EUR" for k in range(5)]},
                {
                    "label": "Notes",
                    "cells": [{"text": "Good fit", "tone": "ok"}] * 3 + ["None", "Unknown"],
                },
            ],
        }
    ],
    "t184-long": _every_role(LONG_WORD),
    # B7a: a 13-digit maximum (and a tiny bar beside it)
    "t184-13digit": [
        {
            "id": "root",
            "component": "Bars",
            "title": "Thirteen digits",
            "unit": "units",
            "bars": [
                {"label": "Huge", "value": 4321098765432},
                {"label": "Tiny", "value": 7},
            ],
        }
    ],
}

#: B5a: the newest card, pushed last (older cards are ``inert``, so only the newest is reachable by
#: Tab): a 5-column Comparison long enough to scroll inside a card at 1440 px, then two Actions.
TAB_SID = "turn-t100"
TAB_LABELS = ["Tab stop one", "Tab stop two"]
TAB_COMPONENTS: list[dict[str, Any]] = [
    {"id": "root", "component": "Column", "children": ["cmp", "act"]},
    {
        "id": "cmp",
        "component": "Comparison",
        "columns": [
            {"title": f"Option number {k} with a long column title", "badge": "Pick"}
            for k in range(5)
        ],
        "highlight": 1,
        "rows": [
            {"label": "Price per month", "cells": [f"{k * 7} EUR a month" for k in range(5)]},
            {
                "label": "Notes",
                "cells": [{"text": "Good fit", "tone": "ok"}] * 3
                + [{"text": "Slow", "tone": "warn"}, "Unknown"],
            },
        ],
    },
    {"id": "act", "component": "Actions", "labels": TAB_LABELS},
]
#: Tab presses at most (the HUD's own controls come first; the walk stops when focus wraps).
TAB_MAX = 250

#: B7a: the Bars cards whose tick labels are measured at every B9 view. The B7 cases (all bars 0,
#: a single bar, 1e12) and the 13-digit maximum, plus every other Bars card pushed here (the boards,
#: a tie, 12 long labels), which the criterion does not name but which must not overlap either.
TICK_KEYS = (
    "bars-all-zero",
    "bars-single",
    "bars-huge",
    "t184-13digit",
    "bars-titled",
    "board-chart",
    "t184-tie",
    "t184-bars12",
)
#: The closed list of tick steps (the mantissa of a "nice" step), as tests/hud/ui_cards_v2.test.mjs
#: reads B4 (ii): one significant digit, or 2.5, times a power of ten.
NICE = (1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
#: ``String(n)`` of a finite number in [0, 1e21): plain digits, an optional fraction without a
#: trailing zero; no sign, separator, exponent, suffix or space.
JS_NUMBER = re.compile(r"(0|[1-9][0-9]*)(\.[0-9]*[1-9])?")


def surfaces() -> dict[str, AstraUiSurface]:
    """key -> the surface the HUD gets (exemplars, the v1 card, the synthetic payloads)."""
    out: dict[str, AstraUiSurface] = {}
    for key, row in _exemplars().items():
        messages = row["messages"]
        sid = next(m["createSurface"]["surfaceId"] for m in messages if "createSurface" in m)
        out[key] = AstraUiSurface(
            surface_id=sid, canvas_idx=0, final=True, messages=tuple(messages)
        )
    out[V1_KEY] = payload_surface(V1_KEY)
    for key, comps in SYNTHETIC.items():
        out[key] = AstraUiSurface(
            surface_id=key, canvas_idx=0, final=True, messages=tuple(_v2(key, comps))
        )
    return out


# --- page scripts ---------------------------------------------------------------------------

NEUTRALISE_CSS = (
    "*, *::before, *::after { transition-property: none !important; "
    "transition-duration: 0s !important; animation: none !important; }"
)
HIDE_TEXT_CSS = (
    "[data-a2ui-root], [data-a2ui-root] * { color: transparent !important; "
    "-webkit-text-fill-color: transparent !important; text-shadow: none !important; "
    "outline-color: transparent !important; caret-color: transparent !important; }"
)
TWO_FRAMES_JS = (
    "new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))"
)

#: Everything one card shows, in viewport coordinates. ``arg = [surfaceId, tileTitles]``.
DUMP_JS = r"""
([sid, tileTitles]) => {
  const w = document.querySelector('[data-ui-surface="' + sid + '"]');
  const root = w ? w.querySelector('[data-a2ui-root]') : null;
  if (!root) return null;
  const px = (v) => parseFloat(v) || 0;
  const cs = (el, p) => getComputedStyle(el, p);
  const R = (el) => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
  const alpha = (c) => { const m = (c || '').match(/[\d.]+/g); return !m ? 0 : m.length >= 4 ? parseFloat(m[3]) : 1; };
  const opacity = (el) => { let o = 1; for (let n = el; n && n.nodeType === 1; n = n.parentElement) o *= parseFloat(cs(n).opacity); return o; };
  const kindOf = (el) => { const k = el.closest('[data-a2-kind]'); return k && root.contains(k) ? [k.getAttribute('data-a2-kind'), k.getAttribute('data-a2-id')] : [null, null]; };
  const texts = [];
  const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let n = tw.nextNode(); n; n = tw.nextNode()) {
    if (!n.data.trim()) continue;
    const el = n.parentElement, s = cs(el);
    const rg = document.createRange(); rg.selectNodeContents(n);
    const g = rg.getBoundingClientRect();
    const lines = [...rg.getClientRects()].slice(0, 6).map((r) => [r.left, r.top, r.right, r.bottom]);
    const [kind, cid] = kindOf(el);
    texts.push({text: n.data.length > 120 ? n.data.slice(0, 120) : n.data, len: n.data.length,
      el: R(el), glyph: [g.left, g.top, g.right, g.bottom], lines, color: s.color, fs: px(s.fontSize),
      fw: parseInt(s.fontWeight, 10) || 400, ls: s.letterSpacing, tt: s.textTransform,
      fvn: s.fontVariantNumeric, opacity: opacity(el), kind, cid, radius: px(s.borderTopLeftRadius),
      bg: s.backgroundColor});
  }
  const all = [...root.querySelectorAll('*')];
  const paintedLeaf = (el) => {
    if (!(el instanceof HTMLElement) || el.textContent.trim()) return false;
    const s = cs(el);
    return alpha(s.backgroundColor) > 0 || s.backgroundImage !== 'none';
  };
  const painted = all.filter(paintedLeaf);
  const fills = painted.filter((el) => !painted.some((o) => o !== el && el.contains(o))).map((el) => {
    const s = cs(el); const [kind, cid] = kindOf(el);
    return {rect: R(el), bg: s.backgroundColor, bgimg: s.backgroundImage, kind, cid};
  });
  const buttons = [...root.querySelectorAll('button')].map((b) => {
    const s = cs(b); const [kind, cid] = kindOf(b);
    return {text: b.textContent, rect: R(b), bg: s.backgroundColor, color: s.color, bw: px(s.borderTopWidth),
      bstyle: s.borderTopStyle, bcolor: s.borderTopColor, radius: px(s.borderTopLeftRadius), kind, cid,
      type: b.getAttribute('type')};
  });
  const svgs = [...root.querySelectorAll('svg')].map((g) => { const [kind, cid] = kindOf(g); return {rect: R(g), kind, cid}; });
  const kinds = [...root.querySelectorAll('[data-a2-kind]')].map((k) => ({kind: k.getAttribute('data-a2-kind'), cid: k.getAttribute('data-a2-id'), rect: R(k)}));
  const pseudo = [];
  for (const el of [root, ...all]) for (const which of ['::before', '::after']) {
    const c = cs(el, which).content;
    if (c && c !== 'none' && c !== 'normal') pseudo.push([el.tagName.toLowerCase(), which, c]);
  }
  const markers = all.filter((el) => cs(el).display === 'list-item' && cs(el).listStyleType !== 'none').map((el) => el.tagName.toLowerCase());
  const tiles = [];
  for (const k of root.querySelectorAll('[data-a2-kind="StatTiles"]')) {
    const titleEls = (tileTitles || []).map((t) => [...k.querySelectorAll('*')].find((e) => [...e.childNodes].some((c) => c.nodeType === 3 && c.data === t))).filter(Boolean);
    for (const t of titleEls) {
      let box = t;
      while (box.parentElement && box.parentElement !== k && !titleEls.some((o) => o !== t && box.parentElement.contains(o))) box = box.parentElement;
      tiles.push({title: t.textContent, rect: R(box), cid: k.getAttribute('data-a2-id')});
    }
  }
  const s = cs(root);
  const frame = {rect: R(root), radius: [s.borderTopLeftRadius, s.borderTopRightRadius, s.borderBottomRightRadius, s.borderBottomLeftRadius].map(px),
    bw: [s.borderTopWidth, s.borderRightWidth, s.borderBottomWidth, s.borderLeftWidth].map(px), bstyle: s.borderTopStyle,
    bcolor: s.borderTopColor, pad: [s.paddingTop, s.paddingRight, s.paddingBottom, s.paddingLeft].map(px),
    shadow: s.boxShadow, bgimg: s.backgroundImage, bg: s.backgroundColor};
  return {texts, fills, buttons, svgs, kinds, pseudo, markers, tiles, frame,
          theme: document.documentElement.getAttribute('data-theme')};
}
"""

#: The page's tokens, resolved through probes (colours as rgb, the shadow and the spark as computed).
TOKENS_JS = r"""
() => {
  const out = {};
  const probe = (prop, value) => {
    const p = document.createElement('div');
    p.style.cssText = 'position:fixed;left:-60px;top:0;width:4px;height:4px;' + prop + ':' + value;
    document.body.appendChild(p);
    const s = getComputedStyle(p);
    const v = prop === 'box-shadow' ? s.boxShadow : prop === 'background-image' ? s.backgroundImage : s.backgroundColor;
    p.remove();
    return v;
  };
  for (const t of ['--muted', '--tonal', '--tonal-ink', '--ok', '--ok-soft', '--warn', '--warn-soft', '--accent', '--ink'])
    out[t] = probe('background-color', 'var(' + t + ')');
  out['--card-shadow'] = probe('box-shadow', 'var(--card-shadow)');
  out['--spark'] = probe('background-image', 'var(--spark)');
  return out;
}
"""

SCROLL_JS = r"""
(sid) => {
  const w = document.querySelector('[data-ui-surface="' + sid + '"]');
  const root = w ? w.querySelector('[data-a2ui-root]') : null;
  if (!root) return null;
  const tall = root.getBoundingClientRect().height > innerHeight - 24;
  root.scrollIntoView({block: tall ? 'start' : 'center', inline: 'nearest', behavior: 'instant'});
  const r = root.getBoundingClientRect();
  return [r.left, r.top, r.right, r.bottom];
}
"""

#: B9: every card against its slot; who overflows the card and whether a Comparison scroller holds it.
LAYOUT_JS = r"""
(sids) => {
  const out = {cards: {}, doc: null};
  const R = (el) => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
  for (const [key, sid] of Object.entries(sids)) {
    const w = document.querySelector('[data-ui-surface="' + sid + '"]');
    const root = w ? w.querySelector('[data-a2ui-root]') : null;
    if (!root) { out.cards[key] = null; continue; }
    const slot = root.closest('[data-card-slot]');
    const rr = root.getBoundingClientRect();
    const scrolls = (el) => ['auto', 'scroll'].includes(getComputedStyle(el).overflowX) && el.scrollWidth > el.clientWidth + 1;
    const bad = [], scrollers = [];
    for (const el of root.querySelectorAll('*')) {
      if (scrolls(el)) scrollers.push({inCmp: !!el.closest('[data-a2-kind="Comparison"]'), rect: R(el)});
      const r = el.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) continue;
      if (r.right <= rr.right + 0.5 && r.left >= rr.left - 0.5) continue;
      let s = el.parentElement, held = false;
      while (s && s !== root) { if (scrolls(s)) { const sr = s.getBoundingClientRect(); held = sr.left >= rr.left - 0.5 && sr.right <= rr.right + 0.5 && !!s.closest('[data-a2-kind="Comparison"]'); break; } s = s.parentElement; }
      if (!held) bad.push({tag: el.tagName.toLowerCase(), kind: (el.closest('[data-a2-kind]') || {getAttribute: () => null}).getAttribute('data-a2-kind'), rect: R(el), card: R(root)});
    }
    const tiles = {};
    for (const k of root.querySelectorAll('[data-a2-kind="StatTiles"]')) {
      const titles = [...k.querySelectorAll('*')].filter((e) => [...e.childNodes].some((c) => c.nodeType === 3 && /^Q[1-4]$/.test(c.data)));
      tiles[k.getAttribute('data-a2-id')] = titles.map((t) => { let b = t; while (b.parentElement && b.parentElement !== k && !titles.some((o) => o !== t && b.parentElement.contains(o))) b = b.parentElement; return R(b); });
    }
    out.cards[key] = {root: R(root), slot: slot ? R(slot) : null, scroll: [root.scrollWidth, root.clientWidth],
                      bad: bad.slice(0, 5), nbad: bad.length, scrollers, tiles};
  }
  const se = document.scrollingElement;
  out.doc = {scroll: [se.scrollWidth, se.scrollHeight], view: [innerWidth, innerHeight]};
  return out;
}
"""

#: B7a: the text nodes of each card's one Bars component, in document order, with their glyph
#: boxes (one box per client rect). ``arg = {key: surfaceId}``.
TICKS_JS = r"""
(sids) => {
  const out = {};
  for (const [key, sid] of Object.entries(sids)) {
    const w = document.querySelector('[data-ui-surface="' + sid + '"]');
    const root = w ? w.querySelector('[data-a2ui-root]') : null;
    if (!root) { out[key] = null; continue; }
    const bars = [...root.querySelectorAll('[data-a2-kind="Bars"]')];
    if (bars.length !== 1) { out[key] = {bars: bars.length, texts: []}; continue; }
    const texts = [];
    const tw = document.createTreeWalker(bars[0], NodeFilter.SHOW_TEXT);
    for (let n = tw.nextNode(); n; n = tw.nextNode()) {
      if (!n.data.trim()) continue;
      const rg = document.createRange(); rg.selectNodeContents(n);
      const g = rg.getBoundingClientRect();
      texts.push({text: n.data, box: [g.left, g.top, g.right, g.bottom],
                  rects: [...rg.getClientRects()].map((r) => [r.left, r.top, r.right, r.bottom])});
    }
    out[key] = {bars: 1, texts};
  }
  return out;
}
"""

#: B5a: what has focus after a Tab press, described without class names.
ACTIVE_JS = r"""
() => {
  const a = document.activeElement;
  if (!a || a === document.body || a === document.documentElement) return {path: 'body', inRoot: false};
  const path = [];
  for (let n = a; n && n.nodeType === 1; n = n.parentElement) {
    const p = n.parentElement;
    path.push(n.tagName.toLowerCase() + (p ? ':' + [...p.children].indexOf(n) : ''));
  }
  const root = a.closest('[data-a2ui-root]');
  const k = a.closest('[data-a2-kind]');
  const s = a.closest('[data-ui-surface]');
  return {path: path.join('<'), inRoot: !!root, tag: a.tagName.toLowerCase(),
          kind: k && root && root.contains(k) ? k.getAttribute('data-a2-kind') : null,
          surface: s ? s.getAttribute('data-ui-surface') : null,
          text: (a.textContent || '').trim().slice(0, 60), tabindex: a.getAttribute('tabindex'),
          scrolls: a.scrollWidth > a.clientWidth + 1 && ['auto', 'scroll'].includes(getComputedStyle(a).overflowX)};
}
"""

#: B5a premise: the Comparison of the card scrolls sideways inside it.
CMP_SCROLLERS_JS = r"""
(sid) => {
  const w = document.querySelector('[data-ui-surface="' + sid + '"]');
  const cmp = w ? w.querySelector('[data-a2ui-root] [data-a2-kind="Comparison"]') : null;
  if (!cmp) return null;
  return [cmp, ...cmp.querySelectorAll('*')].filter((el) =>
    ['auto', 'scroll'].includes(getComputedStyle(el).overflowX) && el.scrollWidth > el.clientWidth + 1).length;
}
"""

#: Decodes a PNG (base64) in a blank page and returns the RGB at each [x, y] (image coordinates).
SAMPLE_JS = r"""
async ([b64, points]) => {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const bmp = await createImageBitmap(new Blob([bytes], {type: 'image/png'}));
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const g = c.getContext('2d', {colorSpace: 'srgb'});
  g.drawImage(bmp, 0, 0);
  const data = g.getImageData(0, 0, bmp.width, bmp.height).data;
  return points.map(([x, y]) => {
    const xi = Math.max(0, Math.min(bmp.width - 1, Math.floor(x))), yi = Math.max(0, Math.min(bmp.height - 1, Math.floor(y)));
    const i = (yi * bmp.width + xi) * 4;
    return [data[i], data[i + 1], data[i + 2]];
  });
}
"""

FONTS_JS = r"""
() => {
  let faces = 0, unreadable = 0;
  for (const sheet of document.styleSheets) {
    let rules;
    try { rules = sheet.cssRules; } catch (e) { unreadable += 1; continue; }
    const walk = (rs) => { for (const r of rs) { if (r.constructor.name === 'CSSFontFaceRule') faces += 1; if (r.cssRules) walk(r.cssRules); } };
    walk(rules);
  }
  return {faces, unreadable, fontset: document.fonts.size};
}
"""


# --- the session ----------------------------------------------------------------------------


@dataclass
class Shot:
    """One card in one theme state: the dump, and the hidden-text pixel at each sampled point."""

    dump: dict[str, Any]
    pixels: dict[tuple[int, int], tuple[float, float, float]] = field(default_factory=dict)


@dataclass
class Run:
    error: str = ""
    sids: dict[str, str] = field(default_factory=dict)
    rendered: dict[str, bool] = field(default_factory=dict)
    tokens: dict[str, dict[str, str]] = field(default_factory=dict)
    shots: dict[str, dict[str, Shot]] = field(default_factory=dict)
    mutant: dict[str, Any] = field(default_factory=dict)
    layout: dict[str, Any] = field(default_factory=dict)
    focus: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)
    stray_actions: int = -1
    requests: list[str] = field(default_factory=list)
    fonts: dict[str, Any] = field(default_factory=dict)
    #: B7a: view -> key -> the Bars text nodes (TICKS_JS)
    ticks: dict[str, Any] = field(default_factory=dict)
    #: B5a: the Tab walk over the newest card
    tab: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0


def _grid(rect: list[float], nx: int = 5, ny: int = 3) -> list[tuple[int, int]]:
    left, top, right, bottom = rect
    if right - left < 1 or bottom - top < 1:
        return []
    return [
        (int(left + (right - left) * (i + 0.5) / nx), int(top + (bottom - top) * (j + 0.5) / ny))
        for i in range(nx)
        for j in range(ny)
    ]


def _points(dump: dict[str, Any]) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for t in dump["texts"]:
        pts += _grid(t["glyph"])
        for line in t["lines"]:
            pts += _grid(line)
    for f in dump["fills"]:
        (left, top, right, bottom) = f["rect"]
        mid = int((top + bottom) / 2)
        pts += [(int((left + right) / 2), mid), (int(right + 2), mid)]
    for b in dump["buttons"]:
        (left, top, right, bottom) = b["rect"]
        mid = int((top + bottom) / 2)
        pts += [(int(left - 3), mid), (int(left - 8), mid), (int(left + 2), mid)]
        pts.append((int((left + right) / 2), int(top + 3)))
    for tile in dump["tiles"]:
        (left, top, _r, _b) = tile["rect"]
        pts += _grid([left + 4, top + 4, left + 16, top + 12])
    fr = dump["frame"]["rect"]
    cx = int((fr[0] + fr[2]) / 2)
    pts += [(cx, int(fr[1] + 5)), (cx, int(fr[3] - 5)), (int(fr[0] - 4), int((fr[1] + fr[3]) / 2))]
    pts += [(int(fr[0] + 8), int(fr[1] + (fr[3] - fr[1]) * k / 10)) for k in range(1, 10)]
    return pts


def _sample(sampler: Any, png: bytes, origin: tuple[float, float], pts: list[tuple[int, int]]):
    rel = [[x - origin[0], y - origin[1]] for x, y in pts]
    got = sampler.evaluate(SAMPLE_JS, [base64.b64encode(png).decode(), rel])
    return {p: (float(c[0]), float(c[1]), float(c[2])) for p, c in zip(pts, got, strict=True)}


def _theme(page: Any, state: str) -> None:
    scheme, pref = {
        "system-dark": ("dark", "system"),
        "system-light": ("light", "system"),
        "data-dark": ("light", "dark"),
        "data-light": ("dark", "light"),
    }[state]
    page.emulate_media(color_scheme=scheme, reduced_motion="reduce")
    page.evaluate(choose_theme_js(pref))
    page.evaluate(TWO_FRAMES_JS)


def _shoot(page: Any, sampler: Any, sid: str, tiles: list[str]) -> Shot | None:
    rect = page.evaluate(SCROLL_JS, sid)
    if rect is None:
        return None
    page.evaluate(TWO_FRAMES_JS)
    dump = page.evaluate(DUMP_JS, [sid, tiles])
    if dump is None:
        return None
    fr = dump["frame"]["rect"]
    x0, y0 = max(0.0, fr[0] - 12), max(0.0, fr[1] - 12)
    x1, y1 = min(float(VIEW[0]), fr[2] + 12), min(float(VIEW[1]), fr[3] + 12)
    style = page.add_style_tag(content=HIDE_TEXT_CSS)
    page.evaluate(TWO_FRAMES_JS)
    png = page.screenshot(clip={"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0})
    style.evaluate("(el) => el.remove()")
    page.evaluate(TWO_FRAMES_JS)
    pts = [(x, y) for x, y in _points(dump) if x0 <= x < x1 and y0 <= y < y1]
    return Shot(dump=dump, pixels=_sample(sampler, png, (x0, y0), pts))


def _tile_titles(surface: AstraUiSurface) -> list[str]:
    out: list[str] = []
    for m in surface.messages:
        for c in (m.get("updateComponents") or {}).get("components") or []:
            if c.get("component") == "StatTiles":
                out += [str(t.get("title")) for t in c.get("tiles") or []]
    return out


def _actions(stub: HudStub) -> list[dict[str, Any]]:
    out = []
    for raw in stub.snapshot():
        with contextlib.suppress(ValueError):
            obj = json.loads(raw)
            if isinstance(obj, dict) and obj.get("type") == "astra.ui.action":
                out.append(obj)
    return out


def _gate_made_button(label: str) -> AstraUiSurface:
    """The gate's own Button (RB repair): a basic Button with ``text`` and no action, gated."""
    raw = "\n".join(
        json.dumps(m)
        for m in (
            {
                "version": "v0.9.1",
                "createSurface": {"surfaceId": "s", "catalogId": GATE_CATALOG_ID},
            },
            {
                "version": "v0.9.1",
                "updateComponents": {
                    "surfaceId": "s",
                    "components": [
                        {"id": "root", "component": "Column", "children": ["btn"]},
                        {"id": "btn", "component": "Button", "text": label},
                    ],
                },
            },
        )
    )
    result = UiGate().gate(raw, turn_id="t99", canvas_idx=0, final=True)
    assert result.surface is not None, result.report
    return result.surface


CLICK_LABELS = ["Show the deluxe edition", "Compare prices"]


def _run(run: Run, p: Any) -> None:
    all_surfaces = surfaces()
    run.sids = {k: s.surface_id for k, s in all_surfaces.items()}
    browser = p.chromium.launch()
    stub = HudStub([]).start()
    server = HudStaticServer(HUD, 0)
    try:
        context = browser.new_context(
            viewport={"width": VIEW[0], "height": VIEW[1]},
            device_scale_factor=1,
            color_scheme="dark",
            reduced_motion="reduce",
        )
        page = context.new_page()
        page.on("request", lambda r: run.requests.append(r.url))
        page.on("websocket", lambda ws: run.requests.append(ws.url))
        page.goto(
            f"http://127.0.0.1:{server.port}/?source=canvas&ws="
            + urllib.parse.quote(stub.url, safe="")
        )
        if not stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20):
            raise RuntimeError("the page never said astra.hello")
        stub.wait_for_client(10)
        page.wait_for_function(
            f"document.querySelectorAll('main > .pane').length >= {PANES}", timeout=25_000
        )
        page.add_style_tag(content=NEUTRALISE_CSS)
        page.mouse.move(0, 0)
        for surface in all_surfaces.values():
            stub.push(surface)
            time.sleep(0.02)
        time.sleep(SETTLE_S)
        for key, sid in run.sids.items():
            run.rendered[key] = bool(
                page.evaluate(
                    "(sid) => !!document.querySelector('[data-ui-surface=\"' + sid + '\"] "
                    "[data-a2ui-root] [data-a2-id]')",
                    sid,
                )
            )
        sampler = context.new_page()  # about:blank: decodes shots, requests nothing
        for state in STATES:
            _theme(page, state)
            run.tokens[state] = page.evaluate(TOKENS_JS)
            run.shots[state] = {}
            for key in CONTRAST_KEYS:
                shot = _shoot(page, sampler, run.sids[key], _tile_titles(all_surfaces[key]))
                if shot is not None:
                    run.shots[state][key] = shot
        # B8's mutant arm (system-dark): each muted bar painted in the card colour measured right
        # next to it (the pixel the fill row reads it against).
        _theme(page, "system-dark")
        sid = run.sids["board-chart"]
        real = run.shots["system-dark"].get("board-chart")
        paint = []
        if real is not None:
            for f in _fills(real, "Bars")[1:]:
                (_l, top, right, bottom) = f["rect"]
                px = real.pixels.get((int(right + 2), int((top + bottom) / 2)))
                paint.append(None if px is None else "rgb({:.0f}, {:.0f}, {:.0f})".format(*px))
        page.evaluate(
            r"""([sid, paint]) => {
              const root = document.querySelector('[data-ui-surface="' + sid + '"] [data-a2ui-root]');
              if (!root) return;
              const alpha = (c) => { const m = (c || '').match(/[\d.]+/g); return !m ? 0 : m.length >= 4 ? +m[3] : 1; };
              const painted = [...root.querySelectorAll('[data-a2-kind="Bars"] *')].filter((el) => !el.textContent.trim() && (alpha(getComputedStyle(el).backgroundColor) > 0 || getComputedStyle(el).backgroundImage !== 'none'));
              const leaves = painted.filter((el) => !painted.some((o) => o !== el && el.contains(o)));
              leaves.slice(1).forEach((el, k) => { if (paint[k]) el.style.setProperty('background', paint[k], 'important'); });
            }""",
            [sid, paint],
        )
        mshot = _shoot(page, sampler, sid, [])
        run.mutant = {"shot": mshot}
        page.reload()
        stub.wait_for_client(10)
        page.wait_for_function(
            f"document.querySelectorAll('main > .pane').length >= {PANES}", timeout=25_000
        )
        page.add_style_tag(content=NEUTRALISE_CSS)
        page.mouse.move(0, 0)
        # after a reload the HUD replays nothing of ours: push every card again (fresh ids)
        for surface in all_surfaces.values():
            stub.push(surface.model_copy(update={"canvas_idx": 0}))
            time.sleep(0.02)
        time.sleep(SETTLE_S)
        _theme(page, "system-dark")
        for w, h in LAYOUT_VIEWS:
            page.set_viewport_size({"width": w, "height": h})
            page.evaluate(TWO_FRAMES_JS)
            time.sleep(0.15)
            run.layout[f"{w}x{h}"] = page.evaluate(LAYOUT_JS, run.sids)
            run.ticks[f"{w}x{h}"] = page.evaluate(TICKS_JS, {k: run.sids[k] for k in TICK_KEYS})
        page.set_viewport_size({"width": VIEW[0], "height": VIEW[1]})
        page.evaluate(TWO_FRAMES_JS)
        # B5 and the focus ring, on the newest card
        stub.push(
            AstraUiSurface(
                surface_id="turn-t98",
                canvas_idx=0,
                final=True,
                messages=tuple(
                    _v2(
                        "turn-t98",
                        [
                            {"id": "root", "component": "Column", "children": ["btn"]},
                            {"id": "btn", "component": "Actions", "labels": CLICK_LABELS},
                        ],
                    )
                ),
            )
        )
        page.wait_for_function(
            "!!document.querySelector('[data-ui-surface=\"turn-t98\"] button')", timeout=10_000
        )
        page.evaluate(SCROLL_JS, "turn-t98")
        page.evaluate(TWO_FRAMES_JS)
        before = _shoot(page, sampler, "turn-t98", [])
        page.evaluate(
            "() => { const b = document.querySelector('[data-ui-surface=\"turn-t98\"] button');"
            " b.focus({focusVisible: true}); }"
        )
        page.evaluate(TWO_FRAMES_JS)
        run.focus = {
            "before": before,
            "ring": page.evaluate(
                "() => { const b = document.activeElement; const s = getComputedStyle(b);"
                " const r = b.getBoundingClientRect();"
                " return {inCard: !!b.closest('[data-ui-surface=\"turn-t98\"]'),"
                " style: s.outlineStyle, width: parseFloat(s.outlineWidth) || 0,"
                " offset: parseFloat(s.outlineOffset) || 0, color: s.outlineColor,"
                " rect: [r.left, r.top, r.right, r.bottom]}; }"
            ),
        }
        page.evaluate("() => document.activeElement && document.activeElement.blur()")
        n0 = len(_actions(stub))
        # a click on the card body, its text and the frame sends nothing
        page.evaluate(SCROLL_JS, "turn-t98")
        box = page.locator('[data-ui-surface="turn-t98"] [data-a2ui-root]').bounding_box()
        assert box is not None
        page.mouse.click(box["x"] + 6, box["y"] + 6)
        page.keyboard.press("Escape")
        time.sleep(0.4)
        run.stray_actions = len(_actions(stub)) - n0
        page.get_by_role("button", name=CLICK_LABELS[1], exact=True).last.click()
        stub.wait_for(lambda s: len(_actions(s)) >= n0 + 1, 10)
        stub.push(_gate_made_button(CLICK_LABELS[1]))
        page.wait_for_function(
            "!!document.querySelector('[data-ui-surface=\"turn-t99\"] button')", timeout=10_000
        )
        page.locator('[data-ui-surface="turn-t99"] button').click()
        stub.wait_for(lambda s: len(_actions(s)) >= n0 + 2, 10)
        time.sleep(0.3)
        run.actions = _actions(stub)[n0:]
        # B5a: the newest card holds a scrolling 5-column Comparison; walk the page with Tab
        stub.push(
            AstraUiSurface(
                surface_id=TAB_SID,
                canvas_idx=0,
                final=True,
                messages=tuple(_v2(TAB_SID, TAB_COMPONENTS)),
            )
        )
        page.wait_for_function(
            f"!!document.querySelector('[data-ui-surface=\"{TAB_SID}\"] button')", timeout=10_000
        )
        page.evaluate(SCROLL_JS, TAB_SID)
        page.evaluate(TWO_FRAMES_JS)
        sent0 = len(_actions(stub))
        page.evaluate("() => document.activeElement && document.activeElement.blur()")
        stops: list[dict[str, Any]] = []
        for _ in range(TAB_MAX):
            page.keyboard.press("Tab")
            stop = page.evaluate(ACTIVE_JS)
            if stops and stop["path"] != "body" and stop["path"] == stops[0]["path"]:
                break
            stops.append(stop)
        time.sleep(0.2)
        run.tab = {
            "scrollers": page.evaluate(CMP_SCROLLERS_JS, TAB_SID),
            "stops": stops,
            "sent": len(_actions(stub)) - sent0,
        }
        run.fonts = page.evaluate(FONTS_JS)
        context.close()
    finally:
        browser.close()
        server.close()
        stub.stop()


@pytest.fixture(scope="module")
def genui() -> Iterator[Run]:
    """Every browser measurement, in one Chromium closed again before any row runs. Never raises."""
    why = _playwright_chromium()
    if why:
        pytest.skip(why)
    from playwright.sync_api import sync_playwright

    run = Run()
    started = time.monotonic()
    try:
        with sync_playwright() as p:
            _run(run, p)
    except Exception as exc:  # a broken session fails in a row, not here
        import traceback

        where = traceback.extract_tb(exc.__traceback__)
        inside = [f for f in where if f.filename == __file__]
        line = f" (at line {inside[-1].lineno}: {inside[-1].line})" if inside else ""
        run.error = f"{type(exc).__name__}: {exc}{line}"
    run.seconds = time.monotonic() - started
    yield run


def _ok(run: Run) -> None:
    assert not run.error, f"the T-0184 B browser session failed before any assertion: {run.error}"
    missing = sorted(k for k, ok in run.rendered.items() if not ok)
    assert not missing, f"cards that never rendered in the HUD: {missing}"


# --- colour helpers ----------------------------------------------------------------------------


def _parse(css: str) -> tuple[float, float, float, float]:
    """``parse_css_color``, plus the ``color(srgb r g b / a)`` form Chrome computes for a
    ``color-mix(in srgb, …)`` (implementation note 6), whose channels are 0-1."""
    m = re.fullmatch(
        r"\s*color\(srgb\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)(?:\s*/\s*([\d.e]+))?\)\s*",
        css or "",
    )
    if m:
        r, g, b = (max(0.0, min(1.0, float(m.group(i)))) * 255.0 for i in (1, 2, 3))
        return (r, g, b, float(m.group(4)) if m.group(4) is not None else 1.0)
    assert not (css or "").strip().startswith(("color(", "oklab(", "oklch(", "lab(", "lch(")), (
        f"a computed colour this module cannot read: {css!r}"
    )
    return parse_css_color(css)


def _rgb(css: str) -> tuple[float, float, float]:
    r, g, b, _a = _parse(css)
    return (r, g, b)


def _over(css: str, behind: tuple[float, float, float], opacity: float = 1.0):
    r, g, b, a = _parse(css)
    return composite((r, g, b, a * opacity), behind)


def _text(shot: Shot, s: str) -> dict[str, Any]:
    hits = [t for t in shot.dump["texts"] if t["text"] == s]
    assert hits, (
        f"no text node {s!r} in the card (shown: {[t['text'] for t in shot.dump['texts']][:30]})"
    )
    return hits[0]


def _px(shot: Shot, x: float, y: float) -> tuple[float, float, float]:
    key = (int(x), int(y))
    assert key in shot.pixels, f"pixel {key} was not sampled"
    return shot.pixels[key]


def _mean(shot: Shot, rect: list[float]) -> tuple[float, float, float] | None:
    got = [shot.pixels[p] for p in _grid(rect) if p in shot.pixels]
    if not got:
        return None
    return (
        sum(c[0] for c in got) / len(got),
        sum(c[1] for c in got) / len(got),
        sum(c[2] for c in got) / len(got),
    )


def _behind(shot: Shot, s: str) -> tuple[float, float, float]:
    """The mean backdrop over the first line box of text ``s`` (text hidden for the shot)."""
    got = _mean(shot, _text(shot, s)["lines"][0])
    assert got is not None, f"the backdrop of {s!r} was not sampled"
    return got


def _fills(shot: Shot, kind: str) -> list[dict[str, Any]]:
    return [f for f in shot.dump["fills"] if f["kind"] == kind]


# =========================================================================================
# premise (not pending)
# =========================================================================================


def _manifest_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in FIXTURES_MANIFEST.read_text(encoding="utf-8").splitlines():
        m = re.search(r"genui_v2_20261028/([A-Za-z0-9_.-]+)`.*?\b([0-9a-f]{64})\b", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def test_premise_the_approved_v2_boards_are_committed_byte_for_byte_as_the_manifest_records():
    """H-T184-8: Main, Compare and Chart, sha256-equal to the MANIFEST rows; reference only, so
    nothing under ``hud/`` names the directory."""
    hashes = _manifest_hashes()
    assert sorted(hashes) == sorted(MOCK_FILES), hashes
    for name in MOCK_FILES:
        digest = hashlib.sha256((MOCK_DIR / name).read_bytes()).hexdigest()
        assert digest == hashes[name], f"{name} is {digest}, the MANIFEST says {hashes[name]}"
    strays = [
        str(p.relative_to(REPO))
        for p in HUD.rglob("*")
        if p.is_file() and "genui_v2_20261028" in p.read_text(errors="replace")
    ]
    assert strays == []


def test_premise_the_synthetic_payloads_validate_against_the_v2_catalog():
    """This module's synthetic payloads are well-formed v2 surfaces (group A's validator), so a
    rendering failure is the renderer's, not the payload's."""
    from local_astra.ui import a2ui_schema

    validator = a2ui_schema.A2UIValidator(catalog=a2ui_schema.load_cards_v2_catalog())
    for key, comps in [*SYNTHETIC.items(), (TAB_SID, TAB_COMPONENTS)]:
        report = validator.validate_messages(_v2(key, comps), final=True)
        assert report.ok, f"{key}: {report}"


# =========================================================================================
# B3: the look
# =========================================================================================


def test_b3_every_pushed_card_rendered(genui):
    """Premise of every row below: all exemplars, the v1 card and the synthetic payloads render."""
    _ok(genui)
    assert len(genui.rendered) == len(surfaces())


@pytest.mark.parametrize("theme", LOOK_STATES)
@pytest.mark.parametrize("key", ["board-main", "board-chart", V1_KEY])
def test_b3_card_frame(genui, key, theme):
    """B3 card frame, for a v2 card and the v1 card (H-T184-5: one frame for every card): radius
    20 px, a 1 px solid border, padding 22-24 px, ``--card-shadow``; in dark a vertical gradient
    dE00 <= 3 of #1b2030 -> #171a22 and a border dE00 <= 3 of white at 0.08 composited; in light
    the frame stands off the pane."""
    _ok(genui)
    shot = genui.shots[theme][key]
    fr = shot.dump["frame"]
    problems: list[str] = []
    if any(abs(r - FRAME_RADIUS) > LEN for r in fr["radius"]):
        problems.append(f"radius {fr['radius']}, want {FRAME_RADIUS}")
    if any(abs(w - 1.0) > 0.25 for w in fr["bw"]) or fr["bstyle"] != "solid":
        problems.append(f"border {fr['bw']} {fr['bstyle']}, want 1 px solid")
    if any(not (FRAME_PAD[0] - LEN <= p <= FRAME_PAD[1] + LEN) for p in fr["pad"]):
        problems.append(f"padding {fr['pad']}, want 22-24 px")
    if fr["shadow"] != genui.tokens[theme]["--card-shadow"]:
        problems.append(f"box-shadow {fr['shadow']!r} is not --card-shadow")
    (left, top, right, bottom) = fr["rect"]
    cx = (left + right) / 2
    top_px, bottom_px = _px(shot, cx, top + 5), _px(shot, cx, bottom - 5)
    pane = _px(shot, left - 4, (top + bottom) / 2)
    if theme == "system-dark":
        if delta_e(top_px, DARK_TOP) > FRAME_DE:
            problems.append(f"top {hexc(top_px)} vs #1b2030 dE00 {delta_e(top_px, DARK_TOP):.2f}")
        if delta_e(bottom_px, DARK_BOTTOM) > FRAME_DE:
            problems.append(
                f"bottom {hexc(bottom_px)} vs #171a22 dE00 {delta_e(bottom_px, DARK_BOTTOM):.2f}"
            )
        got = _over(fr["bcolor"], top_px)
        want = composite(DARK_BORDER, top_px)
        if delta_e(got, want) > FRAME_DE:
            problems.append(f"border {fr['bcolor']} vs white/0.08, dE00 {delta_e(got, want):.2f}")
    else:
        edge = _over(fr["bcolor"], top_px)
        if max(delta_e(top_px, pane), delta_e(edge, pane)) < BORDER_MIN_DE:
            problems.append(f"the frame {hexc(top_px)} does not stand off the pane {hexc(pane)}")
    assert problems == [], f"T-0184 B3 frame ({key}, {theme}):\n  " + "\n  ".join(problems)


def _token_de(shot_rgb: tuple[float, float, float], tokens: dict[str, str], name: str) -> float:
    return delta_e(shot_rgb, _rgb(tokens[name]))


@pytest.mark.parametrize("theme", LOOK_STATES)
def test_b3_stat(genui, theme):
    """B3 Stat (board Main): the label 12 px, ``--muted``, letter-spaced, not transformed; the value
    56 px with tabular numbers; the condition line; one SVG glyph inside the Stat."""
    _ok(genui)
    shot, tok = genui.shots[theme]["board-main"], genui.tokens[theme]
    label, value = _text(shot, "PARIS · NOW"), _text(shot, "[17°]")
    _text(shot, "Partly cloudy")
    problems = []
    if abs(label["fs"] - 12) > LEN:
        problems.append(f"label {label['fs']} px")
    if _token_de(_rgb(label["color"]), tok, "--muted") > TOKEN_DE:
        problems.append(f"label colour {label['color']} is not --muted {tok['--muted']}")
    if (float(re.sub(r"[^\d.\-]", "", label["ls"]) or 0) if label["ls"] != "normal" else 0) <= 0:
        problems.append(f"label letter-spacing {label['ls']}")
    if label["tt"] != "none":
        problems.append(f"label text-transform {label['tt']} (H-T184-2)")
    if abs(value["fs"] - 56) > LEN:
        problems.append(f"value {value['fs']} px")
    if "tabular-nums" not in value["fvn"]:
        problems.append(f"value font-variant-numeric {value['fvn']}")
    stat = [k for k in shot.dump["kinds"] if k["kind"] == "Stat"]
    glyphs = [g for g in shot.dump["svgs"] if g["kind"] == "Stat"]
    if len(stat) != 1 or len(glyphs) != 1:
        problems.append(f"{len(stat)} Stat(s), {len(glyphs)} Stat glyph(s)")
    assert problems == [], f"T-0184 B3 Stat ({theme}): {problems}"


@pytest.mark.parametrize("theme", LOOK_STATES)
def test_b3_stat_tiles(genui, theme):
    """B3 StatTiles (board Main): one equal-width row at 1440 px; title above value; percent drawn
    as a thin bar whose length is proportional to the percent, plus its label; the highlighted
    tile (Saturday) is tinted towards ``--accent``. The percent bounds 0 and 100 (exemplar)."""
    _ok(genui)
    shot, tok = genui.shots[theme]["board-main"], genui.tokens[theme]
    tiles = shot.dump["tiles"]
    assert [t["title"] for t in tiles] == ["Friday", "Saturday", "Sunday"], tiles
    widths = [t["rect"][2] - t["rect"][0] for t in tiles]
    tops = [t["rect"][1] for t in tiles]
    problems = []
    if max(widths) - min(widths) > 1 or max(tops) - min(tops) > 1:
        problems.append(f"tiles are not one equal-width row: widths {widths}, tops {tops}")
    for title, value in (("Friday", "[19°]"), ("Saturday", "[22°]"), ("Sunday", "[16°]")):
        if _text(shot, title)["glyph"][1] >= _text(shot, value)["glyph"][1]:
            problems.append(f"{title}: the value is not below the title")
    fills = _fills(shot, "StatTiles")
    if len(fills) != 3:
        problems.append(f"{len(fills)} progress fills, want 3")
    else:
        w = [f["rect"][2] - f["rect"][0] for f in fills]
        for wi, pct in zip(w, (20, 10, 60), strict=True):
            if abs(wi - w[2] * pct / 60) > 1.5:
                problems.append(f"fill widths {w} are not proportional to 20/10/60")
                break
        hs = [f["rect"][3] - f["rect"][1] for f in fills]
        if any(not (2 <= h <= 10) for h in hs):
            problems.append(f"fill heights {hs}: not a thin bar")
    backs = [
        _mean(shot, [t["rect"][0] + 4, t["rect"][1] + 4, t["rect"][0] + 16, t["rect"][1] + 12])
        or (0.0, 0.0, 0.0)
        for t in tiles
    ]
    accent = _rgb(tok["--accent"])
    plain = backs[0]
    if delta_e(backs[1], plain) < TINT_MIN_DE or delta_e(backs[1], accent) >= delta_e(
        plain, accent
    ):
        problems.append(
            f"Saturday {hexc(backs[1])} is not tinted towards --accent vs {hexc(plain)}"
        )
    if delta_e(backs[0], backs[2]) > SAME_DE:
        problems.append(f"the two plain tiles differ: {hexc(backs[0])} {hexc(backs[2])}")
    bounds = genui.shots[theme]["stattiles-percent-bounds"]
    bf = _fills(bounds, "StatTiles")
    if (
        not bf
        or any((f["rect"][2] - f["rect"][0]) > 0.5 for f in bf[:-1])
        or bf[-1]["rect"][2] - bf[-1]["rect"][0] <= 0
    ):
        problems.append(f"percent 0 / 100 fills: {[f['rect'] for f in bf]}")
    assert problems == [], f"T-0184 B3 StatTiles ({theme}):\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("theme", LOOK_STATES)
def test_b3_comparison(genui, theme):
    """B3 Comparison (board Compare): a label column plus one column per entry (cells of a column
    aligned across rows, columns left to right); the badge a pill in its header; the highlighted
    column tinted; toned cells are pills in --ok/--ok-soft and --warn/--warn-soft. Extra cells
    become a further column; no column is tinted for an out-of-range or absent highlight."""
    _ok(genui)
    shot, tok = genui.shots[theme]["board-compare"], genui.tokens[theme]
    left = lambda s: _text(shot, s)["glyph"][0]  # noqa: E731
    problems = []
    labels = ["Volumes", "Price per book", "Format", "Best for"]
    if max(left(s) for s in labels) - min(left(s) for s in labels) > 1:
        problems.append("the row labels are not one column")
    col_a, col_b = ["[1]", "[€45]", "[Hardcover, slipcase]"], ["[3]", "[€18]", "[Paperback]"]
    for col in (col_a, col_b):
        if max(left(s) for s in col) - min(left(s) for s in col) > 2:
            problems.append(f"cells {col} are not one column")
    if not (left("Volumes") < left("[1]") < left("[3]")):
        problems.append("columns are not label, first, second from left to right")
    if (
        abs(left("Deluxe hardcover") - left("[1]")) > 2
        or abs(left("Regular volumes") - left("[3]")) > 2
    ):
        problems.append("a header does not sit over its column")
    badge = _text(shot, "Recommended")
    bh = badge["el"][3] - badge["el"][1]
    if badge["radius"] + 0.5 < bh / 2:
        problems.append(f"the badge is not a pill (radius {badge['radius']}, height {bh})")
    hl, plain = _behind(shot, "[1]"), _behind(shot, "[3]")
    accent = _rgb(tok["--accent"])
    if delta_e(hl, plain) < TINT_MIN_DE or delta_e(hl, accent) >= delta_e(plain, accent):
        problems.append(f"the highlighted column {hexc(hl)} is not tinted vs {hexc(plain)}")
    for s, ink, soft in (
        ("[Gifts and display]", "--ok", "--ok-soft"),
        ("[Reading on the go]", "--warn", "--warn-soft"),
    ):
        t = _text(shot, s)
        behind = _behind(shot, s)
        if (
            _token_de(_rgb(t["color"]), tok, ink) > TOKEN_DE
            or _token_de(behind, tok, soft) > TOKEN_DE
        ):
            problems.append(f"{s}: ink {t['color']} on {hexc(behind)}, want {ink} on {soft}")
        if t["radius"] + 0.5 < (t["el"][3] - t["el"][1]) / 2:
            problems.append(f"{s}: not a pill")
    for key, pairs in (("comparison-laptops", [("1.6 kg", "1.4 kg")]),):
        other = genui.shots[theme][key]
        for a, b in pairs:
            if delta_e(_behind(other, a), _behind(other, b)) > SAME_DE:
                problems.append(f"{key}: untinted columns differ ({a} vs {b})")
    assert problems == [], f"T-0184 B3 Comparison ({theme}):\n  " + "\n  ".join(problems)


def test_b3_comparison_extra_cells_are_a_further_column_and_out_of_range_tints_nothing(genui):
    """B4/B7 in the page: x1-extra sits right of b1 (a column with no header); a2 sits under a1;
    ``highlight: 7`` tints neither column (exemplar), read from the layout run at 1440 px."""
    _ok(genui)
    problems = []
    for theme in LOOK_STATES:
        cells = genui.shots[theme]["t184-cells"]
        g = {s: _text(cells, s)["glyph"] for s in ("a1", "b1", "x1-extra", "a2", "Col A", "Col B")}
        if g["x1-extra"][0] < g["b1"][2] - 0.5:
            problems.append(f"{theme}: the extra cell is not a further column right of b1")
        if abs(g["a2"][0] - g["a1"][0]) > 2 or abs(g["Col A"][0] - g["a1"][0]) > 2:
            problems.append(f"{theme}: a2 / the header are not over a1's column")
        oor = genui.shots[theme]["comparison-highlight-out-of-range"]
        for a, b in (("Train", "Plane"), ("3 h 10", "3 h 40")):
            if delta_e(_behind(oor, a), _behind(oor, b)) > SAME_DE:
                problems.append(f"{theme}: highlight 7 tints a column ({a} vs {b})")
    assert problems == [], "\n  ".join(problems)


@pytest.mark.parametrize("theme", LOOK_STATES)
@pytest.mark.parametrize("key", ["board-chart", "t184-tie"])
def test_b3_bars(genui, key, theme):
    """B3 Bars: labels in one column; one fill per bar starting at one x; widths proportional to
    the values; the largest (the first on a tie) in ``--spark``, the rest muted (no gradient, not
    the accent); the value then the unit after each bar's end; the model's order kept; the tick
    labels left to right in one line beyond the bars."""
    _ok(genui)
    shot, tok = genui.shots[theme][key], genui.tokens[theme]
    if key == "board-chart":
        bars = [("Everest", 8849), ("K2", 8611), ("Kangchenjunga", 8586), ("Mont Blanc", 4806)]
        unit, top = "m", 0
    else:
        bars = [("First", 5), ("Second", 9), ("Third", 9), ("Fourth", 2)]
        unit, top = "pts", 1
    problems = []
    lab = [_text(shot, s) for s, _ in bars]
    if max(t["glyph"][0] for t in lab) - min(t["glyph"][0] for t in lab) > 1:
        problems.append("labels are not one column")
    if [t["glyph"][1] for t in lab] != sorted(t["glyph"][1] for t in lab):
        problems.append("the model's order is not kept")
    fills = _fills(shot, "Bars")
    if len(fills) != len(bars):
        problems.append(f"{len(fills)} fills for {len(bars)} bars")
    else:
        lefts = [f["rect"][0] for f in fills]
        widths = [f["rect"][2] - f["rect"][0] for f in fills]
        vmax = max(v for _, v in bars)
        if max(lefts) - min(lefts) > 1:
            problems.append(f"fills start at {lefts}")
        for (_s, v), w in zip(bars, widths, strict=True):
            if abs(w - widths[top] * v / vmax) > 1.5:
                problems.append(f"widths {widths} are not proportional to {[v for _, v in bars]}")
                break
        if fills[top]["bgimg"] != tok["--spark"]:
            problems.append(f"the largest bar's background {fills[top]['bgimg']!r} is not --spark")
        for k, f in enumerate(fills):
            if k == top:
                continue
            fill_px = _px(
                shot, (f["rect"][0] + f["rect"][2]) / 2, (f["rect"][1] + f["rect"][3]) / 2
            )
            if f["bgimg"] != "none" or _token_de(fill_px, tok, "--accent") < 10:
                problems.append(f"bar {k} is not muted ({f['bgimg']}, {hexc(fill_px)})")
        for (s, v), f in zip(bars, fills, strict=True):
            val, un = _text(shot, str(v)), [t for t in shot.dump["texts"] if t["text"] == unit]
            if val["glyph"][0] < f["rect"][2] - 0.5:
                problems.append(f"{s}: the value {v} is not after the bar's end")
            if not any(
                abs(u["glyph"][1] - val["glyph"][1]) < 2 and u["glyph"][0] >= val["glyph"][2] - 0.5
                for u in un
            ):
                problems.append(f"{s}: the unit {unit!r} does not follow the value")
        known = {s for s, _ in bars} | {str(v) for _, v in bars} | {unit, "en.wikipedia.org"}
        ticks = [t for t in shot.dump["texts"] if t["text"] not in known]
        xs = [t["glyph"][0] for t in ticks]
        if (
            len(ticks) < 2
            or xs != sorted(xs)
            or max(t["glyph"][1] for t in ticks) - min(t["glyph"][1] for t in ticks) > 2
        ):
            problems.append(
                f"tick labels {[t['text'] for t in ticks]} are not one left-to-right line"
            )
        elif not all(
            t["glyph"][1] >= max(f["rect"][3] for f in fills) - 0.5
            or t["glyph"][3] <= min(f["rect"][1] for f in fills) + 0.5
            for t in ticks
        ):
            problems.append("tick labels overlap the bars")
    assert problems == [], f"T-0184 B3 Bars ({key}, {theme}):\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("theme", LOOK_STATES)
def test_b3_steps_source_and_actions(genui, theme):
    """B3 Steps: the ordinal left of its title, the detail under the title, no list marker and no
    generated content in any card. Source: a pill with the glyph before the text. Actions: one
    button per label, >= 44 px tall, the first ``--tonal`` with ``--tonal-ink``, the rest not."""
    _ok(genui)
    tok = genui.tokens[theme]
    problems = []
    steps = genui.shots[theme]["steps-omelette"]
    for k, (title, detail) in enumerate(
        [
            ("Whisk the eggs", "Three eggs, a pinch of salt."),
            ("Heat the pan", None),
            ("Cook and fold", "Low heat, fold when just set."),
        ]
    ):
        o, t = _text(steps, str(k + 1)), _text(steps, title)
        if o["glyph"][2] > t["glyph"][0] + 0.5 or abs(o["glyph"][1] - t["glyph"][1]) > t["fs"]:
            problems.append(f"ordinal {k + 1} is not left of its title")
        if detail and _text(steps, detail)["glyph"][1] < t["glyph"][3] - 1:
            problems.append(f"the detail of {title!r} is not under it")
    for key, shot in genui.shots[theme].items():
        if shot.dump["markers"] or shot.dump["pseudo"]:
            problems.append(
                f"{key}: list markers {shot.dump['markers']}, generated content {shot.dump['pseudo']}"
            )
    src = genui.shots[theme]["source-only"]
    t = _text(src, "From a web search")
    glyph = [g for g in src.dump["svgs"] if g["kind"] == "Source"]
    if len(glyph) != 1 or glyph[0]["rect"][2] > t["glyph"][0] + 0.5:
        problems.append("the Source glyph is not before its text")
    chip = next((k["rect"] for k in src.dump["kinds"] if k["kind"] == "Source"), None)
    behind = _behind(src, "From a web search")
    card = _px(
        src,
        (src.dump["frame"]["rect"][0] + src.dump["frame"]["rect"][2]) / 2,
        src.dump["frame"]["rect"][1] + 5,
    )
    if delta_e(behind, card) < BORDER_MIN_DE and chip is not None:
        problems.append(f"the Source chip {hexc(behind)} does not stand off the card {hexc(card)}")
    acts = genui.shots[theme]["actions-three"].dump["buttons"]
    if [b["text"] for b in acts] != ["Show more", "Compare prices", "Start over"]:
        problems.append(f"buttons {[b['text'] for b in acts]}")
    else:
        if any(b["rect"][3] - b["rect"][1] < 44 - 0.01 for b in acts):
            problems.append(f"button heights {[b['rect'][3] - b['rect'][1] for b in acts]} < 44")
        first = acts[0]
        if (
            _token_de(_rgb(first["bg"]), tok, "--tonal") > TOKEN_DE
            or _token_de(_rgb(first["color"]), tok, "--tonal-ink") > TOKEN_DE
        ):
            problems.append(
                f"the first button {first['bg']}/{first['color']} is not --tonal/--tonal-ink"
            )
        sec = genui.shots[theme]["actions-three"]
        for b in acts[1:]:
            fill = _px(sec, b["rect"][0] + 2, (b["rect"][1] + b["rect"][3]) / 2)
            if delta_e(fill, _rgb(first["bg"])) < 10:
                problems.append(f"{b['text']!r} looks primary ({hexc(fill)})")
    assert problems == [], f"T-0184 B3 ({theme}):\n  " + "\n  ".join(problems)


# =========================================================================================
# B8: contrast
# =========================================================================================


def _text_ratios(shot: Shot) -> list[tuple[str, float, float]]:
    """(text, worst ratio over its glyph box, the floor) for every text node of the card."""
    out = []
    for t in shot.dump["texts"]:
        # per line box: a wrapped text's bounding box also covers the start of its first line
        samples = [shot.pixels[p] for line in t["lines"] for p in _grid(line) if p in shot.pixels]
        if not samples:
            continue
        worst = min(contrast_ratio(_over(t["color"], bg, t["opacity"]), bg) for bg in samples)
        out.append((t["text"][:40], worst, threshold_for(t["fs"], t["fw"])))
    return out


def _fill_ratios(shot: Shot) -> list[tuple[str, float]]:
    out = []
    for k, f in enumerate(shot.dump["fills"]):
        if f["kind"] not in ("Bars", "StatTiles"):
            continue
        (left, top, right, bottom) = f["rect"]
        if right - left < 2:
            continue
        mid = int((top + bottom) / 2)
        a, b = (
            shot.pixels.get((int((left + right) / 2), mid)),
            shot.pixels.get((int(right + 2), mid)),
        )
        if a is None or b is None:
            continue
        out.append((f"{f['kind']} fill {k}", contrast_ratio(a, b)))
    return out


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("key", CONTRAST_KEYS)
def test_b8_every_text_clears_its_floor_against_what_is_behind_it(genui, key, state):
    """B8: every text node of the card, against the pixels actually behind its glyphs (text made
    transparent for the shot): 4.5:1, or 3:1 for large text (the 56 px value)."""
    _ok(genui)
    shot = genui.shots[state].get(key)
    assert shot is not None, f"{key} was not measured in {state}"
    ratios = _text_ratios(shot)
    assert ratios, "premise: the card has text"
    low = [f"{t!r}: {r:.2f} < {floor}" for t, r, floor in ratios if r + 1e-9 < floor]
    assert low == [], f"T-0184 B8 text ({key}, {state}):\n  " + "\n  ".join(low)


@pytest.mark.parametrize("state", STATES)
def test_b8_fills_and_the_secondary_button_boundary_clear_3_to_1(genui, state):
    """B8: bar and progress fills against their neighbour, and each secondary Actions button's
    boundary (its border, or its fill) against the card, clear 3:1."""
    _ok(genui)
    problems = []
    measured = 0
    for key in ("board-main", "board-chart", "bars-titled", "stattiles-percent-bounds", "t184-tie"):
        for name, r in _fill_ratios(genui.shots[state][key]):
            measured += 1
            if r < 3.0:
                problems.append(f"{key} {name}: {r:.2f}")
    assert measured >= 12, f"premise: {measured} fills measured"
    shot = genui.shots[state]["actions-three"]
    for b in shot.dump["buttons"][1:]:
        mid = (b["rect"][1] + b["rect"][3]) / 2
        outside = _px(shot, b["rect"][0] - 3, mid)
        fill = _px(shot, b["rect"][0] + 2, mid)
        edge = _over(b["bcolor"], fill) if b["bw"] >= 1 and b["bstyle"] != "none" else fill
        r = max(contrast_ratio(edge, outside), contrast_ratio(fill, outside))
        if r < 3.0:
            problems.append(f"secondary {b['text']!r}: boundary {r:.2f}")
    assert problems == [], f"T-0184 B8 fills/boundaries ({state}):\n  " + "\n  ".join(problems)


def test_b8_the_focus_ring_clears_3_to_1(genui):
    """B8: the focus ring of the newest card's first Actions button (``focus({focusVisible})``)
    is drawn and clears 3:1 against the card just outside it (system-dark)."""
    _ok(genui)
    ring, before = genui.focus.get("ring"), genui.focus.get("before")
    assert ring and ring["inCard"], f"the Actions button took no focus: {ring}"
    assert ring["style"] not in ("none", "") and ring["width"] >= 1, (
        f"no visible focus ring: {ring}"
    )
    (left, top, _r, bottom) = ring["rect"]
    assert before is not None, "the newest card was not measured before focus"
    assert ring["offset"] + ring["width"] < 8, (
        f"the ring reaches past the sampled neighbour: {ring}"
    )
    behind = _px(before, left - 8, (top + bottom) / 2)
    r = contrast_ratio(_over(ring["color"], behind), behind)
    assert r >= 3.0, f"focus ring {ring['color']} on {hexc(behind)}: {r:.2f}"


def test_b8_mutant_muted_bars_in_the_card_colour_turn_the_fill_row_red(genui):
    """B8 adversarial: with the muted bars painted in the card's own colour (injected in the page),
    the same fill measurement reports each of them below 3:1; the unmutated chart passes."""
    _ok(genui)
    real = _fill_ratios(genui.shots["system-dark"]["board-chart"])
    assert len(real) == 4 and all(r >= 3.0 for _, r in real), (
        f"premise: the real chart passes {real}"
    )
    mshot = genui.mutant.get("shot")
    assert mshot is not None, "the mutant chart was not measured"
    mut = _fill_ratios(mshot)
    assert len(mut) >= 3 and sum(1 for _, r in mut if r < 3.0) >= 3, (
        f"the fill row did not catch muted bars in the card colour: {mut}"
    )


# =========================================================================================
# B9: no overflow
# =========================================================================================


@pytest.mark.parametrize("view", [f"{w}x{h}" for w, h in LAYOUT_VIEWS])
def test_b9_nothing_overflows_its_card_and_only_a_comparison_scrolls(genui, view):
    """B9 at five views, every card (stress cases included): the card lies inside its slot and
    does not scroll sideways; nothing inside it is wider than the card unless a Comparison
    scroller inside the card holds it; no other element scrolls; the document never scrolls at
    >= 1100 px."""
    _ok(genui)
    lay = genui.layout.get(view) or {}
    assert lay.get("cards"), f"{view}: no layout"
    problems = []
    for key, c in lay["cards"].items():
        if c is None or c["slot"] is None:
            problems.append(f"{key}: card or slot missing")
            continue
        (rl, _t, rr, _b), (sl, _st, sr, _sb) = c["root"], c["slot"]
        if rl < sl - 0.5 or rr > sr + 0.5:
            problems.append(f"{key}: card {rl:.1f}..{rr:.1f} outside its slot {sl:.1f}..{sr:.1f}")
        if c["scroll"][0] > c["scroll"][1] + 1:
            problems.append(f"{key}: the card scrolls sideways {c['scroll']}")
        if c["nbad"]:
            problems.append(f"{key}: {c['nbad']} element(s) overflow the card, e.g. {c['bad'][:2]}")
        stray = [s for s in c["scrollers"] if not s["inCmp"]]
        if stray:
            problems.append(f"{key}: {len(stray)} scroller(s) outside a Comparison")
    w = int(view.split("x")[0])
    if w >= 1100:
        (sw, sh), (vw, vh) = lay["doc"]["scroll"], lay["doc"]["view"]
        if sw > vw or sh > vh:
            problems.append(f"the document scrolls: {sw}x{sh} in {vw}x{vh}")
    assert problems == [], f"T-0184 B9 at {view}:\n  " + "\n  ".join(problems)


def test_b9_stat_tiles_are_one_row_wide_and_wrap_in_a_narrow_slot(genui):
    """B3/B9: "an equal-width row that wraps in a narrow slot": the four 12-digit tiles wrap
    (more than one row) at 420 px (B9's overflow row keeps them inside the card); at 1440 px every
    tile row is equal-width. One row at 1440 px is board Main's (three tiles, ``test_b3_stat_tiles``)."""
    _ok(genui)
    wide = genui.layout["1440x900"]["cards"]["stattiles-four-wide"]["tiles"]
    narrow = genui.layout["420x900"]["cards"]["stattiles-four-wide"]["tiles"]
    (wide_tiles,) = wide.values()
    (narrow_tiles,) = narrow.values()
    assert len(wide_tiles) == 4 and len(narrow_tiles) == 4
    rows: dict[int, list[float]] = {}
    for t in wide_tiles:
        rows.setdefault(round(t[1]), []).append(t[2] - t[0])
    assert all(max(w) - min(w) <= 1 for w in rows.values()), f"unequal tiles in a row: {wide_tiles}"
    assert len({round(t[1]) for t in narrow_tiles}) >= 2, f"no wrap at 420 px: {narrow_tiles}"


# =========================================================================================
# B5: a real click
# =========================================================================================


def test_b5_a_real_click_on_actions_equals_a_gate_made_button_frame(genui):
    """B5: a mouse click on the newest card's "Compare prices" sends exactly one astra.ui.action;
    decoded, it equals the frame a gate-made Button (``UiGate``'s RB repair of the same label,
    component id "btn") sends, but for the surface id and the click's timestamp. A click on the
    card body and Escape send nothing."""
    _ok(genui)
    assert genui.stray_actions == 0, f"{genui.stray_actions} action(s) from a body click / a key"
    assert len(genui.actions) == 2, (
        f"want the Actions click then the gate button's: {genui.actions}"
    )
    v2_frame, gate_frame = genui.actions
    for frame, sid in ((v2_frame, "turn-t98"), (gate_frame, "turn-t99")):
        assert frame["surface_id"] == sid
        ts = frame["context"]["a2ui"]["timestamp"]
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", ts), ts

    def norm(f: dict[str, Any]) -> dict[str, Any]:
        g = json.loads(json.dumps(f))
        g["surface_id"] = "<sid>"
        g["context"]["a2ui"]["timestamp"] = "<ts>"
        return g

    assert norm(v2_frame) == norm(gate_frame), (v2_frame, gate_frame)
    assert v2_frame["name"] == "__gate.clicked"
    assert v2_frame["context"] == {
        "label": CLICK_LABELS[1],
        "a2ui": {"sourceComponentId": "btn", "timestamp": v2_frame["context"]["a2ui"]["timestamp"]},
    }


def test_b5a_tab_never_focuses_anything_in_a_card_but_its_actions_buttons(genui):
    """B5a: with the newest card holding a 5-column Comparison that scrolls sideways inside it
    (Chrome makes such a scroller keyboard-focusable on its own), Tab from the top of the page
    until focus wraps never lands on anything inside ``[data-a2ui-root]`` except an ``Actions``
    ``<button>``; it does reach both Actions buttons (so the walk entered the card), and the walk
    sends no ``astra.ui.action``."""
    _ok(genui)
    tab = genui.tab
    assert tab, "the Tab walk did not run"
    assert tab["scrollers"] and tab["scrollers"] >= 1, (
        f"premise: the 5-column Comparison does not scroll inside the card ({tab['scrollers']})"
    )
    stops = tab["stops"]
    assert len(stops) < TAB_MAX, f"premise: focus never wrapped in {TAB_MAX} Tab presses"
    reached = [
        s["text"] for s in stops if s.get("surface") == TAB_SID and s.get("kind") == "Actions"
    ]
    assert reached == TAB_LABELS, (
        f"premise: Tab reached {reached} of the newest card's Actions buttons {TAB_LABELS}; "
        f"stops in cards: {[s for s in stops if s.get('inRoot')]}"
    )
    bad = [
        s for s in stops if s["inRoot"] and not (s["tag"] == "button" and s["kind"] == "Actions")
    ]
    assert bad == [], (
        f"T-0184 B5a: Tab focused something in a card that is not an Actions button: {bad}"
    )
    assert tab["sent"] == 0, f"the Tab walk sent {tab['sent']} astra.ui.action(s)"


# =========================================================================================
# B7a: tick labels never overlap
# =========================================================================================


def _js_string(v: Any) -> str:
    """``String(v)`` for the JSON numbers of these payloads (integers, or a short decimal)."""
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer()):
        return str(int(v))
    return repr(float(v))


def _bars_of(key: str) -> dict[str, Any]:
    surface = surfaces()[key]
    found = [
        c
        for m in surface.messages
        for c in (m.get("updateComponents") or {}).get("components") or []
        if c.get("component") == "Bars"
    ]
    assert len(found) == 1, f"premise: {key} holds one Bars, not {len(found)}"
    return found[0]


def _tick_texts(texts: list[dict[str, Any]], bars: dict[str, Any]) -> list[dict[str, Any]]:
    """The tick labels: every text node of the Bars component that is not its title, a bar's
    label, a bar's value (``String(value)``, the first such node after its label) or the unit after
    that value. Read from the payload, in the order B4 fixes (label, value, unit per bar)."""
    labels = [str(b["label"]) for b in bars["bars"]]
    values = [_js_string(b["value"]) for b in bars["bars"]]
    title, unit = bars.get("title"), bars.get("unit")
    out: list[dict[str, Any]] = []
    k, want_value, after_value, title_seen = 0, None, False, False
    for t in texts:
        s = t["text"]
        if title is not None and not title_seen and s == title:
            title_seen = True
        elif k < len(labels) and s == labels[k]:
            want_value, after_value, k = values[k], False, k + 1
        elif want_value is not None and s == want_value:
            want_value, after_value = None, True
        elif after_value and unit is not None and s == unit:
            after_value = False
        else:
            out.append(t)
            after_value = False
    return out


def tick_problem(ticks: list[str], vmax: float, allow_none: bool = False) -> str | None:
    """None when ``ticks`` (in document order) are B4 (ii)'s ticks for a largest value ``vmax``:
    each ``String(n)`` (no abbreviation, no suffix), the first 0, equally spaced by a step from the
    closed list ``NICE`` times a power of ten, at most five, the last at least half the largest
    value and less than one step beyond it (the reading of tests/hud/ui_cards_v2.test.mjs).
    ``allow_none``: no tick at all is also valid (B7a ruling, 195866ca: where even 0 and the scale
    end cannot sit side by side, no tick labels are drawn); one label alone never is."""
    if vmax == 0:
        return (
            None if ticks in ([], ["0"]) else f"all bars 0: ticks may be none or '0', got {ticks}"
        )
    if allow_none and ticks == []:
        return None
    if not 2 <= len(ticks) <= 5:
        return f"{len(ticks)} ticks {ticks}: a scale needs 2 (0 and its end), at most 5"
    for t in ticks:
        if not JS_NUMBER.fullmatch(t):
            return f"tick {t!r} is not String(n) of a number (no abbreviation, suffix or separator)"
    nums = [float(t) for t in ticks]
    if nums[0] != 0:
        return f"the first tick is {ticks[0]}, not 0"
    step = nums[1] - nums[0]
    if any(abs(nums[i] - nums[i - 1] - step) > 1e-9 * max(1.0, step) for i in range(1, len(nums))):
        return f"ticks {ticks} are not equally spaced"
    exp = 10.0 ** math.floor(math.log10(step))
    if not any(abs(step / exp - n) < 1e-9 for n in NICE):
        return f"the step {step} is not from the closed list {NICE} x 10^k"
    if nums[-1] < vmax / 2:
        return f"the last tick {ticks[-1]} covers less than half the largest value {vmax}"
    if nums[-1] - step >= vmax:
        return f"the last tick {ticks[-1]} is a whole step beyond the largest value {vmax}"
    return None


def _overlap(a: list[float], b: list[float]) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > 0.01 and min(a[3], b[3]) - max(a[1], b[1]) > 0.01


def test_b7a_premise_tick_problem_and_overlap_by_hand():
    """The B7a oracles on hand-made cases: the approved Chart board's ticks pass; a thinned scale
    (0 and its end, or every other tick) passes; abbreviations, suffixes, separators, a missing 0,
    six ticks and a non-list step fail; touching boxes do not overlap, crossing ones do."""
    assert tick_problem(["0", "3000", "6000", "9000"], 8849) is None
    assert tick_problem(["0", "10000"], 8849) is None
    assert tick_problem(["0", "500000000000", "1000000000000"], 1e12) is None
    assert tick_problem(["0", "5000000000000"], 4321098765432) is None
    for bad, vmax in (
        (["0", "1T"], 1e12),
        (["0", "1000000000000 USD"], 1e12),
        (["0", "1,000,000,000,000"], 1e12),
        (["0", "1e12"], 1e12),
        (["0", "1.0e+12"], 1e12),
        (["250000000000", "500000000000", "1000000000000"], 1e12),
        (["0", "1", "2", "3", "4", "5"], 5),
        (["0", "3500", "7000"], 7000),
        (["0", "2000"], 8849),
    ):
        assert tick_problem(bad, vmax) is not None, bad
    assert tick_problem([], 0) is None and tick_problem(["0", "1"], 0) is not None
    # B7a ruling (195866ca): no ticks is valid only where allowed; one label alone never is
    assert tick_problem([], 8849) is not None
    assert tick_problem([], 8849, allow_none=True) is None
    for alone in (["0"], ["10000"], ["1000000000000"]):
        assert tick_problem(alone, 8849, allow_none=True) is not None, alone
    assert tick_problem(["0", "1T"], 1e12, allow_none=True) is not None
    assert not _overlap([0, 0, 10, 10], [10, 0, 20, 10])
    assert _overlap([0, 0, 10, 10], [9, 2, 20, 8])
    texts = [{"text": s} for s in ["T", "A", "5", "u", "B", "1000000000000", "u", "0", "1e12"]]
    bars = {
        "title": "T",
        "unit": "u",
        "bars": [{"label": "A", "value": 5}, {"label": "B", "value": 1e12}],
    }
    assert [t["text"] for t in _tick_texts(texts, bars)] == ["0", "1e12"]


#: B7a's views: every B9 view (ruling 195866ca, 420x900 included).
TICK_VIEWS = tuple(f"{w}x{h}" for w, h in LAYOUT_VIEWS)
#: Views where "no tick labels at all" is also valid. The ruling allows it where even 0 and the
#: scale end cannot sit side by side; that is read as the views narrower than 1000 px (420, 540,
#: 580), where a card's scale can be under 100 px (83 px for 1e12 at 580, reviewer B). At 1000 px
#: and wider (about 142 px or more) two labels fit, so a scale there still needs 0 and its end.
NO_TICK_VIEWS = frozenset(f"{w}x{h}" for w, h in LAYOUT_VIEWS if w < 1000)


@pytest.mark.parametrize("view", TICK_VIEWS)
def test_b7a_tick_labels_never_overlap_and_stay_a_closed_list_scale(genui, view):
    """B7a, at each B9 view: on every B7 Bars case (all 0, one bar, 1e12), a 13-digit maximum and
    the other Bars cards, no two tick-label glyph boxes intersect; the ticks that are drawn are a
    scale from 0 to its end with a step from the closed list (``tick_problem``), each ``String(n)``
    (no abbreviation, no suffix), and they read left to right in value order. At 420x900 no tick
    at all is also valid; one label alone is not."""
    _ok(genui)
    got = genui.ticks.get(view) or {}
    problems = []
    for key in TICK_KEYS:
        card = got.get(key)
        if not card or card["bars"] != 1:
            problems.append(f"{key}: no single Bars component measured ({card and card['bars']})")
            continue
        bars = _bars_of(key)
        vmax = max(float(b["value"]) for b in bars["bars"])
        ticks = _tick_texts(card["texts"], bars)
        names = [t["text"] for t in ticks]
        p = tick_problem(names, vmax, allow_none=view in NO_TICK_VIEWS)
        if p:
            problems.append(f"{key}: {p}")
        boxes = [r for t in ticks for r in (t["rects"] or [t["box"]])]
        hits = [
            (names[i], names[j])
            for i in range(len(ticks))
            for j in range(i + 1, len(ticks))
            if any(
                _overlap(a, b)
                for a in (ticks[i]["rects"] or [ticks[i]["box"]])
                for b in (ticks[j]["rects"] or [ticks[j]["box"]])
            )
        ]
        if hits:
            problems.append(f"{key}: tick labels intersect: {hits} (boxes {boxes})")
        xs = [t["box"][0] for t in ticks]
        if not p and len(ticks) >= 2 and xs != sorted(xs):
            problems.append(f"{key}: ticks {names} are not left to right ({xs})")
    assert problems == [], f"T-0184 B7a at {view}:\n  " + "\n  ".join(problems)


# =========================================================================================
# B11: loopback, no @font-face
# =========================================================================================


def test_b11_the_session_requests_only_loopback_and_loads_no_font_face(genui):
    """B11 / T-0128 criterion 2: every request of the whole session (HTTP and WebSocket) is on
    loopback, no font file is requested, and no ``@font-face`` is declared in any sheet."""
    _ok(genui)
    assert genui.requests, "the request log is empty, so this row would be vacuous"
    off = [
        u
        for u in genui.requests
        if urllib.parse.urlsplit(u).scheme in {"http", "https", "ws", "wss"}
        and urllib.parse.urlsplit(u).hostname not in {"127.0.0.1", "localhost", "::1"}
    ]
    assert off == []
    fonts = [u for u in genui.requests if re.search(r"\.(woff2?|ttf|otf)(\?|$)", u)]
    assert fonts == []
    assert genui.fonts.get("faces") == 0 and genui.fonts.get("fontset") == 0, genui.fonts


# =========================================================================================
# Group B amendment after reviewer A, round 1 (2026-10-28): B6a, B6b, B7b, B7c, B9a, B5b,
# B11a, B3a. One more Chromium session (``genui_r1``), run after ``genui``'s has closed.
#
# Readings, stated so a reviewer can disagree:
# - *The scale* of a Bars card is located without class names. With ticks drawn: the lowest
#   common ancestor of the tick labels' elements (the box the ticks are placed in). With no tick
#   drawn: the deepest element of the Bars component that holds no text and no fill and whose left
#   edge is the fills' common left edge (the empty axis). "Fill width / scale width ~ value / last
#   tick" is then ``|w_fill - w_scale * v / last| <= 1`` px; with no ticks the longest fill's width
#   is the scale's within 1 px.
# - *The axis row's height* (B7c) is that scale element's height, with labels (1440 px) and
#   without (420 px), within 1 px. The resize sweep listens for window ``error`` events and console
#   errors naming "ResizeObserver loop".
# - *One line* (B9a): every rect of the text node's ``Range.getClientRects`` with a width shares
#   one top (rounded to the pixel); a value and its unit, and a percent split as "<p>" then "%",
#   share it too. Which nodes are derived comes from the payload (B4's order), never a class.
# - *B6b*: the re-sends are three identical canvases with rising ``canvas_idx`` (the last
#   ``final``), after the card's layout settled at that width; mutation records of every type on
#   the card root's subtree are counted.
# =========================================================================================

R1_VIEWS = tuple(f"{w}x{h}" for w, h in LAYOUT_VIEWS)
#: B9a: the cards whose derived numbers are read (values+units, ticks, percents, ordinals).
B9A_KEYS = (
    "board-chart",
    "bars-titled",
    "bars-single",
    "bars-huge",
    "t184-13digit",
    "t184-tie",
    "t184-bars12",
    "stattiles-storage",
    "stattiles-percent-bounds",
    "board-main",
    "steps-omelette",
)
#: B7b: the Bars cards measured against their scale (all bars 0 has no scale to measure).
B7B_KEYS = tuple(k for k in TICK_KEYS if k != "bars-all-zero")
B6B_KEY = "bars-huge"
GROW_SID = "t184-grow"
GROW_FIRST = [{"label": "Alpha", "value": 40}, {"label": "Beta", "value": 100}]
GROW_EXTRA = {"label": "Gamma", "value": 12.75}
B3A_SID = "t184-b3a"
#: B3a: two headers, highlight 2 (out of range of the headers, inside the widest row's cells).
B3A_COMPONENTS: list[dict[str, Any]] = [
    {
        "id": "root",
        "component": "Comparison",
        "columns": [{"title": "Head A"}, {"title": "Head B"}],
        "highlight": 2,
        "rows": [
            {"label": "Row one", "cells": ["aa1", "bb1", "xx1"]},
            {"label": "Row two", "cells": ["aa2", "bb2", "xx2"]},
        ],
    }
]
B5B_SID = "turn-t101"
B5B_INNER = "Inner choice"
#: B5b: a gate-admitted Button (the gate's own ``__gate.clicked`` action, as its RB repair writes
#: it) that wraps an Actions.
B5B_COMPONENTS: list[dict[str, Any]] = [
    {"id": "root", "component": "Column", "children": ["wrap"]},
    {
        "id": "wrap",
        "component": "Button",
        "child": "act",
        "action": {"event": {"name": "__gate.clicked", "context": {"label": "Outer button"}}},
    },
    {"id": "act", "component": "Actions", "labels": [B5B_INNER]},
]
#: B7c': the sweep crosses the views where bars-huge's axis goes bare (580, 540) and back.
RESIZE_SWEEP = (1440, 1200, 1000, 800, 600, 580, 540, 500, 420, 540, 580, 760, 1100, 1440)
#: B7c': the card and the views where its axis is drawn with labels and bare (reviewer B: 1e12 at
#: 580x900 draws "scale 83 ticks []").
B7C_KEY, B7C_LABELLED = "bars-huge", "1440x900"
#: B7c' fixture after 0664495e (ruling ad799aff): no payload reaches a bare axis at the B9 views,
#: so the card's slot is narrowed in-page to ``B7C_SLOT_PX`` (premise: no tick drawn there), and
#: the resize sweep also sweeps that slot's width through the bare state and back, 1440x900.
B7C_SLOT_PX = 120
B7C_SLOT_SWEEP = (600, 400, 300, 220, 160, 120, 100, 120, 160, 220, 300, 400, 600)
#: Narrows (``width`` px) or restores (``null``) the slot holding card ``sid``.
SLOT_JS = r"""
([sid, width]) => {
  const root = document.querySelector('[data-ui-surface="' + sid + '"] [data-a2ui-root]');
  const slot = root ? root.closest('[data-card-slot]') : null;
  if (!slot) return null;
  for (const p of ['width', 'max-width', 'min-width', 'flex']) slot.style.removeProperty(p);
  if (width !== null) {
    slot.style.setProperty('width', width + 'px', 'important');
    slot.style.setProperty('max-width', width + 'px', 'important');
    slot.style.setProperty('min-width', '0', 'important');
    slot.style.setProperty('flex', 'none', 'important');
  }
  const r = root.getBoundingClientRect();
  return [r.left, r.right];
}
"""
#: B9b: a Bars inside a basic Row beside two Stats (reviewer A's probe shape).
ROWBARS_SID = "t184-rowbars"
ROWBARS_COMPONENTS: list[dict[str, Any]] = [
    {"id": "root", "component": "Row", "children": ["rb", "rs1", "rs2"]},
    {
        "id": "rb",
        "component": "Bars",
        "unit": "u",
        "bars": [{"label": "Huge", "value": 4321098765432}, {"label": "Tiny", "value": 7}],
    },
    {"id": "rs1", "component": "Stat", "label": "LEFT STAT", "value": "12"},
    {"id": "rs2", "component": "Stat", "label": "RIGHT STAT", "value": "34"},
]
#: B9b' (reviewer A round 3, aee92609): the Bars nested in a Row child, in a Column under a Text and
#: in a Card (reviewer A's probe shapes; values 8849 and 4806 m).
_B9B_BARS = {
    "id": "nb",
    "component": "Bars",
    "unit": "m",
    "bars": [{"label": "Everest", "value": 8849}, {"label": "Mont Blanc", "value": 4806}],
}
#: sid -> (components, the Row's children (the siblings), the Bars' id)
ROWBARS_CASES: dict[str, tuple[list[dict[str, Any]], list[str], str]] = {
    ROWBARS_SID: (ROWBARS_COMPONENTS, ["rb", "rs1", "rs2"], "rb"),
    "t184-row-col-bars": (
        [
            {"id": "root", "component": "Row", "children": ["nc", "ns"]},
            {"id": "nc", "component": "Column", "children": ["nt", "nb"]},
            {"id": "nt", "component": "Text", "text": "Heights"},
            _B9B_BARS,
            {"id": "ns", "component": "Stat", "label": "X", "value": "12"},
        ],
        ["nc", "ns"],
        "nb",
    ),
    "t184-row-card-bars": (
        [
            {"id": "root", "component": "Row", "children": ["nc", "ns"]},
            {"id": "nc", "component": "Card", "child": "nb"},
            _B9B_BARS,
            {"id": "ns", "component": "Stat", "label": "X", "value": "12"},
        ],
        ["nc", "ns"],
        "nb",
    ),
    # reviewer B, N7: a Bars in a Column aligned center or start (not stretched) still takes the
    # whole width (reviewer B's probe shape: Bars then a caption Text)
    **{
        f"t184-col-{align}-bars": (
            [
                {"id": "root", "component": "Column", "align": align, "children": ["cb", "ct"]},
                {
                    "id": "cb",
                    "component": "Bars",
                    "unit": "m",
                    "bars": [{"label": "Everest", "value": 8849}, {"label": "K2", "value": 8611}],
                },
                {"id": "ct", "component": "Text", "text": "caption"},
            ],
            ["cb", "ct"],
            "cb",
        )
        for align in ("center", "start")
    },
}
#: B9e (ruling 7f3562a9): the generated cross product of parents around a Bars. Containers, each
#: as (name, kind, extra props, takes children):
B9E_CONTAINERS: tuple[tuple[str, str, dict[str, Any], bool], ...] = (
    ("row", "Row", {}, True),
    ("col-start", "Column", {"align": "start"}, True),
    ("col-center", "Column", {"align": "center"}, True),
    ("col-end", "Column", {"align": "end"}, True),
    ("col-stretch", "Column", {"align": "stretch"}, True),
    ("list-v", "List", {"direction": "vertical"}, True),
    ("list-h", "List", {"direction": "horizontal"}, True),
    ("card", "Card", {}, False),
)
B9E_VIEWS = ("420x900", "580x900", "1440x900")
_B9E_BARS = {
    "component": "Bars",
    "unit": "m",
    "bars": [{"label": "Everest", "value": 8849}, {"label": "K2", "value": 8611}],
}


def _b9e_wrap(
    cid: str, container: tuple[str, str, dict[str, Any], bool], inner: str, sib: str
) -> list[dict[str, Any]]:
    """Container ``cid`` around component ``inner``; a sibling ``sib`` (a Stat beside a Row's
    child, else a caption Text) where the container takes children."""
    _name, kind, props, many = container
    if not many:
        return [{"id": cid, "component": kind, "child": inner, **props}]
    sibling = (
        {"id": sib, "component": "Stat", "label": "X", "value": "12"}
        if kind == "Row"
        else {"id": sib, "component": "Text", "text": "caption"}
    )
    return [{"id": cid, "component": kind, "children": [inner, sib], **props}, sibling]


def b9e_cases() -> dict[str, tuple[list[dict[str, Any]], list[str], str]]:
    """sid -> (components, the leaves to compare: the Bars and every sibling, the Bars' id):
    depth 1 (C[Bars, sibling]) and depth 2 (C1[C2[Bars, sibling], sibling]) over every container."""
    out: dict[str, tuple[list[dict[str, Any]], list[str], str]] = {}
    for c1 in B9E_CONTAINERS:
        comps = [*_b9e_wrap("root", c1, "eb", "es1"), {"id": "eb", **_B9E_BARS}]
        out[f"b9e-{c1[0]}"] = (comps, [c["id"] for c in comps if c["id"] in ("eb", "es1")], "eb")
        for c2 in B9E_CONTAINERS:
            comps = [
                *_b9e_wrap("root", c1, "ec2", "es1"),
                *_b9e_wrap("ec2", c2, "eb", "es2"),
                {"id": "eb", **_B9E_BARS},
            ]
            leaves = [c["id"] for c in comps if c["id"] in ("eb", "es1", "es2")]
            out[f"b9e-{c1[0]}--{c2[0]}"] = (comps, leaves, "eb")
    return out


B9E_CASES = b9e_cases()


#: B9b: each sibling's element box plus its text's rects; the card root's box; the Bars' fills and
#: its text nodes (data, line tops). ``arg = [sid, siblingIds, barsId]``.
ROWBARS_JS = r"""
([sid, ids, barsId]) => {
  const root = document.querySelector('[data-ui-surface="' + sid + '"] [data-a2ui-root]');
  if (!root) return null;
  const R = (r) => [r.left, r.top, r.right, r.bottom];
  const alpha = (c) => { const m = (c || '').match(/[\d.]+/g); return !m ? 0 : m.length >= 4 ? parseFloat(m[3]) : 1; };
  const kids = {};
  for (const id of ids) {
    const el = root.querySelector('[data-a2-id="' + id + '"]');
    if (!el) { kids[id] = null; continue; }
    const rects = [R(el.getBoundingClientRect())];
    const tw = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    for (let n = tw.nextNode(); n; n = tw.nextNode()) {
      if (!n.data.trim()) continue;
      const rg = document.createRange(); rg.selectNodeContents(n);
      for (const r of rg.getClientRects()) if (r.width > 0) rects.push(R(r));
    }
    kids[id] = rects;
  }
  const b = root.querySelector('[data-a2-id="' + barsId + '"]');
  const painted = b ? [...b.querySelectorAll('*')].filter((el) => el instanceof HTMLElement && !el.textContent.trim()
    && (alpha(getComputedStyle(el).backgroundColor) > 0 || getComputedStyle(el).backgroundImage !== 'none')) : [];
  const fills = painted.filter((el) => !painted.some((o) => o !== el && el.contains(o)));
  const texts = [];
  if (b) {
    const tw = document.createTreeWalker(b, NodeFilter.SHOW_TEXT);
    for (let n = tw.nextNode(); n; n = tw.nextNode()) {
      if (!n.data.trim()) continue;
      const rg = document.createRange(); rg.selectNodeContents(n);
      const rects = [...rg.getClientRects()].filter((r) => r.width > 0);
      texts.push({text: n.data, kind: 'Bars', tops: [...new Set(rects.map((r) => Math.round(r.top)))]});
    }
  }
  const fs = getComputedStyle(root), px = (v) => parseFloat(v) || 0;
  const rr = root.getBoundingClientRect();
  // the frame's content edges, from its computed border and padding (B9e: not from a sibling)
  const content = [rr.left + px(fs.borderLeftWidth) + px(fs.paddingLeft), rr.right - px(fs.borderRightWidth) - px(fs.paddingRight)];
  // a leaf held by a sideways scroller inside the card (a horizontal List) may sit outside the
  // card's box: the scroller's own box is what must stay inside
  const held = {};
  for (const id of ids) {
    const el = root.querySelector('[data-a2-id="' + id + '"]');
    let s = el ? el.parentElement : null, box = null;
    while (s && s !== root) {
      const cs = getComputedStyle(s);
      if (['auto', 'scroll', 'hidden'].includes(cs.overflowX) && s.scrollWidth > s.clientWidth + 1) { box = R(s.getBoundingClientRect()); break; }
      s = s.parentElement;
    }
    held[id] = box;
  }
  return {root: R(root.getBoundingClientRect()), content, held, kids, texts,
          bars_w: b ? b.getBoundingClientRect().width : null,
          fills: fills.map((f) => f.getBoundingClientRect().width)};
}
"""

#: Every non-blank text node of the cards, with its kind and its line tops. ``arg = {key: sid}``.
R1_TEXTS_JS = r"""
(sids) => {
  const out = {};
  for (const [key, sid] of Object.entries(sids)) {
    const w = document.querySelector('[data-ui-surface="' + sid + '"]');
    const root = w ? w.querySelector('[data-a2ui-root]') : null;
    if (!root) { out[key] = null; continue; }
    const texts = [];
    const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = tw.nextNode(); n; n = tw.nextNode()) {
      if (!n.data.trim()) continue;
      const rg = document.createRange(); rg.selectNodeContents(n);
      const rects = [...rg.getClientRects()].filter((r) => r.width > 0);
      const k = n.parentElement.closest('[data-a2-kind]');
      texts.push({text: n.data, kind: k && root.contains(k) ? k.getAttribute('data-a2-kind') : null,
                  tops: [...new Set(rects.map((r) => Math.round(r.top)))]});
    }
    out[key] = texts;
  }
  return out;
}
"""

#: B7b/B7c: the fills of the card's Bars and its scale. ``arg = [sid, tickIdx]``: the indices of the
#: tick labels among the Bars' non-blank text nodes (classified in Python from the payload).
R1_SCALE_JS = r"""
([sid, tickIdx]) => {
  const b = document.querySelector('[data-ui-surface="' + sid + '"] [data-a2ui-root] [data-a2-kind="Bars"]');
  if (!b) return null;
  const R = (el) => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
  const alpha = (c) => { const m = (c || '').match(/[\d.]+/g); return !m ? 0 : m.length >= 4 ? parseFloat(m[3]) : 1; };
  const all = [...b.querySelectorAll('*')];
  const painted = all.filter((el) => el instanceof HTMLElement && !el.textContent.trim()
    && (alpha(getComputedStyle(el).backgroundColor) > 0 || getComputedStyle(el).backgroundImage !== 'none'));
  const fills = painted.filter((el) => !painted.some((o) => o !== el && el.contains(o)));
  const nodes = [];
  const tw = document.createTreeWalker(b, NodeFilter.SHOW_TEXT);
  for (let n = tw.nextNode(); n; n = tw.nextNode()) if (n.data.trim()) nodes.push(n);
  let scale = null, how = 'none';
  const tickEls = tickIdx.map((i) => nodes[i].parentElement);
  if (tickEls.length >= 2) {
    let a = tickEls[0];
    while (a && !tickEls.every((t) => a.contains(t))) a = a.parentElement;
    scale = a; how = 'ticks';
  } else if (fills.length) {
    const left = Math.min(...fills.map((f) => f.getBoundingClientRect().left));
    const cands = all.filter((el) => !el.textContent.trim() && !fills.some((f) => el.contains(f) || f.contains(el))
      && Math.abs(el.getBoundingClientRect().left - left) <= 0.5);
    const deepest = cands.filter((el) => !cands.some((o) => o !== el && el.contains(o)));
    if (deepest.length) { scale = deepest[deepest.length - 1]; how = 'empty axis'; }
  }
  return {fills: fills.map(R), scale: scale ? R(scale) : null, how};
}
"""

#: B6b: count every mutation record on the card root's subtree from now on.
OBSERVE_JS = r"""
(sid) => {
  const root = document.querySelector('[data-ui-surface="' + sid + '"] [data-a2ui-root]');
  if (!root) return false;
  if (window.__t184mo) window.__t184mo.disconnect();
  window.__t184muts = [];
  window.__t184mo = new MutationObserver((rs) => { for (const r of rs) window.__t184muts.push(r.type + ':' + (r.attributeName || '') + ':' + r.target.nodeName); });
  window.__t184mo.observe(root, {subtree: true, childList: true, attributes: true, characterData: true});
  return true;
}
"""

#: B6a: remember the Bars fills now, and count attribute writes on them from now on.
GROW_MARK_JS = r"""
(sid) => {
  const b = document.querySelector('[data-ui-surface="' + sid + '"] [data-a2ui-root] [data-a2-kind="Bars"]');
  const alpha = (c) => { const m = (c || '').match(/[\d.]+/g); return !m ? 0 : m.length >= 4 ? parseFloat(m[3]) : 1; };
  const all = [...b.querySelectorAll('*')];
  const painted = all.filter((el) => el instanceof HTMLElement && !el.textContent.trim()
    && (alpha(getComputedStyle(el).backgroundColor) > 0 || getComputedStyle(el).backgroundImage !== 'none'));
  window.__t184fills = painted.filter((el) => !painted.some((o) => o !== el && el.contains(o)));
  window.__t184fillw = [];
  window.__t184fmo = new MutationObserver((rs) => { for (const r of rs) if (window.__t184fills.includes(r.target)) window.__t184fillw.push(r.attributeName); });
  for (const f of window.__t184fills) window.__t184fmo.observe(f, {attributes: true});
  return window.__t184fills.map((f) => f.getBoundingClientRect().width);
}
"""
GROW_READ_JS = (
    "() => ({widths: window.__t184fills.map((f) => f.getBoundingClientRect().width),"
    " connected: window.__t184fills.map((f) => f.isConnected), writes: window.__t184fillw})"
)
ERRORS_INIT_JS = (
    "window.__t184errors = []; window.addEventListener('error', (e) =>"
    " window.__t184errors.push(String(e.message || e)));"
)


@dataclass
class RunR1:
    error: str = ""
    sids: dict[str, str] = field(default_factory=dict)
    texts: dict[str, Any] = field(default_factory=dict)
    scales: dict[str, dict[str, Any]] = field(default_factory=dict)
    b6b: dict[str, list[str]] = field(default_factory=dict)
    grow: dict[str, Any] = field(default_factory=dict)
    resize_errors: list[str] = field(default_factory=list)
    b7c_slot: dict[str, Any] = field(default_factory=dict)
    b3a: dict[str, Shot] = field(default_factory=dict)
    b5b: list[dict[str, Any]] = field(default_factory=list)
    rowbars: dict[str, Any] = field(default_factory=dict)
    b9e: dict[str, Any] = field(default_factory=dict)


def _tick_idx(texts: list[dict[str, Any]], key: str) -> list[int]:
    ticks = _tick_texts(texts, _bars_of(key))
    return [i for i, t in enumerate(texts) if any(t is x for x in ticks)]


def _run_r1(run: RunR1, p: Any) -> None:
    all_surfaces = surfaces()
    browser = p.chromium.launch()
    stub = HudStub([]).start()
    server = HudStaticServer(HUD, 0)
    console: list[str] = []
    try:
        context = browser.new_context(
            viewport={"width": VIEW[0], "height": VIEW[1]},
            device_scale_factor=1,
            color_scheme="dark",
            reduced_motion="reduce",
        )
        context.add_init_script(ERRORS_INIT_JS)
        page = context.new_page()
        page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: console.append(str(e)))
        page.goto(
            f"http://127.0.0.1:{server.port}/?source=canvas&ws="
            + urllib.parse.quote(stub.url, safe="")
        )
        if not stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20):
            raise RuntimeError("the page never said astra.hello")
        stub.wait_for_client(10)
        page.wait_for_function(
            f"document.querySelectorAll('main > .pane').length >= {PANES}", timeout=25_000
        )
        page.add_style_tag(content=NEUTRALISE_CSS)
        page.mouse.move(0, 0)
        keys = sorted(set(B9A_KEYS) | set(TICK_KEYS))
        for key in keys:
            stub.push(all_surfaces[key])
            run.sids[key] = all_surfaces[key].surface_id
            time.sleep(0.02)
        extra = {
            GROW_SID: _v2(
                GROW_SID, [{"id": "root", "component": "Bars", "unit": "m", "bars": GROW_FIRST}]
            ),
            B3A_SID: _v2(B3A_SID, B3A_COMPONENTS),
            **{sid: _v2(sid, case[0]) for sid, case in ROWBARS_CASES.items()},
        }
        for sid, msgs in extra.items():
            stub.push(
                AstraUiSurface(surface_id=sid, canvas_idx=0, final=True, messages=tuple(msgs))
            )
            run.sids[sid] = sid
        # B9e: the generated cross product, one card each
        for sid, case in B9E_CASES.items():
            stub.push(
                AstraUiSurface(
                    surface_id=sid, canvas_idx=0, final=True, messages=tuple(_v2(sid, case[0]))
                )
            )
            time.sleep(0.02)
        page.wait_for_function(
            "(ids) => ids.every((id) => !!document.querySelector('[data-ui-surface=\"' + id"
            ' + \'"] [data-a2ui-root] [data-a2-id="eb"]\'))',
            arg=list(B9E_CASES),
            timeout=20_000,
        )
        time.sleep(SETTLE_S)
        _theme(page, "system-dark")
        # B9a / B7b / B7c at every view
        for w, h in LAYOUT_VIEWS:
            page.set_viewport_size({"width": w, "height": h})
            page.evaluate(TWO_FRAMES_JS)
            time.sleep(0.25)
            view = f"{w}x{h}"
            run.texts[view] = page.evaluate(R1_TEXTS_JS, {k: run.sids[k] for k in keys})
            run.rowbars[view] = {
                sid: page.evaluate(ROWBARS_JS, [sid, case[1], case[2]])
                for sid, case in ROWBARS_CASES.items()
            }
            if view in B9E_VIEWS:
                run.b9e[view] = {
                    sid: page.evaluate(ROWBARS_JS, [sid, c[1], c[2]])
                    for sid, c in B9E_CASES.items()
                }
            ticks = page.evaluate(TICKS_JS, {k: run.sids[k] for k in B7B_KEYS})
            run.scales[view] = {}
            for k in B7B_KEYS:
                idx = _tick_idx(ticks[k]["texts"], k) if ticks.get(k) else []
                run.scales[view][k] = {
                    "ticks": [ticks[k]["texts"][i]["text"] for i in idx] if ticks.get(k) else [],
                    **(page.evaluate(R1_SCALE_JS, [run.sids[k], idx]) or {}),
                }
        # B6b: identical re-sends after layout, at 1440 (the fit thins the 1e12 ticks), 1000, 420
        surface = all_surfaces[B6B_KEY]
        idx_next = 1
        for w in (1440, 1000, 420):
            page.set_viewport_size({"width": w, "height": VIEW[1]})
            page.evaluate(TWO_FRAMES_JS)
            time.sleep(0.4)
            page.evaluate(OBSERVE_JS, surface.surface_id)
            for k in range(3):
                stub.push(surface.model_copy(update={"canvas_idx": idx_next, "final": k == 2}))
                idx_next += 1
                time.sleep(0.25)
            page.evaluate(TWO_FRAMES_JS)
            time.sleep(0.3)
            run.b6b[str(w)] = page.evaluate("() => window.__t184muts.slice()")
        page.evaluate("() => window.__t184mo && window.__t184mo.disconnect()")
        # B6a: grow by a smaller value with a longer string, at 1440
        page.set_viewport_size({"width": VIEW[0], "height": VIEW[1]})
        page.evaluate(TWO_FRAMES_JS)
        time.sleep(0.4)
        ticks0 = page.evaluate(TICKS_JS, {GROW_SID: GROW_SID})[GROW_SID]["texts"]
        grow_bars = {"unit": "m", "bars": GROW_FIRST}
        idx0 = [
            i for i, t in enumerate(ticks0) if any(t is x for x in _tick_texts(ticks0, grow_bars))
        ]
        before_scale = page.evaluate(R1_SCALE_JS, [GROW_SID, idx0])
        before_w = page.evaluate(GROW_MARK_JS, GROW_SID)
        stub.push(
            AstraUiSurface(
                surface_id=GROW_SID,
                canvas_idx=1,
                final=True,
                messages=(
                    {
                        "version": "v0.9.1",
                        "updateComponents": {
                            "surfaceId": GROW_SID,
                            "components": [
                                {
                                    "id": "root",
                                    "component": "Bars",
                                    "unit": "m",
                                    "bars": [*GROW_FIRST, GROW_EXTRA],
                                }
                            ],
                        },
                    },
                ),
            )
        )
        time.sleep(0.5)
        page.evaluate(TWO_FRAMES_JS)
        ticks1 = page.evaluate(TICKS_JS, {GROW_SID: GROW_SID})[GROW_SID]["texts"]
        grow_bars1 = {"unit": "m", "bars": [*GROW_FIRST, GROW_EXTRA]}
        idx1 = [
            i for i, t in enumerate(ticks1) if any(t is x for x in _tick_texts(ticks1, grow_bars1))
        ]
        run.grow = {
            "before_w": before_w,
            "before_scale": before_scale,
            "before_ticks": [ticks0[i]["text"] for i in idx0],
            "after": page.evaluate(GROW_READ_JS),
            "after_scale": page.evaluate(R1_SCALE_JS, [GROW_SID, idx1]),
            "after_texts": [t["text"] for t in ticks1],
        }
        # B3a: the backdrop of each column, both system themes
        sampler = context.new_page()
        for state in LOOK_STATES:
            _theme(page, state)
            shot = _shoot(page, sampler, B3A_SID, [])
            if shot is not None:
                run.b3a[state] = shot
        _theme(page, "system-dark")
        # B7c': the 1e12 card in a slot narrowed in-page (ruling ad799aff), at 1440x900
        page.set_viewport_size({"width": VIEW[0], "height": VIEW[1]})
        page.evaluate(TWO_FRAMES_JS)
        hsid = run.sids[B7C_KEY]
        page.evaluate(SCROLL_JS, hsid)
        run.b7c_slot = {"card": page.evaluate(SLOT_JS, [hsid, B7C_SLOT_PX])}
        page.evaluate(TWO_FRAMES_JS)
        time.sleep(0.4)
        page.evaluate(TWO_FRAMES_JS)
        bt = page.evaluate(TICKS_JS, {B7C_KEY: hsid})[B7C_KEY]["texts"]
        bidx = _tick_idx(bt, B7C_KEY)
        run.b7c_slot.update(
            ticks=[bt[i]["text"] for i in bidx], **(page.evaluate(R1_SCALE_JS, [hsid, bidx]) or {})
        )
        # B7c: a resize sweep of that slot through the bare state and back, then of the viewport;
        # then the window errors and console errors seen in the whole session
        for w in B7C_SLOT_SWEEP:
            page.evaluate(SLOT_JS, [hsid, w])
            page.evaluate(TWO_FRAMES_JS)
            time.sleep(0.08)
        page.evaluate(SLOT_JS, [hsid, None])
        for w in RESIZE_SWEEP:
            page.set_viewport_size({"width": w, "height": VIEW[1]})
            page.evaluate(TWO_FRAMES_JS)
            time.sleep(0.12)
        time.sleep(0.3)
        run.resize_errors = list(page.evaluate("() => window.__t184errors.slice()")) + console
        # B5b last: the newest card, one click on the Actions inside a gate-admitted Button
        page.set_viewport_size({"width": VIEW[0], "height": VIEW[1]})
        stub.push(
            AstraUiSurface(
                surface_id=B5B_SID,
                canvas_idx=0,
                final=True,
                messages=tuple(_v2(B5B_SID, B5B_COMPONENTS)),
            )
        )
        page.wait_for_function(
            f'!!document.querySelector(\'[data-ui-surface="{B5B_SID}"] [data-a2-kind="Actions"] button\')',
            timeout=10_000,
        )
        page.evaluate(SCROLL_JS, B5B_SID)
        n0 = len(_actions(stub))
        page.locator(f'[data-ui-surface="{B5B_SID}"] [data-a2-kind="Actions"] button').first.click()
        stub.wait_for(lambda s: len(_actions(s)) >= n0 + 1, 10)
        time.sleep(0.6)
        run.b5b = _actions(stub)[n0:]
        context.close()
    finally:
        browser.close()
        server.close()
        stub.stop()


@pytest.fixture(scope="module")
def genui_r1(genui: Run) -> Iterator[RunR1]:
    """The round-1 rows' Chromium session, after ``genui``'s has closed. Never raises."""
    why = _playwright_chromium()
    if why:
        pytest.skip(why)
    from playwright.sync_api import sync_playwright

    run = RunR1()
    try:
        with sync_playwright() as p:
            _run_r1(run, p)
    except Exception as exc:  # a broken session fails in a row, not here
        import traceback

        where = traceback.extract_tb(exc.__traceback__)
        inside = [f for f in where if f.filename == __file__]
        line = f" (at line {inside[-1].lineno}: {inside[-1].line})" if inside else ""
        run.error = f"{type(exc).__name__}: {exc}{line}"
    yield run


def _ok_r1(run: RunR1) -> None:
    assert not run.error, f"the T-0184 B round-1 browser session failed: {run.error}"


def test_r1_premise_the_round_1_payloads_validate_against_the_v2_catalog():
    """The B3a, B5b and B6a payloads are well-formed v2 surfaces (group A's validator), so a
    failure below is the renderer's. B5b's Button wrapping an Actions is admitted."""
    from local_astra.ui import a2ui_schema

    validator = a2ui_schema.A2UIValidator(catalog=a2ui_schema.load_cards_v2_catalog())
    grown = [{"id": "root", "component": "Bars", "unit": "m", "bars": [*GROW_FIRST, GROW_EXTRA]}]
    for sid, comps in (
        (B3A_SID, B3A_COMPONENTS),
        (B5B_SID, B5B_COMPONENTS),
        (GROW_SID, grown),
        *((sid, case[0]) for sid, case in ROWBARS_CASES.items()),
    ):
        report = validator.validate_messages(_v2(sid, comps), final=True)
        assert report.ok, f"{sid}: {report}"


def test_b6b_an_identical_canvas_after_layout_makes_zero_mutations_in_chrome(genui_r1):
    """B6b: a MutationObserver on the 1e12 card's root; three identical canvases re-sent after
    layout at 1440 px (the fit thins the ticks), 1000 px and 420 px: 0 mutation records each."""
    _ok_r1(genui_r1)
    assert set(genui_r1.b6b) == {"1440", "1000", "420"}, genui_r1.b6b
    bad = {w: (len(m), m[:6]) for w, m in genui_r1.b6b.items() if m}
    assert bad == {}, f"T-0184 B6b: mutations on identical re-sends (width: count, first): {bad}"


def _derived_nodes(texts: list[dict[str, Any]], key: str) -> list[tuple[str, list[dict[str, Any]]]]:
    """B9a: (role, the text nodes that must share one line) for each derived number of card
    ``key``, read from its payload: a Bars value with its unit, a tick, a tile percent (one node,
    or "<p>" then "%"), a step ordinal."""
    comps = [
        c
        for m in surfaces()[key].messages
        for c in (m.get("updateComponents") or {}).get("components") or []
    ]
    out: list[tuple[str, list[dict[str, Any]]]] = []
    bars = [t for t in texts if t["kind"] == "Bars"]
    for c in comps:
        if c.get("component") == "Bars":
            labels = [str(b["label"]) for b in c["bars"]]
            values = [_js_string(b["value"]) for b in c["bars"]]
            unit = c.get("unit")
            k = 0
            i = 0
            while i < len(bars):
                if k < len(labels) and bars[i]["text"] == labels[k]:
                    j = next(
                        (x for x in range(i + 1, len(bars)) if bars[x]["text"] == values[k]), None
                    )
                    if j is not None:
                        group = [bars[j]]
                        if unit is not None and j + 1 < len(bars) and bars[j + 1]["text"] == unit:
                            group.append(bars[j + 1])
                        out.append((f"value {values[k]}", group))
                    k += 1
                i += 1
            for t in _tick_texts(bars, c):
                out.append((f"tick {t['text']}", [t]))
        if c.get("component") == "StatTiles":
            tiles = [t for t in texts if t["kind"] == "StatTiles"]
            for tile in c["tiles"]:
                if "percent" not in tile:
                    continue
                p = _js_string(tile["percent"])
                for i, t in enumerate(tiles):
                    if t["text"] == p + "%":
                        out.append((f"percent {p}%", [t]))
                    elif t["text"] == p and i + 1 < len(tiles) and tiles[i + 1]["text"] == "%":
                        out.append((f"percent {p}%", [t, tiles[i + 1]]))
        if c.get("component") == "Steps":
            steps = [t for t in texts if t["kind"] == "Steps"]
            titles = [str(s["title"]) for s in c["steps"]]
            for i, t in enumerate(steps):
                if i + 1 < len(steps) and steps[i + 1]["text"] in titles and t["text"].isdigit():
                    out.append((f"ordinal {t['text']}", [t]))
    return out


@pytest.mark.parametrize("view", R1_VIEWS)
def test_b9a_no_derived_number_splits_across_lines(genui_r1, view):
    """B9a, at every B9 view: each ``String(value)`` with its unit, each tick, each percent and
    each ordinal sits on one line (one distinct top over its text nodes' client rects)."""
    _ok_r1(genui_r1)
    texts = genui_r1.texts.get(view) or {}
    problems, seen = [], 0
    for key in B9A_KEYS:
        if not texts.get(key):
            problems.append(f"{key}: not measured")
            continue
        for role, group in _derived_nodes(texts[key], key):
            seen += 1
            tops = sorted({top for t in group for top in t["tops"]})
            if len(tops) != 1 and not (len(tops) == 2 and tops[1] - tops[0] <= 1):
                problems.append(f"{key}: {role} {[t['text'] for t in group]} on lines {tops}")
    # 39 at 420 px (no ticks there), more where ticks are drawn
    assert seen >= 30, f"premise: {seen} derived numbers found"
    assert problems == [], f"T-0184 B9a at {view}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("view", R1_VIEWS)
def test_b7b_bars_are_drawn_to_the_tick_scale(genui_r1, view):
    """B7b: with ticks drawn, each fill's width over the scale's equals value over the last tick
    (within 1 px); with no tick drawn, the longest fill fills the scale (within 1 px)."""
    _ok_r1(genui_r1)
    problems = []
    for key in B7B_KEYS:
        s = (genui_r1.scales.get(view) or {}).get(key) or {}
        if not s.get("scale") or not s.get("fills"):
            problems.append(f"{key}: no scale or fills located ({s.get('how')})")
            continue
        values = [float(b["value"]) for b in _bars_of(key)["bars"]]
        widths = [f[2] - f[0] for f in s["fills"]]
        sw = s["scale"][2] - s["scale"][0]
        if len(widths) != len(values):
            problems.append(f"{key}: {len(widths)} fills for {len(values)} bars")
            continue
        if len(s["ticks"]) >= 2:
            last = float(s["ticks"][-1])
            for v, w in zip(values, widths, strict=True):
                if abs(w - sw * v / last) > 1.0:
                    problems.append(
                        f"{key}: fill {w:.1f} px for {v:g} on a {sw:.1f} px scale ending at "
                        f"{s['ticks'][-1]}: want {sw * v / last:.1f}"
                    )
                    break
        elif abs(max(widths) - sw) > 1.0:
            problems.append(
                f"{key}: no ticks; the longest fill {max(widths):.1f} px vs the scale {sw:.1f} px"
            )
    assert problems == [], f"T-0184 B7b at {view}:\n  " + "\n  ".join(problems)


def test_b7c_the_axis_keeps_its_height_and_no_resize_observer_loop(genui_r1):
    """B7c / B7c': the 1e12 card's axis (the scale element) has the same height with labels
    (1440x900) and bare (its slot narrowed in-page to 120 px, ruling ad799aff, where it draws no
    tick: a premise, so the bare branch runs), within 1 px; a sweep of that slot's width through
    the bare state and back, then a viewport sweep, raise no "ResizeObserver loop" error."""
    _ok_r1(genui_r1)
    wide = genui_r1.scales[B7C_LABELLED][B7C_KEY]
    narrow = genui_r1.b7c_slot
    assert narrow.get("card"), f"premise: the card's slot was not found ({narrow})"
    (cl, cr) = narrow["card"]
    assert cr - cl <= B7C_SLOT_PX + 0.5, (
        f"premise: the card is {cr - cl:.1f} px in the narrowed slot"
    )
    assert wide.get("scale") and narrow.get("scale"), (wide, narrow)
    assert len(wide["ticks"]) >= 2, f"premise: ticks are drawn at {B7C_LABELLED} ({wide['ticks']})"
    assert narrow["ticks"] == [], (
        f"premise (B7c'): no tick is drawn in the {B7C_SLOT_PX} px slot ({narrow['ticks']}), "
        "so the bare branch runs"
    )
    hw, hn = wide["scale"][3] - wide["scale"][1], narrow["scale"][3] - narrow["scale"][1]
    problems = []
    if abs(hw - hn) > 1.0:
        problems.append(f"the axis is {hw:.1f} px tall with labels, {hn:.1f} px bare")
    loops = [e for e in genui_r1.resize_errors if "ResizeObserver loop" in e]
    if loops:
        problems.append(f"{len(loops)} ResizeObserver loop error(s): {loops[:2]}")
    assert problems == [], "T-0184 B7c:\n  " + "\n  ".join(problems)


def _intersects(a: list[float], b: list[float]) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > 0.5 and min(a[3], b[3]) - max(a[1], b[1]) > 0.5


def _value_unit_split(texts: list[dict[str, Any]], bars: dict[str, Any]) -> list[str]:
    """B9a on a Bars' text nodes (in order): each ``String(value)`` and the unit after it share
    one line (one distinct top, a 1 px rounding step allowed)."""
    labels = [str(b["label"]) for b in bars["bars"]]
    values = [_js_string(b["value"]) for b in bars["bars"]]
    unit, out, k = bars.get("unit"), [], 0
    for i, t in enumerate(texts):
        if k < len(labels) and t["text"] == labels[k]:
            j = next((x for x in range(i + 1, len(texts)) if texts[x]["text"] == values[k]), None)
            if j is not None:
                group = [texts[j]]
                if unit is not None and j + 1 < len(texts) and texts[j + 1]["text"] == unit:
                    group.append(texts[j + 1])
                tops = sorted({top for g in group for top in g["tops"]})
                if len(tops) != 1 and not (len(tops) == 2 and tops[1] - tops[0] <= 1):
                    out.append(f"value {values[k]} {[g['text'] for g in group]} on lines {tops}")
            k += 1
    return out


@pytest.mark.parametrize("view", R1_VIEWS)
@pytest.mark.parametrize("case", list(ROWBARS_CASES))
def test_b9b_a_bars_in_a_row_gets_a_definite_share(genui_r1, case, view):
    """B9b / B9b', at every B9 view: Row[Bars{4321098765432, 7}, Stat, Stat],
    Row[Column[Text, Bars], Stat] and Row[Card[Bars], Stat] on a v2 surface. No two siblings
    overlap (each sibling is its element box plus its text's rects), no text spills outside its
    sibling's box or the card, the longest fill is wider than 0, and (B9a) each value and its unit
    share one line."""
    _ok_r1(genui_r1)
    got = (genui_r1.rowbars.get(view) or {}).get(case)
    assert got, f"{view}: the Row card {case} was not measured"
    problems = []
    bars_comp = next(c for c in ROWBARS_CASES[case][0] if c["id"] == ROWBARS_CASES[case][2])
    problems += _value_unit_split(got["texts"], bars_comp)
    kids = got["kids"]
    if any(v is None for v in kids.values()):
        problems.append(f"a sibling did not render: {[k for k, v in kids.items() if v is None]}")
    else:
        names = list(kids)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                hits = [(a, b) for a in kids[names[i]] for b in kids[names[j]] if _intersects(a, b)]
                if hits:
                    problems.append(f"{names[i]} overlaps {names[j]}: e.g. {hits[0]}")
        (rl, _rt, rr, _rb) = got["root"]
        out = [(k, r) for k, rs in kids.items() for r in rs if r[0] < rl - 0.5 or r[2] > rr + 0.5]
        if out:
            problems.append(f"{len(out)} box(es) outside the card {rl:.1f}..{rr:.1f}: {out[:2]}")
        # the definite share (B9b'): the Row child holding the Bars takes the Row's free space, so
        # the children reach the card's content edge; the edge comes from the card frame's
        # computed border and padding (B9e), never from a sibling under test
        edge = got["content"][1]
        reach = max(rs[0][2] for rs in kids.values())
        if reach < edge - 1.0:
            problems.append(
                f"the root's children end at {reach:.1f}, short of the content edge "
                f"{edge:.1f}: the Bars' child has no share (Bars {got['bars_w']} px)"
            )
        # a sibling's text stays inside its own element's columns (no spill into a neighbour's)
        for k, rs in kids.items():
            (el, _t, er, _b) = rs[0]
            spill = [r for r in rs[1:] if r[0] < el - 0.5 or r[2] > er + 0.5]
            if spill:
                problems.append(f"{k}: text spills outside its box {el:.1f}..{er:.1f}: {spill[:2]}")
    if not got["fills"] or max(got["fills"]) <= 0:
        problems.append(
            f"the longest fill is {max(got['fills'], default=0):.1f} px wide "
            f"(the Bars is {got['bars_w']} px)"
        )
    assert problems == [], f"T-0184 B9b {case} at {view}:\n  " + "\n  ".join(problems)


def b9e_problems(got: dict[str, Any] | None, bars: dict[str, Any]) -> list[str]:
    """B9e's oracles on one generated card: the longest fill wider than 0; no two leaves (the
    Bars and each sibling: element box plus text rects) overlap; no overflow (every leaf box and
    text rect inside the card, or, when a sideways scroller inside the card holds the leaf, that
    scroller inside the card; each leaf's text inside its own element's columns); and B9a (each
    value and its unit on one line)."""
    if not got:
        return ["not measured"]
    out = []
    kids = got["kids"]
    missing = [k for k, v in kids.items() if v is None]
    if missing:
        return [f"leaves that did not render: {missing}"]
    if not got["fills"] or max(got["fills"]) <= 0:
        out.append(
            f"the longest fill is {max(got['fills'], default=0):.1f} px (Bars {got['bars_w']} px)"
        )
    names = list(kids)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            hits = [(a, b) for a in kids[names[i]] for b in kids[names[j]] if _intersects(a, b)]
            if hits:
                out.append(f"{names[i]} overlaps {names[j]}: e.g. {hits[0]}")
    (rl, _rt, rr, _rb) = got["root"]
    for k, rs in kids.items():
        held = got["held"].get(k)
        outer = [held] if held else rs
        bad = [r for r in outer if r[0] < rl - 0.5 or r[2] > rr + 0.5]
        if bad:
            out.append(f"{k}: outside the card {rl:.1f}..{rr:.1f}: {bad[:2]}")
        (el, _t, er, _b) = rs[0]
        spill = [r for r in rs[1:] if r[0] < el - 0.5 or r[2] > er + 0.5]
        if spill:
            out.append(f"{k}: text spills outside its box {el:.1f}..{er:.1f}: {spill[:2]}")
    out += _value_unit_split(got["texts"], bars)
    return out


def test_b9e_premise_the_generated_cases_cover_the_cross_product_and_validate():
    """B9e premise: 8 containers at depth 1 and 8 x 8 at depth 2 (72 cards), each a valid v2
    surface (group A's validator), each with one Bars."""
    from local_astra.ui import a2ui_schema

    assert len(B9E_CASES) == 8 + 8 * 8
    validator = a2ui_schema.A2UIValidator(catalog=a2ui_schema.load_cards_v2_catalog())
    for sid, (comps, leaves, bars_id) in B9E_CASES.items():
        assert [c["component"] for c in comps].count("Bars") == 1 and bars_id in leaves, sid
        report = validator.validate_messages(_v2(sid, comps), final=True)
        assert report.ok, f"{sid}: {report}"


@pytest.mark.parametrize("view", B9E_VIEWS)
@pytest.mark.parametrize("case", list(B9E_CASES))
def test_b9e_a_bars_in_any_parent_shape_keeps_its_width(genui_r1, case, view):
    """B9e (ruling 7f3562a9): every generated parent shape around a Bars (Row, Column{start,
    center, end, stretch}, List{vertical, horizontal}, Card; depth 1 and 2; a Stat or Text
    sibling), at 420, 580 and 1440 px: the B9b oracles and B9a (``b9e_problems``)."""
    _ok_r1(genui_r1)
    got = (genui_r1.b9e.get(view) or {}).get(case)
    problems = b9e_problems(got, _B9E_BARS)
    assert problems == [], f"T-0184 B9e {case} at {view}:\n  " + "\n  ".join(problems)


CARDS_V2_JS = HUD / "src" / "ui" / "cards_v2.js"
CARD_CSS_PATH = HUD / "src" / "ui" / "card.css"


def narrow_threshold_problems(js: str, css: str) -> list[str]:
    """B9c: ``NARROW_BARS`` in cards_v2.js, every ``@container (max-width: Npx)`` in card.css and
    every px literal subtracted from ``100cqi`` (the cqi gate, in either file) are one number.
    Comments are stripped first (a comment can name the number in prose)."""
    js_code = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", " ", js)
    css_code = re.sub(r"/\*[\s\S]*?\*/", " ", css)
    m = re.findall(r"\bconst\s+NARROW_BARS\s*=\s*(\d+(?:\.\d+)?)\s*;", js_code)
    if len(m) != 1:
        return [f"cards_v2.js declares NARROW_BARS {len(m)} times (want one numeric const)"]
    want = float(m[0])
    out = []
    queries = re.findall(r"@container[^{]*?max-width:\s*(\d+(?:\.\d+)?)px", css_code)
    if not queries:
        out.append("card.css has no @container (max-width: …px) query")
    out += [
        f"card.css @container max-width {q}px != NARROW_BARS {m[0]}"
        for q in queries
        if float(q) != want
    ]
    gates = re.findall(r"100cqi\s*-\s*(\d+(?:\.\d+)?)px", css_code + "\n" + js_code)
    out += [
        f"a 100cqi gate subtracts {g}px != NARROW_BARS {m[0]}" for g in gates if float(g) != want
    ]
    named = re.findall(r"100cqi\s*-\s*\$\{\s*([A-Za-z_]\w*)\s*\}px", js_code)
    if not gates and not named:
        out.append("no 100cqi gate found in card.css or cards_v2.js")
    out += [
        f"the 100cqi gate subtracts ${{{n}}}px, not NARROW_BARS"
        for n in named
        if n != "NARROW_BARS"
    ]
    return out


def test_b9c_premise_the_threshold_check_by_hand():
    """The B9c check on hand-written sources: one number passes; each drift (JS or CSS to 300 or
    400, a literal cqi gate that disagrees, another name in the gate) fails."""
    js = "const NARROW_BARS = 340;\nconst room = `max(0px, (100cqi - ${NARROW_BARS}px) * 1000)`;"
    css = ".x { container-type: inline-size; }\n@container (max-width: 340px) { .y { a: b; } }"
    assert narrow_threshold_problems(js, css) == []
    for bad_js, bad_css in (
        (js.replace("= 340", "= 300"), css),
        (js.replace("= 340", "= 400"), css),
        (js, css.replace("340px", "300px")),
        (js, css.replace("340px", "400px")),
        (js.replace("${NARROW_BARS}px", "300px"), css),
        (js.replace("${NARROW_BARS}", "${OTHER}"), css),
        (js, css + "\n.z { width: calc(100cqi - 400px); }"),
    ):
        assert narrow_threshold_problems(bad_js, bad_css) != [], (bad_js, bad_css)
    # prose in a comment does not count
    assert narrow_threshold_problems(js + "\n// was 300 once", css + "/* 400px */") == []


def test_b9c_one_narrow_threshold_in_the_renderer_and_the_sheet():
    """B9c: ``NARROW_BARS`` in hud/src/ui/cards_v2.js equals the px of card.css's container
    query and of the ``100cqi`` gate."""
    problems = narrow_threshold_problems(
        CARDS_V2_JS.read_text(encoding="utf-8"), CARD_CSS_PATH.read_text(encoding="utf-8")
    )
    assert problems == [], "T-0184 B9c:\n  " + "\n  ".join(problems)


def test_b6a_growing_by_a_smaller_value_with_a_longer_string_rewrites_no_fill(genui_r1):
    """B6a, in Chrome at 1440 px: Bars 40, 100 grown by 12.75 (a smaller value, a longer string):
    no attribute write on an old fill, the old fills keep their widths, and the scale (the ticks'
    box) keeps its width."""
    _ok_r1(genui_r1)
    g = genui_r1.grow
    assert g and g["before_w"] and len(g["before_w"]) == 2, f"premise: two fills before ({g})"
    assert len(g["before_ticks"]) >= 2, (
        f"premise: ticks drawn before the growth ({g['before_ticks']})"
    )
    assert "Gamma" in g["after_texts"], "the new bar rendered"
    problems = []
    if not all(g["after"]["connected"]):
        problems.append("an old fill was replaced")
    if g["after"]["writes"]:
        problems.append(f"old fills rewritten: {g['after']['writes']}")
    if any(abs(a - b) > 0.5 for a, b in zip(g["before_w"], g["after"]["widths"], strict=True)):
        problems.append(f"old fill widths {g['before_w']} -> {g['after']['widths']}")
    bs, as_ = g["before_scale"]["scale"], (g["after_scale"] or {}).get("scale")
    if not as_ or abs((bs[2] - bs[0]) - (as_[2] - as_[0])) > 0.5:
        problems.append(f"the ticks box width {bs[2] - bs[0]:.1f} -> {as_ and as_[2] - as_[0]}")
    assert problems == [], "T-0184 B6a:\n  " + "\n  ".join(problems)


def test_b3a_an_out_of_range_highlight_with_extra_cells_tints_no_column(genui_r1):
    """B3a: two headers, ``highlight: 2`` and three cells a row: the extra column is not tinted;
    every cell of a row has the same backdrop (dE00 <= 0.8), both system themes."""
    _ok_r1(genui_r1)
    problems = []
    for state in LOOK_STATES:
        shot = genui_r1.b3a.get(state)
        assert shot is not None, f"B3a not measured in {state}"
        for row in (("aa1", "bb1", "xx1"), ("aa2", "bb2", "xx2")):
            backs = [_behind(shot, s) for s in row]
            for s, b in zip(row[1:], backs[1:], strict=True):
                if delta_e(backs[0], b) > SAME_DE:
                    problems.append(f"{state}: {s} {hexc(b)} vs {row[0]} {hexc(backs[0])}")
    assert problems == [], "T-0184 B3a (a column is tinted):\n  " + "\n  ".join(problems)


def test_b5b_one_click_on_an_actions_inside_a_gate_button_sends_one_action(genui_r1):
    """B5b: one click on the Actions button inside a gate-admitted Button sends exactly one
    ``astra.ui.action``, ``__gate.clicked`` with the Actions label (the outer Button's own
    ``__gate.clicked`` is not sent too)."""
    _ok_r1(genui_r1)
    got = [(a.get("name"), (a.get("context") or {}).get("label")) for a in genui_r1.b5b]
    assert got == [("__gate.clicked", B5B_INNER)], f"T-0184 B5b: one click sent {got}"


# --- B11a: the fixed-ref row --------------------------------------------------------------------

#: B's base (the main tip T-0184 B branched from) and the implementation tip under review. The
#: coordinator replaces ``B11_TIP`` with the merged tip when B lands (closed range, T-0094).
B11_BASE = "e45f39a5"
B11_TIP = "c9cc8d4d"  # the last implementation commit (round 8, fast-forwarded T-0184-b-renderer)
B11_PATHS = (
    "hud/styles.css",
    "hud/src/ui/bridge.js",
    "hud/src/ui/action.js",
    "hud/src/ui/fence.js",
    "hud/src/panels",
    "hud/src/main.js",
    "hud/index.html",
    "hud/src/protocol.js",
    "local_astra",
    "tests/fakes",
)


def _b11_changed(base: str, tip: str) -> list[str]:
    from tests import _git_range as gr

    return gr.changed(base, tip, *B11_PATHS)


def test_b11a_the_byte_identical_paths_did_not_change_over_base_to_tip():
    """B11a: none of B11's byte-identical paths changed over the closed range BASE..TIP (literal
    shas); BASE is an ancestor of TIP. Skips (never passes) when a commit is absent."""
    from tests import _git_range as gr

    gr.require_commits(B11_BASE, B11_TIP)
    assert gr.is_ancestor(B11_BASE, B11_TIP), f"{B11_BASE} is not an ancestor of {B11_TIP}"
    assert _b11_changed(B11_BASE, B11_TIP) == []


def test_b11a_premise_the_range_check_sees_a_change_under_local_astra():
    """Adversarial premise: over group A's implementation commit (2f587843, which added the v2
    catalog under local_astra/ui), the same check reports the change, so B11a is not vacuous."""
    from tests import _git_range as gr

    gr.require_commits("2f587843")
    got = _b11_changed("2f587843~1", "2f587843")
    assert any(p.startswith("local_astra/ui/") for p in got), got


# =========================================================================================
# B10: the screenshot script
# =========================================================================================


@dataclass
class Screens:
    error: str = ""
    out: Path | None = None
    stdout: str = ""


@pytest.fixture(scope="module")
def screens(genui: Run, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Screens]:
    """``hud_screens.py --out DIR --genui-v2``, run in this process after the session's browser
    has closed (``genui`` is a dependency, so the two Chromiums never overlap)."""
    why = _playwright_chromium()
    if why:
        pytest.skip(why)
    result = Screens(out=tmp_path_factory.mktemp("t0184b-shots"))
    spec = importlib.util.spec_from_file_location("_t0184b_hud_screens", SCREENS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    try:
        spec.loader.exec_module(module)
        with contextlib.redirect_stdout(buf):
            code = module.main(["--out", str(result.out), "--genui-v2"])
        if code:
            result.error = f"hud_screens.py returned {code}"
    except SystemExit as exc:
        result.error = f"hud_screens.py exited: SystemExit({exc.code!r})"
    except Exception as exc:
        result.error = f"hud_screens.py raised {type(exc).__name__}: {exc}"
    result.stdout = buf.getvalue()
    yield result


def test_b10_the_script_writes_the_v2_composites_cropped_to_the_cards(screens):
    """B10: ``genui-v2-{dark,light}.png`` are written and listed on stdout, cropped to the cards
    (narrower than two 1440 px pages), tall enough for every exemplar kind and the three boards
    (ten card rows of at least 60 px), and the run's requests stayed on loopback."""
    assert not screens.error, f"B10: {screens.error}\n{screens.stdout[-2000:]}"
    assert screens.out is not None
    for theme in ("dark", "light"):
        path = screens.out / f"genui-v2-{theme}.png"
        assert path.is_file() and path.stat().st_size > 0, f"B10: {path.name} was not written"
        assert str(path) in screens.stdout, f"B10: {path.name} is not listed on stdout"
        w, h = _png_size(path.read_bytes())
        assert 0 < w < 2 * VIEW[0] and h >= 10 * 60, f"B10: {path.name} is {w}x{h}"
    req = screens.out / "requests.txt"
    if req.is_file():
        off = [
            u
            for u in req.read_text(encoding="utf-8").split()
            if urllib.parse.urlsplit(u).hostname not in {"127.0.0.1", "localhost", "::1", None}
        ]
        assert off == [], off


_ = FIND_THEME_CONTROL_JS  # imported for choose_theme_js's contract; kept for readers
