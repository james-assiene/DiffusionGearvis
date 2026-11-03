"""T-0056 criteria 6-13 and 18 — the HUD chrome in headless Chrome, with no socket at all.

The sibling modules ``tests/hud/shell.test.mjs`` and ``tests/hud/styles.test.mjs`` read the shell
and the sheet as text. This one asks the questions only a renderer can answer: what a colour
*computes* to once four stylesheets and a custom-property chain have had their say, what a grid
does when a seventh pane mounts itself, whether a state that is styled is actually distinguishable
from the five next to it, and whether a token binding has painted a component of the model's
surface the colour of the thing behind it.

**No WebSocket.** The page is opened with ``?source=canvas`` (the synthetic capture provider) and a
``?ws=`` pointing at a port nothing is listening on, so the socket sits in reconnect backoff for
the whole run. That is the deterministic state the task's *Regime* names: six panes mount (``ui``,
``assists``, ``ask``, ``log`` plus the four declared ones minus… see ``PANE_COUNT`` below), no
message ever arrives, and nothing on the page is timing-dependent. Group B's rows are the one
exception: they need a surface, so they run a second page session against the same ``HudStub``
``scripts/demo/hud_ui_demo.py`` uses, fed the real quiz fixture through the real ``UiGate``.
T-0154 C4 adds a third session of the same shape (``card_page``: the gated t9 card and a synthetic
card, measured inside each surface's shadow root under each data-theme).

**The contrast oracle is Python.** Chrome is asked for ``rgb()``/``rgba()`` strings, font sizes,
weights, opacities and the background chain; WCAG 2.1 relative luminance and the ratio are
computed here, by hand, and unit-tested against two pairs whose ratios are known from the
specification's own examples (black on white = 21:1; #767676 on white = 4.54:1, the canonical
"smallest grey that passes AA on white"). A test that asserted a CSS *declaration* instead would
pass for a rule that exists and does not apply, which is the failure this whole module exists for.

Driving: the CDP helpers of ``tests/integration/test_hud_headless.py`` (T-0008's file, untouched;
``Chrome``, ``CHROME``, ``requires_chrome`` are imported, never copied), and
``local_astra.hud_static.HudStaticServer`` — the production static server, on a loopback ephemeral
port.

Regime (docs/tasks/T-0056.md): Chrome 147 headless, dpr 1, viewports 1280x800 and 1920x1080,
``prefers-reduced-motion: reduce`` and ``prefers-color-scheme: light`` via
``Emulation.setEmulatedMedia``. n = 1 per viewport per scheme, no seed, no statistics: a differing
result is a bug, not noise.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.protocol import (
    AstraSpoken,
    AstraSpokenPlayed,
    AstraToolCall,
    AstraTranscript,
    AstraTurnState,
    AstraUiFence,
    PlayedUtterance,
    TurnState,
)
from local_astra.hud_static import HudStaticServer
from tests.fakes.hud_server import HudStub
from tests.integration.test_hud_headless import (
    A2UI_ROOT,
    CHROME,
    HUD,
    T0164_B,
    Chrome,
    requires_chrome,
)
from tests.integration.test_hud_ui_surface import gated_canvases

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]

#: The panes that mount in this configuration (task Regime): the panes declared in the shell plus
#: the assists panel and the session-log panel. ``denoise`` (T-0011) and ``listen`` (T-0047) are
#: absent. Criterion 6's row injects four more on top of these.
#:
#: **Amended by T-0128** (docs/tasks/T-0128.md criterion 3: "every declared pane is a direct child
#: of [main]", and criterion 4's rail adds the declared panes ``pane-thinking`` and
#: ``pane-diagnostics``). T-0056 pinned the literal 6 (four declared + two appended); the count of
#: declared panes is now read from ``hud/index.html`` itself — every ``<section class="pane …">`` —
#: plus the same two appended panels, so the T-0056 layout rows below keep measuring every pane
#: that mounts, before T-0128 lands (4 + 2) and after it (6 + 2). Removed assertion: none; the
#: literal ``PANE_COUNT = 6`` premise is what changed.
_SHELL = re.sub(r"<!--[\s\S]*?-->", " ", (REPO / "hud" / "index.html").read_text(encoding="utf-8"))
DECLARED_PANES = len(
    re.findall(r"<section\b[^>]*\bclass=\"(?:[^\"]*\s)?pane(?:\s[^\"]*)?\"", _SHELL)
)
PANE_COUNT = DECLARED_PANES + 2

#: Criterion 6's floor, verbatim from the task.
MIN_PANE_WIDTH_PX = 280.0
#: T-0056 ruling A12: the tonal alternative to a painted card edge. A borderless card is legible
#: because its surface is a step away from the page behind it, not because it is outlined, and the
#: step has to be a real one. 1.15:1 is the ruling's floor.
#:
#: **Measured, because the ruling's worked examples do not both survive arithmetic.** Ruling A12
#: reads "#1e1f20 on #131314 is about 1.2:1; #f0f4f9 on #ffffff about 1.1:1, so light may need
#: --surface one step darker". The second is right (1.104:1); the first is not — that pair is
#: **1.125:1** by WCAG 2.1 relative luminance, hand-checked and reproduced by the oracle in this
#: module, so the *dark* surface needs a step too. Clearing 1.15 from the Gemini palette:
#: #202123 on #131314 = 1.152, #232426 = 1.195, --surface-2 #282a2c = 1.289; light on #ffffff,
#: --surface-2 #e9eef6 = 1.165, --surface-3 #dde3ea = 1.292. Recorded here rather than silently
#: rounded: the floor is the orchestrator's number, and if it should be 1.10 instead that is a
#: ruling, not a test-author's edit.
MIN_TONAL_STEP = 1.15
#: Criterion 7's floor, verbatim from the task.
MAX_DEAD_SPACE_PX = 64.0

VIEWPORTS = {"1280x800": (1280, 800), "1920x1080": (1920, 1080)}


# --- the contrast oracle (WCAG 2.1, computed here, never read off a declaration) ---------------


def parse_css_color(value: str) -> tuple[float, float, float, float]:
    """``rgb(r, g, b)`` / ``rgba(r, g, b, a)`` / ``rgb(r g b / a)`` -> (r, g, b, alpha)."""
    numbers = re.findall(r"[-+]?\d*\.?\d+", value or "")
    if not numbers:
        # ``transparent`` and the initial value both mean "nothing painted here".
        return (0.0, 0.0, 0.0, 0.0)
    parts = [float(n) for n in numbers]
    if len(parts) == 3:
        parts.append(1.0)
    return (parts[0], parts[1], parts[2], parts[3])


def relative_luminance(rgb: tuple[float, float, float]) -> float:
    """WCAG 2.1 relative luminance of an opaque sRGB colour."""
    channels = []
    for raw in rgb:
        c = raw / 255.0
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """WCAG 2.1 contrast ratio between two opaque colours: (L_lighter + .05)/(L_darker + .05)."""
    la, lb = relative_luminance(a), relative_luminance(b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


def composite(
    top: tuple[float, float, float, float], bottom: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Source-over: ``top`` with its alpha painted on the opaque ``bottom``."""
    alpha = max(0.0, min(1.0, top[3]))
    return tuple(top[i] * alpha + bottom[i] * (1.0 - alpha) for i in range(3))  # type: ignore[return-value]


#: What a browser paints under everything. The page sets its own body background; this is only
#: reached if every level in the chain is transparent, which would itself be a finding.
CANVAS_WHITE = (255.0, 255.0, 255.0)


def painted(chain: list[list[Any]]) -> tuple[float, float, float]:
    """The opaque colour behind an element's text, from its background chain.

    ``chain`` is [[label, backgroundColor, opacity], …] from the element itself outwards to
    ``<html>`` (through shadow hosts, for group B). An ancestor's ``opacity`` fades its whole
    subtree, so a level's effective alpha is its own alpha times the product of the opacities of
    itself and every ancestor — which is why the walk carries a cumulative factor rather than
    treating each level independently.
    """
    cumulative = [1.0] * len(chain)
    running = 1.0
    for i in range(len(chain) - 1, -1, -1):
        running *= float(chain[i][2])
        cumulative[i] = running
    backdrop = CANVAS_WHITE
    for i in range(len(chain) - 1, -1, -1):
        r, g, b, a = parse_css_color(str(chain[i][1]))
        backdrop = composite((r, g, b, a * cumulative[i]), backdrop)
    return backdrop


def foreground(color: str, chain: list[list[Any]]) -> tuple[float, float, float]:
    """The colour the text is actually painted in, faded by every opacity above it."""
    cumulative = 1.0
    for level in chain:
        cumulative *= float(level[2])
    r, g, b, a = parse_css_color(color)
    return composite((r, g, b, a * cumulative), painted(chain))


def threshold_for(font_size: float, font_weight: int) -> float:
    """WCAG 2.1 large text: >= 18 px, or >= 14 px at weight >= 700 (task Regime says so too)."""
    large = font_size >= 18.0 or (font_size >= 14.0 and font_weight >= 700)
    return 3.0 if large else 4.5


def ratio_of(measurement: dict[str, Any]) -> float:
    chain = list(measurement["chain"])
    return contrast_ratio(foreground(str(measurement["color"]), chain), painted(chain))


def hexc(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(c):02x}" for c in rgb)


def test_the_contrast_oracle_matches_two_pairs_whose_ratio_is_known_by_hand():
    """The oracle every row below leans on, checked against WCAG 2.1's own worked examples.

    Black on white is the maximum the formula can produce, 21:1 exactly. #767676 on white is the
    darkest grey that clears the 4.5:1 body-text floor — 4.54:1 — and is the pair that catches a
    luminance function with the sRGB gamma ramp wrong, because a naive linear luminance puts it at
    about 4.0 and a gamma-2.2 one at about 4.6.
    """
    assert contrast_ratio((0, 0, 0), (255, 255, 255)) == pytest.approx(21.0, abs=1e-9)
    assert contrast_ratio((0x76, 0x76, 0x76), (255, 255, 255)) == pytest.approx(4.54, abs=0.005)
    # …and the compositing half, which the [data-unplayed] row depends on: 55 % of #8fa0aa over
    # #111a1f, by hand — 143*0.55 + 17*0.45 = 86.30, 160*0.55 + 26*0.45 = 99.70,
    # 170*0.55 + 31*0.45 = 107.45 — and that is a *lower* contrast against the pane than the
    # unfaded colour: 2.87 against 6.53. A checker that ignored opacity would report the unplayed
    # tail as passing at 6.5:1 when what the human sees is 2.9:1.
    faded = composite((0x8F, 0xA0, 0xAA, 0.55), (0x11, 0x1A, 0x1F))
    assert [round(c, 2) for c in faded] == [86.30, 99.70, 107.45]
    assert contrast_ratio(faded, (0x11, 0x1A, 0x1F)) == pytest.approx(2.87, abs=0.01)
    assert contrast_ratio((0x8F, 0xA0, 0xAA), (0x11, 0x1A, 0x1F)) == pytest.approx(6.53, abs=0.01)


# --- the page under measurement ---------------------------------------------------------------

#: The frozen (name, selector, pseudo) list of criterion 8. Frozen means: this module does not get
#: to drop a row because the restyle made it inconvenient.
#:
#: **Amended by T-0154 B1/B3/B7** (docs/tasks/T-0154.md G2). Three rows named elements B1 retires:
#: ``.line.partial`` (the partial line of the retired ``#transcript-pane``), ``.line.empty`` (its
#: empty line; the conversation pane's empty state is ``[data-empty]``, measured by criterion 12's
#: row below) and ``.placeholder`` (the ``no surface yet (T-0010)`` placeholder of the retired
#: ``#shown-channel``). Removed rows: those three. In their place, every text-bearing element of
#: the B3 turn layout, seeded through the page's own ``createThread`` (``SEED_JS``): the user
#: bubble (``[data-transcript]``, a partial and a final one — the partial line's successor), the
#: assistant prose (``[data-utterance-id]``), the meta line (``[data-turn-meta]``, interrupted mark
#: and fence notice), the action card's line and status chip (``[data-tool-line]``,
#: ``[data-tool-status]``, B7), and the composer's typed text (``[data-ask-input]``'s ``.value``;
#: the ``::value`` pseudo below is this module's own marker for "measure the element, read its
#: value", not a CSS pseudo-element). Every other row is kept verbatim, and the list is measured
#: in both emulated schemes and, through ``CONTRAST_SELECTORS_T0128``, both data-themes.
CONTRAST_SELECTORS: list[tuple[str, str, str | None]] = [
    ("pane heading", ".pane h2", None),
    ("user bubble", "[data-thread] [data-transcript]", None),
    ("assistant prose", "[data-thread] [data-utterance-id]", None),
    ("turn meta line", "[data-thread] [data-turn-meta]", None),
    ("action line", "[data-thread] [data-tool-line]", None),
    ("action status", "[data-thread] [data-tool-status]", None),
    ("composer input (typed text)", "[data-ask-input]", "::value"),
    ("top-bar counters", ".counters", None),
    ("counter value", ".counters b", None),
    ("connection status", ".status", None),
    ("turn chip", ".chip", None),
    ("button", "button", None),
    ("disabled button", "button[disabled]", None),
    ("ask input placeholder", ".ask-input", "::placeholder"),
    ("error line", ".error-line", None),
    ("capture banner", ".banner", None),
    # T-0162 B-9 (named spec change): the assist panel is found inside the Settings popover.
    ("assist explanation", "[data-settings-popover] #assist-panel .assist-why", None),
    ("session-log counters", "#log-panel .log-counters", None),
]

TURN_STATES = ["IDLE", "LISTENING", "ENDPOINTING", "THINKING", "SPEAKING", "INTERRUPTED"]

#: **Amended by T-0154 B1/B3** (G2): the conversation is no longer hand-built markup in
#: ``#transcript-pane`` / ``#spoken-pane`` (both retired). The page's own thread module renders it:
#: ``createThread`` is imported in-page from ``/src/panels/thread.js`` (the URL ``hud/src/main.js``
#: imports it from, so it is the same module instance) and handed the shell's ``[data-thread]``
#: plus decoded messages in the wire shapes of docs/02-architecture.md §5 (the same shapes
#: tests/hud/thread.test.mjs builds). A second thread instance on the page's root is safe here:
#: this module has no socket, so the page's own instance never receives a message. The result
#: says whether the thread could be seeded, so a missing module fails a row with its reason
#: rather than as "matched nothing".
THREAD_IMPORT_JS = """
  const seedThread = async (messages) => {
    const root = document.querySelector('[data-thread]');
    if (!root) return 'no [data-thread] in the shell (T-0154 B1)';
    let mod;
    try { mod = await import('/src/panels/thread.js'); }
    catch (e) { return 'could not import /src/panels/thread.js: ' + e.message; }
    if (typeof mod.createThread !== 'function') return 'thread.js exports no createThread';
    try {
      const t = mod.createThread(root);
      for (const m of messages) t.handle(m);
    } catch (e) { return 'the thread threw: ' + e.message; }
    return 'seeded';
  };
"""


def _wire(message: Any) -> dict[str, Any]:
    """A contract as the page's decoder hands it on (``codec.encode`` is ``exclude_none``)."""
    return dict(message.model_dump(mode="json", exclude_none=True))


#: The seeded conversation, built from the contracts so the shapes are the wire's (turn ids
#: prefixed ``c`` so they cannot collide with anything). c1: a final transcript and an utterance
#: whose tail the device never played (one ``astra.spoken.played`` entry, T-0030 criterion 19 —
#: the only source of ``[data-unplayed]``; "the part that was heard" is 5 words). c2: a partial
#: transcript, an utterance, a fence closed ``open`` (the meta line's notice), two action cards
#: (``proposed`` and ``dry_run``), then ``INTERRUPTED`` (the meta line's mark).
SEED_MESSAGES: list[dict[str, Any]] = [
    _wire(AstraTranscript(turn_id="c1", text="what is on my screen?", final=True)),
    _wire(
        AstraSpoken(
            turn_id="c1", utterance_id="c1-u1", text="the part that was heard the tail nobody heard"
        )
    ),
    _wire(
        AstraSpokenPlayed(
            turn_id="c1",
            played_ms=1000,
            utterances=[
                PlayedUtterance(
                    utterance_id="c1-u1",
                    played_text="the part that was heard",
                    played_words=5,
                    complete=False,
                )
            ],
        )
    ),
    _wire(AstraTranscript(turn_id="c2", text="and show me a", final=False)),
    _wire(AstraSpoken(turn_id="c2", utterance_id="c2-u1", text="Here is a packing list.")),
    _wire(AstraUiFence(turn_id="c2", canvases=1, close="open")),
    _wire(
        AstraToolCall(
            turn_id="c2",
            call_id="c2-k1",
            name="scroll",
            arguments={"direction": "down"},
            status="proposed",
            source="draft",
        )
    ),
    _wire(
        AstraToolCall(
            turn_id="c2",
            call_id="c2-k2",
            name="google_search",
            arguments={"query": "packing list"},
            status="dry_run",
            source="draft",
            action="would search for: packing list",
        )
    ),
    _wire(AstraTurnState(state=TurnState.INTERRUPTED, turn_id="c2")),
]

#: The composer's typed text (measured as ``[data-ask-input]``'s value).
COMPOSER_TEXT = "what is on my screen right now?"

#: Elements the page only creates when a message arrives. This module has no socket, so the
#: error line and the banner are created with *exactly* the markup hud/src/main.js produces
#: (renderErrors(): ``error-line`` inside #error-strip; the synthetic-capture banner), and the
#: conversation through the page's own thread (above). If a renderer's markup ever changes, the
#: hand-built half of this seed is wrong and the contrast it measures is fiction — which is why
#: each one names its source.
SEED_JS = (
    "(async () => {"
    + THREAD_IMPORT_JS
    + "  const MESSAGES = "
    + json.dumps(SEED_MESSAGES)
    + ";\n"
    + "  const COMPOSER = "
    + json.dumps(COMPOSER_TEXT)
    + ";\n"
    + """
  const doc = document;
  const mk = (tag, cls, text, attrs) => {
    const el = doc.createElement(tag);
    if (cls) el.className = cls;
    if (text) el.textContent = text;
    for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
    return el;
  };
  // hud/src/main.js renderErrors(): `p.error-line[data-error-code]` inside #error-strip, shown.
  const errors = doc.getElementById('error-strip');
  errors.hidden = false;
  errors.append(mk('p', 'error-line', 'llm_unavailable: vllm is not answering',
                   {'data-error-code': 'llm_unavailable'}));
  // hud/src/main.js: the synthetic-capture banner (T-0008 criterion 7).
  const banner = doc.getElementById('capture-banner');
  banner.hidden = false;
  banner.textContent = 'SYNTHETIC CAPTURE';
  // hud/src/main.js enables the mute button once capture starts; the shell ships it disabled and
  // it returns to that state whenever there is nothing to mute.
  doc.getElementById('mute-button').disabled = true;
  // The composer's typed text: a value, as a human leaves it after typing.
  const input = doc.querySelector('[data-ask-input]');
  if (input) input.value = COMPOSER;
  const thread = await seedThread(MESSAGES);
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  return {thread, articles: doc.querySelectorAll('[data-thread] article').length};
})()"""
)

CHAIN_JS = """
  const chainOf = (el) => {
    const out = [];
    let node = el;
    while (node) {
      const s = getComputedStyle(node);
      out.push([node.tagName + (node.id ? '#' + node.id : ''), s.backgroundColor, s.opacity]);
      node = node.parentElement || (node.getRootNode && node.getRootNode().host) || null;
    }
    return out;
  };
"""


#: T-0162 B-9: the assist panel lives in the Settings popover, closed by default, so its matches
#: are 0x0 boxes until the gear is pressed. The contrast probes open it, measure, and close it
#: again (Escape, B-6), so every other phase sees the page as before.
_SETTINGS_OPEN_JS = (
    "(() => { const g = document.querySelector('header button[aria-label=\"Settings\"]');"
    " const p = document.querySelector('[data-settings-popover] #assist-panel');"
    " if (g && !(p && p.checkVisibility())) { g.click(); return true; } return false; })()"
)
_SETTINGS_VISIBLE_JS = (
    "(() => { const p = document.querySelector('[data-settings-popover]');"
    " return !!(p && p.checkVisibility()); })()"
)


def _measure_with_settings_open(chrome: Any, specs: list[tuple[str, str, str | None]]) -> Any:
    opened = chrome.evaluate(_SETTINGS_OPEN_JS) is True
    try:
        return chrome.evaluate(measure_js(specs))
    finally:
        if opened and chrome.evaluate(_SETTINGS_VISIBLE_JS):
            chrome.key("Escape", code=27)


def measure_js(specs: list[tuple[str, str, str | None]]) -> str:
    """Every match of each (name, selector): colour, size, weight, opacity, background chain.

    ``querySelectorAll``, not ``querySelector`` — T-0056 ruling A7. The first match of
    ``#assist-panel .assist-why`` is the ``page_text`` row's empty 371x0 ``<small>``: measuring it
    reported 8.12:1 and hid the three explanations that really carry text, which sit under
    ``hud/src/assists/index.js``'s ``opacity: 0.55`` fade at 3.40:1 dark / 2.48:1 light. A
    one-element oracle passes a page whose failing text happens to come second, and the shape is
    latent on every selector with more than one match (``.pane h2``, ``button``, ``.counters b``,
    ``.line.*``).
    """
    return (
        "(() => {"
        + CHAIN_JS
        + "  const SPECS = "
        + json.dumps([[n, s, p] for n, s, p in specs])
        + ";\n"
        + """
  // The text a match carries. For a pseudo-element the host's own textContent is the wrong
  // question (an <input> has none): ::placeholder shows the placeholder attribute, and any other
  // pseudo shows its own `content`.
  const textFor = (el, pseudo) => {
    if (pseudo === '::placeholder') return (el.getAttribute('placeholder') || '').trim();
    // T-0154: the composer's typed text is the input's value, styled by the element itself.
    if (pseudo === '::value') return String(el.value || '').trim();
    if (pseudo) {
      const c = getComputedStyle(el, pseudo).content;
      return c && c !== 'none' && c !== 'normal' ? c.replace(/^["']|["']$/g, '').trim() : '';
    }
    return (el.textContent || '').trim();
  };
  const out = {};
  for (const [name, sel, pseudo] of SPECS) {
    const matches = [...document.querySelectorAll(sel)].map((el, index) => {
      const s = getComputedStyle(el, pseudo && pseudo !== '::value' ? pseudo : undefined);
      const own = getComputedStyle(el);
      const rect = el.getBoundingClientRect();
      return {
        index, selector: sel, pseudo: pseudo || null,
        color: s.color, background: s.backgroundColor,
        font_size: parseFloat(s.fontSize), font_weight: parseInt(s.fontWeight, 10) || 400,
        opacity: parseFloat(own.opacity), decoration: s.textDecorationLine,
        border_color: own.borderTopColor, border_width: parseFloat(own.borderTopWidth) || 0,
        width: rect.width, height: rect.height,
        text: textFor(el, pseudo).slice(0, 48),
        chain: chainOf(el),
        parent_chain: chainOf(el.parentElement || el),
      };
    });
    out[name] = {selector: sel, pseudo: pseudo || null, matches};
  }
  return out;
})()"""
    )


GEOMETRY_JS = """
(() => {
  // Descendant, not a direct-child selector: whether a pane is a *direct* child of <main>
  // is criterion 4's question and tests/hud/shell.test.mjs's row. This module's job is the
  // geometry of every pane on the page, wherever the markup put it — so that a wrapper
  // element makes the shell row go red and these rows keep measuring, instead of both
  // failing for the same reason and telling the implementer one thing twice.
  const panes = [...document.querySelectorAll('main .pane')];
  const doc = document.documentElement;
  return {
    count: panes.length,
    scroll_width: doc.scrollWidth, client_width: doc.clientWidth,
    scroll_height: doc.scrollHeight,
    inner_height: window.innerHeight,
    // The regime the number was taken in, so a bottom-of-page figure can never be quoted
    // without it (the banner is ~29 px of chrome and `?source=canvas` puts it up).
    banner_up: !document.getElementById('capture-banner').hidden,
    error_strip_up: !document.getElementById('error-strip').hidden,
    rects: panes.map((p) => {
      const r = p.getBoundingClientRect();
      const label = (p.querySelector('h2') || {}).textContent || p.id || p.className;
      return {label: label.trim().slice(0, 32), x: r.x, y: r.y, w: r.width, h: r.height,
              right: r.right, bottom: r.bottom,
              // T-0154 B1: the one conversation pane is the pane holding [data-thread] (was
              // .pane-spoken, .pane-said, .pane-shown — the three retired panes).
              conversation: p.matches(':has([data-thread])')};
    }),
  };
})()
"""

#: T-0128 (ruling on d4a36e1): the rail scrolls as a whole, and since there is no rail wrapper the
#: scroller is ``<main>``. Scroll it to its end, read every rail pane's rect (every ``main .pane``
#: that is not the conversation pane — T-0154 B1, was "not spoken, you-said or shown"), then put
#: the scroll back so later phases measure the
#: page as it loads. ``main_scrolled`` is the scrollTop actually reached, recorded so a failure
#: message can say whether ``<main>`` scrolled at all.
RAIL_REACH_JS = """
(() => {
  const main = document.querySelector('main');
  const before = main.scrollTop;
  main.scrollTop = main.scrollHeight;
  const reached = main.scrollTop;
  const panes = [...document.querySelectorAll('main .pane')]
    .filter((p) => !p.matches(':has([data-thread])'));  // T-0154 B1: every pane but the thread's
  const rects = panes.map((p) => {
    const r = p.getBoundingClientRect();
    const label = (p.querySelector('h2') || {}).textContent || p.id || p.className;
    return {label: label.trim().slice(0, 32), top: r.top, bottom: r.bottom};
  });
  const out = {
    inner_height: window.innerHeight,
    main_scroll_height: main.scrollHeight, main_client_height: main.clientHeight,
    main_scrolled: reached,
    doc_scroll_height: document.documentElement.scrollHeight,
    rects,
  };
  main.scrollTop = before;
  return out;
})()
"""

INJECT_PANES_JS = """
(() => {
  const main = document.querySelector('main');
  for (let i = 0; i < 4; i += 1) {
    const section = document.createElement('section');
    section.className = 'pane';
    section.dataset.injected = 'true';
    const h2 = document.createElement('h2');
    h2.textContent = 'x';
    section.append(h2);
    main.append(section);
  }
  return document.querySelectorAll('main .pane').length;
})()
"""

MOTION_JS = """
(() => {
  const out = [];
  for (const el of document.querySelectorAll('*')) {
    const s = getComputedStyle(el);
    if (s.animationName !== 'none' || s.animationDuration !== '0s' ||
        s.transitionDuration !== '0s') {
      out.push({tag: el.tagName + (el.id ? '#' + el.id : '') + (el.className ? '.' + el.className : ''),
                animation_name: s.animationName, animation_duration: s.animationDuration,
                transition_duration: s.transitionDuration});
    }
  }
  return out;
})()
"""

CHIP_STATES_JS = """
(() => {
  const chip = document.querySelector('[data-turn-state]');
  const before = chip.getAttribute('data-turn-state');
  const out = {};
  for (const state of __STATES__) {
    chip.setAttribute('data-turn-state', state);
    const s = getComputedStyle(chip);
    out[state] = {color: s.color, background: s.backgroundColor, border: s.borderTopColor,
                  border_width: s.borderTopWidth, box_shadow: s.boxShadow,
                  font_weight: s.fontWeight, text_decoration: s.textDecorationLine,
                  outline: s.outlineColor + ' ' + s.outlineWidth};
  }
  chip.setAttribute('data-turn-state', before);
  return out;
})()
""".replace("__STATES__", json.dumps(TURN_STATES))

FILL_JS = """
(() => {
  const bar = document.getElementById('thinking-bar');
  const out = {paints: {}, ticks: [], animation_name: getComputedStyle(bar).animationName};
  for (const fill of ['0%', '50%', '100%']) {
    bar.style.setProperty('--fill', fill);
    const s = getComputedStyle(bar);
    out.paints[fill] = [s.backgroundImage, s.backgroundColor, s.backgroundSize,
                        s.backgroundPositionX, s.boxShadow, s.animationName].join(' | ');
  }
  bar.style.setProperty('--fill', '0%');
  out.bar_width = bar.clientWidth;
  out.ticks = [...bar.querySelectorAll('.tick')].map((t) => ({
    inline_left: t.style.left,
    computed_left: getComputedStyle(t).left,
    mark_ms: t.dataset.markMs,
  }));
  return out;
})()
"""

FOCUS_JS = (
    """
(() => {
"""
    + CHAIN_JS
    + """
  const out = {};
  for (const sel of ['#share-button', '#mute-button', '[data-ask-input]', '[data-ask-submit]']) {
    const el = document.querySelector(sel);
    if (!el) { out[sel] = {present: false}; continue; }
    // hud/src/ask/index.js enables both ask controls on mount; a disabled control cannot take
    // focus, so it is enabled for the measurement and put back exactly as it was afterwards —
    // #mute-button is measured as `button[disabled]` by the contrast rows.
    const was_disabled = el.disabled;
    el.disabled = false;
    el.focus();
    const s = getComputedStyle(el);
    out[sel] = {present: true, focused: document.activeElement === el,
                focus_visible: el.matches(':focus-visible'),
                outline_width: parseFloat(s.outlineWidth) || 0, outline_style: s.outlineStyle,
                outline_color: s.outlineColor, box_shadow: s.boxShadow,
                parent_chain: chainOf(el.parentElement || el)};
    el.blur();
    el.disabled = was_disabled;
  }
  return out;
})()
"""
)

#: **Amended by T-0154 B1** (G2): the pane under test is the conversation pane (the direct child
#: of ``<main>`` holding ``[data-thread]``; was the pane holding the retired ``#spoken-pane``), and
#: the content is given to it the way the page gives it — through the page's own thread
#: (``THREAD_IMPORT_JS``), one transcript and one utterance — instead of a hand-built
#: ``section.turn > p.utterance``. The article is removed afterwards so phase 1 measures the
#: Regime state.
EMPTY_STATE_MESSAGES: list[dict[str, Any]] = [
    _wire(AstraTranscript(turn_id="e1", text="is anything here?", final=True)),
    _wire(
        AstraSpoken(
            turn_id="e1", utterance_id="e1-u1", text="a sentence the Conductor really spoke"
        )
    ),
]

EMPTY_STATE_JS = (
    """
(async () => {
"""
    + CHAIN_JS
    + THREAD_IMPORT_JS
    + "  const MESSAGES = "
    + json.dumps(EMPTY_STATE_MESSAGES)
    + ";\n"
    + """
  const pane = document.querySelector('main > :has([data-thread])');
  const line = pane ? pane.querySelector('[data-empty]') : null;
  const snap = (el) => {
    if (!el) return {present: false};
    const s = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    return {present: true, display: s.display, visibility: s.visibility, opacity: s.opacity,
            color: s.color, font_size: parseFloat(s.fontSize),
            font_weight: parseInt(s.fontWeight, 10) || 400,
            text: (el.textContent || '').trim().slice(0, 64),
            x: r.x, y: r.y, w: r.width, h: r.height, right: r.right, bottom: r.bottom,
            chain: chainOf(el)};
  };
  const out = {empty: snap(line), pane_present: !!pane};
  // Now give the pane real content, the way the page does: through its own thread.
  out.seeded = await seedThread(MESSAGES);
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const utterance = document.querySelector('[data-thread] [data-utterance-id="e1-u1"]');
  out.with_content = {empty: snap(line), content: snap(utterance)};
  for (const a of document.querySelectorAll('[data-thread] article')) a.remove();
  return out;
})()
"""
)


def closed_loopback_port() -> int:
    """A loopback port with nothing listening on it (the task's deterministic socket state)."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


@dataclass
class Chrome56:
    """Everything one page session measured, keyed by phase."""

    error: str = ""
    seconds: float = 0.0
    empty_state: dict[str, Any] = field(default_factory=dict)
    #: T-0154: SEED_JS's report — whether the page's own thread could be seeded, and how many
    #: articles it made (``{"thread": "seeded", "articles": 2}`` when it could).
    seed: dict[str, Any] = field(default_factory=dict)
    contrast_dark: dict[str, Any] = field(default_factory=dict)
    contrast_light: dict[str, Any] = field(default_factory=dict)
    unplayed_dark: dict[str, Any] = field(default_factory=dict)
    unplayed_light: dict[str, Any] = field(default_factory=dict)
    borders_dark: dict[str, Any] = field(default_factory=dict)
    borders_light: dict[str, Any] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)
    geometry_injected: dict[str, Any] = field(default_factory=dict)
    #: T-0128: the rail panes' rects at 1280x800 with ``<main>`` scrolled to its end.
    rail_reach: dict[str, Any] = field(default_factory=dict)
    chips_dark: dict[str, Any] = field(default_factory=dict)
    chips_reduced: dict[str, Any] = field(default_factory=dict)
    fill: dict[str, Any] = field(default_factory=dict)
    focus: dict[str, Any] = field(default_factory=dict)
    motion_reduced: list[Any] = field(default_factory=list)
    body_background: dict[str, str] = field(default_factory=dict)
    #: T-0128 phase 6: kept apart from ``error`` so a failure of the T-0128 phase (no Settings
    #: control yet) cannot turn the T-0056 rows red.
    t0128_error: str = ""
    theme_chosen: dict[str, Any] = field(default_factory=dict)
    theme_after_flip: dict[str, Any] = field(default_factory=dict)
    contrast_theme: dict[str, Any] = field(default_factory=dict)
    unplayed_theme: dict[str, Any] = field(default_factory=dict)
    borders_theme: dict[str, Any] = field(default_factory=dict)
    #: T-0154 A7: the new text-bearing selectors, per data-theme, measured in phase 6.
    contrast_t0154: dict[str, Any] = field(default_factory=dict)


# --- T-0128: the theme is chosen through the Settings control, and measured under data-theme -----
#
# docs/tasks/T-0128.md *Tests to write* T4: "Run the contrast rows under data-theme light and dark."
# The theme is set the way a human sets it — the header's Settings control — and then the system
# scheme is emulated as the *opposite*, so the measurement is of the data-theme tokens and a
# left-over prefers-color-scheme rule shows up as the wrong colours (criterion 6 (a)). The control
# is found by what criterion 8 says it is: the one <select> in <header> whose option values are
# system/light/dark; nothing else about its markup is assumed.

FIND_THEME_CONTROL_JS = """
  const themeControl = () => {
    const bar = document.querySelector('header');
    if (!bar) return null;
    const hits = [...bar.querySelectorAll('select')].filter((s) =>
      [...s.options].map((o) => o.value).sort().join(',') === 'dark,light,system');
    return hits.length === 1 ? hits[0] : null;
  };
  const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const themeState = () => ({
    theme: document.documentElement.getAttribute('data-theme'),
    scheme: getComputedStyle(document.documentElement).colorScheme,
    body: getComputedStyle(document.body).backgroundColor,
  });
"""


def choose_theme_js(preference: str) -> str:
    """Pick ``preference`` in the Settings control as a change event; report the page after a frame."""
    return (
        "(async () => {"
        + FIND_THEME_CONTROL_JS
        + "  const c = themeControl();\n"
        + "  if (!c) return {ok: false, why: 'no single <select> in <header> whose options are "
        + "system/light/dark (criterion 8)', ...themeState()};\n"
        + f"  c.value = {json.dumps(preference)};\n"
        + "  c.dispatchEvent(new Event('input', {bubbles: true}));\n"
        + "  c.dispatchEvent(new Event('change', {bubbles: true}));\n"
        + "  await frame();\n"
        + "  return {ok: true, value: c.value, ...themeState()};\n"
        + "})()"
    )


THEME_STATE_JS = (
    "(async () => {" + FIND_THEME_CONTROL_JS + "  await frame();\n  return themeState();\n})()"
)

#: The Thinking pane's empty line, verbatim from T-0128 criterion 5.
DENOISE_EMPTY_T0128 = (
    "no denoising data yet — this page does not receive the plugin's per-step canvases"
)
#: **Amended by T-0128c B5** (H-T128c-6; O-B gives this copy to group B's test-author): the page
#: now receives the real canvases, so criterion 5's copy is no longer true. Only the copy changes.
DENOISE_EMPTY_T0128C = (
    "no denoising data yet — the model's real canvases appear here while it thinks"
)


def _t0128c_b_activated() -> bool:
    """The switch tests/hud/shell.test.mjs uses (its ``T0128C_B_PENDING``), in Python.

    T-0128 criterion 5's copy is checked until docs/tasks/T-0128c.md's **Status:** records group
    B's activation ("group B: tests <a>..<b>, activated <sha>"); from then on B5's. ``--pending``
    on the command line or ``ASTRA_RUN_PENDING=1`` checks B5's copy now (it fails until B lands).
    """
    if os.environ.get("ASTRA_RUN_PENDING") == "1" or "--pending" in sys.argv:
        return True
    task = Path(__file__).resolve().parents[2] / "docs" / "tasks" / "T-0128c.md"
    if not task.is_file():
        return False
    m = re.search(r"^\*\*Status:\*\*\s*(.*)$", task.read_text(encoding="utf-8"), re.M)
    status = m.group(1) if m else ""
    return bool(re.search(r"\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b", status, re.I))


DENOISE_EMPTY = DENOISE_EMPTY_T0128C if _t0128c_b_activated() else DENOISE_EMPTY_T0128

#: Tags the two T-0128 elements that have no class of their own to select on, by what they are:
#: the Settings control (as above) and the Thinking pane's empty line (by its text). Test-side
#: attributes only; nothing the page reads.
TAG_PROBES_JS = (
    "(() => {"
    + FIND_THEME_CONTROL_JS
    + "  const c = themeControl();\n"
    + "  if (c) c.setAttribute('data-t0128-probe', 'theme-control');\n"
    + "  const pane = document.querySelector('main > .pane-thinking');\n"
    + "  let line = null;\n"
    + "  if (pane) for (const el of pane.querySelectorAll('*')) {\n"
    + "    if (el.closest('[data-denoise-root]')) continue;\n"
    + f"    if ((el.textContent || '').replace(/\\s+/g, ' ').trim() === {json.dumps(DENOISE_EMPTY)}) line = el;\n"
    + "  }\n"
    + "  if (line) line.setAttribute('data-t0128-probe', 'thinking-empty');\n"
    + "  return {control: !!c, empty_line: !!line};\n"
    + "})()"
)

#: Criterion 10's frozen list: T-0056's, extended with the header pill, the settings control, the
#: rail headings, Diagnostics and the Thinking empty line. The T-0056 entries are kept verbatim.
CONTRAST_SELECTORS_T0128: list[tuple[str, str, str | None]] = [
    *CONTRAST_SELECTORS,
    ("header turn pill", "header [data-turn-state]", None),
    ("settings control", "[data-t0128-probe='theme-control']", None),
    (
        "rail headings",
        "main > .pane.thumb h2, main > .pane-thinking h2, [data-settings-popover] #assist-panel h2, #log-panel h2, "
        "main > .pane-diagnostics h2",
        None,
    ),
    (
        "diagnostics",
        ".pane-diagnostics #frames-sent, .pane-diagnostics #frames-dropped, "
        ".pane-diagnostics #capture-size, .pane-diagnostics #ignored-count, "
        ".pane-diagnostics #conn-status",
        None,
    ),
    ("thinking empty line", "[data-t0128-probe='thinking-empty']", None),
]


#: T-0154 A7 (T-0128 criterion 10 kept, "with the selector list extended to every new text-bearing
#: selector"): group A's new text — the Context pane's human labels and the What I see caption.
#: The one-line reasons stay ``#assist-panel .assist-why`` (T-0056's frozen row above). The hooks
#: are pinned by tests/integration/test_hud_assists.py (``[data-assist-label]``) and
#: tests/integration/test_hud_redesign.py (``[data-capture-caption]``); "Sharing screen" is measured
#: there too, because this module's capture is synthetic and never shows it.
CONTRAST_SELECTORS_T0154: list[tuple[str, str, str | None]] = [
    (
        "context labels",
        "[data-settings-popover] #assist-panel [data-assist-label]",
        None,
    ),  # T-0162 B-9: in the popover
    ("capture caption", "main > .pane.thumb [data-capture-caption]", None),
]


BORDER_SELECTORS: list[tuple[str, str, str | None]] = [
    ("chip border", ".chip", None),
    ("pane border", ".pane", None),
]

UNPLAYED: list[tuple[str, str, str | None]] = [("unplayed tail", "[data-unplayed]", None)]


def _set_viewport(chrome: Chrome, width: int, height: int) -> None:
    chrome._cmd(
        "Emulation.setDeviceMetricsOverride",
        width=width,
        height=height,
        deviceScaleFactor=1,
        mobile=False,
    )
    chrome.evaluate("document.documentElement.getBoundingClientRect().height")
    time.sleep(0.2)


def _media(chrome: Chrome, features: list[dict[str, str]]) -> None:
    chrome._cmd("Emulation.setEmulatedMedia", features=features)
    chrome.evaluate("document.documentElement.getBoundingClientRect().height")
    time.sleep(0.2)


@pytest.fixture(scope="module")
def page() -> Any:
    """One page load in six phases. Never raises: a failure lands on ``Chrome56.error``."""
    if CHROME is None:  # pragma: no cover - requires_chrome is the real guard
        yield Chrome56(error="no chrome")
        return
    result = Chrome56()
    started = time.monotonic()
    server = HudStaticServer(HUD, 0)
    try:
        ws = f"ws://127.0.0.1:{closed_loopback_port()}/v1/realtime"
        url = f"http://127.0.0.1:{server.port}/?source=canvas&ws=" + urllib.parse.quote(ws, safe="")
        with Chrome(CHROME) as chrome:
            chrome.navigate(url)
            mounted = chrome.wait_until(
                f"document.querySelectorAll('main .pane').length >= {PANE_COUNT}", 25
            )
            if not mounted:
                result.error = (
                    "only "
                    + str(chrome.evaluate("document.querySelectorAll('main .pane').length"))
                    + f" of the {PANE_COUNT} panes of the task's Regime mounted in 25 s "
                    "(ui, assists, ask, log plus the four declared sections)"
                )
            _set_viewport(chrome, *VIEWPORTS["1280x800"])

            # --- phase 0: the empty state, on a pane that is genuinely empty ------------------
            result.empty_state = chrome.evaluate(EMPTY_STATE_JS) or {}

            # --- phase 1: geometry, in the Regime state and nothing else ----------------------
            # Before SEED_JS, deliberately (ruling A8). The seed unhides #error-strip and adds a
            # line to it, which is ~29 px of chrome the Regime state does not have: measured after
            # the seed, the lowest pane at f1f9ade read 888.5 of 800 instead of 820.5, and the
            # number criterion 7 is about would have been the test's own doing. Criteria 6 and 7
            # are measured on `?source=canvas` with the six panes that mount and the page's own
            # synthetic-capture banner, which GEOMETRY_JS records so the number carries its state.
            result.geometry = {}
            for name, (width, height) in VIEWPORTS.items():
                _set_viewport(chrome, width, height)
                result.geometry[name] = chrome.evaluate(GEOMETRY_JS)
            # T-0128: rail reachability, same Regime state, before the injected panes.
            _set_viewport(chrome, *VIEWPORTS["1280x800"])
            result.rail_reach = chrome.evaluate(RAIL_REACH_JS) or {}

            # --- phase 2: criterion 6, four more panes than exist today -----------------------
            chrome.evaluate(INJECT_PANES_JS)
            for name, (width, height) in VIEWPORTS.items():
                _set_viewport(chrome, width, height)
                result.geometry_injected[name] = chrome.evaluate(GEOMETRY_JS)
            chrome.evaluate(
                "[...document.querySelectorAll('[data-injected]')].forEach(e=>e.remove())"
            )
            _set_viewport(chrome, *VIEWPORTS["1280x800"])

            # --- phase 3: dark colour, with the message-driven elements seeded ----------------
            result.seed = chrome.evaluate(SEED_JS) or {}
            result.contrast_dark = _measure_with_settings_open(chrome, CONTRAST_SELECTORS) or {}
            result.unplayed_dark = chrome.evaluate(measure_js(UNPLAYED)) or {}
            result.borders_dark = chrome.evaluate(measure_js(BORDER_SELECTORS)) or {}
            result.chips_dark = chrome.evaluate(CHIP_STATES_JS) or {}
            result.fill = chrome.evaluate(FILL_JS) or {}
            chrome.key("Tab")  # keyboard modality, so :focus-visible can match
            result.focus = chrome.evaluate(FOCUS_JS) or {}
            result.body_background["dark"] = str(
                chrome.evaluate("getComputedStyle(document.body).backgroundColor")
            )

            # --- phase 4: reduced motion ------------------------------------------------------
            _media(chrome, [{"name": "prefers-reduced-motion", "value": "reduce"}])
            result.motion_reduced = list(chrome.evaluate(MOTION_JS) or [])
            result.chips_reduced = chrome.evaluate(CHIP_STATES_JS) or {}

            # --- phase 5: the light scheme ----------------------------------------------------
            _media(chrome, [{"name": "prefers-color-scheme", "value": "light"}])
            result.body_background["light"] = str(
                chrome.evaluate("getComputedStyle(document.body).backgroundColor")
            )
            result.contrast_light = _measure_with_settings_open(chrome, CONTRAST_SELECTORS) or {}
            result.unplayed_light = chrome.evaluate(measure_js(UNPLAYED)) or {}
            result.borders_light = chrome.evaluate(measure_js(BORDER_SELECTORS)) or {}
            _media(chrome, [])

            # --- phase 6 (T-0128): each theme chosen in the control, system emulated opposite --
            try:
                chrome.evaluate(TAG_PROBES_JS)
                for theme, opposite in (("light", "dark"), ("dark", "light")):
                    result.theme_chosen[theme] = chrome.evaluate(choose_theme_js(theme)) or {}
                    _media(chrome, [{"name": "prefers-color-scheme", "value": opposite}])
                    result.theme_after_flip[theme] = chrome.evaluate(THEME_STATE_JS) or {}
                    result.contrast_theme[theme] = (
                        _measure_with_settings_open(chrome, CONTRAST_SELECTORS_T0128) or {}
                    )
                    result.unplayed_theme[theme] = chrome.evaluate(measure_js(UNPLAYED)) or {}
                    result.borders_theme[theme] = (
                        chrome.evaluate(measure_js(BORDER_SELECTORS)) or {}
                    )
                    result.contrast_t0154[theme] = (
                        _measure_with_settings_open(chrome, CONTRAST_SELECTORS_T0154) or {}
                    )
                chrome.evaluate(choose_theme_js("system"))
                _media(chrome, [])
            except Exception as exc:  # pragma: no cover - lands on the T-0128 rows only
                result.t0128_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - a broken session must fail in a row, not here
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        server.close()
    result.seconds = time.monotonic() - started
    yield result


def _ok(page: Chrome56) -> None:
    assert not page.error, f"the page session failed before any assertion ran: {page.error}"


def _seeded(page: Chrome56) -> None:
    """Premise of every row that measures the conversation (T-0154 B1-B3): the page's own thread
    rendered SEED_MESSAGES, as two articles (turns c1 and c2, B2: one per turn with content)."""
    assert page.seed.get("thread") == "seeded", (
        "premise (T-0154 B1): the conversation is seeded through the page's own createThread "
        f"(/src/panels/thread.js) into the shell's [data-thread]; it could not be: {page.seed!r}"
    )
    assert page.seed.get("articles") == 2, (
        "premise (T-0154 B2): SEED_MESSAGES name two turns with content (c1, c2), so the thread "
        f"holds two articles; SEED_JS counted {page.seed.get('articles')!r}"
    )


def _describe(name: str, m: dict[str, Any]) -> str:
    fg, bg = foreground(str(m["color"]), m["chain"]), painted(m["chain"])
    where = f" match #{m['index']}" if "index" in m else ""
    return (
        f"{name}{where} ({m['selector']}{m.get('pseudo') or ''}): text {m['color']} -> "
        f"{hexc(fg)} on {hexc(bg)}, {m['font_size']}px/{m['font_weight']}, "
        f"opacity {m['opacity']}, ratio {ratio_of(m):.2f}:1, text={m.get('text')!r}"
    )


def unmeasurable(m: dict[str, Any]) -> str:
    """Why this match carries no measurable text, or "" if it does (T-0056 ruling A7).

    A match is excluded only for a reason that is printed; it is never dropped silently, and a
    selector whose matches are *all* excluded fails as vacuous rather than passing.
    """
    reasons = []
    if not str(m.get("text") or "").strip():
        reasons.append("no text")
    if float(m["width"]) <= 0 or float(m["height"]) <= 0:
        reasons.append(f"{m['width']:.0f}x{m['height']:.0f} box")
    return ", ".join(reasons)


def _partition(entry: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """(matches that carry measurable text, one printable line per excluded match)."""
    usable, excluded = [], []
    for m in entry["matches"]:
        reason = unmeasurable(m)
        if reason:
            excluded.append(f"match #{m['index']} ({reason}, text={m.get('text')!r})")
        else:
            usable.append(m)
    return usable, excluded


# --- criterion 8: contrast floors, in both schemes ---------------------------------------------


@requires_chrome
@pytest.mark.parametrize("scheme", ["dark", "light"])
def test_every_frozen_selector_clears_its_wcag_floor(page, scheme):
    """Criterion 8 (and criterion 13's "in both schemes"): text the human can actually read.

    The floor is WCAG 2.1 AA: 4.5:1 under 18 px (or under 14 px bold), 3.0:1 above. The ratio is
    computed in this process from the computed colour, the background chain and every opacity in
    between; nothing here asserts that a rule exists, because a rule that exists and loses the
    cascade is exactly the bug. Today ``.line.empty`` computes to 1.34:1 — legible only if you
    know it is there — which is the measurement that put this criterion in the task.

    **Amended by T-0154 B1/B3/B7** (G2): the list's three rows on retired elements are replaced by
    the thread's text (see ``CONTRAST_SELECTORS``), seeded through the page's own thread.
    """
    _ok(page)
    _seeded(page)
    measured = page.contrast_dark if scheme == "dark" else page.contrast_light
    assert measured, "no measurements were taken"
    missing = [name for name, e in measured.items() if not e["matches"]]
    assert missing == [], (
        f"these frozen selectors matched nothing on the page in the {scheme} scheme: {missing}. "
        "Criterion 8 freezes the list, so a selector that stops matching is a finding, not a row "
        "to drop: either the restyle renamed a class four other files use, or this module's "
        "SEED_JS no longer mirrors what hud/src/panels/*.js render."
    )
    failures, vacuous, excluded_lines = [], [], []
    for name, entry in measured.items():
        usable, excluded = _partition(entry)
        for line in excluded:
            excluded_lines.append(f"{name} ({entry['selector']}): {line}")
        if not usable:
            vacuous.append(
                f"{name} ({entry['selector']}): {len(entry['matches'])} match(es), none with "
                f"measurable text — {'; '.join(excluded)}"
            )
            continue
        for m in usable:
            floor = threshold_for(float(m["font_size"]), int(m["font_weight"]))
            if ratio_of(m) + 1e-9 < floor:
                failures.append(f"{_describe(name, m)} — needs {floor}:1")
    if excluded_lines:  # never dropped silently (ruling A7); shown with the captured output
        print(f"[criterion 8, {scheme}] matches excluded as unmeasurable:")
        for line in excluded_lines:
            print("   " + line)
    assert vacuous == [], (
        f"criterion 8, {scheme} scheme: these frozen selectors have matches but none that carry "
        "text in a non-zero box, so the row would have measured nothing:\n  "
        + "\n  ".join(vacuous)
        + "\n(Ruling A7: an unmeasurable match is excluded with its reason printed, never counted "
        "as a pass. A selector with no measurable match at all is a finding — the element the "
        "criterion names has stopped carrying the text it was frozen for.)"
    )
    assert failures == [], (
        f"criterion 8, {scheme} scheme: these text elements are below their WCAG 2.1 floor.\n  "
        + "\n  ".join(failures)
        + "\n(4.5:1 under 18 px or under 14 px bold, 3.0:1 above. Every match of every frozen "
        "selector is measured, not the first — ruling A7: the first #assist-panel .assist-why is "
        "an empty 371x0 <small> at 8.12:1, and the three that carry text sit under that row's "
        "opacity fade. The whole point of the token layer is that fixing one of these is a value "
        "change at :root, not a hunt through the sheet.)"
        + ("\nExcluded as unmeasurable: " + "; ".join(excluded_lines) if excluded_lines else "")
    )


@requires_chrome
@pytest.mark.parametrize("scheme", ["dark", "light"])
def test_the_unplayed_tail_is_dimmed_struck_and_still_readable(page, scheme):
    """Criterion 8's named carve-out, as one row so the two rules cannot be traded off.

    T-0030 criteria 19-20 say the words the device never spoke are visible but plainly not spoken:
    struck through and dimmed. Criterion 8 says nothing on the page may be under 3.0:1. Today the
    tail computes to about 2.9:1, so *something* has to move — and the task says which: lift the
    dimmed colour, do not remove the dimming. Asserting all three in one row is what stops the
    implementer from buying the ratio by setting opacity to 1 (which passes a contrast-only row
    and silently deletes the distinction the measurement exists to show) or by deleting the
    strike-through (which tests/hud/state.test.mjs would catch, but only because that rule is
    pinned elsewhere).
    """
    _ok(page)
    _seeded(page)
    entry = (page.unplayed_dark if scheme == "dark" else page.unplayed_light).get(
        "unplayed tail", {"matches": []}
    )
    assert entry["matches"], (
        "[data-unplayed] was not on the page. T-0154 (G2): this module seeds it through the "
        "page's own thread — one astra.spoken.played entry for c1-u1 with played_words 5 of 9 — "
        "so the thread's renderer (renderUtteranceText, reused unchanged per implementation "
        "note 2) must strike the four unplayed words."
    )
    # One tail is seeded, so one match is expected; asserting it keeps the row honest if the
    # page ever renders several (ruling A7: measure every match, never just the first).
    assert len(entry["matches"]) == 1, (
        f"{len(entry['matches'])} [data-unplayed] elements on the page; this module seeds exactly "
        "one, so more than one means the page is marking text as unheard on its own"
    )
    measured = entry["matches"][0]
    assert not unmeasurable(measured), (
        f"the seeded [data-unplayed] is not measurable ({unmeasurable(measured)}); the tail the "
        "device never played must stay visible (T-0030 criterion 19), not be collapsed"
    )
    ratio = ratio_of(measured)
    opacity = float(measured["opacity"])
    decoration = str(measured["decoration"])
    assert "line-through" in decoration, (
        f"[data-unplayed] computes text-decoration-line={decoration!r} in the {scheme} scheme. "
        "T-0030 criterion 19: the tail the device never rendered is struck through, and "
        "tests/hud/state.test.mjs allows exactly one line-through rule in hud/styles.css with "
        "exactly the selector [data-unplayed]. Losing it means the page claims the user heard "
        "words they did not."
    )
    assert 0.0 < opacity < 1.0, (
        f"[data-unplayed] computes opacity={opacity} in the {scheme} scheme. T-0030 criterion 20 "
        "needs it strictly between 0 and 1: at 1 the tail is indistinguishable from speech that "
        "was actually played, at 0 it is gone. Criterion 8 forbids buying contrast by setting it "
        "to 1."
    )
    assert ratio >= 3.0, (
        f"criterion 8's carve-out, {scheme} scheme: {_describe('unplayed tail', measured)} — the "
        "floor for this one element is 3.0:1, not 4.5:1, because it is deliberately dimmed. "
        "Today's #8fa0aa at 55 % over the pane computes to about 2.87:1. The task names the fix: "
        "lift the dimmed colour (a lighter --dim, or a dedicated token, or a higher opacity short "
        "of 1). Removing the dimming to clear the bar is forbidden by the two assertions above."
    )


def _painted_edge(m: dict[str, Any]) -> tuple[bool, float, str]:
    """(is a border painted at all, its ratio against the better of the two sides, a description).

    A boundary is visible if it contrasts with *either* surface it separates, so the better of the
    two is what it is judged on; that is deliberately the forgiving reading, and both of today's
    failures are below the floor on both sides anyway.
    """
    colour = parse_css_color(str(m["border_color"]))
    if float(m["border_width"]) <= 0 or colour[3] <= 0:
        why = "width 0" if float(m["border_width"]) <= 0 else f"colour {m['border_color']}"
        return (False, 0.0, f"no painted border ({why})")
    inside = painted(m["chain"])
    outside = painted(m["parent_chain"])
    ratio = max(
        contrast_ratio(composite(colour, inside), inside),
        contrast_ratio(composite(colour, outside), outside),
    )
    return (
        True,
        ratio,
        f"border {m['border_color']} at {ratio:.2f}:1 against the better of {hexc(inside)} "
        f"(inside) and {hexc(outside)} (outside)",
    )


def _tonal_step(m: dict[str, Any]) -> tuple[float, str]:
    """(the card's surface against the page behind it, a description) — ruling A12's arm (b)."""
    surface = painted(m["chain"])
    behind = painted(m["parent_chain"])
    ratio = contrast_ratio(surface, behind)
    return ratio, f"surface {hexc(surface)} on {hexc(behind)} at {ratio:.2f}:1"


@requires_chrome
@pytest.mark.parametrize("scheme", ["dark", "light"])
def test_the_chip_and_pane_borders_are_visible_against_what_they_separate(page, scheme):
    """Criterion 8's non-text floors, as amended by T-0056 ruling A12.

    The property is *the card is legible as a card*, and there are two honest ways to draw one.
    The original row asserted only the first, and that is why the Gemini pass shipped a 1 px
    #6b6f74 hairline its own sheet calls out as design debt: a tonal design has no hairlines, and
    a test that demands one dictates the visual language rather than the property underneath it.

      (a) **a painted edge** — width > 0 and a non-transparent colour, at >= 3.0:1 against the
          better of the two surfaces it separates. Today's pre-restyle --edge is 1.42:1: not a
          frame, a rumour, and half of why the page reads as one field of black.
      (b) **no painted edge at all**, and a tonal step instead: the card's own surface at
          >= 1.15:1 against the page background behind it. A borderless card on a surface equal
          to the background is 1.00:1 — an invisible card, which is the failure this arm exists
          to catch, and the only way arm (b) can be abused.

    The chip keeps the painted-edge requirement alone (it is a small pill inside a bar, not a
    card: a tonal step at its size is not a boundary a human resolves), and so does the focus
    ring, in its own row. Run under both schemes: a tonal step is exactly the thing that survives
    a dark palette and collapses in a light one, where surface and background are both near white.

    **Amended by T-0154 A3** (docs/tasks/T-0154.md: "T-0056 A12's borderless panes are superseded by
    name"): for the pane, arm (a)'s 3.0:1 hairline is no longer an alternative and a painted edge
    below it is no longer a failure; every pane must clear arm (b)'s tonal step, edged or not. Removed
    assertion: "a pane with a painted edge below 3.0:1 fails". The chip keeps arm (a) as before.
    """
    _ok(page)
    measured = page.borders_dark if scheme == "dark" else page.borders_light
    assert measured, f"no border measurements were taken in the {scheme} scheme"
    failures = []
    for name, entry in measured.items():
        assert entry["matches"], f"{name}: {entry['selector']} matched nothing"
        # Every match, not the first (ruling A7): six panes carry this treatment and a rule keyed
        # on :nth-child or on a panel's id would give one of them a different edge.
        for m in entry["matches"]:
            has_edge, edge_ratio, edge_text = _painted_edge(m)
            where = f"{name} match #{m['index']} ({m['selector']}), {scheme} scheme"
            # T-0154 A3 supersedes ruling A12's two arms for the *pane* by name: the approved
            # mockup's pane carries a 1 px --line hairline (well under 3.0:1) AND the tonal step.
            # Amended (G2): a pane with a painted edge below 3.0:1 no longer fails arm (a); every
            # pane, edged or not, must clear the tonal step. The hairline's visibility is A3's
            # pixel row (tests/integration/test_hud_mockup_parity.py, dE00 >= 1). The chip keeps
            # arm (a) unchanged.
            if name == "pane border":
                step, step_text = _tonal_step(m)
                if step + 1e-9 < MIN_TONAL_STEP:
                    failures.append(
                        f"{where}: arm (b), tonal — {step_text}, needs {MIN_TONAL_STEP}:1. "
                        f"({edge_text}; since T-0154 A3 every pane needs the step.) A card with no "
                        "tonal step is not a card: it is text floating on the page background, "
                        "and nothing on the page says where the model's output stops and the "
                        "page's own chrome starts."
                    )
                continue
            if not has_edge:  # the chip, and the focus ring in its own row, keep arm (a) only
                failures.append(
                    f"{where}: arm (a) — {edge_text}. Criterion 8 asks this element for a border "
                    "at >= 3.0:1; ruling A12's tonal alternative is for the *pane* only. A chip "
                    "is a small pill inside the top bar, and a tonal step at that size is not a "
                    "boundary a human resolves."
                )
                continue
            if edge_ratio < 3.0:
                failures.append(
                    f"{where}: arm (a), painted — {edge_text}, needs 3.0:1. "
                    + (
                        "Ruling A12's other option is to remove the border entirely and carry the "
                        f"card on a tonal step instead (>= {MIN_TONAL_STEP}:1 of surface against "
                        "the page); a faint hairline is the one thing that is neither."
                        if name == "pane border"
                        else ""
                    )
                )
    assert failures == [], (
        f"criterion 8 ({scheme} scheme), edges that do not separate what they sit between:\n  "
        + "\n  ".join(failures)
        + "\n(Ruling A12: a pane passes on a painted border at >= 3.0:1 OR on no painted border "
        f"plus a surface-to-background step of >= {MIN_TONAL_STEP}:1, in both schemes. The arm "
        "that was evaluated is named on each line, so which of the two the design chose is never "
        "a guess.)"
    )


# --- criterion 6: the layout holds an unknown number of panes ----------------------------------


def _overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Rectangle intersection with a half-pixel slack for subpixel layout."""
    return (
        a["x"] < b["right"] - 0.5
        and b["x"] < a["right"] - 0.5
        and a["y"] < b["bottom"] - 0.5
        and b["y"] < a["bottom"] - 0.5
    )


def _check_layout(geometry: dict[str, Any], viewport: str, expected_panes: int) -> list[str]:
    problems = []
    rects = geometry["rects"]
    if len(rects) != expected_panes:
        problems.append(f"{len(rects)} panes in <main>, expected {expected_panes}")
    if geometry["scroll_width"] > geometry["client_width"]:
        problems.append(
            f"horizontal overflow at {viewport}: documentElement.scrollWidth "
            f"{geometry['scroll_width']} > clientWidth {geometry['client_width']}"
        )
    for rect in rects:
        if rect["w"] + 1e-6 < MIN_PANE_WIDTH_PX:
            problems.append(
                f"pane {rect['label']!r} is {rect['w']:.0f} px wide at {viewport} "
                f"(floor {MIN_PANE_WIDTH_PX:.0f} px)"
            )
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            if _overlaps(a, b):
                problems.append(f"panes {a['label']!r} and {b['label']!r} overlap at {viewport}")
    return problems


@requires_chrome
@pytest.mark.parametrize("viewport", list(VIEWPORTS))
def test_the_grid_lays_out_the_panes_that_exist_today(page, viewport):
    """Criterion 6, the six panes of the task's Regime.

    Today's ``grid-template-columns: 2fr 1fr 2fr 1fr`` puts the frame thumbnail in a 188 px cell
    at 1280 px — under the 280 px floor — and gives the assists panel twice the width of the
    session log for no reason at all.
    """
    _ok(page)
    problems = _check_layout(page.geometry[viewport], viewport, PANE_COUNT)
    assert problems == [], (
        f"criterion 6 at {viewport} with the {PANE_COUNT} panes that mount today:\n  "
        + "\n  ".join(problems)
        + "\n(No horizontal scrollbar, no pane under 280 px, no two panes overlapping. 280 px is "
        "the width below which the session log's path and the assists' explanations wrap to "
        "uselessness.)"
    )


@requires_chrome
@pytest.mark.parametrize("viewport", list(VIEWPORTS))
def test_the_grid_survives_four_panes_it_was_not_written_for(page, viewport):
    """Criterion 6, adversarial: the failure this task is most likely to ship.

    A grid written for exactly the panes that exist today looks perfect in the implementer's
    screenshot and breaks the moment T-0047's listen panel mounts itself — in a *different* task,
    whose author will reasonably assume the page can hold their panel, because criteria 4 and 6
    promise it can. Four extra ``section.pane`` are injected into ``<main>`` with
    ``Runtime.evaluate``, which is exactly what ``hud/src/assists/index.js`` and
    ``hud/src/log/index.js`` do at load, and the same three properties are asserted.
    """
    _ok(page)
    problems = _check_layout(page.geometry_injected[viewport], viewport, PANE_COUNT + 4)
    assert problems == [], (
        f"criterion 6 at {viewport} with {PANE_COUNT + 4} panes (four injected):\n  "
        + "\n  ".join(problems)
        + "\n(The task's Implementation notes suggest repeat(auto-fit, minmax(320px, 1fr)) — any "
        "rule that names a fixed number of tracks, or keys a pane's treatment off :nth-child, "
        "fails here.)"
    )


@requires_chrome
def test_the_page_fills_the_window(page):
    """Criterion 7, two-sided (T-0056 ruling A8): the content ends *within* 64 px of the fold.

    Measured, not eyeballed: the greatest pane bottom against the viewport height, at 1280x800
    with the six panes of the Regime and the synthetic-capture banner up. Two failures, not one.
    Under-filling is what the task was written about — roughly 300 px of a 1280x800 window was
    empty below the content. Over-filling is what the one-sided form let through: a subtraction
    that only forbids a *shelf* passes a page whose last pane is anywhere below the fold, and at
    the implementation's first pass the lowest pane sat at 820.5 of 800, scrolling the assists
    panel's counter line off the bottom. ``abs()`` is the criterion's own word, "within".

    The second assertion bounds the document itself: a pane may end 20 px past the fold while the
    document scrolls much further, because the bottom-most *pane* is not the bottom-most thing on
    the page. 64 px is the same budget — one line of text and a gap, not a screenful.

    **Amended by T-0128** (docs/tasks/T-0128.md, Rulings, "on d4a36e1", which supersedes the
    earlier "the fill-the-window row stays"). Criterion 4 lets the rail scroll as a whole, and
    with its panes at natural height (about 951 px at 1280x800) the rail ends below the fold by
    design. So the two-sided bottom-edge budget now applies to the conversation column only —
    the spoken, you-said and shown panes — and the rail panes get a reachability check instead:
    with ``<main>`` scrolled to its end, every rail pane's bottom is within the viewport, give or
    take the same 64 px. The document itself still must not scroll (the scrollHeight assertion
    is kept unchanged); scrolling is ``<main>``'s job. **Removed assertion:** ``abs(innerHeight -
    max(bottom over every pane)) <= 64`` — the lowest of *all* panes against the fold. Its
    conversation-column half is kept as the restricted assertion below; its rail half is
    replaced by the reachability assertion.

    **Amended by T-0154 B1** (G2): the conversation column is one pane (the pane holding
    ``[data-thread]``), not three. Changed assertions: ``len(conversation) == 3`` becomes ``== 1``
    and the rail count ``PANE_COUNT - 3`` becomes ``PANE_COUNT - 1``. The budgets are unchanged.
    """
    _ok(page)
    geometry = page.geometry["1280x800"]
    assert geometry["count"] == PANE_COUNT, (
        f"criterion 7 is measured on the {PANE_COUNT} panes of the task's Regime and not on the "
        f"injected ones ({geometry['count']} panes found); the ten-pane rows belong to criterion 6"
    )
    conversation = [r for r in geometry["rects"] if r["conversation"]]
    assert len(conversation) == 1, (
        "T-0154 B1: the bottom-edge budget is measured on the conversation column — the one "
        "conversation pane, the pane holding [data-thread] — and "
        f"{len(conversation)} such panes were found: "
        f"{[(r['label'], r['conversation']) for r in geometry['rects']]}"
    )
    bottom = max(r["bottom"] for r in conversation)
    dead = geometry["inner_height"] - bottom
    assert abs(dead) <= MAX_DEAD_SPACE_PX, (
        f"criterion 7 (T-0128: conversation column only): the conversation pane "
        f"ends {abs(dead):.1f} px "
        f"{'above' if dead > 0 else 'below'} the bottom of a {geometry['inner_height']} px "
        f"viewport (budget: {MAX_DEAD_SPACE_PX:.0f} px on either side; synthetic-capture banner "
        f"up: {geometry['banner_up']}, error strip up: {geometry['error_strip_up']}). "
        f"Conversation panes: {[(r['label'], round(r['bottom'], 1)) for r in conversation]}. "
        "What moves it is the conversation column's height split, the gaps and the chrome above "
        "<main>."
    )
    reach = page.rail_reach
    rail = reach.get("rects", [])
    assert len(rail) == PANE_COUNT - 1, (
        f"T-0128/T-0154 B1: expected the {PANE_COUNT - 1} rail panes (every pane but the "
        f"conversation pane) to be measured with <main> scrolled to its end; got {len(rail)}: {reach}"
    )
    stranded = [
        (r["label"], round(r["bottom"], 1))
        for r in rail
        if r["bottom"] - reach["inner_height"] > MAX_DEAD_SPACE_PX
    ]
    assert stranded == [], (
        f"T-0128 (criterion 4, the rail scrolls as a whole): with <main> scrolled to its end "
        f"(scrollTop reached {reach['main_scrolled']}, scrollHeight {reach['main_scroll_height']}, "
        f"clientHeight {reach['main_client_height']}), these rail panes still end more than "
        f"{MAX_DEAD_SPACE_PX:.0f} px below the bottom of a {reach['inner_height']} px viewport, "
        f"so no scroll reaches them: {stranded}. All rail panes: "
        f"{[(r['label'], round(r['bottom'], 1)) for r in rail]}. A scrollTop of 0 means <main> "
        "is not the scroller (overflow-y, or a height that is not bounded by the window)."
    )
    overflow = geometry["scroll_height"] - geometry["inner_height"]
    assert overflow <= MAX_DEAD_SPACE_PX, (
        f"criterion 7: the document scrolls {overflow:.1f} px at 1280x800 "
        f"(scrollHeight {geometry['scroll_height']}, innerHeight {geometry['inner_height']}, "
        f"budget {MAX_DEAD_SPACE_PX:.0f} px). The pane assertion above can pass while this one "
        "fails: the lowest pane is not the lowest thing on the page, and what ends up under the "
        "fold is the last line of the last panel — at the implementation's first pass, the "
        "assists panel's counter line."
    )


# --- criteria 9 and 11: motion, and states that do not depend on it ------------------------------


def _durations(value: str) -> list[float]:
    return [float(part.strip().rstrip("s")) for part in str(value).split(",") if part.strip()]


@requires_chrome
def test_nothing_animates_when_the_system_asks_for_reduced_motion(page):
    """Criterion 9, behaviourally: under emulated ``reduce``, every duration is off.

    The textual twin is in tests/hud/styles.test.mjs (the block exists, is universal, carries
    !important and is the sheet's last word on motion). This row is the one that notices a rule
    the block does not reach — an animation declared inside a ``<style>`` a panel injects into
    document.head after styles.css loads, for instance, which is how hud/src/assists/index.js and
    hud/src/log/index.js style themselves and how T-0047's listen panel will.
    """
    _ok(page)
    offenders = []
    for row in page.motion_reduced:
        animating = (
            row["animation_name"] != "none"
            and max(_durations(row["animation_duration"]) or [0.0]) > 0.01
        )
        transitioning = max(_durations(row["transition_duration"]) or [0.0]) > 0.01
        if animating or transitioning:
            offenders.append(
                f"{row['tag']}: animation-name={row['animation_name']} "
                f"animation-duration={row['animation_duration']} "
                f"transition-duration={row['transition_duration']}"
            )
    assert offenders == [], (
        "criterion 9: with prefers-reduced-motion: reduce emulated, these elements still animate "
        "or transition for longer than 0.01 s:\n  " + "\n  ".join(offenders) + "\n"
        "The human sits in front of this page while talking; motion at the edge of vision during "
        "a conversation is the thing the media query exists to stop. A universal "
        "`*, *::before, *::after { animation: none !important; transition: none !important }` "
        "inside the block covers a panel's injected <style> as well as this sheet."
    )


@requires_chrome
@pytest.mark.parametrize("motion", ["normal", "reduce"])
def test_all_six_turn_states_look_different(page, motion):
    """Criterion 11, and criterion 9's "no information carried by motion alone".

    ``data-turn-state`` takes six values and today only two of them are styled: IDLE, LISTENING,
    ENDPOINTING and SPEAKING render identically, so the chip answers "what is the assistant doing"
    with "something". The same six are asserted again under emulated reduced motion, because a
    state told apart only by a pulse is a state that disappears for the user who asked the
    system to stop moving things.
    """
    _ok(page)
    chips = page.chips_dark if motion == "normal" else page.chips_reduced
    assert set(chips) == set(TURN_STATES), f"measured {sorted(chips)}, expected {TURN_STATES}"
    seen: dict[str, str] = {}
    collisions = []
    for state in TURN_STATES:
        style = chips[state]
        key = json.dumps(style, sort_keys=True)
        if key in seen:
            collisions.append(f"{state} is drawn exactly like {seen[key]}: {style}")
        seen[key] = state
    assert collisions == [], (
        f"criterion 11 (motion: {motion}): every data-turn-state value must get its own chip "
        "treatment, and these do not:\n  " + "\n  ".join(collisions) + "\n"
        "The chip's *text* is written by hud/src/panels/thinking.js and is not this task's to "
        "touch, so colour, border, weight or shape is the whole vocabulary available. The task's "
        "notes suggest IDLE neutral, LISTENING accent, ENDPOINTING accent-dim, THINKING accent "
        "filled, SPEAKING accent outline, INTERRUPTED warn."
    )


# --- criterion 10: the thinking bar shows a measurement and never invents one --------------------


@requires_chrome
def test_the_thinking_bar_tracks_the_fill_it_is_given_and_never_animates(page):
    """Criterion 10: the bar is a measurement, and the project rule underneath it.

    ``hud/src/panels/thinking.js`` writes ``--fill`` as an inline custom property and positions
    ``.tick`` with an inline ``left``; any bar the restyle draws has to consume both. The marks
    are at 3000 ms and 4000 ms on a scale of 1.5 x 4000 ms (``THINKING_MARKS_MS`` in
    hud/src/state.js), so their positions are 50 % and 66.6667 % — computed here by hand rather
    than read back from the page. And the bar carries no animation in any state: a bar that
    sweeps or pulses while ``thinking_ms`` is unknown is a fake trajectory presented as a real
    measurement, the same rule that keeps the denoising view showing real per-step canvases.
    """
    _ok(page)
    paints = page.fill.get("paints", {})
    assert set(paints) == {"0%", "50%", "100%"}, f"measured fills: {sorted(paints)}"
    distinct = set(paints.values())
    assert len(distinct) == 3, (
        "criterion 10: --fill at 0 %, 50 % and 100 % must produce three different paints on "
        f"#thinking-bar; {len(distinct)} distinct value(s) were measured.\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in sorted(paints.items()))
        + "\nA bar that ignores the inline --fill hud/src/panels/thinking.js sets shows the same "
        "thing whatever the Conductor measured."
    )
    assert page.fill["animation_name"] == "none", (
        f"#thinking-bar computes animation-name={page.fill['animation_name']!r}. Criterion 10: "
        "the bar carries no animation in any state. An indeterminate sweep on a meter whose whole "
        "job is to report a measured number is the page telling the human something it does not "
        "know — CLAUDE.md's rule about never presenting a fake trajectory as real."
    )
    ticks = page.fill["ticks"]
    assert [t["mark_ms"] for t in ticks] == ["3000", "4000"], (
        f"expected the two marks hud/src/panels/thinking.js draws (3000 ms and 4000 ms), got "
        f"{ticks!r}"
    )
    expected = {"3000": 0.5, "4000": 4000 / 6000}
    bar_width = float(page.fill["bar_width"])
    assert bar_width > 0, "the thinking bar has no width"
    for tick in ticks:
        fraction = expected[str(tick["mark_ms"])]
        assert tick["inline_left"], (
            f"the {tick['mark_ms']} ms mark lost its inline left. hud/src/panels/thinking.js "
            "drawMarks() sets it; the sheet has to leave it in charge of the position."
        )
        used = float(str(tick["computed_left"]).rstrip("px") or "nan")
        assert not math.isnan(used) and abs(used - fraction * bar_width) <= 3.0, (
            f"the {tick['mark_ms']} ms mark computes left={tick['computed_left']} on a "
            f"{bar_width:.0f} px bar; hand-computed from THINKING_MARKS_MS on a 1.5x scale it "
            f"belongs at {fraction * bar_width:.1f} px ({fraction * 100:.4f} %). A rule that "
            "positions .tick by class overrides the inline value and the marks stop meaning "
            "3000 ms and 4000 ms."
        )


# --- the focus ring: the typed path is keyboard-only ---------------------------------------------


@requires_chrome
def test_every_control_shows_where_the_keyboard_is(page):
    """The focus indicator (task *Tests to write*, "Focus"; Implementation notes, *Focus.*).

    The ask box is the whole typed path and it is reached with Tab. There is no ring of the
    page's own today: what the four controls show is Chrome's UA ring, whose computed colour on
    this box is rgb(16, 16, 16) — a near-black ring on a near-black bar, 1.06:1. The task's notes
    prescribe ``:focus-visible { outline: 2px solid var(--ring); outline-offset: 2px }`` on every
    interactive element; this row asserts the property that makes it worth having — an indicator
    that exists *and* can be seen against what is behind it.
    """
    _ok(page)
    failures = []
    for selector, m in page.focus.items():
        assert m.get("present"), f"{selector} is not on the page"
        assert m["focused"], f"{selector} did not take focus; the measurement below is meaningless"
        visible = (m["outline_width"] > 0 and m["outline_style"] != "none") or m[
            "box_shadow"
        ] != "none"
        if not visible:
            failures.append(
                f"{selector}: no indicator at all (outline {m['outline_width']}px "
                f"{m['outline_style']}, box-shadow {m['box_shadow']})"
            )
            continue
        behind = painted(m["parent_chain"])
        indicator = parse_css_color(str(m["outline_color"]))
        ratio = contrast_ratio(composite(indicator, behind), behind)
        if m["box_shadow"] != "none" and m["outline_width"] == 0:
            continue  # a ring drawn as a shadow: the colour is inside the shorthand, not measurable
        if ratio < 3.0:
            failures.append(
                f"{selector}: outline {m['outline_color']} is {ratio:.2f}:1 against "
                f"{hexc(behind)} (outline-style: {m['outline_style']}, "
                f":focus-visible matched: {m['focus_visible']})"
            )
    assert failures == [], (
        "the focus indicator is missing or invisible on:\n  " + "\n  ".join(failures) + "\n"
        "outline-style: auto means the UA ring, whose computed colour is not the two-tone it "
        "actually paints and is not under the page's control in either scheme — declare the ring "
        "explicitly against var(--ring). The typed path (T-0039) is reached only by keyboard."
    )


# --- criterion 12: the empty state, behaviourally -----------------------------------------------


@requires_chrome
def test_a_pane_says_it_is_empty_and_gets_out_of_the_way_when_it_is_not(page):
    """Criterion 12, both halves, on the one pane that is genuinely empty in this configuration.

    The half everybody writes is "show a line when there is nothing". The half that matters is
    the negative: the line may never cover, clip or replace content. The row therefore measures
    the pane empty, appends exactly what hud/src/panels/spoken.js appends, and measures again.

    **Amended by T-0154 B1** (G2): the pane is the conversation pane (the one holding
    ``[data-thread]``; was the pane holding the retired ``#spoken-pane``) and the content arrives
    through the page's own thread (one transcript and one utterance, ``EMPTY_STATE_MESSAGES``)
    instead of a hand-built ``section.turn``. Every assertion is kept.
    """
    _ok(page)
    assert page.empty_state.get("pane_present"), (
        "T-0154 B1: no direct child of <main> holds [data-thread] (the conversation pane); "
        f"measured {page.empty_state!r}"
    )
    empty = page.empty_state["empty"]
    assert empty.get("present"), (
        "the conversation pane has no [data-empty] element. Criterion 12: an empty pane says so in "
        "words that live in hud/index.html. Today it is a blank box 220 px tall — the same thing "
        "a broken panel looks like, which is why the human cannot tell the two apart."
    )
    assert empty["display"] != "none" and empty["w"] > 0 and empty["h"] > 0, (
        f"the empty line is not rendered while the pane is empty: display={empty['display']}, "
        f"box {empty['w']}x{empty['h']}. Hiding it with the :has() rule when there is *no* "
        "content inverts the criterion."
    )
    ratio = contrast_ratio(foreground(str(empty["color"]), empty["chain"]), painted(empty["chain"]))
    assert ratio >= 4.5, (
        f"the empty line is {ratio:.2f}:1 ({empty['color']} on {hexc(painted(empty['chain']))}, "
        f"{empty['font_size']}px, text={empty['text']!r}). Criterion 8's floor applies to it like "
        "any other text: today's equivalent, .line.empty in var(--edge), computes to 1.34:1 — "
        "present in the DOM, invisible to the eye, which is worse than no empty state at all "
        "because it cannot be reported as missing."
    )
    after = page.empty_state["with_content"]
    content = after["content"]
    assert content.get("present"), (
        "premise (T-0154 B1/B3): the page's own thread did not render utterance e1-u1 into "
        f"[data-thread] (seed: {page.empty_state.get('seeded')!r})"
    )
    assert content["w"] > 0 and content["h"] > 0, (
        "after the page's own thread rendered an utterance, the content itself "
        f"has no box ({content['w']}x{content['h']}) — the empty-state rule is hiding the pane's "
        "content instead of itself."
    )
    assert after["empty"]["display"] == "none", (
        f"with content in the pane the empty line still computes display={after['empty']['display']!r}. "
        "Criterion 12 hides it with a :has() rule (the task's notes: "
        "a :has() rule keyed on [data-thread], T-0154 B1). If :has() is "
        "unavailable the line may sit *alongside* content — never over it — so the assertion "
        "below is the one that must never be traded away."
    )
    if after["empty"]["display"] != "none":  # pragma: no cover - the fallback the criterion allows
        assert not _overlaps(after["empty"], content), (
            "the empty-state line overlaps the pane's real content. Criterion 12: it may never "
            "cover, clip or replace what the model or the user actually said."
        )


# --- criterion 13: the light scheme is real, not a declaration ----------------------------------


@requires_chrome
def test_the_page_really_follows_the_system_scheme(page):
    """Criterion 13, behaviourally: emulate light, and the page has to be light.

    tests/hud/styles.test.mjs asserts that the light block exists and re-binds only tokens. This
    row asserts the thing that block is *for*: with ``prefers-color-scheme: light`` emulated, the
    page's own background changes, and changes in the direction that word means. A sheet that
    declares ``color-scheme: light dark`` and forgets the palette gives a page whose native
    controls turn light while every surface stays black.
    """
    _ok(page)
    dark = painted([["body", page.body_background["dark"], "1"]])
    light = painted([["body", page.body_background["light"], "1"]])
    assert page.body_background["dark"] != page.body_background["light"], (
        "the body background is "
        f"{page.body_background['dark']} under both emulated schemes. Criterion 13 and ruling "
        "A1 (1): the light theme ships in this task, as a declarations-only re-binding of the "
        "token layer inside @media (prefers-color-scheme: light)."
    )
    assert relative_luminance(light) > relative_luminance(dark), (
        f"the 'light' scheme background {hexc(light)} (luminance "
        f"{relative_luminance(light):.3f}) is not lighter than the dark one {hexc(dark)} "
        f"({relative_luminance(dark):.3f}). Dark is the default and the desk; light is what the "
        "system asks for and what a screenshot in docs/ looks like."
    )


# --- group B: the generated surface, themed through the tokens the bundle already reads ---------

#: Hand-read from tests/fixtures/a2ui/quiz_example.jsonl (the same list
#: tests/integration/test_hud_ui_surface.py freezes): the two tab titles, the five question
#: prompts without their markdown, question 1's four choice labels, and the submit label. Twelve
#: components the model chose to show; criterion 18 says a theme may not remove one of them.
SURFACE_TEXTS = [
    "Quiz",
    "Flashcards",
    "What is 1/2 + 1/4?",
    "Which fraction is equivalent to 0.75?",
    "Simplify 6/8.",
    "2/3 × 3/5 = ?",
    "Which is larger: 5/6 or 7/9?",
    "2/6",
    "3/4",
    "1/8",
    "2/4",
    "Check answers",
]

SURFACE_JS = (
    "(() => {"
    + CHAIN_JS
    + "  const NEEDLES = "
    + json.dumps(SURFACE_TEXTS)
    + """;
  const all = [];
  const walk = (root) => {
    if (!root || !root.querySelectorAll) return;
    for (const el of root.querySelectorAll('*')) { all.push(el); if (el.shadowRoot) walk(el.shadowRoot); }
  };
  walk(document);
  const out = {};
  for (const needle of NEEDLES) {
    const hits = all.filter((e) => (e.textContent || '').includes(needle) &&
      ![...e.children].some((c) => (c.textContent || '').includes(needle)));
    const el = hits[hits.length - 1];
    if (!el) { out[needle] = {present: false}; continue; }
    const s = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    out[needle] = {present: true, selector: needle, tag: el.tagName,
      color: s.color, background: s.backgroundColor,
      font_size: parseFloat(s.fontSize), font_weight: parseInt(s.fontWeight, 10) || 400,
      opacity: parseFloat(s.opacity), decoration: s.textDecorationLine,
      border_color: s.borderTopColor, border_width: parseFloat(s.borderTopWidth) || 0,
      width: r.width, height: r.height, text: needle, chain: chainOf(el)};
  }
  return out;
})()"""
)

#: Criterion 18's adversarial arm. ``--a2ui-color-on-surface`` and ``--a2ui-color-on-background``
#: are two of the ~60 custom properties the vendored bundle reads and two of the ones group B is
#: required to bind (criterion 16); ``color`` is set as well because custom properties and
#: ``color`` both inherit into the renderer's shadow roots, and a theme could hide text through
#: either. Bound to the pane's own background, this is a one-line "theme" that leaves every
#: structural test green and shows the human an empty box.
SABOTAGE_JS = """
(async () => {
  const host = document.querySelector('.a2ui-panel ' + __A2UI_ROOT__) ||
               document.querySelector(__A2UI_ROOT__);
  if (!host) return false;
  // T-0164 B5: once group B is activated the arm is re-aimed at card.css's own tokens — every
  // var(--x) the served sheet reads is bound on the card root to the pane's background (the
  // --a2ui-* names are the vendor's, which nothing reads any more).
  const sheetTokens = __AIM_AT_CARD_CSS__
    ? [...new Set([...(await (await fetch('/src/ui/card.css', {cache: 'no-store'})).text())
        .matchAll(/var\\(\\s*(--[a-zA-Z0-9_-]+)/g)].map((m) => m[1]))]
    : [];
  // T-0154 B1/B4 (G2): the card sits in its turn's slot in the thread, so "the pane's own
  // background" is the background of the <main> child holding it (was #shown-channel's pane).
  const pane = host.closest('main > *');
  if (!pane) return false;
  const behind = getComputedStyle(pane).backgroundColor;
  for (const prop of ['--a2ui-color-on-surface', '--a2ui-color-on-background',
                      '--a2ui-text-color-text', '--a2ui-color-on-primary', ...sheetTokens, 'color']) {
    host.style.setProperty(prop, behind);
  }
  return behind;
})()
""".replace("__A2UI_ROOT__", json.dumps(A2UI_ROOT)).replace(
    "__AIM_AT_CARD_CSS__", "true" if T0164_B else "false"
)


@dataclass
class Surface:
    error: str = ""
    seconds: float = 0.0
    dark: dict[str, Any] = field(default_factory=dict)
    light: dict[str, Any] = field(default_factory=dict)
    sabotaged: dict[str, Any] = field(default_factory=dict)
    sabotage_colour: Any = None
    #: T-0128: the surface under each data-theme (see the phase in ``surface``).
    t0128_error: str = ""
    theme_chosen: dict[str, Any] = field(default_factory=dict)
    theme_after_flip: dict[str, Any] = field(default_factory=dict)
    theme_measured: dict[str, Any] = field(default_factory=dict)
    theme_controls: dict[str, Any] = field(default_factory=dict)


#: T-0128 criterion 10: "the fixture surface's components: >= 3.0:1 for controls". Read here as
#: WCAG 2.1 SC 1.4.11 for the surface's buttons: the control's boundary — its painted fill, or its
#: painted border if it has one — against what is behind it, the better of the two.
SURFACE_CONTROLS_JS = (
    "(() => {"
    + CHAIN_JS
    + """
  const out = [];
  const walk = (root) => {
    if (!root || !root.querySelectorAll) return;
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) walk(el.shadowRoot);
      if (el.tagName !== 'BUTTON' || !el.closest) continue;
      const s = getComputedStyle(el);
      const r = el.getBoundingClientRect();
      if (r.width <= 0 || r.height <= 0) continue;
      const parent = el.parentElement || (el.getRootNode && el.getRootNode().host) || el;
      out.push({text: (el.textContent || '').trim().slice(0, 40),
                border_color: s.borderTopColor, border_width: parseFloat(s.borderTopWidth) || 0,
                chain: chainOf(el), parent_chain: chainOf(parent)});
    }
  };
  const host = document.querySelector('.a2ui-panel ' + __A2UI_ROOT__) ||
               document.querySelector(__A2UI_ROOT__);
  if (host) { walk(host); if (host.shadowRoot) walk(host.shadowRoot); }
  return out;
})()"""
).replace("__A2UI_ROOT__", json.dumps(A2UI_ROOT))  # T-0164 B2: the card host, re-keyed by name


@pytest.fixture(scope="module")
def surface() -> Any:
    """The quiz fixture, through the real UiGate and the real page. Never raises."""
    if CHROME is None:  # pragma: no cover
        yield Surface(error="no chrome")
        return
    result = Surface()
    started = time.monotonic()
    stub = HudStub([]).start()
    server = HudStaticServer(HUD, 0)
    try:
        canvases = gated_canvases()
        url = f"http://127.0.0.1:{server.port}/?source=canvas&ws=" + urllib.parse.quote(
            stub.url, safe=""
        )
        with Chrome(CHROME) as chrome:
            chrome.navigate(url)
            if not stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20):
                result.error = "the page never said astra.hello to the stub"
            for canvas in canvases:
                stub.push(canvas)
                time.sleep(0.4)
            settled = chrome.wait_until(
                "(() => { const all = []; const walk = (r) => { if (!r || !r.querySelectorAll) "
                "return; for (const e of r.querySelectorAll('*')) { all.push(e); if (e.shadowRoot)"
                " walk(e.shadowRoot); } }; walk(document); return all.some((e) => "
                "(e.textContent || '').includes('Which is larger: 5/6 or 7/9?')); })()",
                20,
            )
            if not settled:
                result.error = (
                    "the reassembled quiz surface never rendered (T-0052); group B's rows have "
                    "nothing to judge"
                )
            time.sleep(0.5)  # one more frame for the catalog's injected styles to settle
            result.dark = chrome.evaluate(SURFACE_JS) or {}
            chrome._cmd(
                "Emulation.setEmulatedMedia",
                features=[{"name": "prefers-color-scheme", "value": "light"}],
            )
            time.sleep(0.3)
            result.light = chrome.evaluate(SURFACE_JS) or {}
            chrome._cmd("Emulation.setEmulatedMedia", features=[])
            time.sleep(0.3)
            # T-0128 (criterion 10): the same surface under each data-theme, chosen in the
            # Settings control with the system scheme emulated opposite. Before the sabotage arm,
            # which edits the host's style. Isolated so it cannot fail the T-0056 rows.
            try:
                for theme, opposite in (("light", "dark"), ("dark", "light")):
                    result.theme_chosen[theme] = chrome.evaluate(choose_theme_js(theme)) or {}
                    _media(chrome, [{"name": "prefers-color-scheme", "value": opposite}])
                    result.theme_after_flip[theme] = chrome.evaluate(THEME_STATE_JS) or {}
                    result.theme_measured[theme] = chrome.evaluate(SURFACE_JS) or {}
                    result.theme_controls[theme] = chrome.evaluate(SURFACE_CONTROLS_JS) or []
                chrome.evaluate(choose_theme_js("system"))
                _media(chrome, [])
            except Exception as exc:  # pragma: no cover
                result.t0128_error = f"{type(exc).__name__}: {exc}"
            result.sabotage_colour = chrome.evaluate(SABOTAGE_JS)
            time.sleep(0.3)
            result.sabotaged = chrome.evaluate(SURFACE_JS) or {}
    except Exception as exc:  # pragma: no cover
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        server.close()
        stub.stop()
    result.seconds = time.monotonic() - started
    yield result


@requires_chrome
@pytest.mark.parametrize("scheme", ["dark", "light"])
def test_the_theme_does_not_hide_anything_the_model_chose_to_show(surface, scheme):
    """Criterion 18: every component of the rendered surface is present, boxed and readable.

    This is the shown channel's honesty rule expressed in CSS. Group B changes exactly one file,
    ``hud/src/ui/style.css``, and it changes it by binding the ``--a2ui-*`` custom properties the
    vendored bundle already reads — no vendor diff, no catalog change, no JS. The failure mode
    that buys is a colour binding that removes something the model decided to show *without*
    removing it from the DOM: ``tests/integration/test_hud_ui_surface.py`` stays green because the
    component is still there, and the human sees an empty card.

    The twelve strings are hand-read from tests/fixtures/a2ui/quiz_example.jsonl, and the surface
    under judgement is the reassembled one (T-0052, criterion 19), driven through the real
    ``UiGate`` and the same ``HudStub`` scripts/demo/hud_ui_demo.py uses.
    """
    assert not surface.error, f"the surface session failed: {surface.error}"
    measured = surface.dark if scheme == "dark" else surface.light
    missing = [needle for needle, m in measured.items() if not m.get("present")]
    assert missing == [], (
        f"these components of the quiz surface are not in the DOM at all in the {scheme} scheme "
        f"(shadow roots included): {missing}. Criterion 18: the theme may not remove one."
    )
    collapsed = [
        f"{needle!r} has a {m['width']}x{m['height']} box"
        for needle, m in measured.items()
        if m["width"] <= 0 or m["height"] <= 0
    ]
    assert collapsed == [], (
        f"criterion 18, {scheme} scheme: these components are in the DOM with no box:\n  "
        + "\n  ".join(collapsed)
        + "\n(a card collapsed by a padding or a height binding removes the component as surely "
        "as deleting it, and leaves every structural test green)"
    )
    failures = []
    for needle, m in measured.items():
        ratio = ratio_of(m)
        floor = threshold_for(float(m["font_size"]), int(m["font_weight"]))
        if ratio + 1e-9 < floor:
            failures.append(f"{_describe(needle, m)} — needs {floor}:1")
    assert failures == [], (
        f"criterion 18, {scheme} scheme: components of the model's surface below their contrast "
        "floor:\n  " + "\n  ".join(failures) + "\n(group B binds --a2ui-color-* to group A's "
        "tokens; if a binding cannot clear the floor in both schemes it is the wrong token, not "
        "a threshold to lower)"
    )


@requires_chrome
def test_the_surface_check_catches_text_painted_the_colour_of_its_background(surface):
    """Criterion 18, adversarial — and the proof that the row above is not vacuous.

    Everything the row above asserts is also true of a surface nobody can read, unless the check
    composites and compares. So the same page is re-measured after a one-line "theme": four
    ``--a2ui-color-*`` bindings and ``color`` set to the pane's own background on the
    ``<a2ui-surface>`` host — which is precisely the shape of change group B is allowed to make,
    and precisely the shape that would pass ``test_hud_ui_surface.py`` untouched. If this row does
    not go red, the check above cannot see the failure it exists to see.
    """
    assert not surface.error, f"the surface session failed: {surface.error}"
    assert surface.sabotage_colour, (
        "the sabotage arm could not find the <a2ui-surface> host, so the check below was never "
        "exercised. hud/src/ui/index.js mounts it in its turn's card slot inside the "
        "conversation pane (T-0154 B4)."
    )
    measured = surface.sabotaged
    present = [m for m in measured.values() if m.get("present")]
    assert len(present) >= len(SURFACE_TEXTS) - 1, (
        "the sabotage removed components from the DOM, which is not what it is meant to prove: "
        "the point is a surface that is *technically all there* and invisible."
    )
    invisible = [
        f"{needle!r}: {ratio_of(m):.2f}:1"
        for needle, m in measured.items()
        if m.get("present") and ratio_of(m) < 1.5
    ]
    assert invisible, (
        "after binding --a2ui-color-on-surface (and friends) to the pane's own background colour "
        f"({surface.sabotage_colour}), the contrast walk found nothing below 1.5:1. Either the "
        "sabotage no longer reaches the renderer's shadow roots — in which case the property "
        "names in SABOTAGE_JS need updating against hud/vendor/'s theming surface — or the walk "
        "is not compositing the chain, and the row above would pass for a surface the human "
        "cannot see. Measured: "
        + ", ".join(f"{k}={ratio_of(v):.2f}" for k, v in measured.items() if v.get("present"))
    )


# --- T-0128 criterion 10: the floors, under each data-theme -----------------------------------------
#
# T-0128 Tests to write T4: the contrast rows run under data-theme light and dark, with the frozen
# list extended, and ``MIN_TONAL_STEP`` stays 1.15 (Ruling O1). The T-0056 rows above are kept as
# they are: they measure the page with no stored preference under each emulated scheme, which is
# criterion 6 (b)'s path and must keep holding. These rows measure criterion 6 (a)'s path — the
# theme chosen in the control, the system emulated the other way.


def _theme_premise(chosen: dict[str, Any], after: dict[str, Any], theme: str) -> None:
    assert chosen.get("ok"), (
        f"T-0128 criterion 8: the Settings control could not be used to choose {theme!r}: "
        f"{chosen.get('why') or chosen!r}"
    )
    assert after.get("theme") == theme, (
        f"criterion 6 (a): {theme!r} was chosen in the control and the system scheme then emulated "
        f"the other way; <html data-theme> reads {after.get('theme')!r}. A stored choice wins over "
        "the system, so every measurement below would be of the wrong theme."
    )
    assert after.get("scheme") == theme, (
        f"criterion 6: color-scheme must match data-theme; it computes {after.get('scheme')!r} under "
        f"data-theme={theme!r}. Native controls (the assist checkboxes, the select, the scrollbars) "
        "are drawn in the color-scheme, not in the tokens."
    )


def _contrast_report(measured: dict[str, Any]) -> tuple[list[str], list[str], list[str], list[str]]:
    """(missing selectors, vacuous selectors, failures, excluded lines) — T-0056's rules (A7)."""
    missing = [name for name, e in measured.items() if not e["matches"]]
    failures, vacuous, excluded_lines = [], [], []
    for name, entry in measured.items():
        if not entry["matches"]:
            continue
        usable, excluded = _partition(entry)
        excluded_lines += [f"{name} ({entry['selector']}): {line}" for line in excluded]
        if not usable:
            vacuous.append(f"{name} ({entry['selector']}): no match carries measurable text")
            continue
        for m in usable:
            floor = threshold_for(float(m["font_size"]), int(m["font_weight"]))
            if ratio_of(m) + 1e-9 < floor:
                failures.append(f"{_describe(name, m)} — needs {floor}:1")
    return missing, vacuous, failures, excluded_lines


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_t0128_every_frozen_selector_clears_its_floor_under_data_theme(page, theme):
    """T-0128 criterion 10: the extended frozen list, both themes, chosen in the control.

    The extension: the header pill, the settings control, the rail headings, Diagnostics and the
    Thinking pane's empty line. The floors are T-0056's (4.5:1, 3.0:1 for large text) and do not
    move: "Token values move until the floors hold."

    T-0154 (B1/B3/B7, C4's "both themes"): the thread's rows of ``CONTRAST_SELECTORS`` run here
    too, under each data-theme.
    """
    _ok(page)
    _seeded(page)
    assert not page.t0128_error, f"the T-0128 phase failed: {page.t0128_error}"
    _theme_premise(page.theme_chosen.get(theme, {}), page.theme_after_flip.get(theme, {}), theme)
    missing, vacuous, failures, excluded = _contrast_report(page.contrast_theme.get(theme, {}))
    if excluded:
        print(f"[T-0128 criterion 10, data-theme={theme}] excluded as unmeasurable:")
        for line in excluded:
            print("   " + line)
    assert missing == [], (
        f"data-theme={theme}: these frozen selectors match nothing: {missing}. Criterion 10 extends "
        "T-0056's list with the header pill, the settings control, the rail headings, Diagnostics "
        "and the Thinking empty line; each must exist to be measured."
    )
    assert vacuous == [], f"data-theme={theme}: vacuous selectors:\n  " + "\n  ".join(vacuous)
    assert failures == [], (
        f"T-0128 criterion 10, data-theme={theme}: text below its WCAG 2.1 floor:\n  "
        + "\n  ".join(failures)
        + "\n(The task names the mockup's known failures — light green/amber/red on the card "
        "surface, the pill's accent on its soft fill, muted text on --surface-3, --faint at about "
        "2.3:1 — and the fix: move the token values, not the floors.)"
    )


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_t0154_the_new_context_and_caption_text_clears_its_floor_under_data_theme(page, theme):
    """T-0154 A7 (T-0128 criterion 10, list extended): the Context labels exist, carry text and
    clear 4.5:1 (3.0:1 large) in both themes. The floors do not move.

    **Amended by T-0177 C4** (which supersedes T-0154 A6's caption by name): the caption now
    describes the image the session sent (an ``astra.vision.frame``), and is empty with no frame.
    This module's page has no session (a closed loopback port), so its caption must be present and
    empty although ``?source=canvas`` runs a capture; the caption's contrast floor is measured with a
    frame shown, in both themes, by tests/integration/test_t0177_hud_vision.py
    (``test_c4_the_caption_and_the_combo_clear_their_contrast_floor``)."""
    _ok(page)
    assert not page.t0128_error, f"the T-0128 phase failed: {page.t0128_error}"
    _theme_premise(page.theme_chosen.get(theme, {}), page.theme_after_flip.get(theme, {}), theme)
    measured = dict(page.contrast_t0154.get(theme, {}))
    caption = measured.pop("capture caption", {"matches": []})
    assert len(caption["matches"]) == 1, (
        f"T-0177 C4 (data-theme={theme}): one main > .pane.thumb [data-capture-caption]; "
        f"found {len(caption['matches'])}"
    )
    assert not str(caption["matches"][0].get("text") or "").strip(), (
        f"T-0177 C4 (data-theme={theme}): with no astra.vision.frame the caption is empty, although "
        f"a capture runs; it reads {caption['matches'][0].get('text')!r}"
    )
    missing, vacuous, failures, _excluded = _contrast_report(measured)
    assert missing == [], f"T-0154 A5/A6 (data-theme={theme}): no element matches {missing}"
    assert vacuous == [], f"data-theme={theme}: vacuous selectors:\n  " + "\n  ".join(vacuous)
    assert failures == [], (
        f"T-0154 A7, data-theme={theme}: text below its WCAG 2.1 floor:\n  " + "\n  ".join(failures)
    )


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_t0128_the_unplayed_tail_is_struck_dimmed_and_readable_under_data_theme(page, theme):
    """T-0128 criteria 10 and 11: [data-unplayed] keeps line-through and 0 < opacity < 1, >= 3.0:1."""
    _ok(page)
    _seeded(page)
    assert not page.t0128_error, f"the T-0128 phase failed: {page.t0128_error}"
    _theme_premise(page.theme_chosen.get(theme, {}), page.theme_after_flip.get(theme, {}), theme)
    entry = page.unplayed_theme.get(theme, {}).get("unplayed tail", {"matches": []})
    assert len(entry["matches"]) == 1, "the seeded [data-unplayed] tail is on the page, once"
    m = entry["matches"][0]
    assert "line-through" in str(m["decoration"]), (
        f"criterion 11, data-theme={theme}: [data-unplayed] computes "
        f"text-decoration-line={m['decoration']!r}; it keeps line-through"
    )
    assert 0.0 < float(m["opacity"]) < 1.0, (
        f"criterion 11, data-theme={theme}: [data-unplayed] opacity {m['opacity']} must stay "
        "strictly between 0 and 1"
    )
    assert ratio_of(m) >= 3.0, f"criterion 10, data-theme={theme}: {_describe('unplayed tail', m)}"


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_t0128_chip_edges_and_the_page_to_pane_step_hold_under_data_theme(page, theme):
    """T-0128 criterion 10: chip edges >= 3.0:1, and page to pane the 1.15 tonal floor (Ruling O1).

    The same two arms as T-0056 ruling A12 (a painted pane edge at >= 3.0:1, or none and a tonal
    step of at least ``MIN_TONAL_STEP``), measured on every pane — the rail's included — under the
    theme chosen in the control. The task records the mockup's own page-to-panel step as 1.07
    dark and 1.09 light: both fail this, which is the point of keeping the floor.
    """
    _ok(page)
    assert not page.t0128_error, f"the T-0128 phase failed: {page.t0128_error}"
    assert MIN_TONAL_STEP == 1.15, "Ruling O1 keeps T-0056's 1.15 tonal floor between page and pane"
    _theme_premise(page.theme_chosen.get(theme, {}), page.theme_after_flip.get(theme, {}), theme)
    measured = page.borders_theme.get(theme, {})
    failures = []
    for name, entry in measured.items():
        assert entry["matches"], f"{name}: {entry['selector']} matched nothing"
        for m in entry["matches"]:
            has_edge, edge_ratio, edge_text = _painted_edge(m)
            where = f"{name} match #{m['index']}, data-theme={theme}"
            # Amended by T-0154 A3 (G2), as in the T-0056 row above: a pane's 1 px --line hairline
            # below 3.0:1 is the approved design, not a failure; the tonal step holds for every pane.
            if name == "pane border":
                step, step_text = _tonal_step(m)
                if step + 1e-9 < MIN_TONAL_STEP:
                    failures.append(f"{where}: tonal — {step_text}, needs {MIN_TONAL_STEP}:1")
                continue
            if not has_edge:
                failures.append(f"{where}: {edge_text}; the chip keeps a painted edge")
                continue
            if edge_ratio < 3.0:
                failures.append(f"{where}: {edge_text}, needs 3.0:1")
    assert failures == [], f"T-0128 criterion 10 (data-theme={theme}):\n  " + "\n  ".join(failures)


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_t0128_the_surface_is_readable_and_its_controls_visible_under_data_theme(surface, theme):
    """T-0128 criterion 10: the fixture surface's components, >= 4.5:1 text and >= 3.0:1 controls.

    The quiz fixture through the real UiGate (T-0056 group B's session), re-measured with the
    theme chosen in the control. "Controls" is read as WCAG 2.1 SC 1.4.11 on the surface's
    buttons: the button's painted fill (or its painted border) against what is behind it.
    """
    assert not surface.error, f"the surface session failed: {surface.error}"
    assert not surface.t0128_error, f"the T-0128 surface phase failed: {surface.t0128_error}"
    _theme_premise(
        surface.theme_chosen.get(theme, {}), surface.theme_after_flip.get(theme, {}), theme
    )
    measured = surface.theme_measured.get(theme, {})
    missing = [n for n, m in measured.items() if not m.get("present")]
    assert missing == [], f"data-theme={theme}: components missing from the DOM: {missing}"
    failures = []
    for needle, m in measured.items():
        floor = threshold_for(float(m["font_size"]), int(m["font_weight"]))
        if m["width"] <= 0 or m["height"] <= 0:
            failures.append(f"{needle!r} has a {m['width']}x{m['height']} box")
        elif ratio_of(m) + 1e-9 < floor:
            failures.append(f"{_describe(needle, m)} — needs {floor}:1")
    controls = surface.theme_controls.get(theme) or []
    assert controls, (
        "the quiz surface's 'Check answers' button was not found among the surface's buttons, so "
        "the controls floor would be vacuous"
    )
    for c in controls:
        fill = painted(c["chain"])
        behind = painted(c["parent_chain"])
        ratio = contrast_ratio(fill, behind)
        colour = parse_css_color(str(c["border_color"]))
        if float(c["border_width"]) > 0 and colour[3] > 0:
            ratio = max(ratio, contrast_ratio(composite(colour, behind), behind))
        if ratio + 1e-9 < 3.0:
            failures.append(
                f"button {c['text']!r}: fill {hexc(fill)} on {hexc(behind)} at {ratio:.2f}:1 "
                "(best of fill and border), needs 3.0:1"
            )
    assert failures == [], f"T-0128 criterion 10, surface, data-theme={theme}:\n  " + "\n  ".join(
        failures
    )


# --- T-0164 C2: WCAG AA for the renderer's roles, in both themes --------------------------------
#
# docs/tasks/T-0164.md C2 (T-0128 criterion 10 extended; T-0056 criterion 18). Written by the
# T-0164 C test-author, 2026-10-27. Replaces T-0154 C4's three rows, which measured the vendored
# catalog's classes inside shadow roots and skipped as "superseded by T-0164 C" from group B on:
#   test_t0154_c4_the_cards_text_and_caption_clear_their_floor_under_data_theme -> the C2 text rows;
#   test_t0154_c4_the_cards_checkbox_and_dividers_are_visible_under_data_theme -> the C2 control
#     rows (checkbox edge, O-C (1) kept: the box must be page-drawn) and the divider rows (A3's
#     border rule, O-C (5) kept);
#   test_t0154_c4_the_card_check_catches_text_painted_the_cards_own_colour -> the C2 mutant arm.
#
# One fresh page: the four mock payloads of the census fixture and the role-probe card of
# ``tests/integration/test_hud_card_parity.py`` (every input kind, a default-variant Button, a
# Modal trigger, a caption, tabs; labelled synthetic there), all through ``HudStub`` as one surface
# each. Text is found by the model's own string (the element owning a text node equal to it), never
# by a class, so the implementation's class names stay free; inputs by value and placeholder.
# Floors: 4.5:1, 3.0:1 for large text (T-0128's ``threshold_for``), 3.0:1 for every control edge.
#
# Readings, so a reviewer can disagree:
# - "against its neighbour" for a control: the larger of its painted fill against what is behind
#   it and its painted border (composited over its own fill) against what is behind it (WCAG 1.4.11
#   "visual boundary"), as T-0154's checkbox row read it.
# - Buttons (B reviewers' hand-over: "the default-variant Button and the Modal trigger need a
#   visible boundary, about 1.1:1 today"): the primary, default and Modal-trigger buttons are held
#   to the same 3.0:1 boundary. The borderless variant is a text button by definition and is held
#   to its label's text floor only.
# - Tabs: both pills (selected and unselected) are controls with a 3.0:1 boundary (the dark
#   unselected tab sat at 3.02:1 at B's tip; a token move must re-check it, which this row does).
# - The focus ring: ``focus({focusVisible: true})`` on a text input, a chip, a default button, a
#   tab and a checkbox of the newest card (older cards are inert); a painted outline whose colour,
#   composited over the control's backdrop, clears 3.0:1 against it.
# - The slider: pixels at the thumb (value 50) and on the filled and unfilled track, each 3.0:1
#   against the card behind it.
# - Placeholders carry the model's words (ruling "placeholders are shown text") and get the text
#   floor; the mock's ``--faint`` is the likely failure (task text).

#: role -> [(payload key, the model's string, kind)]: ``text`` (a text node), ``placeholder``,
#: ``value`` (an input's value). The strings are hand-read from the census fixture's final canvases
#: and from the role-probe canvas.
C2_SYNTH = "synthetic"
C2_TEXT: dict[str, list[tuple[str, str, str]]] = {
    "card heading": [
        ("desk-t9", "Shopping List", "text"),
        ("desk-t8", "Expense Summary", "text"),
        ("trial-omelette", "Simple Omelette", "text"),
        ("trial-flight", "Flight Booking", "text"),
        (C2_SYNTH, "Role probe card", "text"),
    ],
    "card text": [
        ("desk-t9", "225g unsalted butter", "text"),
        ("desk-t9", "175g granulated sugar", "text"),
        ("trial-omelette", "Whisk 2 eggs with salt and pepper.", "text"),
        ("trial-omelette", "Slide onto a plate and serve immediately.", "text"),
        (C2_SYNTH, "Body text this test wrote", "text"),
        (C2_SYNTH, "First panel", "text"),
    ],
    "caption": [(C2_SYNTH, "A caption this test wrote", "text")],
    "key": [
        ("desk-t8", "Rent:", "text"),
        ("desk-t8", "Phone:", "text"),
        (C2_SYNTH, "Key one", "text"),
    ],
    "value": [
        ("desk-t8", "1200", "text"),
        ("desk-t8", "30", "text"),
        (C2_SYNTH, "Value two", "text"),
    ],
    "row after a Divider": [("desk-t8", "Total:", "text"), ("desk-t8", "1925", "text")],
    "field label": [
        ("trial-flight", "From:", "text"),
        ("trial-flight", "Passengers:", "text"),
        (C2_SYNTH, "Field label", "text"),
        (C2_SYNTH, "Amount", "text"),
        (C2_SYNTH, "Size", "text"),
    ],
    "checkbox label": [(C2_SYNTH, "Unticked", "text"), (C2_SYNTH, "Ticked", "text")],
    "chip label, unselected": [
        ("trial-flight", "1", "text"),
        ("trial-flight", "5", "text"),
        (C2_SYNTH, "Small", "text"),
    ],
    "chip label, selected": [(C2_SYNTH, "Large", "text")],
    "button label": [
        ("trial-flight", "Search Flights", "text"),
        (C2_SYNTH, "Primary button", "text"),
        (C2_SYNTH, "Default button", "text"),
        (C2_SYNTH, "Borderless button", "text"),
        (C2_SYNTH, "More details", "text"),
        # ruling 2396676a (b): the disclosure control drawn for a Modal whose trigger is a Text
        (C2_SYNTH, "Show more", "text"),
    ],
    "tab label, selected": [(C2_SYNTH, "First tab", "text")],
    "tab label, unselected": [(C2_SYNTH, "Second tab", "text")],
    "input value": [(C2_SYNTH, "Typed value", "value")],
    "input placeholder": [
        ("trial-flight", "Origin city", "placeholder"),
        ("trial-flight", "Destination city", "placeholder"),
    ],
}


def _ids(*ids: str, inner: str | None = None) -> str:
    out = []
    for cid in ids:
        base = f'[data-a2-id="{cid}"]'
        out += [f"{base}:is({inner})", f"{base} :is({inner})"] if inner else [base]
    return ", ".join(out)


#: role -> [(payload key, selector)], each found element held to 3.0:1 (``control``) or, for the
#: dividers, to A3's dE00 >= 1 border rule. Hand counts in ``C2_CONTROL_COUNTS``.
C2_CONTROLS: dict[str, list[tuple[str, str]]] = {
    "input edge": [
        ("trial-flight", _ids("f1", "f2", "d1", "d2", inner="input, textarea")),
        (C2_SYNTH, _ids("tf", inner="input, textarea")),
    ],
    "chip edge": [
        ("trial-flight", _ids("p1", inner="button")),
        (C2_SYNTH, _ids("cp", inner="button")),
    ],
    "checkbox edge": [(C2_SYNTH, _ids("cb0", "cb1", inner="input[type=checkbox]"))],
    "button boundary, primary": [
        ("trial-flight", _ids("btn", inner="button")),
        (C2_SYNTH, _ids("bpri", inner="button")),
    ],
    "button boundary, default": [(C2_SYNTH, _ids("bdef", inner="button"))],
    "button boundary, Modal trigger": [(C2_SYNTH, _ids("mdt", inner="button"))],
    "button boundary, Modal disclosure": [(C2_SYNTH, _ids("md2", inner="button"))],
    "tab boundary": [(C2_SYNTH, '[data-a2-id="tabs"] [role="tab"]')],
    "divider": [
        ("desk-t8", _ids("d")),
        ("desk-t8", _ids("r2", "r3", "r4", "r5")),
        ("desk-t9", '[data-a2-id="l"] > * + *'),
    ],
}
C2_CONTROL_COUNTS = {
    "input edge": 5,
    "chip edge": 7,
    "checkbox edge": 2,
    "button boundary, primary": 2,
    "button boundary, default": 1,
    "button boundary, Modal trigger": 1,
    "button boundary, Modal disclosure": 1,
    "tab boundary": 2,
    "divider": 1 + 4 + 8,
}
#: Focus-ring targets on the newest (role-probe) card.
C2_FOCUS: list[tuple[str, str]] = [
    (C2_SYNTH, _ids("tf", inner="input, textarea")),
    (C2_SYNTH, _ids("cp", inner="button")),
    (C2_SYNTH, _ids("bdef", inner="button")),
    (C2_SYNTH, '[data-a2-id="tabs"] [role="tab"]'),
    (C2_SYNTH, _ids("cb0", inner="input[type=checkbox]")),
    # ruling 2396676a (b): the Text-trigger Modal's disclosure control and the slider
    (C2_SYNTH, _ids("md2", inner="button")),
    (C2_SYNTH, _ids("sl", inner="input[type=range]")),
]
#: The roles whose text sits directly on the card's own fill: the mutant arm (text painted the
#: card's colour) must turn every one of them red.
C2_ON_CARD = (
    "card heading",
    "card text",
    "caption",
    "key",
    "value",
    "row after a Divider",
    "field label",
    "checkbox label",
    "chip label, unselected",
)
#: Roles red at group B's tip 77094da0 (run 2026-10-27, Chrome 147.0.7727.101, both themes); they
#: were pending until group C's activation (markers removed 2026-10-27 against 313654e6). Measured: the default-variant Button and the Modal
#: trigger are --surface-3 on the --surface-2 card with a transparent border, 1.15:1 dark and
#: 1.10:1 light; the native slider's unfilled track is #3b3b3b on #22262f (1.35:1) dark and
#: #efefef on #f4f5f9 (1.06:1) light. Every other C2 role holds at B's tip and is a guard.
C2_RED_AT_B: frozenset[str] = frozenset(
    {"button boundary, default", "button boundary, Modal trigger", "slider"}
)
CONTROL_FLOOR = 3.0

C2_JS = (
    "(() => {"
    + CHAIN_JS
    + """
  const SIDS = __SIDS__, TEXT = __TEXT__, CTRL = __CTRL__;
  const rootOf = (key) => {
    const w = document.querySelector('[data-ui-surface="' + SIDS[key] + '"]');
    return w ? w.querySelector('[data-a2ui-root]') : null;
  };
  const vis = (el) => {
    if (!el) return false;
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
  };
  const meas = (el, color) => {
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    return {selector: el.tagName.toLowerCase(), index: 0, pseudo: null, color,
            width: r.width, height: r.height, text: (el.textContent || el.value || '').trim().slice(0, 48),
            font_size: parseFloat(s.fontSize), font_weight: parseInt(s.fontWeight, 10) || 400,
            opacity: parseFloat(s.opacity), chain: chainOf(el)};
  };
  const text = [];
  for (const [role, key, str, kind] of TEXT) {
    const root = rootOf(key), found = [];
    if (root && kind === 'text') {
      const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      for (let n = w.nextNode(); n; n = w.nextNode()) {
        if (n.textContent.trim() === str && vis(n.parentElement)) {
          found.push(meas(n.parentElement, getComputedStyle(n.parentElement).color));
        }
      }
    } else if (root) {
      for (const i of root.querySelectorAll('input, textarea')) {
        if (!vis(i)) continue;
        if (kind === 'placeholder' && i.placeholder === str && i.value === '') {
          found.push(Object.assign(meas(i, getComputedStyle(i, '::placeholder').color), {text: str}));
        }
        if (kind === 'value' && i.value === str) found.push(meas(i, getComputedStyle(i).color));
      }
    }
    text.push({role, key, str, kind, found});
  }
  const box = (el) => {
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    return {tag: el.tagName.toLowerCase(), id: el.getAttribute('data-a2-id'),
            border_color: s.borderTopColor, border_width: parseFloat(s.borderTopWidth) || 0,
            border_style: s.borderTopStyle, appearance: s.appearance, w: r.width, h: r.height,
            checked: el.checked === true,
            chain: chainOf(el), parent_chain: chainOf(el.parentElement || el)};
  };
  const ctrl = [];
  for (const [role, key, sel] of CTRL) {
    const root = rootOf(key);
    ctrl.push({role, key, sel, found: root ? [...root.querySelectorAll(sel)].filter(vis).map(box) : []});
  }
  const probe = document.createElement('div');
  probe.style.cssText = 'position:fixed;left:-50px;top:0;width:1px;height:1px;background-color:var(--accent)';
  document.body.appendChild(probe);
  const accent = getComputedStyle(probe).backgroundColor;
  probe.remove();
  return {text, ctrl, accent, theme: document.documentElement.getAttribute('data-theme')};
})()"""
)

#: C2 "the focus ring": each control focused with ``focusVisible``, its computed outline read, then
#: blurred. The ring sits on whatever is behind the control (its parent's painted backdrop).
C2_FOCUS_JS = (
    "(() => {"
    + CHAIN_JS
    + """
  const SIDS = __SIDS__, FOCUS = __FOCUS__;
  const out = [];
  for (const [key, sel] of FOCUS) {
    const w = document.querySelector('[data-ui-surface="' + SIDS[key] + '"]');
    const el = w && w.querySelector('[data-a2ui-root]') && w.querySelector('[data-a2ui-root]').querySelector(sel);
    if (!el) { out.push({sel, found: false}); continue; }
    el.focus({focusVisible: true});
    const s = getComputedStyle(el);
    out.push({sel, found: true, focused: document.activeElement === el,
              visible: el.matches(':focus-visible'), style: s.outlineStyle,
              width: parseFloat(s.outlineWidth) || 0, color: s.outlineColor,
              parent_chain: chainOf(el.parentElement || el)});
    el.blur();
  }
  return out;
})()"""
)

#: C2 "the slider track and thumb": the range scrolled into view and its box read; the pixels are
#: captured from Python (thumb at the middle, value 50 of 0..100; the filled and the unfilled track
#: at a quarter and at four fifths of the width, all on the vertical middle).
C2_SLIDER_JS = (
    "(() => {"
    + CHAIN_JS
    + """
  const w = document.querySelector('[data-ui-surface="' + __SID__ + '"]');
  const el = w && w.querySelector('[data-a2ui-root] [data-a2-id="sl"] input[type=range], '
                                  + '[data-a2ui-root] input[type=range][data-a2-id="sl"]');
  if (!el) return null;
  el.scrollIntoView({block: 'center', inline: 'nearest', behavior: 'instant'});
  const r = el.getBoundingClientRect();
  return {left: r.left, top: r.top, w: r.width, h: r.height, value: el.value,
          parent_chain: chainOf(el.parentElement || el)};
})()"""
)

#: C2's adversarial arm (T-0056 criterion 18's shape; task: "a mutant sheet that paints card text in
#: the card's own colour turns the row red"): one adopted document sheet per card that paints its
#: text, and its inputs' placeholders, the card's own computed background.
CARD_SABOTAGE_JS = """
(() => {
  let n = 0;
  const rules = [];
  for (const w of document.querySelectorAll('[data-ui-surface]')) {
    const card = w.querySelector('[data-a2ui-root]');
    if (!card) continue;
    const t = '[data-ui-surface="' + w.getAttribute('data-ui-surface') + '"] [data-a2ui-root]';
    rules.push(t + ', ' + t + ' *, ' + t + ' *::placeholder { color: '
               + getComputedStyle(card).backgroundColor + ' !important; }');
    n += 1;
  }
  const sheet = new CSSStyleSheet();
  sheet.replaceSync(rules.join('\\n'));
  document.adoptedStyleSheets = [...document.adoptedStyleSheets, sheet];
  return n;
})()
"""


@dataclass
class CardPage:
    error: str = ""
    seconds: float = 0.0
    sids: dict[str, str] = field(default_factory=dict)
    theme_chosen: dict[str, Any] = field(default_factory=dict)
    theme_after_flip: dict[str, Any] = field(default_factory=dict)
    measured: dict[str, Any] = field(default_factory=dict)
    focus: dict[str, Any] = field(default_factory=dict)
    slider: dict[str, Any] = field(default_factory=dict)
    sabotaged_cards: int = 0
    sabotaged: dict[str, Any] = field(default_factory=dict)


@pytest.fixture(scope="module")
def card_page() -> Any:
    """The four mock payloads and the role-probe card in one fresh page, measured under each
    data-theme (chosen in the Settings control, the system scheme emulated opposite), then once
    more under the mutant sheet. Never raises."""
    if CHROME is None:  # pragma: no cover
        yield CardPage(error="no chrome")
        return
    # imported here: test_hud_card_parity imports this module at load time
    from tests.integration.test_hud_card_parity import (
        PAYLOAD_KEYS,
        payload_surface,
        surface_ids,
        synthetic_surface,
    )
    from tests.integration.test_hud_mockup_parity import _pixel

    result = CardPage()
    started = time.monotonic()
    stub = HudStub([]).start()
    server = HudStaticServer(HUD, 0)
    try:
        sids = surface_ids()
        result.sids = sids
        text_spec = [[r, k, s, kind] for r, rows in C2_TEXT.items() for k, s, kind in rows]
        ctrl_spec = [[r, k, sel] for r, rows in C2_CONTROLS.items() for k, sel in rows]
        measure = (
            C2_JS.replace("__SIDS__", json.dumps(sids))
            .replace("__TEXT__", json.dumps(text_spec))
            .replace("__CTRL__", json.dumps(ctrl_spec))
        )
        focus_js = C2_FOCUS_JS.replace("__SIDS__", json.dumps(sids)).replace(
            "__FOCUS__", json.dumps(C2_FOCUS)
        )
        slider_js = C2_SLIDER_JS.replace("__SID__", json.dumps(sids[C2_SYNTH]))
        url = f"http://127.0.0.1:{server.port}/?ws=" + urllib.parse.quote(stub.url, safe="")
        with Chrome(CHROME) as chrome:
            _set_viewport(chrome, 1440, 900)
            _media(chrome, [{"name": "prefers-reduced-motion", "value": "reduce"}])
            # a headless page never has window focus, so :focus-visible never matches without it
            chrome._cmd("Emulation.setFocusEmulationEnabled", enabled=True)
            chrome.navigate(url)
            if not stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20):
                result.error = "the page never said astra.hello to the stub"
            stub.wait_for_client(10)
            # the thread and the ui module mount after the socket opens; a surface pushed before
            # them is dropped, so wait for every pane (as test_hud_redesign's wait_panes does)
            chrome.wait_until(f"document.querySelectorAll('main > .pane').length >= {PANE_COUNT}")
            time.sleep(0.5)
            for key in PAYLOAD_KEYS:
                stub.push(payload_surface(key))
                time.sleep(0.05)
            stub.push(synthetic_surface())
            ready = chrome.wait_until(
                "(() => " + json.dumps(list(sids.values())) + ".every((id) => { const w = "
                "document.querySelector('[data-ui-surface=\"' + id + '\"]');"
                " return !!(w && w.querySelector('[data-a2ui-root] [data-a2-id]')); }))()",
                20,
            )
            if not ready:
                result.error = "the five cards never all rendered"
            time.sleep(0.3)
            for theme, opposite in (("light", "dark"), ("dark", "light")):
                result.theme_chosen[theme] = chrome.evaluate(choose_theme_js(theme)) or {}
                _media(
                    chrome,
                    [
                        {"name": "prefers-color-scheme", "value": opposite},
                        {"name": "prefers-reduced-motion", "value": "reduce"},
                    ],
                )
                result.theme_after_flip[theme] = chrome.evaluate(THEME_STATE_JS) or {}
                result.measured[theme] = chrome.evaluate(measure) or {}
                result.focus[theme] = chrome.evaluate(focus_js) or []
                box = chrome.evaluate(slider_js)
                if box:
                    chrome.evaluate(
                        "new Promise((r) => requestAnimationFrame(() => "
                        "requestAnimationFrame(() => r(true))))"
                    )
                    y = int(box["top"] + box["h"] / 2)
                    box["pixels"] = {
                        name: _pixel(chrome, int(box["left"] + box["w"] * share), y)
                        for name, share in (("thumb", 0.5), ("filled", 0.25), ("unfilled", 0.8))
                    }
                result.slider[theme] = box or {}
            result.sabotaged_cards = int(chrome.evaluate(CARD_SABOTAGE_JS) or 0)
            time.sleep(0.2)
            result.sabotaged = chrome.evaluate(measure) or {}
            chrome.evaluate(choose_theme_js("system"))
            _media(chrome, [])
    except Exception as exc:  # pragma: no cover
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        server.close()
        stub.stop()
    result.seconds = time.monotonic() - started
    yield result


def _card_premise(card_page: CardPage, theme: str) -> dict[str, Any]:
    assert not card_page.error, f"the card session failed: {card_page.error}"
    _theme_premise(
        card_page.theme_chosen.get(theme, {}), card_page.theme_after_flip.get(theme, {}), theme
    )
    report = dict(card_page.measured.get(theme) or {})
    assert report.get("theme") == theme, f"premise: data-theme {report.get('theme')} != {theme}"
    return report


def _text_failures(report: dict[str, Any], role: str) -> tuple[list[str], list[str]]:
    """(missing strings, floor failures) for one text role."""
    missing, failures = [], []
    for entry in report.get("text") or []:
        if entry["role"] != role:
            continue
        where = f"{entry['key']} {entry['str']!r} ({entry['kind']})"
        if not entry["found"]:
            missing.append(where)
        for m in entry["found"]:
            reason = unmeasurable(m)
            if reason:
                failures.append(f"{where}: {reason}")
                continue
            floor = threshold_for(float(m["font_size"]), int(m["font_weight"]))
            ratio = ratio_of(m)
            if ratio + 1e-9 < floor:
                fg = parse_css_color(str(m["color"]))
                failures.append(
                    f"{where}: {m['color']} ({hexc(fg[:3])}) on {hexc(painted(list(m['chain'])))} "
                    f"at {ratio:.2f}:1, needs {floor}:1"
                )
    return missing, failures


@requires_chrome
@pytest.mark.parametrize(
    ("role", "theme"),
    [pytest.param(r, t, id=f"{r}-{t}") for r in C2_TEXT for t in ("dark", "light")],
)
def test_t0164_c2_the_text_role_clears_its_wcag_floor(card_page, role, theme):
    """C2: every text role clears 4.5:1 (3.0:1 large) against its painted background, opacity
    included, under each data-theme; every string of the role is found on the page (a role that
    did not render is not passed silently). A token that fails is fixed in card.css, never here."""
    report = _card_premise(card_page, theme)
    missing, failures = _text_failures(report, role)
    assert missing == [], f"T-0164 C2 {role}, data-theme={theme}: not rendered: {missing}"
    assert failures == [], f"T-0164 C2 {role}, data-theme={theme}:\n  " + "\n  ".join(failures)


def _edge_ratio(m: dict[str, Any]) -> tuple[float, tuple[float, float, float]]:
    behind = painted(list(m["parent_chain"]))
    fill = painted(list(m["chain"]))
    ratio = contrast_ratio(fill, behind)
    colour = parse_css_color(str(m["border_color"]))
    if m["border_width"] > 0 and colour[3] > 0 and m["border_style"] not in ("none", "hidden"):
        ratio = max(ratio, contrast_ratio(composite(colour, fill), behind))
    return ratio, behind


@requires_chrome
@pytest.mark.parametrize(
    ("role", "theme"),
    [pytest.param(r, t, id=f"{r}-{t}") for r in C2_CONTROLS for t in ("dark", "light")],
)
def test_t0164_c2_the_control_is_visible_against_its_neighbour(card_page, role, theme):
    """C2: every non-text control clears 3.0:1 against what is behind it (its fill or its border,
    whichever is stronger); a checkbox is page-drawn (``appearance: none``, O-C (1) kept: a native
    box's computed border is not what Chrome paints). A divider (the Divider, the hairline between
    key/value rows and between list rows) is a painted top border dE00 >= 1 from its backdrop (A3's
    border rule, O-C (5) kept). The hand counts of ``C2_CONTROL_COUNTS`` must all be found."""
    from tests.integration.test_hud_mockup_parity import BORDER_MIN_DE, delta_e

    report = _card_premise(card_page, theme)
    found = [m for e in report.get("ctrl") or [] if e["role"] == role for m in e["found"]]
    assert len(found) == C2_CONTROL_COUNTS[role], (
        f"{role}: found {len(found)} elements, hand count {C2_CONTROL_COUNTS[role]}"
    )
    failures = []
    for m in found:
        where = f"{role} {m['tag']}[data-a2-id={m['id']}]"
        if role == "divider":
            colour = parse_css_color(str(m["border_color"]))
            if not (m["border_width"] > 0 and colour[3] > 0 and m["border_style"] == "solid"):
                failures.append(f"{where}: no painted top border ({m['border_color']})")
                continue
            own = painted(list(m["chain"]))
            de = delta_e(composite(colour, own), own)
            if de + 1e-9 < BORDER_MIN_DE:
                failures.append(f"{where}: line dE00 {de:.2f} from its backdrop")
            continue
        if role == "checkbox edge" and m.get("appearance") != "none":
            failures.append(f"{where}: appearance {m.get('appearance')!r} (native box)")
            continue
        ratio, behind = _edge_ratio(m)
        if ratio + 1e-9 < CONTROL_FLOOR:
            failures.append(
                f"{where}: fill {hexc(painted(list(m['chain'])))} / border {m['border_color']} "
                f"on {hexc(behind)}: {ratio:.2f}:1, needs {CONTROL_FLOOR}:1"
            )
    assert failures == [], f"T-0164 C2 {role}, data-theme={theme}:\n  " + "\n  ".join(failures)


@requires_chrome
@pytest.mark.parametrize(
    "theme",
    ["dark", "light"],
)
def test_t0164_c2_the_focus_ring_clears_the_control_floor(card_page, theme):
    """C2 "the focus ring": a text input, a chip, a default button, a tab and a checkbox, each
    focused visibly, paint an outline (style not none, width >= 1 px) whose colour composited over
    the control's backdrop clears 3.0:1 against that backdrop."""
    _card_premise(card_page, theme)
    rows = card_page.focus.get(theme) or []
    assert len(rows) == len(C2_FOCUS), rows
    failures = []
    for r in rows:
        if not r.get("found") or not r.get("focused") or not r.get("visible"):
            failures.append(f"{r['sel']}: not focused visibly ({r})")
            continue
        behind = painted(list(r["parent_chain"]))
        ring = composite(parse_css_color(str(r["color"])), behind)
        ratio = contrast_ratio(ring, behind)
        if r["style"] in ("none", "hidden") or r["width"] < 1 or ratio + 1e-9 < CONTROL_FLOOR:
            failures.append(
                f"{r['sel']}: outline {r['style']} {r['width']}px {r['color']} on {hexc(behind)}: "
                f"{ratio:.2f}:1, needs {CONTROL_FLOOR}:1"
            )
    assert failures == [], f"T-0164 C2 focus ring, data-theme={theme}:\n  " + "\n  ".join(failures)


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_t0164_c2_the_slider_track_and_thumb_clear_the_control_floor(card_page, theme):
    """C2 "the slider track and thumb": the pixels at the thumb (value 50 of 0..100, the middle),
    on the filled track (a quarter in) and on the unfilled track (four fifths in) each clear 3.0:1
    against the card behind the slider."""
    _card_premise(card_page, theme)
    box = card_page.slider.get(theme) or {}
    assert box.get("pixels"), f"the slider was not found or not captured: {box}"
    assert str(box.get("value")) == "50", f"premise: the slider's value is {box.get('value')}"
    behind = painted(list(box["parent_chain"]))
    failures = []
    for name, px in box["pixels"].items():
        ratio = contrast_ratio(tuple(px), behind)
        if ratio + 1e-9 < CONTROL_FLOOR:
            failures.append(f"{name}: {hexc(tuple(px))} on {hexc(behind)} at {ratio:.2f}:1")
    assert failures == [], f"T-0164 C2 slider, data-theme={theme}:\n  " + "\n  ".join(failures)


#: dE00 by which a ticked box's fill must differ from an unticked one's ("just noticeable").
TICK_JND = 2.3


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_t0164_c2_a_ticked_checkbox_is_filled_with_the_accent_and_differs_from_an_unticked_one(
    card_page, theme
):
    """Ruling 2396676a (b), replacing the retired T-0154 C3 checkbox rows: the ticked box ``cb1``
    paints the page's ``--accent`` (resolved in the page, dE00 <= 2), and its painted fill differs
    from the unticked ``cb0``'s by dE00 >= 2.3 (a just-noticeable difference)."""
    from tests.integration.test_hud_mockup_parity import COLOUR_TOL_DE, delta_e

    report = _card_premise(card_page, theme)
    boxes = [
        m for e in report.get("ctrl") or [] if e["role"] == "checkbox edge" for m in e["found"]
    ]
    ticked = [m for m in boxes if m["checked"]]
    unticked = [m for m in boxes if not m["checked"]]
    assert len(ticked) == 1 and len(unticked) == 1, f"premise: one ticked, one unticked: {boxes}"
    on, off = painted(list(ticked[0]["chain"])), painted(list(unticked[0]["chain"]))
    accent = parse_css_color(str(report["accent"]))[:3]
    assert delta_e(on, accent) <= COLOUR_TOL_DE, (
        f"data-theme={theme}: the ticked box paints {hexc(on)}, not --accent {hexc(accent)} "
        f"(dE00 {delta_e(on, accent):.2f})"
    )
    assert delta_e(on, off) >= TICK_JND, (
        f"data-theme={theme}: ticked {hexc(on)} and unticked {hexc(off)} differ by dE00 "
        f"{delta_e(on, off):.2f} < {TICK_JND}"
    )


@requires_chrome
def test_t0164_c2_the_check_catches_card_text_painted_the_cards_own_colour(card_page):
    """C2 / T-0056 criterion 18, adversarial (kept as a live arm): after one document sheet paints
    every card's text and placeholders the card's own background, every text role that sits on the
    card's own fill (``C2_ON_CARD``) finds failures. If not, the measurement
    reads the wrong element or does not composite the chain, and the rows above would pass a card
    nobody can read."""
    assert not card_page.error, f"the card session failed: {card_page.error}"
    assert card_page.sabotaged_cards >= 5, (
        f"the mutant reached {card_page.sabotaged_cards} card(s); all five are needed"
    )
    report = dict(card_page.sabotaged or {})
    clean = [r for r in C2_ON_CARD if not _text_failures(report, r)[1]]
    assert clean == [], (
        f"after painting card text the card's colour these roles still pass: {clean}"
    )
