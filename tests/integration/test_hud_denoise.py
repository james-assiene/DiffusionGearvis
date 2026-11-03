"""T-0128c group B / T-0159 group B — the real denoising view in the Thinking pane, in headless Chrome.

docs/tasks/T-0128c.md criteria B3, B5, B7, B8, B9 (the browser half) and docs/tasks/T-0159.md
B-2 .. B-10 (the browser half; *Tests to write* "B, Chrome"). The node half (cell model, labels,
focus, retention, the digest, the token cells and sections) is ``tests/hud/denoise.test.mjs``; the
label parity and the text contrast rows are ``tests/integration/test_hud_denoise_parity.py``.

**T-0159 rewrite.** T-0159 B-6 replaces the 2D-canvas bitmap with a DOM grid of token cells, one
native ``<details>`` section per canvas, and B-9 replaces T-0128c B10's pixel rows. The rows that
pinned the bitmap (a ``<canvas>`` created at the first frame; ``getImageData`` reads; 32 square
columns filling the pane; the bitmap sized for devicePixelRatio; a theme switch bumping the draw
counter) are rewritten to the DOM form below (written pending T-0159 at 13122b5, activated with group B). The T-0128c rows
that still hold (B5's empty state and the two refusals, the harness self-checks) are unchanged.

The DOM API read here is the one ``tests/hud/denoise.test.mjs``'s header pins: the root's element
children are the ``<details>`` sections in arrival order; a ``<summary>`` per section with its
label; ``[data-denoise-cell]`` cells in position order; a cell's state in ``data-state`` (or its
one class token naming a state); its text the display form of its piece, its ``title`` the full
piece; a special piece marked ``data-special`` (or class ``special``).

**Driving.** One headless Chrome for the module (``_Recorder`` from test_hud_redesign.py, the CDP
helpers of test_hud_headless.py); one ``HudStub`` and one production ``HudStaticServer`` per row,
each on its own loopback ephemeral port (so each row's origin and ``localStorage`` are fresh). The
stub pushes **hand-built** ``AstraCanvasStep`` frames (built through the real contract, so every
id/mask is one the relay could publish). Frames carrying the T-0159 A1 fields (``argmax_pieces``,
``committed_pieces``) are pushed as raw JSON (the contract's encoding plus the pieces): until group A
lands, ``hud/src/protocol.js`` refuses them as malformed, which is why every such row is pending on
T-0159 and B merges after A.

**The frames** (CL 256, max_steps 48, one request, canvas 0):

- ``F1`` step 6: positions 0-127 have ``canvas == argmax``, 128-255 do not; stable 0-47.
- ``F2`` step 7: argmax of 32-47 changed (they were stable in F1, so they are **revised**, O-B (4));
  stable 0-31; 48-127 ``canvas == argmax``, not stable (**kept**); 128-255 **noise**.
  Counts: stable 32, kept 80, noise 128, revised 16. Label ``canvas 0 · step 7/48``.
- ``C`` commit at step 20 with 100 committed ids: positions 0-99 committed, 100-255 empty. Label
  ``canvas 0 · committed at step 20/48``.

**Readings the task leaves open** (stated so a reviewer can disagree):

- a cell's fill is read from the screen at the cell's horizontal centre, 2 px below its top border
  (inside any pill radius, above a vertically centred one-letter piece), after the cell is scrolled
  into view (B-6: the grid scrolls inside the pane). The expected colour is T-0128c B3's token at its
  alpha composited over the Thinking pane's computed background ("keep the current colours").
- "8 columns in the 340 px rail" is checked at 1440x900, where the rail's content box is ~342 px; at
  1280 and 1920 the column count is reported, and every viewport must show equal-width cells and no
  horizontal overflow.
- "the canvas grid scrolls inside the pane (a bounded height)": some element from a cell up to and
  including ``[data-denoise-root]`` scrolls vertically (``overflow-y`` auto/scroll, content taller
  than its box) for a CL 256 canvas. A pane that scrolls as a whole does not count.
- "text changes in place without transitions": no cell, nor anything inside one, has a transition
  of a non-zero duration on ``all``, ``color``, ``opacity`` or ``font-*``, or an animation.
- the aria-label (T-0128c B3, O-B (3)) is not restated by T-0159; it is kept, carried by the newest
  section or an element inside it.
- B-10 "stated in the pane in the same way as denoise_no_tap": the ``[data-denoise-empty]`` line
  carries the message and is **visible while the cells are drawn** (a stated message that is hidden
  is not stated).

Regime: Google Chrome 147 headless, CDP, DPR 1, 1440x900 unless stated (the viewport row:
1280x800, 1440x900, 1920x1080), data-theme dark and light, frames pushed at 40 ms (the recorded
806da12 trajectory's ~25 Hz); n = 1, deterministic, no seed. No GPU.
"""

from __future__ import annotations

import base64
import contextlib
import json
import re
import time
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from local_astra.contracts.base import Contract
from local_astra.contracts.codec import encode
from local_astra.contracts.protocol import (
    AstraCanvasStep,
    AstraCanvasSynthetic,
    AstraError,
    AstraTurnState,
)
from local_astra.contracts.turn import TurnState
from local_astra.hud_static import HudStaticServer
from tests.fakes.hud_server import HudStub
from tests.integration.test_hud_chrome import (
    DENOISE_EMPTY_T0128C,
    FIND_THEME_CONTROL_JS,
    choose_theme_js,
    parse_css_color,
)
from tests.integration.test_hud_headless import CHROME, HUD, requires_chrome
from tests.integration.test_hud_mockup_parity import _pixel, decode_png, delta_e, hex_rgb, hexc
from tests.integration.test_hud_redesign import ALL_PANES, _panels_line, _Recorder

pytestmark = pytest.mark.integration

#: T-0128c B3's colours, kept by T-0159 ("keep the current colours"): state -> (token, alpha).
STATE_PAINT: dict[str, tuple[str, float]] = {
    "stable": ("accent", 1.0),
    "committed": ("accent", 1.0),
    "kept": ("accent", 0.45),
    "noise": ("accent", 0.12),
    "revised": ("accent-2", 1.0),
    "empty": ("accent", 0.06),
}
COLOUR_TOL_DE = 2.0
VIEW = (1440, 900)
#: T-0159 B-9: "the rail at 1280/1440/1920 px".
VIEWPORTS = ((1280, 800), (1440, 900), (1920, 1080))
#: T-0159 B-6: "8 in the 340 px rail" (the rail at 1440x900).
RAIL_COLUMNS = 8
RID = "chatcmpl-t0128c0b0000000-0a1b2c3d"
CL = 256
MAX_STEPS = 48
FRAME_PERIOD_S = 0.040
BURST = 50
NO_TAP_MESSAGE = "the plugin tap sent no frame this session (is the plugin loaded?)"
NO_PIECES_MESSAGE = "denoising text off: tokenizer.json is not the pinned file"
SYNTHETIC_REASON = "plugin not loaded"


# --- hand-built frames ------------------------------------------------------------------------


def stable_mask(positions: set[int], n: int) -> str:
    """The wire's ``stable``: one bit per position, MSB first, zero-padded to whole bytes (the
    contract's docstring). Bit p lives in byte p // 8 at weight 2 ** (7 - p % 8)."""
    out = bytearray((n + 7) // 8)
    for p in positions:
        out[p // 8] |= 1 << (7 - p % 8)
    return out.hex()


def test_the_stable_mask_helper_is_msb_first():
    """Anti-vacuity for the frame builder (hand: position 0 alone is 0x80, 7 alone is 0x01, 8 is
    the second byte's MSB; 10 positions need two bytes)."""
    assert stable_mask({0}, 8) == "80"
    assert stable_mask({7}, 8) == "01"
    assert stable_mask({8}, 10) == "0080"
    assert stable_mask(set(range(8)), 16) == "ff00"
    assert (
        AstraCanvasStep(
            request_id="r",
            canvas_idx=0,
            step=0,
            max_steps=1,
            canvas_length=10,
            canvas=tuple(range(10)),
            argmax=tuple(range(10)),
            stable=stable_mask({9}, 10),
        ).stable
        == "0040"
    )


def denoise(
    step: int,
    *,
    kept: range,
    stable: set[int],
    argmax_override: dict[int, int] | None = None,
    canvas_idx: int = 0,
    max_steps: int = MAX_STEPS,
    cl: int = CL,
    rid: str = RID,
) -> AstraCanvasStep:
    """A denoise frame: argmax[p] = 100 + p (or the override), canvas[p] = argmax[p] for p in
    ``kept`` and for every stable p, else 9000 + p (never equal to an argmax here)."""
    argmax = [100 + p for p in range(cl)]
    for p, v in (argmax_override or {}).items():
        argmax[p] = v
    canvas = [argmax[p] if (p in kept or p in stable) else 9000 + p for p in range(cl)]
    return AstraCanvasStep(
        request_id=rid,
        canvas_idx=canvas_idx,
        step=step,
        max_steps=max_steps,
        canvas_length=cl,
        canvas=tuple(canvas),
        argmax=tuple(argmax),
        stable=stable_mask(stable, cl),
        phase="denoise",
    )


def f1() -> AstraCanvasStep:
    return denoise(6, kept=range(0, 128), stable=set(range(0, 48)))


def f2() -> AstraCanvasStep:
    # 32-47 were stable in F1; their argmax changes now and they are no longer stable: revised.
    # Their canvas is 9000 + p (not the new argmax), so without the revised rule they read noise.
    return denoise(
        7,
        kept=range(48, 128),
        stable=set(range(0, 32)),
        argmax_override={p: 5000 + p for p in range(32, 48)},
    )


def commit(step: int = 20, n_committed: int = 100, canvas_idx: int = 0) -> AstraCanvasStep:
    return AstraCanvasStep(
        request_id=RID,
        canvas_idx=canvas_idx,
        step=step,
        max_steps=MAX_STEPS,
        canvas_length=CL,
        committed_tokens=tuple(700 + p for p in range(n_committed)),
        phase="commit",
    )


def with_pieces(frame: AstraCanvasStep, pieces: list[str]) -> str:
    """The frame on the wire with its T-0159 A1 piece row: ``argmax_pieces`` (length
    ``canvas_length``) on a denoise frame, ``committed_pieces`` (length ``len(committed_tokens)``)
    on a commit frame. Raw JSON, so the row does not depend on the Python contract having the
    fields; the page's decoder is what must accept them (group A's ``protocol.js``)."""
    wire = json.loads(encode(frame))
    if frame.phase == "commit":
        assert len(pieces) == len(frame.committed_tokens or ()), "committed_pieces length"
        wire["committed_pieces"] = pieces
    else:
        assert len(pieces) == frame.canvas_length, "argmax_pieces length"
        wire["argmax_pieces"] = pieces
    return json.dumps(wire, ensure_ascii=False)


def test_with_pieces_puts_the_row_on_the_frame_by_phase():
    """Anti-vacuity for the wire builder (hand-built, two positions)."""
    d = AstraCanvasStep(
        request_id="r",
        canvas_idx=0,
        step=1,
        max_steps=2,
        canvas_length=2,
        canvas=(1, 2),
        argmax=(1, 3),
        stable="00",
    )
    got = json.loads(with_pieces(d, [" a", "<eos>"]))
    assert got["argmax_pieces"] == [" a", "<eos>"] and "committed_pieces" not in got
    assert got["type"] == "astra.canvas.step" and got["argmax"] == [1, 3]
    c = AstraCanvasStep(
        request_id="r",
        canvas_idx=0,
        step=1,
        max_steps=2,
        canvas_length=2,
        committed_tokens=(5,),
        phase="commit",
    )
    got = json.loads(with_pieces(c, ["x"]))
    assert got["committed_pieces"] == ["x"] and "argmax_pieces" not in got


#: Hand-derived from the frame descriptions in the module docstring.
F2_LABEL = "canvas 0 · step 7/48"
F2_COUNTS = {"stable": 32, "kept": 80, "noise": 128, "revised": 16}
COMMIT_LABEL = "canvas 0 · committed at step 20/48"
#: position -> state, sampled cells of F2 and of the commit frame (hand-derived).
F2_SAMPLES = {
    5: "stable",
    31: "stable",
    35: "revised",
    47: "revised",
    48: "kept",
    74: "kept",
    127: "kept",
    128: "noise",
    212: "noise",
    255: "noise",
}
COMMIT_SAMPLES = {5: "committed", 95: "committed", 99: "committed", 100: "empty", 254: "empty"}


def test_the_hand_derived_samples_follow_from_the_frames():
    """The sample tables restated from the frames' rules, so a typo in one of them is caught here
    and not blamed on the page."""
    for p, state in F2_SAMPLES.items():
        want = "stable" if p < 32 else "revised" if p < 48 else "kept" if p < 128 else "noise"
        assert state == want, (p, state, want)
    for p, state in COMMIT_SAMPLES.items():
        assert state == ("committed" if p < 100 else "empty"), p
    assert sum(F2_COUNTS.values()) == CL and F2_COUNTS["revised"] == 48 - 32


# --- colour oracles (T-0128c B3's colours, kept by T-0159) ---------------------------------------


def expected_colour(
    state: str, tokens: dict[str, str], backdrop: tuple[float, float, float]
) -> tuple[float, float, float]:
    token, alpha = STATE_PAINT[state]
    r, g, b = hex_or_rgb(tokens[token])
    return tuple(v * alpha + bd * (1 - alpha) for v, bd in zip((r, g, b), backdrop, strict=True))  # type: ignore[return-value]


def hex_or_rgb(value: str) -> tuple[float, float, float]:
    v = value.strip()
    if v.startswith("#"):
        return hex_rgb(v)
    r, g, b, _a = parse_css_color(v)
    return (r, g, b)


def classify(
    got: tuple[float, float, float],
    want: str,
    tokens: dict[str, str],
    backdrop: tuple[float, float, float],
) -> str | None:
    """None when ``got`` is within dE00 2 of ``want``'s colour and nearer to it than to any other
    state's colour; else the problem."""
    target = expected_colour(want, tokens, backdrop)
    de = delta_e(got, target)
    others = {
        s: delta_e(got, expected_colour(s, tokens, backdrop))
        for s in STATE_PAINT
        if STATE_PAINT[s] != STATE_PAINT[want]
    }
    nearest = min(others, key=lambda s: others[s]) if others else None
    if de > COLOUR_TOL_DE:
        return (
            f"{hexc(got)} vs {want} {hexc(target)} (--{STATE_PAINT[want][0]} at "
            f"{STATE_PAINT[want][1]} over {hexc(backdrop)}): dE00 {de:.2f} > {COLOUR_TOL_DE}"
            + (f"; nearest other state {nearest} dE00 {others[nearest]:.2f}" if nearest else "")
        )
    if nearest is not None and others[nearest] < de:
        return (
            f"{hexc(got)} is nearer {nearest} (dE00 {others[nearest]:.2f}) than {want} ({de:.2f})"
        )
    return None


# --- page probes ----------------------------------------------------------------------------------

#: Shared JS: the sections, a cell's state, the newest section.
CELLS_PRELUDE = """
  const STATES = ['stable', 'kept', 'noise', 'revised', 'committed', 'empty'];
  const root = document.querySelector('[data-denoise-root]');
  const sections = root ? [...root.children].filter((e) => e.localName === 'details') : [];
  const newest = sections[sections.length - 1] || null;
  const cellsOf = (s) => (s ? [...s.querySelectorAll('[data-denoise-cell]')] : []);
  const stateOf = (c) => {
    const d = c.getAttribute('data-state');
    if (d !== null) return d;
    const hits = [...c.classList].filter((x) => STATES.includes(x));
    return hits.length === 1 ? hits[0] : null;
  };
"""

STATE_JS = (
    "(() => {"
    + CELLS_PRELUDE
    + """
  const px = (v) => parseFloat(v) || 0;
  const content = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    const l = px(s.borderLeftWidth) + px(s.paddingLeft), t = px(s.borderTopWidth) + px(s.paddingTop);
    return {x: r.x + l, y: r.y + t,
            w: r.width - l - px(s.borderRightWidth) - px(s.paddingRight),
            h: r.height - t - px(s.borderBottomWidth) - px(s.paddingBottom),
            right: r.right, bottom: r.bottom, border_w: r.width, border_h: r.height};
  };
  const visible = (el) => {
    if (!el) return false;
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    return s.display !== 'none' && s.visibility !== 'hidden' && parseFloat(s.opacity) > 0 &&
           r.width > 0 && r.height > 0 && !el.closest('[hidden]');
  };
  const pane = document.querySelector('main > .pane-thinking');
  const label = document.querySelector('[data-denoise-label]');
  const empty = document.querySelector('[data-denoise-empty]');
  const rs = getComputedStyle(document.documentElement);
  const ign = (document.getElementById('ignored-count') || {}).textContent || '';
  const arias = newest ? [newest, ...newest.querySelectorAll('[aria-label]')]
      .map((e) => e.getAttribute('aria-label')).filter((a) => a) : [];
  return {
    root: !!root,
    root_children: root ? [...root.children].map((e) => e.tagName.toLowerCase()) : null,
    canvases: root ? root.querySelectorAll('canvas').length : 0,
    sections: sections.map((s) => ({
      open: s.open,
      first: s.firstElementChild ? s.firstElementChild.localName : null,
      summary: ((s.querySelector('summary') || {}).textContent || '').replace(/\\s+/g, ' ').trim(),
      cells: cellsOf(s).length,
    })),
    newest_states: cellsOf(newest).map(stateOf),
    newest_texts: cellsOf(newest).map((c) => c.textContent),
    arias,
    root_box: content(root),
    pane_bg: pane ? getComputedStyle(pane).backgroundColor : null,
    pane_bg_image: pane ? getComputedStyle(pane).backgroundImage : null,
    label: label ? (label.textContent || '').trim() : null,
    label_visible: visible(label),
    empty: empty ? (empty.textContent || '').replace(/\\s+/g, ' ').trim() : null,
    empty_visible: visible(empty),
    frames: root ? root.getAttribute('data-denoise-frames') : null,
    draws: root ? root.getAttribute('data-denoise-draws') : null,
    digest: root ? root.getAttribute('data-denoise-digest') : null,
    ignored: ign,
    tokens: {accent: rs.getPropertyValue('--accent').trim(),
             'accent-2': rs.getPropertyValue('--accent-2').trim(),
             mono: rs.getPropertyValue('--mono').trim()},
    theme: document.documentElement.getAttribute('data-theme'),
    dpr: window.devicePixelRatio,
    scroll_w: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
    inner_w: window.innerWidth,
    scroll: [window.scrollX, window.scrollY],
  };
})()"""
)

#: Scrolls cell ``__P__`` of the newest section into view and returns its viewport rect, the point
#: whose pixel is its fill (centre x, 2 px under its top border) and its state.
CELL_POINT_JS = (
    "(() => {"
    + CELLS_PRELUDE
    + """
  const c = cellsOf(newest)[__P__];
  if (!c) return {ok: false, why: 'no cell __P__ in the newest section (' + cellsOf(newest).length + ' cells)'};
  c.scrollIntoView({block: 'nearest', inline: 'nearest', behavior: 'instant'});
  const r = c.getBoundingClientRect(), s = getComputedStyle(c);
  const top = r.y + (parseFloat(s.borderTopWidth) || 0) + 2;
  return {ok: true, x: Math.floor(r.x + r.width / 2), y: Math.floor(top), w: r.width, h: r.height,
          state: stateOf(c), text: c.textContent};
})()"""
)


def cell_point_js(p: int) -> str:
    return CELL_POINT_JS.replace("__P__", str(int(p)))


#: Records, on the page clock, every change of the draw counter and every distinct label text.
RECORD_JS = """
(() => {
  const root = document.querySelector('[data-denoise-root]');
  const label = document.querySelector('[data-denoise-label]');
  const rec = window.__t0128c = {draws: [], labels: [], frames: [], digests: []};
  if (root) new MutationObserver((ms) => {
    for (const m of ms) {
      const t = performance.now();
      if (m.attributeName === 'data-denoise-draws') rec.draws.push([t, root.getAttribute('data-denoise-draws')]);
      if (m.attributeName === 'data-denoise-frames') rec.frames.push([t, root.getAttribute('data-denoise-frames')]);
      // O-B3 F1: every digest write, as the value before it (batched records lose none); the
      // value after write i is the old value of write i + 1, or the final attribute.
      if (m.attributeName === 'data-denoise-digest') rec.digests.push([t, m.oldValue]);
    }
  }).observe(root, {attributes: true, attributeOldValue: true,
                    attributeFilter: ['data-denoise-draws', 'data-denoise-frames', 'data-denoise-digest']});
  if (label) new MutationObserver(() => {
    const t = (label.textContent || '').trim(), L = rec.labels;
    if (L[L.length - 1] !== t) L.push(t);
  }).observe(label, {childList: true, characterData: true, subtree: true});
  return {root: !!root, label: !!label};
})()
"""

THEME_SWITCH_JS = (
    "(async () => {"
    + FIND_THEME_CONTROL_JS
    + """
  const ctl = themeControl();
  if (!ctl) return {ok: false, why: 'no Settings theme control'};
  ctl.value = __THEME__;
  ctl.dispatchEvent(new Event('input', {bubbles: true}));
  ctl.dispatchEvent(new Event('change', {bubbles: true}));
  await frame();
  const rs = getComputedStyle(document.documentElement);
  const pane = document.querySelector('main > .pane-thinking');
  return {ok: true, theme: document.documentElement.getAttribute('data-theme'),
          tokens: {accent: rs.getPropertyValue('--accent').trim(), 'accent-2': rs.getPropertyValue('--accent-2').trim()},
          pane_bg: pane ? getComputedStyle(pane).backgroundColor : null};
})()"""
)


def theme_switch_js(theme: str) -> str:
    return THEME_SWITCH_JS.replace("__THEME__", json.dumps(theme))


#: B-6/B-9 geometry of the newest section: the cells' rects, the root's, every scroller between a
#: cell and the root, the page's horizontal extent.
GEOMETRY_JS = (
    "(() => {"
    + CELLS_PRELUDE
    + """
  if (!root) return {ok: false, why: 'no [data-denoise-root]'};
  const rr = root.getBoundingClientRect();
  const cells = cellsOf(newest).map((c) => {
    const r = c.getBoundingClientRect();
    return {x: r.x, y: r.y, w: r.width, h: r.height, right: r.right};
  });
  const scrollers = [];
  const first = cellsOf(newest)[0];
  for (let n = first ? first.parentElement : null; n; n = n.parentElement) {
    const s = getComputedStyle(n);
    if (/(auto|scroll)/.test(s.overflowY) && n.scrollHeight > n.clientHeight + 1)
      scrollers.push({tag: n.localName, cls: n.className, sh: n.scrollHeight, ch: n.clientHeight});
    if (n === root) break;
  }
  return {ok: true, root: {x: rr.x, right: rr.right, w: rr.width, sw: root.scrollWidth, cw: root.clientWidth},
          cells, scrollers,
          scroll_w: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
          inner_w: window.innerWidth};
})()"""
)

#: The text-bearing element of cell ``__P__`` of the newest section and the styles B-6/B-8 name.
TEXT_STYLE_JS = (
    "(() => {"
    + CELLS_PRELUDE
    + """
  const c = cellsOf(newest)[__P__];
  if (!c) return {ok: false, why: 'no cell __P__'};
  let t = c;
  for (;;) {
    const kids = [...t.children].filter((k) => (k.textContent || '') === (t.textContent || '') && k.textContent);
    if (kids.length !== 1) break;
    t = kids[0];
  }
  const ts = getComputedStyle(t), cs = getComputedStyle(c), r = c.getBoundingClientRect();
  const special = [c, ...c.querySelectorAll('*')].some((e) => e.hasAttribute('data-special') || e.classList.contains('special'));
  const trans = [c, ...c.querySelectorAll('*')].map((e) => {
    const s = getComputedStyle(e);
    return {props: s.transitionProperty, durs: s.transitionDuration, anim: s.animationName};
  });
  return {ok: true, text: c.textContent, title: c.getAttribute('title'), special,
          font_family: ts.fontFamily, font_size: parseFloat(ts.fontSize),
          text_overflow: ts.textOverflow, overflow_x: ts.overflowX, white_space: ts.whiteSpace,
          cell_text_overflow: cs.textOverflow, cell_overflow_x: cs.overflowX,
          t_scroll_w: t.scrollWidth, t_client_w: t.clientWidth, w: r.width, right: r.right, trans};
})()"""
)


def text_style_js(p: int) -> str:
    return TEXT_STYLE_JS.replace("__P__", str(int(p)))


# --- the session ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    """The module's one headless Chrome."""
    if CHROME is None:  # pragma: no cover - requires_chrome is the real guard
        yield "no chrome"
        return
    stack = contextlib.ExitStack()
    try:
        chrome = stack.enter_context(_Recorder(CHROME))
    except Exception as exc:
        stack.close()
        yield f"headless Chrome did not start: {type(exc).__name__}: {exc}"
        return
    with stack:
        yield chrome


@dataclass
class Page:
    chrome: _Recorder
    stub: HudStub

    def state(self) -> dict[str, Any]:
        return dict(self.chrome.evaluate(STATE_JS) or {})

    def push(self, *messages: Contract | str, gap_s: float = 0.0) -> None:
        for m in messages:
            self.stub.push(m)
            if gap_s:
                time.sleep(gap_s)

    def wait(self, expression: str, timeout_s: float = 5.0) -> bool:
        return bool(self.chrome.wait_until(expression, timeout_s))

    def wait_section(self, timeout_s: float = 5.0) -> bool:
        return self.wait(
            "!!document.querySelector('[data-denoise-root] > details [data-denoise-cell]')",
            timeout_s,
        )

    def wait_label(self, text: str, timeout_s: float = 5.0) -> bool:
        return self.wait(
            "((document.querySelector('[data-denoise-label]') || {}).textContent || '').trim() === "
            + json.dumps(text),
            timeout_s,
        )

    def settle(self, seconds: float = 0.7) -> None:
        """Wait out a throttled redraw (B8: ≤ 2/s) and any transition, then two frames."""
        time.sleep(seconds)
        self.chrome.evaluate(
            "new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))"
        )

    def missing_panel(self) -> str:
        lines = [x for x in self.chrome.console if "denoise" in x]
        return f" (console lines naming denoise: {lines})" if lines else ""

    def cell_fill(self, p: int) -> tuple[dict[str, Any], tuple[float, float, float] | None]:
        pt = dict(self.chrome.evaluate(cell_point_js(p)) or {})
        if not pt.get("ok"):
            return pt, None
        return pt, _pixel(self.chrome, int(pt["x"]), int(pt["y"]))


def _metrics(chrome: _Recorder, view: tuple[int, int], dpr: float) -> None:
    chrome._cmd(
        "Emulation.setDeviceMetricsOverride",
        width=view[0],
        height=view[1],
        deviceScaleFactor=dpr,
        mobile=False,
    )
    chrome.evaluate("document.documentElement.getBoundingClientRect().height")
    time.sleep(0.2)


@contextlib.contextmanager
def hud_page(
    browser: Any,
    *,
    view: tuple[int, int] = VIEW,
    dpr: float = 1,
    reduce: bool = False,
    theme: str = "dark",
) -> Iterator[Page]:
    if isinstance(browser, str):
        pytest.fail(browser)
    chrome: _Recorder = browser
    chrome.reset()
    chrome._cmd("Runtime.enable")
    chrome._cmd("Network.enable")
    chrome.media(
        [
            {"name": "prefers-reduced-motion", "value": "reduce" if reduce else "no-preference"},
            {"name": "prefers-color-scheme", "value": theme},
        ]
    )
    _metrics(chrome, view, dpr)
    stub = HudStub([]).start()
    server = HudStaticServer(HUD, 0)
    try:
        chrome.navigate(
            f"http://127.0.0.1:{server.port}/?ws=" + urllib.parse.quote(stub.url, safe="")
        )
        assert stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20), (
            "the page never said astra.hello to the stub"
        )
        assert stub.wait_for_client(10)
        chrome.wait_panes(ALL_PANES)
        deadline = time.monotonic() + 15
        while _panels_line(chrome.console[chrome._mark :]) is None and time.monotonic() < deadline:
            chrome.drain(0.1)
        chosen = chrome.evaluate(choose_theme_js(theme)) or {}
        assert chosen.get("ok") and chosen.get("theme") == theme, f"theme {theme}: {chosen}"
        chrome.evaluate(RECORD_JS)
        yield Page(chrome, stub)
    finally:
        server.close()
        stub.stop()


def _ignored_malformed(text: str) -> int | None:
    m = re.search(r"(\d+)m", text or "")
    return int(m.group(1)) if m else None


def _draws(state: dict[str, Any]) -> int:
    return int(state.get("draws") or 0)


def _backdrop(s: dict[str, Any]) -> tuple[float, float, float] | None:
    bg = parse_css_color(str(s.get("pane_bg")))
    if bg[3] < 1 or str(s.get("pane_bg_image")) not in ("none", ""):
        return None
    return (bg[0], bg[1], bg[2])


# --- B5: before the first frame, and the two refusals (T-0128c, unchanged) ------------------------


@requires_chrome
def test_before_any_frame_the_root_is_empty_and_the_line_reads_the_b5_copy(browser):
    """B5 (and T-0128 criterion 5 kept): no element child in ``[data-denoise-root]``, no canvas,
    no draw, an empty label, and the empty line reads B5's copy, visibly."""
    with hud_page(browser) as page:
        page.settle(0.5)
        s = page.state()
    assert s["root"] and s["root_children"] == [], (
        f"root children before a frame: {s['root_children']}"
    )
    assert s["canvases"] == 0
    assert _draws(s) == 0, f"data-denoise-draws {s['draws']!r} before any frame"
    assert s["label"] == "", (
        f"[data-denoise-label] is {s['label']!r} before any frame (B4: from a frame only)"
    )
    assert s["empty"] == DENOISE_EMPTY_T0128C and s["empty_visible"], (
        f"B5: the empty line reads {s['empty']!r} (visible {s['empty_visible']}), want "
        f"{DENOISE_EMPTY_T0128C!r}"
    )


@requires_chrome
def test_a_synthetic_canvas_is_refused_by_name_and_nothing_is_drawn(browser):
    """B5 (adversarial, the fake-trajectory failure mode): ``astra.canvas.synthetic`` draws
    nothing, creates no canvas, and the line reads "synthetic canvas refused: {reason}"."""
    with hud_page(browser) as page:
        page.push(AstraCanvasSynthetic(request_id=RID, reason=SYNTHETIC_REASON))
        want = f"synthetic canvas refused: {SYNTHETIC_REASON}"
        page.wait(
            "((document.querySelector('[data-denoise-empty]') || {}).textContent || '')"
            f".includes({json.dumps(want)})",
            5,
        )
        page.settle(0.5)
        s = page.state()
    assert s["empty"] == want and s["empty_visible"], (
        f"the line reads {s['empty']!r}, want {want!r}"
    )
    assert s["canvases"] == 0 and s["root_children"] == [], s["root_children"]
    assert _draws(s) == 0, f"draws {s['draws']!r} after a synthetic canvas"
    assert s["label"] == "", f"label {s['label']!r} after a synthetic canvas"


@requires_chrome
def test_denoise_no_tap_shows_its_message_and_another_error_code_does_not(browser):
    """B5: on ``astra.error`` ``denoise_no_tap`` the empty line shows that message. Both sides of
    the guard: an ``llm_unavailable`` error pushed first leaves B5's copy in place."""
    with hud_page(browser) as page:
        page.push(AstraError(code="llm_unavailable", message="vllm is not answering"))
        page.settle(0.4)
        first = page.state()
        page.push(AstraError(code="denoise_no_tap", message=NO_TAP_MESSAGE))
        page.wait(
            "((document.querySelector('[data-denoise-empty]') || {}).textContent || '')"
            f".includes({json.dumps(NO_TAP_MESSAGE)})",
            5,
        )
        s = page.state()
    assert first["empty"] == DENOISE_EMPTY_T0128C, (
        f"another error code rewrote the Thinking line: {first['empty']!r}"
    )
    assert s["empty"] is not None and NO_TAP_MESSAGE in s["empty"] and s["empty_visible"], (
        f"B5: after denoise_no_tap the line reads {s['empty']!r} (visible {s['empty_visible']})"
    )
    assert s["canvases"] == 0 and _draws(s) == 0


# --- T-0159 B-10: denoise_no_pieces -----------------------------------------------------------------


@requires_chrome
def test_denoise_no_pieces_is_stated_visibly_in_the_pane_while_the_cells_draw(browser):
    """T-0159 B-10 (in the wire order of A5: frames without pieces, the error at the first of them,
    more frames): the message is on ``[data-denoise-empty]`` and visible while the grid is drawn;
    the guard's other side, an ``llm_unavailable`` error, leaves the line alone."""
    with hud_page(browser) as page:
        page.push(f1())
        page.wait_section(5)
        page.push(AstraError(code="llm_unavailable", message="vllm is not answering"))
        page.settle(0.4)
        first = page.state()
        page.push(AstraError(code="denoise_no_pieces", message=NO_PIECES_MESSAGE))
        page.push(f2(), gap_s=0.05)
        page.wait_label(F2_LABEL, 5)
        page.settle(0.6)
        s = page.state()
    assert first["empty"] == DENOISE_EMPTY_T0128C, (
        f"another error code rewrote the Thinking line: {first['empty']!r}"
    )
    assert len(s["sections"]) == 1 and s["sections"][0]["cells"] == CL, (
        f"premise: the frames without pieces still draw their cells: {s['sections']}"
        + page.missing_panel()
    )
    assert s["empty"] is not None and NO_PIECES_MESSAGE in s["empty"], (
        f"B-10: the line reads {s['empty']!r}"
    )
    assert s["empty_visible"], "B-10: the denoise_no_pieces message is in the pane but not visible"
    assert all(t == "" for t in s["newest_texts"]), "B-2: no pieces, no text"


@requires_chrome
def test_a_drawn_frame_hides_the_no_tap_notice_and_pieces_hide_the_no_pieces_notice(browser):
    """T-0159 B review T2. After ``denoise_no_tap``, a drawn frame leaves the notice line hidden or
    cleared, as at BASE. After ``denoise_no_pieces``, frames without pieces keep it visible (B-10)
    and a frame with pieces hides or clears it (the text is back)."""
    with hud_page(browser) as page:
        page.push(AstraError(code="denoise_no_tap", message=NO_TAP_MESSAGE))
        page.settle(0.3)
        tap_before = page.state()
        page.push(f1())
        page.wait_section(5)
        page.settle(0.4)
        tap_after = page.state()
        page.push(AstraError(code="denoise_no_pieces", message=NO_PIECES_MESSAGE))
        page.push(f2(), gap_s=0.05)
        page.wait_label(F2_LABEL, 5)
        page.settle(0.4)
        bare = page.state()
        page.push(with_pieces(denoise(8, kept=range(0), stable=set()), ["back"] * CL))
        page.wait_label("canvas 0 · step 8/48", 5)
        page.settle(0.4)
        pieced = page.state()
    assert NO_TAP_MESSAGE in str(tap_before["empty"]) and tap_before["empty_visible"], (
        f"premise: the no_tap notice shows: {tap_before['empty']!r}"
    )
    assert not (NO_TAP_MESSAGE in str(tap_after["empty"]) and tap_after["empty_visible"]), (
        "T2: a drawn frame leaves the denoise_no_tap notice visible"
    )
    assert NO_PIECES_MESSAGE in str(bare["empty"]) and bare["empty_visible"], (
        f"B-10: frames without pieces keep the notice: {bare['empty']!r}"
    )
    assert pieced["newest_texts"][:1] == ["back"], (
        f"premise: pieces drawn {pieced['newest_texts'][:2]}"
    )
    assert not (NO_PIECES_MESSAGE in str(pieced["empty"]) and pieced["empty_visible"]), (
        "T2: a frame with pieces leaves the denoise_no_pieces notice visible"
    )


# --- B5/B7/B9 + T-0159 B-6/B-7: the first frame, and between turns ---------------------------------


@requires_chrome
def test_the_section_is_created_at_the_first_frame_and_frames_with_pieces_decode(browser):
    """T-0128c B5 in the T-0159 form: one ``<details>`` section appears at the first frame (with a
    ``<summary>`` first and 256 cells), the empty line leaves; frames carrying the A1 piece rows count
    as zero malformed in ``#ignored-count`` (needs group A's ``protocol.js``); B9: frames 2, an 8-hex
    digest; B4: the label; B-6: the cells show F2's pieces; B7: after an IDLE turn state the section,
    its tokens and the label stay."""
    pieces = [f"w{p}" for p in range(CL)]
    with hud_page(browser) as page:
        before = page.state()
        page.push(with_pieces(f1(), [f"v{p}" for p in range(CL)]))
        created = page.wait_section(5)
        after_one = page.state()
        page.push(with_pieces(f2(), pieces), gap_s=0.05)
        page.wait_label(F2_LABEL, 5)
        page.settle(0.5)
        two = page.state()
        page.push(AstraTurnState(state=TurnState.IDLE, turn_id="t1"))
        page.settle(0.8)
        idle = page.state()
    assert before["root_children"] == [], "premise: nothing before the first frame"
    assert _ignored_malformed(two["ignored"]) == 0, (
        f"#ignored-count {two['ignored']!r}: frames with argmax_pieces decode malformed (T-0159 A6)"
    )
    assert created and after_one["root_children"] == ["details"], (
        f"B-7: one <details> at the first frame; root children {after_one['root_children']}"
        + page.missing_panel()
    )
    sec = after_one["sections"][0]
    assert sec["first"] == "summary" and sec["cells"] == CL, sec
    assert after_one["canvases"] == 0, "B-6: no bitmap any more"
    assert not after_one["empty_visible"], "the empty line still shows beside a drawn grid"
    assert two["frames"] == "2", f"data-denoise-frames {two['frames']!r} after two frames"
    assert _draws(two) >= 1
    assert re.fullmatch(r"[0-9a-f]{8}", str(two["digest"] or "")), two["digest"]
    assert two["label"] == F2_LABEL and two["label_visible"], two["label"]
    assert F2_LABEL in two["sections"][0]["summary"], two["sections"]
    assert two["newest_texts"] == pieces, "B-6: cell i shows argmax_pieces[i]"
    assert [two["newest_states"][p] for p in F2_SAMPLES] == list(F2_SAMPLES.values()), (
        "B-5: the cell states are F2's"
    )
    assert idle["root_children"] == ["details"] and idle["label"] == F2_LABEL, (
        f"B7: after IDLE the section ({idle['root_children']}) and label ({idle['label']!r}) stay"
    )
    assert idle["digest"] == two["digest"] and idle["newest_texts"] == pieces


# --- T-0159 B-9: each state's fill in both themes ---------------------------------------------------


def _fill_problems(page: Page, s: dict[str, Any], samples: dict[int, str]) -> list[str]:
    backdrop = _backdrop(s)
    if backdrop is None:
        return [f"premise: the Thinking pane's background is not one opaque colour: {s['pane_bg']}"]
    problems = []
    for p, state in samples.items():
        pt, got = page.cell_fill(p)
        if got is None:
            problems.append(f"cell {p}: {pt.get('why')}")
            continue
        if pt.get("state") != state:
            problems.append(f"cell {p}: state {pt.get('state')!r}, want {state} (B-5)")
        why = classify(got, state, s["tokens"], backdrop)
        if why:
            problems.append(f"cell {p} {state}: {why}")
    return problems


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_each_state_is_filled_in_its_current_colour(browser, theme):
    """T-0159 B-9 (1): stable/committed ``--accent`` 1.0, kept 0.45, noise 0.12, revised
    ``--accent-2`` 1.0, empty 0.06 (T-0128c B3's colours, kept), read from the screen inside each
    sampled cell and composited over the pane; frames carry one-letter pieces."""
    with hud_page(browser, theme=theme) as page:
        page.push(with_pieces(f1(), ["a"] * CL), with_pieces(f2(), ["a"] * CL), gap_s=0.05)
        page.wait_label(F2_LABEL, 5)
        page.settle(0.8)
        s = page.state()
        problems = _fill_problems(page, s, F2_SAMPLES)
        page.push(with_pieces(commit(), ["a"] * 100))
        page.wait_label(COMMIT_LABEL, 5)
        page.settle(0.8)
        sc = page.state()
        problems += _fill_problems(page, sc, COMMIT_SAMPLES)
    assert s["theme"] == theme and len(s["sections"]) == 1, (
        s["theme"],
        s["sections"],
        page.missing_panel(),
    )
    assert sc["label"] == COMMIT_LABEL, sc["label"]
    assert problems == [], f"B-9 (data-theme={theme}):\n  " + "\n  ".join(problems)


@requires_chrome
def test_a_theme_change_recolours_the_cells_within_one_frame(browser):
    """T-0128c B3 / T-0159 B-9: drawn in dark, then light chosen in Settings: one frame later the
    stable and revised cells hold the *light* tokens (screen pixels). (The T-0128c row also wanted
    the draw counter to move; a DOM grid recolours through CSS and need not redraw.)"""
    with hud_page(browser, theme="dark") as page:
        page.push(with_pieces(f1(), ["a"] * CL), with_pieces(f2(), ["a"] * CL), gap_s=0.05)
        page.wait_label(F2_LABEL, 5)
        page.settle(0.5)
        s = page.state()
        out = page.chrome.evaluate(theme_switch_js("light")) or {}
        fills = [page.cell_fill(5), page.cell_fill(35)]
    assert out.get("ok"), f"{out}{page.missing_panel()}"
    assert out["theme"] == "light"
    dark_accent = hex_or_rgb(s["tokens"]["accent"])
    light_accent = hex_or_rgb(out["tokens"]["accent"])
    assert delta_e(dark_accent, light_accent) > 5, "premise: the two themes' accents differ"
    bg = parse_css_color(str(out["pane_bg"]))
    backdrop = (bg[0], bg[1], bg[2])
    problems = []
    for (pt, px), state in zip(fills, ("stable", "revised"), strict=True):
        if px is None:
            problems.append(f"{state}: {pt.get('why')}")
            continue
        why = classify(px, state, out["tokens"], backdrop)
        if why:
            problems.append(f"{state}: {why}")
    assert problems == [], (
        "B-9: one frame after the switch the cells are not in the light tokens:\n  "
        + "\n  ".join(problems)
    )


# --- T-0159 B-6/B-9: columns, fixed widths, no horizontal overflow, bounded height -------------------


def _geometry_problems(g: dict[str, Any], where: str, *, rail: bool) -> list[str]:
    if not g.get("ok"):
        return [f"{where}: {g.get('why')}"]
    cells = g["cells"]
    if len(cells) != CL:
        return [f"{where}: {len(cells)} cells in the newest section, want {CL}"]
    problems = []
    if g["scroll_w"] > g["inner_w"]:
        problems.append(f"{where}: the page scrolls sideways ({g['scroll_w']} > {g['inner_w']})")
    root = g["root"]
    if root["sw"] > root["cw"] + 1:
        problems.append(
            f"{where}: [data-denoise-root] overflows sideways ({root['sw']} > {root['cw']})"
        )
    out = [
        i for i, c in enumerate(cells) if c["right"] > root["right"] + 1 or c["x"] < root["x"] - 1
    ]
    if out:
        problems.append(f"{where}: cells {out[:8]} lie outside the root horizontally")
    widths = sorted({round(c["w"], 1) for c in cells})
    if widths[-1] - widths[0] > 0.5:
        problems.append(f"{where}: cells are not one fixed width: {widths[:6]}…")
    top = cells[0]["y"]
    columns = sum(1 for c in cells if abs(c["y"] - top) < 1)
    if columns < 1 or [round(c["x"]) for c in cells[:columns]] != sorted(
        round(c["x"]) for c in cells[:columns]
    ):
        problems.append(f"{where}: the first row is not in position order")
    if rail and columns != RAIL_COLUMNS:
        problems.append(
            f"{where}: {columns} columns in the rail (root {root['w']:.0f} px), want {RAIL_COLUMNS}"
        )
    if not g["scrollers"]:
        problems.append(
            f"{where}: nothing between the cells and [data-denoise-root] scrolls vertically "
            "(B-6: the grid scrolls inside the pane at a bounded height)"
        )
    return problems


@requires_chrome
def test_the_grid_fits_the_rail_without_horizontal_overflow_at_three_viewports(browser):
    """T-0159 B-6/B-9 (3): at 1280x800, 1440x900 and 1920x1080, a CL 256 canvas with two 60-letter
    pieces: equal-width cells in position order, none outside the root, the root and the page never
    scroll sideways, the grid scrolls vertically inside the root; at 1440 the rail holds 8 columns."""
    pieces = ["a"] * CL
    pieces[3] = "x" * 60
    pieces[200] = " " + "y" * 59
    problems: list[str] = []
    columns: dict[str, int] = {}
    with hud_page(browser, view=VIEWPORTS[0]) as page:
        page.push(with_pieces(f2(), pieces))
        page.wait_section(5)
        page.settle(0.5)
        for view in VIEWPORTS:
            _metrics(page.chrome, view, 1)
            page.settle(0.6)
            g = dict(page.chrome.evaluate(GEOMETRY_JS) or {})
            where = f"{view[0]}x{view[1]}"
            problems += _geometry_problems(g, where, rail=view == VIEW)
            if g.get("ok") and g["cells"]:
                top = g["cells"][0]["y"]
                columns[where] = sum(1 for c in g["cells"] if abs(c["y"] - top) < 1)
    assert problems == [], (
        f"B-6/B-9 geometry (columns seen {columns}):\n  "
        + "\n  ".join(problems)
        + page.missing_panel()
    )


@requires_chrome
def test_a_long_piece_is_cut_inside_its_fixed_width_cell_and_whole_in_the_title(browser):
    """T-0159 B-6: a 60-letter piece keeps its cell the width of its neighbours, is cut with an
    ellipsis (CSS ``text-overflow: ellipsis`` on a clipped element, or a text ending in …), and the
    whole piece, its leading space made visible, is the cell's title."""
    long = " " + "z" * 59
    pieces = ["a"] * CL
    pieces[9] = long
    with hud_page(browser) as page:
        page.push(with_pieces(f2(), pieces))
        page.wait_section(5)
        page.settle(0.5)
        cell = dict(page.chrome.evaluate(text_style_js(9)) or {})
        neighbour = dict(page.chrome.evaluate(text_style_js(10)) or {})
    assert cell.get("ok") and neighbour.get("ok"), (cell, neighbour, page.missing_panel())
    assert abs(cell["w"] - neighbour["w"]) <= 0.5, (
        f"B-6: the long piece's cell is {cell['w']:.1f} px, its neighbour {neighbour['w']:.1f} px"
    )
    css_cut = (
        "ellipsis" in (cell["text_overflow"], cell["cell_text_overflow"])
        and any(v in ("hidden", "clip") for v in (cell["overflow_x"], cell["cell_overflow_x"]))
        and cell["t_scroll_w"] > cell["t_client_w"]
    )
    text_cut = cell["text"].endswith("…") and ("z" * 59).startswith(cell["text"][:-1])
    assert css_cut or text_cut, f"B-6: the long piece is not cut with …: {cell}"
    assert cell["title"] == "␣" + "z" * 59, f"B-6: the title is {cell['title']!r}"


@requires_chrome
def test_special_pieces_are_in_the_monospace_face_at_a_smaller_size(browser):
    """T-0159 B-6: ``<|channel>`` and ``<0xE2>`` are marked special and drawn in the monospace
    utility face (``--mono``) at a font size below an ordinary piece's; an ordinary piece is not."""
    pieces = ["word"] * CL
    pieces[0] = "<|channel>"
    pieces[1] = "<0xE2>"
    with hud_page(browser) as page:
        page.push(with_pieces(f2(), pieces))
        page.wait_section(5)
        page.settle(0.5)
        s = page.state()
        spec = [dict(page.chrome.evaluate(text_style_js(p)) or {}) for p in (0, 1)]
        plain = dict(page.chrome.evaluate(text_style_js(2)) or {})
    assert all(x.get("ok") for x in [*spec, plain]), (spec, plain, page.missing_panel())
    norm = lambda f: re.sub(r"[\s\"']", "", f)  # noqa: E731
    mono = norm(s["tokens"]["mono"])
    assert mono, "premise: --mono is defined"
    for x in spec:
        assert x["special"], f"B-6: {x['text']!r} is not marked special"
        assert norm(x["font_family"]) == mono, (
            f"B-6: {x['text']!r} in {x['font_family']!r}, want --mono {s['tokens']['mono']!r}"
        )
        assert x["font_size"] < plain["font_size"], (
            f"B-6: {x['text']!r} at {x['font_size']} px, an ordinary piece at {plain['font_size']}"
        )
    assert not plain["special"] and norm(plain["font_family"]) != mono, (
        f"B-6: an ordinary piece is marked or set in the utility face: {plain}"
    )


@requires_chrome
def test_text_changes_in_place_without_transitions(browser):
    """T-0159 B-8: no cell (nor anything inside one) transitions its text: no non-zero transition on
    ``all``, ``color``, ``opacity`` or ``font-*``, and no animation (checked without reduced motion,
    where a transition would show)."""
    with hud_page(browser, reduce=False) as page:
        page.push(with_pieces(f1(), ["a"] * CL), with_pieces(f2(), ["b"] * CL), gap_s=0.05)
        page.wait_label(F2_LABEL, 5)
        page.settle(0.4)
        cells = [dict(page.chrome.evaluate(text_style_js(p)) or {}) for p in (5, 35, 74, 200)]
    assert all(c.get("ok") for c in cells), (cells, page.missing_panel())
    problems = []
    for p, c in zip((5, 35, 74, 200), cells, strict=True):
        for t in c["trans"]:
            props = [x.strip() for x in t["props"].split(",")]
            durs = [float(x.strip().rstrip("s") or 0) for x in t["durs"].split(",")]
            for i, prop in enumerate(props):
                d = durs[i % len(durs)]
                if d > 0 and (prop in ("all", "color", "opacity") or prop.startswith("font")):
                    problems.append(f"cell {p}: transition {prop} {d}s")
            if t["anim"] not in ("none", ""):
                problems.append(f"cell {p}: animation {t['anim']}")
    assert problems == [], "B-8: text changes in place:\n  " + "\n  ".join(problems)


# --- T-0159 B-3/B-4: text is text, and stays in the denoising view ---------------------------------------

ISOLATION_JS = """
(() => {
  const root = document.querySelector('[data-denoise-root]');
  if (!root) return {ok: false, why: 'no root'};
  const clone = document.documentElement.cloneNode(true);
  const r = clone.querySelector('[data-denoise-root]');
  if (r) r.remove();
  return {ok: true, outside: clone.outerHTML.includes(__M__),
          inside: root.textContent.includes(__M__),
          title: document.title.includes(__M__),
          made: root.querySelectorAll('b, script, img').length,
          texts: [...root.querySelectorAll('[data-denoise-cell]')].slice(0, 3).map((c) => c.textContent)};
})()
"""


@requires_chrome
def test_pieces_are_shown_literally_and_only_inside_the_denoising_view(browser):
    """T-0159 B-3 and B-4: ``</script><b>x`` and an ``<img onerror>`` piece show literally and make
    no element; a marker piece is inside ``[data-denoise-root]`` and nowhere else in the document
    (the other panes, the label, the title), through a thought canvas and a commit."""
    mark = "QZXthoughtQZX"
    pieces = [f"{mark}{p}" for p in range(CL)]
    pieces[0] = "</script><b>x"
    pieces[1] = "<img src=x onerror=window.__t0159_xss=1>"
    pieces[2] = "<|channel>thought"
    with hud_page(browser) as page:
        page.push(with_pieces(f1(), pieces))
        page.push(with_pieces(commit(), [f" {mark}c{p}" for p in range(100)]), gap_s=0.05)
        page.wait_label(COMMIT_LABEL, 5)
        page.push(with_pieces(f2(), pieces))
        page.wait_label(F2_LABEL, 5)
        page.settle(0.5)
        out = dict(page.chrome.evaluate(ISOLATION_JS.replace("__M__", json.dumps(mark))) or {})
        xss = page.chrome.evaluate("window.__t0159_xss === 1")
    assert out.get("ok") and out["inside"], (
        f"premise: the pieces are shown: {out}{page.missing_panel()}"
    )
    assert out["texts"][:2] == ["</script><b>x", "<img src=x onerror=window.__t0159_xss=1>"], (
        f"B-3: pieces shown literally: {out['texts']}"
    )
    assert out["made"] == 0 and not xss, "B-3: a piece became markup"
    assert not out["outside"] and not out["title"], (
        "B-4: model text from a canvas frame appears outside [data-denoise-root]"
    )


# --- T-0159 B-9 (4): the disclosure from the keyboard -----------------------------------------------

DISCLOSURE_JS = """
(() => {
  const secs = [...document.querySelectorAll('[data-denoise-root] > details')];
  const s = secs[__I__];
  if (!s) return {ok: false, why: 'no section __I__ of ' + secs.length};
  const sum = s.querySelector('summary');
  if (!sum) return {ok: false, why: 'no summary'};
  if (__FOCUS__) sum.focus();
  return {ok: true, open: s.open, focused: document.activeElement === sum, tab: sum.tabIndex,
          opens: secs.map((x) => x.open)};
})()
"""


def disclosure_js(i: int, focus: bool) -> str:
    return DISCLOSURE_JS.replace("__I__", str(i)).replace("__FOCUS__", "true" if focus else "false")


@requires_chrome
def test_a_section_opens_and_closes_from_the_keyboard(browser):
    """T-0159 B-7 + B-9 (4): canvas 0, then canvas 1 (canvas 0's section closes, canvas 1's is
    open); the summary of canvas 0 is in the tab order and takes the focus; Enter opens it, Enter
    closes it, Space opens it again (real key events through CDP)."""
    with hud_page(browser) as page:
        page.push(with_pieces(commit(), ["a"] * 100))
        page.wait_label(COMMIT_LABEL, 5)
        c1 = denoise(3, kept=range(0, 10), stable=set(), canvas_idx=1)
        page.push(with_pieces(c1, ["b"] * CL), gap_s=0.05)
        page.wait_label("canvas 1 · step 3/48", 5)
        page.settle(0.4)
        s = page.state()
        start = dict(page.chrome.evaluate(disclosure_js(0, True)) or {})
        page.chrome.enter()
        page.settle(0.2)
        after_enter = dict(page.chrome.evaluate(disclosure_js(0, False)) or {})
        page.chrome.enter()
        page.settle(0.2)
        after_enter2 = dict(page.chrome.evaluate(disclosure_js(0, False)) or {})
        page.chrome.key(" ", text=" ", code=32)
        page.settle(0.2)
        after_space = dict(page.chrome.evaluate(disclosure_js(0, False)) or {})
    assert [x["open"] for x in s["sections"]] == [False, True], (
        f"B-7: the previous newest closed, the new one open: {s['sections']}{page.missing_panel()}"
    )
    assert COMMIT_LABEL in s["sections"][0]["summary"], s["sections"]
    assert start.get("ok") and start["focused"] and start["tab"] >= 0, (
        f"B-9: the summary takes the keyboard focus: {start}"
    )
    assert start["open"] is False
    assert after_enter.get("open") is True, f"B-9: Enter does not open the section: {after_enter}"
    assert after_enter2.get("open") is False, f"B-9: Enter does not close it: {after_enter2}"
    assert after_space.get("open") is True, f"B-9: Space does not open it: {after_space}"


# --- T-0159 B2: the trailing end-of-sequence canvas ---------------------------------------------------

#: Every cell of every section with what Chrome computes for it: rendered (``checkVisibility``,
#: which is false inside a closed ``<details>``, and a non-empty box) and its text.
VISIBLE_CELLS_JS = (
    "(() => {"
    + CELLS_PRELUDE
    + """
  const shown = (c) => {
    const r = c.getBoundingClientRect();
    const vis = typeof c.checkVisibility === 'function'
      ? c.checkVisibility({checkOpacity: true, checkVisibilityCSS: true})
      : getComputedStyle(c).visibility !== 'hidden';
    return vis && r.width > 0 && r.height > 0;
  };
  return {ok: !!root, sections: sections.map((s) => ({
    open: s.open,
    summary: ((s.querySelector('summary') || {}).textContent || '').replace(/\\s+/g, ' ').trim(),
    shown: cellsOf(s).filter(shown).map((c) => c.textContent),
    cells: cellsOf(s).length,
  }))};
})()"""
)


@requires_chrome
def test_a_trailing_end_of_sequence_canvas_is_folded_and_the_answer_is_on_screen(browser):
    """T-0159 B2-2/B2-3 (the B2-6 Chrome row): canvas 0 commits 100 pieces " w0" .. " w99", then
    canvas 1's first frame has ``<eos>`` at all 256 positions. Checked through what Chrome renders:
    canvas 1's section is closed and its summary reads "canvas 1 · end of sequence"; canvas 0's
    section is open, and the rendered cells are exactly canvas 0's (display "w0" .. "w99", then 156
    empty cells), with no rendered ``<eos>``. B2-2/B2-5: canvas 1 still has its 256 cells, the
    pane label is the newest frame's B4 label and both frames are counted."""
    want_c0 = [f"w{i}" for i in range(100)] + [""] * (CL - 100)
    with hud_page(browser) as page:
        page.push(with_pieces(commit(), [f" w{i}" for i in range(100)]))
        page.wait_label(COMMIT_LABEL, 5)
        c1 = denoise(1, kept=range(0, 0), stable=set(), canvas_idx=1)
        page.push(with_pieces(c1, ["<eos>"] * CL), gap_s=0.05)
        page.wait_label("canvas 1 · step 1/48", 5)
        page.settle(0.7)
        s = page.state()
        v = dict(page.chrome.evaluate(VISIBLE_CELLS_JS) or {})
    secs = v.get("sections") or []
    assert v.get("ok") and len(secs) == 2, f"B2-5: two sections: {v}{page.missing_panel()}"
    s0, s1 = secs
    assert "canvas 1 · end of sequence" in s1["summary"] and "step" not in s1["summary"], (
        f"B2-2: the folded section's summary: {s1['summary']!r}"
    )
    assert s1["open"] is False, f"B2-2: the end-of-sequence section is closed: {s1}"
    assert s1["shown"] == [], f"B2-2: none of its cells is rendered while closed: {s1['shown'][:8]}"
    assert s1["cells"] == CL, "B2-2/B2-5: it keeps its cells"
    assert s0["open"] is True, f"B2-3: the answer's section is open: {s0['summary']!r}"
    assert COMMIT_LABEL in s0["summary"], s0["summary"]
    assert s0["shown"] == want_c0, (
        f"B2-3: the rendered cells are canvas 0's text; first shown: {s0['shown'][:6]}"
    )
    assert s["label"] == "canvas 1 · step 1/48", f"B2-2: the pane label: {s['label']!r}"
    assert s["frames"] == "2", f"B2-5: both frames counted: {s['frames']!r}"


# --- B3 aria-label (kept, T-0159 reading) --------------------------------------------------------------


def _count_near(text: str, name: str) -> list[int]:
    found = re.findall(rf"(\d+)\s*{name}\b", text) + re.findall(rf"\b{name}\W{{0,3}}(\d+)", text)
    return [int(x) for x in found]


@requires_chrome
def test_the_newest_section_states_the_label_and_the_four_counts_in_an_aria_label(browser):
    """T-0128c B3 + O-B (3) in the T-0159 form: the newest section (or an element in it) has an
    ``aria-label`` holding the label and stable 32, kept 80, noise 128, revised 16 (hand-counted from
    F2); after the commit, its label with committed 100 and empty 156."""
    with hud_page(browser) as page:
        page.push(f1(), f2(), gap_s=0.05)
        page.wait_label(F2_LABEL, 5)
        page.settle(0.6)
        s = page.state()
        page.push(commit())
        page.wait_label(COMMIT_LABEL, 5)
        page.settle(0.6)
        sc = page.state()
    aria = next((a for a in s.get("arias") or [] if F2_LABEL in a), "")
    assert aria, f"no aria-label with {F2_LABEL!r} in the newest section: {s.get('arias')}"
    wrong = {n: _count_near(aria, n) for n, v in F2_COUNTS.items() if v not in _count_near(aria, n)}
    assert wrong == {}, (
        f"aria-label {aria!r}: counts not stated next to their names: {wrong} (want {F2_COUNTS})"
    )
    caria = next((a for a in sc.get("arias") or [] if COMMIT_LABEL in a), "")
    assert caria, f"after the commit the aria-labels are {sc.get('arias')}"
    assert 100 in _count_near(caria, "committed") and 156 in _count_near(caria, "empty"), (
        f"commit aria-label {caria!r}: want committed 100 and empty 156 (O-B (3))"
    )


# --- B8: reduced motion -------------------------------------------------------------------------------


def burst_frames() -> list[AstraCanvasStep]:
    """50 frames, steps 1..50 of 64; frame i has positions 0..4i+3 stable (so the last frame's
    state differs from every earlier one at cells 196-199)."""
    return [
        denoise(i + 1, kept=range(0), stable=set(range(0, 4 * i + 4)), max_steps=64)
        for i in range(BURST)
    ]


def _burst(page: Page) -> tuple[dict[str, Any], dict[str, Any], float]:
    frames = burst_frames()
    start_draws = _draws(page.state())
    t0 = time.perf_counter()
    for i, frame in enumerate(frames):
        delay = t0 + i * FRAME_PERIOD_S - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        page.stub.push(frame)
    span = time.perf_counter() - t0
    page.settle(1.2)
    s = page.state()
    rec = page.chrome.evaluate("window.__t0128c") or {}
    s["start_draws"] = start_draws
    return s, rec, span


def _labels_want() -> list[str]:
    return [f"canvas 0 · step {i + 1}/64" for i in range(BURST)]


def _latest_wins(s: dict[str, Any]) -> list[str]:
    """The burst's last frame: 0..199 stable, 200.. noise; read from the newest section's cells."""
    states = s.get("newest_states") or []
    if len(states) != CL:
        return [f"the newest section has {len(states)} cells, want {CL}"]
    return [
        f"cell {p}: {states[p]!r}, want {want}"
        for p, want in ((0, "stable"), (199, "stable"), (200, "noise"), (255, "noise"))
        if states[p] != want
    ]


@requires_chrome
def test_reduced_motion_redraws_at_most_twice_a_second_and_the_label_takes_every_frame(browser):
    """B8 + O-B (2): under ``prefers-reduced-motion: reduce``, 50 frames at 25 fps: no 1 s window
    on the page clock holds more than two draws, at most 5 draws in all (500 ms apart over a
    ~2 s burst: 0, .5, 1, 1.5, 2), at least 2 (it still draws), the label shows all 50 labels in
    order, and the last frame wins (cells 199 stable, 200 noise — T-0159: read from the cells'
    states instead of the bitmap)."""
    with hud_page(browser, reduce=True) as page:
        s, rec, span = _burst(page)
    draws = _draws(s) - s["start_draws"]
    times = sorted(t for t, _ in (rec.get("draws") or []))
    crowded = [round(t) for t in times if sum(1 for u in times if t <= u < t + 980.0) > 2]
    assert s["frames"] == str(BURST), (
        f"data-denoise-frames {s['frames']!r} (O-B (6)){page.missing_panel()}"
    )
    assert span < 2.4, (
        f"premise: the burst took {span:.2f} s (host too loaded to pace 40 ms frames)"
    )
    assert 2 <= draws <= 5, f"B8: {draws} draws for {BURST} frames over {span:.2f} s under reduce"
    assert crowded == [], f"B8: more than two draws within 1 s starting at page ms {crowded}"
    assert (rec.get("labels") or [])[-BURST:] == _labels_want(), (
        f"O-B (2): the label must take every frame's label under reduce; got {rec.get('labels')}"
    )
    problems = _latest_wins(s)
    assert problems == [], "B8 latest wins: the last frame is not what is drawn:\n  " + "\n  ".join(
        problems
    )


@requires_chrome
def test_without_reduced_motion_most_frames_are_drawn(browser):
    """B8's other side (the guard too wide): without ``reduce`` the same burst is drawn at least
    80 % of the time (>= 40 draws for 50 frames), so the throttle is not always on."""
    with hud_page(browser, reduce=False) as page:
        s, rec, span = _burst(page)
    draws = _draws(s) - s["start_draws"]
    assert s["frames"] == str(BURST), f"data-denoise-frames {s['frames']!r}{page.missing_panel()}"
    assert draws >= int(0.8 * BURST), (
        f"{draws} draws for {BURST} frames over {span:.2f} s without reduced motion"
    )
    assert (rec.get("labels") or [])[-BURST:] == _labels_want()


def new_frame_draws(rec: dict[str, Any], final: str | None) -> int:
    """O-B3 F1: the number of digest writes that changed the digest's value. Write i set the value
    that is write i + 1's old value (the last one set ``final``)."""
    olds = [old for _t, old in (rec.get("digests") or [])]
    if not olds:
        return 0
    after = [*olds[1:], final]
    return sum(1 for before, now in zip(olds, after, strict=True) if before != now)


def test_new_frame_draws_counts_value_changes_only():
    """Anti-vacuity for the counter (hand-built records): writes None->a, a->a (a repaint), a->b,
    b->a give 3 changes; no write gives 0."""
    rec = {"digests": [[0, None], [1, "a"], [2, "a"], [3, "b"]]}
    assert new_frame_draws(rec, "a") == 3
    assert new_frame_draws({"digests": []}, "a") == 0


@requires_chrome
def test_f1_reduced_motion_with_the_rail_resizing_every_frame_draws_at_most_5_new_frames(browser):
    """O-B3 F1 (Chrome twin of the node row): under reduced motion, 50 frames at 25 Hz with the
    viewport switching 1440 <-> 1000 px wide before every frame (the rail is 342 px, then the
    one-column pane). Every burst frame has its own state array (frame i: positions 0..4i+3
    stable), so a repaint of the same frame at a new size keeps the digest (allowed) and drawing a
    newer frame changes it: at most 5 digest changes, at least 2, the label takes all 50, and
    latest wins at the end (T-0159: read from the cells' states)."""
    with hud_page(browser, reduce=True) as page:
        frames = burst_frames()
        t0 = time.perf_counter()
        for i, frame in enumerate(frames):
            delay = t0 + i * FRAME_PERIOD_S - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            w, h = (1000, 800) if i % 2 else (1440, 900)
            page.chrome._cmd(
                "Emulation.setDeviceMetricsOverride",
                width=w,
                height=h,
                deviceScaleFactor=1,
                mobile=False,
            )
            page.stub.push(frame)
        span = time.perf_counter() - t0
        _metrics(page.chrome, VIEW, 1)
        page.settle(1.2)
        s = page.state()
        rec = page.chrome.evaluate("window.__t0128c") or {}
    changes = new_frame_draws(rec, s["digest"])
    assert s["frames"] == str(BURST), f"data-denoise-frames {s['frames']!r}{page.missing_panel()}"
    assert span < 2.6, f"premise: the burst took {span:.2f} s (host too loaded to pace it)"
    assert changes >= 2, f"premise: reduced motion still draws new frames ({changes})"
    assert changes <= 5, (
        f"O-B3 F1: {changes} new-frame draws for {BURST} frames over {span:.2f} s with the rail "
        f"resizing each frame under reduced motion (<= 5); total draws {s['draws']}"
    )
    assert (rec.get("labels") or [])[-BURST:] == _labels_want(), (
        f"O-B (2): the label takes every frame; got {rec.get('labels')}"
    )
    problems = _latest_wins(s)
    assert problems == [], "latest wins: " + "; ".join(problems)


# --- B7: a reload shows the empty state; canvas frames are not replayed ---------------------------------


@requires_chrome
def test_a_reload_shows_the_empty_state_until_the_next_frame(browser):
    """B7: drawn, then reloaded (the stub replays nothing, like ``HudServer``'s hello journal, which
    ignores ``astra.canvas.*``): the root is empty, the label empty, B5's copy shows; the next
    frame draws a section again (T-0159: a section, not a bitmap)."""
    with hud_page(browser) as page:
        page.push(f2())
        drawn = page.wait_section(5)
        hellos = page.stub.received_types().count("astra.hello")
        page.chrome.reload()
        page.stub.wait_for(lambda st: st.received_types().count("astra.hello") > hellos, 20)
        page.chrome.wait_panes(ALL_PANES)
        page.settle(1.0)
        s = page.state()
        page.push(f1())
        again = page.wait_section(5)
    assert drawn, f"premise: the frame was drawn before the reload{page.missing_panel()}"
    assert s["root_children"] == [] and s["canvases"] == 0, s["root_children"]
    assert s["label"] == "" and _draws(s) == 0, (s["label"], s["draws"])
    assert s["empty"] == DENOISE_EMPTY_T0128C and s["empty_visible"], s["empty"]
    assert again, "the next frame after a reload draws"


# --- harness self-checks (no implementation needed) ------------------------------------------------------


@requires_chrome
def test_the_screen_sampler_reads_what_the_page_painted(browser):
    """Anti-vacuity for the sampling path the colour rows lean on: a page-painted 40x40 block of
    a known colour at a known place reads back within dE00 1 through ``_pixel``, and a 1x1
    ``Page.captureScreenshot`` decodes through ``decode_png`` to the same value."""
    with hud_page(browser) as page:
        page.chrome.evaluate(
            "(() => { const d = document.createElement('div');"
            " d.style.cssText = 'position:fixed;left:100px;top:120px;width:40px;height:40px;"
            "background:#3c7a2e;z-index:99999'; d.id = 't0128c-probe';"
            " document.body.appendChild(d); return true; })()"
        )
        page.settle(0.2)
        got = _pixel(page.chrome, 120, 140)
        shot = page.chrome._cmd(
            "Page.captureScreenshot",
            format="png",
            clip={"x": 110, "y": 130, "width": 1, "height": 1, "scale": 1},
        )
    _w, _h, rows = decode_png(base64.b64decode(shot["data"]))
    assert delta_e(got, hex_rgb("#3c7a2e")) < 1.0, hexc(got)
    assert delta_e(tuple(float(v) for v in rows[0][0]), hex_rgb("#3c7a2e")) < 1.0  # type: ignore[arg-type]


def test_the_colour_oracle_tells_the_six_states_apart_in_both_themes():
    """Anti-vacuity for ``classify``: with the HUD's own tokens (hud/styles.css, hand-copied:
    dark --accent #8ea2ff --accent-2 #b48cff --surface #1b1e26; light #4052d6 #7a4fe0 #ffffff),
    every state's exact colour classifies as itself, and noise vs empty (the closest pair) are
    more than dE00 2 apart, so the 2.0 tolerance can tell them apart."""
    for tokens, surface in (
        ({"accent": "#8ea2ff", "accent-2": "#b48cff"}, "#1b1e26"),
        ({"accent": "#4052d6", "accent-2": "#7a4fe0"}, "#ffffff"),
    ):
        bd = hex_rgb(surface)
        for state in STATE_PAINT:
            assert classify(expected_colour(state, tokens, bd), state, tokens, bd) is None, state
        assert (
            delta_e(expected_colour("noise", tokens, bd), expected_colour("empty", tokens, bd))
            > 2.0
        )
        assert classify(expected_colour("stable", tokens, bd), "noise", tokens, bd) is not None


def test_the_geometry_evaluator_on_hand_built_layouts():
    """Anti-vacuity for ``_geometry_problems`` (no browser): a hand-built 8-column rail layout of
    256 equal cells inside a 342 px root with a vertical scroller passes; the same with one wide
    cell, with 7 columns, with the page scrolling sideways, or with no scroller fails, each for
    its own reason."""

    def layout(cols: int = 8, wide: int | None = None, scroll_w: int = 1440, scrollers=True):
        w = (342 - (cols - 1) * 4) / cols
        cells = []
        for p in range(CL):
            x = 10 + (p % cols) * (w + 4)
            cw = w + (30 if p == wide else 0)
            cells.append({"x": x, "y": 100 + (p // cols) * 24, "w": cw, "h": 20, "right": x + cw})
        return {
            "ok": True,
            "root": {"x": 10, "right": 352, "w": 342, "sw": 342, "cw": 342},
            "cells": cells,
            "scrollers": [{"tag": "div"}] if scrollers else [],
            "scroll_w": scroll_w,
            "inner_w": 1440,
        }

    assert _geometry_problems(layout(), "ok", rail=True) == []
    assert any("fixed width" in p for p in _geometry_problems(layout(wide=3), "w", rail=True))
    assert any("columns" in p for p in _geometry_problems(layout(cols=7), "c", rail=True))
    assert _geometry_problems(layout(cols=7), "c", rail=False) == []
    assert any("sideways" in p for p in _geometry_problems(layout(scroll_w=1500), "s", rail=True))
    assert any(
        "scrolls vertically" in p
        for p in _geometry_problems(layout(scrollers=False), "v", rail=True)
    )
