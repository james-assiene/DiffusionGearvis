"""T-0128c B10 / T-0159 B-9 — the Thinking pane's label against the mockup, and the token text's contrast.

**T-0159 rewrite.** T-0159 B-9 ("Pixel and DOM rows replace T-0128c B10") supersedes B10's grid rows:
the bitmap's 32 columns, its cell width and height, its gap and its lit-cell colour are no longer the
design (James's reference image: a grid of token cells, as many columns as fit the rail). Those five
rows are removed here; their replacements (each state's fill in both themes, no horizontal overflow
at 1280/1440/1920, the disclosure from the keyboard) are in tests/integration/test_hud_denoise.py.
The label rows stay (T-0159 B-7: the pane label still shows the newest drawn frame's label). B-9's
contrast rows (text on its fill: >= 4.5:1 settled/committed, >= 3:1 noise) are added at the end of
this module, pending T-0159 (they feed frames carrying pieces, so they need group A's decoder too).

The rest of this docstring is T-0128c's, kept for the mockup premise rows and the label rows.

docs/tasks/T-0128c.md criterion B10 and *Tests to write* "tests/integration/test_hud_denoise_parity.py".
The harness of tests/integration/test_hud_mockup_parity.py (T-0154) is imported, never copied or
edited: ``decode_png``, ``_pixel`` (which adds the window's scroll offsets itself), ``over``,
``delta_e``, ``Row``, ``T0128_DEVIATIONS``, the static server and the media/settle helpers.

**What is compared with what.** The mockup (``tests/fixtures/hud/mockup_20261021/hud.html``) is
served from its fixture directory; its Thinking pane is ``.rail > .panel:nth-child(2)``, its grid
``.noise`` with one ``<i>`` per cell (32 x 8), its label the heading row's ``span`` ("canvas 2 ·
step 31/48", 12 px, ``--muted``). The HUD is served by ``HudStaticServer`` with ``HudStub`` pushing
**one hand-built all-stable denoise frame**: CL 256 (8 rows of 32, the mockup's shape), canvas 2,
step 31 of 48 (the mockup's label, so the two labels are the same text), every position stable
and ``canvas == argmax``, so every cell is lit (``--accent`` alpha 1.0, B3). Both pages at
1440x900, DPR 1, reduced motion emulated, each theme set the way T-0154 sets it.

**How the grid is measured.** The mockup's cells are DOM elements: column count, cell size and
gap come from their rects. The HUD's grid is pixels on a ``<canvas>``, so it is measured from the
screen: a 1-px-wide vertical strip down column 0 finds the first lit row, then a 1-px-high strip
across that row's middle gives 32 lit runs; the mean run length is the cell size, the mean dark
run between them the gap (a pixel is lit when it is farther than dE00 2.5 from the pane's own
background). The same strip method is checked against the mockup itself (whose cell size is known
from its CSS: (380 - 2 - 36 - 31 x 2) / 32 = 8.75 px), so the measuring is not vacuous.
``_strip`` reads a 1-px strip with one ``Page.captureScreenshot``; like ``_pixel`` it adds the
window's scroll offsets (a clip is in document coordinates), and a row checks it agrees with
``_pixel`` at three points.

**Readings, stated so a reviewer can disagree:**

- "lit cell dE <= 2" compares the HUD's painted stable cell with the mockup's ``.noise i``
  background at alpha 1 (its computed colour, ``--accent``); for the light theme T-0128 moved
  ``--accent`` (``T0128_DEVIATIONS``), and B3 says the cell is ``--accent``, so the light target is
  T-0128's accent, as in T-0154 A2. The label colour likewise: ``--muted`` is a T-0128 move in both
  themes.
- "label right edge" is the label element's rect right against the mockup span's rect right.
- **Named deviation (B10):** the thinking-time meter sits between the heading and the grid, so the
  grid's top is not compared, and nothing vertical about the grid's position is.

**Adversarial (proved by the test-author against the group B implementation at 12c29a5, harness
deleted):** a scratch copy of the tree with ``.denoise canvas { width: 50% }`` appended to
``hud/styles.css``, run from inside the copy with ``PYTHONPATH`` set to it (origin printed from
inside the pytest process: ``local_astra`` and the served ``hud/`` were the copy's). The B10 row
went red in both themes, the cell-width row included: "cell width: 7.60 px vs the mockup's 8.75"
(and also column count 20 vs 32, cell height 42.00 vs 8.75, lit cell dE00 48). The same copy with
the sheet restored gave 6 passed.

Regime: Google Chrome 147 headless, CDP, DPR 1, 1440x900, data-theme dark and light, reduced
motion on; n = 1, deterministic, no seed. No GPU.
"""

from __future__ import annotations

import base64
import contextlib
import json
import statistics
import time
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from local_astra.contracts.protocol import AstraCanvasStep
from local_astra.hud_static import HudStaticServer
from tests.fakes.hud_server import HudStub
from tests.integration.test_hud_chrome import choose_theme_js, contrast_ratio, parse_css_color
from tests.integration.test_hud_denoise import (
    CELLS_PRELUDE,
    CL,
    COMMIT_LABEL,
    F2_LABEL,
    commit,
    f1,
    f2,
    hud_page,
    with_pieces,
)
from tests.integration.test_hud_headless import CHROME, HUD, requires_chrome
from tests.integration.test_hud_mockup_parity import (
    COLOUR_TOL_DE,
    FONT_SIZE_TOL_PX,
    LENGTH_TOL_PX,
    MOCKUP,
    T0128_DEVIATIONS,
    THEMES,
    VIEW,
    Row,
    _media,
    _pixel,
    _serve_dir,
    _settle,
    decode_png,
    delta_e,
    hex_rgb,
    hexc,
    over,
)
from tests.integration.test_hud_redesign import ALL_PANES, _panels_line, _Recorder

pytestmark = pytest.mark.integration

#: B10's right-edge tolerance, verbatim (the others are T-0154's A2 constants, the same numbers).
RIGHT_EDGE_TOL_PX = 2.0
#: A pixel is "lit" when farther than this from the pane background (dE00).
LIT_DE = 2.5
#: Hand-read from the mockup's CSS: rail 380, panel border 1 + padding 18 each side, 32 columns,
#: gap 2 -> (380 - 2 - 36 - 62) / 32.
MOCK_CELL_PX = 8.75
MOCK_GAP_PX = 2.0
MOCK_LABEL = "canvas 2 · step 31/48"

M_PANE = ".rail > .panel:nth-child(2)"
M_GRID = f"{M_PANE} .noise"
M_CELLS = f"{M_PANE} .noise i"
M_LABEL = f"{M_PANE} .rh span"
H_PANE = "main > .pane-thinking"
H_LABEL = "[data-denoise-label]"

L, F, C = LENGTH_TOL_PX, FONT_SIZE_TOL_PX, COLOUR_TOL_DE

#: B10's rows, in T-0154's Row shape (``prop`` names the measurement below). T-0159 B-9 removed the
#: five grid rows (column count, cell width, cell height, gap, lit cell colour); see the docstring.
B10: list[Row] = [
    Row("label: font size", M_LABEL, H_LABEL, "font_size", F),
    Row("label: colour", M_LABEL, H_LABEL, "color", C, "muted", backdrop=M_PANE),
    Row("label: right edge", M_LABEL, H_LABEL, "right", RIGHT_EDGE_TOL_PX),
]


def all_stable_frame() -> AstraCanvasStep:
    n = 256
    return AstraCanvasStep(
        request_id="chatcmpl-t0128cparity00000-0a1b2c3d",
        canvas_idx=2,
        step=31,
        max_steps=48,
        canvas_length=n,
        canvas=tuple(100 + p for p in range(n)),
        argmax=tuple(100 + p for p in range(n)),
        stable="ff" * (n // 8),
        phase="denoise",
    )


# --- measuring ---------------------------------------------------------------------------------------

MEASURE_JS = """
(() => {
  window.scrollTo(0, 0);
  const S = __S__;
  const look = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect(), s = getComputedStyle(el), px = (v) => parseFloat(v) || 0;
    return {x: r.x, y: r.y, w: r.width, h: r.height, right: r.right, bottom: r.bottom,
            cx: r.x + px(s.borderLeftWidth) + px(s.paddingLeft),
            cy: r.y + px(s.borderTopWidth) + px(s.paddingTop),
            cw: r.width - px(s.borderLeftWidth) - px(s.borderRightWidth) - px(s.paddingLeft) - px(s.paddingRight),
            ch: r.height - px(s.borderTopWidth) - px(s.borderBottomWidth) - px(s.paddingTop) - px(s.paddingBottom),
            font_size: px(s.fontSize), color: s.color, background: s.backgroundColor,
            background_image: s.backgroundImage, opacity: s.opacity,
            text: (el.textContent || '').trim()};
  };
  const out = {};
  for (const [k, sel] of Object.entries(S)) out[k] = [...document.querySelectorAll(sel)].map(look);
  const rs = getComputedStyle(document.documentElement);
  out.__tokens__ = Object.fromEntries(['--accent', '--muted', '--surface'].map((t) => [t, rs.getPropertyValue(t).trim()]));
  out.__theme__ = document.documentElement.getAttribute('data-theme');
  out.__scroll__ = [window.scrollX, window.scrollY];
  return out;
})()
"""


def measure_js(selectors: dict[str, str]) -> str:
    return MEASURE_JS.replace("__S__", json.dumps(selectors))


def _strip(
    chrome: _Recorder, x: int, y: int, w: int, h: int
) -> list[list[tuple[float, float, float]]]:
    """A ``w x h`` block of pixels at *viewport* point (x, y), one capture; the scroll offsets are
    added here exactly as ``_pixel`` adds them (``_pixel`` is not called, so nothing is added twice)."""
    off = chrome.evaluate("[window.scrollX, window.scrollY]") or [0, 0]
    shot = chrome._cmd(
        "Page.captureScreenshot",
        format="png",
        clip={"x": x + off[0], "y": y + off[1], "width": w, "height": h, "scale": 1},
    )
    _w, _h, rows = decode_png(base64.b64decode(shot["data"]))
    return [[(float(p[0]), float(p[1]), float(p[2])) for p in row] for row in rows]


def runs(
    line: list[tuple[float, float, float]], backdrop: tuple[float, float, float]
) -> tuple[list[tuple[int, int]], list[int]]:
    """(lit runs as (start, length), dark run lengths strictly between two lit runs)."""
    lit = [delta_e(p, backdrop) > LIT_DE for p in line]
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(lit):
        if lit[i]:
            j = i
            while j < len(lit) and lit[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    gaps = [out[k + 1][0] - (out[k][0] + out[k][1]) for k in range(len(out) - 1)]
    return out, gaps


def test_the_run_finder_on_a_hand_built_line():
    """Anti-vacuity for ``runs``: 3 lit, 2 dark, 4 lit, 1 dark, 1 lit (backdrop black, lit white)."""
    b, w = (0.0, 0.0, 0.0), (255.0, 255.0, 255.0)
    line = [w] * 3 + [b] * 2 + [w] * 4 + [b] + [w]
    assert runs(line, b) == ([(0, 3), (5, 4), (10, 1)], [2, 1])
    assert runs([b] * 5, b) == ([], [])


@dataclass
class Grid:
    columns: int = 0
    cell_w: float = 0.0
    cell_h: float = 0.0
    gap: float = 0.0
    lit: tuple[float, float, float] | None = None
    backdrop: tuple[float, float, float] | None = None
    note: str = ""


def measure_grid(
    chrome: _Recorder, box: dict[str, Any], backdrop: tuple[float, float, float]
) -> Grid:
    """Strip-measure a 32-column grid whose content box is ``box`` (viewport coordinates)."""
    g = Grid(backdrop=backdrop)
    x0, y0 = int(box["cx"]), int(box["cy"])
    w, h = max(1, int(box["cw"]) + 2), max(1, int(box["ch"]) + 2)
    col0_x = int(box["cx"] + box["cw"] / 64)
    vertical = [row[0] for row in _strip(chrome, col0_x, y0, 1, h)]
    vruns, _ = runs(vertical, backdrop)
    if not vruns:
        g.note = f"no lit pixel down column 0 at x={col0_x}"
        return g
    g.cell_h = statistics.mean(length for _s, length in vruns)
    mid = y0 + vruns[0][0] + vruns[0][1] // 2
    across = _strip(chrome, x0, mid, w, 1)[0]
    hruns, gaps = runs(across, backdrop)
    g.columns = len(hruns)
    if hruns:
        g.cell_w = statistics.mean(length for _s, length in hruns)
        start, length = hruns[len(hruns) // 2]
        g.lit = across[start + length // 2]
    if gaps:
        g.gap = statistics.mean(gaps)
    g.note = f"strip y={mid}: {len(hruns)} runs {[n for _s, n in hruns]}, gaps {gaps}"
    return g


@dataclass
class Side:
    styles: dict[str, Any] = field(default_factory=dict)
    grid: Grid = field(default_factory=Grid)
    strip_check: list[tuple[Any, Any]] = field(default_factory=list)


@dataclass
class Parity:
    error: str = ""
    mock: dict[str, Side] = field(default_factory=dict)
    hud: dict[str, Side] = field(default_factory=dict)
    hud_error: dict[str, str] = field(default_factory=dict)


def _opaque(colour: str, page: tuple[float, float, float]) -> tuple[float, float, float]:
    return over(colour, page)


def _measure_mock(chrome: _Recorder, theme: str) -> Side:
    side = Side()
    side.styles = dict(
        chrome.evaluate(
            measure_js({"pane": M_PANE, "grid": M_GRID, "cells": M_CELLS, "label": M_LABEL})
        )
        or {}
    )
    pane = side.styles["pane"][0]
    page_bg = hex_rgb(side.styles["__tokens__"]["--surface"])
    backdrop = _opaque(pane["background"], page_bg)
    grid = side.styles["grid"][0]
    side.grid = measure_grid(chrome, grid, backdrop)
    # _strip vs _pixel at three points of the across strip (the sampler self-check).
    cells = side.styles["cells"]
    y = int(cells[0]["y"] + cells[0]["h"] / 2)
    xs = [int(cells[0]["x"]) + 3, int(cells[1]["x"]) - 1, int(cells[5]["x"]) + 4]
    block = _strip(chrome, xs[0], y, xs[-1] - xs[0] + 1, 1)[0]
    side.strip_check = [(block[x - xs[0]], _pixel(chrome, x, y)) for x in xs]
    return side


def _measure_hud(chrome: _Recorder) -> Side:
    side = Side()
    side.styles = dict(chrome.evaluate(measure_js({"pane": H_PANE, "label": H_LABEL})) or {})
    return side


def _run(result: Parity, browser: Any) -> None:
    if isinstance(browser, str):
        raise RuntimeError(browser)
    chrome: _Recorder = browser
    mock_server, mock_port = _serve_dir(MOCKUP)
    stub = HudStub([]).start()
    hud_server = HudStaticServer(HUD, 0)
    try:
        chrome.reset()
        chrome._cmd("Runtime.enable")
        chrome._cmd("Network.enable")
        for theme in THEMES:
            _media(chrome, theme)
            chrome.viewport(*VIEW)
            chrome.navigate(f"http://127.0.0.1:{mock_port}/hud.html?theme={theme}")
            chrome.wait_until("document.readyState === 'complete'", 15)
            _settle(chrome)
            result.mock[theme] = _measure_mock(chrome, theme)

        _media(chrome, "dark")
        chrome.viewport(*VIEW)
        chrome.navigate(
            f"http://127.0.0.1:{hud_server.port}/?ws=" + urllib.parse.quote(stub.url, safe="")
        )
        stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20)
        stub.wait_for_client(10)
        chrome.wait_panes(ALL_PANES)
        deadline = time.monotonic() + 15
        while _panels_line(chrome.console[chrome._mark :]) is None and time.monotonic() < deadline:
            chrome.drain(0.1)
        stub.push(all_stable_frame())
        drawn = chrome.wait_until(
            "Number((document.querySelector('[data-denoise-root]') || {getAttribute: () => 0})"
            ".getAttribute('data-denoise-draws') || 0) >= 1",
            8,
        )
        for theme in THEMES:
            chosen = chrome.evaluate(choose_theme_js(theme)) or {}
            _media(chrome, theme)
            # reduced motion: a redraw may wait up to 500 ms (B8); give it that and a frame
            time.sleep(0.8)
            _settle(chrome)
            if not (chosen.get("ok") and chosen.get("theme") == theme):
                result.hud_error[theme] = f"theme control: {chosen}"
            if not drawn:
                lines = [x for x in chrome.console if "denoise" in x]
                result.hud_error[theme] = f"the frame was never drawn (console: {lines})"
            result.hud[theme] = _measure_hud(chrome)
    finally:
        hud_server.close()
        stub.stop()
        mock_server.shutdown()
        mock_server.server_close()


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
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


@pytest.fixture(scope="module")
def parity(browser: Any) -> Iterator[Parity]:
    result = Parity()
    try:
        _run(result, browser)
    except Exception as exc:  # a broken session fails in a row, not here
        result.error = f"{type(exc).__name__}: {exc}"
    yield result


def _ok(p: Parity) -> None:
    assert not p.error, f"the parity session failed before any assertion ran: {p.error}"


# --- the premise: the mockup's grid is what its CSS says, and the strip method reads it ------------


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_the_mockup_thinking_pane_is_what_its_css_says_and_the_strip_measures_it(parity, theme):
    """Hand-read from hud.html: 256 cells, 32 per row, 8.75 x 8.75 px, gap 2, background
    ``--accent``; the label reads "canvas 2 · step 31/48" at 12 px. Then the strip method on the
    mockup's own pixels finds 32 runs, cell 8.75 +/- 1, gap 2 +/- 1; and ``_strip`` agrees with
    ``_pixel`` at three points."""
    _ok(parity)
    m = parity.mock[theme]
    s = m.styles
    assert s["__theme__"] == theme
    cells = s["cells"]
    assert len(cells) == 256
    first_row = [c for c in cells if abs(c["y"] - cells[0]["y"]) < 0.5]
    assert len(first_row) == 32
    assert abs(cells[0]["w"] - MOCK_CELL_PX) < 0.01 and abs(cells[0]["h"] - MOCK_CELL_PX) < 0.01, (
        cells[0]
    )
    assert abs((cells[1]["x"] - cells[0]["right"]) - MOCK_GAP_PX) < 0.01
    assert parse_css_color(cells[0]["background"])[:3] == hex_rgb(s["__tokens__"]["--accent"])
    label = s["label"][0]
    assert label["text"] == MOCK_LABEL and label["font_size"] == 12, label
    g = m.grid
    assert g.columns == 32, g.note
    assert (
        abs(g.cell_w - MOCK_CELL_PX) <= LENGTH_TOL_PX and abs(g.gap - MOCK_GAP_PX) <= LENGTH_TOL_PX
    ), (
        g.cell_w,
        g.gap,
        g.note,
    )
    for a, b in m.strip_check:
        assert delta_e(a, b) < 1.0, (hexc(a), hexc(b))


# --- B10 --------------------------------------------------------------------------------------------


def _target(
    row: Row, theme: str, measured: str, backdrop: tuple[float, float, float]
) -> tuple[tuple[float, float, float], str]:
    if row.deviation and (row.deviation, theme) in T0128_DEVIATIONS:
        hexv = T0128_DEVIATIONS[(row.deviation, theme)]
        return hex_rgb(hexv), f"T-0128 --{row.deviation} {hexv} (named deviation)"
    return over(measured, backdrop), f"the mockup's {measured}"


def evaluate_b10(row: Row, mock: Side, hud: Side, theme: str) -> list[str]:
    ms, hs = mock.styles, hud.styles
    mpane = ms["pane"][0]
    mbackdrop = over(mpane["background"], hex_rgb(ms["__tokens__"]["--surface"]))
    if not hs.get("label"):
        return [f"HUD {H_LABEL} matched nothing"]
    hl, ml = hs["label"][0], ms["label"][0]
    if row.prop == "font_size":
        d = hl["font_size"] - ml["font_size"]
        return (
            []
            if abs(d) <= row.tol
            else [f"{hl['font_size']} px vs the mockup's {ml['font_size']} px"]
        )
    if row.prop == "color":
        hpane = (hs.get("pane") or [{}])[0]
        hbg = parse_css_color(str(hpane.get("background")))
        got = over(hl["color"], (hbg[0], hbg[1], hbg[2]))
        target, label = _target(row, theme, ml["color"], mbackdrop)
        de = delta_e(got, target)
        return (
            []
            if de <= row.tol
            else [f"label {hl['color']} ({hexc(got)}) vs {label} ({hexc(target)}), dE00 {de:.2f}"]
        )
    if row.prop == "right":
        d = hl["right"] - ml["right"]
        return (
            []
            if abs(d) <= row.tol
            else [f"right edge {hl['right']:.1f} vs the mockup's {ml['right']:.1f}"]
        )
    return [f"unknown row property {row.prop}"]


@requires_chrome
def test_the_b10_evaluator_passes_the_mockup_against_itself_and_fails_a_moved_label(parity):
    """Anti-vacuity for ``evaluate_b10`` (no implementation needed): the mockup's own label in the
    HUD's place passes the font-size and right-edge rows; the same label moved 5 px left fails the
    right-edge row and only it; 2 px larger fails the font-size row and only it. (T-0159 removed
    the grid rows this row used to exercise with a halved cell.)"""
    _ok(parity)
    mock = parity.mock["dark"]
    as_hud = Side(styles={"label": mock.styles["label"], "pane": mock.styles["pane"]})
    geometry = [r for r in B10 if r.prop in ("font_size", "right")]
    assert len(geometry) == 2
    assert [p for r in geometry for p in evaluate_b10(r, mock, as_hud, "dark")] == []
    label = dict(mock.styles["label"][0])
    moved = Side(styles={**as_hud.styles, "label": [{**label, "right": label["right"] - 5}]})
    assert [r.prop for r in geometry if evaluate_b10(r, mock, moved, "dark")] == ["right"]
    bigger = Side(
        styles={**as_hud.styles, "label": [{**label, "font_size": label["font_size"] + 2}]}
    )
    assert [r.prop for r in geometry if evaluate_b10(r, mock, bigger, "dark")] == ["font_size"]


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_b10_the_thinking_label_matches_the_mockup(parity, theme):
    """B10's label rows (the grid rows are superseded by T-0159 B-9): label font size +/- 0.5 px,
    colour dE00 <= 2 and right edge +/- 2 px, both themes; every failing row listed. Named
    deviation: the thinking-time meter between heading and grid (no vertical row)."""
    _ok(parity)
    assert theme not in parity.hud_error, parity.hud_error[theme]
    hud, mock = parity.hud[theme], parity.mock[theme]
    assert hud.styles.get("__theme__") == theme
    labels = hud.styles.get("label") or []
    assert labels and labels[0]["text"] == MOCK_LABEL, (
        f"HUD label {labels}: the frame's label is the mockup's text"
    )
    failures = [
        f"{row.role}: {problem}" for row in B10 for problem in evaluate_b10(row, mock, hud, theme)
    ]
    assert failures == [], (
        f"T-0128c B10 (data-theme={theme}): {len(failures)} row(s) differ from the mockup:\n  "
        + "\n  ".join(failures)
    )


# --- T-0159 B-9 (2): each cell's text contrasts with its fill --------------------------------------
#
# Readings (stated so a reviewer can disagree):
# - "settled/committed text" (>= 4.5:1) is read as the stable, revised and committed states; kept
#   (canvas == argmax, not yet stable) is held to the noise floor (>= 3:1). Empty cells carry no text.
# - "noise ... deliberately dim" (B-6 "noise cells show their current guess dimmed"): the noise
#   text's contrast on its fill is lower than the stable text's on its fill, same theme.
# - The fill is the screen pixel inside the cell (as in test_hud_denoise.py's fill rows); the text
#   colour is the text-bearing element's computed ``color`` with its alpha and the opacity of every
#   element between it and the cell, composited over that fill; the cell's own (and its ancestors')
#   opacity is applied to the text against the pane, as it is to the sampled fill. Glyph pixels are
#   not sampled (anti-aliasing makes them unreliable).

#: B-9's floors, verbatim.
SETTLED_MIN, NOISE_MIN = 4.5, 3.0
CONTRAST_FLOOR = {"stable": SETTLED_MIN, "revised": SETTLED_MIN, "committed": SETTLED_MIN}
CONTRAST_FLOOR |= {"kept": NOISE_MIN, "noise": NOISE_MIN}
#: One sampled cell per state (positions of test_hud_denoise's F2 and commit frames).
CONTRAST_CELLS = {"stable": 5, "revised": 35, "kept": 74, "noise": 212}
COMMITTED_CELL = 5

TEXT_COLOUR_JS = (
    "(() => {"
    + CELLS_PRELUDE
    + """
  const c = cellsOf(newest)[__P__];
  if (!c) return {ok: false, why: 'no cell __P__ in the newest section'};
  let t = c;
  for (;;) {
    const kids = [...t.children].filter((k) => (k.textContent || '') === (t.textContent || '') && k.textContent);
    if (kids.length !== 1) break;
    t = kids[0];
  }
  let inner = 1;
  for (let n = t; n && n !== c; n = n.parentElement) inner *= parseFloat(getComputedStyle(n).opacity);
  let outer = 1;
  for (let n = c; n; n = n.parentElement) {
    outer *= parseFloat(getComputedStyle(n).opacity);
    if (n === root) break;
  }
  return {ok: true, text: c.textContent, state: stateOf(c), color: getComputedStyle(t).color,
          inner, outer};
})()"""
)


def text_on_fill(
    color: str,
    inner: float,
    outer: float,
    fill: tuple[float, float, float],
    pane: tuple[float, float, float],
) -> tuple[float, float, float]:
    """The text's colour as painted: alpha ``a`` (its own times ``inner``) over the sampled fill,
    the ink itself seen through ``outer`` against the pane (derivation: with fill_raw under
    opacity ``outer``, sampled = outer*fill_raw + (1-outer)*pane, and the text pixel is
    outer*(a*t + (1-a)*fill_raw) + (1-outer)*pane = (1-a)*sampled + a*(outer*t + (1-outer)*pane))."""
    r, g, b, alpha = parse_css_color(color)
    a = max(0.0, min(1.0, alpha * inner))
    ink = tuple(outer * v + (1 - outer) * p for v, p in zip((r, g, b), pane, strict=True))
    return tuple((1 - a) * f + a * i for f, i in zip(fill, ink, strict=True))  # type: ignore[return-value]


def test_text_on_fill_by_hand():
    """Anti-vacuity for ``text_on_fill`` (hand arithmetic): opaque white text, no opacity, on a
    black fill is white; at alpha 0.5 it is mid-grey 127.5; under a cell opacity of 0.5 over a
    black pane, ink (255) becomes 127.5 and opaque text shows it; contrast white/black is 21."""
    black, white = (0.0, 0.0, 0.0), (255.0, 255.0, 255.0)
    assert text_on_fill("rgb(255, 255, 255)", 1, 1, black, black) == white
    assert text_on_fill("rgba(255, 255, 255, 0.5)", 1, 1, black, black) == (127.5, 127.5, 127.5)
    assert text_on_fill("rgb(255, 255, 255)", 1, 0.5, black, black) == (127.5, 127.5, 127.5)
    assert abs(contrast_ratio(white, black) - 21.0) < 1e-9


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_each_cells_text_contrasts_with_its_fill(browser, theme):
    """T-0159 B-9 (2): with the two-letter piece "Wg" in every cell, stable, revised and committed
    text is >= 4.5:1 on its fill, kept and noise >= 3:1, and noise text is dimmer (lower contrast)
    than stable text, in both themes."""
    ratios: dict[str, float] = {}
    problems: list[str] = []
    with hud_page(browser, theme=theme) as page:
        page.push(with_pieces(f1(), ["Wg"] * CL), with_pieces(f2(), ["Wg"] * CL), gap_s=0.05)
        page.wait_label(F2_LABEL, 5)
        page.settle(0.8)
        s = page.state()
        pane = parse_css_color(str(s.get("pane_bg")))[:3] if s.get("pane_bg") else (0.0, 0.0, 0.0)
        cells = dict(CONTRAST_CELLS)
        for state, p in cells.items():
            problems += _contrast(page, state, p, pane, ratios)
        page.push(with_pieces(commit(), ["Wg"] * 100))
        page.wait_label(COMMIT_LABEL, 5)
        page.settle(0.8)
        problems += _contrast(page, "committed", COMMITTED_CELL, pane, ratios)
    for state, ratio in ratios.items():
        if ratio < CONTRAST_FLOOR[state]:
            problems.append(
                f"{state}: text {ratio:.2f}:1 on its fill, want >= {CONTRAST_FLOOR[state]}"
            )
    if "noise" in ratios and "stable" in ratios and not ratios["noise"] < ratios["stable"]:
        problems.append(
            f"noise text {ratios['noise']:.2f}:1 is not dimmer than stable {ratios['stable']:.2f}:1"
        )
    assert problems == [], (
        f"T-0159 B-9 contrast (data-theme={theme}, ratios {ratios}):\n  " + "\n  ".join(problems)
    )


def _contrast(
    page: Any,
    state: str,
    p: int,
    pane: tuple[float, float, float],
    ratios: dict[str, float],
) -> list[str]:
    info = dict(page.chrome.evaluate(TEXT_COLOUR_JS.replace("__P__", str(p))) or {})
    pt, fill = page.cell_fill(p)
    if not info.get("ok") or fill is None:
        return [f"{state} cell {p}: {info.get('why') or pt.get('why')}{page.missing_panel()}"]
    if info["state"] != state or info["text"] != "Wg":
        return [f"cell {p}: state {info['state']!r} text {info['text']!r}, want {state} 'Wg'"]
    ink = text_on_fill(info["color"], info["inner"], info["outer"], fill, pane)
    ratios[state] = contrast_ratio(ink, fill)
    return []
