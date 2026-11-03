"""T-0154 — the HUD measured against the approved mockup, in the same headless Chrome (P1; A1-A3, A8).

docs/tasks/T-0154.md *Tests to write* P1 (group A's frame roles). P2 (group B's thread) and P3
(group C's card) are added to the role tables below by those groups; the harness is shared.

**What is compared with what.** The committed mockup (``tests/fixtures/hud/mockup_20261021/``,
MANIFEST.md has the SHA-256 of each file) is served from its fixture directory by a loopback static
server, its theme passed as ``?theme=`` (the mockup's own script reads it). The HUD is served by the
production ``HudStaticServer`` with ``HudStub`` replaying ``session_20261020_t3_t11.jsonl`` up to
the end of t9, the t9 card gated as ``scripts/demo/hud_screens.py`` gates it (the helpers are
imported from ``tests/integration/test_hud_redesign.py``), then one ``astra.turn.state`` SPEAKING
so the state pill is in the state the mockup draws ("Speaking"). No ``?source=``: no capture runs,
so ``#share-button`` is the plain header button the mockup's "Mute" button is compared with.
Both pages are measured in the one Chrome, at 1440x900 (and 1920x1080 for the max-width rows),
DPR 1, reduced motion emulated (so a decorative glyph animation cannot move a pixel probe), and
the theme set on the HUD through its Settings control with the system scheme emulated the same way.

**The reference is measured, never typed.** Every expected value comes from the mockup's computed
style or rect in the same run. The only hand-typed colours are the T-0128 token moves A2 names
(``T0128_DEVIATIONS``): for those roles the HUD is held to T-0128's token instead of the mockup's,
and the row says so.

**The role table (P1).** Each row is (role, mockup selector, HUD selector, property, tolerance,
deviation). Readings the task leaves open, stated so a reviewer can disagree with them:

- "app frame max-width 1480 and padding" is measured as geometry, not as declarations: the header's
  content box (rect minus its own padding and border) sits where the mockup's ``header`` sits, at
  1440x900 and at 1920x1080 (where the 1480 px cap binds), and the rail's right edge matches the
  mockup's ``.rail``. The implementer may put the padding on ``body``, a wrapper, ``main`` or the
  header itself.
- "header height" is the header's content-box height (the mockup's header has no padding).
- "rail panes ... padding 16/18" is measured as the inset of the pane's heading content box from
  the pane's border box (top, left): the HUD's sticky ``h2`` carries its pane's top padding today,
  and either structure renders the same.
- the header button compared with the mockup's plain ``.btn`` ("Mute") is ``#share-button`` while
  no capture runs; ``#mute-button`` is disabled then, so only its lengths are compared.
- colours with alpha (``--line``, ``--accent-soft``) are composited over one backdrop, the
  mockup's own colour behind the element, for both pages: the row compares the token, the pixel
  rows (A3) compare what is painted.
- shadows are compared layer by layer: same count, same inset flags, offsets/blur/spread +/- 1 px,
  each layer's colour composited over the backdrop within the colour tolerance.

**Pixels (A3).** Each probe is a 1x1 ``Page.captureScreenshot`` clip, decoded here with ``zlib``
(no new dependency; Pillow is not installed by default). CIEDE2000 is implemented here and checked
against published test pairs (Sharma, Wu and Dalal 2005).

**Adversarial (proved by the test-author, harness deleted).** A mutant HUD sheet with
``.pane { border-color: transparent }`` appended turns the border row red; a mutant
``.bar { background: var(--bg) }`` turns the gradient-under-header row red. The header must keep the
class ``bar`` for the second mutant to mean anything; ``test_the_header_keeps_the_class_the_mutant_names``
pins that.

Regime (T-0154 *Regime*): Google Chrome 147.0.7727.101 headless, CDP, DPR 1; 1440x900 and
1920x1080; data-theme dark and light; reduced motion on; n = 1, deterministic, no seed. No GPU.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.util
import json
import math
import re
import socketserver
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.protocol import AstraTurnState
from local_astra.contracts.turn import TurnState
from local_astra.hud_static import HudStaticServer
from tests.fakes.hud_server import HudStub
from tests.integration.test_hud_chrome import choose_theme_js, parse_css_color
from tests.integration.test_hud_headless import (
    A2UI_ROOT,
    CHROME,
    HUD,
    _QuietHandler,
    requires_chrome,
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

REPO = Path(__file__).resolve().parents[2]
MOCKUP = REPO / "tests" / "fixtures" / "hud" / "mockup_20261021"
MANIFEST = REPO / "tests" / "fixtures" / "hud" / "MANIFEST.md"
MOCKUP_FILES = (
    "hud.html",
    "cards.html",
    "tokens.css",
    "hud-dark.png",
    "hud-light.png",
    "cards-dark.png",
    "cards-light.png",
)
ARTIFACT_LINK = "https://claude.ai/artifact/TbSJM8AuFP8vwP7NfPDcbZ"
THEMES = ("dark", "light")
VIEW = (1440, 900)
WIDE = (1920, 1080)

#: A2's tolerances, verbatim.
LENGTH_TOL_PX = 1.0
FONT_SIZE_TOL_PX = 0.5
COLOUR_TOL_DE = 2.0
#: A3's tolerances, verbatim.
PIXEL_TOL_DE = 3.0
GRADIENT_MIN_DE = 2.0
BORDER_MIN_DE = 1.0

#: A2: "For the token moves T-0128 made for contrast (--bg, --surface, --muted, --faint, the light
#: accent, green, amber and red), the target is the T-0128 token". Hand-typed from hud/styles.css
#: at f6b027c (the T-0128 values, with their measured reasons in that file's comments). Only the
#: tokens T-0128 actually moved in a theme are listed: light --surface is #ffffff in both.
T0128_DEVIATIONS: dict[tuple[str, str], str] = {
    ("bg", "dark"): "#0b0d11",
    ("surface", "dark"): "#1b1e26",
    ("muted", "dark"): "#9aa1b2",
    ("faint", "dark"): "#939bad",
    ("bg", "light"): "#e9ebf1",
    ("muted", "light"): "#5d6579",
    ("faint", "light"): "#596175",
    ("accent", "light"): "#4052d6",
    ("green", "light"): "#107a47",
    ("amber", "light"): "#8a5a0f",
    ("red", "light"): "#b8283e",
}


# --- colour science (CIEDE2000, by hand) ------------------------------------------------------


def _srgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """sRGB (0-255) -> CIE L*a*b* under D65 (IEC 61966-2-1 transfer, CIE 1976 Lab)."""

    def lin(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in rgb)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > (6 / 29) ** 3 else t / (3 * (6 / 29) ** 2) + 4 / 29

    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def ciede2000(lab1: tuple[float, float, float], lab2: tuple[float, float, float]) -> float:
    """CIEDE2000 colour difference, kL = kC = kH = 1 (Sharma, Wu and Dalal 2005, eqs. 1-22)."""
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    cbar = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(cbar**7 / (cbar**7 + 25**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if (a1p or b1) else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if (a2p or b2) else 0.0
    dlp = l2 - l1
    dcp = c2p - c1p
    if c1p * c2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360
    dhp_big = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dhp / 2))
    lbarp = (l1 + l2) / 2
    cbarp = (c1p + c2p) / 2
    if c1p * c2p == 0:
        hbarp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbarp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbarp = (h1p + h2p + 360) / 2
    else:
        hbarp = (h1p + h2p - 360) / 2
    t = (
        1
        - 0.17 * math.cos(math.radians(hbarp - 30))
        + 0.24 * math.cos(math.radians(2 * hbarp))
        + 0.32 * math.cos(math.radians(3 * hbarp + 6))
        - 0.20 * math.cos(math.radians(4 * hbarp - 63))
    )
    dtheta = 30 * math.exp(-(((hbarp - 275) / 25) ** 2))
    rc = 2 * math.sqrt(cbarp**7 / (cbarp**7 + 25**7))
    sl = 1 + (0.015 * (lbarp - 50) ** 2) / math.sqrt(20 + (lbarp - 50) ** 2)
    sc = 1 + 0.045 * cbarp
    sh = 1 + 0.015 * cbarp * t
    rt = -math.sin(math.radians(2 * dtheta)) * rc
    return math.sqrt(
        (dlp / sl) ** 2 + (dcp / sc) ** 2 + (dhp_big / sh) ** 2 + rt * (dcp / sc) * (dhp_big / sh)
    )


def delta_e(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ciede2000(_srgb_to_lab(a), _srgb_to_lab(b))


def hex_rgb(value: str) -> tuple[float, float, float]:
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    return (float(int(v[0:2], 16)), float(int(v[2:4], 16)), float(int(v[4:6], 16)))


def over(colour: str, backdrop: tuple[float, float, float]) -> tuple[float, float, float]:
    """A computed CSS colour string, composited source-over on an opaque backdrop."""
    r, g, b, a = parse_css_color(colour)
    a = max(0.0, min(1.0, a))
    return (
        r * a + backdrop[0] * (1 - a),
        g * a + backdrop[1] * (1 - a),
        b * a + backdrop[2] * (1 - a),
    )


def hexc(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c))):02x}" for c in rgb)


@pytest.mark.parametrize(
    ("lab1", "lab2", "expected"),
    [
        # Sharma, Wu and Dalal (2005), Table 1: pairs 1, 7, 17 and 25.
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
        ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ],
)
def test_ciede2000_matches_the_published_test_pairs(lab1, lab2, expected):
    """The oracle every colour row leans on, against the formula paper's own worked pairs."""
    assert ciede2000(lab1, lab2) == pytest.approx(expected, abs=1e-4)
    assert ciede2000(lab2, lab1) == pytest.approx(expected, abs=1e-4), "symmetric"


def test_srgb_to_lab_matches_the_white_point_and_a_known_grey():
    """White is L*=100, a*=b*=0 under D65; sRGB #777777 is L* 50.03 (hand: Y = 0.18447)."""
    lab = _srgb_to_lab((255.0, 255.0, 255.0))
    assert lab[0] == pytest.approx(100.0, abs=0.01)
    assert abs(lab[1]) < 0.01 and abs(lab[2]) < 0.01
    assert _srgb_to_lab((0x77, 0x77, 0x77))[0] == pytest.approx(50.03, abs=0.05)


# --- a tiny PNG decoder (the 1x1 probes; stdlib only) -----------------------------------------


def decode_png(data: bytes) -> tuple[int, int, list[list[tuple[int, int, int]]]]:
    """8-bit RGB/RGBA, non-interlaced PNG -> (width, height, rows of RGB). Enough for CDP clips."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos, idat, width, height, ctype = 8, b"", 0, 0, 0
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, ctype, _c, _f, interlace = struct.unpack(">IIBBBBB", body)
            assert depth == 8 and ctype in (2, 6) and interlace == 0, (depth, ctype, interlace)
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    raw = zlib.decompress(idat)
    bpp = 4 if ctype == 6 else 3
    stride = width * bpp
    rows: list[list[tuple[int, int, int]]] = []
    prev = bytearray(stride)
    i = 0
    for _ in range(height):
        ftype = raw[i]
        line = bytearray(raw[i + 1 : i + 1 + stride])
        i += 1 + stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            if ftype == 1:
                line[x] = (line[x] + a) & 0xFF
            elif ftype == 2:
                line[x] = (line[x] + b) & 0xFF
            elif ftype == 3:
                line[x] = (line[x] + (a + b) // 2) & 0xFF
            elif ftype == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[x] = (line[x] + pred) & 0xFF
        rows.append([(line[x], line[x + 1], line[x + 2]) for x in range(0, stride, bpp)])
        prev = line
    return width, height, rows


def test_the_png_decoder_reads_a_hand_built_image_with_every_filter():
    """Anti-vacuity for the probe decoder: a 3x5 RGB image, one row per filter type 0-4, built here
    by hand (the filtered bytes computed from the known pixels), decodes back to the pixels."""
    px = [[(10 * r + x, 200 - 7 * x, (r * 31 + x * 17) % 256) for x in range(3)] for r in range(5)]
    raw = bytearray()
    prev = [0] * 9
    for r, row in enumerate(px):
        flat = [c for p in row for c in p]
        out = []
        for x, v in enumerate(flat):
            a = flat[x - 3] if x >= 3 else 0
            b = prev[x]
            c = prev[x - 3] if x >= 3 else 0
            if r == 0:
                pred = 0
            elif r == 1:
                pred = a
            elif r == 2:
                pred = b
            elif r == 3:
                pred = (a + b) // 2
            else:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
            out.append((v - pred) & 0xFF)
        raw += bytes([r]) + bytes(out)
        prev = flat

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 5, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )
    w, h, rows = decode_png(png)
    assert (w, h) == (3, 5)
    assert rows == px


# --- A1: the reference is committed and kept apart from the page --------------------------------


def _manifest_hashes() -> dict[str, str]:
    """``file -> sha256`` from the MANIFEST rows that name ``mockup_20261021/<file>``."""
    text = MANIFEST.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.search(r"mockup_20261021/([A-Za-z0-9_.-]+)`?.*?\b([0-9a-f]{64})\b", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def test_the_mockup_is_committed_byte_for_byte_as_the_manifest_records():
    """A1: the seven files, their SHA-256 equal to the MANIFEST's, with the source and the link."""
    hashes = _manifest_hashes()
    assert sorted(hashes) == sorted(MOCKUP_FILES), (
        f"A1: MANIFEST.md records a SHA-256 for exactly {sorted(MOCKUP_FILES)}; it records "
        f"{sorted(hashes)}"
    )
    for name in MOCKUP_FILES:
        path = MOCKUP / name
        assert path.is_file(), f"A1: {path} is committed"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == hashes[name], f"A1: {name} is {digest}, the MANIFEST says {hashes[name]}"
    text = MANIFEST.read_text(encoding="utf-8")
    assert ARTIFACT_LINK in text, "A1: the MANIFEST entry records the claude.ai artifact link"
    assert "scratchpad/mock/" in text, "A1: the MANIFEST entry records the source directory"


def test_nothing_under_hud_references_the_mockup_and_the_hud_server_does_not_serve_it():
    """A1 (negative): no file under hud/ names the fixture, and ``astra hud``'s server refuses it."""
    offenders = []
    for path in HUD.rglob("*"):
        if not path.is_file() or "node_modules" in path.parts:
            continue
        if path.suffix not in {".html", ".css", ".js", ".mjs", ".json", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "mockup_20261021" in text or "tests/fixtures" in text:
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == [], f"A1: these hud/ files reference the mockup fixture: {offenders}"
    server = HudStaticServer(HUD, 0)
    try:
        for probe in (
            "/tests/fixtures/hud/mockup_20261021/hud.html",
            "/../tests/fixtures/hud/mockup_20261021/hud.html",
            "/mockup_20261021/hud.html",
            "/%2e%2e/tests/fixtures/hud/mockup_20261021/tokens.css",
        ):
            status = None
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.port}{probe}", timeout=5
                ) as r:
                    status = r.status
                    body = r.read()
            except urllib.error.HTTPError as exc:
                status = exc.code
                body = b""
            assert status != 200 or b"local_astra HUD (mockup)" not in body, (
                f"A1: HudStaticServer served the mockup at {probe}"
            )
    finally:
        server.close()


def test_the_header_keeps_the_class_the_mutant_names():
    """The gradient-under-header mutant (``.bar { background: var(--bg) }``) is meaningful only if
    the header element still carries ``class="bar"``; a rename would make the proof vacuous."""
    shell = re.sub(r"<!--[\s\S]*?-->", " ", (HUD / "index.html").read_text(encoding="utf-8"))
    m = re.search(r"<header\b[^>]*\bclass=\"([^\"]*)\"", shell)
    assert m and "bar" in m.group(1).split(), (
        "the page's <header> carries class 'bar' (the T-0154 mutant harness targets `.bar`)"
    )


# --- the role table (P1) -------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    role: str
    mock: str
    hud: str
    prop: str
    tol: float
    deviation: str | None = None
    #: which viewport the row is measured at
    view: str = "1440x900"
    #: the mockup element whose painted colour is the backdrop for alpha colours
    backdrop: str = "body"
    note: str = ""


M_BODY, H_BODY = "body", "body"
M_HEADER, H_HEADER = "body > .app > header", "header"
M_BTN, H_BTN = ".controls > .btn:nth-child(3)", "#share-button"
H_MUTE = "#mute-button"
M_PILL, H_PILL = ".brand > .status", "#turn-chip"
M_RAIL = ".rail"
M_PANE, H_PANE = ".rail > .panel:nth-child(1)", "main > .pane.thumb"
M_PANE2, H_PANE2 = ".rail > .panel:nth-child(2)", "main > .pane-thinking"
#: **Amended by T-0162 B-9** (named spec change, James 2026-10-26): the Tools pane replaces the
#: Context pane in the rail, so ``main > #assist-panel`` becomes ``main > #tools-panel``; nothing
#: else in the selector moves.
H_RAIL_ALL = (
    "main > .pane.thumb, main > .pane-thinking, main > #tools-panel, main > #log-panel, "
    "main > .pane-diagnostics"
)
M_HEADING = ".rail > .panel:nth-child(1) h3"

L, F, C = LENGTH_TOL_PX, FONT_SIZE_TOL_PX, COLOUR_TOL_DE

P1: list[Row] = [
    Row("page background: a gradient is painted", M_BODY, H_BODY, "gradient", 0),
    Row("page background: colour", M_BODY, H_BODY, "color:background", C, "bg"),
    Row("body text: 15 px", M_BODY, H_BODY, "font_size", F),
    Row("body text: colour", M_BODY, H_BODY, "color:text", C),
    Row("app frame: header left", M_HEADER, H_HEADER, "content_x", L),
    Row("app frame: header top", M_HEADER, H_HEADER, "content_y", L),
    Row("app frame: header width", M_HEADER, H_HEADER, "content_w", L),
    Row(
        "app frame: header left at 1920",
        M_HEADER,
        H_HEADER,
        "content_x",
        L,
        view="1920x1080",
        note="max-width 1480, centred",
    ),
    Row("app frame: header width at 1920", M_HEADER, H_HEADER, "content_w", L, view="1920x1080"),
    Row("header: height", M_HEADER, H_HEADER, "content_h", L),
    Row("header: transparent", M_HEADER, H_HEADER, "transparent", 0),
    Row("rail: right edge", M_RAIL, H_RAIL_ALL, "right", L),
    Row("rail: right edge at 1920", M_RAIL, H_RAIL_ALL, "right", L, view="1920x1080"),
    Row("rail: top (18 below the header)", M_RAIL, H_PANE, "y", L),
    Row("rail: width 380", M_RAIL, H_RAIL_ALL, "w", L),
    Row("rail: gap 18 between panes", f"{M_PANE}|{M_PANE2}", f"{H_PANE}|{H_PANE2}", "gap", L),
    Row("rail pane: radius 22", M_PANE, H_RAIL_ALL, "radius", L),
    Row("rail pane: 1 px border", M_PANE, H_RAIL_ALL, "border_w", L),
    Row("rail pane: line border colour", M_PANE, H_RAIL_ALL, "color:border", C, backdrop=M_PANE),
    Row("rail pane: shadow", M_PANE, H_RAIL_ALL, "shadow", L, backdrop=M_BODY),
    Row("rail pane: background", M_PANE, H_RAIL_ALL, "color:background", C, "surface"),
    Row("rail pane: padding top 16", M_PANE, H_RAIL_ALL, "inset_top", L),
    Row("rail pane: padding left 18", M_PANE, H_RAIL_ALL, "inset_left", L),
    Row("rail heading: 13 px", M_HEADING, "@heading", "font_size", F),
    Row("rail heading: weight 650", M_HEADING, "@heading", "font_weight", 0),
    Row("rail heading: colour", M_HEADING, "@heading", "color:text", C, backdrop=M_PANE),
    Row("header button: 36 px", M_BTN, H_BTN, "h", L),
    Row("header button: radius 12", M_BTN, H_BTN, "radius", L),
    Row("header button: 1 px border", M_BTN, H_BTN, "border_w", L),
    Row("header button: line border colour", M_BTN, H_BTN, "color:border", C, backdrop=M_BTN),
    Row("header button: shadow", M_BTN, H_BTN, "shadow", L),
    Row("header button: fill", M_BTN, H_BTN, "color:background", C, "surface"),
    Row("Mute button: 36 px", M_BTN, H_MUTE, "h", L),
    Row("Mute button: radius 12", M_BTN, H_MUTE, "radius", L),
    Row("Mute button: 1 px border", M_BTN, H_MUTE, "border_w", L),
    Row("state pill: radius", M_PILL, H_PILL, "radius", L),
    Row("state pill: padding top", M_PILL, H_PILL, "pad_t", L),
    Row("state pill: padding right", M_PILL, H_PILL, "pad_r", L),
    Row("state pill: padding bottom", M_PILL, H_PILL, "pad_b", L),
    Row("state pill: padding left", M_PILL, H_PILL, "pad_l", L),
    Row("state pill: soft fill", M_PILL, H_PILL, "color:background", C),
    Row("state pill: 13.5 px", M_PILL, H_PILL, "font_size", F),
    Row("state pill: weight 600", M_PILL, H_PILL, "font_weight", 0),
    Row("state pill: text colour", M_PILL, H_PILL, "color:text", C, "accent"),
]

#: Every selector a row reads, per page; ``@heading`` is each HUD rail pane's first ``h2``.
MOCK_SELECTORS = sorted({s for r in P1 for s in r.mock.split("|")} | {M_BODY})
HUD_SELECTORS = sorted({s for r in P1 for s in r.hud.split("|") if s != "@heading"})

MEASURE_JS = """
(() => {
  const SELS = __SELS__;
  const look = (el) => {
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    const px = (v) => parseFloat(v) || 0;
    const pad = [px(s.paddingTop), px(s.paddingRight), px(s.paddingBottom), px(s.paddingLeft)];
    const bw = [px(s.borderTopWidth), px(s.borderRightWidth), px(s.borderBottomWidth), px(s.borderLeftWidth)];
    const h = el.matches('main > .pane') ? el.querySelector('h2') : (el.querySelector('h3') || el.querySelector('h2'));
    let heading = null;
    if (h) {
      const hs = getComputedStyle(h), hr = h.getBoundingClientRect();
      heading = {top: hr.top + px(hs.paddingTop) + px(hs.borderTopWidth),
                 left: hr.left + px(hs.paddingLeft) + px(hs.borderLeftWidth),
                 font_size: px(hs.fontSize), font_weight: hs.fontWeight, color: hs.color};
    }
    return {tag: el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') +
                 (typeof el.className === 'string' && el.className ? '.' + el.className.trim().split(/\\s+/).join('.') : ''),
            x: r.x, y: r.y, w: r.width, h: r.height, right: r.right, bottom: r.bottom,
            content_x: r.x + bw[3] + pad[3], content_y: r.y + bw[0] + pad[0],
            content_w: r.width - bw[1] - bw[3] - pad[1] - pad[3],
            content_h: r.height - bw[0] - bw[2] - pad[0] - pad[2],
            pad_t: pad[0], pad_r: pad[1], pad_b: pad[2], pad_l: pad[3],
            border_w: bw[0], border_color: s.borderTopColor, border_style: s.borderTopStyle,
            radius: px(s.borderTopLeftRadius), shadow: s.boxShadow,
            background: s.backgroundColor, background_image: s.backgroundImage,
            color: s.color, font_size: px(s.fontSize), font_weight: s.fontWeight,
            text: (el.innerText || '').trim().slice(0, 40), heading};
  };
  const out = {};
  for (const sel of SELS) out[sel] = [...document.querySelectorAll(sel)].map(look);
  const root = getComputedStyle(document.documentElement);
  out.__tokens__ = Object.fromEntries(['--bg', '--surface', '--muted', '--faint', '--accent', '--green',
    '--amber', '--red', '--line'].map((t) => [t, root.getPropertyValue(t).trim()]));
  out.__theme__ = document.documentElement.getAttribute('data-theme');
  out.__viewport__ = [innerWidth, innerHeight];
  return out;
})()
"""


def measure_js(selectors: list[str]) -> str:
    return MEASURE_JS.replace("__SELS__", json.dumps(selectors))


#: The A3 probe points, computed in the page from its own rects so that the same *role* is probed
#: on both pages; the page-level probes are fixed coordinates. ``owner`` is what
#: ``elementFromPoint`` returns there, so a probe that lands on content is reported, not trusted.
PROBES_JS = """
(() => {
  const P = __P__;
  const q = (s) => document.querySelector(s);
  const pane = q(P.pane), btn = q(P.btn), header = q(P.header);
  const pt = (x, y) => ({x: Math.floor(x), y: Math.floor(y)});
  const r = (el) => el ? el.getBoundingClientRect() : null;
  const pr = r(pane), br = r(btn), hr = r(header);
  const probes = {
    'page corner': pt(3, 3),
    'gradient top-left': pt(420, hr ? hr.top + hr.height / 2 : 36),
    // "bottom-centre" is read as the bottom of the page background between the two columns (the
    // gutter left of the rail, 6 px from the window's bottom): at 1440x900 the mockup's
    // conversation panel runs past the window's bottom edge, so the literal centre is a panel.
    'bottom-centre': pt(pr ? pr.left - 6 : innerWidth / 2, innerHeight - 6),
  };
  if (pr) {
    probes['rail pane interior'] = pt(pr.left + 9, pr.top + pr.height / 2);
    probes['rail pane edge'] = pt(pr.left, pr.top + pr.height / 2);
  }
  if (br) probes['header button interior'] = pt(br.left + 4, br.top + br.height / 2);
  const owner = {};
  for (const [k, p] of Object.entries(probes)) {
    const el = document.elementFromPoint(p.x + 0.5, p.y + 0.5);
    const s = el ? getComputedStyle(el) : null;
    owner[k] = el ? {tag: el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') +
                           (typeof el.className === 'string' && el.className ? '.' + el.className.trim().split(/\\s+/).join('.') : ''),
                     is_header: !!header && el === header,
                     in_header: !!header && header.contains(el),
                     empty: !(el.textContent || '').trim(),
                     transparent: s.backgroundColor === 'rgba(0, 0, 0, 0)' && s.backgroundImage === 'none',
                     // the header itself, or an empty transparent child of it (a flex spacer)
                     bare_header: !!header && (el === header || (header.contains(el) &&
                       !(el.textContent || '').trim() && s.backgroundColor === 'rgba(0, 0, 0, 0)' &&
                       s.backgroundImage === 'none')),
                     page: el === document.body || el === document.documentElement || el.tagName === 'MAIN' ||
                           // a transparent layout container hit directly (no child at the point)
                           (!el.closest('.pane, .panel, header') && s.backgroundColor === 'rgba(0, 0, 0, 0)' &&
                            s.backgroundImage === 'none')} : null;
  }
  return {probes, owner, pane_rect: pr ? [pr.left, pr.top, pr.width, pr.height] : null};
})()
"""


def probes_js(pane: str, btn: str, header: str) -> str:
    return PROBES_JS.replace("__P__", json.dumps({"pane": pane, "btn": btn, "header": header}))


def _pixel(chrome: _Recorder, x: int, y: int) -> tuple[float, float, float]:
    """The pixel at *viewport* point ``(x, y)`` (what ``getBoundingClientRect``/``elementFromPoint`` use).

    A ``Page.captureScreenshot`` clip is in *document* coordinates, so the window's scroll offsets are
    added here (T-0154 B, G2: measured on Chrome with a 1000 px red / 2000 px blue page scrolled to 900,
    viewport y 150 is blue on screen; a clip at y 150 read white and one at y 1050 read blue). Without
    this every probe on a scrolled document reads the wrong place.
    """
    off = chrome.evaluate("[window.scrollX, window.scrollY]") or [0, 0]
    shot = chrome._cmd(
        "Page.captureScreenshot",
        format="png",
        clip={"x": x + off[0], "y": y + off[1], "width": 1, "height": 1, "scale": 1},
    )
    import base64

    _w, _h, rows = decode_png(base64.b64decode(shot["data"]))
    p = rows[0][0]
    return (float(p[0]), float(p[1]), float(p[2]))


@dataclass
class PageMeasure:
    styles: dict[str, Any] = field(default_factory=dict)
    wide: dict[str, Any] = field(default_factory=dict)
    probes: dict[str, Any] = field(default_factory=dict)
    pixels: dict[str, tuple[float, float, float]] = field(default_factory=dict)


@dataclass
class Parity:
    error: str = ""
    seconds: float = 0.0
    mock: dict[str, PageMeasure] = field(default_factory=dict)
    hud: dict[str, PageMeasure] = field(default_factory=dict)
    hud_theme: dict[str, Any] = field(default_factory=dict)
    card_rendered: bool = False
    chip: str = ""
    requests: list[str] = field(default_factory=list)


def _serve_dir(root: Path) -> tuple[socketserver.TCPServer, int]:
    handler = functools.partial(_QuietHandler, directory=str(root))
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, int(server.server_address[1])


def _media(chrome: _Recorder, scheme: str) -> None:
    chrome.media(
        [
            {"name": "prefers-reduced-motion", "value": "reduce"},
            {"name": "prefers-color-scheme", "value": scheme},
        ]
    )


def _settle(chrome: _Recorder) -> None:
    chrome.evaluate(
        "document.fonts.ready.then(() => new Promise((r) => "
        "requestAnimationFrame(() => requestAnimationFrame(() => r(true)))))"
    )


def _measure_page(
    chrome: _Recorder, selectors: list[str], pane: str, btn: str, header: str
) -> PageMeasure:
    m = PageMeasure()
    chrome.viewport(*WIDE)
    _settle(chrome)
    m.wide = dict(chrome.evaluate(measure_js(selectors)) or {})
    chrome.viewport(*VIEW)
    _settle(chrome)
    m.styles = dict(chrome.evaluate(measure_js(selectors)) or {})
    m.probes = dict(chrome.evaluate(probes_js(pane, btn, header)) or {})
    for name, p in (m.probes.get("probes") or {}).items():
        m.pixels[name] = _pixel(chrome, int(p["x"]), int(p["y"]))
    return m


def _run(result: Parity, browser: _Recorder | str) -> None:
    if isinstance(browser, str):
        raise RuntimeError(browser)
    mock_server, mock_port = _serve_dir(MOCKUP)
    lines = fixture_lines()
    t9_end = _last_index_of(CARD_TURN)
    surfaces = card_surfaces(gated_card())
    stub = HudStub([]).start()
    hud_server = HudStaticServer(HUD, 0)
    try:
        chrome = browser
        chrome.reset()
        chrome._cmd("Runtime.enable")
        chrome._cmd("Network.enable")
        for theme in THEMES:
            _media(chrome, theme)
            chrome.viewport(*VIEW)
            chrome.navigate(f"http://127.0.0.1:{mock_port}/hud.html?theme={theme}")
            chrome.wait_until("document.readyState === 'complete'", 15)
            result.mock[theme] = _measure_page(chrome, MOCK_SELECTORS, M_PANE, M_BTN, M_HEADER)

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
            chrome.wait_until(f"!!document.querySelector('{A2UI_ROOT}')", 10)  # T-0164 B2
        )
        chrome.wait_until(
            "(document.querySelector('[data-turn-state]') || {}).textContent === 'SPEAKING'", 5
        )
        result.chip = str(
            chrome.evaluate("(document.querySelector('[data-turn-state]') || {}).textContent")
        )
        for theme in THEMES:
            result.hud_theme[theme] = chrome.evaluate(choose_theme_js(theme)) or {}
            _media(chrome, theme)
            result.hud[theme] = _measure_page(chrome, HUD_SELECTORS, H_PANE, H_BTN, H_HEADER)
        chrome.evaluate(choose_theme_js("system"))
        chrome.drain(0.3)
        result.requests = list(chrome.requests)
    finally:
        hud_server.close()
        stub.stop()
        mock_server.shutdown()
        mock_server.server_close()


@pytest.fixture(scope="module")
def browser() -> Any:
    """The module's one headless Chrome (T-0154 Tests to write: one Chrome for the module)."""
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


@pytest.fixture(scope="module")
def parity(browser: Any) -> Iterator[Parity]:
    result = Parity()
    if CHROME is None:  # pragma: no cover
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


def _ok(p: Parity) -> None:
    assert not p.error, f"the parity session failed before any assertion ran: {p.error}"


# --- evaluating a row ---------------------------------------------------------------------------


def _token_rgb(tokens: dict[str, str], name: str) -> tuple[float, float, float] | None:
    value = (tokens or {}).get(f"--{name}") or ""
    if value.startswith("#"):
        return hex_rgb(value)
    if value.startswith("rgb"):
        r, g, b, _a = parse_css_color(value)
        return (r, g, b)
    return None


def _backdrop(mock: dict[str, Any], sel: str) -> tuple[float, float, float]:
    """The mockup's opaque colour behind ``sel``: its own background if opaque, else the page's."""
    tokens = mock.get("__tokens__") or {}
    page = _token_rgb(tokens, "bg") or (255.0, 255.0, 255.0)
    els = mock.get(sel) or []
    if els:
        colour = parse_css_color(str(els[0]["background"]))
        if colour[3] > 0:
            return over(str(els[0]["background"]), page)
    return page


_LENGTH = re.compile(r"(-?\d*\.?\d+)px")
_COLOUR = re.compile(r"rgba?\([^)]*\)")


def _shadow_layers(value: str) -> list[dict[str, Any]]:
    if not value or value == "none":
        return []
    layers = []
    depth, start = 0, 0
    parts = []
    for i, ch in enumerate(value):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(value[start:i])
            start = i + 1
    parts.append(value[start:])
    for part in parts:
        colour = _COLOUR.search(part)
        layers.append(
            {
                "colour": colour.group(0) if colour else "rgb(0, 0, 0)",
                "lengths": [float(x) for x in _LENGTH.findall(_COLOUR.sub("", part))],
                "inset": "inset" in part,
            }
        )
    return layers


def _compare_shadow(mock: str, hud: str, backdrop: tuple[float, float, float]) -> str | None:
    ml, hl = _shadow_layers(mock), _shadow_layers(hud)
    if len(ml) != len(hl):
        return f"{len(hl)} shadow layer(s) {hud!r}, the mockup has {len(ml)} {mock!r}"
    for i, (a, b) in enumerate(zip(ml, hl, strict=True)):
        if a["inset"] != b["inset"]:
            return f"layer {i}: inset {b['inset']} vs the mockup's {a['inset']}"
        la = (a["lengths"] + [0.0] * 4)[:4]
        lb = (b["lengths"] + [0.0] * 4)[:4]
        if any(abs(x - y) > LENGTH_TOL_PX for x, y in zip(la, lb, strict=True)):
            return f"layer {i}: offsets/blur/spread {lb} vs the mockup's {la}"
        de = delta_e(over(a["colour"], backdrop), over(b["colour"], backdrop))
        if de > COLOUR_TOL_DE:
            return f"layer {i}: colour {b['colour']} vs {a['colour']}, dE00 {de:.2f} over {hexc(backdrop)}"
    return None


def _value(el: dict[str, Any], prop: str) -> Any:
    if prop in ("font_size", "font_weight", "color:text") and el.get("__heading__"):
        h = el.get("heading") or {}
        return h.get({"color:text": "color"}.get(prop, prop))
    return el.get(prop)


def evaluate_row(row: Row, mock: dict[str, Any], hud: dict[str, Any], theme: str) -> list[str]:
    """Problems with one row (empty: the row holds)."""
    if row.prop == "gap":
        m1, m2 = row.mock.split("|")
        h1, h2 = row.hud.split("|")
        if not (mock.get(m1) and mock.get(m2)):
            return [f"mockup {m1} / {m2} matched nothing"]
        if not (hud.get(h1) and hud.get(h2)):
            return [f"HUD {h1} / {h2} matched nothing"]
        want = mock[m2][0]["y"] - mock[m1][0]["bottom"]
        got = hud[h2][0]["y"] - hud[h1][0]["bottom"]
        if abs(got - want) > row.tol:
            return [f"gap {got:.1f} px, the mockup's is {want:.1f} px (+/- {row.tol})"]
        return []

    mels = mock.get(row.mock) or []
    if not mels:
        return [f"mockup selector {row.mock!r} matched nothing (the reference is missing)"]
    ref = dict(mels[0])
    if row.hud == "@heading":
        ref = {**ref, "__heading__": False}
        hels = [
            {"__heading__": True, "heading": e.get("heading"), "tag": e["tag"] + " h2"}
            for e in hud.get(H_RAIL_ALL) or []
        ]
        if not hels or any(not e["heading"] for e in hels):
            return [f"a HUD rail pane has no h2: {[e['tag'] for e in hels]}"]
        ref_value = {
            "font_size": ref["font_size"],
            "font_weight": ref["font_weight"],
            "color:text": ref["color"],
        }[row.prop]
    else:
        hels = hud.get(row.hud) or []
        ref_value = _value(ref, row.prop) if not row.prop.startswith("color:") else None
    if not hels:
        return [f"HUD selector {row.hud!r} matched nothing"]

    problems: list[str] = []
    backdrop = _backdrop(mock, row.backdrop)
    for el in hels:
        where = el.get("tag", row.hud)
        if row.prop == "gradient":
            for name, image in (
                ("mockup", ref["background_image"]),
                ("HUD", el["background_image"]),
            ):
                if "gradient" not in str(image):
                    problems.append(
                        f"{name} {where}: background-image {image!r} paints no gradient"
                    )
        elif row.prop == "transparent":
            for name, e in (("mockup", ref), ("HUD", el)):
                alpha = parse_css_color(str(e["background"]))[3]
                if alpha > 0.01 or str(e["background_image"]) != "none":
                    problems.append(
                        f"{name} {e['tag']}: background {e['background']} / {e['background_image']} "
                        "is not transparent (the page gradient must show under the header)"
                    )
        elif row.prop == "shadow":
            if str(ref["shadow"]) == "none":
                problems.append("the mockup's element has no shadow: the row is vacuous")
            why = _compare_shadow(str(ref["shadow"]), str(el["shadow"]), backdrop)
            if why:
                problems.append(f"{where}: {why}")
        elif row.prop.startswith("color:"):
            key = {
                "color:background": "background",
                "color:border": "border_color",
                "color:text": "color",
            }[row.prop]
            mine = (el.get("heading") or {}).get("color") if row.hud == "@heading" else el[key]
            ref_colour = ref[key]
            got = over(str(mine), backdrop)
            if row.deviation and (row.deviation, theme) in T0128_DEVIATIONS:
                target = hex_rgb(T0128_DEVIATIONS[(row.deviation, theme)])
                label = f"T-0128 --{row.deviation} {T0128_DEVIATIONS[(row.deviation, theme)]} (named deviation)"
            else:
                target = over(str(ref_colour), backdrop)
                label = f"the mockup's {ref_colour}"
            de = delta_e(got, target)
            if de > row.tol:
                problems.append(
                    f"{where}: {key} {mine} ({hexc(got)}) vs {label} ({hexc(target)}), "
                    f"dE00 {de:.2f} > {row.tol}"
                )
            if (
                key == "border_color"
                and float(el["border_w"]) > 0
                and str(el.get("border_style")) == "none"
            ):
                problems.append(f"{where}: border-style none")
        elif row.prop == "font_weight":
            got = _value(el, row.prop)
            if str(got) != str(ref_value):
                problems.append(f"{where}: font-weight {got} vs the mockup's {ref_value} (exact)")
        elif row.prop in ("inset_top", "inset_left"):
            h = el.get("heading")
            rh = ref.get("heading")
            if not h or not rh:
                problems.append(f"{where}: no heading to measure the inset from")
                continue
            if row.prop == "inset_top":
                got, want = h["top"] - el["y"], rh["top"] - ref["y"]
            else:
                got, want = h["left"] - el["x"], rh["left"] - ref["x"]
            if abs(got - want) > row.tol:
                problems.append(
                    f"{where}: heading inset {got:.1f} px vs the mockup's {want:.1f} px"
                )
        else:
            got = _value(el, row.prop)
            if got is None or ref_value is None:
                problems.append(f"{where}: {row.prop} not measured ({got!r} / {ref_value!r})")
            elif abs(float(got) - float(ref_value)) > row.tol:
                problems.append(
                    f"{where}: {row.prop} {float(got):.2f} vs the mockup's {float(ref_value):.2f} "
                    f"(+/- {row.tol})"
                )
    return problems


def _tables(p: Parity, theme: str, view: str) -> tuple[dict[str, Any], dict[str, Any]]:
    m, h = p.mock.get(theme), p.hud.get(theme)
    assert m is not None and h is not None, f"{theme}: a page was not measured"
    if view == "1920x1080":
        return m.wide, h.wide
    return m.styles, h.styles


# --- the premise: the mockup was really loaded, in the theme asked for ------------------------------


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_the_mockup_renders_what_its_own_css_says(parity, theme):
    """Premise of every row (anti-vacuity, and A3's "holds in the mockup" halves).

    Hand-read from the committed ``hud.html``/``tokens.css``: header buttons 36 px with radius 12;
    rail 380 px; rail panels radius 22 with a 1 px border; rail heading 13 px/650; body 15 px; pill
    13.5 px/600; a gradient on body; a transparent header. Then A3's two checks on the mockup's own
    pixels: the top-left probe against the bottom-centre probe is dE00 >= 2 (the gradient shows),
    and the pane edge against the pane interior is dE00 >= 1 (the border shows).
    """
    _ok(parity)
    m = parity.mock[theme]
    s = m.styles
    assert s.get("__theme__") == theme, (
        f"the mockup did not take ?theme={theme}: {s.get('__theme__')}"
    )
    assert s.get("__viewport__") == list(VIEW), s.get("__viewport__")
    btn, rail, pane = s[M_BTN][0], s[M_RAIL][0], s[M_PANE][0]
    heading, body, pill = s[M_HEADING][0], s[M_BODY][0], s[M_PILL][0]
    assert btn["text"] == "Mute", f"premise: the plain header button is the mockup's Mute: {btn}"
    assert (btn["h"], btn["radius"], btn["border_w"]) == (36, 12, 1), btn
    assert rail["w"] == 380 and pane["radius"] == 22 and pane["border_w"] == 1, (rail, pane)
    assert (heading["font_size"], heading["font_weight"]) == (13, "650"), heading
    assert body["font_size"] == 15 and "gradient" in body["background_image"], body
    assert (pill["font_size"], pill["font_weight"]) == (13.5, "600"), pill
    header = s[M_HEADER][0]
    assert parse_css_color(header["background"])[3] == 0, header
    owner = (m.probes.get("owner") or {}).get("gradient top-left") or {}
    assert owner.get("bare_header"), (
        f"premise: the mockup's gradient probe sits on the header's empty middle: {owner}"
    )
    bottom = (m.probes.get("owner") or {}).get("bottom-centre") or {}
    assert bottom.get("page"), f"premise: the mockup's bottom probe is page background: {bottom}"
    grad = delta_e(m.pixels["gradient top-left"], m.pixels["bottom-centre"])
    assert grad >= GRADIENT_MIN_DE, f"A3 in the mockup ({theme}): gradient dE00 {grad:.2f}"
    edge = delta_e(m.pixels["rail pane edge"], m.pixels["rail pane interior"])
    assert edge >= BORDER_MIN_DE, f"A3 in the mockup ({theme}): border dE00 {edge:.2f}"


@requires_chrome
def test_the_hud_session_reached_t9_with_the_card_and_the_pill_speaking(parity):
    """Premise of the HUD side: the replay reached t9, the gated card mounted, the pill reads the
    pushed SPEAKING verbatim, and both themes were chosen through the Settings control."""
    _ok(parity)
    assert parity.card_rendered, "the t9 card never mounted an <a2ui-surface>"
    assert parity.chip == "SPEAKING", f"the turn chip reads {parity.chip!r}"
    for theme in THEMES:
        chosen = parity.hud_theme.get(theme) or {}
        assert chosen.get("ok") and chosen.get("theme") == theme, chosen
        assert parity.hud[theme].styles.get("__theme__") == theme


# --- A2: every P1 row ---------------------------------------------------------------------------------


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_a2_every_frame_role_matches_the_mockup(parity, theme):
    """A2: the P1 role table, measured live from the mockup, both themes; every failing row listed."""
    _ok(parity)
    failures: list[str] = []
    for row in P1:
        mock, hud = _tables(parity, theme, row.view)
        for problem in evaluate_row(row, mock, hud, theme):
            dev = f" [deviation: --{row.deviation}]" if row.deviation else ""
            failures.append(f"{row.role} @ {row.view}{dev}: {problem}")
    assert failures == [], (
        f"T-0154 A2 (data-theme={theme}): {len(failures)} frame role(s) differ from the mockup:\n  "
        + "\n  ".join(failures)
    )


# --- A3: pixels ---------------------------------------------------------------------------------------

#: probe -> the T-0128 token the painted colour is built on (A3: "the named deviations of A2 allowed
#: up to their own token dE + 1").
PROBE_DEVIATION = {
    "page corner": "bg",
    "gradient top-left": "bg",
    "bottom-centre": "bg",
    "rail pane interior": "surface",
    "rail pane edge": "surface",
    "header button interior": "surface",
}


def _probe_tolerance(mock: PageMeasure, probe: str, theme: str) -> tuple[float, str]:
    token = PROBE_DEVIATION.get(probe)
    if token and (token, theme) in T0128_DEVIATIONS:
        ref = _token_rgb(mock.styles.get("__tokens__") or {}, token)
        assert ref is not None, f"the mockup's --{token} was not measured"
        tde = delta_e(ref, hex_rgb(T0128_DEVIATIONS[(token, theme)]))
        return max(PIXEL_TOL_DE, tde + 1.0), f"--{token} deviation (token dE00 {tde:.2f} + 1)"
    return PIXEL_TOL_DE, "no deviation"


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_a3_the_probe_pixels_match_the_mockup(parity, theme):
    """A3: page corner, gradient top-left, bottom-centre, rail pane interior and edge, header button
    interior: the HUD's pixel within dE00 3 of the mockup's (or its named deviation's dE + 1)."""
    _ok(parity)
    m, h = parity.mock[theme], parity.hud[theme]
    failures = []
    for probe in PROBE_DEVIATION:
        if probe not in m.pixels:
            failures.append(f"{probe}: not probed in the mockup")
            continue
        if probe not in h.pixels:
            failures.append(f"{probe}: not probed in the HUD (its element is missing)")
            continue
        tol, why = _probe_tolerance(m, probe, theme)
        de = delta_e(m.pixels[probe], h.pixels[probe])
        if de > tol:
            failures.append(
                f"{probe} at HUD {h.probes['probes'][probe]} (on {h.probes['owner'][probe]['tag']}): "
                f"{hexc(h.pixels[probe])} vs the mockup's {hexc(m.pixels[probe])} at "
                f"{m.probes['probes'][probe]}, dE00 {de:.2f} > {tol:.2f} ({why})"
            )
    owner = h.probes.get("owner") or {}
    bottom = owner.get("bottom-centre") or {}
    if not bottom.get("page"):
        failures.append(
            f"bottom-centre lands on {bottom.get('tag')}, not the page background: the frame's "
            "bottom padding is missing"
        )
    assert failures == [], f"T-0154 A3 (data-theme={theme}):\n  " + "\n  ".join(failures)


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_a3_the_gradient_shows_under_the_header(parity, theme):
    """A3 gradient check on the HUD: the top-left probe (inside the header row, on no content) is
    dE00 >= 2 from the bottom-centre probe. Mutant ``.bar { background: var(--bg) }`` turns it red."""
    _ok(parity)
    h = parity.hud[theme]
    owner = (h.probes.get("owner") or {}).get("gradient top-left") or {}
    assert owner.get("bare_header"), (
        f"the gradient probe must sit on the header's empty middle (the header row, no content): {owner}"
    )
    bottom = (h.probes.get("owner") or {}).get("bottom-centre") or {}
    assert bottom.get("page"), (
        f"the bottom probe (the gutter left of the rail) must be page background: {bottom}"
    )
    de = delta_e(h.pixels["gradient top-left"], h.pixels["bottom-centre"])
    assert de >= GRADIENT_MIN_DE, (
        f"T-0154 A3 (data-theme={theme}): under the header {hexc(h.pixels['gradient top-left'])} "
        f"vs bottom-centre {hexc(h.pixels['bottom-centre'])} is dE00 {de:.2f} < {GRADIENT_MIN_DE}: "
        "the page's background gradient does not show under the header"
    )


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_a3_the_rail_pane_edge_is_visible_against_its_interior(parity, theme):
    """A3 border check on the HUD (supersedes T-0056 A12's borderless panes by name): the rail pane's
    edge pixel is dE00 >= 1 from its interior. Mutant ``.pane { border-color: transparent }`` turns
    it red."""
    _ok(parity)
    h = parity.hud[theme]
    assert "rail pane edge" in h.pixels, f"no rail pane was probed: {h.probes}"
    de = delta_e(h.pixels["rail pane edge"], h.pixels["rail pane interior"])
    assert de >= BORDER_MIN_DE, (
        f"T-0154 A3 (data-theme={theme}): pane edge {hexc(h.pixels['rail pane edge'])} vs interior "
        f"{hexc(h.pixels['rail pane interior'])} is dE00 {de:.2f} < {BORDER_MIN_DE}: no visible border"
    )


@requires_chrome
def test_the_parity_session_requests_nothing_off_loopback(parity):
    """T-0128 criterion 2 kept (A7): both pages, every request on 127.0.0.1 (the mockup's Inter is a
    font-family name, never a download)."""
    _ok(parity)
    assert parity.requests, "the request log is empty, so this row would be vacuous"
    offenders = [
        u
        for u in parity.requests
        if urllib.parse.urlsplit(u).scheme in {"http", "https", "ws", "wss"}
        and urllib.parse.urlsplit(u).hostname not in {"127.0.0.1", "localhost", "::1"}
    ]
    assert offenders == [], f"non-loopback requests: {offenders}"


# --- A8: the side-by-side evidence (Playwright) ----------------------------------------------------


def _playwright_chromium() -> str | None:
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


def test_a8_the_screenshot_script_writes_the_composites_fonts_and_loopback_requests(tmp_path):
    """A8: ``hud_screens.py --out DIR --compare tests/fixtures/hud/mockup_20261021/`` writes the 8 HUD
    PNGs, ``compare-{dark,light}.png`` (mockup left, HUD right, each 1440x900), ``requests.txt``
    (loopback only) and ``fonts.txt`` (``fc-match`` for ``system-ui`` and ``sans-serif``)."""
    why = _playwright_chromium()
    if why:
        pytest.skip(why)
    script = REPO / "scripts" / "demo" / "hud_screens.py"
    out = tmp_path / "shots"
    proc = subprocess.run(
        [sys.executable, str(script), "--out", str(out), "--compare", str(MOCKUP)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"hud_screens.py --compare failed:\n{proc.stdout}\n{proc.stderr}"
    pngs = sorted(p.name for p in out.glob("*.png"))
    # Amended by T-0154 C6 (G2): the script also writes compare-cards-{dark,light}.png (the card
    # composites, required and sized by tests/integration/test_hud_card_parity.py's C6 row). The
    # thread composites below are still exactly these two; the card ones are the only other
    # compare-* names allowed.
    # Amended by T-0164 C5 (docs/tasks/T-0164.md, amendment 2026-10-27): the script also writes
    # compare-genui-{dark,light}.png (sized by test_hud_card_parity.py's C5 row) and may keep its
    # per-card crops in one subdirectory, genui-cards/. Exactly those are admitted, nothing else.
    composites = [
        n
        for n in pngs
        if n.startswith("compare-")
        and not n.startswith("compare-cards-")
        and not n.startswith("compare-genui-")
    ]
    assert sorted(composites) == ["compare-dark.png", "compare-light.png"], pngs
    card_composites = {n for n in pngs if n.startswith("compare-cards-")}
    assert card_composites <= {"compare-cards-dark.png", "compare-cards-light.png"}, pngs
    genui_composites = {n for n in pngs if n.startswith("compare-genui-")}
    assert genui_composites == {"compare-genui-dark.png", "compare-genui-light.png"}, pngs
    subdirs = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert subdirs in ([], ["genui-cards"]), f"only genui-cards/ is admitted: {subdirs}"
    hud_pngs = [n for n in pngs if not n.startswith("compare-")]
    assert len(hud_pngs) == 8, f"T-0128 criterion 15's 8 HUD PNGs: {hud_pngs}"
    for name in hud_pngs:
        assert _png_size((out / name).read_bytes()) == VIEW, name
    for name in composites:
        data = (out / name).read_bytes()
        w, h = _png_size(data)
        assert w >= 2 * VIEW[0] and h >= VIEW[1], (
            f"{name} is {w}x{h}: two 1440x900 renders side by side need at least 2880x900"
        )
    requests_txt = (out / "requests.txt").read_text(encoding="utf-8")
    hosts = {
        urllib.parse.urlsplit(u).hostname
        for u in re.findall(r"[a-zA-Z][a-zA-Z0-9+.-]*://\S+", requests_txt)
        if urllib.parse.urlsplit(u).scheme not in {"data", "blob", "about"}
    }
    assert hosts == {"127.0.0.1"}, f"requests.txt hosts: {hosts}"
    fonts = (out / "fonts.txt").read_text(encoding="utf-8")
    assert "system-ui" in fonts and "sans-serif" in fonts, fonts
    expected = subprocess.run(
        ["fc-match", "sans-serif"], capture_output=True, text=True, timeout=10
    ).stdout.strip()
    assert expected and expected in fonts, (
        f"fonts.txt carries the fc-match output for sans-serif ({expected!r}): {fonts!r}"
    )
