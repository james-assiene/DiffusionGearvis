#!/usr/bin/env python3
"""Render the T-0223 handwritten maths/physics attempts to PNG pages and one PDF.

    uv run --with matplotlib==3.11.2 --with pillow==12.3.0 python scripts/demo/render_handwritten.py

Reads ``demos/handwritten/attempts.json`` and writes ``demos/handwritten/out/<id>.png`` (1700 x 2200
px, 200 dpi, US letter) and ``demos/handwritten/out/attempts.pdf`` (ten pages, M1..M5 then P1..P5).

Each page is a worksheet: a printed header and problem statement in a plain font (DejaVu Sans), then
lined paper with the attempt written in the vendored handwriting font (Patrick Hand, SIL OFL,
``demos/handwritten/fonts/``) in dark-blue ink, each line with a small seeded offset and rotation.
Glyphs the handwriting font lacks (theta, mu, parallel, ...) fall back to DejaVu Sans. Diagrams are
drawn with matplotlib's sketch path effect (the xkcd look). Fixed seed and fixed metadata: two runs
give byte-identical PNGs. No project dependency: matplotlib and pillow come from ``uv run --with``,
pinned to MATPLOTLIB_VERSION and PILLOW_VERSION (the committed ``out/`` was rendered with them; other
versions may change the bytes, and the script warns on stderr when they differ).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon, Rectangle

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "demos" / "handwritten"
SEED = 20261101
MATPLOTLIB_VERSION = "3.11.2"  # the versions the committed out/ was rendered with
PILLOW_VERSION = "12.3.0"
ORDER = ["M1", "M2", "M3", "M4", "M5", "P1", "P2", "P3", "P4", "P5"]

# page geometry (inches at DPI): US letter at 200 dpi = 1700 x 2200 px
PAGE_W, PAGE_H, DPI = 8.5, 11.0, 200
ROW = 0.040  # ruled-line spacing as a fraction of the page height (88 px)
MARGIN_X = 0.12  # the red margin line
TEXT_X = 0.15  # where the handwriting starts
INK = "#16247a"
PRINT = "#222222"
RULE = "#9cc3e6"
MARGIN = "#e08a8a"
PAPER = "#fdfdf8"
HAND_SIZE = 25
PRINT_SIZE = 15
SKETCH = (1.4, 90.0, 2.0)  # matplotlib sketch params: scale, length, randomness (the xkcd look)

PNG_META = {"Software": None}
PDF_META = {
    "Creator": "scripts/demo/render_handwritten.py (T-0223)",
    "Producer": None,
    "CreationDate": None,
    "ModDate": None,
}


def setup_fonts(font_dir: Path) -> str:
    """Register the vendored handwriting font; return its family name."""
    ttf = font_dir / "PatrickHand-Regular.ttf"
    if not ttf.is_file():
        sys.exit(f"render_handwritten: missing handwriting font {ttf}")
    font_manager.fontManager.addfont(str(ttf))
    family = font_manager.FontProperties(fname=str(ttf)).get_name()
    plt.rcParams.update(
        {
            "font.family": [family, "DejaVu Sans"],  # per-glyph fallback for symbols it lacks
            "mathtext.fontset": "custom",
            "mathtext.rm": family,
            "mathtext.it": family,
            "mathtext.bf": family,
            "mathtext.sf": family,
            "mathtext.tt": family,
            "mathtext.cal": family,
            "mathtext.fallback": "stixsans",
            "svg.hashsalt": str(SEED),
            "pdf.fonttype": 42,
        }
    )
    return family


def wrap_math(fig: Figure, text: str, max_w: float, **kw: Any) -> list[str]:
    """Wrap at spaces outside $...$ (a mathtext span is never split), measuring rendered width."""
    words: list[str] = []
    buf = ""
    for tok in text.split(" "):
        buf = f"{buf} {tok}" if buf else tok
        if buf.count("$") % 2 == 0:
            words.append(buf)
            buf = ""
    if buf:
        words.append(buf)
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]

    def width(s: str) -> float:
        t = fig.text(0, 0, s, **kw)
        w = float(t.get_window_extent(renderer).width) / fig.bbox.width
        t.remove()
        return w

    lines: list[str] = []
    cur = ""
    for w in words:
        cand = f"{cur} {w}" if cur else w
        if cur and width(cand) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def paper(fig: Figure, first_rule: float) -> None:
    """Lined paper: blue rules from first_rule down, red margin line."""
    y = first_rule
    while y > 0.03:
        fig.add_artist(
            plt.Line2D([0.0, 1.0], [y, y], color=RULE, lw=1.1, transform=fig.transFigure)
        )
        y -= ROW
    fig.add_artist(
        plt.Line2D(
            [MARGIN_X, MARGIN_X], [0.0, 1.0], color=MARGIN, lw=1.3, transform=fig.transFigure
        )
    )


def header(fig: Figure, att: dict[str, Any]) -> float:
    """Printed worksheet header; returns the y (figure fraction) below it."""
    plain = {"family": "DejaVu Sans", "math_fontfamily": "dejavusans", "color": PRINT}
    fig.text(TEXT_X, 0.965, att["course"], fontsize=PRINT_SIZE - 2, **plain)
    fig.text(0.95, 0.965, "Name: ______________", fontsize=PRINT_SIZE - 2, ha="right", **plain)
    fig.text(
        TEXT_X,
        0.935,
        f"Problem {att['id']}.  {att['title']}",
        fontsize=PRINT_SIZE + 2,
        weight="bold",
        **plain,
    )
    y = 0.905
    for line in wrap_math(fig, att["problem"], 0.95 - TEXT_X, fontsize=PRINT_SIZE, **plain):
        fig.text(TEXT_X, y, line, fontsize=PRINT_SIZE, va="baseline", **plain)
        y -= 0.024
    return y - 0.01


def hand_line(fig: Figure, text: str, x: float, y: float, rng: random.Random) -> Any:
    dx = rng.uniform(-0.008, 0.012)
    dy = rng.uniform(-0.002, 0.002)
    rot = rng.uniform(-0.9, 0.9)
    return fig.text(
        x + dx,
        y + dy,
        text,
        fontsize=HAND_SIZE,
        color=INK,
        va="baseline",
        rotation=rot,
        rotation_mode="anchor",
        math_fontfamily="custom",
    )


def rows_needed(fig: Figure, text: str) -> int:
    """Ruled rows a written line takes: two for tall mathtext (fractions, limits)."""
    tall = ("\\frac", "\\int_", "\\left[", "\\sqrt")
    return 2 if any(t in text for t in tall) else 1


def write_lines(fig: Figure, att: dict[str, Any], start_rule: float, rng: random.Random) -> float:
    """Write the attempt on the ruled lines below start_rule; returns the next free rule."""
    rule = start_rule
    last = len(att["lines"]) - 1
    for i, line in enumerate(att["lines"]):
        n = rows_needed(fig, line["text"])
        base_off = 0.22 if n == 1 else 0.62
        if i == last:  # one extra row so the box clears the line above
            n += 1
            base_off += 0.5
        rule -= ROW * n
        base = rule + ROW * base_off
        t = hand_line(fig, line["text"], TEXT_X, base, rng)
        if i == last:  # the student boxes the final answer
            fig.canvas.draw()
            bb = t.get_window_extent().transformed(fig.transFigure.inverted())
            box = Rectangle(
                (bb.x0 - 0.012, bb.y0 - 0.008),
                bb.width + 0.024,
                bb.height + 0.016,
                transform=fig.transFigure,
                fill=False,
                ec=INK,
                lw=2.0,
            )
            box.set_sketch_params(*SKETCH)
            fig.add_artist(box)
    return rule


def ink_axes(fig: Figure, rect: tuple[float, float, float, float]) -> Axes:
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return ax


def sk(artist: Any) -> Any:
    artist.set_sketch_params(*SKETCH)
    return artist


def line(ax: Axes, xs: list[float], ys: list[float], **kw: Any) -> None:
    kw.setdefault("color", INK)
    kw.setdefault("lw", 2.6)
    for a in ax.plot(xs, ys, **kw):
        sk(a)


def arrow(ax: Axes, p0: tuple[float, float], p1: tuple[float, float], lw: float = 2.6) -> None:
    a = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=26, color=INK, lw=lw, shrinkA=0, shrinkB=0
    )
    ax.add_patch(sk(a))


def label(ax: Axes, x: float, y: float, s: str, size: int = HAND_SIZE - 2, **kw: Any) -> None:
    ax.text(x, y, s, fontsize=size, color=INK, math_fontfamily="custom", **kw)


def zigzag(
    x0: float, y0: float, x1: float, y1: float, n: int = 6, amp: float = 0.12
) -> tuple[list[float], list[float]]:
    """A resistor symbol between two points."""
    L = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    px, py = -uy, ux
    lead = 0.2 * L
    xs, ys = [x0, x0 + ux * lead], [y0, y0 + uy * lead]
    seg = (L - 2 * lead) / (2 * n)
    for k in range(1, 2 * n):
        s = lead + k * seg
        sgn = amp if k % 2 else -amp
        xs.append(x0 + ux * s + px * sgn)
        ys.append(y0 + uy * s + py * sgn)
    xs += [x1 - ux * lead, x1]
    ys += [y1 - uy * lead, y1]
    return xs, ys


# ---- diagrams -------------------------------------------------------------------------------


def d_incline(ax: Axes) -> None:
    th = math.radians(30)
    W = 6.0
    H = W * math.tan(th)
    line(ax, [0, W, W, 0], [0, 0, H, 0])
    ax.add_patch(sk(Arc((0, 0), 2.2, 2.2, theta1=0, theta2=30, color=INK, lw=2.2)))
    label(ax, 1.25, 0.12, r"$30°$")
    # block centred on the slope
    s = 3.6
    cx, cy = s * math.cos(th), s * math.sin(th)
    ux, uy = math.cos(th), math.sin(th)
    nx, ny = -math.sin(th), math.cos(th)
    b = 0.9
    pts = [
        (cx - ux * b / 2, cy - uy * b / 2),
        (cx + ux * b / 2, cy + uy * b / 2),
        (cx + ux * b / 2 + nx * b, cy + uy * b / 2 + ny * b),
        (cx - ux * b / 2 + nx * b, cy - uy * b / 2 + ny * b),
    ]
    ax.add_patch(sk(Polygon(pts, closed=True, fill=False, ec=INK, lw=2.6)))
    mx, my = cx + nx * b / 2, cy + ny * b / 2
    arrow(ax, (mx, my), (mx, my - 1.7))
    label(ax, mx + 0.15, my - 1.75, r"$mg$")
    arrow(ax, (mx, my), (mx + nx * 2.0, my + ny * 2.0))
    label(ax, mx + nx * 2.0 - 0.45, my + ny * 2.0 + 0.05, r"$N$")
    arrow(ax, (mx, my), (mx + ux * 1.8, my + uy * 1.8))
    label(ax, mx + ux * 1.8 + 0.1, my + uy * 1.8 - 0.1, r"$f$")
    label(ax, 3.9, 0.15, r"$m = 2$ kg", size=HAND_SIZE - 5)
    ax.set_xlim(-0.3, 6.5)
    ax.set_ylim(-0.3, 4.6)


def d_cliff(ax: Axes) -> None:
    line(ax, [-1.5, 0, 0, 12.5], [20, 20, 0, 0])
    ax.plot([0.3], [20.6], "o", color=INK, ms=13)
    t_land = math.sqrt(40 / 9.8)
    ts = [t_land * k / 30 for k in range(31)]
    line(
        ax, [0.3 + 5 * t for t in ts], [20.6 - 4.9 * t * t * 20.6 / 20 for t in ts], ls="--", lw=2.0
    )
    arrow(ax, (0.9, 20.6), (3.6, 20.6))
    label(ax, 1.0, 21.6, "5 m/s")
    arrow(ax, (-0.8, 10), (-0.8, 19.6), lw=1.8)
    arrow(ax, (-0.8, 10), (-0.8, 0.4), lw=1.8)
    label(ax, -3.6, 9.3, "20 m")
    arrow(ax, (5.0, -1.4), (0.2, -1.4), lw=1.8)
    arrow(ax, (5.0, -1.4), (10.0, -1.4), lw=1.8)
    label(ax, 4.2, -4.0, "x = ?")
    ax.set_xlim(-4.2, 13)
    ax.set_ylim(-4.6, 23.5)
    ax.set_aspect(0.45)


def d_circuit(ax: Axes) -> None:
    # battery on the left, R1 along the top, R2 and R3 in parallel on the right
    line(ax, [0, 0], [0, 1.6])
    line(ax, [-0.45, 0.45], [1.6, 1.6], lw=3.4)
    line(ax, [-0.25, 0.25], [1.85, 1.85], lw=3.4)
    line(ax, [0, 0], [1.85, 4])
    label(ax, -2.05, 1.45, "12 V")
    xs, ys = zigzag(0, 4, 4, 4)
    line(ax, xs, ys)
    label(ax, 1.2, 4.35, r"$R_1 = 6\ \Omega$")
    line(ax, [4, 6], [4, 4])
    line(ax, [0, 6], [0, 0])
    line(ax, [4, 4], [4, 3.6])
    line(ax, [6, 6], [4, 3.6])
    line(ax, [4, 4], [0, 0.4])
    line(ax, [6, 6], [0, 0.4])
    xs, ys = zigzag(4, 3.6, 4, 0.4, amp=0.14)
    line(ax, xs, ys)
    xs, ys = zigzag(6, 3.6, 6, 0.4, amp=0.14)
    line(ax, xs, ys)
    label(ax, 1.7, 1.85, r"$R_2 = 3\ \Omega$", size=HAND_SIZE - 5)
    label(ax, 6.35, 1.85, r"$R_3 = 6\ \Omega$", size=HAND_SIZE - 5)
    ax.set_xlim(-2.4, 8.5)
    ax.set_ylim(-0.5, 5.0)


def d_pulley(ax: Axes) -> None:
    r = 0.9
    line(ax, [-1.8, 1.8], [5.6, 5.6], lw=3.2)
    line(ax, [0, 0], [5.6, 4.4])
    ax.add_patch(sk(Circle((0, 4.4), r, fill=False, ec=INK, lw=2.6)))
    line(ax, [-r, -r], [4.4, 2.5])
    line(ax, [r, r], [4.4, 1.5])
    ax.add_patch(sk(Rectangle((-r - 0.55, 1.5), 1.1, 1.0, fill=False, ec=INK, lw=2.6)))
    ax.add_patch(sk(Rectangle((r - 0.65, 0.2), 1.3, 1.3, fill=False, ec=INK, lw=2.6)))
    label(ax, -r - 0.45, 1.8, "3 kg", size=HAND_SIZE - 7)
    label(ax, r - 0.5, 0.65, "5 kg", size=HAND_SIZE - 7)
    label(ax, -3.0, 1.75, r"$m_1$")
    label(ax, r + 0.85, 0.65, r"$m_2$")
    label(ax, -r - 0.55, 3.2, r"$T$", size=HAND_SIZE - 4)
    label(ax, r + 0.2, 2.6, r"$T$", size=HAND_SIZE - 4)
    arrow(ax, (-r, 1.5), (-r, 0.4), lw=2.0)
    label(ax, -r - 1.35, 0.5, r"$m_1 g$", size=HAND_SIZE - 6)
    arrow(ax, (r, 0.2), (r, -0.9), lw=2.0)
    label(ax, r + 0.2, -0.9, r"$m_2 g$", size=HAND_SIZE - 6)
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-1.3, 6.0)


def d_pendulum(ax: Axes) -> None:
    line(ax, [-1.0, 1.0], [0, 0], lw=3.2)
    th = math.radians(45)
    L = 3.0
    bx, by = L * math.sin(th), -L * math.cos(th)
    line(ax, [0, 0], [0, -L - 0.4], ls="--", lw=1.8)
    line(ax, [0, bx], [0, by])
    ax.add_patch(sk(Circle((bx, by), 0.28, fill=True, color=INK)))
    ax.add_patch(sk(Circle((0, -L), 0.28, fill=False, ec=INK, lw=2.0, ls="--")))
    ax.add_patch(sk(Arc((0, 0), 1.6, 1.6, theta1=-90, theta2=-45, color=INK, lw=2.2)))
    label(ax, 0.2, -1.35, r"$45°$", size=HAND_SIZE - 5)
    label(ax, bx * 0.55 + 0.15, by * 0.55 + 0.2, r"$L = 1$ m", size=HAND_SIZE - 5)
    # no height marker: drawing the true drop L(1 - cos 45°) would hint the answer (T-0223 R-2)
    label(ax, 0.45, -L - 0.25, "bottom", size=HAND_SIZE - 7)
    ax.set_xlim(-1.4, 4.6)
    ax.set_ylim(-3.6, 0.5)


def axes2d(ax: Axes, xmax: float, ymax: float) -> None:
    arrow(ax, (0, 0), (xmax, 0), lw=2.0)
    arrow(ax, (0, 0), (0, ymax), lw=2.0)
    label(ax, xmax + 0.03 * xmax, -0.03 * ymax, r"$x$")
    label(ax, -0.09 * xmax, ymax, r"$y$")


def d_area(ax: Axes) -> None:
    xs = [k / 40 for k in range(41)]
    ax.fill_between(xs, [x * x for x in xs], xs, color=INK, alpha=0.18, lw=0)
    axes2d(ax, 1.3, 1.3)
    line(ax, [0, 1.15], [0, 1.15])
    xx = [k / 40 * 1.1 for k in range(41)]
    line(ax, xx, [x * x for x in xx])
    label(ax, 0.62, 0.9, r"$y = x$", size=HAND_SIZE - 4)
    label(ax, 0.78, 0.25, r"$y = x^2$", size=HAND_SIZE - 4)
    ax.plot([1], [1], "o", color=INK, ms=8)
    label(ax, 1.03, 0.9, "(1, 1)", size=HAND_SIZE - 6)
    label(ax, -0.12, -0.12, "0", size=HAND_SIZE - 6)
    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(-0.2, 1.4)


def d_solution(ax: Axes) -> None:
    sy = 1 / 4.0  # squash y so 0..9.4 fits
    axes2d(ax, 1.25, 10.5 * sy)
    xs = [k / 40 for k in range(41)]
    line(ax, xs, [(math.exp(2 * x) + 2) * sy for x in xs])
    ax.plot([0], [3 * sy], "o", color=INK, ms=8)
    label(ax, 0.06, 3 * sy - 0.06, "(0, 3)", size=HAND_SIZE - 6)
    label(ax, 0.72, 4.6 * sy, r"$y = e^{2x} + 2$", size=HAND_SIZE - 4)
    line(ax, [1, 1], [-0.05, 0.05], lw=1.8)
    label(ax, 0.97, -0.25, "1", size=HAND_SIZE - 6)
    ax.set_xlim(-0.2, 1.45)
    ax.set_ylim(-0.3, 11 * sy)


DIAGRAMS = {
    "M3": (d_area, "after"),
    "M5": (d_solution, "after"),
    "P1": (d_incline, "before"),
    "P2": (d_cliff, "before"),
    "P3": (d_circuit, "before"),
    "P4": (d_pulley, "before"),
    "P5": (d_pendulum, "before"),
}
DIAGRAM_ROWS = 8


def render_page(att: dict[str, Any]) -> Figure:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), dpi=DPI, facecolor=PAPER)
    rng = random.Random(f"{SEED}-{att['id']}")
    below = header(fig, att)
    first_rule = below - 0.012
    paper(fig, first_rule)
    diagram = DIAGRAMS.get(att["id"])
    rule = first_rule
    if diagram and diagram[1] == "before":
        h = ROW * DIAGRAM_ROWS
        diagram[0](ink_axes(fig, (TEXT_X, rule - h, 0.55, h - 0.005)))
        rule -= h
    rule = write_lines(fig, att, rule, rng)
    if diagram and diagram[1] == "after":
        h = ROW * DIAGRAM_ROWS
        rule -= ROW
        diagram[0](ink_axes(fig, (TEXT_X + 0.05, rule - h, 0.5, h - 0.005)))
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--attempts", type=Path, default=DEMO / "attempts.json")
    ap.add_argument("--font-dir", type=Path, default=DEMO / "fonts")
    ap.add_argument("--out", type=Path, default=DEMO / "out")
    args = ap.parse_args(argv)
    os.environ["SOURCE_DATE_EPOCH"] = "1761955200"  # 2025-11-01, fixed
    import PIL

    if (matplotlib.__version__, PIL.__version__) != (MATPLOTLIB_VERSION, PILLOW_VERSION):
        print(
            f"render_handwritten: warning: matplotlib {matplotlib.__version__} / pillow "
            f"{PIL.__version__}, the committed out/ used {MATPLOTLIB_VERSION} / {PILLOW_VERSION}; "
            "the bytes may differ",
            file=sys.stderr,
        )
    setup_fonts(args.font_dir)
    data = json.loads(args.attempts.read_text(encoding="utf-8"))
    by_id = {a["id"]: a for a in data["attempts"]}
    if sorted(by_id) != sorted(ORDER):
        sys.exit(f"render_handwritten: expected ids {ORDER}, got {sorted(by_id)}")
    args.out.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out / "attempts.pdf", metadata=PDF_META) as pdf:
        for aid in ORDER:
            fig = render_page(by_id[aid])
            fig.savefig(args.out / f"{aid}.png", dpi=DPI, facecolor=PAPER, metadata=PNG_META)
            pdf.savefig(fig, facecolor=PAPER)
            plt.close(fig)
            print(f"wrote {args.out / f'{aid}.png'}")
    print(f"wrote {args.out / 'attempts.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
