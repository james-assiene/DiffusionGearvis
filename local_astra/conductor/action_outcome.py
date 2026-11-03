"""T-0136 A2: did the last action visibly change the screen, and where (docs/tasks/T-0136.md, A2-1..A2-7).

A pure pixel diff of the frame the model saw against the frame it will see next. No model call, no DOM, no I/O,
no clock; numpy and the standard library at import, PIL only inside :func:`gray`.

The rule shape is fixed by A2-3 (only the constants below are calibrated):

1. grayscale frames, decoded at ``1 / DRAFT_SCALE``, are cut into ``CELL_PX`` cells (array px, grid at (0, 0));
2. a pixel is changed when ``|delta| > NOISE_DELTA``; inside the hover zone (``+-HOVER_HALF_W x +-HOVER_HALF_H``
   full-frame px around each pointer) the threshold is the higher ``HOVER_DELTA`` (tolerated, not masked out);
3. a cell is changed when it holds at least ``MIN_CELL_PIXELS`` changed pixels;
3'. (A2b-4) such a cell is tolerated (not counted) when all its changed pixels lie within at most
   ``UNDERLINE_ROWS`` consecutive array rows and those rows meet some pointer's row band, the closed full-px rows
   ``[py - HOVER_HALF_H, py + HOVER_HALF_H]``, at any x (the ``a:hover`` underline of a long link);
4. ``page`` when the frame sizes differ (in :func:`outcome_of_jpeg`, the JPEGs' declared sizes) or changed cells
   are at least ``PAGE_FRACTION`` of all cells;
5. else ``near`` when a changed cell touches the closed near box ``+-W/4 x +-H/10`` around ``point``;
6. else ``elsewhere`` in the 3x3 ninth holding the most changed cells (ties: first in reading order);
7. else ``none``.

Units: ``point`` and ``pointers`` are full-frame viewport px; an array pixel ``j`` stands at full px
``DRAFT_SCALE * j``. The hover zone is the closed box ``[px - HOVER_HALF_W, px + HOVER_HALF_W] x
[py - HOVER_HALF_H, py + HOVER_HALF_H]`` in full px.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

OutcomeKind = Literal["none", "near", "elsewhere", "page"]
Region = Literal[
    "top left",
    "top",
    "top right",
    "left",
    "centre",
    "right",
    "bottom left",
    "bottom",
    "bottom right",
]

_REGIONS: tuple[Region, ...] = (
    "top left",
    "top",
    "top right",
    "left",
    "centre",
    "right",
    "bottom left",
    "bottom",
    "bottom right",
)

NO_VISIBLE_CHANGE = "no visible change"

# --- A2-3 / A2b-4 constants, re-tuned on the dev split of corpus 610f36f (A2b-5) -----------------------------------
# ``scripts/t0136_outcome_sweep.py`` (dev records only): 124 gated dev groups (36 none, 48 near, 22 elsewhere,
# 18 page), every bar met; each continuous constant holds the dev bars over +-15 % (or +-1 sweep step), varied one
# at a time. The dev band of each constant is in its comment.
#: Draft decode scale (JPEG DCT scaling): frames are compared at half resolution. Dev band {2}: at 1 a hover tint
#: reads ``near`` (none kept fails); 2 is the top of the domain {1, 2}.
DRAFT_SCALE = 2
#: Cell side in array px (16 full px at DRAFT_SCALE 2). Dev band 8..16: at 6 two navigations fall below the page
#: fraction.
CELL_PX = 8
#: A pixel outside every hover zone is changed above this gray |delta|. Above the JPEG jitter (max 5, O-A1). Dev
#: band 8..36 (the top is HOVER_DELTA's floor).
NOISE_DELTA = 24
#: Inside a hover zone the threshold is this higher |delta|: above ``button:hover`` brightness(1.08) and the hover
#: tints, below the weakest near colour change (an icon recolour, lost at 72). The underline is rule 3''s, not this
#: threshold's. Dev band 32..64.
HOVER_DELTA = 40
#: Hover zone half extents and the row band's half height, full-frame px (a closed box around each pointer).
#: Dev bands: HOVER_HALF_W 128..416+ (a menu-item tint reads ``near`` at 112), HOVER_HALF_H 20..96+ (the same tint
#: at 16).
HOVER_HALF_W = 200
HOVER_HALF_H = 32
#: Changed pixels a cell needs to count as changed (a 16 px checkbox gives 64 per cell). Dev band 1..6: from 7 the
#: 12 px count and qty changes are lost; 2 keeps the "5 items -> 6 items" count change.
MIN_CELL_PIXELS = 2
#: Changed-cell fraction at or above which the whole page changed: navigations change 12-14 % of cells, the largest
#: in-place change seen in A2 (a form replaced by a card) 6.4 %. Dev band 0.02..0.11 (two navigations lost at 0.12).
PAGE_FRACTION = 0.09
#: Rule 3' (A2b-4): a changed cell whose changed pixels span at most this many array rows, inside a pointer's row
#: band, is a hover underline and is tolerated. Unconstrained by dev (every value in {1, 2, 3} meets the dev bars),
#: so set to its premise value: an underline spans 1-2 rows at DRAFT_SCALE 2, small text 3-6 (the A2b planner probe).
UNDERLINE_ROWS = 2

Point = tuple[int, int]
GrayFrame = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Outcome:
    kind: OutcomeKind
    region: Region | None
    changed_cells: int
    cells: int

    @property
    def text(self) -> str:
        if self.kind == "none":
            return NO_VISIBLE_CHANGE
        if self.kind == "near":
            return "changed near your click"
        if self.kind == "page":
            return "page changed"
        return f"changed elsewhere ({self.region})"


def gray(jpeg: bytes) -> GrayFrame:
    """Decode ``jpeg`` to grayscale at ``1 / DRAFT_SCALE``: shape ``(H // DRAFT_SCALE, W // DRAFT_SCALE)``, and at
    least 1 x 1 (a frame smaller than ``DRAFT_SCALE`` px keeps one array pixel per axis, A2b-7 (b))."""
    from PIL import Image

    s = DRAFT_SCALE
    with Image.open(io.BytesIO(jpeg)) as img:
        width, height = img.size
        tw, th = max(1, width // s), max(1, height // s)
        if s > 1:
            img.draft("L", (tw, th))
        g = img.convert("L")
    if s > 1 and g.size == (width, height):  # the decoder could not scale: reduce in full
        g = g.reduce(s)
    arr: GrayFrame = np.asarray(g, dtype=np.uint8)[:th, :tw]
    return arr


#: JPEG start-of-frame markers (SOF0-SOF15 without DHT, JPG and DAC).
_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def _declared_size(jpeg: bytes) -> tuple[int, int] | None:
    """The ``(width, height)`` a JPEG declares in its start-of-frame segment, or ``None`` when no SOF is found
    before the scan (stdlib only; PIL stays inside :func:`gray`)."""
    n = len(jpeg)
    if jpeg[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 4 <= n:
        if jpeg[i] != 0xFF:
            return None
        marker = jpeg[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:  # standalone markers
            i += 2
            continue
        if marker in (0xD9, 0xDA):  # end of image, or the scan began without a frame header
            return None
        (length,) = struct.unpack_from(">H", jpeg, i + 2)
        if marker in _SOF_MARKERS:
            if i + 9 > n:
                return None
            height, width = struct.unpack_from(">HH", jpeg, i + 5)
            return (width, height) if height > 0 else None
        i += 2 + length
    return None


def _span_hits(
    lo: npt.NDArray[np.int64], hi: npt.NDArray[np.int64], c: int, num: int, den: int
) -> npt.NDArray[np.bool_]:
    """True where some x in the closed interval ``[lo, hi]`` (full px) has ``den * |x - c| <= num``.

    With ``num`` the frame extent this is the closed near box ``|x - c| <= extent / den``, in exact integers."""
    dist = np.maximum(np.maximum(lo - c, c - hi), 0)
    return np.asarray(den * dist <= num, dtype=np.bool_)


def outcome_of(
    before: GrayFrame, after: GrayFrame, point: Point, pointers: tuple[Point, ...] | list[Point]
) -> Outcome:
    """A2-3 on two gray frames as :func:`gray` returns them; ``point``/``pointers`` in full-frame px."""
    s = DRAFT_SCALE
    c = CELL_PX
    b = np.asarray(before)
    a = np.asarray(after)
    h, w = b.shape
    gh, gw = -(-h // c), -(-w // c)
    cells = gh * gw
    if a.shape != b.shape:
        return Outcome(kind="page", region=None, changed_cells=cells, cells=cells)

    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    changed = diff > NOISE_DELTA
    for px, py in pointers:
        x0 = max(0, -(-(px - HOVER_HALF_W) // s))
        x1 = min(w, (px + HOVER_HALF_W) // s + 1)
        y0 = max(0, -(-(py - HOVER_HALF_H) // s))
        y1 = min(h, (py + HOVER_HALF_H) // s + 1)
        if x0 < x1 and y0 < y1:
            changed[y0:y1, x0:x1] = diff[y0:y1, x0:x1] > HOVER_DELTA

    padded = np.zeros((gh * c, gw * c), dtype=np.bool_)
    padded[:h, :w] = changed
    blocks = padded.reshape(gh, c, gw, c)
    counts = blocks.sum(axis=(1, 3))
    cell_changed = counts >= MIN_CELL_PIXELS

    # cell extents in array px: first pixel and one past the last
    col_lo = np.arange(gw, dtype=np.int64) * c
    col_hi = np.minimum(col_lo + c, w)
    row_lo = np.arange(gh, dtype=np.int64) * c
    row_hi = np.minimum(row_lo + c, h)

    # rule 3': thin changed cells inside some pointer's row band (closed full-px rows, any x) are tolerated
    if len(pointers) > 0 and bool(cell_changed.any()):
        rows_hit: npt.NDArray[np.bool_] = np.asarray(
            blocks.any(axis=3)
        )  # (cell row, row in cell, cell col)
        first = rows_hit.argmax(axis=1)
        last = c - 1 - rows_hit[:, ::-1, :].argmax(axis=1)
        thin = (last - first + 1) <= UNDERLINE_ROWS
        top = row_lo[:, None] + first  # array rows of the first and last changed pixel
        bottom = row_lo[:, None] + last
        in_band = np.zeros_like(cell_changed)
        for _, py in pointers:
            j_lo = -(-(py - HOVER_HALF_H) // s)  # first array row with s * j >= py - HOVER_HALF_H
            j_hi = (py + HOVER_HALF_H) // s  # last array row with s * j <= py + HOVER_HALF_H
            in_band |= (top <= j_hi) & (bottom >= j_lo)
        cell_changed &= ~(thin & in_band)

    n_changed = int(cell_changed.sum())
    if n_changed == 0:
        return Outcome(kind="none", region=None, changed_cells=0, cells=cells)
    if n_changed >= PAGE_FRACTION * cells:
        return Outcome(kind="page", region=None, changed_cells=n_changed, cells=cells)

    full_w, full_h = w * s, h * s
    px, py = point
    near_x = _span_hits(col_lo * s, (col_hi - 1) * s, px, full_w, 4)
    near_y = _span_hits(row_lo * s, (row_hi - 1) * s, py, full_h, 10)
    if bool((cell_changed & np.outer(near_y, near_x)).any()):
        return Outcome(kind="near", region=None, changed_cells=n_changed, cells=cells)

    ninth_x = np.minimum(3 * (col_lo + col_hi) // (2 * w), 2)
    ninth_y = np.minimum(3 * (row_lo + row_hi) // (2 * h), 2)
    ninth = (3 * ninth_y[:, None] + ninth_x[None, :])[cell_changed]
    per_ninth = np.bincount(ninth, minlength=9)
    region = _REGIONS[int(np.argmax(per_ninth))]
    return Outcome(kind="elsewhere", region=region, changed_cells=n_changed, cells=cells)


def outcome_of_jpeg(
    before_jpeg: bytes, after_jpeg: bytes, point: Point, pointers: tuple[Point, ...] | list[Point]
) -> Outcome:
    """:func:`outcome_of` on two JPEG frames; rule 4's size test compares the JPEGs' declared sizes (A2b-7 (a)),
    which a draft decode can hide (1281 and 1280 px both decode to 640 at ``DRAFT_SCALE`` 2)."""
    before, after = gray(before_jpeg), gray(after_jpeg)
    size_b, size_a = _declared_size(before_jpeg), _declared_size(after_jpeg)
    if size_b is not None and size_a is not None and size_b != size_a:
        h, w = before.shape
        cells = -(-h // CELL_PX) * -(-w // CELL_PX)
        return Outcome(kind="page", region=None, changed_cells=cells, cells=cells)
    return outcome_of(before, after, point, pointers)
