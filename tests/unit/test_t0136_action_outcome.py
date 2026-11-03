"""T-0136 group A2 (unit + Hypothesis): the pixel-diff outcome detector ``local_astra/conductor/action_outcome.py``
(docs/tasks/T-0136.md, A2-1..A2-4 and A2-7; "Tests to write (A2)"; reviewer mutations 1-5).

**Status.** Activated for group A2 (the ``pending`` markers removed at ``/implement``). Until the module exists every
detector row fails on the named assertion in :func:`ao` -- never on an ImportError at collection (the module is
imported by name inside a ``try``). The fixture-integrity rows (``labels.jsonl`` and the committed frame subset) and
the A2-7 scope row are green at BASE_A2.

BASE_A2 = 2688714 (main after the A1-7 corpus run was recorded; the A2 branch point).

**Readings these rows pin (flagged in the test-author report; the task fixes the rule shape, not every unit).**

- R0. The calibrated constants of A2-3 are module attributes under the names A2-3 uses: ``DRAFT_SCALE`` (1 or 2),
  ``CELL_PX``, ``NOISE_DELTA``, ``HOVER_DELTA``, ``HOVER_HALF_W``, ``HOVER_HALF_H``, ``MIN_CELL_PIXELS``,
  ``PAGE_FRACTION``. The Hypothesis properties of A2-4 are stated in terms of them, so they must be readable.
- R1. ``gray(jpeg)`` returns a 2-D array of shape ``(H // DRAFT_SCALE, W // DRAFT_SCALE)`` (the draft decode).
- R2. ``outcome_of(before, after, point, pointers)`` takes arrays in that shape (what ``gray`` returns); ``point``
  and ``pointers`` are full-frame viewport px (A2-1), so the full frame is the array shape times ``DRAFT_SCALE``.
- R3. ``CELL_PX`` is in array px (the frames "are cut into ``CELL_PX`` cells"), the grid anchored at (0, 0). Every
  synthetic change here is aligned to that grid, so a reading with ``CELL_PX`` in full-frame px (smaller cells) sees
  the same whole cells.
- R4. The near box is closed: a changed cell intersects it when any of its pixels lies within ``+-W/4`` and ``+-H/10``
  of ``point`` inclusive (``+-320 x +-80`` at 1280x800). The +-1 px rows put a cell edge exactly on the box edge and
  one array pixel outside it.
- R5. ``HOVER_HALF_W``/``HOVER_HALF_H`` are not assumed to be in either unit: a change "inside the zone" is inside
  ``+-HOVER_HALF`` full px (inside under both readings) and "outside" is beyond ``+-HOVER_HALF * DRAFT_SCALE`` plus
  one array pixel (outside under both).

**A2b (amendment A1c/A2b, H-T136-A2b-1: rule 3', F5, the frame-group split).** The A2b rows are activated
(markers removed at the A2b implementation c94f529); the label, fixture and count rows were never pending.

- R5 (re-pinned for A2b-4): ``HOVER_HALF_W``/``HOVER_HALF_H`` are full-frame px and the zone is the closed box
  ``[px - HW, px + HW] x [py - HH, py + HH]``; array pixel ``j`` stands at full px ``DRAFT_SCALE * j``. The A2 rows
  that read R5 both ways still hold under this reading (their "inside" is a subset of it, their "outside" a superset).
- R6. ``UNDERLINE_ROWS`` is a module attribute in {1, 2, 3}. Rule 3' tolerates a cell with >= ``MIN_CELL_PIXELS``
  changed pixels when its changed pixels lie within <= ``UNDERLINE_ROWS`` consecutive array rows and those rows
  intersect some pointer's row band, the closed full-px rows ``[py - HOVER_HALF_H, py + HOVER_HALF_H]``, at any x.
- R7. ``PAGE_FRACTION`` is compared as ``changed_cells >= PAGE_FRACTION * cells`` and ``MIN_CELL_PIXELS`` as
  ``changed_pixels >= MIN_CELL_PIXELS`` (A2-3 rules 3 and 4, "at least"). ``cells`` counts partial edge cells.
- Known equivalences, recorded rather than asserted: an open bottom edge on the *zone* alone is equivalent under
  rule 3' (the band shares ``HOVER_HALF_H``); the page ``>``-for-``>=`` mutant is equivalent when
  ``PAGE_FRACTION * cells`` is not an integer; an edge-column mutant is equivalent when ``CELL_PX * 1`` column of
  pixels cannot reach ``MIN_CELL_PIXELS``. Each such row prints the equivalence it found.

Expected values are by hand (the geometry of A2-3: the near box, the ninths, the precedence) or from the hand labels
in ``tests/fixtures/t0136/labels.jsonl`` (labelled blind from the A1-6 contact sheets, H-T136-1); nothing is captured
from an implementation, none existing.

Regime: CPU only, numpy + PIL (test side), no browser, no network; frames 1280x800; Hypothesis default profile.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import io
import json
import numbers
import subprocess
import sys
from fractions import Fraction
from math import ceil, floor
from pathlib import Path
from types import ModuleType
from typing import Any, get_args

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "local_astra" / "conductor" / "action_outcome.py"
FIXTURES = REPO / "tests" / "fixtures" / "t0136"
LABELS = FIXTURES / "labels.jsonl"
FRAMES = FIXTURES / "frames"

#: The commit group A2 branched from (main at "T-0136: A1-7 corpus run recorded"). A literal, never a moving ref.
BASE_A2 = "2688714"
#: The implementation tip for the A2-7 scope row: starts at BASE_A2 and is bumped by the test-author or the
#: orchestrator to the last implementation commit before review (the implementer never edits tests).
TIP_A2 = "c94f529"

#: A2b: the labels at the A2 test-author's last commit (corpus 20104cc); carried labels must not move (A2b-N).
LABELS_A2 = "ad8268d"
#: The 20104cc ids that vanished in the 610f36f rebuild (JPEG jitter changed their bytes); dropped (A2b-1).
DROPPED_20104CC = frozenset(
    {"016b6ce03a8ff5b9", "07366c56e5ac3ebc", "16cab72b9df10a4b", "36e252c8ca3bed15", "536da5c3d1f70f80",
     "5f39ffcb230c1372", "72350e7b93e5dc60", "781a7b56a4cbdbb4", "79f73f3a7a173306", "7aac35429b85cabe",
     "cba4539b01dc9e15", "d33892830bc18af2", "e27ff3d42006c2ba", "e80732de9cb48a5c"}
)  # fmt: skip
#: New replay ids that inherit their label from a jitter-equivalent 20104cc pair with the same points (A2b-1),
#: each checked on its own contact sheet: new id -> (parent id, label, region).
INHERITED = {
    "0573be5a4fbc853b": ("5afa76caccc35b68", "elsewhere", "top right"),
    "1c6a275fd9b7bde0": ("7aac35429b85cabe", "elsewhere", "top right"),
    "678f72f1ec307eaf": ("07366c56e5ac3ebc", "elsewhere", "top right"),
    "9fed2cc24022619e": ("3556aa6508ec5a3e", "near", None),
    "a3100286d17ce023": ("5f39ffcb230c1372", "near", None),
    "eebb6a062fb64648": ("cdea44042cc56fc5", "elsewhere", "top right"),
}
#: F3: the landmine pairs added to the committed subset in A2b (pending detector rows until A2b is implemented).
LANDMINES = {
    "5e772b5f725f3376": "n-hover-3 underline (page a)",
    "7915c1b58d13ca93": "c-label-3 Start -> Stop (page a)",
    "095f56f028782c71": "b-ul-09 540 px link underline",
    "a05099282f2dab36": "b-ul-12 560 px link underline",
    "52ba90a0337b03f3": "b-small-07 qty 1 -> 2",
    "b38e02246a6b0719": "b-small-08 5 items -> 6 items",
    "ab0abced9820bc7b": "b-small-10 red Invalid hint",
    "b7d5cabe1dec6a0c": "b-colour-05 star outline -> filled",
    "ed07b542a520b5fb": "b-pre-05 un-hover then a no-effect click",
}


W, H = 1280, 800
KINDS = ("none", "near", "elsewhere", "page")
REGIONS = (
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
#: A2-2, typed from the task file: exactly these 12 strings.
VOCAB = frozenset(
    {"no visible change", "changed near your click", "page changed"}
    | {f"changed elsewhere ({r})" for r in REGIONS}
)
HYP = settings(deadline=None)

try:
    _AO: ModuleType | None = importlib.import_module("local_astra.conductor.action_outcome")
except (
    ModuleNotFoundError
) as _exc:  # only the module itself may be missing; anything else is a real error
    if _exc.name != "local_astra.conductor.action_outcome":
        raise
    _AO = None


# =====================================================================================================
# Access to the module under test
# =====================================================================================================


def ao() -> ModuleType:
    assert _AO is not None, (
        "T-0136 A2-1: local_astra/conductor/action_outcome.py does not exist yet"
    )
    return _AO


def api(name: str) -> Any:
    m = ao()
    assert hasattr(m, name), f"T-0136 A2-1: action_outcome has no public name {name!r}"
    return getattr(m, name)


def const(name: str) -> Any:
    m = ao()
    assert hasattr(m, name), (
        f"T-0136 A2-3: action_outcome has no calibrated constant {name!r} (reading R0 of this module's docstring)"
    )
    value = getattr(m, name)
    assert isinstance(value, int | float) and not isinstance(value, bool), (name, value)
    return value


def scale() -> int:
    s = const("DRAFT_SCALE")
    assert s in (1, 2), f"T-0136 A2-3: DRAFT_SCALE in {{1, 2}}, got {s!r}"
    return int(s)


def cell_full() -> int:
    """One grid cell in full-frame px (R3)."""
    c = const("CELL_PX")
    assert isinstance(c, int) and c > 0, c
    return c * scale()


def check_constants() -> None:
    """The order A2-3 states: the hover threshold is the *higher* one, and both are below a strong change."""
    nd, hd = const("NOISE_DELTA"), const("HOVER_DELTA")
    assert 0 <= nd < hd < 255, f"T-0136 A2-3: 0 <= NOISE_DELTA < HOVER_DELTA < 255; got {nd}, {hd}"
    assert 0 < const("PAGE_FRACTION") <= 1, const("PAGE_FRACTION")
    assert const("MIN_CELL_PIXELS") >= 1, const("MIN_CELL_PIXELS")
    assert const("HOVER_HALF_W") > 0 and const("HOVER_HALF_H") > 0


def underline_rows() -> int:
    """R6: the rule-3' constant, an integer in {1, 2, 3} (A2b-4), and narrower than a cell."""
    u = const("UNDERLINE_ROWS")
    assert isinstance(u, int) and u in (1, 2, 3), (
        f"T-0136 A2b-4: UNDERLINE_ROWS in {{1, 2, 3}}, got {u!r}"
    )
    assert int(const("CELL_PX")) > u, (const("CELL_PX"), u)
    return u


# =====================================================================================================
# Frames (full-frame px in, array px inside)
# =====================================================================================================


def canvas(value: int = 0) -> np.ndarray:
    s = scale()
    return np.full((H // s, W // s), value, dtype=np.uint8)


def fill(arr: np.ndarray, rect: tuple[int, int, int, int], value: int | np.ndarray) -> None:
    """Paint the half-open full-px rect ``(x0, y0, x1, y1)``; the bounds must be multiples of ``DRAFT_SCALE``."""
    s = scale()
    x0, y0, x1, y1 = rect
    assert all(v % s == 0 for v in rect), f"premise: rect {rect} not on the {s}-px grid"
    arr[y0 // s : y1 // s, x0 // s : x1 // s] = value


def texture(rng: np.random.Generator, lo: int = 0, hi: int = 256) -> np.ndarray:
    """A page-like frame: 8x8 blocks of random levels."""
    s = scale()
    h, w = H // s, W // s
    lo, hi = max(0, lo), min(256, hi)
    small = rng.integers(lo, hi, size=(ceil(h / 8), ceil(w / 8)), dtype=np.int64)
    return np.kron(small, np.ones((8, 8), dtype=np.int64))[:h, :w].astype(np.uint8)


def jpeg(arr_full: np.ndarray, quality: int = 85) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(arr_full).save(out, format="JPEG", quality=quality)
    return out.getvalue()


def ninth_of(rect: tuple[int, int, int, int]) -> str:
    """The ninth wholly containing the half-open full-px rect (hand oracle; asserts the premise)."""
    x0, y0, x1, y1 = rect
    third_w, third_h = Fraction(W, 3), Fraction(H, 3)

    def band(a: int, b: int, third: Fraction) -> int:
        for i in range(3):
            if a >= i * third and b <= (i + 1) * third:
                return i
        raise AssertionError(f"premise: [{a}, {b}) straddles a ninth boundary")

    return REGIONS[3 * band(y0, y1, third_h) + band(x0, x1, third_w)]


def in_near_box(rect: tuple[int, int, int, int], point: tuple[int, int], margin: int = 0) -> bool:
    """Does the half-open full-px rect touch the closed near box (R4), grown by ``margin``?"""
    x0, y0, x1, y1 = rect
    px, py = point
    hw, hh = W // 4 + margin, H // 10 + margin
    return x0 <= px + hw and x1 - 1 >= px - hw and y0 <= py + hh and y1 - 1 >= py - hh


def in_outer_zone(
    rect: tuple[int, int, int, int], pointer: tuple[int, int], margin: int = 0
) -> bool:
    """Does the rect touch the hover zone under the wider of the two unit readings (R5), grown by ``margin``?"""
    s = scale()
    hw = int(const("HOVER_HALF_W")) * s + s + margin
    hh = int(const("HOVER_HALF_H")) * s + s + margin
    x0, y0, x1, y1 = rect
    px, py = pointer
    return x0 <= px + hw and x1 - 1 >= px - hw and y0 <= py + hh and y1 - 1 >= py - hh


def inner_zone_slices(pointer: tuple[int, int]) -> tuple[slice, slice]:
    """Array slices of the pixels inside ``+-HOVER_HALF`` full px of ``pointer`` under both readings (R5)."""
    s = scale()
    px, py = pointer
    hw, hh = int(const("HOVER_HALF_W")), int(const("HOVER_HALF_H"))
    # array pixel j covers full px [s*j, s*j + s - 1]; keep it only when that span is inside the zone
    jx0, jx1 = max(0, ceil((px - hw) / s)), min(W // s, floor((px + hw - (s - 1)) / s) + 1)
    jy0, jy1 = max(0, ceil((py - hh) / s)), min(H // s, floor((py + hh - (s - 1)) / s) + 1)
    return slice(jy0, jy1), slice(jx0, jx1)


def total_cells_lower_bound() -> int:
    c = const("CELL_PX")
    s = scale()
    return (W // s // c) * (H // s // c)


# =====================================================================================================
# Calling the detector
# =====================================================================================================


Point = tuple[int, int]


def check_outcome(o: Any) -> None:
    """Every outcome, every row: A2-2's closed vocabulary and the region/kind coupling."""
    assert o.kind in KINDS, o
    assert o.text in VOCAB, f"T-0136 A2-2: {o.text!r} is not one of the 12 strings"
    if o.kind == "elsewhere":
        assert o.region in REGIONS, o
        assert o.text == f"changed elsewhere ({o.region})", o
    else:
        assert o.region is None, f"T-0136 A2-2: region is set exactly when kind == 'elsewhere': {o}"
    assert isinstance(o.changed_cells, numbers.Integral) and isinstance(
        o.cells, numbers.Integral
    ), o
    assert o.cells > 0 and 0 <= o.changed_cells <= o.cells, o


def run(
    before: np.ndarray, after: np.ndarray, point: Point, pointers: tuple[Point, ...] | None = None
) -> Any:
    ptrs = (point,) if pointers is None else pointers
    o = api("outcome_of")(before, after, point, ptrs)
    check_outcome(o)
    return o


def run_jpeg(
    before: bytes, after: bytes, point: Point, pointers: tuple[Point, ...] | None = None
) -> Any:
    ptrs = (point,) if pointers is None else pointers
    o = api("outcome_of_jpeg")(before, after, point, ptrs)
    check_outcome(o)
    return o


def label_of(o: Any) -> tuple[str, str | None]:
    return (o.kind, o.region)


# =====================================================================================================
# A2-1 / A2-2: names, vocabulary, the literal
# =====================================================================================================


def test_public_names_and_literals() -> None:
    """A2-1: the closed ``OutcomeKind`` and ``Region`` literals, exactly (order-free)."""
    assert set(get_args(api("OutcomeKind"))) == set(KINDS)
    assert len(get_args(api("OutcomeKind"))) == 4
    assert set(get_args(api("Region"))) == set(REGIONS)
    assert len(get_args(api("Region"))) == 9
    for name in ("Outcome", "gray", "outcome_of", "outcome_of_jpeg", "NO_VISIBLE_CHANGE"):
        api(name)


def test_no_visible_change_literal() -> None:
    """A2-1, B-5 (the T-0137 contract): the literal, and the ``none`` outcome's text is it."""
    assert api("NO_VISIBLE_CHANGE") == "no visible change"
    o = api("Outcome")(kind="none", region=None, changed_cells=0, cells=100)
    assert o.text == "no visible change" == api("NO_VISIBLE_CHANGE")


def test_vocabulary_is_exactly_twelve_strings() -> None:
    """A2-2: every valid ``Outcome`` renders one of the 12 strings, and all 12 are reachable."""
    Outcome = api("Outcome")
    texts = {
        Outcome(kind="none", region=None, changed_cells=0, cells=100).text,
        Outcome(kind="near", region=None, changed_cells=1, cells=100).text,
        Outcome(kind="page", region=None, changed_cells=90, cells=100).text,
    } | {Outcome(kind="elsewhere", region=r, changed_cells=1, cells=100).text for r in REGIONS}
    assert texts == VOCAB, sorted(texts ^ VOCAB)
    assert len(VOCAB) == 12
    assert Outcome(kind="elsewhere", region="top right", changed_cells=1, cells=100).text == (
        "changed elsewhere (top right)"
    )
    assert (
        Outcome(kind="near", region=None, changed_cells=1, cells=100).text
        == "changed near your click"
    )
    assert Outcome(kind="page", region=None, changed_cells=90, cells=100).text == "page changed"


def test_outcome_is_frozen_and_compares_by_value() -> None:
    Outcome = api("Outcome")
    a = Outcome(kind="elsewhere", region="bottom", changed_cells=3, cells=100)
    b = Outcome(kind="elsewhere", region="bottom", changed_cells=3, cells=100)
    assert a == b and hash(a) == hash(b)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        a.kind = "near"


def test_gray_shape_and_level() -> None:
    """R1: a 1280x800 JPEG decodes to ``(800 // DRAFT_SCALE, 1280 // DRAFT_SCALE)``, 2-D, at the right level."""
    s = scale()
    frame = np.full((H, W), 128, dtype=np.uint8)
    g = api("gray")(jpeg(frame))
    assert isinstance(g, np.ndarray) and g.ndim == 2, type(g)
    assert g.shape == (H // s, W // s), g.shape
    assert abs(float(np.asarray(g, dtype=np.float64).mean()) - 128.0) <= 2.0


# =====================================================================================================
# A2-4: the Hypothesis properties
# =====================================================================================================


def draw_point(data: st.DataObject) -> Point:
    s = scale()
    x = data.draw(st.integers(0, (W - 1) // s)) * s
    y = data.draw(st.integers(0, (H - 1) // s)) * s
    return (x, y)


def draw_pointers(data: st.DataObject, point: Point) -> tuple[Point, ...]:
    """0-2 pointers (A2-1); when there are any, the last one is the click point (B-2's shape)."""
    n = data.draw(st.integers(0, 2))
    if n == 0:
        return ()
    if n == 1:
        return (point,)
    return (draw_point(data), point)


@HYP
@given(data=st.data())
def test_identical_frames_give_none_for_every_point(data: st.DataObject) -> None:
    """A2-4: identical frames -> ``none`` whatever the point and pointers (and zero changed cells)."""
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    frame = (
        texture(rng)
        if data.draw(st.booleans())
        else rng.integers(0, 256, canvas().shape, dtype=np.uint8)
    )
    point = draw_point(data)
    o = run(frame, frame.copy(), point, draw_pointers(data, point))
    assert o.kind == "none" and o.changed_cells == 0, o


def random_after(rng: np.random.Generator, before: np.ndarray, data: st.DataObject) -> np.ndarray:
    s = scale()
    after = before.copy()
    for _ in range(data.draw(st.integers(0, 4))):
        x0 = data.draw(st.integers(0, W // s - 1))
        y0 = data.draw(st.integers(0, H // s - 1))
        w = data.draw(st.integers(1, W // s))
        h = data.draw(st.integers(1, H // s))
        after[y0 : y0 + h, x0 : x0 + w] = data.draw(st.integers(0, 255))
    if data.draw(st.booleans()):
        noise = rng.integers(-40, 41, size=before.shape)
        after = np.clip(after.astype(np.int64) + noise, 0, 255).astype(np.uint8)
    return after


@HYP
@given(data=st.data())
def test_kind_is_symmetric_under_swap(data: st.DataObject) -> None:
    """A2-4: swapping before and after never changes ``kind`` (an unsigned-subtraction wraparound fails here)."""
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    before = texture(rng)
    after = random_after(rng, before, data)
    point = draw_point(data)
    pointers = draw_pointers(data, point)
    assert run(before, after, point, pointers).kind == run(after, before, point, pointers).kind


@HYP
@given(data=st.data())
def test_deterministic_and_inputs_untouched(data: st.DataObject) -> None:
    """A2-1/A2-4: two calls give equal outcomes, and neither frame is modified."""
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    before = texture(rng)
    after = random_after(rng, before, data)
    b0, a0 = before.copy(), after.copy()
    point = draw_point(data)
    pointers = draw_pointers(data, point)
    first = run(before, after, point, pointers)
    second = run(before, after, point, pointers)
    assert first == second
    assert np.array_equal(before, b0) and np.array_equal(after, a0), (
        "the detector modified its input"
    )


@HYP
@given(data=st.data())
def test_change_within_hover_delta_inside_the_zone_is_none(data: st.DataObject) -> None:
    """A2-4: any change with ``|delta| <= HOVER_DELTA`` confined to the hover zone gives ``none`` -- the zone of
    the *previous* pointer as well as the current one. ``exact`` puts ``+-HOVER_DELTA`` on every zone pixel, which
    is above ``NOISE_DELTA``: a detector without the zone (mutation 1) sees it."""
    check_constants()
    hd = int(const("HOVER_DELTA"))
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    level = data.draw(st.integers(hd, 255 - hd))
    before = np.clip(texture(rng, level - 5, level + 6), 0, 255).astype(np.uint8)
    point = draw_point(data)
    pointers = draw_pointers(data, point)
    assume(pointers)
    which = data.draw(st.integers(0, len(pointers) - 1))
    ys, xs = inner_zone_slices(pointers[which])
    region = before[ys, xs].astype(np.int64)
    assume(region.size > 0)
    if data.draw(st.booleans()):  # exact
        sign = data.draw(st.sampled_from((-1, 1)))
        delta = np.full(region.shape, sign * hd, dtype=np.int64)
    else:
        delta = rng.integers(-hd, hd + 1, size=region.shape)
    after = before.copy()
    after[ys, xs] = np.clip(region + delta, 0, 255).astype(np.uint8)
    assert int(np.abs(after.astype(np.int64) - before.astype(np.int64)).max()) <= hd
    o = run(before, after, point, pointers)
    assert o.kind == "none", (o, pointers, which)


@HYP
@given(data=st.data())
def test_noise_within_noise_delta_anywhere_is_none(data: st.DataObject) -> None:
    """A2-4: uniform noise with ``|delta| <= NOISE_DELTA`` over the whole frame gives ``none`` (not ``page``)."""
    check_constants()
    nd = int(const("NOISE_DELTA"))
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    before = texture(rng)
    if data.draw(st.booleans()):  # every pixel at the bound
        noise = np.where(rng.random(before.shape) < 0.5, -nd, nd)
    else:
        noise = rng.integers(-nd, nd + 1, size=before.shape)
    after = np.clip(before.astype(np.int64) + noise, 0, 255).astype(np.uint8)
    point = draw_point(data)
    o = run(before, after, point, draw_pointers(data, point))
    assert o.kind == "none", o


def aligned_span(lo: Fraction, hi: Fraction, step: int) -> tuple[int, int]:
    """The widest ``[a, b)`` on the ``step`` grid inside ``[lo, hi)``."""
    return ceil(lo / step) * step, floor(hi / step) * step


@HYP
@given(data=st.data())
def test_single_strong_block_far_from_pointers_is_elsewhere_in_its_ninth(
    data: st.DataObject,
) -> None:
    """A2-4: one strong, grid-aligned block wholly inside a ninth, clear of the near box and every hover zone,
    gives ``elsewhere`` with that ninth (hand oracle :func:`ninth_of`)."""
    check_constants()
    cf = cell_full()
    hd = int(const("HOVER_DELTA"))
    i = data.draw(st.integers(0, 8))
    r, c = divmod(i, 3)
    xa, xb = aligned_span(Fraction(W * c, 3), Fraction(W * (c + 1), 3), cf)
    ya, yb = aligned_span(Fraction(H * r, 3), Fraction(H * (r + 1), 3), cf)
    assume(xb - xa >= cf and yb - ya >= cf)
    nw = data.draw(st.integers(1, min(2, (xb - xa) // cf)))
    nh = data.draw(st.integers(1, min(2, (yb - ya) // cf)))
    gx = data.draw(st.integers(0, (xb - xa) // cf - nw))
    gy = data.draw(st.integers(0, (yb - ya) // cf - nh))
    rect = (xa + gx * cf, ya + gy * cf, xa + (gx + nw) * cf, ya + (gy + nh) * cf)
    assert ninth_of(rect) == REGIONS[i]
    s = scale()
    # the block must stay below the page fraction under either cell reading
    assume(nw * nh * s * s < const("PAGE_FRACTION") * total_cells_lower_bound())
    point = draw_point(data)
    pointers = draw_pointers(data, point)
    assume(not in_near_box(rect, point, margin=cf))
    assume(not any(in_outer_zone(rect, p, margin=cf) for p in (*pointers, point)))
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    before = texture(rng)
    after = before.copy()
    d = data.draw(st.integers(hd + 1, 255))
    base = data.draw(st.integers(0, 255 - d))
    fill(before, rect, base)
    fill(after, rect, base + d)
    o = run(before, after, point, pointers)
    assert label_of(o) == ("elsewhere", REGIONS[i]), (o, rect, point, pointers)


@HYP
@given(data=st.data())
def test_strong_block_inside_the_hover_zone_is_near(data: st.DataObject) -> None:
    """A2-4: a strong block inside the click's hover zone gives ``near`` -- the zone is tolerated, not masked out
    (mutation 2). The block covers at least ``MIN_CELL_PIXELS`` pixels of one cell (hand count on the R3 grid).
    A2b-4: the block is drawn taller than ``UNDERLINE_ROWS`` array rows, and some cell holding >=
    ``MIN_CELL_PIXELS`` changed pixels spans more than ``UNDERLINE_ROWS`` rows (rule 3' cannot tolerate it)."""
    check_constants()
    u = underline_rows()
    s = scale()
    point = draw_point(data)
    ys, xs = inner_zone_slices(point)
    # ... and inside the near box (the zone is the click's; A2-4 does not let a zone reach past the box)
    px, py = point
    ys = slice(max(ys.start, ceil((py - H // 10) / s)), min(ys.stop, (py + H // 10) // s + 1))
    xs = slice(max(xs.start, ceil((px - W // 4) / s)), min(xs.stop, (px + W // 4) // s + 1))
    zh, zw = ys.stop - ys.start, xs.stop - xs.start
    assume(zh > u and zw > 0)
    bh = data.draw(st.integers(u + 1, zh))
    bw = data.draw(st.integers(1, zw))
    oy = ys.start + data.draw(st.integers(0, zh - bh))
    ox = xs.start + data.draw(st.integers(0, zw - bw))
    mask = np.zeros(canvas().shape, dtype=bool)
    mask[oy : oy + bh, ox : ox + bw] = True
    c = int(const("CELL_PX"))
    h, w = mask.shape
    padded = np.pad(mask, ((0, (-h) % c), (0, (-w) % c))).reshape(ceil(h / c), c, ceil(w / c), c)
    per_cell = padded.sum(axis=(1, 3))
    rows_hit = padded.any(axis=3)  # (cell row, row in cell, cell col)
    span = rows_hit.sum(axis=1)  # a solid block: rows hit in a cell are consecutive
    assume(bool(((per_cell >= const("MIN_CELL_PIXELS")) & (span > u)).any()))
    # stay below the page fraction under either cell reading (hand count of touched cells, times s^2)
    assume(int((per_cell > 0).sum()) * s * s < const("PAGE_FRACTION") * total_cells_lower_bound())
    before = canvas(0)
    after = before.copy()
    after[mask] = 255
    pointers = (point,) if data.draw(st.booleans()) else (draw_point(data), point)
    o = run(before, after, point, pointers)
    assert o.kind == "near", (o, point, (oy, ox, bh, bw))


@HYP
@given(data=st.data())
def test_full_frame_shift_of_a_texture_is_page(data: st.DataObject) -> None:
    """A2-4: a textured frame shifted by its period's half (every pixel flips between two strong levels) gives
    ``page``."""
    check_constants()
    s = scale()
    h, w = H // s, W // s
    q = data.draw(st.integers(1, 8))
    lo = data.draw(st.integers(0, 254 - int(const("HOVER_DELTA"))))
    hi = data.draw(st.integers(lo + int(const("HOVER_DELTA")) + 1, 255))
    axis = data.draw(st.sampled_from(("x", "y")))
    yy, xx = np.mgrid[0:h, 0:w]

    def board(dx: int, dy: int) -> np.ndarray:
        return np.where(((yy + dy) // q + (xx + dx) // q) % 2 == 0, lo, hi).astype(np.uint8)

    before = board(0, 0)
    after = board(q, 0) if axis == "x" else board(0, q)
    assert np.all(before != after), "premise: every pixel flips"
    point = draw_point(data)
    o = run(before, after, point, draw_pointers(data, point))
    assert o.kind == "page", o


@HYP
@given(data=st.data())
def test_text_is_always_in_the_vocabulary(data: st.DataObject) -> None:
    """A2-4: ``.text`` is always one of the 12 strings, over arbitrary frames, points and 0-2 pointers (the check
    lives in :func:`check_outcome`, run on both orders)."""
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    before = (
        texture(rng)
        if data.draw(st.booleans())
        else rng.integers(0, 256, canvas().shape, dtype=np.uint8)
    )
    after = random_after(rng, before, data)
    point = draw_point(data)
    pointers = draw_pointers(data, point)
    run(before, after, point, pointers)
    run(after, before, point, pointers)


# =====================================================================================================
# A2-3: precedence, regions, the near box at +-1 px
# =====================================================================================================


def aligned_cells(region: str, n: int) -> list[tuple[int, int, int, int]]:
    """``n`` grid cells (full px) wholly inside ``region``, taken from its corner farthest from the centre."""
    cf = cell_full()
    i = REGIONS.index(region)
    r, c = divmod(i, 3)
    xa, xb = aligned_span(Fraction(W * c, 3), Fraction(W * (c + 1), 3), cf)
    ya, yb = aligned_span(Fraction(H * r, 3), Fraction(H * (r + 1), 3), cf)
    cols = list(range(xa, xb, cf))
    rows = list(range(ya, yb, cf))
    if c == 2:
        cols.reverse()
    if r == 2:
        rows.reverse()
    cells = [(x, y, x + cf, y + cf) for y in rows for x in cols][:n]
    assert len(cells) == n, f"premise: {region} holds fewer than {n} cells of {cf} px"
    for cell in cells:
        assert ninth_of(cell) == region
    return cells


CENTRE: Point = (640, 400)


def strong(
    cells: list[tuple[int, int, int, int]], before: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    b = canvas(0) if before is None else before
    a = b.copy()
    for cell in cells:
        fill(a, cell, 255)
    return b, a


def clear_of_centre(cells: list[tuple[int, int, int, int]]) -> None:
    for cell in cells:
        assert not in_near_box(cell, CENTRE, margin=cell_full()), (
            f"premise: {cell} touches the near box"
        )


def test_precedence_near_over_elsewhere() -> None:
    """A2-3 precedence (mutation 5): one changed cell at the click and two (more) in the bottom-right ninth ->
    ``near``."""
    check_constants()
    cf = cell_full()
    far = aligned_cells("bottom right", 2)
    clear_of_centre(far)
    at_click = (640, 384, 640 + cf, 384 + cf)
    before, after = strong([at_click, *far])
    o = run(before, after, (640 + cf // 2, 384 + cf // 2))
    assert o.kind == "near", o


def test_precedence_page_over_near() -> None:
    """A2-3 precedence: a whole-frame change (it touches the near box too) is ``page``, not ``near``."""
    check_constants()
    before = canvas(0)
    after = canvas(255)
    for point in (CENTRE, (0, 0), (1279, 799)):
        assert run(before, after, point).kind == "page"


def test_page_when_frame_sizes_differ() -> None:
    """A2-3 rule 4: different frame sizes -> ``page``, both orders, even when the common part is identical."""
    s = scale()
    small = canvas(128)
    tall = np.full(((H + 100) // s, W // s), 128, dtype=np.uint8)
    assert run(small, tall, CENTRE).kind == "page"
    assert run(tall, small, CENTRE).kind == "page"


def test_page_when_jpeg_sizes_differ() -> None:
    a = jpeg(np.full((H, W), 200, dtype=np.uint8))
    b = jpeg(np.full((H + 100, W), 200, dtype=np.uint8))
    assert run_jpeg(a, b, CENTRE).kind == "page"


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("top right", "bottom left", "top right"),
        ("bottom left", "top right", "top right"),
        ("bottom right", "top", "top"),
        ("right", "left", "left"),
    ],
)
def test_region_tie_goes_to_reading_order(a: str, b: str, expected: str) -> None:
    """A2-3 rule 6: equal changed-cell counts in two ninths -> the first in reading order."""
    check_constants()
    cells = aligned_cells(a, 1) + aligned_cells(b, 1)
    clear_of_centre(cells)
    before, after = strong(cells)
    assert label_of(run(before, after, CENTRE)) == ("elsewhere", expected)


@pytest.mark.parametrize(
    ("many", "few", "expected"),
    [("bottom", "top left", "bottom"), ("bottom right", "top", "bottom right")],
)
def test_region_is_the_ninth_with_most_changed_cells(many: str, few: str, expected: str) -> None:
    """A2-3 rule 6: two changed cells in one ninth beat one earlier in reading order."""
    check_constants()
    cells = aligned_cells(many, 2) + aligned_cells(few, 1)
    clear_of_centre(cells)
    before, after = strong(cells)
    assert label_of(run(before, after, CENTRE)) == ("elsewhere", expected)


def boundary_case(side: str, inside: bool) -> tuple[tuple[int, int, int, int], Point, str]:
    """A cell and a click point with the cell's nearest edge exactly on the closed near box (``inside``) or one array
    pixel beyond it; returns (cell, point, the cell's ninth). Hand geometry: the box is +-320 x +-80 at 1280x800."""
    cf = cell_full()
    s = scale()
    off = 0 if inside else s
    if side == "right":  # cell left edge b, box right edge px + 320
        b = ceil(900 / cf) * cf
        y0 = (400 // cf) * cf
        cell, point = (b, y0, b + cf, y0 + cf), (b - W // 4 - off, y0 + (cf // 2 // s) * s)
    elif side == "left":  # cell right edge e (last pixel e - 1), box left edge px - 320
        e = (380 // cf) * cf
        y0 = (400 // cf) * cf
        cell, point = (e - cf, y0, e, y0 + cf), (e - s + W // 4 + off, y0 + (cf // 2 // s) * s)
    elif side == "below":  # cell top edge b, box bottom edge py + 80
        b = ceil(540 / cf) * cf
        x0 = (640 // cf) * cf
        cell, point = (x0, b, x0 + cf, b + cf), (x0 + (cf // 2 // s) * s, b - H // 10 - off)
    else:  # above: cell bottom edge e (last pixel e - 1), box top edge py - 80
        e = (266 // cf) * cf
        x0 = (640 // cf) * cf
        cell, point = (x0, e - cf, x0 + cf, e), (x0 + (cf // 2 // s) * s, e - s + H // 10 + off)
    assert in_near_box(cell, point) is inside, (side, inside, cell, point)
    assert 0 <= point[0] < W and 0 <= point[1] < H, point
    return cell, point, ninth_of(cell)


@pytest.mark.parametrize("side", ["right", "left", "below", "above"])
@pytest.mark.parametrize("inside", [True, False], ids=["on-edge", "one-px-out"])
def test_near_box_boundary_plus_minus_one_px(side: str, inside: bool) -> None:
    """A2-3 rule 5 at the edge (mutation 4, adversarial off-by-one): a strong changed cell whose nearest pixel is
    exactly ``+-W/4`` (or ``+-H/10``) from the click is ``near``; one array pixel further it is ``elsewhere`` in its
    ninth. ``pointers`` is the click only, and the cell is strong, so the hover zone plays no part."""
    check_constants()
    cell, point, ninth = boundary_case(side, inside)
    before, after = strong([cell])
    o = run(before, after, point)
    assert label_of(o) == (("near", None) if inside else ("elsewhere", ninth)), (o, cell, point)


@pytest.mark.parametrize("fraction", ["0.25", "0.10"])
def test_near_box_scales_with_the_frame(fraction: str) -> None:
    """A2-3 rule 5: the near box is a fraction of the frame, not fixed pixels -- on a 2560x1600 frame a cell 400 px
    right of (or 120 px below) the click is still near (320/80 px would miss it)."""
    check_constants()
    s = scale()
    cf = cell_full()
    big_w, big_h = 2 * W, 2 * H
    before = np.zeros((big_h // s, big_w // s), dtype=np.uint8)
    after = before.copy()
    y0, x0 = (800 // cf) * cf, (1280 // cf) * cf
    if fraction == "0.25":
        cell, point = (x0, y0, x0 + cf, y0 + cf), (x0 - 400, y0)
    else:
        cell, point = (x0, y0, x0 + cf, y0 + cf), (x0, y0 - 120)
    after[cell[1] // s : cell[3] // s, cell[0] // s : cell[2] // s] = 255
    o = run(before, after, point)
    assert o.kind == "near", o


# =====================================================================================================
# The adversarial rows of "Tests to write (A2)"
# =====================================================================================================


CHECKBOX = (640, 384, 656, 400)  # 16x16 full px, on every power-of-two grid up to 128


def test_checkbox_at_the_click_is_near() -> None:
    """Checkbox (adversarial; the mutant fully masks the hover zone, mutation 2): a strong 16x16 change exactly at
    the click point -- inside its hover zone -- gives ``near``."""
    check_constants()
    s, cf = scale(), cell_full()
    per_cell = (min(cf, 16) // s) ** 2
    assert const("MIN_CELL_PIXELS") <= per_cell, (
        f"T-0136 A2-3: MIN_CELL_PIXELS={const('MIN_CELL_PIXELS')} cannot see a 16 px checkbox ({per_cell} array px "
        "per cell)"
    )
    before = canvas(250)
    fill(before, (640, 384, 656, 386), 120)  # the unticked box's border
    after = before.copy()
    fill(after, CHECKBOX, 40)  # ticked: filled dark
    fill(after, (644, 392, 652, 394), 250)  # the tick
    click = (648, 392)
    for pointers in ((click,), ((100, 700), click)):
        o = run(before, after, click, pointers)
        assert o.kind == "near", (o, pointers)


def test_hover_brightness_on_a_button_under_the_pointer_is_none() -> None:
    """Hover (adversarial; the mutant drops the zone, mutation 1): CSS ``brightness(1.08)`` on a button-sized rect
    (100x32 full px, light-grey face, dark label) centred on the pointer gives ``none``. The rect lies inside
    ``+-HOVER_HALF`` of the pointer (a zone narrower than a button cannot tolerate hover at all)."""
    check_constants()
    assert const("HOVER_HALF_W") >= 50 and const("HOVER_HALF_H") >= 16, (
        "T-0136 A2-3: the hover zone must cover a button under the pointer; "
        f"HOVER_HALF_W={const('HOVER_HALF_W')}, HOVER_HALF_H={const('HOVER_HALF_H')}"
    )
    s = scale()
    pointer = (648, 392)
    button = (598, 376, 698, 408)
    before = canvas(245)
    fill(before, button, 230)
    for y in range(388, 396, s):  # the label's strokes
        fill(before, (620, y, 676, y + s), 55)
    after = before.copy()
    x0, y0, x1, y1 = button
    face = before[y0 // s : y1 // s, x0 // s : x1 // s].astype(np.float64)
    after[y0 // s : y1 // s, x0 // s : x1 // s] = np.clip(np.rint(face * 1.08), 0, 255).astype(
        np.uint8
    )
    assert int(np.abs(after.astype(np.int64) - before.astype(np.int64)).max()) == 18  # 230 -> 248
    for pointers in ((pointer,), (pointer, (100, 700)), ((100, 700), pointer)):
        o = run(before, after, pointer if pointers[-1] == pointer else pointers[-1], pointers)
        assert o.kind == "none", (o, pointers)


def test_hover_delta_just_outside_the_zone_counts() -> None:
    """The zone too wide (the companion of the hover rows): a change of exactly ``HOVER_DELTA`` (above
    ``NOISE_DELTA``) on a whole cell just beyond the previous pointer's zone, clear of the click's near box, is
    ``elsewhere`` in its ninth -- not tolerated."""
    check_constants()
    cf = cell_full()
    hd = int(const("HOVER_DELTA"))
    prev: Point = (64, 400)
    click: Point = (1100, 700)
    x = prev[0] + int(const("HOVER_HALF_W")) * scale() + scale()
    x = ceil(x / cf) * cf + cf
    y = (384 // cf) * cf
    while any(x < b < x + cf for b in (Fraction(W, 3), Fraction(2 * W, 3))):
        x += cf  # the next cell wholly inside one ninth
    cell = (x, y, x + cf, y + cf)
    assert cell[2] <= W, (
        f"premise: HOVER_HALF_W={const('HOVER_HALF_W')} leaves no room right of the pointer"
    )
    assert not in_outer_zone(cell, prev) and not in_outer_zone(cell, click), (
        "premise: outside both zones"
    )
    assert not in_near_box(cell, click, margin=cf), "premise: clear of the near box"
    before = canvas(0)
    after = before.copy()
    fill(after, cell, hd)
    o = run(before, after, click, (prev, click))
    assert label_of(o) == ("elsewhere", ninth_of(cell)), (o, cell)


SUBSET = [
    r
    for r in (json.loads(line) for line in LABELS.read_text().splitlines() if line.strip())
    if r["subset"]
]
JPEG_SOURCES = (
    "7eff7112c610dbb198aa6b0a484aae808db49b0dd6682be173bdc0dd62b58f34",  # the probe page
    "dcfa23d792dcd97f48d542d705013735c1b8c689b331a0ee4e6572f1b61c5cc4",  # Mugs
    "935f09a570e94816d247a02edc29e0af1b7f97cce64f46300fc73b91fd87e5ad",  # search results
)


@pytest.mark.parametrize("sha", JPEG_SOURCES, ids=["probe", "mugs", "results"])
def test_jpeg_reencode_is_none(sha: str) -> None:
    """JPEG re-encode (adversarial; the mutant compares bytes, mutation 3): one rendered page saved at quality 80
    and at 90 -- different bytes, same picture -- gives ``none`` in both orders, wherever the click."""
    from PIL import Image

    with Image.open(FRAMES / f"{sha}.jpg") as img:
        rgb = img.convert("RGB")
    q80, q90 = io.BytesIO(), io.BytesIO()
    rgb.save(q80, format="JPEG", quality=80)
    rgb.save(q90, format="JPEG", quality=90)
    assert q80.getvalue() != q90.getvalue(), "premise: the bytes differ"
    for point in (CENTRE, (100, 100), (1200, 60)):
        assert run_jpeg(q80.getvalue(), q90.getvalue(), point).kind == "none", point
        assert run_jpeg(q90.getvalue(), q80.getvalue(), point).kind == "none", point


def frame_bytes(sha: str) -> bytes:
    return (FRAMES / f"{sha}.jpg").read_bytes()


@pytest.mark.parametrize(
    "rec",
    SUBSET,
    ids=[
        f"{r['pair_id']}-{r['label']}" + (f"-{r['region']}" if r["region"] else "") for r in SUBSET
    ],
)
def test_committed_subset_pairs_match_their_labels(rec: dict[str, Any]) -> None:
    """A2-5 / "Tests to write": the committed corpus pairs give their hand label (kind and region), through
    ``outcome_of_jpeg`` with B-2's pointers (the previous executed pointer point, if any; then the click).
    A2b (F3): the landmine pairs (``LANDMINES``) are pending until A2b is implemented; the A2 pairs are not."""
    point = (int(rec["point"][0]), int(rec["point"][1]))
    prev = rec["prev_point"]
    pointers = ((int(prev[0]), int(prev[1])), point) if prev else (point,)
    o = run_jpeg(frame_bytes(rec["before"]), frame_bytes(rec["after"]), point, pointers)
    assert label_of(o) == (rec["label"], rec["region"]), (o, rec["note"])


# =====================================================================================================
# A2-1: purity
# =====================================================================================================


def test_import_loads_no_pil() -> None:
    """A2-1: importing the module pulls in no PIL (a fresh interpreter; PIL is imported inside ``gray`` only)."""
    ao()
    code = (
        "import sys; before = set(sys.modules); "
        "import local_astra.conductor.action_outcome; "
        "print(sorted(m for m in set(sys.modules) - before if m.split('.')[0] in ('PIL', 'asyncio', 'httpx')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=REPO
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", (
        f"T-0136 A2-1: importing action_outcome loads {out.stdout.strip()}"
    )


def test_module_is_pure() -> None:
    """A2-1: numpy and the standard library only at import, PIL only inside ``gray``, no I/O or clock modules, no
    project imports, no ``open``/``print`` calls (an AST check of the source)."""
    ao()
    tree = ast.parse(MODULE_PATH.read_text())
    found: list[tuple[str, str | None]] = []

    def visit(node: ast.AST, fn: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                visit(child, child.name)
                continue
            if isinstance(child, ast.Import):
                found.extend((a.name.split(".")[0], fn) for a in child.names)
            elif isinstance(child, ast.ImportFrom):
                assert child.level == 0, "T-0136 A2-1: no relative imports"
                found.append(((child.module or "").split(".")[0], fn))
            visit(child, fn)

    visit(tree, None)
    mods = {m for m, _ in found} - {"__future__"}
    allowed = set(sys.stdlib_module_names) | {"numpy", "PIL"}
    assert mods <= allowed, f"T-0136 A2-1: non-stdlib imports {sorted(mods - allowed)}"
    pil_sites = {fn for m, fn in found if m == "PIL"}
    assert pil_sites <= {"gray"}, f"T-0136 A2-1: PIL imported outside gray(): {pil_sites}"
    banned = {"os", "pathlib", "socket", "subprocess", "time", "datetime", "random", "asyncio", "threading",
              "logging", "shutil", "tempfile", "urllib", "http", "local_astra"}  # fmt: skip
    assert not (mods & banned), f"T-0136 A2-1: pure; imports {sorted(mods & banned)}"
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert not (called & {"open", "print", "input", "eval", "exec"}), sorted(called)


# =====================================================================================================
# The fixtures (green at BASE_A2: they pin the labels file and the committed frames, not the detector)
# =====================================================================================================


def all_labels() -> list[dict[str, Any]]:
    return [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]


HEX64 = frozenset("0123456789abcdef")


def is_sha256(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 64 and set(s) <= HEX64


def test_labels_cover_every_distinct_pair_once() -> None:
    """A2b-1 (re-pinned to corpus 610f36f): 442 distinct pairs (260 carried from 20104cc, 182 new: 176 page-b
    probes and 6 jitter-drifted replay pairs), each labelled once, with the closed classes, A2-2's region coupling
    and the A2b-1 fields: ``group`` = ``"<before cluster>:<after cluster>"`` (two sha256), ``trivial`` exactly when
    the two clusters are equal, ``stress``/``seen``/``subset``/``carried`` booleans, ``kind`` a string (``replay``
    for replay pairs), ``split`` in {dev, test}."""
    recs = all_labels()
    ids = [r["pair_id"] for r in recs]
    assert len(recs) == 442 and len(set(ids)) == 442
    for r in recs:
        assert (
            len(r["pair_id"]) == 16
            and int(r["pair_id"], 16) >= 0
            and r["pair_id"] == r["pair_id"].lower()
        )
        assert r["label"] in KINDS, r
        assert (r["region"] in REGIONS) if r["label"] == "elsewhere" else (r["region"] is None), r
        assert r["split"] in ("dev", "test"), r
        assert r["frame_kind"] in ("live", "overview"), r
        assert isinstance(r["point"], list) and len(r["point"]) == 2, r
        b, sep, a = r["group"].partition(":")
        assert sep == ":" and is_sha256(b) and is_sha256(a), r["group"]
        assert r["trivial"] is (b == a), r
        for k in ("stress", "seen", "subset", "carried"):
            assert isinstance(r[k], bool), (k, r)
        assert isinstance(r["kind"], str) and r["kind"], r
        assert (r["kind"] == "replay") is (r["probe_page"] is None), r
        assert r["probe_page"] in (None, "outcomes.html", "outcomes_b.html"), r
        assert not r["stress"] or r["probe_page"] == "outcomes_b.html", r
    overview = [r for r in recs if r["frame_kind"] == "overview"]
    assert len(overview) == 6 and all(r["label"] == "page" for r in overview)
    counts = {k: sum(r["label"] == k for r in recs) for k in KINDS}
    assert counts == {"none": 182, "near": 125, "elsewhere": 81, "page": 54}, counts
    assert sum(r["probe_page"] == "outcomes_b.html" for r in recs) == 176
    assert sum(not r["carried"] for r in recs) == 182


def groups_of(recs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        out.setdefault(r["group"], []).append(r)
    return out


def test_split_is_by_frame_group() -> None:
    """A2b-2: one split per group; seen groups (and every group holding a committed-subset pair) are dev; unseen
    groups are stratified by class (the label, or the sorted label set when mixed), ordered by sha256 of the group
    string and assigned test, dev, test, dev, ... -- reproduced here from ``group`` and the labels alone.
    *Adversarial, mutation 20:* the pair-id-parity split disagrees with this on the committed labels."""
    recs = all_labels()
    groups = groups_of(recs)
    for gid, rs in groups.items():
        assert len({r["split"] for r in rs}) == 1, (gid, [(r["pair_id"], r["split"]) for r in rs])
        assert len({r["seen"] for r in rs}) == 1, gid
        if any(r["subset"] for r in rs):
            assert rs[0]["seen"], f"T-0136 A2b-2: a subset group must be seen: {gid}"
        if rs[0]["seen"]:
            assert rs[0]["split"] == "dev", f"T-0136 A2b-2 (mutation 22): seen group in test: {gid}"
    by_class: dict[str, list[str]] = {}
    for gid, rs in groups.items():
        if not rs[0]["seen"]:
            by_class.setdefault("+".join(sorted({r["label"] for r in rs})), []).append(gid)
    expected: dict[str, str] = {}
    for gids in by_class.values():
        gids.sort(key=lambda s: hashlib.sha256(s.encode()).hexdigest())
        for i, gid in enumerate(gids):
            expected[gid] = "test" if i % 2 == 0 else "dev"
    got = {gid: rs[0]["split"] for gid, rs in groups.items() if not rs[0]["seen"]}
    assert got == expected, sorted(g for g in got if got[g] != expected[g])[:10]
    parity = {r["pair_id"]: ("dev" if int(r["pair_id"], 16) % 2 == 0 else "test") for r in recs}
    assert any(parity[r["pair_id"]] != r["split"] for r in recs), (
        "premise: the parity split is not the rule"
    )
    assert len(groups) == 226, len(groups)


def test_carried_labels_unchanged() -> None:
    """A2b-1 / A2b-N (negative): every 20104cc pair id that survives keeps its label and region from the A2 labels
    at ``LABELS_A2`` (a fixed commit); the 14 vanished ids are dropped; the 6 jitter-drifted replay pairs inherit
    their parent's label; ``carried`` is true exactly for the surviving ids."""
    gr.require_commits(LABELS_A2)
    proc = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{LABELS_A2}:tests/fixtures/t0136/labels.jsonl"],
        capture_output=True,
        text=True,
        check=True,
    )
    old = {r["pair_id"]: r for r in map(json.loads, proc.stdout.splitlines()) if r}
    assert len(old) == 274
    recs = {r["pair_id"]: r for r in all_labels()}
    survived = set(old) & set(recs)
    assert set(old) - set(recs) == DROPPED_20104CC
    assert len(survived) == 260
    for pid, r in recs.items():
        assert r["carried"] is (pid in survived), pid
    for pid in survived:
        assert (recs[pid]["label"], recs[pid]["region"]) == (
            old[pid]["label"],
            old[pid]["region"],
        ), pid
    for pid, (parent, label, region) in INHERITED.items():
        assert parent in old and (old[parent]["label"], old[parent]["region"]) == (label, region), (
            parent
        )
        assert (recs[pid]["label"], recs[pid]["region"]) == (label, region), pid
        assert recs[pid]["kind"] == "replay" and not recs[pid]["carried"], pid
    new_replay = {pid for pid, r in recs.items() if r["kind"] == "replay" and not r["carried"]}
    assert new_replay == set(INHERITED), sorted(new_replay ^ set(INHERITED))


#: A2b-3 minima on the test split, in independent groups (never lowered; short -> A1d adds probes).
TEST_MINIMA = {"none": 20, "near": 20, "elsewhere": 10, "page": 8}


def test_test_split_minimum_counts() -> None:
    """A2b-3 (a property of the labels, not pending): the test split holds >= 20 non-trivial non-stress ``none``
    groups, >= 20 ``near``, >= 10 ``elsewhere`` and >= 8 ``page`` groups (live pairs), and no test group is seen."""
    groups = groups_of([r for r in all_labels() if r["frame_kind"] == "live"])
    n = dict.fromkeys(TEST_MINIMA, 0)
    for gid, rs in groups.items():
        if rs[0]["split"] != "test":
            continue
        assert not rs[0]["seen"], gid
        if rs[0]["trivial"] or any(r["stress"] for r in rs):
            continue
        for label in {r["label"] for r in rs}:
            n[label] += 1
    short = {k: (n[k], m) for k, m in TEST_MINIMA.items() if n[k] < m}
    assert not short, (
        f"T-0136 A2b-3 / Q-A2b-2: test-split groups short {short}; stop for an A1d, never lower n"
    )


def page_b_probes() -> dict[str, dict[str, Any]]:
    import re

    html = (FIXTURES / "outcomes_b.html").read_text()
    m = re.search(
        r'<script type="application/json" id="probes">\s*(\[.*?\])\s*</script>', html, re.S
    )
    assert m, "the page-b probe table"
    return {p["id"]: p for p in json.loads(m.group(1))}


def test_committed_subset_has_the_landmines() -> None:
    """F3 (A2b, "Tests to write"): the committed subset includes n-hover-3's underline pair, the Start -> Stop
    relabel (c-label-3), >= 2 page-b long-link (>= 400 px) underlines, >= 3 C-SMALL pairs, 1 C-COLOUR pair and 1
    N-PRE pair, each with the hand label its kind implies; every subset group is seen and dev."""
    recs = all_labels()
    sub = [r for r in recs if r["subset"]]
    by_id = {r["pair_id"]: r for r in sub}
    assert set(LANDMINES) <= set(by_id), sorted(set(LANDMINES) - set(by_id))
    probes = page_b_probes()

    def probe_of(r: dict[str, Any]) -> str:
        return r["source"].split()[1]

    assert (
        probe_of(by_id["5e772b5f725f3376"]) == "n-hover-3"
        and by_id["5e772b5f725f3376"]["label"] == "none"
    )
    assert (
        probe_of(by_id["7915c1b58d13ca93"]) == "c-label-3"
        and by_id["7915c1b58d13ca93"]["label"] == "near"
    )
    b = [r for r in sub if r["probe_page"] == "outcomes_b.html"]
    long_ul = [r for r in b if r["kind"] == "n-ul" and probes[probe_of(r)]["width_px"] >= 400]
    assert len(long_ul) >= 2 and all(r["label"] == "none" for r in long_ul), long_ul
    small = [r for r in b if r["kind"].startswith("c-small-")]
    assert len(small) >= 3 and all(r["label"] == "near" for r in small), small
    assert sum(r["kind"] == "c-colour" and r["label"] == "near" for r in b) >= 1
    assert sum(r["kind"] == "n-pre" and r["label"] == "none" for r in b) >= 1
    subset_groups = {r["group"] for r in sub}
    for r in recs:
        if r["group"] in subset_groups:
            assert r["seen"] and r["split"] == "dev", r["pair_id"]


def test_committed_frames_are_the_subset_and_under_2_mb() -> None:
    """The committed subset: every frame a subset record names is present, named by its own sha256, nothing else is
    committed, the total is at most 2 MB, and every class is represented (stratified)."""
    subset = [r for r in all_labels() if r["subset"]]
    named = {r[k] for r in subset for k in ("before", "after")}
    on_disk = {p.stem for p in FRAMES.glob("*.jpg")}
    assert named == on_disk, sorted(named ^ on_disk)
    total = 0
    for sha in on_disk:
        data = (FRAMES / f"{sha}.jpg").read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha, sha
        total += len(data)
    assert total <= 2 * 1024 * 1024, total
    per = {k: sum(r["label"] == k for r in subset) for k in KINDS}
    assert all(n >= 3 for n in per.values()), per


# =====================================================================================================
# A2-7: negative (green at BASE_A2)
# =====================================================================================================


def test_a2_changes_only_the_detector_and_nothing_imports_it() -> None:
    """A2-7: over ``BASE_A2..TIP_A2`` (two literals) the only ``local_astra/`` change is the new module, and at
    ``TIP_A2`` no other ``local_astra/`` file mentions ``action_outcome``."""
    gr.require_commits(BASE_A2, TIP_A2)
    assert gr.is_ancestor(BASE_A2, TIP_A2), (BASE_A2, TIP_A2)
    assert set(gr.changed(BASE_A2, TIP_A2, "local_astra")) <= {
        "local_astra/conductor/action_outcome.py"
    }
    proc = subprocess.run(
        ["git", "-C", str(REPO), "grep", "-l", "action_outcome", TIP_A2, "--", "local_astra"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stderr
    hits = {line.split(":", 1)[1] for line in proc.stdout.split()}
    assert hits <= {"local_astra/conductor/action_outcome.py"}, sorted(hits)


def test_a2_scope_predicate_has_power() -> None:
    """The A2-7 predicates see a change when there is one: T-0151 A1 (cad8538) changed ``local_astra/``, and
    ``git grep`` at a commit finds a known importer (``degeneracy`` is imported by ``session.py`` at BASE_A2)."""
    gr.require_commits("d49fa5b", "cad8538", BASE_A2)
    assert gr.changed("d49fa5b", "cad8538", "local_astra") != []
    proc = subprocess.run(
        ["git", "-C", str(REPO), "grep", "-l", "degeneracy", BASE_A2, "--", "local_astra"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert f"{BASE_A2}:local_astra/conductor/session.py" in proc.stdout.split(), proc.stdout


# =====================================================================================================
# A2b-4: rule 3' (the thin-row tolerance in the pointer row bands), its closed edges (R5/R6)
# =====================================================================================================


def band_rows(py: int) -> tuple[int, int]:
    """Array rows ``j`` whose full px ``DRAFT_SCALE * j`` lie in the closed band ``[py - HH, py + HH]`` (R5/R6)."""
    s = scale()
    hh = int(const("HOVER_HALF_H"))
    return max(0, ceil((py - hh) / s)), min(H // s - 1, floor((py + hh) / s))


def thin_line(rows: range, x0: int, x1: int, delta: int) -> tuple[np.ndarray, np.ndarray]:
    """Frames that differ only by ``+delta`` on array rows ``rows`` and array columns ``[x0, x1)``."""
    before = canvas(0)
    after = before.copy()
    after[rows.start : rows.stop, x0:x1] = delta
    return before, after


def equivalent(note: str) -> None:
    """Record (print, never assert) that a named mutant is equivalent at the module's constants."""
    print(f"\nT-0136 A2b equivalence at these constants: {note}")


@HYP
@given(data=st.data())
def test_thin_change_in_a_row_band_is_none_at_any_x(data: st.DataObject) -> None:
    """A2b-4 property: a change confined, within each cell, to <= ``UNDERLINE_ROWS`` array rows and lying inside
    some pointer's row band gives ``none`` -- at any |delta| above the noise and at any x, far beyond
    ``HOVER_HALF_W`` included. The band may be the previous pointer's."""
    check_constants()
    u = underline_rows()
    s = scale()
    nd = int(const("NOISE_DELTA"))
    point = draw_point(data)
    pointers = (point,) if data.draw(st.booleans()) else (draw_point(data), point)
    owner = pointers[data.draw(st.integers(0, len(pointers) - 1))]
    lo, hi = band_rows(owner[1])
    t = data.draw(st.integers(1, u))
    assume(hi - lo + 1 >= t)
    j0 = data.draw(st.integers(lo, hi - t + 1))
    x0 = data.draw(st.integers(0, W // s - 1))
    x1 = data.draw(st.integers(x0 + 1, W // s))
    delta = data.draw(st.integers(nd + 1, 255))
    before, after = thin_line(range(j0, j0 + t), x0, x1, delta)
    o = run(before, after, point, pointers)
    assert o.kind == "none", (o, pointers, owner, (j0, t), (x0, x1), delta)


@pytest.mark.parametrize("thickness", ["one-row", "UNDERLINE_ROWS"])
def test_long_underline_with_the_pointer_at_its_start_is_none(thickness: str) -> None:
    """*Adversarial, mutation 13* (the band clipped to the zone's x-extent): a 600 px underline from x=40 to x=640,
    with the pointer at its start, just under the pointer's row, gives ``none``."""
    check_constants()
    u = underline_rows()
    s = scale()
    hh = int(const("HOVER_HALF_H"))
    pointer: Point = (40, 400)
    assert 40 + int(const("HOVER_HALF_W")) + cell_full() < 640, (
        "premise: the line reaches a whole cell past the zone's right edge"
    )
    t = 1 if thickness == "one-row" else u
    j0 = 400 // s + 1
    assert s * (j0 + t - 1) <= 400 + hh, "premise: the rows lie in the band"
    before, after = thin_line(range(j0, j0 + t), 40 // s, 640 // s, 200)
    for pointers in ((pointer,), ((1200, 100), pointer)):
        o = run(before, after, pointer, pointers)
        assert o.kind == "none", (o, pointers)


@HYP
@given(data=st.data())
def test_thin_change_outside_every_band_counts(data: st.DataObject) -> None:
    """A2b-4 property (mutation 14, rule 3' applied outside the bands): the same thin change, outside every
    pointer's row band by at least one array row, with >= ``MIN_CELL_PIXELS`` changed pixels per cell, is counted:
    ``near`` when one of its cells meets the near box, else ``elsewhere`` in its ninth (hand geometry)."""
    check_constants()
    u = underline_rows()
    s = scale()
    c = int(const("CELL_PX"))
    cf = cell_full()
    hd = int(const("HOVER_DELTA"))
    t = data.draw(st.integers(1, u))
    assume(t * c >= const("MIN_CELL_PIXELS"))
    i = data.draw(st.integers(0, 8))
    r, col = divmod(i, 3)
    xa, xb = aligned_span(Fraction(W * col, 3), Fraction(W * (col + 1), 3), cf)
    ya, yb = aligned_span(Fraction(H * r, 3), Fraction(H * (r + 1), 3), cf)
    ncell = data.draw(st.integers(1, (xb - xa) // cf))
    gx = data.draw(st.integers(0, (xb - xa) // cf - ncell))
    gy = data.draw(st.integers(0, (yb - ya) // cf - 1))
    x0f, x1f = xa + gx * cf, xa + (gx + ncell) * cf
    top = ya + gy * cf
    j0 = top // s + data.draw(st.integers(0, c - t))
    cells_rect = (x0f, top, x1f, top + cf)
    assert ninth_of(cells_rect) == REGIONS[i]
    point = draw_point(data)
    pointers = (point,) if data.draw(st.booleans()) else (draw_point(data), point)
    for p in pointers:
        lo, hi = band_rows(p[1])
        assume(j0 + t - 1 < lo - 1 or j0 > hi + 1)
    assume(ncell * s * s < const("PAGE_FRACTION") * total_cells_lower_bound())
    delta = data.draw(st.integers(hd + 1, 255))
    before, after = thin_line(range(j0, j0 + t), x0f // s, x1f // s, delta)
    o = run(before, after, point, pointers)
    expected = ("near", None) if in_near_box(cells_rect, point) else ("elsewhere", REGIONS[i])
    assert label_of(o) == expected, (o, cells_rect, (j0, t), point, pointers)


@pytest.mark.parametrize("extra", [0, 1], ids=["UNDERLINE_ROWS", "one-more-row"])
def test_underline_rows_boundary(extra: int) -> None:
    """A2b-4 (mutation 15, the ``UNDERLINE_ROWS`` comparison off by one): a strong change of exactly
    ``UNDERLINE_ROWS`` rows within each cell, in the pointer's band, is tolerated (``none``); one row more is
    counted (``near``: it sits at the click)."""
    check_constants()
    u = underline_rows()
    s = scale()
    c = int(const("CELL_PX"))
    cf = cell_full()
    hh = int(const("HOVER_HALF_H"))
    k = u + extra
    py = (400 // cf) * cf  # the top row of a cell, so all k rows share one cell row
    pointer: Point = (640, py)
    assert s * (k - 1) <= hh, "premise: the rows lie in the band"
    x0f = (640 // cf) * cf - 2 * cf
    before, after = thin_line(range(py // s, py // s + k), x0f // s, (x0f + 4 * cf) // s, 255)
    o = run(before, after, pointer)
    if k * c < const("MIN_CELL_PIXELS"):
        equivalent(
            f"{k} rows x {c} columns < MIN_CELL_PIXELS: rule 3 alone leaves the cell unchanged"
        )
        assert o.kind == "none", o
    elif extra == 0:
        assert o.kind == "none", (o, "exactly UNDERLINE_ROWS rows in the band must be tolerated")
    else:
        assert o.kind == "near", (o, "UNDERLINE_ROWS + 1 rows must be counted")


@pytest.mark.parametrize("edge", ["bottom", "top"])
@pytest.mark.parametrize("inside", [True, False], ids=["on-the-edge-row", "one-row-out"])
def test_band_edges_closed_in_full_px(edge: str, inside: bool) -> None:
    """R5/R6 (mutation 18, an open bottom or top edge on the band): a thin strong change whose only band row is
    the one at full px exactly ``py + HOVER_HALF_H`` (bottom) or ``py - HOVER_HALF_H`` (top) is tolerated; the same
    change one array row further out is counted. The pointer is far left; the change is far right."""
    check_constants()
    u = underline_rows()
    s = scale()
    c = int(const("CELL_PX"))
    hh = int(const("HOVER_HALF_H"))
    t = u
    rc = (H // s // c) // 2  # a cell row in the middle of the frame
    if edge == "bottom":
        je = rc * c  # first row of a cell
        py = s * je - hh
        rows = range(je, je + t) if inside else range(je + 1, je + 1 + t)
    else:
        je = rc * c + c - 1  # last row of a cell
        py = s * je + hh
        rows = range(je - t + 1, je + 1) if inside else range(je - t, je)
    assert 0 <= py < H, py
    assert rows.start // c == (rows.stop - 1) // c == rc, "premise: one cell row"
    cols = (W // s // c) * 3 // 4 * c
    before, after = thin_line(rows, cols, cols + 4 * c, 255)
    o = run(before, after, (100, py))
    if inside:
        assert o.kind == "none", (o, edge, "the edge row is inside the closed band")
    elif t * c < const("MIN_CELL_PIXELS"):
        equivalent(f"band {edge} edge: {t} x {c} pixels < MIN_CELL_PIXELS")
        assert o.kind == "none", o
    else:
        assert o.kind in ("near", "elsewhere"), (
            o,
            edge,
            "one row outside the band must be counted",
        )


@pytest.mark.parametrize("inside", [True, False], ids=["on-the-edge-column", "one-column-out"])
def test_zone_right_edge_closed_in_full_px(inside: bool) -> None:
    """R5 (mutation 18, an open right edge on the hover zone): a change of exactly ``HOVER_DELTA`` (above the
    noise) on one array column, a whole cell tall (taller than ``UNDERLINE_ROWS``), at full px exactly
    ``px + HOVER_HALF_W`` is inside the closed zone (``none``); one array column further it is counted."""
    check_constants()
    underline_rows()
    s = scale()
    c = int(const("CELL_PX"))
    hw, hh = int(const("HOVER_HALF_W")), int(const("HOVER_HALF_H"))
    hd = int(const("HOVER_DELTA"))
    q = ceil(hw / (s * c)) + 1
    ie = q * c  # first column of a cell
    px = s * ie - hw
    assert 0 <= px < W and s * (ie + c) <= W, (px, ie)
    rc = (H // s // c) // 2
    py = s * rc * c
    assert s * (c - 1) <= hh, "premise: the cell's rows lie inside the zone's y-extent"
    col = ie if inside else ie + 1
    before = canvas(0)
    after = before.copy()
    after[rc * c : rc * c + c, col] = hd
    o = run(before, after, (px, py))
    if inside:
        assert o.kind == "none", (
            o,
            "|delta| == HOVER_DELTA on the zone's edge column is tolerated",
        )
    elif c < const("MIN_CELL_PIXELS"):
        equivalent(f"zone right edge: one column of {c} pixels < MIN_CELL_PIXELS")
        assert o.kind == "none", o
    else:
        assert o.kind in ("near", "elsewhere"), (o, "one column outside the zone must be counted")


def test_zone_bottom_edge_closed_in_full_px() -> None:
    """R5 (reviewer F3; mutation 18 on the zone's bottom edge, e.g. ``y1 = (py + HOVER_HALF_H - 1) // s + 1``): the
    zone's last row is the array row at full px exactly ``py + HOVER_HALF_H``. That edge row carries
    ``|delta| == HOVER_DELTA`` across a cell (inside the closed zone: not changed); edge + 1 is unchanged; edge +
    ``UNDERLINE_ROWS`` (edge + 2 at the tip) carries one strong pixel, fewer than ``MIN_CELL_PIXELS`` alone. The
    closed zone gives ``none``. An open bottom edge changes the edge row too, and the cell then spans
    ``UNDERLINE_ROWS + 1`` rows (not thin, so rule 3' cannot tolerate it) and is counted."""
    check_constants()
    u = underline_rows()
    s = scale()
    c = int(const("CELL_PX"))
    hh = int(const("HOVER_HALF_H"))
    hd = int(const("HOVER_DELTA"))
    rc = (H // s // c) // 2
    je = (
        rc * c
    )  # the edge row is the first row of a cell; edge + UNDERLINE_ROWS stays in that cell (c > u)
    py = s * je - hh
    px = 640
    assert 0 <= py < H
    x0 = (px // s // c) * c
    before = canvas(0)
    after = before.copy()
    after[je, x0 : x0 + c] = hd
    after[je + u, x0] = 255
    o = run(before, after, (px, py))
    if const("MIN_CELL_PIXELS") < 2:
        equivalent("MIN_CELL_PIXELS < 2: the lone pixel below the edge is counted either way")
        assert o.kind == "near", o
    else:
        assert o.kind == "none", (o, "the edge row at py + HOVER_HALF_H is inside the closed zone")


@pytest.mark.parametrize("inside", [False, True], ids=["row-above-band", "first-band-row"])
def test_band_first_row_at_an_odd_offset(inside: bool) -> None:
    """R6 (reviewer F3; floor-for-ceil on the band's first array row): with ``py - HOVER_HALF_H`` odd, the band's
    first array row ``j`` has full px ``DRAFT_SCALE * j == py - HOVER_HALF_H + 1``; the row above sits at
    ``py - HOVER_HALF_H - 1``, outside the closed band. A strong thin change (``UNDERLINE_ROWS`` rows) ending on
    that row is counted (``near``: about 33 px above the click); the same change starting on the first band row is
    tolerated (``none``). Flooring the first row pulls the row above into the band and tolerates the first case."""
    check_constants()
    u = underline_rows()
    s = scale()
    c = int(const("CELL_PX"))
    hh = int(const("HOVER_HALF_H"))
    rc = (H // s // c) // 2
    jt = (rc + 1) * c  # the band's first row: the first row of a cell
    py = hh + s * jt - 1
    assert 0 <= py < H
    if s == 1:
        equivalent("DRAFT_SCALE 1: py - HOVER_HALF_H is always a whole row; floor and ceil agree")
    else:
        assert (py - hh) % s != 0 and ceil((py - hh) / s) == jt, (py, hh, s)
    rows = range(jt, jt + u) if inside else range(jt - u, jt)
    px = 640
    x0 = (px // s // c) * c
    before, after = thin_line(rows, x0, x0 + c, 255)
    o = run(before, after, (px, py))
    if inside:
        assert o.kind == "none", (o, "a thin change on the band's first row is tolerated")
    elif u * c < const("MIN_CELL_PIXELS"):
        equivalent(f"{u} x {c} pixels < MIN_CELL_PIXELS")
        assert o.kind == "none", o
    else:
        assert o.kind == "near", (o, "the row above the band is outside it: the thin change counts")


# =====================================================================================================
# A2b: exact boundaries of rules 3 and 4 (R7)
# =====================================================================================================


@pytest.mark.parametrize(
    "offset", [0, -1], ids=["n0", "n0-minus-1"]
)  # holds at 967a5f5: not pending
def test_page_fraction_boundary_exact(offset: int) -> None:
    """Rule 4 at its boundary (mutation 16, ``>=`` becomes ``>``): with n0 = ceil(PAGE_FRACTION * cells) whole
    cells changed the outcome is ``page``; with n0 - 1 it is not (``near`` or ``elsewhere``). ``cells`` is the
    outcome's own count, checked to be the floor or ceiling grid of R3; ``Fraction`` keeps the product exact."""
    check_constants()
    s = scale()
    c = int(const("CELL_PX"))
    h, w = H // s, W // s
    probe = run(canvas(0), canvas(0), CENTRE)
    floor_n, ceil_n = (h // c) * (w // c), ceil(h / c) * ceil(w / c)
    assert probe.cells in (floor_n, ceil_n), (probe.cells, floor_n, ceil_n)
    product = Fraction(const("PAGE_FRACTION")) * probe.cells
    n0 = ceil(product)
    n = n0 + offset
    full_cols = w // c
    assert n0 <= (h // c) * full_cols, "premise: n0 whole cells fit"
    before = canvas(0)
    after = before.copy()
    for k in range(n):
        r, q = divmod(k, full_cols)
        after[r * c : r * c + c, q * c : q * c + c] = 255
    o = run(before, after, (1279, 799))
    if product.denominator != 1:
        equivalent(
            f"PAGE_FRACTION * cells = {product} is not an integer: the >-for->= mutant is equivalent"
        )
    if offset == 0:
        assert o.kind == "page", (o, n0, product)
    elif n >= 1:
        assert o.kind in ("near", "elsewhere"), (o, n, product)
    else:
        assert o.kind == "none", o


@pytest.mark.parametrize(
    "offset", [0, -1], ids=["MIN_CELL_PIXELS", "one-fewer"]
)  # holds at 967a5f5
def test_min_cell_pixels_boundary_exact(offset: int) -> None:
    """Rule 3 at its boundary (mutation 17, ``>=`` becomes ``>``): exactly ``MIN_CELL_PIXELS`` strong pixels in
    one cell, far from every pointer, band and near box, give ``elsewhere (top left)``; one fewer gives ``none``.
    The pixels fill the cell column by column, so they span more than ``UNDERLINE_ROWS`` rows once there are more."""
    check_constants()
    s = scale()
    c = int(const("CELL_PX"))
    cf = cell_full()
    m = int(const("MIN_CELL_PIXELS"))
    assert m <= c * c, "premise: MIN_CELL_PIXELS fits in a cell"
    cy0 = cx0 = 2 * c
    assert ninth_of((cx0 * s, cy0 * s, cx0 * s + cf, cy0 * s + cf)) == "top left"
    before = canvas(0)
    after = before.copy()
    for k in range(m + offset):
        after[cy0 + k % c, cx0 + k // c] = 255
    pointer: Point = (1200, 700)
    o = run(before, after, pointer)
    if offset == 0:
        assert label_of(o) == ("elsewhere", "top left"), o
    else:
        assert o.kind == "none", o


# =====================================================================================================
# A2b-7 (F5): declared JPEG sizes and tiny frames
# =====================================================================================================


SIZE_CASES = [((1281, 801), (1280, 800)), ((1280, 801), (1280, 800)), ((1279, 800), (1280, 800))]


@pytest.mark.parametrize(
    ("a", "b"),
    [
        SIZE_CASES[0],
        SIZE_CASES[1],
        SIZE_CASES[2],
    ],
    ids=["1281x801", "1280x801", "1279x800"],
)
def test_jpeg_declared_size_off_by_one_is_page(a: tuple[int, int], b: tuple[int, int]) -> None:
    """A2b-7 (a), mutation 19 (the size test on decoded arrays only): two uniform JPEGs of the same level whose
    declared sizes differ by one pixel give ``page`` in both orders (at ``DRAFT_SCALE`` 2 the decoded arrays of
    1281 and 1280 px are equal in size, so only the declared size can tell)."""
    ja = jpeg(np.full((a[1], a[0]), 200, dtype=np.uint8))
    jb = jpeg(np.full((b[1], b[0]), 200, dtype=np.uint8))
    assert run_jpeg(ja, jb, CENTRE).kind == "page"
    assert run_jpeg(jb, ja, CENTRE).kind == "page"


@pytest.mark.parametrize(
    "size",
    [
        (1, 1),
        (1, 2),
        (2, 1),
        (3, 3),
    ],
    ids=lambda s: f"{s[0]}x{s[1]}",
)
def test_tiny_jpeg_does_not_raise(size: tuple[int, int]) -> None:
    """A2b-7 (b): ``gray()`` decodes a tiny JPEG without raising and returns a 2-D array; ``outcome_of_jpeg``
    returns a vocabulary outcome: ``none`` for identical frames, ``page`` for different sizes (both orders)."""
    w, h = size
    j = jpeg(np.full((h, w), 120, dtype=np.uint8))
    other = jpeg(np.full((h + 1, w + 1), 120, dtype=np.uint8))
    try:
        g = api("gray")(j)
        cases = [(x, y, want, api("outcome_of_jpeg")(x, y, (0, 0), ((0, 0),)))
                 for x, y, want in ((j, j, "none"), (j, other, "page"), (other, j, "page"))]  # fmt: skip
    except (ArithmeticError, ValueError, IndexError) as exc:
        raise AssertionError(f"T-0136 A2b-7 (b): a decodable {w}x{h} JPEG raised {exc!r}") from exc
    assert isinstance(g, np.ndarray) and g.ndim == 2, type(g)
    for _x, _y, want, o in cases:
        assert o.kind == want and o.text in VOCAB, (o, want, size)
