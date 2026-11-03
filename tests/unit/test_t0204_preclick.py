"""T-0204 group B at the seam: the pre-click "did the page move?" guard of ``BrowserToolExecutor`` over scripted fake
drivers, no browser (docs/tasks/T-0204.md, B1-B7; "Tests to write" B).

**The capability these rows pin** (B1, from group A's driver): ``screenshot_region(x, y, w, h) -> bytes`` (async, a
JPEG of the viewport clipped to the rectangle, clamped), ``arm_shift_recorder() -> bool`` (async),
``shift_mark() -> object`` and ``shifts_since(mark) -> tuple[rect, ...] | None`` (both synchronous), all four looked
up on the driver's *class*. The guard's constants are module constants of ``executor`` read at call time:
``PRECLICK_GUARD = True``, ``PRECLICK_HALF_PX = 64``, ``PRECLICK_SHIFT_MARGIN_PX = 40``, ``PRECLICK_BLOCK_DELTA = 16``,
``PRECLICK_CHANGED_SHARE = 0.10``, ``PRECLICK_TIMEOUT_S = 0.3``, ``PRECLICK_REF_FRAMES = 4``,
``MAX_MOVED_REFUSALS = 2`` and ``MOVED_REASON`` (== ``session.FRESHNESS_REASON``).

**The fake.** ``GuardMixin`` adds the four methods to T-0111c's ``FakeDriver`` (or to a T-0151/T-0160/T-0174 fake):

- the page is ``page_image()``: for ``GuardDriver`` a hand-drawn PIL image (``self.page``) that ``screenshot()`` also
  encodes (JPEG quality 85, as ``runtime.JPEG_QUALITY``), so a row "moves the page" by drawing on ``self.page``; for
  the other fakes the decoded ``fake_jpeg`` of the current scene;
- ``screenshot_region`` clamps the rectangle to the viewport as A1 does, raises ``ValueError`` when nothing is
  inside, crops ``page_image()`` and encodes the crop on its own (a fresh JPEG, so its 8x8 grid is not the frame's);
  ``region_script`` items override one call each: an exception to raise, ``("sleep", s)`` then the crop, or
  ``"hang"`` (until cancelled);
- ``arm_shift_recorder`` answers ``arm_ok``; ``shift_mark`` issues a fresh object and remembers how many
  ``screenshot`` calls preceded it; ``shifts_since`` answers ``shift_script`` items one per call (``None``, a tuple of
  ``(x, y, w, h)`` rects), then ``shift_default``.

Every call is recorded in ``calls`` (the four included), so BASE's sequence is the record with the four removed.

**Oracles.** BASE is the executor at ``BASE_SHA`` (group A's branch: main with T-0203 B, before any B code), run
from ``git show`` as T-0203's differential does; with the phash salt patched, a negative row compares the driver
call sequence, outcomes, bytes and phashes byte for byte. Geometry is by hand: ``to_pixels(500, 1280) = 640``,
``to_pixels(500, 800) = 400``, so a click at (500, 500) has the box (576, 336, 128, 128); grown by the margin it is
x 536-744, y 296-504. ``to_pixels(50, 1280) = 64``, ``to_pixels(50, 800) = 40``; ``to_pixels(502, 1280) = 643``,
``to_pixels(496, 800) = 397`` (a crop whose origin (579, 333) is off the frame's 8x8 JPEG grid). Changed shares are
counted by hand from blocks painted on the 8x8 grid of the crop (16 x 16 = 256 blocks): 32 blocks = 0.125, 24 blocks
= 0.09375. Luma deltas use greys (formula-free): 128 -> 140 is 12, 128 -> 152 is 24.

**Readings** (reported, not guessed silently):

- B4 "the next guarded call is not checked": pinned as no ``screenshot_region`` and no ``shifts_since`` call, and
  the note's ``shift`` and ``changed_share`` are ``None``.
- B5 "no reference": nothing was measured, so the note's ``shift`` and ``changed_share`` are ``None``.
- B2/B5 overlap: a shift near the point together with a crop that raises is not pinned (B2 says moved, B5 says fail
  open); escalated to the orchestrator.
- Orchestrator ruling (2026-10-29): zero-area rects (``{0,0,0,0}``) are dropped from the shift test, and a source that
  slid fully out still refuses through its non-zero ``previousRect``.
- "Arming at most one round trip per page" is the driver's (A3, idempotent); here only "arm before the kept
  screenshot" and "a newly adopted page is armed before its first capture" are pinned, not the number of arm calls.
- ``page_overview`` as a run reset (B4) is not pinned here (it needs T-0151's page fake under the guard mixin); the
  five other resets are.

**T-0234 B (named re-pins).** Since T-0234 a shift rect near the point decides only when the crop could not be
taken (B2); with a crop, pixels decide (B1). Every row below that used a shift alone on a static page to obtain a
moved refusal now makes the crop fail (``region_script`` gets a ``Boom``) so the shift decides by BASE's rule:
``test_shift_near_point_refuses``, ``test_margin_read_at_call_time``, ``test_zero_area_rect_dropped_near_origin``,
``test_cap_allows_third_and_notes``, ``test_cap_read_at_call_time``, ``test_executed_action_resets_run``,
``test_other_refusals_neither_count_nor_reset``, ``test_new_turn_resets``, ``test_google_search_does_not_reset_run`` and
``test_same_phash_recapture_after_seen_change_executes[far-whole-viewport]``. The inverse (a shift near the point
with an unchanged crop executes) is ``tests/unit/test_t0234_refused_reask_and_preclick.py``.

Regime: CPU over fakes, real asyncio time (budgets patched small where a row waits; well under 2 s in total),
1280x800, n = 1, deterministic salt.
"""

from __future__ import annotations

import asyncio
import io
import json
import secrets
import sys
import time
import types
from collections import deque
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from local_astra.computer import executor as executor_mod
from local_astra.conductor.tool_executor import to_pixels
from local_astra.contracts.perception import Frame
from local_astra.contracts.toolcall import FrameGeometry, ToolInvocation, ToolOutcome
from tests import _git_range as gr
from tests.unit.test_computer_executor import CLOCK_MS, FakeDriver
from tests.unit.test_computer_web_executor import FieldDriver
from tests.unit.test_t0151_executor import PageDriver, register
from tests.unit.test_t0174_navigated_executor import NavMixin

#: group B's BASE: group A's branch head (main with T-0203 B merged and T-0204 A), before any B code. A literal.
BASE_SHA = "e715c33994a8321cc0b9a1b18d8677825ea91379"
SALT = 0x5EED_0204_0204_5EED
W, H = 1280, 800
#: typed from the task (H-T204-1) and session.FRESHNESS_REASON as merged (T-0111g B9); never imported for the oracle
FRESHNESS = "the screen changed since this frame; take a new frame"
NOTE_WHAT = "live click: page moved check"
NOTE_KEYS = {"what", "result", "shift", "changed_share", "ms"}
GUARD = ("screenshot_region", "arm_shift_recorder", "shift_mark", "shifts_since")
INPUT = frozenset(
    {"mouse_move", "mouse_down", "mouse_up", "key_down", "key_up", "type_text", "scroll_by", "goto"}
)

#: (500, 500) -> (640, 400); its box and the grown box (by hand, module docstring)
CLICK = {"x": 500, "y": 500}
TYPE = {"x": 500, "y": 500, "text": "kettle", "press_enter": False}
POINT = (640, 400)
BOX = (576, 336, 128, 128)
#: the uniform grey block under POINT on ``block_page`` (x 440-839, y 250-549)
BLOCK = (440, 250, 839, 549)
GREY = 128


# ============================================================================================ pages


def _font(size: int) -> Any:
    return ImageFont.load_default(size=size)


_PAGES: dict[str, Image.Image] = {}


def detailed_page() -> Image.Image:
    """A busy page: hairline grids and dark text of five sizes at fixed pseudo-random places (seeded, by hand); the
    worst case for JPEG re-encoding noise (sharp edges everywhere)."""
    if "detailed" not in _PAGES:
        import random

        rnd = random.Random(7)
        img = Image.new("RGB", (W, H), (250, 250, 250))
        d = ImageDraw.Draw(img)
        for i in range(0, W, 40):
            d.line([(i, 0), (i, H)], fill=(200, 200, 210), width=1)
        for j in range(0, H, 40):
            d.line([(0, j), (W, j)], fill=(200, 210, 200), width=1)
        words = "Vinland Saga Vol. 7 - EUR 12.99 - add to cart"
        for _ in range(400):
            x, y = rnd.randrange(0, W - 60), rnd.randrange(0, H - 20)
            size = rnd.choice([11, 14, 18, 24, 32])
            colour = (rnd.randrange(0, 120), rnd.randrange(0, 120), rnd.randrange(0, 120))
            d.text((x, y), words[: rnd.randrange(3, len(words))], fill=colour, font=_font(size))
        _PAGES["detailed"] = img
    return _PAGES["detailed"].copy()


def block_page(grey: int = GREY) -> Image.Image:
    """``detailed_page`` with a uniform grey block under POINT (BLOCK): the crop at (500, 500) is all one grey."""
    img = detailed_page()
    ImageDraw.Draw(img).rectangle(BLOCK, fill=(grey, grey, grey))
    return img


def field_page(caret: bool) -> Image.Image:
    """``detailed_page`` with a white text field (x 340-940, y 380-420) under POINT and, with ``caret``, a 2x18 px
    black caret at x 639-640, y 391-408 (Chromium's blink, on or off)."""
    img = detailed_page()
    d = ImageDraw.Draw(img)
    d.rectangle((340, 380, 940, 420), fill=(255, 255, 255), outline=(120, 120, 120))
    d.text((350, 388), "vinland saga", fill=(20, 20, 20), font=_font(20))
    if caret:
        d.rectangle((639, 391, 640, 408), fill=(0, 0, 0))
    return img


def lazy_insert(img: Image.Image, at_y: int, px: int, colour: tuple[int, int, int]) -> Image.Image:
    """The page after a ``px``-high block was inserted at ``at_y``: everything from ``at_y`` down moves by ``px``."""
    out = img.copy()
    out.paste(img.crop((0, at_y, W, H - px)), (0, at_y + px))
    ImageDraw.Draw(out).rectangle((0, at_y, W - 1, at_y + px - 1), fill=colour)
    return out


def jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ============================================================================================ the fakes


class GuardMixin:
    """The four B1 capabilities over a fake driver (module docstring)."""

    calls: list[tuple[str, tuple[Any, ...]]]
    viewport: tuple[int, int]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.region_script: deque[Any] = deque()
        self.shift_script: deque[Any] = deque()
        self.shift_default: Any = ()
        self.arm_ok = True
        self.shift_marks: list[object] = []
        #: id(mark) -> the number of ``screenshot`` calls before it was issued
        self.mark_after: dict[int, int] = {}
        #: the marks ``shifts_since`` was asked about, in order
        self.asked: list[object] = []
        self.region_log: list[tuple[float, tuple[float, float, float, float]]] = []
        #: B-R4 rows: a mark-aware record (``None``: off). ``add_shift`` appends; ``shifts_since(mark)`` then answers the
        #: rects recorded after ``mark`` was issued (when ``shift_script`` is empty), as the real recorder does
        self.shift_log: list[tuple[float, float, float, float]] | None = None
        self.mark_log_at: dict[int, int] = {}
        self.moves_at: list[float] = []

    # -- what the page shows now
    def page_image(self) -> Image.Image:
        from tests.unit.test_computer_executor import fake_jpeg

        d: Any = self
        if d.collide is not None:
            data = fake_jpeg(self.viewport, *d.collide)
        else:
            data = fake_jpeg(self.viewport, d.generation, d.scene)
        return Image.open(io.BytesIO(data)).convert("RGB")

    def shots_so_far(self) -> int:
        return sum(1 for c in self.calls if c[0] == "screenshot")

    # -- the capability
    async def screenshot_region(self, x: float, y: float, w: float, h: float) -> bytes:
        self._enter("screenshot_region", x, y, w, h)  # type: ignore[attr-defined]
        self.region_log.append((time.monotonic(), (x, y, w, h)))
        item = self.region_script.popleft() if self.region_script else None
        if isinstance(item, BaseException):
            raise item
        if item == "hang":
            await asyncio.Event().wait()
        if isinstance(item, tuple) and item[0] == "sleep":
            await asyncio.sleep(item[1])
        vw, vh = self.viewport
        x0, y0 = max(0.0, float(x)), max(0.0, float(y))
        x1, y1 = min(float(vw), float(x) + float(w)), min(float(vh), float(y) + float(h))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"region {(x, y, w, h)} has nothing inside the viewport")
        crop = self.page_image().crop((round(x0), round(y0), round(x1), round(y1)))
        return jpeg(crop)

    async def arm_shift_recorder(self) -> bool:
        self._enter("arm_shift_recorder")  # type: ignore[attr-defined]
        return self.arm_ok

    def shift_mark(self) -> object:
        self.calls.append(("shift_mark", ()))
        m = object()
        self.shift_marks.append(m)
        self.mark_after[id(m)] = self.shots_so_far()
        self.mark_log_at[id(m)] = len(self.shift_log) if self.shift_log is not None else 0
        return m

    def add_shift(self, *rects: tuple[float, float, float, float]) -> None:
        if self.shift_log is None:
            self.shift_log = []
        self.shift_log.extend(rects)

    def shifts_since(self, mark: object) -> Any:
        self.calls.append(("shifts_since", (mark,)))
        self.asked.append(mark)
        if not any(mark is m for m in self.shift_marks):
            return None  # a foreign mark: the real driver answers None (A2)
        if self.shift_script:
            return self.shift_script.popleft()
        if self.shift_log is not None:
            return tuple(self.shift_log[self.mark_log_at[id(mark)] :])
        return self.shift_default

    async def mouse_move(self, x: float, y: float) -> None:
        self.moves_at.append(time.monotonic())
        await super().mouse_move(x, y)  # type: ignore[misc]


class GuardDriver(GuardMixin, FakeDriver):
    """A hand-drawn page (``self.page``): ``screenshot()`` encodes it, ``screenshot_region`` crops it."""

    def __init__(self, page: Image.Image | None = None) -> None:
        super().__init__()
        self.page = page if page is not None else detailed_page()

    def page_image(self) -> Image.Image:
        return self.page

    async def screenshot(self) -> bytes:
        self._enter("screenshot")
        self.shots.append(self.active)
        return jpeg(self.page)

    async def bring_to_front(self) -> None:
        self.calls.append(("bring_to_front", ()))


class GuardField(GuardMixin, FieldDriver):
    """T-0160's ``focused_field`` fake under the guard (the guarded type path)."""


class GuardNav(GuardMixin, NavMixin, FakeDriver):
    """T-0174's navigation capability under the guard (the mark/answer order)."""


class GuardPage(GuardMixin, PageDriver):
    """T-0151's overview fake under the guard."""


def missing(name: str) -> type:
    """``GuardDriver`` (whose four methods record their calls) with ``name`` absent from the class: the guard must not
    run at all (B1: all four, not any)."""
    return type(f"GuardWithout_{name}", (GuardDriver,), {name: None})


class ThreeOfFour(FakeDriver):
    """Three of the four B1 methods: ``shifts_since`` is missing, so the guard must not run (B1)."""

    async def screenshot_region(self, x: float, y: float, w: float, h: float) -> bytes:
        raise AssertionError("B1: the guard ran on a driver class without shifts_since")

    async def arm_shift_recorder(self) -> bool:
        raise AssertionError("B1: the guard armed a driver class without shifts_since")

    def shift_mark(self) -> object:
        raise AssertionError("B1: the guard took a mark on a driver class without shifts_since")


class Boom(RuntimeError):
    pass


# ============================================================================================ helpers


@pytest.fixture(autouse=True)
def fixed_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secrets, "randbits", lambda n: SALT)


@pytest.fixture(autouse=True)
def settle_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """T-0204 C10: these rows pin the guard alone, against BASE ``e715c339`` (before any settle), so the visual settle
    (group C, on by default for a driver class with the B1 capability) is switched off here. The guard and the settle
    together are pinned in ``tests/unit/test_t0204_settle.py`` (C8)."""
    monkeypatch.setattr(executor_mod, "STABLE_SCREEN", False, raising=False)


def const(name: str) -> Any:
    value = getattr(executor_mod, name, None)
    assert value is not None, f"T-0204 B: executor.{name} does not exist yet"
    return value


_BASE: types.ModuleType | None = None


def base_module() -> types.ModuleType:
    global _BASE
    if _BASE is None:
        gr.require_commits(BASE_SHA)
        src = gr._git("show", f"{BASE_SHA}:local_astra/computer/executor.py").stdout
        assert "def _register" in src and "screenshot_region" not in src, (
            "premise: BASE has no pre-click guard"
        )
        name = "_t0204b_base_executor"
        mod = types.ModuleType(name)
        mod.__file__ = f"<{BASE_SHA[:8]}>/local_astra/computer/executor.py"
        sys.modules[name] = mod
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        _BASE = mod
    return _BASE


def build(
    d: Any, *, module: Any = executor_mod, notes: list[dict[str, Any]] | None = None, **kw: Any
) -> Any:
    ex = module.BrowserToolExecutor(now_ms=lambda: CLOCK_MS, viewport=d.viewport, driver=d, **kw)
    if notes is not None:
        ex.on_note = notes.append
    return ex


_N = iter(range(1_000_000))


def call(
    name: str, frame: Frame | None = None, *, turn: str = "t1", **arguments: Any
) -> ToolInvocation:
    g = (
        FrameGeometry(
            phash=frame.phash, source=frame.source, width=frame.width, height=frame.height
        )
        if frame is not None
        else None
    )
    return ToolInvocation(
        call_id=f"g{next(_N)}",
        turn_id=turn,
        name=name,
        arguments=arguments,
        source="draft",
        frame=g,
    )


async def own(ex: Any) -> Frame:
    f = await ex.frame()
    assert isinstance(f, Frame)
    return f


async def act(
    ex: Any, tool: str, args: dict[str, Any], frame: Frame | None = None, *, turn: str = "t1"
) -> ToolOutcome:
    if tool in ("click_at", "type_at") and frame is None:
        frame = ex.latest_frame
        if frame is None:
            frame = await own(ex)
    out = await ex.execute(
        call(tool, frame if tool in ("click_at", "type_at") else None, turn=turn, **args)
    )
    assert isinstance(out, ToolOutcome)
    return out


def names(d: Any, since: int = 0) -> list[str]:
    return [c[0] for c in d.calls[since:]]


def count(d: Any, name: str, since: int = 0) -> int:
    return sum(1 for c in d.calls[since:] if c[0] == name)


def checked(d: Any, since: int) -> None:
    """The call was checked (one region call and one shift read since ``since``): a "not refused" row is about the
    guard's verdict, not about a guard that never ran."""
    seq = names(d, since)
    assert seq.count("screenshot_region") == 1 and seq.count("shifts_since") == 1, (
        f"T-0204 B2: the guarded call was not checked (one region call, one shift read): {seq}"
    )


def order(seq: list[str], *want: str) -> list[int]:
    missing = [w for w in want if w not in seq]
    assert not missing, f"T-0204 B2: {missing} not called: {seq}"
    return [seq.index(w) for w in want]


def dispatched(d: Any, since: int = 0) -> list[tuple[str, tuple[Any, ...]]]:
    return [c for c in d.calls[since:] if c[0] in INPUT]


def guard_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """B6's notes, each checked for the exact shape."""
    out = [n for n in notes if n.get("what") == NOTE_WHAT]
    for n in out:
        assert set(n) == NOTE_KEYS, f"T-0204 B6: the note has exactly {sorted(NOTE_KEYS)}: {n}"
        assert n["result"] in ("refused", "allowed_after_cap", "unavailable"), n
        assert n["shift"] is None or type(n["shift"]) is bool, n
        assert n["changed_share"] is None or type(n["changed_share"]) is float, n
        assert type(n["ms"]) is int and n["ms"] >= 0, n
    return out


def assert_moved_refusal(out: ToolOutcome, tool: str = "click_at") -> None:
    reason = const("MOVED_REASON")
    assert (out.status, out.reason) == ("refused", reason), out
    assert out.action == f"refused {tool}: {reason}", out.action
    assert out.resolved == {}, f"B3/B6: a refusal carries no resolved key: {out.resolved}"


def _norm(calls: list[tuple[str, tuple[Any, ...]]]) -> list[Any]:
    out: list[Any] = []
    for name, args in calls:
        out.append([name, [a if isinstance(a, (int, float, str)) else repr(a) for a in args]])
    return out


async def trace(
    module: Any, make: Any, steps: list[tuple[str, dict[str, Any]]], **kw: Any
) -> dict[str, Any]:
    """One executor over a fresh driver: a first frame, then ``steps``; the call sequence, outcomes, kept bytes and
    phashes (what B7 compares with BASE)."""
    d = make()
    ex = build(d, module=module, **kw)
    rec: dict[str, Any] = {"outs": [], "frames": []}
    await own(ex)
    for tool, args in steps:
        out = await act(ex, tool, args)
        rec["outs"].append([out.status, out.action, out.reason, out.resolved])
        rec["frames"].append([ex.latest_frame.phash, ex.latest_frame.jpeg.hex()])
    rec["calls"] = _norm(d.calls)
    rec["driver"] = d
    return rec


def same(
    got: dict[str, Any], want: dict[str, Any], what: str, *, drop: frozenset[str] = frozenset()
) -> None:
    for key in ("outs", "frames", "calls"):
        g = got[key]
        if key == "calls" and drop:
            g = [c for c in g if c[0] not in drop]
        assert json.dumps(g, default=str) == json.dumps(want[key], default=str), (
            f"T-0204 B7 {what}: {key} differs from BASE {BASE_SHA[:8]}:\n got  {g}\n base {want[key]}"
        )


# ============================================================================================ B1-B3: constants


def test_constants_exports_and_reason_text() -> None:
    """B1-B4, H-T204-1..3: the constants at their proposed defaults, exported; ``MOVED_REASON`` is the session's
    freshness text byte for byte (both sides pinned against the typed literal)."""

    want = {
        "PRECLICK_GUARD": True,
        "PRECLICK_HALF_PX": 64,
        "PRECLICK_SHIFT_MARGIN_PX": 40,
        "PRECLICK_BLOCK_DELTA": 16,
        "PRECLICK_CHANGED_SHARE": 0.10,
        "PRECLICK_TIMEOUT_S": 0.3,
        "PRECLICK_REF_FRAMES": 4,
        "MAX_MOVED_REFUSALS": 2,
        "MOVED_REASON": FRESHNESS,
    }
    for name, value in want.items():
        got = getattr(executor_mod, name, None)
        assert got is not None, f"T-0204 B: executor.{name} does not exist yet"
        assert got == value and type(got) is type(value), (
            f"T-0204 B: {name} = {got!r}, want {value!r}"
        )
        assert name in executor_mod.__all__, f"T-0204 B: {name} is not in executor.__all__"


def test_reason_equals_session_freshness_text() -> None:
    """B3 (H-T204-1, T-0187's pattern): the moved refusal reuses the session's text, so ``session.py`` needs no
    change: ``executor.MOVED_REASON == session.FRESHNESS_REASON`` byte for byte, and that text is one of T-0187's
    "take a new frame" reasons."""
    from local_astra.conductor import session

    reason = const("MOVED_REASON")
    assert session.FRESHNESS_REASON == FRESHNESS, "premise: the session's text as merged"
    assert reason.encode() == session.FRESHNESS_REASON.encode()
    assert reason in session.NEW_FRAME_REASONS, "premise: T-0187 refreshes the frame after it"


# ============================================================================================ B1: references


async def test_arm_before_kept_screenshot_and_mark_after() -> None:
    """B1: every own capture arms before its kept screenshot and reads ``shift_mark()`` right after it. Re-pinned for
    ruling B-R4: a same-page capture with the same phash (a scroll's after-frame of a page that did not change)
    REPLACES that phash's reference, so the guard's ``shifts_since`` gets the newest capture's mark; a newer own frame
    with another phash leaves the first frame's reference (and mark) alone."""
    d = GuardDriver()
    ex = build(d)
    f = await own(ex)
    seq = names(d)
    shot = seq.index("screenshot")
    assert "arm_shift_recorder" in seq[:shot], f"B1: armed before the kept screenshot: {seq}"
    assert "shift_mark" in seq[shot + 1 :], f"B1: the mark is read after the kept screenshot: {seq}"
    first_mark = d.shift_marks[0]
    assert d.mark_after[id(first_mark)] == 1, "B1: the mark is read after the first screenshot"
    # the same pixels captured again on the same page: B-R4, the newest capture and mark win
    out = await act(ex, "scroll", {"direction": "down"})
    assert out.status == "executed" and out.resolved["frame_after"] == f.phash, (
        f"premise: the after-frame has the same phash: {out}"
    )
    newest = d.shift_marks[-1]
    assert newest is not first_mark and d.mark_after[id(newest)] == 2, names(d)
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    assert out.status == "executed", out
    asked = [c[1][0] for c in d.calls[since:] if c[0] == "shifts_since"]
    assert len(asked) == 1 and asked[0] is newest, (
        "B-R4: a same-phash recapture replaces the reference: shifts_since gets the newest capture's mark"
    )
    # a newer own frame with another phash (a far corner changed) does not touch f's reference
    keep = d.shift_marks[-1]
    ImageDraw.Draw(d.page).rectangle((0, 0, 299, 199), fill=(0, 0, 0))
    g = await own(ex)
    assert g.phash != f.phash, "premise: a distinct own frame"
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    assert out.status == "executed", out
    asked = [c[1][0] for c in d.calls[since:] if c[0] == "shifts_since"]
    assert len(asked) == 1 and asked[0] is keep, (
        "B1: another phash's capture leaves f's reference alone"
    )


async def test_new_page_armed_before_its_first_capture() -> None:
    """B1: a page adopted by ``ensure()`` (``page_serial`` +1) is armed before its first kept screenshot."""
    d = GuardDriver()
    ex = build(d)
    await own(ex)
    d.open_popup()
    since = len(d.calls)
    await own(ex)
    seq = names(d, since)
    assert d.page_serial == 1, "premise: frame()'s ensure() adopted the popup"
    e, shot = order(seq, "ensure", "screenshot")
    assert "arm_shift_recorder" in seq[e:shot], (
        f"B1: the new page is armed after ensure() and before its kept screenshot: {seq}"
    )


async def test_reference_lru_bound() -> None:
    """B1/B5: at most ``PRECLICK_REF_FRAMES = 4`` references, oldest dropped first. Five own frames of a page that
    differs only far from the point: a click on the first is unchecked (no reference: ``unavailable``, executed as
    BASE); a click on the second is checked with its own mark and passes silently."""
    assert const("PRECLICK_REF_FRAMES") == 4
    d = GuardDriver(detailed_page())
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    frames: list[Frame] = []
    marks: list[object] = []
    corners = [None, (0, 0), (980, 0), (0, 600), (980, 600)]
    for c in corners:
        page = detailed_page()
        if c is not None:
            ImageDraw.Draw(page).rectangle((c[0], c[1], c[0] + 299, c[1] + 199), fill=(0, 0, 0))
        d.page = page
        frames.append(await own(ex))
        marks.append(d.shift_marks[-1])
    assert len({f.phash for f in frames}) == 5, "premise: five distinct own frames"
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, frames[0])
    assert out.status == "executed", f"B5: no reference fails open: {out}"
    assert len(dispatched(d, since)) == 3, "B5: the click went through as at BASE"
    got = guard_notes(notes)
    assert [(n["result"], n["shift"], n["changed_share"]) for n in got] == [
        ("unavailable", None, None)
    ], f"B5/B6: one 'unavailable' note, nothing measured: {notes}"
    notes.clear()
    d.page = detailed_page()  # frames[1]'s pixels at the point (the far corner block differs only)
    ImageDraw.Draw(d.page).rectangle((0, 0, 299, 199), fill=(0, 0, 0))
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, frames[1])
    assert out.status == "executed", out
    asked = [c[1][0] for c in d.calls[since:] if c[0] == "shifts_since"]
    assert len(asked) == 1 and asked[0] is marks[1], "B1: frames[1] kept its reference and mark"
    assert count(d, "screenshot_region", since) == 1
    assert guard_notes(notes) == [], f"B6: a plain 'not moved' writes no note: {notes}"


# ============================================================================================ B2/B7: the static click


@pytest.mark.parametrize(("tool", "args"), [("click_at", CLICK), ("type_at", TYPE)])
async def test_static_click_executes_one_region_call(tool: str, args: dict[str, Any]) -> None:
    """B2/B7: on a static page a guarded click/type adds exactly one ``screenshot_region`` (the box of half-size 64
    around the point, (576, 336, 128, 128)) and one synchronous ``shifts_since`` read, after ``ensure()`` and before
    the first ``mouse_move``; with those and the capture's arm/mark calls removed, the call sequence, the outcome,
    its ``resolved`` and the after-frame bytes and phash are BASE's. No note."""
    steps = [(tool, args)]
    notes: list[dict[str, Any]] = []
    got = await trace(executor_mod, GuardDriver, steps, notes=notes)
    want = await trace(base_module(), GuardDriver, steps)
    same(got, want, f"static {tool}", drop=frozenset(GUARD))
    d = got["driver"]
    first_click = max(i for i, c in enumerate(d.calls) if c[0] == "ensure")
    seq = names(d, first_click)
    assert seq.count("screenshot_region") == 1 and seq.count("shifts_since") == 1, (
        f"B7: exactly one region call and one shift read: {seq}"
    )
    r, s_, move = order(seq, "screenshot_region", "shifts_since", "mouse_move")
    assert r < move and s_ < move, f"B2: the check runs before the first mouse_move: {seq}"
    region = next(c[1] for c in d.calls[first_click:] if c[0] == "screenshot_region")
    assert tuple(round(float(v), 3) for v in region) == BOX, f"B2: the box around {POINT}: {region}"
    assert guard_notes(notes) == [], notes


async def test_check_after_front_and_before_guarded_type_click() -> None:
    """B2: with a headed executor the check runs after ``_front`` (``bring_to_front``) and before the first
    ``mouse_move``; on the guarded type path (``focused_field``) before ``_dispatch_guarded``'s click."""
    d = GuardDriver()
    ex = build(d, headless=False)
    await own(ex)
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK)
    assert out.status == "executed", out
    seq = names(d, since)
    a, b, c = order(seq, "bring_to_front", "screenshot_region", "mouse_move")
    assert a < b < c, seq

    g = GuardField()
    ex = build(g)
    await own(ex)
    since = len(g.calls)
    out = await act(ex, "type_at", TYPE)
    assert out.status == "executed", out
    seq = names(g, since)
    a, b, c = order(seq, "screenshot_region", "mouse_move", "focused_field")
    assert a < b < c, seq


async def test_check_after_navigation_mark_and_refusal_checks() -> None:
    """B2: the check runs after ``ensure()`` and every refusal check: with T-0174's capability the navigation mark is
    taken first; a page-changed refusal makes no region call at all."""
    d = GuardNav()
    ex = build(d)
    f = await own(ex)
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    assert out.status == "executed", out
    seq = names(d, since)
    a, b, c, e = order(seq, "ensure", "navigation_mark", "screenshot_region", "mouse_move")
    assert a < b < c < e, seq
    d.open_popup()
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    assert (out.status, out.reason) == ("refused", executor_mod.PAGE_CHANGED_REASON), out
    assert count(d, "screenshot_region", since) == 0 and count(d, "shifts_since", since) == 0, (
        f"B2: the page-changed refusal comes first: {names(d, since)}"
    )


# ============================================================================================ B2 (a): the shift signal


NEAR = [
    pytest.param(((630.0, 390.0, 20.0, 20.0),), id="at-point"),
    pytest.param(((720.0, 390.0, 10.0, 10.0),), id="margin-right"),
    pytest.param(((630.0, 300.0, 10.0, 10.0),), id="margin-top"),
    pytest.param(((540.0, 495.0, 10.0, 6.0),), id="margin-corner"),
    pytest.param(((0.0, 0.0, 1280.0, 800.0),), id="whole-viewport"),
    pytest.param(((900.0, 0.0, 50.0, 50.0), (600.0, 380.0, 30.0, 30.0)), id="second-rect"),
    pytest.param(((650.0, 380.0, 100.0, 40.0), (0.0, 0.0, 0.0, 0.0)), id="slid-out"),
]


@pytest.mark.parametrize("rects", NEAR)
@pytest.mark.parametrize(("tool", "args"), [("click_at", CLICK), ("type_at", TYPE)])
async def test_shift_near_point_refuses(rects: Any, tool: str, args: dict[str, Any]) -> None:
    """B2 (a)/B3, re-pinned by T-0234 B2: with the crop failing (a ``Boom``), a shift rect intersecting the box grown
    by the 40 px margin (x 536-744, y 296-504) refuses with ``MOVED_REASON``: nothing dispatched, no screenshot,
    ``latest_frame`` unchanged, one ``refused`` note with ``shift`` ``True``. ``slid-out`` (orchestrator ruling): a
    source that moved fully out of the viewport reads ``{0,0,0,0}`` as its current rect, and its non-zero previous
    rect still refuses. (Before T-0234 the crop was taken and unchanged here; that case now executes.)"""
    d = GuardDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.shift_script.append(rects)
    d.region_script.append(Boom("no crop (T-0234 B2: the shift decides)"))
    since = len(d.calls)
    out = await act(ex, tool, args, f)
    assert_moved_refusal(out, tool)
    assert dispatched(d, since) == [], f"B3: nothing dispatched: {dispatched(d, since)}"
    assert count(d, "screenshot", since) == 0, f"B3: no frame registered: {names(d, since)}"
    assert ex.latest_frame is f, "B3: latest_frame is unchanged"
    got = guard_notes(notes)
    assert [(n["result"], n["shift"]) for n in got] == [("refused", True)], notes


FAR = [
    pytest.param(((760.0, 390.0, 10.0, 10.0),), id="right-of-margin"),
    pytest.param(((630.0, 520.0, 10.0, 10.0),), id="below-margin"),
    pytest.param(((0.0, 0.0, 100.0, 100.0),), id="top-left"),
    pytest.param(((40.0, 600.0, 1200.0, 200.0), (40.0, 700.0, 1200.0, 100.0)), id="far-below"),
]


@pytest.mark.parametrize("rects", FAR)
async def test_shift_outside_margin_executes(rects: Any) -> None:
    """B2 (a)/B9.4 at the seam: shifts outside the grown box (x 536-744, y 296-504) and static pixels: executed as
    BASE, one region call, no note."""
    d = GuardDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.shift_script.append(rects)
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    assert out.status == "executed", out
    checked(d, since)
    assert len(dispatched(d, since)) == 3
    assert guard_notes(notes) == [], notes


async def test_margin_read_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """B2: ``PRECLICK_SHIFT_MARGIN_PX`` is read at call time: a rect 20 px right of the box (x 724) refuses at 40 and
    executes at 10. Re-pinned by T-0234 B2: the crop fails both times, so the shift decides."""
    rect = ((724.0, 390.0, 4.0, 4.0),)
    d = GuardDriver()
    ex = build(d)
    f = await own(ex)
    d.shift_script.append(rect)
    d.region_script.append(Boom("no crop"))
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    assert const("PRECLICK_SHIFT_MARGIN_PX") == 40
    monkeypatch.setattr(executor_mod, "PRECLICK_SHIFT_MARGIN_PX", 10)
    d.shift_script.append(rect)
    d.region_script.append(Boom("no crop"))
    out = await act(ex, "click_at", CLICK, f, turn="t2")
    assert out.status == "executed", out


async def test_zero_area_rect_dropped_near_origin() -> None:
    """Orchestrator ruling: a zero-area rect ``(0, 0, 0, 0)`` (a source that moved in or out of the viewport) is not a
    shift by itself. The point (64, 40) (``(50, 50)``) has a grown box that contains the origin; with a static page
    the click executes and writes no note. The same rect next to the previous rect of the source that slid out
    refuses (``test_shift_near_point_refuses[slid-out]``). The crop here is clamped to (0, 0, 128, 104), so the row
    also pins that the reference region is clamped the same way (else the pixels misalign). T-0234 B2 re-pin: the
    second call's crop fails, so its shift decides (with a crop, the unchanged pixels would let it execute); a third
    call, zero-area rect alone and a failed crop, executes ``unavailable`` (the zero-area rect still is no shift)."""
    d = GuardDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.shift_script.append(((0.0, 0.0, 0.0, 0.0),))
    since = len(d.calls)
    out = await act(ex, "click_at", {"x": 50, "y": 50}, f)
    assert out.status == "executed", out
    checked(d, since)
    assert guard_notes(notes) == [], notes
    d.shift_script.append(((20.0, 10.0, 60.0, 40.0), (0.0, 0.0, 0.0, 0.0)))
    d.region_script.append(Boom("no crop"))
    out = await act(ex, "click_at", {"x": 50, "y": 50}, f)
    assert_moved_refusal(out)
    notes.clear()
    d.shift_script.append(((0.0, 0.0, 0.0, 0.0),))
    d.region_script.append(Boom("no crop"))
    out = await act(ex, "click_at", {"x": 50, "y": 50}, f, turn="t2")
    assert out.status == "executed", f"the zero-area rect alone is no shift: {out}"
    assert [n["result"] for n in guard_notes(notes)] == ["unavailable"], notes


@pytest.mark.parametrize(
    "units", [pytest.param((999, 999), id="bottom-right"), pytest.param((0, 999), id="bottom-left")]
)
async def test_clamped_crop_at_viewport_edge_static_executes(units: tuple[int, int]) -> None:
    """B2 adversarial (alignment): at a viewport corner the crop is clamped; on a busy static page the click executes
    with no note, so the reference region is the same clamped rectangle."""
    d = GuardDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    since = len(d.calls)
    out = await act(ex, "click_at", {"x": units[0], "y": units[1]}, f)
    assert out.status == "executed", out
    checked(d, since)
    assert guard_notes(notes) == [], notes


# ============================================================================================ B2 (b): the pixel signal


@pytest.mark.parametrize(
    "shifts", [pytest.param((), id="no-shift"), pytest.param(None, id="shift-unknown")]
)
async def test_pixels_changed_refuses_without_shift(shifts: Any) -> None:
    """B2 (b), B5, threat model (transform moves, iframes, CDP failing): no shift entry (or ``None``: the pixel signal
    alone decides) and the grey block under the point turned dark grey: every block of the crop differs (luma 128 ->
    60), so the share is 1.0 (by hand) and the call is refused; the note says ``shift`` ``False`` (or ``None``) and
    ``changed_share`` 1.0."""
    d = GuardDriver(block_page())
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.page = block_page(60)
    d.shift_script.append(shifts)
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    assert_moved_refusal(out)
    assert dispatched(d, since) == [] and ex.latest_frame is f
    got = guard_notes(notes)
    assert len(got) == 1 and got[0]["result"] == "refused", notes
    assert got[0]["shift"] is (False if shifts == () else None), got[0]
    assert got[0]["changed_share"] == pytest.approx(1.0, abs=0.01), got[0]


async def test_lazy_insert_above_refuses_by_pixels() -> None:
    """B9.1 at the seam: a 300 px block inserted above the point (all content below y 200 moves down 300 px), no
    shift entry reported: refused by pixels."""
    d = GuardDriver(detailed_page())
    ex = build(d)
    f = await own(ex)
    d.page = lazy_insert(d.page, 200, 300, (200, 30, 30))
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))


@pytest.mark.parametrize(
    ("rect", "verdict"),
    [
        pytest.param((576, 336, 607, 399), "refused", id="32-blocks-0.125"),
        pytest.param((576, 336, 607, 383), "executed", id="24-blocks-0.094"),
    ],
)
async def test_changed_share_threshold(rect: tuple[int, int, int, int], verdict: str) -> None:
    """B2 (b) threshold ``PRECLICK_CHANGED_SHARE = 0.10``: a dark patch on the crop's 8x8 grid covering 4x8 = 32 of
    256 blocks (0.125) refuses; 4x6 = 24 blocks (0.094) does not (by hand; luma delta 68 per changed block)."""
    d = GuardDriver(block_page())
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    ImageDraw.Draw(d.page).rectangle(rect, fill=(60, 60, 60))
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    checked(d, since)
    if verdict == "refused":
        assert_moved_refusal(out)
        (n,) = guard_notes(notes)
        assert n["changed_share"] == pytest.approx(32 / 256, abs=0.02), n
    else:
        assert out.status == "executed", out
        assert guard_notes(notes) == [], notes


@pytest.mark.parametrize(
    ("grey", "verdict"),
    [pytest.param(140, "executed", id="delta-12"), pytest.param(152, "refused", id="delta-24")],
)
async def test_block_delta_threshold(grey: int, verdict: str) -> None:
    """B2 (b) threshold ``PRECLICK_BLOCK_DELTA = 16``: the whole grey block under the point brightens by 12 (every
    block's mean |delta| is 12 <= 16: not changed) or by 24 (every block changed: share 1.0)."""
    d = GuardDriver(block_page())
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.page = block_page(grey)
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    checked(d, since)
    if verdict == "refused":
        assert_moved_refusal(out)
    else:
        assert out.status == "executed", out
        assert guard_notes(notes) == [], notes


async def test_thresholds_read_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """B2: ``PRECLICK_CHANGED_SHARE`` and ``PRECLICK_BLOCK_DELTA`` are read at call time: the 24-block patch refuses
    once the share is 0.05; the +12 grey refuses once the delta is 8."""
    d = GuardDriver(block_page())
    ex = build(d)
    f = await own(ex)
    ImageDraw.Draw(d.page).rectangle((576, 336, 607, 383), fill=(60, 60, 60))
    assert const("PRECLICK_CHANGED_SHARE") == 0.10 and const("PRECLICK_BLOCK_DELTA") == 16
    monkeypatch.setattr(executor_mod, "PRECLICK_CHANGED_SHARE", 0.05)
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    monkeypatch.setattr(executor_mod, "PRECLICK_CHANGED_SHARE", 0.10)
    d2 = GuardDriver(block_page())
    ex2 = build(d2)
    f2 = await own(ex2)
    d2.page = block_page(140)
    monkeypatch.setattr(executor_mod, "PRECLICK_BLOCK_DELTA", 8)
    assert_moved_refusal(await act(ex2, "click_at", CLICK, f2))


@pytest.mark.parametrize("on_in_frame", [True, False], ids=["caret-then-off", "off-then-caret"])
async def test_caret_sized_change_not_refused(on_in_frame: bool) -> None:
    """Adversarial (a threshold too tight): a focused field whose 2x18 px caret blinks at the point between the frame
    and the crop. The caret changes about 6 blocks of 256 (each about 32 luma on a white field): not moved; the
    ``type_at`` executes with no note."""
    d = GuardDriver(field_page(caret=on_in_frame))
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.page = field_page(caret=not on_in_frame)
    since = len(d.calls)
    out = await act(ex, "type_at", TYPE, f)
    assert out.status == "executed", out
    checked(d, since)
    assert guard_notes(notes) == [], notes


@pytest.mark.parametrize(
    "units",
    [
        pytest.param((502, 496), id="off-grid-643-397"),
        pytest.param((79, 74), id="near-top-left"),
        pytest.param((948, 964), id="near-bottom-right"),
        pytest.param((260, 694), id="text-dense"),
    ],
)
async def test_jpeg_reencode_noise_not_refused(units: tuple[int, int]) -> None:
    """Adversarial (a threshold too tight): a busy static page (hairlines, text of five sizes). The crop is a fresh
    JPEG encode whose 8x8 grid is off the frame's, so the two decode with per-pixel differences up to about 30 luma
    (premise checked here) on identical pixels. Not moved: executed, no note."""
    d = GuardDriver(detailed_page())
    # premise: the noise is real (the two encodes differ), so the row would catch a threshold near zero
    x0 = max(0, to_pixels(units[0], W) - 64)
    y0 = max(0, to_pixels(units[1], H) - 64)
    crop = Image.open(io.BytesIO(jpeg(d.page.crop((x0, y0, x0 + 64, y0 + 64))))).convert("L")
    ref = Image.open(io.BytesIO(jpeg(d.page))).convert("L").crop((x0, y0, x0 + 64, y0 + 64))
    diff = int(np.abs(np.asarray(crop, dtype=np.int16) - np.asarray(ref, dtype=np.int16)).max())
    assert diff >= 8, f"premise: re-encoding noise is present ({diff})"
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    since = len(d.calls)
    out = await act(ex, "click_at", {"x": units[0], "y": units[1]}, f)
    assert out.status == "executed", out
    checked(d, since)
    assert guard_notes(notes) == [], notes


# ============================================================================================ B3: refusal


@pytest.mark.parametrize(("tool", "args"), [("click_at", CLICK), ("type_at", TYPE)])
async def test_refusal_dispatches_nothing_registers_nothing(
    tool: str, args: dict[str, Any]
) -> None:
    """B3: on "moved" no mouse, keyboard or scroll call, no screenshot (no frame registered), ``latest_frame`` and its
    phash unchanged, and the frame the call carried is still an own frame (a later static call on it executes)."""
    d = GuardDriver(block_page())
    ex = build(d)
    f = await own(ex)
    d.page = block_page(60)
    since = len(d.calls)
    out = await act(ex, tool, args, f)
    assert_moved_refusal(out, tool)
    assert dispatched(d, since) == [], dispatched(d, since)
    assert count(d, "screenshot", since) == 0 and count(d, "settle", since) == 0, names(d, since)
    assert ex.latest_frame is f and ex.latest_frame.phash == f.phash
    d.page = block_page()
    out = await act(ex, tool, args, f, turn="t2")
    assert out.status == "executed", out


# ============================================================================================ B4: the loop bound


async def test_cap_allows_third_and_notes() -> None:
    """B4/B6: a page moving near the point (a shift every call): two refusals, then the third call in the turn is not
    checked (no region call, no shift read), runs as BASE and is noted ``allowed_after_cap`` with nothing measured;
    that executed click resets the run, so the fourth is checked and refused again."""
    assert const("MAX_MOVED_REFUSALS") == 2
    d = GuardDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.shift_default = ((630.0, 390.0, 20.0, 20.0),)
    # T-0234 B2 re-pin: the checked calls (1, 2 and 4) have no crop, so the shift decides
    d.region_script.extend([Boom("no crop")] * 3)
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    assert out.status == "executed", out
    assert count(d, "screenshot_region", since) == 0 and count(d, "shifts_since", since) == 0, (
        f"B4: the capped call is not checked: {names(d, since)}"
    )
    assert len(dispatched(d, since)) == 3
    got = guard_notes(notes)
    assert [n["result"] for n in got] == ["refused", "refused", "allowed_after_cap"], notes
    assert (got[2]["shift"], got[2]["changed_share"]) == (None, None), got[2]
    assert_moved_refusal(await act(ex, "click_at", CLICK, ex.latest_frame))


async def test_cap_read_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """B4: ``MAX_MOVED_REFUSALS`` is read at call time: at 1, the second call is already allowed after the cap."""
    assert const("MAX_MOVED_REFUSALS") == 2
    monkeypatch.setattr(executor_mod, "MAX_MOVED_REFUSALS", 1)
    d = GuardDriver()
    ex = build(d)
    f = await own(ex)
    d.shift_default = ((630.0, 390.0, 20.0, 20.0),)
    d.region_script.append(Boom("no crop"))  # T-0234 B2 re-pin: the shift decides
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    out = await act(ex, "click_at", CLICK, f)
    assert out.status == "executed", out


RESETS = [
    pytest.param("scroll", {"direction": "down"}, id="scroll"),
    pytest.param("press_key", {"combo": "Tab"}, id="press_key"),
    pytest.param("wait", {}, id="wait"),
    pytest.param("click_at", CLICK, id="click_at"),
    pytest.param("type_at", TYPE, id="type_at"),
]


@pytest.mark.parametrize(("tool", "args"), RESETS)
async def test_executed_action_resets_run(
    tool: str, args: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4: refused (1), then an executed ``tool`` of the same turn resets the run, so the next two moved calls are
    both refused (1, 2) and only the one after is allowed after the cap. Without the reset the second of them would
    already be allowed."""
    monkeypatch.setattr(executor_mod, "WAIT_S", 0.01)
    near = ((630.0, 390.0, 20.0, 20.0),)
    d = GuardDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.shift_script.append(near)
    # T-0234 B2 re-pin: every moved refusal here has no crop, so the shift decides; ``None`` takes a real crop
    d.region_script.append(Boom("no crop"))
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    if tool in ("click_at", "type_at"):
        # T-0204 B-R2: only a resetting click/type is itself checked (B7: scroll, press_key and wait never read
        # shifts_since), so the quiet answer is queued for it alone
        d.shift_script.append(())
        d.region_script.append(None)
    out = await act(ex, tool, args, f)
    assert out.status == "executed", out
    d.shift_default = near
    d.region_script.extend([Boom("no crop")] * 2)
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    out = await act(ex, "click_at", CLICK, f)
    assert out.status == "executed", out
    assert [n["result"] for n in guard_notes(notes)] == [
        "refused",
        "refused",
        "refused",
        "allowed_after_cap",
    ], notes


async def test_other_refusals_neither_count_nor_reset() -> None:
    """B4: a refusal for another reason (bad coordinates) between two moved refusals neither counts (the call after
    the first moved refusal and it is refused moved, not allowed) nor resets (after two moved refusals and it, the
    next is allowed after the cap)."""
    d = GuardDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.shift_default = ((630.0, 390.0, 20.0, 20.0),)
    d.region_script.extend([Boom("no crop")] * 2)  # T-0234 B2 re-pin: the shift decides
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    bad = await act(ex, "click_at", {"x": 5000, "y": 1}, f)
    assert (bad.status, bad.reason) == ("refused", "bad coordinates"), bad
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    bad = await act(ex, "click_at", {"x": 5000, "y": 1}, f)
    assert bad.status == "refused" and bad.reason == "bad coordinates", bad
    out = await act(ex, "click_at", CLICK, f)
    assert out.status == "executed", out
    assert [n["result"] for n in guard_notes(notes)] == ["refused", "refused", "allowed_after_cap"]


async def test_new_turn_resets() -> None:
    """B4: two moved refusals in turn t1; the first call of turn t2 is checked again (a region call) and refused."""
    d = GuardDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.shift_default = ((630.0, 390.0, 20.0, 20.0),)
    d.region_script.extend([Boom("no crop")] * 3)  # T-0234 B2 re-pin: the shift decides
    assert_moved_refusal(await act(ex, "click_at", CLICK, f, turn="t1"))
    assert_moved_refusal(await act(ex, "click_at", CLICK, f, turn="t1"))
    since = len(d.calls)
    assert_moved_refusal(await act(ex, "click_at", CLICK, f, turn="t2"))
    assert count(d, "shifts_since", since) == 1, names(d, since)
    assert [n["result"] for n in guard_notes(notes)] == ["refused"] * 3


# ============================================================================================ B5: fail open


@pytest.mark.parametrize(
    "fault",
    [
        pytest.param(Boom("synthetic region failure"), id="raises"),
        pytest.param(ValueError("nothing inside"), id="valueerror"),
        pytest.param("hang", id="hangs"),
    ],
)
async def test_region_timeout_fails_open(fault: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """B5: the region screenshot raises or outlives ``PRECLICK_TIMEOUT_S`` (patched to 0.1 s, read at call time):
    the click proceeds as at BASE (executed, the same outcome keys), noted ``unavailable`` with ``changed_share``
    ``None``; a hang costs at most the bound (region start to ``mouse_move`` < 0.28 s, under the default 0.3)."""
    assert const("PRECLICK_TIMEOUT_S") == 0.3
    monkeypatch.setattr(executor_mod, "PRECLICK_TIMEOUT_S", 0.1)
    d = GuardDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.region_script.append(fault)
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    assert out.status == "executed", out
    assert sorted(out.resolved) == ["frame_after", "x_px", "y_px"], out.resolved
    assert len(dispatched(d, since)) == 3
    got = guard_notes(notes)
    assert [(n["result"], n["changed_share"]) for n in got] == [("unavailable", None)], notes
    if fault == "hang":
        took = d.moves_at[-1] - d.region_log[-1][0]
        assert took < 0.28, f"B2/B5: the check is bounded by PRECLICK_TIMEOUT_S: {took:.3f} s"


@pytest.mark.parametrize(
    "fault",
    [pytest.param(Boom("synthetic region failure"), id="raises"), pytest.param("hang", id="hangs")],
)
async def test_shift_decides_when_region_fails(fault: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """B2/B5 ruling (Status, 2026-10-29): a shift rect near the point refuses the call even when the region
    screenshot raises or outlives ``PRECLICK_TIMEOUT_S`` (patched to 0.1 s): ``refused`` with ``MOVED_REASON``,
    nothing dispatched, one note ``refused`` with ``shift`` ``True`` and ``changed_share`` ``None``. Fail-open
    ``unavailable`` is only for neither signal deciding (``test_region_timeout_fails_open``,
    ``test_both_signals_unavailable_fail_open``)."""
    assert const("PRECLICK_TIMEOUT_S") == 0.3
    monkeypatch.setattr(executor_mod, "PRECLICK_TIMEOUT_S", 0.1)
    d = GuardDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.shift_script.append(((630.0, 390.0, 20.0, 20.0),))
    d.region_script.append(fault)
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    assert_moved_refusal(out)
    assert dispatched(d, since) == [] and count(d, "screenshot", since) == 0, names(d, since)
    assert ex.latest_frame is f
    got = guard_notes(notes)
    assert [(n["result"], n["shift"], n["changed_share"]) for n in got] == [
        ("refused", True, None)
    ], notes


async def test_both_signals_unavailable_fail_open() -> None:
    """B5 / threat model: arming failed (``shifts_since`` answers ``None``) and the crop raises: executed as BASE,
    noted ``unavailable`` with ``shift`` and ``changed_share`` ``None``."""
    d = GuardDriver()
    d.arm_ok = False
    d.shift_default = None
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.region_script.append(Boom("no crop"))
    out = await act(ex, "click_at", CLICK, f)
    assert out.status == "executed", out
    got = guard_notes(notes)
    assert [(n["result"], n["shift"], n["changed_share"]) for n in got] == [
        ("unavailable", None, None)
    ], notes


async def test_shift_unknown_static_pixels_decide_silently() -> None:
    """B5: ``shifts_since`` gives ``None`` (arming failed) but the crop ran: the pixel signal alone decides; on a
    static page that is a plain "not moved" (executed, no note)."""
    d = GuardDriver()
    d.arm_ok = False
    d.shift_default = None
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    assert out.status == "executed", out
    checked(d, since)
    assert guard_notes(notes) == [], notes


@pytest.mark.parametrize(("tool", "args"), [("click_at", CLICK), ("type_at", TYPE)])
async def test_cancel_propagates(tool: str, args: dict[str, Any]) -> None:
    """B5: a cancellation during the region screenshot propagates; nothing is dispatched or captured and
    ``latest_frame`` is unchanged."""
    d = GuardDriver()
    ex = build(d)
    f = await own(ex)
    d.region_script.append("hang")
    since = len(d.calls)
    task = asyncio.ensure_future(ex.execute(call(tool, f, **args)))
    for _ in range(200):
        await asyncio.sleep(0.005)
        if count(d, "screenshot_region", since):
            break
    assert count(d, "screenshot_region", since) == 1, (
        f"premise: the check started: {names(d, since)}"
    )
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert dispatched(d, since) == [] and count(d, "screenshot", since) == 0, names(d, since)
    assert ex.latest_frame is f


# ============================================================================================ B6: the note


async def test_note_shape() -> None:
    """B6: each non-plain check calls ``on_note`` once with exactly the five keys; no URL, host, title, phash or
    coordinate is in it; ``ms`` is the check's own time; a raising ``on_note`` is swallowed and changes nothing."""
    d = GuardDriver(block_page())
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.page = block_page(60)
    d.region_script.append(("sleep", 0.05))
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    (n,) = guard_notes(notes)
    assert n["ms"] >= 40, f"B6: ms is the check's time (the crop slept 50 ms): {n}"
    text = json.dumps(n)
    for leak in (str(f.phash), "640", "400", "127.0.0.1", "http"):
        assert leak not in text, f"B6: the note leaks {leak!r}: {n}"

    def boom(_: dict[str, Any]) -> None:
        raise RuntimeError("note sink failed")

    ex.on_note = boom
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    d.page = block_page()
    out = await act(ex, "click_at", CLICK, f, turn="t3")
    assert out.status == "executed" and "resolved" not in json.dumps(n), out


# ============================================================================================ B7: negatives


async def test_overview_frame_path_unguarded() -> None:
    """B2/B7: a click on a registered overview frame (T-0151) is not checked: no region call and no shift read in that
    call, and its outcome is BASE's."""

    async def run(module: Any) -> tuple[list[Any], list[str]]:
        d = GuardPage(content_height=1220.0)
        ex = build(d, module=module, echo="model")
        await own(ex)
        ov = await register(ex)
        since = len(d.calls)
        out = await ex.execute(call("click_at", ov, x=500, y=900))
        return [out.status, out.action, out.reason, sorted(out.resolved)], names(d, since)

    got, seq = await run(executor_mod)
    want, base_seq = await run(base_module())
    assert got == want, f"B7: the overview-frame click differs from BASE: {got} vs {want}"
    assert "screenshot_region" not in seq and "shifts_since" not in seq, seq
    assert [s for s in seq if s not in GUARD] == base_seq, f"B7: {seq} vs BASE {base_seq}"


SEQUENCE = [
    ("click_at", CLICK),
    ("type_at", TYPE),
    ("scroll", {"direction": "down"}),
    ("press_key", {"combo": "Enter"}),
    ("wait", {}),
    ("click_at", {"x": 10, "y": 990}),
    ("type_at", {"x": 900, "y": 20, "text": "abc", "press_enter": True}),
]


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(FakeDriver, id="FakeDriver"),
        pytest.param(FieldDriver, id="FieldDriver"),
        pytest.param(lambda: PageDriver(content_height=1220.0), id="PageDriver"),
        pytest.param(ThreeOfFour, id="three-of-four"),
        *[pytest.param(missing(n), id=f"without-{n}") for n in GUARD],
    ],
)
async def test_fakes_without_capability_base_sequence(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1/B7 (negative): a driver class without all four methods (every existing fake; and one with three of the four)
    makes BASE's driver calls and outcomes, bytes and phashes for every action."""
    monkeypatch.setattr(executor_mod, "WAIT_S", 0.01)
    base = base_module()
    monkeypatch.setattr(base, "WAIT_S", 0.01)
    got = await trace(executor_mod, make, SEQUENCE)
    want = await trace(base, make, SEQUENCE)
    same(got, want, "without the capability")


async def test_guard_off_base_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """B7 (negative): with ``PRECLICK_GUARD = False`` (read at call time) a capability driver's call sequence is
    BASE's: no arm, mark, shift read or region call, and the same outcomes, bytes and phashes."""
    assert const("PRECLICK_GUARD") is True
    monkeypatch.setattr(executor_mod, "PRECLICK_GUARD", False)
    monkeypatch.setattr(executor_mod, "WAIT_S", 0.01)
    base = base_module()
    monkeypatch.setattr(base, "WAIT_S", 0.01)
    got = await trace(executor_mod, GuardDriver, SEQUENCE)
    want = await trace(base, GuardDriver, SEQUENCE)
    same(got, want, "guard off")


async def test_non_click_tools_never_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    """B7: ``scroll``, ``press_key`` and ``wait`` are never checked, even with a shift at every point: no region call,
    no shift read, BASE's outcomes; only the capture's arm/mark calls are added."""
    monkeypatch.setattr(executor_mod, "WAIT_S", 0.01)
    base = base_module()
    monkeypatch.setattr(base, "WAIT_S", 0.01)
    steps = [("scroll", {"direction": "down"}), ("press_key", {"combo": "Tab"}), ("wait", {})]

    def moving() -> GuardDriver:
        d = GuardDriver()
        d.shift_default = ((0.0, 0.0, 1280.0, 800.0),)
        return d

    got = await trace(executor_mod, moving, steps)
    want = await trace(base, moving, steps)
    same(got, want, "non-click tools", drop=frozenset({"arm_shift_recorder", "shift_mark"}))
    assert (
        count(got["driver"], "shifts_since") == 0 and count(got["driver"], "screenshot_region") == 0
    )


async def test_google_search_untouched() -> None:
    """B7: ``google_search`` stays the dry run: no driver call at all."""
    d = GuardDriver()
    ex = build(d)
    out = await ex.execute(call("google_search", None, query="kettle"))
    assert out.status == "dry_run", out
    assert d.calls == [], d.calls


# ============================================================================================ B-R4 and mutation gaps


@pytest.mark.parametrize("change", ["far-whole-viewport", "reflow-at-point"])
async def test_same_phash_recapture_after_seen_change_executes(change: str) -> None:
    """Ruling B-R4 (adversarial: a false refusal). After frame ``f`` the page changes in a way that leaves the phash
    alone: a layout-shift entry read as the whole viewport (A-R1: five or more sources, far from the point), or a
    reflow at the point (a shift there and a band 28 luma darker over 2 of the crop's 16 block rows: 32 of 256
    blocks, 0.125). A click carrying ``f`` is
    refused (checked before any recapture). The session then shows a fresh frame with the same phash; a click carrying
    it executes, since that capture replaced the reference (newest capture and mark)."""
    d = GuardDriver(block_page())
    d.shift_log = []
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    if change == "far-whole-viewport":
        d.add_shift((0.0, 0.0, 1280.0, 800.0))
        # T-0234 B2 re-pin: the pixels at the point are unchanged, so with a crop the click would execute; the crop
        # fails here so the whole-viewport shift decides (BASE's rule), and the recapture's crop is taken
        d.region_script.append(Boom("no crop"))
    else:
        d.add_shift((560.0, 330.0, 160.0, 60.0), (560.0, 336.0, 160.0, 60.0))
        ImageDraw.Draw(d.page).rectangle((576, 336, 703, 351), fill=(100, 100, 100))
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    g = await own(ex)
    assert g.phash == f.phash, (
        "premise: the change leaves the phash alone (the same-phash recapture)"
    )
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, g)
    assert out.status == "executed", f"B-R4: the fresh same-phash frame is the reference now: {out}"
    checked(d, since)
    assert [n["result"] for n in guard_notes(notes)] == ["refused"], notes


async def test_block_size_is_8() -> None:
    """B2 (b): the blocks are 8x8. One 8x8 sub-block brightened by 40 in every 16x16 cell of the crop: 64 of the 256
    8x8 blocks differ by 40 (> 16): share 0.25, refused. On a 16x16 grid every block would differ by 10 (<= 16) and
    pass, so a wrong block size is caught (by hand)."""
    d = GuardDriver(block_page())
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    draw = ImageDraw.Draw(d.page)
    for by in range(8):
        for bx in range(8):
            x0, y0 = 576 + 16 * bx, 336 + 16 * by
            draw.rectangle((x0, y0, x0 + 7, y0 + 7), fill=(168, 168, 168))
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    (n,) = guard_notes(notes)
    assert n["changed_share"] == pytest.approx(0.25, abs=0.03), n


class BlankFirst(GuardDriver):
    """The first screenshot is white (a page still loading, T-0200's blank capture); then the page paints, with a
    shift at the point recorded after the first shot's mark and before the second screenshot, and shows
    ``block_page``."""

    def __init__(self) -> None:
        super().__init__(Image.new("RGB", (W, H), (255, 255, 255)))
        self.shift_log = []
        self.blank_left = 1

    async def screenshot(self) -> bytes:
        if self.blank_left == 0 and not self.shift_log:
            # the page painted after the blank capture's mark and before this one: its shift precedes this shot
            self.add_shift((560.0, 330.0, 160.0, 140.0))
            self.page = block_page()
        out = await super().screenshot()
        self.blank_left = max(0, self.blank_left - 1)
        return out


async def test_blank_recapture_threads_the_kept_shots_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    """B1 with T-0200's blank re-capture: the reference's mark is read after the KEPT (non-blank) screenshot, not the
    first (blank) one, so the shift recorded while the page loaded is before the mark; the first click on the frame
    executes with no note."""
    monkeypatch.setattr(executor_mod, "BLANK_RECAPTURE_INTERVAL_S", 0.01)
    d = BlankFirst()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    assert count(d, "screenshot") == 2, f"premise: the blank capture was retaken: {names(d)}"
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, f)
    assert out.status == "executed", f"B1: the kept shot's mark is the reference's: {out}"
    checked(d, since)
    assert guard_notes(notes) == [], notes


async def test_google_search_does_not_reset_run() -> None:
    """B4: only an executed ``click_at``, ``type_at``, ``scroll``, ``press_key``, ``wait`` or ``page_overview`` ends a
    run of moved refusals. Moved (1), ``google_search`` (dry run), moved (2): the next call is allowed after the cap.
    T-0234 B2 (named re-pin): the two checked calls have no crop, so the shift decides."""
    d = GuardDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await own(ex)
    d.shift_default = ((630.0, 390.0, 20.0, 20.0),)
    d.region_script.extend([Boom("no crop")] * 2)
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    out = await ex.execute(call("google_search", None, query="kettle"))
    assert out.status in ("dry_run", "executed"), out
    assert_moved_refusal(await act(ex, "click_at", CLICK, f))
    out = await act(ex, "click_at", CLICK, f)
    assert out.status == "executed", out
    assert [n["result"] for n in guard_notes(notes)] == ["refused", "refused", "allowed_after_cap"]


@pytest.mark.parametrize("name", GUARD)
async def test_partial_capability_never_calls_the_guard(name: str) -> None:
    """B1: the guard runs only when the driver's class has all four methods. With any one of them absent, a frame, a
    click and a type make no call to the other three (no arm, mark, shift read or region capture)."""
    d = missing(name)()
    ex = build(d)
    f = await own(ex)
    assert (await act(ex, "click_at", CLICK, f)).status == "executed"
    assert (await act(ex, "type_at", TYPE)).status == "executed"
    used = [c[0] for c in d.calls if c[0] in GUARD]
    assert used == [], f"B1: all four, not any: {name} is missing but the guard called {used}"
