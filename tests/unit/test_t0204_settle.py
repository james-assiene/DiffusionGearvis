"""T-0204 group C at the seam: the visual settle before an action's after-frame, over scripted fake drivers, no
browser (docs/tasks/T-0204.md, C1-C10; "Tests to write" C).

**What these rows pin** (the task's words, nothing from an implementation). With ``STABLE_SCREEN = True`` (read at
call time) and a driver *class* with B1's four methods (``screenshot_region``, ``arm_shift_recorder``,
``shift_mark``, ``shifts_since``), an action's capture (``_take("action", ...)``, after T-0200's blank loop) takes
more screenshots at ``STABLE_SCHEDULE_S = (0.1, 0.25, 0.5)`` after the previous one (the last interval repeating)
until two consecutive shots match on T-0200's ``BLANK_GRID_W = 64`` luma grid: a cell differs when |delta| >
``STABLE_CELL_DELTA = 6``; stable is at most ``STABLE_MAX_CELLS = 4`` differing cells, and from the third shot on a
cell that also differed in the previous pair is ignored (animation). The last shot is kept and registered once;
``STABLE_CAP_S = 1.0`` from the first shot, never past T-0200's ``deadline``. A settle of more than 2 shots or at the
cap calls ``on_note`` once with exactly ``{"what": "live frame: waited for a stable screen", "path": "action",
"shots", "stable", "waited_ms"}``.

**The fake.** ``SettleMixin`` puts a script of hand-drawn pages in front of B's ``GuardMixin`` (the four B1 methods,
imported from ``test_t0204_preclick``): each ``screenshot()`` pops the next item of ``pages`` (a PIL page, a
``(page, metrics)`` pair, ``"hang"`` or an exception), makes it the page (so ``screenshot_region`` crops what the
last shot showed), encodes it (JPEG q85, as ``runtime.JPEG_QUALITY``) and logs ``(start, end, bytes)``; once the
script is empty the page stays as it is (a static page). ``on_shot(n)`` runs after the n-th shot.

**Pages and grids, by hand.** Greys only (luma = the grey). 1280x800 box-downscaled to 64 px wide is a 64x40 grid of
20x20 px cells. Every change is drawn in 40x40 squares on multiples of 40 px, so it covers whole grid cells and whole
JPEG 8x8 blocks (no edge inside a block, no bleed into a neighbour cell): one square is exactly 4 cells. ``ANIM`` is
7x6 squares (168 cells) whose grey steps through multiples of 30 (any two states differ by >= 30); ``X`` and ``Y`` are
2x2 squares (16 cells each). Never-stable pages flip a seeded random half of the 640 squares by 100 grey levels on each
shot, so about a quarter of the cells differ in a pair and not in the pair before (far above 4).

**Oracles.** BASE is the executor at ``BASE_SHA`` (Merge T-0204-b, before any C code), run from ``git show``; a
discarded shot's phash is BASE's ``frame()`` of the same bytes under the same salt (``secrets.randbits`` patched).
Negative rows compare driver calls, outcomes, bytes and phashes with BASE byte for byte.

**Readings** (reported, not guessed silently):

- "offsets after the previous shot": pinned start to start (a lower bound ``offset - 5 ms`` and an upper bound
  ``offset + 0.1 s``), which an implementation timing from either end of the previous shot meets; every shot is on
  the schedule (no extra shot clipped to the cap), so a never-stable page takes exactly 4 shots (0, 0.1, 0.35, 0.85).
- "each shot bounded as in T-0200's RV1": a settle shot that hangs is cut at most one interval past the cap (and
  ``FRAME_BUILD_RESERVE_S`` short of the deadline); the action is still ``executed`` with the last successful shot,
  as T-0200's raising re-capture keeps the last successful capture.
- C9 "shots" counts the settle's shots from the capture the blank loop kept (a blank re-capture is T-0200's note).
- C1 "gated on the B1 driver capability": pinned as the four methods on the class. Ruling C-R3: ``PRECLICK_GUARD =
  False`` does not turn the settle off (``STABLE_SCREEN`` is its only switch), and the settle still reads shift marks
  then (C-R1); ``test_guard_off_settle_still_runs_and_reads_shifts``.
- C10 "dry-run and off" are not ``BrowserToolExecutor`` paths (no capture); not pinned here.

Regime: CPU over fakes, real asyncio time (the default schedule where a row is about it, patched small elsewhere;
about 8 s in total), 1280x800, n = 1, deterministic salt.
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import sys
import time
import types
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from local_astra.computer import executor as executor_mod
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.toolcall import FrameGeometry, ToolInvocation, ToolOutcome
from tests import _git_range as gr
from tests.unit.test_computer_executor import CLOCK_MS, FakeDriver
from tests.unit.test_computer_web_executor import FieldDriver
from tests.unit.test_t0151_executor import PageDriver, register
from tests.unit.test_t0200_blank_recapture import ShotDriver
from tests.unit.test_t0203_follow_executor import FollowMixin
from tests.unit.test_t0204_preclick import GuardMixin, GuardPage, ThreeOfFour, jpeg

#: group C's BASE: Merge T-0204-b (main), before any C code. A literal.
BASE_SHA = "cbb88b4b89ad9eafdae7a12a54ed9dfe8caa0953"
SALT = 0x5EED_0204_000C_5EED
W, H = 1280, 800
NOTE_WHAT = "live frame: waited for a stable screen"
NOTE_KEYS = {"what", "path", "shots", "stable", "waited_ms"}
BLANK_WHAT = "live frame: blank capture retaken"
#: typed from the task (C2, H-T204-5); never read from the module for an oracle
SCHEDULE = (0.1, 0.25, 0.5)
CAP = 1.0
#: a sleep is a lower bound; asyncio wakes a little early at most by the clock's resolution
TOL = 0.005
CLICK = {"x": 500, "y": 500}
TYPE = {"x": 500, "y": 500, "text": "kettle", "press_enter": False}
TYPE_ENTER = {"x": 500, "y": 500, "text": "kettle", "press_enter": True}
PASSWORD = {"tag": "input", "type": "password", "name": "pw"}

# ============================================================================================ pages (by hand)

#: squares (col0, row0, col1, row1), 40 px each, exclusive ends
ANIM = (
    12,
    7,
    19,
    13,
)  # x 480-759, y 280-519: 7x6 squares = 168 cells (under the click point (640, 400))
ANIM_GREYS = (30, 90, 150, 210, 60, 120, 180, 240, 0)
X = (26, 3, 28, 5)  # x 1040-1119, y 120-199: 16 cells
Y = (4, 15, 6, 17)  # x 160-239, y 600-679: 16 cells
ONE = (30, 17, 31, 18)  # x 1200-1239, y 680-719: 4 cells
OTHER = (2, 18, 3, 19)  # x 80-119, y 720-759: 4 more cells
BIG = (
    11,
    6,
    21,
    14,
)  # x 440-839, y 240-559: grey 20 over the page (320 cells differ from base_page)


def base_page() -> Image.Image:
    """A static page: grey 230 with a dark header and two blocks (not blank: luma 60..230)."""
    img = Image.new("RGB", (W, H), (230, 230, 230))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W - 1, 79), fill=(60, 60, 60))
    d.rectangle((80, 160, 399, 399), fill=(120, 120, 120))
    d.rectangle((880, 480, 1199, 719), fill=(90, 90, 90))
    return img


def paint(img: Image.Image, area: tuple[int, int, int, int], grey: int) -> Image.Image:
    """A copy of ``img`` with the squares ``area`` filled with ``grey``."""
    out = img.copy()
    c0, r0, c1, r1 = area
    ImageDraw.Draw(out).rectangle((40 * c0, 40 * r0, 40 * c1 - 1, 40 * r1 - 1), fill=(grey,) * 3)
    return out


def anim(k: int, img: Image.Image | None = None) -> Image.Image:
    return paint(img if img is not None else base_page(), ANIM, ANIM_GREYS[k])


def results_page() -> Image.Image:
    """Another page (a navigation's target): grey 200, header 20, a block lower left."""
    img = Image.new("RGB", (W, H), (200, 200, 200))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W - 1, 119), fill=(20, 20, 20))
    d.rectangle((40, 200, 639, 599), fill=(100, 100, 100))
    return img


def white() -> Image.Image:
    return Image.new("RGB", (W, H), (255, 255, 255))


def noise_pages(n: int, seed: int = 204) -> list[Image.Image]:
    """``n`` pages of 32x20 squares; each page flips a seeded random half of the squares by 100 grey levels."""
    rng = np.random.default_rng(seed)
    greys = rng.integers(20, 220, size=(20, 32))
    out = []
    for _ in range(n):
        flip = rng.random((20, 32)) < 0.5
        greys = np.where(flip, (greys - 20 + 100) % 200 + 20, greys)
        arr = np.kron(greys, np.ones((40, 40))).astype(np.uint8)
        out.append(Image.fromarray(arr, mode="L").convert("RGB"))
    return out


# ============================================================================================ the fakes


class SettleMixin:
    """Scripted pages in front of the B1 capability fake (module docstring)."""

    calls: list[tuple[str, tuple[Any, ...]]]
    shots: list[str]
    active: str

    def __init__(self, *args: Any, page: Image.Image | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.page = page if page is not None else base_page()
        self.pages: deque[Any] = deque()
        self.shot_log: list[tuple[float, float, bytes]] = []
        self.on_shot: Callable[[int], None] | None = None
        self.metrics_now: dict[str, float] = {
            "scroll_y": 0.0,
            "viewport_height": 800.0,
            "content_height": 1600.0,
        }

    def page_image(self) -> Image.Image:
        return self.page

    async def screenshot(self) -> bytes:
        self._enter("screenshot")  # type: ignore[attr-defined]
        self.shots.append(self.active)
        start = time.monotonic()
        item = self.pages.popleft() if self.pages else None
        if isinstance(item, BaseException):
            raise item
        if item == "hang":
            await asyncio.Event().wait()
        if isinstance(item, tuple):
            self.page, self.metrics_now = item[0], dict(item[1])
        elif isinstance(item, Image.Image):
            self.page = item
        data = jpeg(self.page)
        self.shot_log.append((start, time.monotonic(), data))
        if self.on_shot is not None:
            self.on_shot(len(self.shot_log))
        return data

    async def bring_to_front(self) -> None:
        self.calls.append(("bring_to_front", ()))


class SettleDriver(SettleMixin, GuardMixin, FakeDriver):
    pass


class SettleMetrics(SettleDriver):
    """With T-0134's ``layout_metrics``: the metrics the last shot set (``(page, metrics)`` items)."""

    async def layout_metrics(self) -> dict[str, float]:
        self._enter("layout_metrics")
        return dict(self.metrics_now)


class SettleField(SettleMixin, GuardMixin, FieldDriver):
    """T-0160's ``focused_field`` fake (the guarded type path) under the scripted pages."""


class SettleFollow(SettleMixin, GuardMixin, FollowMixin, FakeDriver):
    """T-0203's ``follow_navigation`` and scripted ``navigated_since`` answers under the scripted pages."""


class PlainScripted(SettleMixin, FakeDriver):
    """Scripted, changing pages on a class WITHOUT the B1 capability: BASE (C10)."""


class Boom(RuntimeError):
    pass


# ============================================================================================ helpers


@pytest.fixture(autouse=True)
def fixed_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(secrets, "randbits", lambda n: SALT)


def const(name: str) -> Any:
    value = getattr(executor_mod, name, None)
    assert value is not None, f"T-0204 C: executor.{name} does not exist yet"
    return value


_BASE: types.ModuleType | None = None


def base_module() -> types.ModuleType:
    global _BASE
    if _BASE is None:
        gr.require_commits(BASE_SHA)
        src = gr._git("show", f"{BASE_SHA}:local_astra/computer/executor.py").stdout
        assert "def _take" in src and "STABLE_SCREEN" not in src, "premise: BASE has no settle"
        name = "_t0204c_base_executor"
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


def geometry(frame: Frame) -> FrameGeometry:
    return FrameGeometry(
        phash=frame.phash, source=frame.source, width=frame.width, height=frame.height
    )


def call(
    name: str, frame: Frame | FrameGeometry | None = None, *, turn: str = "t1", **arguments: Any
) -> ToolInvocation:
    g = geometry(frame) if isinstance(frame, Frame) else frame
    return ToolInvocation(
        call_id=f"c{next(_N)}",
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
        frame = ex.latest_frame if ex.latest_frame is not None else await own(ex)
    out = await ex.execute(
        call(tool, frame if tool in ("click_at", "type_at") else None, turn=turn, **args)
    )
    assert isinstance(out, ToolOutcome)
    return out


def names(d: Any, since: int = 0) -> list[str]:
    return [c[0] for c in d.calls[since:]]


def settle_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """C9's notes, each checked for the exact shape."""
    out = [n for n in notes if n.get("what") == NOTE_WHAT]
    for n in out:
        assert set(n) == NOTE_KEYS, f"T-0204 C9: the note has exactly {sorted(NOTE_KEYS)}: {n}"
        assert n["path"] == "action", n
        assert type(n["shots"]) is int and n["shots"] >= 2, n
        assert type(n["stable"]) is bool, n
        assert type(n["waited_ms"]) is int and n["waited_ms"] >= 0, n
    return out


def starts(log: list[tuple[float, float, bytes]]) -> list[float]:
    return [s[0] for s in log]


def check_schedule(log: list[tuple[float, float, bytes]], schedule: tuple[float, ...]) -> None:
    """Every gap between two shots' starts is the schedule's offset (the last repeating): at least ``offset - TOL``
    and at most ``offset + 0.1`` (module docstring, reading)."""
    t = starts(log)
    for i in range(1, len(t)):
        want = schedule[min(i - 1, len(schedule) - 1)]
        gap = t[i] - t[i - 1]
        assert want - TOL <= gap <= want + 0.1, (
            f"T-0204 C2: shot {i + 1} came {gap * 1000:.0f} ms after shot {i}, want the offset "
            f"{want * 1000:.0f} ms (schedule {schedule}); starts {[round(x - t[0], 3) for x in t]}"
        )


async def base_phash(data: bytes) -> str:
    """BASE's ``frame()`` phash of ``data`` under the same salt (an independent oracle; frame() is untouched by C)."""
    d = ShotDriver()
    d.queue(data)
    ex = base_module().BrowserToolExecutor(now_ms=lambda: CLOCK_MS, viewport=(W, H), driver=d)
    f = await ex.frame()
    return str(f.phash)


def stale(phash: str) -> FrameGeometry:
    return FrameGeometry(phash=phash, source=Source.WINDOW, width=W, height=H)


async def acted(
    d: Any, ex: Any, tool: str, args: dict[str, Any], **kw: Any
) -> tuple[ToolOutcome, list[tuple[float, float, bytes]], float]:
    """Run one action; its outcome, the shots it took and its wall time."""
    n0 = len(d.shot_log)
    t0 = time.monotonic()
    out = await act(ex, tool, args, **kw)
    return out, d.shot_log[n0:], time.monotonic() - t0


# ============================================================================================ C1/C2: constants


def test_constants_exports() -> None:
    """C1/C2 (H-T204-5): the switch and the rule's constants at their proposed defaults, exported."""
    want: dict[str, Any] = {
        "STABLE_SCREEN": True,
        "STABLE_SCHEDULE_S": SCHEDULE,
        "STABLE_CELL_DELTA": 6,
        "STABLE_MAX_CELLS": 4,
        "STABLE_CAP_S": CAP,
    }
    for name, value in want.items():
        got = getattr(executor_mod, name, None)
        assert got is not None, f"T-0204 C: executor.{name} does not exist yet"
        assert got == value and type(got) is type(value), (
            f"T-0204 C: {name} = {got!r}, want {value!r}"
        )
        assert name in executor_mod.__all__, f"T-0204 C: {name} is not in executor.__all__"
    assert executor_mod.BLANK_GRID_W == 64, "premise: T-0200's grid width"


# ============================================================================================ C1/C3: a static page


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        pytest.param("click_at", CLICK, id="click_at"),
        pytest.param("type_at", TYPE, id="type_at"),
        pytest.param("press_key", {"combo": "Escape"}, id="press_key"),
        pytest.param("scroll", {"direction": "down"}, id="scroll"),
        pytest.param("wait", {}, id="wait"),
    ],
)
async def test_static_two_shots(
    tool: str, args: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """C1/C3: on a static page an executed action's after-frame takes exactly 2 screenshots, the second
    ``STABLE_SCHEDULE_S[0]`` after the first; the second is kept (``latest_frame``, ``frame_after``); no settle note.
    BASE takes one."""
    monkeypatch.setattr(executor_mod, "WAIT_S", 0.01)
    d = SettleDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    await own(ex)
    out, shots, _ = await acted(d, ex, tool, args)
    assert out.status == "executed", out
    assert len(shots) == 2, (
        f"T-0204 C3: a static page's after-frame is exactly 2 screenshots, got {len(shots)}"
    )
    check_schedule(shots, SCHEDULE)
    assert (
        ex.latest_frame.jpeg == shots[-1][2]
        and out.resolved["frame_after"] == ex.latest_frame.phash
    )
    assert settle_notes(notes) == [], f"T-0204 C9: no note for a 2-shot settle: {notes}"


async def test_static_two_shots_overview_frame_action() -> None:
    """C1: the overview-frame action's after-frame (T-0151) settles too: one more screenshot than BASE on a static
    page."""

    async def run(module: Any) -> tuple[str, int]:
        d = GuardPage(content_height=1220.0)
        ex = build(d, module=module, echo="model")
        await own(ex)
        ov = await register(ex)
        since = len(d.calls)
        out = await ex.execute(call("click_at", ov, x=500, y=900))
        return out.status, names(d, since).count("screenshot")

    got, base = await run(executor_mod), await run(base_module())
    assert got[0] == base[0] == "executed", (got, base)
    assert got[1] == base[1] + 1, (
        f"T-0204 C1: the overview-frame action settles (BASE {base[1]} screenshots + 1), got {got[1]}"
    )


async def test_static_two_shots_guarded_type_stop() -> None:
    """C1: the guarded-type stop capture (T-0160 B-3: a password field, ``failed`` before any key) settles too:
    2 screenshots after the call on a static page; BASE takes 1."""

    async def run(module: Any) -> tuple[str, str | None, int]:
        d = SettleField(attrs=PASSWORD)
        ex = build(d, module=module)
        await own(ex)
        n0 = len(d.shot_log)
        out = await act(ex, "type_at", TYPE)
        return out.status, out.reason, len(d.shot_log) - n0

    got, base = await run(executor_mod), await run(base_module())
    assert got[:2] == base[:2] and got[0] == "failed", (got, base)
    assert base[2] == 1, "premise: BASE's stop capture is one screenshot"
    assert got[2] == 2, f"T-0204 C1: the guarded-type stop capture settles: {got[2]} screenshots"


# ============================================================================================ C2: the rule


async def test_changing_then_stable_keeps_last() -> None:
    """C2: the page changes between the first two shots (320 cells) and by only 4 cells between the 2nd and 3rd:
    stable at the 3rd shot, which is kept (not the 2nd, whose bytes differ), with the extent from the metrics read
    around the 3rd shot; the shots follow the default schedule."""
    m1 = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 1600.0}
    m2 = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 2000.0}
    d = SettleMetrics()
    ex = build(d)
    await own(ex)
    b = base_page()
    d.pages.extend([(paint(b, BIG, 20), m1), (b, m2), (paint(b, ONE, 0), m2)])
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed", out
    assert len(shots) == 3, f"T-0204 C2: stable at the 3rd shot, got {len(shots)} shots"
    check_schedule(shots, SCHEDULE)
    assert shots[1][2] != shots[2][2], "premise: the 2nd and 3rd shots differ (by 4 cells)"
    assert ex.latest_frame.jpeg == shots[2][2], "T-0204 C2: the kept capture is the last shot"
    assert out.resolved["frame_after"] == ex.latest_frame.phash
    assert ex.extent_of(ex.latest_frame.phash) == {
        "above_px": 0.0,
        "below_px": 1200.0,
        "viewport_px": 800.0,
    }, "T-0204 C2: the kept frame's extent is from the metrics read around the last shot"


@pytest.mark.parametrize(
    ("area2", "shots_want"),
    [
        pytest.param(ONE, 2, id="4-cells-stable"),
        pytest.param((30, 17, 31, 19), 3, id="8-cells-not-stable"),
    ],
)
async def test_max_cells_threshold(area2: tuple[int, int, int, int], shots_want: int) -> None:
    """C2: at most ``STABLE_MAX_CELLS = 4`` differing cells is stable (one 40x40 square = 4 cells: 2 shots); 8 cells
    is not (3 shots, the 3rd equal to the 2nd)."""
    d = SettleDriver()
    ex = build(d)
    await own(ex)
    changed = paint(base_page(), area2, 0)
    d.pages.extend([base_page(), changed, changed])
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed", out
    assert len(shots) == shots_want, f"T-0204 C2: want {shots_want} shots, got {len(shots)}"


@pytest.mark.parametrize(
    ("grey", "shots_want"),
    [
        pytest.param(234, 2, id="delta-4-same"),
        pytest.param(236, 2, id="delta-6-same"),
        pytest.param(237, 3, id="delta-7-differs"),
        pytest.param(242, 3, id="delta-12-differs"),
    ],
)
async def test_cell_delta_threshold(grey: int, shots_want: int) -> None:
    """C2: a cell differs only when |delta luma| > ``STABLE_CELL_DELTA = 6``: the whole lower half of the background
    (grey 230) going to 234 (delta 4, hundreds of cells) is stable at 2 shots; to 242 (delta 12) it is not."""
    d = SettleDriver()
    ex = build(d)
    await own(ex)
    changed = paint(
        base_page(), (0, 10, 20, 20), grey
    )  # x 0-799, y 400-799: background and part of a block
    d.pages.extend([paint(base_page(), (0, 10, 20, 20), 230), changed, changed])
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed", out
    assert len(shots) == shots_want, f"T-0204 C2: want {shots_want} shots, got {len(shots)}"


async def test_rule_constants_read_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """C2 (implementation notes: read at call time): a patched schedule is the gaps; ``STABLE_MAX_CELLS = 8`` makes an
    8-cell change stable; ``STABLE_CELL_DELTA = 20`` makes a delta-12 change the same."""
    monkeypatch.setattr(executor_mod, "STABLE_SCHEDULE_S", (0.02, 0.04, 0.06), raising=False)
    d = SettleDriver()
    ex = build(d)
    await own(ex)
    b = base_page()
    # pair 1 changes X, pair 2 changes Y (new cells, so not animation), pair 3 nothing: stable at the 4th shot
    x = paint(b, X, 0)
    d.pages.extend([b, x, paint(x, Y, 0)])
    _, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert len(shots) == 4, f"premise: X then Y then nothing is 4 shots: {len(shots)}"
    check_schedule(shots, (0.02, 0.04, 0.06))

    monkeypatch.setattr(executor_mod, "STABLE_MAX_CELLS", 8, raising=False)
    eight = paint(b, (30, 17, 31, 19), 0)
    d.pages.extend(
        [b, eight]
    )  # stable at the 2nd shot: a third scripted page would leak into the next action
    _, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert len(shots) == 2, f"T-0204 C2: STABLE_MAX_CELLS read at call time: {len(shots)} shots"

    monkeypatch.setattr(executor_mod, "STABLE_MAX_CELLS", 4, raising=False)
    monkeypatch.setattr(executor_mod, "STABLE_CELL_DELTA", 20, raising=False)
    lower = paint(b, (0, 10, 20, 20), 242)
    d.pages.extend([paint(b, (0, 10, 20, 20), 230), lower])
    _, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert len(shots) == 2, f"T-0204 C2: STABLE_CELL_DELTA read at call time: {len(shots)} shots"


# ============================================================================================ C2/C5: animation


async def test_animation_cells_ignored_from_third_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    """C2/C5: static except a 168-cell block that changes on every shot: the first pair differs (nothing is ignored
    in the first pair), so not stable at 2; at the 3rd shot every differing cell also differed in the previous pair,
    so it is stable: exactly 3 shots, the 3rd kept, one note ``shots: 3, stable: true``."""
    monkeypatch.setattr(executor_mod, "STABLE_SCHEDULE_S", (0.02, 0.04, 0.06), raising=False)
    d = SettleDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    await own(ex)
    d.pages.extend([anim(k) for k in range(8)])
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed", out
    assert len(shots) == 3, (
        f"T-0204 C2: an animated block does not hold the settle: {len(shots)} shots"
    )
    assert ex.latest_frame.jpeg == shots[2][2]
    (n,) = settle_notes(notes)
    assert (n["shots"], n["stable"]) == (3, True), n


async def test_new_cells_in_a_later_pair_are_not_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """C2 (adversarial: everything ignored): the block animates, and at the 3rd shot 16 cells elsewhere change for the
    first time: those are not animation, so not stable at 3; the 4th shot changes only the block: stable, 4 shots."""
    monkeypatch.setattr(executor_mod, "STABLE_SCHEDULE_S", (0.02, 0.04, 0.06), raising=False)
    d = SettleDriver()
    ex = build(d)
    await own(ex)
    withx = paint(base_page(), X, 0)
    d.pages.extend(
        [anim(0), anim(1), anim(2, withx), anim(3, withx), anim(4, withx), anim(5, withx)]
    )
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed", out
    assert len(shots) == 4, (
        f"T-0204 C2: new cells at the 3rd shot count: want 4 shots, got {len(shots)}"
    )
    assert ex.latest_frame.jpeg == shots[3][2]


async def test_only_the_previous_pair_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """C2 (adversarial: a cumulative ignore set): X changes in pair 1, Y in pair 2, X changes back in pair 3. X did not
    differ in the previous pair (2), so pair 3 is not stable; pair 4 is unchanged: stable at the 5th shot."""
    monkeypatch.setattr(executor_mod, "STABLE_SCHEDULE_S", (0.02, 0.04, 0.06), raising=False)
    d = SettleDriver()
    ex = build(d)
    await own(ex)
    b = base_page()
    xy = paint(paint(b, X, 0), Y, 0)
    d.pages.extend([b, paint(b, X, 0), xy, paint(b, Y, 0), paint(b, Y, 0)])
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed", out
    assert len(shots) == 5, (
        f"T-0204 C2: only the previous pair's cells are animation: want 5 shots, got {len(shots)}"
    )


# ============================================================================================ C-R1: layout shifts


SHIFT = (480.0, 280.0, 280.0, 240.0)  # the ANIM block's rect, CSS px


def shift_at(
    d: SettleDriver, n0: int, k: int, rect: tuple[float, float, float, float] = SHIFT
) -> None:
    """A layout-shift entry recorded during the ``k``-th shot of the action (before that shot's mark): between shot
    ``k - 1`` and shot ``k``."""

    def hook(n: int) -> None:
        if n == n0 + k:
            d.add_shift(rect)

    d.on_shot = hook


async def test_shift_between_shots_holds_animation_like_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ruling C-R1 (adversarial: a lazy insert changes the same cells every pair). The ANIM cells change on every
    shot, which alone is stable at the 3rd shot (``test_animation_cells_ignored_from_third_shot``); with the recorder
    reporting a layout shift between shots 2 and 3 the 2-3 pair is not stable, so a 4th shot is taken; no shift
    between 3 and 4: stable, 4 shots, the 4th kept."""
    monkeypatch.setattr(executor_mod, "STABLE_SCHEDULE_S", (0.02, 0.04, 0.06), raising=False)
    d = SettleDriver()
    ex = build(d)
    await own(ex)
    d.pages.extend([anim(k) for k in range(8)])
    n0 = len(d.shot_log)
    shift_at(d, n0, 3)
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed", out
    assert len(shots) == 4, (
        f"T-0204 C-R1: a shift between shots 2 and 3 holds the pair: want 4 shots, got {len(shots)}"
    )
    assert ex.latest_frame.jpeg == shots[3][2]


async def test_shift_between_identical_shots_holds_the_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C-R1: a static page whose first two shots are pixel-identical but with a shift entry between them is not
    stable at 2 (the shift may not have painted yet): 3 shots."""
    monkeypatch.setattr(executor_mod, "STABLE_SCHEDULE_S", (0.02, 0.04, 0.06), raising=False)
    d = SettleDriver()
    ex = build(d)
    await own(ex)
    shift_at(d, len(d.shot_log), 2)
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed" and len(shots) == 3, (out.status, len(shots))


async def test_zero_area_shift_does_not_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """C-R1: a zero-area entry (a source that moved fully in or out, A's R1) is not a shift: the animated page is
    stable at the 3rd shot as without the recorder."""
    monkeypatch.setattr(executor_mod, "STABLE_SCHEDULE_S", (0.02, 0.04, 0.06), raising=False)
    d = SettleDriver()
    ex = build(d)
    await own(ex)
    d.pages.extend([anim(k) for k in range(8)])
    shift_at(d, len(d.shot_log), 3, (0.0, 0.0, 0.0, 0.0))
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed" and len(shots) == 3, (out.status, len(shots))


@pytest.mark.parametrize("answer", [None, ()], ids=["recorder-none", "no-entries"])
async def test_recorder_unavailable_cell_rule_alone(
    answer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C-R1: ``shifts_since`` answering ``None`` (recorder unavailable) or no entries: the cell rule alone decides,
    so the animated page is stable at the 3rd shot and a static page at the 2nd; the settle read the recorder."""
    monkeypatch.setattr(executor_mod, "STABLE_SCHEDULE_S", (0.02, 0.04, 0.06), raising=False)
    d = SettleDriver()
    ex = build(d)
    await own(ex)
    d.shift_default = answer
    d.pages.extend([anim(k) for k in range(8)])
    since = len(d.calls)
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed" and len(shots) == 3, (out.status, len(shots))
    assert "shifts_since" in names(d, since), (
        "T-0204 C-R1: the settle reads the shift recorder between shots"
    )
    d.pages.clear()
    d.pages.append(base_page())
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed" and len(shots) == 2, (out.status, len(shots))


# ============================================================================================ C2/C6: bounds


async def test_never_stable_bounded_by_cap_and_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """C2/C6 (adversarial: a page that never settles). Cap: shots at 0, 0.1, 0.35, 0.85 s from the first (the next,
    1.35, is past ``STABLE_CAP_S``): exactly 4, the last kept, ``executed`` within ``STABLE_CAP_S + 0.3`` s, one note
    ``stable: false``. Deadline: with ``ACTION_TIMEOUT_S = 1.5`` (deadline 0.5 s after the call starts) no shot starts
    past the deadline and the call is still ``executed`` within 1.5 s."""
    d = SettleDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    await own(ex)
    d.pages.extend(noise_pages(12))
    out, shots, took = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed", out
    assert len(shots) == 4, (
        f"T-0204 C2: a never-stable page takes the 4 scheduled shots, got {len(shots)}"
    )
    check_schedule(shots, SCHEDULE)
    t = starts(shots)
    assert t[-1] - t[0] <= CAP + TOL, f"T-0204 C2: a shot past the cap: {t[-1] - t[0]:.3f} s"
    assert took <= CAP + 0.3, f"T-0204 C6: {took:.2f} s on a never-stable page"
    assert ex.latest_frame.jpeg == shots[-1][2]
    (n,) = settle_notes(notes)
    assert (n["shots"], n["stable"]) == (4, False), n
    assert 840 <= n["waited_ms"] <= 1300, n

    monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", 1.5)
    d2 = SettleDriver()
    ex2 = build(d2)
    await own(ex2)
    d2.pages.extend(noise_pages(12, seed=7))
    t0 = time.monotonic()
    out, shots, took = await acted(d2, ex2, "press_key", {"combo": "Escape"})
    assert out.status == "executed", out
    deadline = t0 + 1.5 - executor_mod.CAPTURE_RESERVE_S
    late = [round(s - deadline, 3) for s in starts(shots) if s > deadline + 0.02]
    assert late == [], f"T-0204 C2: shots started past T-0200's deadline by {late} s"
    assert len(shots) >= 2, f"premise: the settle ran before the deadline: {len(shots)}"
    assert took <= 1.5, f"T-0204 C6: {took:.2f} s with ACTION_TIMEOUT_S = 1.5"


async def test_cap_read_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """C2: ``STABLE_CAP_S = 0.4`` (read at call time): a never-stable page takes the shots at 0, 0.1, 0.35 only."""
    monkeypatch.setattr(executor_mod, "STABLE_CAP_S", 0.4, raising=False)
    d = SettleDriver()
    ex = build(d)
    await own(ex)
    d.pages.extend(noise_pages(8))
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed" and len(shots) == 3, (out.status, len(shots))


@pytest.mark.parametrize("timeout", [1.5, None], ids=["deadline", "default-timeout"])
async def test_hanging_settle_shot_is_bounded(
    timeout: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C2 "each shot bounded as in T-0200's RV1" (reading): the settle's 2nd screenshot hangs. With
    ``ACTION_TIMEOUT_S = 1.5`` the call ends within 1.5 s; with the default 8 s within one interval past the cap
    (<= 2.0 s). Either way ``executed`` with the first (last successful) shot kept."""
    if timeout is not None:
        monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", timeout)
    d = SettleDriver()
    ex = build(d)
    await own(ex)
    first = paint(base_page(), BIG, 20)
    d.pages.extend([first, "hang"])
    n0 = len(d.calls)
    t0 = time.monotonic()
    out = await asyncio.wait_for(act(ex, "press_key", {"combo": "Escape"}), 10)
    took = time.monotonic() - t0
    assert names(d, n0).count("screenshot") >= 2, "premise: the settle's 2nd shot was started"
    bound = 1.5 if timeout is not None else 2.0
    assert took <= bound, f"T-0204 C2/RV1: a hanging settle shot held the call {took:.2f} s"
    assert out.status == "executed", out
    assert ex.latest_frame.jpeg == jpeg(first), "the last successful shot is kept"


# ============================================================================================ C7: composition


async def test_blank_loop_runs_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """C1/C7: two blank captures, then the painted page: T-0200 retakes every ``BLANK_RECAPTURE_INTERVAL_S``; the
    settle starts from the first painted capture (one more shot ``STABLE_SCHEDULE_S[0]`` later): 4 shots, the 4th
    kept, T-0200's note ``captures: 3``, no settle note (2 settle shots). With the blank budget spent, the settle still
    runs once on the last (blank) capture."""
    monkeypatch.setattr(executor_mod, "BLANK_RECAPTURE_INTERVAL_S", 0.02)
    d = SettleDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    await own(ex)
    d.pages.extend([white(), white(), base_page()])
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed", out
    assert len(shots) == 4, f"T-0204 C7: 3 blank-loop shots + 1 settle shot, got {len(shots)}"
    t = starts(shots)
    assert t[1] - t[0] < 0.08 and t[2] - t[1] < 0.08, f"premise: T-0200's interval: {t}"
    assert t[3] - t[2] >= SCHEDULE[0] - TOL, (
        f"T-0204 C7: the settle's offset after the kept capture: {t}"
    )
    assert ex.latest_frame.jpeg == shots[3][2]
    blank = [n for n in notes if n.get("what") == BLANK_WHAT]
    assert [(n["captures"], n["blank"]) for n in blank] == [(3, False)], notes
    assert settle_notes(notes) == [], notes

    monkeypatch.setattr(executor_mod, "BLANK_RECAPTURE_ACTION_S", 0.05)
    notes.clear()
    d.pages.extend([white()] * 12)
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed", out
    t = starts(shots)
    assert len(shots) >= 2 and t[-1] - t[-2] >= SCHEDULE[0] - TOL, (
        f"T-0204 C1: the settle runs after a spent blank budget: {[round(x - t[0], 3) for x in t]}"
    )
    assert all(t[i + 1] - t[i] < 0.08 for i in range(len(t) - 2)), t
    blank = [n for n in notes if n.get("what") == BLANK_WHAT]
    assert (
        len(blank) == 1 and blank[0]["blank"] is True and blank[0]["captures"] == len(shots) - 1
    ), notes


async def test_follow_runs_before_the_settled_capture() -> None:
    """C7: T-0203's follow runs before the capture, and the capture after it settles (2 shots of the new page)."""
    d = SettleFollow()
    ex = build(d)
    await own(ex)
    d.read_script.extend([None, True])
    d.follow_script.append("loaded")
    d.pages.extend([results_page(), results_page()])
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK)
    assert out.status == "executed" and out.resolved.get("navigated") is True, out
    seq = names(d, since)
    f = seq.index("follow_navigation")
    assert "screenshot" not in seq[:f], f"T-0203 B1: the follow precedes the capture: {seq}"
    assert seq[f:].count("screenshot") == 2, (
        f"T-0204 C1: the capture after the follow settles: {seq}"
    )


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        pytest.param("type_at", TYPE_ENTER, id="type-enter"),
        pytest.param("press_key", {"combo": "Enter"}, id="press-enter"),
    ],
)
async def test_late_retake_also_settles(tool: str, args: dict[str, Any]) -> None:
    """C1/C7: an Enter action reads ``False``, its capture settles (2 shots of the old page), the late read sees the
    navigation, the follow ends ``loaded``, and T-0203 B2's re-capture settles too: the new page half-painted, then
    painted twice (3 shots); the last is kept, ``navigated: True``."""
    d = SettleFollow()
    ex = build(d)
    await own(ex)
    d.read_script.extend([False, None, True])
    d.follow_script.append("loaded")
    r = results_page()
    half = paint(r, (0, 10, 32, 20), 200)
    half = paint(half, (1, 5, 16, 10), 230)  # the lower block not painted yet
    d.pages.extend([base_page(), base_page(), half, r, r])
    since = len(d.calls)
    out, shots, _ = await acted(d, ex, tool, args)
    assert out.status == "executed", out
    seq = [
        n for n in names(d, since) if n in ("screenshot", "navigated_since", "follow_navigation")
    ]
    first_shot = seq.index("screenshot")
    tail = seq[first_shot:]
    assert tail == [
        "screenshot",
        "screenshot",
        "navigated_since",
        "follow_navigation",
        "navigated_since",
        "screenshot",
        "screenshot",
        "screenshot",
    ], f"T-0204 C1/C7: both captures settle, the late re-capture after the follow: {seq}"
    assert ex.latest_frame.jpeg == shots[-1][2] == jpeg(r)
    assert (
        out.resolved.get("navigated") is True
        and out.resolved["frame_after"] == ex.latest_frame.phash
    )


# ============================================================================================ C8: identity


async def test_intermediate_shot_not_own_frame() -> None:
    """C8: only the kept (3rd) shot is registered: a ``click_at`` carrying the 1st shot's phash (BASE's frame() of
    its bytes) is refused and dispatches nothing; one carrying the kept frame executes."""
    d = SettleDriver()
    ex = build(d)
    await own(ex)
    b = base_page()
    d.pages.extend([paint(b, BIG, 20), b, paint(b, ONE, 0)])
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed" and len(shots) == 3, (out, len(shots))
    first = await base_phash(shots[0][2])
    kept = ex.latest_frame
    assert first != kept.phash, "premise: the 1st shot's phash is not the kept frame's"
    since = len(d.calls)
    refused = await ex.execute(call("click_at", stale(first), **CLICK))
    assert refused.status == "refused", (
        f"T-0204 C8: an intermediate shot is not an own frame: {refused}"
    )
    assert [c for c in names(d, since) if c.startswith(("mouse", "key", "type"))] == []
    assert ex.latest_frame is kept
    out = await act(ex, "click_at", CLICK, kept)
    assert out.status == "executed", out


async def test_guard_reference_is_the_kept_shot() -> None:
    """C8: B's guard reference is the kept shot's: a layout shift at the click point recorded between the 2nd shot's
    mark and the 3rd shot holds the 2-3 pair (ruling C-R1), so the 4th shot (equal to the 3rd) is kept; the shift is
    before the kept shot's mark, so a click on the kept frame executes and its ``shifts_since`` read is given the mark
    read right after the kept screenshot."""
    d = SettleDriver()
    ex = build(d)
    await own(ex)
    b = base_page()
    d.pages.extend([paint(b, BIG, 20), b, paint(b, ONE, 0)])
    n0 = len(d.shot_log)

    def shift_before_third(n: int) -> None:
        if n == n0 + 3:
            d.add_shift((560.0, 330.0, 160.0, 140.0))  # around (640, 400)

    d.on_shot = shift_before_third
    out = await act(ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed" and len(d.shot_log) - n0 == 4, (out, len(d.shot_log) - n0)
    d.on_shot = None
    kept_shots = d.shots_so_far()
    since = len(d.calls)
    out = await act(ex, "click_at", CLICK, ex.latest_frame)
    assert out.status == "executed", (
        f"T-0204 C8: the reference is the kept shot's (mark and pixels): {out}"
    )
    # the guard's read is the one before the first mouse_move (the click's own after-frame settle reads the
    # recorder too, ruling C-R1)
    calls = d.calls[since:]
    move = next(i for i, c in enumerate(calls) if c[0] == "mouse_move")
    asked = [c[1][0] for c in calls[:move] if c[0] == "shifts_since"]
    assert len(asked) == 1 and d.mark_after[id(asked[0])] == kept_shots, (
        "T-0204 C8: shifts_since gets the mark read after the kept screenshot"
    )


async def test_cancel_registers_nothing() -> None:
    """C10: a cancellation during the settle's wait propagates; nothing is registered (``latest_frame`` unchanged, the
    1st shot's phash is not an own frame) and no shot is taken after it."""
    d = SettleDriver()
    ex = build(d)
    f = await own(ex)
    first = paint(base_page(), BIG, 20)
    d.pages.extend([first, base_page()])
    n0 = len(d.shot_log)
    got = asyncio.Event()
    d.on_shot = lambda n: got.set() if n == n0 + 1 else None
    task = asyncio.ensure_future(act(ex, "press_key", {"combo": "Escape"}))
    await asyncio.wait_for(got.wait(), 5)
    await asyncio.sleep(0.03)
    task.cancel()
    try:
        done = await task
    except asyncio.CancelledError:
        done = None
    assert done is None, (
        f"T-0204 C10: the action finished before the cancellation (no settle wait): {done}"
    )
    assert ex.latest_frame is f, "T-0204 C10: a cancelled settle registers nothing"
    taken = len(d.shot_log)
    await asyncio.sleep(0.15)
    assert len(d.shot_log) == taken == n0 + 1, "no shot after the cancellation"
    d.on_shot = None
    refused = await ex.execute(call("click_at", stale(await base_phash(jpeg(first))), **CLICK))
    assert refused.status == "refused", refused


async def test_cancel_during_a_hanging_settle_shot() -> None:
    """C10 (review F2): a cancellation while the settle's 2nd screenshot hangs propagates (it is not taken for a failed
    shot, which would keep the 1st); nothing is registered and ``latest_frame`` is unchanged."""
    d = SettleDriver()
    ex = build(d)
    f = await own(ex)
    first = paint(base_page(), BIG, 20)
    d.pages.extend([first, "hang"])
    since = len(d.calls)
    task = asyncio.ensure_future(act(ex, "press_key", {"combo": "Escape"}))
    for _ in range(400):
        await asyncio.sleep(0.005)
        if names(d, since).count("screenshot") >= 2:
            break
    assert names(d, since).count("screenshot") == 2, (
        f"premise: the 2nd shot hangs: {names(d, since)}"
    )
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        done = await asyncio.wait_for(task, 5)
    except asyncio.CancelledError:
        done = None
    assert done is None, f"T-0204 C10: the cancellation was swallowed by the settle: {done}"
    assert ex.latest_frame is f, "T-0204 C10: a cancelled settle registers nothing"
    refused = await ex.execute(call("click_at", stale(await base_phash(jpeg(first))), **CLICK))
    assert refused.status == "refused", refused


async def test_guard_off_settle_still_runs_and_reads_shifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ruling C-R3: with ``PRECLICK_GUARD = False`` the settle still runs (2 shots on a static page) and still takes its
    shift marks (C-R1): the animated page with a shift between shots 2 and 3 takes 4 shots, not the cell rule's 3."""
    monkeypatch.setattr(executor_mod, "PRECLICK_GUARD", False)
    monkeypatch.setattr(executor_mod, "STABLE_SCHEDULE_S", (0.02, 0.04, 0.06), raising=False)
    d = SettleDriver()
    ex = build(d)
    await own(ex)
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed" and len(shots) == 2, (
        f"T-0204 C-R3: the guard switch does not turn the settle off: {len(shots)} shots"
    )
    d.pages.extend([anim(k) for k in range(8)])
    shift_at(d, len(d.shot_log), 3)
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed" and len(shots) == 4, (
        f"T-0204 C-R1/C-R3: with the guard off the settle still reads the shift between shots 2 and 3: "
        f"{len(shots)} shots"
    )


# ============================================================================================ C9: the note


async def test_note_only_past_two_shots(monkeypatch: pytest.MonkeyPatch) -> None:
    """C9: a 2-shot settle writes no note; a 3-shot settle writes one with exactly the five keys (``shots: 3``,
    ``stable: true``, ``waited_ms`` its time: >= 0.35 s on the default schedule); the outcome's ``resolved`` keys are
    BASE's (no key added); a raising ``on_note`` is swallowed; the note carries no phash."""
    d = SettleDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    await own(ex)
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert len(shots) == 2 and settle_notes(notes) == [], notes
    b = base_page()
    d.pages.extend([paint(b, BIG, 20), b, b])
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert len(shots) == 3, len(shots)
    (n,) = settle_notes(notes)
    assert (n["shots"], n["stable"]) == (3, True), n
    assert 340 <= n["waited_ms"] <= 1300, n
    assert ex.latest_frame.phash not in json.dumps(n)

    bd = SettleDriver()
    bex = build(bd, module=base_module())
    await own(bex)
    bd.pages.extend([paint(b, BIG, 20), b, b])
    bout = await act(bex, "press_key", {"combo": "Escape"})
    assert sorted(out.resolved) == sorted(bout.resolved), (
        f"T-0204 C9: no resolved key added: {sorted(out.resolved)} vs BASE {sorted(bout.resolved)}"
    )

    def boom(_: dict[str, Any]) -> None:
        raise Boom("note sink failed")

    ex.on_note = boom
    d.pages.extend([paint(b, BIG, 20), b, b])
    out, shots, _ = await acted(d, ex, "press_key", {"combo": "Escape"})
    assert out.status == "executed" and len(shots) == 3, (out, len(shots))


# ============================================================================================ C10: negatives


SEQUENCE = [
    ("click_at", CLICK),
    ("type_at", TYPE),
    ("scroll", {"direction": "down"}),
    ("press_key", {"combo": "Enter"}),
    ("wait", {}),
    ("type_at", {"x": 900, "y": 20, "text": "abc", "press_enter": True}),
]


def _arg(a: Any) -> Any:
    # a shift or navigation mark is a fresh object: its repr carries a memory address, never equal across two runs
    return a if isinstance(a, (int, float, str)) else re.sub(r" at 0x[0-9a-fA-F]+", "", repr(a))


def _norm(calls: list[tuple[str, tuple[Any, ...]]]) -> list[Any]:
    return [[n, [_arg(a) for a in args]] for n, args in calls]


async def trace(
    module: Any, make: Callable[[], Any], steps: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    d = make()
    ex = build(d, module=module)
    rec: dict[str, Any] = {"outs": [], "frames": []}
    await own(ex)
    for tool, args in steps:
        out = await act(ex, tool, args)
        rec["outs"].append([out.status, out.action, out.reason, out.resolved])
        rec["frames"].append([ex.latest_frame.phash, ex.latest_frame.jpeg.hex()])
    rec["calls"] = _norm(d.calls)
    return rec


def same(got: dict[str, Any], want: dict[str, Any], what: str) -> None:
    for key in ("outs", "frames", "calls"):
        g, w = json.dumps(got[key], default=str), json.dumps(want[key], default=str)
        assert g == w, (
            f"T-0204 C10 {what}: {key} differs from BASE {BASE_SHA[:8]}:\n got  {got[key]}\n base {want[key]}"
        )


def _changing() -> list[Image.Image]:
    b = base_page()
    return [paint(b, BIG, 20), b, anim(0), anim(1), paint(b, X, 0), b] * 3


def plain_scripted() -> PlainScripted:
    d = PlainScripted()
    d.pages.extend(_changing())
    return d


def settle_scripted() -> SettleDriver:
    d = SettleDriver()
    d.pages.extend(_changing())
    return d


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(FakeDriver, id="FakeDriver"),
        pytest.param(FieldDriver, id="FieldDriver"),
        pytest.param(lambda: PageDriver(content_height=1220.0), id="PageDriver"),
        pytest.param(ThreeOfFour, id="three-of-four"),
        pytest.param(plain_scripted, id="changing-page-no-capability"),
    ],
)
async def test_fakes_without_capability_base(make: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """C10 (negative): a driver class without B1's four methods (every pre-B fake, three of the four, and a changing
    page without them) makes BASE's driver calls, outcomes, bytes and phashes for every action: no settle."""
    base = base_module()
    for mod in (executor_mod, base):
        monkeypatch.setattr(mod, "WAIT_S", 0.01)
    same(
        await trace(executor_mod, make, SEQUENCE),
        await trace(base, make, SEQUENCE),
        "without the capability",
    )


async def test_stable_screen_off_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """C10 (negative): ``STABLE_SCREEN = False`` (read at call time) on a capability driver over a changing page is
    BASE: the same calls, outcomes, bytes and phashes."""
    assert const("STABLE_SCREEN") is True
    monkeypatch.setattr(executor_mod, "STABLE_SCREEN", False)
    base = base_module()
    for mod in (executor_mod, base):
        monkeypatch.setattr(mod, "WAIT_S", 0.01)
    same(
        await trace(executor_mod, settle_scripted, SEQUENCE),
        await trace(base, settle_scripted, SEQUENCE),
        "off",
    )


async def test_frame_and_overview_untouched() -> None:
    """C1/C10 (negative, H-T204-6): ``frame()`` over a changing page and ``page_overview`` on a capability driver are
    BASE: one screenshot for ``frame()``, and the same calls, outcomes, bytes and phashes."""

    async def frames(module: Any) -> dict[str, Any]:
        d = settle_scripted()
        ex = build(d, module=module)
        got = [await own(ex) for _ in range(3)]
        return {
            "outs": [],
            "frames": [[f.phash, f.jpeg.hex()] for f in got],
            "calls": _norm(d.calls),
        }

    tip, base = await frames(executor_mod), await frames(base_module())
    same(tip, base, "frame()")
    assert [c[0] for c in tip["calls"]].count("screenshot") == 3, "one screenshot per frame()"

    async def overview(module: Any) -> dict[str, Any]:
        d = GuardPage(content_height=1220.0)
        ex = build(d, module=module)
        await own(ex)
        out = await ex.execute(call("page_overview"))
        return {
            "outs": [[out.status, out.action, out.reason, sorted(out.resolved)]],
            "frames": [],
            "calls": _norm(d.calls),
        }

    same(await overview(executor_mod), await overview(base_module()), "page_overview")
