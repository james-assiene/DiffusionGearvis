"""T-0200 criteria 1-9 against fake drivers: an own capture whose pixels are near-uniform is taken again, every
``BLANK_RECAPTURE_INTERVAL_S``, for a bounded time; the first painted capture (or the last when the budget is spent) is
the frame and the only one registered; normal pages cost one screenshot and nothing else (docs/tasks/T-0200.md).

**The seam these rows pin** (the task's definitions). ``executor.blank_frame(jpeg: bytes) -> bool`` is pure and takes
the screenshot's JPEG bytes (reading: "decode in reduced/draft mode if needed" says it decodes, so it takes bytes, not
a decoded ``Image``). Constants module-level, read at call time (the rows monkeypatch them): ``BLANK_GRID_W = 64``,
``BLANK_LUMA_RANGE = 8``, ``BLANK_RECAPTURE_INTERVAL_S = 0.1``, ``BLANK_RECAPTURE_ACTION_S = 3.0``,
``BLANK_RECAPTURE_FRAME_S = 1.5``, ``CAPTURE_RESERVE_S = 1.0``. ``BrowserToolExecutor.on_note`` (default ``None``)
is set after construction.

**The fakes.** ``ShotsMixin`` puts a script of JPEG bytes (or exceptions) in front of any T-0111c-shaped fake's
``screenshot()``: each call is recorded in ``driver.calls`` as ``("screenshot", ())`` (through the fake's own
``_enter``, so a killed fake still raises), pops the next scripted item, and falls back to the fake's own picture once
the script is empty (``forever`` repeats one item instead). ``on_shot`` runs after each call (the extent and
cancellation rows move the page or signal there). The capability fakes are imported, not copied: T-0198's
``FrontMixin``, T-0174's ``NavMixin``, T-0192's ``UrlsMixin``, T-0151's ``PageDriver``, T-0160's ``FieldDriver``.

**Oracles.** Never the code under test: the uniform and sparse pictures are drawn by hand with their luma ranges
computed below; a kept frame is identified by its JPEG bytes (1280x800 is not resized, so a frame's ``jpeg`` is the
screenshot's bytes) and its phash by **BASE's** executor (c9614f62, loaded from git) capturing the same bytes under the
same salt (``secrets.randbits`` patched to 0); the normal-page rows are a differential against that BASE executor.
"Sleep" is observed by wrapping ``asyncio.sleep`` (the implementation notes require it, and it is cancellable);
``time.sleep`` raises.

**Readings (reported):**

- C2 "waits ``BLANK_RECAPTURE_INTERVAL_S``": the interval is a lower bound between two screenshots of one capture.
- C3 "stopping once less than ``CAPTURE_RESERVE_S`` remains": with ``ACTION_TIMEOUT_S = 1.0`` and a 0.7 s settle,
  0.3 s remains at the first capture, so the after-frame is one screenshot.
- C9 ``waited_ms``: which clock is not said (the executor's ``now_ms`` is the session clock, constant in the fakes);
  pinned only as an ``int`` in ``0 .. budget + 500 ms``.
- C1 timing: "median <= 5 ms" is measured as CPU time of this thread (``time.thread_time``, .claude/rules/tests.md
  on load), median of 15 calls, on a busy 1280x800 q85 page.

Regime: CPU over fakes; real asyncio time with the budgets patched small except in the deadline and default-bound
rows (about 3 s of wall time in total).
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import statistics
import sys
import time
import types
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from PIL import Image, ImageDraw

from local_astra.computer import executor as executor_mod
from local_astra.contracts.perception import Frame
from local_astra.contracts.toolcall import FrameGeometry, ToolInvocation, ToolOutcome
from tests import _git_range as gr
from tests.unit.test_computer_executor import CLOCK_MS, INPUT, FakeDriver, fake_jpeg
from tests.unit.test_computer_web_executor import FieldDriver
from tests.unit.test_t0151_executor import PageDriver, page_jpeg
from tests.unit.test_t0174_navigated_executor import NavMixin
from tests.unit.test_t0192_checkout_executor import START, UrlsMixin
from tests.unit.test_t0198_front_executor import FRONT, FrontMixin

#: T-0200's BASE (docs/tasks/T-0200.md: the branch's fork, the ruling commit). A literal, never a moving ref.
BASE_SHA = "c9614f62f05f1feec1dd0479a81efca711a6e0c3"
W, H = 1280, 800
NOTE_WHAT = "live frame: blank capture retaken"
NOTE_KEYS = {"what", "path", "captures", "blank", "waited_ms"}
PASSWORD = {"tag": "input", "type": "password", "name": "pw"}


# ============================================================================================ pictures (by hand)


def _jpeg(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


_CACHE: dict[Any, bytes] = {}


def flat(rgb: tuple[int, int, int]) -> bytes:
    """A uniform 1280x800 q85 JPEG (luma range 0 at any grid: one colour)."""
    key = ("flat", rgb)
    if key not in _CACHE:
        _CACHE[key] = _jpeg(Image.new("RGB", (W, H), rgb))
    return _CACHE[key]


WHITE = flat((255, 255, 255))
BLACK = flat((0, 0, 0))
GREY = flat((128, 128, 128))
DARK = flat((0x2B, 0x2B, 0x2B))
RED = flat((255, 0, 0))
#: distinct uniform captures (so "the last" is identifiable by bytes)
UNIFORMS = [flat((v, v, v)) for v in (255, 250, 245, 240, 235, 230)]


def text_line() -> bytes:
    """White with one 14 px dark text line: 40 glyph strokes 6 px wide, 3 px apart, x 100..460, y 100..113. At the
    64-px grid (20x20 px cells) a cell under the line is about 0.7 x 0.67 covered: luma about 255 - 120, range far
    above 8; its variance over the 2560 cells is about 18 cells' worth, small."""
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i in range(40):
        x = 100 + 9 * i
        d.rectangle([x, 100, x + 5, 113], fill=(20, 20, 20))
    return _jpeg(img)


def hairline() -> bytes:
    """White with one 1 px black line across the width at y = 10: a 20-px cell row averages 255 - 255/20 = 242.25,
    so the grid's luma range is about 12.75 (> 8: content), while the variance over 2560 cells is
    (64/2560) * (1 - 64/2560) * 12.75**2 = 3.96 (< 8): a variance test calls it blank."""
    img = Image.new("RGB", (W, H), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([0, 10, W - 1, 10], fill=(0, 0, 0))
    return _jpeg(img)


def skeleton() -> bytes:
    """A #f0f0f0 loading skeleton on white: six cell-aligned bars 600x40 px; luma range 255 - 240 = 15 (> 8)."""
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i in range(6):
        y = 100 + 80 * i
        d.rectangle([100, y, 699, y + 39], fill=(0xF0, 0xF0, 0xF0))
    return _jpeg(img)


def two_tone(a: int, b: int) -> bytes:
    """Grey ``a`` with a cell-aligned 640x400 block of grey ``b`` at (320, 200): grid range ``|a - b|``."""
    img = Image.new("RGB", (W, H), (a, a, a))
    ImageDraw.Draw(img).rectangle([320, 200, 959, 599], fill=(b, b, b))
    return _jpeg(img)


def busy_page() -> bytes:
    """A busy 1280x800 q85 page (text-like strokes everywhere): a realistic decode cost for the timing row."""
    key = ("busy",)
    if key not in _CACHE:
        img = Image.new("RGB", (W, H), (250, 250, 250))
        d = ImageDraw.Draw(img)
        for row in range(0, H, 18):
            for col in range(0, W, 9):
                v = (row * 7 + col * 13) % 97
                if v < 60:
                    d.rectangle(
                        [col, row, col + 5, row + 12], fill=(v, (v * 3) % 255, (v * 5) % 255)
                    )
        _CACHE[key] = _jpeg(img)
    return _CACHE[key]


#: a painted page (T-0111c's fake picture: a gradient with a block)
CONTENT = fake_jpeg((W, H), 1, 0)
CONTENT_B = fake_jpeg((W, H), 1, 2)


# ============================================================================================ the fakes


class ShotsMixin:
    """Scripted screenshots in front of a T-0111c-shaped fake (module docstring)."""

    calls: list[tuple[str, tuple[Any, ...]]]
    shots: list[str]
    active: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.script: deque[bytes | BaseException] = deque()
        self.forever: bytes | None = None
        self.on_shot: Callable[[int], None] | None = None
        self.shot_times: list[float] = []
        self.returned: list[bytes] = []

    def queue(self, *items: bytes | BaseException) -> None:
        self.script.extend(items)

    async def screenshot(self) -> bytes:
        self.shot_times.append(time.monotonic())
        if self.script or self.forever is not None:
            self._enter("screenshot")  # type: ignore[attr-defined]
            self.shots.append(self.active)
            item = self.script.popleft() if self.script else self.forever
            assert item is not None
            if self.on_shot is not None:
                self.on_shot(len(self.shot_times))
            if isinstance(item, BaseException):
                raise item
            self.returned.append(item)
            return item
        out: bytes = await super().screenshot()  # type: ignore[misc]
        self.returned.append(out)
        if self.on_shot is not None:
            self.on_shot(len(self.shot_times))
        return out


class ShotDriver(ShotsMixin, FakeDriver):
    pass


class ShotPage(ShotsMixin, PageDriver):
    pass


class ShotField(ShotsMixin, FieldDriver):
    pass


class SlowSettle(ShotsMixin, FakeDriver):
    """``settle()`` takes ``settle_s`` of real time (a page still loading)."""

    settle_s = 0.0

    async def settle(self) -> None:
        await super().settle()
        if self.settle_s:
            await asyncio.sleep(self.settle_s)


class AllDriver(ShotsMixin, FrontMixin, NavMixin, UrlsMixin, PageDriver):
    """Every optional capability the executor looks up (fronting, navigation mark, checkout read, metrics,
    whole-page capture) over one page, so a loop can be checked against all of them at once."""


class AllField(ShotsMixin, FrontMixin, NavMixin, UrlsMixin, FieldDriver):
    pass


class SlowShotsMixin:
    """A screenshot that awaits before answering (RV1/RV2): ``slow(n, since_s) -> float | None`` is asked at the
    n-th screenshot call (``since_s``: seconds since ``t0``); a float delays it that long, ``math.inf`` blocks until
    cancelled. ``entered`` is set when a delayed call starts; ``cancelled_in_shot`` counts cancellations there."""

    slow: Callable[[int, float], float | None] | None = None
    t0: float = 0.0
    cancelled_in_shot: int = 0
    entered: asyncio.Event | None = None
    shot_times: list[float]

    async def screenshot(self) -> bytes:
        n = len(self.shot_times) + 1
        delay = self.slow(n, time.monotonic() - self.t0) if self.slow is not None else None
        if delay is not None:
            if self.entered is not None:
                self.entered.set()
            try:
                if math.isinf(delay):
                    await asyncio.Event().wait()
                else:
                    await asyncio.sleep(delay)
            except asyncio.CancelledError:
                self.cancelled_in_shot += 1
                raise
        out: bytes = await super().screenshot()  # type: ignore[misc]
        return out


class SlowDriver(SlowShotsMixin, ShotDriver):
    pass


class SlowPage(SlowShotsMixin, ShotPage):
    pass


class SlowField(SlowShotsMixin, ShotField):
    pass


class Boom(RuntimeError):
    """A scripted screenshot failure."""


# ============================================================================================ helpers


def const(name: str) -> Any:
    value = getattr(executor_mod, name, None)
    assert value is not None, f"T-0200: executor.{name} does not exist yet"
    return value


def blank_frame(jpeg: bytes) -> bool:
    fn = const("blank_frame")
    out = fn(jpeg)
    assert isinstance(out, bool), f"T-0200 C1: blank_frame returns a bool, not {out!r}"
    return out


def build(
    d: Any, *, module: Any = executor_mod, notes: list[dict[str, Any]] | None = None, **kw: Any
) -> Any:
    if kw.get("web"):
        kw.setdefault("start_origin", START)
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
        call_id=f"z{next(_N)}",
        turn_id=turn,
        name=name,
        arguments=arguments,
        source="draft",
        frame=g,
    )


async def act(ex: Any, d: Any, i: ToolInvocation, timeout: float = 15.0) -> ToolOutcome:
    d.turn = i.turn_id
    out = await asyncio.wait_for(ex.execute(i), timeout)
    assert isinstance(out, ToolOutcome), type(out)
    return out


async def frame_of(ex: Any, timeout: float = 15.0) -> Frame:
    f = await asyncio.wait_for(ex.frame(), timeout)
    assert isinstance(f, Frame), type(f)
    return f


def names(d: Any, since: int = 0) -> list[str]:
    return [c[0] for c in d.calls[since:]]


def shot_idx(d: Any, since: int = 0) -> list[int]:
    return [i for i, c in enumerate(d.calls) if i >= since and c[0] == "screenshot"]


@pytest.fixture(autouse=True)
def zero_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fixed phash salt (0) for this module's executor and BASE's (both read ``secrets.randbits``)."""
    monkeypatch.setattr(executor_mod.secrets, "randbits", lambda k: 0)


class Sleeps(list[float]):
    pass


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> Iterator[Sleeps]:
    """Every ``asyncio.sleep`` delay awaited (the fakes here never sleep unless a row asks); ``time.sleep`` raises."""
    seen = Sleeps()
    real = asyncio.sleep

    async def rec(delay: float, result: Any = None) -> Any:
        seen.append(float(delay))
        return await real(delay, result)

    def blocking(*a: Any) -> None:
        raise AssertionError("T-0200: a blocking time.sleep on the event loop")

    monkeypatch.setattr(asyncio, "sleep", rec)
    monkeypatch.setattr(time, "sleep", blocking)
    yield seen


# -- BASE c9614f62's executor, for the differential and as the phash oracle

_BASE: types.ModuleType | None = None


def base_module() -> types.ModuleType:
    global _BASE
    if _BASE is None:
        gr.require_commits(BASE_SHA)
        src = gr._git("show", f"{BASE_SHA}:local_astra/computer/executor.py").stdout
        assert "def _capture" in src and "BLANK_" not in src, (
            "premise: BASE's executor has no re-capture"
        )
        name = "_t0200_base_executor"
        mod = types.ModuleType(name)
        mod.__file__ = f"<{BASE_SHA[:8]}>/local_astra/computer/executor.py"
        sys.modules[name] = mod
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        _BASE = mod
    return _BASE


async def base_phash(jpeg: bytes) -> str:
    """BASE's own frame of exactly these bytes, salt 0: the phash a kept capture of them must carry."""
    d = ShotDriver()
    d.queue(jpeg)
    f = await build(d, module=base_module()).frame()
    assert f.jpeg == jpeg
    return str(f.phash)


def check_note(
    n: dict[str, Any], *, path: str, captures: int, blank: bool, budget_s: float
) -> None:
    assert set(n) == NOTE_KEYS, f"T-0200 C9: the note has exactly {sorted(NOTE_KEYS)}; got {n}"
    assert n["what"] == NOTE_WHAT, n
    assert n["path"] == path, n
    assert type(n["captures"]) is int and n["captures"] == captures, n
    assert type(n["blank"]) is bool and n["blank"] is blank, n
    assert type(n["waited_ms"]) is int and 0 <= n["waited_ms"] <= budget_s * 1000 + 500, n


# ============================================================================================ criterion 1


def test_blank_frame_true_for_uniform_any_colour() -> None:
    """C1: uniform white, black, grey, #2b2b2b (and red: "any colour") 1280x800 q85 JPEGs are blank; two cell-aligned
    tones 6 apart are blank (<= 8), 12 apart are not (the threshold from both sides); the constants are the task's;
    a busy page decodes and decides in a median <= 5 ms of thread CPU time."""
    assert const("BLANK_GRID_W") == 64 and const("BLANK_LUMA_RANGE") == 8
    assert "blank_frame" in executor_mod.__all__
    for name, jpeg in (
        ("white", WHITE),
        ("black", BLACK),
        ("grey", GREY),
        ("#2b2b2b", DARK),
        ("red", RED),
    ):
        assert blank_frame(jpeg) is True, f"T-0200 C1: a uniform {name} capture is blank"
    assert blank_frame(two_tone(128, 134)) is True, "T-0200 C1: a grid range of 6 (<= 8) is blank"
    assert blank_frame(two_tone(128, 140)) is False, (
        "T-0200 C1: a grid range of 12 (> 8) is content"
    )
    page = busy_page()
    assert blank_frame(page) is False
    times = []
    for _ in range(15):
        t0 = time.thread_time()
        blank_frame(page)
        times.append(time.thread_time() - t0)
    med = statistics.median(times)
    assert med <= 0.005, (
        f"T-0200 C1 (James 2026-10-29): median {med * 1000:.2f} ms > 5 ms on a 1280x800 q85 page"
    )


def test_blank_frame_false_for_sparse_content() -> None:
    """C1 adversarial (H-T200-1: max - min, so one small element counts): one 14 px text line, one 1 px hairline (a
    variance test misfires here: 3.96), a #f0f0f0 skeleton on white, and every existing fake screenshot are content."""
    from tests.unit._live_browser_support import jpeg_bytes, scene_jpeg

    cases = {
        "text line": text_line(),
        "hairline": hairline(),
        "skeleton": skeleton(),
        "fake_jpeg g0": fake_jpeg((W, H), 0, 0),
        "fake_jpeg g1": fake_jpeg((W, H), 1, 3),
        "jpeg_bytes": jpeg_bytes(W, H, seed=0),
        "jpeg_bytes s7": jpeg_bytes(W, H, seed=7),
        "scene_jpeg": scene_jpeg(W, H, blocks=[(100, 100, 300, 160, (30, 30, 30))]),
        "page_jpeg": page_jpeg(W, 1600, 0),
    }
    wrong = [k for k, v in cases.items() if blank_frame(v)]
    assert wrong == [], f"T-0200 C1: content called blank: {wrong}"


# ============================================================================================ criteria 2, 4


async def test_blank_then_content_after_frame_is_content(sleeps: Sleeps) -> None:
    """C2/C4: after a click the page screenshots white, white, then painted: three screenshots, the kept frame is the
    painted capture's bytes, ``frame_after`` and ``latest_frame`` carry the phash BASE gives those bytes; the two
    waits between them are the interval."""
    const("BLANK_RECAPTURE_INTERVAL_S")
    d = ShotDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await frame_of(ex)
    since = len(d.calls)
    d.queue(WHITE, WHITE, CONTENT)
    out = await act(ex, d, call("click_at", f, x=500, y=500))
    assert out.status == "executed", out
    assert len(shot_idx(d, since)) == 3, names(d, since)
    want = await base_phash(CONTENT)
    assert ex.latest_frame.jpeg == CONTENT, (
        "T-0200 C2: the kept frame is the first non-blank capture"
    )
    assert out.resolved["frame_after"] == want == ex.latest_frame.phash, (out.resolved, want)
    t = d.shot_times[-3:]
    assert t[1] - t[0] >= 0.09 and t[2] - t[1] >= 0.09, f"T-0200 C2: 100 ms between captures; {t}"
    assert len([s for s in sleeps if s > 0]) == 2 and all(
        abs(s - 0.1) < 1e-9 for s in sleeps if s > 0
    ), sleeps
    assert len(notes) == 1, notes


async def test_blank_then_content_frame_path() -> None:
    """C2/C4 on ``frame()``: black, then painted: the frame is the painted capture with BASE's phash for it."""
    const("BLANK_RECAPTURE_FRAME_S")
    d = ShotDriver()
    ex = build(d)
    await frame_of(ex)
    since = len(d.calls)
    d.queue(BLACK, CONTENT_B)
    f = await frame_of(ex)
    assert len(shot_idx(d, since)) == 2
    assert f.jpeg == CONTENT_B and f.phash == await base_phash(CONTENT_B) == ex.latest_frame.phash


async def test_intermediate_blank_not_own_frame() -> None:
    """C4 adversarial: an intermediate blank capture is never an own frame: a ``click_at`` carrying its phash (BASE's
    phash of the white bytes, salt 0) is refused and dispatches nothing; the kept frame's phash executes."""
    const("BLANK_RECAPTURE_INTERVAL_S")
    d = ShotPage()
    ex = build(d)
    await frame_of(ex)
    d.queue(WHITE, CONTENT)
    kept = await frame_of(ex)
    blank_ph = await base_phash(WHITE)
    assert blank_ph != kept.phash, "premise: the white and painted captures hash apart"
    ghost = kept.model_copy(update={"phash": blank_ph})
    since = len(d.calls)
    out = await act(ex, d, call("click_at", ghost, x=500, y=500))
    assert out.status == "refused" and d.dispatched(since) == [], (
        f"T-0200 C4: a click on an intermediate blank capture must be refused; {out}"
    )
    assert ex.extent_of(blank_ph) is None
    ok = await act(ex, d, call("click_at", kept, x=500, y=500))
    assert ok.status == "executed", ok


async def test_extent_brackets_kept_screenshot() -> None:
    """C4: the page scrolls while it loads (to 300 during the first blank screenshot, to 120 during the second), then
    holds: the kept extent is the one read around the painted screenshot: above 120, below 1220 - 120 - 800 = 300,
    viewport 800; a metrics read precedes and follows the kept screenshot with no screenshot between."""
    const("BLANK_RECAPTURE_INTERVAL_S")
    d = ShotPage(content_height=1220.0)
    ex = build(d)
    await frame_of(ex)
    moves = {1: 300.0, 2: 120.0}
    base_n = len(d.shot_times)

    def on_shot(n: int) -> None:
        if n - base_n in moves:
            d.scroll_y = moves[n - base_n]

    d.on_shot = on_shot
    since = len(d.calls)
    d.queue(WHITE, WHITE, CONTENT)
    f = await frame_of(ex)
    assert f.jpeg == CONTENT
    assert ex.extent_of(f.phash) == {"above_px": 120.0, "below_px": 300.0, "viewport_px": 800.0}, (
        f"T-0200 C4: the kept extent is read around the kept screenshot; {ex.extent_of(f.phash)}"
    )
    seq = names(d, since)
    k = max(i for i, n in enumerate(seq) if n == "screenshot")
    assert seq[k - 1] == "layout_metrics" and seq[k + 1] == "layout_metrics", seq


# ============================================================================================ criterion 5


def _norm(calls: list[tuple[str, tuple[Any, ...]]]) -> list[tuple[str, tuple[Any, ...]]]:
    keep = (int, float, str, bool, type(None))
    return [
        (n, tuple(a if isinstance(a, keep) else type(a).__name__ for a in args))
        for n, args in calls
    ]


async def _scenario(module: Any, notes: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Normal pages through every capture path: frames, a click, a replacing type, a secret-field stop, a scroll, a
    key, a page overview and a click on the overview frame; headed (fronting), web mode (checkout read), navigation
    marks and metrics on."""
    d = AllField(dict(PASSWORD))
    ex = build(d, module=module, notes=notes, headless=False, web=True, echo="model")
    rec: dict[str, Any] = {"outs": [], "frames": []}

    async def snap() -> None:
        f = await frame_of(ex)
        rec["frames"].append([f.phash, f.width, f.height, f.jpeg.hex(), ex.extent_of(f.phash)])

    await snap()
    f = ex.latest_frame
    for i in (
        call("click_at", f, x=500, y=500),
        call("type_at", f, x=300, y=120, text="hello", press_enter=False),
        call("scroll", direction="down"),
        call("press_key", key="Enter"),
        call("scroll", None, turn="t2", direction="up"),
    ):
        if i.frame is not None:
            i = i.model_copy(update={"frame": call("x", ex.latest_frame).frame})
        out = await act(ex, d, i)
        rec["outs"].append([out.status, out.action, out.reason, out.resolved])
    await snap()
    rec["calls"] = _norm(d.calls)
    rec["latest"] = ex.latest_frame.phash
    return rec


async def _overview_scenario(module: Any, notes: list[dict[str, Any]] | None) -> dict[str, Any]:
    d = AllDriver(content_height=1220.0)
    ex = build(d, module=module, notes=notes, headless=False, echo="model")
    rec: dict[str, Any] = {"outs": []}
    await frame_of(ex)
    out = await act(ex, d, call("page_overview"))
    rec["outs"].append([out.status, out.action, out.reason, out.resolved])
    ov = ex.overview_frame_of(out.resolved["overview"])
    assert isinstance(ov, Frame), "premise: the overview frame registers"
    out = await act(ex, d, call("click_at", ov, x=500, y=900))
    rec["outs"].append([out.status, out.action, out.reason, out.resolved])
    rec["calls"] = _norm(d.calls)
    rec["latest"] = [ex.latest_frame.phash, ex.latest_frame.jpeg.hex()]
    return rec


async def test_normal_page_one_screenshot_no_sleep_no_note(sleeps: Sleeps) -> None:
    """C5 (negative, BASE differential): painted first captures through every capture path give BASE c9614f62's
    driver call sequence (one ``screenshot`` per capture), outcomes, ``resolved``, bytes, phashes and extents, with
    no ``asyncio.sleep`` and no note; ``on_note`` defaults to ``None``."""
    for name in ("BLANK_RECAPTURE_INTERVAL_S", "BLANK_RECAPTURE_ACTION_S", "CAPTURE_RESERVE_S"):
        const(name)
    plain = executor_mod.BrowserToolExecutor(now_ms=lambda: 0, driver=FakeDriver())
    assert plain.on_note is None, "T-0200 C9: on_note defaults to None"
    base = base_module()
    for scenario in (_scenario, _overview_scenario):
        want = await scenario(base, None)
        sleeps.clear()
        notes: list[dict[str, Any]] = []
        got = await scenario(executor_mod, notes)
        assert sleeps == [], (
            f"T-0200 C5 (James 2026-10-29): no wait on a painted page; slept {sleeps}"
        )
        assert notes == [], f"T-0200 C5: no note on a painted page; {notes}"
        for key in want:
            assert json.dumps(got[key], default=str) == json.dumps(want[key], default=str), (
                f"T-0200 C5 {scenario.__name__}: {key} differs from BASE {BASE_SHA[:7]}"
            )


# ============================================================================================ criteria 3, 6


@pytest.fixture
def small(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    """The re-capture budgets patched small (read at call time, C2): interval 20 ms, action 200 ms, frame 150 ms."""
    vals = {
        "BLANK_RECAPTURE_INTERVAL_S": 0.02,
        "BLANK_RECAPTURE_ACTION_S": 0.2,
        "BLANK_RECAPTURE_FRAME_S": 0.15,
    }
    for k, v in vals.items():
        # raising=False: at BASE the names are absent and the row fails on its own assertion, not in setup
        monkeypatch.setattr(executor_mod, k, v, raising=False)
    return vals


async def test_uniform_forever_bounded_action(small: dict[str, float]) -> None:
    """C6 (negative): a page that screenshots uniform forever: the click is ``executed`` with at most
    ``1 + ceil(0.2 / 0.02) = 11`` screenshots, the loop ran its budget (>= 0.15 s, so it is not cut short), and the
    kept frame is the *last* capture (distinct uniform greys, identified by bytes), registered as ``latest_frame``."""
    d = ShotDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await frame_of(ex)
    n = 1 + math.ceil(0.2 / 0.02)
    script = [UNIFORMS[i % len(UNIFORMS)] for i in range(n + 5)]
    d.queue(*script)
    since, t_since = len(d.calls), len(d.shot_times)
    t0 = time.monotonic()
    out = await act(ex, d, call("click_at", f, x=500, y=500))
    took = time.monotonic() - t0
    assert out.status == "executed", out
    k = len(shot_idx(d, since))
    assert 2 <= k <= n, f"T-0200 C2/C6: {k} screenshots (bound {n})"
    span = d.shot_times[-1] - d.shot_times[t_since]
    assert span >= 0.15 and took < 0.2 + 0.5, (span, took)
    assert ex.latest_frame.jpeg == script[k - 1], (
        "T-0200 C2: the last capture is kept when the budget is spent"
    )
    assert out.resolved["frame_after"] == ex.latest_frame.phash
    assert len(notes) == 1
    check_note(notes[0], path="action", captures=k, blank=True, budget_s=0.2)


async def test_uniform_forever_bounded_frame(small: dict[str, float]) -> None:
    """C6 (negative) on ``frame()``: at most ``1 + ceil(0.15 / 0.02) = 9`` screenshots, the budget used (>= 0.1 s),
    the last capture returned and registered, within the budget + 0.5 s."""
    d = ShotDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    await frame_of(ex)
    n = 1 + math.ceil(0.15 / 0.02)
    script = [UNIFORMS[i % len(UNIFORMS)] for i in range(n + 5)]
    d.queue(*script)
    t_since = len(d.shot_times)
    t0 = time.monotonic()
    f = await frame_of(ex)
    took = time.monotonic() - t0
    k = len(d.shot_times) - t_since
    assert 2 <= k <= n, f"T-0200 C6: {k} screenshots (bound {n})"
    assert d.shot_times[-1] - d.shot_times[t_since] >= 0.1 and took < 0.15 + 0.5, took
    assert f.jpeg == script[k - 1] and ex.latest_frame is f
    check_note(notes[0], path="frame", captures=k, blank=True, budget_s=0.15)


async def test_uniform_forever_frame_default_bound() -> None:
    """C3/C6 at the task's constants: ``frame()`` on an always-white page returns within 1.5 + 0.5 s with at most
    ``1 + ceil(1.5 / 0.1) = 16`` screenshots, having used most of its budget (>= 1.2 s)."""
    assert const("BLANK_RECAPTURE_FRAME_S") == 1.5 and const("BLANK_RECAPTURE_INTERVAL_S") == 0.1
    d = ShotDriver()
    ex = build(d)
    await frame_of(ex)
    d.forever = WHITE
    t_since = len(d.shot_times)
    t0 = time.monotonic()
    f = await frame_of(ex)
    took = time.monotonic() - t0
    k = len(d.shot_times) - t_since
    assert f.jpeg == WHITE and 2 <= k <= 16, k
    assert 1.2 <= took < 2.0, f"T-0200 C3/C6: frame() took {took:.2f} s"


async def test_recapture_never_fails_action_near_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """C3 adversarial: blank forever, ``ACTION_TIMEOUT_S = 1.0``, a 0.7 s settle: ``executed``, never a timeout
    failure, one screenshot (0.3 s < ``CAPTURE_RESERVE_S`` remains). With ``ACTION_TIMEOUT_S = 2.0`` and a 0.2 s settle
    the loop re-captures (>= 2 screenshots) and still returns ``executed`` with at least the reserve to spare."""
    assert const("BLANK_RECAPTURE_ACTION_S") == 3.0 and const("CAPTURE_RESERVE_S") == 1.0
    for timeout_s, settle_s, at_least, at_most in ((1.0, 0.7, 1, 1), (2.0, 0.2, 2, 11)):
        monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", timeout_s)
        d = SlowSettle()
        ex = build(d)
        f = await frame_of(ex)
        d.settle_s = settle_s
        d.forever = WHITE
        since = len(d.calls)
        t0 = time.monotonic()
        out = await act(ex, d, call("click_at", f, x=500, y=500))
        took = time.monotonic() - t0
        assert out.status == "executed", f"T-0200 C3: ACTION_TIMEOUT_S={timeout_s}: {out}"
        k = len(shot_idx(d, since))
        assert at_least <= k <= at_most, f"T-0200 C3: ACTION_TIMEOUT_S={timeout_s}: {k} screenshots"
        # the loop stops once less than the 1.0 s reserve remains: by ``timeout_s - 1.0`` (+0.3 s slack), or right
        # after the settle when the reserve was already reached
        assert took < max(settle_s + 0.25, timeout_s - 1.0 + 0.3), (
            f"T-0200 C3: stops once less than the reserve remains; took {took:.2f}"
        )


async def test_guarded_type_stop_recaptures(small: dict[str, float]) -> None:
    """C3: the guarded-type stop's capture (a password field) is an action path: white then painted gives the
    painted frame and one ``action`` note; nothing is typed."""
    d = ShotField(dict(PASSWORD))
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await frame_of(ex)
    d.queue(WHITE, CONTENT)
    out = await act(ex, d, call("type_at", f, x=300, y=120, text="secret", press_enter=False))
    assert out.status == "failed" and out.resolved.get("typed") == 0 and d.typed == [], out
    assert out.resolved["frame_after"] == await base_phash(CONTENT) == ex.latest_frame.phash
    assert len(notes) == 1
    check_note(notes[0], path="action", captures=2, blank=False, budget_s=0.2)


async def test_overview_frame_click_recaptures(small: dict[str, float]) -> None:
    """C3: the overview-frame click's after-frame is an action path: white then painted gives the painted frame."""
    d = ShotPage(content_height=1220.0)
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes, echo="model")
    await frame_of(ex)
    out = await act(ex, d, call("page_overview"))
    ov = ex.overview_frame_of(out.resolved["overview"])
    assert isinstance(ov, Frame), "premise: the overview frame registers"
    d.queue(WHITE, CONTENT)
    out = await act(ex, d, call("click_at", ov, x=500, y=900))
    assert out.status == "executed", out
    assert out.resolved["frame_after"] == await base_phash(CONTENT) == ex.latest_frame.phash
    check_note(notes[0], path="action", captures=2, blank=False, budget_s=0.2)


# ============================================================================================ criterion 7


async def test_loop_calls_only_screenshot_and_metrics(small: dict[str, float]) -> None:
    """C7: between the first and the kept screenshot of an after-frame only ``screenshot`` and ``layout_metrics``
    are called (no settle, front, focus read, checkout read, navigation read, input or event drain)."""
    d = AllDriver(content_height=1220.0)
    ex = build(d, headless=False, web=True)
    f = await frame_of(ex)
    d.queue(WHITE, GREY, WHITE, CONTENT)
    since = len(d.calls)
    out = await act(ex, d, call("click_at", f, x=500, y=500))
    assert out.status == "executed", out
    idx = shot_idx(d, since)
    assert len(idx) == 4, names(d, since)
    between = set(names(d)[idx[0] : idx[-1] + 1])
    assert between <= {"screenshot", "layout_metrics"}, f"T-0200 C7: the loop called {between}"


async def test_front_count_unchanged_headed(small: dict[str, float]) -> None:
    """C7: headed, one click on one page whose after-frame is re-taken: one ``bring_to_front`` (T-0198: once per page
    key), before the click's first input, none during the loop."""
    d = AllDriver(content_height=1220.0)
    ex = build(d, headless=False)
    f = await frame_of(ex)
    d.queue(WHITE, WHITE, CONTENT)
    since = len(d.calls)
    out = await act(ex, d, call("click_at", f, x=500, y=500))
    assert out.status == "executed", out
    seq = names(d, since)
    assert seq.count("screenshot") == 3, f"premise: the after-frame was re-taken; {seq}"
    assert seq.count(FRONT) == 1, seq
    assert seq.index(FRONT) < min(i for i, n in enumerate(seq) if n in INPUT), seq
    assert seq.index(FRONT) < seq.index("screenshot")


async def test_navigated_checkout_read_before_loop(small: dict[str, float]) -> None:
    """C7: ``navigated_since`` once and the checkout read twice per action (T-0174, T-0192: before dispatch, after the
    last settle), both before the after-frame's first screenshot; none after it."""
    d = AllDriver(content_height=1220.0)
    ex = build(d, web=True)
    f = await frame_of(ex)
    d.queue(WHITE, WHITE, CONTENT)
    since = len(d.calls)
    out = await act(ex, d, call("click_at", f, x=500, y=500))
    assert out.status == "executed" and "navigated" in out.resolved, out
    seq = names(d, since)
    first = seq.index("screenshot")
    assert seq.count("screenshot") == 3, f"premise: the after-frame was re-taken; {seq}"
    assert seq.count("navigated_since") == 1 and seq.count("main_frame_urls") == 2, seq
    assert "navigated_since" not in seq[first:] and "main_frame_urls" not in seq[first:], seq
    assert seq.count("navigation_mark") == 1


async def test_overview_never_recaptured(sleeps: Sleeps) -> None:
    """C7: a ``page_overview`` whose whole-page picture is uniform white is taken once: one
    ``full_page_screenshot``, no ``screenshot``, no wait."""
    const("BLANK_RECAPTURE_INTERVAL_S")

    class WhitePage(AllDriver):
        async def full_page_screenshot(self, max_height_css: int) -> bytes:
            await super().full_page_screenshot(max_height_css)
            return _jpeg(
                Image.new(
                    "RGB", (W, int(min(self.content_height, max_height_css))), (255, 255, 255)
                )
            )

    d = WhitePage(content_height=1220.0)
    ex = build(d)
    await frame_of(ex)
    sleeps.clear()
    since = len(d.calls)
    out = await act(ex, d, call("page_overview"))
    assert out.status == "executed" and "overview" in out.resolved, out
    seq = names(d, since)
    assert seq.count("full_page_screenshot") == 1 and "screenshot" not in seq, seq
    assert sleeps == [], sleeps


# ============================================================================================ criterion 8


async def test_recapture_exception_keeps_previous(small: dict[str, float]) -> None:
    """C8 (amended 0e8c4e4a: a raising re-capture is one blank attempt, retried within the budget; when the budget
    is spent the last *successful* capture is kept): on ``frame()``, white, then black, then every re-capture raises:
    the black capture (the last that succeeded, not the first) is the frame and ``latest_frame``, its phash BASE's
    for those bytes, within the frame budget + 0.5 s, with more than one failed attempt.
    (Before the amendment this row ended the loop at the first raise; it now pins the retry and "last successful".)"""
    d = ShotDriver()
    ex = build(d)
    await frame_of(ex)
    d.queue(WHITE, BLACK)
    d.forever = Boom("synthetic re-capture failure")  # type: ignore[assignment]
    t_since = len(d.shot_times)
    t0 = time.monotonic()
    g = await frame_of(ex)
    took = time.monotonic() - t0
    k = len(d.shot_times) - t_since
    assert g.jpeg == BLACK and ex.latest_frame is g, (
        "T-0200 C8: the last successful capture is kept"
    )
    assert g.phash == await base_phash(BLACK)
    assert 4 <= k <= 1 + math.ceil(0.15 / 0.02), (
        f"T-0200 C8: raises are retried within the budget; {k} shots"
    )
    assert took < 0.15 + 0.5, took


async def test_recapture_exception_retries_within_budget(small: dict[str, float]) -> None:
    """C8 (amended 0e8c4e4a): after a click, [white, raise, painted]: the raise is one blank attempt, so the loop
    waits and takes the painted capture: ``executed``, three screenshots, the kept frame is the painted bytes with
    BASE's phash for them. Then [white, raise forever]: ``executed`` with the white capture (the last successful)
    kept, at most ``1 + ceil(0.2 / 0.02) = 11`` screenshots, within the action budget + 0.5 s."""
    d = ShotDriver()
    ex = build(d)
    f = await frame_of(ex)
    d.queue(WHITE, Boom("synthetic re-capture failure"), CONTENT)
    since = len(d.calls)
    out = await act(ex, d, call("click_at", f, x=500, y=500))
    assert out.status == "executed", out
    assert len(shot_idx(d, since)) == 3, (
        f"T-0200 C8: a raising re-capture is retried; {names(d, since)}"
    )
    assert ex.latest_frame.jpeg == CONTENT
    assert out.resolved["frame_after"] == await base_phash(CONTENT) == ex.latest_frame.phash

    d.queue(WHITE)
    d.forever = Boom("raises forever")  # type: ignore[assignment]
    since = len(d.calls)
    t0 = time.monotonic()
    out = await act(ex, d, call("click_at", ex.latest_frame, x=500, y=500))
    took = time.monotonic() - t0
    assert out.status == "executed", (
        f"T-0200 C8: a capture that cannot be re-taken still executes; {out}"
    )
    k = len(shot_idx(d, since))
    assert 3 <= k <= 1 + math.ceil(0.2 / 0.02), f"T-0200 C8: retried within the budget; {k} shots"
    assert ex.latest_frame.jpeg == WHITE and out.resolved["frame_after"] == await base_phash(WHITE)
    assert took < 0.2 + 0.5, took


@pytest.mark.parametrize("shape", ["hang", "slow_late"])
async def test_slow_recapture_near_deadline_still_executed(
    monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """RV1 (C3 + C8): blank forever, ``ACTION_TIMEOUT_S = 2.0``. ``hang``: every re-capture screenshot blocks until
    cancelled; ``slow_late``: a re-capture started 0.8 s or more into the action takes 1.2 s, so it runs past the action bound. Either way the click is ``executed`` (never a ``TimeoutError``
    failure), the white first capture (the last successful) is kept, and the call returns before 2.0 s: a
    re-capture is bounded by the time left, and its timeout is C8's raising re-capture."""
    assert const("CAPTURE_RESERVE_S") == 1.0
    monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", 2.0)
    d = SlowDriver()
    ex = build(d)
    f = await frame_of(ex)
    n0 = len(d.shot_times)
    d.forever = WHITE

    def slow(n: int, since: float) -> float | None:
        if n <= n0 + 1:
            return None  # the action's first capture is quick and white
        if shape == "hang":
            return math.inf
        return 1.2 if since >= 0.8 else None

    d.slow = slow
    d.t0 = time.monotonic()
    t0 = time.monotonic()
    out = await act(ex, d, call("click_at", f, x=500, y=500))
    took = time.monotonic() - t0
    assert out.status == "executed", (
        f"T-0200 RV1/C3: a slow re-capture must not fail the click; {out}"
    )
    assert ex.latest_frame.jpeg == WHITE and out.resolved["frame_after"] == await base_phash(WHITE)
    assert took < 2.0, f"T-0200 RV1/C3: returned after {took:.2f} s (ACTION_TIMEOUT_S 2.0)"
    assert d.cancelled_in_shot >= 1, "premise: a re-capture was cut while it awaited"


async def test_hanging_late_recapture_capped_at_action_budget_default_timeout() -> None:
    """RV3 (C3, F1): at the default ``ACTION_TIMEOUT_S`` (8.0) the action's re-capture budget
    ``BLANK_RECAPTURE_ACTION_S = 3.0`` is a cap, not just a bound on when the last re-capture *starts*: blank forever,
    and every re-capture screenshot started 2.8 s or more into the action blocks until cancelled. The click is
    ``executed`` with the white capture kept, and returns within 3.0 s + one interval (0.1 s) + 0.5 s of load slack,
    not at the call's ``ACTION_TIMEOUT_S`` end less the frame-build reserve (about 7.7 s). The RV1 row above patches
    ``ACTION_TIMEOUT_S`` to 2.0 and cannot tell those two bounds apart."""
    assert const("ACTION_TIMEOUT_S") == 8.0 and const("BLANK_RECAPTURE_ACTION_S") == 3.0
    assert const("BLANK_RECAPTURE_INTERVAL_S") == 0.1
    d = SlowDriver()
    ex = build(d)
    f = await frame_of(ex)
    n0 = len(d.shot_times)
    d.forever = WHITE
    # the action's first capture (n0 + 1) is quick and white; a re-capture started at >= 2.8 s hangs
    d.slow = lambda n, since: math.inf if n > n0 + 1 and since >= 2.8 else None
    d.t0 = time.monotonic()
    t0 = time.monotonic()
    out = await act(ex, d, call("click_at", f, x=500, y=500))
    took = time.monotonic() - t0
    assert out.status == "executed", (
        f"T-0200 RV3/C3: a hanging re-capture must not fail the click; {out}"
    )
    assert ex.latest_frame.jpeg == WHITE and out.resolved["frame_after"] == await base_phash(WHITE)
    assert d.cancelled_in_shot >= 1, (
        "premise: a late re-capture was started and cut while it awaited"
    )
    assert took < 3.0 + 0.1 + 0.5, (
        f"T-0200 RV3/C3: BLANK_RECAPTURE_ACTION_S 3.0 is a cap; the click returned after {took:.2f} s"
    )


async def test_hanging_recapture_frame_path_bounded() -> None:
    """RV3 (C3/C6, the frame path's hard bound): ``frame()`` on an always-white page whose re-capture screenshots
    started 1.3 s or more after the call block until cancelled: the white capture is returned (and registered)
    within ``BLANK_RECAPTURE_FRAME_S`` (1.5) + 0.5 s; the hanging re-capture was cut, not awaited."""
    assert const("BLANK_RECAPTURE_FRAME_S") == 1.5
    d = SlowDriver()
    ex = build(d)
    await frame_of(ex)
    n0 = len(d.shot_times)
    d.forever = WHITE
    d.slow = lambda n, since: math.inf if n > n0 + 1 and since >= 1.3 else None
    d.t0 = time.monotonic()
    t0 = time.monotonic()
    fr = await frame_of(ex, timeout=5.0)
    took = time.monotonic() - t0
    assert fr.jpeg == WHITE and ex.latest_frame is fr
    assert d.cancelled_in_shot >= 1, (
        "premise: a late re-capture was started and cut while it awaited"
    )
    assert took < 1.5 + 0.5, f"T-0200 RV3/C3: frame() returned after {took:.2f} s"


async def test_cancel_inside_recapture_screenshot_propagates(small: dict[str, float]) -> None:
    """RV2 (C8): ``frame()`` cancelled while a *re-capture* screenshot is awaiting (not during the sleep):
    ``CancelledError`` reaches the caller (a loop catching ``BaseException`` would swallow it), the screenshot saw
    the cancellation, nothing runs on, ``latest_frame`` is unchanged and the white capture is not an own frame."""
    d = SlowDriver()
    ex = build(d)
    prev = await frame_of(ex)
    n0 = len(d.shot_times)
    d.forever = WHITE
    d.entered = asyncio.Event()
    d.slow = lambda n, since: math.inf if n >= n0 + 2 else None
    task = asyncio.ensure_future(ex.frame())
    await asyncio.wait_for(d.entered.wait(), 5)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert task.cancelled(), "T-0200 RV2: the cancellation propagates out of frame()"
    assert d.cancelled_in_shot == 1, d.cancelled_in_shot
    taken = len(d.shot_times)
    await asyncio.sleep(0.1)
    assert len(d.shot_times) == taken, "T-0200 RV2: nothing runs on after the cancellation"
    assert ex.latest_frame is prev
    d.slow, d.forever = None, None
    white = prev.model_copy(update={"phash": await base_phash(WHITE)})
    out = await act(ex, d, call("click_at", white, x=500, y=500))
    assert out.status == "refused", out


async def test_overview_click_near_deadline_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    """RV3 (C3 on ``_act_on_overview``): ``ACTION_TIMEOUT_S = 2.0``, the action budget left at 3.0 s, blank forever
    after an overview-frame click: ``executed`` (the loop stops at the call's deadline less the reserve), within 2 s,
    the white capture kept. Without the deadline the 3 s budget overruns the 2 s bound."""
    assert const("BLANK_RECAPTURE_ACTION_S") == 3.0
    monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", 2.0)
    d = ShotPage(content_height=1220.0)
    ex = build(d, echo="model")
    await frame_of(ex)
    out = await act(ex, d, call("page_overview"))
    ov = ex.overview_frame_of(out.resolved["overview"])
    assert isinstance(ov, Frame), "premise: the overview frame registers"
    d.forever = WHITE
    t0 = time.monotonic()
    out = await act(ex, d, call("click_at", ov, x=500, y=900))
    took = time.monotonic() - t0
    assert out.status == "executed", f"T-0200 RV3: the overview click ran past its bound; {out}"
    assert ex.latest_frame.jpeg == WHITE and took < 2.0, took


async def test_guarded_stop_near_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """RV3 (C3 on ``_dispatch_guarded``): the same on a password field's stop capture: the outcome is the guard's
    own ``failed`` (``typed: 0``, the secret-field reason), not a ``TimeoutError`` failure, within 2 s."""
    assert const("BLANK_RECAPTURE_ACTION_S") == 3.0
    monkeypatch.setattr(executor_mod, "ACTION_TIMEOUT_S", 2.0)
    d = ShotField(dict(PASSWORD))
    ex = build(d)
    f = await frame_of(ex)
    d.forever = WHITE
    t0 = time.monotonic()
    out = await act(ex, d, call("type_at", f, x=300, y=120, text="secret", press_enter=False))
    took = time.monotonic() - t0
    assert out.status == "failed" and out.reason == executor_mod.SECRET_FIELD_REASON, out
    assert out.resolved.get("typed") == 0 and d.typed == [], out
    assert ex.latest_frame.jpeg == WHITE and took < 2.0, took


def test_blank_frame_threshold_edge() -> None:
    """C1 edge: cell-aligned two tones 8 apart are blank (``<= 8``), 9 apart are not."""
    assert blank_frame(two_tone(128, 136)) is True, "T-0200 C1: a grid range of exactly 8 is blank"
    assert blank_frame(two_tone(128, 137)) is False, "T-0200 C1: a grid range of 9 is content"


async def test_first_screenshot_exception_propagates_as_base(small: dict[str, float]) -> None:
    """C8: the first screenshot raising is BASE's: ``execute()`` gives BASE's ``failed`` outcome and ``frame()``
    raises the driver's exception; nothing is re-tried."""
    assert hasattr(executor_mod, "blank_frame"), "T-0200: no blank re-capture yet"
    outs = []
    for module in (base_module(), executor_mod):
        d = ShotDriver()
        ex = build(d, module=module)
        f = await frame_of(ex)
        d.queue(Boom("synthetic first-capture failure"))
        since = len(d.calls)
        out = await act(ex, d, call("click_at", f, x=500, y=500))
        outs.append([out.status, out.action, out.reason, out.resolved])
        assert len(shot_idx(d, since)) == 1
        d.queue(Boom("frame failure"))
        with pytest.raises(Boom):
            await frame_of(ex)
    assert outs[0][0] == "failed" and outs[1] == outs[0], outs


async def test_cancel_mid_loop_registers_nothing(small: dict[str, float]) -> None:
    """C8: ``frame()`` cancelled between two captures of a white page: ``CancelledError`` reaches the caller, no
    further screenshot, ``latest_frame`` is still the previous frame and the white capture is not an own frame."""
    d = ShotDriver()
    ex = build(d)
    prev = await frame_of(ex)
    first = asyncio.Event()
    n0 = len(d.shot_times)
    d.on_shot = lambda n: first.set() if n == n0 + 1 else None
    d.forever = WHITE
    task = asyncio.ensure_future(ex.frame())
    await asyncio.wait_for(first.wait(), 5)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled(), "T-0200 C8: the cancellation propagates"
    taken = len(d.shot_times)
    await asyncio.sleep(0.1)
    assert len(d.shot_times) == taken, "T-0200 C8: nothing runs on after the cancellation"
    assert ex.latest_frame is prev, "T-0200 C8: latest_frame unchanged"
    d.forever = None
    white = prev.model_copy(update={"phash": await base_phash(WHITE)})
    out = await act(ex, d, call("click_at", white, x=500, y=500))
    assert out.status == "refused", f"T-0200 C8: the cancelled capture is not an own frame; {out}"


# ============================================================================================ criterion 9


async def test_note_shape_once_per_capture(small: dict[str, float]) -> None:
    """C9: one note per capture that took more than one screenshot, exact keys, the path named; no URL, host, title,
    phash or exception text anywhere in it."""
    d = AllDriver(content_height=1220.0)
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes, web=True)
    d.queue(WHITE, WHITE, CONTENT)
    f = await frame_of(ex)
    check_note(notes[0], path="frame", captures=3, blank=False, budget_s=0.15)
    d.queue(BLACK, CONTENT_B)
    out = await act(ex, d, call("click_at", f, x=500, y=500))
    assert out.status == "executed"
    assert len(notes) == 2
    check_note(notes[1], path="action", captures=2, blank=False, budget_s=0.2)
    text = json.dumps(notes)
    for leak in (
        "http",
        "example",
        "127.0.0.1",
        str(f.phash),
        str(out.resolved["frame_after"]),
        "Boom",
    ):
        assert leak not in text, f"T-0200 C9: the note carries {leak!r}: {text}"


async def test_note_absent_on_normal(small: dict[str, float]) -> None:
    """C9: painted captures give no note, on either path."""
    d = ShotDriver()
    notes: list[dict[str, Any]] = []
    ex = build(d, notes=notes)
    f = await frame_of(ex)
    out = await act(ex, d, call("click_at", f, x=500, y=500))
    assert out.status == "executed" and notes == [], notes
    d.queue(WHITE, CONTENT)
    await frame_of(ex)
    assert len(notes) == 1, "premise: a re-taken capture notes"


async def test_on_note_raising_swallowed(small: dict[str, float]) -> None:
    """C9: a raising ``on_note`` is swallowed: the capture still returns the painted frame and the action executes."""
    d = ShotDriver()
    ex = build(d)
    seen: list[dict[str, Any]] = []

    def bad(note: dict[str, Any]) -> None:
        seen.append(note)
        raise RuntimeError("log is gone")

    ex.on_note = bad
    d.queue(WHITE, CONTENT)
    f = await frame_of(ex)
    assert f.jpeg == CONTENT and len(seen) == 1
    d.queue(WHITE, CONTENT_B)
    out = await act(ex, d, call("click_at", f, x=500, y=500))
    assert out.status == "executed" and ex.latest_frame.jpeg == CONTENT_B and len(seen) == 2, out


async def test_resolved_keys_unchanged_with_recapture(small: dict[str, float]) -> None:
    """C9 (H-T200-4: no ``resolved`` key): a re-taken after-frame gives exactly BASE's outcome for a page that was
    painted at once (same bytes, so the same ``frame_after``), web mode and navigation reads on."""
    outs = []
    for module, script in ((base_module(), (CONTENT,)), (executor_mod, (WHITE, WHITE, CONTENT))):
        d = AllDriver(content_height=1220.0)
        ex = build(d, module=module, web=True)
        f = await frame_of(ex)
        d.queue(*script)
        out = await act(ex, d, call("click_at", f, x=500, y=500))
        outs.append(json.dumps([out.status, out.action, out.reason, out.resolved], default=str))
    assert outs[1] == outs[0], outs
