"""T-0111c criteria 1-4, 6, 10-13 against a fake driver: ``BrowserToolExecutor`` without a browser.

**The driver seam these rows pin.** ``BrowserToolExecutor(*, now_ms, viewport=(1280, 800), guard=None,
user_data_dir=None, driver=None)`` drives the browser only through ``driver`` (``None`` = the Playwright runtime of
``local_astra/computer/runtime.py``). The fake below implements exactly this interface; any other public name raises a
named ``AssertionError`` (a test double closes by delegation to a raising sentinel, .claude/rules/tests.md):

==============================================  =====================================================================
``viewport: tuple[int, int]``                   the page's CSS size (DSF 1)
``generation: int``                             0 before the first launch, +1 per (re)launch
``page_serial: int``                            +1 each time a different page becomes the active one after the
                                                first launch (a popup adopted, a closed page replaced, a
                                                relaunch); never decreases (ruling 683cc1e, criterion 15)
``async ensure() -> None``                      launch if not running; relaunch once if the browser died; adopt a
                                                popup that opened since the last settle (``page_serial`` +1). The
                                                fake makes every page change, the ``page_serial`` bump included,
                                                *before* any hang armed with ``arm_hang("ensure")``, as
                                                ``runtime.PlaywrightDriver.ensure`` does before its first await:
                                                a caller timed out inside ``ensure()`` still sees the new page
``async screenshot() -> bytes``                 a JPEG of the viewport
``async settle() -> None``                      the bounded settle (navigation, two animation frames)
``async mouse_move(x, y)``                      CSS pixels
``async mouse_down()`` / ``async mouse_up()``   the primary button
``async key_down(key)`` / ``async key_up(key)`` Playwright key names (``"Control"``, ``"Alt"``, ``"Enter"``, ``"a"``)
``async type_text(text)``                       keystrokes for ``text`` (``keyboard.type``)
``async scroll_by(dx, dy)``                     CSS pixels; positive is right/down
``async goto(url)``                             navigation (``ex.open``)
``take_events() -> list[dict]``                 blocked downloads/dialogs and closed popups since the last call
==============================================  =====================================================================

"Dispatched" means one of the input calls (``mouse_*``, ``key_*``, ``type_text``, ``scroll_by``, ``goto``). The
other seam calls (``ensure``, ``screenshot``, ``settle``, ``take_events``) dispatch nothing.

Ruling 683cc1e (on the re-review of fb39e3a): a frame's ``phash`` is salted per browser generation (deterministic for
identical pixels within one generation, different across a restart); a ``page_serial`` change across
``execute()``'s ``ensure()`` in the same generation refuses every non-search action with ``the active page
changed; take a new frame`` (a generation change wins: ``failed: browser restarted``); and ``ensure()`` runs under
``executor.ACTION_TIMEOUT_S``.

Expected pixels are by hand: ``to_pixels(500, 1280) = (2*500*1279 + 999) // 1998 = 640``, ``to_pixels(500, 800) =
400``, ``to_pixels(999, 1600) = 1599``, ``to_pixels(500, 1600) = 800``, ``to_pixels(500, 1000) = 500``.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import io
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.conductor.tool_executor import DryRunToolExecutor
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.protocols import ToolExecutor
from local_astra.contracts.toolcall import FrameGeometry, ToolInvocation, ToolOutcome

CLOCK_MS = 1_700_000_001_234
INPUT = frozenset(
    {"mouse_move", "mouse_down", "mouse_up", "key_down", "key_up", "type_text", "scroll_by", "goto"}
)
CONTROL = [chr(i) for i in range(0x20)] + ["\x7f"]


def computer_executor() -> Any:
    try:
        return importlib.import_module("local_astra.computer.executor")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("local_astra.computer"):
            raise AssertionError(f"T-0111c: {exc.name} does not exist yet") from exc
        raise


class DriverBoom(RuntimeError):
    """The fake's injected mid-action failure."""


class FakeBrowserGone(RuntimeError):
    """What the fake raises once ``kill()`` has been called (Playwright raises ``TargetClosedError``)."""


class FakeDriver:
    """The driver seam above, with no browser. Records every call; never opens a socket, a process or a file."""

    def __init__(self, viewport: tuple[int, int] = (1280, 800)) -> None:
        self.viewport = viewport
        self.generation = 0
        self.page_serial = 0
        #: the active page's name, and the active page at each screenshot
        self.active = "opener"
        self.shots: list[str] = []
        self._popup: str | None = None
        self.launches = 0
        self.alive = False
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.pointer: tuple[int, int] | None = None
        self.button_down = False
        self.presses: list[tuple[int, int] | None] = []
        self.held: set[str] = set()
        self.typed: list[str] = []
        self.scrolls: list[tuple[int, int]] = []
        self.scene = 0
        self.events: list[dict[str, Any]] = []
        #: ``(generation, scene)`` to draw on every screenshot instead of the live ones (``None``: the live ones)
        self.collide: tuple[int, int] | None = None
        self.raised = False
        self._raise_on: int | None = None
        self._counted = 0
        self._hang: tuple[str, int] | None = None
        self._seen: dict[str, int] = {}
        self._waiters: list[asyncio.Future[None]] = []
        self.hanging: asyncio.Event | None = None

    # -- the sentinel: every public name the seam does not have is a named failure, never an AttributeError
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise AssertionError(
            f"the executor used driver.{name}, which is not in the T-0111c driver seam "
            "(tests/unit/test_computer_executor.py docstring)"
        )

    # -- test controls
    def arm_raise(self, n: int) -> None:
        """Raise ``DriverBoom`` on the n-th seam call from now (``ensure`` excluded)."""
        self._raise_on, self._counted, self.raised = n, 0, False

    def arm_hang(self, name: str, k: int = 1) -> None:
        """Block forever in the k-th ``name`` call from now (after its effect), until cancelled or killed."""
        self._hang, self._seen = (name, k), {}
        self.hanging = asyncio.Event()

    def open_popup(self, name: str = "popup") -> None:
        """A popup opened after the last settle: the next ``ensure()`` of a live browser adopts it."""
        self._popup = name

    def kill(self) -> None:
        self.alive = False
        for w in self._waiters:
            if not w.done():
                w.set_exception(FakeBrowserGone("Target page, context or browser has been closed"))

    def dispatched(self, since: int = 0) -> list[tuple[str, tuple[Any, ...]]]:
        return [c for c in self.calls[since:] if c[0] in INPUT]

    def index(self, name: str, since: int = 0) -> list[int]:
        return [i for i, c in enumerate(self.calls) if i >= since and c[0] == name]

    # -- the seam
    def _enter(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))
        if name != "ensure":
            if not self.alive:
                raise FakeBrowserGone("Target page, context or browser has been closed")
            if self._raise_on is not None:
                self._counted += 1
                if self._counted == self._raise_on:
                    self.raised = True
                    raise DriverBoom(f"synthetic failure in {name}")

    async def _maybe_hang(self, name: str) -> None:
        if self._hang is None or self._hang[0] != name:
            return
        self._seen[name] = self._seen.get(name, 0) + 1
        if self._seen[name] != self._hang[1]:
            return
        self._hang = None
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        assert self.hanging is not None
        self.hanging.set()
        await fut

    async def ensure(self) -> None:
        self._enter("ensure")
        if not self.alive:
            self.alive = True
            self.launches += 1
            if (
                self.generation
            ):  # a relaunch makes a fresh page active (a popup of the dead browser died with it)
                self.page_serial += 1
            self.generation += 1
            self.scene = 0
            self.active, self._popup = "opener", None
        elif self._popup is not None:
            self.active, self._popup = self._popup, None
            self.page_serial += 1
            self.scene += 1
        await self._maybe_hang(
            "ensure"
        )  # after the effect: the adopted page's load wait, or a cap close

    async def screenshot(self) -> bytes:
        self._enter("screenshot")
        self.shots.append(self.active)
        if self.collide is not None:  # the same pixels whatever the generation or the scene
            return fake_jpeg(self.viewport, *self.collide)
        return fake_jpeg(self.viewport, self.generation, self.scene)

    async def settle(self) -> None:
        self._enter("settle")

    async def mouse_move(self, x: float, y: float) -> None:
        self._enter("mouse_move", x, y)
        self.pointer = (round(x), round(y))
        self.scene += 1
        await self._maybe_hang("mouse_move")

    async def mouse_down(self) -> None:
        self._enter("mouse_down")
        self.button_down = True
        self.presses.append(self.pointer)
        self.scene += 1
        await self._maybe_hang("mouse_down")

    async def mouse_up(self) -> None:
        self._enter("mouse_up")
        self.button_down = False
        self.scene += 1
        await self._maybe_hang("mouse_up")

    async def key_down(self, key: str) -> None:
        self._enter("key_down", key)
        self.held.add(key)
        self.scene += 1
        await self._maybe_hang("key_down")

    async def key_up(self, key: str) -> None:
        self._enter("key_up", key)
        self.held.discard(key)
        self.scene += 1
        await self._maybe_hang("key_up")

    async def type_text(self, text: str) -> None:
        self._enter("type_text", text)
        self.typed.append(text)
        self.scene += 1
        await self._maybe_hang("type_text")

    async def scroll_by(self, dx: float, dy: float) -> None:
        self._enter("scroll_by", dx, dy)
        self.scrolls.append((round(dx), round(dy)))
        self.scene += 1
        await self._maybe_hang("scroll_by")

    async def goto(self, url: str) -> None:
        self._enter("goto", url)
        self.scene += 1

    def take_events(self) -> list[dict[str, Any]]:
        self._enter("take_events")
        out, self.events = self.events, []
        return out


def fake_jpeg(viewport: tuple[int, int], generation: int, scene: int) -> bytes:
    """A JPEG whose coarse structure depends on (generation, scene): generations differ in gradient direction, so a
    pre-restart frame never shares a perceptual hash with a post-restart one."""
    from PIL import Image, ImageDraw

    w, h = viewport
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    steps = 16
    for i in range(steps):
        v = int(255 * i / (steps - 1))
        if generation % 2:
            d.rectangle([i * w // steps, 0, (i + 1) * w // steps, h], fill=(v, v, v))
        else:
            d.rectangle([0, i * h // steps, w, (i + 1) * h // steps], fill=(v, v, v))
    k = scene % 4
    d.rectangle(
        [k * w // 4, h // 3, (k + 1) * w // 4, 2 * h // 3],
        fill=(0, 0, 0) if generation % 2 else (255, 255, 255),
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def make(
    fake: FakeDriver | None = None,
    *,
    guard: Callable[[ToolInvocation], str | None] | None = None,
    **kw: Any,
) -> tuple[Any, FakeDriver]:
    fake = fake or FakeDriver()
    ex = computer_executor().BrowserToolExecutor(
        now_ms=lambda: CLOCK_MS, viewport=fake.viewport, driver=fake, guard=guard, **kw
    )
    return ex, fake


_N = iter(range(1_000_000))


def inv(name: str, frame: FrameGeometry | Frame | None = None, **arguments: Any) -> ToolInvocation:
    g = (
        FrameGeometry(
            phash=frame.phash, source=frame.source, width=frame.width, height=frame.height
        )
        if isinstance(frame, Frame)
        else frame
    )
    return ToolInvocation(
        call_id=f"u{next(_N)}",
        turn_id="t1",
        name=name,
        arguments=arguments,
        source="draft",
        frame=g,
    )


async def own_frame(ex: Any) -> Frame:
    f = await ex.frame()
    assert isinstance(f, Frame), type(f)
    return f


# --------------------------------------------------------------------------------------------- criterion 1, 2, 3


async def test_protocol_and_statuses() -> None:
    """Criterion 1: a live ``ToolExecutor``; ``executed`` for each dispatched action (never ``dry_run``), and the
    outcome's ``resolved`` carries the mapped pixels and the post-action frame's phash. Criterion 2's frame fields.
    Criterion 9 (mutation 11) at the seam: the post-action screenshot follows a settle that follows the last input."""
    from PIL import Image

    ex, fake = make()
    assert isinstance(ex, ToolExecutor)
    assert ex.mode == "live"
    f = await own_frame(ex)
    assert f.source == Source.WINDOW and f.ts_ms == CLOCK_MS
    assert (f.width, f.height) == (1280, 800)
    assert Image.open(io.BytesIO(f.jpeg)).size == (1280, 800)
    assert re.fullmatch(r"[0-9a-f]{16}", f.phash), f.phash
    assert (await own_frame(ex)).phash == f.phash, "identical pixels, identical phash"

    mark = len(fake.calls)
    out = await ex.execute(inv("click_at", f, x=500, y=500))
    assert isinstance(out, ToolOutcome) and out.status == "executed", out
    assert (out.resolved["x_px"], out.resolved["y_px"]) == (640, 400)
    assert fake.presses == [(640, 400)] and not fake.button_down
    latest = ex.latest_frame
    assert isinstance(latest, Frame) and out.resolved["frame_after"] == latest.phash
    last_input = max(i for i, c in enumerate(fake.calls) if c[0] in INPUT)
    shots = [i for i in fake.index("screenshot", mark) if i > last_input]
    settles = [i for i in fake.index("settle", mark) if last_input < i]
    assert shots and settles and min(settles) < max(shots), (
        f"the post-action frame must be taken after a settle that follows the last input: {fake.calls[mark:]}"
    )

    f = latest
    out = await ex.execute(inv("type_at", f, x=500, y=500, text="hello", press_enter=True))
    assert out.status == "executed", out
    assert fake.typed == ["hello"]
    i_down = fake.index("mouse_down")[-1]
    i_type = fake.index("type_text")[-1]
    i_enter = [i for i, c in enumerate(fake.calls) if c == ("key_down", ("Enter",))]
    assert i_down < i_type < i_enter[-1], "type_at clicks, types, then presses Enter"
    assert not fake.held and not fake.button_down

    for direction, sign in (
        ("down", (0, 1)),
        ("up", (0, -1)),
        ("right", (1, 0)),
        ("left", (-1, 0)),
    ):
        n = len(fake.scrolls)
        out = await ex.execute(inv("scroll", direction=direction))
        assert out.status == "executed", (direction, out)
        assert fake.scrolls[n:], f"scroll {direction} did not scroll: {fake.calls[-6:]}"
        dx, dy = (sum(v) for v in zip(*fake.scrolls[n:], strict=True))
        size = 1280 if sign[0] else 800
        moved = dx * sign[0] + dy * sign[1]
        assert size // 2 <= moved <= size and (dy if sign[0] else dx) == 0, (
            direction,
            fake.scrolls[n:],
        )

    out = await ex.execute(inv("press_key", combo="Ctrl+A"))
    assert out.status == "executed", out
    downs = [c[1][0] for c in fake.calls if c[0] == "key_down"]
    letters = [i for i, k in enumerate(downs) if k in ("a", "A")]
    assert "Control" in downs and letters and downs.index("Control") < max(letters), downs
    assert not fake.held, f"keys left down: {fake.held}"

    out = await ex.execute(inv("open_the_pod_bay_doors"))
    assert out.status == "refused", out


async def test_mapping_against_viewport_when_frame_is_downscaled() -> None:
    """Criterion 3 (mutation 1) at the seam: a 1600x1000 viewport's frame is 1568x980 (``screen_max`` scale 0.98);
    (999,999) still presses (1599,999) and (500,500) presses (800,500), not (1567,979) and (784,490)."""
    ex, fake = make(FakeDriver(viewport=(1600, 1000)))
    f = await own_frame(ex)
    assert (f.width, f.height) == (1568, 980)
    for (x, y), want in (((999, 999), (1599, 999)), ((500, 500), (800, 500)), ((0, 0), (0, 0))):
        out = await ex.execute(inv("click_at", f, x=x, y=y))
        assert out.status == "executed", out
        assert (out.resolved["x_px"], out.resolved["y_px"]) == want
        assert fake.presses[-1] == want, fake.presses


async def test_search_equals_dry_run() -> None:
    """Criterion 1: ``google_search`` is described exactly as ``DryRunToolExecutor`` does (equal outcomes, the
    malformed ones included) and dispatches nothing."""
    ex, fake = make()
    for args in ({"query": "omelette recipe"}, {}, {"query": 5}, {"query": "a\nb"}):
        i = inv("google_search", **args)
        assert await ex.execute(i) == await DryRunToolExecutor().execute(i), args
    assert fake.dispatched() == []


# --------------------------------------------------------------------------------------------- criterion 4, 6, 8


async def test_refusals_dispatch_nothing() -> None:
    """Criterion 4 (mutation 2), 6 and 8: every refusal, with its exact reason, dispatches nothing. The adversarial
    row is a HUD-shaped ``FrameGeometry(source=SCREEN)`` of the viewport's size with an invented phash."""
    ex, fake = make()
    own = await own_frame(ex)
    invented = "0123456789abcdef" if own.phash != "0123456789abcdef" else "fedcba9876543210"
    hud = FrameGeometry(phash=invented, source=Source.SCREEN, width=1280, height=800)
    cases: list[tuple[ToolInvocation, str]] = [
        (inv("click_at", None, x=1, y=1), "no frame"),
        (inv("type_at", None, x=1, y=1, text="a", press_enter=False), "no frame"),
        (inv("click_at", hud, x=1, y=1), "frame not from this browser"),
        (
            inv("click_at", hud.model_copy(update={"source": Source.WINDOW}), x=1, y=1),
            "frame not from this browser",
        ),
        (
            inv("click_at", hud.model_copy(update={"phash": None}), x=1, y=1),
            "frame not from this browser",
        ),
        (inv("type_at", hud, x=1, y=1, text="a", press_enter=False), "frame not from this browser"),
        (inv("click_at", own, x=1000, y=1), "bad coordinates"),
        (inv("click_at", own, x=1, y=-1), "bad coordinates"),
        (inv("click_at", own, x=False, y=1), "bad coordinates"),
        (inv("click_at", own, x=1.5, y=1), "bad coordinates"),
        (inv("click_at", own, y=1), "bad coordinates"),
        (inv("type_at", own, x=1, y=1, text=None, press_enter=False), "bad arguments"),
        (inv("type_at", own, x=1, y=1, text="a"), "bad arguments"),
        (inv("type_at", own, x=1, y=1, text="a", press_enter=1), "bad arguments"),
        (
            inv("type_at", own, x=1, y=1, text="a\nb", press_enter=False),
            "bad arguments: control character",
        ),
        (inv("press_key", combo="Alt+Tab"), "not available in the browser"),
        (inv("press_key", combo="Ctrl+T"), "not available in the browser"),
        (inv("press_key", combo="Ctrl+W"), "not available in the browser"),
        (inv("press_key", combo="Ctrl+L"), "not available in the browser"),
    ]
    for i, reason in cases:
        mark = len(fake.calls)
        out = await ex.execute(i)
        assert (out.status, out.reason) == ("refused", reason), (i.name, i.arguments, i.frame, out)
        assert fake.dispatched(mark) == [], (i.arguments, fake.dispatched(mark))
    assert fake.presses == [] and fake.typed == []


_SAFE = st.characters(blacklist_categories=("Cs", "Cc"))


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    head=st.text(alphabet=_SAFE, max_size=8),
    ctl=st.sampled_from(CONTROL),
    tail=st.text(alphabet=_SAFE, max_size=8),
    enter=st.booleans(),
)
def test_control_characters_refused(head: str, ctl: str, tail: str, enter: bool) -> None:
    """Criterion 6 (mutation 5), both directions of the guard: any text with a C0 control or DEL is refused as
    ``bad arguments: control character`` with nothing clicked or typed; the same text without it is typed exactly."""

    async def body() -> None:
        ex, fake = make()
        f = await own_frame(ex)
        mark = len(fake.calls)
        out = await ex.execute(
            inv("type_at", f, x=10, y=10, text=head + ctl + tail, press_enter=enter)
        )
        assert (out.status, out.reason) == ("refused", "bad arguments: control character"), out
        assert fake.dispatched(mark) == []
        clean = head + tail
        if clean:
            out = await ex.execute(inv("type_at", f, x=10, y=10, text=clean, press_enter=enter))
            assert out.status == "executed", (clean, out)
            assert "".join(fake.typed) == clean, (clean, fake.typed)

    asyncio.run(body())


# --------------------------------------------------------------------------------------------- criterion 10


async def _action(ex: Any, f: Frame, kind: str) -> ToolOutcome:
    if kind == "click":
        return await ex.execute(inv("click_at", f, x=500, y=500))
    if kind == "type":
        return await ex.execute(inv("type_at", f, x=500, y=500, text="abc", press_enter=True))
    if kind == "scroll":
        return await ex.execute(inv("scroll", direction="down"))
    return await ex.execute(inv("press_key", combo="Ctrl+A"))


@pytest.mark.parametrize("kind", ["click", "type", "scroll", "press_key"])
async def test_exception_never_executed(kind: str) -> None:
    """Criterion 10 (mutation 3): a ``DriverBoom`` raised on the n-th seam call of an action, for every n the
    action reaches, is ``failed`` naming ``DriverBoom``, never ``executed``."""
    raised = 0
    for n in range(1, 40):
        ex, fake = make()
        f = await own_frame(ex)
        fake.arm_raise(n)
        out = await _action(ex, f, kind)
        if not fake.raised:
            assert out.status == "executed", (n, out)
            break
        raised += 1
        assert out.status == "failed", (kind, n, fake.calls[-1], out)
        assert out.reason and "DriverBoom" in out.reason, out.reason
    else:
        raise AssertionError("the action never completed")
    assert raised >= 2, (
        f"{kind}: only {raised} seam call(s) before completion; the fake saw {fake.calls}"
    )


async def test_kill_mid_action_then_restart_once_no_retry() -> None:
    """Criterion 10 (mutation 4): a kill during a click is ``failed`` naming the exception and does not relaunch;
    the next call relaunches exactly once, dispatches nothing and is ``failed: browser restarted``; a pre-kill
    frame is ``refused: frame from before a browser restart``; the call after that runs normally."""
    ex, fake = make()
    f0 = await own_frame(ex)
    fake.arm_hang("mouse_down")
    task = asyncio.ensure_future(ex.execute(inv("click_at", f0, x=500, y=500)))
    assert fake.hanging is not None
    await asyncio.wait_for(fake.hanging.wait(), 5)
    fake.kill()
    out = await asyncio.wait_for(task, 5)
    assert out.status == "failed", out
    assert out.reason and "FakeBrowserGone" in out.reason, out.reason
    assert fake.launches == 1, "relaunched inside the failed call"

    mark = len(fake.calls)
    out = await ex.execute(inv("press_key", combo="Tab"))
    assert (out.status, out.reason) == ("failed", "browser restarted"), out
    assert fake.launches == 2 and fake.generation == 2
    assert fake.dispatched(mark) == [], f"the action ran after the restart: {fake.dispatched(mark)}"

    mark = len(fake.calls)
    out = await ex.execute(inv("click_at", f0, x=500, y=500))
    assert (out.status, out.reason) == ("refused", "frame from before a browser restart"), out
    assert fake.dispatched(mark) == []

    out = await ex.execute(inv("press_key", combo="Tab"))
    assert out.status == "executed", out
    assert fake.launches == 2
    f1 = await own_frame(ex)
    assert f1.phash != f0.phash
    out = await ex.execute(inv("click_at", f1, x=500, y=500))
    assert out.status == "executed", out
    assert len(fake.presses) == 2, f"a click was replayed: {fake.presses}"


def _pixels(f: Frame) -> bytes:
    from PIL import Image

    return Image.open(io.BytesIO(f.jpeg)).convert("RGB").tobytes()


async def _restart(ex: Any, fake: FakeDriver) -> None:
    fake.kill()
    out = await ex.execute(inv("press_key", combo="Tab"))
    assert (out.status, out.reason) == ("failed", "browser restarted"), out


async def test_retired_phash_refused_after_restart_even_when_seen_again() -> None:
    """Criterion 2/4 as ruled at 683cc1e (per-generation salt; the reviewer's livelock). The fake draws the same
    pixels in every generation, as a uniform page re-opened after a restart does. A pre-restart frame is still
    refused (``frame from before a browser restart``, nothing dispatched); a fresh post-restart frame of identical
    pixels carries a different phash and is accepted and dispatched (before the ruling it was refused for good,
    which left the page unclickable until its pixels changed). The same holds across a second restart, for both
    earlier generations' frames."""
    ex, fake = make()
    fake.collide = (1, 0)
    f0 = await own_frame(ex)
    await _restart(ex, fake)
    f1 = await own_frame(ex)
    assert _pixels(f1) == _pixels(f0), (
        "the fake did not draw identical pixels; the row tests nothing"
    )
    assert f1.phash != f0.phash, (
        "identical pixels after a restart carry the pre-restart phash: the phash is not salted per generation "
        "(ruling 683cc1e)"
    )
    mark = len(fake.calls)
    out = await ex.execute(inv("click_at", f0, x=500, y=500))
    assert (out.status, out.reason) == ("refused", "frame from before a browser restart"), out
    assert fake.dispatched(mark) == []
    mark = len(fake.calls)
    out = await ex.execute(inv("click_at", f1, x=500, y=500))
    assert out.status == "executed", (
        f"a fresh post-restart frame of a re-opened page is not clickable (the livelock): {out}"
    )
    assert fake.presses == [(640, 400)], fake.presses
    assert [c[0] for c in fake.dispatched(mark)].count("mouse_down") == 1

    await _restart(ex, fake)
    f2 = await own_frame(ex)
    assert _pixels(f2) == _pixels(f0)
    assert len({f0.phash, f1.phash, f2.phash}) == 3, (f0.phash, f1.phash, f2.phash)
    for f in (f0, f1):
        mark = len(fake.calls)
        out = await ex.execute(inv("click_at", f, x=500, y=500))
        assert (out.status, out.reason) == ("refused", "frame from before a browser restart"), out
        assert fake.dispatched(mark) == []
    out = await ex.execute(inv("click_at", f2, x=500, y=500))
    assert out.status == "executed", out
    assert fake.presses == [(640, 400), (640, 400)], fake.presses


async def test_phash_equal_within_a_generation_and_salted_across() -> None:
    """Criterion 2 as ruled at 683cc1e: two frames of identical pixels within one generation have equal phashes
    (a salt drawn per capture would break this and make no frame findable again), each is 16 lowercase hex, and
    different pixels in the same generation still differ; across a restart the same pixels get a different
    phash (the salt is per generation, not a constant)."""
    ex, fake = make()
    fake.collide = (1, 0)
    a, b = await own_frame(ex), await own_frame(ex)
    for f in (a, b):
        assert re.fullmatch(r"[0-9a-f]{16}", f.phash), f.phash
    assert _pixels(a) == _pixels(b) and a.phash == b.phash, (a.phash, b.phash)
    fake.collide = (2, 3)
    other = await own_frame(ex)
    assert other.phash != a.phash, "different pixels in one generation share a phash"

    fake.collide = (1, 0)
    await _restart(ex, fake)
    c, d = await own_frame(ex), await own_frame(ex)
    for f in (c, d):
        assert re.fullmatch(r"[0-9a-f]{16}", f.phash), f.phash
    assert c.phash == d.phash, (
        "identical pixels within generation 2 differ: the salt is not per generation"
    )
    assert _pixels(c) == _pixels(a)
    assert c.phash != a.phash, (
        "identical pixels across a restart share a phash: no per-generation salt"
    )


# --------------------------------------------------------------------------------------------- criterion 15

ADOPTED = "the active page changed; take a new frame"


@pytest.mark.parametrize("kind", ["click", "type", "scroll", "press_key"])
async def test_adoption_in_ensure_refuses_the_pending_action(kind: str) -> None:
    """Criterion 15 as ruled at 683cc1e: a popup that opened after the last settle is adopted by ``execute()``'s
    ``ensure()``; the pending action, whatever it is (frame-carrying or not), is refused with ``the active page
    changed; take a new frame`` and dispatches nothing. The next own frame shows the popup, and the same action
    against that frame runs (the guard's too-wide side)."""
    ex, fake = make()
    f = await own_frame(ex)
    assert fake.shots == ["opener"]
    fake.open_popup()
    mark = len(fake.calls)
    out = await _action(ex, f, kind)
    assert fake.active == "popup" and fake.page_serial == 1, (
        "the premise failed: execute() did not call ensure(), so the fake never adopted the popup"
    )
    assert (out.status, out.reason) == ("refused", ADOPTED), out
    assert fake.dispatched(mark) == [], fake.dispatched(mark)
    nxt = await own_frame(ex)
    assert fake.shots[-1] == "popup", fake.shots
    out = await _action(ex, nxt, kind)
    assert out.status == "executed", out
    assert fake.dispatched(mark), "the action after a new frame dispatched nothing"


async def test_adoption_does_not_touch_search() -> None:
    """Criterion 1 and 15 (ruling 683cc1e): with a popup waiting to be adopted, ``google_search`` is still exactly
    the dry-run description, and dispatches nothing."""
    ex, fake = make()
    await own_frame(ex)
    fake.open_popup()
    for args in ({"query": "weather"}, {}):
        i = inv("google_search", **args)
        assert await ex.execute(i) == await DryRunToolExecutor().execute(i), args
    assert fake.dispatched() == []


async def test_restart_wins_over_adoption() -> None:
    """Criterion 10 with criterion 15's ruling: a relaunch also makes a different page active (the fake bumps
    ``page_serial``), but a call that relaunches is ``failed: browser restarted``, not the adoption refusal."""
    ex, fake = make()
    await own_frame(ex)
    fake.open_popup()
    fake.kill()
    mark = len(fake.calls)
    out = await ex.execute(inv("press_key", combo="Tab"))
    assert (out.status, out.reason) == ("failed", "browser restarted"), out
    assert fake.dispatched(mark) == []


# --------------------------------------------------------------------------------------------- criterion 10 (ensure)

SMALL_TIMEOUT_S = 0.3


@pytest.mark.parametrize(
    "popup", [True, False], ids=["adopting-load-or-close", "closed-page-replaced"]
)
async def test_ensure_runs_under_the_action_timeout(
    popup: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 10 as ruled at 683cc1e: ``ensure()`` runs under ``ACTION_TIMEOUT_S`` (patched to 0.3 s here rather
    than waiting 8 s). A load wait or a page close that hangs inside ``execute()``'s ``ensure()`` settles within
    the timeout (wall bound 0.3 s + 1.5 s slack, far under the real 8 s) as the executor's own timeout outcome,
    the oracle being the outcome of the same call whose dispatch hangs instead; nothing is dispatched."""
    mod = computer_executor()
    assert mod.ACTION_TIMEOUT_S == 8.0, (
        "the ruled worst case is 8 s, under TOOL_EXEC_TIMEOUT_MS = 10_000"
    )
    monkeypatch.setattr(mod, "ACTION_TIMEOUT_S", SMALL_TIMEOUT_S)

    ex, fake = make()
    f = await own_frame(ex)
    fake.arm_hang("mouse_down")
    oracle = await asyncio.wait_for(ex.execute(inv("click_at", f, x=500, y=500)), 5)
    assert oracle.status != "executed", oracle

    ex, fake = make()
    f = await own_frame(ex)
    if popup:
        fake.open_popup()
    fake.arm_hang("ensure")
    mark = len(fake.calls)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    try:
        out = await asyncio.wait_for(
            ex.execute(inv("click_at", f, x=500, y=500)), SMALL_TIMEOUT_S + 1.5
        )
    except TimeoutError:
        raise AssertionError(
            f"a hang inside ensure() was not bounded by ACTION_TIMEOUT_S ({SMALL_TIMEOUT_S} s patched)"
        ) from None
    elapsed = loop.time() - t0
    assert fake.hanging is not None and fake.hanging.is_set(), (
        "the premise failed: ensure() never hung"
    )
    assert elapsed >= SMALL_TIMEOUT_S * 0.9, (
        f"settled after {elapsed:.3f} s, before the timeout could fire"
    )
    assert (out.status, out.reason) == (oracle.status, oracle.reason), (out, oracle)
    assert fake.dispatched(mark) == [], fake.dispatched(mark)


# --------------------------------------------------------------------------------------------- criterion 4 (frame page)


async def test_adoption_then_ensure_timeout_refuses_the_openers_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Criterion 4 as widened on the round-3 review of e345ef2 (kills S3, the frame-page check removed). A popup is
    queued and a hang is armed inside ``ensure()`` (after the adoption, so ``page_serial`` has already moved), with
    ``ACTION_TIMEOUT_S`` patched to 0.3 s. The first call is the executor's own timeout outcome, the same
    ``(status, reason)`` as a dispatch that hangs, and dispatches nothing. The next calls carry the *opener's* frame:
    their ``ensure()`` changes nothing (the popup is already active), so only the frame's own page tells them
    apart, and each is ``refused`` with ``the active page changed; take a new frame``, dispatching nothing. The
    guard's too-wide side: a new own frame (of the popup) is accepted and its click lands once."""
    mod = computer_executor()
    monkeypatch.setattr(mod, "ACTION_TIMEOUT_S", SMALL_TIMEOUT_S)

    ex, fake = make()
    f = await own_frame(ex)
    fake.arm_hang("mouse_down")
    oracle = await asyncio.wait_for(ex.execute(inv("click_at", f, x=500, y=500)), 5)
    assert oracle.status == "failed", oracle

    ex, fake = make()
    opener = await own_frame(ex)
    assert fake.shots == ["opener"]
    fake.open_popup()
    fake.arm_hang("ensure")
    mark = len(fake.calls)
    try:
        out = await asyncio.wait_for(
            ex.execute(inv("click_at", opener, x=500, y=500)), SMALL_TIMEOUT_S + 1.5
        )
    except TimeoutError:
        raise AssertionError("a hang inside ensure() was not bounded by ACTION_TIMEOUT_S") from None
    assert fake.hanging is not None and fake.hanging.is_set(), (
        "the premise failed: ensure() never hung"
    )
    assert fake.active == "popup" and fake.page_serial == 1, (
        "the premise failed: the popup was not adopted before the hang"
    )
    assert (out.status, out.reason) == (oracle.status, oracle.reason), (out, oracle)
    assert fake.dispatched(mark) == [], fake.dispatched(mark)

    for i in (
        inv("click_at", opener, x=500, y=500),
        inv("type_at", opener, x=500, y=500, text="abc", press_enter=True),
    ):
        mark, serial = len(fake.calls), fake.page_serial
        out = await ex.execute(i)
        assert fake.page_serial == serial and fake.active == "popup", (
            "the premise failed: this call's ensure() changed the page, so the serial check alone refuses it"
        )
        assert (out.status, out.reason) == ("refused", ADOPTED), (i.name, out)
        assert fake.dispatched(mark) == [], (i.name, fake.dispatched(mark))
    assert fake.presses == [] and fake.typed == []

    nxt = await own_frame(ex)
    assert fake.shots[-1] == "popup", fake.shots
    out = await ex.execute(inv("click_at", nxt, x=500, y=500))
    assert out.status == "executed", out
    assert fake.presses == [(640, 400)], fake.presses


async def test_openers_frame_after_a_newer_own_frame_is_refused() -> None:
    """Criterion 4 as widened on the round-3 review of e345ef2 (S3, the second case the ruling names): the popup
    is adopted by ``frame()``'s ``ensure()``, not by ``execute()``'s, so the call's own ``ensure()`` sees no page
    change. A click or a type carrying the opener's older own frame is refused (``the active page changed; take a
    new frame``) and dispatches nothing; the popup's frame is accepted."""
    ex, fake = make()
    opener = await own_frame(ex)
    fake.open_popup()
    popup = await own_frame(ex)
    assert fake.shots == ["opener", "popup"] and fake.page_serial == 1, fake.shots
    for i in (
        inv("click_at", opener, x=500, y=500),
        inv("type_at", opener, x=500, y=500, text="abc", press_enter=False),
    ):
        mark = len(fake.calls)
        out = await ex.execute(i)
        assert (out.status, out.reason) == ("refused", ADOPTED), (i.name, out)
        assert fake.dispatched(mark) == [], (i.name, fake.dispatched(mark))
    out = await ex.execute(inv("click_at", popup, x=500, y=500))
    assert out.status == "executed", out
    assert fake.presses == [(640, 400)], fake.presses


# --------------------------------------------------------------------------------------------- criterion 11


@pytest.mark.parametrize(
    ("hang", "make_inv"),
    [
        (("mouse_down", 1), lambda f: inv("click_at", f, x=500, y=500)),
        (
            ("type_text", 1),
            lambda f: inv("type_at", f, x=500, y=500, text="x" * 2000, press_enter=True),
        ),
        (("key_down", 2), lambda f: inv("press_key", combo="Ctrl+A")),
        (("key_down", 2), lambda f: inv("press_key", combo="Ctrl+Enter")),
    ],
    ids=["click-between-down-and-up", "type", "ctrl+a-modifier-held", "ctrl+enter-modifier-held"],
)
async def test_cancel_between_down_and_up_releases(
    hang: tuple[str, int], make_inv: Callable[[Frame], ToolInvocation]
) -> None:
    """Criterion 11 (mutation 6): cancelled while a button or a modifier is down, ``CancelledError`` propagates and
    nothing stays held; nothing more is pressed or typed after the cancel."""
    ex, fake = make()
    f = await own_frame(ex)
    fake.arm_hang(*hang)
    task = asyncio.ensure_future(ex.execute(make_inv(f)))
    assert fake.hanging is not None
    await asyncio.wait_for(fake.hanging.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert not fake.button_down, "the primary button is still down"
    assert not fake.held, f"keys still down: {fake.held}"
    after = len(fake.calls)
    await asyncio.sleep(0.05)
    late = [c for c in fake.dispatched(after) if c[0] in ("mouse_down", "key_down", "type_text")]
    assert late == [], late
    assert fake.typed in ([], ["x" * 2000]), "typing resumed after the cancel"


# --------------------------------------------------------------------------------------------- criterion 12


async def test_guard_called_once_and_refuses() -> None:
    """Criterion 12 (mutation 10): the guard sees each non-search invocation exactly once, unchanged; a string
    refuses with that reason and dispatches nothing; ``None`` lets the action run."""
    seen: list[ToolInvocation] = []
    verdict: list[str | None] = [None]

    def guard(i: ToolInvocation) -> str | None:
        seen.append(i)
        return verdict[0]

    ex, fake = make(guard=guard)
    f = await own_frame(ex)
    calls = [
        inv("click_at", f, x=500, y=500),
        inv("type_at", f, x=500, y=500, text="hi", press_enter=False),
        inv("scroll", direction="down"),
        inv("press_key", combo="Tab"),
    ]
    for i in calls:
        frozen = copy.deepcopy(i)
        seen.clear()
        out = await ex.execute(i)
        assert out.status == "executed", out
        assert seen == [frozen], f"guard calls for {i.name}: {seen}"
        assert i == frozen
    verdict[0] = "the user said no"
    for i in calls:
        seen.clear()
        mark = len(fake.calls)
        out = await ex.execute(i)
        assert (out.status, out.reason) == ("refused", "the user said no"), out
        assert len(seen) == 1
        assert fake.dispatched(mark) == []


async def test_guard_not_consulted_for_search() -> None:
    """Criterion 12: ``google_search`` never reaches the guard and stays the dry-run description."""
    seen: list[ToolInvocation] = []

    def guard(i: ToolInvocation) -> str | None:
        seen.append(i)
        return "refused by the guard"

    ex, _fake = make(guard=guard)
    i = inv("google_search", query="weather")
    assert await ex.execute(i) == await DryRunToolExecutor().execute(i)
    assert seen == []


# --------------------------------------------------------------------------------------------- criterion 13


def test_profile_paths_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 13: a ``user_data_dir`` resolving inside ``~/.config/google-chrome``, ``~/.config/chromium`` or
    ``~/.mozilla`` is refused at construction (``ValueError``), through ``~``, ``..`` and a symlink alike; a
    temporary directory, or another directory under ``~/.config``, is not (the guard's too-wide side). Nothing is
    launched by construction."""
    home = tmp_path / "home"
    for d in (".config/google-chrome/Default", ".config/chromium", ".mozilla/firefox/abc.default"):
        (home / d).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    (tmp_path / "link").symlink_to(home / ".config" / "chromium")
    refused = [
        home / ".config" / "google-chrome",
        home / ".config" / "google-chrome" / "Default",
        home / ".config" / "chromium",
        home / ".mozilla",
        home / ".mozilla" / "firefox" / "abc.default",
        "~/.config/chromium",
        str(home / ".config" / "astra" / ".." / "google-chrome"),
        tmp_path / "link",
        tmp_path / "link" / "Profile 1",
    ]
    for p in refused:
        fake = FakeDriver()
        with pytest.raises(ValueError):
            make(fake, user_data_dir=p)
        assert fake.calls == [], f"construction touched the driver: {fake.calls}"
    for p in (tmp_path / "astra-profile", home / ".config" / "astra-browser", None):
        fake = FakeDriver()
        make(fake, user_data_dir=p)
        assert fake.calls == []
    assert os.environ["HOME"] == str(home)
