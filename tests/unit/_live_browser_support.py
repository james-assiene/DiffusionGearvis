"""T-0111g group B: the fake live browser, the live kit and the rig over the session fakes (docs/tasks/T-0111g.md).

**The interfaces the group B rows pin** (the task fixes behaviour; these are the names a row needs, listed in the
test-author's report for the orchestrator to accept):

``local_astra/conductor/session.py``

- ``LiveBrowser``: a ``typing.Protocol`` defined in ``session.py`` (never in ``contracts/protocols.py``, which is
  frozen), the session-side live browser seam. Members, exactly the executor's public surface plus A2's state::

      closed: bool                                   # property; True once the browser ended (A2)
      latest_frame: Frame | None                     # property; the post-action capture
      egress_log: tuple[dict[str, str], ...]         # property; blocked attempts, {host, scheme}
      async def frame(self) -> Frame                 # an own frame of the active page
      async def open(self, url: str) -> None
      async def close(self) -> None
      async def execute(self, inv: ToolInvocation) -> ToolOutcome

- ``ToolKit.mode: Literal["dry_run", "live"]`` and a new keyword field ``ToolKit.live_browser: LiveBrowser | None =
  None``. In ``live``: ``kit.live_browser is kit.executor`` (B5).
- ``LIVE_FRAME_TIMEOUT_MS: int``, a module constant, ``0 < LIVE_FRAME_TIMEOUT_MS <= TOOL_EXEC_TIMEOUT_MS // 2``
  (B6: "well under the 10 s deadline"; the rows read the constant, the value is the implementer's).
- Turn-line keys (``TurnRecord.extras``, like T-0114's), ``live`` only: ``frame_source`` (``"window"`` |
  ``"none"``), ``browser`` (``"open"`` | ``"closed"``), ``browser_egress_blocked`` (list of ``{"host", "scheme"}``
  dicts, unique, exactly those two keys) and ``assists_dropped: True`` when any assist flag was on (absent or
  ``False`` otherwise).

``local_astra/toolcall/loop.py`` (pure, contracts-only imports)

- ``swap_newest_image(prompt: Prompt, frame: Frame) -> Prompt``: the newest ``ImagePart`` of the prompt's **final
  user message** (the current turn) is replaced by the part the prompt builder makes for ``frame``
  (``ImagePart(url="data:image/jpeg;base64,<b64(frame.jpeg)>", phash=frame.phash)``); every other part and field is
  unchanged. A prompt whose final user message holds no image comes back equal (``==``), history images untouched.
- ``screenshot_payload(payload: Mapping[str, Any], updated: bool) -> dict[str, Any]``: ``payload`` plus
  ``"screenshot": "updated" | "unchanged"``; the input is not mutated.

**The fake** (``FakeLiveBrowser``) closes by delegation, the T-0111c way: every name it does not deliberately fake
raises a named ``AssertionError`` (there is no real module to delegate to without launching a browser). It models
the executor's rules the rows lean on: a coordinate call needs an own frame of the current generation (else
``refused: frame not from this browser``), an executed action captures a post-action frame (``latest_frame``, its
phash in ``resolved["frame_after"]``), ``closed`` refuses every non-search action with A2's reason.

**T-0131 group B additions** (at the end of the module; ``live_rig`` and ``FakeLiveBrowser`` are unchanged):

- ``jpeg_bytes(width=W, height=H, *, seed=0)`` / ``jpeg_frame(tag, *, seed=None)``: own frames with **real** JPEG
  bytes (solid 64-px blocks whose colours depend on the block and the seed), because the session cuts the crop from
  them with PIL;
- ``make_reground_llm(browser, script)``: a ``ToolLlm`` subclass (defined inside the factory) that tells a pass-2
  request apart structurally (its prompt holds an image whose data URL is no frame the browser returned) and serves
  it from ``script_pass2`` (a list of commits, an exception, or ``HANG`` until aborted); it records ``pass2_prompts``,
  ``pass2_ids``,
  ``pass2_overlaps`` (generates live at each pass-2 start), ``events`` and calls ``on_pass2`` at each start; the
  abort log is ``FakeLlm.aborted``;
- ``reground_rig(tmp_path, browser, *, script, reground=True, session_kwargs=None)``: the live session with that LLM
  and ``Session(..., reground=...)`` (``None``: the keyword is not passed); a session without the keyword is a
  named ``AssertionError``.

**T-0134 group A additions** (at the end of the module and in ``FakeLiveBrowser``, additive): ``FakeLiveBrowser(...,
echo="pixels" | "model" | None)`` (``"model"``: ``clicked at (x, y)`` in the call's 0-999 units, ``resolved`` keeps
pixels; ``None``: no ``echo`` attribute), ``browser.extents`` (phash -> ``{"above_px", "below_px", "viewport_px"}``
or ``None``) read by ``extent_of(key)`` (phash, capture identity or frame), ``live_feedback(...)`` and
``feedback_rig(...)``.

**T-0140 group A1 additions** (in ``FakeLiveBrowser``, additive; docs/tasks/T-0140.md "Scope within shared files"):
a scripted full-page image and scripted page metrics for the ``page_overview`` tool, which the group A2 session rows
drive. ``browser.overview_image`` (the *scaled* overview frame the executor would keep) and ``browser.page_metrics``
(``{"scroll_y", "viewport_height", "content_height"}``, CSS px) decide what ``execute(page_overview)`` returns, by
A3's rules: ``executed`` with the fits string and no image when ``content_height <= 1.05 * viewport_height``; else
``executed`` with ``showed the whole page (about {s:.1f} screens tall{C}) as a picture before the screen`` and
``resolved = {"overview": <id>, "screens": s, "clipped": bool}``; ``failed`` (``the page moved during the
overview``) when ``overview_moves`` is set; ``refused`` (``page overview is not available``) when
``overview_available`` is ``False``. ``overview_of(id)`` returns the frame once (then ``None``). The overview frame is
never registered as an own frame (A3 "Not an own frame"). With no ``overview_image`` set and no fit, the call fails
loudly (a row forgot its script). ``screens``/``clipped`` in ``resolved`` are the test-author's reading of "Tests to
write" (``gives `clipped`, `screens` 9.3``); see the T-0140 A1 report.

**T-0150 group A additions** (in ``FakeLiveBrowser``, additive; docs/tasks/T-0150.md "Scope within shared files"), all
off by default so every earlier row sees what it saw: ``overview_gate`` (an ``asyncio.Event``: a ``page_overview``
execute waits on it, a slow capture) and ``overview_hang`` (a capture that never returns until cancelled); either one
records ``("overview.start", id)`` and then ``("overview.end", id)`` or ``("overview.cancelled", id)`` in ``calls``, so
a row reads the interleaving with ``frame()``; ``overview_cancel_ticks`` event-loop turns pass between a cancel and
the capture's end (a real capture restores the viewport first). ``url`` (set by ``open`` and the ``navigate(url)`` control) and
``url_metrics`` (url -> metrics) script the page metrics per URL (``scripted_metrics()``; ``page_metrics`` otherwise).
``overview_captures()`` counts the ``page_overview`` invocations that reached the browser. These are test controls,
not seam members: the session never calls them.

**T-0151 group A2 additions** (in ``FakeLiveBrowser``, additive; docs/tasks/T-0151.md "Scope within shared files":
``FakeLiveBrowser.overview_frame_of``, scripted registration results, and a record of the ``ToolInvocation`` frame
geometry it received), all off by default so every earlier row sees what it saw. They mirror group A1's executor
contract as merged (e4ef5cf; docs/tasks/T-0151.md A1-1..A1-5):

- ``perform_overview`` also keeps the capture's facts per id (``overview_capture``: the latest own frame's phash at
  the capture, the generation, ``clipped``, the captured CSS size, the viewport and content heights);
- ``overview_frame_of(key)`` reads the same once-only store as ``overview_of`` (whichever reads first takes it) and
  registers the picture as an *overview frame* (``overview_frames``, keyed by its phash, apart from ``_own`` and
  ``latest_frame``) only when it was not clipped, its CSS width is the viewport's (``overview_wide`` scripts a wider
  capture), the generation is the capture's and ``latest_frame`` is still the own frame that was latest at the capture
  (there was one); else ``None``. ``overview_frame_none`` scripts a ``None`` registration whatever the facts. Every
  read is in ``overview_frame_reads`` and ``calls`` (``("overview_frame_of", key)``);
- a ``click_at``/``type_at`` whose frame geometry (phash, ``WINDOW``, width and height) is a registered overview frame
  is refused ``frame from before a browser restart`` after a restart and ``OVERVIEW_STALE`` once ``latest_frame`` is
  not the frame recorded at the capture; otherwise it is planned against the page (``P = (to_px(x, page_w),
  to_px(y, page_h))``), scrolled once into view (``dy = clamp(P_y - H/2, 0, C - H) - S``, the scripted metrics'
  ``scroll_y`` moves) and "clicked" at ``(P_x, P_y - S')``. Its action string is BASE's in the call's units (echo
  ``model``) or the viewport pixels (``pixels``), prefixed ``scrolled down|up by N px and `` after a scroll;
  ``resolved`` holds ``x_px``, ``y_px``, ``page_y_px``, ``scrolled_px`` and ``frame_after``;
- ``land(frame)``: a test control, an own frame captured by someone else (a takeover snapshot), which becomes
  ``latest_frame``. ``invocations`` already records every ``ToolInvocation`` and so its ``frame`` geometry.

**T-0136 group B additions** (additive, all off by default; docs/tasks/T-0136.md "Owned files": scriptable
decodable JPEG frames and ``resolved`` points):

- ``FakeLiveBrowser.resolved_extra``: call id -> keys merged into that executed ``click_at``/``type_at``'s
  ``resolved`` (normal and overview-frame paths), e.g. ``{"x_px": 901, "y_px": 444}`` (a point that is not the
  arguments' mapping) or ``{"scrolled_px": 120}``. Empty: the fake's ``resolved`` is unchanged;
- ``scene_jpeg(blocks=..., background=..., quality=...)`` / ``scene_frame(tag, ...)``: real JPEG frames, a solid
  background with solid rectangles ``(x0, y0, x1, y1, (r, g, b))`` in viewport px, so a row can script a large,
  unambiguous change at a known place (independent of the detector's calibrated constants);
- ``TurnLogSpy`` / ``turn_logs(monkeypatch)``: every ``session._TurnLog`` the session creates, in order (a recording
  subclass installed on the module), so a row can read per-turn fields after the turn ended.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, n = 1, deterministic.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import importlib
from collections import deque
from collections.abc import Callable, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.toolcall import ToolInvocation, ToolOutcome

CLOSED = "the browser window was closed"
FRESHNESS = "the screen changed since this frame; take a new frame"
NOT_OWN = "frame not from this browser"
ADOPTION = "the active page changed; take a new frame"
RESTARTED = "browser restarted"
W, H = 1280, 800
LIVE_KEYS = ("frame_source", "browser", "browser_egress_blocked")
#: T-0140 A3's fixed strings and constants (docs/tasks/T-0140.md), copied here by hand from the task, never imported
OVERVIEW_FITS = "the whole page fits on the browser screen; no overview was taken"
OVERVIEW_MOVED = "the page moved during the overview"
OVERVIEW_UNAVAILABLE = "page overview is not available"
OVERVIEW_FITS_RATIO = 1.05
MAX_OVERVIEW_SCREENS = 4
#: T-0151 A1-3's reasons, copied by hand from the task (never imported from the executor)
OVERVIEW_STALE = "the screen changed since the whole-page picture; take a new frame"
RESTART_FRAME = "frame from before a browser restart"


def to_px(v: int, n: int) -> int:
    """``to_pixels`` by hand (0-999 model units onto ``n`` pixels, rounded half up): ``(2 v (n - 1) + 999) // 1998``
    (T-0151 A1 test module's oracle)."""
    return (2 * v * (n - 1) + 999) // 1998


def overview_action(screens: float, clipped: bool) -> str:
    """A3's executed action string, by the task's template."""
    c = f", top {MAX_OVERVIEW_SCREENS:.1f} screens shown" if clipped else ""
    return f"showed the whole page (about {screens:.1f} screens tall{c}) as a picture before the screen"


class FakeBrowserClosedError(RuntimeError):
    """What the fake's ``frame()`` raises once ``closed`` (the real one raises ``executor.BrowserClosedError``)."""


def live_frame(tag: str, *, phash: str | None = None, ts_ms: int = 0) -> Frame:
    """An own frame of the agent browser: ``Source.WINDOW``, 1280x800, bytes and phash derived from ``tag``."""
    return Frame(
        source=Source.WINDOW,
        ts_ms=ts_ms,
        width=W,
        height=H,
        jpeg=b"\xff\xd8" + f"LIVE-{tag}|".encode() * 8,
        phash=phash if phash is not None else hashlib.sha256(tag.encode()).hexdigest()[:16],
    )


def hud_frame(ts_ms: int = 0) -> Frame:
    """A distinctive HUD (perception) frame: its bytes must appear in no live request body (H-T111c-5)."""
    return Frame(
        source=Source.SCREEN,
        ts_ms=ts_ms,
        width=1280,
        height=720,
        jpeg=b"\xff\xd8" + b"HUD-SENTINEL-T0111G|" * 8,
        phash="feedfacefeedface",
        title="the HUD's shared tab",
        app="firefox",
    )


def data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def b64(jpeg: bytes) -> str:
    return base64.b64encode(jpeg).decode()


def builder_url(frame: Frame) -> str:
    """Oracle (B8 "Encoding"): the data URL the real ``PromptBuilder`` writes for ``frame`` as a current frame."""
    from local_astra.conductor.prompt_builder import PromptBuilder
    from local_astra.contracts.llm import ImagePart

    prompt = PromptBuilder("system").prefix([], frames=(frame,))
    parts = [p for p in prompt.messages[-1].content if isinstance(p, ImagePart)]
    assert len(parts) == 1, parts
    return parts[0].url


Handler = Callable[["FakeLiveBrowser", ToolInvocation], Any]


class FakeLiveBrowser:
    """The live browser seam over no browser (module docstring). Test controls are plain attributes/methods."""

    #: the ``ToolExecutor`` protocol's ``mode`` (``BrowserToolExecutor.mode``)
    mode = "live"

    def __init__(
        self,
        frames: Iterable[Frame] = (),
        *,
        post_frames: Iterable[Frame] = (),
        now_ms: Callable[[], int] = lambda: 0,
        echo: str | None = "pixels",
    ) -> None:
        #: T-0134: the executor's ``echo`` ("pixels" | "model"); ``None``: the attribute is absent (duck typing)
        if echo is not None:
            self.echo = echo
        #: T-0134: the scripted page extent per own frame phash, ``{"above_px", "below_px", "viewport_px"}`` (CSS
        #: px) or ``None`` (an unstable or failed read); a phash not in the map has no extent either
        self.extents: dict[str, dict[str, float] | None] = {}
        #: frames ``frame()`` returns, in order (a fresh ``auto-N`` frame once empty)
        self.frames: deque[Frame] = deque(frames)
        #: post-action frames of executed actions, in order (a fresh ``post-N`` frame once empty)
        self.post_frames: deque[Frame] = deque(post_frames)
        self.now_ms = now_ms
        self.calls: list[tuple[Any, ...]] = []
        self.invocations: list[ToolInvocation] = []
        self.frames_returned: list[Frame] = []
        self.cancelled: list[str] = []
        self.handlers: dict[str, Handler] = {}
        #: "ok" | "raise" | "hang" for ``frame()``
        self.frame_mode = "ok"
        self.frame_hang_cancelled = 0
        #: when set, every ``frame()`` call blocks on this event before taking its frame (a capture that takes
        #: time; the group B review rows A3/A4 release it at a chosen moment)
        self.frame_gate: asyncio.Event | None = None
        #: ``frame()`` calls in flight now, and the most ever in flight at once (A4: calls are serialised)
        self.frame_active = 0
        self.frame_max_active = 0
        self.close_error: BaseException | None = None
        self.generation = 1
        self._own: dict[str, int] = {}
        self._latest: Frame | None = None
        self._closed = False
        self._egress: list[dict[str, str]] = []
        self._release = asyncio.Event()
        self._n = 0
        #: T-0140 A1: the scripted overview (module docstring, "T-0140 group A1 additions")
        self.overview_image: Frame | None = None
        self.page_metrics: dict[str, float] = {
            "scroll_y": 0.0,
            "viewport_height": float(H),
            "content_height": float(H),
        }
        self.overview_moves = False
        self.overview_available = True
        #: overview id -> frame, until read once by ``overview_of``
        self.overviews: dict[str, Frame] = {}
        #: every ``overview_of`` read, in order: (id, returned a frame)
        self.overview_reads: list[tuple[str, bool]] = []
        #: T-0150 A1 (module docstring, "T-0150 group A additions"): a slow or never-returning capture, the page's
        #: URL and per-URL scripted metrics. All off by default, so every earlier row sees what it saw.
        self.overview_gate: asyncio.Event | None = None
        self.overview_hang = False
        self.overview_cancel_ticks = 0
        self.url: str | None = None
        self.url_metrics: dict[str, dict[str, float]] = {}
        #: T-0151 A2 (module docstring, "T-0151 group A2 additions"): the capture facts per overview id, the
        #: registered overview frames (phash -> geometry), every ``overview_frame_of`` read, and the scripted
        #: registration controls. All off by default.
        self.overview_capture: dict[str, dict[str, Any]] = {}
        self.overview_frames: dict[str, dict[str, Any]] = {}
        self.overview_frame_reads: list[tuple[str, bool]] = []
        self.overview_frame_none = False
        self.overview_wide = False
        #: T-0136 B (module docstring, "T-0136 group B additions"): call id -> keys merged into an executed
        #: ``click_at``/``type_at``'s ``resolved`` (e.g. ``{"x_px": 901, "y_px": 444}`` or ``{"scrolled_px": 120}``),
        #: before ``frame_after``. Empty by default, so every earlier row sees what it saw.
        self.resolved_extra: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ closes by delegation
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "echo":
            # ``echo`` absent (T-0134 A5's duck typing): ``getattr(browser, "echo", None)`` is ``None``
            raise AttributeError(name)
        raise AssertionError(
            f"the session used live_browser.{name}, which is not in the LiveBrowser seam (frame, latest_frame, "
            "open, egress_log, close, execute, closed; tests/unit/_live_browser_support.py docstring)"
        )

    # ------------------------------------------------------------------ the seam
    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def latest_frame(self) -> Frame | None:
        return self._latest

    @property
    def egress_log(self) -> tuple[dict[str, str], ...]:
        return tuple(dict(r) for r in self._egress)

    async def frame(self) -> Frame:
        self.calls.append(("frame",))
        if self._closed:
            raise FakeBrowserClosedError(CLOSED)
        if self.frame_mode == "raise":
            raise RuntimeError("the fake browser could not take a frame")
        self.frame_active += 1
        self.frame_max_active = max(self.frame_max_active, self.frame_active)
        try:
            if self.frame_mode == "hang":
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.frame_hang_cancelled += 1
                    raise
            if self.frame_gate is not None:
                await self.frame_gate.wait()
            f = self.frames.popleft() if self.frames else self._auto("auto")
            return self._register(f)
        finally:
            self.frame_active -= 1

    async def open(self, url: str) -> None:
        self.calls.append(("open", url))
        self.url = url  # T-0150: the page whose metrics ``url_metrics`` may script

    async def close(self) -> None:
        self.calls.append(("close",))
        if self.close_error is not None:
            raise self.close_error

    async def execute(self, inv: ToolInvocation) -> ToolOutcome:
        self.calls.append(("execute", inv.call_id))
        self.invocations.append(inv)
        if inv.name == "google_search":
            return ToolOutcome(
                call_id=inv.call_id, status="dry_run", action="search", reason=None, resolved={}
            )
        if self._closed:
            return refused(inv, CLOSED)
        if inv.name == "page_overview" and (self.overview_gate is not None or self.overview_hang):
            # T-0150 A1.3/A1.5: a capture that takes time (released through ``overview_gate``) or never returns;
            # its start and its end (or cancellation) are recorded in ``calls`` so a row can read the interleaving
            # with ``frame()``
            self.calls.append(("overview.start", inv.call_id))
            try:
                if self.overview_hang:
                    await asyncio.Event().wait()
                else:
                    assert self.overview_gate is not None
                    await self.overview_gate.wait()
            except asyncio.CancelledError:
                # a real capture restores the viewport after a cancel (O-A1-2) before it is done; model that with
                # ``overview_cancel_ticks`` event-loop turns, so a ``frame()`` that does not wait for the capture's
                # end lands inside it
                # a second cancel during the unwind ends it early; the end is recorded either way
                try:
                    for _ in range(self.overview_cancel_ticks):
                        await asyncio.sleep(0)
                finally:
                    self.calls.append(("overview.cancelled", inv.call_id))
                raise
            self.calls.append(("overview.end", inv.call_id))
        handler = self.handlers.get(inv.call_id)
        if handler is not None:
            out = handler(self, inv)
            if asyncio.iscoroutine(out):
                out = await out
            return out  # type: ignore[no-any-return]
        return self.perform(inv)

    # ------------------------------------------------------------------ the executor's rules
    def perform(self, inv: ToolInvocation) -> ToolOutcome:
        """The default: validate the frame like the executor, then 'act' and capture the post-action frame."""
        resolved: dict[str, Any] = {}
        if inv.name == "page_overview":
            return self.perform_overview(inv)
        if inv.name in ("click_at", "type_at"):
            fr = inv.frame
            if fr is None:
                return refused(inv, "no frame")
            ov = (
                self.overview_frames.get(fr.phash)
                if fr.phash is not None and fr.phash not in self._own
                else None
            )
            if (
                ov is not None
                and fr.source == Source.WINDOW
                and (fr.width, fr.height) == (ov["width"], ov["height"])
            ):
                return self.perform_on_overview(inv, ov)  # T-0151 A1-2..A1-5, mirrored
            if (
                fr.phash is None
                or self._own.get(fr.phash) != self.generation
                or fr.source != Source.WINDOW
                or (fr.width, fr.height) != (W, H)
            ):
                return refused(inv, NOT_OWN)
            x, y = int(inv.arguments.get("x", 0)), int(inv.arguments.get("y", 0))
            resolved = {"x_px": x * W // 1000, "y_px": y * H // 1000}
            # T-0136 B: scripted ``resolved`` keys for this call (empty by default: every earlier row unchanged)
            resolved.update(self.resolved_extra.get(inv.call_id, {}))
            if getattr(self, "echo", None) == "model":
                # T-0134 A2: the call's own 0-999 units (the arguments it executed); resolved keeps pixels
                action = f"clicked at ({x}, {y})"
            else:
                action = f"clicked at ({resolved['x_px']}, {resolved['y_px']})"
        elif inv.name == "scroll":
            action = f"scrolled {inv.arguments.get('direction')} by 600 px"
        else:
            action = f"pressed {inv.arguments.get('combo')}"
        after = self.capture_post()
        resolved["frame_after"] = after.phash
        return ToolOutcome(
            call_id=inv.call_id, status="executed", action=action, reason=None, resolved=resolved
        )

    def perform_overview(self, inv: ToolInvocation) -> ToolOutcome:
        """T-0140 A3 over the scripted image and metrics (module docstring). Dispatches nothing, captures no own
        frame, leaves ``latest_frame`` alone."""
        if not self.overview_available:
            return refused(inv, OVERVIEW_UNAVAILABLE)
        if self.overview_moves:
            return failed(inv, OVERVIEW_MOVED)
        metrics = self.scripted_metrics()
        content = float(metrics["content_height"])
        viewport = float(metrics["viewport_height"])
        if content <= OVERVIEW_FITS_RATIO * viewport:
            return ToolOutcome(
                call_id=inv.call_id,
                status="executed",
                action=OVERVIEW_FITS,
                reason=None,
                resolved={},
            )
        assert self.overview_image is not None, (
            "FakeLiveBrowser: page_overview on a page taller than the screen, but the row set no overview_image"
        )
        screens = content / viewport
        clipped = screens > MAX_OVERVIEW_SCREENS
        key = f"overview-{len(self.overviews) + len(self.overview_reads) + len(self.overview_frame_reads) + 1}"
        self.overviews[key] = self.overview_image
        # T-0151 A2: the capture's facts (A1-1's registration conditions read them)
        self.overview_capture[key] = {
            "latest": self._latest.phash if self._latest is not None else None,
            "generation": self.generation,
            "clipped": clipped,
            "page_width": W + 1 if self.overview_wide else W,
            "page_height": content,
            "viewport_height": viewport,
            "content_height": content,
        }
        return ToolOutcome(
            call_id=inv.call_id,
            status="executed",
            action=overview_action(screens, clipped),
            reason=None,
            resolved={"overview": key, "screens": screens, "clipped": clipped},
        )

    def scripted_metrics(self) -> dict[str, float]:
        """T-0150: the current page's metrics: ``url_metrics[url]`` when the page's URL is scripted there, else
        ``page_metrics`` (every T-0140 row's)."""
        if self.url is not None and self.url in self.url_metrics:
            return self.url_metrics[self.url]
        return self.page_metrics

    def navigate(self, url: str) -> None:
        """T-0150 test control: the active page is now ``url`` (an action that navigated)."""
        self.url = url

    def overview_captures(self) -> int:
        """T-0150: how many ``page_overview`` invocations reached the browser (the capture-call counter)."""
        return sum(1 for i in self.invocations if i.name == "page_overview")

    def overview_of(self, key: Any) -> Frame | None:
        """T-0140 A3: the overview frame of ``resolved["overview"]``, once; a second read (or an unknown id) is
        ``None``."""
        self.calls.append(("overview_of", key))
        f = self.overviews.pop(key, None) if isinstance(key, str) else None
        self.overview_reads.append((str(key), f is not None))
        return f

    def overview_frame_of(self, key: Any) -> Frame | None:
        """T-0151 A1-1, mirrored (module docstring, "T-0151 group A2 additions"): the picture of ``key`` from
        ``overview_of``'s once-only store, registered as an overview frame when A1-1's conditions hold; else
        ``None`` and the picture is dropped."""
        self.calls.append(("overview_frame_of", key))
        f = self.overviews.pop(key, None) if isinstance(key, str) else None
        facts = self.overview_capture.get(key) if isinstance(key, str) else None
        ok = (
            f is not None
            and facts is not None
            and not self.overview_frame_none
            and not facts["clipped"]
            and facts["page_width"] == W
            and facts["generation"] == self.generation
            and facts["latest"] is not None
            and self._latest is not None
            and self._latest.phash == facts["latest"]
        )
        self.overview_frame_reads.append((str(key), bool(ok)))
        if not ok:
            return None
        assert f is not None and facts is not None and f.phash is not None
        self.overview_frames[f.phash] = {
            "generation": facts["generation"],
            "latest": facts["latest"],
            "width": f.width,
            "height": f.height,
            "page_width": facts["page_width"],
            "page_height": facts["page_height"],
            "viewport_height": facts["viewport_height"],
            "content_height": facts["content_height"],
        }
        return f

    def perform_on_overview(self, inv: ToolInvocation, ov: dict[str, Any]) -> ToolOutcome:
        """T-0151 A1-2..A1-5 over the scripted metrics (module docstring): refused after a restart or once the screen
        changed; else mapped onto the page, scrolled once into view, "clicked" at the viewport point, captured."""
        if ov["generation"] != self.generation:
            return refused(inv, RESTART_FRAME)
        if self._latest is None or self._latest.phash != ov["latest"]:
            return refused(inv, OVERVIEW_STALE)
        x, y = int(inv.arguments.get("x", 0)), int(inv.arguments.get("y", 0))
        px = to_px(x, int(ov["page_width"]))
        py = to_px(y, round(float(ov["page_height"])))
        metrics = self.scripted_metrics()
        s = float(metrics["scroll_y"])
        h = float(metrics["viewport_height"])
        c = float(metrics["content_height"])
        scrolled = 0
        if not s <= py < s + h:
            target = min(max(py - h / 2, 0.0), max(0.0, c - h))
            scrolled = round(target - s)
            metrics["scroll_y"] = s + scrolled
        vy = round(py - (s + scrolled))
        shown = (x, y) if getattr(self, "echo", None) == "model" else (px, vy)
        action = f"clicked at ({shown[0]}, {shown[1]})"
        resolved: dict[str, Any] = {
            "x_px": px,
            "y_px": vy,
            "page_y_px": py,
            "scrolled_px": scrolled,
        }
        if inv.name == "type_at":
            text = str(inv.arguments.get("text", ""))
            enter = bool(inv.arguments.get("press_enter", False))
            action += f" and typed {len(text)} characters" + (" and pressed Enter" if enter else "")
            resolved |= {"text": text, "press_enter": enter}
        if scrolled:
            action = (
                f"scrolled {'down' if scrolled > 0 else 'up'} by {abs(scrolled)} px and {action}"
            )
        resolved.update(self.resolved_extra.get(inv.call_id, {}))  # T-0136 B (empty by default)
        after = self.capture_post()
        resolved["frame_after"] = after.phash
        return ToolOutcome(
            call_id=inv.call_id, status="executed", action=action, reason=None, resolved=resolved
        )

    def land(self, f: Frame) -> Frame:
        """T-0151 test control: an own frame captured by someone else (a takeover snapshot); it becomes
        ``latest_frame``."""
        return self._register(f)

    def capture_post(self) -> Frame:
        f = self.post_frames.popleft() if self.post_frames else self._auto("post")
        return self._register(f)

    def _auto(self, kind: str) -> Frame:
        self._n += 1
        return live_frame(f"{kind}-{self._n}-g{self.generation}")

    def _register(self, f: Frame) -> Frame:
        f = f.model_copy(update={"ts_ms": self.now_ms()})
        assert f.phash is not None
        self._own[f.phash] = self.generation
        self._latest = f
        self.frames_returned.append(f)
        return f

    # ------------------------------------------------------------------ test controls
    def close_window(self) -> None:
        """A2: the human closed the last window."""
        self._closed = True

    def restart(self) -> None:
        """A browser restart: every earlier frame is from an old generation."""
        self.generation += 1

    def block(self, *hosts: tuple[str, str], extra: dict[str, str] | None = None) -> None:
        """Blocked egress attempts; ``extra`` keys model a log that carries more than ``{host, scheme}``."""
        for host, scheme in hosts:
            self._egress.append({"host": host, "scheme": scheme, **(extra or {})})

    def hang(self, call_id: str) -> None:
        async def wait(fb: FakeLiveBrowser, inv: ToolInvocation) -> ToolOutcome:
            try:
                await fb._release.wait()
            except asyncio.CancelledError:
                fb.cancelled.append(inv.call_id)
                fb.calls.append(("execute.cancelled", inv.call_id))
                raise
            return fb.perform(inv)

        self.handlers[call_id] = wait

    def outcome(self, call_id: str, make: Callable[[ToolInvocation], ToolOutcome]) -> None:
        self.handlers[call_id] = lambda fb, inv: make(inv)

    def executes(self) -> int:
        return sum(1 for c in self.calls if c[0] == "execute")

    # ------------------------------------------------------------------ T-0134: the page extent
    def extent_of(self, key: Any) -> dict[str, float] | None:
        """T-0134 A4: the extent captured with an own frame. ``key`` may be the executor's phash, the session's
        capture identity (``capture:<sha256(jpeg)[:32]>``) or a frame/geometry with a ``phash``; the fake resolves
        each to the frame's phash. Unknown: ``None``."""
        self.calls.append(("extent_of", key))
        phash: str | None
        if isinstance(key, str) and key.startswith("capture:"):
            found = [
                f.phash
                for f in self.frames_returned
                if "capture:" + hashlib.sha256(f.jpeg).hexdigest()[:32] == key
            ]
            phash = found[-1] if found else None
        elif isinstance(key, str):
            phash = key
        else:
            phash = getattr(key, "phash", None)
        if phash is None:
            return None
        value = self.extents.get(phash)
        return dict(value) if value is not None else None


def refused(inv: ToolInvocation, reason: str) -> ToolOutcome:
    return ToolOutcome(
        call_id=inv.call_id,
        status="refused",
        action=f"refused {inv.name}: {reason}",
        reason=reason,
        resolved={},
    )


def failed(inv: ToolInvocation, reason: str) -> ToolOutcome:
    return ToolOutcome(
        call_id=inv.call_id,
        status="failed",
        action=f"failed {inv.name}: {reason}",
        reason=reason,
        resolved={},
    )


# --------------------------------------------------------------------------------------------- the kit and the rig


def session_mod() -> Any:
    return importlib.import_module("local_astra.conductor.session")


def live_frame_timeout_ms() -> int:
    s = session_mod()
    value = getattr(s, "LIVE_FRAME_TIMEOUT_MS", None)
    assert isinstance(value, int), (
        "T-0111g B6: session.py has no int module constant LIVE_FRAME_TIMEOUT_MS (_live_browser_support docstring)"
    )
    assert 0 < value <= s.TOOL_EXEC_TIMEOUT_MS // 2, (
        f"T-0111g B6: LIVE_FRAME_TIMEOUT_MS is well under the {s.TOOL_EXEC_TIMEOUT_MS} ms deadline; got {value}"
    )
    return value


def live_kit(browser: Any, *, repair_off: str | None = None) -> Any:
    """``ToolKit(mode="live", executor=browser, live_browser=browser, ...)``; a kit without the field is a named
    failure, never a ``TypeError``."""
    from tests.unit import _tools_switch_support as T

    base = T.make_kit(executor=browser, repair_off=repair_off)
    names = {f.name for f in dataclasses.fields(base)}
    assert "live_browser" in names, (
        "T-0111g B5: ToolKit gains `live_browser: LiveBrowser | None = None` (and mode 'live'); its fields are "
        f"{sorted(names)}"
    )
    return dataclasses.replace(base, mode="live", live_browser=browser)


@asynccontextmanager
async def live_rig(
    tmp_path: Path,
    browser: FakeLiveBrowser | None = None,
    *,
    script: Any = (),
    hud_frames: Any = None,
    repair_off: str | None = None,
    machine_cls: Any = None,
) -> Any:
    """``tool_rig`` with the loop on and a live kit over ``browser``; the HUD perception serves ``hud_frames``
    (default: the sentinel). ``rig.browser`` is the fake."""
    from tests.unit import test_session_tool_loop as tsl
    from tests.unit._tool_turn_support import tool_rig

    browser = browser if browser is not None else FakeLiveBrowser()
    kit = live_kit(browser, repair_off=repair_off)
    extra: dict[str, Any] = {}
    if machine_cls is not None:
        extra["machine_cls"] = machine_cls
    async with tool_rig(
        tmp_path,
        kit=kit,
        policy=tsl.LOOP,
        script=script,
        frames=hud_frames if hud_frames is not None else [hud_frame(0)],
        **extra,
    ) as rig:
        browser.now_ms = lambda: rig.clock.now_ms
        rig.browser = browser  # type: ignore[attr-defined]
        yield rig


def image_urls(body: dict[str, Any]) -> list[str]:
    """Every image URL of a request body, in order."""
    out: list[str] = []
    for m in body.get("messages", []):
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    out.append(part["image_url"]["url"])
    return out


def turn_extras(rig: Any, turn_id: str = "t1") -> dict[str, Any]:
    return dict(rig.record(turn_id).extras)


# --------------------------------------------------------------------------------------------- T-0131 group B
#
# ReGround needs frames whose bytes are a real JPEG (the session cuts the crop from them with PIL), a fake LLM that
# tells a pass-2 request apart and scripts it, and a session built with ``reground=True``. Everything below is
# additive: ``live_rig`` and ``FakeLiveBrowser`` are unchanged, so every T-0111g row sees exactly what it saw.

#: what a pass-2 script entry may be besides a list of commits: hold the request until it is aborted
HANG = "hang"


def jpeg_bytes(width: int = W, height: int = H, *, seed: int = 0) -> bytes:
    """A real JPEG, ``width`` x ``height``: a grid of solid 64-px blocks whose colours depend on the block and on
    ``seed`` (so two seeds give two different captures, and any crop has recognisable content)."""
    import io

    from PIL import Image

    img = Image.new("RGB", (width, height))
    for by in range((height + 63) // 64):
        for bx in range((width + 63) // 64):
            colour = (
                (bx * 37 + seed * 11) % 256,
                (by * 53 + seed * 29) % 256,
                (bx + by + seed) * 17 % 256,
            )
            box = (bx * 64, by * 64, min(width, bx * 64 + 64), min(height, by * 64 + 64))
            img.paste(colour, box)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


_JPEG_CACHE: dict[tuple[int, int, int], bytes] = {}


def jpeg_frame(tag: str, *, seed: int | None = None) -> Frame:
    """``live_frame(tag)`` with real 1280x800 JPEG bytes (cached per seed; the seed defaults to the tag's hash)."""
    s = seed if seed is not None else int(hashlib.sha256(tag.encode()).hexdigest()[:4], 16)
    key = (W, H, s)
    if key not in _JPEG_CACHE:
        _JPEG_CACHE[key] = jpeg_bytes(seed=s)
    return live_frame(tag).model_copy(update={"jpeg": _JPEG_CACHE[key]})


def _tool_llm_class() -> Any:
    from tests.unit._tool_turn_support import ToolLlm

    return ToolLlm


def make_reground_llm(browser: FakeLiveBrowser, script: Any = ()) -> Any:
    """A ``ToolLlm`` that recognises a pass-2 request and serves it from ``script_pass2`` (never from ``script``,
    the continuation or the step scripts).

    A request is pass 2 when its prompt holds an image part whose URL is not the data URL of any frame the fake
    browser has returned (the crop is the only image the session makes itself). Recorded: ``pass2_prompts``
    (engine id, prompt), ``pass2_ids``, ``pass2_overlaps`` (the generates/continuations live at each pass-2
    start, B6), ``events`` (``("start"|"end", kind, id, clock ms)`` for every generate and pass 2) and the
    inherited ``aborted``."""
    base = _tool_llm_class()

    class RegroundLlm(base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__(list(script))
            self.browser = browser
            self.pass2_scripts: list[Any] = []
            self.pass2_prompts: list[tuple[str, Any]] = []
            self.pass2_ids: list[str] = []
            self.pass2_overlaps: list[tuple[str, ...]] = []
            self.events: list[tuple[str, str, str, int]] = []
            self._hangs: dict[str, asyncio.Event] = {}
            #: called (no arguments) at the start of every pass-2 request, before its script is served
            self.on_pass2: Any = None

        def script_pass2(self, *entries: Any) -> None:
            """One entry per pass-2 request, in order: a list of commits, an exception (raised at the first
            ``__anext__``, like the real client) or ``HANG``. No entry left: the request yields nothing."""
            self.pass2_scripts = list(entries)

        def _now(self) -> int:
            return int(self.clock.now_ms) if getattr(self, "clock", None) is not None else 0

        def is_pass2(self, prompt: Any) -> bool:
            from local_astra.contracts.llm import ImagePart

            known = {data_url(f.jpeg) for f in self.browser.frames_returned}
            return any(
                isinstance(p, ImagePart) and p.url not in known
                for m in prompt.messages
                for p in m.content
            )

        async def generate(  # type: ignore[override]
            self,
            prompt: Any,
            request_id: str,
            options: Any = None,
            *,
            xargs: Any = None,
            cd: Any = None,
        ) -> Any:
            if self.is_pass2(prompt):
                async for c in self._pass2(prompt, request_id):
                    yield c
                return
            self.events.append(("start", "generate", request_id, self._now()))
            try:
                async for c in super().generate(prompt, request_id, options, xargs=xargs, cd=cd):
                    yield c
            finally:
                self.events.append(("end", "generate", request_id, self._now()))

        async def _pass2(self, prompt: Any, request_id: str) -> Any:
            self.pass2_prompts.append((request_id, prompt))
            self.pass2_ids.append(request_id)
            self.pass2_overlaps.append(tuple(sorted(self.live_generates)))
            if self.on_pass2 is not None:
                self.on_pass2()
            self.events.append(("start", "pass2", request_id, self._now()))
            try:
                entry = self.pass2_scripts.pop(0) if self.pass2_scripts else []
                if isinstance(entry, BaseException):
                    raise entry
                self._active.add(request_id)
                if entry == HANG:
                    ev = self._hangs.setdefault(request_id, asyncio.Event())
                    await ev.wait()
                    return
                for c in entry:
                    if request_id not in self._active:
                        return
                    await asyncio.sleep(0)
                    yield c.model_copy(update={"request_id": request_id})
                self._active.discard(request_id)
            finally:
                self.events.append(("end", "pass2", request_id, self._now()))

        async def abort(self, request_id: str) -> None:
            await super().abort(request_id)
            ev = self._hangs.get(request_id)
            if ev is not None:
                ev.set()

    return RegroundLlm()


@asynccontextmanager
async def reground_rig(
    tmp_path: Path,
    browser: FakeLiveBrowser,
    *,
    script: Any = (),
    reground: bool | None = True,
    session_kwargs: dict[str, Any] | None = None,
) -> Any:
    """``live_rig``'s session (loop on, live kit over ``browser``, the HUD sentinel frame) with the pass-2 fake LLM
    (``rig.llm``) and ``Session(..., reground=reground, **session_kwargs)``. A session without the keyword is a
    named ``AssertionError`` (T-0131 B1's pinned seam), never a ``TypeError``."""
    import inspect

    from tests.unit import _tool_turn_support as tts
    from tests.unit import test_session_tool_loop as tsl

    real = tts.Session
    # ``reground=None``: the keyword is not passed at all (the session's default)
    kwargs = {**({} if reground is None else {"reground": reground}), **(session_kwargs or {})}
    params = inspect.signature(real.__init__).parameters
    missing = [k for k in kwargs if k not in params]
    assert not missing, (
        f"T-0131 B1: Session(..., {', '.join(missing)}=...) does not exist yet (pinned seam)"
    )

    class Regrounding(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            for key, value in kwargs.items():
                k.setdefault(key, value)
            super().__init__(*a, **k)

    llm = make_reground_llm(browser, script)
    tts.Session = Regrounding  # type: ignore[misc]
    try:
        async with tts.tool_rig(
            tmp_path,
            kit=live_kit(browser),
            policy=tsl.LOOP,
            llm=llm,
            frames=[hud_frame(0)],
        ) as rig:
            browser.now_ms = lambda: rig.clock.now_ms
            rig.browser = browser  # type: ignore[attr-defined]
            yield rig
    finally:
        tts.Session = real  # type: ignore[misc]


# --------------------------------------------------------------------------------------------- T-0134 group A
#
# Additive, like T-0131's: ``FakeLiveBrowser`` gains ``echo`` (default "pixels", the executor's default; ``None``
# leaves the attribute absent) with the model-unit wording under ``"model"``, and a scripted ``extents`` map read by
# ``extent_of``. Every earlier row sees exactly what it saw: with the defaults the fake's actions are unchanged.


def live_feedback(*, screen_label: bool = True, page_extent: bool = False) -> Any:
    """``session.LiveFeedback(screen_label=..., page_extent=...)``; its absence is a named ``AssertionError``. A
    ``ValueError`` from the constructor propagates (A1 lets the check sit there or in ``Session``)."""
    s = session_mod()
    cls = getattr(s, "LiveFeedback", None)
    assert isinstance(cls, type) and dataclasses.is_dataclass(cls), (
        "T-0134 A1: session.LiveFeedback (a frozen dataclass in session.py) does not exist yet"
    )
    return cls(screen_label=screen_label, page_extent=page_extent)


@asynccontextmanager
async def feedback_rig(
    tmp_path: Path,
    browser: Any,
    *,
    script: Any = (),
    session_kwargs: dict[str, Any] | None = None,
    reground_llm: bool = False,
    kit: Any = None,
    machine_cls: Any = None,
) -> Any:
    """``live_rig``'s session (loop on, live kit over ``browser``, the HUD sentinel frame) built with
    ``Session(..., **session_kwargs)`` (e.g. ``live_feedback=...``, ``reground=True``). A keyword the session does
    not take is a named ``AssertionError`` (T-0134 A1), never a ``TypeError``. ``reground_llm``: the pass-2 fake LLM
    of ``make_reground_llm`` (``browser`` then needs ``frames_returned``). ``kit``: another kit (default the live
    kit over ``browser``)."""
    import inspect

    from tests.unit import _tool_turn_support as tts
    from tests.unit import test_session_tool_loop as tsl

    real = tts.Session
    kwargs = dict(session_kwargs or {})
    params = inspect.signature(real.__init__).parameters
    missing = [k for k in kwargs if k not in params]
    assert not missing, f"T-0134 A1: Session(..., {', '.join(missing)}=...) does not exist yet"

    class WithFeedback(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            for key, value in kwargs.items():
                k.setdefault(key, value)
            super().__init__(*a, **k)

    extra: dict[str, Any] = {}
    if reground_llm:
        extra["llm"] = make_reground_llm(browser, script)
    else:
        extra["script"] = script
    if machine_cls is not None:
        extra["machine_cls"] = machine_cls
    tts.Session = WithFeedback  # type: ignore[misc]
    try:
        async with tts.tool_rig(
            tmp_path,
            kit=kit if kit is not None else live_kit(browser),
            policy=tsl.LOOP,
            frames=[hud_frame(0)],
            **extra,
        ) as rig:
            if hasattr(browser, "now_ms"):
                browser.now_ms = lambda: rig.clock.now_ms
            rig.browser = browser  # type: ignore[attr-defined]
            yield rig
    finally:
        tts.Session = real  # type: ignore[misc]


# --------------------------------------------------------------------------------------------- T-0136 group B
#
# Additive, like the groups above: real JPEG scenes with a known change, and a spy on the per-turn telemetry
# object. ``FakeLiveBrowser.resolved_extra`` (in the class) is empty by default.

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
Block = tuple[int, int, int, int, tuple[int, int, int]]

_SCENE_CACHE: dict[tuple[Any, ...], bytes] = {}


def scene_jpeg(
    width: int = W,
    height: int = H,
    *,
    blocks: Iterable[Block] = (),
    background: tuple[int, int, int] = WHITE,
    quality: int = 90,
) -> bytes:
    """A real JPEG, ``width`` x ``height``: ``background`` with solid rectangles ``(x0, y0, x1, y1, colour)``
    (viewport px, ``x1``/``y1`` exclusive) painted in order. Cached per arguments."""
    import io

    from PIL import Image

    key = (width, height, tuple(blocks), background, quality)
    if key not in _SCENE_CACHE:
        img = Image.new("RGB", (width, height), background)
        for x0, y0, x1, y1, colour in key[2]:
            img.paste(colour, (x0, y0, x1, y1))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=quality)
        _SCENE_CACHE[key] = out.getvalue()
    return _SCENE_CACHE[key]


def scene_frame(
    tag: str,
    *,
    blocks: Iterable[Block] = (),
    background: tuple[int, int, int] = WHITE,
    quality: int = 90,
    width: int = W,
    height: int = H,
    phash: str | None = None,
) -> Frame:
    """``live_frame(tag)`` (an own frame, phash from ``tag``) whose bytes are ``scene_jpeg(...)``."""
    jpeg = scene_jpeg(width, height, blocks=blocks, background=background, quality=quality)
    return live_frame(tag, phash=phash).model_copy(
        update={"jpeg": jpeg, "width": width, "height": height}
    )


class TurnLogSpy:
    """Every ``session._TurnLog`` created while installed, in creation order (``logs``)."""

    def __init__(self) -> None:
        self.logs: list[Any] = []

    def by_turn(self, turn_id: str) -> Any:
        found = [t for t in self.logs if getattr(t, "turn_id", None) == turn_id]
        assert found, f"no _TurnLog was created for {turn_id}; {[t.turn_id for t in self.logs]}"
        return found[-1]


def turn_logs(monkeypatch: Any) -> TurnLogSpy:
    """Install a recording subclass of ``session._TurnLog`` on the session module (undone by ``monkeypatch``)."""
    s = session_mod()
    real = s._TurnLog
    spy = TurnLogSpy()

    class Recorded(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            spy.logs.append(self)

    Recorded.__name__ = Recorded.__qualname__ = "_TurnLog"
    monkeypatch.setattr(s, "_TurnLog", Recorded)
    return spy
