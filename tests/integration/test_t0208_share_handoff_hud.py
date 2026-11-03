"""T-0208 group A1: a real Share click on the real HUD page reaches the real live Session's step-1 body.

docs/tasks/T-0208.md A1.1-A1.5 and "Tests to write" for this module. The chain under test: ``#share-button`` (a real
mouse click over CDP) -> the page's ``startCapture("screen", "share")`` -> ``astra.vision.select shared`` on the
socket -> the real ``HudServer.route`` -> ``Session.apply_vision_select`` -> the echo on every page; then the page's
``astra.frame`` posts -> ``BrowserFrameSource`` (one object that is both the server's frame sink and the session's
perception, the production shape of ``cli``) -> a typed ``astra.text`` question -> the step-1 request body, the turn
line, the published ``astra.vision.frame`` and ``#preview``.

**Rows** (expected green at BASE; A1.5: a red row is reported, never edited to pass):

- ``test_share_click_selects_shared_on_the_real_session`` (A1.1)
- ``test_next_turn_step1_is_the_posted_frame_and_labelled`` (A1.2)
- ``test_stop_sharing_returns_to_browser`` (A1.3)
- ``test_share_and_ask_at_once_never_shows_the_browser`` (A1.4, adversarial; prints ``shared`` or ``blind`` with the
  1-minute load, asserts neither)
- ``test_url_capture_never_switches_the_real_session`` (negative: ``?source=synthetic`` never selects shared)

**The rig.** ``tests/integration/test_hud_rehearsal.py``'s ``Live`` (the T-0036/T-0039 harness ``test_hud_text_mode``
imports), subclassed here only to build a *live* session: the live kit over T-0177's ``VisionBrowser`` (a fake agent
browser whose one frame ``F0`` is distinctive), ``ToolLlm`` with the loop policy, a real ``SessionLog``, and the real
``BrowserFrameSource`` as perception and frame sink. ``Live``'s wire tap, ``until``, ``idle`` and ``turn_lines`` are
used unchanged; ``test_hud_text_mode.type_and_submit_js`` types the question; T-0177 C's ``_click`` is the real mouse
click. ``getDisplayMedia`` is replaced by an init script returning a real ``canvas.captureStream`` (so
``capture.synthetic`` is false, as for a human's Share), painted with a solid, distinctive pattern standing in for
H1's image, which does not exist until group B (the task allows this).

**Q-2 reading (stated):** byte equality is checked against the posted ``astra.frame``'s own ``jpeg_b64`` (decoded),
and, separately, against the bytes ``BrowserFrameSource`` stored for that frame; the row says which one fails.

**Regime** (T-0208 Regime, A1): the box's headless Chrome (``google-chrome``) over CDP, 1440x900 at DSF 1; real
``HudServer(port=0)`` and live ``Session`` on ``SessionClock`` (wall time); fake LLM (``ToolLlm``), ``VisionBrowser``,
fake STT/TTS/audio; the page served from ``hud/`` over loopback http with no ``?source`` (Share goes through the
page's real screen provider); n = 1 per row. No number here is a latency claim; every wait is a generous ceiling.
Runs only with ``ASTRA_BROWSER_TESTS=1`` (``make browser-handoff``, B8).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import types
import urllib.parse
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.hud_server import HudServer
from local_astra.conductor.prompt_builder import NO_IMAGE_STATEMENT
from local_astra.conductor.session import Session, SessionClock, SessionConfig
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.contracts.protocol import AstraVisionFrame, AstraVisionState
from local_astra.perception.browser_source import BrowserFrameSource
from tests.fakes import FakeMemory, FakeStt
from tests.integration.test_hud_headless import CHROME, Chrome, requires_chrome, serve_hud
from tests.integration.test_hud_rehearsal import PAGE_TIMEOUT_S, TURN_TIMEOUT_S, Live, Page
from tests.integration.test_hud_text_mode import type_and_submit_js
from tests.integration.test_t0177_hud_vision import _click
from tests.unit import _tools_switch_support as T
from tests.unit import test_session_tool_loop as tsl
from tests.unit._hud_rig import RecordingAudio, RecordingMachine, RecordingTts
from tests.unit._live_browser_support import b64, builder_url, data_url, live_frame, live_kit
from tests.unit._session_log_rig import make_log, of_kind, read_records
from tests.unit._tool_turn_support import ToolLlm, answer
from tests.unit.test_session_live import _wire, last_user
from tests.unit.test_t0177_shared_frame import SHARED_LABEL_SCREEN, VisionBrowser

GATE_ENV = "ASTRA_BROWSER_TESTS"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.browser,
    pytest.mark.skipif(
        os.environ.get(GATE_ENV) != "1",
        reason=f"{GATE_ENV} is not 1: T-0208 A1 drives headless Chrome against a real Session "
        "(make browser-handoff, docs/tasks/T-0208.md B8)",
    ),
]

#: The agent browser's own frame: distinctive bytes, so "no agent-browser byte" is a substring search.
F0 = live_frame("t0208-agent-browser")

QUESTION = "what is on my shared screen"
ANSWER = "I see a pink screen. "

#: The stand-in for H1's image (group B has not merged): a 1280x800 canvas, solid magenta with a cyan block top
#: left and a yellow block bottom right; the centre pixel is magenta. Painted the same way every 200 ms, so every
#: frame the page grabs shows the same pixels.
#:
#: The init script also wraps ``WebSocket`` (the page builds its socket as ``new globalThis.WebSocket(url)``,
#: hud/src/ws.js) to record, page side, every ``astra.vision.state`` received and every ``astra.frame`` /
#: ``astra.vision.select`` sent. A1.4's race: when ``window.__t0208_race`` holds a question, the very send call that
#: carries the page's ``select shared`` sends ``astra.text`` right after it, in the same JS task. Until then the
#: wrapper holds back any ``astra.frame`` the new capture produces (the capture can grab its first frame before
#: ``startCapture`` sends the select: seen in one run of two) and sends the held frames right after the question, in
#: order: the premise "before any frame from the new capture arrived" is made true, not hoped for.
INIT_JS = r"""
(() => {
  const rec = {streams: [], calls: 0, states: [], selects: [], frames_sent: 0, frames_held: 0, held: [], race: null};
  Object.defineProperty(window, '__t0208', {value: rec, configurable: true});
  const fake = () => {
    rec.calls += 1;
    const c = document.createElement('canvas');
    c.width = 1280; c.height = 800;
    const g = c.getContext('2d');
    const paint = () => {
      g.fillStyle = 'rgb(255,0,255)'; g.fillRect(0, 0, 1280, 800);
      g.fillStyle = 'rgb(0,255,255)'; g.fillRect(0, 0, 320, 200);
      g.fillStyle = 'rgb(255,255,0)'; g.fillRect(960, 600, 320, 200);
    };
    paint();
    setInterval(paint, 200);
    const stream = c.captureStream(5);
    rec.streams.push(stream);
    return Promise.resolve(stream);
  };
  if (window.MediaDevices) {
    MediaDevices.prototype.getDisplayMedia = function () { return fake(); };
    MediaDevices.prototype.getUserMedia = function () { return fake(); };
  }
  const Orig = window.WebSocket;
  class Tapped extends Orig {
    constructor(...args) {
      super(...args);
      this.addEventListener('message', (e) => {
        try {
          const m = JSON.parse(e.data);
          if (m && m.type === 'astra.vision.state') rec.states.push({source: m.source, replay: !!m.replay});
        } catch (_) {}
      });
    }
    send(data) {
      let m = null;
      try { m = JSON.parse(data); } catch (_) {}
      const racing = typeof window.__t0208_race === 'string';
      if (m && m.type === 'astra.frame') {
        rec.frames_sent += 1;
        if (racing) { rec.held.push(data); rec.frames_held += 1; return undefined; }
      }
      if (m && m.type === 'astra.vision.select') rec.selects.push(m.source);
      const out = super.send(data);
      if (m && m.type === 'astra.vision.select' && m.source === 'shared' && racing) {
        const text = window.__t0208_race;
        window.__t0208_race = null;
        rec.race = {frames_on_wire_before: rec.frames_sent - rec.frames_held, frames_held: rec.frames_held};
        super.send(JSON.stringify({type: 'astra.text', text}));
        for (const held of rec.held.splice(0)) super.send(held);
      }
      return out;
    }
  }
  window.WebSocket = Tapped;
})();
"""

COMBO = "main > .pane.thumb select[data-vision-source]"

PANE_JS = """
(() => {
  const sel = document.querySelector(__COMBO__);
  const opt = sel ? sel.options[sel.selectedIndex] : null;
  const prev = document.getElementById('preview');
  const vis = (el) => !!el && el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true});
  let rgb = null;
  if (vis(prev) && prev.complete && prev.naturalWidth) {
    const c = document.createElement('canvas');
    c.width = prev.naturalWidth; c.height = prev.naturalHeight;
    const g = c.getContext('2d');
    g.drawImage(prev, 0, 0);
    const d = g.getImageData(c.width >> 1, c.height >> 1, 1, 1).data;
    rgb = [d[0], d[1], d[2]];
  }
  const cap = document.querySelector('main > .pane.thumb [data-capture-caption]');
  return {
    combo_value: sel ? sel.value : null,
    combo_text: opt ? opt.textContent.replace(/\\s+/g, ' ').trim() : null,
    preview_src: prev ? prev.getAttribute('src') : null,
    preview_rgb: rgb,
    caption: cap ? cap.textContent.replace(/\\s+/g, ' ').trim() : null,
    capturing: !document.getElementById('mute-button').disabled,
    rec: window.__t0208 ? {states: window.__t0208.states, selects: window.__t0208.selects,
                           frames_sent: window.__t0208.frames_sent, race: window.__t0208.race,
                           calls: window.__t0208.calls} : null,
  };
})()
""".replace("__COMBO__", json.dumps(COMBO))

CAPTURING_JS = "!document.getElementById('mute-button').disabled"


def _magenta(rgb: Any) -> bool:
    if not isinstance(rgb, list) or len(rgb) < 3:
        return False
    r, g, b = (int(x) for x in rgb[:3])
    return r > 170 and b > 170 and g < 80


# --------------------------------------------------------------------------------------------- the rig


class PostedSink:
    """The server's ``frame_sink``: the real ``BrowserFrameSource`` plus the posted messages, in arrival order, each
    with the ``Frame`` the source stored for it (``None``: refused)."""

    def __init__(self, inner: BrowserFrameSource) -> None:
        self.inner = inner
        self.posted: list[tuple[Any, Any]] = []

    def ingest(self, message: Any) -> Any:
        frame = self.inner.ingest(message)
        self.posted.append((message, frame))
        return frame

    def accepted(self) -> list[tuple[bytes, Any]]:
        """(posted bytes, stored Frame) of every accepted frame."""
        return [(base64.b64decode(m.jpeg_b64), f) for m, f in self.posted if f is not None]


class ShareLive(Live):
    """``Live`` with a live session: the live kit over ``VisionBrowser([F0])``, ``ToolLlm`` with the loop policy, a
    real ``SessionLog``, and one ``BrowserFrameSource`` as both perception and frame sink. ``Live.__init__`` builds a
    session without a kit and cannot be parameterised, so this constructor replaces it; ``_tap`` and every driving and
    observation method are ``Live``'s own."""

    def __init__(self, tmp_path: Path, script: Sequence[Commit]) -> None:
        self.clock = SessionClock()
        self.machine = RecordingMachine(tsl.LOOP)
        self.frames = BrowserFrameSource(now_ms=lambda: self.clock.now_ms)
        self.sink = PostedSink(self.frames)
        self.llm = ToolLlm(list(script))
        self.llm.clock = self.clock
        self.stt = FakeStt()
        self.tts = RecordingTts(ms_per_word=50)
        self.audio = RecordingAudio(self.clock)  # type: ignore[arg-type]
        self.memory = FakeMemory()
        self.browser = VisionBrowser([F0])
        self.browser.now_ms = lambda: self.clock.now_ms
        self.log = make_log(tmp_path / "logs")
        self.turns_path = tmp_path / "turns.jsonl"
        self.config = SessionConfig(data_dir=tmp_path, turns_path=self.turns_path)
        self.session = Session(
            self.machine,
            self.llm,
            self.stt,
            self.tts,
            self.audio,
            self.frames,
            self.memory,
            clock=self.clock,
            config=self.config,
            session_log=self.log,
            **{T.KIT_KWARG: live_kit(self.browser)},
        )
        self.server = HudServer(self.session, frame_sink=self.sink, port=0)
        self.wire: list[Any] = []
        self.hello_replies: list[list[Any]] = []
        self.hellos = 0
        self._tap()
        #: every message handed to ``Session.apply_vision_select`` (A1.1: "reaches"), in order
        self.selects: list[str] = []
        apply = self.session.apply_vision_select

        def recording(message: Any) -> Any:
            self.selects.append(str(getattr(message.source, "value", message.source)))
            return apply(message)

        self.session.apply_vision_select = recording  # type: ignore[method-assign]
        #: per typed question (``UserText`` handed to ``Session.submit`` by ``HudServer.route``), how many frames the
        #: sink had accepted at that moment: the frames on the socket before the question. "The newest posted
        #: astra.frame" (A1.2) is the last of those; a frame posted after the question is not the question's.
        self.posted_at_question: list[int] = []
        submit = self.session.submit

        def stamping(event: Any) -> Any:
            if isinstance(event, E.UserText):
                self.posted_at_question.append(len(self.sink.accepted()))
            return submit(event)

        self.session.submit = stamping  # type: ignore[method-assign]
        self.log_records: list[dict[str, Any]] = []

    async def aclose(self) -> None:
        await super().aclose()
        self.log_records = read_records(self.log.path)

    # --- observation ------------------------------------------------------------------------------

    def rig(self) -> Any:
        return types.SimpleNamespace(llm=self.llm)

    def gen_body(self, turn: str) -> dict[str, Any]:
        return _wire(tsl.sent(self.rig(), f"{turn}-gen"))

    def states(self) -> list[AstraVisionState]:
        return [m for m in self.wire if isinstance(m, AstraVisionState)]

    def vision_frames(self, turn: str) -> list[AstraVisionFrame]:
        return [m for m in self.wire if isinstance(m, AstraVisionFrame) and m.turn_id == turn]

    def line(self, turn: str) -> dict[str, Any]:
        found = [line for line in self.turn_lines() if line.get("turn_id") == turn]
        assert len(found) == 1, f"one turns.jsonl line for {turn}; got {self.turn_lines()}"
        return found[0]

    def frame_calls(self) -> int:
        return sum(1 for c in self.browser.calls if c[0] == "frame")


def _field(line: dict[str, Any], key: str) -> Any:
    """A10's per-turn key, top level or under ``extras`` (the turn line's extras are flattened or nested)."""
    if key in line:
        return line[key]
    return (line.get("extras") or {}).get(key)


def current_parts(body: dict[str, Any]) -> list[dict[str, Any]]:
    content = last_user(body)["content"]
    return content if isinstance(content, list) else [{"type": "text", "text": content}]


def current_images(body: dict[str, Any]) -> list[str]:
    return [p["image_url"]["url"] for p in current_parts(body) if p.get("type") == "image_url"]


def url_bytes(url: str) -> bytes:
    assert url.startswith("data:image/jpeg;base64,"), url[:60]
    return base64.b64decode(url.split(",", 1)[1])


def page_url(hud_port: int, ws_url: str, source: str | None = None) -> str:
    q = f"source={source}&" if source else ""
    return f"http://127.0.0.1:{hud_port}/?{q}ws=" + urllib.parse.quote(ws_url, safe="")


@contextlib.asynccontextmanager
async def share_page(
    tmp_path: Path, script: Sequence[Commit], *, source: str | None = None
) -> AsyncIterator[tuple[ShareLive, Page]]:
    """A started ``ShareLive``, the page served from ``hud/`` with the init script installed before load, connected
    and greeted (the live state echo has arrived)."""
    assert CHROME is not None
    live = ShareLive(tmp_path, script)
    await live.start()
    hud, hud_port = serve_hud()
    chrome = Chrome(CHROME).__enter__()
    try:
        chrome._cmd("Page.enable")
        chrome._cmd("Page.addScriptToEvaluateOnNewDocument", source=INIT_JS)
        chrome._cmd(
            "Emulation.setDeviceMetricsOverride",
            width=1440,
            height=900,
            deviceScaleFactor=1,
            mobile=False,
        )
        page = Page(chrome)
        await page.navigate(page_url(hud_port, live.server.url, source))
        await page.connected()
        await page.until(
            f"!!document.querySelector({json.dumps(COMBO)})",
            "the combo is rendered (the hello reply's astra.vision.state arrived)",
        )
        yield live, page
    finally:
        with contextlib.suppress(Exception):
            chrome.__exit__()
        hud.shutdown()
        await live.aclose()


async def pane(page: Page) -> dict[str, Any]:
    got = await page.evaluate(PANE_JS)
    assert isinstance(got, dict), f"the pane probe threw: {got!r}"
    return got


async def click(page: Page, selector: str) -> None:
    ok = await asyncio.to_thread(_click, page.chrome, selector)
    assert ok, f"{selector} is not on the page"


async def share(live: ShareLive, page: Page) -> None:
    """A real click on Share; returns once the capture runs and one select has reached the session."""
    before = len(live.selects)
    await click(page, "#share-button")
    await page.until(CAPTURING_JS, "the Share click starts a capture")
    await live.until(
        lambda: len(live.selects) > before,
        "the Share click's astra.vision.select never reached Session.apply_vision_select",
        timeout_s=PAGE_TIMEOUT_S,
    )
    await asyncio.sleep(1.0)  # a generous settle, so a second select for one Share is seen too


def _centre_rgb(jpeg: bytes) -> list[int]:
    import io

    from PIL import Image

    with Image.open(io.BytesIO(jpeg)) as im:
        rgb = im.convert("RGB")
        return list(rgb.getpixel((rgb.width // 2, rgb.height // 2)))  # type: ignore[arg-type]


async def first_frame(live: ShareLive, *, painted: bool = True) -> None:
    """Wait for a posted frame. ``painted``: the newest posted frame shows the stand-in picture (magenta centre).
    The capture's first grabbed video frame can be black, before the canvas reached the video (seen once in A1.2's
    #preview, 2026-10-30); a question asked then is about a black screen, which is not this row's premise."""
    await live.until(
        lambda: (
            bool(live.sink.accepted())
            and (not painted or _magenta(_centre_rgb(live.sink.accepted()[-1][0])))
        ),
        "no astra.frame showing the shared picture reached BrowserFrameSource",
        timeout_s=PAGE_TIMEOUT_S,
    )


async def ask_typed(live: ShareLive, page: Page, question: str, turn: str) -> None:
    await page.until(
        "!(document.querySelector('[data-ask-input]') || {disabled: true}).disabled",
        "the ask box takes a question",
    )
    outcome = await page.evaluate(type_and_submit_js(question))
    assert outcome == "submitted", f"the page could not be typed into: {outcome!r}"
    await turn_line(live, turn)


async def turn_line(live: ShareLive, turn: str) -> None:
    await live.until(
        lambda: any(line.get("turn_id") == turn for line in live.turn_lines()),
        f"{turn} never wrote its turns.jsonl line",
        timeout_s=TURN_TIMEOUT_S,
    )
    await live.idle()


def notes(records: list[dict[str, Any]], what: str) -> list[dict[str, Any]]:
    out = []
    for r in of_kind(records, "note"):
        data = r.get("data") if isinstance(r.get("data"), dict) else r
        if data.get("what") == what:
            out.append(data)
    return out


# --------------------------------------------------------------------------------------------- rows


@requires_chrome
async def test_share_click_selects_shared_on_the_real_session(tmp_path: Path) -> None:
    """A1.1 (T-0177 A3, C2, H-T177-1): one real Share click -> exactly one ``select shared`` reaches
    ``Session.apply_vision_select``; ``session.vision_source == "shared"``; the page receives one (non-replay)
    ``astra.vision.state {source: "shared"}`` and the server published exactly one; one note
    ``vision.select {from: "browser", to: "shared"}``; the combo reads "Shared screen"."""
    async with share_page(tmp_path, [answer(ANSWER)]) as (live, page):
        assert live.session.vision_source == "browser", (
            "premise: a live session starts on the browser"
        )
        assert live.selects == [], f"premise: nothing selected before the click; {live.selects}"
        published_before = len(live.states())
        before = await pane(page)
        assert before["combo_value"] == "browser", f"premise: the combo echoes browser; {before}"
        page_states_before = len(before["rec"]["states"])

        await share(live, page)
        await page.until(
            f"(document.querySelector({json.dumps(COMBO)}) || {{}}).value === 'shared'",
            "the combo follows the shared echo",
        )
        after = await pane(page)
        published = live.states()[published_before:]
        vision_source = live.session.vision_source

    assert after["rec"]["calls"] == 1, (
        f"premise: the Share went through getDisplayMedia once; {after['rec']}"
    )
    assert live.selects == ["shared"], (
        f"A1.1: exactly one astra.vision.select {{source: shared}} reaches Session.apply_vision_select; "
        f"it saw {live.selects} (page sent {after['rec']['selects']})"
    )
    assert vision_source == "shared", (
        f"A1.1: session.vision_source is 'shared'; it is {vision_source!r}"
    )
    assert [(str(m.source), m.replay) for m in published] == [("shared", False)], (
        f"A1.1: one astra.vision.state {{source: shared}} published to every page; "
        f"published {[(m.source, m.replay) for m in published]}"
    )
    received = after["rec"]["states"][page_states_before:]
    assert received == [{"source": "shared", "replay": False}], (
        f"A1.1: the page receives one astra.vision.state {{source: shared}}; it received {received}"
    )
    changes = notes(live.log_records, "vision.select")
    assert [(c.get("from"), c.get("to")) for c in changes] == [("browser", "shared")], (
        f"A1.1: one note vision.select {{from: browser, to: shared}}; the log has {changes}"
    )
    assert after["combo_text"] == "Shared screen", (
        f"A1.1: the combo reads 'Shared screen'; it reads {after['combo_text']!r}"
    )


@requires_chrome
async def test_next_turn_step1_is_the_posted_frame_and_labelled(tmp_path: Path) -> None:
    """A1.2 (T-0177 A6, A7, A10, B7, C3, C4): after a Share and at least one posted ``astra.frame``, a typed question
    opens a turn whose line says ``vision_source: "shared"``, ``frame_source: "screen"``; whose step-1 body's current
    image is the bytes of the newest posted frame, followed by ``SHARED_LABEL_SCREEN``, with no agent-browser byte;
    whose published ``astra.vision.frame`` (step 1) is ``source: "shared"`` with those bytes; and ``#preview`` shows
    it with the caption "Shared screen · WxH · turn t1 step 1"."""
    async with share_page(tmp_path, [answer(ANSWER)]) as (live, page):
        await share(live, page)
        await first_frame(live)
        await ask_typed(live, page, QUESTION, "t1")
        await page.until(
            "(document.querySelector('main > .pane.thumb [data-capture-caption]') || {textContent: ''})"
            ".textContent.includes('turn t1 step 1')",
            "the caption names turn t1 step 1",
        )
        shown = await pane(page)
        body = live.gen_body("t1")
        assert len(live.posted_at_question) == 1, live.posted_at_question
        n_posted = live.posted_at_question[0]
        accepted = live.sink.accepted()
        bodies = json.dumps(live.llm.bodies)
        frames = live.vision_frames("t1")
        line = live.line("t1")
        frame_calls = live.frame_calls()

    assert n_posted >= 1, (
        "premise: at least one frame from the capture reached the server before step 1"
    )
    newest, stored = accepted[n_posted - 1]
    assert (_field(line, "vision_source"), _field(line, "frame_source")) == ("shared", "screen"), (
        f"A1.2 (T-0177 A10): the turn line says vision_source shared, frame_source screen; it reads {line}"
    )
    images = current_images(body)
    assert len(images) == 1, f"A1.2: step 1 carries one current image; it carries {len(images)}"
    sent = url_bytes(images[0])
    assert sent == newest, (
        "A1.2: the step-1 image is not byte-equal to the newest posted astra.frame's jpeg_b64 (Q-2: the ingress "
        "re-encoded, or the snapshot was older than the newest frame)"
    )
    assert sent == stored.jpeg, (
        "A1.2: the step-1 image is not the bytes BrowserFrameSource stored for the newest posted frame"
    )
    parts = current_parts(body)
    at = next(i for i, p in enumerate(parts) if p.get("type") == "image_url")
    assert parts[at + 1 : at + 2] == [{"type": "text", "text": SHARED_LABEL_SCREEN}], (
        f"A1.2 (T-0177 A7): SHARED_LABEL_SCREEN right after the shared image; the parts after it are "
        f"{parts[at + 1 : at + 3]}"
    )
    assert b64(F0.jpeg) not in bodies, "A1.2: the agent browser's frame reached a request body"
    assert frame_calls == 0, (
        f"A1.2 (T-0177 A6): the shared step 1 asked the agent browser for {frame_calls} frame(s)"
    )
    step1 = [m for m in frames if m.step == 1]
    assert step1, (
        f"A1.2 (T-0177 B7): step 1 publishes one astra.vision.frame; t1 published {frames}"
    )
    assert (step1[-1].source, step1[-1].blind) == ("shared", False), step1[-1]
    assert [base64.b64decode(i.jpeg_b64 or "") for i in step1[-1].images] == [sent], (
        "A1.2 (T-0177 B7): the published frame's bytes are exactly the step-1 body's"
    )
    assert shown["preview_src"] == data_url(sent), (
        "A1.2 (T-0177 C3): #preview shows the session's published bytes, verbatim"
    )
    assert _magenta(shown["preview_rgb"]), (
        f"A1.2 (T-0177 C3): #preview renders the shared picture (magenta centre); it reads {shown['preview_rgb']}"
    )
    posted_message = [m for m, f in live.sink.posted if f is not None][n_posted - 1]
    expect_caption = (
        f"Shared screen · {posted_message.width}x{posted_message.height} · turn t1 step 1"
    )
    assert shown["caption"] == expect_caption, (
        f"A1.2 (T-0177 C4): the caption is {expect_caption!r}; it reads {shown['caption']!r}"
    )


@requires_chrome
async def test_stop_sharing_returns_to_browser(tmp_path: Path) -> None:
    """A1.3 (T-0177 C2, A6): after a shared turn, "Stop sharing" (a real click on ``#mute-button``) sends exactly one
    ``select browser``; the next typed turn is a browser turn: ``vision_source: "browser"``, its step-1 image is the
    agent browser's frame, and no posted shared byte is in its body."""
    async with share_page(tmp_path, [answer(ANSWER)]) as (live, page):
        await share(live, page)
        await first_frame(live)
        await ask_typed(live, page, QUESTION, "t1")
        assert live.selects == ["shared"], f"premise: one select shared; {live.selects}"
        before = len(live.selects)

        await click(page, "#mute-button")
        await page.until(
            "document.getElementById('mute-button').disabled", "Stop sharing stops the capture"
        )
        await live.until(
            lambda: len(live.selects) > before,
            "Stop sharing's select never reached the session",
            timeout_s=PAGE_TIMEOUT_S,
        )
        await asyncio.sleep(1.0)
        stop_selects = live.selects[before:]
        source_after = live.session.vision_source
        await page.until(
            f"(document.querySelector({json.dumps(COMBO)}) || {{}}).value === 'browser'",
            "the combo follows the browser echo",
        )
        await ask_typed(live, page, "and now what do you see", "t2")
        body2 = json.dumps(live.gen_body("t2"))
        images2 = current_images(live.gen_body("t2"))
        line2 = live.line("t2")
        shared = [b64(raw) for raw, _ in live.sink.accepted()]
        t1_images = current_images(live.gen_body("t1"))

    assert t1_images and url_bytes(t1_images[0]) != F0.jpeg, "premise: t1 was a shared turn"
    assert stop_selects == ["browser"], (
        f"A1.3: Stop sharing sends exactly one select browser; got {stop_selects}"
    )
    assert source_after == "browser", (
        f"A1.3: the session is back on the browser; it reads {source_after!r}"
    )
    assert _field(line2, "vision_source") == "browser", (
        f"A1.3: t2 is a browser turn; its line reads {line2}"
    )
    assert images2 == [builder_url(F0)], "A1.3: t2's step-1 image is the agent browser's frame"
    assert shared, "premise: frames were posted"
    # Ruling R-A1 (T-0208 Status, 2026-10-30): "one source" means the current turn's images; history images of
    # the other source are kept by design (T-0177 H-T177-4, docs/02 §3 two-turn aging). So "no shared bytes" is
    # read over t2's current (final user) message.
    current2 = json.dumps(last_user(json.loads(body2)))
    assert not any(s in current2 for s in shared), (
        "A1.3 (R-A1): t2's current (final user) message carries a posted shared frame"
    )
    assert SHARED_LABEL_SCREEN not in current2, (
        "A1.3 (R-A1): t2's current (final user) message carries the shared label"
    )
    assert SHARED_LABEL_SCREEN not in body2, "A1.3: t2's body carries the shared label"
    # H-T177-4 pinned (R-A1): the held t1 shared image is present in t2's body, and only in history, i.e. in a
    # message before the current user message; and it is exactly t1's step-1 image.
    messages = json.loads(body2)["messages"]
    current_at = max(k for k, m in enumerate(messages) if m.get("role") == "user")
    held = b64(url_bytes(t1_images[0]))
    where = [k for k, m in enumerate(messages) if held in json.dumps(m)]
    assert where and all(k < current_at for k in where), (
        f"A1.3 (R-A1, H-T177-4): t1's shared image is held in t2's history only; it sits in message(s) {where}, "
        f"the current user message is {current_at} of {[m.get('role') for m in messages]}"
    )
    stray = [
        k
        for k, m in enumerate(messages)
        if k not in where and any(s in json.dumps(m) for s in shared)
    ]
    assert not stray, (
        f"A1.3 (R-A1): posted shared bytes other than t1's held image appear in t2's message(s) {stray}"
    )


@requires_chrome
async def test_share_and_ask_at_once_never_shows_the_browser(tmp_path: Path) -> None:
    """A1.4 (adversarial, the likely race; T-0177 A6 "no fallback"). The question goes out in the same JS task as the
    Share click's ``select shared`` (the page's own send call sends ``astra.text`` right after it), before the page
    has sent any frame of the new capture. Step 1 carries the shared frame or nothing (blind: the existing no-image
    wording), never the agent browser's frame. The row prints which happened with the 1-minute load (a reading for
    H-T208-3); it asserts neither."""
    async with share_page(tmp_path, [answer(ANSWER)]) as (live, page):
        await page.evaluate(f"window.__t0208_race = {json.dumps(QUESTION)}; true")
        await click(page, "#share-button")
        await page.until(CAPTURING_JS, "the Share click starts a capture")
        await live.until(
            lambda: bool(live.selects),
            "the Share click's astra.vision.select never reached Session.apply_vision_select",
            timeout_s=PAGE_TIMEOUT_S,
        )
        await turn_line(live, "t1")
        state = await pane(page)
        body = live.gen_body("t1")
        bodies = json.dumps(live.llm.bodies)
        line = live.line("t1")
        accepted = live.sink.accepted()
        frame_calls = live.frame_calls()

    race = state["rec"]["race"]
    assert race is not None, (
        f"premise: the page sent the question with its select shared; {state['rec']}"
    )
    assert race["frames_on_wire_before"] == 0 and live.posted_at_question == [0], (
        f"premise (A1.4): no frame of the new capture reached the server before the question; page {race}, "
        f"frames accepted at the question {live.posted_at_question}"
    )
    assert live.selects == ["shared"], (
        f"premise: the Share's one select reached the session; {live.selects}"
    )
    assert _field(line, "vision_source") == "shared", (
        f"premise: the turn froze the shared choice; {line}"
    )
    assert b64(F0.jpeg) not in bodies, (
        "A1.4 (T-0177 A6): the agent browser's frame reached a request body"
    )
    assert frame_calls == 0, (
        f"A1.4 (T-0177 A6): step 1 asked the agent browser for {frame_calls} frame(s)"
    )
    images = current_images(body)
    if images:
        sent = url_bytes(images[0])
        assert len(images) == 1 and any(sent == raw for raw, _ in accepted), (
            "A1.4: step 1's image is neither the agent browser's nor a posted shared frame"
        )
        outcome = "shared"
    else:
        assert NO_IMAGE_STATEMENT in json.dumps(body), (
            "A1.4: a blind step 1 carries the existing no-image wording"
        )
        assert _field(line, "frame_source") == "none", (
            f"A1.4: a blind shared turn logs frame_source none; {line}"
        )
        outcome = "blind"
    load1 = os.getloadavg()[0]
    print(
        f"T-0208 A1.4: {outcome} (load1={load1:.2f}, frames held back until after the question="
        f"{race['frames_held']}, frames accepted by turn end={len(accepted)})"
    )


@requires_chrome
async def test_url_capture_never_switches_the_real_session(tmp_path: Path) -> None:
    """Negative (T-0177 C2): a page loaded with ``?source=synthetic`` captures from load and posts frames, but never
    selects shared: no select reaches the session, the session stays on the browser, and a typed turn is a browser
    turn whose body carries none of the posted bytes."""
    async with share_page(tmp_path, [answer(ANSWER)], source="synthetic") as (live, page):
        await first_frame(live, painted=False)
        await asyncio.sleep(1.5)  # more capture ticks: a late auto-switch would show here
        selects = list(live.selects)
        source = live.session.vision_source
        await ask_typed(live, page, QUESTION, "t1")
        body = json.dumps(live.gen_body("t1"))
        images = current_images(live.gen_body("t1"))
        line = live.line("t1")
        posted = [b64(raw) for raw, _ in live.sink.accepted()]
        state = await pane(page)

    assert posted, "premise: the synthetic capture posted frames"
    assert selects == [] and state["rec"]["selects"] == [], (
        f"T-0177 C2: a ?source= capture never auto-switches; the session saw {selects}, the page sent "
        f"{state['rec']['selects']}"
    )
    assert source == "browser", f"T-0177 C2: the session stays on the browser; it reads {source!r}"
    assert _field(line, "vision_source") == "browser", line
    assert images == [builder_url(F0)], "the browser turn's step 1 is the agent browser's frame"
    assert not any(p in body for p in posted), (
        "a posted synthetic frame reached a browser turn's body"
    )
