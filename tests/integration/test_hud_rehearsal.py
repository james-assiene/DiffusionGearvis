"""T-0036: the real ``hud/`` page in headless Chrome against the real ``HudServer`` and the real
``Session`` — the one module where both halves of the wire are production code.

Everything else that touches the HUD tests one half: ``tests/integration/test_hud_headless.py``
drives the page against ``tests/fakes/hud_server.HudStub``, ``tests/unit/test_hud_egress.py`` and
friends drive the server against a Python client, and the only crossing module before this one
(``tests/integration/test_hud_live.py``) is ``gpu``-gated. This module is the 2026-09-14 rehearsal
landed as a test: no GPU, no audio device, no model — the *engine* end is the fakes and recorders
of ``tests/unit/_hud_rig.py`` (``RecordingMachine`` over the production ``TurnMachine``,
``RecordingLlm`` replaying a scripted commit list, ``FakeStt``, ``RecordingTts``,
``RecordingAudio``, ``FakeMemory``), and everything between the session's publisher seam and the
pixels in the browser is the real thing.

**Regime** (docs/tasks/T-0036.md): fakes and recorders from ``tests/unit/_hud_rig.py``, real
``Session`` on ``SessionClock`` (wall time, *not* the virtual clock — the browser has a clock of
its own and the two must agree), real ``HudServer(port=0)``, the page served over loopback http
with ``?source=canvas`` (the synthetic ``canvas.captureStream()`` provider, so no camera and no
screen grant) and ``?ws=<server url>``, headless Chrome 147, ``tts ms_per_word`` as each row sets
it, n=1 per row. **No number here is a latency claim**: the rows wait on wall-clock deadlines
because a browser cannot be single-stepped, and every deadline is a generous ceiling.

**Where the assertions come from.** Each row asserts from the DOM (the attribute contract pinned
in ``tests/integration/test_hud_headless.py``'s docstring: ``[data-turn-state]``,
``[data-transcript]``, ``[data-utterance-id]``, ``[data-unplayed]``, ``[data-ui-surface]``,
``[data-frames-sent]``) or from the wire (what ``HudServer.publish``/``hello_reply`` fanned out,
and what the ``RecordingMachine`` was fed). Nothing asserts a private attribute of ``local_astra``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import typing
import urllib.parse
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.hud_server import RECONNECT_GRACE_MS, HudServer
from local_astra.conductor.session import Session, SessionClock, SessionConfig
from local_astra.contracts import events as E
from local_astra.contracts.codec import encode
from local_astra.contracts.llm import Commit
from local_astra.contracts.perception import Source
from local_astra.contracts.protocol import AstraMessage, AstraSpoken, AstraTurnState
from local_astra.contracts.turn import TurnState
from local_astra.perception import policy
from local_astra.perception.browser_source import BrowserFrameSource
from tests.fakes import FakeMemory, FakeStt
from tests.integration.test_hud_headless import CHROME, Chrome, requires_chrome, serve_hud
from tests.unit._hud_rig import (
    POLICY,
    RecordingAudio,
    RecordingLlm,
    RecordingMachine,
    RecordingTts,
)

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
QUIZ = REPO / "tests" / "fixtures" / "a2ui" / "quiz_example.jsonl"

#: The label on the quiz fixture's primary button and the action it carries, read out of
#: ``tests/fixtures/a2ui/quiz_example.jsonl`` by hand (component ``check_btn``) rather than from
#: the page: row (d)'s point is that the page turns *this* button into *this* wire message.
BUTTON_LABEL = "Check answers"
ACTION_NAME = "check_quiz"

#: T-0164 group B (docs/tasks/T-0164.md B2/B4): the surface count is re-keyed by name from the
#: vendored ``<a2ui-surface>`` to ``[data-a2ui-root]`` once T-0164's **Status:** records "group B:
#: tests <a>..<b>, activated <sha>" (``ASTRA_RUN_PENDING=1`` flips it now); before that the rows are
#: today's. The B4 transcript row is ``pending(task="T-0164")`` and clicks through the light DOM.
T0164_TASK = REPO / "docs" / "tasks" / "T-0164.md"


def _t0164_b_activated() -> bool:
    import os
    import re

    if os.environ.get("ASTRA_RUN_PENDING") == "1":
        return True
    text = T0164_TASK.read_text(encoding="utf-8") if T0164_TASK.is_file() else ""
    status = re.search(r"^\*\*Status:\*\*\s*(.*)$", text, re.M)
    line = status.group(1) if status else ""
    return re.search(r"\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b", line, re.I) is not None


T0164_B = _t0164_b_activated()
ROOT_SEL = "[data-a2ui-root]" if T0164_B else "a2ui-surface"

#: T-0164 B4: a canvas whose Button the model wrote with a bare ``text`` — UiGate's RB repair makes
#: it the gate's ``__gate.clicked`` with ``context.label`` = the model's string.
GATE_LABEL = "Show me more"
GATE_BODY = (
    "\n".join(
        json.dumps(m)
        for m in (
            {
                "version": "v0.9.1",
                "createSurface": {
                    "surfaceId": "more",
                    "catalogId": "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json",
                },
            },
            {
                "version": "v0.9.1",
                "updateComponents": {
                    "surfaceId": "more",
                    "components": [
                        {"id": "root", "component": "Column", "children": ["t", "go"]},
                        {"id": "t", "component": "Text", "text": "Want more?"},
                        {"id": "go", "component": "Button", "text": GATE_LABEL},
                    ],
                },
            },
        )
    )
    + "\n"
)


#: The transcript of a gate click as it is today, measured on main 9b6ee6e0 with the vendored
#: renderer (2026-10-27, this rehearsal, real HudServer + Session, fake LLM):
#:   '[ui] __gate.clicked {"label":"Show me more","a2ui":{"sourceComponentId":"go",
#:   "timestamp":"2026-10-27T02:44:44.584Z"}}'
#: i.e. the page's context, compact JSON, key order label, a2ui; sourceComponentId, timestamp. The
#: unit layer's ``[ui] __gate.clicked {"label":"No"}`` has no ``a2ui`` because it builds the event
#: by hand; the real page always adds it. Only the timestamp varies, and it must be the action's own.
def gate_transcript(timestamp: str) -> str:
    return (
        '[ui] __gate.clicked {"label":"Show me more","a2ui":{"sourceComponentId":"go",'
        f'"timestamp":"{timestamp}"}}}}'
    )


#: Row (f): a thought whose text exists nowhere else, so "the thought never leaves the conductor"
#: is a substring search over the page, the wire and ``turns.jsonl`` rather than a statement about
#: a field name (CLAUDE.md: never store assistant thinking in history).
SECRET_THOUGHT = "SECRETTHOUGHT the user is confused about eighths"

#: T-0180 B9 (ruling R-1): the one message type allowed to carry thought text, in its ``text`` only.
THOUGHT_TYPE = "astra.thought"


def thought_leaks(messages: Sequence[Any], marker: str) -> list[str]:
    """T-0180 B9: the encoded ``messages`` that carry ``marker`` anywhere but an ``astra.thought``'s
    ``text`` (another type, or another field of an ``astra.thought``)."""
    out = []
    for m in messages:
        raw = encode(m)
        if marker not in raw:
            continue
        if getattr(m, "type", None) != THOUGHT_TYPE:
            out.append(raw)
            continue
        rest = {k: v for k, v in json.loads(raw).items() if k != "text"}
        if marker in json.dumps(rest):
            out.append(raw)
    return out


def thought_texts(messages: Sequence[Any]) -> list[str]:
    """T-0180 B9 anti-vacuity: the ``text`` of every ``astra.thought`` in ``messages``."""
    return [m.text for m in messages if getattr(m, "type", None) == THOUGHT_TYPE]


#: T-0180 B9, the DOM half (group C): the page's HTML with every ``[data-model-thoughts]`` cut out of a
#: clone, and the text inside those disclosures. The one DOM region allowed to hold thought text is
#: ``[data-model-thoughts]`` (ruling R-1; the denoising view's per-position pieces are the other, and these
#: rows' fake LLM sends no canvas tap).
SPLIT_THOUGHTS_JS = """
(() => {
  const live = [...document.querySelectorAll('[data-model-thoughts]')];
  const clone = document.documentElement.cloneNode(true);
  for (const d of clone.querySelectorAll('[data-model-thoughts]')) d.remove();
  return {outside: clone.outerHTML, inside: live.map((d) => d.textContent).join('\\n'),
          count: live.length};
})()
"""


async def thought_dom(page: Any) -> tuple[str, str]:
    """(the DOM outside ``[data-model-thoughts]``, the text inside it), read in one evaluation."""
    got = await page.evaluate(SPLIT_THOUGHTS_JS)
    assert isinstance(got, dict) and "outside" in got, f"the split-DOM probe threw: {got!r}"
    return str(got["outside"]), str(got["inside"])


#: Generous wall-clock ceilings; a row fails by timing out *with* what it was waiting for.
PAGE_TIMEOUT_S = 20.0
TURN_TIMEOUT_S = 30.0


def answer(text: str, canvas_idx: int = 0) -> Commit:
    return Commit(request_id="?", canvas_idx=canvas_idx, segment="answer", text=text)


def thought(text: str, canvas_idx: int = 0) -> Commit:
    return Commit(request_id="?", canvas_idx=canvas_idx, segment="thought", text=text)


def ui_commit(text: str | None = None, canvas_idx: int = 1) -> Commit:
    return Commit(
        request_id="?",
        canvas_idx=canvas_idx,
        segment="ui",
        text=QUIZ.read_text() if text is None else text,
    )


class CountingSink:
    """The server's ``frame_sink``: the real ``BrowserFrameSource`` plus a tally.

    ``BrowserFrameSource`` exposes no "how many frames did you take" accessor and row (b) may not
    read its private ring. This wrapper is exactly the ``ingest(AstraFrame) -> Frame | None``
    protocol ``HudServer`` documents; the frames land in the very source the session snapshots.
    """

    def __init__(self, inner: BrowserFrameSource) -> None:
        self.inner = inner
        self.accepted: list[Any] = []
        self.refused = 0

    def ingest(self, message: Any) -> Any:
        frame = self.inner.ingest(message)
        if frame is None:
            self.refused += 1
        else:
            self.accepted.append(frame)
        return frame


class Live:
    """Real ``Session`` + real ``HudServer`` over the fakes, with the wire tapped.

    The tap: ``HudServer.publish`` is the one place every server→client message the session
    produces passes through, and ``HudServer.hello_reply`` is the one place the handshake's live
    messages come from (three since T-0043 criterion 18 put ``astra.log.state`` third; every row
    here selects by type, so the count is not pinned anywhere in this module). Wrapping both gives
    a faithful copy of everything the browser could have
    been sent without opening a second WebSocket — which would hold the connection count above
    zero and make row (f)'s ``ClientDisconnected`` unobservable.
    """

    def __init__(self, tmp_path: Path, script: Sequence[Commit], *, ms_per_word: int = 400) -> None:
        self.clock = SessionClock()
        self.machine = RecordingMachine(POLICY)
        self.llm = RecordingLlm(list(script))
        self.llm.clock = self.clock  # type: ignore[assignment]
        self.stt = FakeStt()
        self.tts = RecordingTts(ms_per_word=ms_per_word)
        self.audio = RecordingAudio(self.clock)  # type: ignore[arg-type]
        self.frames = BrowserFrameSource(now_ms=lambda: self.clock.now_ms)
        self.sink = CountingSink(self.frames)
        self.memory = FakeMemory()
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
        )
        self.server = HudServer(self.session, frame_sink=self.sink, port=0)
        self.wire: list[Any] = []
        #: one entry per handshake: exactly what ``hello_reply`` gave that connection
        self.hello_replies: list[list[Any]] = []
        self.hellos = 0
        self._tap()

    def _tap(self) -> None:
        publish = self.server.publish
        hello_reply = self.server.hello_reply

        def tapped_publish(message: AstraMessage) -> None:
            self.wire.append(message)
            publish(message)

        def tapped_hello_reply() -> Iterator[AstraMessage]:
            self.hellos += 1
            out = list(hello_reply())
            self.hello_replies.append(out)
            self.wire.extend(out)
            return iter(out)

        # The session holds the bound ``server.publish`` from the server's constructor, so the
        # publisher is re-bound explicitly; the connection reads ``hello_reply`` off the server on
        # every handshake, so the instance attribute is enough there.
        self.server.publish = tapped_publish  # type: ignore[method-assign]
        self.server.hello_reply = tapped_hello_reply  # type: ignore[method-assign]
        self.session.set_publisher(tapped_publish)

    # --- lifecycle ---------------------------------------------------------------------------

    async def start(self) -> None:
        await self.session.start()
        await self.server.start()

    async def aclose(self) -> None:
        await self.server.aclose()
        await self.session.aclose()

    # --- driving -----------------------------------------------------------------------------

    @property
    def now(self) -> int:
        return int(self.clock.now_ms)

    async def ask(self, text: str, *, word_ms: int = 60) -> None:
        """One spoken question, as the STT events ``Session._pump_stt`` would have delivered."""
        at = self.now
        self.session.submit(E.UserSpeechOnset(t_ms=at, onset_ms=at))
        await asyncio.sleep(0.05)
        for word in text.split():
            await asyncio.sleep(word_ms / 1000)
            now = self.now
            self.session.submit(E.SttWord(t_ms=now, text=word, start_ms=now - word_ms, end_ms=now))
        await asyncio.sleep(0.1)
        now = self.now
        self.session.submit(E.SttEndOfTurn(t_ms=now, t_eot_ms=now - 20, p_end=0.9))
        self.session.submit(E.SttFlushed(t_ms=now, t_eot_ms=now - 20))
        await asyncio.sleep((POLICY.eot_settle_ms + 80) / 1000)

    def asking(self, text: str) -> asyncio.Task[None]:
        """``ask`` in the background: the browser has to be watched while the turn runs."""
        return asyncio.ensure_future(self.ask(text))

    async def until(
        self, predicate: Callable[[], bool], what: str, timeout_s: float = TURN_TIMEOUT_S
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.02)
        raise AssertionError(
            f"{what}: still false after {timeout_s:.0f} s of wall clock "
            f"(state={self.session.state.state}, counts={dict(self.server.counts)})"
        )

    async def idle(self, timeout_s: float = TURN_TIMEOUT_S) -> None:
        await self.until(
            lambda: self.session.state.state is TurnState.IDLE,
            "the turn never came back to IDLE",
            timeout_s,
        )

    # --- observation -------------------------------------------------------------------------

    def spoken(self) -> list[Any]:
        return [m for m in self.wire if isinstance(m, AstraSpoken)]

    def turn_lines(self) -> list[dict[str, Any]]:
        if not self.turns_path.exists():
            return []
        return [
            json.loads(line) for line in self.turns_path.read_text().splitlines() if line.strip()
        ]


# --- the browser side ---------------------------------------------------------------------------

#: Walk the page *and* every open shadow root: the A2UI renderer lives inside custom elements, so
#: ``document.querySelectorAll`` alone sees neither ``<a2ui-surface>`` nor the button's label.
DEEP_JS = """
  const deepAll = () => {
    const out = [];
    const walk = (root) => {
      if (!root || !root.querySelectorAll) return;
      for (const el of root.querySelectorAll('*')) {
        out.push(el);
        if (el.shadowRoot) walk(el.shadowRoot);
      }
    };
    walk(document);
    return out;
  };
"""

PROBE_JS = (
    """
(() => {
  const attr = (sel, name) => {
    const el = document.querySelector(sel);
    return el ? el.getAttribute(name) : null;
  };
  const text = (sel) => {
    const el = document.querySelector(sel);
    return el ? el.textContent : null;
  };
"""
    + DEEP_JS
    + """
  return {
    status: text('#conn-status'),
    chip: attr('[data-turn-state]', 'data-turn-state'),
    transcript: text('[data-transcript]'),
    frames_sent: attr('[data-frames-sent]', 'data-frames-sent'),
    frames_dropped: attr('[data-frames-dropped]', 'data-frames-dropped'),
    capture_size: attr('[data-capture-size]', 'data-capture-size'),
    // T-0154 B4/B8 (G2, implementation note 7): several cards can be in the thread, and only the
    // newest speaks; read the last card wrapper that is not inert (was: the first [data-ui-surface]).
    surface: (() => {
      const live = [...document.querySelectorAll('[data-ui-surface]')]
        .filter((e) => !e.inert && !e.closest('[inert]'));
      return live.length ? live[live.length - 1].getAttribute('data-ui-surface') : null;
    })(),
    errors: text('#error-strip'),
    body: document.body.innerText || '',
    spoken: [...document.querySelectorAll('[data-utterance-id]')].map((e) => ({
      utterance_id: e.getAttribute('data-utterance-id'),
      turn_id: e.getAttribute('data-turn-id'),
      text: e.textContent.trim(),
      unplayed: [...e.querySelectorAll('[data-unplayed]')].map((t) => ({
        text: t.textContent.trim(),
        decoration: getComputedStyle(t).textDecorationLine,
      })),
    })),
    a2ui_count: deepAll().filter((e) => e.matches && e.matches(__ROOT_SEL__)).length,
    deep_text: deepAll()
      .filter((e) => e.children.length === 0)
      .map((e) => (e.textContent || '').trim())
      .filter((t) => t)
      .join(' | '),
  };
})()
"""
).replace("__ROOT_SEL__", json.dumps(ROOT_SEL))


def light_click_js(label: str) -> str:
    """T-0164 B4: click the one ``<button>`` reading ``label`` inside the newest card's light-DOM
    root (``[data-ui-surface] > [data-a2ui-root]``), with no shadow piercing; or say why not."""
    return """(() => {
  const roots = [...document.querySelectorAll('[data-ui-surface] > [data-a2ui-root]')];
  if (!roots.length) return 'no light-DOM [data-a2ui-root] on the page';
  const hits = [...roots[roots.length - 1].querySelectorAll('button')]
    .filter((b) => (b.textContent || '').trim() === __WANT__);
  if (hits.length !== 1) return hits.length + ' buttons read ' + __WANT__;
  hits[0].click();
  return 'clicked';
})()""".replace("__WANT__", json.dumps(label))


def click_js(label: str) -> str:
    """Click the innermost actionable ancestor of the leaf whose text is exactly ``label``."""
    return (
        "(() => {"
        + DEEP_JS
        + """
  const want = __WANT__;
  const leaves = deepAll().filter(
    (e) => e.children.length === 0 && (e.textContent || '').trim() === want
  );
  if (!leaves.length) return 'no element reads ' + want;
  let node = leaves[0];
  let target = null;
  for (let i = 0; i < 8 && node; i++) {
    const tag = node.tagName ? node.tagName.toLowerCase() : '';
    const role = node.getAttribute ? node.getAttribute('role') : null;
    if (tag === 'button' || tag === 'a' || role === 'button') { target = node; break; }
    node = node.parentElement || (node.parentNode && node.parentNode.host) || null;
  }
  if (!target) return 'no button around ' + want;
  target.click();
  return 'clicked';
})()""".replace("__WANT__", json.dumps(label))
    )


class Page:
    """A headless tab driven from the event loop (the CDP calls block, so they go off-thread)."""

    def __init__(self, chrome: Chrome) -> None:
        self.chrome = chrome

    async def evaluate(self, expression: str) -> Any:
        return await asyncio.to_thread(self.chrome.evaluate, expression)

    async def probe(self) -> dict[str, Any]:
        value = await self.evaluate(PROBE_JS)
        assert isinstance(value, dict), f"the probe threw inside the page: {value!r}"
        return value

    async def navigate(self, url: str) -> None:
        await asyncio.to_thread(self.chrome.navigate, url)

    async def dom(self) -> str:
        return str(await self.evaluate("document.documentElement.outerHTML"))

    async def until(self, expression: str, what: str, timeout_s: float = PAGE_TIMEOUT_S) -> None:
        deadline = time.monotonic() + timeout_s
        last: Any = None
        while time.monotonic() < deadline:
            last = await self.evaluate(expression)
            # T-0225 R-1: Chrome.evaluate returns {"__exception__": ...} when the expression throws (e.g. before
            # the HUD has loaded); that dict is truthy but is never "condition met".
            if last and not (isinstance(last, dict) and "__exception__" in last):
                return
            await asyncio.sleep(0.1)
        raise AssertionError(
            f"the page never satisfied {what} within {timeout_s:.0f} s (last value {last!r})"
        )

    async def connected(self) -> None:
        await self.until(
            "(() => { const c = document.querySelector('#conn-status');"
            " return Boolean(c) && c.textContent === 'connected'; })()",
            "#conn-status === 'connected'",
        )


def page_url(hud_port: int, ws_url: str) -> str:
    return f"http://127.0.0.1:{hud_port}/?source=canvas&ws=" + urllib.parse.quote(ws_url, safe="")


@contextlib.asynccontextmanager
async def open_page(url: str) -> typing.AsyncIterator[Page]:
    """One headless tab on ``url``, connected, closed on the way out."""
    assert CHROME is not None
    chrome = Chrome(CHROME).__enter__()
    try:
        page = Page(chrome)
        await page.navigate(url)
        await page.connected()
        yield page
    finally:
        with contextlib.suppress(Exception):
            chrome.__exit__()


@contextlib.asynccontextmanager
async def live_page(
    tmp_path: Path, script: Sequence[Commit], *, ms_per_word: int = 400
) -> typing.AsyncIterator[tuple[Live, Page]]:
    """A started ``Live``, the page served over loopback http, and one connected tab."""
    live = Live(tmp_path, script, ms_per_word=ms_per_word)
    await live.start()
    hud, hud_port = serve_hud()
    try:
        async with open_page(page_url(hud_port, live.server.url)) as page:
            yield live, page
    finally:
        hud.shutdown()
        await live.aclose()


QUESTION = "what is on this screen"
FIRST = "Here is a quick check. "
SECOND = "Try the five questions carefully please. "


def turn_script() -> list[Commit]:
    """A thought, two spoken answers and the quiz canvas — the rehearsal's script."""
    return [thought("Fractions again."), answer(FIRST), answer(SECOND), ui_commit()]


# --- (a) the handshake ---------------------------------------------------------------------------


@requires_chrome
async def test_a_handshake_is_one_hello_the_servers_state_and_the_banner(tmp_path):
    """2(a): one ``astra.hello``, a chip that reads the *server's* state, the capture banner.

    The session is put — and held — in ``THINKING`` *before* the browser is started, so the chip
    cannot be right by defaulting: the page's own initial value is IDLE, and the only way it can
    read THINKING is the ``astra.turn.state`` that ``hello_reply`` sends. The banner is T-0008
    criterion 7 — a ``?source=canvas`` page says so on its face. "One hello" is the adversarial
    half (a page that greets on every retry would replay the handshake and, with it, the state).
    """
    live = Live(tmp_path, turn_script())
    await live.start()
    hud, hud_port = serve_hud()
    live.llm.hold()
    asking = live.asking(QUESTION)
    try:
        await live.until(
            lambda: live.session.state.state is TurnState.THINKING,
            "the session never reached THINKING before the page was opened",
        )
        async with open_page(page_url(hud_port, live.server.url)) as page:
            probe = await page.probe()
            assert len(live.hello_replies) == 1, live.hello_replies
            # The handshake's own messages, not the wire at large: the session keeps publishing
            # ``astra.turn.state`` while it is held in THINKING.
            handshake = [m for m in live.hello_replies[0] if isinstance(m, AstraTurnState)]

            assert live.hellos == 1, (
                "2(a): one connection is one astra.hello; the server answered "
                f"{live.hellos} of them (counts={dict(live.server.counts)})"
            )
            assert probe["status"] == "connected", probe
            # T-0037 ruling A3: the reply opens with the live state and, when a journal was
            # replayed, closes by repeating it so the chip is not left reading a state the server
            # has already left. Both say THINKING; the repeat is marked ``replay=True``, and the
            # live one is never marked (a page that could not tell them apart would treat a
            # replayed turn state as a new transition).
            assert handshake, (
                f"the hello reply carried no turn state at all: {live.hello_replies[0]}"
            )
            assert (handshake[0].state.name, handshake[0].replay) == (
                TurnState.THINKING.name,
                False,
            ), (
                "2(a): the hello reply opens with the session's current state, live; it opened "
                f"with {(handshake[0].state, handshake[0].replay)}"
            )
            assert all(
                (m.state.name, m.replay) == (TurnState.THINKING.name, True) for m in handshake[1:]
            ), (
                "2(a)/T-0037 A3: any further turn state in the reply is the closing repeat of the "
                "same state, marked as a replay; the reply carried "
                f"{[(m.state, m.replay) for m in handshake]}"
            )
            assert probe["chip"] == TurnState.THINKING.name == live.session.state.state.name, (
                "2(a): the chip shows the state the server sent in its hello reply (THINKING, "
                f"held); the page reads {probe['chip']!r}"
            )
            assert "SYNTHETIC CAPTURE" in (probe["body"] or ""), (
                "2(a): ?source=canvas must say so on the page (T-0008 criterion 7); the page "
                f"reads {probe['body']!r}"
            )
    finally:
        live.llm.release()
        with contextlib.suppress(Exception):
            await asking
        hud.shutdown()
        await live.aclose()


# --- (b) a frame crosses --------------------------------------------------------------------------


@requires_chrome
async def test_b_a_canvas_frame_is_routed_and_none_is_rejected(tmp_path):
    """2(b): the synthetic capture's frames reach the real ``BrowserFrameSource``.

    The page encodes the JPEG and the dHash; the server validates both. ``frames_rejected == 0``
    is the load-bearing half: a page whose phash spelling drifted would still send frames, the
    sink would refuse every one of them, and perception would be blind while the HUD says
    "frames sent: n".
    """
    async with live_page(tmp_path, [answer("Hello. ")]) as (live, page):
        await live.until(
            lambda: len(live.sink.accepted) >= 1,
            "no astra.frame ever reached the frame sink",
            timeout_s=PAGE_TIMEOUT_S,
        )
        await page.until(
            "Number(document.querySelector('[data-frames-sent]')"
            ".getAttribute('data-frames-sent')) >= 1",
            "[data-frames-sent] >= 1",
        )
        probe = await page.probe()
        counts = dict(live.server.counts)

        assert counts["frames_rejected"] == 0, (
            "2(b): every frame the page sends must pass the sink's validation; "
            f"{counts['frames_rejected']} were refused (accepted {len(live.sink.accepted)})"
        )
        assert live.sink.refused == 0, "the sink itself refused a frame"
        assert counts["routed"] >= 1, counts
        assert int(probe["frames_sent"]) >= len(live.sink.accepted), (
            "the page's own counter cannot be behind what the server took: page says "
            f"{probe['frames_sent']!r}, the sink took {len(live.sink.accepted)}"
        )
        frame = live.sink.accepted[0]
        assert frame.jpeg[:2] == b"\xff\xd8", (
            f"2(b): a real JPEG crossed the wire, not a placeholder: {frame.jpeg[:8]!r}"
        )
        assert frame.source is Source.SCREEN, frame.source
        # The page downscales before it encodes, and the size it must land on is the one
        # ``perception.policy`` prescribes for the capture the page reports — computed here from
        # the HUD's own [data-capture-size], not copied from the frame.
        capture_w, capture_h = (int(part) for part in str(probe["capture_size"]).split("x"))
        bucket = policy.bucket_for(Source.SCREEN, capture_w, capture_h)
        assert (frame.width, frame.height) == policy.target_size(capture_w, capture_h, bucket), (
            f"2(b): a {capture_w}x{capture_h} capture must arrive at "
            f"{policy.target_size(capture_w, capture_h, bucket)} (bucket {bucket.name}); it "
            f"arrived at {frame.width}x{frame.height}"
        )
        assert probe["errors"] in (None, ""), f"the page put an error up: {probe['errors']!r}"


# --- (c) one whole turn ---------------------------------------------------------------------------


@requires_chrome
async def test_c_one_turn_transcript_thinking_two_utterances_one_surface(tmp_path):
    """2(c): the transcript pane, the chip in THINKING, two utterances, exactly one surface.

    The LLM is *held* between end-of-turn and its first commit, so THINKING is observed on the
    page while the session is genuinely in it rather than inferred after the fact. The surface
    count is taken through the shadow roots: a canvas that renders twice (once per commit) or not
    at all both fail here, and a page that shows the placeholder instead fails on the count.
    """
    async with live_page(tmp_path, turn_script()) as (live, page):
        live.llm.hold()
        asking = live.asking(QUESTION)
        try:
            await page.until(
                "document.querySelector('[data-turn-state]')"
                ".getAttribute('data-turn-state') === 'THINKING'",
                "the chip in THINKING while the LLM is held",
            )
            held = await page.probe()
            assert live.session.state.state is TurnState.THINKING, (
                "the page said THINKING while the session said "
                f"{live.session.state.state}: the chip is ahead of the server"
            )
            assert QUESTION in " ".join(str(held["transcript"] or "").split()), (
                f"2(c): the transcript pane shows what the user said; it reads {held['transcript']!r}"
            )
        finally:
            live.llm.release()
            await asking

        await page.until(
            "document.querySelectorAll('[data-utterance-id]').length >= 2",
            "two spoken utterances in the transcript",
        )
        await page.until("!!document.querySelector('[data-ui-surface]')", "an A2UI surface")
        await live.idle()
        probe = await page.probe()

        spoken_wire = [m.text for m in live.spoken()]
        assert spoken_wire == [FIRST.strip(), SECOND.strip()], (
            f"2(c): the turn speaks the two scripted answers; the wire carried {spoken_wire}"
        )
        shown = [u["text"] for u in probe["spoken"]]
        assert len(shown) == 2, f"2(c): two utterances on the page; it shows {shown}"
        for said, element in zip(spoken_wire, shown, strict=True):
            assert " ".join(said.split()) in " ".join(element.split()), (
                f"the page must show what was spoken: {said!r} is not inside {element!r}"
            )
        assert live.tts.texts() == spoken_wire, (
            "2(c): what the page shows as spoken is what TTS was asked for; TTS got "
            f"{live.tts.texts()} and the wire carried {spoken_wire}"
        )
        assert probe["a2ui_count"] == 1, (
            "2(c): the quiz fixture renders exactly one <a2ui-surface>; the page has "
            f"{probe['a2ui_count']} (surface attribute {probe['surface']!r})"
        )
        surfaces = [m for m in live.wire if getattr(m, "type", "") == "astra.ui.surface"]
        # The canvas goes out more than once (a streamed surface and its final), and all of them
        # are the same surface: one <a2ui-surface> in the page for n messages is the assertion.
        assert surfaces and surfaces[-1].final, (
            f"the canvas must end with a final astra.ui.surface; the wire carried {surfaces}"
        )
        assert len({m.surface_id for m in surfaces}) == 1, (
            f"one turn is one surface id; the wire used {[m.surface_id for m in surfaces]}"
        )
        assert probe["surface"] == surfaces[-1].surface_id, (
            "the rendered surface is the one the server sent: page says "
            f"{probe['surface']!r}, the wire said {surfaces[0].surface_id!r}"
        )
        assert BUTTON_LABEL in str(probe["deep_text"]), (
            "the quiz's own button must be in the rendered tree; the page reads "
            f"{str(probe['deep_text'])[:400]!r}"
        )
        assert "no surface yet" not in str(probe["body"]), "the placeholder is still up"
        assert probe["errors"] in (None, ""), f"the page put an error up: {probe['errors']!r}"


# --- (d) the click round trip ----------------------------------------------------------------------


@requires_chrome
async def test_d_clicking_the_surface_sends_one_action_and_opens_a_second_turn(tmp_path):
    """2(d): one click is exactly one ``astra.ui.action`` and exactly one new turn.

    The click goes through the rendered A2UI button, so the action's ``name``, ``surface_id`` and
    ``context`` are the fixture's (``check_quiz`` on ``lesson_1``) and nothing the test made up.
    "Exactly one" is the adversarial half: a page that re-sends on every render, or a renderer
    that fires both the button and its label, opens two turns out of one gesture.
    """
    script = [*turn_script(), thought("They answered."), answer("Two of five are right. ")]
    async with live_page(tmp_path, script, ms_per_word=200) as (live, page):
        await live.ask(QUESTION)
        await page.until("!!document.querySelector('[data-ui-surface]')", "the surface to click")
        await live.idle()
        transcripts_before = list(live.llm.transcripts())
        rendered = (await page.probe())["surface"]
        assert rendered, "the page must have a surface id on [data-ui-surface] before the click"

        result = await page.evaluate(click_js(BUTTON_LABEL))
        assert result == "clicked", f"the button could not be clicked: {result!r}"

        await live.until(
            lambda: len(live.llm.transcripts()) > len(transcripts_before),
            "the click never opened a turn",
        )
        await live.idle()
        await asyncio.sleep(0.5)  # a duplicate action would have arrived by now

        actions = live.machine.events(E.UiAction)
        assert len(actions) == 1, (
            f"2(d): one click is one astra.ui.action; the session saw {actions}"
        )
        # ``UiGate`` re-ids the canvas per turn, so the id the click must echo is the one the page
        # was *rendering* when it was clicked, read off [data-ui-surface] above; the event name
        # and context are the fixture's, which is how we know the click came from the A2UI tree.
        assert (actions[0].surface_id, actions[0].name) == (rendered, ACTION_NAME), (
            "2(d): the action echoes the rendered surface and the fixture's event name; it "
            f"carried {actions[0].surface_id!r}/{actions[0].name!r} against "
            f"{rendered!r}/{ACTION_NAME!r}"
        )
        assert actions[0].context.get("quizId") == "fractions_1", (
            f"the action's context is the fixture's: {actions[0].context}"
        )
        transcripts = live.llm.transcripts()
        assert len(transcripts) == len(transcripts_before) + 1, (
            f"2(d): exactly one new turn; the LLM was asked {transcripts}"
        )
        assert transcripts[-1].startswith("[ui] "), (
            f"docs/02 §2.2: a clicked turn's transcript begins '[ui] '; it is {transcripts[-1]!r}"
        )
        assert ACTION_NAME in transcripts[-1], transcripts[-1]
        assert len(live.turn_lines()) == 2, (
            f"two turns, two turns.jsonl lines; got {len(live.turn_lines())}"
        )


@requires_chrome
async def test_t0164_b4_the_gate_clicks_second_turn_transcript_is_byte_equal_to_the_pinned_ui_line(
    tmp_path,
):
    """T-0164 B4: real ``HudServer`` and ``Session``, fake LLM. The model writes a Button with a
    bare ``text``; the gate makes it ``__gate.clicked``; a click on it in the light-DOM card (no
    shadow piercing) opens exactly one second turn whose transcript is byte-equal to the line
    pinned today, ``[ui] __gate.clicked {"label":"Show me more"}``."""
    script = [
        thought("Offer more."),
        answer(FIRST),
        ui_commit(GATE_BODY),
        thought("They want more."),
        answer("Here is more. "),
    ]
    async with live_page(tmp_path, script, ms_per_word=200) as (live, page):
        await live.ask(QUESTION)
        await page.until("!!document.querySelector('[data-ui-surface]')", "the gate card")
        await live.idle()
        before = list(live.llm.transcripts())
        result = await page.evaluate(light_click_js(GATE_LABEL))
        assert result == "clicked", f"B1/B4: the gate button in the light-DOM card: {result!r}"
        await live.until(
            lambda: len(live.llm.transcripts()) > len(before), "the click never opened a turn"
        )
        await live.idle()
        await asyncio.sleep(0.5)  # a duplicate action would have arrived by now
        actions = live.machine.events(E.UiAction)
        assert len(actions) == 1, f"B4: one click is one astra.ui.action; the session saw {actions}"
        transcripts = live.llm.transcripts()
        assert len(transcripts) == len(before) + 1, transcripts
        stamp = (actions[0].context.get("a2ui") or {}).get("timestamp")
        assert isinstance(stamp, str) and len(stamp) == 24 and stamp.endswith("Z"), actions[0]
        want = gate_transcript(stamp)
        assert transcripts[-1] == want, (
            f"B4: the second turn's transcript is byte-equal to today's {want!r}; it is "
            f"{transcripts[-1]!r}"
        )


# --- (e) barge-in ----------------------------------------------------------------------------------


@requires_chrome
async def test_e_barge_in_strikes_the_unplayed_tail_and_removes_nothing(tmp_path):
    """2(e): ``UserSpeechOnset`` mid-utterance strikes the tail nobody heard, and only that.

    The cut is not a guess: the session publishes ``astra.spoken.played`` with the word index it
    actually reached, and the page must strike from that index on — so the expected tail is
    recomputed here from the published ``astra.spoken`` text and that index, and compared with
    what the DOM struck. ``text-decoration-line`` is the *computed* style, so a rule that exists
    but does not apply fails. Nothing may be deleted: the words the user did not hear stay
    readable (T-0030 criterion 19).
    """
    script = [
        thought("Slowly then."),
        answer("One two three four five six. "),
        answer("Seven eight nine ten eleven twelve. "),
    ]
    async with live_page(tmp_path, script, ms_per_word=400) as (live, page):
        asking = live.asking(QUESTION)
        await live.until(
            lambda: live.session.state.state is TurnState.SPEAKING, "the turn never spoke"
        )
        await asking
        await asyncio.sleep(0.9)  # let the first utterance get under way
        now = live.now
        live.session.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))

        await page.until(
            "document.querySelectorAll('[data-unplayed]').length >= 1",
            "a struck tail after the barge-in",
        )
        probe = await page.probe()

        played = [m for m in live.wire if getattr(m, "type", "") == "astra.spoken.played"]
        assert played, "2(e): the session must publish astra.spoken.played when playback stops"
        report = played[-1]
        said = {m.utterance_id: m.text for m in live.spoken()}
        #: what the page *must* strike: for each published utterance, the words from the index
        #: the session says it reached. Computed from the wire, never from the page.
        expected = {
            u.utterance_id: said[u.utterance_id].split()[u.played_words :]
            for u in report.utterances
        }
        cut = [u for u in report.utterances if not u.complete and u.played_words > 0]
        assert cut, (
            "the barge-in must land *inside* an utterance, otherwise this row proves nothing "
            f"about a tail; the report is {report}"
        )
        assert any(expected.values()), f"nothing was left unheard; the report is {report}"

        shown = {u["utterance_id"]: u for u in probe["spoken"]}
        assert set(expected) <= set(shown), (
            f"every published utterance is on the page; the page shows {sorted(shown)} of "
            f"{sorted(expected)}"
        )
        for utterance_id, tail_words in expected.items():
            tails = shown[utterance_id]["unplayed"]
            struck = [word for tail in tails for word in tail["text"].split()]
            assert struck == tail_words, (
                f"2(e): {utterance_id} was heard up to word "
                f"{len(said[utterance_id].split()) - len(tail_words)} of "
                f"{said[utterance_id]!r}, so {tail_words} is the tail to strike; the page struck "
                f"{struck}"
            )
            for tail in tails:
                assert "line-through" in tail["decoration"], (
                    "2(e): the unheard tail is struck through (computed style); it computes to "
                    f"{tail['decoration']!r}"
                )
            assert said[utterance_id].split() == shown[utterance_id]["text"].split(), (
                "2(e): the text is struck, never removed; the page shows "
                f"{shown[utterance_id]['text']!r} where the utterance was "
                f"{said[utterance_id]!r}"
            )


# --- (f) disconnect, reconnect, and the thought that never leaves ------------------------------------


@requires_chrome
async def test_f_closing_the_browser_disconnects_once_and_a_fresh_page_reconnects(tmp_path):
    """2(f): one ``ClientDisconnected`` after the grace, a fresh page reconnects, no thought leaks.

    ``RECONNECT_GRACE_MS`` is why the event is *one* and why it is late: a reload must not look
    like the user walking away. The thought scan is the negative half — ``SECRET_THOUGHT`` is
    committed by the LLM on the turn below and must appear in no DOM, no server→client message
    and no ``turns.jsonl`` line (CLAUDE.md: only what was spoken is stored).

    **Narrowed by T-0180** (B9, ruling R-1): on the wire the thought may appear in exactly one
    place, the ``text`` of an ``astra.thought``; and (anti-vacuity) it **does** appear there,
    since the script committed it. It stays absent from every other message, every other field,
    history and ``turns.jsonl``. **The DOM half (T-0180 group C):** on both pages the thought
    may appear only inside ``[data-model-thoughts]``; on the first page it **does** appear there
    (anti-vacuity). The reconnected page follows a departure, which clears the hello journal
    (T-0037 criterion 4), so it holds the thought nowhere, not even in a disclosure.
    """
    script = [thought(SECRET_THOUGHT), answer(FIRST), ui_commit()]
    live = Live(tmp_path, script, ms_per_word=200)
    await live.start()
    hud, hud_port = serve_hud()
    url = page_url(hud_port, live.server.url)
    try:
        async with open_page(url) as page:
            await live.ask(QUESTION)
            await page.until(
                "document.querySelectorAll('[data-utterance-id]').length >= 1",
                "the turn on the page",
            )
            await live.idle()
            first_dom = await thought_dom(page)
            before = len(live.machine.events(E.ClientDisconnected))
            assert before == 0, f"nothing disconnected yet; the machine saw {before}"

        # the browser is gone: the event comes after RECONNECT_GRACE_MS, not before
        await asyncio.sleep(RECONNECT_GRACE_MS / 2000)
        assert live.machine.events(E.ClientDisconnected) == [], (
            "2(f): a page that vanished for less than the reconnect grace is a reload, not a "
            "departure; the session was told after "
            f"{RECONNECT_GRACE_MS / 2:.0f} ms"
        )
        await live.until(
            lambda: len(live.machine.events(E.ClientDisconnected)) >= 1,
            "ClientDisconnected after the grace",
            timeout_s=10.0,
        )
        await asyncio.sleep(1.0)
        assert len(live.machine.events(E.ClientDisconnected)) == 1, (
            "2(f): one departure is one ClientDisconnected; the machine saw "
            f"{live.machine.events(E.ClientDisconnected)}"
        )

        async with open_page(url) as page2:
            assert live.hellos == 2, (
                f"2(f): the fresh page greets the server again; hellos={live.hellos}"
            )
            second_dom = await thought_dom(page2)
            probe = await page2.probe()
            assert probe["chip"] == live.session.state.state.name, (
                "the reconnected page starts from the server's state, not from a guess: "
                f"{probe['chip']!r} vs {live.session.state.state.name}"
            )
            assert len(live.machine.events(E.ClientDisconnected)) == 1, (
                "a reconnect is not another departure"
            )

        # Anti-vacuity: the thought really did enter the conductor on this turn. Without this the
        # three scans below would also pass for a session that never generated anything.
        thoughts = [
            e
            for e in live.machine.events(E.LlmCommit)
            if e.segment == "thought" and "SECRETTHOUGHT" in e.text
        ]
        assert thoughts, (
            "the turn machine never saw the thought commit, so the scans below prove nothing; "
            f"it saw {[(e.segment, e.text[:40]) for e in live.machine.events(E.LlmCommit)]}"
        )
        for name, (outside, _inside) in (
            ("the first page", first_dom),
            ("the reconnected page", second_dom),
        ):
            assert "SECRETTHOUGHT" not in outside, (
                f"2(f), narrowed by T-0180 C: the assistant's thinking reached {name}'s DOM "
                "outside [data-model-thoughts]"
            )
        assert SECRET_THOUGHT in first_dom[1], (
            "T-0180 B9 anti-vacuity (DOM): the committed thought is not in the first page's "
            f"[data-model-thoughts]; the disclosures read {first_dom[1][:200]!r}"
        )
        # The reconnected page comes after a *departure* (ClientDisconnected above), and a departure
        # clears the hello journal (T-0037 criterion 4: nothing outlives it; T-0180 A4 keeps the
        # thoughts in that same journal), so there is nothing to replay: no thought, not even inside
        # a disclosure. A reload inside the grace (the replay case) is T-0180 C's Chrome row (d).
        assert "SECRETTHOUGHT" not in second_dom[1], (
            "T-0037 criterion 4 / T-0180 A4: a thought outlived the departure into the reconnected "
            f"page's [data-model-thoughts]: {second_dom[1][:200]!r}"
        )
        leaked = thought_leaks(live.wire, "SECRETTHOUGHT")
        assert not leaked, (
            f"2(f), narrowed by T-0180: the thought went out on the wire outside astra.thought's "
            f"text: {leaked}"
        )
        shown = thought_texts(live.wire)
        assert any(SECRET_THOUGHT in text for text in shown), (
            "T-0180 B9 anti-vacuity: the committed thought is in no astra.thought text; the "
            f"astra.thought texts on the wire were {shown}"
        )
        history = json.dumps([h.model_dump(mode="json") for h in live.session.state.history])
        assert "SECRETTHOUGHT" not in history, "T-0180 B9: the thought reached history"
        lines = live.turns_path.read_text() if live.turns_path.exists() else ""
        assert lines.strip(), "the turn must have been recorded at all"
        assert "SECRETTHOUGHT" not in lines, (
            f"2(f): the thought was written to turns.jsonl:\n{lines}"
        )
        record = live.turn_lines()[0]
        assert record["outcome"] == "spoken" and record["canvases"] >= 1, (
            f"the scan above must be reading the record of a turn that actually spoke: {record}"
        )
    finally:
        hud.shutdown()
        await live.aclose()
