"""The HUD page in headless Chrome, against the stub Conductor (T-0008, the load-bearing test).

One browser run per module: the page is served from ``hud/`` over loopback http (ES modules do not
load over ``file://`` — docs/tasks/T-0008.md implementation notes), told to use the synthetic
``canvas.captureStream()`` provider with ``?source=canvas``, and pointed at
``tests/fakes/hud_server.HudStub`` with ``?ws=``. The script the stub replays contains the
adversarial interleave of criterion 2; the assertions are split over the criteria below.

**Driving:** Chrome DevTools Protocol rather than ``--dump-dom --virtual-time-budget``. Measured on
this box (Chrome 147.0.7727.101, 2026-09-13): under ``--virtual-time-budget=6000`` the page's
virtual clock burns ~15x faster than real time, so the browser exits after ~0.4 s of wall clock and
a scripted WebSocket sequence is truncated mid-way (6 of 8 messages arrived; the 8 MiB message and
the message after it never did). Real-time polling over CDP delivers all of it. ``--dump-dom`` stays
the right tool for ``scripts/hud_dom.sh`` screenshots of a page that fills itself from timers.

**The DOM contract this test pins** (the task fixes the *content*, not the markup; these are the
hooks the assertions need, and the implementer must provide them):

===========================  ==========================================================
``[data-turn-state]``        the turn chip; the attribute value is the ``TurnState`` name
``[data-thinking-ms]``       the thinking meter; the attribute value is integer ms
``[data-utterance-id]``      one element per spoken utterance, also ``[data-turn-id]``
``[data-unplayed]``          (T-0030 criterion 19) inside an utterance element: the words from
                             index ``played_words`` on, struck through and dimmed, never removed
``[data-transcript]``        the user transcript line (since T-0154 B3: the user bubble of the
                             turn's ``article[data-turn-id]`` inside ``[data-thread]``)
``[data-capture-size]``      ``"WxH"`` of the capture *before* the resolution policy
``[data-frames-sent]``       frames sent (criterion 4)
``[data-frames-dropped]``    frames dropped (criterion 3, "the drop count is displayed")
text ``SYNTHETIC CAPTURE``   criterion 7 banner
===========================  ==========================================================

**Amended by T-0154 group B** (docs/tasks/T-0154.md G2). The ``no surface yet (T-0010)``
placeholder row of the table is retired with ``#shown-channel`` (B1); criterion 10's "the page
renders without ./ui/index.js" is re-pinned from the placeholder form to B11's: the
``degraded`` session below serves ``/src/ui/index.js`` as a 404, as a module that throws at
import and as one whose ``mount()`` throws, and the thread must still render the text and the
action cards while the socket keeps working. The T-0090 ``[data-tool-calls]`` rows are re-keyed
(B7): the cards live in the t7 article of the thread, not inside a ``[data-ui-surface]`` card.
"""

from __future__ import annotations

import base64
import functools
import http.server
import itertools
import json
import os
import re
import shutil
import socketserver
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from websockets.sync.client import connect

from local_astra.contracts.codec import decode
from local_astra.contracts.perception import Source
from local_astra.contracts.protocol import (
    AstraError,
    AstraFrame,
    AstraSpoken,
    AstraTranscript,
    AstraTurnState,
)
from local_astra.contracts.turn import TurnState
from local_astra.perception import policy
from tests.fakes.hud_server import HudStub, oversized_message

#: T-0030 group A's models. Guarded so that this module still *collects* before that group lands
#: (a pending test must fail with an assertion, never an ImportError at collection time); the
#: scripted entries below appear only once the contract exists, and the CDP row says so.
try:
    from local_astra.contracts.protocol import AstraSpokenPlayed, PlayedUtterance
except ImportError:  # the pre-implementation path
    AstraSpokenPlayed = PlayedUtterance = None  # type: ignore[assignment,misc]

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
HUD = REPO / "hud"
CHROME = shutil.which("google-chrome") or shutil.which("chromium")


def _t0164_b_activated() -> bool:
    """T-0164 group B (docs/tasks/T-0164.md B2): true once T-0164's **Status:** records "group B:
    tests <a>..<b>, activated <sha>" (or ``ASTRA_RUN_PENDING=1``). The HUD Chrome modules that
    count or find the rendered card re-key it from ``<a2ui-surface>`` to ``[data-a2ui-root]`` then;
    before it they are today's rows."""
    if os.environ.get("ASTRA_RUN_PENDING") == "1":
        return True
    task = REPO / "docs" / "tasks" / "T-0164.md"
    text = task.read_text(encoding="utf-8") if task.is_file() else ""
    status = re.search(r"^\*\*Status:\*\*\s*(.*)$", text, re.M)
    line = status.group(1) if status else ""
    return re.search(r"\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b", line, re.I) is not None


T0164_B = _t0164_b_activated()
#: The CSS selector of the rendered card element (T-0164 B2's re-key by name).
A2UI_ROOT = "[data-a2ui-root]" if T0164_B else "a2ui-surface"

requires_chrome = pytest.mark.skipif(
    CHROME is None,
    reason="headless Chrome is a dev prerequisite for the HUD integration test "
    "(docs/tasks/T-0008.md Regime: Google Chrome 147.0.7727.101 as `google-chrome`)",
)

TRANSCRIPT_TEXT = "what is on my screen"
SPOKEN_1 = "It is a stack trace."
SPOKEN_2 = "The error is on line four."
SPOKEN_FINAL = "Still here after all that."
ERROR_CODE = "llm_unavailable"
ERROR_MESSAGE = "vllm is not answering"
#: T-0030: the user heard all of ``SPOKEN_1`` and the first three words of ``SPOKEN_2``, so
#: ``"on line four."`` is the tail nobody heard. Hand-split from SPOKEN_2's six words.
PLAYED_WORDS = 3
PLAYED_PREFIX = " ".join(SPOKEN_2.split()[:PLAYED_WORDS])
UNHEARD_TAIL = " ".join(SPOKEN_2.split()[PLAYED_WORDS:])
PLAYED_MS = 2_400
PLACEHOLDER = "no surface yet (T-0010)"
SYNTHETIC_BANNER = "SYNTHETIC CAPTURE"


def script() -> list[Any]:
    """What the stub replays after ``astra.hello`` — criterion 2's adversarial interleave."""
    canvas_step = json.dumps(
        {
            "type": "astra.canvas.step",
            "request_id": "r1",
            "canvas_idx": 0,
            "step": 3,
            "max_steps": 16,
            "canvas_length": 8,
            "argmax": [1, 2, 3, 4, 5, 6, 7, 8],
            "accepted": "f0",
            "entropy": [0.1] * 8,
        }
    )
    played: list[Any] = []
    if AstraSpokenPlayed is not None:  # T-0030 criteria 8 and 19
        played = [
            AstraSpokenPlayed(
                turn_id="t1",
                played_ms=PLAYED_MS,
                utterances=[
                    PlayedUtterance(
                        utterance_id="u1",
                        played_text=SPOKEN_1 + " ",
                        played_words=len(SPOKEN_1.split()),
                        complete=True,
                    ),
                    PlayedUtterance(
                        utterance_id="u2",
                        played_text=PLAYED_PREFIX,
                        played_words=PLAYED_WORDS,
                        complete=False,
                    ),
                ],
            )
        ]
    return [
        # T-0030: the turn is marked interrupted *before* the THINKING row, not after it, so the
        # chip and the meter still read THINKING at the end of the script (the T-0008 assertions
        # below stay unmodified). The strike-through must come from the played message and from
        # nothing else — criterion 22 — and this ordering is what proves it here too: by the time
        # the page dims anything, the INTERRUPTED message is ten messages old.
        AstraTurnState(state=TurnState.INTERRUPTED, turn_id="t1"),
        AstraTurnState(state=TurnState.THINKING, turn_id="t1", thinking_ms=3_200),
        AstraTranscript(turn_id="t1", text=TRANSCRIPT_TEXT, final=True),
        AstraSpoken(turn_id="t1", utterance_id="u1", text=SPOKEN_1),
        AstraSpoken(turn_id="t1", utterance_id="u2", text=SPOKEN_2),
        AstraError(code=ERROR_CODE, message=ERROR_MESSAGE),
        "{this is not json at all",  # malformed
        '{"type":"astra.zzz","payload":{"a":1}}',  # unknown astra.*
        '{"type":"response.created","response":{"id":"r1"}}',  # RealtimeEvent passthrough
        '{"type":42}',  # non-string type
        oversized_message(),  # over codec.MAX_MESSAGE_BYTES
        canvas_step,  # well formed, but T-0011's panel is absent
        AstraSpoken(turn_id="t2", utterance_id="u3", text=SPOKEN_FINAL),  # must still render
        *played,
    ]


# --- headless Chrome over CDP ---------------------------------------------------------------


class Chrome:
    """Just enough DevTools Protocol to navigate, poll in real time and read the DOM."""

    def __init__(self, binary: str) -> None:
        self.binary = binary
        self.profile = tempfile.mkdtemp(prefix="astra-hud-")
        self.proc: subprocess.Popen[bytes] | None = None
        self.ws: Any = None
        self._id = 0

    def __enter__(self) -> Chrome:
        self.proc = subprocess.Popen(
            [
                self.binary,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--hide-scrollbars",
                "--window-size=1280,800",
                "--no-first-run",
                "--disable-extensions",
                f"--user-data-dir={self.profile}",
                "--remote-debugging-port=0",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        port_file = Path(self.profile) / "DevToolsActivePort"
        deadline = time.monotonic() + 30
        port = None
        while time.monotonic() < deadline:
            if port_file.exists():
                lines = port_file.read_text().splitlines()
                if lines:
                    port = int(lines[0])
                    break
            time.sleep(0.05)
        assert port, "headless Chrome never wrote DevToolsActivePort"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as r:
            targets = json.loads(r.read())
        page = next(t for t in targets if t["type"] == "page")
        self.ws = connect(page["webSocketDebuggerUrl"], max_size=64 * 1024 * 1024, open_timeout=15)
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            if self.ws is not None:
                self.ws.close()
        finally:
            if self.proc is not None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            shutil.rmtree(self.profile, ignore_errors=True)

    def _cmd(self, method: str, **params: Any) -> dict[str, Any]:
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv(timeout=30))
            if msg.get("id") == self._id:
                assert "error" not in msg, f"{method}: {msg['error']}"
                return dict(msg["result"])

    def navigate(self, url: str) -> None:
        self._cmd("Page.enable")
        self._cmd("Page.navigate", url=url)

    def evaluate(self, expression: str) -> Any:
        r = self._cmd(
            "Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True
        )
        if "exceptionDetails" in r:
            return {"__exception__": r["exceptionDetails"].get("text")}
        return r["result"].get("value")

    # --- real input, not a synthesised DOM event -------------------------------------------
    #
    # ``Runtime.evaluate`` with ``dispatchEvent`` builds an untrusted event that never travels
    # the browser's own key pipeline: no character is inserted, and (T-0039 ruling A19) the
    # document-level shortcut listener in ``hud/src/main.js`` is never given the chance to fire.
    # ``Input.dispatchKeyEvent`` is the key path a human has, which is what a row about a
    # keystroke must use.

    def key(self, key: str, *, text: str | None = None, code: int | None = None) -> None:
        """One keystroke: keyDown (with its character, if it has one) then keyUp."""
        down: dict[str, Any] = {"type": "keyDown", "key": key}
        if text is not None:
            down["text"] = text
            down["unmodifiedText"] = text
        if code is not None:
            down["windowsVirtualKeyCode"] = code
            down["nativeVirtualKeyCode"] = code
        self._cmd("Input.dispatchKeyEvent", **down)
        up: dict[str, Any] = {"type": "keyUp", "key": key}
        if code is not None:
            up["windowsVirtualKeyCode"] = code
            up["nativeVirtualKeyCode"] = code
        self._cmd("Input.dispatchKeyEvent", **up)

    def type_text(self, text: str) -> None:
        """Type ``text`` one character at a time, into whatever holds the focus."""
        for char in text:
            self.key(char, text=char)

    def enter(self) -> None:
        self.key("Enter", text="\r", code=13)

    def wait_until(self, expression: str, timeout_s: float = 20.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.evaluate(expression):
                return True
            time.sleep(0.1)
        return bool(self.evaluate(expression))

    def dom(self) -> str:
        return str(self.evaluate("document.documentElement.outerHTML"))


def serve_hud() -> tuple[socketserver.TCPServer, int]:
    handler = functools.partial(_QuietHandler, directory=str(HUD))
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, int(server.server_address[1])


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass


PROBE_JS = """
(() => {
  const attr = (sel, name) => {
    const el = document.querySelector(sel);
    return el ? el.getAttribute(name) : null;
  };
  return {
    thread_present: !!document.querySelector('[data-thread]'),
    chip: attr('[data-turn-state]', 'data-turn-state'),
    chip_text: (document.querySelector('[data-turn-state]') || {}).textContent || null,
    thinking_ms: attr('[data-thinking-ms]', 'data-thinking-ms'),
    transcript: (document.querySelector('[data-transcript]') || {}).textContent || null,
    capture_size: attr('[data-capture-size]', 'data-capture-size'),
    frames_sent: attr('[data-frames-sent]', 'data-frames-sent'),
    frames_dropped: attr('[data-frames-dropped]', 'data-frames-dropped'),
    spoken: [...document.querySelectorAll('[data-utterance-id]')].map((e) => ({
      utterance_id: e.getAttribute('data-utterance-id'),
      turn_id: e.getAttribute('data-turn-id'),
      text: e.textContent.trim(),
      // T-0030 ruling A8: the decoration of the utterance element *itself*, so "the played part
      // is not struck" is a style assertion and not a string comparison.
      decoration: getComputedStyle(e).textDecorationLine,
    })),
    // T-0030 criteria 19 and 20: the tail nobody heard, and what the stylesheet does to it.
    // The *computed* style, so a rule that exists but does not apply fails here.
    unplayed: [...document.querySelectorAll('[data-utterance-id]')].map((e) => {
      const tails = [...e.querySelectorAll('[data-unplayed]')];
      const tail = tails[0] || null;
      const style = tail ? getComputedStyle(tail) : null;
      return {
        utterance_id: e.getAttribute('data-utterance-id'),
        count: tails.length,
        text: tail ? tail.textContent.trim() : null,
        decoration: style ? style.textDecorationLine : null,
        opacity: style ? style.opacity : null,
      };
    }),
  };
})()
"""


@dataclass
class Run:
    dom: str
    probe: dict[str, Any]
    stub: HudStub
    frames_after_script: int = 0
    #: T-0030: did a ``[data-unplayed]`` element ever appear? (a note, not a failure: the other
    #: rows in this module must not go red because group C has not landed)
    struck: bool = False
    notes: list[str] = field(default_factory=list)


@pytest.fixture(scope="module")
def run() -> Any:
    """One page load, shared by the assertions below (a browser launch per criterion is waste)."""
    assert (HUD / "index.html").is_file(), (
        "T-0008 adds hud/index.html (plus src/main.js and friends) — the page this test drives"
    )
    assert CHROME is not None
    stub = HudStub(script()).start()
    server, port = serve_hud()
    try:
        url = f"http://127.0.0.1:{port}/?source=canvas&ws=" + urllib.parse.quote(stub.url, safe="")
        with Chrome(CHROME) as chrome:
            chrome.navigate(url)
            ok_hello = stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20)
            assert ok_hello, (
                "criterion 1: the page must send exactly one astra.hello on open; the stub saw "
                f"{stub.received_types()!r} in 20 s"
            )
            stub.script_sent.wait(timeout=20)
            frames_at_script = len(stub.frames())
            rendered = chrome.wait_until(
                "document.querySelectorAll('[data-utterance-id]').length >= 3", 25
            )
            stub.wait_for(lambda s: len(s.frames()) > frames_at_script, 10)
            struck = chrome.wait_until(
                "document.querySelectorAll('[data-unplayed]').length >= 1", 8
            )
            probe = chrome.evaluate(PROBE_JS)
            dom = chrome.dom()
        result = Run(
            dom=dom,
            probe=probe if isinstance(probe, dict) else {"__probe__": probe},
            stub=stub,
            frames_after_script=len(stub.frames()) - frames_at_script,
            struck=bool(struck),
        )
        if not rendered:
            result.notes.append("the third spoken utterance never rendered")
        yield result
    finally:
        server.shutdown()
        stub.stop()


def spoken_of(run: Run) -> list[dict[str, Any]]:
    return list(run.probe.get("spoken") or [])


def unplayed_of(run: Run, utterance_id: str) -> dict[str, Any]:
    rows = [r for r in (run.probe.get("unplayed") or []) if r["utterance_id"] == utterance_id]
    assert rows, (
        f"T-0030 criterion 19: no [data-utterance-id={utterance_id!r}] element in the pane at "
        f"all; probe={run.probe!r}"
    )
    return dict(rows[0])


@requires_chrome
def test_the_unheard_tail_is_struck_through_in_the_spoken_pane(run):
    """T-0030 criteria 19 and 20, end to end: the page dims what the device never rendered.

    The stub sent ``astra.spoken.played(t1, u1 complete, u2 played_words=3)`` after the two
    utterances, so the page has both the text and the measurement. What this row pins, in a real
    browser with the real stylesheet:

    * the utterance that was heard whole carries **no** ``[data-unplayed]`` element (not an empty
      one, not one with a zero-width tail) — the reviewer's mutation "keep the class, drop the
      element" dies here;
    * the cut utterance carries exactly one, holding exactly the words from index 3 on;
    * its *computed* ``text-decoration-line`` contains ``line-through`` and its opacity is below
      1 (criterion 20: a rule that exists but does not apply is not a rule);
    * the full sentence is still in the DOM (docs/02 §4: "the HUD may show unspoken text struck
      through" — struck, never removed. A page that deletes the tail loses what the assistant was
      about to say, which is the whole point of showing it).
    """
    assert AstraSpokenPlayed is not None, (
        "T-0030 group A is not implemented: contracts/protocol.py has no AstraSpokenPlayed, so "
        "the stub could not script one (docs/tasks/T-0030.md criterion 1)"
    )
    heard = unplayed_of(run, "u1")
    assert heard["count"] == 0, (
        f"criterion 19: complete === true means no [data-unplayed] element at all; got {heard!r}"
    )
    cut = unplayed_of(run, "u2")
    assert run.struck, (
        "criterion 19: the page never rendered a [data-unplayed] element for the utterance the "
        f"server said was cut after {PLAYED_WORDS} words; probe={run.probe.get('unplayed')!r}"
    )
    assert cut["count"] == 1, f"criterion 19: exactly one tail element, got {cut!r}"
    assert cut["text"] == UNHEARD_TAIL, (
        f"criterion 19: the tail is the words from index {PLAYED_WORDS} on of "
        f"{SPOKEN_2!r}, i.e. {UNHEARD_TAIL!r}; the page shows {cut['text']!r}"
    )
    assert "line-through" in (cut["decoration"] or ""), (
        "criterion 20: hud/styles.css gives [data-unplayed] a line-through; the computed "
        f"text-decoration-line is {cut['decoration']!r}"
    )
    assert 0.0 < float(cut["opacity"] or 1.0) < 1.0, (
        f"criterion 20: and reduced opacity — visible, but plainly not spoken; got "
        f"{cut['opacity']!r}"
    )
    whole = next(u for u in spoken_of(run) if u["utterance_id"] == "u2")
    assert whole["text"] == SPOKEN_2, (
        "criterion 19: the full text is always present in the DOM, dimmed and struck, never "
        f"removed; the utterance element reads {whole['text']!r}"
    )
    played_part = whole["text"].replace(cut["text"], "").strip()
    assert played_part == PLAYED_PREFIX, (
        "criterion 19: removing the unheard tail from the utterance's text leaves exactly the "
        f"words the server said were played; got {played_part!r}"
    )
    # ... and the played part is not struck: the decoration lives on the tail element, not on the
    # utterance element (ruling A8 — asserted as a computed style, not inferred from the text).
    assert "line-through" not in (whole["decoration"] or ""), (
        "criterion 20: only the [data-unplayed] tail is struck through; the utterance element "
        f"computes text-decoration-line={whole['decoration']!r}, so the words the user did hear "
        "are struck too"
    )
    heard_whole = next(u for u in spoken_of(run) if u["utterance_id"] == "u1")
    assert "line-through" not in (heard_whole["decoration"] or ""), (
        "criterion 20: and an utterance heard in full is not struck at all; got "
        f"{heard_whole['decoration']!r}"
    )


# --- criteria 8, 9, 11: the two channels, the chip and the meter ------------------------------


@requires_chrome
def test_the_chip_reads_thinking_and_the_meter_is_past_the_three_second_mark(run):
    """Criterion 9 (docs/01 D11) — the page's own evidence line in the task's screenshot list."""
    assert run.probe.get("chip") == TurnState.THINKING.value, (
        f"criterion 9: the state chip equals astra.turn.state.state; probe={run.probe!r}"
    )
    assert TurnState.THINKING.value in (run.probe.get("chip_text") or ""), (
        "criterion 9: the chip is legible in the DOM, not only in an attribute"
    )
    raw = run.probe.get("thinking_ms")
    assert raw not in (None, ""), (
        "criterion 9: the meter shows thinking_ms while state == THINKING "
        f"([data-thinking-ms]); probe={run.probe!r}"
    )
    assert int(float(raw)) >= 3_000, (
        f"criterion 9: the server sent thinking_ms=3200 and the page clock only adds to it; "
        f"the meter must be past the 3 s mark, got {raw}"
    )


@requires_chrome
def test_both_spoken_utterances_render_in_turn_order_and_the_user_transcript_shows(run):
    """Criterion 8: grouped by ``turn_id`` in arrival order, keyed by ``utterance_id``."""
    got = spoken_of(run)
    assert [u["utterance_id"] for u in got] == ["u1", "u2", "u3"], (
        f"criterion 8: one element per utterance, in arrival order; got {got!r}"
    )
    assert [u["turn_id"] for u in got] == ["t1", "t1", "t2"], "criterion 8: grouped by turn_id"
    assert [u["text"] for u in got] == [SPOKEN_1, SPOKEN_2, SPOKEN_FINAL]
    assert TRANSCRIPT_TEXT in (run.probe.get("transcript") or ""), (
        f"criterion 8: the final user transcript line; probe={run.probe!r}"
    )


@requires_chrome
def test_an_astra_error_is_shown_in_the_error_strip(run):
    """Criterion 2: ``AstraError`` renders its ``code``/``message``."""
    assert ERROR_CODE in run.dom and ERROR_MESSAGE in run.dom, (
        "criterion 2: AstraError renders its code and message in an error strip"
    )


@requires_chrome
def test_the_retired_placeholder_is_gone_and_the_capture_is_labelled_synthetic(run):
    """Criteria 10 and 7. **Amended by T-0154 B1** (G2): the assertion that the page shows
    ``no surface yet (T-0010)`` is removed with ``#shown-channel``, and replaced by its negation;
    criterion 10's degradation half is re-pinned in the ``degraded`` rows below (B11)."""
    assert PLACEHOLDER not in run.dom, (
        f"T-0154 B1: the {PLACEHOLDER!r} placeholder is retired with #shown-channel; the page "
        "still carries it"
    )
    assert run.probe.get("thread_present"), (
        "T-0154 B1: the conversation is one [data-thread]; the page has none"
    )
    assert SYNTHETIC_BANNER in run.dom, (
        "criterion 7: a stream that did not come from getDisplayMedia/getUserMedia makes the page "
        f"show a persistent {SYNTHETIC_BANNER!r} banner (docs/01 D15, the honesty rule)"
    )


@requires_chrome
def test_the_frame_and_drop_counters_are_displayed(run):
    """Criteria 3 and 4: "the drop count is displayed", and a frame counter at ~1/s."""
    assert run.probe.get("frames_sent") not in (None, ""), "criterion 4: a frame counter is shown"
    assert run.probe.get("frames_dropped") not in (None, ""), (
        "criterion 3: older frames are dropped and the drop count is displayed"
    )
    assert int(float(run.probe["frames_sent"])) >= 1


# --- criteria 1, 5: what the page put on the wire, through the real codec ----------------------


@requires_chrome
def test_the_page_sends_exactly_one_hello_and_nothing_but_frames_afterwards(run):
    """Criterion 1, including the negative: no astra.frame.text / assist.toggle / ui.action."""
    types = run.stub.received_types()
    assert types.count("astra.hello") == 1, f"criterion 1: exactly one astra.hello, got {types!r}"
    assert types[0] == "astra.hello", f"criterion 1: hello is the first message, got {types[:3]!r}"
    assert set(types) <= {"astra.hello", "astra.frame"}, (
        "criterion 1 negative: the only client->server types this task's code may emit are "
        f"astra.hello and astra.frame; the stub recorded {sorted(set(types))!r}"
    )


@requires_chrome
def test_every_frame_decodes_through_the_real_codec_with_the_criterion_5_properties(run):
    """Criterion 5, validated by ``local_astra.contracts.codec.decode``, not by a JS mirror."""
    raws = [r for r in run.stub.snapshot() if '"astra.frame"' in r]
    assert raws, "criterion 4: the page must post frames at ~1 fps; the stub received none"
    capture = str(run.probe.get("capture_size") or "")
    assert re.fullmatch(r"\d+x\d+", capture), (
        "criterion 5: the page must report the capture size before the resolution policy in "
        f"[data-capture-size] so this test can compute the expected bucket size; got {capture!r}"
    )
    cap_w, cap_h = (int(v) for v in capture.split("x"))
    expected = policy.target_size(cap_w, cap_h, policy.bucket_for(Source.SCREEN, cap_w, cap_h))
    for raw in raws:
        msg = decode(raw)
        assert isinstance(msg, AstraFrame), f"criterion 5: {msg!r} is not a valid AstraFrame"
        assert msg.source is Source.SCREEN
        assert (msg.width, msg.height) == expected, (
            f"criterion 5: {cap_w}x{cap_h} from the {policy.SCREEN_BUCKETS[0].name} bucket is "
            f"{expected} by policy.target_size (never upscaled); the page sent "
            f"{(msg.width, msg.height)}"
        )
        jpeg = base64.b64decode(msg.jpeg_b64, validate=True)
        assert jpeg[:2].hex() == "ffd8" and jpeg[-2:].hex() == "ffd9", (
            "criterion 5: jpeg_b64 base64-decodes to a JPEG (ffd8 … ffd9)"
        )
        assert msg.phash is not None and re.fullmatch(r"[0-9a-f]{16}", msg.phash), (
            f"criterion 5: phash is a 16-character lowercase-hex dHash, got {msg.phash!r}"
        )
        now_ms = time.time() * 1000
        assert abs(msg.ts_ms - now_ms) < 10 * 60 * 1000, (
            f"criterion 5: ts_ms == Date.now() in the browser, got {msg.ts_ms}"
        )
        assert len(raw) < 1024 * 1024, (
            "implementation notes: refuse and log a frame that would exceed 1 MB rather than "
            f"shipping one the codec would accept and the socket would stall on ({len(raw)} bytes)"
        )


@requires_chrome
def test_frames_arrive_at_about_one_per_second(run):
    """Criterion 4 against the wall clock (the JS test pins the exact throttle on a fake clock)."""
    frames = run.stub.frames()
    assert len(frames) >= 1
    if len(frames) >= 2:
        gaps = [b.ts_ms - a.ts_ms for a, b in itertools.pairwise(frames)]
        assert all(g >= 800 for g in gaps), (
            f"criterion 4: at most one astra.frame per source per second, gaps were {gaps}"
        )


# --- criterion 2, adversarial ------------------------------------------------------------------


@requires_chrome
def test_the_adversarial_interleave_leaves_the_socket_open_and_the_page_rendering(run):
    """Criterion 2: malformed, unknown, Realtime, oversized and an orphan canvas step, then a
    final ``astra.spoken`` that must still render. The expected failure mode is a throw inside the
    message handler that silently kills the socket while the page still looks fine."""
    assert run.stub.script_error is None, run.stub.script_error
    assert not run.notes, f"criterion 2: {run.notes}"
    got = spoken_of(run)
    assert got and got[-1]["text"] == SPOKEN_FINAL, (
        "criterion 2: the astra.spoken after the malformed line, the unknown astra.zzz, the "
        "RealtimeEvent, the oversized message and the orphan astra.canvas.step must still render; "
        f"the spoken pane holds {got!r}"
    )
    assert run.frames_after_script >= 1, (
        "criterion 2: the page kept sending frames after the adversarial batch, which is how we "
        "know the socket is still open and the message handler did not throw"
    )
    assert "astra.zzz" not in run.dom, "an unknown astra.* type is ignored and counted, not shown"


# --- the fake itself (no browser needed) --------------------------------------------------------


def test_the_hud_stub_records_the_client_and_replays_its_script():
    """``tests/fakes/hud_server.HudStub`` is new in T-0008; this is its own coverage.

    The real endpoint it stands in for is T-0007's ``conductor/session.py``; the matching
    real-service contract test belongs to that task, not this one.
    """
    with HudStub([AstraSpoken(turn_id="t1", utterance_id="u1", text="hi"), "{bad"]) as stub:
        assert stub.url.startswith("ws://127.0.0.1:")
        with connect(stub.url, max_size=32 * 1024 * 1024, open_timeout=10) as ws:
            ws.send(json.dumps({"type": "astra.hello", "protocol_version": 1, "client": "hud"}))
            first = ws.recv(timeout=10)
            second = ws.recv(timeout=10)
            frame = AstraFrame(
                source=Source.SCREEN,
                ts_ms=1_700_000_000_000,
                width=16,
                height=16,
                jpeg_b64=base64.b64encode(b"\xff\xd8\xff\xd9").decode(),
                phash="0" * 16,
            )
            ws.send(frame.model_dump_json(exclude_none=True))
            assert stub.wait_for(lambda s: len(s.frames()) == 1, 10)
    assert decode(first) == AstraSpoken(turn_id="t1", utterance_id="u1", text="hi")
    assert second == "{bad", "raw script entries go on the wire unchanged, warts and all"
    assert stub.received_types() == ["astra.hello", "astra.frame"]
    assert stub.frames()[0].phash == "0" * 16
    assert stub.connections == 1


def test_the_oversized_helper_is_actually_over_the_codec_limit():
    """The stub's adversarial entry must really be the ``Malformed`` case, not a near miss."""
    from local_astra.contracts.codec import MAX_MESSAGE_BYTES, Malformed

    raw = oversized_message()
    assert len(raw) > MAX_MESSAGE_BYTES
    assert isinstance(decode(raw), Malformed)


def test_chrome_and_the_page_are_present_for_the_integration_layer():
    """A named skip reason beats a silent green: say what is missing and why it matters."""
    if CHROME is None:
        pytest.skip(
            "no google-chrome/chromium on PATH; the HUD integration layer cannot run "
            "(docs/tasks/T-0008.md Regime)"
        )
    assert (HUD / "index.html").is_file(), "T-0008 adds hud/index.html"
    assert os.access(CHROME, os.X_OK)


def test_the_stub_can_push_one_message_at_a_time_after_the_script():
    """``HudStub.push`` is T-0024's addition to this fake (progressive render, criterion 7).

    Covered here because this file is the fake's home; the real endpoint it stands in for is
    T-0007's ``conductor/session.py``, whose matching contract test belongs to that task.
    """
    from local_astra.contracts.protocol import AstraUiSurface

    surface = AstraUiSurface(
        surface_id="turn-t1",
        canvas_idx=0,
        final=False,
        messages=({"version": "v0.9.1", "createSurface": {"surfaceId": "turn-t1"}},),
    )
    with (
        HudStub([]) as stub,
        connect(stub.url, max_size=32 * 1024 * 1024, open_timeout=10) as ws,
    ):
        ws.send(json.dumps({"type": "astra.hello", "protocol_version": 1, "client": "hud"}))
        assert stub.wait_for_client(10), "push needs a connected client"
        stub.push(surface)
        stub.push('{"type":"astra.ui.surface","canvas_idx":-1}')  # raw, warts and all
        first = ws.recv(timeout=10)
        second = ws.recv(timeout=10)
    assert decode(first) == surface, "a Contract is encoded with the production codec"
    assert second == '{"type":"astra.ui.surface","canvas_idx":-1}', "a raw str goes on unchanged"


# --- T-0090: the action card in a real browser (criteria 4 and 5) -------------------------------
#
# docs/tasks/T-0090.md, "Tests to write": the card renders, and no network request is issued (the
# request log is empty except loopback). The session's own lifecycle publisher is T-0092's, so the
# stub pushes the messages in the order the server publishes them: the turn's spoken line, then the
# call's proposed -> dry_run, then a second call whose query is markup, then a spoken line that must
# still render (criterion 5: every other type is handled exactly as before).
#
# DOM hooks pinned here and in tests/hud/tool_call.test.mjs: ``[data-tool-calls]`` (the container
# in hud/index.html), ``[data-tool-call-id]``, ``[data-tool-line]``, ``[data-tool-status]``,
# ``[data-tool-source]``, ``[data-tool-warning]``.

TOOL_TURN = "t7"
EVIL_QUERY = "<img src=x onerror=window.__pwned=1>"
DRY_RUN_NOTICE = "dry run — nothing was done"
#: The pane headings, static panes plus the two the assists and session-log panels append. T-0090
#: pinned them in document order on main at e0174f9; T-0128 criterion 3 ("Named pin changes",
#: ruling R1 of docs/tasks/T-0128.md) re-keys the pin: the thumb pane's heading becomes "What I
#: see", the headings become sentence case, and the shell declares the two rail panes Thinking and
#: Diagnostics. The comparison is case-folded and order-free, the same reading as the pin in
#: tests/hud/shell.test.mjs: sentence case is a rendered property (asserted in
#: tests/integration/test_hud_redesign.py), and the rail's order is the layout's business.
#: Removed by T-0128: the T-0090 assertion that the list is exactly ["spoken", "you said",
#: "shown", "what the model sees", "assists", "session log"], in that order and in that case.
#: **Amended by T-0154 A5** (docs/tasks/T-0154.md G2): the assist pane's heading renders "Context",
#: so "assists" becomes "context" in this pin; nothing else in the list moves. The row reading it
#: is red until T-0154 group A's implementation lands (activated with the task's status).
#: **Amended by T-0154 B1** (G2): "spoken", "you said" and "shown" leave the list with their
#: panes; the one conversation pane has no heading (ruling O-B (1)), so the pin is the rail's five.
#: **Amended by T-0162 B-9** (named spec change, James 2026-10-26): "context" becomes "tools"; the
#: assist panel's "Context helpers" heading lives in the Settings popover, outside ``<main>``.
TOOL_HEADINGS = sorted(
    [
        "what i see",
        "thinking",
        "diagnostics",
        "tools",  # **Amended by T-0162 B-9**: the Tools pane replaces Context in the rail
        "session log",
    ]
)


def _tool_call(call_id: str, **kw: Any) -> str:
    """One ``astra.tool.call`` as raw JSON (the contract may not exist yet; the wire shape does)."""
    body: dict[str, Any] = {"type": "astra.tool.call", "turn_id": TOOL_TURN, "call_id": call_id}
    body.update(kw)
    return json.dumps(body)


def tool_script() -> list[str]:
    click = {"name": "click_at", "arguments": {"x": 120, "y": 845}, "source": "draft"}
    search = {"name": "google_search", "arguments": {"query": EVIL_QUERY}, "source": "draft"}
    return [
        json.dumps(
            {"type": "astra.spoken", "turn_id": TOOL_TURN, "utterance_id": "u1", "text": "Sure."}
        ),
        _tool_call("c1", status="proposed", **click),
        _tool_call("c1", status="dry_run", action="would click at (153, 676) on screen", **click),
        _tool_call("c2", status="proposed", query_problem="query looks like markup", **search),
        _tool_call(
            "c2",
            status="dry_run",
            action=f"would search for: {EVIL_QUERY}",
            query_problem="query looks like markup",
            **search,
        ),
        # T-0154 B7: the turn's own A2UI card, so "the action cards are not inside the card
        # wrapper" is asserted against a wrapper that exists in the same article.
        t7_card_surface().model_dump_json(exclude_none=True),
        json.dumps(
            {
                "type": "astra.spoken",
                "turn_id": TOOL_TURN,
                "utterance_id": "u2",
                "text": "After the card.",
            }
        ),
    ]


T7_CARD_TEXT = "the t7 card, beside the action cards"


def t7_card_surface() -> Any:
    from local_astra.contracts.protocol import AstraUiSurface

    sid = f"turn-{TOOL_TURN}"
    catalog = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
    return AstraUiSurface(
        surface_id=sid,
        canvas_idx=0,
        final=True,
        messages=(
            {"version": "v0.9.1", "createSurface": {"surfaceId": sid, "catalogId": catalog}},
            {
                "version": "v0.9.1",
                "updateComponents": {
                    "surfaceId": sid,
                    "components": [{"id": "root", "component": "Text", "text": T7_CARD_TEXT}],
                },
            },
        ),
    )


TOOL_PROBE_JS = """
(() => {
  const cards = [...document.querySelectorAll('[data-tool-call-id]')].map((c) => {
    const q = (sel) => c.querySelector(sel);
    const warn = [...c.querySelectorAll('[data-tool-warning]')].filter((w) => !w.closest('[hidden]'));
    return {
      id: c.getAttribute('data-tool-call-id'),
      line: q('[data-tool-line]') ? q('[data-tool-line]').textContent.replace(/\\s+/g, ' ').trim() : null,
      status: q('[data-tool-status]') ? q('[data-tool-status]').getAttribute('data-tool-status') : null,
      source: q('[data-tool-source]') ? q('[data-tool-source]').getAttribute('data-tool-source') : null,
      warnings: warn.map((w) => w.textContent.trim()),
      text: c.textContent.replace(/\\s+/g, ' ').trim(),
      links: c.querySelectorAll('a, [href]').length,
      imgs: c.querySelectorAll('img, script, iframe').length,
      buttons: c.querySelectorAll('button, [onclick]').length,
      // T-0154 B7: where the card lives — the article of its turn, inside the thread, and never
      // inside an A2UI card wrapper (the renderer owns that subtree).
      article: (c.closest('article[data-turn-id]') || {getAttribute: () => null}).getAttribute('data-turn-id'),
      in_thread: !!c.closest('[data-thread]'),
      in_card: !!c.closest('[data-ui-surface]'),
    };
  });
  return {
    cards,
    surfaces: [...document.querySelectorAll('[data-ui-surface]')].map((s) => ({
      id: s.getAttribute('data-ui-surface'),
      article: (s.closest('article[data-turn-id]') || {getAttribute: () => null}).getAttribute('data-turn-id'),
    })),
    pwned: window.__pwned === undefined ? null : String(window.__pwned),
    headings: [...document.querySelectorAll('main h2')].map((h) => h.textContent.trim()),
    spoken: [...document.querySelectorAll('[data-utterance-id]')].map((e) => e.getAttribute('data-utterance-id')),
  };
})()
"""


class _RecordingChrome(Chrome):
    """``Chrome`` plus every ``Network.requestWillBeSent`` URL (the T-0024 pattern)."""

    def __init__(self, binary: str) -> None:
        super().__init__(binary)
        self.requests: list[str] = []

    def _note(self, msg: dict[str, Any]) -> None:
        if msg.get("method") == "Network.requestWillBeSent":
            url = ((msg.get("params") or {}).get("request") or {}).get("url")
            if isinstance(url, str):
                self.requests.append(url)

    def _cmd(self, method: str, **params: Any) -> dict[str, Any]:
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv(timeout=30))
            self._note(msg)
            if msg.get("id") == self._id:
                assert "error" not in msg, f"{method}: {msg['error']}"
                return dict(msg["result"])

    def drain(self, seconds: float = 1.0) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                self._note(json.loads(self.ws.recv(timeout=0.2)))
            except Exception:
                return


@dataclass
class ToolRun:
    probe: dict[str, Any] = field(default_factory=dict)
    requests: list[str] = field(default_factory=list)
    sent_types: list[str] = field(default_factory=list)
    error: str | None = None


@pytest.fixture(scope="module")
def tool_run() -> Any:
    """One page session for T-0090; never raises — a broken harness lands on ``error``."""
    if CHROME is None:  # pragma: no cover - requires_chrome is the real guard
        yield ToolRun(error="no chrome")
        return
    result = ToolRun()
    stub = HudStub([]).start()
    server, port = serve_hud()
    try:
        url = f"http://127.0.0.1:{port}/?source=canvas&ws=" + urllib.parse.quote(stub.url, safe="")
        with _RecordingChrome(CHROME) as chrome:
            chrome._cmd("Network.enable")
            chrome.navigate(url)
            stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20)
            stub.wait_for_client(10)
            for raw in tool_script():
                stub.push(raw)
                time.sleep(0.3)
            chrome.wait_until(
                "[...document.querySelectorAll('[data-tool-call-id]')].length >= 2 && "
                "document.querySelectorAll('[data-utterance-id]').length >= 2 && "
                "!!document.querySelector('[data-ui-surface]')",
                15,
            )
            value = chrome.evaluate(TOOL_PROBE_JS)
            result.probe = value if isinstance(value, dict) else {"__probe__": value}
            chrome.drain(1.5)
            result.requests = list(chrome.requests)
        result.sent_types = stub.received_types()
    except Exception as exc:  # a broken harness is data, not an error inside a test
        result.error = f"the headless run failed: {type(exc).__name__}: {exc}"
    finally:
        server.shutdown()
        stub.stop()
    yield result


def _tool_cards(run: ToolRun) -> dict[str, dict[str, Any]]:
    assert run.error is None, run.error
    return {c["id"]: c for c in run.probe.get("cards") or []}


@requires_chrome
@pytest.mark.desktop
def test_the_action_card_renders_the_dry_run_in_the_shown_pane(tool_run):
    """T-0090 criterion 4, in Chrome with the real stylesheet and main.js mount line."""
    cards = _tool_cards(tool_run)
    assert set(cards) == {"c1", "c2"}, (
        "criterion 4: one card per call_id, the second message of each call updating the first; "
        f"the page shows {sorted(cards)} (probe={tool_run.probe!r})"
    )
    c1, c2 = cards["c1"], cards["c2"]
    assert (c1["line"] or "").startswith("Would click at (120, 845)"), f"criterion 4: {c1!r}"
    assert c1["status"] == "dry_run" and c1["source"] == "draft", f"criterion 4: {c1!r}"
    assert DRY_RUN_NOTICE in c1["text"], (
        f"criterion 4: the dry-run notice; card reads {c1['text']!r}"
    )
    assert c2["line"] == f"Would search for: {EVIL_QUERY}", (
        f"criterion 4 / H2: the query is shown as text; got {c2['line']!r}"
    )
    assert any("query looks like markup" in w for w in c2["warnings"]), (
        f"criterion 4 / H7: query_problem is a visible warning; got {c2['warnings']!r}"
    )
    assert not c1["warnings"], f"…and only where it is set; c1 shows {c1['warnings']!r}"


@requires_chrome
@pytest.mark.desktop
def test_the_card_has_no_link_no_markup_and_nothing_to_press(tool_run):
    """T-0090 criteria 4 and 5 (adversarial): the markup query stays text and runs nothing."""
    cards = _tool_cards(tool_run)
    assert cards, f"premise: the cards rendered; probe={tool_run.probe!r}"
    for card in cards.values():
        assert card["links"] == 0, f"criterion 5 / H2: no link on a card; {card!r}"
        assert card["imgs"] == 0, f"criterion 4: model text creates no element; {card!r}"
        assert card["buttons"] == 0, f"implementation notes: the card is display-only; {card!r}"
    assert tool_run.probe.get("pwned") is None, (
        "criterion 4: the onerror handler in the query must never run (textContent, not innerHTML)"
    )
    assert set(tool_run.sent_types) <= {"astra.hello", "astra.frame"}, (
        f"the card sends nothing back; the page sent {sorted(set(tool_run.sent_types))}"
    )


@requires_chrome
@pytest.mark.desktop
def test_no_request_leaves_loopback_and_no_request_is_made_for_the_query(tool_run):
    """T-0090 criterion 5 / H2: the request log is empty except loopback, and the query is never
    fetched — not as a search, and not as the ``src=x`` of the markup in it."""
    assert _tool_cards(tool_run), (
        "premise: the cards rendered — a clean request log for a page that never drew the card "
        f"says nothing about the card; probe={tool_run.probe!r}"
    )
    assert tool_run.requests, (
        "the CDP network log is empty, so this row would be vacuous — Network.enable must be on "
        "before navigation and the page fetches at least its own modules"
    )
    offenders = [
        u
        for u in tool_run.requests
        if u.split("://", 1)[0] in {"http", "https", "ws", "wss"}
        and not any(u.split("://", 1)[1].startswith(h) for h in ("127.0.0.1", "localhost", "[::1]"))
    ]
    assert not offenders, f"criterion 5: zero non-loopback requests; got {offenders}"
    about_the_query = [
        u
        for u in tool_run.requests
        if "search" in u.lower() or urllib.parse.urlsplit(u).path.rstrip("/").endswith("/x")
    ]
    assert not about_the_query, f"H2: the page made a request for the query: {about_the_query}"


@requires_chrome
@pytest.mark.desktop
def test_the_cards_live_in_their_turns_article_and_the_rail_headings_are_unchanged(tool_run):
    """T-0090 criterion 5, **re-keyed by T-0154 B7** (G2): the cards used to live in the
    ``[data-tool-calls]`` container of ``.pane-shown``, outside ``#shown-channel``; both regions
    are retired (B1). Now every card lives in its turn's article (t7) inside ``[data-thread]``, and
    never inside a ``[data-ui-surface]`` card wrapper (the renderer owns that subtree) — asserted
    with t7's own card present in the same article. Removed assertions: ``has_box``,
    ``in_shown_pane``, ``not inside_a2ui_root``, ``cards_outside_box == 0``; each is replaced by
    the per-card location below. The heading pin is T-0128 criterion 3's as amended by T-0154
    (see ``TOOL_HEADINGS``): case-folded, order-free."""
    assert tool_run.error is None, tool_run.error
    probe = tool_run.probe
    cards = probe.get("cards") or []
    assert len(cards) == 2, f"premise: the two cards rendered; probe={probe!r}"
    surfaces = probe.get("surfaces") or []
    assert surfaces == [{"id": f"turn-{TOOL_TURN}", "article": TOOL_TURN}], (
        "premise (B4): t7's own card renders in t7's article, so the row below is asserted "
        f"beside a real card wrapper; the page has {surfaces!r}"
    )
    misplaced = [
        (c["id"], c.get("in_thread"), c.get("article"), c.get("in_card"))
        for c in cards
        if not (c.get("in_thread") and c.get("article") == TOOL_TURN and not c.get("in_card"))
    ]
    assert misplaced == [], (
        "T-0154 B7: each turn's astra.tool.call cards render in that turn's article inside "
        "[data-thread], outside the A2UI card wrapper; (call_id, in_thread, article, in_card) "
        f"for the misplaced ones: {misplaced}"
    )
    headings = sorted(h.lower() for h in probe.get("headings") or [])
    assert headings == TOOL_HEADINGS, (
        "T-0090 criterion 5 as amended by T-0128 criterion 3: the pane headings, case-folded, are "
        f"{TOOL_HEADINGS!r}; got {probe.get('headings')!r}"
    )


@requires_chrome
@pytest.mark.desktop
def test_other_messages_still_render_around_the_cards(tool_run):
    """T-0090 criterion 5: the page's handling of every other type is unchanged."""
    assert _tool_cards(tool_run), f"premise: the cards rendered; probe={tool_run.probe!r}"
    assert tool_run.probe.get("spoken") == ["u1", "u2"], (
        "criterion 5: the spoken lines before and after the tool calls both render; got "
        f"{tool_run.probe.get('spoken')!r}"
    )


# --- T-0154 B11: the thread without a working ui panel -------------------------------------------
#
# docs/tasks/T-0154.md B11: "With ./ui/index.js absent or throwing, the thread still renders text
# and action cards and the socket keeps working (T-0008 criterion 10, re-pinned from the
# placeholder form)." The page is served from hud/ by a handler that answers /src/ui/index.js
# three ways — a 404, a module that throws while it is evaluated, and a module whose mount()
# throws — and everything else from disk. One headless Chrome for the three, navigated between
# servers (each server is a fresh origin). The script is the wire order of one turn with an action
# card and a card (transcript, utterance, tool call proposed -> dry_run, the surface), then a
# second turn's utterance that must still render after the surface nobody could draw.

UI_INDEX_PATH = "/src/ui/index.js"
UI_BROKEN: dict[str, str | None] = {
    "ui/index.js 404s": None,
    "ui/index.js throws at import": (
        "throw new Error('the ui panel is broken at import (T-0154 B11 test)');\n"
    ),
    "ui/index.js mount() throws": (
        "export function mount() {\n"
        "  throw new Error('the ui panel is broken in mount (T-0154 B11 test)');\n"
        "}\n"
    ),
}
DEGRADED_CARD_TEXT = "DEGRADED-CARD-NOBODY-CAN-DRAW"


def degraded_script() -> list[Any]:
    click = {"name": "click_at", "arguments": {"x": 10, "y": 20}, "source": "draft"}
    card = t7_card_surface()
    surface = card.model_copy(
        update={
            "surface_id": "turn-t1",
            "messages": tuple(
                json.loads(
                    json.dumps(m)
                    .replace(f"turn-{TOOL_TURN}", "turn-t1")
                    .replace(T7_CARD_TEXT, DEGRADED_CARD_TEXT)
                )
                for m in card.messages
            ),
        }
    )
    tool = {"type": "astra.tool.call", "turn_id": "t1", "call_id": "d1", **click}
    return [
        AstraTranscript(turn_id="t1", text=TRANSCRIPT_TEXT, final=True),
        AstraSpoken(turn_id="t1", utterance_id="u1", text=SPOKEN_1),
        json.dumps({**tool, "status": "proposed"}),
        json.dumps({**tool, "status": "dry_run", "action": "would click at (10, 20) on screen"}),
        surface,
        AstraSpoken(turn_id="t2", utterance_id="u2", text=SPOKEN_FINAL),
    ]


DEGRADED_PROBE_JS = """
(() => {
  const thread = document.querySelector('[data-thread]');
  const info = (turn) => {
    const a = thread ? thread.querySelector('article[data-turn-id="' + turn + '"]') : null;
    if (!a) return null;
    return {
      transcript: [...a.querySelectorAll('[data-transcript]')].map((e) => e.textContent.trim()),
      utterances: [...a.querySelectorAll('[data-utterance-id]')].map((e) => [e.getAttribute('data-utterance-id'), e.textContent.trim()]),
      cards: [...a.querySelectorAll('[data-tool-call-id]')].map((c) => [
        c.getAttribute('data-tool-call-id'),
        (c.querySelector('[data-tool-status]') || {getAttribute: () => null}).getAttribute('data-tool-status'),
      ]),
    };
  };
  return {
    thread: !!thread,
    t1: info('t1'),
    t2: info('t2'),
    wrappers: document.querySelectorAll('[data-ui-surface]').length,
    a2ui: document.querySelectorAll(__A2UI_ROOT__).length,
    body: document.body.innerText || '',
  };
})()
""".replace("__A2UI_ROOT__", json.dumps(A2UI_ROOT))


def _broken_ui_handler(body: str | None, hits: list[str]) -> Any:
    class _BrokenUi(_QuietHandler):
        def do_GET(self) -> None:
            if urllib.parse.urlsplit(self.path).path == UI_INDEX_PATH:
                hits.append(self.path)
                if body is None:
                    self.send_error(404, "absent (T-0154 B11 test)")
                    return
                raw = body.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            super().do_GET()

    return functools.partial(_BrokenUi, directory=str(HUD))


@dataclass
class Degraded:
    probe: dict[str, Any] = field(default_factory=dict)
    ui_hits: list[str] = field(default_factory=list)
    frames_after_script: int = 0
    sent_types: list[str] = field(default_factory=list)
    error: str | None = None


@pytest.fixture(scope="module")
def degraded() -> Any:
    """The three broken-panel sessions in one Chrome. Never raises: errors land per kind."""
    out: dict[str, Degraded] = {kind: Degraded() for kind in UI_BROKEN}
    if CHROME is None:  # pragma: no cover - requires_chrome is the real guard
        for row in out.values():
            row.error = "no chrome"
        yield out
        return
    try:
        with Chrome(CHROME) as chrome:
            for kind, body in UI_BROKEN.items():
                row = out[kind]
                stub = HudStub([]).start()
                server = socketserver.ThreadingTCPServer(
                    ("127.0.0.1", 0), _broken_ui_handler(body, row.ui_hits)
                )
                server.daemon_threads = True
                threading.Thread(target=server.serve_forever, daemon=True).start()
                port = int(server.server_address[1])
                try:
                    url = f"http://127.0.0.1:{port}/?source=canvas&ws=" + urllib.parse.quote(
                        stub.url, safe=""
                    )
                    chrome.navigate(url)
                    stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20)
                    stub.wait_for_client(10)
                    for message in degraded_script():
                        stub.push(message)
                        time.sleep(0.2)
                    frames_at_script = len(stub.frames())
                    chrome.wait_until(
                        "document.querySelectorAll('[data-utterance-id]').length >= 2", 10
                    )
                    stub.wait_for(lambda s, n=frames_at_script: len(s.frames()) > n, 10)
                    value = chrome.evaluate(DEGRADED_PROBE_JS)
                    row.probe = value if isinstance(value, dict) else {"__probe__": value}
                    row.frames_after_script = len(stub.frames()) - frames_at_script
                    row.sent_types = stub.received_types()
                except Exception as exc:  # a broken harness is data, not an error in a row
                    row.error = f"the {kind!r} session failed: {type(exc).__name__}: {exc}"
                finally:
                    server.shutdown()
                    stub.stop()
    except Exception as exc:  # pragma: no cover - Chrome itself did not start
        for row in out.values():
            row.error = row.error or f"headless Chrome failed: {type(exc).__name__}: {exc}"
    yield out


@requires_chrome
@pytest.mark.parametrize("kind", list(UI_BROKEN))
def test_without_a_working_ui_panel_the_thread_renders_text_and_action_cards(degraded, kind):
    """B11 (T-0008 criterion 10 re-pinned): the thread renders the transcript, the utterances and
    the action card in their turns' articles with the ui panel broken, and draws no card."""
    row = degraded[kind]
    assert row.error is None, row.error
    assert row.ui_hits, (
        f"premise ({kind}): the page never requested {UI_INDEX_PATH}, so this session did not "
        "exercise a broken ui panel at all (hud/src/panels.js PANEL_SPECS loads ./ui/index.js)"
    )
    p = row.probe
    assert p.get("thread"), f"B1/B11 ({kind}): the page has no [data-thread]; probe={p!r}"
    t1, t2 = p.get("t1") or {}, p.get("t2") or {}
    assert t1.get("transcript") == [TRANSCRIPT_TEXT], (
        f"B11 ({kind}): t1's user bubble reads the transcript; got {t1!r}"
    )
    assert t1.get("utterances") == [["u1", SPOKEN_1]], (
        f"B11 ({kind}): t1's utterance renders in t1's article; got {t1!r}"
    )
    assert t1.get("cards") == [["d1", "dry_run"]], (
        f"B11 ({kind}): t1's action card renders (dry run) in t1's article; got {t1!r}"
    )
    assert t2.get("utterances") == [["u2", SPOKEN_FINAL]], (
        f"B11 ({kind}): the utterance after the undrawable surface still renders, in t2's "
        f"article; got {t2!r}"
    )
    assert p.get("a2ui") == 0 and DEGRADED_CARD_TEXT not in str(p.get("body")), (
        f"B11 ({kind}): with no working ui panel nothing renders the surface; probe={p!r}"
    )
    assert PLACEHOLDER not in str(p.get("body")), (
        f"B1 ({kind}): the retired placeholder does not come back when the panel is missing"
    )


@requires_chrome
@pytest.mark.parametrize("kind", list(UI_BROKEN))
def test_without_a_working_ui_panel_the_socket_keeps_working(degraded, kind):
    """B11: the socket keeps working — frames keep arriving after the whole script (including a
    surface no panel could take), and the page sends nothing but hello and frames."""
    row = degraded[kind]
    assert row.error is None, row.error
    assert row.ui_hits, f"premise ({kind}): {UI_INDEX_PATH} was never requested"
    assert row.frames_after_script >= 1, (
        f"B11 ({kind}): no frame arrived after the script — the socket or the frame loop died "
        "when the surface arrived with no panel to take it"
    )
    assert row.sent_types.count("astra.hello") == 1, (
        f"B11 ({kind}): one hello (no reconnect loop from a thrown handler); got {row.sent_types}"
    )
    assert set(row.sent_types) <= {"astra.hello", "astra.frame"}, (
        f"B11 ({kind}): the page sent {sorted(set(row.sent_types))}"
    )
