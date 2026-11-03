"""The "Model thoughts" disclosure in the HUD thread, in headless Chrome (docs/tasks/T-0180.md group C).

*Tests to write*, Group C (Playwright/Chrome, via ``HudStub``):

- (a) collapsed by default: no ``open`` attribute, and the body is not visible;
- (b) expands on a real click and with the keyboard (Tab reaches the summary; Enter, Space), showing
  ``COMMITTED_SENTINEL``;
- (c) never shows draft text: ``astra.canvas.step`` frames carrying ``DRAFT_SENTINEL`` are sent before and
  between the thoughts; ``DRAFT_SENTINEL`` is in the denoising root and not in ``[data-model-thoughts]``;
- (d) after a reload with replayed messages, the blocks are closed (H-T180-2);
- (e) contrast (the body text at >= 4.5:1, C8) and no horizontal overflow at 1280, 1440 and 1920, both themes;

plus C2 in the browser (an update keeps the viewer's open or closed state), C8's 24 rem scroll box and
``white-space: pre-wrap`` (C3), and C9's second half (the denoise digests are unchanged by interleaved
thoughts, a negative that runs ungated).

Regime: the box's Chrome (or Playwright's pinned Chromium) headless over CDP, the T-0177 C rig shape
(``tests/integration/test_t0177_hud_vision.py``: one module fixture drives the page through every phase and
records probes; each test asserts on one), 1440x900 unless a row says otherwise, the stub Conductor
``tests/fakes/hud_server.HudStub`` (no Session: every ``astra.thought`` is A1's real ``AstraThought`` model,
encoded by the real codec). A click is a real mouse event (``Input.dispatchMouseEvent``); a key a real
``Input.dispatchKeyEvent``. Nothing is measured.

**The DOM contract this module pins** (the hooks of ``tests/hud/thoughts.test.mjs``):

==========================================  =====================================================================
``details[data-model-thoughts]``            one per turn with thoughts, in the turn's article; never ``open`` by
                                            the page
``[data-model-thoughts] > summary``         the toggle the viewer clicks or focuses
``[data-thought-attempt]``                  one block per attempt
``[data-thought-text]``                     the committed text (``white-space: pre-wrap``)
``[data-thought-status]``, ``-omitted``     the status line, the omitted line
``[data-denoise-root]``                     the Thinking pane's denoising view (T-0128c, unchanged)
==========================================  =====================================================================

Readings (stated, not guessed silently):

- "the body is not visible" is read as: ``[data-thought-text]`` fails ``checkVisibility()`` or has an empty box.
- "Muted body text at >= 4.5:1" is applied to ``[data-thought-text]`` at a flat 4.5:1 whatever its size (C8
  says 4.5 without the large-text allowance); the summary, label, status and omitted lines get T-0056's floor
  (4.5:1, 3.0:1 large).
- "The block scrolls inside a 24 rem max-height" is read as: the nearest element at or above
  ``[data-thought-text]`` inside the disclosure whose ``overflow-y`` is ``auto`` or ``scroll`` has a computed
  ``max-height`` of 24 x the root font size, and with a 16000-character thought it really scrolls
  (``scrollHeight > clientHeight``).
- "never overflows sideways" is read for the document (``scrollWidth <= clientWidth``) and for the thread
  element ``[data-thread]``, with a 600-character unbroken token in the open block.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import pytest

from local_astra.contracts.protocol import AstraSpoken, AstraTranscript
from local_astra.hud_static import HudStaticServer
from tests.fakes.hud_server import HudStub
from tests.integration.test_hud_chrome import (
    _contrast_report,
    _describe,
    _partition,
    choose_theme_js,
    measure_js,
    ratio_of,
)
from tests.integration.test_hud_headless import CHROME, HUD, requires_chrome
from tests.integration.test_hud_redesign import _panels_line, _Recorder

pytestmark = pytest.mark.integration

COMMITTED_SENTINEL = "COMMITTEDSENTINEL"
DRAFT_SENTINEL = "DRAFTSENTINEL"
WIDTHS = (1280, 1440, 1920)
THEMES = ("dark", "light")
SUMMARY = "[data-model-thoughts] > summary"

#: C8: the body text measured at a flat 4.5:1; the other lines at their WCAG floor.
BODY_SPEC = [("thought text", "[data-model-thoughts] [data-thought-text]", None)]
OTHER_SPECS = [
    ("thought summary", SUMMARY, None),
    ("thought label", "[data-model-thoughts] [data-thought-label]", None),
    ("thought status", "[data-model-thoughts] [data-thought-status]", None),
    ("thought omitted", "[data-model-thoughts] [data-thought-omitted]", None),
]


def _thought(
    turn_id: str,
    attempt_id: str,
    *,
    step: int = 1,
    kind: str = "step",
    text: str,
    end: str = "open",
    omitted_chars: int = 0,
    replay: bool = False,
) -> Any:
    from local_astra.contracts import protocol

    model = protocol.AstraThought  # T-0180 A1 (merged)
    return model(
        turn_id=turn_id,
        request_id=f"{attempt_id}-req",
        attempt_id=attempt_id,
        step=step,
        kind=kind,
        text=text,
        omitted_chars=omitted_chars,
        end=end,
        replay=replay,
    )


def _canvas_step(request_id: str, i: int, piece: str) -> str:
    """One denoise ``astra.canvas.step`` with per-position pieces (T-0159), the draft in position 0."""
    n = 4
    return json.dumps(
        {
            "type": "astra.canvas.step",
            "request_id": request_id,
            "canvas_idx": 0,
            "step": i + 1,
            "max_steps": 16,
            "canvas_length": n,
            "canvas": [100 + i + k for k in range(n)],
            "argmax": [200 + i + k for k in range(n)],
            "stable": "0f" if i % 2 else "03",
            "phase": "denoise",
            "segment": "thought",
            "text_preview": f"{piece} preview",
            "argmax_pieces": [piece, " a", " b", " c"],
        }
    )


#: Every probe: the disclosure's state as a viewer would see it, and the sentinels by region.
PROBE_JS = """
(() => {
  const vis = (el) => !!el && el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true})
                      && el.getBoundingClientRect().width > 0 && el.getBoundingClientRect().height > 0;
  const ds = [...document.querySelectorAll('[data-model-thoughts]')];
  const d = ds[0] || null;
  const texts = d ? [...d.querySelectorAll('[data-thought-text]')] : [];
  const denoise = document.querySelector('[data-denoise-root]');
  const sq = (s) => String(s || '').replace(/\\s+/g, ' ').trim();
  return {
    count: ds.length,
    tag: d ? d.localName : null,
    open_attr: d ? d.hasAttribute('open') : null,
    open_prop: d ? !!d.open : null,
    any_open: ds.some((x) => x.hasAttribute('open')),
    blocks: d ? d.querySelectorAll('[data-thought-attempt]').length : 0,
    text_visible: texts.map(vis),
    visible_text: texts.filter(vis).map((t) => t.innerText).join('\\n'),
    thoughts_text: ds.map((x) => x.textContent).join('\\n'),
    summary: d && d.querySelector(':scope > summary') ? sq(d.querySelector(':scope > summary').textContent) : null,
    summary_visible: d && d.querySelector(':scope > summary') ? vis(d.querySelector(':scope > summary')) : null,
    active_is_summary: !!d && document.activeElement === d.querySelector(':scope > summary'),
    denoise_text: denoise ? denoise.textContent : null,
    denoise_frames: denoise ? denoise.getAttribute('data-denoise-frames') : null,
    denoise_digest: denoise ? denoise.getAttribute('data-denoise-digest') : null,
    thread_has_draft: (document.querySelector('[data-thread]') || {textContent: ''}).textContent.includes('DRAFTSENTINEL'),
    theme: document.documentElement.getAttribute('data-theme'),
  };
})()
"""

#: C8: layout at the current viewport, with the first disclosure open.
LAYOUT_JS = """
(() => {
  const d = document.querySelector('[data-model-thoughts]');
  const text = d ? d.querySelector('[data-thought-text]') : null;
  let box = null;
  for (let n = text; n && d && d.contains(n); n = n.parentElement) {
    const oy = getComputedStyle(n).overflowY;
    if (oy === 'auto' || oy === 'scroll') { box = n; break; }
  }
  const rootPx = parseFloat(getComputedStyle(document.documentElement).fontSize);
  const thread = document.querySelector('[data-thread]');
  const de = document.documentElement;
  return {
    open: !!d && d.hasAttribute('open'),
    white_space: text ? getComputedStyle(text).whiteSpace : null,
    root_px: rootPx,
    box: !!box,
    box_max_height: box ? getComputedStyle(box).maxHeight : null,
    box_scrolls: box ? box.scrollHeight > box.clientHeight + 1 : null,
    box_client_height: box ? box.clientHeight : null,
    doc_scroll_width: de.scrollWidth, doc_client_width: de.clientWidth,
    thread_scroll_width: thread ? thread.scrollWidth : null,
    thread_client_width: thread ? thread.clientWidth : null,
  };
})()
"""


@dataclass
class Run:
    error: str | None = None
    probes: dict[str, Any] = field(default_factory=dict)
    layout: dict[str, Any] = field(default_factory=dict)
    contrast: dict[str, Any] = field(default_factory=dict)
    tab_steps: int | None = None
    anchor: dict[str, Any] = field(default_factory=dict)
    pins: dict[str, Any] = field(default_factory=dict)
    requests: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _click(chrome: Any, selector: str) -> bool:
    """A real mouse click (Input.dispatchMouseEvent), not a synthesised DOM event."""
    got = chrome.evaluate(
        "(() => { const e = document.querySelector(" + json.dumps(selector) + ");"
        " if (!e) return null; e.scrollIntoView({block: 'center'}); const r = e.getBoundingClientRect();"
        " return [r.left + Math.min(20, r.width / 2), r.top + r.height / 2]; })()"
    )
    if not isinstance(got, list):
        return False
    x, y = float(got[0]), float(got[1])
    for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
        params: dict[str, Any] = {"type": kind, "x": x, "y": y}
        if kind != "mouseMoved":
            params.update(button="left", clickCount=1)
        chrome._cmd("Input.dispatchMouseEvent", **params)
    return True


def _summary_of(turn_id: str) -> str:
    return f'article[data-turn-id="{turn_id}"] [data-model-thoughts] > summary'


#: C-R1: where a summary is on screen, against the thread's box and its scroll state.
GEOM_JS = """
((sel) => {
  const s = document.querySelector(sel);
  const t = document.querySelector('[data-thread]');
  if (!s || !t) return {missing: !s ? 'summary' : 'thread'};
  const r = s.getBoundingClientRect(), tr = t.getBoundingClientRect();
  return {
    top: r.top, bottom: r.bottom, left: r.left, width: r.width, height: r.height,
    thread_top: tr.top, thread_bottom: tr.bottom,
    inside: r.top >= tr.top - 0.5 && r.bottom <= tr.bottom + 0.5,
    open: s.parentElement.hasAttribute('open'),
    scroll_top: t.scrollTop, scroll_height: t.scrollHeight, client_height: t.clientHeight,
    from_bottom: t.scrollHeight - t.scrollTop - t.clientHeight,
    focused: document.activeElement === s,
  };
})
"""


def _geom(chrome: Any, selector: str) -> dict[str, Any]:
    got = chrome.evaluate(GEOM_JS + "(" + json.dumps(selector) + ")")
    return got if isinstance(got, dict) else {"__geom__": got}


def _click_in_place(chrome: Any, selector: str) -> bool:
    """A real mouse click at the element's current rect: nothing scrolls it into view first (C-R1)."""
    g = _geom(chrome, selector)
    if "top" not in g:
        return False
    x = float(g["left"]) + min(20.0, float(g["width"]) / 2)
    y = float(g["top"]) + float(g["height"]) / 2
    for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
        params: dict[str, Any] = {"type": kind, "x": x, "y": y}
        if kind != "mouseMoved":
            params.update(button="left", clickCount=1)
        chrome._cmd("Input.dispatchMouseEvent", **params)
    return True


def _mounted(chrome: Any, timeout_s: float = 25.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and _panels_line(chrome.console[chrome._mark :]) is None:
        chrome.drain(0.1)
    return _panels_line(chrome.console[chrome._mark :]) is not None


def _drive(result: Run, chrome: Any, stub: HudStub, port: int) -> None:
    ws = urllib.parse.quote(stub.url, safe="")
    url = f"http://127.0.0.1:{port}/?ws={ws}"
    probes = result.probes

    def frame() -> None:
        chrome.evaluate(
            "new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))"
        )

    def probe(name: str) -> dict[str, Any]:
        frame()
        got = chrome.evaluate(PROBE_JS)
        probes[name] = got if isinstance(got, dict) else {"__probe__": got}
        return probes[name]

    def push(msg: Any, settle: float = 0.05) -> None:
        stub.push(msg)
        chrome.drain(settle)

    def load() -> None:
        hellos = stub.received_types().count("astra.hello")
        chrome.navigate(url)
        if not _mounted(chrome):
            raise RuntimeError("the page never logged 'panels mounted:'")
        if not stub.wait_for(lambda s: s.received_types().count("astra.hello") > hellos, 15):
            raise RuntimeError(f"no astra.hello; the stub saw {stub.received_types()!r}")
        stub.wait_for_client(10)
        chrome.drain(0.3)

    def key(name: str) -> None:
        if name == "Enter":
            chrome.key("Enter", text="\r", code=13)
        elif name == " ":
            chrome.key(" ", text=" ", code=32)
        else:
            chrome.key("Tab", code=9)
        chrome.drain(0.1)

    chrome.viewport(1440, 900)
    load()

    # --- (a)/(c): drafts before and between the thoughts; collapsed ------------------------------------
    for i in range(3):
        push(_canvas_step("t1-a1-req", i, DRAFT_SENTINEL))
    push(AstraTranscript(turn_id="t1", text="What is on the page?", final=True))
    push(AstraSpoken(turn_id="t1", utterance_id="t1-u1", text="A search box and a button."))
    push(_thought("t1", "t1-a1", text="I see a box.", end="open"))
    for i in range(3, 6):
        push(_canvas_step("t1-a1-req", i, DRAFT_SENTINEL))
    push(
        _thought(
            "t1", "t1-a1", text=f"I see a box.\n{COMMITTED_SENTINEL} then click it.", end="done"
        )
    )
    push(_canvas_step("t1-a2-req", 0, DRAFT_SENTINEL))
    push(_thought("t1", "t1-a2", step=2, text="Second step thought.", end="done"), settle=0.4)
    probe("collapsed")

    # --- (b): a real click opens and closes it ----------------------------------------------------------
    _click(chrome, SUMMARY)
    chrome.drain(0.2)
    probe("clicked")
    _click(chrome, SUMMARY)
    chrome.drain(0.2)
    probe("click_closed")

    # --- (b): the keyboard: Tab reaches the summary, Enter opens, Space closes ---------------------------
    chrome.evaluate(
        "(() => { document.activeElement && document.activeElement.blur(); window.scrollTo(0, 0); return true; })()"
    )
    for n in range(1, 121):
        key("Tab")
        if chrome.evaluate(
            "document.activeElement === document.querySelector(" + json.dumps(SUMMARY) + ")"
        ):
            result.tab_steps = n
            break
    probe("tabbed")
    key("Enter")
    probe("enter")
    key(" ")
    probe("space")

    # --- C2: an update keeps the viewer's state, closed and open ------------------------------------------
    push(_thought("t1", "t1-a3", step=3, text="Third step, open.", end="open"), settle=0.3)
    probe("update_closed")
    _click(chrome, SUMMARY)
    chrome.drain(0.2)
    push(_thought("t1", "t1-a3", step=3, text="Third step, finished.", end="done"))
    push(_thought("t1", "t1-a4", step=4, text="Fourth.", end="error"), settle=0.3)
    probe("update_open")

    # --- (d): reload; the hello's replay; everything closed -----------------------------------------------
    load()
    for attempt, step, text in (
        ("t1-a1", 1, f"I see a box.\n{COMMITTED_SENTINEL} then click it."),
        ("t1-a2", 2, "Second step thought."),
        ("t1-a3", 3, "Third step, finished."),
    ):
        push(_thought("t1", attempt, step=step, text=text, end="done", replay=True))
    chrome.drain(0.4)
    probe("reloaded")
    # C1 in the browser: the reloaded turn carries thoughts and nothing else (no transcript, no
    # answer), so its disclosure must still be on screen and open on a click. The thread hides an
    # assistant row with no answer, meta, card or tool card (T-0154); a thought-only row is not empty.
    _click(chrome, SUMMARY)
    chrome.drain(0.2)
    probe("reloaded_clicked")

    # --- C9: the denoise digests with and without interleaved thoughts ------------------------------------
    for name, with_thoughts in (("denoise_plain", False), ("denoise_mixed", True)):
        load()
        for i in range(6):
            if with_thoughts:
                push(_thought("t5", "t5-a1", text=f"thinking {i}", end="open"))
            push(_canvas_step("t5-a1-req", i, f"piece{i}"))
        if with_thoughts:
            push(_thought("t5", "t5-a1", text="thinking done", end="done"))
        chrome.drain(0.6)
        probe(name)

    # --- (e)/C8: a long thought, open, at three widths and both themes ------------------------------------
    load()
    unbroken = "x" * 600
    long_text = (unbroken + "\n" + "word " * 3080)[:16000]
    push(AstraSpoken(turn_id="t9", utterance_id="t9-u1", text="Done."))
    push(_thought("t9", "t9-a1", text=long_text, end="done", omitted_chars=1234))
    push(_thought("t9", "t9-a2", step=2, text="Short and still thinking.", end="open"), settle=0.4)
    _click(chrome, SUMMARY)
    chrome.drain(0.3)
    for width in WIDTHS:
        chrome.viewport(width, 900)
        for theme in THEMES:
            chrome.evaluate(choose_theme_js(theme))
            chrome.drain(0.2)
            frame()
            result.layout[f"{width}_{theme}"] = chrome.evaluate(LAYOUT_JS) or {}
            if width == 1440:
                result.contrast[f"body_{theme}"] = chrome.evaluate(measure_js(BODY_SPEC)) or {}
                result.contrast[f"other_{theme}"] = chrome.evaluate(measure_js(OTHER_SPECS)) or {}
    chrome.evaluate(choose_theme_js("dark"))

    # --- C-R1: a viewer's toggle never moves the reader ---------------------------------------------------
    chrome.viewport(1440, 900)
    load()
    answer = "A longer answer that wraps over a few lines of the thread. " * 10
    for n in range(1, 7):
        tid = f"r{n}"
        push(AstraTranscript(turn_id=tid, text=f"Question number {n}?", final=True))
        push(AstraSpoken(turn_id=tid, utterance_id=f"{tid}-u1", text=answer))
        if n == 6:
            for k in range(1, 4):
                body = (f"Block {k}. " + "a long committed thought " * 160)[:4000]
                push(_thought(tid, f"{tid}-a{k}", step=k, text=body, end="done"))
        elif n == 5:
            push(_thought(tid, f"{tid}-a1", text=("older thought " * 300)[:4000], end="done"))
        else:
            push(_thought(tid, f"{tid}-a1", text=f"Thought of turn {n}.", end="done"))
    chrome.drain(0.4)

    def pin() -> None:
        chrome.evaluate(
            "(() => { const t = document.querySelector('[data-thread]');"
            " t.scrollTop = t.scrollHeight; return true; })()"
        )
        chrome.drain(0.2)
        frame()

    def toggle(name: str, selector: str, how: str) -> None:
        frame()
        before = _geom(chrome, selector)
        if how == "click":
            _click_in_place(chrome, selector)
        else:
            chrome.evaluate(
                "(() => { const s = document.querySelector(" + json.dumps(selector) + ");"
                " s.focus({preventScroll: true}); return document.activeElement === s; })()"
            )
            before = _geom(chrome, selector)
            chrome.key("Enter", text="\r", code=13)
        chrome.drain(0.25)
        frame()
        frame()
        result.anchor[name] = {"before": before, "after": _geom(chrome, selector)}

    # An older turn's disclosure, the reader pinned at the bottom.
    pin()
    toggle("older_pinned_open", _summary_of("r5"), "click")
    toggle("older_pinned_close", _summary_of("r5"), "click")
    # The last turn's disclosure, pinned: a real click, then Enter twice with focus on the summary.
    pin()
    toggle("last_click_open", _summary_of("r6"), "click")
    toggle("last_enter_close", _summary_of("r6"), "enter")
    toggle("last_enter_open", _summary_of("r6"), "enter")
    # An older turn's disclosure, the reader scrolled up to it (not pinned).
    chrome.evaluate(
        "(() => { const t = document.querySelector('[data-thread]');"
        " const s = document.querySelector(" + json.dumps(_summary_of("r2")) + ");"
        " const tr = t.getBoundingClientRect(), r = s.getBoundingClientRect();"
        " t.scrollTop += r.top - (tr.top + t.clientHeight / 2); return true; })()"
    )
    chrome.drain(0.2)
    toggle("older_unpinned_open", _summary_of("r2"), "click")

    _drive_c_r2(result, push, load, pin, frame, chrome)


def _drive_c_r2(result: Run, push: Any, load: Any, pin: Any, frame: Any, chrome: Any) -> None:
    """C-R2: a toggle does not unpin. Each scenario starts from a fresh page: six turns, a thread taller
    than its viewport, the reader pinned at the bottom by setting ``scrollTop`` (the B9 premise), then a
    real click at the summary's current rect (no ``scrollIntoView``)."""
    answer = "A longer answer that wraps over a few lines of the thread. " * 10
    rows = result.pins

    def turn(tid: str, *, end: str = "done") -> None:
        # Back to back, no drain between (a drain costs up to one 0.2 s CDP read); settle() follows.
        push(AstraTranscript(turn_id=tid, text=f"Question {tid}?", final=True), 0)
        push(AstraSpoken(turn_id=tid, utterance_id=f"{tid}-u1", text=answer), 0)
        body = (f"Thought of {tid}. " + "a long committed thought " * 160)[:4000]
        push(_thought(tid, f"{tid}-a1", text=body, end=end), 0)

    def settle() -> None:
        chrome.drain(0.3)
        frame()
        frame()

    def fresh(*, live_last: bool = False) -> None:
        load()
        for n in range(1, 7):
            turn(f"r{n}", end="open" if live_last and n == 6 else "done")
        chrome.drain(0.4)
        pin()

    def has_turn(tid: str) -> bool:
        return bool(
            chrome.evaluate(
                "!!document.querySelector('article[data-turn-id=" + json.dumps(tid) + "]')"
            )
        )

    def new_turn_after_toggle(name: str, toggled: str) -> None:
        fresh()
        sel = _summary_of(toggled)
        frame()
        before = _geom(chrome, sel)
        _click_in_place(chrome, sel)
        settle()
        toggled_geom = _geom(chrome, sel)
        had = has_turn("r7")
        turn("r7")
        settle()
        rows[name] = {
            "before": before,
            "toggled": toggled_geom,
            "after": _geom(chrome, sel),
            "had_new_turn_before": had,
            "new_turn": has_turn("r7"),
        }

    # (1) the last turn's disclosure opened, then a new turn; (2) an older turn's, left open.
    new_turn_after_toggle("open_last_then_new_turn", "r6")
    new_turn_after_toggle("open_older_then_new_turn", "r5")

    # (3) the live turn's disclosure opened, then thought updates to that same turn.
    fresh(live_last=True)
    sel = _summary_of("r6")
    frame()
    before = _geom(chrome, sel)
    _click_in_place(chrome, sel)
    settle()
    toggled_geom = _geom(chrome, sel)
    body = (
        "Thought of r6. "
        + "a long committed thought " * 160
        + "UPDATEDSENTINEL and the thought goes on growing. " * 20
    )
    push(_thought("r6", "r6-a1", text=body[:4000], end="done"), 0.05)
    push(_thought("r6", "r6-a2", step=2, text="Second step, UPDATEDSENTINEL. " * 30, end="open"))
    settle()
    rows["live_turn_thought_update"] = {
        "before": before,
        "toggled": toggled_geom,
        "after": _geom(chrome, sel),
        "updated": bool(
            chrome.evaluate(
                "(document.querySelector('article[data-turn-id=\"r6\"] [data-model-thoughts]')"
                " || {textContent: ''}).textContent.includes('UPDATEDSENTINEL')"
            )
        ),
    }

    # (4) a reader scrolled up (not pinned): a new turn with no toggle, then a toggle and another turn.
    fresh()
    sel = _summary_of("r2")
    chrome.evaluate(
        "(() => { const t = document.querySelector('[data-thread]');"
        " const s = document.querySelector(" + json.dumps(sel) + ");"
        " const tr = t.getBoundingClientRect(), r = s.getBoundingClientRect();"
        " t.scrollTop += r.top - (tr.top + t.clientHeight / 2); return true; })()"
    )
    chrome.drain(0.2)
    frame()
    before = _geom(chrome, sel)
    turn("r7")
    settle()
    rows["unpinned_new_turn"] = {
        "before": before,
        "after": _geom(chrome, sel),
        "new_turn": has_turn("r7"),
    }
    _click_in_place(chrome, sel)
    settle()
    before = _geom(chrome, sel)
    turn("r8")
    settle()
    rows["unpinned_toggle_then_new_turn"] = {
        "before": before,
        "after": _geom(chrome, sel),
        "new_turn": has_turn("r8"),
    }

    # (5) pinned, the last summary toggled open and closed again inside one task (two ``click()`` calls,
    # so no resize separates them and nothing re-reads the reader), then a new turn: still pinned.
    fresh()
    sel = _summary_of("r6")
    frame()
    before = _geom(chrome, sel)
    chrome.evaluate(
        "(() => { const s = document.querySelector(" + json.dumps(sel) + ");"
        " s.click(); s.click(); return true; })()"
    )
    settle()
    toggled_geom = _geom(chrome, sel)
    turn("r7")
    settle()
    rows["double_toggle_then_new_turn"] = {
        "before": before,
        "toggled": toggled_geom,
        "after": _geom(chrome, sel),
        "new_turn": has_turn("r7"),
    }

    _drive_c_r3(rows, push, fresh, turn, settle, has_turn, frame, chrome)


R1_GROWTH = "R1GROWTHSENTINEL, an utterance added to the first turn. " * 12
R6_GROWTH = "R6GROWTHSENTINEL, the live turn keeps talking. " * 12


def _text_in(chrome: Any, turn_id: str, needle: str) -> bool:
    return bool(
        chrome.evaluate(
            "(document.querySelector('article[data-turn-id=" + json.dumps(turn_id) + "]')"
            " || {textContent: ''}).textContent.includes(" + json.dumps(needle) + ")"
        )
    )


def _wheel_to_bottom(chrome: Any, frame: Any, tries: int = 12) -> int:
    """The reader scrolls the thread back to its bottom with a real mouse wheel
    (``Input.dispatchMouseEvent`` ``mouseWheel`` over the thread), never by setting ``scrollTop``.
    Returns how many wheel events it took (``tries + 1`` if it never got there)."""
    for n in range(1, tries + 1):
        pt = chrome.evaluate(
            "(() => { const r = document.querySelector('[data-thread]').getBoundingClientRect();"
            " return [r.left + r.width / 2, r.top + r.height / 2]; })()"
        )
        x, y = float(pt[0]), float(pt[1])
        chrome._cmd("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
        chrome._cmd("Input.dispatchMouseEvent", type="mouseWheel", x=x, y=y, deltaX=0, deltaY=1200)
        chrome.drain(0.15)
        frame()
        g = _geom(chrome, "[data-thread]")
        if "from_bottom" in g and g["from_bottom"] <= PINNED_PX:
            return n
    return tries + 1


def _drive_c_r3(
    rows: dict[str, Any],
    push: Any,
    fresh: Any,
    turn: Any,
    settle: Any,
    has_turn: Any,
    frame: Any,
    chrome: Any,
) -> None:
    """C-R3: only the reader's own input releases the hold (docs/tasks/T-0180.md, group C review rulings)."""
    # (1) pinned, a real click opens r6's thoughts (held); an utterance grows r1, far above the viewport
    # (Chrome's scroll anchoring answers with a scroll the reader did not make); then a new turn r7.
    fresh()
    sel = _summary_of("r6")
    frame()
    before = _geom(chrome, sel)
    _click_in_place(chrome, sel)
    settle()
    toggled_geom = _geom(chrome, sel)
    r1_top = _geom(chrome, _summary_of("r1"))
    push(AstraSpoken(turn_id="r1", utterance_id="r1-u2", text=R1_GROWTH), 0)
    settle()
    grown = _geom(chrome, sel)
    had = has_turn("r7")
    turn("r7")
    settle()
    rows["r3_growth_above_then_new_turn"] = {
        "before": before,
        "toggled": toggled_geom,
        "r1_before_growth": r1_top,
        "grown": grown,
        "r1_grew": _text_in(chrome, "r1", "R1GROWTHSENTINEL"),
        "after": _geom(chrome, sel),
        "had_new_turn_before": had,
        "new_turn": has_turn("r7"),
    }

    # (2) pinned with r6 live, a real click opens r6's thoughts (held); the reader wheels back to the
    # bottom by hand; then the live turn grows (an utterance and a thought update to r6, no new article).
    fresh(live_last=True)
    frame()
    before = _geom(chrome, sel)
    _click_in_place(chrome, sel)
    settle()
    toggled_geom = _geom(chrome, sel)
    wheels = _wheel_to_bottom(chrome, frame)
    settle()
    wheeled = _geom(chrome, sel)
    push(AstraSpoken(turn_id="r6", utterance_id="r6-u2", text=R6_GROWTH), 0)
    push(
        _thought("r6", "r6-a2", step=2, text="Second step, R6GROWTHSENTINEL. " * 30, end="open"),
        0,
    )
    settle()
    rows["r3_wheel_back_then_live_growth"] = {
        "before": before,
        "toggled": toggled_geom,
        "wheels": wheels,
        "wheeled": wheeled,
        "r6_grew": _text_in(chrome, "r6", "R6GROWTHSENTINEL"),
        "after": _geom(chrome, sel),
        "new_turn": has_turn("r7"),
    }

    # (3) pinned, r6's thoughts opened (held), then the reader presses PageUp three times with the
    # browser's own key pipeline; ~0.8 s later a new turn r7 arrives. The keyboard scroll is the
    # reader's own input: they stay where they scrolled. Focus varies: on the summary after an
    # Enter-open, on the summary after a click-open, and on <body> after a click on plain thread text.
    for focus in R3_KEY_FOCUS:
        fresh()
        frame()
        before = _geom(chrome, sel)
        if focus == "enter_open":
            chrome.evaluate(
                "(() => { const s = document.querySelector(" + json.dumps(sel) + ");"
                " s.focus({preventScroll: true}); return true; })()"
            )
            chrome.key("Enter", text="\r", code=13)
        else:
            _click_in_place(chrome, sel)
        settle()
        toggled_geom = _geom(chrome, sel)
        clicked_text = None
        if focus == "text_click":
            clicked_text = _click_plain_text(chrome)
            settle()
        active = chrome.evaluate(
            "(() => { const a = document.activeElement; return a ? a.tagName : null; })()"
        )
        for _ in range(3):
            _page_up(chrome)
            chrome.drain(0.05)
        chrome.drain(0.8)
        frame()
        scrolled = _geom(chrome, sel)
        had = has_turn("r7")
        turn("r7")
        settle()
        rows[f"r3_page_up_{focus}"] = {
            "before": before,
            "toggled": toggled_geom,
            "clicked_text": clicked_text,
            "active": active,
            "scrolled": scrolled,
            "after": _geom(chrome, sel),
            "had_new_turn_before": had,
            "new_turn": has_turn("r7"),
        }

    def grow_r1_soon_after(keyed_at: float) -> dict[str, Any]:
        """The anchoring stimulus of (1), pushed inside the 600 ms window that follows a scroll key:
        an utterance grows r1 far above the viewport, so Chrome's scroll anchoring fires a scroll the
        reader did not make. If the keys before it had been counted as the reader's scroll, that
        anchoring scroll would release the hold and unpin them, and the new turn would not stick."""
        r1 = _geom(chrome, _summary_of("r1"))
        pushed_after = time.monotonic() - keyed_at
        push(AstraSpoken(turn_id="r1", utterance_id="r1-u2", text=R1_GROWTH), 0)
        settle()
        grown = _geom(chrome, sel)
        had = has_turn("r7")
        turn("r7")
        settle()
        return {
            "r1_before_growth": r1,
            "pushed_after_s": pushed_after,
            "r1_grew": _text_in(chrome, "r1", "R1GROWTHSENTINEL"),
            "grown": grown,
            "had_new_turn_before": had,
            "after": _geom(chrome, sel),
            "new_turn": has_turn("r7"),
        }

    # (4) pinned, the reader focuses r6's summary and presses Space: it toggles open (a toggle, not a
    # scroll); within 600 ms r1 grows above the viewport (the anchoring stimulus of (1)); then a new
    # turn r7 arrives and, the hold notwithstanding, sticks them to the bottom.
    fresh()
    frame()
    before = _geom(chrome, sel)
    chrome.evaluate(
        "(() => { const s = document.querySelector(" + json.dumps(sel) + ");"
        " s.focus({preventScroll: true}); return true; })()"
    )
    focused = _geom(chrome, sel)
    keyed_at = time.monotonic()
    chrome.key(" ", text=" ", code=32)
    chrome.drain(0.05)
    frame()
    toggled_geom = _geom(chrome, sel)
    rows["r3_space_open_then_new_turn"] = {
        "before": before,
        "focused": focused,
        "toggled": toggled_geom,
        **grow_r1_soon_after(keyed_at),
    }

    # (5) pinned, a real click opens r6's thoughts (held); the reader focuses the ask box and types
    # Space and ArrowUp into it (text, not scrolling); within 600 ms r1 grows above the viewport;
    # then a new turn r7 arrives and sticks them to the bottom.
    fresh()
    frame()
    before = _geom(chrome, sel)
    _click_in_place(chrome, sel)
    settle()
    toggled_geom = _geom(chrome, sel)
    focused_input = chrome.evaluate(
        "(() => { const i = document.querySelector('[data-ask-input]');"
        " if (!i) return null; i.focus({preventScroll: true});"
        " return {active: document.activeElement === i, disabled: i.disabled}; })()"
    )
    keyed_at = time.monotonic()
    chrome.key(" ", text=" ", code=32)
    for kind in ("rawKeyDown", "keyUp"):
        chrome._cmd(
            "Input.dispatchKeyEvent",
            type=kind,
            key="ArrowUp",
            code="ArrowUp",
            windowsVirtualKeyCode=38,
            nativeVirtualKeyCode=38,
        )
    chrome.drain(0.05)
    typed = chrome.evaluate(
        "(() => { const i = document.querySelector('[data-ask-input]');"
        " return {value: i.value, active: document.activeElement === i}; })()"
    )
    keyed_geom = _geom(chrome, sel)
    rows["r3_keys_in_the_ask_box_then_new_turn"] = {
        "before": before,
        "toggled": toggled_geom,
        "focused_input": focused_input,
        "typed": typed,
        "keyed": keyed_geom,
        **grow_r1_soon_after(keyed_at),
    }


#: C-R3 keyboard rows: where the focus is when the reader presses PageUp.
R3_KEY_FOCUS = ("enter_open", "click_open", "text_click")


def _page_up(chrome: Any) -> None:
    """One PageUp through the browser's own key pipeline (``Input.dispatchKeyEvent``); no character."""
    for kind in ("rawKeyDown", "keyUp"):
        chrome._cmd(
            "Input.dispatchKeyEvent",
            type=kind,
            key="PageUp",
            code="PageUp",
            windowsVirtualKeyCode=33,
            nativeVirtualKeyCode=33,
        )


def _click_plain_text(chrome: Any) -> dict[str, Any] | None:
    """A real mouse click on spoken text inside the thread's viewport (an utterance, never a summary or a
    disclosure body). Returns where it clicked, or None when no utterance was on screen."""
    got = chrome.evaluate(
        "(() => { const t = document.querySelector('[data-thread]'); const tr = t.getBoundingClientRect();"
        " const us = Array.from(t.querySelectorAll('[data-utterance-id]')).reverse();"
        " for (const u of us) { if (u.closest('details')) continue; const r = u.getBoundingClientRect();"
        "  const y = Math.max(r.top, tr.top) + 8; if (r.height < 4 || y > Math.min(r.bottom, tr.bottom) - 4) continue;"
        "  return {id: u.getAttribute('data-utterance-id'), x: r.left + Math.min(40, r.width / 2), y: y}; }"
        " return null; })()"
    )
    if not isinstance(got, dict):
        return None
    x, y = float(got["x"]), float(got["y"])
    for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
        params: dict[str, Any] = {"type": kind, "x": x, "y": y}
        if kind != "mouseMoved":
            params.update(button="left", clickCount=1)
        chrome._cmd("Input.dispatchMouseEvent", **params)
    return got


@pytest.fixture(scope="module")
def run() -> Any:
    if CHROME is None:  # pragma: no cover - requires_chrome is the real guard
        yield Run(error="no chrome")
        return
    result = Run()
    started = time.monotonic()
    stub = HudStub([]).start()
    server = HudStaticServer(HUD, 0)
    try:
        with _Recorder(CHROME) as chrome:
            chrome._cmd("Runtime.enable")
            chrome._cmd("Network.enable")
            try:
                _drive(result, chrome, stub, server.port)
            finally:
                chrome.drain(0.3)
                result.requests = list(chrome.requests)
                result.exceptions = list(chrome.exceptions)
    except Exception as exc:  # a broken harness is data, not an error inside a test
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        server.close()
        stub.stop()
    result.seconds = time.monotonic() - started
    print(f"[T-0180 C] headless session wall time {result.seconds:.1f} s")
    yield result


def _p(run: Run, name: str) -> dict[str, Any]:
    assert run.error is None, f"the T-0180 C headless session failed: {run.error}"
    got = run.probes.get(name)
    assert isinstance(got, dict) and "__probe__" not in got, f"probe {name!r} broken: {got!r}"
    return got


# --- (a) collapsed by default ---------------------------------------------------------------------------


@requires_chrome
def test_a_collapsed_by_default_no_open_attribute_and_the_body_is_not_visible(run):
    p = _p(run, "collapsed")
    assert p["count"] == 1 and p["tag"] == "details", (
        f"C1: one native <details data-model-thoughts> for the turn; found {p['count']} ({p['tag']})"
    )
    assert p["blocks"] == 2, f"C3: two attempts, two blocks; found {p['blocks']}"
    assert p["open_attr"] is False and p["open_prop"] is False, "C2: collapsed by default"
    assert p["text_visible"] and not any(p["text_visible"]), (
        f"(a): the thought body is not visible while collapsed: {p['text_visible']}"
    )
    assert COMMITTED_SENTINEL not in (p["visible_text"] or ""), "(a): nothing of the thought shows"
    assert p["summary"] and p["summary"].replace("▸", "").replace("▾", "").strip() == (
        "Model thoughts · 2 steps"
    ), f"C1: the summary reads {p['summary']!r}"


# --- (b) click and keyboard -------------------------------------------------------------------------------


@requires_chrome
def test_b_a_click_on_the_summary_expands_it_and_shows_the_committed_thought(run):
    p = _p(run, "clicked")
    assert p["open_attr"] is True, "C2/(b): a real click on the summary opens the disclosure"
    assert all(p["text_visible"]) and len(p["text_visible"]) == 2, (
        f"(b): both blocks' bodies are visible once open: {p['text_visible']}"
    )
    assert COMMITTED_SENTINEL in p["visible_text"], (
        f"(b): the committed thought shows; visible text {p['visible_text'][:200]!r}"
    )
    assert "I see a box.\n" in p["visible_text"], "C3: the newline is kept (pre-wrap)"
    q = _p(run, "click_closed")
    assert q["open_attr"] is False and not any(q["text_visible"]), "(b): a second click closes it"


@requires_chrome
def test_b_the_keyboard_reaches_the_summary_and_enter_and_space_toggle_it(run):
    assert run.tab_steps is not None, (
        "C8: the summary is reachable by keyboard (Tab); 120 Tabs never focused it"
    )
    assert _p(run, "tabbed")["open_attr"] is False, "premise: closed before Enter"
    e = _p(run, "enter")
    assert e["open_attr"] is True and e["active_is_summary"], (
        "C2/(b): Enter on the summary opens it"
    )
    assert COMMITTED_SENTINEL in e["visible_text"], "(b): and shows the committed thought"
    s = _p(run, "space")
    assert s["open_attr"] is False, "C2/(b): Space on the summary toggles it closed"


# --- (c) never a draft ----------------------------------------------------------------------------------


@requires_chrome
def test_c_draft_text_is_in_the_denoising_view_and_never_in_the_thoughts(run):
    for name in ("collapsed", "clicked"):
        p = _p(run, name)
        assert p["count"] == 1, f"premise ({name}): the disclosure exists"
        assert COMMITTED_SENTINEL in p["thoughts_text"], (
            f"anti-vacuity ({name}): the committed thought is in [data-model-thoughts]"
        )
        assert DRAFT_SENTINEL in (p["denoise_text"] or ""), (
            f"premise ({name}): the drafts reached the denoising root, so their absence below "
            f"means something; it reads {str(p['denoise_text'])[:200]!r}"
        )
        assert DRAFT_SENTINEL not in p["thoughts_text"], (
            f"(c)/B2 ({name}): a draft reached the Model thoughts disclosure"
        )
        assert not p["thread_has_draft"], f"(c) ({name}): a draft reached the thread"


# --- C2 in the browser ------------------------------------------------------------------------------------


@requires_chrome
def test_c2_an_update_keeps_the_viewers_closed_or_open_state(run):
    c = _p(run, "update_closed")
    assert c["blocks"] == 3, f"premise: the update rendered a third block; {c['blocks']}"
    assert c["open_attr"] is False, "C2: an update to a closed disclosure leaves it closed"
    o = _p(run, "update_open")
    assert o["blocks"] == 4, f"premise: the updates rendered; {o['blocks']}"
    assert o["open_attr"] is True, "C2: an update (and an error end) never closes an open one"
    assert "Third step, finished." in o["visible_text"], "C4: the latest message shows"


# --- (d) reload -------------------------------------------------------------------------------------------


@requires_chrome
def test_d_after_a_reload_the_replayed_blocks_are_closed(run):
    assert _p(run, "update_open")["open_attr"] is True, "premise: open before the reload"
    p = _p(run, "reloaded")
    assert p["count"] == 1 and p["blocks"] == 3, (
        f"premise: the replay rebuilt the disclosure with three blocks; {p['count']}, {p['blocks']}"
    )
    assert p["any_open"] is False, "C2/H-T180-2: nothing persists across a reload; all closed"
    assert COMMITTED_SENTINEL in p["thoughts_text"], "the replayed text is there"


@requires_chrome
def test_c1_a_thought_only_turn_is_on_screen_and_opens_on_a_click(run):
    """C1 ("the first astra.thought for a turn creates its article if needed") in a real browser.

    Adversarial for T-0154's rule that hides an assistant row with nothing in it: after the reload the
    turn holds only replayed thoughts, so a disclosure the stylesheet hides with its row would pass every
    textContent probe above and still be invisible and unclickable.
    """
    p = _p(run, "reloaded")
    assert p["count"] == 1, f"premise: the replayed disclosure exists; {p['count']}"
    assert p["summary_visible"] is True, (
        "C1: a turn with thoughts and no answer still shows its 'Model thoughts' summary"
    )
    q = _p(run, "reloaded_clicked")
    assert q["open_attr"] is True, "C1/C2: a click on the thought-only turn's summary opens it"
    assert COMMITTED_SENTINEL in q["visible_text"], (
        f"C1: the opened thought-only turn shows its thought; visible {q['visible_text'][:200]!r}"
    )


# --- (e) contrast, overflow, the scroll box ---------------------------------------------------------------


@requires_chrome
@pytest.mark.parametrize("theme", THEMES)
def test_e_the_thought_body_clears_4_5_to_1_and_the_other_lines_their_floor(run, theme):
    assert run.error is None, run.error
    body = run.contrast.get(f"body_{theme}", {})
    entry = body.get("thought text") or {}
    assert entry.get("matches"), f"C8 ({theme}): no [data-thought-text] to measure"
    usable, excluded = _partition(entry)
    assert usable, f"C8 ({theme}): no measurable thought text: {excluded}"
    low = [_describe("thought text", m) for m in usable if ratio_of(m) + 1e-9 < 4.5]
    assert low == [], f"C8 ({theme}): the muted body text is below 4.5:1: {low}"
    missing, vacuous, failures, _ = _contrast_report(run.contrast.get(f"other_{theme}", {}))
    assert missing == [], f"C3/C8 ({theme}): no element matches {missing}"
    assert vacuous == [], f"({theme}) vacuous selectors: {vacuous}"
    assert failures == [], f"C8 ({theme}): text below its WCAG floor: {failures}"


@requires_chrome
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("theme", THEMES)
def test_e_no_horizontal_overflow_and_a_24rem_scroll_box(run, width, theme):
    assert run.error is None, run.error
    lay = run.layout.get(f"{width}_{theme}") or {}
    assert lay.get("open") is True, f"premise ({width}, {theme}): the disclosure is open: {lay}"
    assert lay["white_space"] == "pre-wrap", f"C3: white-space is {lay['white_space']!r}"
    assert lay["doc_scroll_width"] <= lay["doc_client_width"], (
        f"C8 ({width}, {theme}): the page overflows sideways: {lay}"
    )
    assert lay["thread_scroll_width"] <= lay["thread_client_width"], (
        f"C8 ({width}, {theme}): the thread overflows sideways: {lay}"
    )
    assert lay["box"], f"C8: no scrolling box around [data-thought-text]: {lay}"
    want = 24 * float(lay["root_px"])
    assert lay["box_max_height"] == f"{want:g}px", (
        f"C8: max-height {lay['box_max_height']!r}, want 24rem = {want:g}px"
    )
    assert lay["box_scrolls"] is True and lay["box_client_height"] <= want + 0.5, (
        f"C8: a 16000-character thought scrolls inside the box: {lay}"
    )


# --- C-R1: opening or closing "Model thoughts" never moves the reader ------------------------------------

#: "doesn't move by more than a few px".
ANCHOR_PX = 4.0


def _anchored(run: Run, name: str, *, want_open: bool, pinned: bool | None) -> None:
    assert run.error is None, f"the T-0180 C headless session failed: {run.error}"
    row = run.anchor.get(name)
    assert isinstance(row, dict), f"probe {name!r} missing: {sorted(run.anchor)}"
    b, a = row["before"], row["after"]
    assert "top" in b and "top" in a, f"premise ({name}): the summary exists: {row}"
    assert b["scroll_height"] > b["client_height"] + 200, (
        f"premise ({name}): the thread is taller than its viewport: {b}"
    )
    assert b["inside"], (
        f"premise ({name}): the summary is on screen before the toggle (the rows run in sequence, so an "
        f"earlier toggle that moved it also turns this row red): {b}"
    )
    if pinned is True:
        assert b["from_bottom"] <= 2, f"premise ({name}): the reader is pinned at the bottom: {b}"
    elif pinned is False:
        assert b["from_bottom"] > 100, (
            f"premise ({name}): the reader scrolled away from the bottom: {b}"
        )
    assert a["open"] is want_open, (
        f"premise ({name}): the toggle {'opened' if want_open else 'closed'} the disclosure: {a}"
    )
    assert a["inside"], (
        f"C-R1 ({name}): after the viewer's toggle the summary left the thread's box: summary top "
        f"{b['top']:.0f} -> {a['top']:.0f} px, thread {a['thread_top']:.0f}..{a['thread_bottom']:.0f}, "
        f"scrollTop {b['scroll_top']:.0f} -> {a['scroll_top']:.0f}"
    )
    assert abs(a["top"] - b["top"]) <= ANCHOR_PX, (
        f"C-R1 ({name}): the summary moved {a['top'] - b['top']:+.0f} px on its own toggle "
        f"({b['top']:.0f} -> {a['top']:.0f}); it stays anchored"
    )


@requires_chrome
def test_c_r1_a_real_click_on_the_last_turns_summary_never_moves_it_while_pinned(run):
    _anchored(run, "last_click_open", want_open=True, pinned=True)


@requires_chrome
@pytest.mark.parametrize(
    ("name", "want_open"), [("last_enter_close", False), ("last_enter_open", True)]
)
def test_c_r1_enter_on_the_focused_summary_never_moves_it(run, name, want_open):
    row = run.anchor.get(name) or {}
    assert (row.get("before") or {}).get("focused") is True, (
        f"premise ({name}): keyboard focus is on the summary before Enter: {row}"
    )
    _anchored(run, name, want_open=want_open, pinned=None)


@requires_chrome
@pytest.mark.parametrize(
    ("name", "want_open"), [("older_pinned_open", True), ("older_pinned_close", False)]
)
def test_c_r1_an_older_turns_toggle_never_shifts_the_view_while_pinned(run, name, want_open):
    _anchored(run, name, want_open=want_open, pinned=True if want_open else None)


@requires_chrome
def test_c_r1_an_older_turns_toggle_never_shifts_the_view_when_scrolled_up(run):
    _anchored(run, "older_unpinned_open", want_open=True, pinned=False)


# --- C-R2: a toggle does not unpin --------------------------------------------------------------------------

#: "stuck to the bottom again": the reader sees the last pixel of the thread (sub-pixel rounding allowed).
PINNED_PX = 2.0


def _pin_row(run: Run, name: str) -> dict[str, Any]:
    assert run.error is None, f"the T-0180 C headless session failed: {run.error}"
    row = run.pins.get(name)
    assert isinstance(row, dict), f"probe {name!r} missing: {sorted(run.pins)}"
    for key in ("before", "after", "toggled"):
        if key in row:
            assert "top" in row[key], f"premise ({name}): the summary exists ({key}): {row[key]}"
    b = row["before"]
    assert b["scroll_height"] > b["client_height"] + 200, (
        f"premise ({name}): the thread is taller than its viewport: {b}"
    )
    assert b["inside"], f"premise ({name}): the summary is on screen before the toggle: {b}"
    return row


@requires_chrome
@pytest.mark.parametrize(
    "name", ["open_last_then_new_turn", "open_older_then_new_turn"], ids=["last", "older"]
)
def test_c_r2_a_pinned_reader_who_opened_a_disclosure_is_stuck_to_the_bottom_by_a_new_turn(
    run, name
):
    """(1) and (2): pinned, a real click opens the last (or an older) turn's disclosure, and it stays open;
    then a new turn (a new article) arrives. C-R2: the reader was pinned before the toggle, so the new turn
    sticks them to the bottom again."""
    row = _pin_row(run, name)
    b, t, a = row["before"], row["toggled"], row["after"]
    assert b["from_bottom"] <= PINNED_PX, f"premise ({name}): pinned before the toggle: {b}"
    assert t["open"] is True and a["open"] is True, (
        f"premise ({name}): the click opened the disclosure and it stayed open: {t['open']}, {a['open']}"
    )
    assert row["had_new_turn_before"] is False and row["new_turn"] is True, (
        f"premise ({name}): the new turn's article arrived only after the toggle: {row}"
    )
    assert a["scroll_height"] > t["scroll_height"], (
        f"premise ({name}): the new turn made the thread taller: {t['scroll_height']} -> "
        f"{a['scroll_height']}"
    )
    assert a["from_bottom"] <= PINNED_PX, (
        f"C-R2 ({name}): the reader was pinned before toggling, so a new turn sticks them to the bottom; "
        f"from_bottom is {a['from_bottom']:.0f} px (after the toggle {t['from_bottom']:.0f} px)"
    )


@requires_chrome
def test_c_r2_an_update_to_the_live_turn_with_its_disclosure_open_does_not_move_the_summary(run):
    """(3): pinned, a real click opens the live turn's disclosure; then thought updates to that same turn
    (no new article). C-R2: "updates to existing turns ... do not move them" until a new turn arrives."""
    row = _pin_row(run, "live_turn_thought_update")
    b, t, a = row["before"], row["toggled"], row["after"]
    assert b["from_bottom"] <= PINNED_PX, f"premise: pinned before the toggle: {b}"
    assert t["open"] is True and a["open"] is True, f"premise: the disclosure is open: {t}, {a}"
    assert row["updated"] is True, "premise: the thought update rendered in the open disclosure"
    assert a["scroll_height"] > t["scroll_height"], (
        f"premise: the update made the thread taller: {t['scroll_height']} -> {a['scroll_height']}"
    )
    assert abs(a["top"] - t["top"]) <= ANCHOR_PX, (
        f"C-R2: a thought update to the turn being read moved its summary {a['top'] - t['top']:+.0f} px "
        f"({t['top']:.0f} -> {a['top']:.0f}; scrollTop {t['scroll_top']:.0f} -> {a['scroll_top']:.0f})"
    )


@requires_chrome
@pytest.mark.parametrize(
    "name",
    ["unpinned_new_turn", "unpinned_toggle_then_new_turn"],
    ids=["no_toggle", "after_toggle"],
)
def test_c_r2_a_reader_not_pinned_stays_where_they_are_when_a_new_turn_arrives(run, name):
    """(4): the reader scrolled up to an older turn (not pinned); a new turn arrives, with no toggle and
    then after a real click opened that older turn's disclosure. C-R2: "A reader who was not pinned stays
    unpinned" - the summary they are reading does not move."""
    row = _pin_row(run, name)
    b, a = row["before"], row["after"]
    assert b["from_bottom"] > 100, (
        f"premise ({name}): the reader is scrolled away from the bottom: {b}"
    )
    assert row["new_turn"] is True, f"premise ({name}): the new turn arrived"
    assert a["scroll_height"] > b["scroll_height"], f"premise ({name}): the thread grew: {b}, {a}"
    if name == "unpinned_toggle_then_new_turn":
        assert b["open"] is True, f"premise: the click opened the older disclosure: {b}"
    assert abs(a["top"] - b["top"]) <= ANCHOR_PX, (
        f"C-R2 ({name}): a new turn moved an unpinned reader's summary {a['top'] - b['top']:+.0f} px "
        f"(from_bottom {b['from_bottom']:.0f} -> {a['from_bottom']:.0f})"
    )
    assert a["from_bottom"] > 100, f"C-R2 ({name}): the reader was pulled to the bottom: {a}"


@requires_chrome
def test_c_r2_a_pinned_reader_whose_toggles_cancel_out_is_stuck_by_the_next_new_turn(run):
    """Adversarial (review mutant M1): open and close the last summary inside one task, so no resize follows
    the toggle and the record of it is never consumed by one; a new turn must still stick a reader who was
    pinned. The two clicks are ``HTMLElement.click()`` in one ``evaluate`` (real input events cannot be
    forced into one frame); C-R2 does not depend on how the toggle was made."""
    row = _pin_row(run, "double_toggle_then_new_turn")
    b, t, a = row["before"], row["toggled"], row["after"]
    assert b["from_bottom"] <= PINNED_PX, f"premise: pinned before the toggles: {b}"
    assert t["open"] is False and a["open"] is False, f"premise: open then closed again: {t}, {a}"
    assert row["new_turn"] is True, "premise: the new turn arrived"
    assert a["from_bottom"] <= PINNED_PX, (
        f"C-R2: after toggles that cancel out, a new turn leaves the pinned reader {a['from_bottom']:.0f} px "
        f"from the bottom"
    )


# --- C-R3: only the reader's own input releases the hold ------------------------------------------------------


@requires_chrome
def test_c_r3_growth_above_the_viewport_does_not_release_the_hold_so_a_new_turn_still_sticks(run):
    """(1) Pinned, a real click opens the last turn's (r6) disclosure; then an utterance grows r1, far above
    the viewport. Chrome's scroll anchoring may answer with a scroll event the reader did not make; C-R3:
    that neither releases the hold nor unpins. Then a new turn (r7) arrives: the reader is at the bottom."""
    row = _pin_row(run, "r3_growth_above_then_new_turn")
    b, t, g, a = row["before"], row["toggled"], row["grown"], row["after"]
    assert b["from_bottom"] <= PINNED_PX, f"premise: pinned before the toggle: {b}"
    assert t["open"] is True and a["open"] is True, (
        f"premise: the click opened r6's thoughts: {t}, {a}"
    )
    r1 = row["r1_before_growth"]
    assert "top" in r1 and r1["bottom"] < r1["thread_top"], (
        f"premise: r1 is above the viewport when it grows: {r1}"
    )
    assert row["r1_grew"] is True and g["scroll_height"] > t["scroll_height"], (
        f"premise: the utterance rendered in r1 and the thread grew: {t['scroll_height']} -> "
        f"{g['scroll_height']}"
    )
    assert row["had_new_turn_before"] is False and row["new_turn"] is True, (
        f"premise: r7's article arrived only after the growth: {row}"
    )
    assert a["from_bottom"] <= PINNED_PX, (
        f"C-R3: growth above the viewport is not the reader's input, so the hold stands and the new turn "
        f"sticks them to the bottom; from_bottom is {a['from_bottom']:.0f} px (after the toggle "
        f"{t['from_bottom']:.0f} px, after r1 grew {g['from_bottom']:.0f} px)"
    )


@requires_chrome
def test_c_r3_a_held_reader_who_wheels_back_to_the_bottom_follows_the_live_turns_growth(run):
    """(2) Pinned with r6 live, a real click opens r6's disclosure (held); the reader scrolls back to the
    bottom with a real mouse wheel; then the live turn grows (an utterance and a thought to r6, no new
    article). C-R3: "A held reader who scrolls back to the bottom by hand is pinned again and follows the
    live turn's growth"."""
    row = _pin_row(run, "r3_wheel_back_then_live_growth")
    b, t, w, a = row["before"], row["toggled"], row["wheeled"], row["after"]
    assert b["from_bottom"] <= PINNED_PX, f"premise: pinned before the toggle: {b}"
    assert t["open"] is True and a["open"] is True, (
        f"premise: the click opened r6's thoughts: {t}, {a}"
    )
    assert t["from_bottom"] > 100, (
        f"premise: the opened disclosure left the reader away from the bottom (a hold to release): {t}"
    )
    assert w["from_bottom"] <= PINNED_PX, (
        f"premise: the wheel brought the reader back to the bottom ({row['wheels']} wheel events): {w}"
    )
    assert row["r6_grew"] is True and a["scroll_height"] > w["scroll_height"], (
        f"premise: the live turn grew: {w['scroll_height']} -> {a['scroll_height']}"
    )
    assert row["new_turn"] is False, "premise: no new article; only the live turn grew"
    assert a["from_bottom"] <= PINNED_PX, (
        f"C-R3: the reader wheeled back to the bottom, so they are pinned again and follow the live "
        f"turn's growth; from_bottom is {a['from_bottom']:.0f} px"
    )


@requires_chrome
@pytest.mark.parametrize("focus", R3_KEY_FOCUS)
def test_c_r3_a_reader_who_pages_up_with_the_keyboard_stays_where_they_scrolled(run, focus):
    """(3) Pinned, r6's thoughts opened (held); the reader presses PageUp three times with real
    ``Input.dispatchKeyEvent`` and, ~0.8 s later, a new turn r7 arrives. C-R3: only the reader's own input
    releases the hold, and a keyboard scroll is the reader's own input, so the new turn does not pull them
    to the bottom. Focus: on the summary after an Enter-open (``enter_open``), on the summary after a
    click-open (``click_open``), on <body> after a click on plain thread text (``text_click``)."""
    row = _pin_row(run, f"r3_page_up_{focus}")
    b, t, s, a = row["before"], row["toggled"], row["scrolled"], row["after"]
    assert b["from_bottom"] <= PINNED_PX, f"premise ({focus}): pinned before the toggle: {b}"
    assert t["open"] is True and a["open"] is True, (
        f"premise ({focus}): r6's thoughts opened and stayed open: {t}, {a}"
    )
    if focus == "text_click":
        assert row["clicked_text"] is not None, (
            f"premise: an utterance was on screen to click: {row}"
        )
        assert row["active"] == "BODY", (
            f"premise: a click on plain text leaves the focus on <body>: {row['active']}"
        )
    else:
        assert t["focused"] is True, f"premise ({focus}): focus is on r6's summary: {t}"
    assert s["from_bottom"] > 1000 and s["scroll_top"] < t["scroll_top"], (
        f"premise ({focus}): three PageUps scrolled the thread up: toggled {t['from_bottom']:.0f} px from "
        f"the bottom, after PageUp {s['from_bottom']:.0f} px"
    )
    assert row["had_new_turn_before"] is False and row["new_turn"] is True, (
        f"premise ({focus}): r7's article arrived after the PageUps: {row}"
    )
    assert a["from_bottom"] > 1000 and a["scroll_top"] <= s["scroll_top"] + 4, (
        f"C-R3 ({focus}): a reader who paged up with the keyboard was pulled toward the bottom by a new "
        f"turn: from_bottom {s['from_bottom']:.0f} -> {a['from_bottom']:.0f} px, scroll_top "
        f"{s['scroll_top']:.0f} -> {a['scroll_top']:.0f}"
    )


@requires_chrome
def test_c_r3_space_on_a_focused_summary_toggles_it_and_is_not_taken_for_a_scroll(run):
    """(4) Pinned, focus on r6's summary, Space: the disclosure opens (Space is the summary's toggle key,
    not a page scroll), and a new turn r7 then sticks the pinned reader to the bottom (C-R2: a toggle
    does not unpin; C-R3: a toggle key is not a scroll that releases the reader)."""
    row = _pin_row(run, "r3_space_open_then_new_turn")
    b, f, t, a = row["before"], row["focused"], row["toggled"], row["after"]
    assert b["from_bottom"] <= PINNED_PX, f"premise: pinned before Space: {b}"
    assert f["focused"] is True and f["open"] is False, f"premise: focused and closed: {f}"
    assert t["open"] is True and a["open"] is True, (
        f"Space on the focused summary toggles it open: {t}"
    )
    _assert_anchoring_stimulus(row, "Space")
    assert a["from_bottom"] <= PINNED_PX, (
        f"C-R2/C-R3: Space toggled the summary, it did not scroll, so r1's growth above the viewport "
        f"inside the 600 ms after it ({row['pushed_after_s']:.2f} s) did not release the hold and the new "
        f"turn sticks the pinned reader to the bottom; from_bottom is {a['from_bottom']:.0f} px (after the "
        f"toggle {t['from_bottom']:.0f} px, after r1 grew {row['grown']['from_bottom']:.0f} px)"
    )


def _assert_anchoring_stimulus(row: dict[str, Any], keys: str) -> None:
    """Premises shared by (4) and (5): r1 grew above the viewport inside the key's 600 ms window and the
    new turn arrived only after it."""
    r1 = row["r1_before_growth"]
    assert "top" in r1 and r1["bottom"] < r1["thread_top"], (
        f"premise: r1 is above the viewport when it grows: {r1}"
    )
    assert row["pushed_after_s"] < 0.5, (
        f"premise: r1 grew inside the 600 ms after {keys} ({row['pushed_after_s']:.2f} s), so a key "
        f"wrongly counted as scrolling would still be live"
    )
    assert (
        row["r1_grew"] is True and row["grown"]["scroll_height"] > row["toggled"]["scroll_height"]
    ), (
        f"premise: the utterance rendered in r1 and the thread grew: {row['toggled']} -> {row['grown']}"
    )
    assert row["had_new_turn_before"] is False and row["new_turn"] is True, (
        f"premise: r7's article arrived only after the growth: {row}"
    )
    assert row["toggled"]["from_bottom"] > 100, (
        f"premise: the opened disclosure left the reader away from the bottom (held, not merely at the "
        f"bottom): {row['toggled']}"
    )


@requires_chrome
def test_c_r3_keys_typed_into_the_ask_box_are_not_taken_for_a_scroll(run):
    """(5) Pinned, a real click opens r6's thoughts (held); the reader focuses the ask box and types Space
    and ArrowUp into it with real ``Input.dispatchKeyEvent``; within 600 ms r1 grows above the viewport,
    then a new turn r7 arrives. C-R3: only the reader's own scroll releases the hold, and keys typed into a
    text field are text, not scrolling, so the anchoring scroll leaves the hold standing and r7 sticks the
    reader to the bottom."""
    row = _pin_row(run, "r3_keys_in_the_ask_box_then_new_turn")
    b, t, k, a = row["before"], row["toggled"], row["keyed"], row["after"]
    assert b["from_bottom"] <= PINNED_PX, f"premise: pinned before the toggle: {b}"
    assert t["open"] is True and a["open"] is True, (
        f"premise: the click opened r6's thoughts: {t}, {a}"
    )
    fi, typed = row["focused_input"], row["typed"]
    assert isinstance(fi, dict) and fi["active"] is True and fi["disabled"] is False, (
        f"premise: the ask box is mounted, enabled and focused: {fi}"
    )
    assert isinstance(typed, dict) and typed["active"] is True and " " in typed["value"], (
        f"premise: the Space went into the ask box as text and the focus stayed there: {typed}"
    )
    assert abs(k["scroll_top"] - t["scroll_top"]) < 1, (
        f"premise: the keys in the field did not scroll the thread: {t['scroll_top']} -> {k['scroll_top']}"
    )
    _assert_anchoring_stimulus(row, "Space and ArrowUp in the ask box")
    assert a["from_bottom"] <= PINNED_PX, (
        f"C-R3: keys typed into the ask box are not a scroll, so r1's growth above the viewport inside "
        f"the 600 ms after them ({row['pushed_after_s']:.2f} s) did not release the hold and the new turn "
        f"sticks the pinned reader to the bottom; from_bottom is {a['from_bottom']:.0f} px (after the "
        f"keys {k['from_bottom']:.0f} px, after r1 grew {row['grown']['from_bottom']:.0f} px)"
    )


# --- negatives and hygiene (ungated: green before and after group C) ---------------------------------------


@requires_chrome
def test_c9_interleaved_thoughts_leave_the_denoise_digest_and_frame_count_unchanged(run):
    plain, mixed = _p(run, "denoise_plain"), _p(run, "denoise_mixed")
    assert plain["denoise_frames"] == "6" and plain["denoise_digest"], (
        f"premise: the denoising view drew the six frames: {plain}"
    )
    assert (mixed["denoise_frames"], mixed["denoise_digest"]) == (
        plain["denoise_frames"],
        plain["denoise_digest"],
    ), f"C9: thoughts changed the denoising view: plain {plain}, mixed {mixed}"
    assert mixed["denoise_text"] == plain["denoise_text"], "C9: the denoise text too"


@requires_chrome
def test_no_page_exception_and_no_request_off_loopback(run):
    assert run.error is None, run.error
    assert run.exceptions == [], f"uncaught exceptions in the page: {run.exceptions}"
    off = [
        u
        for u in run.requests
        if not u.startswith(("http://127.0.0.1", "ws://127.0.0.1", "data:", "blob:", "about:"))
    ]
    assert off == [], f"requests left loopback: {off}"


@requires_chrome
def test_the_session_fits_the_module_budget(run):
    assert run.error is None, run.error
    assert run.seconds < 90, f"the headless session took {run.seconds:.1f} s"
