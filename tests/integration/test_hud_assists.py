"""The assist panel in headless Chrome, against the stub Conductor (T-0025 criteria 3-11).

This is the end-to-end half of ``docs/tasks/T-0025.md``: the page really renders four rows, the
click really puts an ``astra.assist.toggle`` on a socket, the badge really is the string the
server sent, and the ``astra.frame.text`` messages the page emits are decoded by the **production
codec** and fed to a fresh ``local_astra.perception.assists.AssistState`` — the page and the
Conductor half are shown to agree through the contract, never through a JS mirror of it.

Driving: Chrome DevTools Protocol polling in real time, T-0008's harness and ruling R3
(``--virtual-time-budget`` runs ~15x real time and truncates scripted sockets).
``tests/integration/test_hud_headless.py`` is T-0008's file and is not touched; the pieces reused
from it (``Chrome``, ``serve_hud``) are imported, not copied.

**The DOM contract this test pins** (the task fixes the behaviour; these are the hooks the
assertions need, and the implementer must provide them):

==================================  ====================================================
``[data-assist-kind]``              one element per kind, in ``AssistFlags`` field order,
                                    each containing an ``input[type=checkbox]``
``[data-assist-on]``                ``"true"`` only after an ``astra.assist.state`` said so
``[data-assist-available]``         ``"false"`` for a kind with no extractor
``[data-assist-reason]``            non-empty when unavailable
``[data-assist-badge]``             text == ``astra.assist.state.badge``, verbatim
``[data-assist-sent]``              assist texts sent (criterion 7)
``[data-assist-dropped]``           assist texts dropped (criterion 7)
==================================  ====================================================

The fixture runs the whole session once (three phases) and records snapshots; each test asserts
on one phase, because a browser launch per criterion is waste. Setup never raises: a failure is
recorded on the ``Run`` and asserted in the call phase, so a pending run reports a clean failure
instead of a collection error.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import pytest

from local_astra.contracts.codec import decode
from local_astra.contracts.perception import AssistFlags, Source
from local_astra.contracts.protocol import AstraAssistState, AstraAssistToggle, AstraFrameText
from local_astra.perception.assists import ASSIST_HORIZON_MS, AssistState
from tests.fakes.hud_server import HudStub
from tests.integration.test_hud_headless import CHROME, HUD, Chrome, requires_chrome, serve_hud

pytestmark = pytest.mark.integration

ASSISTS_INDEX = HUD / "src" / "assists" / "index.js"
KINDS = ("page_text", "pdf_text", "cells", "at_spi")
BADGE_AFTER = "pixels+dom"
DISAGREEING_BADGE = "pixels"
#: Drawn into the synthetic capture by ``hud/src/capture.js`` (``synthetic capture <tick>``), and
#: therefore the one piece of text on this page that the page itself drew and genuinely knows —
#: which is exactly what criterion 8 says the only v1 extractor may report.
SENTINEL = "synthetic capture"

RATE_WINDOW_S = 1.0  # 1000 / policy.MAX_FPS, the window criterion 7 shares with the frames


def script() -> list[Any]:
    """What the stub replays once the first ``astra.assist.toggle`` arrives (criterion 4's
    adversarial interleave: a malformed line, then a state whose badge and flags disagree —
    which the wire permits — then the good one)."""
    return [
        "{this is not json at all",
        AstraAssistState(flags=AssistFlags(cells=True), badge=DISAGREEING_BADGE),
        AstraAssistState(flags=AssistFlags(page_text=True), badge=BADGE_AFTER),
    ]


#: T-0154 A5: what the Context pane renders (the rendered text, via innerText), per row and for the
#: heading. Hooks pinned here for T-0154 (the task fixes the copy; these are the elements the
#: assertions need): the row's human label is the element ``[data-assist-label]`` inside the row;
#: its one-line reason stays ``.assist-why`` (T-0056's contrast row selects it); the full reason
#: stays in ``data-assist-reason`` and is also a ``title`` on the row or on the reason element.
CONTEXT_JS = """
() => {
  // T-0162 B-9 (named spec change): the panel is found inside the Settings popover.
  const panel = document.querySelector('[data-settings-popover] #assist-panel');
  if (!panel) return {present: false};
  const h = panel.querySelector('h2');
  const badge = panel.querySelector('[data-assist-badge]');
  const box = (el) => { if (!el) return null; const r = el.getBoundingClientRect();
                        return {top: r.top, bottom: r.bottom, left: r.left, right: r.right, w: r.width, h: r.height}; };
  const lines = (el) => {
    if (!el || !(el.textContent || '').trim()) return 0;
    const range = document.createRange();
    range.selectNodeContents(el);
    const tops = new Set([...range.getClientRects()].filter((r) => r.width > 0).map((r) => Math.round(r.top)));
    return tops.size;
  };
  return {
    present: true,
    heading: h ? (h.innerText || '').trim() : null,
    heading_box: box(h),
    badge_box: badge ? box(badge) : null,
    badge_text: badge ? badge.textContent : null,
    text: (panel.innerText || ''),
    rows: [...panel.querySelectorAll('[data-assist-kind]')].map((row) => {
      const label = row.querySelector('[data-assist-label]');
      const why = row.querySelector('.assist-why');
      return {
        kind: row.getAttribute('data-assist-kind'),
        available: row.getAttribute('data-assist-available'),
        reason: row.getAttribute('data-assist-reason'),
        label: label ? (label.innerText || '').trim() : null,
        why: why ? (why.innerText || '').trim() : null,
        why_lines: lines(why),
        titles: [row.getAttribute('title'), why && why.getAttribute('title'),
                 label && label.getAttribute('title')].filter(Boolean),
      };
    }),
  };
}
"""

PROBE_JS = """
(() => {
  const rows = [...document.querySelectorAll('[data-assist-kind]')].map((el) => ({
    kind: el.getAttribute('data-assist-kind'),
    on: el.getAttribute('data-assist-on'),
    available: el.getAttribute('data-assist-available'),
    reason: el.getAttribute('data-assist-reason'),
    disabled: (() => {
      const box = el.querySelector('input[type=checkbox]');
      return box ? Boolean(box.disabled) : null;
    })(),
    checkbox: Boolean(el.querySelector('input[type=checkbox]')),
  }));
  const attr = (sel, name) => {
    const el = document.querySelector(sel);
    return el ? el.getAttribute(name) : null;
  };
  const badgeEl = document.querySelector('[data-assist-badge]');
  return {
    rows,
    badge_text: badgeEl ? badgeEl.textContent : null,
    sent: attr('[data-assist-sent]', 'data-assist-sent'),
    dropped: attr('[data-assist-dropped]', 'data-assist-dropped'),
    banner: document.documentElement.outerHTML.includes('SYNTHETIC CAPTURE'),
    context: CONTEXT(),
  };
})()
""".replace("CONTEXT()", "(" + CONTEXT_JS.strip() + ")()")

CLICK_JS = """
(() => {
  const row = document.querySelector('[data-assist-kind="page_text"]');
  if (!row) return 'no page_text row';
  const box = row.querySelector('input[type=checkbox]');
  if (!box) return 'no checkbox in the page_text row';
  if (box.disabled) return 'the page_text checkbox is disabled';
  box.click();
  return 'clicked';
})()
"""


@dataclass
class Run:
    """One page session, recorded in three phases."""

    error: str | None = None
    stub: HudStub | None = None
    idle_types: list[str] = field(default_factory=list)
    idle_probe: dict[str, Any] = field(default_factory=dict)
    click_result: str | None = None
    after_probe: dict[str, Any] = field(default_factory=dict)
    after_raw: list[str] = field(default_factory=list)
    untick_result: str | None = None
    texts_at_untick: int = 0
    texts_after_untick: int = 0
    final_raw: list[str] = field(default_factory=list)


def _types(raws: list[str]) -> list[str]:
    out = []
    for raw in raws:
        try:
            obj = json.loads(raw)
        except ValueError:
            out.append("<invalid json>")
            continue
        out.append(str(obj.get("type")) if isinstance(obj, dict) else "<not an object>")
    return out


def _texts(raws: list[str]) -> list[AstraFrameText]:
    return [m for m in (decode(r) for r in raws) if isinstance(m, AstraFrameText)]


def _toggles(raws: list[str]) -> list[AstraAssistToggle]:
    return [m for m in (decode(r) for r in raws) if isinstance(m, AstraAssistToggle)]


@pytest.fixture(scope="module")
def run() -> Any:
    """One page load, three phases. Never raises: setup failures land on ``Run.error``."""
    if not ASSISTS_INDEX.is_file():
        yield Run(
            error=(
                "T-0025 adds hud/src/assists/index.js (plus gate.js and extract.js) — the assist "
                "panel this test drives (docs/tasks/T-0025.md criteria 1-11)"
            )
        )
        return
    if CHROME is None:  # pragma: no cover - the skipif below is the real guard
        yield Run(error="no chrome")
        return
    result = Run()
    stub = HudStub(script(), send_after="astra.assist.toggle").start()
    result.stub = stub
    server, port = serve_hud()
    try:
        url = f"http://127.0.0.1:{port}/?source=canvas&ws=" + urllib.parse.quote(stub.url, safe="")
        with Chrome(CHROME) as chrome:
            chrome.navigate(url)
            stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20)
            # --- phase A: nothing ticked, frames flowing, for >= 5 s of real time -------------
            stub.wait_for_frames(2, 20)
            # T-0162 B-9: the panel lives in the Settings popover; open it so its rows are laid
            # out for the measuring probes (a no-op on a page without the gear).
            chrome.evaluate(
                "(() => { const g = document.querySelector('header button[aria-label=\"Settings\"]');"
                " if (g) g.click(); })()"
            )
            time.sleep(5.0)
            result.idle_types = stub.received_types()
            probe = chrome.evaluate(PROBE_JS)
            result.idle_probe = probe if isinstance(probe, dict) else {"__probe__": probe}

            # --- phase B: tick page_text; the stub replays its script ------------------------
            result.click_result = str(chrome.evaluate(CLICK_JS))
            stub.wait_for(lambda s: "astra.assist.toggle" in s.received_types(), 20)
            stub.script_sent.wait(timeout=20)
            stub.wait_for(lambda s: len(_texts(s.snapshot())) >= 2, 20)
            chrome.wait_until(
                "(document.querySelector('[data-assist-badge]') || {}).textContent === "
                f"{BADGE_AFTER!r}".replace("'", '"'),
                10,
            )
            probe = chrome.evaluate(PROBE_JS)
            result.after_probe = probe if isinstance(probe, dict) else {"__probe__": probe}
            result.after_raw = stub.snapshot()

            # --- phase C: untick; nothing more for >= 3 rate windows -------------------------
            result.texts_at_untick = len(_texts(stub.snapshot()))
            result.untick_result = str(chrome.evaluate(CLICK_JS))
            time.sleep(3.5 * RATE_WINDOW_S)
            result.final_raw = stub.snapshot()
            result.texts_after_untick = len(_texts(result.final_raw))
    except Exception as exc:  # a broken harness is data, not an error inside a test
        result.error = f"the headless run failed: {type(exc).__name__}: {exc}"
    finally:
        server.shutdown()
        stub.stop()
    yield result


def rows_of(run: Run, phase: str) -> list[dict[str, Any]]:
    probe = run.after_probe if phase == "after" else run.idle_probe
    return list(probe.get("rows") or [])


# --- criteria 2 and 8: the rows, their order, and what cannot be ticked ------------------------


@requires_chrome
def test_the_panel_renders_one_row_per_kind_in_assist_flags_order(run):
    """Criterion 2, and criterion 8's disabled rows."""
    assert run.error is None, run.error
    rows = rows_of(run, "idle")
    assert [r["kind"] for r in rows] == list(KINDS), (
        "criterion 2: exactly four rows keyed data-assist-kind in AssistFlags declaration order "
        f"(mirrored from local_astra/contracts/perception.py); got {rows!r}"
    )
    assert [r["kind"] for r in rows] == list(AssistFlags.model_fields), (
        "criterion 2: and that order is the contract's field order, not a hand-written list"
    )
    for row in rows:
        assert row["checkbox"], f"criterion 2: {row['kind']} needs a checkbox"
        assert row["on"] == "false", (
            f"criterion 3: {row['kind']} starts off — no astra.assist.state has arrived yet"
        )
    by_kind = {r["kind"]: r for r in rows}
    assert by_kind["page_text"]["available"] == "true", (
        "criterion 8: v1 registers exactly one extractor, page_text over the synthetic canvas "
        "source (?source=canvas), whose text the page itself drew"
    )
    for kind in ("pdf_text", "cells", "at_spi"):
        assert by_kind[kind]["available"] == "false", (
            f"criterion 8: {kind} has no extractor in a browser tab (a tab cannot read another "
            "tab's DOM, PDF text layer, cells or accessibility tree — planner's finding 1)"
        )
        assert (by_kind[kind]["reason"] or "").strip(), (
            f"criterion 8: {kind}'s row must say why, in data-assist-reason"
        )
        assert by_kind[kind]["disabled"] is True, (
            f"criterion 8: {kind}'s checkbox is disabled — the user cannot tick a box into a "
            "regime the page cannot honour"
        )


# --- criterion 5: with no box ticked, no assist text ever -------------------------------------


@requires_chrome
def test_with_no_box_ticked_the_page_sends_no_assist_text(run):
    """Criterion 5 end to end: the headline negative, over >= 5 s with frames flowing."""
    assert run.error is None, run.error
    types = run.idle_types
    assert "astra.frame" in types, (
        f"premise: the capture pipeline is running and posting frames; the stub saw {set(types)!r}"
    )
    assert "astra.frame.text" not in types, (
        "criterion 5: with no box ticked no assist text leaves the page — not buffered, not "
        f"queued, not 'just once at startup' (docs/01 D12a); the stub recorded {types!r}"
    )
    assert "astra.assist.toggle" not in types, (
        "criterion 3: the page sends a toggle only when the user clicks one, never at startup"
    )
    assert set(types) <= {"astra.hello", "astra.frame"}, (
        f"criterion 12: the shell's whitelist is unchanged; got {sorted(set(types))!r}"
    )
    assert run.idle_probe.get("banner"), "T-0008 criterion 7: the SYNTHETIC CAPTURE banner is up"


# --- criteria 3, 4, 10, 11: the round trip ------------------------------------------------------


@requires_chrome
def test_ticking_the_box_round_trips_into_assist_state(run):
    """Criteria 3, 4, 10 and 11: click -> toggle -> echo -> text -> ``AssistState``."""
    assert run.error is None, run.error
    assert run.click_result == "clicked", (
        f"the page_text checkbox was not clickable: {run.click_result}"
    )

    toggles = _toggles(run.after_raw)
    assert len(toggles) == 1, (
        f"criterion 3: one click, one astra.assist.toggle; got {_types(run.after_raw)!r}"
    )
    assert toggles[0] == AstraAssistToggle(flags=AssistFlags(page_text=True)), (
        "criterion 10: the toggle carries all four booleans explicitly — a whole flag set, never "
        f"a per-kind merge; the page sent {toggles[0]!r}"
    )

    rows = {r["kind"]: r for r in rows_of(run, "after")}
    assert rows["page_text"]["on"] == "true", (
        "criterion 3: a row reads data-assist-on=true only after an astra.assist.state whose "
        "flags[kind] is true — and that echo has now arrived"
    )
    for kind in ("pdf_text", "cells", "at_spi"):
        assert rows[kind]["on"] == "false", (
            "criterion 4: the last state message says page_text only; the earlier one (cells "
            "true, badge 'pixels') is superseded, and the page reconciles nothing"
        )
    assert (run.after_probe.get("badge_text") or "").strip() == BADGE_AFTER, (
        "criterion 4: [data-assist-badge] is astra.assist.state.badge character for character "
        f"({BADGE_AFTER!r}); got {run.after_probe.get('badge_text')!r}"
    )

    texts = _texts(run.after_raw)
    assert len(texts) >= 1, (
        "criterion 5: after the echo the page may send assist text for the confirmed kind; the "
        f"stub recorded {_types(run.after_raw)!r}"
    )
    assert {t.kind for t in texts} == {"page_text"}, (
        f"criterion 5: only the confirmed kind may be sent, got {sorted({t.kind for t in texts})}"
    )
    for text in texts:
        assert text.source is Source.SCREEN, (
            "criterion 10: AstraFrameText.source equals the live capture's source"
        )
        assert SENTINEL in text.text.lower(), (
            "criterion 8: the only v1 extractor reports the text the page itself drew into the "
            f"synthetic canvas ({SENTINEL!r}, hud/src/capture.js); got {text.text[:200]!r}"
        )

    # Criterion 11: through the production codec into the Conductor's own authority.
    state = AssistState()
    state.toggle(toggles[0])
    for text in texts:
        state.accept_frame_text(text)
    state.freeze()
    at_ms = max(t.ts_ms for t in texts)
    got = state.texts_for(at_ms)
    assert [t.kind for t in got] == ["page_text"], (
        "criterion 11: a fresh AssistState fed the page's own messages yields exactly one "
        f"page_text entry; got {got!r}"
    )
    assert SENTINEL in got[0].text.lower()
    assert dict(state.dropped) == dict.fromkeys(KINDS, 0), (
        "criterion 11: nothing the page sent was refused at ingress — state.dropped is all "
        f"zeros; got {dict(state.dropped)}"
    )
    assert at_ms - min(t.ts_ms for t in texts) < ASSIST_HORIZON_MS, (
        "premise: the whole exchange fits inside the 60 s horizon"
    )

    sent = run.after_probe.get("sent")
    assert sent not in (None, ""), "criterion 7: the sent counter is displayed (data-assist-sent)"
    assert run.after_probe.get("dropped") not in (None, ""), (
        "criterion 7: and so is the dropped counter (data-assist-dropped)"
    )
    assert int(float(str(sent))) >= 1


@requires_chrome
def test_the_adversarial_script_leaves_the_page_connected_and_honest(run):
    """Criterion 4 adversarial: a malformed line and a badge/flags pair that disagree, before the
    good state. The page must survive both, show the *last* server badge verbatim, and must not
    have sent text for a kind it could not honour (``cells`` was 'confirmed' by the disagreeing
    message and has no extractor)."""
    assert run.error is None, run.error
    assert run.stub is not None and run.stub.script_error is None, (
        run.stub.script_error if run.stub else "no stub"
    )
    assert (run.after_probe.get("badge_text") or "").strip() == BADGE_AFTER, (
        "criterion 4: the badge is the last server string, verbatim — the page invents no "
        "reconciliation between a badge and the flags it disagrees with"
    )
    kinds = {t.kind for t in _texts(run.final_raw)}
    assert kinds <= {"page_text"}, (
        f"criteria 5 and 8: no text for an unconfirmed or unhonourable kind, got {sorted(kinds)}"
    )
    assert _types(run.final_raw).count("astra.frame") >= 2, (
        "criterion 2 of T-0008 still holds: the malformed line did not kill the socket — the "
        "page kept posting frames afterwards"
    )


# --- criterion 6: unticking stops the text -------------------------------------------------------


@requires_chrome
def test_unticking_stops_the_text_within_one_rate_window(run):
    """Criterion 6: after the untick, no further ``astra.frame.text`` for >= 3 rate windows."""
    assert run.error is None, run.error
    assert run.untick_result == "clicked", f"the untick did not happen: {run.untick_result}"
    toggles = _toggles(run.final_raw)
    assert len(toggles) == 2, f"criterion 6: the untick sends its own toggle; got {toggles!r}"
    assert toggles[-1] == AstraAssistToggle(flags=AssistFlags()), (
        "criterion 6: the untick toggle sets page_text false and leaves the other three "
        f"unchanged; got {toggles[-1]!r}"
    )
    assert run.texts_after_untick == run.texts_at_untick, (
        "criterion 6: unticking stops extraction for that kind and drops anything already "
        f"extracted or queued — {run.texts_after_untick - run.texts_at_untick} more "
        f"astra.frame.text arrived in {3.5 * RATE_WINDOW_S:.1f} s (>= 3 rate windows)"
    )


# --- T-0154 A5: the Context pane (the copy changes; the semantics above do not) -----------------------

#: H-T154-2 (default ruled 2026-10-25): the heading and the human labels, verbatim.
#: **Amended by T-0162 B-6** (H-T162-6): the moved panel is headed "Context helpers".
CONTEXT_HEADING = "Context helpers"
CONTEXT_LABELS = {
    "page_text": "Page text",
    "pdf_text": "PDF text",
    "cells": "Spreadsheet cells",
    "at_spi": "Accessibility tree",
}
#: H-T154-2: "unavailable rows say 'Needs the desktop helper', full reason in the tooltip".
UNAVAILABLE_LINE = "Needs the desktop helper"


def _context(run: Run, phase: str) -> dict[str, Any]:
    probe = run.after_probe if phase == "after" else run.idle_probe
    ctx = dict(probe.get("context") or {})
    assert ctx.get("present"), f"#assist-panel is not on the page: {probe!r}"
    return ctx


@requires_chrome
def test_t0154_the_context_pane_shows_human_labels_and_one_line_reasons(run):
    """A5 (H-T154-2): the heading renders "Context"; each row's ``[data-assist-label]`` is its human
    label; an unavailable row's ``.assist-why`` reads "Needs the desktop helper" on one line, while
    ``data-assist-reason`` keeps the full reason and a ``title`` repeats it."""
    assert run.error is None, run.error
    ctx = _context(run, "idle")
    assert ctx["heading"] == CONTEXT_HEADING, (
        f"A5: the pane's heading renders {ctx['heading']!r}; it must render {CONTEXT_HEADING!r}"
    )
    by_kind = {r["kind"]: r for r in ctx["rows"]}
    assert list(by_kind) == list(KINDS), f"premise: the four rows in order: {list(by_kind)}"
    problems = []
    for kind, label in CONTEXT_LABELS.items():
        row = by_kind[kind]
        if row["label"] != label:
            problems.append(f"{kind}: [data-assist-label] renders {row['label']!r}, not {label!r}")
        if row["available"] == "false":
            full = (row["reason"] or "").strip()
            if not full:
                problems.append(f"{kind}: data-assist-reason lost the full reason")
            if row["why"] != UNAVAILABLE_LINE:
                problems.append(
                    f"{kind}: the reason line reads {row['why']!r}, not {UNAVAILABLE_LINE!r}"
                )
            if row["why_lines"] != 1:
                problems.append(f"{kind}: the reason renders on {row['why_lines']} lines, not one")
            if full not in [t.strip() for t in row["titles"]]:
                problems.append(
                    f"{kind}: no title carries the full reason {full!r}: {row['titles']}"
                )
    assert problems == [], "T-0154 A5, the Context copy:\n  " + "\n  ".join(problems)


@requires_chrome
def test_t0154_no_badge_word_before_any_state_and_the_badge_sits_on_the_heading_row(run):
    """A5: with no ``astra.assist.state`` received the pane shows no "badge" word (the old
    ``badge `` prefix line); once the state arrives, ``[data-assist-badge]`` reads the server's
    string verbatim (criterion 4, unchanged) on the heading's row."""
    assert run.error is None, run.error
    idle = _context(run, "idle")
    assert idle["badge_text"] == "", (
        f"premise (T-0025 criterion 4): before any state the badge is empty: {idle['badge_text']!r}"
    )
    assert not re.search(r"\bbadge\b", idle["text"], re.IGNORECASE), (
        f"A5: with no astra.assist.state the Context pane renders the word 'badge': {idle['text']!r}"
    )
    after = _context(run, "after")
    assert (after["badge_text"] or "").strip() == BADGE_AFTER, after["badge_text"]
    h, b = after["heading_box"], after["badge_box"]
    assert h and b and b["w"] > 0, f"the heading and the badge are both laid out: {h} {b}"
    centre = (b["top"] + b["bottom"]) / 2
    assert h["top"] - 0.5 <= centre <= h["bottom"] + 0.5, (
        f"A5: the badge sits on the heading row; its centre y {centre:.1f} is outside the heading's "
        f"{h['top']:.1f}..{h['bottom']:.1f}"
    )
    assert b["left"] >= h["left"], "the badge follows the heading on its row"


# --- the file the whole task is about -------------------------------------------------------------


@requires_chrome
def test_the_assist_panel_files_exist_where_the_seam_expects_them(run):
    """Criterion 1: ``hud/src/assists/index.js`` is what ``panels.js`` imports; criterion 13: the
    seam it needs is ``host.send``/``host.subscribe`` in ``hud/src/main.js``, and nothing else."""
    for name in ("index.js", "gate.js", "extract.js"):
        assert (HUD / "src" / "assists" / name).is_file(), (
            f"T-0025 Owned files: hud/src/assists/{name}"
        )
    main_js = (HUD / "src" / "main.js").read_text()
    assert "subscribe" in main_js and "send" in main_js, (
        "criterion 13: host gains send (the raw socket.send) and subscribe (a registry called "
        "with every decoded astra.* message); the panel has no other way to reach the socket"
    )
    for forbidden in ("astra.assist.toggle", "astra.frame.text"):
        assert forbidden not in main_js, (
            f"criterion 13: main.js mentions no assist type string ({forbidden}), so criterion "
            "12's scan still covers it unchanged"
        )
