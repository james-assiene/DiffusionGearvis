"""The Tools pane and the Settings menu in headless Chrome (docs/tasks/T-0162.md group B).

*Tests to write*, Chrome: "there is no ``#assist-panel`` in the rail; the gear opens the popover,
and the switches work through the echo; the badge shows; Escape closes the popover, and ``m``
typed inside it does not mute; there are zero off-loopback requests." Plus B-1 (the pane in the
slot where Context was), B-6 (heading, sentence, title, outside click), B-7 (the badge rule) and
B-8 (the theme select stays in the header; the inline tool cards stay).

Regime (T-0162 *Regime*): Playwright's pinned Chromium or the box's Chrome, headless, loopback
only, the stub Conductor (``tests/fakes/hud_server.HudStub``) instead of a session, 1440x900, the
popover rows in both data-themes. No measured criterion.

**The DOM contract this module pins** (the task fixes behaviour; these hooks are the
test-author's names, which the implementer provides):

=====================================  ===========================================================
``header button[aria-label=Settings]``  the gear; its ``title`` is D12a's sentence (B-6)
``[data-settings-popover]``             the popover; outside ``<main>`` (it is header chrome, not a
                                        rail pane); hidden until the gear is pressed
``#assist-panel``                       inside ``[data-settings-popover]``, never in ``<main>``
``[data-header-badge]``                 in ``<header>``: the server's badge, verbatim, while it is
                                        not ``pixels``; empty or not rendered otherwise (B-7)
``main > #tools-panel``                 the Tools pane, ``h2`` "Tools", after ``.pane-thinking`` and
                                        before ``#log-panel`` in document order (B-1)
=====================================  ===========================================================

"Beside the gear" (B-7) is read as: in the header, with at most 32 px between the badge's box
and the gear's box — the task gives no number; 32 px is the test-author's.

One fixture drives one page through every phase and records probes; each test asserts on one.
Setup never raises: a failure lands on ``Run.error`` and fails the rows that need it.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import pytest

from local_astra.contracts.perception import AssistFlags
from local_astra.contracts.protocol import AstraAssistState
from tests.fakes.hud_server import HudStub
from tests.integration.test_hud_headless import (
    CHROME,
    _RecordingChrome,
    requires_chrome,
    serve_hud,
)

pytestmark = pytest.mark.integration

D12A_SENTENCE = (
    "Gearvis reads your screen from pixels only; these optional helpers also send it the page's "
    "text, a PDF's text layer, spreadsheet cells or the accessibility tree. They are all off by "
    "default, and every turn records which ones were on."
)
BADGE_ON = "pixels+dom"
# Pushed while the popover is closed, and on a reconnect: strings no earlier phase has shown,
# so a badge that is only recomputed while the popover is open cannot pass by leftover text.
BADGE_CLOSED = "pixels+dom+pdf"
BADGE_REPLAYED = "pixels+cells"
EVIL = "<img src=x onerror=window.__pwned=1>"
BESIDE_PX = 32.0


def _plain(s: Any) -> str:
    return " ".join(str(s or "").replace("’", "'").replace("‘", "'").split())


PROBE_JS = """
(() => {
  const vis = (el) => !!el && el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true});
  const box = (el) => { if (!el) return null; const r = el.getBoundingClientRect();
                        return {left: r.left, right: r.right, top: r.top, bottom: r.bottom, w: r.width, h: r.height}; };
  const gear = document.querySelector('header button[aria-label="Settings"]');
  const pop = document.querySelector('[data-settings-popover]');
  const panel = document.getElementById('assist-panel');
  const tools = document.querySelector('main > #tools-panel');
  const badge = document.querySelector('header [data-header-badge]');
  const order = [...document.querySelectorAll('main > *')].map((e) =>
      e.id ? '#' + e.id : (e.classList.contains('pane-thinking') ? '.pane-thinking' : e.className));
  const mute = document.getElementById('mute-button');
  return {
    gear: !!gear, gear_title: gear ? gear.getAttribute('title') : null, gear_box: box(gear),
    gear_font: gear ? getComputedStyle(gear).fontFamily : null,
    popover: !!pop, popover_visible: vis(pop), popover_in_main: !!(pop && pop.closest('main')),
    panel_in_popover: !!(panel && pop && pop.contains(panel)),
    panel_in_main: !!(panel && panel.closest('main')), panel_visible: vis(panel),
    rail_assist: !!document.querySelector('main > #assist-panel, main #assist-panel'),
    panel_heading: panel && panel.querySelector('h2, h3') ? panel.querySelector('h2, h3').innerText : null,
    panel_paras: panel ? [...panel.querySelectorAll('p')].map((p) => p.innerText) : [],
    tools: !!tools, tools_heading: tools && tools.querySelector('h2') ? tools.querySelector('h2').innerText : null,
    tools_text: tools ? tools.textContent : null,
    main_order: order,
    badge: !!badge, badge_text: badge ? badge.textContent : null, badge_visible: vis(badge) && !!(badge && badge.textContent.trim()),
    badge_box: box(badge),
    theme_in_header: !!document.querySelector('header [data-theme-control]'),
    thread_cards: document.querySelectorAll('[data-thread] [data-tool-call-id]').length,
    pwned: window.__pwned === 1,
    mute_enabled: !!(mute && !mute.disabled),
    page_text_on: (document.querySelector('[data-settings-popover] [data-assist-kind="page_text"]') || {getAttribute: () => null}).getAttribute('data-assist-on'),
    theme: document.documentElement.getAttribute('data-theme'),
  };
})()
"""


FOCUS_JS = """
(() => {
  const gear = document.querySelector('header button[aria-label="Settings"]');
  const pop = document.querySelector('[data-settings-popover]');
  const a = document.activeElement;
  const first = pop ? pop.querySelector('input[type=checkbox]:not([disabled])') : null;
  return {
    popover_visible: !!pop && pop.checkVisibility({checkOpacity: true, checkVisibilityCSS: true}),
    expanded: gear ? gear.getAttribute('aria-expanded') : null,
    on_gear: !!gear && a === gear,
    on_first_switch: !!first && a === first,
    in_popover: !!(pop && a && pop.contains(a)),
    active: a ? (a.tagName + (a.id ? '#' + a.id : '') + (a.type ? '[' + a.type + ']' : '')) : null,
  };
})()
"""


def _center(chrome: Any, selector: str) -> tuple[float, float] | None:
    got = chrome.evaluate(
        "(() => { const e = document.querySelector(" + json.dumps(selector) + ");"
        " if (!e) return null; const r = e.getBoundingClientRect();"
        " return [r.left + r.width / 2, r.top + r.height / 2]; })()"
    )
    return (float(got[0]), float(got[1])) if isinstance(got, list) else None


def _click(chrome: Any, selector: str) -> bool:
    """A real mouse click (Input.dispatchMouseEvent), not a synthesised DOM event."""
    at = _center(chrome, selector)
    if at is None:
        return False
    for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
        params: dict[str, Any] = {"type": kind, "x": at[0], "y": at[1]}
        if kind != "mouseMoved":
            params.update(button="left", clickCount=1)
        chrome._cmd("Input.dispatchMouseEvent", **params)
    return True


def _set_theme(chrome: Any, theme: str) -> None:
    chrome.evaluate(
        "(() => { const s = document.querySelector('[data-theme-control]'); if (!s) return;"
        f" s.value = {json.dumps(theme)}; s.dispatchEvent(new Event('change', {{bubbles: true}})); }})()"
    )


@dataclass
class Run:
    error: str | None = None
    probes: dict[str, Any] = field(default_factory=dict)
    toggles: int = 0
    hellos: int = 0
    requests: list[str] = field(default_factory=list)
    m_outside_frames: int = 0
    m_outside_keys: list[str] = field(default_factory=list)


def _tool_call(call_id: str, **kw: Any) -> str:
    body = {"type": "astra.tool.call", "turn_id": "t1", "call_id": call_id, "source": "draft"}
    body.update(kw)
    return json.dumps(body)


@pytest.fixture(scope="module")
def run() -> Any:
    if CHROME is None:  # pragma: no cover - requires_chrome is the real guard
        yield Run(error="no chrome")
        return
    result = Run()
    stub = HudStub([]).start()
    server, port = serve_hud()
    probes = result.probes

    def probe(name: str, chrome: Any) -> None:
        got = chrome.evaluate(PROBE_JS)
        probes[name] = got if isinstance(got, dict) else {"__probe__": got}

    def focus(name: str, chrome: Any) -> None:
        got = chrome.evaluate(FOCUS_JS)
        probes[name] = got if isinstance(got, dict) else {"__probe__": got}

    try:
        url = f"http://127.0.0.1:{port}/?source=canvas&ws=" + urllib.parse.quote(stub.url, safe="")
        with _RecordingChrome(CHROME) as chrome:
            chrome._cmd("Network.enable")
            chrome._cmd(
                "Emulation.setDeviceMetricsOverride",
                width=1440,
                height=900,
                deviceScaleFactor=1,
                mobile=False,
            )
            chrome.navigate(url)
            stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20)
            stub.wait_for_frames(1, 20)
            chrome.wait_until("!!document.querySelector('main > #log-panel')", 15)
            probe("idle", chrome)  # no astra.assist.state received yet

            # Tool calls: the pane fills from astra.tool.call (and the inline card stays, B-8).
            stub.push(
                json.dumps({"type": "astra.turn.state", "state": "THINKING", "turn_id": "t1"})
            )
            stub.push(
                _tool_call("c1", name="click_at", arguments={"x": 120, "y": 845}, status="proposed")
            )
            stub.push(
                _tool_call("c2", name="google_search", arguments={"query": EVIL}, status="dry_run")
            )
            chrome.wait_until("document.querySelectorAll('[data-tools-call]').length >= 2", 10)
            probe("tools", chrome)

            # The popover in both data-themes: the gear opens it, Escape closes it.
            for theme in ("dark", "light"):
                _set_theme(chrome, theme)
                _click(chrome, 'header button[aria-label="Settings"]')
                chrome.wait_until(
                    "(document.querySelector('[data-settings-popover]') || {checkVisibility: () => false}).checkVisibility()",
                    5,
                )
                probe(f"open_{theme}", chrome)
                chrome.evaluate(
                    "(() => { const b = document.querySelector('[data-settings-popover] input[type=checkbox]');"
                    " if (b) b.focus(); })()"
                )
                chrome.key("Escape", code=27)
                time.sleep(0.3)
                probe(f"escaped_{theme}", chrome)

            # Keyboard (T-0162 B-6, reviewer T2): Enter on the focused gear opens the popover,
            # marks the gear expanded and moves focus to the first enabled switch; Escape there
            # closes it, marks the gear collapsed and gives focus back to the gear. chrome.enter()
            # carries text="\r": a CDP Enter without it does not activate a button.
            chrome.evaluate(
                "document.querySelector('header button[aria-label=\"Settings\"]').focus()"
            )
            focus("kb_before", chrome)
            chrome.enter()
            chrome.wait_until(
                "(document.querySelector('[data-settings-popover]') || {checkVisibility: () => false}).checkVisibility()",
                5,
            )
            focus("kb_open", chrome)
            chrome.key("Escape", code=27)
            time.sleep(0.3)
            focus("kb_escaped", chrome)

            # Escape with the focus outside the popover (the document's handler) closes it too.
            _click(chrome, 'header button[aria-label="Settings"]')
            chrome.wait_until(
                "(document.querySelector('[data-settings-popover]') || {checkVisibility: () => false}).checkVisibility()",
                5,
            )
            chrome.evaluate("document.activeElement && document.activeElement.blur()")
            focus("doc_open", chrome)
            chrome.key("Escape", code=27)
            time.sleep(0.3)
            focus("doc_escaped", chrome)

            # Outside click closes it.
            _click(chrome, 'header button[aria-label="Settings"]')
            time.sleep(0.3)
            probe("reopened", chrome)
            _click(chrome, "main .pane-convo [data-empty]")
            time.sleep(0.3)
            probe("outside", chrome)

            # `m` typed inside the popover does not mute.
            _click(chrome, 'header button[aria-label="Settings"]')
            time.sleep(0.3)
            chrome.evaluate(
                "(() => { const b = document.querySelector('[data-settings-popover] input[type=checkbox]');"
                " if (b) b.focus(); })()"
            )
            chrome.key("m", text="m", code=77)
            time.sleep(0.3)
            probe("m_inside", chrome)

            # The switch works through the echo; the header badge shows the server's string.
            _click(
                chrome,
                '[data-settings-popover] [data-assist-kind="page_text"] input[type=checkbox]',
            )
            stub.wait_for(lambda s: "astra.assist.toggle" in s.received_types(), 10)
            result.toggles = stub.received_types().count("astra.assist.toggle")
            stub.push(AstraAssistState(flags=AssistFlags(page_text=True), badge=BADGE_ON))
            chrome.wait_until(
                "(document.querySelector('header [data-header-badge]') || {}).textContent === "
                + json.dumps(BADGE_ON),
                10,
            )
            chrome.key("Escape", code=27)  # closed popover: the badge must still show (D12a)
            time.sleep(0.3)
            probe("badge_on", chrome)
            # A second assist-on state while the popover is closed: the badge follows it there
            # too, not only while the switches are on screen (reviewer T1).
            stub.push(
                AstraAssistState(
                    flags=AssistFlags(page_text=True, pdf_text=True), badge=BADGE_CLOSED
                )
            )
            chrome.wait_until(
                "(document.querySelector('header [data-header-badge]') || {}).textContent === "
                + json.dumps(BADGE_CLOSED),
                5,
            )
            probe("badge_closed", chrome)
            stub.push(AstraAssistState(flags=AssistFlags(), badge="pixels"))
            time.sleep(0.8)
            probe("badge_off", chrome)

            # Reconnect: the server drops the socket, the page reconnects and says hello again,
            # and the state replayed on the new socket shows with the popover never reopened.
            stub.drop_client()
            stub.wait_for(lambda s: s.received_types().count("astra.hello") >= 2, 20)
            result.hellos = stub.received_types().count("astra.hello")
            stub.push(AstraAssistState(flags=AssistFlags(cells=True), badge=BADGE_REPLAYED))
            chrome.wait_until(
                "(document.querySelector('header [data-header-badge]') || {}).textContent === "
                + json.dumps(BADGE_REPLAYED),
                5,
            )
            probe("badge_replayed", chrome)

            # T-0181 B3 (H-T181-4): `m` outside the popover does nothing; the capture still runs. A window
            # capture listener proves the key reached the page; the stub's frame count shows the capture
            # still sending afterwards.
            chrome.evaluate(
                "(() => { window.__t0181_keys = []; window.addEventListener('keydown',"
                " (e) => window.__t0181_keys.push(e.key), true); return true; })()"
            )
            chrome.evaluate("document.activeElement && document.activeElement.blur()")
            frames_before_m = len(stub.frames())
            chrome.key("m", text="m", code=77)
            time.sleep(0.5)
            probe("m_outside", chrome)
            stub.wait_for(lambda s: len(s.frames()) >= frames_before_m + 2, 3)
            result.m_outside_frames = len(stub.frames()) - frames_before_m
            result.m_outside_keys = chrome.evaluate("window.__t0181_keys || []")
            chrome.drain(1.0)
            result.requests = list(chrome.requests)
    except Exception as exc:  # a broken harness is data, not an error inside a test
        result.error = f"the headless run failed: {type(exc).__name__}: {exc}"
    finally:
        server.shutdown()
        stub.stop()
    yield result


def _p(run: Run, name: str) -> dict[str, Any]:
    assert run.error is None, run.error
    got = run.probes.get(name)
    assert isinstance(got, dict) and "__probe__" not in got, f"probe {name}: {got!r}"
    return got


@requires_chrome
def test_b9_there_is_no_assist_panel_in_the_rail_and_the_tools_pane_takes_its_slot(run):
    """B-1 / B-9: ``#assist-panel`` is not in ``<main>``; ``main > #tools-panel`` headed "Tools"
    sits after Thinking and before the session log."""
    p = _p(run, "idle")
    assert p["rail_assist"] is False, (
        f"B-9: #assist-panel is still in the rail; main = {p['main_order']}"
    )
    assert p["tools"] and _plain(p["tools_heading"]) == "Tools", f"B-1: no Tools pane: {p!r}"
    order = p["main_order"]
    assert ".pane-thinking" in order and "#log-panel" in order, f"premise: {order}"
    assert (
        order.index(".pane-thinking") < order.index("#tools-panel") < order.index("#log-panel")
    ), (
        f"B-1: the Tools pane is in the slot where Context was (between Thinking and the session "
        f"log); main = {order}"
    )


@requires_chrome
def test_b6_the_assist_panel_lives_in_a_hidden_popover_outside_main(run):
    p = _p(run, "idle")
    assert p["gear"], "B-6: a header button with aria-label=Settings"
    assert p["popover"] and not p["popover_in_main"], (
        f"B-6: the popover exists, outside <main>: {p!r}"
    )
    assert p["panel_in_popover"] and not p["panel_in_main"], (
        f"B-6: #assist-panel is in the popover: {p!r}"
    )
    assert not p["popover_visible"], "B-6: the popover is closed until the gear is pressed"


@requires_chrome
def test_b6_the_gear_title_and_the_panel_carry_the_d12a_sentence_and_heading(run):
    p = _p(run, "open_dark")
    assert _plain(p["gear_title"]) == D12A_SENTENCE, (
        f"B-6: the gear's title; got {p['gear_title']!r}"
    )
    assert _plain(p["panel_heading"]) == "Context helpers", (
        f"B-6: heading; got {p['panel_heading']!r}"
    )
    assert D12A_SENTENCE in [_plain(t) for t in p["panel_paras"]], (
        f"B-6: the paragraph; {p['panel_paras']!r}"
    )


@requires_chrome
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_b6_the_gear_opens_the_popover_and_escape_closes_it(run, theme):
    opened, closed = _p(run, f"open_{theme}"), _p(run, f"escaped_{theme}")
    assert opened["theme"] == theme, (
        f"premise: the {theme} data-theme is in force: {opened['theme']!r}"
    )
    assert opened["popover_visible"] and opened["panel_visible"], (
        f"B-6: the gear opens it ({theme})"
    )
    assert not closed["popover_visible"], f"B-6: Escape closes it ({theme})"


@requires_chrome
def test_b6_an_outside_click_closes_the_popover(run):
    assert _p(run, "reopened")["popover_visible"], "premise: the gear reopened the popover"
    assert not _p(run, "outside")["popover_visible"], "B-6: an outside click closes the popover"


@requires_chrome
def test_b6_m_typed_inside_the_popover_does_not_mute(run):
    inside = _p(run, "m_inside")
    assert inside["popover_visible"], "premise: the key went to the open popover"
    assert inside["mute_enabled"], "B-6: `m` typed in the popover never triggers mute"


@requires_chrome
def test_b6_m_outside_the_popover_does_nothing_and_the_capture_still_runs(run):
    """T-0181 B3 (H-T181-4) re-scoped the old control half ("`m` outside the popover still mutes"): the
    ``m`` shortcut is removed, so ``m`` with nothing focused stops nothing and the capture keeps sending.
    The key did reach the page (a window capture listener saw it)."""
    outside = _p(run, "m_outside")
    assert "m" in run.m_outside_keys, (
        f"premise: the m keydown reached the page: {run.m_outside_keys}"
    )
    assert outside["mute_enabled"], "B3: `m` outside the popover stopped the capture"
    assert run.m_outside_frames >= 2, (
        f"B3: the capture stopped sending after m (frames at the stub grew by {run.m_outside_frames})"
    )


@requires_chrome
def test_b6_the_switches_work_through_the_echo(run):
    p = _p(run, "badge_on")
    assert run.toggles == 1, (
        f"B-6: one click in the popover sends one astra.assist.toggle; got {run.toggles}"
    )
    assert p["page_text_on"] == "true", "B-6: the popover row turns on from the server's echo"


@requires_chrome
def test_b7_the_badge_shows_in_the_header_beside_the_gear_only_while_an_assist_is_on(run):
    idle, on, off = _p(run, "idle"), _p(run, "badge_on"), _p(run, "badge_off")
    assert not idle["badge_visible"], (
        f"B-7: nothing shows before any astra.assist.state: {idle['badge_text']!r}"
    )
    assert not on["popover_visible"], (
        "premise: the popover is closed (hiding the switches must not hide the badge)"
    )
    assert on["badge_visible"] and on["badge_text"] == BADGE_ON, (
        f"B-7: the badge, verbatim: {on['badge_text']!r}"
    )
    g, b = on["gear_box"], on["badge_box"]
    gap = max(b["left"] - g["right"], g["left"] - b["right"], 0.0)
    assert gap <= BESIDE_PX, f"B-7: beside the gear (gap {gap:.1f} px > {BESIDE_PX})"
    assert not off["badge_visible"], (
        f"B-7: with every assist off nothing shows: {off['badge_text']!r}"
    )


@requires_chrome
def test_b7_a_state_pushed_while_the_popover_is_closed_updates_the_badge(run):
    """Reviewer T1: the badge is recomputed on every ``astra.assist.state``, popover or not.

    ``badge_on`` received its state while the popover was open; this one arrives after it closed,
    with a string no earlier phase showed, so a badge only rendered while the switches are on
    screen is caught here rather than surviving on leftover text.
    """
    p = _p(run, "badge_closed")
    assert not p["popover_visible"], "premise: the state arrived with the popover closed"
    assert p["badge_visible"] and p["badge_text"] == BADGE_CLOSED, (
        f"B-7: the closed-popover state's badge, verbatim ({BADGE_CLOSED!r}); "
        f"got {p['badge_text']!r}, visible={p['badge_visible']}"
    )


@requires_chrome
def test_b7_the_state_replayed_after_a_reconnect_shows_the_badge(run):
    """B-7 over a dropped socket: the page reconnects (a second hello), the state on the new
    socket shows its badge verbatim, and the popover was never reopened."""
    p = _p(run, "badge_replayed")
    assert run.hellos >= 2, f"premise: the page reconnected and said hello again ({run.hellos})"
    assert not _p(run, "badge_off")["badge_visible"], "premise: the badge was off before the drop"
    assert not p["popover_visible"], "premise: the popover is closed"
    assert p["badge_visible"] and p["badge_text"] == BADGE_REPLAYED, (
        f"B-7: the replayed badge, verbatim ({BADGE_REPLAYED!r}); got {p['badge_text']!r}, "
        f"visible={p['badge_visible']}"
    )


@requires_chrome
def test_b6_enter_on_the_focused_gear_opens_and_focuses_the_first_enabled_switch(run):
    """Reviewer T2 (a disclosure's keyboard contract; B-6 names no focus rule, see the report)."""
    before, opened = _p(run, "kb_before"), _p(run, "kb_open")
    assert before["on_gear"] and not before["popover_visible"], f"premise: {before!r}"
    assert opened["popover_visible"], "B-6: Enter on the focused gear opens the popover"
    assert opened["expanded"] == "true", (
        f'the open gear says aria-expanded="true"; got {opened["expanded"]!r}'
    )
    assert opened["on_first_switch"], (
        f"opening moves focus to the popover's first enabled switch; focus is on {opened['active']}"
    )


@requires_chrome
def test_b6_escape_in_the_popover_closes_it_and_returns_focus_to_the_gear(run):
    p = _p(run, "kb_escaped")
    assert _p(run, "kb_open")["in_popover"], "premise: Escape was pressed inside the popover"
    assert not p["popover_visible"], "B-6: Escape closes the popover"
    assert p["on_gear"], f"Escape gives focus back to the gear; focus is on {p['active']}"
    assert p["expanded"] == "false", (
        f'the closed gear says aria-expanded="false"; got {p["expanded"]!r}'
    )


@requires_chrome
def test_b6_escape_with_focus_outside_the_popover_closes_it_too(run):
    """The document-level Escape: the key does not reach the popover's own handler."""
    opened, closed = _p(run, "doc_open"), _p(run, "doc_escaped")
    assert opened["popover_visible"] and not opened["in_popover"], (
        f"premise: open, with the focus outside it ({opened['active']})"
    )
    assert not closed["popover_visible"], "B-6: Escape closes the popover wherever the focus is"
    assert closed["expanded"] == "false", f"aria-expanded after Escape: {closed['expanded']!r}"


@requires_chrome
def test_b1_b2_the_pane_shows_the_calls_and_markup_in_arguments_is_text(run):
    p = _p(run, "tools")
    text = p["tools_text"] or ""
    assert "click_at (120, 845)" in text, f"B-2: short form in the pane: {text!r}"
    assert EVIL in text and not p["pwned"], "B-2: an <img onerror> argument is shown as text"


@requires_chrome
def test_b8_the_theme_select_and_the_inline_tool_cards_stay(run):
    p = _p(run, "tools")
    assert p["theme_in_header"], "B-8 / H-T154-3: the theme select stays in the header"
    assert p["thread_cards"] == 2, (
        f"B-8 / H-T162-5: the inline tool cards stay; got {p['thread_cards']}"
    )


@requires_chrome
def test_b8_the_page_requests_nothing_off_loopback(run):
    assert run.error is None, run.error
    assert run.requests, "premise: the page's own requests were recorded"
    off = [
        u
        for u in run.requests
        if urllib.parse.urlsplit(u).hostname not in ("127.0.0.1", "localhost", "::1")
        and not u.startswith(("data:", "blob:"))
    ]
    assert off == [], f"B-8: off-loopback requests (no web font, no CDN): {off}"
