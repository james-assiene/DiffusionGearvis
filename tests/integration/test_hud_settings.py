"""T-0213 group B-read: the read-only Settings panel in the real page (B-1 .. B-4).

``docs/tasks/T-0213.md`` B-1 to B-4 and *Tests to write* "test_hud_settings.py" (B-read rows; A2-apply extends
this module later). The page side only: the panel renders what an ``astra.settings.state`` (A2r-1) says.

**Two sources for that message** (A2-read is built in parallel):

- *synthetic*: the rows post a hand-built ``astra.settings.state`` into the page's own socket (a ``MessageEvent``
  dispatched on the page's ``WebSocket``, so it goes through ``ws.js`` and the page's decoder exactly as a server
  line would). The init script drops any ``astra.settings.state`` the server itself sends while a synthetic row
  runs, so the panel shows the synthetic payload only. The payload is built from ``local_astra/settings_catalog.py``
  (the task's single source of truth for the text), with distinctive values and sources so a page that renders a
  hard-coded catalog, or the defaults, fails. When ``contracts.protocol`` has ``AstraSettingsState`` (A2-read
  merged), the payload is validated by that model and encoded by the production codec; until then it is a dict in
  A2r-1's shape.
- *server*: ``test_server_state_is_what_the_panel_shows`` reads the ``astra.settings.state`` the real
  ``HudServer.hello_reply`` sent (A2r-3) and checks the page against it.

**DOM contract these rows pin** (the task fixes the behaviour; the hooks are named here, and the implementer
provides them):

==========================================  =================================================================
``[data-settings-popover] [data-settings-panel]``  the panel, inside the existing Settings popover, after
                                            ``#assist-panel`` ("below Context helpers", B-1)
``[data-settings-group="<group>"]``         one per group, in B-1 order; its text names the group
``[data-setting-key="<key>"]``              one per catalog row, inside its group, in catalog order
``[data-setting-label]``                    the label (textContent == catalog ``label``)
``[data-setting-value]``                    the current value (non-empty; contains ``str(value)`` for a str/int)
``[data-setting-source="<source>"]``        the source badge (attribute == the item's ``source``; text non-empty)
``[data-setting-kind="<kind>"]``            the kind chip (attribute == kind; text == the B-1 chip)
``[data-setting-what]``, ``[data-setting-useful]``, ``[data-setting-not-useful]``, ``[data-setting-evidence]``,
``[data-setting-cost]``                     the catalog's ``what``, ``useful_when``, ``not_useful_when``,
                                            ``evidence``, ``cost`` (textContent, trimmed, == the string)
``code[data-setting-relaunch]``             a row with a relaunch command: the command, verbatim (B-2)
``button[data-setting-copy]``               beside it: Copy, enabled; puts the command on the clipboard
``[data-setting-control]``                  a row's value control (input/select/button); in B-read every one is
                                            ``disabled`` with ``title`` == :data:`DISABLED_TITLE` (B-3)
==========================================  =================================================================

Chips (B-1): ``next_turn`` "next turn", ``next_turn_cold`` "next turn, slower first reply", ``relaunch`` "needs
relaunch". A ``fixed`` row's chip text is not pinned (the task gives three texts for four kinds); it must not be one
of the two changeable chips.

**Rows** (the new-code rows were activated with A2-read; the negative rows hold at BASE by design and guard B-4):

- ``test_the_gear_opens_the_panel_with_every_catalog_row`` (B-1, synthetic)
- ``test_relaunch_rows_are_read_only_with_their_command_and_a_copy_button`` (B-2, synthetic)
- ``test_in_b_read_every_control_is_disabled_with_the_title_and_nothing_is_sent`` (B-3, synthetic; re-pinned by
  R-24: relaunch, not_here and fixed rows stay disabled with the title; nothing is sent until a control changes),
  with ``test_a_row_the_session_cannot_change_has_no_enabled_control_and_sends_nothing`` (adversarial companion)
- A2-apply page rows (T-0213 group C tests, activated): see the A2-apply section below, with its
  ``LiveTools`` rig (a live session over ``FakeLiveBrowser``) and a rig self-check that holds at 96a4d27b
- ``test_markup_in_the_evidence_renders_as_text`` (B-1 adversarial, synthetic)
- ``test_escape_outside_click_and_focus_behave_as_t0162_b6`` (B-4, with the panel)
- ``test_server_state_is_what_the_panel_shows`` (B-1 with A2-read's real hello message)
- ``test_the_assist_panel_and_the_theme_select_behave_as_before`` (B-4 negative; holds at BASE)
- ``test_the_page_requests_nothing_off_loopback`` (B-4 negative; holds at BASE)
- ``test_b4_frozen_page_files_are_byte_identical`` (B-4: ``hud/vendor/**``, ``ws.js``, ``assists/gate.js``;
  ``RANGE_BASE..TIP`` and ``BASE..TIP``, ``TIP`` = 2eee3f82)
- panel review rulings (2026-10-31): ``test_r10_a_not_changeable_next_turn_item_shows_its_items_chip`` (R-10/R-11),
  ``test_r12_the_relaunch_box_shows_only_the_flag_to_add`` (R-12),
  ``test_r12_adversarial_copy_in_a_no_web_search_off_session_gives_only_the_flag`` (R-12 adversarial; holds at
  BASE), ``test_a_later_state_without_a_catalog_updates_the_values`` (A2-apply hello row; holds at BASE, kills
  "ignore states without a catalog"). ``_check_rows`` now also pins R-10 (chip from ``item.kind``, ``not_here``),
  R-11 (a null value reads "doesn't apply here") and R-12 (the box label); activated with R-10a/R-12.

**The rig**: ``tests/integration/test_hud_rehearsal.py``'s ``Live`` (a real ``Session`` + real ``HudServer(port=0)``
over the fakes of ``tests/unit/_hud_rig.py``, ``SessionClock`` wall time, fake LLM), the page served from ``hud/``
over loopback http with ``?source=canvas``, the box's headless Chrome over CDP (``test_hud_headless.Chrome``, with
``Network.enable`` for the request log). No turn runs; no number here is a latency claim; every wait is a generous
ceiling. n = 1 per row. Runs only with ``ASTRA_BROWSER_TESTS=1`` (the frozen-files row runs without it).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import time
import typing
from pathlib import Path
from typing import Any

import pytest

from local_astra import settings_catalog as sc
from local_astra.conductor import live_settings
from local_astra.conductor.hud_server import HudServer
from local_astra.conductor.session import Session, SessionClock, SessionConfig
from local_astra.config import action_outcome_without_feedback_refusal
from local_astra.contracts import events as E
from local_astra.contracts import protocol
from local_astra.contracts.codec import encode
from local_astra.contracts.perception import AssistFlags
from local_astra.contracts.turn import TurnState
from local_astra.perception.browser_source import BrowserFrameSource
from tests import _git_range as gr
from tests.fakes import FakeMemory, FakeStt
from tests.integration.test_hud_headless import (
    CHROME,
    _RecordingChrome,
    requires_chrome,
    serve_hud,
)
from tests.integration.test_hud_rehearsal import CountingSink, Live, Page, answer, page_url
from tests.unit import test_session_tool_loop as tsl
from tests.unit._hud_rig import RecordingAudio, RecordingMachine, RecordingTts
from tests.unit._live_browser_support import (
    BLACK,
    FakeLiveBrowser,
    live_feedback,
    live_kit,
    scene_frame,
)
from tests.unit._tool_turn_support import VALID_CLICK, ToolLlm, tool

GATE_ENV = "ASTRA_BROWSER_TESTS"

pytestmark = [pytest.mark.integration]

BROWSER = [
    pytest.mark.browser,
    requires_chrome,
    pytest.mark.skipif(
        os.environ.get(GATE_ENV) != "1",
        reason=f"{GATE_ENV} is not 1: T-0213 B-read drives headless Chrome against a real Session",
    ),
]


def browser(fn: Any) -> Any:
    for mark in reversed(BROWSER):
        fn = mark(fn)
    return fn


#: B-1, verbatim.
B1_GROUPS = (
    "Seeing the screen",
    "Acting in the browser",
    "Checking its work",
    "Guidance and prompt",
    "Search and web",
    "Speech",
    "Engine (relaunch only)",
)
#: B-1, verbatim.
CHIPS = {
    "next_turn": "next turn",
    "next_turn_cold": "next turn, slower first reply",
    "relaunch": "needs relaunch",
    "not_here": "doesn't apply in this mode",  # R-10, verbatim
}
#: R-11, verbatim: what a null value reads.
NULL_VALUE_TEXT = "doesn't apply here"
#: R-12, verbatim: the relaunch box's label.
RELAUNCH_LABEL = "Add to your session command:"
#: R-12 as amended: browser_viewport keeps its config text under this label.
VIEWPORT_LABEL = "To change it:"
#: R-10: the effective kind of a live session's not-changeable rows that have a relaunch (hand-typed; the catalog
#: says next_turn_cold for both).
LIVE_EFFECTIVE_KIND = {"page_overview": "relaunch", "wait_tool": "relaunch"}
#: R-12: the flag(s) to add, per row with a relaunch, in a default live session (hand-typed from the catalog's
#: ``flag`` and the value its relaunch sets; the engine shows ``--profile <the profile in force>``). browser_viewport
#: has no flag (``flag == ""``; set in astra.toml) and keeps its catalog sentence (reported).
LIVE_RELAUNCH = {
    "page_overview": "--page-overview",
    "wait_tool": "--wait-tool off",
    "action_echo": "--action-echo pixels",
    "max_tool_steps": "--max-tool-steps N",
    "max_tool_steps_web": "--max-tool-steps N",
    "browser": "--browser headless",
    "browser_viewport": "set browser_viewport in astra.toml, then relaunch diffusion_gearvis session",
    "tools": "--tools live",
    "search": "--search off",
    "browser_web": "--no-web",
    "filler_clips": "--filler-clips",
    "engine": "--profile lean",
}
#: B-3, verbatim.
DISABLED_TITLE = "Changes from the panel arrive with the next update"
#: The A1-1 text fields, under settings_catalog's attribute names (A2r-1: "the A1-1 text fields").
TEXT_FIELDS = (
    "key",
    "label",
    "group",
    "kind",
    "what",
    "useful_when",
    "not_useful_when",
    "evidence",
    "cost",
)
#: row field in the probe -> catalog attribute
SHOWN = {
    "label": "label",
    "what": "what",
    "useful": "useful_when",
    "not_useful": "not_useful_when",
    "evidence": "evidence",
    "cost": "cost",
}

#: Distinctive values and sources in the synthetic state (none is the default/default pair).
OVERRIDES: dict[str, tuple[Any, str]] = {
    "action_outcome": (True, "flag"),
    "visual_tokens": (560, "profile"),
    "live_guidance": ("v2", "ui"),
    "live_feedback": ("label+extent", "flag"),
    "engine": ("lean", "profile"),
}
SYNTHETIC_REV = 7
SYNTHETIC_SHA = "0123456789ab"
EVIL_EVIDENCE = '<img src=x onerror="window.__t0213_pwned=1">'

PAGE_WAIT_S = 12.0


# --- the synthetic astra.settings.state -------------------------------------------------------------------------


def _jsonable(v: Any) -> Any:
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    return v


def _model_fields(name: str) -> set[str] | None:
    model = getattr(protocol, name, None)
    fields = getattr(model, "model_fields", None)
    return set(fields) if isinstance(fields, dict) else None


def synthetic_items(
    evidence: dict[str, str] | None = None,
    item_over: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(items, catalog) for every catalog row, in catalog order, with :data:`OVERRIDES` applied. Items are a
    default live session's as R-10/R-12 describe them: kind from :data:`LIVE_EFFECTIVE_KIND` (else the catalog's),
    relaunch from :data:`LIVE_RELAUNCH`; ``item_over`` replaces item fields per key."""
    catalog_fields = _model_fields("CatalogEntry")
    item_fields = _model_fields("SettingItem")
    items: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []
    for e in sc.CATALOG:
        value, source = OVERRIDES.get(e.key, (e.default, "default"))
        item = {
            "key": e.key,
            "value": _jsonable(value),
            "source": source,
            "kind": LIVE_EFFECTIVE_KIND.get(e.key, e.kind),
            "changeable": bool(e.v1_changeable),
            "pending": None,
            "relaunch": LIVE_RELAUNCH.get(e.key),
        }
        item.update((item_over or {}).get(e.key, {}))
        if item_fields is not None:
            item = {k: v for k, v in item.items() if k in item_fields}
        items.append(item)
        full = {f.name: _jsonable(getattr(e, f.name)) for f in dataclasses.fields(e)}
        if evidence and e.key in evidence:
            full["evidence"] = evidence[e.key]
        if catalog_fields is not None:
            entry = {k: v for k, v in full.items() if k in catalog_fields}
        else:
            entry = {k: full[k] for k in TEXT_FIELDS}
        catalog.append(entry)
    return items, catalog


def synthetic_state(
    evidence: dict[str, str] | None = None,
    item_over: dict[str, dict[str, Any]] | None = None,
    *,
    rev: int = SYNTHETIC_REV,
    with_catalog: bool = True,
) -> tuple[str, dict[str, Any]]:
    """The raw line to post, and the dict it was built from. ``with_catalog=False`` is a later state (A2r-1: the
    catalog rides on the hello's state only)."""
    items, catalog = synthetic_items(evidence, item_over)
    body: dict[str, Any] = {
        "type": "astra.settings.state",
        "rev": rev,
        "items": items,
        "catalog": catalog if with_catalog else None,
        "system_sha12": SYNTHETIC_SHA,
        "instructions": "",
        "instructions_max": 1000,
        "refused": [],
        "replay": False,
    }
    model = getattr(protocol, "AstraSettingsState", None)
    if model is not None:  # A2-read merged: the production model and codec make the line
        try:
            built = model.model_validate(body)
        except ValueError as err:  # pydantic's ValidationError is a ValueError
            raise AssertionError(f"the wire model refuses the synthetic state: {err}") from err
        return encode(built), body
    return json.dumps(body), body


# --- the page side ----------------------------------------------------------------------------------------------

#: Installed before any page script. Records every socket the page opens and every message type it sends; stubs
#: the clipboard (records what Copy writes); and, when ``block`` is true, drops the server's own
#: ``astra.settings.state`` so a synthetic row sees its payload only.
INIT_JS = r"""
(() => {
  const rec = {sockets: [], sent: [], copied: [], blocked: 0, injecting: false, block: __BLOCK__};
  Object.defineProperty(window, '__t0213', {value: rec, configurable: true});
  const clip = {
    writeText: (t) => { rec.copied.push(String(t)); return Promise.resolve(); },
    readText: () => Promise.resolve(''),
  };
  try { Object.defineProperty(Navigator.prototype, 'clipboard', {configurable: true, get() { return clip; }}); } catch (_) {}
  const exec = Document.prototype.execCommand;
  Document.prototype.execCommand = function (cmd, ...rest) {
    if (String(cmd).toLowerCase() === 'copy') {
      const a = document.activeElement;
      const sel = (a && typeof a.value === 'string' && a.selectionEnd > a.selectionStart)
        ? a.value.slice(a.selectionStart, a.selectionEnd) : String(window.getSelection() || '');
      rec.copied.push(sel);
      return true;
    }
    return exec.call(this, cmd, ...rest);
  };
  const Orig = window.WebSocket;
  const desc = Object.getOwnPropertyDescriptor(Orig.prototype, 'onmessage');
  class Tapped extends Orig {
    constructor(...args) { super(...args); rec.sockets.push(this); }
    send(data) {
      try { const m = JSON.parse(data); rec.sent.push(m && m.type); } catch (_) { rec.sent.push('<not json>'); }
      return super.send(data);
    }
  }
  Object.defineProperty(Tapped.prototype, 'onmessage', {
    configurable: true,
    get() { return this.__t0213_h || null; },
    set(h) {
      this.__t0213_h = h;
      desc.set.call(this, h == null ? h : (event) => {
        if (rec.block && !rec.injecting) {
          try {
            const m = JSON.parse(event.data);
            if (m && m.type === 'astra.settings.state') { rec.blocked += 1; return undefined; }
          } catch (_) {}
        }
        return h.call(this, event);
      });
    },
  });
  window.WebSocket = Tapped;
})();
"""


def init_js(*, block: bool) -> str:
    return INIT_JS.replace("__BLOCK__", "true" if block else "false")


def post_js(raw: str) -> str:
    return """(() => {
  const rec = window.__t0213;
  const s = rec && rec.sockets[rec.sockets.length - 1];
  if (!s) return 'no socket';
  rec.injecting = true;
  try { s.dispatchEvent(new MessageEvent('message', {data: __RAW__})); } finally { rec.injecting = false; }
  return 'posted';
})()""".replace("__RAW__", json.dumps(raw))


OPEN_JS = """(() => {
  const g = document.querySelector('header button[aria-label="Settings"]');
  if (!g) return 'no Settings gear';
  const p = document.querySelector('[data-settings-popover]');
  if (p && !p.hidden) return 'open';
  g.click();
  return (p && !p.hidden) ? 'open' : 'still closed';
})()"""

PANEL_JS = r"""
(() => {
  const pop = document.querySelector('[data-settings-popover]');
  const panel = pop && pop.querySelector('[data-settings-panel]');
  const vis = (el) => !!el && el.checkVisibility();
  if (!panel) return {present: false, popover_open: Boolean(pop) && !pop.hidden};
  const assist = pop.querySelector('#assist-panel');
  const one = (row, sel) => row.querySelector(sel);
  const text = (row, sel) => { const e = one(row, sel); return e ? e.textContent : null; };
  return {
    present: true,
    visible: vis(panel),
    popover_open: !pop.hidden,
    after_assist: assist ? Boolean(assist.compareDocumentPosition(panel) & Node.DOCUMENT_POSITION_FOLLOWING) : null,
    imgs: panel.querySelectorAll('img').length,
    scripts: panel.querySelectorAll('script').length,
    pwned: window.__t0213_pwned === undefined ? null : String(window.__t0213_pwned),
    groups: [...panel.querySelectorAll('[data-settings-group]')].map((g) => ({
      name: g.getAttribute('data-settings-group'),
      text: g.textContent,
      keys: [...g.querySelectorAll('[data-setting-key]')].map((r) => r.getAttribute('data-setting-key')),
    })),
    rows: [...panel.querySelectorAll('[data-setting-key]')].map((row) => {
      const src = one(row, '[data-setting-source]');
      const kind = one(row, '[data-setting-kind]');
      const code = one(row, '[data-setting-relaunch]');
      const copy = one(row, '[data-setting-copy]');
      return {
        key: row.getAttribute('data-setting-key'),
        label: text(row, '[data-setting-label]'),
        value: text(row, '[data-setting-value]'),
        source: src ? src.getAttribute('data-setting-source') : null,
        source_text: src ? src.textContent : null,
        kind: kind ? kind.getAttribute('data-setting-kind') : null,
        chip: kind ? kind.textContent : null,
        what: text(row, '[data-setting-what]'),
        useful: text(row, '[data-setting-useful]'),
        not_useful: text(row, '[data-setting-not-useful]'),
        evidence: text(row, '[data-setting-evidence]'),
        cost: text(row, '[data-setting-cost]'),
        relaunch_tag: code ? code.tagName.toLowerCase() : null,
        relaunch: code ? code.textContent : null,
        relaunch_box: code && code.parentElement ? code.parentElement.textContent : null,
        copy: copy ? {tag: copy.tagName.toLowerCase(), disabled: Boolean(copy.disabled), text: copy.textContent.trim()} : null,
        controls: [...row.querySelectorAll('[data-setting-control]')].map((c) => ({
          tag: c.tagName.toLowerCase(), disabled: Boolean(c.disabled), title: c.getAttribute('title'),
        })),
        enabled_inputs: [...row.querySelectorAll('input, select, textarea, button')]
          .filter((e) => !e.disabled && !e.matches('[data-setting-copy]'))
          .map((e) => e.outerHTML.slice(0, 120)),
      };
    }),
  };
})()
"""

REC_JS = (
    "(() => { const r = window.__t0213; return r ? {sent: r.sent.slice(), copied: r.copied.slice(), "
    "blocked: r.blocked, sockets: r.sockets.length} : null; })()"
)


class _SettingsChrome(_RecordingChrome):
    """``_RecordingChrome`` (the request log) plus an init script."""

    def add_init_script(self, source: str) -> None:
        self._cmd("Page.enable")
        self._cmd("Page.addScriptToEvaluateOnNewDocument", source=source)


@contextlib.asynccontextmanager
async def settings_page(
    tmp_path: Path,
    *,
    block: bool,
    make_live: typing.Callable[[Path], Live] | None = None,
    script: str | None = None,
) -> typing.AsyncIterator[tuple[Live, Page, _SettingsChrome]]:
    """A started ``Live``, the page over loopback http, one connected tab with the init script and the network log.

    ``make_live`` builds another rig (A2-apply: :class:`LiveTools`, a live session); ``script`` replaces the init
    script (A2-apply: :func:`init_js_full`, which also records every message the page sends)."""
    assert CHROME is not None
    live = make_live(tmp_path) if make_live is not None else Live(tmp_path, [answer("Fine. ")])
    await live.start()
    hud, hud_port = serve_hud()
    chrome = _SettingsChrome(CHROME).__enter__()
    try:
        await asyncio.to_thread(chrome._cmd, "Network.enable")
        await asyncio.to_thread(
            chrome.add_init_script, script if script is not None else init_js(block=block)
        )
        page = Page(chrome)
        await page.navigate(page_url(hud_port, live.server.url))
        await page.connected()
        yield live, page, chrome
    finally:
        with contextlib.suppress(Exception):
            chrome.__exit__()
        hud.shutdown()
        await live.aclose()


#: T-0218 B: the page's last recorded socket is OPEN (what :func:`post_js` dispatches on).
SOCKET_OPEN_JS = (
    "(() => { const r = window.__t0213; const s = r && r.sockets[r.sockets.length - 1];"
    " return Boolean(s) && s.readyState === 1; })()"
)
#: what the page looked like when no open socket appeared: a reload or a reconnect shows here.
SOCKET_DIAG_JS = """(() => {
  const r = window.__t0213;
  const nav = performance.getEntriesByType('navigation');
  const c = document.querySelector('#conn-status');
  return {
    recorder: Boolean(r),
    sockets: r ? r.sockets.length : null,
    ready_states: r ? r.sockets.map((s) => s.readyState) : null,
    urls: r ? r.sockets.map((s) => s.url) : null,
    href: location.href,
    navigation_entries: nav.length,
    navigation_type: nav.length ? nav[0].type : null,
    legacy_navigation_type: performance.navigation ? performance.navigation.type : null,
    conn_status: c ? c.textContent : null,
    time_origin_age_ms: Math.round(performance.now()),
  };
})()"""


async def post_synthetic(page: Page, raw: str) -> None:
    if not await page.evaluate(SOCKET_OPEN_JS):
        # not open at the first look: note what the page was doing (opt-in file, for diagnosing the race)
        note = os.environ.get("ASTRA_T0218_SOCKET_DIAG")
        if note:
            diag = await page.evaluate(SOCKET_DIAG_JS)
            with open(note, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"first_look": diag}) + "\n")
    open_now = await settle(page, SOCKET_OPEN_JS)
    if not open_now:
        diag = await page.evaluate(SOCKET_DIAG_JS)
        raise AssertionError(
            f"rig: no OPEN socket in window.__t0213.sockets after {PAGE_WAIT_S:.0f} s, so the synthetic state "
            f"cannot be posted; page diagnostics: {diag!r}"
        )
    outcome = await page.evaluate(post_js(raw))
    assert outcome == "posted", (
        f"rig: could not post the synthetic astra.settings.state: {outcome!r}"
    )


async def open_panel(page: Page, n_rows: int) -> dict[str, Any]:
    outcome = await page.evaluate(OPEN_JS)
    assert outcome == "open", f"B-1: the Settings gear opens the popover: {outcome!r}"
    await page.until(
        "(() => { const p = document.querySelector('[data-settings-popover] [data-settings-panel]');"
        f" return Boolean(p) && p.querySelectorAll('[data-setting-key]').length === {n_rows}; }})()",
        f"B-1: the Settings panel shows {n_rows} rows (one per catalog entry) inside the popover",
        timeout_s=PAGE_WAIT_S,
    )
    probe = await page.evaluate(PANEL_JS)
    assert isinstance(probe, dict) and "__exception__" not in probe, (
        f"the panel probe threw: {probe!r}"
    )
    return probe


async def rec(page: Page) -> dict[str, Any]:
    value = await page.evaluate(REC_JS)
    assert isinstance(value, dict), f"rig: the init script's recorder is missing: {value!r}"
    return value


#: T-0218 B: a window that proves *nothing more* happens (no extra copy, no extra set) after a settled condition.
QUIET_S = 0.5


async def settle(page: Page, expression: str, timeout_s: float = PAGE_WAIT_S) -> Any:
    """Poll ``expression`` until it is truthy or ``timeout_s`` passes, and return its last value. Never raises on a
    timeout: the row's own assertion, unchanged, then reads the page and fails with its own message (T-0218 B: a
    condition wait in place of a fixed sleep for an effect)."""
    deadline = time.monotonic() + timeout_s
    last: Any = None
    while time.monotonic() < deadline:
        last = await page.evaluate(expression)
        if last and not (isinstance(last, dict) and "__exception__" in last):
            return last
        await asyncio.sleep(0.05)
    return last


async def copied_settled(page: Page, n: int) -> dict[str, Any]:
    """The recorder once ``n`` Copy payloads are on the clipboard stub (bounded by :data:`PAGE_WAIT_S`), then a
    :data:`QUIET_S` window started after that, so an extra copy still shows; the row asserts the list itself."""
    await settle(
        page,
        f"(() => {{ const r = window.__t0213; return Boolean(r) && r.copied.length >= {n}; }})()",
    )
    await asyncio.sleep(QUIET_S)
    return await rec(page)


def expected_groups() -> list[tuple[str, list[str]]]:
    """B-1's groups in B-1 order, each with its catalog keys in catalog order (the data module is the oracle)."""
    return [
        (g, [e.key for e in sc.CATALOG if e.group == g])
        for g in B1_GROUPS
        if any(e.group == g for e in sc.CATALOG)
    ]


def _check_rows(probe: dict[str, Any], body: dict[str, Any]) -> list[str]:
    """Every row against the state it came from: text fields, value, source, chip. Returns the problems."""
    problems: list[str] = []
    catalog = {e["key"]: e for e in body["catalog"]}
    items = {i["key"]: i for i in body["items"]}
    rows = {r["key"]: r for r in probe["rows"]}
    for key, entry in catalog.items():
        row = rows.get(key)
        if row is None:
            problems.append(f"{key}: no [data-setting-key] row")
            continue
        for shown, attr in SHOWN.items():
            want = entry.get(attr)
            if want is None:
                continue
            got = row[shown]
            if got is None or got.strip() != str(want).strip():
                problems.append(f"{key}: [{shown}] reads {got!r}, the catalog's {attr} is {want!r}")
        item = items[key]
        value_text = row["value"]
        if not (value_text or "").strip():
            problems.append(f"{key}: [data-setting-value] is empty")
        elif item.get("value") is None and value_text.strip() != NULL_VALUE_TEXT:
            problems.append(
                f"{key}: a null value reads {value_text!r}, not {NULL_VALUE_TEXT!r} (R-11)"
            )
        elif "not reported" in value_text:
            problems.append(f"{key}: the value reads 'not reported' (R-11)")
        elif (
            isinstance(item.get("value"), (str, int))
            and not isinstance(item.get("value"), bool)
            and item["value"] != ""
            and str(item["value"]) not in value_text
        ):
            problems.append(f"{key}: the value {item['value']!r} is not shown: {value_text!r}")
        if row["source"] != item["source"]:
            problems.append(
                f"{key}: data-setting-source is {row['source']!r}, the item says {item['source']!r}"
            )
        if not (row["source_text"] or "").strip():
            problems.append(f"{key}: the source badge has no text")
        # R-10 re-pin: the chip comes from the item's effective kind (item.kind), never the catalog entry's kind.
        kind = item["kind"]
        if row["kind"] != kind:
            problems.append(f"{key}: data-setting-kind is {row['kind']!r}, not {kind!r}")
        chip = (row["chip"] or "").strip()
        if kind in CHIPS and chip != CHIPS[kind]:
            problems.append(f"{key}: the {kind} chip reads {chip!r}, not {CHIPS[kind]!r}")
        if kind not in CHIPS and chip in (CHIPS["next_turn"], CHIPS["next_turn_cold"]):
            problems.append(f"{key}: the {kind} row claims a changeable chip {chip!r}")
        if kind == "not_here" and (row["relaunch"] is not None or row["copy"] is not None):
            # R-10a: the wire may still carry relaunch text for a not_here row; the page shows no box for it.
            problems.append(
                f"{key}: a not_here row shows a relaunch box or Copy ({row['relaunch']!r})"
            )
        if row["relaunch"] is not None:
            box = (row["relaunch_box"] or "").strip()
            label = VIEWPORT_LABEL if key == "browser_viewport" else RELAUNCH_LABEL
            if not box.startswith(label):
                problems.append(f"{key}: the relaunch box is not labelled {label!r}: {box!r}")
    return problems


# --- rows -------------------------------------------------------------------------------------------------------


def test_premise_the_catalog_groups_are_b1s() -> None:
    """Premise of every browser row: the data module's groups are B-1's, in B-1's order (A1 holds this)."""
    assert tuple(sc.GROUPS) == B1_GROUPS
    assert {e.group for e in sc.CATALOG} <= set(B1_GROUPS)


@browser
async def test_the_gear_opens_the_panel_with_every_catalog_row(tmp_path):
    """B-1 (synthetic state): the gear opens the panel below Context helpers; the groups are in B-1 order; every
    catalog row appears once, in its group, in catalog order, with label, value, source badge, kind chip and the
    five text fields verbatim. The synthetic values/sources (:data:`OVERRIDES`) must show, not the defaults."""
    raw, body = synthetic_state()
    async with settings_page(tmp_path, block=True) as (_live, page, _chrome):
        await post_synthetic(page, raw)
        probe = await open_panel(page, len(sc.CATALOG))
    assert probe["visible"] and probe["popover_open"], (
        f"B-1: the panel is visible in the open popover: {probe}"
    )
    assert probe["after_assist"] is True, (
        "B-1: the panel mounts below Context helpers (#assist-panel)"
    )
    got = [(g["name"], g["keys"]) for g in probe["groups"]]
    assert got == expected_groups(), (
        f"B-1: groups in B-1 order with their rows in catalog order; got {got}"
    )
    for g in probe["groups"]:
        assert g["name"] in (g["text"] or ""), (
            f"B-1: the group section names its group: {g['name']!r}"
        )
    keys = [r["key"] for r in probe["rows"]]
    assert sorted(keys) == sorted(e.key for e in sc.CATALOG) and len(set(keys)) == len(keys), keys
    problems = _check_rows(probe, body)
    assert problems == [], "B-1, per row:\n  " + "\n  ".join(problems)
    rows = {r["key"]: r for r in probe["rows"]}
    assert rows["action_outcome"]["source"] == "flag" and rows["live_guidance"]["source"] == "ui", (
        "anti-vacuity: the synthetic sources are shown (a page rendering defaults fails here)"
    )
    assert "560" in (rows["visual_tokens"]["value"] or ""), rows["visual_tokens"]


@browser
async def test_relaunch_rows_are_read_only_with_their_command_and_a_copy_button(tmp_path):
    """B-2 (synthetic state): every row with a relaunch command shows it verbatim in a ``<code>`` with an enabled
    Copy button that puts exactly that command on the clipboard; a relaunch or fixed row has no enabled input; a
    row with no command shows no ``<code>`` relaunch and no Copy."""
    raw, body = synthetic_state()
    async with settings_page(tmp_path, block=True) as (_live, page, _chrome):
        await post_synthetic(page, raw)
        probe = await open_panel(page, len(sc.CATALOG))
        commands = {i["key"]: i["relaunch"] for i in body["items"] if i.get("relaunch")}
        assert commands, "premise: the catalog has relaunch rows"
        clicked = []
        for key in commands:
            outcome = await page.evaluate(
                "(() => { const b = document.querySelector("
                + json.dumps(f'[data-setting-key="{key}"] [data-setting-copy]')
                + "); if (!b) return 'no copy'; if (b.disabled) return 'disabled'; b.click(); return 'clicked'; })()"
            )
            clicked.append((key, outcome))
        recorded = await copied_settled(page, len(commands))
    rows = {r["key"]: r for r in probe["rows"]}
    problems = []
    for key, command in commands.items():
        row = rows[key]
        if row["relaunch_tag"] != "code":
            problems.append(f"{key}: the command is not in a <code> ({row['relaunch_tag']!r})")
        if row["relaunch"] != command:
            problems.append(f"{key}: the <code> reads {row['relaunch']!r}, not {command!r}")
        copy = row["copy"]
        if not copy or copy["tag"] != "button" or copy["disabled"] or copy["text"] != "Copy":
            problems.append(f"{key}: no enabled <button> reading Copy: {copy!r}")
    for e in sc.CATALOG:
        row = rows[e.key]
        if e.kind in ("relaunch", "fixed") and row["enabled_inputs"]:
            problems.append(
                f"{e.key}: a {e.kind} row has an enabled control: {row['enabled_inputs']}"
            )
        if e.key not in commands and (row["relaunch"] is not None or row["copy"] is not None):
            problems.append(f"{e.key}: no command, yet a relaunch <code> or Copy is shown")
    assert problems == [], "B-2:\n  " + "\n  ".join(problems)
    assert all(o == "clicked" for _, o in clicked), clicked
    assert recorded["copied"] == list(commands.values()), (
        f"B-2: each Copy puts its command on the clipboard, verbatim; got {recorded['copied']!r}"
    )


@browser
async def test_in_b_read_every_control_is_disabled_with_the_title_and_nothing_is_sent(tmp_path):
    """B-3 re-pinned by R-24 (2026-10-31: C supersedes B-3's "every control disabled"; named spec change in the C
    tests). Synthetic state of a default live session: every row whose item kind is ``relaunch``, ``not_here`` or
    ``fixed`` has no enabled input besides Copy, and each ``[data-setting-control]`` it has is disabled with the
    title :data:`DISABLED_TITLE`; with the panel open and nothing touched, nothing is sent (no ``astra.settings.*``
    leaves the page within 0.8 s); forcing a click and a change event on those disabled controls sends nothing
    either. (B-read's "every v1-changeable row has a control" moved to
    ``test_a2_apply_changeable_rows_have_enabled_controls``.)"""
    raw, body = synthetic_state()
    read_only = {i["key"] for i in body["items"] if i["kind"] in ("relaunch", "not_here", "fixed")}
    assert {"engine", "thinking", "page_overview"} <= read_only, f"premise: {sorted(read_only)}"
    async with settings_page(tmp_path, block=True) as (live, page, _chrome):
        await post_synthetic(page, raw)
        probe = await open_panel(page, len(sc.CATALOG))
        before = await rec(page)
        await asyncio.sleep(0.8)
        untouched = await rec(page)
        keys = json.dumps(sorted(read_only))
        forced = await page.evaluate(
            "(() => { let n = 0; for (const k of "
            + keys
            + ") for (const c of document.querySelectorAll("
            "'[data-settings-panel] [data-setting-key=\"' + k + '\"] [data-setting-control]'))"
            " { n += 1; c.click(); if ('checked' in c) c.checked = !c.checked;"
            " if (c.tagName === 'SELECT' && c.options.length > 1) c.selectedIndex = (c.selectedIndex + 1) % c.options.length;"
            " c.dispatchEvent(new Event('input', {bubbles: true})); c.dispatchEvent(new Event('change', {bubbles: true})); }"
            " return n; })()"
        )
        await asyncio.sleep(0.8)
        after = await rec(page)
        settings_states = [
            m for m in live.wire if getattr(m, "type", None) == "astra.settings.state"
        ]
    rows = {r["key"]: r for r in probe["rows"]}
    problems = []
    for key in sorted(read_only):
        row = rows[key]
        for c in row["controls"]:
            if not c["disabled"]:
                problems.append(f"{key}: a {c['tag']} control is enabled")
            if c["title"] != DISABLED_TITLE:
                problems.append(
                    f"{key}: the control's title is {c['title']!r}, not {DISABLED_TITLE!r}"
                )
        if row["enabled_inputs"]:
            problems.append(f"{key}: enabled inputs besides Copy: {row['enabled_inputs']}")
    assert problems == [], "B-3 / R-24:\n  " + "\n  ".join(problems)
    assert isinstance(forced, int), f"the forcing probe threw: {forced!r}"
    idle = untouched["sent"][len(before["sent"]) :]
    assert not [t for t in idle if str(t).startswith("astra.settings")], (
        f"R-24: nothing is sent until a control is changed; the page sent {idle}"
    )
    new = after["sent"][len(before["sent"]) :]
    assert all(t in ("astra.frame",) for t in new), (
        f"nothing but frames left the page after forcing the read-only controls; it sent {new}"
    )
    assert not any(str(t).startswith("astra.settings") for t in after["sent"]), after["sent"]
    assert len(settings_states) == 1, (
        f"no set was routed: the only settings state on the wire is the hello's; got {len(settings_states)}"
    )


@browser
async def test_a_row_the_session_cannot_change_has_no_enabled_control_and_sends_nothing(tmp_path):
    """R-24's adversarial companion (holds at BASE and after C): the page enables by the item's ``changeable``, not
    by its kind. In a synthetic state where **no** item is changeable (``changeable: false``, every item
    keeping its catalog kind, so a page that enables by kind instead of by ``changeable`` fails), no row has an
    enabled control besides Copy, every ``[data-setting-control]`` is disabled, and forcing a click and a change
    event on each sends nothing (no ``astra.settings.*`` leaves the page, the server routes none). A page that renders
    no control for such a row passes the enabled half; the forcing half then has nothing to force (allowed)."""
    over = {e.key: {"changeable": False} for e in sc.CATALOG}
    raw, _body = synthetic_state(item_over=over)
    async with settings_page(tmp_path, block=True) as (live, page, _chrome):
        await post_synthetic(page, raw)
        probe = await open_panel(page, len(sc.CATALOG))
        before = await rec(page)
        forced = await page.evaluate(
            "(() => { let n = 0; for (const c of document.querySelectorAll('[data-settings-panel] [data-setting-control]'))"
            " { n += 1; c.click(); if ('checked' in c) c.checked = !c.checked;"
            " if (c.tagName === 'SELECT' && c.options.length > 1) c.selectedIndex = (c.selectedIndex + 1) % c.options.length;"
            " c.dispatchEvent(new Event('input', {bubbles: true})); c.dispatchEvent(new Event('change', {bubbles: true})); }"
            " return n; })()"
        )
        await asyncio.sleep(0.8)
        after = await rec(page)
        settings_states = [
            m for m in live.wire if getattr(m, "type", None) == "astra.settings.state"
        ]
    rows = {r["key"]: r for r in probe["rows"]}
    problems = []
    for e in sc.CATALOG:
        row = rows[e.key]
        for c in row["controls"]:
            if not c["disabled"]:
                problems.append(
                    f"{e.key}: a {c['tag']} control is enabled on a not-changeable item"
                )
        if row["enabled_inputs"]:
            problems.append(f"{e.key}: enabled inputs besides Copy: {row['enabled_inputs']}")
    assert problems == [], "B-3 / A2-apply:\n  " + "\n  ".join(problems)
    assert isinstance(forced, int), f"the forcing probe threw: {forced!r}"
    new = after["sent"][len(before["sent"]) :]
    assert all(t in ("astra.frame",) for t in new), (
        f"nothing but frames left the page after forcing the disabled controls; it sent {new}"
    )
    assert not any(str(t).startswith("astra.settings") for t in after["sent"]), after["sent"]
    assert len(settings_states) == 1, (
        "no set was routed: the only settings state on the wire is the hello's; got "
        f"{len(settings_states)}"
    )


@browser
async def test_markup_in_the_evidence_renders_as_text(tmp_path):
    """B-1 adversarial (synthetic state): an ``evidence`` holding ``<img src=x onerror=...>`` shows as that text,
    creates no element, runs nothing, and the page never requests ``x``."""
    raw, _body = synthetic_state(evidence={"reground": EVIL_EVIDENCE})
    async with settings_page(tmp_path, block=True) as (_live, page, chrome):
        await post_synthetic(page, raw)
        probe = await open_panel(page, len(sc.CATALOG))
        await asyncio.sleep(0.5)
        await asyncio.to_thread(chrome.drain, 1.0)
        requests = list(chrome.requests)
    rows = {r["key"]: r for r in probe["rows"]}
    assert (rows["reground"]["evidence"] or "").strip() == EVIL_EVIDENCE, (
        f"B-1: the evidence is set with textContent and reads verbatim; got {rows['reground']['evidence']!r}"
    )
    assert probe["imgs"] == 0 and probe["scripts"] == 0, (
        f"B-1: catalog text created markup: {probe['imgs']} img"
    )
    assert probe["pwned"] is None, "B-1: the onerror handler in the evidence ran"
    assert not [u for u in requests if u.split("?", 1)[0].rstrip("/").endswith("/x")], (
        f"B-1: the page fetched the src=x of the evidence markup: {requests}"
    )


@browser
async def test_escape_outside_click_and_focus_behave_as_t0162_b6(tmp_path):
    """B-4 (with the panel rendered): opening moves the focus into the popover; a click inside the settings panel
    keeps it open; Escape closes it and returns the focus to the gear; an outside click closes it."""
    raw, _body = synthetic_state()
    async with settings_page(tmp_path, block=True) as (_live, page, chrome):
        await post_synthetic(page, raw)
        await open_panel(page, len(sc.CATALOG))
        focus_in = await page.evaluate(
            "document.querySelector('[data-settings-popover]').contains(document.activeElement)"
        )
        inside = await page.evaluate(
            "(() => { const e = document.querySelector('[data-settings-panel] [data-setting-what]');"
            " if (!e) return 'no what'; e.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true})); e.click();"
            " return document.querySelector('[data-settings-popover]').hidden ? 'closed' : 'open'; })()"
        )
        await asyncio.to_thread(chrome.key, "Escape", code=27)
        await settle(
            page,
            "(() => { const p = document.querySelector('[data-settings-popover]');"
            " const g = document.getElementById('settings-button');"
            " return Boolean(p) && p.hidden && document.activeElement === g"
            " && g.getAttribute('aria-expanded') === 'false'; })()",
        )
        after_escape = await page.evaluate(
            "({hidden: document.querySelector('[data-settings-popover]').hidden,"
            " focus_on_gear: document.activeElement === document.getElementById('settings-button'),"
            " expanded: document.getElementById('settings-button').getAttribute('aria-expanded')})"
        )
        reopened = await page.evaluate(OPEN_JS)
        outside = await page.evaluate(
            "(() => { const m = document.querySelector('main'); m.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true}));"
            " m.click(); return document.querySelector('[data-settings-popover]').hidden ? 'closed' : 'open'; })()"
        )
    assert focus_in is True, "T-0162 B-6: opening the popover moves the focus into it"
    assert inside == "open", (
        f"B-4: a click inside the settings panel must not close the popover: {inside}"
    )
    assert after_escape == {"hidden": True, "focus_on_gear": True, "expanded": "false"}, (
        f"T-0162 B-6: Escape closes the popover and returns the focus to the gear: {after_escape}"
    )
    assert reopened == "open", reopened
    assert outside == "closed", f"T-0162 B-6: an outside click closes the popover: {outside}"


@browser
async def test_server_state_is_what_the_panel_shows(tmp_path):
    """B-1 against A2-read's real message: the ``astra.settings.state`` (with ``catalog``) that
    ``HudServer.hello_reply`` sends (A2r-3) is what the panel shows, row for row (needs A2-read and B)."""
    async with settings_page(tmp_path, block=False) as (live, page, _chrome):

        def states() -> list[Any]:
            return [
                m
                for reply in live.hello_replies
                for m in reply
                if getattr(m, "type", None) == "astra.settings.state"
            ]

        # T-0218 B: page.connected() can return before the tap records the hello reply. Wait for at least one
        # state (bounded, never raising), then a QUIET_S window so a second state would show; then assert exactly 1.
        with contextlib.suppress(AssertionError):
            await live.until(
                lambda: bool(states()), "the hello reply was recorded", timeout_s=PAGE_WAIT_S
            )
        await asyncio.sleep(QUIET_S)
        sent = states()
        assert len(sent) == 1, (
            f"A2r-3: one astra.settings.state on the hello reply; got {[getattr(m, 'type', m) for r in live.hello_replies for m in r]}"
        )
        body = json.loads(encode(sent[0]))
        assert body.get("catalog"), "A2r-1: the hello's state carries the catalog"
        probe = await open_panel(page, len(body["items"]))
        recorded = await rec(page)
    assert recorded["blocked"] == 0, "rig: nothing was blocked in the server-sourced row"
    keys = [r["key"] for r in probe["rows"]]
    assert sorted(keys) == sorted(i["key"] for i in body["items"]), keys
    by_group: dict[str, list[str]] = {}
    for e in body["catalog"]:
        by_group.setdefault(e["group"], []).append(e["key"])
    want = [(g, by_group[g]) for g in B1_GROUPS if g in by_group]
    got = [(g["name"], g["keys"]) for g in probe["groups"]]
    assert got == want, (
        f"B-1: groups in B-1 order with their rows in the server catalog's order; got {got}"
    )
    problems = _check_rows(probe, body)
    assert problems == [], "B-1 against the server's state:\n  " + "\n  ".join(problems)
    # R-12 against the real server: every shown relaunch is the flag to add, never a whole command.
    catalog = {e.key: e for e in sc.CATALOG}
    items = {i["key"]: i for i in body["items"]}
    r12 = []
    for row in probe["rows"]:
        code = row["relaunch"]
        if code is None or row["key"] == "browser_viewport":
            continue
        if "diffusion_gearvis" in code:
            r12.append(f"{row['key']}: a whole command {code!r}")
        flag = catalog[row["key"]].flag
        if flag and not code.startswith(flag):
            r12.append(f"{row['key']}: {code!r} does not start with the catalog flag {flag!r}")
    engine = {r["key"]: r for r in probe["rows"]}["engine"]["relaunch"]
    # The bare-Session rig has no cli start, so the engine item may carry no value; then the relaunch is the
    # catalog's fallback (its whole command ends in ``--profile lean``; hand-typed). The value case (the profile in
    # force) is pinned by tests/unit/test_t0213_settings_wire.py::test_r12_engine_relaunch_is_the_profile_in_force.
    engine_value = items["engine"].get("value")
    want_engine = "--profile lean" if engine_value is None else f"--profile {engine_value}"
    if engine != want_engine:
        r12.append(f"engine: {engine!r}, not {want_engine!r} (engine value {engine_value!r})")
    assert r12 == [], "R-12 against the server's state:\n  " + "\n  ".join(r12)


async def _copy(page: Page, key: str) -> str:
    return typing.cast(
        str,
        await page.evaluate(
            "(() => { const b = document.querySelector("
            + json.dumps(f'[data-setting-key="{key}"] [data-setting-copy]')
            + "); if (!b) return 'no copy'; if (b.disabled) return 'disabled'; b.click(); return 'clicked'; })()"
        ),
    )


@browser
async def test_r10_a_not_changeable_next_turn_item_shows_its_items_chip(tmp_path):
    """R-10 (synthetic state): the catalog says ``next_turn`` for ReGround and Action outcome; the items say
    ``not_here`` (changeable false, no relaunch; outside live) and ``relaunch`` (changeable false, a flag). Each row
    shows its item's chip, never "next turn". Kills a page reading ``entry.kind``."""
    over = {
        "reground": {"kind": "not_here", "changeable": False, "value": None, "relaunch": None},
        "action_outcome": {"kind": "relaunch", "changeable": False, "relaunch": "--action-outcome"},
    }
    raw, _body = synthetic_state(item_over=over)
    assert {e.key: e.kind for e in sc.CATALOG if e.key in over} == {
        "reground": "next_turn",
        "action_outcome": "next_turn",
    }, "premise: both catalog kinds are next_turn"
    async with settings_page(tmp_path, block=True) as (_live, page, _chrome):
        await post_synthetic(page, raw)
        probe = await open_panel(page, len(sc.CATALOG))
    rows = {r["key"]: r for r in probe["rows"]}
    got = {k: ((rows[k]["chip"] or "").strip(), rows[k]["kind"]) for k in over}
    assert got == {
        "reground": ("doesn't apply in this mode", "not_here"),
        "action_outcome": ("needs relaunch", "relaunch"),
    }, f"R-10: the chip is the item's; got {got}"
    assert (rows["reground"]["value"] or "").strip() == NULL_VALUE_TEXT, (
        f"R-11: {rows['reground']['value']!r}"
    )
    for k in over:
        assert rows[k]["enabled_inputs"] == [], f"{k}: a not-changeable row has an enabled control"


@browser
async def test_r10a_a_not_here_row_with_relaunch_text_shows_no_relaunch_box(tmp_path):
    """R-10a (synthetic state, outside live): a live-only row is ``not_here`` even when it has a command, and the
    wire still carries that command as ``relaunch`` (``tools`` carries ``--tools live``; Page overview
    ``--page-overview``). The page shows the not_here chip and no relaunch ``<code>`` and no Copy for such a row.
    Kills a page that shows ``item.relaunch`` whenever it is non-empty, whatever the kind."""
    over = {
        "page_overview": {"kind": "not_here", "changeable": False, "relaunch": "--page-overview"},
        "wait_tool": {"kind": "not_here", "changeable": False, "relaunch": "--wait-tool off"},
    }
    raw, _body = synthetic_state(item_over=over)
    async with settings_page(tmp_path, block=True) as (_live, page, _chrome):
        await post_synthetic(page, raw)
        probe = await open_panel(page, len(sc.CATALOG))
    rows = {r["key"]: r for r in probe["rows"]}
    got = {
        k: (rows[k]["kind"], (rows[k]["chip"] or "").strip(), rows[k]["relaunch"], rows[k]["copy"])
        for k in over
    }
    assert got == {k: ("not_here", "doesn't apply in this mode", None, None) for k in over}, (
        f"R-10a: a not_here row shows no relaunch box and no Copy; got {got}"
    )
    # control: a relaunch row in the same state still shows its box (the probe can see one)
    assert rows["engine"]["relaunch"] == "--profile lean", rows["engine"]


@browser
async def test_r12_the_relaunch_box_shows_only_the_flag_to_add(tmp_path):
    """R-12 (synthetic state of a default live session): every relaunch box is labelled "Add to your session
    command:"; its ``<code>`` and its Copy payload are the flag(s) to add (Page overview exactly ``--page-overview``,
    the engine ``--profile lean``), never a whole ``diffusion_gearvis session`` command; the catalog's own whole
    command (``CatalogEntry.relaunch``, also on the wire) is not what the page shows."""
    raw, body = synthetic_state()
    async with settings_page(tmp_path, block=True) as (_live, page, _chrome):
        await post_synthetic(page, raw)
        probe = await open_panel(page, len(sc.CATALOG))
        clicked = [(k, await _copy(page, k)) for k in ("page_overview", "engine")]
        recorded = await copied_settled(page, 2)
    rows = {r["key"]: r for r in probe["rows"]}
    problems = []
    for key, flag in LIVE_RELAUNCH.items():
        row = rows[key]
        box = (row["relaunch_box"] or "").strip()
        label = VIEWPORT_LABEL if key == "browser_viewport" else RELAUNCH_LABEL
        if not box.startswith(label):
            problems.append(f"{key}: the box is not labelled {label!r}: {box!r}")
        if row["relaunch"] != flag:
            problems.append(f"{key}: the <code> reads {row['relaunch']!r}, not {flag!r}")
    assert problems == [], "R-12:\n  " + "\n  ".join(problems)
    assert rows["page_overview"]["relaunch"] == "--page-overview"
    assert rows["engine"]["relaunch"] == "--profile lean"
    assert clicked == [("page_overview", "clicked"), ("engine", "clicked")], clicked
    assert recorded["copied"] == ["--page-overview", "--profile lean"], recorded["copied"]
    assert body["items"], "premise: the state has items"


@browser
async def test_r12_adversarial_copy_in_a_no_web_search_off_session_gives_only_the_flag(tmp_path):
    """R-12 adversarial (synthetic state of a session started with ``--no-web --search off``): Copy on Page overview
    puts only ``--page-overview`` on the clipboard. The catalog entry on the same wire carries the whole command
    ``diffusion_gearvis session --tools live --page-overview``, which, pasted, would turn web and search back on; a
    page that shows or copies the catalog's command fails here. (Holds at BASE, whose page copies ``item.relaunch``;
    the real-server half is ``tests/unit/test_t0213_settings_wire.py::test_r12_adversarial_...``.)"""
    over = {
        "browser_web": {"value": False, "source": "flag"},
        "search": {"value": "off", "source": "flag"},
        "page_overview": {"relaunch": "--page-overview"},
    }
    raw, body = synthetic_state(item_over=over)
    entry = {e["key"]: e for e in body["catalog"]}["page_overview"]
    assert "diffusion_gearvis session" in str(
        entry.get("relaunch") or sc.by_key("page_overview").relaunch
    ), "premise: the catalog's whole command is the trap"
    async with settings_page(tmp_path, block=True) as (_live, page, _chrome):
        await post_synthetic(page, raw)
        probe = await open_panel(page, len(sc.CATALOG))
        clicked = await _copy(page, "page_overview")
        recorded = await copied_settled(page, 1)
    rows = {r["key"]: r for r in probe["rows"]}
    assert rows["page_overview"]["relaunch"] == "--page-overview", rows["page_overview"]
    assert clicked == "clicked", clicked
    assert recorded["copied"] == ["--page-overview"], (
        f"R-12: Copy gives only the flag to add; got {recorded['copied']!r}"
    )


@browser
async def test_a_later_state_without_a_catalog_updates_the_values(tmp_path):
    """A2r-1 / A2-apply hello row, in the wire order: the hello's state (with the catalog) and then a later state
    without one (rev 8; the A2-apply echo) whose values and sources differ. The panel shows the later values. Kills
    "ignore states without a catalog" (the panel would keep showing 560/profile and action outcome on)."""
    first, _ = synthetic_state()
    later_over = {
        "visual_tokens": {"value": 280, "source": "ui"},
        "action_outcome": {"value": False, "source": "ui"},
        "live_guidance": {"value": "v1", "source": "ui"},
    }
    later, later_body = synthetic_state(
        item_over=later_over, rev=SYNTHETIC_REV + 1, with_catalog=False
    )
    assert "catalog" not in json.loads(later), "premise: the later state carries no catalog"
    async with settings_page(tmp_path, block=True) as (_live, page, _chrome):
        await post_synthetic(page, first)
        before = await open_panel(page, len(sc.CATALOG))
        await post_synthetic(page, later)
        await page.until(
            "(() => { const r = document.querySelector('[data-setting-key=\"visual_tokens\"] [data-setting-source]');"
            " return Boolean(r) && r.getAttribute('data-setting-source') === 'ui'; })()",
            "the later state (no catalog) reached the panel: visual_tokens source ui",
            timeout_s=PAGE_WAIT_S,
        )
        after = await page.evaluate(PANEL_JS)
    b = {r["key"]: r for r in before["rows"]}
    a = {r["key"]: r for r in after["rows"]}
    assert (
        "560" in (b["visual_tokens"]["value"] or "") and b["visual_tokens"]["source"] == "profile"
    )
    assert "280" in (a["visual_tokens"]["value"] or "") and "560" not in (
        a["visual_tokens"]["value"] or ""
    )
    for key, want in later_over.items():
        assert a[key]["source"] == want["source"], (key, a[key])
    assert a["action_outcome"]["value"] != b["action_outcome"]["value"], (
        b["action_outcome"],
        a["action_outcome"],
    )
    assert len(after["rows"]) == len(sc.CATALOG), (
        "the catalog of the hello is kept for the later state"
    )
    assert [(g["name"], g["keys"]) for g in after["groups"]] == expected_groups()
    assert later_body["rev"] == SYNTHETIC_REV + 1


@browser
async def test_the_assist_panel_and_the_theme_select_behave_as_before(tmp_path):
    """B-4 negative (holds at BASE): with a settings state delivered, the Settings popover still holds the
    "Context helpers" panel with its four rows; ticking page_text still sends exactly one ``astra.assist.toggle``;
    the theme select stays in the header, outside the popover."""
    raw, _body = synthetic_state()
    async with settings_page(tmp_path, block=True) as (_live, page, _chrome):
        await post_synthetic(page, raw)
        opened = await page.evaluate(OPEN_JS)
        await settle(
            page,
            "(() => { const pop = document.querySelector('[data-settings-popover]');"
            " const a = pop && !pop.hidden && pop.querySelector('#assist-panel');"
            " return Boolean(a) && Boolean(a.querySelector('h2'))"
            " && a.querySelectorAll('[data-assist-kind]').length >= 4"
            " && Boolean(document.querySelector('[data-theme-control]')); })()",
        )
        state = await page.evaluate(
            "(() => { const pop = document.querySelector('[data-settings-popover]');"
            " const a = pop && pop.querySelector('#assist-panel');"
            " const t = document.querySelector('[data-theme-control]');"
            " return {assist: Boolean(a), heading: a && a.querySelector('h2') ? a.querySelector('h2').innerText.trim() : null,"
            " kinds: a ? [...a.querySelectorAll('[data-assist-kind]')].map((r) => r.getAttribute('data-assist-kind')) : [],"
            " theme_in_header: Boolean(t && t.closest('header')), theme_in_popover: Boolean(t && pop && pop.contains(t)),"
            " theme_options: t ? [...t.options].map((o) => o.value) : []}; })()"
        )
        before = await rec(page)
        clicked = await page.evaluate(
            "(() => { const b = document.querySelector('[data-assist-kind=\"page_text\"] input[type=checkbox]');"
            " if (!b) return 'no box'; if (b.disabled) return 'disabled'; b.click(); return 'clicked'; })()"
        )
        deadline = time.monotonic() + 5
        after = before
        while time.monotonic() < deadline:
            after = await rec(page)
            if "astra.assist.toggle" in after["sent"]:
                break
            await asyncio.sleep(0.1)
    assert opened == "open", opened
    assert state["assist"] and state["heading"] == "Context helpers", f"T-0162 B-6/B-9: {state}"
    assert state["kinds"] == ["page_text", "pdf_text", "cells", "at_spi"], state["kinds"]
    assert state["theme_in_header"] and not state["theme_in_popover"], (
        f"B-4: the theme select stays in the header: {state}"
    )
    assert state["theme_options"] == ["system", "light", "dark"], state["theme_options"]
    assert clicked == "clicked", clicked
    toggles = after["sent"].count("astra.assist.toggle") - before["sent"].count(
        "astra.assist.toggle"
    )
    assert toggles == 1, (
        f"T-0025 criterion 3: one tick, one astra.assist.toggle; the page sent {after['sent']}"
    )


@browser
async def test_rig_a_posted_line_reaches_the_pages_decoder(tmp_path):
    """Rig self-check (holds at BASE): a line posted with :func:`post_js` goes through ``ws.js`` and the page's
    decoder like a server line. An ``astra.assist.state`` posted this way sets ``[data-assist-badge]`` verbatim, so
    the synthetic rows' "the page never showed the panel" is about the panel, not about a dead injection path."""
    badge = "pixels+t0213-rig"
    raw = encode(protocol.AstraAssistState(flags=AssistFlags(page_text=True), badge=badge))
    async with settings_page(tmp_path, block=True) as (_live, page, _chrome):
        await post_synthetic(page, raw)
        await page.until(
            f"(document.querySelector('[data-assist-badge]') || {{}}).textContent === {json.dumps(badge)}",
            "rig: the posted astra.assist.state reached the page",
            timeout_s=PAGE_WAIT_S,
        )


@browser
async def test_the_page_requests_nothing_off_loopback(tmp_path):
    """B-4 negative (holds at BASE): over a page load, a settings state (with markup in one evidence), the panel
    opened and every Copy clicked, every request the page makes is to loopback."""
    raw, _body = synthetic_state(evidence={"reground": EVIL_EVIDENCE})
    async with settings_page(tmp_path, block=True) as (_live, page, chrome):
        await post_synthetic(page, raw)
        await page.evaluate(OPEN_JS)
        await page.until(
            "(() => { const p = document.querySelector('[data-settings-popover] [data-settings-panel]');"
            f" return Boolean(p) && p.querySelectorAll('[data-setting-key]').length === {len(sc.CATALOG)}; }})()",
            "rig: the Settings panel shows every row before its Copy buttons are clicked",
            timeout_s=PAGE_WAIT_S,
        )
        await page.evaluate(
            "(() => { for (const b of document.querySelectorAll('[data-setting-copy]')) if (!b.disabled) b.click(); })()"
        )
        # nothing-happens window (no off-loopback request), started once the panel was shown and the clicks returned
        await asyncio.sleep(QUIET_S)
        await asyncio.to_thread(chrome.drain, 1.0)
        requests = list(chrome.requests)
    assert requests, (
        "the CDP request log is empty, so this row would be vacuous (Network.enable before navigation)"
    )
    offenders = [
        u
        for u in requests
        if u.split("://", 1)[0] in {"http", "https", "ws", "wss"}
        and not any(u.split("://", 1)[1].startswith(h) for h in ("127.0.0.1", "localhost", "[::1]"))
    ]
    assert not offenders, f"B-4: zero non-loopback requests; got {offenders}"
    assert not [
        u for u in requests if u.split("://", 1)[0] not in {"http", "ws", "data", "blob"}
    ], requests


# --- A2-apply: changes from the panel (T-0213 A2a-1..A2a-3 page half, "Tests to write" A2-apply rows) -------------
#
# Rig: :class:`LiveTools`, a live session over ``FakeLiveBrowser`` (loop on, ``visual_tokens=1120``, the T-0134c
# screen-now label on, so ``action_outcome`` passes Profile's cross-rule), real ``HudServer(port=0)``, the page over
# loopback, the server's own states (``block=False``). Every turn is spoken through ``Live.ask``; the fake LLM's pass
# 1 clicks (``VALID_CLICK``) and its step answers, so a step request exists for the outcome clause to ride on. The
# frames are real JPEGs (a white screen; after each click a black badge at the top right), so T-0136's detector
# measures the click and the clause is placed when the setting is on.
#
# DOM readings these rows add (the task names the behaviour, not the hooks; stated in the report):
#
# - a changeable row's value control is its ``[data-setting-control]``: a checkbox for a two-valued bool row, a
#   ``<select>`` whose option values are the catalog values as text for a choice row; a toggle or a change sends one
#   ``astra.settings.set`` at once (no Apply button outside the instructions box);
# - a pending change shows the words "applies after this turn" in its row, and the panel one enabled ``<button>``
#   reading "Stop and apply";
# - a refusal's reason is shown inside its row, in ``[data-setting-refusal]``.

STATE_TYPE = "astra.settings.state"
SET_TYPE = "astra.settings.set"
#: T-0136 B-3's clause stem, typed from that criterion (never imported from session.py)
CLAUSE_STEM = "Compared with the previous screenshot"
#: the words a pending change shows (Tests to write, A2-apply), verbatim
PENDING_TEXT = "applies after this turn"
#: the button beside it (Tests to write, A2-apply), verbatim
STOP_AND_APPLY = "Stop and apply"
#: the tool loop's step texts (``tests/unit/test_session_tool_loop``): S1 announces, S3 ends the turn
S1, S3 = tsl.S1, tsl.S3
QUESTION_A = "please open the settings menu"
QUESTION_B = "and open it once more"
#: the badge a click leaves at the top right (far from VALID_CLICK's resolved point (640, 200)): T-0136 "elsewhere"
BADGE_BLOCK = (1100, 20, 1220, 80, BLACK)


class LiveTools(Live):
    """``Live`` in live mode: a real ``Session`` with a live kit over ``FakeLiveBrowser``, the loop on, the HUD tap of
    ``Live`` (``wire``, ``hello_replies``), the fake LLM a ``ToolLlm`` (pass 1 clicks, each step answers ``S3``)."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        ms_per_word: int = 60,
        session_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.clock = SessionClock()
        self.machine = RecordingMachine(tsl.LOOP)
        self.llm = ToolLlm([answer(S1), tool(VALID_CLICK)])
        self.llm.clock = self.clock  # type: ignore[assignment]
        self.llm.script_steps(*([answer(S3)] for _ in range(8)))
        self.stt = FakeStt()
        self.tts = RecordingTts(ms_per_word=ms_per_word)
        self.audio = RecordingAudio(self.clock)  # type: ignore[arg-type]
        self.frames = BrowserFrameSource(now_ms=lambda: self.clock.now_ms)
        self.sink = CountingSink(self.frames)
        self.memory = FakeMemory()
        self.browser = FakeLiveBrowser(
            [scene_frame(f"t0213c-white-{k}") for k in range(40)],
            post_frames=[scene_frame(f"t0213c-badge-{k}", blocks=[BADGE_BLOCK]) for k in range(20)],
            now_ms=lambda: int(self.clock.now_ms),
        )
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
            tool_kit=live_kit(self.browser),
            visual_tokens=1120,
            live_feedback=live_feedback(screen_label=True),
            **(session_kwargs or {}),
        )
        self.server = HudServer(self.session, frame_sink=self.sink, port=0)
        self.wire = []
        self.hello_replies = []
        self.hellos = 0
        self._tap()

    def step_bodies(self) -> list[str]:
        """Every step request's body as the real client would send it, JSON, in order."""
        return [json.dumps(b) for b in self.llm.step_bodies]


def live_rig(tmp_path: Path) -> Live:
    return LiveTools(tmp_path)


def live_rig_overview_frame(tmp_path: Path) -> Live:
    """:func:`live_rig` with the T-0151 overview frame on (``Session(overview_frame=True)``), so R-4's premise holds:
    the overview frame is on in the state the page holds when ReGround is turned on."""
    return LiveTools(tmp_path, session_kwargs={"overview_frame": True})


def steps_of(live: Live) -> list[str]:
    assert isinstance(live, LiveTools), "rig: a LiveTools row"
    return live.step_bodies()


def init_js_full(*, block: bool = False) -> str:
    """:data:`INIT_JS` that also keeps every message the page sends, parsed, in ``window.__t0213.msgs``."""
    out = init_js(block=block)
    for old, new in (
        ("sent: [], ", "sent: [], msgs: [], "),
        (
            "rec.sent.push(m && m.type); }",
            "rec.sent.push(m && m.type); rec.msgs.push(m); }",
        ),
    ):
        assert old in out, f"rig: INIT_JS no longer holds {old!r}"
        out = out.replace(old, new)
    return out


MSGS_JS = "(() => { const r = window.__t0213; return r && r.msgs ? r.msgs.slice() : null; })()"


async def sent_msgs(page: Page, type_: str = SET_TYPE) -> list[dict[str, Any]]:
    got = await page.evaluate(MSGS_JS)
    assert isinstance(got, list), f"rig: the full recorder is missing: {got!r}"
    return [m for m in got if isinstance(m, dict) and m.get("type") == type_]


def wire_states(live: Live) -> list[Any]:
    return [m for m in live.wire if getattr(m, "type", None) == STATE_TYPE]


def wire_item(state: Any, key: str) -> Any:
    got = [i for i in state.items if i.key == key]
    assert len(got) == 1, f"one item for {key}; got {[i.key for i in state.items]}"
    return got[0]


async def hello_state(live: Live) -> Any:
    """The hello reply's ``astra.settings.state``. T-0218 B (N1): ``page.connected()`` can return before the tap has
    recorded the hello reply, so wait on it (bounded by :data:`PAGE_WAIT_S`) before the unchanged assertion."""

    def sent() -> list[Any]:
        return [m for r in live.hello_replies for m in r if getattr(m, "type", None) == STATE_TYPE]

    with contextlib.suppress(AssertionError):
        await live.until(
            lambda: bool(sent()), "the hello reply was recorded", timeout_s=PAGE_WAIT_S
        )
    got = sent()
    assert got, "A2r-3: the hello reply carries an astra.settings.state"
    return got[-1]


async def wait_live(live: Live, predicate: typing.Callable[[], bool], what: str) -> None:
    await live.until(predicate, what, timeout_s=PAGE_WAIT_S)


def control_js(key: str, value: Any = None) -> str:
    """Operate ``key``'s first enabled ``[data-setting-control]``: a checkbox is clicked; a ``<select>`` is set to
    ``value`` (or, with no value, to the next option) with ``input`` and ``change`` events. Returns what it did."""
    return """(() => {
  const row = document.querySelector('[data-settings-panel] [data-setting-key=' + JSON.stringify(__KEY__) + ']');
  if (!row) return 'no row';
  const c = [...row.querySelectorAll('[data-setting-control]')].find((e) => !e.disabled);
  if (!c) return 'no enabled control';
  if (c.tagName === 'INPUT' && c.type === 'checkbox') { c.click(); return 'toggled ' + c.checked; }
  if (c.tagName === 'SELECT') {
    const want = __VALUE__;
    const opts = [...c.options];
    const i = want === null ? (c.selectedIndex + 1) % opts.length : opts.findIndex((o) => o.value === want);
    if (i < 0) return 'no option ' + want + ' in ' + opts.map((o) => o.value).join(',');
    c.selectedIndex = i;
    c.dispatchEvent(new Event('input', {bubbles: true}));
    c.dispatchEvent(new Event('change', {bubbles: true}));
    return 'selected ' + c.value;
  }
  return 'unknown control ' + c.tagName;
})()""".replace("__KEY__", json.dumps(key)).replace(
        "__VALUE__", json.dumps(None if value is None else str(value))
    )


ROW_JS = """(() => {
  const row = document.querySelector('[data-settings-panel] [data-setting-key=' + JSON.stringify(__KEY__) + ']');
  if (!row) return null;
  const src = row.querySelector('[data-setting-source]');
  const ref = row.querySelector('[data-setting-refusal]');
  const stop = [...document.querySelectorAll('[data-settings-panel] button')]
    .filter((b) => b.textContent.trim() === __STOP__);
  return {
    text: row.textContent,
    value: (row.querySelector('[data-setting-value]') || {}).textContent || null,
    source: src ? src.getAttribute('data-setting-source') : null,
    refusal: ref ? ref.textContent : null,
    stop_buttons: stop.map((b) => ({disabled: Boolean(b.disabled), visible: b.checkVisibility()})),
  };
})()"""


async def row_state(page: Page, key: str) -> dict[str, Any]:
    got = await page.evaluate(
        ROW_JS.replace("__KEY__", json.dumps(key)).replace("__STOP__", json.dumps(STOP_AND_APPLY))
    )
    assert isinstance(got, dict), f"the [data-setting-key={key!r}] row is not in the panel: {got!r}"
    return got


async def turn(live: Live, question: str, n_lines: int) -> None:
    """One spoken question, to IDLE with its turn line written."""
    await live.ask(question)
    await live.until(
        lambda: live.session.state.state is TurnState.IDLE and len(live.turn_lines()) >= n_lines,
        f"premise: the question {question!r} was answered and its turn line written",
    )


async def send_raw_set(page: Page, body: dict[str, Any]) -> None:
    sent = await page.evaluate(
        "(() => { const s = window.__t0213.sockets.at(-1); s.send(JSON.stringify("
        + json.dumps(body)
        + ")); return 'sent'; })()"
    )
    assert sent == "sent", sent


@browser
async def test_rig_a_set_on_the_pages_socket_turns_the_outcome_clause_on(tmp_path):
    """Rig self-check (holds at 96a4d27b, the server half being merged): in :class:`LiveTools`, turn 1 (outcome off)
    sends a step request without T-0136's clause; one ``astra.settings.set {rev: 0, changes: {action_outcome: true}}``
    written straight onto the page's socket is answered with rev 1, ``action_outcome`` true, source ``ui``; turn 2's
    step request carries the clause. So the page rows below fail on the page, not on the rig."""
    async with settings_page(tmp_path, block=False, make_live=live_rig, script=init_js_full()) as (
        live,
        page,
        _chrome,
    ):
        await turn(live, QUESTION_A, 1)
        steps_1 = steps_of(live)
        assert steps_1, "premise: turn 1 sent a step request (the click executed)"
        assert not any(CLAUSE_STEM in b for b in steps_1), "premise: outcome off, no clause"
        await send_raw_set(page, {"type": SET_TYPE, "rev": 0, "changes": {"action_outcome": True}})
        await wait_live(
            live,
            lambda: any(s.rev == 1 for s in wire_states(live)),
            "rig: the set was applied (a state with rev 1)",
        )
        applied = [s for s in wire_states(live) if s.rev == 1][-1]
        ao = wire_item(applied, "action_outcome")
        assert (ao.value, ao.source) == (True, "ui"), ao
        await turn(live, QUESTION_B, 2)
        steps_2 = steps_of(live)[len(steps_1) :]
    assert steps_2 and any(CLAUSE_STEM in b for b in steps_2), (
        "rig: with action_outcome on, turn 2's step request carries the clause"
    )


@browser
async def test_a2_apply_changeable_rows_have_enabled_controls(tmp_path):
    """A2-apply (page): in a live session's real hello state, every item the session says it can change
    (``changeable: true``, kind ``next_turn`` or ``next_turn_cold``) has at least one ``[data-setting-control]`` and
    every one of them is enabled, without B-read's :data:`DISABLED_TITLE`; ``user_instructions``' control is its
    enabled textarea. Every ``relaunch``, ``not_here`` or ``fixed`` item, and every item with ``changeable: false``,
    stays read-only: no enabled input besides Copy."""
    async with settings_page(tmp_path, block=False, make_live=live_rig, script=init_js_full()) as (
        live,
        page,
        _chrome,
    ):
        state = await hello_state(live)
        probe = await open_panel(page, len(state.items))
        textarea = await page.evaluate(
            "(() => { const t = document.querySelector('[data-setting-key=\"user_instructions\"] textarea');"
            " return t ? {disabled: t.disabled, readonly: t.readOnly} : null; })()"
        )
    rows = {r["key"]: r for r in probe["rows"]}
    changeable = [
        i.key
        for i in state.items
        if i.changeable
        and i.kind in ("next_turn", "next_turn_cold")
        and i.key != "user_instructions"
    ]
    assert {"action_outcome", "visual_tokens", "live_feedback", "completion_check"} <= set(
        changeable
    ), f"premise: a live session can change the v1 rows; changeable {changeable}"
    problems = []
    for key in changeable:
        controls = rows[key]["controls"]
        if not controls:
            problems.append(f"{key}: a changeable row has no [data-setting-control]")
        for c in controls:
            if c["disabled"]:
                problems.append(f"{key}: a {c['tag']} control is disabled on a changeable row")
            if c["title"] == DISABLED_TITLE:
                problems.append(f"{key}: the control still says {DISABLED_TITLE!r}")
    for i in state.items:
        if (i.kind in ("relaunch", "not_here", "fixed") or not i.changeable) and rows[i.key][
            "enabled_inputs"
        ]:
            problems.append(
                f"{i.key} ({i.kind}, changeable {i.changeable}): enabled inputs {rows[i.key]['enabled_inputs']}"
            )
    assert problems == [], "A2-apply:\n  " + "\n  ".join(problems)
    assert textarea == {"disabled": False, "readonly": False}, (
        f"C-1: the instructions textarea is enabled in a session that can change them: {textarea}"
    )


@browser
async def test_a2_apply_toggling_action_outcome_sends_one_set_echo_ui_next_turn_has_the_clause(
    tmp_path,
):
    """A2-apply (page; Tests to write): after one turn (outcome off, its step request has no clause), toggling Action
    outcome sends exactly one ``astra.settings.set`` with the rev in force (0), ``changes == {action_outcome: true}``
    and no ``stop_turn: true``; the server's echo (rev 1) reaches the row: source badge ``ui``, the value changed; the
    next turn's step request carries T-0136's clause."""
    async with settings_page(tmp_path, block=False, make_live=live_rig, script=init_js_full()) as (
        live,
        page,
        _chrome,
    ):
        await turn(live, QUESTION_A, 1)
        steps_1 = steps_of(live)
        assert steps_1 and not any(CLAUSE_STEM in b for b in steps_1), "premise: turn 1, no clause"
        rev = wire_states(live)[-1].rev
        assert rev == 0, f"premise: nothing changed yet; rev {rev}"
        probe = await open_panel(page, len(sc.CATALOG))
        before = {r["key"]: r for r in probe["rows"]}["action_outcome"]
        did = await page.evaluate(control_js("action_outcome"))
        assert isinstance(did, str) and did.startswith("toggled"), (
            f"A2-apply: Action outcome's control toggles: {did!r}"
        )
        await page.until(
            "(() => { const s = document.querySelector('[data-setting-key=\"action_outcome\"] [data-setting-source]');"
            " return Boolean(s) && s.getAttribute('data-setting-source') === 'ui'; })()",
            "A2-apply: the echo shows source=ui on the Action outcome row",
            timeout_s=PAGE_WAIT_S,
        )
        await asyncio.sleep(0.5)
        sets = await sent_msgs(page)
        after = await row_state(page, "action_outcome")
        applied = wire_states(live)[-1]
        await turn(live, QUESTION_B, 2)
        steps_2 = steps_of(live)[len(steps_1) :]
    assert len(sets) == 1, f"A2-apply: one toggle, one astra.settings.set; the page sent {sets}"
    assert sets[0]["rev"] == 0 and sets[0]["changes"] == {"action_outcome": True}, sets[0]
    assert sets[0].get("stop_turn", False) is False, sets[0]
    assert applied.rev == 1 and wire_item(applied, "action_outcome").value is True, applied
    assert after["source"] == "ui", after
    assert after["value"] != before["value"], (before["value"], after["value"])
    assert steps_2 and any(CLAUSE_STEM in b for b in steps_2), (
        "A2-apply: the next turn's step request carries the outcome clause"
    )


@browser
async def test_r26a_choosing_visual_tokens_560_sends_the_int(tmp_path):
    """R-26 (a): a select row whose catalog values are ints sends the typed value. In the live rig (1120 in force),
    choosing 560 in Visual tokens' ``<select>`` sends exactly one ``astra.settings.set`` with ``rev == 0`` and
    ``changes == {"visual_tokens": 560}`` where 560 is a JSON number (``int``, not the option's text ``"560"``); the
    session applies it (rev 1, value 560)."""
    async with settings_page(tmp_path, block=False, make_live=live_rig, script=init_js_full()) as (
        live,
        page,
        _chrome,
    ):
        assert wire_item(await hello_state(live), "visual_tokens").value == 1120, (
            "premise: 1120 in force"
        )
        await open_panel(page, len(sc.CATALOG))
        did = await page.evaluate(control_js("visual_tokens", 560))
        assert did == "selected 560", f"Visual tokens' select offers 560: {did!r}"
        await page.until(
            "(() => { const r = window.__t0213; return Boolean(r) && r.msgs.some((m) => m && m.type === "
            + json.dumps(SET_TYPE)
            + "); })()",
            "the page sent an astra.settings.set for Visual tokens",
            timeout_s=PAGE_WAIT_S,
        )
        await asyncio.sleep(0.5)
        sets = await sent_msgs(page)
        # the page's message first: a string "560" is refused by the session, so the apply wait below would also
        # go red, but for the session's reason, not this row's
        assert len(sets) == 1, f"one change, one astra.settings.set; the page sent {sets}"
        assert sets[0]["rev"] == 0, sets[0]
        value = sets[0]["changes"].get("visual_tokens")
        assert type(value) is int, (
            f"R-26 (a): the int value is sent as a JSON number, not {type(value).__name__} {value!r}"
        )
        assert sets[0]["changes"] == {"visual_tokens": 560}, sets[0]
        await wait_live(
            live,
            lambda: any(s.rev == 1 for s in wire_states(live)),
            "the session applied the panel's visual_tokens change (rev 1)",
        )
        applied = [s for s in wire_states(live) if s.rev == 1][-1]
    assert wire_item(applied, "visual_tokens").value == 560, applied


@browser
async def test_r26b_turning_reground_on_beside_the_overview_frame_also_sends_overview_frame_false(
    tmp_path,
):
    """R-26 (b), R-4: in a live session with the overview frame on and ReGround off, turning ReGround on from the
    panel sends exactly one ``astra.settings.set`` whose changes carry both ``reground: true`` and
    ``overview_frame: false`` (Profile refuses the two together), with ``rev == 0``. Only the page's message is
    checked; whether the session can run a pass 2 here is the A2-apply rows' business."""
    async with settings_page(
        tmp_path, block=False, make_live=live_rig_overview_frame, script=init_js_full()
    ) as (
        live,
        page,
        _chrome,
    ):
        state = await hello_state(live)
        rg, of = wire_item(state, "reground"), wire_item(state, "overview_frame")
        assert of.value is True and rg.value is False, (
            f"premise: overview frame on, ReGround off; {of}, {rg}"
        )
        assert rg.changeable and rg.kind in ("next_turn", "next_turn_cold"), (
            f"premise: ReGround changeable; {rg}"
        )
        await open_panel(page, len(sc.CATALOG))
        did = await page.evaluate(control_js("reground"))
        assert did == "toggled true", f"ReGround's control toggles on: {did!r}"
        await page.until(
            "(() => { const r = window.__t0213; return Boolean(r) && r.msgs.some((m) => m && m.type === "
            + json.dumps(SET_TYPE)
            + "); })()",
            "the page sent an astra.settings.set for ReGround",
            timeout_s=PAGE_WAIT_S,
        )
        await asyncio.sleep(0.5)
        sets = await sent_msgs(page)
    assert len(sets) == 1, f"one toggle, one astra.settings.set; the page sent {sets}"
    assert sets[0]["rev"] == 0, sets[0]
    assert sets[0]["changes"] == {"reground": True, "overview_frame": False}, (
        f"R-4: ReGround on beside the overview frame sends overview_frame false in the same set; {sets[0]}"
    )


@browser
async def test_a2_apply_a_change_mid_turn_applies_after_it_and_stop_and_apply_sends_stop_turn(
    tmp_path,
):
    """A2-apply (page; A2a-3): with a turn in flight (the fake LLM held mid-generate), toggling Action outcome sends
    one set; the server answers with the change pending (``pending: true``) and the row shows "applies after this
    turn" with one enabled "Stop and apply" button; at IDLE before the toggle neither shows. The button sends one
    more ``astra.settings.set`` with ``stop_turn: true`` and the rev in force (0); the session stops the turn (the
    LLM is still held, so only the stop can end it) and applies (rev 1, value true); the pending words and the
    button go."""
    async with settings_page(tmp_path, block=False, make_live=live_rig, script=init_js_full()) as (
        live,
        page,
        _chrome,
    ):
        await open_panel(page, len(sc.CATALOG))
        idle_row = await row_state(page, "action_outcome")
        live.llm.hold()
        asking = live.asking(QUESTION_A)
        try:
            await live.until(
                lambda: live.session.state.active_generate is not None,
                "premise: the turn's generate is in flight",
            )
            did = await page.evaluate(control_js("action_outcome"))
            assert isinstance(did, str) and did.startswith("toggled"), did
            await wait_live(
                live,
                lambda: any(
                    wire_item(s, "action_outcome").pending is True for s in wire_states(live)
                ),
                "A2a-3: the mid-turn change is pending",
            )
            await page.until(
                "(document.querySelector('[data-setting-key=\"action_outcome\"]') || {textContent: ''})"
                f".textContent.toLowerCase().includes({json.dumps(PENDING_TEXT)})",
                f"the row shows {PENDING_TEXT!r}",
                timeout_s=PAGE_WAIT_S,
            )
            pending_row = await row_state(page, "action_outcome")
            assert live.session.state.state is not TurnState.IDLE, (
                "premise: the turn is still in flight"
            )
            clicked = await page.evaluate(
                "(() => { const b = [...document.querySelectorAll('[data-settings-panel] button')]"
                f".filter((e) => e.textContent.trim() === {json.dumps(STOP_AND_APPLY)} && !e.disabled);"
                " if (b.length !== 1) return b.length + ' buttons'; b[0].click(); return 'clicked'; })()"
            )
            assert clicked == "clicked", f"one enabled {STOP_AND_APPLY!r} button: {clicked}"
            await wait_live(
                live,
                lambda: any(
                    s.rev == 1 and wire_item(s, "action_outcome").value is True
                    for s in wire_states(live)
                ),
                "A2a-3: stop_turn stopped the turn and the change applied (rev 1), the LLM still held",
            )
            await page.until(
                "!(document.querySelector('[data-setting-key=\"action_outcome\"]') || {textContent: ''})"
                f".textContent.toLowerCase().includes({json.dumps(PENDING_TEXT)})",
                "the pending words go once the change applied",
                timeout_s=PAGE_WAIT_S,
            )
            done_row = await row_state(page, "action_outcome")
            sets = await sent_msgs(page)
            # R-23: the turn really stopped (the LLM is still held, so nothing else could end it)
            stopped_idle = live.session.state.state is TurnState.IDLE
            stops = live.machine.events(E.UserStop)
        finally:
            live.llm.release()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asking, timeout=10)
    assert PENDING_TEXT not in idle_row["text"].lower() and idle_row["stop_buttons"] == [], idle_row
    assert pending_row["stop_buttons"] == [{"disabled": False, "visible": True}], pending_row
    assert len(sets) == 2, f"one set for the toggle, one for Stop and apply; got {sets}"
    assert sets[0]["changes"] == {"action_outcome": True}, sets[0]
    assert not sets[0].get("stop_turn", False), sets[0]
    assert sets[1].get("stop_turn") is True and sets[1]["rev"] == 0, (
        f"Stop and apply sends stop_turn=true with the rev in force: {sets[1]}"
    )
    assert all(not b["visible"] for b in done_row["stop_buttons"]), done_row
    assert stopped_idle, "R-23: Stop and apply stopped the turn in flight (IDLE with the LLM held)"
    assert len(stops) == 1, f"R-23: exactly one UserStop was submitted; got {stops}"


@browser
async def test_a2_apply_a_server_refusal_is_shown_on_its_row(tmp_path):
    """A2-apply (page): with Action outcome on (toggled from the panel, applied), choosing Screen-now label ``off``
    is refused by the session with config's own reason (Profile's cross-rule, A1-3); the page shows the server's
    reason inside the refused key's row (``[data-setting-refusal]``), and the value it shows stays the one in force
    (``label``). The expected pair is the server's ``refused`` on the wire, checked against the independent oracle
    ``live_settings.validate`` and ``config.action_outcome_without_feedback_refusal()``."""
    async with settings_page(tmp_path, block=False, make_live=live_rig, script=init_js_full()) as (
        live,
        page,
        _chrome,
    ):
        await open_panel(page, len(sc.CATALOG))
        did = await page.evaluate(control_js("action_outcome"))
        assert isinstance(did, str) and did.startswith("toggled"), did
        await wait_live(
            live,
            lambda: any(s.rev == 1 for s in wire_states(live)),
            "premise: Action outcome applied from the panel (rev 1)",
        )
        # T-0218 B: the server's rev 1 is on the wire before the page has read it; a change made before then goes
        # out with rev 0 and is refused as stale ("*"), not by the cross-rule this row is about. Wait on the echo
        # reaching the page (A2-apply: the row's source badge reads ui).
        await page.until(
            "(() => { const s = document.querySelector('[data-setting-key=\"action_outcome\"] [data-setting-source]');"
            " return Boolean(s) && s.getAttribute('data-setting-source') === 'ui'; })()",
            "premise: the rev 1 echo reached the page (Action outcome source ui)",
            timeout_s=PAGE_WAIT_S,
        )
        snapshot = live.session.settings
        did = await page.evaluate(control_js("live_feedback", "off"))
        assert did == "selected off", f"Screen-now label's select offers 'off': {did!r}"
        await wait_live(
            live,
            lambda: bool(wire_states(live)[-1].refused),
            "the session refused live_feedback off beside action_outcome on",
        )
        refused = [(r.key, r.reason) for r in wire_states(live)[-1].refused]
        await settle(
            page,
            "(() => "
            + json.dumps([[k, r] for k, r in refused])
            + ".every(([k, r]) => { const e = document.querySelector("
            "'[data-settings-panel] [data-setting-key=' + JSON.stringify(k) + '] [data-setting-refusal]');"
            " return Boolean(e) && e.textContent.includes(r); }))()",
        )
        rows = {k: await row_state(page, k) for k in sorted({k for k, _ in refused})}
        lf = await row_state(page, "live_feedback")
    new, oracle = live_settings.validate(snapshot, {"live_feedback": "off"}, mode="live", web=False)
    assert oracle and new is None, "premise: the oracle refuses the change"
    assert any(action_outcome_without_feedback_refusal() in reason for _, reason in refused), (
        refused
    )
    problems = []
    for key, reason in refused:
        shown = rows[key]["refusal"]
        if shown is None or reason not in shown:
            problems.append(
                f"{key}: the row's [data-setting-refusal] reads {shown!r}, not the reason {reason!r}"
            )
    assert problems == [], "A2-apply refusal on its row:\n  " + "\n  ".join(problems)
    assert "label" in (lf["value"] or ""), f"the value in force stays label: {lf}"


# --- B-4: frozen page files ---------------------------------------------------------------------------------------

#: main when the B-read tests were written (82ee4503, T-0213's planning commit; A1's BASE too).
BASE = "82ee450339b8fbf25c3e85246346eb365b05b28a"
#: the B-read implementation commit (2eee3f82, "T-0213 B-read: read-only Settings panel in the HUD popover"), set at
#: the A2-read + B-read activation (the T-0211 B8 pattern).
TIP: str | None = "2eee3f825118658bb74a5ef34b7ee25dd4961edd"
#: B-4's range base: ``git merge-base main 2eee3f82`` = 78afde2a (checked at activation, main f79123ed; ``--all``
#: also lists 6c517b2c, the A1 combine). Main moved past BASE while B-read was open (the B branches merged main at
#: df4602ab and 8f329075), so ``RANGE_BASE..TIP`` is the range B-read's own commits live in. ``BASE..TIP`` is
#: checked as well; both must be empty for the frozen paths.
RANGE_BASE = "78afde2a689f870f74267c075ed894734f0d809a"
#: B-4's byte-identical list. ``hud/vendor`` is absent at BASE (deleted by T-0164 D) and must stay absent.
FROZEN = ("hud/vendor", "hud/src/ws.js", "hud/src/assists/gate.js")


def test_b4_frozen_page_files_are_byte_identical() -> None:
    """B-4: ``hud/vendor/**``, ``hud/src/ws.js`` and ``hud/src/assists/gate.js`` are byte-identical over
    ``RANGE_BASE..TIP`` and ``BASE..TIP`` (literals; ``TIP`` is B-read's implementation commit, ``RANGE_BASE`` its
    merge-base with main)."""
    assert TIP is not None, (
        "T-0213 B-4: the TIP literal (B-read's implementation commit) does not exist yet"
    )
    gr.require_commits(BASE, RANGE_BASE, TIP)
    assert gr.is_ancestor(BASE, RANGE_BASE), f"{BASE[:8]} is not an ancestor of {RANGE_BASE[:8]}"
    assert gr.is_ancestor(RANGE_BASE, TIP), f"{RANGE_BASE[:8]} is not an ancestor of {TIP[:8]}"
    assert not gr.is_ancestor(TIP, RANGE_BASE), "premise: the range is not empty"
    for lo in (RANGE_BASE, BASE):
        changed = gr.changed(lo, TIP, *FROZEN)
        assert changed == [], (
            f"T-0213 B-4: frozen page files differ in {lo[:8]}..{TIP[:8]}: {changed}"
        )
