"""The A2UI renderer in headless Chrome, against the stub Conductor (T-0024, load-bearing).

This is the end-to-end half of ``docs/tasks/T-0024.md``: the quiz fixture, taken through the
**real** ``UiGate``, really renders; three canvases really morph one surface; a real click really
puts one ``astra.ui.action`` on the socket, decoded by the **production codec**; and over the
whole session the browser makes no non-loopback request and constructs no ``AudioContext``.

Driving: Chrome DevTools Protocol, polling in real time (T-0008 ruling R3:
``--virtual-time-budget`` runs ~15x real time and truncates a socket-fed page).
``tests/integration/test_hud_headless.py`` is T-0008's file and is not touched; ``Chrome``,
``serve_hud`` and ``requires_chrome`` are imported from it, never copied. ``HudStub.push`` is new
in this pass (``tests/fakes/hud_server.py``) because criterion 7 is about what happens *between*
canvases and a scripted burst cannot observe that.

**The DOM contract this test pins** (the task fixes the behaviour; these are the hooks the
assertions need, and the implementer provides them):

==================================  ============================================================
``[data-thread]``                   the one conversation thread (T-0154 B1); every card lives in
                                    it, in the card slot of ``article[data-turn-id]`` of the turn
                                    its ``surface_id`` names (``turn-<turn_id>``, B4), mounted
                                    through ``host.cardSlot(surfaceId)`` (tests/hud/thread.test.mjs)
``[data-ui-surface]``               one card wrapper per surface id, value == the ``surface_id``
``[data-ui-canvas-idx]``            on the same wrapper: the last ``canvas_idx`` applied to it
``<a2ui-surface>``                  the renderer's own element — exactly one per surface id,
                                    morphed in place canvas after canvas
``inert``                           on every card wrapper but the newest surface's (B8, H-T154-4)
==================================  ============================================================

**Amended by T-0154** (docs/tasks/T-0154.md G2). ``#shown-channel`` and the ``no surface yet
(T-0010)`` placeholder are retired (B1): the pins on them are re-keyed to the thread and the card
slot. T-0024 criterion 7's "one surface element, not n" is amended **by name** to "one
``<a2ui-surface>`` per surface id" (B4), so the session now pushes three surface ids (``turn-t1``,
the dangling ``turn-t9`` and a ``turn-t10`` with a button) and the per-id count is asserted. B8 adds
the adversarial phase: a click on the *older* card's button — both a synthetic ``.click()`` and a
real CDP mouse click — sends no ``astra.ui.action``, while a CDP click on the newest card's button
sends exactly one (the guard is pinned in both directions).

Every DOM assertion pierces open shadow roots (``deepText``/``deepTags`` below); the A2UI Lit
renderer puts almost everything inside one.

The fixture runs one page session in phases and records snapshots; each test asserts on one
phase, because a browser launch per criterion is waste. Setup never raises: failures land on
``Run.error`` and are asserted in the call phase, so a pending run fails cleanly instead of
erroring at collection.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.ui_gate import UiGate
from local_astra.contracts.codec import decode
from local_astra.contracts.protocol import AstraUiAction, AstraUiSurface
from tests.fakes.hud_server import HudStub
from tests.integration.test_hud_headless import CHROME, HUD, Chrome, requires_chrome, serve_hud

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
UI_INDEX = HUD / "src" / "ui" / "index.js"
FIXTURE = REPO / "tests" / "fixtures" / "a2ui" / "quiz_example.jsonl"
#: T-0008 criterion 10's placeholder, retired by T-0154 B1: it must no longer be on the page.
PLACEHOLDER = "no surface yet (T-0010)"

#: Read off ``tests/fixtures/a2ui/quiz_example.jsonl`` by hand: the two tab titles, the five
#: question prompts (without their ``**n.**`` markdown so the assertion does not depend on
#: whether the basic catalog's Text renders markdown), and question 1's four choice labels.
TAB_TITLES = ("Quiz", "Flashcards")
PROMPTS = (
    "What is 1/2 + 1/4?",
    "Which fraction is equivalent to 0.75?",
    "Simplify 6/8.",
    "2/3 × 3/5 = ?",
    "Which is larger: 5/6 or 7/9?",
)
Q1_CHOICES = ("2/6", "3/4", "1/8", "2/4")
CHECK_LABEL = "Check answers"

#: T-0164 group B (docs/tasks/T-0164.md B1/B2): the card element is re-keyed by name from the
#: vendored ``<a2ui-surface>`` to our renderer's light-DOM ``[data-a2ui-root]`` once T-0164's
#: **Status:** records "group B: tests <a>..<b>, activated <sha>" (``ASTRA_RUN_PENDING=1`` flips it
#: now). Before that every pre-existing row is byte-for-byte today's, so the module stays green on
#: main. The new B rows carry ``pending(task="T-0164")`` and key on ``[data-a2ui-root]`` directly.
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
#: The selector of "the rendered card element" the pre-existing rows count and mark.
ROOT_SEL = "[data-a2ui-root]" if T0164_B else "a2ui-surface"
#: T-0164 B3: the vendored renderer's own catalog id (the gate never emits it) and an arbitrary one.
FOREIGN_CATALOG_IDS = (
    "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json",
    "urn:astra-test:foreign-catalog",
)
FOREIGN_SENTINEL = "FOREIGN-CATALOG-MUST-NOT-RENDER"

#: The gate's rewritten id for ``turn_id="t1"`` (``surface_id_prefix="turn"``, T-0010 A7). The
#: model wrote ``lesson_1``; sending that back would address a surface the Conductor never had.
GATE_SURFACE_ID = "turn-t1"
MODEL_SURFACE_ID = "lesson_1"


def gated_canvases() -> list[AstraUiSurface]:
    """The fixture through the real ``UiGate``, canvas by canvas (regime: 3 canvases).

    Split: prefix-cumulative at message boundaries — canvas 0 is the ``createSurface``, canvas 1
    adds the 38-component ``updateComponents``, canvas 2 adds the ``updateDataModel``. That is
    what a growing canvas looks like on the wire, and it reproduces criterion 7's landmine: the
    gate re-sends ``createSurface`` on every canvas (T-0010 A4) and the real
    ``MessageProcessor.processCreateSurfaceMessage`` throws ``Surface … already exists`` on the
    second one. (The task's Regime says "say so if you split differently" — this is the split.)
    """
    lines = FIXTURE.read_text().splitlines(keepends=True)
    boundaries = [1, 41, len(lines)]
    gate = UiGate()
    out: list[AstraUiSurface] = []
    for idx, end in enumerate(boundaries):
        result = gate.gate(
            "".join(lines[:end]), turn_id="t1", canvas_idx=idx, final=(idx == len(boundaries) - 1)
        )
        assert result.surface is not None, (
            "the acceptance fixture must pass the real gate — otherwise this test could pass "
            f"against a surface the gate would have rejected. Canvas {idx}: {result.report}"
        )
        out.append(result.surface)
    return out


def smuggled_a2ui() -> str:
    """A complete, well-formed A2UI body — the payload criterion 4 says may never be rendered."""
    return json.dumps(
        [
            {
                "version": "v0.9.1",
                "createSurface": {
                    "surfaceId": "smuggled",
                    "catalogId": (
                        "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
                    ),
                },
            },
            {
                "version": "v0.9.1",
                "updateComponents": {
                    "surfaceId": "smuggled",
                    "components": [
                        {"id": "root", "component": "Text", "text": SMUGGLED_SENTINEL},
                    ],
                },
            },
        ]
    )


SMUGGLED_SENTINEL = "SMUGGLED-SURFACE-MUST-NEVER-RENDER"
GHOST_CHILD_ID = "q_ghost"
SIBLING_TEXTS = ("first sibling stays", "third sibling stays")


def dangling_surface() -> AstraUiSurface:
    """Criterion 8: hand-built (the gate would have pruned the reference), one child missing."""
    return AstraUiSurface(
        surface_id="turn-t9",
        canvas_idx=0,
        final=True,
        messages=(
            {
                "version": "v0.9.1",
                "createSurface": {
                    "surfaceId": "turn-t9",
                    "catalogId": (
                        "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
                    ),
                },
            },
            {
                "version": "v0.9.1",
                "updateComponents": {
                    "surfaceId": "turn-t9",
                    "components": [
                        {
                            "id": "root",
                            "component": "Column",
                            "children": ["s1", GHOST_CHILD_ID, "s3"],
                        },
                        {"id": "s1", "component": "Text", "text": SIBLING_TEXTS[0]},
                        {"id": "s3", "component": "Text", "text": SIBLING_TEXTS[1]},
                    ],
                },
            },
        ),
    )


#: B8's "too wide" half: the newest surface, with a button of its own. Written here by hand in the
#: quiz fixture's own Button shape (``check_btn``) and taken through the real ``UiGate`` for turn
#: ``t10``, so the id the click must echo is the gate's ``turn-t10``.
NEWEST_TURN = "t10"
NEWEST_SURFACE_ID = "turn-t10"
NEWEST_LABEL = "Confirm newest card"
NEWEST_ACTION = "newest_ok"
OLDER_SURFACE_IDS = (GATE_SURFACE_ID, "turn-t9")


def newest_surface() -> AstraUiSurface:
    catalog = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
    body = "\n".join(
        json.dumps(m)
        for m in (
            {"version": "v0.9.1", "createSurface": {"surfaceId": "note", "catalogId": catalog}},
            {
                "version": "v0.9.1",
                "updateComponents": {
                    "surfaceId": "note",
                    "components": [
                        {"id": "root", "component": "Column", "children": ["msg", "ok_btn"]},
                        {"id": "msg", "component": "Text", "text": "The newest card speaks."},
                        {
                            "id": "ok_btn",
                            "component": "Button",
                            "variant": "primary",
                            "child": "ok_lbl",
                            "action": {"event": {"name": NEWEST_ACTION, "context": {"k": "v"}}},
                        },
                        {"id": "ok_lbl", "component": "Text", "text": NEWEST_LABEL},
                    ],
                },
            },
        )
    )
    result = UiGate().gate(body + "\n", turn_id=NEWEST_TURN, canvas_idx=0, final=True)
    assert result.surface is not None, (
        f"the B8 newest-card body must pass the real gate; it reported {result.report}"
    )
    assert result.surface.surface_id == NEWEST_SURFACE_ID, result.surface.surface_id
    return result.surface


# --- the JS this test runs in the page ----------------------------------------------------------

DEEP = """
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
const deepUnder = (host) => {
  const out = [];
  const walk = (root) => {
    if (!root || !root.querySelectorAll) return;
    for (const el of root.querySelectorAll('*')) {
      out.push(el);
      if (el.shadowRoot) walk(el.shadowRoot);
    }
  };
  walk(host);
  return out;
};
const leafText = (els) =>
  els
    .filter((el) => el.children.length === 0)
    .map((el) => (el.textContent || '').trim())
    .filter((t) => t.length > 0);
"""

PROBE_JS = (
    """
(() => {
"""
    + DEEP
    + """
  const all = deepAll();
  // T-0154 B1/B4: the shown channel is the set of card wrappers, wherever the thread put them
  // (#shown-channel is retired). Scoped to the wrappers so the spoken line that displays the
  // smuggled JSON as text (honest) is not mistaken for a rendered surface (not honest).
  const wrappers = all.filter((el) => el.hasAttribute && el.hasAttribute('data-ui-surface'));
  const under = wrappers.flatMap((w) => deepUnder(w));
  const tags = {};
  for (const el of all) {
    const t = el.tagName.toLowerCase();
    tags[t] = (tags[t] || 0) + 1;
  }
  const hooks = wrappers.map((el) => {
    const mine = deepUnder(el);
    const article = el.closest('article[data-turn-id]');
    return {
      surface: el.getAttribute('data-ui-surface'),
      canvas_idx: el.getAttribute('data-ui-canvas-idx'),
      inert: !!(el.inert || el.closest('[inert]')),
      in_thread: !!el.closest('[data-thread]'),
      article: article ? article.getAttribute('data-turn-id') : null,
      surfaces_inside: mine.filter((e) => e.matches(__ROOT_SEL__)).length,
      // T-0164 B1: the card root is the wrapper's direct (and only) element child, in light DOM
      direct_root: el.children.length === 1 && el.children[0].matches('[data-a2ui-root]'),
      text: leafText(mine).join('\\n'),
    };
  });
  const thread = document.querySelector('[data-thread]');
  return {
    elements: all.length,
    roots: all.filter((e) => e.matches(__ROOT_SEL__)).length,
    shadow_hosts: all.filter((e) => !!e.shadowRoot).length,
    attach_shadow_calls: globalThis.__astra_shadow_calls,
    foreign_leaves: all.filter(
      (el) => el.children.length === 0 && (el.textContent || '').trim() === __FOREIGN__,
    ).length,
    shown_elements: under.length,
    tags,
    hooks,
    articles: thread
      ? [...thread.querySelectorAll('article[data-turn-id]')].map((a) => a.getAttribute('data-turn-id'))
      : null,
    // an element whose *whole* text is the smuggled sentinel is a rendered Text component; the
    // spoken line shows the sentinel only inside the larger JSON string it arrived in
    sentinel_leaves: all.filter(
      (el) => el.children.length === 0 && (el.textContent || '').trim() === __SENTINEL__,
    ).length,
    placeholder_on_page: (document.body.innerText || '').includes(__PLACEHOLDER__),
    text: leafText(all).join('\\n'),
    shown_text: leafText(under).join('\\n'),
    audio_hits: globalThis.__astra_audio_hits || [],
    errors: [...document.querySelectorAll('.error-line')].map((e) => e.textContent),
  };
})()
""".replace("__SENTINEL__", json.dumps("SMUGGLED-SURFACE-MUST-NEVER-RENDER"))
    .replace("__PLACEHOLDER__", json.dumps(PLACEHOLDER))
    .replace("__ROOT_SEL__", json.dumps(ROOT_SEL))
    .replace("__FOREIGN__", json.dumps(FOREIGN_SENTINEL))
)


#: B4 "canvas k morphs canvas k-1 in that same element": a test-side expando on the first
#: ``<a2ui-surface>`` inside a wrapper. A re-created element loses it; a morphed one keeps it.
def mark_js(surface_id: str, mark: str) -> str:
    return (
        "(() => {"
        + DEEP
        + """
  const wrap = document.querySelector('[data-ui-surface="' + __ID__ + '"]');
  if (!wrap) return false;
  const host = deepUnder(wrap).find((e) => e.matches(__ROOT_SEL__));
  if (!host) return false;
  host.__astraMark = __MARK__;
  return true;
})()""".replace("__ID__", json.dumps(surface_id))
        .replace("__MARK__", json.dumps(mark))
        .replace("__ROOT_SEL__", json.dumps(ROOT_SEL))
    )


def marked_js(surface_id: str) -> str:
    """The mark on the ``<a2ui-surface>`` now inside ``surface_id``'s wrapper, or null."""
    return (
        "(() => {"
        + DEEP
        + """
  const wrap = document.querySelector('[data-ui-surface="' + __ID__ + '"]');
  if (!wrap) return 'no wrapper';
  const host = deepUnder(wrap).find((e) => e.matches(__ROOT_SEL__));
  if (!host) return 'no ' + __ROOT_SEL__;
  return host.__astraMark === undefined ? null : host.__astraMark;
})()""".replace("__ID__", json.dumps(surface_id)).replace("__ROOT_SEL__", json.dumps(ROOT_SEL))
    )


def _button_in_js(surface_id: str, label: str) -> str:
    """JS that finds the button reading ``label`` inside ``surface_id``'s card wrapper as
    ``target`` (or returns a reason string)."""
    return (
        DEEP
        + """
  const wrap = document.querySelector('[data-ui-surface="' + __ID__ + '"]');
  if (!wrap) return 'no card wrapper for ' + __ID__;
  const want = __WANT__;
  const leaves = deepUnder(wrap).filter(
    (el) => el.children.length === 0 && (el.textContent || '').trim() === want,
  );
  if (!leaves.length) return 'no element in ' + __ID__ + ' reads ' + want;
  let node = leaves[0];
  let target = null;
  for (let i = 0; i < 8 && node; i += 1) {
    const tag = node.tagName ? node.tagName.toLowerCase() : '';
    const role = node.getAttribute ? node.getAttribute('role') : null;
    if (tag === 'button' || role === 'button') { target = node; break; }
    node = node.parentElement || (node.parentNode && node.parentNode.host) || null;
  }
  if (!target) return 'no button around ' + want + ' in ' + __ID__;
""".replace("__ID__", json.dumps(surface_id)).replace("__WANT__", json.dumps(label))
    )


def click_in_js(surface_id: str, label: str) -> str:
    """A synthetic ``.click()`` on the button reading ``label`` inside one card wrapper."""
    return (
        "(() => {"
        + _button_in_js(surface_id, label)
        + """
  target.click();
  return 'clicked ' + target.tagName.toLowerCase();
})()"""
    )


def button_point_js(surface_id: str, label: str) -> str:
    """Scroll the button into view and return its centre (viewport px) for a CDP mouse click,
    with the premises that make that click meaningful: the button has a box, its centre is in
    the viewport and inside its own card wrapper's box."""
    return (
        "(() => {"
        + _button_in_js(surface_id, label)
        + """
  target.scrollIntoView({block: 'center', inline: 'nearest'});
  const r = target.getBoundingClientRect();
  const w = wrap.getBoundingClientRect();
  const x = r.x + r.width / 2, y = r.y + r.height / 2;
  return {x, y, w: r.width, h: r.height,
          in_viewport: x >= 0 && y >= 0 && x <= innerWidth && y <= innerHeight,
          inside_wrapper: x >= w.left && x <= w.right && y >= w.top && y <= w.bottom};
})()"""
    )


DEEP_TEXT_JS = (
    """
(() => {
"""
    + DEEP
    + """
  return leafText(deepAll()).join('\\n');
})()
"""
)


def click_js(label: str) -> str:
    """Click the *component* whose label is ``label``, piercing shadow roots on the way up."""
    return (
        """
(() => {
"""
        + DEEP
        + """
  const want = __WANT__;
  const leaves = deepAll().filter(
    (el) => el.children.length === 0 && (el.textContent || '').trim() === want,
  );
  if (!leaves.length) return 'no element reads ' + want;
  let node = leaves[0];
  let target = null;
  for (let i = 0; i < 8 && node; i += 1) {
    const tag = node.tagName ? node.tagName.toLowerCase() : '';
    const role = node.getAttribute ? node.getAttribute('role') : null;
    if (tag === 'button' || tag === 'a' || role === 'button' || role === 'tab') {
      target = node;
      break;
    }
    node = node.parentElement || (node.parentNode && node.parentNode.host) || null;
  }
  target = target || leaves[0];
  target.click();
  return 'clicked ' + target.tagName.toLowerCase();
})()
""".replace("__WANT__", json.dumps(label))
    )


#: Installed before any page script runs, so a construction by the vendored bundle is recorded
#: even though nothing in our code constructs one (criterion 10, the behavioural half).
AUDIO_SENTINEL_JS = """
globalThis.__astra_audio_hits = [];
for (const key of ['AudioContext', 'webkitAudioContext', 'OfflineAudioContext']) {
  const original = globalThis[key];
  if (typeof original !== 'function') continue;
  const wrapper = function (...args) {
    globalThis.__astra_audio_hits.push(key);
    return new original(...args);
  };
  wrapper.prototype = original.prototype;
  try {
    Object.defineProperty(globalThis, key, { value: wrapper, configurable: true, writable: true });
  } catch (err) {
    globalThis.__astra_audio_hits.push('could not wrap ' + key);
  }
}
"""

#: T-0164 B1: every ``Element.prototype.attachShadow`` call, counted before any page script runs.
#: It delegates, so the vendored renderer still works with it installed (the count is all it adds).
SHADOW_SENTINEL_JS = """
globalThis.__astra_shadow_calls = 0;
(() => {
  const original = Element.prototype.attachShadow;
  Element.prototype.attachShadow = function (...args) {
    globalThis.__astra_shadow_calls += 1;
    return original.apply(this, args);
  };
})();
"""


def light_button_point_js(surface_id: str, label: str, within: str | None = None) -> str:
    """T-0164 B4: the ``<button>`` reading ``label`` inside ``surface_id``'s light-DOM card root
    (``[data-ui-surface] > [data-a2ui-root]``; no shadow piercing), optionally inside the component
    ``[data-a2-id=within]``, scrolled into view; its centre for a CDP click, or a reason string."""
    return (
        """(() => {
  const root = document.querySelector('[data-ui-surface="' + __ID__ + '"] > [data-a2ui-root]');
  if (!root) return 'no light-DOM [data-a2ui-root] for ' + __ID__;
  const scope = __WITHIN__ === null ? root : root.querySelector('[data-a2-id="' + __WITHIN__ + '"]');
  if (!scope) return 'no [data-a2-id=' + __WITHIN__ + '] in ' + __ID__;
  const hits = [...scope.querySelectorAll('button')].filter((b) => (b.textContent || '').trim() === __WANT__);
  if (hits.length !== 1) return hits.length + ' buttons read ' + __WANT__ + ' in ' + __ID__;
  const target = hits[0];
  target.scrollIntoView({block: 'center', inline: 'nearest'});
  const r = target.getBoundingClientRect();
  const x = r.x + r.width / 2, y = r.y + r.height / 2;
  return {x, y, w: r.width, h: r.height, in_viewport: x >= 0 && y >= 0 && x <= innerWidth && y <= innerHeight,
          hit: document.elementFromPoint(x, y) === target || target.contains(document.elementFromPoint(x, y))};
})()""".replace("__ID__", json.dumps(surface_id))
        .replace("__WANT__", json.dumps(label))
        .replace("__WITHIN__", json.dumps(within))
    )


def observe_root_js(surface_id: str) -> str:
    """T-0164 B2: a ``MutationObserver`` on ``surface_id``'s card root (everything: child lists,
    attributes, character data, subtree), recording into ``globalThis.__astra_mut``."""
    return """(() => {
  const root = document.querySelector('[data-ui-surface="' + __ID__ + '"] > [data-a2ui-root]');
  if (!root) return 'no light-DOM [data-a2ui-root] for ' + __ID__;
  globalThis.__astra_mut = [];
  globalThis.__astra_mut_root = root;
  const mo = new MutationObserver((records) => {
    for (const r of records) globalThis.__astra_mut.push(r.type + ':' + (r.attributeName || ''));
  });
  mo.observe(root, {subtree: true, childList: true, attributes: true, characterData: true});
  globalThis.__astra_mo = mo;
  return 'observing';
})()""".replace("__ID__", json.dumps(surface_id))


def observed_js(surface_id: str) -> str:
    """Flush the observer and read what it saw, plus whether the root is still the same node."""
    return """(() => {
  const mo = globalThis.__astra_mo;
  if (!mo) return {error: 'no observer'};
  for (const r of mo.takeRecords()) globalThis.__astra_mut.push(r.type + ':' + (r.attributeName || ''));
  const root = document.querySelector('[data-ui-surface="' + __ID__ + '"] > [data-a2ui-root]');
  const wrap = document.querySelector('[data-ui-surface="' + __ID__ + '"]');
  return {mutations: [...globalThis.__astra_mut], same_root: root === globalThis.__astra_mut_root,
          canvas_idx: wrap ? wrap.getAttribute('data-ui-canvas-idx') : null,
          text: root ? root.textContent : null};
})()""".replace("__ID__", json.dumps(surface_id))


def focus_field_js(surface_id: str, label: str) -> str:
    """Focus the input whose accessible name is ``label`` in ``surface_id``'s light-DOM card."""
    return """(() => {
  const root = document.querySelector('[data-ui-surface="' + __ID__ + '"] > [data-a2ui-root]');
  if (!root) return 'no light-DOM [data-a2ui-root] for ' + __ID__;
  const named = (el) => el.getAttribute('aria-label') === __WANT__ ||
    [...(el.labels || [])].some((l) => (l.textContent || '').trim() === __WANT__);
  const hits = [...root.querySelectorAll('input, textarea')].filter(named);
  if (hits.length !== 1) return hits.length + ' inputs are named ' + __WANT__;
  hits[0].focus();
  globalThis.__astra_field = hits[0];
  return document.activeElement === hits[0] ? 'focused' : 'focus did not land';
})()""".replace("__ID__", json.dumps(surface_id)).replace("__WANT__", json.dumps(label))


FIELD_STATE_JS = """(() => {
  const f = globalThis.__astra_field;
  if (!f) return {error: 'no field'};
  return {value: f.value, start: f.selectionStart, end: f.selectionEnd,
          active: document.activeElement === f, connected: f.isConnected};
})()"""

#: T-0164 B4 / A8: the gate-made card. The model wrote a Button with a bare ``text`` (the UiGate's
#: RB repair makes it ``__gate.clicked`` with ``context.label``) and a literal-valued TextField.
GATE_TURN = "t11"
GATE_CARD_ID = "turn-t11"
GATE_LABEL = "Show me more"
FIELD_LABEL = "Your name"
TYPED = "Ada Lovelace"
LATER_LINE = "A later canvas adds this line."


def gate_card(canvas_idx: int) -> AstraUiSurface:
    """The gate card through the real ``UiGate``; canvas 1 adds one Text after the field."""
    catalog = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"
    children = ["t", "name", "go"] + (["later"] if canvas_idx >= 1 else [])
    components: list[dict[str, Any]] = [
        {"id": "root", "component": "Column", "children": children},
        {"id": "t", "component": "Text", "text": "Pick one."},
        {"id": "name", "component": "TextField", "label": FIELD_LABEL, "value": ""},
        {"id": "go", "component": "Button", "text": GATE_LABEL},
    ]
    if canvas_idx >= 1:
        components.append({"id": "later", "component": "Text", "text": LATER_LINE})
    body = "\n".join(
        json.dumps(m)
        for m in (
            {"version": "v0.9.1", "createSurface": {"surfaceId": "c", "catalogId": catalog}},
            {"version": "v0.9.1", "updateComponents": {"surfaceId": "c", "components": components}},
        )
    )
    result = UiGate().gate(body + "\n", turn_id=GATE_TURN, canvas_idx=canvas_idx, final=True)
    assert result.surface is not None, f"the gate card must pass the real gate: {result.report}"
    assert result.surface.surface_id == GATE_CARD_ID, result.surface.surface_id
    return result.surface


def regated_t1(canvas_idx: int, *, changed: bool = False) -> AstraUiSurface:
    """T-0164 B2: the quiz's last canvas, gated again at a later ``canvas_idx`` (the bridge drops
    an index it has applied, so "re-sent" means the same messages at a new index, as the gate
    sends a turn's final canvas). ``changed`` edits one prompt: the observer's positive control."""
    text = FIXTURE.read_text()
    if changed:
        assert PROMPTS[2] in text, "premise: the third prompt is in the fixture"
        text = text.replace(PROMPTS[2], PROMPTS[2] + " (edited)")
    result = UiGate().gate(text, turn_id="t1", canvas_idx=canvas_idx, final=True)
    assert result.surface is not None, result.report
    return result.surface


def foreign_surface(index: int, catalog_id: str) -> AstraUiSurface:
    """T-0164 B3: hand-built (the gate would refuse it), a createSurface whose catalogId is not the
    gate's — the engine throws for it and nothing renders."""
    sid = f"turn-f{index}"
    return AstraUiSurface(
        surface_id=sid,
        canvas_idx=0,
        final=True,
        messages=(
            {"version": "v0.9.1", "createSurface": {"surfaceId": sid, "catalogId": catalog_id}},
            {
                "version": "v0.9.1",
                "updateComponents": {
                    "surfaceId": sid,
                    "components": [{"id": "root", "component": "Text", "text": FOREIGN_SENTINEL}],
                },
            },
        ),
    )


#: T-0164 B4: the two picks, read off quiz_example.jsonl by hand (q1_c's option "3/4" is value
#: "b"; q3_c's option "2/3" is value "a"), so the resolved answers are q1 ["b"] and q3 ["a"].
QUIZ_PICKS = (("q1_c", "3/4"), ("q3_c", "2/3"))
QUIZ_ANSWERS = {"q1": ["b"], "q2": [], "q3": ["a"], "q4": [], "q5": []}

#: T-0164 B5: copies of the served card.css in the *document* (adopted or linked), matched by the
#: selector list Chrome parses from the served file (the method of test_hud_card_parity.SHEET_JS),
#: plus every shadow root on the page (there must be none).
PAGE_SHEET_JS = r"""
(async () => {
  const resp = await fetch('/src/ui/card.css', {cache: 'no-store'});
  if (!resp.ok) return {ok: false, status: resp.status};
  const oracle = new CSSStyleSheet();
  oracle.replaceSync(await resp.text());
  const sels = (sheet) => {
    const out = [];
    const walk = (rules) => { for (const r of rules) { if (r.selectorText !== undefined) out.push(r.selectorText);
                                                      if (r.cssRules) walk(r.cssRules); } };
    try { walk(sheet.cssRules); } catch (e) { return null; }
    return out;
  };
  const want = JSON.stringify(sels(oracle));
  const sheets = [...(document.adoptedStyleSheets || []), ...document.styleSheets];
  return {
    ok: true, rules: JSON.parse(want).length,
    copies: sheets.filter((s) => JSON.stringify(sels(s)) === want).length,
    hrefs: [...document.styleSheets].map((s) => s.href).filter(Boolean),
    shadow_roots: [...document.querySelectorAll('*')].filter((e) => !!e.shadowRoot).length,
  };
})()
"""

LOOPBACK = ("127.0.0.1", "localhost", "[::1]")


class RecordingChrome(Chrome):
    """``Chrome`` plus every ``Network.requestWillBeSent`` URL seen over the whole session.

    ``Chrome._cmd`` drops protocol *events* while it waits for its reply; this subclass records
    them on the way past. Polling is frequent enough that the events are read continuously, and
    ``drain`` collects whatever is left before the browser is closed.
    """

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
class Snapshot:
    probe: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return str(self.probe.get("text") or "")

    @property
    def shown_text(self) -> str:
        return str(self.probe.get("shown_text") or "")

    @property
    def tags(self) -> dict[str, int]:
        value = self.probe.get("tags")
        return dict(value) if isinstance(value, dict) else {}

    @property
    def hooks(self) -> list[dict[str, Any]]:
        return list(self.probe.get("hooks") or [])

    @property
    def elements(self) -> int:
        return int(self.probe.get("elements") or 0)

    @property
    def roots(self) -> int:
        """Rendered card elements: ``<a2ui-surface>`` until T-0164 B, ``[data-a2ui-root]`` after."""
        return int(self.probe.get("roots") or 0)

    @property
    def shown_elements(self) -> int:
        """Elements under ``#shown-channel``, shadow roots included — the component count."""
        return int(self.probe.get("shown_elements") or 0)


@dataclass
class Run:
    error: str | None = None
    stub: HudStub | None = None
    canvases: list[AstraUiSurface] = field(default_factory=list)
    after_adversarial: Snapshot = field(default_factory=Snapshot)
    per_canvas: list[Snapshot] = field(default_factory=list)
    after_click: Snapshot = field(default_factory=Snapshot)
    click_result: str = ""
    tab_click_result: str = ""
    raw_before_click: list[str] = field(default_factory=list)
    raw_after_click: list[str] = field(default_factory=list)
    raw_final: list[str] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)
    dangling: Snapshot = field(default_factory=Snapshot)
    frames_after_all: int = 0
    #: B4 morph-in-place: the canvas at which the t1 <a2ui-surface> was marked (None: never
    #: found), and the mark read back after the last canvas and after the newer surfaces
    marked_at: int | None = None
    mark_after_canvases: Any = None
    mark_after_newer: Any = None
    #: B8: the older card clicked both ways while turn-t9 is the newest surface; first turn-t1
    #: is switched back from Flashcards to its Quiz tab (the only tab whose child is rendered)
    quiz_tab_back: str = ""
    older_synthetic: str = ""
    older_point: Any = None
    raw_before_older: list[str] = field(default_factory=list)
    raw_after_older: list[str] = field(default_factory=list)
    #: B8's other direction: the newest card (turn-t10) clicked by CDP mouse
    newest: Snapshot = field(default_factory=Snapshot)
    newest_point: Any = None
    raw_after_newest: list[str] = field(default_factory=list)
    #: T-0154 C1: hud/src/ui/card.css's copies in every <a2ui-surface> shadow root, read after
    #: phase F (turn-t1 morphed through three canvases, two newer surfaces created after it)
    card_sheet: dict[str, Any] = field(default_factory=dict)
    #: T-0164 B4: the two chip picks before phase C's click (CDP points, or reason strings)
    picks: list[Any] = field(default_factory=list)
    #: T-0164 B2: the MutationObserver on turn-t1's root over 3 identical re-sends, then a change
    observe_start: str = ""
    observed_identical: Any = None
    observed_changed: Any = None
    #: T-0164 B3: after the foreign-catalog surfaces
    foreign: Snapshot = field(default_factory=Snapshot)
    #: T-0164 A8-in-Chrome and B4: the gate card (turn-t11)
    field_focus: str = ""
    field_before: Any = None
    field_after: Any = None
    gate_point: Any = None
    raw_before_gate: list[str] = field(default_factory=list)
    raw_after_gate: list[str] = field(default_factory=list)
    #: T-0164 B1/B5: the page-level card sheet probe (light DOM)
    page_sheet: dict[str, Any] = field(default_factory=dict)
    end: Snapshot = field(default_factory=Snapshot)


def _probe(chrome: Chrome) -> Snapshot:
    value = chrome.evaluate(PROBE_JS)
    return Snapshot(probe=value if isinstance(value, dict) else {"__probe__": value})


def _mouse_click(chrome: Chrome, point: dict[str, Any]) -> None:
    """A real left click at ``point`` through ``Input.dispatchMouseEvent`` — the path a human's
    mouse takes, which hit-testing (and so ``inert``) applies to; a synthetic ``.click()`` skips
    hit-testing entirely, which is why B8 is asserted both ways."""
    x, y = float(point.get("x") or 0), float(point.get("y") or 0)
    chrome._cmd("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
    for kind in ("mousePressed", "mouseReleased"):
        chrome._cmd("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1)


def _wait_for_text(chrome: Chrome, needle: str, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if needle in str(chrome.evaluate(DEEP_TEXT_JS) or ""):
            return True
        time.sleep(0.15)
    return needle in str(chrome.evaluate(DEEP_TEXT_JS) or "")


@pytest.fixture(scope="module")
def run() -> Any:
    """One page session in five phases. Never raises: failures land on ``Run.error``."""
    if not UI_INDEX.is_file():
        yield Run(
            error=(
                "T-0024 adds hud/src/ui/index.js (plus bridge.js, action.js, a2ui.js and the "
                "vendored bundle) — the panel this test drives (docs/tasks/T-0024.md)"
            )
        )
        return
    if CHROME is None:  # pragma: no cover - requires_chrome is the real guard
        yield Run(error="no chrome")
        return
    result = Run()
    stub = HudStub([]).start()
    result.stub = stub
    server, port = serve_hud()
    try:
        result.canvases = gated_canvases()
        url = f"http://127.0.0.1:{port}/?source=canvas&ws=" + urllib.parse.quote(stub.url, safe="")
        with RecordingChrome(CHROME) as chrome:
            chrome._cmd("Network.enable")
            chrome._cmd("Page.addScriptToEvaluateOnNewDocument", source=AUDIO_SENTINEL_JS)
            chrome._cmd("Page.addScriptToEvaluateOnNewDocument", source=SHADOW_SENTINEL_JS)
            chrome.navigate(url)
            stub.wait_for(lambda s: "astra.hello" in s.received_types(), 20)

            # --- phase A (criterion 4, adversarial): three ways to smuggle a surface ---------
            body = smuggled_a2ui()
            stub.push(
                json.dumps(
                    {
                        "type": "astra.spoken",
                        "turn_id": "t0",
                        "utterance_id": "u0",
                        "text": body,
                    }
                )
            )
            stub.push(json.dumps({"type": "astra.zzz", "messages": json.loads(body)}))
            stub.push(
                json.dumps(
                    {
                        "type": "astra.ui.surface",
                        "surface_id": GATE_SURFACE_ID,
                        "canvas_idx": -1,  # rejected by decode: ge=0
                        "final": False,
                        "messages": json.loads(body),
                    }
                )
            )
            time.sleep(2.0)
            result.after_adversarial = _probe(chrome)

            # --- phase B (criteria 5, 7): the three real canvases, one at a time -------------
            for idx, surface in enumerate(result.canvases):
                stub.push(surface)
                time.sleep(0.8)
                result.per_canvas.append(_probe(chrome))
                # T-0154 B4: mark the element before the next canvas arrives
                if (
                    result.marked_at is None
                    and idx < len(result.canvases) - 1
                    and chrome.evaluate(mark_js(GATE_SURFACE_ID, f"canvas-{idx}")) is True
                ):
                    result.marked_at = idx
            _wait_for_text(chrome, PROMPTS[2], 20)
            result.per_canvas[-1] = _probe(chrome)
            result.mark_after_canvases = chrome.evaluate(marked_js(GATE_SURFACE_ID))

            # --- phase C (criterion 9): one click, then two clicks that raise no action ------
            # T-0164 B4: two chip picks by real mouse clicks in the light-DOM card first (before
            # T-0164 B's swap there is no [data-a2ui-root], so nothing is picked and the reason
            # string is recorded). A pick raises no action (A10), so phase C's counts stand.
            for within, label in QUIZ_PICKS:
                point = chrome.evaluate(light_button_point_js(GATE_SURFACE_ID, label, within))
                result.picks.append(point)
                if isinstance(point, dict):
                    _mouse_click(chrome, point)
                    time.sleep(0.3)
            result.raw_before_click = stub.snapshot()
            result.click_result = str(chrome.evaluate(click_js(CHECK_LABEL)))
            stub.wait_for(lambda s: "astra.ui.action" in s.received_types(), 15)
            time.sleep(1.0)
            result.raw_after_click = stub.snapshot()
            result.after_click = _probe(chrome)
            result.tab_click_result = str(chrome.evaluate(click_js(TAB_TITLES[1])))
            chrome.evaluate("globalThis.dispatchEvent(new Event('scroll')), 1")
            time.sleep(1.5)
            result.raw_final = stub.snapshot()
            # T-0154 B8 premise (G2): the vendored Tabs renders only the active tab's child, so
            # after the Flashcards switch "Check answers" is not in the DOM. Switch turn-t1 back
            # to its Quiz tab now, while it is still the newest surface (not inert), so phase E
            # clicks a button that exists. A tab switch raises no action; phase E's
            # ``len(before) == 1`` premise checks that this one sent nothing either.
            result.quiz_tab_back = str(chrome.evaluate(click_in_js(GATE_SURFACE_ID, TAB_TITLES[0])))
            _wait_for_text(chrome, CHECK_LABEL, 10)

            # --- phase D (criterion 8): a hand-built surface with a dangling child -----------
            frames_before = len(stub.frames())
            stub.push(dangling_surface())
            _wait_for_text(chrome, SIBLING_TEXTS[1], 15)
            result.dangling = _probe(chrome)
            stub.wait_for(lambda s: len(s.frames()) > frames_before, 10)
            result.frames_after_all = len(stub.frames()) - frames_before

            # --- phase E (T-0154 B8, adversarial): turn-t9 is now the newest surface; click the
            # older turn-t1 card's "Check answers" both ways. Nothing may reach the socket.
            result.raw_before_older = stub.snapshot()
            result.older_synthetic = str(chrome.evaluate(click_in_js(GATE_SURFACE_ID, CHECK_LABEL)))
            time.sleep(1.0)
            result.older_point = chrome.evaluate(button_point_js(GATE_SURFACE_ID, CHECK_LABEL))
            if isinstance(result.older_point, dict):
                _mouse_click(chrome, result.older_point)
            time.sleep(1.5)
            result.raw_after_older = stub.snapshot()

            # --- phase F (B8, the guard's other side): a newer surface with a button; a real
            # mouse click on it sends exactly one action, for that surface.
            stub.push(newest_surface())
            _wait_for_text(chrome, NEWEST_LABEL, 15)
            time.sleep(0.5)
            result.newest = _probe(chrome)
            result.mark_after_newer = chrome.evaluate(marked_js(GATE_SURFACE_ID))
            result.newest_point = chrome.evaluate(button_point_js(NEWEST_SURFACE_ID, NEWEST_LABEL))
            if isinstance(result.newest_point, dict):
                _mouse_click(chrome, result.newest_point)
            stub.wait_for(
                lambda s: len(ui_actions(s.snapshot())) > len(ui_actions(result.raw_after_older)),
                10,
            )
            time.sleep(1.0)
            result.raw_after_newest = stub.snapshot()

            # T-0154 C1: the card sheet in each surface's shadow root. The probe is the parity
            # module's (imported, not copied; imported here because that module imports this one).
            from tests.integration.test_hud_card_parity import SHEET_JS

            result.card_sheet = dict(chrome.evaluate(SHEET_JS) or {})

            # --- phase G (T-0164 B2, adversarial): a MutationObserver on turn-t1's root while
            # its last canvas is re-sent 3 times (same messages, canvas 4, 5, 6), then a canvas
            # that edits one prompt (the observer's positive control). Canvas 3 is applied
            # *before* the observer starts: phase C's two picks live in the data model, and the
            # fixture's own updateDataModel (answers all []) legitimately resets them on the first
            # re-send (a later updateDataModel wins for path-bound inputs, implementation note 4;
            # ruling 2026-10-27). From then on the page and the canvas agree, so an identical
            # canvas must write nothing.
            stub.push(regated_t1(3))
            time.sleep(0.6)
            result.observe_start = str(chrome.evaluate(observe_root_js(GATE_SURFACE_ID)))
            for idx in (4, 5, 6):
                stub.push(regated_t1(idx))
                time.sleep(0.6)
            result.observed_identical = chrome.evaluate(observed_js(GATE_SURFACE_ID))
            stub.push(regated_t1(7, changed=True))
            _wait_for_text(chrome, PROMPTS[2] + " (edited)", 10)
            result.observed_changed = chrome.evaluate(observed_js(GATE_SURFACE_ID))

            # --- phase H (T-0164 B3): createSurface with a foreign catalogId renders nothing
            for k, catalog_id in enumerate(FOREIGN_CATALOG_IDS, start=1):
                stub.push(foreign_surface(k, catalog_id))
            time.sleep(1.5)
            result.foreign = _probe(chrome)

            # --- phase I (T-0164 A8 in Chrome, B4): type into the gate card's field, deliver a
            # later canvas, then a CDP click on its gate-made button.
            stub.push(gate_card(0))
            _wait_for_text(chrome, GATE_LABEL, 15)
            result.field_focus = str(chrome.evaluate(focus_field_js(GATE_CARD_ID, FIELD_LABEL)))
            if result.field_focus == "focused":
                chrome._cmd("Input.insertText", text=TYPED)
            result.field_before = chrome.evaluate(FIELD_STATE_JS)
            stub.push(gate_card(1))
            _wait_for_text(chrome, LATER_LINE, 15)
            time.sleep(0.3)
            result.field_after = chrome.evaluate(FIELD_STATE_JS)
            result.raw_before_gate = stub.snapshot()
            result.gate_point = chrome.evaluate(light_button_point_js(GATE_CARD_ID, GATE_LABEL))
            if isinstance(result.gate_point, dict):
                _mouse_click(chrome, result.gate_point)
                stub.wait_for(
                    lambda s: (
                        len(ui_actions(s.snapshot())) > len(ui_actions(result.raw_before_gate))
                    ),
                    10,
                )
            time.sleep(1.0)
            result.raw_after_gate = stub.snapshot()
            result.page_sheet = dict(chrome.evaluate(PAGE_SHEET_JS) or {})
            result.end = _probe(chrome)

            chrome.drain(1.5)
            result.requests = list(chrome.requests)
    except Exception as exc:  # a broken harness is data, not an error inside a test
        result.error = f"the headless run failed: {type(exc).__name__}: {exc}"
    finally:
        server.shutdown()
        stub.stop()
    yield result


def ui_actions(raws: list[str]) -> list[AstraUiAction]:
    return [m for m in (decode(r) for r in raws) if isinstance(m, AstraUiAction)]


def guard(run: Run) -> None:
    assert run.error is None, run.error


# --- criterion 5: the quiz fixture renders --------------------------------------------------


@requires_chrome
def test_both_tab_titles_all_five_prompts_and_question_ones_four_choices_are_on_screen(run):
    """Criterion 5, through the pierced DOM (the Lit renderer hides everything in shadow roots)."""
    guard(run)
    text = run.per_canvas[-1].text if run.per_canvas else ""
    missing = [
        wanted for wanted in (*TAB_TITLES, *PROMPTS, *Q1_CHOICES, CHECK_LABEL) if wanted not in text
    ]
    assert not missing, (
        "criterion 5: with the three messages of tests/fixtures/a2ui/quiz_example.jsonl taken "
        "through the real UiGate, the page shows both tab titles, all five question prompts and "
        f"question 1's four choices. Missing from the pierced DOM: {missing}\n"
        f"--- what the page shows ---\n{text[:2000]}"
    )


@requires_chrome
def test_each_surface_renders_in_the_card_slot_of_the_turn_its_id_names(run):
    """Criterion 2/5, **amended by T-0154 B1/B4** (G2): the panel used to mount into T-0008's
    ``#shown-channel``; that region is retired, and each surface now renders through
    ``host.cardSlot(surfaceId)`` into the card slot of ``article[data-turn-id]`` of the turn its
    ``surface_id`` names (``turn-t1`` -> ``t1``), inside ``[data-thread]``.

    Kept from the original row: the quiz's first prompt is inside the shown channel (now: inside
    the ``turn-t1`` card wrapper), not somewhere else on the page.
    """
    guard(run)
    last = run.per_canvas[-1] if run.per_canvas else Snapshot()
    t1 = [h for h in last.hooks if h.get("surface") == GATE_SURFACE_ID]
    assert len(t1) == 1, (
        f"B4: one card wrapper [data-ui-surface={GATE_SURFACE_ID!r}]; the page has {last.hooks!r}"
    )
    assert PROMPTS[0] in str(t1[0].get("text") or ""), (
        "criterion 2 (amended): the quiz renders inside its own card wrapper; the wrapper holds "
        f"{str(t1[0].get('text') or '')[:600]!r}"
    )
    assert t1[0].get("in_thread") and t1[0].get("article") == "t1", (
        "T-0154 B4: a surface renders in the card slot of the turn its surface_id names "
        f"(turn-t1 -> article[data-turn-id='t1'] inside [data-thread]); the wrapper is "
        f"in_thread={t1[0].get('in_thread')!r}, in the article of {t1[0].get('article')!r}"
    )
    # every surface of the session, each in its own turn's article (t9 and t10 have no other
    # content: the card slot is what opens their articles, B2/B4)
    final = run.newest.hooks
    placed = {h.get("surface"): (h.get("in_thread"), h.get("article")) for h in final}
    assert placed == {
        GATE_SURFACE_ID: (True, "t1"),
        "turn-t9": (True, "t9"),
        NEWEST_SURFACE_ID: (True, NEWEST_TURN),
    }, f"B4: each surface in the card slot of its own turn's article; got {placed!r}"
    assert run.newest.probe.get("articles") == ["t0", "t1", "t9", NEWEST_TURN], (
        "B2: one article per turn with content, in first-seen order (t0's smuggling utterance, "
        f"then the three surfaces); the thread holds {run.newest.probe.get('articles')!r}"
    )
    main = (HUD / "src" / "main.js").read_text(encoding="utf-8")
    code = "\n".join(line.split("//", 1)[0] for line in main.splitlines())
    assert "cardSlot" in code, (
        "T-0154 B4/B11 (the host API pinned in tests/hud/thread.test.mjs): hud/src/main.js hands "
        "the ui panel the thread's cardSlot (host.cardSlot); no code line of main.js names it "
        "(comments are stripped before the search)"
    )


# --- criterion 7: progressive morph ---------------------------------------------------------


@requires_chrome
def test_three_canvases_morph_one_surface_and_never_remove_what_was_shown(run):
    """Criterion 7: one surface element for the whole turn, component count non-decreasing.

    The idempotent ``createSurface`` is the landmine: the gate re-sends it on every canvas of the
    turn (T-0010 A4) and ``processCreateSurfaceMessage`` throws ``Surface … already exists``; a
    bridge that lets that throw escape loses the rest of the canvas (nothing renders) and one
    that re-creates the surface blanks what was shown.
    """
    guard(run)
    assert len(run.per_canvas) == 3, f"three canvases were pushed, got {len(run.per_canvas)}"
    creates = sum(
        1 for surface in run.canvases for message in surface.messages if "createSurface" in message
    )
    assert creates >= 2, (
        "this test is only meaningful if the gate really re-sends createSurface; it emitted "
        f"{creates} of them over the three canvases"
    )
    last = run.per_canvas[-1]
    assert last.roots == 1, (
        "criterion 7: the pane morphs in place — one surface element, not n; the pierced DOM has "
        f"{last.roots} <a2ui-surface> elements"
    )
    assert [h["surface"] for h in last.hooks] == [GATE_SURFACE_ID], (
        f"criterion 7: exactly one mounted surface, and it is the gate's id; got {last.hooks!r}"
    )
    counts = [snap.shown_elements for snap in run.per_canvas]
    assert counts == sorted(counts), (
        "criterion 7 negative: an out-of-order or duplicate canvas never removes components "
        f"already shown — the element count per canvas went {counts}"
    )
    assert counts[-1] > counts[0], (
        f"criterion 7: canvas 1 adds 38 components to canvas 0's empty surface; counts {counts}"
    )
    assert str(last.hooks[0].get("canvas_idx")) == "2", (
        "criterion 7: the panel reports the last canvas it applied, so a stalled render is "
        f"visible; got {last.hooks!r}"
    )


@requires_chrome
def test_one_a2ui_surface_per_surface_id_each_morphed_in_place(run):
    """T-0154 B4, **amending T-0024 criterion 7 by name** ("one surface element, not n" ->
    "one per surface id"; G2), and B6's node identity in a real browser.

    After three surface ids (``turn-t1``, the dangling ``turn-t9``, ``turn-t10``) the page holds
    exactly three ``<a2ui-surface>`` elements, one inside each id's wrapper; the ``turn-t1``
    element marked before its last canvas is the same node after that canvas (canvas k morphs
    canvas k-1 in that same element) and after two newer surfaces were created (a newer card
    never rebuilds an older one).
    """
    guard(run)
    snap = run.newest
    ids = [h.get("surface") for h in snap.hooks]
    assert sorted(ids) == sorted([GATE_SURFACE_ID, "turn-t9", NEWEST_SURFACE_ID]), (
        f"B4: one card wrapper per surface id, three ids pushed; the wrappers are {ids!r}"
    )
    assert snap.roots == 3, (
        "B4 (T-0024 criterion 7 amended): one <a2ui-surface> per surface id — three ids, three "
        f"elements; the pierced DOM has {snap.roots}"
    )
    per_wrapper = {h.get("surface"): h.get("surfaces_inside") for h in snap.hooks}
    assert set(per_wrapper.values()) == {1}, (
        f"B4: exactly one <a2ui-surface> inside each card wrapper; got {per_wrapper!r}"
    )
    assert run.marked_at is not None, (
        "premise: an <a2ui-surface> inside the turn-t1 wrapper existed before the last canvas, so "
        "it could be marked; none was found after canvases 0 and 1"
    )
    want = f"canvas-{run.marked_at}"
    assert run.mark_after_canvases == want, (
        f"B4: canvas 2 morphs canvas {run.marked_at} in the same element; the element inside "
        f"turn-t1's wrapper after canvas 2 carries mark {run.mark_after_canvases!r}, not {want!r} "
        "(a re-created element loses the test's mark)"
    )
    assert run.mark_after_newer == want, (
        "B6: two newer surfaces later, turn-t1's <a2ui-surface> is still the same node; it reads "
        f"{run.mark_after_newer!r} (the older card was rebuilt or moved out of its wrapper)"
    )
    canvas = {h.get("surface"): str(h.get("canvas_idx")) for h in snap.hooks}
    assert canvas.get(GATE_SURFACE_ID) == "2", (
        "B4/B5: [data-ui-canvas-idx] is per surface: turn-t1 keeps its last canvas (2) when "
        f"turn-t9 and turn-t10 arrive at canvas 0; got {canvas!r}"
    )


@requires_chrome
def test_the_card_sheet_is_added_exactly_once_to_each_surfaces_shadow_root(run):
    """T-0154 C1 under B's one-surface-per-id layout: ``hud/src/ui/card.css`` is in the shadow root
    of every ``<a2ui-surface>`` exactly once — after ``turn-t1`` morphed through three canvases and
    two newer surfaces were created after it, so a sheet re-added per canvas, or re-added to every
    root whenever a new surface mounts, reads as more than one copy. It is *added*: the root still
    holds at least one other sheet (the catalog's own), so a sheet that replaced the catalog's
    styles fails too. Copies are matched by the selector list Chrome parsed from the served file
    (``SHEET_JS``, adopted or linked)."""
    guard(run)
    if T0164_B:
        # T-0164 B5 amends T-0154 C1 by name: "adopted into each surface's shadow root" becomes
        # "one page sheet; no shadow root exists". Read at the end of the session, after turn-t1
        # morphed through six more canvases and three newer surfaces mounted.
        page = run.page_sheet
        assert page.get("ok"), f"B5: /src/ui/card.css is not served by the page's origin: {page!r}"
        assert int(page.get("rules") or 0) > 0, (
            f"B5: the served card.css parses to no rule: {page!r}"
        )
        assert page.get("copies") == 1, (
            "B5: card.css is loaded once per page (a <link> or one adopted sheet), never per "
            f"surface or per canvas; the document holds {page.get('copies')!r} copies: {page!r}"
        )
        assert page.get("shadow_roots") == 0, f"B5: no shadow root exists on the page: {page!r}"
        style_css = [
            u for u in run.requests if urllib.parse.urlsplit(u).path == "/src/ui/style.css"
        ]
        assert style_css == [], (
            f"B5: hud/src/ui/style.css is deleted and never requested: {style_css}"
        )
        return
    sheet = run.card_sheet
    assert sheet.get("ok"), f"C1: /src/ui/card.css is not served by the page's origin: {sheet!r}"
    assert int(sheet.get("rules") or 0) > 0, f"C1: the served card.css parses to no rule: {sheet!r}"
    surfaces = list(sheet.get("surfaces") or [])
    ids = sorted(str(s.get("id")) for s in surfaces)
    assert ids == sorted([GATE_SURFACE_ID, "turn-t9", NEWEST_SURFACE_ID]), (
        f"premise (B4): one <a2ui-surface> per pushed surface id; the page has {ids!r}"
    )
    bad = [s for s in surfaces if s.get("copies") != 1]
    assert bad == [], f"C1: surfaces without exactly one copy of card.css: {bad!r}"
    replaced = [s for s in surfaces if int(s.get("sheets") or 0) < 2]
    assert replaced == [], (
        "C1: the card sheet is added to the catalog's sheets, not swapped in for them; these "
        f"shadow roots hold card.css and nothing else: {replaced!r}"
    )


# --- T-0154 B8: only the newest surface speaks -------------------------------------------------


@requires_chrome
def test_older_cards_are_inert_and_only_the_newest_surface_carries_no_inert(run):
    """B8 / H-T154-4: earlier card wrappers carry ``inert``; the newest does not."""
    guard(run)
    inert = {h.get("surface"): h.get("inert") for h in run.newest.hooks}
    assert inert == {GATE_SURFACE_ID: True, "turn-t9": True, NEWEST_SURFACE_ID: False}, (
        "B8 (H-T154-4): every card wrapper but the most recently created surface's is inert "
        f"(the attribute on the wrapper or an ancestor inside the article); got {inert!r}"
    )
    inert_then = {h.get("surface"): h.get("inert") for h in run.dangling.hooks}
    assert inert_then == {GATE_SURFACE_ID: True, "turn-t9": False}, (
        "B8: while turn-t9 was the newest surface, turn-t1 was inert and turn-t9 was not; got "
        f"{inert_then!r}"
    )


@requires_chrome
def test_clicking_an_older_cards_button_both_ways_sends_no_ui_action(run):
    """B8, adversarial: with ``turn-t9`` the newest surface, the older ``turn-t1`` card's "Check
    answers" is clicked by a synthetic ``.click()`` (no hit-testing: only the bridge's own "newest
    surface only" rule can stop it) and by a CDP mouse click at the button's centre (hit-testing:
    ``inert`` stops it). Neither may put an ``astra.ui.action`` on the socket.

    Anti-vacuity: the same button sent exactly one action in phase C, when ``turn-t1`` was the
    newest surface (``test_clicking_check_answers_…``); the CDP click's point is inside the older
    card's box and in the viewport; and the guard's other side is the next row.
    """
    guard(run)
    assert run.quiz_tab_back.startswith("clicked"), (
        f"premise: turn-t1 was switched back to its {TAB_TITLES[0]!r} tab (only the active tab's "
        f"child is rendered) before turn-t9 arrived; the harness reports {run.quiz_tab_back!r}"
    )
    assert run.older_synthetic.startswith("clicked"), (
        "premise: the older card is still in the thread with its button (B7/B4: earlier cards stay "
        f"as history, never removed); the harness reports {run.older_synthetic!r}"
    )
    point = run.older_point if isinstance(run.older_point, dict) else {}
    assert point.get("w", 0) > 0 and point.get("h", 0) > 0, (
        f"premise: the older card's button has a box to click; got {run.older_point!r}"
    )
    assert point.get("in_viewport") and point.get("inside_wrapper"), (
        "premise: the mouse click lands on the older card — its centre is in the viewport and "
        f"inside the turn-t1 card wrapper; got {point!r}"
    )
    before = ui_actions(run.raw_before_older)
    assert len(before) == 1, (
        f"premise: phase C's one action is the only one before the older-card clicks; {before!r}"
    )
    after = ui_actions(run.raw_after_older)
    assert after == before, (
        "B8 (T-0024 criterion 9): astra.ui.action is sent only for the most recently created "
        "surface; clicking the older turn-t1 card (synthetic click, then a real mouse click) sent "
        f"{[(a.surface_id, a.name) for a in after[len(before) :]]}"
    )


@requires_chrome
def test_a_mouse_click_on_the_newest_cards_button_sends_exactly_one_action_for_it(run):
    """B8, the guard's too-wide side (.claude/rules/tests.md): a page that silenced every card
    once a second surface existed would pass the row above. The newest card still speaks."""
    guard(run)
    point = run.newest_point if isinstance(run.newest_point, dict) else {}
    assert point.get("in_viewport") and point.get("inside_wrapper"), (
        f"premise: the newest card's button is on screen inside its wrapper; got {run.newest_point!r}"
    )
    new = ui_actions(run.raw_after_newest)[len(ui_actions(run.raw_after_older)) :]
    assert len(new) == 1, (
        "B8: one real click on the newest surface's button is exactly one astra.ui.action; the "
        f"stub received {[(a.surface_id, a.name) for a in new]}"
    )
    assert (new[0].surface_id, new[0].name) == (NEWEST_SURFACE_ID, NEWEST_ACTION), (
        f"B8: the action names the newest surface and its own event; got {new[0]!r}"
    )


# --- criterion 9: the action round-trip ------------------------------------------------------


@requires_chrome
def test_clicking_check_answers_sends_exactly_one_astra_ui_action_that_the_real_codec_decodes(run):
    """Criterion 9, decoded by ``local_astra.contracts.codec.decode`` — no JS mirror."""
    guard(run)
    assert run.click_result.startswith("clicked"), (
        f"the harness could not click {CHECK_LABEL!r}: {run.click_result}"
    )
    assert not ui_actions(run.raw_before_click), (
        "criterion 9: nothing is sent before the user clicks; the stub had already received "
        f"{[a.name for a in ui_actions(run.raw_before_click)]}"
    )
    actions = ui_actions(run.raw_after_click)
    assert len(actions) == 1, (
        "criterion 9 negative: exactly one message per click — a double-send is the failure mode "
        f"to test; the stub received {len(actions)}: {actions!r}"
    )
    action = actions[0]
    assert action.surface_id == GATE_SURFACE_ID, (
        f"criterion 9: surface_id is the gate's rewritten id (T-0010 A7), not the model's "
        f"{MODEL_SURFACE_ID!r}; got {action.surface_id!r}"
    )
    assert action.name == "check_quiz", f"the model's own event name; got {action.name!r}"
    assert action.context.get("quizId") == "fractions_1", (
        f"criterion 9: the model's own context keys ride along; got {action.context!r}"
    )
    assert "answers" in action.context, (
        "criterion 9: `answers` is resolved by the binder from {'path': '/answers'} to the chosen "
        f"values; got {action.context!r}"
    )
    a2ui = action.context.get("a2ui")
    assert isinstance(a2ui, dict) and a2ui.get("sourceComponentId") == "check_btn", (
        "criterion 9: the A2UI action's sourceComponentId and timestamp are carried under a "
        f"single reserved context['a2ui'] object (AstraUiAction has no field for them); got {a2ui!r}"
    )
    assert action.data_model is not None and "answers" in action.data_model, (
        "criterion 9: data_model is populated because the fixture's createSurface sets "
        f"sendDataModel: true; got {action.data_model!r}"
    )


@requires_chrome
def test_a_tab_switch_and_a_scroll_send_nothing_at_all(run):
    """Criterion 9, negative: "no action is sent for a click the renderer did not raise one for"."""
    guard(run)
    assert run.tab_click_result.startswith("clicked"), (
        f"the harness could not click the {TAB_TITLES[1]!r} tab: {run.tab_click_result}"
    )
    assert len(ui_actions(run.raw_final)) == 1, (
        "criterion 9 negative: switching tab and scrolling raise no A2UI action, so the socket "
        f"still carries exactly the one message from the check button; got "
        f"{[a.name for a in ui_actions(run.raw_final)]}"
    )


@requires_chrome
def test_the_page_emits_no_client_type_other_than_its_own(run):
    """Criterion 3 negative: the shell still emits hello/frame; the panel adds ui.action only."""
    guard(run)
    types = set(HudStub.received_types(run.stub)) if run.stub else set()
    assert types <= {"astra.hello", "astra.frame", "astra.ui.action"}, (
        "criterion 3: hud/src/protocol.js CLIENT_TYPES is unchanged and this task's panel adds "
        f"astra.ui.action and nothing else; the stub recorded {sorted(types)}"
    )


# --- criterion 4: the adversarial one ---------------------------------------------------------


@requires_chrome
def test_a2ui_smuggled_through_spoken_text_an_unknown_type_or_a_bad_surface_never_renders(run):
    """Criterion 4, the honesty rule and the most likely silent failure.

    A complete A2UI body arrives three ways — inside ``astra.spoken.text``, inside an unknown
    ``astra.zzz``, and as an ``astra.ui.surface`` with ``canvas_idx: -1`` that ``decode``
    rejects. After all three no card exists anywhere; the legitimate surface that follows must
    still render and the socket must still be open.

    **Amended by T-0154 B1** (G2): "the pane still reads the placeholder" is replaced — the
    placeholder is retired — by "no card wrapper exists, no element anywhere renders the
    sentinel as its whole text, and the retired placeholder is not on the page".
    """
    guard(run)
    snap = run.after_adversarial
    # Scope: the *shown* channel (the card wrappers). The thread's utterance legitimately displays
    # the text of an astra.spoken verbatim, JSON and all — showing it as text is honest, turning
    # it into a surface is not, and criterion 4 is about the second.
    assert SMUGGLED_SENTINEL not in snap.shown_text, (
        "criterion 4: there is no code path that renders A2UI from anywhere but a decoded "
        "astra.ui.surface — not from spoken text, not from an unknown astra.*, not from a "
        f"message the codec rejected. The shown channel rendered {SMUGGLED_SENTINEL!r}"
    )
    assert snap.hooks == [], (
        "criterion 4 / T-0154 B4: after the three smuggling attempts no card wrapper "
        f"([data-ui-surface]) exists anywhere; got {snap.hooks!r}"
    )
    assert snap.probe.get("sentinel_leaves") == 0, (
        "criterion 4: no element anywhere on the page carries the smuggled Text component's text "
        f"as its whole text (a rendered component); {snap.probe.get('sentinel_leaves')!r} do"
    )
    assert snap.probe.get("placeholder_on_page") is False, (
        f"T-0154 B1: the {PLACEHOLDER!r} placeholder is retired; the page still shows it"
    )
    assert snap.roots == 0, (
        "criterion 4: no surface element is mounted for a message that never passed the gate"
    )
    assert "astra.zzz" not in snap.shown_text, (
        "an unknown astra.* type is ignored and counted, never rendered"
    )
    # and the socket survived all of it: the legitimate surface rendered afterwards
    assert run.per_canvas and PROMPTS[0] in run.per_canvas[-1].text, (
        "criterion 4: a subsequent legitimate surface still renders after the adversarial batch"
    )
    assert run.frames_after_all >= 1, (
        "criterion 4: the page kept sending frames to the end of the session, which is how we "
        "know the socket is still open and no handler threw"
    )


# --- criterion 8: a missing child is a placeholder, never invented content ---------------------


@requires_chrome
def test_a_dangling_child_reference_keeps_the_siblings_and_synthesises_nothing(run):
    """Criterion 8: render the rest of the tree, leave a visibly empty slot, invent no content.

    **Amended by T-0154 B4** (G2): the dangling ``turn-t9`` surface is now the session's second
    surface id, so "one ``<a2ui-surface>`` on the page" becomes "one inside ``turn-t9``'s wrapper,
    two on the page (``turn-t1`` stays as history)", and the invented-token scan is scoped to the
    ``turn-t9`` card (before, the whole shown channel *was* that card).
    """
    guard(run)
    text = run.dangling.text
    for sibling in SIBLING_TEXTS:
        assert sibling in text, (
            "criterion 8 negative: the surrounding components are not dropped; the sibling "
            f"{sibling!r} is missing from {text[:1000]!r}"
        )
    t9 = [h for h in run.dangling.hooks if h.get("surface") == "turn-t9"]
    assert len(t9) == 1, f"B4: one card wrapper for turn-t9; got {run.dangling.hooks!r}"
    for sibling in SIBLING_TEXTS:
        assert sibling in str(t9[0].get("text") or ""), (
            f"criterion 8: the sibling {sibling!r} renders inside turn-t9's own card"
        )
    invented = [
        token
        for token in ("undefined", "[object Object]", "NaN", "null")
        if token in str(t9[0].get("text") or "")
    ]
    assert not invented, (
        "criterion 8 negative: no text, label or default value is synthesised for the absent "
        f"component; the turn-t9 card contains {invented}"
    )
    assert t9[0].get("surfaces_inside") == 1, (
        "criterion 8: the surface still renders (one element inside its wrapper), it is not "
        f"discarded because one child reference dangles; got {t9[0]!r}"
    )
    assert run.dangling.roots == 2, (
        "B4: one <a2ui-surface> per surface id — turn-t1 (history) and turn-t9; the pierced DOM "
        f"has {run.dangling.roots}"
    )


# --- criterion 10: loopback only, and no audio, behaviourally ---------------------------------


@requires_chrome
def test_no_request_left_loopback_over_the_whole_session(run):
    """Criterion 10: every ``Network.requestWillBeSent`` of the session, host by host.

    Strictly stronger than the URL-literal grep it replaces for ``hud/vendor/**``
    (``tests/unit/test_hud_no_audio_no_egress.py``): a URL assembled at runtime defeats a grep and
    not the browser.
    """
    guard(run)
    assert run.requests, (
        "the CDP network log is empty, so this assertion would be vacuous — Network.enable must "
        "be on before navigation and the page must at least fetch its own modules"
    )
    offenders = [
        url
        for url in run.requests
        if url.split("://", 1)[0] in {"http", "https", "ws", "wss"}
        and not any(url.split("://", 1)[1].startswith(host) for host in LOOPBACK)
    ]
    assert not offenders, (
        "criterion 10: zero requests to a non-loopback host over the whole session (CLAUDE.md: "
        f"all services bind to loopback only; the page works with no network route): {offenders}"
    )


@requires_chrome
def test_the_rendered_page_holds_no_audio_element_and_constructs_no_audio_context(run):
    """Criterion 10: the HUD never touches audio (docs/01 D7, docs/02 ADR-013), shadow roots too."""
    guard(run)
    for snap, where in ((run.after_click, "after the click"), (run.dangling, "at the end")):
        assert snap.tags.get("audio", 0) == 0, (
            f"criterion 10: no <audio> element anywhere in the pierced DOM {where}; found "
            f"{snap.tags.get('audio')}"
        )
        assert list(snap.probe.get("audio_hits") or []) == [], (
            "criterion 10: the sentinel installed before any page script records every "
            f"AudioContext construction; it fired {snap.probe.get('audio_hits')!r} {where}"
        )


# --- criterion 11 ------------------------------------------------------------------------------


# --- T-0164 group B: the swap (pending until group B is activated) ------------------------------


@requires_chrome
def test_t0164_b1_no_vendor_request_no_a2ui_surface_and_attach_shadow_never_called(run):
    """B1: over the whole session (CDP, real Chrome) — no request under ``/vendor/``, zero
    ``<a2ui-surface>``, ``attachShadow`` called 0 times (instrumented before page scripts), and
    every card is light DOM: ``[data-ui-surface] > [data-a2ui-root]``. Amends T-0024 criterion 1
    ("the renderer is the vendored bundle") by name."""
    guard(run)
    assert run.requests, "premise: the CDP network log is not empty"
    vendor = [u for u in run.requests if "/vendor/" in urllib.parse.urlsplit(u).path]
    assert vendor == [], f"B1: the page requested the vendored bundle: {vendor}"
    for snap, where in ((run.newest, "after phase F"), (run.end, "at the end of the session")):
        assert snap.probe.get("attach_shadow_calls") == 0, (
            "B1: Element.prototype.attachShadow is never called (the sentinel was installed before "
            f"any page script); it was called {snap.probe.get('attach_shadow_calls')!r} times {where}"
        )
        assert snap.tags.get("a2ui-surface", 0) == 0, (
            f"B1: zero <a2ui-surface> elements {where}; found {snap.tags.get('a2ui-surface')}"
        )
        assert snap.probe.get("shadow_hosts") == 0, f"B1: no shadow root {where}"
        assert snap.hooks, f"premise: cards exist {where}"
        not_light = [h.get("surface") for h in snap.hooks if not h.get("direct_root")]
        assert not_light == [], (
            "B1: every card is light DOM, its wrapper's only element child the card root "
            f"([data-ui-surface] > [data-a2ui-root]); these are not: {not_light} {where}"
        )


@requires_chrome
def test_t0164_b2_re_sending_t1s_last_canvas_3_times_mutates_nothing(run):
    """B2 (A7 in the browser, adversarial): a ``MutationObserver`` on turn-t1's root records 0
    mutations while its last canvas is re-sent 3 times. The bridge drops a canvas_idx it has already
    applied, so the re-sends carry canvas 4, 5 and 6 with identical messages (what the gate sends
    for a turn's final canvas); ``data-ui-canvas-idx == "6"`` proves all three were applied.

    Premise amended (coordinator, 2026-10-27): phase C picked q1 and q3 by mouse, and the
    fixture's own ``updateDataModel`` (answers all ``[]``) legitimately resets those picks when the
    canvas is first re-sent (a later ``updateDataModel`` wins for path-bound inputs). That first
    re-send (canvas 3) is applied before the observer starts. The positive control: a canvas 7
    that edits one prompt is seen by the same observer, on the same root node."""
    guard(run)
    assert run.observe_start == "observing", (
        f"premise: turn-t1's light-DOM [data-a2ui-root] was observed; {run.observe_start!r}"
    )
    seen = run.observed_identical if isinstance(run.observed_identical, dict) else {}
    assert seen.get("canvas_idx") == "6", (
        f"premise: the bridge applied the three re-sends (canvas 4, 5, 6); the wrapper says {seen!r}"
    )
    assert seen.get("same_root") is True, f"B2/A7: the root is the same node: {seen!r}"
    assert seen.get("mutations") == [], (
        "B2/A7: re-applying an identical canvas writes nothing — no element created, inserted, "
        f"moved or removed, no text or attribute written; the observer saw {seen.get('mutations')!r}"
    )
    changed = run.observed_changed if isinstance(run.observed_changed, dict) else {}
    assert changed.get("same_root") is True and changed.get("canvas_idx") == "7", changed
    assert changed.get("mutations"), (
        "anti-vacuity: the same observer sees the canvas that edits a prompt; it saw nothing, so "
        f"the zero above proves nothing: {changed!r}"
    )
    assert PROMPTS[2] + " (edited)" in str(changed.get("text") or ""), changed


@requires_chrome
def test_t0164_b3_a_foreign_catalog_id_renders_nothing(run):
    """B3 (T-0024 criterion 4, kept): a ``createSurface`` whose ``catalogId`` is the vendor's own
    v0_9 id, or any other non-gate id, creates no ``[data-a2ui-root]`` and no card, and its Text is
    nowhere on the page as a component."""
    guard(run)
    snap = run.foreign
    ids = [h.get("surface") for h in snap.hooks]
    assert not [i for i in ids if str(i).startswith("turn-f")], (
        f"B3: no card wrapper for a foreign-catalog surface; the page has {ids!r}"
    )
    assert snap.probe.get("foreign_leaves") == 0, (
        f"B3: the foreign surface's Text {FOREIGN_SENTINEL!r} is rendered "
        f"{snap.probe.get('foreign_leaves')!r} times"
    )
    light = snap.probe.get("roots") if T0164_B else None
    before = run.newest.probe.get("roots") if T0164_B else None
    assert snap.hooks and all(h.get("direct_root") for h in snap.hooks), (
        "premise (B1): the session's legitimate cards are light-DOM [data-a2ui-root] cards, so "
        f"'no root for the foreign ones' is not vacuous; got {snap.hooks!r}"
    )
    assert light == before, (
        f"B3: the foreign surfaces added no [data-a2ui-root]: {before} -> {light}"
    )


@requires_chrome
def test_t0164_b4_check_answers_carries_the_resolved_answers_and_the_surface_model(run):
    """B4: with q1's "3/4" (value b) and q3's "2/3" (value a) picked by real mouse clicks on the
    light-DOM chips, "Check answers" sends one ``astra.ui.action`` that the production codec decodes,
    whose ``context.answers`` is the resolved ``/answers`` and whose ``data_model`` is the surface's
    own model: the fixture's ``updateDataModel`` value with those two picks (ruling 2026-10-27:
    ``{q1:["b"],q2:[],q3:["a"],q4:[],q5:[]}``)."""
    guard(run)
    bad = [p for p in run.picks if not (isinstance(p, dict) and p.get("hit"))]
    assert not bad and len(run.picks) == 2, (
        f"premise: both chips were found in the light DOM and hit by the mouse; got {run.picks!r}"
    )
    actions = ui_actions(run.raw_after_click)
    assert len(actions) == 1, f"one click, one action; got {actions!r}"
    action = actions[0]
    assert action.context.get("answers") == QUIZ_ANSWERS, (
        f"B4 / A9 / A10: context.answers is resolved from the data model; got {action.context!r}"
    )
    model_line = json.loads(FIXTURE.read_text().splitlines()[-1])["updateDataModel"]
    assert model_line["path"] == "/", "premise: the fixture sets the whole model"
    expected = dict(model_line["value"])
    expected["answers"] = QUIZ_ANSWERS
    assert action.data_model == expected, (
        "B1/B4: data_model is engine.dataModelFor(turn-t1) — the surface's own model with the two "
        f"picks; got {action.data_model!r}"
    )


@requires_chrome
def test_t0164_b4_the_gate_button_click_is_one_action_byte_equal_on_the_wire(run):
    """B4: a CDP mouse click on the newest card's gate-made button (light DOM) sends exactly one
    ``astra.ui.action``; the production codec decodes it; ``name == "__gate.clicked"``,
    ``context.label`` is the model's label, ``context.a2ui`` holds the button id and an ISO
    timestamp, and ``context`` has no other key. The surface sets no ``sendDataModel``, so
    ``data_model`` is None."""
    guard(run)
    point = run.gate_point if isinstance(run.gate_point, dict) else {}
    assert point.get("in_viewport") and point.get("hit"), (
        f"premise: the gate button is on screen in the light DOM; got {run.gate_point!r}"
    )
    before = ui_actions(run.raw_before_gate)
    new = ui_actions(run.raw_after_gate)[len(before) :]
    assert len(new) == 1, f"B4: one click, one astra.ui.action; got {new!r}"
    action = new[0]
    assert action.surface_id == GATE_CARD_ID and action.name == "__gate.clicked", action
    assert sorted(action.context) == ["a2ui", "label"], (
        f"B4: context holds label and a2ui and no other key; got {action.context!r}"
    )
    assert action.context["label"] == GATE_LABEL, action.context
    a2ui = action.context["a2ui"]
    assert isinstance(a2ui, dict) and sorted(a2ui) == ["sourceComponentId", "timestamp"], a2ui
    assert a2ui["sourceComponentId"] == "go", a2ui
    import re

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", str(a2ui["timestamp"])), (
        f"B4: the timestamp is Date.prototype.toISOString's shape; got {a2ui['timestamp']!r}"
    )
    assert action.data_model is None, f"no sendDataModel, no data_model; got {action.data_model!r}"


@requires_chrome
def test_t0164_a8_typed_text_survives_a_later_canvas_in_the_real_browser(run):
    """A8 in Chrome (the node fake cannot see focus loss from a real re-render): text typed into the
    gate card's literal-valued TextField keeps its value, caret and focus across the next canvas."""
    guard(run)
    assert run.field_focus == "focused", f"premise: the field was focused; {run.field_focus!r}"
    before = run.field_before if isinstance(run.field_before, dict) else {}
    want = {
        "value": TYPED,
        "start": len(TYPED),
        "end": len(TYPED),
        "active": True,
        "connected": True,
    }
    assert before == want, f"premise: the typed text landed; got {before!r}"
    assert run.field_after == want, (
        f"A8: a later canvas (adds {LATER_LINE!r}) keeps the typed text, the caret and the focus; "
        f"got {run.field_after!r}"
    )


@requires_chrome
def test_the_panel_is_installed_where_the_seam_expects_it(run):
    """Criterion 2: ``hud/src/ui/index.js`` is what ``panels.js`` imports; criterion 3: the panel
    consumes T-0025's seam and edits no file of T-0008's or T-0025's."""
    assert UI_INDEX.is_file(), "T-0024 adds hud/src/ui/index.js, exporting mount(host)"
    main = (HUD / "src" / "main.js").read_text()
    assert "subscribe: (fn) =>" in main, (
        "criterion 3: the subscribe seam is T-0025's and this task does not edit hud/src/main.js"
    )
