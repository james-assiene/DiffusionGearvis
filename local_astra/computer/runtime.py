"""The Playwright runtime behind ``BrowserToolExecutor`` (T-0111c): the only module that imports ``playwright``.

``PlaywrightDriver`` implements the driver seam ``executor.Driver`` (``tests/unit/test_computer_executor.py``
docstring) over Playwright's async API. ``playwright`` is imported lazily, at launch, so importing this module (and
the executor, the conductor or the cli) never loads it (criterion 16).

**Sandbox** (criteria 13-15; H-T111b, H-T111c-2, H-T111c-E):

- Playwright's own pinned Chromium (``services/versions.lock [browser]``), headless by default, in an ephemeral
  context with its own temporary profile, never the user's. A caller-given ``user_data_dir`` is checked by
  ``fence.check_profile_dir`` before it gets here and then used through ``launch_persistent_context``.
- Chromium's own sandbox stays as Playwright launches it (off; ``chromium_sandbox=False`` is Playwright's default).
  That is accepted for loopback fixtures only (H-T111c-S, Rulings 2026-10-20); real sites need an AppArmor profile or
  an explicit acceptance first.
- Egress, browser-wide: (1) ``--host-resolver-rules`` resolves every host except ``localhost`` and ``127.0.0.1`` to
  NOTFOUND (it applies to IP literals too, so ``198.51.100.7`` and ``127.0.1.1`` are unreachable); (2) a context
  route aborts every request whose URL is not ``url_allowed`` and a WebSocket route closes every such socket, each
  logged as ``fence.egress_record`` (host and scheme only); (3) ``service_workers="block"``; (4) background
  networking, component updates, sync, domain reliability and pings are switched off.
- The origin allow-list (T-0111g A3): with ``allowed_origins=None``, ``url_allowed`` is ``fence.is_loopback_url``
  (T-0111c's rule, unchanged). With a set it also requires the URL's ``scheme://host:port`` to be listed, so a
  page on the start-page origin cannot reach another loopback port (vLLM, the HUD, STT/TTS). The resolver rule
  cannot see ports, and Playwright calls a route only for the first URL of a redirect chain (probed 2026-10-21 on
  Playwright 1.64: a fulfilled 302 is followed unrouted), so in set mode every allowed request is fetched by the
  route itself without following redirects (``route.fetch(max_redirects=0)``) and fulfilled from that response; a
  redirect is never followed: it is aborted, and its target is logged when it is outside the set. WebSockets cannot
  redirect; the WebSocket route closes every socket outside the set (``ws://`` to the start page's own port too).
- Downloads are refused (``accept_downloads=False``) and reported; a file chooser is intercepted and left empty;
  every dialog (``alert``/``confirm``/``prompt``/``beforeunload``) is dismissed and reported; geolocation,
  notifications, camera and microphone are set to ``denied``.
- Popups: a new page becomes the active page at the next settle; at most ``MAX_PAGES`` stay open, the oldest is
  closed and reported.
- Replacement (T-0188 A1): ``replace_page(url)`` opens a fresh page in the same context, navigates it, makes it the
  active page and then closes every other page with ``run_before_unload=False`` (a page whose dismissed
  ``beforeunload`` cancels every navigation cannot be left otherwise).

**Settle** (criterion 9; mutation 11): wait for ``domcontentloaded`` on the active page, then two animation frames,
each bounded; a navigation that destroys the context mid-wait is waited for again (at most ``SETTLE_ROUNDS``). Smooth
scrolling is off, so a wheel scroll has landed by the second frame.

**Navigation** (T-0174 B1-1b): each adopted page counts its cross-document main-frame commits from Playwright's own
page events. A main-frame navigation request (``request``, ``is_navigation_request()``, its frame the main frame) is
recorded as pending; a main-frame ``framenavigated`` whose URL (without fragment) equals a pending request's is one
commit. A ``domcontentloaded`` counts only for a document that no such match counted. Same-document navigations
(fragment, ``pushState``/``replaceState``) issue no request and never count; subframes are ignored.
``navigation_mark``/``navigated_since`` compare the active page and that count; a navigation request still pending
at the read is "cannot tell". Nothing runs in the page and no CDP is used.

**Following a navigation** (T-0203 A): the same listeners also number, driver-wide, each main-frame navigation
request's start, each request-matched commit and each download, and wake every waiting follow (an
``asyncio.Event`` per waiter; nothing polls). A mark carries its driver and the sequence number at the mark (the
watermark). ``follow_navigation(mark, timeout_s)`` follows only what started after it on the mark's page (on a page
that became active since, all of that page's requests): a request still in flight or finished without a commit,
or a commit whose document has not reached ``domcontentloaded``. It waits for the commit, then
``domcontentloaded`` and ``_RAF2`` (a document replaced meanwhile is waited for again, ``settle()``'s rule):
``"loaded"``; the request failing, a download or the page closing: ``"ended"``; the budget spent: ``"timeout"``;
nothing to follow: ``"none"`` at once, with no page call. A request that started before the mark is never
followed.

**Main-frame URLs** (T-0192 A2): the same listeners keep, driver-wide, the URL of every main-frame navigation request
(each redirect hop is a request of its own) and of every main-frame commit (same-document ones included) of every
adopted page. ``main_frame_urls()`` returns them with the active page's ``page.url`` last and drains them; a
relaunch drops them. They are a safety input for the executor only (``fence.checkout_url``), never reported.

**Web mode** (T-0160 B-1, B-2, B-6; ``web=True``, with ``start_origin``): every launch starts its own
``EgressProxy`` (``tunnel_start_origin=True``, P1 (d)) before the browser and stops it at teardown, at close, and when
the launch fails or is cancelled. Chromium launches with ``proxy={"server": proxy.origin, "bypass": "<-loopback>"}``
(P1 (b): loopback goes through the proxy too), ``chromium_sandbox=True`` and ``CHROMIUM_ARGS`` plus
``--disable-quic``, always in a fresh non-persistent context (a ``user_data_dir`` is refused). The routes split three
ways: the start origin keeps the no-redirect fetch above; a public URL (``fence.web_url_allowed``) is
``route.continue_()``, so Chromium fetches it through the proxy and follows its redirects through it; everything else
is aborted and logged as above. Between two ``take_events()`` calls (the executor's action boundary) the proxy's
refusals reach ``on_egress`` once per ``{host, scheme}``, and each host it tunnelled is one ``{"kind": "host",
"host": ...}`` event. The proxy calls back on its own thread, so both go through a lock. With ``web=False`` none of this exists and the launch
is BASE's.

**Layout shifts** (T-0204 A2/A3): ``arm_shift_recorder()`` attaches a driver-owned CDP session to the active page and
sends ``PerformanceTimeline.enable`` with ``eventTypes: ["layout-shift"]``, once per page, bounded by
``SHIFT_ARM_TIMEOUT_S``; a failure gives ``False`` and leaves the page unarmed. A failed arm is remembered per page
(A-R3): a later arm on that page answers ``False`` at once with no CDP call, so its reads stay ``None``; the failed
session is detached best-effort, bounded by ``SHIFT_DETACH_TIMEOUT_S``; another page arms normally, and a relaunch
forgets every failure. Chromium then reports each layout shift as ``PerformanceTimeline.timelineEventAdded``. The recorder keeps only geometry: each source's
``previousRect``/``currentRect`` (viewport CSS px, clipped by Chromium) and an arrival counter. ``nodeId`` is never
resolved and nothing runs in the page (no init script, no page-visible global), so page script cannot read, forge or
suppress the record, only cause real shifts. A layout-shift entry whose geometry cannot be read is kept as one rect
covering the viewport (A-R2); ``sources: []`` is kept as no rect. Chromium names at most ``SHIFT_SOURCE_CAP`` (5)
sources per entry, a known Chromium limit (a probe shifted 12 blocks and got 5), so an entry at the cap may have moved
more nodes than it names: it is kept as one rect covering the viewport, whatever its sources' rects (zero-area ones
included). Reading it as moved everywhere is the fail-safe choice (A-R1); 4 or fewer sources are kept as their rects. At most ``MAX_SHIFT_ENTRIES`` entries are kept per page; ``shifts_since(mark)`` reads an
overflow since the mark as one rect covering the viewport, never as quiet, and answers ``None`` when it cannot tell
(unarmed, failed arm, a foreign mark, another generation or page, no active page). The session survives
cross-document navigations of its page (R1: no re-arm). It misses CSS ``transform`` moves and shifts inside
cross-origin iframes. A page's record is dropped when the page closes; a relaunch drops every record.
``screenshot_region`` is a clipped viewport capture (pixels only) for the pixel comparison beside it.

**Submit facts** (T-0206 A2): ``submit_facts(x, y)`` hit-tests the CSS point, descending through open shadow roots and
into same- and cross-origin frames (the point translated into each frame's content box, at most ``_MAX_FOCUS_DEPTH``
frames); ``submit_facts_focused()`` takes the deepest focused element the way ``focused_field`` does. Each resolves
the element to the nearest ``button``, ``input``, ``a`` or ``[role=button]`` in the composed tree (a slotted node's
ancestors run through its slot; a hit that stops on a shadow host takes the host's shadow control under the point;
a slot's label is its assigned text, review R-1) and returns ``fence.irreversible_submit``'s
``facts``: the control (a text-like or other non-button ``input`` has an empty label; a focused field's control is
its form's default button, else a structural empty-label control, and with no form an empty non-structural one), the
scope (the form owner, else the nearest ancestor holding a text-like input, else the document) and at most
``fence.MAX_FIELDS`` fields of it (no ``hidden`` inputs). A field's value never leaves the page: only ``filled`` (and
``visible``); a ``<select>`` is filled only when its value differs from its initial default-selected option (R-4). The focused read also marks the focused field ``"focused": True`` and says whether the focused element
is itself the control (``focused_control``). Both are one ``evaluate`` per frame in the page's main world (no init
script, no event, no focus or scroll change); page script can shadow the DOM APIs it calls, so a hostile page can
forge the answer, which only reopens what the page owner wants open anyway (the stated residual).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import math
import sys
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urldefrag, urljoin, urlsplit

from local_astra.computer import fence
from local_astra.computer.egress_proxy import Connect, EgressProxy, Resolve

if TYPE_CHECKING:
    from playwright.async_api import (
        Browser,
        BrowserContext,
        CDPSession,
        Dialog,
        Download,
        FileChooser,
        Frame,
        Page,
        Playwright,
        Request,
        Route,
        WebSocketRoute,
    )

__all__ = [
    "MAX_PAGES",
    "MAX_SHIFT_ENTRIES",
    "SHIFT_ARM_TIMEOUT_S",
    "BrowserEndedError",
    "FollowResult",
    "PlaywrightDriver",
    "failure_type_class",
    "url_allowed",
]

#: Pages kept open; a popup beyond this closes the oldest (criterion 15).
MAX_PAGES = 4
#: Bound of each settle wait (load state, two animation frames), in ms.
SETTLE_TIMEOUT_MS = 2_000
#: Navigations a settle follows before it stops waiting.
SETTLE_ROUNDS = 3
#: T-0192 A2: main-frame URLs kept between two ``main_frame_urls()`` reads; past it only checkout URLs are added
#: (a read can then miss ordinary URLs, never a checkout one).
MAX_MAIN_FRAME_URLS = 1024
#: Default timeout of Playwright calls that take one (screenshot, goto), in ms.
CALL_TIMEOUT_MS = 6_000
JPEG_QUALITY = 85
#: T-0204 A2: layout-shift entries kept per page (chosen, not measured); an overflow since a mark reads as moved
#: everywhere
MAX_SHIFT_ENTRIES = 256
#: T-0204 A3: bound of one arm (attach a CDP session, enable ``PerformanceTimeline``), in s (chosen, not measured;
#: R1: the enable round trip was 0.4-1.3 ms)
SHIFT_ARM_TIMEOUT_S = 0.5
#: T-0204 A-R3: bound of the best-effort detach of a failed arm's CDP session, in s (chosen, not measured)
SHIFT_DETACH_TIMEOUT_S = 0.2
#: T-0204 A-R1: Chromium attributes at most this many sources to one layout-shift entry; an entry at the cap reads as
#: moved everywhere
SHIFT_SOURCE_CAP = 5
_SHIFT_EVENT = "PerformanceTimeline.timelineEventAdded"
_RECT_KEYS = ("x", "y", "width", "height")
#: T-0203 A1: what ``PlaywrightDriver.follow_navigation`` observed
FollowResult = Literal["none", "loaded", "ended", "timeout"]
_RAF2 = "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))"
_DENIED = ("geolocation", "notifications", "camera", "microphone")
#: T-0160 amendment (reviewer A of group C): web mode also denies writing the clipboard, which in a headed
#: browser is the user's (a page could plant a command for the user to paste)
_WEB_DENIED = ("clipboard-write",)
CHROMIUM_ARGS = (
    "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE localhost , EXCLUDE 127.0.0.1",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-sync",
    "--disable-domain-reliability",
    "--disable-client-side-phishing-detection",
    "--no-pings",
    "--dns-prefetch-disable",
    "--disable-smooth-scrolling",
    "--deny-permission-prompts",
    # Defence in depth behind _NO_WEBRTC: no UDP outside a proxy, and there is no proxy.
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
)
#: T-0160 B-1: added to ``CHROMIUM_ARGS`` in web mode only; QUIC is UDP and would not go through the proxy.
WEB_EXTRA_ARGS = ("--disable-quic",)
#: T-0160 B-1, P1 (b): keep loopback on the proxy even if Playwright's forced ``<-loopback>`` is switched off.
PROXY_BYPASS = "<-loopback>"
#: T-0160 B-3 (b): the deepest focused element through open shadow roots, as the ``fence.secret_field`` keys plus
#: ``contenteditable`` (``"true"`` when editable) and ``opaque`` (``"true"`` for a custom element whose shadow root,
#: if any, is closed, so its inner focus cannot be read; reviewer A item 3). An ``iframe``/``frame``/``object``/
#: ``embed`` that holds the focus is reported as ``{"frame": true}`` so the driver descends into its frame.
_FOCUSED_JS = """() => {
  let el = document.activeElement;
  while (el && el.shadowRoot && el.shadowRoot.activeElement) el = el.shadowRoot.activeElement;
  if (!el) return null;
  const tag = (el.tagName || "").toLowerCase();
  if (tag === "iframe" || tag === "frame" || tag === "object" || tag === "embed") return { frame: true };
  const get = (n) => { const v = el.getAttribute && el.getAttribute(n); return typeof v === "string" ? v : ""; };
  let label = "";
  try {
    if (el.labels) label = Array.from(el.labels).map((l) => l.textContent || "").join(" ");
  } catch (e) {}
  const ids = get("aria-labelledby").split(/\\s+/).filter(Boolean);
  const root = el.getRootNode ? el.getRootNode() : document;
  for (const id of ids) {
    const ref = (root.getElementById ? root.getElementById(id) : null) || document.getElementById(id);
    if (ref) label += " " + (ref.textContent || "");
  }
  return {
    frame: false,
    attrs: {
      tag: tag,
      type: typeof el.type === "string" ? el.type : get("type"),
      autocomplete: get("autocomplete"),
      name: get("name"),
      id: get("id"),
      aria_label: get("aria-label"),
      placeholder: get("placeholder"),
      label: label.trim(),
      contenteditable: el.isContentEditable ? "true" : "",
      opaque: tag.indexOf("-") >= 0 && !el.shadowRoot ? "true" : "",
    },
  };
}"""
#: T-0160 B-3 (b): whether a frame element (in its parent's context) is the deepest focused element there.
_IS_FOCUSED_JS = """(frameEl) => {
  let el = document.activeElement;
  while (el && el.shadowRoot && el.shadowRoot.activeElement) el = el.shadowRoot.activeElement;
  return el === frameEl;
}"""
#: Frames ``focused_field`` descends through at most (a nesting bound, not a timeout).
_MAX_FOCUS_DEPTH = 8
#: T-0206 A2: the shared part of the two submit reads (module docstring, Submit facts). Labels are text nodes only,
#: skipping form controls, so a ``textarea``'s default text (a value) is never read as a label; strings are capped.
_SUBMIT_LIB = """
  const PRESS = new Set(["submit", "button", "image", "reset"]);
  const TEXTLIKE = new Set(["", "text", "email", "tel", "search", "url", "number", "password"]);
  const SKIP = new Set(["INPUT", "TEXTAREA", "SELECT", "OPTION", "SCRIPT", "STYLE", "TEMPLATE", "BUTTON"]);
  const MAX = 64, CAP = 300;
  const str = (v) => (typeof v === "string" ? v : "");
  const cap = (v) => str(v).slice(0, CAP);
  const attr = (el, n) => str(el && el.getAttribute ? el.getAttribute(n) : "");
  const tagOf = (el) => str(el && el.tagName).toLowerCase();
  const typeOf = (el) => str(el.type).toLowerCase();
  const up = (el) =>
    el.assignedSlot || el.parentElement || (el.parentNode && el.parentNode.host) || null;
  const textOf = (node) => {
    let out = "";
    const walk = (n) => {
      if (out.length >= CAP) return;
      if (n.nodeType === 3) { out += n.data; return; }
      if (n !== node && n.nodeType === 1 && SKIP.has(n.tagName)) return;
      if (n.nodeType === 1 && n.tagName === "SLOT" && n.assignedNodes) {
        const got = n.assignedNodes({ flatten: true });
        if (got.length) { for (const c of got) walk(c); return; }
      }
      for (const c of n.childNodes) walk(c);
    };
    walk(node);
    return out.slice(0, CAP);
  };
  const labelledBy = (el) => {
    const root = el.getRootNode ? el.getRootNode() : document;
    let out = "";
    for (const id of attr(el, "aria-labelledby").split(/\\s+/).filter(Boolean)) {
      const ref = (root.getElementById ? root.getElementById(id) : null) || document.getElementById(id);
      if (ref) out += " " + textOf(ref);
    }
    return out.trim();
  };
  const isField = (el) => {
    const t = tagOf(el);
    if (t === "textarea" || t === "select") return true;
    return t === "input" && !PRESS.has(typeOf(el)) && typeOf(el) !== "hidden";
  };
  const isTextLike = (el) =>
    tagOf(el) === "textarea" || (tagOf(el) === "input" && TEXTLIKE.has(typeOf(el)));
  const isControl = (el) => {
    const t = tagOf(el);
    if (t === "button" || t === "input" || t === "a") return true;
    return attr(el, "role").trim().toLowerCase().split(/\\s+/)[0] === "button";
  };
  const controlFrom = (el) => {
    for (let e = el; e; e = up(e)) if (isControl(e)) return e;
    return null;
  };
  const formOf = (el) => {
    const t = tagOf(el);
    if (t === "button" || t === "input" || t === "select" || t === "textarea") return el.form || null;
    return el.closest ? el.closest("form") : null;
  };
  const structuralOf = (el, form) => {
    if (!form) return false;
    const t = tagOf(el);
    return (t === "button" && typeOf(el) === "submit")
      || (t === "input" && (typeOf(el) === "submit" || typeOf(el) === "image"));
  };
  const controlLabel = (el) => {
    const t = tagOf(el);
    if (t === "input" && !PRESS.has(typeOf(el))) return "";
    const aria = attr(el, "aria-label").trim();
    if (aria) return aria;
    const by = labelledBy(el);
    if (by) return by;
    if (t === "input") return typeOf(el) === "image" ? attr(el, "alt") : str(el.value);
    return textOf(el) || attr(el, "title");
  };
  const hrefNav = (el) => {
    if (tagOf(el) !== "a" || !el.hasAttribute("href")) return false;
    const h = attr(el, "href").trim().toLowerCase();
    return h !== "" && !h.startsWith("#") && !h.startsWith("javascript:");
  };
  const controlFacts = (el, form) => ({
    tag: tagOf(el), type: cap(str(el.type || attr(el, "type")).toLowerCase()),
    role: cap(attr(el, "role").toLowerCase()), label: cap(controlLabel(el)),
    href_nav: hrefNav(el), structural: structuralOf(el, form),
  });
  const blank = (el, structural) => ({
    tag: el ? tagOf(el) : "", type: "", role: "", label: "", href_nav: false, structural: structural,
  });
  const fieldLabel = (el) => {
    let out = attr(el, "aria-label") + " " + labelledBy(el);
    try { if (el.labels) for (const l of el.labels) out += " " + textOf(l); } catch (e) {}
    return out.trim();
  };
  const filledOf = (el) => {
    const ty = typeOf(el);
    if (tagOf(el) === "input" && (ty === "checkbox" || ty === "radio")) return !!el.checked;
    if (tagOf(el) === "input" && ty === "file") return !!(el.files && el.files.length);
    if (tagOf(el) === "select") {
      const opts = Array.from(el.options || []);
      if (el.multiple) return opts.some((o) => o.selected !== o.defaultSelected);
      let dflt = null;
      for (const o of opts) if (o.defaultSelected) dflt = o;
      if (!dflt && !(el.size > 1)) dflt = opts.find((o) => !o.disabled) || null;
      return str(el.value) !== (dflt ? str(dflt.value) : "");
    }
    return str(el.value) !== "";
  };
  const visibleOf = (el) => {
    try { if (el.checkVisibility) return !!el.checkVisibility({ visibilityProperty: true }); } catch (e) {}
    return el.getClientRects().length > 0;
  };
  const fieldFacts = (el, focused) => {
    const out = {
      tag: tagOf(el), type: cap(typeOf(el)), autocomplete: cap(attr(el, "autocomplete")),
      name: cap(attr(el, "name")), id: cap(attr(el, "id")), placeholder: cap(attr(el, "placeholder")),
      label: cap(fieldLabel(el)), filled: filledOf(el), visible: visibleOf(el),
    };
    if (el === focused) out.focused = true;
    return out;
  };
  const deep = (root, pred, limit) => {
    const out = [];
    const visit = (r) => {
      for (const el of r.querySelectorAll("*")) {
        if (out.length >= limit) return;
        if (pred(el)) out.push(el);
        if (el.shadowRoot) visit(el.shadowRoot);
      }
    };
    if (root.shadowRoot) visit(root.shadowRoot);
    visit(root);
    return out;
  };
  const SEARCH_SEL = '[role="search" i], search';
  const factsFor = (anchor, form, control, focused) => {
    let scope = "form", els;
    if (form) {
      els = Array.from(form.elements).filter(isField).slice(0, MAX);
    } else {
      scope = "document";
      let a = up(anchor);
      while (a && deep(a, isTextLike, 1).length === 0) a = up(a);
      els = deep(a || document.documentElement, isField, MAX);
    }
    const searchScope = !!((anchor.closest && anchor.closest(SEARCH_SEL)) || (form && form.closest(SEARCH_SEL)));
    return {
      control: control, scope: scope, search_scope: searchScope,
      fields: els.map((e) => fieldFacts(e, focused)),
    };
  };
"""
#: T-0206 A2: the deepest element at a point of one frame, through open shadow roots.
_DEEP_HIT = """
  const deepHit = (x, y) => {
    let el = document.elementFromPoint(x, y);
    while (el && el.shadowRoot) {
      const inner = el.shadowRoot.elementFromPoint(x, y);
      if (!inner || inner === el) break;
      el = inner;
    }
    return el;
  };
"""
#: T-0206 review R-1: a hit that stops on a shadow host (a slotted text node is hit-tested to its light-DOM parent, the
#: host) resolves to the last control of the host's shadow tree whose box holds the point (the composed tree).
_HOST_CONTROL = """
  const hostControlAt = (host, x, y) => {
    if (!host || !host.shadowRoot) return null;
    let hit = null;
    const visit = (r) => {
      for (const e of r.querySelectorAll("*")) {
        if (isControl(e)) {
          const b = e.getBoundingClientRect();
          if (x >= b.left && x <= b.right && y >= b.top && y <= b.bottom) hit = e;
        }
        if (e.shadowRoot) visit(e.shadowRoot);
      }
    };
    visit(host.shadowRoot);
    return hit;
  };
"""
#: T-0206 A2: ``submit_facts`` in one frame: ``{"frame": true, "x", "y"}`` (the point translated into the hit frame's
#: content box) or ``{"frame": false, "facts"}``.
_SUBMIT_AT_JS = (
    "([x, y]) => {"
    + _SUBMIT_LIB
    + _DEEP_HIT
    + _HOST_CONTROL
    + """
  const el = deepHit(x, y);
  if (!el) return null;
  const t = tagOf(el);
  if (t === "iframe" || t === "frame") {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    return {
      frame: true,
      x: x - r.left - el.clientLeft - (parseFloat(cs.paddingLeft) || 0),
      y: y - r.top - el.clientTop - (parseFloat(cs.paddingTop) || 0),
    };
  }
  const ctl = controlFrom(el) || hostControlAt(el, x, y);
  if (!ctl) {
    return { frame: false, facts: { control: blank(el, false), scope: "document", search_scope: false, fields: [] } };
  }
  const form = formOf(ctl);
  return { frame: false, facts: factsFor(ctl, form, controlFacts(ctl, form), null) };
}"""
)
#: T-0206 A2: whether a frame element (in its parent's context) is the deepest element at the point there.
_IS_HIT_JS = "([frameEl, x, y]) => {" + _DEEP_HIT + "  return deepHit(x, y) === frameEl;\n}"
#: T-0206 A2: ``submit_facts_focused`` in one frame: ``{"frame": true}`` when a frame holds the focus, else
#: ``{"frame": false, "facts"}``. A focused control is the control (``focused_control: true``); a focused field's
#: control is its form's default button, else a structural empty-label one, and with no form an empty one.
_SUBMIT_FOCUSED_JS = (
    "() => {"
    + _SUBMIT_LIB
    + """
  let el = document.activeElement;
  while (el && el.shadowRoot && el.shadowRoot.activeElement) el = el.shadowRoot.activeElement;
  const t = tagOf(el);
  if (t === "iframe" || t === "frame") return { frame: true };
  if (!el || t === "body" || t === "html") {
    const none = { control: blank(el, false), scope: "document", search_scope: false, fields: [] };
    none.focused_control = false;
    return { frame: false, facts: none };
  }
  const form = formOf(el);
  if (isControl(el) && !isField(el)) {
    const facts = factsFor(el, form, controlFacts(el, form), el);
    facts.focused_control = true;
    return { frame: false, facts: facts };
  }
  let control = blank(el, false);
  if (form) {
    const dflt = Array.from(form.elements).find((e) => structuralOf(e, form));
    control = dflt ? controlFacts(dflt, form) : blank(null, true);
  }
  const facts = factsFor(el, form, control, el);
  facts.focused_control = false;
  return { frame: false, facts: facts };
}"""
)
#: WebRTC is disabled outright (ruling a918d74): an init script runs in every frame, about:blank children included,
#: before any page script, and removes the constructors, so no STUN/TURN/mDNS socket is ever opened. ``SharedWorker``
#: goes the same way (T-0111g A2, defence in depth): its context is not covered by the context routes.
_NO_WEBRTC = """(() => {
  for (const n of ["RTCPeerConnection", "webkitRTCPeerConnection", "RTCDataChannel", "RTCSessionDescription",
                   "SharedWorker"]) {
    try { delete window[n]; } catch (e) {}
    if (n in window) {
      try { Object.defineProperty(window, n, { value: undefined, writable: false, configurable: false }); } catch (e) {}
    }
  }
})();"""
#: T-0160 amendment (reviewer A of group C), web mode only: a page cannot write the clipboard. ``execCommand``
#: ``copy``/``cut`` return false with no effect (other commands are untouched), and ``navigator.clipboard.write`` /
#: ``writeText`` reject without writing. Runs in every frame before any page script, as ``_NO_WEBRTC``; behind the
#: ``clipboard-write`` permission denial (``_WEB_DENIED``).
_NO_CLIPBOARD_WRITE = """(() => {
  try {
    const exec = Document.prototype.execCommand;
    const guarded = function (command, ...rest) {
      const c = String(command).toLowerCase();
      if (c === "copy" || c === "cut") return false;
      return exec.call(this, command, ...rest);
    };
    Object.defineProperty(Document.prototype, "execCommand", { value: guarded, writable: false, configurable: false });
  } catch (e) {}
  try {
    if (typeof Clipboard !== "undefined") {
      const refuse = function () { return Promise.reject(new DOMException("clipboard writes are off", "NotAllowedError")); };
      for (const n of ["write", "writeText"]) {
        Object.defineProperty(Clipboard.prototype, n, { value: refuse, writable: false, configurable: false });
      }
    }
  } catch (e) {}
  // a keyboard copy/cut (Ctrl+C/X) fires a copy/cut event whose handler could set clipboardData: this capture
  // listener runs before any page listener, stops them and cancels the clipboard write
  for (const t of ["copy", "cut"]) {
    window.addEventListener(t, (e) => { e.stopImmediatePropagation(); e.preventDefault(); }, true);
  }
})();"""


class BrowserEndedError(RuntimeError):
    """``ensure()`` under ``end_when_closed`` found the launched browser gone or its last page closed, and refused
    to relaunch or open a page (T-0111g A3); the executor reports the browser closed."""


class _NavigationMark:
    """T-0174 B1-1: ``PlaywrightDriver.navigation_mark``'s opaque mark: the browser generation, the active page, and
    that page's count of main-frame documents at the mark. T-0203 A: also the driver that issued it (``owner``) and
    that driver's navigation sequence number at the mark (``seq``, the follow's watermark)."""

    __slots__ = ("documents", "generation", "owner", "page", "seq")

    def __init__(
        self, generation: int, page: Page | None, documents: int, owner: object, seq: int
    ) -> None:
        self.generation, self.page, self.documents = generation, page, documents
        self.owner, self.seq = owner, seq


class _ShiftRecorder:
    """T-0204 A2: one page's layout-shift record (module docstring, Layout shifts). ``entries`` holds
    ``(arrival number, rects)``, oldest first, at most ``MAX_SHIFT_ENTRIES``; ``count`` is the last arrival number.
    ``armed`` is set once the enable succeeded and cleared when the record is dropped."""

    __slots__ = ("armed", "count", "entries")

    def __init__(self) -> None:
        self.armed = False
        self.count = 0
        self.entries: deque[tuple[int, tuple[tuple[float, float, float, float], ...]]] = deque()


class _ShiftMark:
    """T-0204 A2: ``PlaywrightDriver.shift_mark``'s opaque mark: the driver, the generation, the active page, its
    record (``None``: not armed then) and that record's arrival count at the mark."""

    __slots__ = ("count", "generation", "owner", "page", "recorder")

    def __init__(
        self,
        owner: object,
        generation: int,
        page: Page | None,
        recorder: _ShiftRecorder | None,
        count: int,
    ) -> None:
        self.owner, self.generation, self.page = owner, generation, page
        self.recorder, self.count = recorder, count


class _PageNavigation:
    """T-0174 B1-1b: one adopted page's navigation state (module docstring, Navigation)."""

    __slots__ = ("awaiting_dcl", "commit_seq", "commits", "download_seq", "requests", "started")

    def __init__(self) -> None:
        #: cross-document main-frame commits counted so far
        self.commits = 0
        #: a commit was counted by a request match and its document's ``domcontentloaded`` has not arrived yet
        self.awaiting_dcl = False
        #: main-frame navigation requests not yet matched by a commit: URL without fragment -> response finished
        self.requests: dict[str, bool] = {}
        #: T-0203 A: the driver sequence number at which each entry of ``requests`` started
        self.started: dict[str, int] = {}
        #: T-0203 A: the sequence number of the last request-matched commit, and of the last download (0: none)
        self.commit_seq = 0
        self.download_seq = 0

    def pending(self) -> bool:
        """A main-frame navigation request is in flight (not failed, not finished, not committed)."""
        return any(not finished for finished in self.requests.values())


class PlaywrightDriver:
    """The driver seam over one Playwright Chromium. ``generation`` counts (re)launches; ``page`` is the active
    page. ``on_egress`` receives each blocked off-loopback attempt as ``{"host", "scheme"}``. With
    ``end_when_closed`` (the executor's end mode), ``ensure()`` never relaunches a launched browser and never opens a
    page when the last one closed: it raises ``BrowserEndedError`` instead. With the default ``False`` it does both,
    as in T-0111c."""

    def __init__(
        self,
        *,
        viewport: tuple[int, int],
        on_egress: Callable[[dict[str, str]], None],
        user_data_dir: Path | None = None,
        headless: bool = True,
        allowed_origins: frozenset[str] | None = None,
        end_when_closed: bool = False,
        web: bool = False,
        start_origin: str | None = None,
        proxy_resolve: Resolve | None = None,
        proxy_connect: Connect | None = None,
    ) -> None:
        if web and start_origin is None:
            raise ValueError("web mode needs the start origin (T-0160 B-1)")
        if web and user_data_dir is not None:
            raise ValueError(
                "web mode uses a fresh profile only; a user_data_dir is refused (T-0160 B-2)"
            )
        self.web = web
        self._start_origin = start_origin
        self._proxy_resolve = proxy_resolve
        self._proxy_connect = proxy_connect
        self._proxy: EgressProxy | None = None
        #: T-0160 B-6: the hosts and the proxy refusals already reported since the last ``take_events()``
        self._host_lock = threading.Lock()
        self._hosts_seen: set[str] = set()
        self._blocked_seen: set[tuple[str, str]] = set()
        self.viewport = viewport
        self._end_when_closed = end_when_closed
        self.generation = 0
        #: +1 each time a different page becomes the active one after the first launch (ruling 683cc1e)
        self.page_serial = 0
        self._on_egress = on_egress
        self._user_data_dir = user_data_dir
        self._headless = headless
        #: web mode: the start origin is the set (the no-redirect route logs a redirect leaving it)
        self._allowed_origins = (
            allowed_origins
            if not web or allowed_origins is not None
            else frozenset({start_origin or ""})
        )
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._alive = False
        #: set by the close listener of the current context; a context closed during ``_launch`` fails the launch
        self._context_closed = False
        self._pages: list[Page] = []
        self._page: Page | None = None
        self._new_pages: list[Page] = []
        self._events: list[dict[str, Any]] = []
        self._permission_session: CDPSession | None = None
        #: T-0134 A4: the CDP session ``layout_metrics`` reads through, and the page it is attached to
        self._metrics_session: tuple[Page, CDPSession] | None = None
        #: T-0174 B1-1b: navigation state per adopted, open page (dropped when the page closes)
        self._navigation: dict[Page, _PageNavigation] = {}
        #: T-0192 A2: main-frame navigation request and commit URLs of every adopted page since the last read
        self._main_urls: list[str] = []
        #: T-0203 A: numbers navigation request starts, commits and downloads (marks hold a watermark of it)
        self._nav_seq = 0
        #: T-0203 A: one event per running ``follow_navigation``, set by every navigation listener
        self._nav_waiters: set[asyncio.Event] = set()
        #: T-0204 A2: the layout-shift record of each armed page (dropped when the page closes and at relaunch)
        self._shift_recorders: dict[Page, _ShiftRecorder] = {}
        #: T-0204 A-R3: pages whose arm failed (never retried; dropped when the page closes and at relaunch)
        self._shift_failed: set[Page] = set()

    # ------------------------------------------------------------------ lifecycle

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("the browser has no active page; call ensure() first")
        return self._page

    async def bring_to_front(self) -> None:
        """T-0198 C8: raise the active page's window (``Page.bring_to_front``); ``RuntimeError`` with no page."""
        await self.page.bring_to_front()

    def _running(self) -> bool:
        if self._context is None or not self._alive:
            return False
        return self._browser is None or self._browser.is_connected()

    @property
    def connected(self) -> bool:
        """The current generation's browser is running (T-0111g A2; read only under ``end_when_closed``)."""
        return self._running()

    @property
    def page_count(self) -> int:
        """Pages open in the current browser; 0 when it is not running (T-0111g A2)."""
        if not self._running() or self._context is None:
            return 0
        return sum(1 for p in self._context.pages if not p.is_closed())

    async def ensure(self) -> None:
        """Launch if not running; relaunch (a new generation) if the browser died. A closed active page is replaced
        by the newest open one, or a new page, without a relaunch. A popup that opened after the last settle becomes
        the active page here, so the next frame shows it (criterion 15). Every change of the active page raises
        ``page_serial`` before any await, so a caller that times out here still sees it.

        The executor runs this under its action timeout, so it may be cancelled at any await: the browser counts as
        running only once ``_launch`` has installed the whole fence, and ``generation`` moves only after that.

        Under ``end_when_closed`` the two reopen paths refuse (class docstring), including a last page that closes
        while a popup is being adopted or its load is awaited."""
        if self._running():
            if await self._switch_to_new_pages():
                page = self.page
                try:
                    await self._wait_loaded(page)
                except Exception:
                    # end mode only: a page that closed under the wait falls through to the closed-page rule below
                    if not (self._end_when_closed and page.is_closed()):
                        raise
            if self._page is None or self._page.is_closed():
                self._pages = [p for p in self._pages if not p.is_closed()]
                self.page_serial += 1
                if self._pages:
                    self._page = self._pages[-1]
                else:
                    self._page = None
                    if self._end_when_closed:
                        raise BrowserEndedError("the last page closed")
                    assert self._context is not None
                    page = await self._context.new_page()
                    self._adopt(page)
                    self._page = page
            return
        if self._end_when_closed and self.generation:
            raise BrowserEndedError("the browser is no longer running")
        await self._teardown()
        await self._launch()
        if self.generation:
            self.page_serial += 1
        self.generation += 1

    async def _launch(self) -> None:
        from playwright.async_api import async_playwright

        if self._pw is None:
            self._pw = await async_playwright().start()
        w, h = self.viewport
        options: dict[str, Any] = {
            "viewport": {"width": w, "height": h},
            "screen": {"width": w, "height": h},
            "device_scale_factor": 1,
            "accept_downloads": False,
            "service_workers": "block",
            "permissions": [],
        }
        chromium = self._pw.chromium
        self._context_closed = False
        # T-0160 B-1: in web mode the proxy runs before the browser exists and is stopped on any failed or
        # cancelled launch (the browser is torn down by the next ensure(), as in BASE)
        proxy = self._start_proxy() if self.web else None
        try:
            if self._user_data_dir is not None:
                context = await chromium.launch_persistent_context(
                    str(self._user_data_dir),
                    headless=self._headless,
                    args=list(CHROMIUM_ARGS),
                    **options,
                )
                self._browser = None
            elif proxy is not None:
                self._browser = await chromium.launch(
                    headless=self._headless,
                    args=[*CHROMIUM_ARGS, *WEB_EXTRA_ARGS],
                    chromium_sandbox=True,
                    proxy={"server": proxy.origin, "bypass": PROXY_BYPASS},
                )
                context = await self._browser.new_context(**options)
            else:
                self._browser = await chromium.launch(
                    headless=self._headless, args=list(CHROMIUM_ARGS)
                )
                context = await self._browser.new_context(**options)
            self._context = context
            context.on("close", self._on_context_close)
            context.set_default_timeout(CALL_TIMEOUT_MS)
            if proxy is not None:
                await self._route_web(context)
            else:
                allowed = self._allowed_origins
                if allowed is not None:
                    # the two route predicates are disjoint (allowed / not allowed), so each URL matches exactly
                    # one route and the registration order does not matter
                    await context.route(lambda u: url_allowed(u, allowed), self._no_redirect_route)
                await context.route(lambda u: not url_allowed(u, allowed), self._block_route)
                await context.route_web_socket(
                    lambda u: not url_allowed(u, allowed), self._block_websocket
                )
            await context.add_init_script(_NO_WEBRTC)
            if proxy is not None:
                await context.add_init_script(
                    _NO_CLIPBOARD_WRITE
                )  # T-0160 amendment: web mode only
            self._pages, self._new_pages, self._events = [], [], []
            page = context.pages[0] if context.pages else await context.new_page()
            self._adopt(page)
            self._page = page
            context.on("page", self._on_new_page)
            await self._deny_permissions(page)
            if self._context_closed:
                # the context closed mid-launch: never mark it alive; the next ensure() tears down and relaunches
                raise RuntimeError("the browser context closed during launch")
        except BaseException:
            if proxy is not None:
                self._stop_proxy()
            raise
        # last: a launch cancelled or failed before this point is torn down and relaunched by the next ensure()
        self._alive = True

    # ------------------------------------------------------------------ web mode (T-0160)

    def _start_proxy(self) -> EgressProxy:
        """A new ``EgressProxy`` for this launch, started; the previous one was stopped by ``_teardown``."""
        self._stop_proxy()
        with self._host_lock:
            self._hosts_seen, self._blocked_seen = set(), set()
        proxy = EgressProxy(
            start_origin=self._start_origin,
            on_blocked=self._on_proxy_blocked,
            on_host=self._on_proxy_host,
            resolve=self._proxy_resolve,
            connect=self._proxy_connect,
            tunnel_start_origin=True,
        )
        proxy.start()
        self._proxy = proxy
        return proxy

    def _stop_proxy(self) -> None:
        proxy, self._proxy = self._proxy, None
        if proxy is not None:
            proxy.stop()

    @property
    def proxy(self) -> EgressProxy | None:
        """The running egress proxy of the current launch (web mode); ``None`` otherwise."""
        return self._proxy

    async def _route_web(self, context: BrowserContext) -> None:
        """B-1's route split; the three page predicates are disjoint, so each URL matches exactly one route."""
        start_origin = self._start_origin
        assert start_origin is not None
        start = frozenset({start_origin})

        def is_start(u: str) -> bool:
            return url_allowed(u, start)

        def is_public(u: str) -> bool:
            return not is_start(u) and fence.web_url_allowed(u, start_origin)

        # Playwright passes (route, request) to a handler that takes two parameters: handlers take one
        await context.route(is_start, self._no_redirect_route)
        await context.route(is_public, self._continue_route)
        await context.route(lambda u: not is_start(u) and not is_public(u), self._block_route)
        await context.route_web_socket(
            lambda u: not fence.web_url_allowed(u, start_origin), self._block_websocket
        )

    async def _continue_route(self, route: Route) -> None:
        with contextlib.suppress(Exception):
            await route.continue_()

    def _on_proxy_blocked(self, record: dict[str, str]) -> None:
        """Proxy thread: a refusal, reported once per ``{host, scheme}`` per action (O-A2 (4))."""
        key = (record.get("host", ""), record.get("scheme", ""))
        with self._host_lock:
            if key in self._blocked_seen:
                return
            self._blocked_seen.add(key)
        self._on_egress(dict(record))

    def _on_proxy_host(self, host: str) -> None:
        """Proxy thread: a host the proxy tunnelled, one ``host`` event per action (O-A2 (4))."""
        name = host.lower()
        with self._host_lock:
            if name in self._hosts_seen:
                return
            self._hosts_seen.add(name)
            self._events.append({"kind": "host", "host": name})

    async def focused_field(self) -> dict[str, str] | None:
        """T-0160 B-3 (b): the attributes (``fence.secret_field`` keys) of the deepest focused element of the active
        page, through open shadow roots and into same- and cross-origin frames; ``None`` when nothing holds the
        focus. Unbounded here: the executor bounds it (``FOCUS_TIMEOUT_S``) and fails closed."""
        frame = self.page.main_frame
        for _ in range(_MAX_FOCUS_DEPTH):
            got = await frame.evaluate(_FOCUSED_JS)
            if not isinstance(got, dict):
                return None
            if not got.get("frame"):
                attrs = got.get("attrs")
                if not isinstance(attrs, dict):
                    return None
                return {str(k): v if isinstance(v, str) else "" for k, v in attrs.items()}
            child = None
            for candidate in frame.child_frames:
                if candidate.is_detached():
                    # Playwright keeps a navigated-away frame in child_frames; its frame_element() raises
                    continue
                handle = await candidate.frame_element()
                try:
                    if await frame.evaluate(_IS_FOCUSED_JS, handle):
                        child = candidate
                        break
                finally:
                    await handle.dispose()
            if child is None:
                raise RuntimeError("the focused frame was not found")
            frame = child
        raise RuntimeError("frames nested too deep to read the focus")

    async def submit_facts(self, x: float, y: float) -> dict[str, Any] | None:
        """T-0206 A2: ``fence.irreversible_submit``'s facts for the element at the CSS point ``(x, y)`` of the active
        page (module docstring, Submit facts); ``None`` when a hit frame could not be matched or frames nest too
        deep. Unbounded here: the executor bounds it (``SUBMIT_TIMEOUT_S``) and fails open."""
        frame = self.page.main_frame
        for _ in range(_MAX_FOCUS_DEPTH):
            got = await frame.evaluate(_SUBMIT_AT_JS, [x, y])
            if not isinstance(got, dict):
                return None
            if not got.get("frame"):
                facts = got.get("facts")
                return facts if isinstance(facts, dict) else None
            nx, ny = got.get("x"), got.get("y")
            if not isinstance(nx, (int, float)) or not isinstance(ny, (int, float)):
                return None
            child = None
            for candidate in frame.child_frames:
                if candidate.is_detached():
                    continue
                handle = await candidate.frame_element()
                try:
                    if await frame.evaluate(_IS_HIT_JS, [handle, x, y]):
                        child = candidate
                        break
                finally:
                    await handle.dispose()
            if child is None:
                return None
            frame, x, y = child, float(nx), float(ny)
        return None

    async def submit_facts_focused(self) -> dict[str, Any] | None:
        """T-0206 A2: ``fence.irreversible_submit``'s facts for the deepest focused element of the active page
        (module docstring, Submit facts), plus ``focused_control`` and the focused field's ``"focused": True``;
        ``None`` when the focused frame could not be found. Unbounded here, as ``submit_facts``."""
        frame = self.page.main_frame
        for _ in range(_MAX_FOCUS_DEPTH):
            got = await frame.evaluate(_SUBMIT_FOCUSED_JS)
            if not isinstance(got, dict):
                return None
            if not got.get("frame"):
                facts = got.get("facts")
                return facts if isinstance(facts, dict) else None
            child = None
            for candidate in frame.child_frames:
                if candidate.is_detached():
                    continue
                handle = await candidate.frame_element()
                try:
                    if await frame.evaluate(_IS_FOCUSED_JS, handle):
                        child = candidate
                        break
                finally:
                    await handle.dispose()
            if child is None:
                return None
            frame = child
        return None

    async def _deny_permissions(self, page: Page) -> None:
        """Set the four permissions to ``denied`` for the whole context. A CDP permission override lasts as long as
        the session that set it, so the session is kept: a browser-level one when there is a browser, else (a
        persistent context) one on the first page."""
        assert self._context is not None
        page_session = await self._context.new_cdp_session(page)
        info = await page_session.send("Target.getTargetInfo")
        context_id = info["targetInfo"]["browserContextId"]
        if self._browser is not None:
            await page_session.detach()
            session = await self._browser.new_browser_cdp_session()
        else:
            session = page_session
        for name in (*_DENIED, *_WEB_DENIED) if self.web else _DENIED:
            await session.send(
                "Browser.setPermission",
                {
                    "permission": {"name": name},
                    "setting": "denied",
                    "browserContextId": context_id,
                },
            )
        self._permission_session = session

    async def _teardown(self) -> None:
        """Best-effort close of a previous (possibly dead) browser; its frames are never used again. The web-mode
        proxy is stopped last, even when a close is cancelled."""
        context, browser = self._context, self._browser
        self._context, self._browser, self._page, self._alive = None, None, None, False
        self._pages, self._new_pages, self._permission_session = [], [], None
        self._metrics_session = None
        self._navigation = {}
        self._main_urls = []
        for recorder in self._shift_recorders.values():
            recorder.armed = False
        self._shift_recorders = {}
        self._shift_failed = set()
        try:
            if context is not None:
                with contextlib.suppress(Exception):
                    await context.close()
            if browser is not None:
                with contextlib.suppress(Exception):
                    await browser.close()
        finally:
            self._stop_proxy()

    async def close(self) -> None:
        await self._teardown()
        if self._pw is not None:
            pw, self._pw = self._pw, None
            await pw.stop()

    def _on_context_close(self, _context: BrowserContext) -> None:
        self._context_closed = True
        self._alive = False

    # ------------------------------------------------------------------ pages, events, fence

    def _adopt(self, page: Page) -> None:
        if page in self._pages:
            return
        self._pages.append(page)
        self._navigation[page] = _PageNavigation()
        page.on("dialog", self._on_dialog)
        page.on("download", self._on_download)
        page.on("download", functools.partial(self._on_nav_download, page))
        page.on("filechooser", self._on_filechooser)
        page.on("close", self._on_page_close)
        page.on("request", functools.partial(self._on_nav_request, page))
        page.on("requestfinished", functools.partial(self._on_nav_request_done, page, True))
        page.on("requestfailed", functools.partial(self._on_nav_request_done, page, False))
        page.on("framenavigated", functools.partial(self._on_frame_navigated, page))
        page.on("domcontentloaded", self._on_document)

    # ------------------------------------------------------------------ navigation (T-0174 B1-1b)

    def _main_frame_navigation(self, page: Page, request: Request) -> _PageNavigation | None:
        """The page's state when ``request`` is a navigation request of its main frame; ``None`` otherwise."""
        state = self._navigation.get(page)
        if state is None:
            return None
        try:
            if not request.is_navigation_request() or request.frame != page.main_frame:
                return None
        except Exception:
            return None  # a request whose frame is gone is not the main frame's
        return state

    def _next_nav_seq(self) -> int:
        self._nav_seq += 1
        return self._nav_seq

    def _wake_followers(self) -> None:
        """T-0203 A2: a navigation listener ran; every waiting ``follow_navigation`` re-reads the state."""
        for event in self._nav_waiters:
            event.set()

    def _on_nav_request(self, page: Page, request: Request) -> None:
        self._wake_followers()
        state = self._main_frame_navigation(page, request)
        if state is None:
            return
        # T-0192 A2: every main-frame request counts, each redirect hop being its own request
        self._record_main_url(request.url)
        # requests that finished without a commit (a 204, a download) are dropped when the next one starts
        state.requests = {u: f for u, f in state.requests.items() if not f}
        url = _without_fragment(request.url)
        state.requests[url] = False
        state.started = {u: n for u, n in state.started.items() if u in state.requests}
        state.started[url] = self._next_nav_seq()

    def _on_nav_request_done(self, page: Page, finished: bool, request: Request) -> None:
        """``requestfinished`` keeps the entry for a later ``framenavigated`` but no longer pending; ``requestfailed``
        drops it."""
        self._wake_followers()
        state = self._main_frame_navigation(page, request)
        if state is None:
            return
        url = _without_fragment(request.url)
        if url not in state.requests:
            return
        if finished:
            state.requests[url] = True
        else:
            del state.requests[url]
            state.started.pop(url, None)

    def _on_frame_navigated(self, page: Page, frame: Frame) -> None:
        self._wake_followers()
        state = self._navigation.get(page)
        if state is None or frame != page.main_frame:
            return
        # T-0192 A2: a commit's URL, same-document ones (``pushState``) included
        self._record_main_url(frame.url)
        url = _without_fragment(frame.url)
        if url in state.requests:
            del state.requests[url]
            state.started.pop(url, None)
            state.commits += 1
            state.awaiting_dcl = True
            state.commit_seq = self._next_nav_seq()

    def _record_main_url(self, url: str) -> None:
        """T-0192 A2: keep ``url`` for the next ``main_frame_urls()`` (once; bounded by ``MAX_MAIN_FRAME_URLS``)."""
        store = self._main_urls
        if url in store:
            return
        if len(store) < MAX_MAIN_FRAME_URLS or fence.checkout_url(url):
            store.append(url)

    def main_frame_urls(self) -> tuple[str, ...]:
        """T-0192 A2: the main-frame navigation request and commit URLs of the adopted pages since the previous call
        (redirect hops included), then the active page's current URL (``page.url``) last; the store is drained.
        Synchronous: it reads what the page listeners recorded and Playwright's own URL, runs nothing in the page,
        reads no DOM or text and uses no CDP. Raises when there is no active page (the store is then kept)."""
        current = self.page.url
        out, self._main_urls = self._main_urls, []
        return (*out, current)

    def _on_document(self, page: Page) -> None:
        """Playwright emits ``domcontentloaded`` for the main frame only, once per new document: it counts only when
        no request match already counted that document."""
        self._wake_followers()
        state = self._navigation.get(page)
        if state is None:
            return
        if state.awaiting_dcl:
            state.awaiting_dcl = False
        else:
            state.commits += 1

    def _on_new_page(self, page: Page) -> None:
        if page in self._pages:
            return  # a page this driver opened itself
        self._adopt(page)
        self._new_pages.append(page)

    def _on_page_close(self, page: Page) -> None:
        self._wake_followers()
        if page in self._pages:
            self._pages.remove(page)
        if page in self._new_pages:
            self._new_pages.remove(page)
        self._navigation.pop(page, None)
        recorder = self._shift_recorders.pop(page, None)
        if recorder is not None:
            recorder.armed = False
        self._shift_failed.discard(page)

    async def _on_dialog(self, dialog: Dialog) -> None:
        # Playwright runs a coroutine listener as its own task; a page that went away has no dialog to dismiss.
        self._events.append({"kind": "dialog", "type": dialog.type})
        with contextlib.suppress(Exception):
            await dialog.dismiss()

    def _on_nav_download(self, page: Page, _download: Download) -> None:
        """T-0203 A3: a download on ``page`` ends a navigation followed there."""
        state = self._navigation.get(page)
        if state is not None:
            state.download_seq = self._next_nav_seq()
        self._wake_followers()

    async def _on_download(self, download: Download) -> None:
        self._events.append({"kind": "download"})
        with contextlib.suppress(Exception):
            await download.cancel()

    def _on_filechooser(self, _chooser: FileChooser) -> None:
        self._events.append({"kind": "file_chooser"})

    async def _block_route(self, route: Route) -> None:
        url = route.request.url
        if url.split(":", 1)[0].lower() in fence.NETWORK_SCHEMES:
            self._on_egress(fence.egress_record(url))
        with contextlib.suppress(Exception):
            await route.abort("blockedbyclient")

    async def _no_redirect_route(self, route: Route) -> None:
        """Set mode only: fetch an allowed request without following redirects and fulfil it from the response. A
        redirect is aborted (never followed, its target unrouted otherwise); a target outside the set is logged as
        an egress record. A request the route cannot fetch is aborted."""
        url = route.request.url
        try:
            response = await route.fetch(max_redirects=0)
        except Exception:
            with contextlib.suppress(Exception):
                await route.abort("failed")
            return
        if 300 <= response.status < 400:
            target = urljoin(url, response.headers.get("location", ""))
            if (
                not url_allowed(target, self._allowed_origins)
                and target.split(":", 1)[0].lower() in fence.NETWORK_SCHEMES
            ):
                self._on_egress(fence.egress_record(target))
            with contextlib.suppress(Exception):
                await route.abort("blockedbyclient")
            return
        with contextlib.suppress(Exception):
            await route.fulfill(response=response)

    async def _block_websocket(self, ws: WebSocketRoute) -> None:
        self._on_egress(fence.egress_record(ws.url))
        with contextlib.suppress(Exception):
            await ws.close()

    def take_events(self) -> list[dict[str, Any]]:
        """The events since the last call; it is also the action boundary of the web-mode dedup (B-6)."""
        with self._host_lock:
            out, self._events = self._events, []
            self._hosts_seen, self._blocked_seen = set(), set()
        return out

    async def _switch_to_new_pages(self) -> bool:
        """Make the newest popup the active page and close the oldest pages beyond ``MAX_PAGES``."""
        fresh = [p for p in self._new_pages if not p.is_closed()]
        self._new_pages = []
        if not fresh:
            return False
        if fresh[-1] is not self._page:
            self.page_serial += 1
        self._page = fresh[-1]
        while len(self._pages) > MAX_PAGES:
            oldest = self._pages.pop(0)
            await oldest.close()
            self._events.append({"kind": "page_closed"})
        return True

    async def _wait_loaded(self, page: Page) -> None:
        """Bounded wait for a newly adopted page's ``domcontentloaded``; a slow page is used as it is."""
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        with contextlib.suppress(PlaywrightTimeoutError):
            await page.wait_for_load_state("domcontentloaded", timeout=SETTLE_TIMEOUT_MS)

    # ------------------------------------------------------------------ the seam

    async def settle(self) -> None:
        from playwright.async_api import Error as PlaywrightError
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        for _ in range(SETTLE_ROUNDS):
            await self._switch_to_new_pages()
            page = self.page
            with contextlib.suppress(PlaywrightTimeoutError):
                await page.wait_for_load_state("domcontentloaded", timeout=SETTLE_TIMEOUT_MS)
            try:
                await asyncio.wait_for(page.evaluate(_RAF2), SETTLE_TIMEOUT_MS / 1000)
            except TimeoutError:
                pass  # bounded: a page that renders no frame is screenshotted as it is
            except PlaywrightError as exc:
                if not _navigated_away(exc):
                    raise
                continue  # a navigation replaced the document: wait for the new one
            if not self._new_pages:
                return
        await self._switch_to_new_pages()

    def navigation_mark(self) -> object:
        """T-0174 B1-1: an opaque mark of the active page's main-frame document (the executor takes it at dispatch
        start). Reads only what this driver already counted; nothing is sent to the page."""
        page = self._page
        state = self._navigation.get(page) if page is not None else None
        return _NavigationMark(
            self.generation, page, state.commits if state is not None else 0, self, self._nav_seq
        )

    def navigated_since(self, mark: object) -> bool | None:
        """T-0174 B1-1/B1-1b: ``True`` when the active page now shows a different main-frame document than at
        ``mark``: a different page is active (a newly opened page the settle switched to, H-T174-1), or the same page
        has committed a cross-document main-frame navigation since. ``None`` when this driver cannot tell: a
        main-frame navigation request is still pending, a mark it did not issue, a restarted browser, no active page
        then or now, or a page it no longer tracks. ``False`` otherwise."""
        if not isinstance(mark, _NavigationMark) or mark.generation != self.generation:
            return None
        page = self._page
        if page is None or mark.page is None:
            return None
        if page is not mark.page:
            return True
        state = self._navigation.get(page)
        if state is None:
            return None
        if state.commits > mark.documents:
            return True
        if state.pending():
            return None
        return False

    async def follow_navigation(self, mark: object, timeout_s: float) -> FollowResult:
        """T-0203 A: follow the active page's main-frame navigation that started after ``mark`` (module docstring,
        Following a navigation) for at most ``timeout_s``. ``"none"`` at once, with no page call, when there is
        nothing to follow, the mark is not this driver's or is from another generation, or there is no active page.
        Woken by the navigation listeners; the only script it runs is ``_RAF2``; a Playwright error ends it
        (``"timeout"`` when the time is up, else ``"ended"``); ``CancelledError`` propagates."""
        if (
            not isinstance(mark, _NavigationMark)
            or mark.owner is not self
            or mark.generation != self.generation
        ):
            return "none"
        page = self._page
        if page is None or mark.page is None:
            return "none"
        state = self._navigation.get(page)
        if state is None:
            return "none"
        # a page that became active after the mark: all of its requests count
        watermark = mark.seq if page is mark.page else 0
        if not _fresh_requests(state, watermark) and not (
            state.awaiting_dcl and state.commit_seq > watermark
        ):
            return "none"
        return await self._follow(page, watermark, timeout_s)

    async def _follow(self, page: Page, watermark: int, timeout_s: float) -> FollowResult:
        """``follow_navigation``'s wait, once it knows there is something to follow."""
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        # a commit counts as this follow's when its document is still loading or it came after the follow started
        start_seq = self._nav_seq
        wake = asyncio.Event()
        self._nav_waiters.add(wake)
        try:
            while True:
                state = self._navigation.get(page)
                if state is None or state.download_seq > watermark:
                    return "ended"  # the page closed, or the navigation became a download
                left = deadline - loop.time()
                if left <= 0:
                    return "timeout"
                if _fresh_requests(state, watermark):
                    wake.clear()
                    try:
                        await asyncio.wait_for(wake.wait(), left)
                    except TimeoutError:
                        return "timeout"
                    continue
                if not (
                    state.commit_seq > watermark
                    and (state.awaiting_dcl or state.commit_seq > start_seq)
                ):
                    return "ended"  # the request failed without a commit
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=left * 1000)
                    left = deadline - loop.time()
                    if left <= 0:
                        return "timeout"
                    await asyncio.wait_for(page.evaluate(_RAF2), left)
                    return "loaded"
                except (TimeoutError, PlaywrightTimeoutError):
                    return "timeout"
                except Exception as exc:
                    if loop.time() >= deadline:
                        return "timeout"
                    if _navigated_away(exc):
                        continue  # a navigation replaced the document: wait for the new one
                    return "ended"
        finally:
            self._nav_waiters.discard(wake)

    async def screenshot(self) -> bytes:
        return await self.page.screenshot(
            type="jpeg", quality=JPEG_QUALITY, scale="css", full_page=False
        )

    async def layout_metrics(self) -> dict[str, float]:
        """T-0134 A4: the active page's geometry in CSS px from Chromium's own layout, CDP
        ``Page.getLayoutMetrics`` on a CDP session attached to that page: ``scroll_y`` (the layout viewport's
        ``pageY``), ``viewport_height`` (its ``clientHeight``) and ``content_height`` (``cssContentSize.height``).
        Page script cannot change these (redefining ``window.scrollY`` or ``scrollingElement`` changes what the
        page reads, not the layout). A scrolling element inside the page is not covered. The session is kept per
        active page and dropped when a read fails, so the next read attaches afresh."""
        page = self.page
        held = self._metrics_session
        if held is None or held[0] is not page:
            assert self._context is not None
            held = (page, await self._context.new_cdp_session(page))
            self._metrics_session = held
        try:
            metrics = await held[1].send("Page.getLayoutMetrics")
        except Exception:
            self._metrics_session = None
            raise
        viewport, content = metrics["cssLayoutViewport"], metrics["cssContentSize"]
        return {
            "scroll_y": float(viewport["pageY"]),
            "viewport_height": float(viewport["clientHeight"]),
            "content_height": float(content["height"]),
        }

    async def screenshot_region(self, x: float, y: float, w: float, h: float) -> bytes:
        """T-0204 A1: one JPEG of the active page's viewport (as ``screenshot()``) clipped to the rectangle
        ``(x, y, w, h)`` in viewport CSS px, clamped to the viewport. ``ValueError`` when nothing of it is inside the
        viewport (or a value is not finite), ``RuntimeError`` with no active page. Pixels only: no input, no scroll,
        no script."""
        page = self.page
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError(f"region {(x, y, w, h)} is not finite")
        vw, vh = float(self.viewport[0]), float(self.viewport[1])
        x0, y0 = max(0.0, float(x)), max(0.0, float(y))
        x1, y1 = min(vw, float(x) + float(w)), min(vh, float(y) + float(h))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"region {(x, y, w, h)} has nothing inside the {vw:g}x{vh:g} viewport")
        return await page.screenshot(
            type="jpeg",
            quality=JPEG_QUALITY,
            scale="css",
            full_page=False,
            clip={"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
        )

    # ------------------------------------------------------------------ layout shifts (T-0204 A2/A3)

    async def arm_shift_recorder(self) -> bool:
        """T-0204 A3: record the active page's layout shifts (module docstring, Layout shifts). ``True`` when armed,
        at once and with no CDP call when this page already is; ``False`` on any failure (no active page, the attach
        or the enable failing or taking longer than ``SHIFT_ARM_TIMEOUT_S``), the page then staying unarmed. A failed
        attach or enable is remembered for the page (A-R3): later arms there answer ``False`` at once with no CDP
        call, and the failed session is detached best-effort within ``SHIFT_DETACH_TIMEOUT_S``.
        ``CancelledError`` propagates."""
        try:
            page = self._page
            context = self._context
            if page is None or context is None:
                return False
            held = self._shift_recorders.get(page)
            if held is not None and held.armed:
                return True
            if page in self._shift_failed:
                return False
            recorder = _ShiftRecorder()
            handler = functools.partial(self._on_layout_shift, recorder)
            attached: list[CDPSession] = []

            async def enable() -> None:
                session = await context.new_cdp_session(page)
                attached.append(session)
                session.on(_SHIFT_EVENT, handler)
                await session.send("PerformanceTimeline.enable", {"eventTypes": ["layout-shift"]})

            ok = False
            try:
                await asyncio.wait_for(enable(), SHIFT_ARM_TIMEOUT_S)
                # a relaunch or a page change meanwhile: the record would not be the active page's
                ok = self._page is page and self._context is context
            except Exception:
                if self._context is context:
                    self._shift_failed.add(page)
            finally:
                if not ok:
                    for session in attached:
                        with contextlib.suppress(Exception):
                            session.remove_listener(_SHIFT_EVENT, handler)
            if not ok:
                for session in attached:
                    await _detach_bounded(session)
                return False
            recorder.armed = True
            self._shift_recorders[page] = recorder
            return True
        except asyncio.CancelledError:
            raise
        except Exception:
            return False

    def _on_layout_shift(self, recorder: _ShiftRecorder, params: Any) -> None:
        """One ``timelineEventAdded``: keep its layout-shift sources' rects (geometry only); never raises."""
        try:
            if not recorder.armed:
                return
            event = params.get("event") if isinstance(params, dict) else None
            if not isinstance(event, dict):
                return
            if event.get("type") != "layout-shift" and "layoutShiftDetails" not in event:
                return
            rects = _shift_rects(event.get("layoutShiftDetails"))
            if rects is None:
                # a shift whose geometry cannot be read, or at the source cap, may have been anywhere
                rects = (self._viewport_rect(),)
            recorder.count += 1
            recorder.entries.append((recorder.count, rects))
            while len(recorder.entries) > max(1, MAX_SHIFT_ENTRIES):
                recorder.entries.popleft()
        except Exception:
            return

    def _viewport_rect(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, float(self.viewport[0]), float(self.viewport[1]))

    def shift_mark(self) -> object:
        """T-0204 A2: an opaque mark of the active page's layout-shift record (synchronous; no page or CDP call)."""
        page = self._page
        recorder = self._shift_recorders.get(page) if page is not None else None
        if recorder is not None and not recorder.armed:
            recorder = None
        return _ShiftMark(
            self, self.generation, page, recorder, recorder.count if recorder is not None else 0
        )

    def shifts_since(self, mark: object) -> tuple[tuple[float, float, float, float], ...] | None:
        """T-0204 A2: the rects ``(x, y, w, h)`` (viewport CSS px) of every layout-shift entry recorded on the active
        page after ``mark``; ``()`` when none. One rect covering the viewport when entries since the mark were
        dropped. ``None`` when it cannot tell: the page was not armed at the mark or is not now, a foreign mark,
        another generation or page, no active page. Synchronous; never raises."""
        try:
            if (
                not isinstance(mark, _ShiftMark)
                or mark.owner is not self
                or mark.generation != self.generation
            ):
                return None
            page = self._page
            if page is None or mark.page is not page or mark.recorder is None:
                return None
            recorder = self._shift_recorders.get(page)
            if recorder is None or recorder is not mark.recorder or not recorder.armed:
                return None
            entries = recorder.entries
            oldest = entries[0][0] if entries else recorder.count + 1
            if recorder.count > mark.count and oldest > mark.count + 1:
                return (self._viewport_rect(),)  # entries since the mark were dropped
            return tuple(r for n, rects in entries if n > mark.count for r in rects)
        except Exception:
            return None

    async def full_page_screenshot(self, max_height_css: int) -> bytes:
        """T-0140 A3: one JPEG of the whole active page at CSS scale (``full_page=True``, Chromium's
        capture-beyond-viewport), clipped to the top ``max_height_css`` CSS px. Its width is the page's CSS content
        width (at least the viewport's), read from CDP ``Page.getLayoutMetrics`` on the session ``layout_metrics``
        keeps. It dispatches no input, does not scroll and reads no page text, title, URL or DOM: pixels and layout
        metrics only (A4)."""
        page = self.page
        held = self._metrics_session
        if held is None or held[0] is not page:
            assert self._context is not None
            held = (page, await self._context.new_cdp_session(page))
            self._metrics_session = held
        try:
            metrics = await held[1].send("Page.getLayoutMetrics")
        except Exception:
            self._metrics_session = None
            raise
        width = max(float(self.viewport[0]), float(metrics["cssContentSize"]["width"]))
        return await page.screenshot(
            type="jpeg",
            quality=JPEG_QUALITY,
            scale="css",
            full_page=True,
            clip={"x": 0, "y": 0, "width": width, "height": max_height_css},
        )

    async def mouse_move(self, x: float, y: float) -> None:
        await self.page.mouse.move(x, y)

    async def mouse_down(self) -> None:
        await self.page.mouse.down()

    async def mouse_up(self) -> None:
        await self.page.mouse.up()

    async def key_down(self, key: str) -> None:
        await self.page.keyboard.down(key)

    async def key_up(self, key: str) -> None:
        await self.page.keyboard.up(key)

    async def type_text(self, text: str) -> None:
        await self.page.keyboard.type(text)

    async def scroll_by(self, dx: float, dy: float) -> None:
        await self.page.mouse.wheel(dx, dy)

    async def goto(self, url: str) -> None:
        await self.page.goto(url, wait_until="domcontentloaded")

    async def replace_page(self, url: str) -> None:
        """T-0188 A1: a new page in the same context, navigated to ``url``, becomes the active page (``page_serial``
        +1); then every other page is closed without running ``beforeunload``. The new page opens first, so
        ``page_count`` never reaches 0 (``end_when_closed``). It is adopted with the usual listeners and kept out of
        the popups. When its navigation fails it is closed again and the active page is unchanged. In a persistent
        context the permission overrides live on a page's CDP session, so they are set again on the new page once the
        others are closed (T-0188 A-R1)."""
        if self._context is None:
            raise RuntimeError("the browser has no context; call ensure() first")
        context = self._context
        page = await context.new_page()
        self._adopt(page)
        if page in self._new_pages:
            self._new_pages.remove(page)
        try:
            await page.goto(url, wait_until="domcontentloaded")
        except BaseException:
            with contextlib.suppress(Exception):
                await page.close(run_before_unload=False)
            raise
        self._page = page
        self.page_serial += 1
        for other in [p for p in context.pages if p is not page and not p.is_closed()]:
            # best effort, as ``_teardown``: a page that went away meanwhile has nothing to close
            with contextlib.suppress(Exception):
                await other.close(run_before_unload=False)
        if self._browser is None:
            # after the closes (T-0188 A-R1): closing the page that held the overrides' CDP session drops them
            await self._deny_permissions(page)


_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def _origin_key(url: str) -> tuple[str, str, int] | None:
    """``(scheme, host, port)`` of a URL, lowercased, the port defaulted by scheme; ``None`` when unparseable."""
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if host is None or scheme not in _DEFAULT_PORTS:
        return None
    return scheme, host.lower(), port if port is not None else _DEFAULT_PORTS[scheme]


def url_allowed(url: str, allowed_origins: frozenset[str] | None) -> bool:
    """The route and WebSocket predicate (T-0111g A3). ``None``: ``fence.is_loopback_url`` (T-0111c). A set: also
    loopback, and the URL's ``scheme://host:port`` is one of the listed origins exactly (``localhost`` is not
    ``127.0.0.1``, ``ws`` is not ``http``, the default port is the scheme's)."""
    if not fence.is_loopback_url(url):
        return False
    if allowed_origins is None:
        return True
    key = _origin_key(url)
    return key is not None and any(_origin_key(o) == key for o in allowed_origins)


def _without_fragment(url: str) -> str:
    """A URL without its fragment: a navigation request's URL never carries one, ``framenavigated``'s may (T-0174)."""
    return urldefrag(url).url


def failure_type_class(exc: BaseException) -> str:
    """T-0188 A2: the part of a failed navigation's reason class decided by the exception's *type*: ``timeout``
    (Playwright's ``TimeoutError`` or the builtin ``TimeoutError``, which is ``asyncio.TimeoutError``),
    ``target_closed`` (Playwright's ``TargetClosedError``) or ``other``. The message is never read. Playwright is not
    imported here: its exception classes are looked up only when it is already loaded (no Playwright exception can
    exist otherwise; ``playwright.async_api.TimeoutError`` is ``playwright._impl._errors.TimeoutError``, and 1.64.0
    does not re-export ``TargetClosedError``)."""
    if isinstance(exc, TimeoutError):
        return "timeout"
    errors = sys.modules.get("playwright._impl._errors")
    if errors is None:
        return "other"
    if isinstance(exc, errors.TimeoutError):
        return "timeout"
    if isinstance(exc, errors.TargetClosedError):
        return "target_closed"
    return "other"


async def _detach_bounded(session: CDPSession) -> None:
    """T-0204 A-R3: best-effort detach of a failed arm's CDP session, bounded by ``SHIFT_DETACH_TIMEOUT_S``; a hang or
    an error is swallowed. ``CancelledError`` propagates."""
    with contextlib.suppress(Exception):
        await asyncio.wait_for(session.detach(), SHIFT_DETACH_TIMEOUT_S)


def _shift_rects(details: Any) -> tuple[tuple[float, float, float, float], ...] | None:
    """T-0204 A2: the previous and current rects of each source of one ``layoutShiftDetails``, or ``None`` when the
    entry must read as moved everywhere: its geometry cannot be read (A-R2), or it names ``SHIFT_SOURCE_CAP`` sources
    or more, Chromium's cap (A-R1). ``nodeId`` is ignored."""
    if not isinstance(details, dict):
        return None
    sources = details.get("sources")
    if not isinstance(sources, list) or len(sources) >= SHIFT_SOURCE_CAP:
        return None
    out: list[tuple[float, float, float, float]] = []
    for src in sources:
        if not isinstance(src, dict):
            return None
        for key in ("previousRect", "currentRect"):
            r = src.get(key)
            if not isinstance(r, dict):
                return None
            values: list[float] = []
            for k in _RECT_KEYS:
                v = r.get(k)
                if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
                    return None
                values.append(float(v))
            out.append((values[0], values[1], values[2], values[3]))
    return tuple(out)


def _fresh_requests(state: _PageNavigation, watermark: int) -> bool:
    """T-0203 A: a main-frame navigation request that started after ``watermark`` has neither failed nor
    committed (in flight, or finished without a commit)."""
    return any(state.started.get(url, 0) > watermark for url in state.requests)


def _navigated_away(exc: Exception) -> bool:
    """Playwright's "Execution context was destroyed, most likely because of a navigation"; a closed target
    (``TargetClosedError``) is not this and propagates."""
    return "Execution context was destroyed" in str(exc)
