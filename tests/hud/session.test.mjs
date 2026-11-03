// T-0162 group D — the Restart button: hud/src/session/gate.js and hud/src/session/index.js.
//
// docs/tasks/T-0162.md D-1 (the button, its two-step confirm, one astra.session.reset through the
// panel's own whitelist, disabled while the socket is closed) and D-4 (never on a key press);
// *Tests to write* "session.test.mjs: one send per confirmed Restart; Cancel sends nothing; the
// whitelist holds; a double-click sends one". The page-wide clear (D-2) has its node rows in
// thread.test.mjs (thread, cards, Tools pane), denoise.test.mjs and state.test.mjs; the browser
// rows are tests/integration/test_hud_restart.py.
//
// The failure modes aimed at: a Restart that fires without its confirm (R6: the memory wipe is
// irreversible); a double-click that wipes twice; a Restart that fires on a key press; a gate that
// is stuck after a refused send or an echo (too wide), next to one that lets a second reset out
// before the first is echoed (too narrow).
//
// API this pins (the task names the files and the whitelist constant; the other names are these):
//
//   hud/src/session/gate.js — importable in node, no DOM at module top level
//     SESSION_CLIENT_TYPES = ["astra.session.reset"]   the panel's own egress whitelist (D-1)
//     createSessionGate({ send, log }) -> {
//       reset() -> bool    hands exactly one { type: "astra.session.reset" } (no other field: A-1
//                          "no fields", unknown fields are malformed) to send; true iff it did.
//                          While one is in flight (sent, not yet echoed) it sends nothing, false.
//                          A send that throws or rejects ends the flight: never a throw.
//       onCleared(m)       feed a decoded server message; an astra.session.cleared ends the flight,
//                          any other message changes nothing
//       pending() -> bool  a reset went out whose astra.session.cleared has not arrived
//     }
//
//   hud/src/session/index.js — mount(host), the panels.js seam; never throws
//     host = { document, send, subscribe, log, sessionSlot, onConnection }
//       sessionSlot        the header element index.html gives the button (main.js passes it)
//       onConnection(fn)   main.js calls fn(true) while the socket is open and fn(false) when it
//                          is not; fn is called once at registration with the current value
//     DOM hooks (inside sessionSlot):
//       [data-restart-button]       <button>, text "Restart", disabled while the socket is closed
//       [data-restart-confirm]      the inline confirm, hidden until Restart is pressed; its text
//                                   says a new session starts, that the conversation and everything
//                                   Gearvis remembers is cleared (H-T162-1, R6), and that the agent's
//                                   browser goes back to its start page (group A ruling (c))
//       [data-restart-confirm-yes]  <button> "Restart" inside the confirm: the one that sends
//       [data-restart-cancel]       <button> "Cancel" inside the confirm: sends nothing, closes it
//
// Strict-xfail gate (the pendingT0162B mechanism of tests/hud/panels.test.mjs): every D row runs
// through `pendingT0162D`, which reads docs/tasks/T-0162.md's **Status:** line. Until it records
// "group D: tests <a>..<b>, activated <sha>" each row must FAIL, and a row that passes throws
// XPASS(strict). `ASTRA_RUN_PENDING=1 node --test tests/hud/session.test.mjs` runs them for real.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);

// --- the group D strict-xfail gate -----------------------------------------------------------

const T0162_TASK = fileURLToPath(new URL("docs/tasks/T-0162.md", REPO));
const T0162_STATUS = existsSync(T0162_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0162_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const D_ACTIVATED = /\bgroup D:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const D_PENDING = !D_ACTIVATED.test(T0162_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0162D(name, fn) {
  test(name + (D_PENDING ? " [pending T-0162 D: strict xfail until group D is activated]" : ""), async (t) => {
    if (!D_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0162 group D row passed while docs/tasks/T-0162.md's status does not " +
          "record group D's activation. Either group D has landed (record 'group D: tests …, " +
          "activated <sha>' in the status) or the row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0162 D pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0162 D gate: the status line is readable and only group D's own phrase activates the rows", () => {
  assert.ok(T0162_STATUS.length > 0, "docs/tasks/T-0162.md must carry a **Status:** line");
  // Hand-written samples: groups A and B's activation wording (today's line) must not switch D on.
  assert.equal(D_ACTIVATED.test("in progress (group B: tests 1697b85b..42d62069, activated a1d0db53; group A: tests 2decefd7..f18ad08c, activated f18ad08c)"), false);
  assert.equal(D_ACTIVATED.test("in progress (group A: …, activated f18ad08c; group D: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6)"), true);
  assert.equal(D_ACTIVATED.test("in progress (group D: tests 1a2b3c4..5d6e7f8, not yet activated)"), false, "activation names its commit");
  assert.equal(D_ACTIVATED.test("in progress (group C: tests …, activated 1234abc)"), false, "group C's activation is not group D's");
});

// --- module loading (the ESM shim of tests/hud/panels.test.mjs) ------------------------------

let ESM_ROOT = null;
function srcRoot() {
  if (ESM_ROOT) return ESM_ROOT;
  const real = fileURLToPath(new URL("hud/src", REPO));
  const dir = mkdtempSync(join(tmpdir(), "astra-hud-esm-"));
  writeFileSync(join(dir, "package.json"), '{"type":"module","private":true}');
  if (existsSync(real)) cpSync(real, join(dir, "src"), { recursive: true });
  ESM_ROOT = pathToFileURL(join(dir, "src") + "/");
  return ESM_ROOT;
}

// Origin proof (.claude/rules/tests.md): where the modules under test were imported from.
console.log(`# session origin: repo=${fileURLToPath(REPO)} modules=${srcRoot().href}`);

async function load(name) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch (err) {
    assert.fail(
      `T-0162 D-1 adds hud/src/${name}, importable in node as an ES module with no DOM at module ` +
        `top level. node could not import it: ${err.message}`,
    );
  }
}

async function gateModule() {
  const mod = await load("session/gate.js");
  assert.equal(typeof mod.createSessionGate, "function", "D-1: session/gate.js exports createSessionGate({ send, log })");
  return mod;
}

// --- a DOM double ----------------------------------------------------------------------------
//
// Closed by delegation in the direction that matters (.claude/rules/tests.md): reading a member the
// double does not implement throws a *named* error rather than returning undefined, so a missing
// feature of the double can never make a row pass by rendering nothing. Expandos the panel sets on
// its own elements are kept and read back. Markup sinks throw (text is text). A disabled element
// does not run its click listeners, as a browser's does not.

const flush = () => new Promise((resolve) => setImmediate(resolve));

function fakeDocument() {
  const docListeners = [];
  const doc = {
    nodeType: 9,
    _listeners: docListeners,
    activeElement: null,
    createElement: (tag) => makeElement(doc, String(tag).toLowerCase()),
    createElementNS: (_ns, tag) => makeElement(doc, String(tag).toLowerCase()),
    createTextNode: (text) => makeText(String(text)),
    createDocumentFragment: () => makeElement(doc, "#fragment"),
    addEventListener: (type, fn) => docListeners.push({ type, fn }),
    removeEventListener: (type, fn) => {
      const at = docListeners.findIndex((l) => l.type === type && l.fn === fn);
      if (at >= 0) docListeners.splice(at, 1);
    },
  };
  doc.documentElement = makeElement(doc, "html");
  doc.documentElement._root = true;
  doc.head = makeElement(doc, "head");
  doc.body = makeElement(doc, "body");
  doc.documentElement.append(doc.head, doc.body);
  doc.querySelector = (s) => doc.documentElement.querySelector(s);
  doc.querySelectorAll = (s) => doc.documentElement.querySelectorAll(s);
  doc.getElementById = (id) => doc.documentElement.querySelectorAll("*").find((e) => e.id === id) || null;
  return doc;
}

function makeText(text) {
  return { nodeType: 3, data: text, parentNode: null, get textContent() { return this.data; }, set textContent(v) { this.data = String(v); } };
}

/** "[data-x]", "[data-x=\"v\"]", "tag", "#id", "tag[data-x]", and comma lists of those. */
function matches(el, selector) {
  return String(selector)
    .split(",")
    .map((s) => s.trim())
    .some((one) => {
      const m = /^([a-z0-9-]*)(#[\w-]+)?((?:\[[^\]]+\])*)$/i.exec(one);
      if (!m) throw new Error(`fake DOM: selector ${JSON.stringify(one)} is not supported — extend tests/hud/session.test.mjs`);
      const [, tag, id, attrs] = m;
      if (tag && tag !== "*" && el.localName !== tag.toLowerCase()) return false;
      if (id && el.id !== id.slice(1)) return false;
      for (const a of attrs.match(/\[[^\]]+\]/g) || []) {
        const am = /^\[([\w-]+)(?:="?([^"\]]*)"?)?\]$/.exec(a);
        if (!am) throw new Error(`fake DOM: attribute selector ${a} is not supported`);
        if (!el.hasAttribute(am[1])) return false;
        if (am[2] !== undefined && el.getAttribute(am[1]) !== am[2]) return false;
      }
      return true;
    });
}

const kebab = (key) => "data-" + key.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase());

function makeElement(doc, tag) {
  const attrs = new Map();
  const kids = [];
  const listeners = [];
  const expando = new Map();
  let proxy = null;
  const descendants = () => kids.filter((k) => k.nodeType === 1).flatMap((k) => [k, ...k.querySelectorAll("*")]);
  const adopt = (node) => {
    if (typeof node === "string" || typeof node === "number") return [makeText(String(node))];
    if (node && node.localName === "#fragment") {
      const moved = [...node.childNodes];
      node.childNodes.length = 0;
      return moved;
    }
    if (node && node.parentNode && node.parentNode !== proxy) node.parentNode.removeChild(node);
    else if (node && node.parentNode === proxy) kids.splice(kids.indexOf(node), 1);
    return [node];
  };
  const dataset = new Proxy({}, {
    get: (_t, k) => (typeof k === "string" && attrs.has(kebab(k)) ? attrs.get(kebab(k)) : undefined),
    set: (_t, k, v) => (attrs.set(kebab(k), String(v)), true),
    deleteProperty: (_t, k) => (attrs.delete(kebab(k)), true),
    has: (_t, k) => attrs.has(kebab(k)),
  });
  const classes = () => (attrs.get("class") || "").split(/\s+/).filter(Boolean);
  const base = {
    nodeType: 1,
    localName: tag,
    tagName: tag.toUpperCase(),
    nodeName: tag.toUpperCase(),
    ownerDocument: doc,
    parentNode: null,
    childNodes: kids,
    style: { setProperty() {}, removeProperty() {} },
    dataset,
    _root: false,
    _listeners: listeners,
    get parentElement() { return this.parentNode; },
    get children() { return kids.filter((k) => k.nodeType === 1); },
    get firstElementChild() { return this.children[0] || null; },
    get firstChild() { return kids[0] || null; },
    get isConnected() { for (let n = proxy; n; n = n.parentNode) if (n._root) return true; return false; },
    get id() { return attrs.get("id") || ""; },
    set id(v) { attrs.set("id", String(v)); },
    get className() { return attrs.get("class") || ""; },
    set className(v) { attrs.set("class", String(v)); },
    get title() { return attrs.get("title") || ""; },
    set title(v) { attrs.set("title", String(v)); },
    get type() { return attrs.get("type") || (tag === "button" ? "submit" : ""); },
    set type(v) { attrs.set("type", String(v)); },
    get hidden() { return attrs.has("hidden"); },
    set hidden(v) { if (v) attrs.set("hidden", ""); else attrs.delete("hidden"); },
    get disabled() { return attrs.has("disabled"); },
    set disabled(v) { if (v) attrs.set("disabled", ""); else attrs.delete("disabled"); },
    get textContent() { return kids.map((k) => k.textContent).join(""); },
    set textContent(v) {
      for (const k of kids) k.parentNode = null;
      kids.length = 0;
      const s = v === null || v === undefined ? "" : String(v);
      if (s) proxy.appendChild(makeText(s));
    },
    get innerText() { return this.textContent; },
    set innerHTML(_v) { throw new Error("fake DOM: innerHTML — the panel builds its DOM with createElement and textContent"); },
    get innerHTML() { throw new Error("fake DOM: innerHTML is not read in this panel"); },
    insertAdjacentHTML() { throw new Error("fake DOM: insertAdjacentHTML — text is text"); },
    classList: {
      add: (...n) => attrs.set("class", [...new Set([...classes(), ...n])].join(" ")),
      remove: (...n) => attrs.set("class", classes().filter((c) => !n.includes(c)).join(" ")),
      contains: (c) => classes().includes(c),
      toggle: (c, force) => {
        const on = force === undefined ? !classes().includes(c) : Boolean(force);
        if (on) proxy.classList.add(c);
        else proxy.classList.remove(c);
        return on;
      },
    },
    setAttribute: (n, v) => attrs.set(String(n).toLowerCase(), String(v)),
    getAttribute: (n) => (attrs.has(String(n).toLowerCase()) ? attrs.get(String(n).toLowerCase()) : null),
    hasAttribute: (n) => attrs.has(String(n).toLowerCase()),
    removeAttribute: (n) => attrs.delete(String(n).toLowerCase()),
    toggleAttribute: (n, force) => {
      const on = force === undefined ? !attrs.has(n) : Boolean(force);
      if (on) attrs.set(n, "");
      else attrs.delete(n);
      return on;
    },
    appendChild: (node) => {
      for (const n of adopt(node)) {
        n.parentNode = proxy;
        kids.push(n);
      }
      return node;
    },
    append: (...nodes) => nodes.forEach((n) => proxy.appendChild(n)),
    prepend: (...nodes) => {
      const moved = nodes.flatMap(adopt);
      for (const n of moved) n.parentNode = proxy;
      kids.unshift(...moved);
    },
    insertBefore: (node, ref) => {
      if (!ref) return proxy.appendChild(node);
      const moved = adopt(node);
      for (const n of moved) n.parentNode = proxy;
      kids.splice(Math.max(0, kids.indexOf(ref)), 0, ...moved);
      return node;
    },
    removeChild: (node) => {
      const at = kids.indexOf(node);
      if (at >= 0) kids.splice(at, 1);
      node.parentNode = null;
      return node;
    },
    replaceChildren: (...nodes) => {
      for (const k of kids) k.parentNode = null;
      kids.length = 0;
      proxy.append(...nodes);
    },
    remove: () => proxy.parentNode && proxy.parentNode.removeChild(proxy),
    before: (...nodes) => nodes.forEach((n) => proxy.parentNode && proxy.parentNode.insertBefore(n, proxy)),
    after: (...nodes) => {
      const p = proxy.parentNode;
      if (!p) return;
      const at = p.childNodes.indexOf(proxy);
      const next = p.childNodes[at + 1] || null;
      nodes.forEach((n) => p.insertBefore(n, next));
    },
    contains: (node) => { for (let n = node; n; n = n.parentNode) if (n === proxy) return true; return false; },
    closest: (sel) => { for (let n = proxy; n && n.nodeType === 1; n = n.parentNode) if (matches(n, sel)) return n; return null; },
    matches: (sel) => matches(proxy, sel),
    querySelectorAll: (sel) => descendants().filter((e) => sel === "*" || matches(e, sel)),
    querySelector: (sel) => descendants().find((e) => sel === "*" || matches(e, sel)) || null,
    addEventListener: (type, fn) => listeners.push({ type, fn }),
    removeEventListener: (type, fn) => {
      const at = listeners.findIndex((l) => l.type === type && l.fn === fn);
      if (at >= 0) listeners.splice(at, 1);
    },
    focus: () => { doc.activeElement = proxy; },
    blur: () => { if (doc.activeElement === proxy) doc.activeElement = null; },
    click: () => proxy._fire("click"),
    dispatchEvent: (event) => {
      fireOn(proxy, event);
      return !event.defaultPrevented;
    },
    /** A trusted-looking event that bubbles to the document, as a user's would. */
    _fire(type, init = {}) {
      if (type === "click" && proxy.disabled) return; // a disabled control runs no click listener
      fireOn(proxy, { type, ...init });
    },
  };
  proxy = new Proxy(base, {
    get(target, key, receiver) {
      if (typeof key === "symbol" || key in target) return Reflect.get(target, key, receiver);
      if (expando.has(key)) return expando.get(key);
      if (key === "then" || key === "toJSON") return undefined;
      if (/^on[a-z]+$/.test(key)) return null;
      throw new Error(`fake DOM: <${tag}>.${String(key)} is not implemented — extend the double in tests/hud/session.test.mjs`);
    },
    set(target, key, value, receiver) {
      if (key in target) return Reflect.set(target, key, value, receiver);
      if (/^on[a-z]+$/.test(key)) {
        listeners.push({ type: key.slice(2), fn: value });
        return true;
      }
      expando.set(key, value);
      return true;
    },
    has(target, key) {
      return key in target || expando.has(key);
    },
  });
  return proxy;
}

/** Run `event` at `el`, then up its ancestors, then on the document (bubbling, no capture). */
function fireOn(el, event) {
  const ev = {
    bubbles: true,
    cancelable: true,
    defaultPrevented: false,
    key: undefined,
    target: el,
    currentTarget: el,
    _stopped: false,
    preventDefault() { this.defaultPrevented = true; },
    stopPropagation() { this._stopped = true; },
    stopImmediatePropagation() { this._stopped = true; },
    ...event,
  };
  ev.target = el;
  for (let n = el; n && !ev._stopped; n = n.parentNode) {
    ev.currentTarget = n;
    for (const l of [...(n._listeners || [])]) if (l.type === ev.type) l.fn.call(n, ev);
  }
  if (!ev._stopped && el.ownerDocument) {
    ev.currentTarget = el.ownerDocument;
    for (const l of [...el.ownerDocument._listeners]) if (l.type === ev.type) l.fn.call(el.ownerDocument, ev);
  }
}

// --- the panel against the double -------------------------------------------------------------

/** A header with an empty session slot, the host main.js passes, and the socket open by default. */
async function mounted({ open = true, sendImpl = null } = {}) {
  const { mount } = await load("session/index.js");
  assert.equal(typeof mount, "function", "D-1: hud/src/session/index.js exports mount(host)");
  const doc = fakeDocument();
  const header = doc.createElement("header");
  const slot = doc.createElement("div");
  slot.setAttribute("data-session-slot", "");
  header.appendChild(slot);
  const main = doc.createElement("main");
  doc.body.append(header, main);
  const sent = [];
  const logged = [];
  const handlers = [];
  const connection = [];
  let isOpen = open;
  const host = {
    document: doc,
    sessionSlot: slot,
    send: (m) => {
      sent.push(m);
      return sendImpl ? sendImpl(m) : Promise.resolve();
    },
    subscribe: (fn) => (handlers.push(fn), () => handlers.splice(handlers.indexOf(fn), 1)),
    onConnection: (fn) => {
      connection.push(fn);
      fn(isOpen);
      return () => connection.splice(connection.indexOf(fn), 1);
    },
    log: (line) => logged.push(String(line)),
  };
  await mount(host);
  const deliver = (m) => handlers.forEach((fn) => fn(m));
  const setOpen = (v) => {
    isOpen = v;
    connection.forEach((fn) => fn(v));
  };
  const one = (sel) => {
    const hits = slot.querySelectorAll(sel);
    assert.equal(hits.length, 1, `D-1: one ${sel} inside the header's session slot; found ${hits.length} (logged ${JSON.stringify(logged)})`);
    return hits[0];
  };
  const resets = () => sent.filter((m) => m && m.type === "astra.session.reset");
  return { doc, slot, header, sent, logged, deliver, setOpen, one, resets };
}

/** Hidden by the attribute or by an ancestor's, up to the slot. */
function shown(el, stop) {
  for (let n = el; n && n !== stop.parentNode; n = n.parentNode) if (n.hidden) return false;
  return true;
}

const squeeze = (s) => String(s).replace(/\s+/g, " ").trim();
const cleared = (resets, extra = {}) => ({ type: "astra.session.cleared", resets, ended_turn_id: null, memory_forgotten: 0, replay: false, ...extra });

// =================================================================================================
// the gate: its own whitelist, one send, the flight
// =================================================================================================

pendingT0162D("T-0162 D-1: SESSION_CLIENT_TYPES is exactly ['astra.session.reset'] and protocol.js CLIENT_TYPES is not widened", async () => {
  const mod = await gateModule();
  assert.deepEqual([...mod.SESSION_CLIENT_TYPES], ["astra.session.reset"], "D-1: the panel declares its own whitelist, one type");
  const { CLIENT_TYPES } = await load("protocol.js");
  // T-0008's statement about T-0008's code; the assist and log panels declare their own (T-0025, T-0043).
  assert.deepEqual([...CLIENT_TYPES], ["astra.hello", "astra.frame"], "D-1: protocol.js CLIENT_TYPES stays T-0008's two types");
});

pendingT0162D("T-0162 D-1: reset() hands exactly one { type: 'astra.session.reset' } to send — no other field — and says so", async () => {
  const { createSessionGate, SESSION_CLIENT_TYPES } = await gateModule();
  const sent = [];
  const gate = createSessionGate({ send: (m) => (sent.push(m), Promise.resolve()) });
  assert.equal(gate.pending(), false, "nothing is in flight before a reset");
  assert.equal(gate.reset(), true, "D-1: reset() reports that it sent");
  assert.equal(sent.length, 1, `D-1: one message; sent ${JSON.stringify(sent)}`);
  assert.deepEqual(sent[0], { type: "astra.session.reset" }, "A-1: AstraSessionReset has no fields, and an unknown field is malformed");
  assert.ok(sent.every((m) => SESSION_CLIENT_TYPES.includes(m.type)), "D-1: everything the gate emitted is in its own whitelist");
  assert.equal(gate.pending(), true, "the reset is in flight until its echo");
});

pendingT0162D("T-0162 D-1 (double-click): a second reset() before the echo sends nothing; after astra.session.cleared the next one sends", async () => {
  const { createSessionGate } = await gateModule();
  const sent = [];
  const gate = createSessionGate({ send: (m) => (sent.push(m), Promise.resolve()) });
  gate.reset();
  assert.equal(gate.reset(), false, "the second reset() reports that it did not send");
  gate.reset();
  assert.equal(sent.length, 1, `a double (or triple) click wipes once; sent ${sent.length}`);
  gate.onCleared({ type: "astra.turn.state", state: "IDLE", turn_id: "t3", thinking_ms: null, replay: false });
  gate.onCleared({ type: "astra.log.state", on: true, path: "/x", records: 1, dropped: 0, bytes: 1, error: null });
  assert.equal(gate.pending(), true, "only astra.session.cleared ends the flight; other messages change nothing");
  assert.equal(gate.reset(), false);
  assert.equal(sent.length, 1);
  // The other side of the guard: once echoed, Restart works again (a stuck gate is a dead button).
  gate.onCleared(cleared(1));
  assert.equal(gate.pending(), false, "the echo ends the flight");
  assert.equal(gate.reset(), true, "a later Restart sends again");
  assert.equal(sent.length, 2);
});

pendingT0162D("T-0162 D-1: a send that rejects or throws ends the flight and never throws into the click", async () => {
  const { createSessionGate } = await gateModule();
  const logged = [];
  let mode = "reject";
  const sent = [];
  const gate = createSessionGate({
    send: (m) => {
      sent.push(m);
      if (mode === "throw") throw new Error("socket is not open");
      return Promise.reject(new Error("socket is not open"));
    },
    log: (l) => logged.push(String(l)),
  });
  gate.reset();
  await flush();
  assert.equal(gate.pending(), false, "ws.js rejects while the socket is closed: nothing is in flight");
  mode = "throw";
  let result;
  assert.doesNotThrow(() => {
    result = gate.reset();
  }, "a throwing send is a log line, never a throw");
  assert.equal(result, false, "a send that threw did not send");
  assert.equal(gate.pending(), false);
  mode = "reject";
  assert.equal(gate.reset(), true, "the gate is usable again after a refused send");
  assert.equal(sent.length, 3);
  assert.ok(logged.length >= 1, "the refusal is logged");
});

// =================================================================================================
// the panel: the button, the two-step confirm, the socket, the keys
// =================================================================================================

pendingT0162D("T-0162 D-1: mount renders a 'Restart' button in the header's slot, the confirm hidden, nothing sent", async () => {
  const p = await mounted();
  const button = p.one("[data-restart-button]");
  assert.equal(button.localName, "button", "D-1: the Restart control is a <button>");
  assert.equal(button.type, "button", "a type=button, so it never submits a form");
  assert.equal(squeeze(button.textContent), "Restart", "D-1: labelled 'Restart'");
  assert.ok(p.header.contains(button), "D-1: a header button");
  const confirm = p.one("[data-restart-confirm]");
  assert.equal(shown(confirm, p.slot), false, "H-T162-3: the confirm is hidden until Restart is pressed");
  assert.deepEqual(p.sent, [], "mounting sends nothing");
});

pendingT0162D("T-0162 D-1: Restart opens the inline confirm and sends nothing; its text names the memory wipe and the browser's start page", async () => {
  const p = await mounted();
  p.one("[data-restart-button]")._fire("click");
  const confirm = p.one("[data-restart-confirm]");
  assert.equal(shown(confirm, p.slot), true, "D-1: the first click opens the confirm");
  assert.deepEqual(p.resets(), [], "D-1 (two steps, R6): the first click sends nothing");
  const text = squeeze(confirm.textContent);
  assert.match(text, /Start a new session\?/, `D-1: the question, verbatim; the confirm reads ${JSON.stringify(text)}`);
  assert.match(text, /clears the conversation/, "D-1: it says the conversation is cleared");
  assert.match(text, /everything Gearvis remembers/, "H-T162-1 / R6: it says the long-term memory is wiped too");
  // Ruling (b), group D reviews 2026-10-27: the wording is now fixed; its key phrases are pinned.
  assert.match(text, /clears the conversation and everything Gearvis remembers/, "ruling (b): conversation and memory");
  assert.match(text, /stops screen sharing/, "ruling (b) / D-3: it says the screen share stops");
  assert.match(text, /Gearvis's own browser, if it is open, back to its start page/, "ruling (b): the agent browser returns to its start page");
  assert.match(text, /Session logs already written stay on disk\./, "ruling (b): the logs on disk are kept");
  const yes = p.one("[data-restart-confirm-yes]");
  const cancel = p.one("[data-restart-cancel]");
  assert.ok(confirm.contains(yes) && confirm.contains(cancel), "both buttons are inside the confirm");
  assert.equal(squeeze(yes.textContent), "Restart", "D-1: the confirm's buttons are **Restart**…");
  assert.equal(squeeze(cancel.textContent), "Cancel", "…and **Cancel**");
});

pendingT0162D("T-0162 D-1: Cancel sends nothing and closes the confirm; Restart → Restart sends exactly one astra.session.reset", async () => {
  const p = await mounted();
  p.one("[data-restart-button]")._fire("click");
  p.one("[data-restart-cancel]")._fire("click");
  assert.deepEqual(p.sent, [], "D-1: Cancel sends nothing");
  assert.equal(shown(p.one("[data-restart-confirm]"), p.slot), false, "Cancel closes the confirm");
  p.one("[data-restart-button]")._fire("click");
  assert.equal(shown(p.one("[data-restart-confirm]"), p.slot), true, "Restart opens it again after a Cancel");
  p.one("[data-restart-confirm-yes]")._fire("click");
  assert.deepEqual(p.sent, [{ type: "astra.session.reset" }], `D-1: one confirmed Restart, one message; sent ${JSON.stringify(p.sent)}`);
});

pendingT0162D("T-0162 D-1 (adversarial): a double-click on the confirm, or on both buttons, sends one reset", async () => {
  const p = await mounted();
  const button = p.one("[data-restart-button]");
  button._fire("click");
  button._fire("click"); // a double-click on Restart is not a confirm
  assert.deepEqual(p.resets(), [], "two clicks on Restart are still only the first step");
  const yes = p.one("[data-restart-confirm-yes]");
  yes._fire("click");
  yes._fire("click");
  yes._fire("click");
  assert.equal(p.resets().length, 1, `D-1: a double-click on the confirm wipes once; sent ${p.resets().length}`);
  // Until the echo, opening the confirm again does not let a second reset out either.
  button._fire("click");
  const again = p.slot.querySelectorAll("[data-restart-confirm-yes]");
  for (const b of again) b._fire("click");
  assert.equal(p.resets().length, 1, "a second confirm before the echo sends nothing");
  // The other side: after the echo, a new confirmed Restart sends.
  p.deliver(cleared(1));
  p.one("[data-restart-button]")._fire("click");
  p.one("[data-restart-confirm-yes]")._fire("click");
  assert.equal(p.resets().length, 2, "after astra.session.cleared, the next confirmed Restart sends (the gate is not stuck)");
});

pendingT0162D("T-0162 D-1: the button is disabled while the socket is closed, and a confirm left open sends nothing then", async () => {
  const p = await mounted({ open: false });
  const button = p.one("[data-restart-button]");
  assert.equal(button.disabled, true, "D-1: disabled while the socket is closed (the page starts before it opens)");
  p.setOpen(true);
  assert.equal(button.disabled, false, "enabled once the socket is open (the other direction)");
  button._fire("click");
  p.setOpen(false);
  assert.equal(p.one("[data-restart-button]").disabled, true, "disabled again when the socket closes");
  for (const b of p.slot.querySelectorAll("[data-restart-confirm-yes]")) b._fire("click");
  assert.deepEqual(p.resets(), [], "a confirm still open when the socket closed sends nothing");
  p.setOpen(true);
  assert.equal(p.one("[data-restart-button]").disabled, false, "a reconnect enables it again");
});

pendingT0162D("T-0162 D-4: Restart never fires on a key press — r, R, Enter, Space and Escape anywhere send nothing and open nothing", async () => {
  const p = await mounted();
  const button = p.one("[data-restart-button]");
  const keys = ["r", "R", "Enter", " ", "Escape", "Delete", "Backspace"];
  for (const key of keys) {
    p.doc.body._fire("keydown", { key });
    p.doc.body._fire("keyup", { key });
    button._fire("keydown", { key });
    button._fire("keyup", { key });
  }
  assert.deepEqual(p.sent, [], "D-4: no key press sends astra.session.reset");
  assert.equal(shown(p.one("[data-restart-confirm]"), p.slot), false, "D-4: no key press opens the confirm");
  // With the confirm open, a key press does not confirm it either.
  button._fire("click");
  const yes = p.one("[data-restart-confirm-yes]");
  for (const key of keys) {
    p.doc.body._fire("keydown", { key });
    yes._fire("keydown", { key });
  }
  assert.deepEqual(p.resets(), [], "D-4: no key press confirms the reset");
});

pendingT0162D("T-0162 D-1: a reset from another tab closes this page's open confirm, and its held Yes then sends nothing", async () => {
  // Reviewer B (2026-10-27): the earlier row clicked Yes first, so the confirm was already closed by
  // the click and the message proved nothing. Here this page never confirms: another tab's reset
  // (a new resets value) answers the question for it.
  const p = await mounted();
  p.one("[data-restart-button]")._fire("click");
  const yes = p.one("[data-restart-confirm-yes]");
  assert.equal(shown(p.one("[data-restart-confirm]"), p.slot), true, "premise: the confirm is open");
  p.deliver(cleared(1, { ended_turn_id: "t4", memory_forgotten: 2 }));
  assert.equal(shown(p.one("[data-restart-confirm]"), p.slot), false, "the session was reset elsewhere: the confirm closes");
  yes._fire("click"); // the node is still held (a stale click, or a click already queued)
  assert.deepEqual(p.resets(), [], "a Yes left over from before the other tab's reset wipes nothing");
  assert.equal(p.one("[data-restart-button]").disabled, false, "Restart is usable for the next session");
});

pendingT0162D("T-0162 D-2 (reviewer A T-4): the page's reset tracker accepts each *different* resets value — a restarted CLI's 1 after 2 clears", async () => {
  // Ruling (a): main.js owns one tracker (createClearTracker in session/gate.js) and panels seed
  // theirs from it. "New" is `!==` the last accepted value, not `>`: a CLI restarted behind a page
  // that saw resets 2 counts from 1 again, and that is a new session too.
  const { createClearTracker } = await gateModule();
  assert.equal(typeof createClearTracker, "function", "ruling (a): session/gate.js exports createClearTracker(seed)");
  const t = createClearTracker();
  const seen = [cleared(1), cleared(1, { replay: true }), cleared(2), cleared(2, { replay: true }), cleared(1, { replay: true }), cleared(1)].map((m) => t.accept(m));
  assert.deepEqual(seen, [true, false, true, false, true, false], "accept: once per change of value, in either direction");
  assert.equal(t.accept({ type: "astra.turn.state", state: "IDLE", turn_id: "t1", thinking_ms: null, replay: false }), false, "other types are never accepted");
  // Seeded (a panel mounted after the hello's replay of reset 3): the same value is not new; 4 is.
  const seeded = createClearTracker(3);
  assert.deepEqual([seeded.accept(cleared(3, { replay: true })), seeded.accept(cleared(4))], [false, true], "a seeded tracker takes the seed as already accepted");
});

pendingT0162D("T-0162 D-1: a socket that closes with a reset in flight abandons it — after the reconnect Restart works again", async () => {
  const p = await mounted();
  p.one("[data-restart-button]")._fire("click");
  p.one("[data-restart-confirm-yes]")._fire("click");
  assert.equal(p.resets().length, 1, "premise: one reset is in flight");
  assert.equal(p.one("[data-restart-button]").disabled, true, "premise: Restart is disabled while the reset waits for its echo");
  p.setOpen(false); // the socket dropped before astra.session.cleared arrived
  p.setOpen(true);
  assert.equal(p.one("[data-restart-button]").disabled, false, "D-1: after the reconnect the abandoned reset no longer blocks Restart");
  p.one("[data-restart-button]")._fire("click");
  p.one("[data-restart-confirm-yes]")._fire("click");
  assert.equal(p.resets().length, 2, "…and the next confirmed Restart sends one reset");
});

pendingT0162D("T-0162 D-1: a host with no document or no send never throws, and sends nothing", async () => {
  const { mount } = await load("session/index.js");
  const sent = [];
  await assert.doesNotReject(async () => mount({ log: () => {} }), "no document: render nothing");
  await assert.doesNotReject(async () => mount(undefined), "no host at all");
  const doc = fakeDocument();
  const slot = doc.createElement("div");
  doc.body.appendChild(slot);
  await assert.doesNotReject(async () => mount({ document: doc, sessionSlot: slot, subscribe: () => () => {}, onConnection: (fn) => fn(true), log: () => {} }));
  for (const b of slot.querySelectorAll("button")) b._fire("click");
  for (const b of slot.querySelectorAll("button")) b._fire("click");
  assert.deepEqual(sent, []);
  // and the panel did render its button (anti-vacuity for the clicks above)
  assert.equal(slot.querySelectorAll("[data-restart-button]").length, 1, "the button is still rendered without send");
});
