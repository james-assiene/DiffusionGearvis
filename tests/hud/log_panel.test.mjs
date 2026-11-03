// T-0043 group B — the session-log panel: hud/src/log/gate.js and hud/src/log/index.js.
//
// Criterion 19 of docs/tasks/T-0043.md, and the failure it exists for, in the task's own words:
// "a page that claims a session is being logged when it is not". Someone reads a ticked box,
// speaks freely believing there is a record, and there is none. So three states are pinned here,
// not two:
//
//   unknown   no astra.log.state has arrived yet -> never a ticked box, never the word "on"
//   waiting   a click went out whose echo has not come back -> "waiting for the server"
//   settled   the server's echo, and *only* the server's echo, decides what the box shows
//
// The server's default is ON (criterion 15), which is exactly why the page may not guess it: a
// page that pre-ticked the box would be right most of the time and catastrophically wrong in the
// one case that matters (--no-log, or a log that disabled itself on an OSError — criterion 4).
//
// This file *skips itself* while the panel is not implemented (the convention of
// tests/hud/ask.test.mjs; node tests cannot carry pytest's `pending` marker and
// tests/unit/test_hud_js.py runs `node --test tests/hud` inside the fast suite). The sentinel is
// the two exports below rather than the files' existence, because the stubs exist already.
//
// API this pins (docs/tasks/T-0043.md leaves the JS names to the implementer; these are them):
//
//   hud/src/log/gate.js — no DOM at module top level, importable in node
//     LOG_CLIENT_TYPES = ["astra.log.toggle"]   the panel's own egress whitelist (criterion 19)
//     LOG_STATE_TYPE   = "astra.log.state"      the one server->client type it listens to
//     createLogGate({ send, log }) -> {
//       onState(message)   feed a decoded astra.log.state; nothing else confirms anything
//       known() -> bool    false until the first astra.log.state has arrived
//       on() -> bool       the server's flag (false while unknown — see `known()`)
//       pending() -> bool  a toggle went out whose echo has not arrived
//       path() -> string|null    the server's path string, verbatim; null while unknown
//       stats() -> { records, dropped, bytes, error }
//       toggle(on) -> bool true iff exactly one astra.log.toggle was handed to send
//     }
//
//   hud/src/log/index.js
//     mount(host)  — host is { shown, document, send, subscribe, log }, T-0025's seam
//     The panel inserts its own element and its own <style>; hud/index.html, hud/styles.css,
//     hud/src/main.js and hud/src/state.js are not touched (criterion 19).
//     DOM contract, as the integration row and this file address it:
//       [data-log-panel]                  the panel element
//       [data-log-panel][data-log-state]  "on" | "off" | "unknown"
//       [data-log-toggle]                 the checkbox (input, .checked)
//       [data-log-path]                   textContent: the server's path string, verbatim
//       [data-log-records] [data-log-dropped]   textContent: the counters
//       [data-log-pending]                textContent: "waiting for the server" while unechoed
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);

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

// Origin proof (.claude/rules/tests.md, 2026-09-16): the editable install and the working
// directory are two separate ways for a run started inside a mutation copy to import the tree it
// meant to leave behind. Printed once, from inside this test process, so any mutation result
// taken from this file says which tree it measured.
console.log(
  `# log panel origin: repo=${fileURLToPath(REPO)} modules=${new URL("log/", srcRoot()).href}`,
);

async function tryLoad(name) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch {
    return null;
  }
}

// The sentinel: the modules exist (they are committed as stubs so this file imports) but export
// nothing until group B lands.
const GATE = await tryLoad("log/gate.js");
const INDEX = await tryLoad("log/index.js");
const INSTALLED =
  GATE !== null &&
  INDEX !== null &&
  typeof GATE.createLogGate === "function" &&
  typeof INDEX.mount === "function";
const opts = INSTALLED
  ? {}
  : {
      skip:
        "T-0043 group B is not implemented yet: hud/src/log/gate.js exports no createLogGate " +
        "and/or hud/src/log/index.js exports no mount. This file is the acceptance suite of " +
        "docs/tasks/T-0043.md criterion 19; it runs the moment those land. (Origin of the two " +
        `modules this run looked at: ${new URL("log/gate.js", srcRoot()).href})`,
    };

async function load(name) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch (err) {
    assert.fail(
      `docs/tasks/T-0043.md §3 (group B) owns hud/src/${name}; it must be importable in node as ` +
        `an ES module with no DOM at module top level (the panel touches the DOM inside ` +
        `mount(host) only). node could not import it: ${err.message}`,
    );
  }
}

/** A recorder for the host's `send`: what left the page, and nothing more. */
function recorder({ rejects = false } = {}) {
  const sent = [];
  return {
    sent,
    ofType: (t) => sent.filter((m) => m && m.type === t),
    send(message) {
      sent.push(message);
      return rejects ? Promise.reject(new Error("socket is not open")) : Promise.resolve();
    },
  };
}

/** One decoded `astra.log.state`, as `contracts/protocol.AstraLogState` encodes (criterion 16). */
function logState({
  on = true,
  path = "/home/u/astra/logs/sessions/20260916-041500-12345.jsonl",
  records = 0,
  dropped = 0,
  bytes = 0,
  error = null,
} = {}) {
  return { type: "astra.log.state", on, path, records, dropped, bytes, error };
}

// --- the fake DOM ------------------------------------------------------------------------------
//
// Closed by *delegation*, not by enumeration (.claude/rules/tests.md, 2026-09-16): every element
// is wrapped in a Proxy, so a member the panel reaches for that this double does not implement is
// a named assertion naming the member — never an `undefined is not a function` inside the test,
// and never a silent no-op that would make a row pass because the rendering never happened.

const ELEMENT_MEMBERS = new Set([
  "tagName", "type", "childNodes", "children", "attributes", "dataset", "listeners", "value",
  "checked", "indeterminate", "disabled", "hidden", "textContent", "className", "id", "title",
  "parentElement", "style", "setAttribute", "getAttribute", "hasAttribute", "removeAttribute",
  "appendChild", "append", "replaceChildren", "addEventListener", "dispatchEvent", "matches",
  "querySelector", "querySelectorAll", "remove", "insertBefore", "contains", "then",
]);

function makeDom() {
  const missing = [];
  class El {
    constructor(tag) {
      this.tagName = String(tag).toUpperCase();
      this.type = "";
      this.childNodes = [];
      this.attributes = new Map();
      this.dataset = {};
      this.listeners = new Map();
      this.value = "";
      this.checked = false;
      this.indeterminate = false;
      this.disabled = false;
      this.hidden = false;
      this.textContent = "";
      this.className = "";
      this.id = "";
      this.title = "";
      this.style = {};
      this.parentElement = null;
    }
    get children() {
      return this.childNodes;
    }
    setAttribute(name, value) {
      this.attributes.set(name, String(value));
      if (name === "id") this.id = String(value);
    }
    getAttribute(name) {
      return this.attributes.has(name) ? this.attributes.get(name) : null;
    }
    hasAttribute(name) {
      return this.attributes.has(name);
    }
    removeAttribute(name) {
      this.attributes.delete(name);
    }
    appendChild(child) {
      child.parentElement = this;
      this.childNodes.push(child);
      return child;
    }
    append(...nodes) {
      for (const n of nodes) if (n && typeof n === "object") this.appendChild(n);
        else this.appendChild(Object.assign(wrap(new El("#text")), { textContent: String(n) }));
    }
    replaceChildren(...nodes) {
      this.childNodes = [];
      this.append(...nodes);
    }
    insertBefore(child) {
      return this.appendChild(child);
    }
    remove() {
      if (!this.parentElement) return;
      const siblings = this.parentElement.childNodes;
      const at = siblings.indexOf(this);
      if (at >= 0) siblings.splice(at, 1);
    }
    contains(node) {
      if (node === this) return true;
      return this.childNodes.some((c) => c.contains && c.contains(node));
    }
    addEventListener(type, handler) {
      if (!this.listeners.has(type)) this.listeners.set(type, []);
      this.listeners.get(type).push(handler);
    }
    dispatchEvent(event) {
      event.target = this;
      for (const handler of this.listeners.get(event.type) || []) handler(event);
      return true;
    }
    matches(selector) {
      if (selector.startsWith("#")) return this.id === selector.slice(1);
      if (selector.startsWith("[") && selector.endsWith("]")) {
        const [name, value] = selector.slice(1, -1).split("=");
        if (value === undefined) return this.attributes.has(name);
        return this.attributes.get(name) === value.replace(/^["']|["']$/g, "");
      }
      return this.tagName === selector.toUpperCase();
    }
    querySelectorAll(selector) {
      const out = [];
      for (const child of this.childNodes) {
        if (child.matches && child.matches(selector)) out.push(child);
        if (child.querySelectorAll) out.push(...child.querySelectorAll(selector));
      }
      return out;
    }
    querySelector(selector) {
      return this.querySelectorAll(selector)[0] || null;
    }
    /** Everything this element and its descendants would show a human. */
    visibleText() {
      const own = this.childNodes.length ? "" : this.textContent;
      return [own, ...this.childNodes.map((c) => (c.visibleText ? c.visibleText() : ""))]
        .filter(Boolean)
        .join(" ");
    }
  }
  const wrap = (el) =>
    new Proxy(el, {
      get(target, prop, receiver) {
        if (typeof prop === "symbol" || prop in target || ELEMENT_MEMBERS.has(prop))
          return Reflect.get(target, prop, receiver);
        if (prop === "visibleText") return Reflect.get(target, prop, receiver);
        missing.push(`${target.tagName}.${String(prop)}`);
        assert.fail(
          `the log panel reached for ${target.tagName}.${String(prop)}, which this test's DOM ` +
            "double does not implement. Extend the double (tests/hud/log_panel.test.mjs) rather " +
            "than working around it — an unimplemented member must be a named failure, never an " +
            "undefined that makes a rendering row pass because nothing was rendered.",
        );
      },
    });
  const create = (tag) => wrap(new El(tag));
  const doc = {
    createElement: create,
    createTextNode: (text) => Object.assign(create("#text"), { textContent: String(text) }),
    listeners: new Map(),
    addEventListener(type, handler) {
      if (!this.listeners.has(type)) this.listeners.set(type, []);
      this.listeners.get(type).push(handler);
    },
  };
  doc.head = create("head");
  doc.body = create("body");
  doc.documentElement = create("html");
  doc.documentElement.appendChild(doc.head);
  doc.documentElement.appendChild(doc.body);
  const main = create("main");
  doc.body.appendChild(main);
  doc.querySelector = (s) => doc.documentElement.querySelector(s);
  doc.querySelectorAll = (s) => doc.documentElement.querySelectorAll(s);
  doc.getElementById = (id) => doc.documentElement.querySelector(`#${id}`);
  return { doc, main, missing };
}

/** The dangerous globals: reaching for one is a named failure, not a real network call. */
function trapGlobals() {
  const saved = {};
  for (const name of ["fetch", "WebSocket", "XMLHttpRequest", "document", "localStorage"]) {
    saved[name] = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, {
      configurable: true,
      get() {
        assert.fail(
          `the log panel reached for globalThis.${name}: criterion 19 says it consumes the panel ` +
            "host (document/send/subscribe) and criterion 25 says it opens no transport of its own",
        );
      },
    });
  }
  return () => {
    for (const [name, descriptor] of Object.entries(saved)) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  };
}

async function mounted(extra = {}, recorderOptions = {}) {
  const index = await load("log/index.js");
  const dom = makeDom();
  const rec = recorder(recorderOptions);
  const handlers = [];
  const host = {
    document: dom.doc,
    send: rec.send,
    subscribe: (fn) => {
      handlers.push(fn);
      return () => {};
    },
    log: () => {},
    ...extra,
  };
  const restore = trapGlobals();
  try {
    await index.mount(host);
  } finally {
    restore();
  }
  const panel = () => dom.doc.querySelector("[data-log-panel]");
  return {
    ...dom,
    rec,
    host,
    handlers,
    panel,
    deliver: (m) => handlers.forEach((fn) => fn(m)),
    at: (selector) => {
      const node = dom.doc.querySelector(selector);
      assert.ok(
        node,
        `criterion 19: the panel must render ${selector} — the rows below address the checkbox, ` +
          "the path and the counters by these attributes, and so does " +
          "tests/integration/test_hud_session_log.py",
      );
      return node;
    },
  };
}

const CLAIMS_ON = /\bon\b|\btrue\b|\benabled\b|\brecording\b|\bis being logged\b/i;

// --- criterion 19: the panel's own egress -------------------------------------------------------

test("criterion 19: LOG_CLIENT_TYPES is exactly astra.log.toggle", opts, async () => {
  const { LOG_CLIENT_TYPES } = await load("log/gate.js");
  assert.deepEqual(
    [...LOG_CLIENT_TYPES],
    ["astra.log.toggle"],
    "criterion 19: the panel declares its own client->server whitelist and emits nothing else; " +
      "protocol.js CLIENT_TYPES stays T-0008's two entries (the per-directory scan in " +
      "tests/hud/protocol.test.mjs is the other half of the oracle)",
  );
});

test("criterion 19: mounting emits nothing at all", opts, async () => {
  const page = await mounted();
  assert.deepEqual(
    page.rec.sent,
    [],
    "criterion 19: the page never asks for a state and never announces one — the server sends " +
      "astra.log.state as the third message of its hello reply (criterion 18), unprompted",
  );
});

// --- criterion 19, the headline: before any state has arrived ----------------------------------

test("criterion 19 ADVERSARIAL: before any astra.log.state the panel renders unknown, never on", opts, async () => {
  // The failure this row exists for: the server's default is ON, so a page that guessed would be
  // right most of the time — and wrong exactly when the session is *not* being recorded, which is
  // when someone speaks freely believing there is a record and there is none.
  const page = await mounted();
  const panel = page.panel();
  assert.ok(panel, "criterion 19: the panel renders as soon as it is mounted, state or no state");
  assert.equal(
    panel.getAttribute("data-log-state"),
    "unknown",
    "criterion 19: with no echo yet the state is unknown — not 'on', not 'off'",
  );
  const box = page.at("[data-log-toggle]");
  assert.equal(
    box.checked,
    false,
    "criterion 19: …and the box is not ticked. A ticked box is the page claiming a session is " +
      "being recorded on no evidence whatever",
  );
  const text = panel.visibleText();
  assert.equal(
    CLAIMS_ON.test(text),
    false,
    `criterion 19: …and nothing rendered claims it is on. The panel shows: ${JSON.stringify(text)}`,
  );
  assert.equal(
    page.at("[data-log-path]").textContent,
    "",
    "criterion 19: there is no path to show before the server has named one, and an invented " +
      "one would be read as the file to go and look at",
  );
});

test("criterion 19: the gate itself reports unknown before the first state", opts, async () => {
  const { createLogGate } = await load("log/gate.js");
  const rec = recorder();
  const gate = createLogGate({ send: rec.send, log: () => {} });
  assert.equal(gate.known(), false, "criterion 19: nothing is known before the server says so");
  assert.equal(gate.on(), false, "criterion 19: and 'unknown' never reads as on");
  assert.equal(gate.path(), null, "criterion 19: no path is known either");
  assert.equal(gate.pending(), false, "nothing has been clicked yet");
});

// --- criterion 19: the echo decides, the click never does --------------------------------------

test("criterion 19: a click sends a toggle and does NOT tick the box", opts, async () => {
  const page = await mounted();
  page.deliver(logState({ on: false, path: null }));
  const box = page.at("[data-log-toggle]");
  assert.equal(box.checked, false, "premise: the server says logging is off");

  box.checked = true; // the browser flips it on click, before any handler runs
  box.dispatchEvent({ type: "change" });

  assert.deepEqual(
    page.rec.ofType("astra.log.toggle"),
    [{ type: "astra.log.toggle", on: true }],
    "criterion 19: the click puts exactly one astra.log.toggle on the wire (T-0025 precedent)",
  );
  assert.equal(
    page.at("[data-log-toggle]").checked,
    false,
    "criterion 19: …and the box goes back to the server's last word. The checked state comes " +
      "from the echo and never from the click: a box that ticks itself says a session is being " +
      "recorded before anything has agreed to record it",
  );
  assert.equal(
    page.panel().getAttribute("data-log-state"),
    "off",
    "criterion 19: the rendered state is still the server's",
  );
  assert.match(
    page.at("[data-log-pending]").textContent,
    /waiting for the server/i,
    "criterion 19: a clicked-but-unechoed box says 'waiting for the server'",
  );
});

test("criterion 19: the echo — and only the echo — settles the box", opts, async () => {
  const page = await mounted();
  page.deliver(logState({ on: false, path: null }));
  const box = page.at("[data-log-toggle]");
  box.checked = true;
  box.dispatchEvent({ type: "change" });
  assert.match(page.at("[data-log-pending]").textContent, /waiting for the server/i);

  page.deliver(logState({ on: true, path: "/tmp/sessions/20260916-041500-1.jsonl", records: 7 }));
  assert.equal(
    page.at("[data-log-toggle]").checked,
    true,
    "criterion 19: the server confirmed it, so now the box is ticked",
  );
  assert.equal(page.panel().getAttribute("data-log-state"), "on");
  assert.equal(
    page.at("[data-log-pending]").textContent.trim(),
    "",
    "criterion 19: …and it is no longer waiting for anything",
  );
});

test("criterion 19 ADVERSARIAL: a click the server refuses leaves the box where the server put it", opts, async () => {
  // The log can disable itself (criterion 4: an OSError takes it down and publishes the state
  // with `error` set). The page must render *that*, not the click that asked for the opposite.
  const page = await mounted();
  page.deliver(logState({ on: false, path: null }));
  const box = page.at("[data-log-toggle]");
  box.checked = true;
  box.dispatchEvent({ type: "change" });
  page.deliver(logState({ on: false, path: null, error: "cannot open /x/y.jsonl: is a directory" }));

  assert.equal(
    page.at("[data-log-toggle]").checked,
    false,
    "criterion 19: the user asked for logging, the server could not, and the box shows what is " +
      "true — this is the one case where a page that echoed the click would lie outright",
  );
  assert.equal(page.panel().getAttribute("data-log-state"), "off");
  assert.equal(
    page.at("[data-log-pending]").textContent.trim(),
    "",
    "criterion 19: the echo arrived, so nothing is waiting any more",
  );
  assert.ok(
    page.panel().visibleText().includes("cannot open /x/y.jsonl: is a directory"),
    "criterion 4/19: the reason logging is off reaches the person looking at the page; it shows " +
      JSON.stringify(page.panel().visibleText()),
  );
});

// --- criterion 19: the server's strings and counters, verbatim ----------------------------------

test("criterion 19: the path is the server's string, character for character", opts, async () => {
  const page = await mounted();
  const path = "/home/user/demos/local_astra/logs/sessions/20260916-041500-12345.jsonl";
  page.deliver(logState({ on: true, path, records: 412, dropped: 3, bytes: 190233 }));
  assert.equal(
    page.at("[data-log-path]").textContent,
    path,
    "criterion 19: the panel renders the server's path verbatim — it is the file the human is " +
      "about to open, so a shortened or prettified one is the wrong file",
  );
  assert.equal(page.at("[data-log-records]").textContent, "412", "criterion 19: the counters too");
  assert.equal(
    page.at("[data-log-dropped]").textContent,
    "3",
    "criterion 19: …including the drops, which are how a reader knows the log is incomplete",
  );
});

test("criterion 19: the gate ignores every message that is not an astra.log.state", opts, async () => {
  const { createLogGate } = await load("log/gate.js");
  const gate = createLogGate({ send: recorder().send, log: () => {} });
  for (const message of [
    { type: "astra.assist.state", flags: {}, badge: "pixels" },
    { type: "astra.turn.state", state: "IDLE", turn_id: "t1" },
    { type: "astra.log.toggle", on: true },
    null,
    undefined,
    "astra.log.state",
    { on: true },
  ]) {
    assert.doesNotThrow(() => gate.onState(message), `onState(${JSON.stringify(message)}) threw`);
    assert.equal(
      gate.known(),
      false,
      `criterion 19: ${JSON.stringify(message)} is not the server's log state and must not ` +
        "settle the checkbox — least of all the page's own toggle echoed back at it",
    );
  }
  gate.onState(logState({ on: true }));
  assert.equal(gate.known(), true, "…and the real thing does settle it");
});

// --- criterion 19: a host that cannot render or cannot send --------------------------------------

test("criterion 19: mount({}) renders nothing, sends nothing and throws nothing", opts, async () => {
  const index = await load("log/index.js");
  const restore = trapGlobals();
  try {
    await assert.doesNotReject(
      async () => index.mount({}),
      "criterion 19 / T-0008 criterion 10: a host with no document, no send and no subscribe " +
        "degrades to nothing rendered, leaving every other panel and the socket working",
    );
  } finally {
    restore();
  }
});

test("criterion 19: a host with no send renders but never emits", opts, async () => {
  const page = await mounted({ send: undefined });
  page.deliver(logState({ on: true }));
  const box = page.at("[data-log-toggle]");
  box.checked = false;
  assert.doesNotThrow(
    () => box.dispatchEvent({ type: "change" }),
    "criterion 19: a click with no socket is a no-op, not an exception in the click handler",
  );
  assert.deepEqual(page.rec.sent, [], "criterion 19: nothing left the page — there is no socket");
});

test("criterion 19: a send that rejects does not throw into the click handler", opts, async () => {
  const page = await mounted({}, { rejects: true });
  page.deliver(logState({ on: true }));
  const box = page.at("[data-log-toggle]");
  box.checked = false;
  assert.doesNotThrow(
    () => box.dispatchEvent({ type: "change" }),
    "ws.js rejects while the socket is closed; the panel turns that into a log line, never an " +
      "unhandled rejection in the user's click",
  );
  assert.equal(page.rec.ofType("astra.log.toggle").length, 1, "…the attempt was still made once");
});

test("criterion 19: a panel with no document survives an inbound state", opts, async () => {
  const index = await load("log/index.js");
  const rec = recorder();
  const handlers = [];
  const restore = trapGlobals();
  try {
    await index.mount({
      send: rec.send,
      subscribe: (fn) => {
        handlers.push(fn);
        return () => {};
      },
      log: () => {},
    });
    for (const fn of handlers)
      assert.doesNotThrow(
        () => fn(logState({ on: true })),
        "criterion 19: a host that predates this seam must survive the message, not take the " +
          "socket down with it",
      );
  } finally {
    restore();
  }
  assert.deepEqual(rec.sent, [], "criterion 19: and it still sends nothing of its own");
});

// --- criterion 19: the panel owns its own element and its own style ----------------------------

test("criterion 19: the panel inserts its own element and its own <style>", opts, async () => {
  const page = await mounted();
  assert.ok(
    page.panel(),
    "criterion 19: hud/index.html and hud/styles.css are not touched, so the panel must insert " +
      "its own element (docs/tasks/T-0043.md criterion 19: 'It inserts its own element and its " +
      "own <style>')",
  );
  const styles = page.doc.querySelectorAll("style");
  assert.ok(
    styles.length >= 1,
    "criterion 19: …and its own <style>; hud/styles.css belongs to T-0008 and is at zero diff",
  );
});
