// T-0039 group B — the ask panel: hud/src/ask/form.js and hud/src/ask/index.js.
//
// Criteria 10-13 of docs/tasks/T-0039.md. The panel is the only thing on the page that may emit
// `astra.text`, and every refusal that matters (empty, whitespace, over the cap, socket closed,
// a turn already running) is *its* job — the server's `malformed`/`busy` answers are backstops,
// not the normal path (criterion 12). The adversarial row is the `<form>`: a submit that is not
// prevented navigates, the page reloads, the socket drops and the turn the user just started
// dies. That is the failure a hand-test finds only after losing an answer.
//
// This file *skips itself* while hud/src/ask/index.js is absent (the convention of
// tests/hud/ui_action.test.mjs and tests/hud/assists.test.mjs; node tests cannot carry pytest's
// `pending` marker).
//
// API this pins (docs/tasks/T-0039.md §4 leaves the JS names to the implementer where the module
// is new; these are them, and §3 owns both files):
//
//   hud/src/ask/form.js — no DOM at module top level
//     ASK_CLIENT_TYPES = ["astra.text"]   the panel's own egress whitelist (criterion 11)
//     MAX_TEXT_CHARS   = 2000             code points, mirroring contracts/protocol.AstraText
//     createAskForm({ send, isOpen, log }) -> {
//       submit(text) -> boolean            true iff exactly one astra.text was handed to send
//       onMessage(message)                 fed every decoded astra.* message (host.subscribe)
//       enabled() -> boolean               false while the last astra.turn.state is not IDLE
//     }
//       isOpen() -> bool, default () => true: nothing is sent while the socket is not open.
//
//   hud/src/ask/index.js
//     mount(host)  — host is { shown, document, send, subscribe } (hud/src/main.js, unmodified)
//
//   hud/index.html — the container the panel mounts into (criterion 10), addressed by the same
//   data attributes tests/integration/test_hud_text_mode.py types into:
//     [data-ask-form]   a <form>          [data-ask-input]  the text box
//     [data-ask-submit] the submit control
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);
const INDEX_JS = fileURLToPath(new URL("hud/src/ask/index.js", REPO));
const INSTALLED = existsSync(INDEX_JS);
const opts = INSTALLED
  ? {}
  : {
      skip:
        "T-0039 group B is not implemented yet: hud/src/ask/index.js does not exist. This file " +
        "is the acceptance suite of docs/tasks/T-0039.md criteria 10-13; it runs the moment that " +
        "module lands.",
    };

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

async function load(name) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch (err) {
    assert.fail(
      `docs/tasks/T-0039.md §3 owns hud/src/${name}; it must be importable in node as an ES ` +
        `module with no DOM at module top level (the panel touches the DOM inside mount(host) ` +
        `only). node could not import it: ${err.message}`,
    );
  }
}

/** A recorder for the host's `send`: what left the page, and nothing more. */
function recorder({ rejects = false } = {}) {
  const sent = [];
  return {
    sent,
    send(message) {
      sent.push(message);
      return rejects ? Promise.reject(new Error("socket is not open")) : Promise.resolve();
    },
  };
}

const turnState = (state) => ({ type: "astra.turn.state", state, turn_id: "t1" });

// --- criterion 11: the panel's own egress whitelist --------------------------------------------

test("criterion 11: ASK_CLIENT_TYPES is exactly astra.text", opts, async () => {
  const { ASK_CLIENT_TYPES } = await load("ask/form.js");
  assert.deepEqual(
    [...ASK_CLIENT_TYPES],
    ["astra.text"],
    "criterion 11: the panel declares its own client->server whitelist and emits nothing else; " +
      "protocol.js CLIENT_TYPES stays T-0008's two entries (the per-directory scan in " +
      "tests/hud/protocol.test.mjs is the other half of the oracle)",
  );
});

test("criterion 12: the client-side cap is 2000 code points", opts, async () => {
  const { MAX_TEXT_CHARS } = await load("ask/form.js");
  assert.equal(
    MAX_TEXT_CHARS,
    2000,
    "criterion 2/12: contracts/protocol.AstraText accepts at most 2000 code points and the page " +
      "enforces the same bound, so the server's malformed answer is a backstop and not the " +
      "normal path (docs/tasks/T-0039.md ambiguity 1: the bound is chosen, and changing it is a " +
      "spec change, not a test edit)",
  );
});

// --- criterion 12: what may and may not leave ---------------------------------------------------

test("criterion 12: a well-formed question leaves once, as one astra.text", opts, async () => {
  const { createAskForm } = await load("ask/form.js");
  const rec = recorder();
  const form = createAskForm({ send: rec.send, log: () => {} });
  const ok = form.submit("what is on this screen");
  assert.equal(ok, true, "submit() reports that one message was handed to send");
  assert.equal(rec.sent.length, 1, `exactly one message per submit; got ${rec.sent.length}`);
  assert.deepEqual(
    rec.sent[0],
    { type: "astra.text", text: "what is on this screen" },
    "criterion 8/10: the wire message is contracts/protocol.AstraText — type and text, and " +
      "(base.Contract is extra='forbid') nothing else",
  );
});

test("criterion 12: empty and whitespace-only input send nothing", opts, async () => {
  const { createAskForm } = await load("ask/form.js");
  const rec = recorder();
  const form = createAskForm({ send: rec.send, log: () => {} });
  for (const text of ["", " ", "\t\n  ", " "]) {
    const ok = form.submit(text);
    assert.equal(ok, false, `submit(${JSON.stringify(text)}) must report nothing sent`);
  }
  assert.deepEqual(
    rec.sent,
    [],
    "criterion 12: an empty or whitespace-only string never reaches the socket — the server " +
      "would answer astra.error(code='malformed') and the user would have learnt nothing",
  );
});

test("criterion 12: 2001 code points are refused client-side, 2000 are not", opts, async () => {
  const { createAskForm, MAX_TEXT_CHARS } = await load("ask/form.js");
  const rec = recorder();
  const form = createAskForm({ send: rec.send, log: () => {} });
  assert.equal(form.submit("q".repeat(MAX_TEXT_CHARS + 1)), false, "2001 is over the cap");
  assert.deepEqual(rec.sent, [], "criterion 12: nothing over the cap goes on the wire");
  assert.equal(form.submit("q".repeat(MAX_TEXT_CHARS)), true, "2000 is exactly the cap");
  assert.equal(rec.sent.length, 1);
  assert.equal(rec.sent[0].text.length, MAX_TEXT_CHARS);

  // The cap is code points, not UTF-16 units and not bytes: 1000 astral characters are 2000
  // units and 4000 bytes, and must still be accepted (T-0027 owns the 8 MB byte cap).
  rec.sent.length = 0;
  const astral = "\u{1F600}".repeat(MAX_TEXT_CHARS);
  assert.equal(
    form.submit(astral),
    true,
    "criterion 2: the bound is 2000 *code points*; 2000 emoji are 4000 UTF-16 units and are " +
      "still one legal question",
  );
  assert.equal(rec.sent.length, 1, "…and it went out whole");
});

test("criterion 12: the question is sent stripped of surrounding whitespace", opts, async () => {
  const { createAskForm } = await load("ask/form.js");
  const rec = recorder();
  const form = createAskForm({ send: rec.send, log: () => {} });
  assert.equal(form.submit("  what is this?  \n"), true);
  assert.equal(
    rec.sent[0].text,
    "what is this?",
    "criterion 2: contracts/protocol.AstraText strips surrounding whitespace; the page sends " +
      "what the session will record as the transcript, verbatim",
  );
});

test("criterion 12: nothing is sent while the socket is not open", opts, async () => {
  const { createAskForm } = await load("ask/form.js");
  const rec = recorder();
  const form = createAskForm({ send: rec.send, isOpen: () => false, log: () => {} });
  assert.equal(form.submit("are you there"), false, "submit() reports nothing sent");
  assert.deepEqual(
    rec.sent,
    [],
    "criterion 12: a closed socket means the question is not sent at all — never queued behind " +
      "the reconnect, where it would open a turn the user is no longer waiting for",
  );
});

test("criterion 12: a send that rejects is reported, never thrown at the page", opts, async () => {
  const { createAskForm } = await load("ask/form.js");
  const rec = recorder({ rejects: true });
  const logged = [];
  const form = createAskForm({ send: rec.send, log: (line) => logged.push(String(line)) });
  await assert.doesNotReject(
    async () => form.submit("what is on this screen"),
    "ws.js rejects while the socket is closing; the panel turns that into a line, not an " +
      "exception in the submit handler (T-0008 criterion 10: a panel never takes the page down)",
  );
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.ok(logged.length >= 1, `the refusal is reported; the panel logged ${logged.length} lines`);
});

// --- criterion 13: the control follows the turn state -------------------------------------------

test("criterion 13: THINKING disables the control and IDLE re-enables it", opts, async () => {
  const { createAskForm } = await load("ask/form.js");
  const rec = recorder();
  const form = createAskForm({ send: rec.send, log: () => {} });
  assert.equal(form.enabled(), true, "the page starts IDLE and ready for a question");

  for (const state of ["LISTENING", "ENDPOINTING", "THINKING", "SPEAKING", "PAUSED", "INTERRUPTED"]) {
    form.onMessage(turnState(state));
    assert.equal(
      form.enabled(),
      false,
      `criterion 13: the input is disabled while the last astra.turn.state is ${state}; a second ` +
        "question there is refused by the session with astra.error(code='busy') (criterion 7)",
    );
  }
  form.onMessage(turnState("IDLE"));
  assert.equal(form.enabled(), true, "criterion 13: IDLE re-enables it");
});

test("criterion 13: a busy error does not disable or clear anything of its own", opts, async () => {
  const { createAskForm } = await load("ask/form.js");
  const rec = recorder();
  const form = createAskForm({ send: rec.send, log: () => {} });
  form.onMessage({ type: "astra.error", code: "busy", message: "a turn is already running" });
  assert.equal(
    form.enabled(),
    true,
    "criterion 13: astra.error(code='busy') is rendered by the existing error strip; the state " +
      "the control follows is astra.turn.state and nothing else",
  );
});

// --- the DOM half: mount(host), the form, and preventDefault -------------------------------------

/** The smallest document the panel can be mounted into: the elements hud/index.html declares. */
function fakeDocument() {
  class El {
    constructor(tag) {
      this.tagName = String(tag).toUpperCase();
      this.childNodes = [];
      this.attributes = new Map();
      this.dataset = {};
      this.listeners = new Map();
      this.value = "";
      this.disabled = false;
      this.hidden = false;
      this.textContent = "";
      this.className = "";
      this.id = "";
      this.parentElement = null;
      this.focused = 0;
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
      for (const n of nodes) this.appendChild(n);
    }
    replaceChildren(...nodes) {
      this.childNodes = [];
      for (const n of nodes) this.appendChild(n);
    }
    focus() {
      this.focused += 1;
    }
    addEventListener(type, handler) {
      if (!this.listeners.has(type)) this.listeners.set(type, []);
      this.listeners.get(type).push(handler);
    }
    dispatchEvent(event) {
      // Real DOM propagation, because two rows below are *about* propagation: a bubbling event
      // dispatched on the input walks its ancestors up to the document unless a listener on the
      // way calls stopPropagation(). A fake that delivers only to the target would make
      // "the m keydown never reaches the document" true of every implementation.
      event.target = this;
      let node = this;
      while (node) {
        for (const handler of node.listeners.get(event.type) || []) {
          event.currentTarget = node;
          handler(event);
        }
        if (event.__stopped || !event.bubbles) break;
        node = node.parentElement || node.__document || null;
      }
      return !event.defaultPrevented;
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
  }

  const doc = {
    createElement: (tag) => new El(tag),
    createTextNode: (text) => Object.assign(new El("#text"), { textContent: text }),
    listeners: new Map(),
    addEventListener(type, handler) {
      if (!this.listeners.has(type)) this.listeners.set(type, []);
      this.listeners.get(type).push(handler);
    },
  };
  doc.head = new El("head");
  doc.body = new El("body");
  doc.documentElement = new El("html");
  doc.documentElement.appendChild(doc.head);
  doc.documentElement.appendChild(doc.body);
  doc.querySelector = (s) => doc.documentElement.querySelector(s);
  doc.querySelectorAll = (s) => doc.documentElement.querySelectorAll(s);
  doc.getElementById = (id) => doc.documentElement.querySelector(`#${id}`);
  doc.documentElement.__document = doc; // the last hop of a bubbling event

  // hud/index.html's connection chip: hud/src/ws.js writes "connected" into it on open and
  // "disconnected"/"socket error"/"socket failed: …" otherwise (hud/src/main.js onStatus). It is
  // the only socket state the page publishes, and therefore the only one a panel can read.
  const status = new El("span");
  status.setAttribute("id", "conn-status");
  status.textContent = "connected"; // the rows below start on a live page and say when they do not
  doc.body.appendChild(status);

  // hud/index.html's ask container (criterion 10), as the page declares it.
  const panel = new El("section");
  panel.setAttribute("data-ask-panel", "");
  const form = new El("form");
  form.setAttribute("data-ask-form", "");
  const input = new El("textarea");
  input.setAttribute("data-ask-input", "");
  const submit = new El("button");
  submit.setAttribute("data-ask-submit", "");
  submit.setAttribute("type", "submit");
  form.appendChild(input);
  form.appendChild(submit);
  panel.appendChild(form);
  doc.body.appendChild(panel);
  return { doc, form, input, submit, status, El };
}

function submitEvent() {
  return {
    type: "submit",
    defaultPrevented: false,
    preventDefault() {
      this.defaultPrevented = true;
    },
  };
}

async function mounted(extra = {}, recorderOptions = {}) {
  const index = await load("ask/index.js");
  assert.equal(
    typeof index.mount,
    "function",
    "criterion 10: hud/src/ask/index.js exports mount(host) — the contract panels.js calls",
  );
  const dom = fakeDocument();
  const rec = recorder(recorderOptions);
  const handlers = [];
  const host = {
    document: dom.doc,
    send: rec.send,
    subscribe: (fn) => {
      handlers.push(fn);
      return () => {};
    },
    ...extra,
  };
  await index.mount(host);
  return { ...dom, rec, host, handlers, deliver: (m) => handlers.forEach((fn) => fn(m)) };
}

test("criterion 10: mount({}) does not throw, renders nothing and sends nothing", opts, async () => {
  const index = await load("ask/index.js");
  const rec = recorder();
  await assert.doesNotReject(
    async () => index.mount({}),
    "T-0008 criterion 10: a host that predates this seam (no document, no send, no subscribe) " +
      "degrades instead of taking the socket down",
  );
  assert.deepEqual(rec.sent, [], "mounting sends nothing");
});

test("criterion 12 adversarial: the submit handler calls preventDefault", opts, async () => {
  // A <form> whose submit is not prevented navigates: the page reloads, the WebSocket drops and
  // the turn the user just opened dies with it. Nothing in the DOM shows this afterwards, which
  // is why it is asserted on the event object itself.
  const m = await mounted();
  m.input.value = "what is on this screen";
  const event = submitEvent();
  m.form.dispatchEvent(event);
  assert.equal(
    event.defaultPrevented,
    true,
    "criterion 12: the form's default is prevented — a navigating form reloads the page, drops " +
      "the socket and kills the turn",
  );
  assert.equal(m.rec.sent.length, 1, `one submit, one astra.text; got ${m.rec.sent.length}`);
  assert.equal(m.rec.sent[0].type, "astra.text");
  assert.equal(m.rec.sent[0].text, "what is on this screen");
});

test("criterion 12: submitting clears the input", opts, async () => {
  const m = await mounted();
  m.input.value = "what is on this screen";
  m.form.dispatchEvent(submitEvent());
  // T-0039 ruling A19(3): the clear happens once `send` has *taken* the message, so this row
  // yields one microtask first. What may not happen is clearing a question the socket refused —
  // that is the row below ("a rejected send keeps the question in the box").
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(
    m.input.value,
    "",
    "criterion 12: the box is cleared, so the next question is not the previous one with a word " +
      "appended",
  );
});

test("criterion 12: an empty box submits nothing and stays mounted", opts, async () => {
  const m = await mounted();
  m.input.value = "   ";
  const event = submitEvent();
  m.form.dispatchEvent(event);
  assert.deepEqual(m.rec.sent, [], "criterion 12: whitespace-only is not a question");
  assert.equal(
    event.defaultPrevented,
    true,
    "…and the form still must not navigate on the empty submit either",
  );
});

test("criterion 13: the real input is disabled while the turn is not IDLE", opts, async () => {
  const m = await mounted();
  assert.equal(m.input.disabled, false, "the page starts ready for a question");
  m.deliver(turnState("THINKING"));
  assert.equal(
    m.input.disabled,
    true,
    "criterion 13: the panel disables the input while the last astra.turn.state is not IDLE " +
      "(host.subscribe is the only inbound seam it has — hud/src/main.js is not modified)",
  );
  m.input.value = "and another thing";
  m.form.dispatchEvent(submitEvent());
  assert.deepEqual(
    m.rec.sent,
    [],
    "criterion 7/13: a question submitted while a turn is running is refused by the page too; " +
      "the session's astra.error(code='busy') is the backstop",
  );
  m.deliver(turnState("IDLE"));
  assert.equal(m.input.disabled, false, "criterion 13: IDLE re-enables it");
});

// --- criterion 10: the page declares the container the panel mounts into --------------------------

test("criterion 10: hud/index.html carries the ask container the tests type into", opts, () => {
  const html = readFileSync(fileURLToPath(new URL("hud/index.html", REPO)), "utf8");
  for (const attribute of ["data-ask-form", "data-ask-input", "data-ask-submit"]) {
    assert.ok(
      html.includes(attribute),
      `criterion 10: hud/index.html gains a text box and a submit control inside a container the ` +
        `new panel mounts into; it has no ${attribute} (tests/integration/test_hud_text_mode.py ` +
        `types into exactly these attributes)`,
    );
  }
});

// --- T-0039 ruling A19/A20: the three defects the group-B review found ---------------------------

/** A key event as the browser dispatches one from a keystroke in the box: it bubbles. */
function keyEvent(type, key) {
  return {
    type,
    key,
    bubbles: true,
    metaKey: false,
    ctrlKey: false,
    altKey: false,
    defaultPrevented: false,
    __stopped: false,
    preventDefault() {
      this.defaultPrevented = true;
    },
    stopPropagation() {
      this.__stopped = true;
    },
  };
}

test("A19(1): keystrokes in the box never reach the document's shortcut listener", opts, async () => {
  // hud/src/main.js:182 binds a document-level `keydown` and stops the capture on a bare "m"
  // (the mute shortcut). Every question with an "m" in it — "how many…" — therefore kills
  // perception mid-typing, and the next turn sees no screen. main.js is zero-diff here
  // (criterion 20), so the panel is what must keep its keystrokes to itself.
  const m = await mounted();
  const heard = [];
  for (const type of ["keydown", "keypress", "keyup"]) {
    m.doc.addEventListener(type, (event) => heard.push(`${type}:${event.key}`));
  }

  // Anti-vacuity: the fake document really does deliver bubbling key events to a document-level
  // listener when nothing stops them — a keystroke anywhere else on the page still arrives.
  m.doc.body.dispatchEvent(keyEvent("keydown", "m"));
  assert.deepEqual(
    heard,
    ["keydown:m"],
    "the fixture's document listener must fire for an event that nothing stops; otherwise the " +
      "assertion below would hold of every implementation",
  );

  heard.length = 0;
  for (const type of ["keydown", "keypress", "keyup"]) {
    m.input.dispatchEvent(keyEvent(type, "m"));
  }
  assert.deepEqual(
    heard,
    [],
    "T-0039 ruling A19(1): the panel stops propagation of keydown/keypress/keyup originating in " +
      "[data-ask-input], so typing a question containing 'm' does not trip main.js's mute " +
      "shortcut and stop the capture",
  );
});

test("A19(1): the same holds for every other character of a typed question", opts, async () => {
  const m = await mounted();
  const heard = [];
  m.doc.addEventListener("keydown", (event) => heard.push(event.key));
  for (const key of [..."how many eighths", "Enter", "Backspace", " "]) {
    m.input.dispatchEvent(keyEvent("keydown", key));
  }
  assert.deepEqual(
    heard,
    [],
    "A19(1): the guard is 'this event came from the box', not 'this key is m' — a panel that " +
      "special-cases the one shortcut it knows about breaks on the next one main.js adds",
  );
});

test("A19(3): a rejected send keeps the question in the box", opts, async () => {
  // The question is the user's typing, and it is the only copy: clearing the box before the
  // socket has taken the message loses it, and the page shows no sign that it did.
  const logged = [];
  const m = await mounted({ log: (line) => logged.push(String(line)) }, { rejects: true });
  m.input.value = "how many eighths are in three quarters";
  m.form.dispatchEvent(submitEvent());
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(
    m.input.value,
    "how many eighths are in three quarters",
    "T-0039 ruling A19(3): the box is cleared only after `send` resolves; on a rejection the " +
      "text stays so the user can press Enter again instead of retyping it",
  );
  assert.ok(
    logged.some((line) => /not sent|socket/i.test(line)),
    `A19(3): …and the rejection is reported rather than swallowed; the panel logged ${JSON.stringify(logged)}`,
  );
});

test("criterion 12: a send that resolves does clear the box", opts, async () => {
  // The other half of the row above: "keep the text on rejection" must not become "never clear".
  const m = await mounted();
  m.input.value = "how many eighths are in three quarters";
  m.form.dispatchEvent(submitEvent());
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(m.rec.sent.length, 1, "the question went out");
  assert.equal(
    m.input.value,
    "",
    "criterion 12: once the socket has taken the message the box is cleared, so the next " +
      "question is not the previous one with a word appended",
  );
});

test("A19(3): isOpen comes from #conn-status, which only says 'connected' when it is", opts, async () => {
  // The real panel host (hud/src/main.js, unmodified) hands panels no socket state, so a default
  // of `() => true` means the page happily "sends" into a dead socket and the user waits for an
  // answer that was never asked for. `#conn-status` is what ws.js writes the socket's state
  // into ("connected" on open; "disconnected"/"socket error"/"socket failed: …" otherwise), and
  // it is the only honest seam that does not touch main.js.
  const m = await mounted();
  for (const text of ["idle", "disconnected", "socket error", "socket failed: boom", ""]) {
    m.status.textContent = text;
    m.input.value = "are you there";
    m.form.dispatchEvent(submitEvent());
    assert.deepEqual(
      m.rec.sent,
      [],
      `T-0039 ruling A19(3): #conn-status reads ${JSON.stringify(text)}, which is not ` +
        "'connected', so nothing may go on the wire (criterion 12: never send while the socket " +
        "is not open)",
    );
    assert.equal(
      m.input.value,
      "are you there",
      "…and the unsent question stays in the box (A19(3))",
    );
  }

  m.status.textContent = "connected";
  m.form.dispatchEvent(submitEvent());
  assert.deepEqual(
    m.rec.sent,
    [{ type: "astra.text", text: "are you there" }],
    "…and the moment ws.js writes 'connected' the same submit goes out: the state is read per " +
      "submit, not cached at mount, because a socket drops while the page stays open",
  );
});
