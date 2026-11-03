// T-0090 criteria 4 and 5 — hud/src/panels/tool_call.js: the action card.
//
// In v1 a proposed action is never executed (T-0080d H1: dry run only) and a search query is
// never fetched or linked (H2). The card says what the assistant *would* have done, shows the
// status, the source and the query_problem warning (H7), and is rendered with textContent only.
// The failure modes this file is aimed at:
//   * a card that turns model text into markup (innerHTML) — a query of
//     "<img src=x onerror=alert(1)>" must stay text and create no element;
//   * a card that offers a click-through (a link, window.open, fetch) for a search query;
//   * a panel that clears what it showed when a later status for the same call_id arrives —
//     tested in the order the server publishes (.claude/rules/tests.md, "a panel that clears
//     state on an id change must be tested in the order the server actually publishes"):
//     proposed -> repairing -> dry_run | refused | cancelled.
//
// API this pins (docs/tasks/T-0090.md names the file, not the exports; these are the names):
//
//   hud/src/panels/tool_call.js — importable in node, no DOM at module top level
//     TOOL_CALL_TYPE = "astra.tool.call"
//     createToolCallPanel(root) -> { handle(message) }
//       root      an element; cards are built with root.ownerDocument.createElement and live
//                 under root, one per call_id, in arrival order
//       handle(m) one decoded server message of any type; only astra.tool.call is acted on
//
//   DOM hooks inside root:
//     [data-tool-call-id="<call_id>"]  the card
//     [data-tool-line]                 inside the card: the one "Would …" line
//     [data-tool-status="<status>"]    inside the card: the status chip (non-empty text)
//     [data-tool-source="<source>"]    inside the card: draft | repair (non-empty text)
//     [data-tool-warning]              inside the card, only while query_problem is set
//     text "dry run — nothing was done" in the card whenever the status is dry_run
//
// Every expected line is criterion 4's text with the arguments substituted by hand.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);
const PANEL_FILE = fileURLToPath(new URL("hud/src/panels/tool_call.js", REPO));

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

// Origin proof (.claude/rules/tests.md): print where the module under test was imported from.
console.log(`# tool card origin: repo=${fileURLToPath(REPO)} modules=${srcRoot().href}`);

async function load() {
  let mod;
  try {
    mod = await import(new URL("panels/tool_call.js", srcRoot()).href);
  } catch (err) {
    assert.fail(
      "T-0090 criterion 4 adds hud/src/panels/tool_call.js, importable in node as an ES module " +
        `with no DOM at module top level. node could not import it: ${err.message}`,
    );
  }
  assert.equal(
    mod.TOOL_CALL_TYPE,
    "astra.tool.call",
    "T-0090: tool_call.js exports TOOL_CALL_TYPE = 'astra.tool.call'",
  );
  assert.equal(
    typeof mod.createToolCallPanel,
    "function",
    "T-0090 criterion 4: tool_call.js exports createToolCallPanel(root) -> { handle(message) }",
  );
  return mod;
}

// --- a DOM double ----------------------------------------------------------------------------
//
// Enough of Element/Document for a card built the ordinary way. The markup sinks are the point:
// innerHTML / outerHTML / insertAdjacentHTML / document.write throw a named assertion and are
// recorded, so "rendered with textContent only" is a behaviour, not a grep. Every element ever
// created is recorded by tag, so "no element is created" from model text is checkable. A name the
// double does not implement throws a named error rather than a bare TypeError, so a gap in the
// double reads as a gap in the double (extend it here; do not reshape the panel around it).

const MARKUP_SINKS = ["innerHTML", "outerHTML"]; // insertAdjacentHTML is a method, trapped below

function fakeDocument() {
  const created = [];
  const violations = [];
  const doc = {
    created,
    violations,
    nodeType: 9,
    createElement(tag) {
      const el = makeElement(doc, String(tag).toLowerCase());
      created.push(el);
      return el;
    },
    createTextNode(text) {
      return makeText(doc, text);
    },
    createDocumentFragment() {
      return makeElement(doc, "#fragment");
    },
    write() {
      violations.push("document.write");
      throw new Error("T-0090 criterion 4: the card never writes markup (document.write)");
    },
  };
  return doc;
}

function makeText(doc, text) {
  return {
    nodeType: 3,
    ownerDocument: doc,
    parentNode: null,
    data: String(text),
    get textContent() {
      return this.data;
    },
    set textContent(v) {
      this.data = String(v);
    },
    remove() {
      if (this.parentNode) this.parentNode._detach(this);
    },
  };
}

const camel = (name) => name.slice(5).replace(/-([a-z])/g, (_m, c) => c.toUpperCase());
const kebab = (key) => "data-" + key.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase());

function makeElement(doc, tag) {
  const attrs = new Map();
  const childNodes = [];
  const classes = new Set();
  const listeners = [];
  const target = {
    nodeType: 1,
    tagName: tag.toUpperCase(),
    localName: tag,
    nodeName: tag.toUpperCase(),
    ownerDocument: doc,
    parentNode: null,
    hidden: false,
    title: "",
    id: "",
    childNodes,
    get children() {
      return childNodes.filter((c) => c.nodeType === 1);
    },
    get firstChild() {
      return childNodes[0] || null;
    },
    get lastChild() {
      return childNodes[childNodes.length - 1] || null;
    },
    get firstElementChild() {
      return this.children[0] || null;
    },
    get parentElement() {
      return this.parentNode;
    },
    get isConnected() {
      return true;
    },
    get className() {
      return [...classes].join(" ");
    },
    set className(v) {
      classes.clear();
      for (const c of String(v).split(/\s+/)) if (c) classes.add(c);
    },
    classList: {
      add: (...n) => n.forEach((c) => classes.add(c)),
      remove: (...n) => n.forEach((c) => classes.delete(c)),
      toggle: (c, force) => {
        const on = force === undefined ? !classes.has(c) : !!force;
        if (on) classes.add(c);
        else classes.delete(c);
        return on;
      },
      contains: (c) => classes.has(c),
      replace: (a, b) => (classes.delete(a) ? (classes.add(b), true) : false),
    },
    style: {},
    get textContent() {
      return childNodes.map((c) => c.textContent).join("");
    },
    set textContent(v) {
      for (const c of childNodes) c.parentNode = null;
      childNodes.length = 0;
      const s = v === null || v === undefined ? "" : String(v);
      if (s) proxy.appendChild(makeText(doc, s));
    },
    get innerText() {
      return this.textContent;
    },
    set innerText(v) {
      this.textContent = v;
    },
    setAttribute(name, value) {
      name = String(name).toLowerCase();
      if (name === "class") return (this.className = value);
      if (name === "id") this.id = String(value);
      if (name === "hidden") this.hidden = true;
      attrs.set(name, String(value));
    },
    getAttribute(name) {
      name = String(name).toLowerCase();
      if (name === "class") return classes.size ? this.className : null;
      if (name.startsWith("data-")) {
        const v = dataset[camel(name)];
        return v === undefined ? null : v;
      }
      if (name === "hidden") return this.hidden ? "" : null;
      return attrs.has(name) ? attrs.get(name) : null;
    },
    hasAttribute(name) {
      return this.getAttribute(name) !== null;
    },
    removeAttribute(name) {
      name = String(name).toLowerCase();
      if (name.startsWith("data-")) delete dataset[camel(name)];
      if (name === "hidden") this.hidden = false;
      attrs.delete(name);
    },
    toggleAttribute(name, force) {
      const on = force === undefined ? !this.hasAttribute(name) : !!force;
      if (on) this.setAttribute(name, "");
      else this.removeAttribute(name);
      return on;
    },
    _detach(child) {
      const i = childNodes.indexOf(child);
      if (i >= 0) childNodes.splice(i, 1);
      child.parentNode = null;
    },
    _adopt(node) {
      if (typeof node === "string" || typeof node === "number") return [makeText(doc, node)];
      if (node && node.localName === "#fragment") {
        const kids = [...node.childNodes];
        for (const k of kids) node._detach(k);
        return kids;
      }
      if (node && node.parentNode) node.parentNode._detach(node);
      return [node];
    },
    appendChild(node) {
      for (const n of proxy._adopt(node)) {
        n.parentNode = proxy;
        childNodes.push(n);
      }
      return node;
    },
    append(...nodes) {
      for (const n of nodes) proxy.appendChild(n);
    },
    prepend(...nodes) {
      const adopted = nodes.flatMap((n) => proxy._adopt(n));
      for (const n of adopted) n.parentNode = proxy;
      childNodes.unshift(...adopted);
    },
    insertBefore(node, ref) {
      if (!ref) return proxy.appendChild(node);
      const adopted = proxy._adopt(node);
      const i = childNodes.indexOf(ref);
      for (const n of adopted) n.parentNode = proxy;
      childNodes.splice(i < 0 ? childNodes.length : i, 0, ...adopted);
      return node;
    },
    removeChild(node) {
      proxy._detach(node);
      return node;
    },
    replaceChild(next, old) {
      proxy.insertBefore(next, old);
      proxy._detach(old);
      return old;
    },
    replaceChildren(...nodes) {
      for (const c of childNodes) c.parentNode = null;
      childNodes.length = 0;
      proxy.append(...nodes);
    },
    replaceWith(...nodes) {
      const parent = proxy.parentNode;
      if (!parent) return;
      for (const n of nodes) parent.insertBefore(n, proxy);
      parent._detach(proxy);
    },
    remove() {
      if (proxy.parentNode) proxy.parentNode._detach(proxy);
    },
    contains(node) {
      for (let n = node; n; n = n.parentNode) if (n === proxy) return true;
      return false;
    },
    matches(selector) {
      return matches(proxy, selector);
    },
    closest(selector) {
      for (let n = proxy; n && n.nodeType === 1; n = n.parentNode) if (matches(n, selector)) return n;
      return null;
    },
    querySelectorAll(selector) {
      return descendants(proxy).filter((e) => matches(e, selector));
    },
    querySelector(selector) {
      return proxy.querySelectorAll(selector)[0] || null;
    },
    addEventListener(type, fn) {
      listeners.push([type, fn]);
    },
    removeEventListener() {},
    _listeners: listeners,
    _attrs: attrs,
  };
  const dataset = new Proxy(
    {},
    {
      set(obj, key, value) {
        obj[key] = String(value);
        return true;
      },
    },
  );
  target.dataset = dataset;
  for (const sink of MARKUP_SINKS) {
    Object.defineProperty(target, sink, {
      get() {
        return sink === "innerHTML" ? "" : undefined;
      },
      set() {
        doc.violations.push(sink);
        throw new Error(`T-0090 criterion 4: model text is rendered with textContent only (${sink} was set)`);
      },
    });
  }
  target.insertAdjacentHTML = () => {
    doc.violations.push("insertAdjacentHTML");
    throw new Error("T-0090 criterion 4: model text is rendered with textContent only (insertAdjacentHTML)");
  };
  const proxy = new Proxy(target, {
    get(obj, key, receiver) {
      if (typeof key === "symbol" || key in obj) return Reflect.get(obj, key, receiver);
      if (key === "then" || key === "toJSON" || key === "nodeValue") return undefined;
      throw new Error(
        `the DOM double in tests/hud/tool_call.test.mjs has no Element.${key}; extend the double ` +
          "(this is a gap in the test, not a defect in the panel)",
      );
    },
    set(obj, key, value) {
      if (key === "href" || key === "src" || key === "onclick" || key === "target") {
        doc.violations.push(`${key}=${value}`);
      }
      obj[key] = value;
      return true;
    },
  });
  return proxy;
}

function descendants(el) {
  const out = [];
  for (const c of el.childNodes || []) {
    if (c.nodeType !== 1) continue;
    out.push(c, ...descendants(c));
  }
  return out;
}

/** Simple selectors only: tag, .class, #id, [attr], [attr="v"]. */
function matches(el, selector) {
  const s = selector.trim();
  const attr = s.match(/^\[([a-z0-9-]+)(?:=["']?([^"'\]]*)["']?)?\]$/i);
  if (attr) {
    const v = el.getAttribute(attr[1]);
    return attr[2] === undefined ? v !== null : v === attr[2];
  }
  if (s.startsWith(".")) return el.classList.contains(s.slice(1));
  if (s.startsWith("#")) return el.id === s.slice(1);
  return el.localName === s.toLowerCase();
}

// --- helpers ----------------------------------------------------------------------------------

async function panel() {
  const { createToolCallPanel } = await load();
  const doc = fakeDocument();
  const root = doc.createElement("div");
  const p = createToolCallPanel(root);
  assert.ok(p && typeof p.handle === "function", "createToolCallPanel(root) returns { handle }");
  // Whatever the panel built at mount is its own chrome; the rows below count what *messages* do.
  const mounted = { created: doc.created.length, children: root.childNodes.length, text: root.textContent };
  return { doc, root, mounted, handle: (m) => p.handle(m) };
}

const call = (over = {}) => ({
  type: "astra.tool.call",
  turn_id: "t7",
  call_id: "t7-c1",
  name: "click_at",
  arguments: { x: 120, y: 845 },
  status: "proposed",
  source: "draft",
  ...over,
});

const squeeze = (s) => String(s).replace(/\s+/g, " ").trim();
const cards = (root) => root.querySelectorAll("[data-tool-call-id]");
function card(root, id) {
  const found = cards(root).filter((c) => c.getAttribute("data-tool-call-id") === id);
  assert.equal(found.length, 1, `exactly one card for call_id ${id}; found ${found.length}`);
  return found[0];
}
function lineOf(el) {
  const lines = el.querySelectorAll("[data-tool-line]");
  assert.equal(lines.length, 1, `one [data-tool-line] per card; found ${lines.length}`);
  return squeeze(lines[0].textContent);
}
function statusOf(el) {
  const chip = el.querySelector("[data-tool-status]");
  assert.ok(chip, "criterion 4: the card shows a status chip ([data-tool-status])");
  assert.ok(squeeze(chip.textContent).length > 0, "…with visible text, not only an attribute");
  return chip.getAttribute("data-tool-status");
}
function sourceOf(el) {
  const s = el.querySelector("[data-tool-source]");
  assert.ok(s, "criterion 4: the card shows the source ([data-tool-source])");
  assert.ok(squeeze(s.textContent).length > 0, "…with visible text");
  return s.getAttribute("data-tool-source");
}
const visibleWarnings = (el) =>
  el.querySelectorAll("[data-tool-warning]").filter((w) => {
    for (let n = w; n && n !== el.parentNode; n = n.parentNode) if (n.hidden) return false;
    return squeeze(w.textContent).length > 0;
  });
const DRY_RUN_NOTICE = "dry run — nothing was done";

// --- criterion 4: the line per tool ------------------------------------------------------------

test("T-0090 criterion 4: click_at reads 'Would click at (x, y) on screen'", async () => {
  const { root, handle } = await panel();
  handle(call({ arguments: { x: 120, y: 845 } }));
  const line = lineOf(card(root, "t7-c1"));
  assert.ok(
    line.startsWith("Would click at (120, 845)"),
    `criterion 4: "Would click at (x, y) on screen [frame]" with x=120, y=845; got ${JSON.stringify(line)}`,
  );
  assert.ok(line.includes("on screen"), `criterion 4: "…on screen…"; got ${JSON.stringify(line)}`);
});

test("T-0090 criterion 4: type_at quotes the text and says 'and press Enter' only when asked", async () => {
  const { root, handle } = await panel();
  handle(call({ call_id: "c-enter", name: "type_at", arguments: { x: 410, y: 72, text: "quarterly report", press_enter: true } }));
  handle(call({ call_id: "c-plain", name: "type_at", arguments: { x: 5, y: 999, text: "hello", press_enter: false } }));
  const withEnter = lineOf(card(root, "c-enter"));
  assert.ok(
    withEnter.startsWith('Would type "quarterly report" at (410, 72)'),
    `criterion 4: 'Would type "…" at (x, y)'; got ${JSON.stringify(withEnter)}`,
  );
  assert.ok(withEnter.includes("and press Enter"), `criterion 4: press_enter=true adds "and press Enter"; got ${JSON.stringify(withEnter)}`);
  const plain = lineOf(card(root, "c-plain"));
  assert.ok(plain.startsWith('Would type "hello" at (5, 999)'), `got ${JSON.stringify(plain)}`);
  assert.ok(
    !/enter/i.test(plain),
    `criterion 4: press_enter=false must not claim Enter would be pressed; got ${JSON.stringify(plain)}`,
  );
});

test("T-0090 criterion 4: scroll, press_key and google_search read exactly as the criterion", async () => {
  const { root, handle } = await panel();
  handle(call({ call_id: "c-down", name: "scroll", arguments: { direction: "down" } }));
  handle(call({ call_id: "c-up", name: "scroll", arguments: { direction: "up" } }));
  handle(call({ call_id: "c-key", name: "press_key", arguments: { combo: "Ctrl+L" } }));
  handle(call({ call_id: "c-alt", name: "press_key", arguments: { combo: "Alt+Tab" } }));
  handle(call({ call_id: "c-q", name: "google_search", arguments: { query: "weather in Paris tomorrow" } }));
  assert.equal(lineOf(card(root, "c-down")), "Would scroll down");
  assert.equal(lineOf(card(root, "c-up")), "Would scroll up", "the direction comes from the arguments");
  assert.equal(lineOf(card(root, "c-key")), "Would press Ctrl+L");
  assert.equal(lineOf(card(root, "c-alt")), "Would press Alt+Tab", "the combo comes from the arguments");
  assert.equal(lineOf(card(root, "c-q")), "Would search for: weather in Paris tomorrow");
});

test("T-0090 criterion 4: one card per call_id, in arrival order", async () => {
  const { root, handle } = await panel();
  handle(call({ call_id: "a" }));
  handle(call({ call_id: "b", name: "scroll", arguments: { direction: "down" } }));
  handle(call({ call_id: "a", status: "dry_run" }));
  assert.deepEqual(
    cards(root).map((c) => c.getAttribute("data-tool-call-id")),
    ["a", "b"],
    "criterion 4: a later message for the same call_id updates its card in place — no second card",
  );
});

// --- criterion 4: chip, source, warning, dry-run notice -----------------------------------------

test("T-0090 criterion 4: the status chip and the source are shown", async () => {
  const { root, handle } = await panel();
  handle(call({ status: "repairing", source: "draft" }));
  assert.equal(statusOf(card(root, "t7-c1")), "repairing");
  assert.equal(sourceOf(card(root, "t7-c1")), "draft");
  handle(call({ status: "dry_run", source: "repair" }));
  assert.equal(statusOf(card(root, "t7-c1")), "dry_run");
  assert.equal(sourceOf(card(root, "t7-c1")), "repair", "the source follows the latest message");
});

test("T-0090 criterion 4 / H7: a query_problem is shown as a warning, and only when set", async () => {
  const { root, handle } = await panel();
  handle(call({ call_id: "bad", name: "google_search", arguments: { query: "the" }, query_problem: "query is a single stopword" }));
  handle(call({ call_id: "ok", name: "google_search", arguments: { query: "weather in Paris" } }));
  const warned = visibleWarnings(card(root, "bad"));
  assert.equal(warned.length, 1, "criterion 4: one visible [data-tool-warning] when query_problem is set");
  assert.ok(
    squeeze(warned[0].textContent).includes("query is a single stopword"),
    `H7: the warning shows the server's query_problem; got ${JSON.stringify(warned[0].textContent)}`,
  );
  assert.equal(
    visibleWarnings(card(root, "ok")).length,
    0,
    "criterion 4, the other direction: no warning on a call whose query_problem is null",
  );
  assert.equal(lineOf(card(root, "bad")), "Would search for: the", "H7: the warning never blocks the card");
});

test("T-0090 criterion 4: 'dry run — nothing was done' whenever the status is dry_run", async () => {
  const { root, handle } = await panel();
  handle(call({ call_id: "p", status: "proposed" }));
  handle(call({ call_id: "d", status: "dry_run", action: "would click at (153, 676) on screen" }));
  handle(call({ call_id: "x", status: "executed" }));
  assert.ok(
    squeeze(card(root, "d").textContent).includes(DRY_RUN_NOTICE),
    `criterion 4: a dry_run card says "${DRY_RUN_NOTICE}"; it reads ${JSON.stringify(squeeze(card(root, "d").textContent))}`,
  );
  assert.ok(
    !squeeze(card(root, "x").textContent).includes(DRY_RUN_NOTICE),
    "the other direction: an executed card must not claim nothing was done",
  );
  assert.ok(
    !squeeze(card(root, "p").textContent).includes(DRY_RUN_NOTICE),
    "…and a proposal has not been dry-run yet",
  );
});

// --- criterion 4: status transitions in wire order ----------------------------------------------

test("T-0090 criterion 4, wire order: proposed then dry_run keeps the card's arguments", async () => {
  const { root, handle } = await panel();
  const args = { x: 410, y: 72, text: "quarterly report", press_enter: true };
  handle(call({ name: "type_at", arguments: args, status: "proposed" }));
  const before = lineOf(card(root, "t7-c1"));
  handle(call({ name: "type_at", arguments: args, status: "dry_run", action: 'would type "quarterly report" at (525, 79) and press Enter' }));
  const c = card(root, "t7-c1");
  assert.equal(lineOf(c), before, "criterion 4: the dry_run that follows the proposal must not clear the line");
  assert.ok(lineOf(c).includes("quarterly report"), "…the arguments are still on the card");
  assert.equal(statusOf(c), "dry_run");
  assert.ok(squeeze(c.textContent).includes(DRY_RUN_NOTICE));
});

test("T-0090 criterion 4, wire order: proposed -> repairing -> dry_run shows the repaired call", async () => {
  const { root, handle } = await panel();
  handle(call({ arguments: { x: 1200, y: 5 }, status: "proposed", source: "draft" }));
  handle(call({ arguments: { x: 1200, y: 5 }, status: "repairing", source: "draft" }));
  assert.equal(statusOf(card(root, "t7-c1")), "repairing");
  handle(call({ arguments: { x: 120, y: 5 }, status: "dry_run", source: "repair", action: "would click at (153, 4) on screen" }));
  const c = card(root, "t7-c1");
  assert.equal(cards(root).length, 1, "one call, one card, across all three messages");
  assert.ok(lineOf(c).startsWith("Would click at (120, 5)"), `the latest arguments win; got ${JSON.stringify(lineOf(c))}`);
  assert.ok(!squeeze(c.textContent).includes("1200"), "the invalid draft's coordinates are gone from the card");
  assert.equal(sourceOf(c), "repair");
  assert.equal(statusOf(c), "dry_run");
});

test("T-0090 criterion 4, wire order: proposed -> refused keeps the line and drops no card", async () => {
  const { root, handle } = await panel();
  handle(call({ status: "proposed" }));
  handle(call({ status: "refused", reason: "no frame" }));
  const c = card(root, "t7-c1");
  assert.equal(statusOf(c), "refused");
  assert.ok(lineOf(c).startsWith("Would click at (120, 845)"), "the refusal does not clear the line");
  assert.ok(!squeeze(c.textContent).includes(DRY_RUN_NOTICE), "a refused call was not dry-run");
});

test("T-0090 adversarial: a cancelled message after dry_run for the same id shows cancelled", async () => {
  const { root, handle } = await panel();
  handle(call({ status: "proposed" }));
  handle(call({ status: "dry_run", action: "would click at (153, 676) on screen" }));
  handle(call({ status: "cancelled", reason: "interrupted" }));
  const c = card(root, "t7-c1");
  assert.equal(cards(root).length, 1);
  assert.equal(statusOf(c), "cancelled", "criterion 4: the later status wins — the card does not stick at dry_run");
  assert.match(squeeze(c.querySelector("[data-tool-status]").textContent), /cancel/i, "…and the chip says so in words");
  assert.ok(lineOf(c).startsWith("Would click at (120, 845)"), "the line survives the cancellation");
});

test("T-0090 criterion 3/4: a replayed message renders the card like a live one", async () => {
  const { root, handle } = await panel();
  handle(call({ status: "dry_run", replay: true, name: "scroll", arguments: { direction: "down" } }));
  assert.equal(lineOf(card(root, "t7-c1")), "Would scroll down", "a reloading page is shown the journalled card");
});

// --- criterion 4 adversarial: model text is text ------------------------------------------------

test("T-0090 adversarial: an HTML query renders as text and creates no element", async () => {
  const { doc, root, handle } = await panel();
  const evil = "<img src=x onerror=alert(1)>";
  handle(call({ name: "google_search", arguments: { query: evil }, status: "proposed", query_problem: evil }));
  handle(call({ name: "google_search", arguments: { query: evil }, status: "dry_run", action: `would search for: ${evil}`, query_problem: evil }));
  handle(call({ call_id: "t7-c2", name: "type_at", arguments: { x: 1, y: 2, text: '<a href="http://example.com">x</a>', press_enter: false } }));
  assert.deepEqual(doc.violations, [], `criterion 4: no markup sink and no href/src/onclick was touched: ${doc.violations}`);
  const tags = new Set(doc.created.map((e) => e.localName));
  for (const banned of ["img", "a", "script", "iframe", "object", "embed", "link", "form"]) {
    assert.ok(!tags.has(banned), `criterion 4: no <${banned}> may be created; the panel created ${[...tags]}`);
  }
  assert.equal(lineOf(card(root, "t7-c1")), `Would search for: ${evil}`, "the query is shown verbatim, as text");
  assert.ok(lineOf(card(root, "t7-c2")).includes('<a href="http://example.com">x</a>'), "typed text is text too");
});

// --- criterion 5 negative: no link, no fetch, no window.open, nothing sent -----------------------

test("T-0090 criterion 5: a search card offers no link and the panel makes no request", async () => {
  const calls = [];
  const saved = {};
  const trap = (name) => (...args) => {
    calls.push(name);
    throw new Error(`T-0090 criterion 5 / H2: the card called ${name}(${JSON.stringify(args[0])})`);
  };
  for (const name of ["fetch", "open", "XMLHttpRequest", "WebSocket", "EventSource", "Image"]) {
    saved[name] = Object.getOwnPropertyDescriptor(globalThis, name);
    globalThis[name] = trap(name);
  }
  const savedNav = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { sendBeacon: trap("navigator.sendBeacon") },
  });
  try {
    const { doc, root, handle } = await panel();
    handle(call({ name: "google_search", arguments: { query: "weather in Paris" }, status: "proposed" }));
    handle(call({ name: "google_search", arguments: { query: "weather in Paris" }, status: "dry_run", action: "would search for: weather in Paris" }));
    assert.deepEqual(calls, [], "criterion 5 / H2: no fetch, no window.open, no request of any kind");
    assert.equal(root.querySelectorAll("a").length, 0, "criterion 5 / H2: no link on the card");
    assert.ok(
      !descendants(root).some((e) => e._attrs.has("href") || e._attrs.has("src") || e._listeners.length),
      "criterion 5: no href, no src and no click handler — the card is display-only",
    );
    assert.ok(!doc.created.some((e) => e.localName === "button"), "the card has nothing to press in v1");
  } finally {
    for (const [name, desc] of Object.entries(saved)) {
      if (desc) Object.defineProperty(globalThis, name, desc);
      else delete globalThis[name];
    }
    if (savedNav) Object.defineProperty(globalThis, "navigator", savedNav);
    else delete globalThis.navigator;
  }
});

/** The panel's source with comments stripped (a guard that greps prose is a false alarm). */
function codeOf(path) {
  return readFileSync(path, "utf8")
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/(^|[^:\\])\/\/[^\n]*/g, "$1 ");
}

test("T-0090 criteria 4 and 5: tool_call.js names no markup sink, no request API and no send", () => {
  assert.ok(existsSync(PANEL_FILE), "T-0090 adds hud/src/panels/tool_call.js");
  const src = codeOf(PANEL_FILE);
  for (const [pattern, why] of [
    [/\binnerHTML\b|\bouterHTML\b|insertAdjacentHTML|\bDOMParser\b|createContextualFragment|document\.write/, "criterion 4: textContent only"],
    [/\bfetch\s*\(|XMLHttpRequest|sendBeacon|\bWebSocket\b|\bEventSource\b/, "criterion 5: no request"],
    [/\bwindow\.open\b|\blocation\s*[.=]|\.href\b|['"]href['"]/, "criterion 5 / H2: no link and no navigation"],
    [/createElement\(\s*['"](a|img|iframe|script|form)['"]/, "criterion 5: no element that loads or links"],
    [/\.send\s*\(/, "implementation notes: the card is display-only and sends nothing back"],
  ]) {
    const m = src.match(pattern);
    assert.equal(m, null, `${why}; hud/src/panels/tool_call.js contains ${m && JSON.stringify(m[0])} (comments are stripped first)`);
  }
});

// --- criterion 5: every other type is left alone ---------------------------------------------------

test("T-0090 criterion 5: messages of every other type do nothing to the card pane", async () => {
  const { doc, root, mounted, handle } = await panel();
  const others = [
    { type: "astra.spoken", turn_id: "t7", utterance_id: "u1", text: "Would click at (1, 2)" },
    { type: "astra.turn.state", state: "SPEAKING", turn_id: "t7" },
    { type: "astra.transcript", turn_id: "t7", text: "click the button" },
    { type: "astra.ui.surface", surface_id: "s", canvas_idx: 0, messages: [] },
    { type: "astra.ui.fence", turn_id: "t7", canvases: 0 },
    { type: "astra.error", code: "x", message: "y" },
    { type: "astra.tool.calls", call_id: "c", name: "click_at", arguments: {}, status: "proposed", source: "draft" },
    null,
    undefined,
    "astra.tool.call",
  ];
  for (const m of others) assert.doesNotThrow(() => handle(m), `handle(${JSON.stringify(m)}) threw`);
  assert.equal(cards(root).length, 0, "criterion 5: no card for a message that is not astra.tool.call");
  assert.equal(root.childNodes.length, mounted.children, "…no change to the pane at all");
  assert.equal(root.textContent, mounted.text, "…not even its text");
  assert.equal(doc.created.length, mounted.created, "…and no element was even built");
});

// --- review rows (docs/tasks/T-0090.md "Rulings after review": U2 iff, U7 on the page) ----------

/** The card's text a human sees: hidden subtrees excluded (a hidden element's text is still text). */
function visibleText(el) {
  if (el.nodeType === 3) return el.textContent;
  if (el.hidden) return "";
  return el.childNodes.map(visibleText).join("");
}

test("T-0090 review (U2 iff): dry_run -> cancelled hides the dry-run notice", async () => {
  const { root, handle } = await panel();
  handle(call({ status: "dry_run", action: "would click at (153, 676) on screen" }));
  assert.ok(squeeze(visibleText(card(root, "t7-c1"))).includes(DRY_RUN_NOTICE), "premise: shown on dry_run");
  handle(call({ status: "cancelled", reason: "interrupted" }));
  assert.ok(
    !squeeze(visibleText(card(root, "t7-c1"))).includes(DRY_RUN_NOTICE),
    "U2: the notice is shown if and only if the status is dry_run — a cancelled card must not " +
      `keep saying it; the card reads ${JSON.stringify(squeeze(visibleText(card(root, "t7-c1"))))}`,
  );
});

// --- T-0154 B7: the "replace on a new turn" rows below are amended by name -----------------------
//
// docs/tasks/T-0154.md B7: "Each turn's astra.tool.call cards render in that turn's article.
// Earlier turns keep theirs as history (H-T128-3 default); T-0090 'replace on a new turn' is
// amended by name. U7 and U10 still hold: a message for a superseded turn is ignored and never
// reopens or moves a turn." The four T-0090 rows that asserted the *replacement* were deleted in
// group B's activation commit (G2): "T-0090 review (U7): the first tool call of a never-seen turn
// replaces the shown cards", "T-0090 review (U7): a late message for a superseded turn neither
// clears nor adds", "T-0090 U8: the page remembers eight superseded turns, not just the last one"
// and "T-0090 U10: sequence A over tool calls — t1, t2, t3 cards, then a late t2 message". Their
// B7 counterparts below run through `pendingT0154B` (strict xfail until the activation is recorded
// in T-0154's **Status:** line; `ASTRA_RUN_PENDING=1` runs them for real). Assertions amended,
// each tied to B7:
//   U7 "a new turn's first card replaces the previous turn's cards"  -> earlier cards stay;
//   U7 late message: cards === ["t2-c1"]                              -> ["t1-c1", "t2-c1"], t1's card unchanged;
//   U8: cards === ["t9-c1"] after each late message                    -> all nine cards, t1–t8 unchanged;
//   U10: cards === ["t3-c1"]                                           -> t1, t2, t3, t2's card unchanged.
// Kept unchanged: "the late message neither clears nor adds", "the same element", "untouched".

const T0154_TASK = fileURLToPath(new URL("docs/tasks/T-0154.md", REPO));
const T0154_STATUS = existsSync(T0154_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0154_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const B_ACTIVE = /\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i.test(T0154_STATUS);
const B_PENDING = !B_ACTIVE && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0154B(name, fn) {
  test(name + (B_PENDING ? " [pending T-0154 B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null)
      assert.fail(
        "XPASS(strict): this T-0154 group B row passed before group B's activation is recorded in " +
          "docs/tasks/T-0154.md's status. Either B landed (record the activation) or the row is vacuous.",
      );
    t.diagnostic("expected failure (T-0154 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

const ids = (root) => cards(root).map((c) => c.getAttribute("data-tool-call-id"));

pendingT0154B("T-0154 B7 (amends U7): a never-seen turn's first card keeps the earlier turns' cards as history", async () => {
  const { root, handle } = await panel();
  handle(call({ turn_id: "t1", call_id: "t1-c1" }));
  handle(call({ turn_id: "t1", call_id: "t1-c2", name: "scroll", arguments: { direction: "up" } }));
  const kept = card(root, "t1-c1");
  handle(call({ turn_id: "t2", call_id: "t2-c1", name: "scroll", arguments: { direction: "down" } }));
  assert.deepEqual(ids(root), ["t1-c1", "t1-c2", "t2-c1"], "B7/H-T128-3: earlier turns keep their cards, in arrival order");
  assert.equal(card(root, "t1-c1"), kept, "…the history card is the same element");
  assert.ok(lineOf(kept).startsWith("Would click at (120, 845)"), "B7: toolLine copy kept on the history card");
});

pendingT0154B("T-0154 B7 (amends U7): a late message for a superseded turn is ignored — its history card is not updated, nothing added", async () => {
  const { root, handle } = await panel();
  handle(call({ turn_id: "t1", call_id: "t1-c1" }));
  handle(call({ turn_id: "t2", call_id: "t2-c1", name: "scroll", arguments: { direction: "down" } }));
  const t1 = card(root, "t1-c1");
  const t2 = card(root, "t2-c1");
  handle(call({ turn_id: "t1", call_id: "t1-c1", status: "cancelled", reason: "interrupted" }));
  handle(call({ turn_id: "t1", call_id: "t1-c9", name: "press_key", arguments: { combo: "Ctrl+L" } }));
  assert.deepEqual(ids(root), ["t1-c1", "t2-c1"], "U7: no t1-c9; t1's card stays as history; nothing moved");
  assert.equal(card(root, "t1-c1"), t1);
  assert.equal(statusOf(t1), "proposed", "U7/U10: the superseded turn's late status is ignored");
  assert.equal(card(root, "t2-c1"), t2, "t2's card is the same element");
  assert.equal(statusOf(t2), "proposed", "…untouched");
  assert.equal(lineOf(t2), "Would scroll down");
});

pendingT0154B("T-0154 B7 (amends U8): nine turns keep nine cards; late messages for the eight superseded ones change none", async () => {
  const { root, handle } = await panel();
  for (let n = 1; n <= 9; n++) handle(call({ turn_id: `t${n}`, call_id: `t${n}-c1` }));
  const want = Array.from({ length: 9 }, (_v, i) => `t${i + 1}-c1`);
  assert.deepEqual(ids(root), want, "B7: every turn keeps its card");
  for (let n = 1; n <= 8; n++) {
    handle(call({ turn_id: `t${n}`, call_id: `t${n}-c1`, status: "cancelled" }));
    handle(call({ turn_id: `t${n}`, call_id: `t${n}-c2` }));
    assert.deepEqual(ids(root), want, `U8: a late message for superseded t${n} neither adds a card nor removes one`);
    assert.equal(statusOf(card(root, `t${n}-c1`)), "proposed", `U8: t${n}'s card is not updated by its late message`);
  }
  assert.equal(statusOf(card(root, "t9-c1")), "proposed", "t9's card is untouched");
});

pendingT0154B("T-0154 B7 (amends U10): sequence A — t1, t2, t3 cards, then a late t2 message: all three stay, t2 unchanged, t3 current", async () => {
  const { root, handle } = await panel();
  handle(call({ turn_id: "t1", call_id: "t1-c1" }));
  handle(call({ turn_id: "t2", call_id: "t2-c1", name: "press_key", arguments: { combo: "Ctrl+L" } }));
  handle(call({ turn_id: "t3", call_id: "t3-c1", name: "scroll", arguments: { direction: "down" } }));
  const t2 = card(root, "t2-c1");
  const t3 = card(root, "t3-c1");
  handle(call({ turn_id: "t2", call_id: "t2-c1", status: "cancelled", reason: "interrupted" }));
  assert.deepEqual(ids(root), ["t1-c1", "t2-c1", "t3-c1"], "U10/B7: no card cleared, none reappears out of order");
  assert.equal(card(root, "t2-c1"), t2);
  assert.equal(statusOf(t2), "proposed", "U10: t2 is superseded; its late cancelled is ignored");
  assert.equal(lineOf(t2), "Would press Ctrl+L");
  assert.equal(card(root, "t3-c1"), t3);
  handle(call({ turn_id: "t3", call_id: "t3-c1", status: "dry_run" }));
  assert.equal(statusOf(card(root, "t3-c1")), "dry_run", "…and t3 is still the current turn");
});

test("T-0090 review (U7): a message for an already-shown call_id updates that card in place", async () => {
  const { root, handle } = await panel();
  handle(call({ turn_id: "t2", call_id: "t2-c1" }));
  const el = card(root, "t2-c1");
  handle(call({ turn_id: "t2", call_id: "t2-c1", status: "dry_run" }));
  assert.equal(card(root, "t2-c1"), el, "the same element, not a rebuilt card");
  assert.equal(statusOf(el), "dry_run");
});
