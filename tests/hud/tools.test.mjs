// T-0162 group B — the Tools pane (docs/tasks/T-0162.md B-1 .. B-5), the node half.
//
// Every row is derived from the acceptance criteria, never from an implementation (none exists
// when this file is written). The Chrome half (placement in the rail, the gear, the badge, no
// egress) is tests/integration/test_hud_tools_pane.py.
//
// Gate. The rows run through `pendingT0162B` — the same Status-phrase strict-xfail gate as
// `pendingT0159B` in tests/hud/denoise.test.mjs, reading docs/tasks/T-0162.md. While its
// **Status:** line does not record "group B: tests <a>..<b>, activated <sha>", each row must FAIL
// (an XPASS fails the row: a stub that satisfies it is vacuous). ASTRA_RUN_PENDING=1 runs them for
// real now. Group A's (or C's, D's) activation does not switch these rows on.
//
// API these rows pin (the task leaves the JS names to the test-author; these are them):
//
//   hud/src/tools/model.js — DOM-free, importable in node
//     shortArgs(name, args) -> string                                       (B-2)
//       click_at     {x, y}                     -> 'click_at (x, y)'
//       type_at      {x, y, text, press_enter}  -> 'type_at "<text>" at (x, y)' + ' + Enter' iff
//                                                  press_enter === true; <text> truncated at 40
//                                                  code points
//       scroll       {direction}                -> 'scroll <direction>'
//       press_key    {combo}                    -> 'press_key <combo>'
//       google_search {query}                   -> 'google_search "<query>"', truncated at 60
//       page_overview {}                        -> 'page_overview'
//       anything else                           -> '<name> <JSON.stringify(args)>', the JSON
//                                                  truncated at 80 code points
//       A truncated part is the first N code points, optionally followed by "…" (U+2026). The
//       task does not say whether a marker is shown; both readings pass, nothing else does.
//
//   hud/src/tools/index.js — the DOM half
//     mount(host)  host = { document, subscribe, ... } exactly as hud/src/main.js builds it.
//       Never throws (T-0008 criterion 10), also for host = {}.
//       Appends one <section id="tools-panel" class="pane"> to <main>, whose <h2> reads "Tools".
//     Inside #tools-panel:
//       [data-tools-turn="<turn_id>"]   one group per turn that has calls, newest turn first (B-1)
//       [data-tools-call="<call_id>"]   one per call_id, a native <details>, in arrival order in
//                                       its turn's group (B-1, B-4)
//         [data-tools-line]             the short form, shortArgs(name, arguments) (B-1, B-2)
//         [data-tools-status="<wire>"]  the status; its text is the label: proposed, repairing,
//                                       dry run, executed, refused, cancelled (B-3)
//         [data-tools-detail]           every argument as "key: value", the source ("from the
//                                       draft" / "repaired"), action, reason, query_problem (B-4)
//         [data-tools-statuses]         the labels of the statuses seen, in order (B-4)
//     "Newest turn" is the turn whose astra.turn.state or astra.tool.call arrived most recently
//     for the first time (the order turns begin).
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);

// --- the T-0162 group B strict-xfail gate --------------------------------------------------------

const TASK = fileURLToPath(new URL("docs/tasks/T-0162.md", REPO));
const STATUS = existsSync(TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(TASK, "utf8")) || [])[1] || "").trim()
  : "";
/** "group B: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6" — the phrase the B test-author writes. */
const B_ACTIVATED = /\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const B_PENDING = !B_ACTIVATED.test(STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0162B(name, fn) {
  test(name + (B_PENDING ? " [pending T-0162 B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0162 group B row passed while docs/tasks/T-0162.md's status does " +
          "not record group B's activation (" + JSON.stringify(STATUS.slice(0, 80)) + "). Either " +
          "group B has landed (record 'group B: tests …, activated <sha>' in the status) or the " +
          "row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0162 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0162 B gate: the status line is readable and only group B's own phrase activates the rows", () => {
  assert.ok(STATUS.length > 0, "docs/tasks/T-0162.md must carry a **Status:** line");
  assert.equal(B_ACTIVATED.test("in progress 2026-10-26 (James's requests ~21:45; rulings below) · **Blast radius:** A high · B medium"), false);
  assert.equal(B_ACTIVATED.test("in progress (group A: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6)"), false);
  assert.equal(B_ACTIVATED.test("in progress (group A: …, activated 9a8b7c6; group B: tests a1b2c3d..e4f5a6b, activated c3d4e5f)"), true);
  assert.equal(B_ACTIVATED.test("in progress (group B: tests a1b2c3d..e4f5a6b, not yet activated)"), false);
  assert.equal(B_ACTIVATED.test("in progress (group B tests written; group D: tests …, activated 1234abc)"), false);
});

// --- loading the modules --------------------------------------------------------------------------

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

// Origin proof (.claude/rules/tests.md): print where the modules under test were imported from.
console.log(`# tools pane origin: repo=${fileURLToPath(REPO)} modules=${srcRoot().href}`);

async function load(name) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch (err) {
    assert.fail(
      `T-0162 B adds hud/src/${name}, importable in node as an ES module with no DOM at module ` +
        `top level (Owned files: "hud/src/tools/model.js — shortArgs, grouping and status ` +
        `tracking (DOM-free)"). node could not import it: ${err.message}`,
    );
  }
}

async function shortArgsFn() {
  const mod = await load("tools/model.js");
  assert.equal(typeof mod.shortArgs, "function", "B-2: tools/model.js exports shortArgs(name, args)");
  return mod.shortArgs;
}

// --- a DOM double ----------------------------------------------------------------------------------
//
// The same shape as tests/hud/tool_call.test.mjs's double, plus a document tree (html > head, body >
// main) so a pane can find <main>. Markup sinks throw and are recorded, so "set with textContent"
// (B-2) is a behaviour, not a grep. A name the double lacks throws a named error: extend the double.

const MARKUP_SINKS = ["innerHTML", "outerHTML"];

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
      throw new Error("B-2: model strings are set with textContent only (document.write)");
    },
    querySelector(sel) {
      return doc.documentElement.querySelector(sel);
    },
    querySelectorAll(sel) {
      return doc.documentElement.querySelectorAll(sel);
    },
    getElementById(id) {
      return doc.documentElement.querySelector("#" + id);
    },
    addEventListener() {},
    removeEventListener() {},
  };
  doc.documentElement = makeElement(doc, "html");
  doc.head = makeElement(doc, "head");
  doc.body = makeElement(doc, "body");
  doc.main = makeElement(doc, "main");
  doc.documentElement.append(doc.head, doc.body);
  doc.body.appendChild(doc.main);
  created.length = 0; // the scaffold is not the pane's
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
    open: false,
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
      if (name === "open") this.open = true;
      if (name.startsWith("data-")) dataset[camel(name)] = String(value);
      attrs.set(name, String(value));
    },
    getAttribute(name) {
      name = String(name).toLowerCase();
      if (name === "class") return classes.size ? this.className : null;
      if (name === "id") return this.id || null;
      if (name.startsWith("data-")) {
        const v = dataset[camel(name)];
        return v === undefined ? null : v;
      }
      if (name === "hidden") return this.hidden ? "" : null;
      if (name === "open") return this.open ? "" : null;
      return attrs.has(name) ? attrs.get(name) : null;
    },
    hasAttribute(name) {
      return this.getAttribute(name) !== null;
    },
    removeAttribute(name) {
      name = String(name).toLowerCase();
      if (name.startsWith("data-")) delete dataset[camel(name)];
      if (name === "hidden") this.hidden = false;
      if (name === "open") this.open = false;
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
        throw new Error(`B-2: model strings are set with textContent only (${sink} was set)`);
      },
    });
  }
  target.insertAdjacentHTML = () => {
    doc.violations.push("insertAdjacentHTML");
    throw new Error("B-2: model strings are set with textContent only (insertAdjacentHTML)");
  };
  const proxy = new Proxy(target, {
    get(obj, key, receiver) {
      if (typeof key === "symbol" || key in obj) return Reflect.get(obj, key, receiver);
      if (key === "then" || key === "toJSON" || key === "nodeValue") return undefined;
      throw new Error(
        `the DOM double in tests/hud/tools.test.mjs has no Element.${key}; extend the double ` +
          "(this is a gap in the test, not a defect in the pane)",
      );
    },
    set(obj, key, value) {
      if (key === "href" || key === "src" || key === "onclick" || key === "onerror") {
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

// --- helpers ---------------------------------------------------------------------------------------

const squeeze = (s) => String(s).replace(/\s+/g, " ").trim();

/** Mount the pane on a fresh document; returns the pane element and a `feed(message)`. */
async function pane() {
  const { mount } = await load("tools/index.js");
  assert.equal(typeof mount, "function", "B-1: tools/index.js exports mount(host)");
  const doc = fakeDocument();
  const subs = [];
  await mount({
    document: doc,
    subscribe: (fn) => (subs.push(fn), () => {}),
    send: () => Promise.resolve(),
    log: () => {},
  });
  const found = doc.main.children.filter((c) => c.id === "tools-panel");
  assert.equal(found.length, 1, "B-1: mount() appends one #tools-panel as a direct child of <main>");
  assert.ok(subs.length >= 1, "B-1: the pane subscribes to the inbound stream");
  const feed = (m) => {
    for (const fn of subs) fn(m);
  };
  return { doc, el: found[0], feed };
}

const call = (over = {}) => ({
  type: "astra.tool.call",
  turn_id: "t1",
  call_id: "t1-c1",
  name: "click_at",
  arguments: { x: 120, y: 845 },
  status: "proposed",
  source: "draft",
  action: null,
  reason: null,
  query_problem: null,
  ...over,
});
const turnState = (turn_id, state = "THINKING") => ({ type: "astra.turn.state", state, turn_id, thinking_ms: null });

const groups = (el) => el.querySelectorAll("[data-tools-turn]").map((g) => g.getAttribute("data-tools-turn"));
const callsIn = (g) => g.querySelectorAll("[data-tools-call]").map((c) => c.getAttribute("data-tools-call"));
function callEl(el, id) {
  const found = el.querySelectorAll("[data-tools-call]").filter((c) => c.getAttribute("data-tools-call") === id);
  assert.equal(found.length, 1, `B-1: exactly one line per call_id; ${found.length} for ${id}`);
  return found[0];
}
function lineOf(c) {
  const l = c.querySelectorAll("[data-tools-line]");
  assert.equal(l.length, 1, "B-1: one [data-tools-line] per call");
  return squeeze(l[0].textContent);
}
function statusOf(c) {
  const s = c.querySelector("[data-tools-status]");
  assert.ok(s, "B-1: each call shows its status ([data-tools-status])");
  return { wire: s.getAttribute("data-tools-status"), label: squeeze(s.textContent) };
}
function detailOf(c) {
  const d = c.querySelector("[data-tools-detail]");
  assert.ok(d, "B-4: each call has a [data-tools-detail] block");
  return squeeze(d.textContent);
}

/** The first n code points of s (an independent oracle: Array.from splits by code point). */
const firstCP = (s, n) => Array.from(s).slice(0, n).join("");

/** `got` is `head` exactly, or `head` + "…" (the task does not say whether a marker is shown). */
function assertTruncatedTo(got, head, what) {
  assert.ok(got === head || got === head + "…", `${what}: expected ${JSON.stringify(head)} (optionally + "…"), got ${JSON.stringify(got)}`);
}

// --- B-2: short forms -----------------------------------------------------------------------------

pendingT0162B("B-2: click_at, scroll, press_key and page_overview short forms", async () => {
  const shortArgs = await shortArgsFn();
  assert.equal(shortArgs("click_at", { x: 120, y: 845 }), "click_at (120, 845)");
  assert.equal(shortArgs("scroll", { direction: "down" }), "scroll down");
  assert.equal(shortArgs("scroll", { direction: "up" }), "scroll up");
  assert.equal(shortArgs("press_key", { combo: "ctrl+l" }), "press_key ctrl+l");
  assert.equal(shortArgs("page_overview", {}), "page_overview");
});

pendingT0162B("B-2: type_at shows the quoted text, at (x, y), and + Enter only when press_enter is true", async () => {
  const shortArgs = await shortArgsFn();
  assert.equal(shortArgs("type_at", { x: 500, y: 300, text: "hello", press_enter: true }), 'type_at "hello" at (500, 300) + Enter');
  assert.equal(shortArgs("type_at", { x: 500, y: 300, text: "hello", press_enter: false }), 'type_at "hello" at (500, 300)');
});

pendingT0162B("B-2 (adversarial): type_at truncates at 40 code points, never inside a surrogate pair", async () => {
  const shortArgs = await shortArgsFn();
  // 45 astral code points: a UTF-16 slice at 40 units would keep 20 emoji and could split a pair.
  const text = "😀".repeat(39) + "🦊" + "ABCDE";
  const got = shortArgs("type_at", { x: 1, y: 2, text, press_enter: false });
  const m = /^type_at "(.*)" at \(1, 2\)$/su.exec(got);
  assert.ok(m, `B-2: the type_at form; got ${JSON.stringify(got)}`);
  assertTruncatedTo(m[1], firstCP(text, 40), "B-2 type_at text at 40 code points");
  assert.ok(!/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/.test(got), "no lone surrogate");
  // exactly 40 code points is not truncated
  const forty = "a".repeat(40);
  assert.equal(shortArgs("type_at", { x: 1, y: 2, text: forty, press_enter: false }), `type_at "${forty}" at (1, 2)`);
});

pendingT0162B("B-2: google_search quotes the query, truncated at 60 code points", async () => {
  const shortArgs = await shortArgsFn();
  assert.equal(shortArgs("google_search", { query: "weather in Lyon" }), 'google_search "weather in Lyon"');
  const q = "q".repeat(59) + "Z" + "tail-that-must-not-show";
  const m = /^google_search "(.*)"$/su.exec(shortArgs("google_search", { query: q }));
  assert.ok(m, "B-2: the google_search form");
  assertTruncatedTo(m[1], "q".repeat(59) + "Z", "B-2 query at 60 code points");
});

pendingT0162B("B-2: any other tool is its name plus compact JSON, truncated at 80", async () => {
  const shortArgs = await shortArgsFn();
  assert.equal(shortArgs("open_url", { url: "http://127.0.0.1/a" }), 'open_url {"url":"http://127.0.0.1/a"}');
  const args = { note: "n".repeat(200) };
  const json = JSON.stringify(args); // compact: no spaces
  const got = shortArgs("annotate", args);
  assert.ok(got.startsWith("annotate "), `B-2: the name first; got ${JSON.stringify(got)}`);
  assertTruncatedTo(got.slice("annotate ".length), firstCP(json, 80), "B-2 compact JSON at 80 code points");
});

// --- B-1, B-3: the pane, grouping, order, status --------------------------------------------------

pendingT0162B("B-1: mount() appends a 'Tools' pane to <main> and survives a host with no document", async () => {
  const { mount } = await load("tools/index.js");
  await mount({}); // T-0008 criterion 10: never throws
  await mount(undefined);
  const { el } = await pane();
  const h = el.querySelector("h2");
  assert.ok(h, "B-1: the pane has an <h2>");
  assert.equal(squeeze(h.textContent), "Tools", "B-1: a rail pane headed \"Tools\"");
  assert.ok(el.classList.contains("pane"), "B-1: a rail pane carries class pane (panes self-append to <main>)");
});

pendingT0162B("B-1: one line per call_id, grouped per turn, newest turn first, arrival order within", async () => {
  const { el, feed } = await pane();
  feed(turnState("t1"));
  feed(call({ turn_id: "t1", call_id: "t1-c1" }));
  feed(call({ turn_id: "t1", call_id: "t1-c2", name: "scroll", arguments: { direction: "down" } }));
  feed(call({ turn_id: "t1", call_id: "t1-c1", status: "dry_run" })); // same id: updated, not added
  feed(turnState("t2"));
  feed(call({ turn_id: "t2", call_id: "t2-c9", name: "press_key", arguments: { combo: "Enter" } }));
  feed(call({ turn_id: "t2", call_id: "t2-c3", name: "page_overview", arguments: {} }));
  assert.deepEqual(groups(el), ["t2", "t1"], "B-1: grouped per turn, newest turn first");
  const [g2, g1] = el.querySelectorAll("[data-tools-turn]");
  assert.deepEqual(callsIn(g1), ["t1-c1", "t1-c2"], "B-1: arrival order within a turn, one line per call_id");
  assert.deepEqual(callsIn(g2), ["t2-c9", "t2-c3"], "B-1: arrival order, not call_id order");
  assert.equal(lineOf(callEl(el, "t1-c2")), "scroll down");
  assert.equal(lineOf(callEl(el, "t2-c9")), "press_key Enter");
  assert.deepEqual(statusOf(callEl(el, "t1-c1")), { wire: "dry_run", label: "dry run" }, "the line updates in place");
});

pendingT0162B("B-1: a call for a turn with no astra.turn.state yet still gets its own group", async () => {
  const { el, feed } = await pane();
  feed(call({ turn_id: "t5", call_id: "a" }));
  feed(call({ turn_id: "t6", call_id: "b" }));
  assert.deepEqual(groups(el), ["t6", "t5"]);
});

pendingT0162B("B-3: the six wire statuses are shown verbatim, dry_run as 'dry run'", async () => {
  const { el, feed } = await pane();
  const want = { proposed: "proposed", repairing: "repairing", dry_run: "dry run", executed: "executed", refused: "refused", cancelled: "cancelled" };
  for (const status of Object.keys(want)) feed(call({ call_id: "c-" + status, status }));
  for (const [status, label] of Object.entries(want)) {
    assert.deepEqual(statusOf(callEl(el, "c-" + status)), { wire: status, label }, `B-3: ${status}`);
  }
  // B-3: failed and timeout arrive with C from a real field; B never derives them from wording.
  feed(call({ call_id: "c-w", status: "refused", reason: "the action failed: timeout while loading" }));
  assert.deepEqual(statusOf(callEl(el, "c-w")), { wire: "refused", label: "refused" }, "B-3: never derived from wording");
});

// --- B-4: expandable detail -----------------------------------------------------------------------

pendingT0162B("B-4: a native <details> per call with arguments, source, action, reason, query_problem and the statuses seen", async () => {
  const { el, feed } = await pane();
  const args = { x: 512, y: 384, text: "pancakes", press_enter: true };
  feed(call({ call_id: "k", name: "type_at", arguments: args, status: "proposed", source: "draft" }));
  feed(call({ call_id: "k", name: "type_at", arguments: args, status: "repairing", source: "repair", query_problem: "qp-marker" }));
  feed(call({ call_id: "k", name: "type_at", arguments: args, status: "refused", source: "repair", action: "act-marker", reason: "why-marker", query_problem: "qp-marker" }));
  const c = callEl(el, "k");
  assert.equal(c.localName, "details", "B-4: a native <details> per call");
  const summary = c.children.find((x) => x.localName === "summary");
  assert.ok(summary, "B-4: the <details> has a <summary>");
  assert.ok(summary.contains(c.querySelector("[data-tools-line]")), "B-4: the short line is in the summary");
  assert.equal(lineOf(c), 'type_at "pancakes" at (512, 384) + Enter');
  const detail = detailOf(c);
  for (const kv of ["x: 512", "y: 384", "text: pancakes", "press_enter: true"]) assert.ok(detail.includes(kv), `B-4: "${kv}" in ${JSON.stringify(detail)}`);
  assert.ok(detail.includes("repaired"), "B-4: the source of the latest message (repaired)");
  for (const marker of ["act-marker", "why-marker", "qp-marker"]) assert.ok(detail.includes(marker), `B-4: ${marker}`);
  const seen = squeeze(c.querySelector("[data-tools-statuses]").textContent);
  const at = ["proposed", "repairing", "refused"].map((s) => seen.indexOf(s));
  assert.ok(at.every((i) => i >= 0) && at[0] < at[1] && at[1] < at[2], `B-4: the statuses seen, in order; got ${JSON.stringify(seen)}`);
});

pendingT0162B("B-4: a draft source reads 'from the draft'", async () => {
  const { el, feed } = await pane();
  feed(call({ call_id: "d", source: "draft" }));
  assert.ok(detailOf(callEl(el, "d")).includes("from the draft"));
});

// --- B-5 (adversarial): no thinking, and model strings are text ---------------------------------

pendingT0162B("B-5 (adversarial): a thought commit and a canvas-step piece never reach the pane", async () => {
  const { el, feed, doc } = await pane();
  const CANARY = "CANARY-thought-7f3e";
  feed(turnState("t1"));
  feed({ type: "astra.canvas.commit", request_id: "r1", canvas_idx: 0, segment: "thought", text: `I should click ${CANARY}`, steps_used: 4 });
  feed({
    type: "astra.canvas.step", request_id: "r1", canvas_idx: 0, step: 1, max_steps: 4, canvas_length: 2,
    canvas: [1, 2], argmax: [1, 2], phase: "denoise", segment: "thought", text_preview: CANARY,
    argmax_pieces: [CANARY, "x"], committed_pieces: [CANARY, "y"],
  });
  feed({ type: "astra.spoken", turn_id: "t1", utterance_id: "u1", text: "Sure." });
  // A T-0135 from_thought call: the pane shows the call, never the thought around it.
  feed(call({ call_id: "t1-c1", name: "click_at", arguments: { x: 3, y: 4 } }));
  assert.equal(lineOf(callEl(el, "t1-c1")), "click_at (3, 4)", "premise: the call itself is shown");
  assert.ok(!el.textContent.includes(CANARY), "B-5: no thinking text in the pane's textContent");
  assert.ok(!doc.created.some((e) => e.textContent.includes(CANARY)), "B-5: …nor in any element the pane created");
});

pendingT0162B("B-2 (adversarial): an <img onerror> in an argument is shown as text and creates no element", async () => {
  const { el, feed, doc } = await pane();
  const EVIL = "<img src=x onerror=window.__pwned=1>";
  feed(call({ call_id: "e", name: "google_search", arguments: { query: EVIL }, status: "dry_run", action: `would search for: ${EVIL}` }));
  assert.equal(lineOf(callEl(el, "e")), `google_search "${EVIL}"`, "B-2: the markup is literal text");
  assert.ok(detailOf(callEl(el, "e")).includes(EVIL), "B-4: …in the detail too");
  assert.deepEqual(doc.violations, [], "B-2: no markup sink was used");
  assert.ok(!doc.created.some((e) => e.localName === "img"), "B-2: no <img> element was created");
});

pendingT0162B("B-1: messages other than astra.tool.call and astra.turn.state add nothing to the pane", async () => {
  const { el, feed } = await pane();
  const before = el.textContent;
  feed({ type: "astra.assist.state", flags: { page_text: true, pdf_text: false, cells: false, at_spi: false }, badge: "pixels + page text" });
  feed({ type: "astra.transcript", turn_id: "t1", text: "hello there", final: true });
  feed({ type: "astra.error", code: "x", message: "boom" });
  // (astra.turn.state is left out: whether a turn with no calls shows an empty group is not specified.)
  assert.equal(el.textContent, before, "B-1: the pane's only content input is astra.tool.call (+ turn.state for grouping)");
  assert.equal(el.querySelectorAll("[data-tools-call]").length, 0);
});

// =================================================================================================
// T-0162 group C — C-3: the outcome and the one-line result (docs/tasks/T-0162.md C-1, C-3)
// =================================================================================================
//
// Gate. `pendingT0162C`: strict xfail until docs/tasks/T-0162.md's **Status:** records
// "group C: tests <a>..<b>, activated <sha>" (group B's or D's activation does not count).
// `ASTRA_RUN_PENDING=1 node --test tests/hud/tools.test.mjs` runs them for real now.
//
// API these rows add (the task leaves the JS names to the test-author; these are them):
//
//   hud/src/tools/model.js
//     shortResult(message) -> string | null    the C-3 one-line result of an astra.tool.call;
//                                              null when the message carries no `result`
//       browser tool (click_at, type_at, scroll, press_key, page_overview) with hosts:
//         '<action> · reached h1, h2'           + ' +<hosts_more>' when hosts_more > 0
//       browser tool without hosts:            '<action>'
//       google_search: results > 0              '<N> results (h1, h2, h3)'  ("1 result (h)" or
//                                              "1 results (h)" for one: the task writes "N results"
//                                              and T-0125's card writes the singular; both pass)
//       google_search: results 0 or null, outcome executed: 'no results'
//       any other tool:                         '<action>'
//     OUTCOME_LABELS: the six outcomes' labels; dry_run reads "dry run", the others verbatim.
//
//   hud/src/tools/index.js, inside each [data-tools-call]:
//     [data-tools-status]   its text is the label of result.outcome when present, else of the
//                           wire status (C-3); the attribute's value is not pinned by group C
//     [data-tools-result]   one element, text = shortResult(message), set with textContent; absent
//                           or empty while the call has no result
//
//       google_search whose outcome is not executed (failed, timeout, cancelled, refused):
//                                              '<action>' (ruling 2026-10-27: "a failed/refused/
//                                              cancelled search shows its action as the result
//                                              line"), never 'no results'
//
// Not pinned (ambiguous in the task, listed in the report): whether a search line shows
// hosts_more; a browser action with a null action.

const C_ACTIVATED = /\bgroup C:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const C_PENDING = !C_ACTIVATED.test(STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0162C(name, fn) {
  test(name + (C_PENDING ? " [pending T-0162 C: strict xfail until group C is activated]" : ""), async (t) => {
    if (!C_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0162 group C row passed while docs/tasks/T-0162.md's status does " +
          "not record group C's activation. Either group C has landed (record 'group C: tests …, " +
          "activated <sha>' in the status) or the row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0162 C pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0162 C gate: only group C's own phrase activates the C rows", () => {
  assert.equal(C_ACTIVATED.test("in progress (group B: tests 1697b85b..42d62069, activated a1d0db53; group D: tests 7d570443..9b5f9d1e, activated 9b5f9d1e)"), false);
  assert.equal(C_ACTIVATED.test("in progress (group D: …, activated 9b5f9d1e; group C: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6)"), true);
  assert.equal(C_ACTIVATED.test("in progress (group C: tests 1a2b3c4..5d6e7f8, not yet activated)"), false);
  assert.equal(C_ACTIVATED.test("C rows: group C tests written; group A: tests …, activated 1234abc"), false);
});

async function shortResultFn() {
  const mod = await load("tools/model.js");
  assert.equal(typeof mod.shortResult, "function", "C-3: tools/model.js exports shortResult(message)");
  return mod.shortResult;
}

const res = (over = {}) => ({ outcome: "executed", hosts: [], hosts_more: 0, ...over });
const finalCall = (over = {}) =>
  call({ call_id: "f", status: "executed", action: "clicked at (640, 180)", step: 1, result: res(), ...over });

function resultOf(c) {
  const r = c.querySelectorAll("[data-tools-result]");
  assert.equal(r.length, 1, `C-3: one [data-tools-result] per call with a result; ${r.length}`);
  return squeeze(r[0].textContent);
}

/** Accepts "1 result (…)" or "1 results (…)" (see the header). */
function assertOneResult(got, hosts) {
  assert.ok(got === `1 result (${hosts})` || got === `1 results (${hosts})`, `C-3: one result; got ${JSON.stringify(got)}`);
}

// --- shortResult (pure) ----------------------------------------------------------------------------

pendingT0162C("C-3: a browser action with hosts reads '<action> · reached h1, h2', with '+n' beyond", async () => {
  const shortResult = await shortResultFn();
  assert.equal(
    shortResult(finalCall({ result: res({ hosts: ["shop.example", "cdn.shop.example"] }) })),
    "clicked at (640, 180) · reached shop.example, cdn.shop.example",
  );
  assert.equal(
    shortResult(finalCall({ result: res({ hosts: ["a.example", "b.example", "c.example"], hosts_more: 4 }) })),
    "clicked at (640, 180) · reached a.example, b.example, c.example +4",
  );
  // a failed live action still reports where it went (T-0160 B-6: a guarded outcome carries hosts)
  assert.equal(
    shortResult(
      finalCall({
        name: "type_at",
        arguments: { x: 1, y: 2, text: "hunter2", press_enter: false },
        status: "refused",
        action: "failed type_at: password field",
        reason: "password field",
        result: res({ outcome: "failed", hosts: ["login.example"] }),
      }),
    ),
    "failed type_at: password field · reached login.example",
  );
});

pendingT0162C("C-3: a browser action without hosts, and any other tool, read as the action", async () => {
  const shortResult = await shortResultFn();
  assert.equal(shortResult(finalCall()), "clicked at (640, 180)");
  assert.equal(
    shortResult(finalCall({ name: "scroll", arguments: { direction: "down" }, status: "dry_run", action: "would scroll down", result: res({ outcome: "dry_run" }) })),
    "would scroll down",
  );
  assert.equal(
    shortResult(finalCall({ name: "annotate", arguments: { note: "n" }, action: "noted", result: res() })),
    "noted",
  );
});

pendingT0162C("C-3: a search reads 'N results (h1, h2, h3)' or 'no results', from the fields only", async () => {
  const shortResult = await shortResultFn();
  const search = (result, action) =>
    finalCall({ name: "google_search", arguments: { query: "weather in Paris" }, action, result });
  assert.equal(
    shortResult(search(res({ hosts: ["a.example", "b.example", "c.example"], hosts_more: 2, results: 7 }), "searched")),
    "7 results (a.example, b.example, c.example)",
  );
  assert.equal(shortResult(search(res({ results: 0 }), "searched")), "no results");
  assert.equal(shortResult(search(res({ results: null }), "searched")), "no results");
  assert.equal(shortResult(search(res({}), "searched")), "no results", "results absent (exclude_none) is no results");
  assertOneResult(shortResult(search(res({ hosts: ["one.example"], results: 1 }), "searched")), "one.example");
  // never from the card's prose: the action text says 9 results at evil.example; the fields say 2
  assert.equal(
    shortResult(search(res({ hosts: ["a.example", "b.example"], results: 2 }), 'searched the web for "x": 9 results (evil.example)')),
    "2 results (a.example, b.example)",
  );
});

pendingT0162C("C-3 (ruling 2026-10-27): a search that did not execute shows its action, never 'no results'", async () => {
  const shortResult = await shortResultFn();
  for (const outcome of ["failed", "timeout", "cancelled", "refused"]) {
    const status = outcome === "refused" ? "refused" : "cancelled";
    const msg = finalCall({
      name: "google_search",
      arguments: { query: "weather in Paris" },
      status,
      action: "search failed",
      result: res({ outcome, results: null }),
    });
    assert.equal(shortResult(msg), "search failed", `C-3: a ${outcome} search reads as its action`);
  }
  const { el, feed } = await pane();
  feed(call({ call_id: "sf", name: "google_search", arguments: { query: "q" }, status: "cancelled", reason: "timeout", action: "search failed", step: 1, result: res({ outcome: "timeout", results: null }) }));
  assert.equal(resultOf(callEl(el, "sf")), "search failed", "C-3: the pane's result line for a timed-out search");
});

pendingT0162C("C-3: no result line before the final message (shortResult is null without a result)", async () => {
  const shortResult = await shortResultFn();
  assert.equal(shortResult(call({ status: "proposed" })), null);
  assert.equal(shortResult(call({ status: "repairing", step: 2 })), null);
});

// --- the pane --------------------------------------------------------------------------------------

pendingT0162C("C-3: the status shows result.outcome when present (failed, timeout), else the wire status", async () => {
  const { el, feed } = await pane();
  feed(call({ call_id: "a", status: "proposed", step: 1 }));
  assert.equal(statusOf(callEl(el, "a")).label, "proposed", "premise: the wire status before the final");
  feed(call({ call_id: "a", status: "refused", reason: "password field", action: "failed click_at: x", step: 1, result: res({ outcome: "failed" }) }));
  assert.equal(statusOf(callEl(el, "a")).label, "failed", "C-3: a live failure reads failed (the wire says refused)");
  feed(call({ call_id: "b", name: "google_search", arguments: { query: "q" }, status: "cancelled", reason: "timeout", step: 1, result: res({ outcome: "timeout" }) }));
  assert.equal(statusOf(callEl(el, "b")).label, "timeout", "C-3: a deadline reads timeout (the wire says cancelled)");
  feed(call({ call_id: "c", status: "dry_run", action: "would click", step: 1, result: res({ outcome: "dry_run" }) }));
  assert.equal(statusOf(callEl(el, "c")).label, "dry run");
  for (const outcome of ["executed", "refused", "cancelled"]) {
    feed(call({ call_id: "o-" + outcome, status: outcome === "executed" ? "executed" : outcome, action: "x", step: 1, result: res({ outcome }) }));
    assert.equal(statusOf(callEl(el, "o-" + outcome)).label, outcome, `C-3: ${outcome}`);
  }
  // the too-wide direction: no result, a reason that reads like a timeout -> still the wire status
  feed(call({ call_id: "w", status: "cancelled", reason: "timeout" }));
  assert.equal(statusOf(callEl(el, "w")).label, "cancelled", "C-3: never derived from wording");
});

pendingT0162C("C-3 (reviewer A T4): data-tools-status carries result.outcome, the hook the failed/timeout colours key on", async () => {
  const { el, feed } = await pane();
  feed(call({ call_id: "f", status: "refused", action: "failed click_at: x", reason: "x", step: 1, result: res({ outcome: "failed" }) }));
  assert.equal(statusOf(callEl(el, "f")).wire, "failed", "T4: a refused card whose outcome is failed");
  feed(call({ call_id: "t", status: "cancelled", reason: "timeout", step: 1, result: res({ outcome: "timeout" }) }));
  assert.equal(statusOf(callEl(el, "t")).wire, "timeout", "T4: a cancelled card whose outcome is timeout");
  feed(call({ call_id: "p", status: "proposed", step: 1 }));
  assert.equal(statusOf(callEl(el, "p")).wire, "proposed", "T4: no result -> the wire status");
});

pendingT0162C("C (privacy amendment): after a masked final, neither the pane nor the inline card shows the typed text", async () => {
  const SECRET = "hunter2-ZQX";
  const MASK = "•••";
  const args = (text) => ({ x: 500, y: 250, text, press_enter: false });
  const msgs = [
    call({ call_id: "s", name: "type_at", arguments: args(SECRET), status: "proposed", step: 1 }),
    call({
      call_id: "s", name: "type_at", arguments: args(MASK), status: "refused", step: 1,
      action: "failed type_at: password field", reason: "will not type into a password or card field; ask the user to type it themselves",
      result: res({ outcome: "failed" }),
    }),
  ];
  const { el, feed, doc } = await pane();
  for (const m of msgs) feed(m);
  assert.ok(el.textContent.includes(MASK), "premise: the pane shows the masked text");
  assert.ok(!el.textContent.includes(SECRET), `the Tools pane drops the original text after the masked final; ${el.textContent}`);
  // the inline card (hud/src/panels/tool_call.js) over the same document double
  let panelMod;
  try {
    panelMod = await import(new URL("panels/tool_call.js", srcRoot()).href);
  } catch (err) {
    assert.fail(`hud/src/panels/tool_call.js must import in node: ${err.message}`);
  }
  const root = doc.createElement("div");
  doc.main.appendChild(root);
  const card = panelMod.createToolCallPanel(root);
  for (const m of msgs) card.handle(m);
  assert.ok(root.textContent.includes(MASK), "premise: the inline card shows the masked text");
  assert.ok(!root.textContent.includes(SECRET), `the inline card drops the original text; ${root.textContent}`);
});

pendingT0162C("C-3: the one-line result appears with the final message and updates in place", async () => {
  const { el, feed } = await pane();
  feed(turnState("t1"));
  feed(call({ call_id: "s", name: "google_search", arguments: { query: "weather in Paris" }, status: "proposed", step: 2 }));
  const before = callEl(el, "s").querySelectorAll("[data-tools-result]");
  assert.ok(before.length === 0 || squeeze(before[0].textContent) === "", "C-3: no result before the final");
  feed(
    call({
      call_id: "s",
      name: "google_search",
      arguments: { query: "weather in Paris" },
      status: "executed",
      action: 'searched the web for "weather in Paris": 2 results (weather.example.org, forecast.example.org)',
      step: 2,
      result: res({ hosts: ["weather.example.org", "forecast.example.org"], results: 2 }),
    }),
  );
  assert.equal(resultOf(callEl(el, "s")), "2 results (weather.example.org, forecast.example.org)");
  assert.equal(el.querySelectorAll("[data-tools-call]").length, 1, "one line per call_id");
  feed(call({ call_id: "k", status: "executed", action: "clicked at (640, 180)", step: 3, result: res({ hosts: ["shop.example"], hosts_more: 1 }) }));
  assert.equal(resultOf(callEl(el, "k")), "clicked at (640, 180) · reached shop.example +1");
});

pendingT0162C("C-3 (adversarial): markup in a host is shown as text and creates no element", async () => {
  const { el, feed, doc } = await pane();
  const EVIL = "<img src=x onerror=window.__pwned=1>";
  feed(call({ call_id: "m", status: "executed", action: "clicked at (1, 2)", step: 1, result: res({ hosts: [EVIL, "good.example"] }) }));
  assert.equal(resultOf(callEl(el, "m")), `clicked at (1, 2) · reached ${EVIL}, good.example`, "C-3: literal text");
  assert.deepEqual(doc.violations, [], "no markup sink was used");
  assert.ok(!doc.created.some((e) => e.localName === "img"), "no <img> element was created");
});

pendingT0162C("C-3 (adversarial): the result line carries no query text", async () => {
  const { el, feed } = await pane();
  const QUERY = "ZQX-QUERY-canary 9 results (evil.example)";
  feed(
    call({
      call_id: "q",
      name: "google_search",
      arguments: { query: QUERY },
      status: "executed",
      action: `searched the web for "${QUERY}": 1 result (a.example)`,
      step: 1,
      result: res({ hosts: ["a.example"], results: 1 }),
    }),
  );
  const line = resultOf(callEl(el, "q"));
  assertOneResult(line, "a.example");
  assert.ok(!line.includes("ZQX-QUERY") && !line.includes("evil.example"), `C-3: no query text in the result; ${line}`);
});

pendingT0162C("C-3 (adversarial): with results shown, a thought commit and a canvas-step piece still never reach the pane", async () => {
  const { el, feed, doc } = await pane();
  const CANARY = "CANARY-thought-c3-91";
  feed(turnState("t1"));
  feed({ type: "astra.canvas.commit", request_id: "r1", canvas_idx: 0, segment: "thought", text: `It found 5 results at ${CANARY}.example`, steps_used: 4 });
  feed({
    type: "astra.canvas.step", request_id: "r1", canvas_idx: 0, step: 1, max_steps: 4, canvas_length: 2,
    canvas: [1, 2], argmax: [1, 2], phase: "denoise", segment: "thought", text_preview: CANARY,
    argmax_pieces: [CANARY, "x"],
  });
  feed(call({ call_id: "t", name: "google_search", arguments: { query: "q" }, status: "executed", action: "searched", step: 1, result: res({ hosts: ["a.example"], results: 3 }) }));
  assert.equal(resultOf(callEl(el, "t")), "3 results (a.example)", "premise: the result is shown");
  assert.ok(!el.textContent.includes(CANARY), "no thinking text in the pane");
  assert.ok(!doc.created.some((e) => e.textContent.includes(CANARY)), "…nor in any element the pane created");
});

// --- T-0202 group A — the `wait` tool in the Tools pane (docs/tasks/T-0202.md A5) -----------------
//
// API these rows pin (A5): in hud/src/tools/model.js
//   shortArgs("wait", args) -> "wait" for any args ({} and, adversarially, {x: 1}, {seconds: 2}, null)
//   "wait" is in BROWSER_TOOLS, so shortResult of a wait with hosts reads
//     "<action> · reached h1, h2" (+ " +n" beyond), exactly like click_at / scroll (T-0162 C-3).
// The action text "waited 1 s" is group B's WAIT_ACTION; the pane reads it as any action string.
//
// Gate. Strict xfail (the T-0193 B pattern) until docs/tasks/T-0202.md's **Status:** records
// "group A: tests <a>..<b>, activated <sha>" (the orchestrator writes it at activation).
// ASTRA_RUN_PENDING=1 runs the rows for real now. Group B's (or C's, D's) activation does not
// switch these rows on.

const T0202_TASK = fileURLToPath(new URL("docs/tasks/T-0202.md", REPO));
const T0202_STATUS = existsSync(T0202_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0202_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const T0202_A_ACTIVATED = /\bgroup A:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const T0202_A_PENDING = !T0202_A_ACTIVATED.test(T0202_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0202A(name, fn) {
  test(name + (T0202_A_PENDING ? " [pending T-0202 A: strict xfail until group A is activated]" : ""), async (t) => {
    if (!T0202_A_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0202 group A row passed while docs/tasks/T-0202.md's status does " +
          "not record group A's activation (" + JSON.stringify(T0202_STATUS.slice(0, 80)) + "). Either " +
          "group A has landed (record 'group A: tests …, activated <sha>' in the status) or the " +
          "row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0202 A pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0202 A gate: the status line is readable and only group A's own phrase activates the rows", () => {
  assert.ok(T0202_STATUS.length > 0, "docs/tasks/T-0202.md must carry a **Status:** line");
  assert.equal(T0202_A_ACTIVATED.test("planned 2026-10-29 at main 76522e39 (T-0200 merged), from James's session"), false);
  assert.equal(T0202_A_ACTIVATED.test("in progress (group A: tests 1a2b3c4..5d6e7f8, not yet activated)"), false);
  assert.equal(T0202_A_ACTIVATED.test("in progress (group B: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6)"), false);
  assert.equal(T0202_A_ACTIVATED.test("in progress (group A: tests a1b2c3d..e4f5a6b, activated c3d4e5f)"), true);
  assert.equal(T0202_A_ACTIVATED.test("in progress. group A: tests a1b2c3d, activated c3d4e5f; group B planned"), true);
});

pendingT0202A("T-0202 A5: shortArgs('wait', {}) is 'wait'", async () => {
  const shortArgs = await shortArgsFn();
  assert.equal(shortArgs("wait", {}), "wait");
});

pendingT0202A("T-0202 A5 (adversarial): a wait with arguments still reads 'wait', never its JSON", async () => {
  const shortArgs = await shortArgsFn();
  assert.equal(shortArgs("wait", { x: 1 }), "wait");
  assert.equal(shortArgs("wait", { seconds: 2 }), "wait");
  assert.equal(shortArgs("wait", null), "wait");
  // power: an unknown no-argument tool still takes the generic form (the case is wait's own, not a catch-all)
  assert.equal(shortArgs("waiting", {}), "waiting {}");
});

pendingT0202A("T-0202 A5: a wait's result line shows the hosts reached, like the other browser tools", async () => {
  const shortResult = await shortResultFn();
  const wait = (result) =>
    finalCall({ name: "wait", arguments: {}, status: "executed", action: "waited 1 s", result });
  assert.equal(shortResult(wait(res({ hosts: ["shop.example", "cdn.shop.example"] }))), "waited 1 s · reached shop.example, cdn.shop.example");
  assert.equal(
    shortResult(wait(res({ hosts: ["a.example", "b.example", "c.example"], hosts_more: 4 }))),
    "waited 1 s · reached a.example, b.example, c.example +4",
  );
  // without hosts it reads as the action; a refused wait shows its action (no hosts invented)
  assert.equal(shortResult(wait(res())), "waited 1 s");
  // power: a non-browser tool with hosts does not show them (BROWSER_TOOLS is what decides)
  assert.equal(
    shortResult(finalCall({ name: "annotate", arguments: {}, action: "noted", result: res({ hosts: ["x.example"] }) })),
    "noted",
  );
});

pendingT0202A("T-0202 A5: in the pane, a wait call's line is 'wait' and its result shows the hosts", async () => {
  const { el, feed } = await pane();
  feed(turnState("t1"));
  feed(call({ turn_id: "t1", call_id: "w1", name: "wait", arguments: {}, status: "executed", action: "waited 1 s", step: 2, result: res({ hosts: ["shop.example"], hosts_more: 1 }) }));
  const c = callEl(el, "w1");
  const line = c.querySelectorAll("[data-tools-line]");
  assert.equal(line.length, 1);
  assert.equal(squeeze(line[0].textContent), "wait");
  assert.equal(resultOf(c), "waited 1 s · reached shop.example +1");
});
