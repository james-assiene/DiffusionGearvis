// T-0154 group B — one conversation thread with cards inline: hud/src/panels/thread.js, and the
// A2UI panel (hud/src/ui/index.js) mounted against the thread's card slots.
//
// docs/tasks/T-0154.md criteria B2, B3, B4, B6, B7, B8 (the node half), B10 and B11; *Tests to
// write* "tests/hud/thread.test.mjs". The browser halves (layout, scrolling, parity, the click on a
// real rendered button) live in tests/integration/test_hud_thread_parity.py and, after group A
// merges, in the amended shared modules (see the group B test-author's report).
//
// The failure modes this file is aimed at (the task names them):
//   * a thread that makes an article for every turn id it hears of — the fixture carries 238
//     astra.turn.state messages, and they must create 0 articles (B2, adversarial);
//   * a thread rebuilt on every message (`replaceChildren`), which throws away the rendered
//     <a2ui-surface> and the action cards — node identity must survive 50 further messages (B6);
//   * a late message for an older turn that moves or duplicates its article (B2), and a superseded
//     turn's late tool call that reopens a turn (B7, U7/U10);
//   * an older card that still speaks (B8, H-T154-4), and a reconnect replay that doubles lines (B10).
//
// API this pins (the task names the file, not the exports; these are the names):
//
//   hud/src/panels/thread.js — importable in node, no DOM at module top level
//     createThread(root) -> { handle(message), cardSlot(surfaceId) }
//       root          the [data-thread] element; everything is built with root.ownerDocument and
//                     lives under root. The thread is the one owner of conversation DOM.
//       handle(m)     one decoded server message of any type (what hud/src/protocol.js decode()
//                     returns as `msg` for kind "astra"); never throws.
//       cardSlot(id)  -> Element | null. The card slot of the article the surface id names
//                     (`turn-<turn_id>`, implementation note 3), creating that article if it does
//                     not exist yet; a string id that names no turn gets its own article at the end
//                     of the thread (B4). The same element for the same id, every time. A
//                     non-string or empty id is null and creates nothing.
//
//   hud/src/ui/index.js — mount(host) with a new host member:
//     host.cardSlot(surfaceId) -> Element   (main.js passes the thread's cardSlot). The panel
//                     renders each surface into the slot its id names. A host without cardSlot:
//                     one log line that names the missing card-slot provider, nothing rendered,
//                     never throws (B11). `host.shown` is retired with #shown-channel (B1).
//
//   DOM hooks (inside root):
//     article[data-turn-id="<turn_id>"]   one per turn with content (B2)
//     [data-transcript]                   the user bubble, one per article that has a transcript
//                                         (the hook test_hud_headless.py already reads as "the
//                                         user transcript line")
//     [data-assistant]                    the assistant row, holding:
//       [data-avatar]                     decorative: aria-hidden="true", no text
//       [data-utterance-id][data-turn-id] one per utterance_id (T-0008 hooks, unchanged)
//       [data-turn-meta]                  the meta line: interrupted mark, fence notice, repaired N
//       the card slot                     whatever cardSlot() returns for the turn's surface id
//       [data-tool-call-id]               the turn's action cards (T-0090 hooks, unchanged)
//     [data-interrupted]                  on the article (or inside it) of an interrupted turn
//     [data-ui-surface] / [data-ui-canvas-idx]  on each card's wrapper, inside its slot (B4)
//     inert                               on every card wrapper but the newest surface's (B8)
//
// Strict-xfail gate (O-P): node tests cannot carry pytest's `pending` marker. Every row below runs
// through `pendingT0154B`, which reads docs/tasks/T-0154.md's **Status:** line: until it records
// group B's activation ("group B: tests <a>..<b>, activated <sha>", written by the test-author when
// /implement T-0154 B starts) each row must FAIL, and a row that passes throws XPASS(strict). Group
// A's activation phrase does not switch these rows on. `ASTRA_RUN_PENDING=1 node --test
// tests/hud/thread.test.mjs` runs them for real now (the node twin of `pytest --pending`).
//
// Every expected value is hand-derived from the criteria or read from the committed fixture
// tests/fixtures/hud/session_20261020_t3_t11.jsonl — never from the module under test.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);
const FIXTURE = fileURLToPath(new URL("tests/fixtures/hud/session_20261020_t3_t11.jsonl", REPO));

// --- the group B strict-xfail gate -----------------------------------------------------------

const T0154_TASK = fileURLToPath(new URL("docs/tasks/T-0154.md", REPO));
const T0154_STATUS = existsSync(T0154_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0154_TASK, "utf8")) || [])[1] || "").trim()
  : "";
/** "group B: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6" — the phrase the B test-author writes. */
const B_ACTIVATED = /\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const B_PENDING = !B_ACTIVATED.test(T0154_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0154B(name, fn) {
  test(name + (B_PENDING ? " [pending T-0154 B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0154 group B row passed while docs/tasks/T-0154.md's status does " +
          "not record group B's activation (" + JSON.stringify(T0154_STATUS.slice(0, 80)) + "). " +
          "Either group B has landed (record 'group B: tests …, activated <sha>' in the status) " +
          "or the row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0154 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0154 B gate: the status line is readable and only group B's own phrase activates the rows", () => {
  assert.ok(T0154_STATUS.length > 0, "docs/tasks/T-0154.md must carry a **Status:** line");
  // Hand-written samples: group A's activation wording must not switch group B's rows on.
  assert.equal(B_ACTIVATED.test("in progress 2026-10-25 (group A: tests 433d454..7a97f95, activated 7030bd5; planned)"), false);
  assert.equal(B_ACTIVATED.test("in progress (group A: …, activated 7030bd5; group B: tests a1b2c3d..e4f5a6b, activated c3d4e5f)"), true);
  assert.equal(B_ACTIVATED.test("in progress (group A: …, activated 7030bd5; group B: tests a1b2c3d..e4f5a6b, not yet activated)"), false, "activation names its commit");
  assert.equal(B_ACTIVATED.test("in progress (group B tests written; group C: tests …, activated 1234abc)"), false, "group C's activation is not group B's");
});

// --- T-0164 group B: the swap (our renderer in place of the vendored bundle) -------------------
//
// docs/tasks/T-0164.md B1, B2 (T-0154 B4/B5/B6/B8/B10/B11 re-keyed by name from <a2ui-surface>
// to [data-a2ui-root]), B4 and B7. Two switches read T-0164's **Status:** line:
//   * the re-keyed T-0154 rows key on <a2ui-surface> and the fake vendor seam until the line
//     records "group B: tests <a>..<b>, activated <sha>", and on [data-a2ui-root] and the real
//     engine (hud/src/ui/engine.js behind index.js) from then on. Before activation they are
//     byte-for-byte today's rows, so they stay green on main; after it they are red until index.js
//     mounts the engine. `ASTRA_RUN_PENDING=1` flips the switch now (see them fail for real).
//   * the new T-0164 B rows run through `pendingT0164B`: strict xfail until that same phrase.
// The ids are read off the DOM contract the task fixes (implementation note 5): the wrapper's
// [data-ui-surface], the card root's [data-a2ui-root], data-a2-id on each component.

const T0164_TASK = fileURLToPath(new URL("docs/tasks/T-0164.md", REPO));
const T0164_STATUS = existsSync(T0164_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0164_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const T0164_B_PHRASE = /\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
/** true once T-0164 group B is activated: the re-keyed rows use our renderer's hooks. */
const T0164_B = T0164_B_PHRASE.test(T0164_STATUS) || process.env.ASTRA_RUN_PENDING === "1";
const T0164_B_PENDING = !T0164_B;

function pendingT0164B(name, fn) {
  test(name + (T0164_B_PENDING ? " [pending T-0164 B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!T0164_B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0164 group B row passed while docs/tasks/T-0164.md's status does " +
          "not record group B's activation (" + JSON.stringify(T0164_STATUS.slice(0, 80)) + ").",
      );
    }
    t.diagnostic("expected failure (T-0164 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0164 B gate: only T-0164's own group B phrase switches the re-keyed rows and the new rows on", () => {
  assert.ok(T0164_STATUS.length > 0, "docs/tasks/T-0164.md must carry a **Status:** line");
  // Hand-written samples. Group A's activation (today's line) must not switch group B on.
  assert.equal(T0164_B_PHRASE.test("in progress 2026-10-27 (group A: tests cf07124b..65245b04, activated 3c1f6079; implementation on T-0164-a-impl-iso)"), false);
  assert.equal(T0164_B_PHRASE.test("in progress (group A: tests a..b, activated 3c1f6079; group B: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6)"), true);
  assert.equal(T0164_B_PHRASE.test("in progress (group A: …, activated 3c1f6079; group B: tests 1a2b3c4..5d6e7f8, not yet activated)"), false);
  assert.equal(T0164_B_PHRASE.test("in progress (group C: tests …, activated 1234abc)"), false, "group C's phrase is not group B's");
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
console.log(`# thread origin: repo=${fileURLToPath(REPO)} modules=${srcRoot().href}`);

async function load(name) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch (err) {
    assert.fail(
      `T-0154 B adds/changes hud/src/${name}, importable in node as an ES module with no DOM at ` +
        `module top level. node could not import it: ${err.message}`,
    );
  }
}

async function threadModule() {
  const mod = await load("panels/thread.js");
  assert.equal(
    typeof mod.createThread,
    "function",
    "T-0154 B: hud/src/panels/thread.js exports createThread(root) -> { handle, cardSlot }",
  );
  return mod;
}

// --- a DOM double ----------------------------------------------------------------------------
//
// Closed by delegation in the one direction that matters (.claude/rules/tests.md): a member the
// double does not implement is a *named* failure, never an undefined that makes a row pass because
// nothing rendered. Markup sinks throw (model text is text). `replaceChildren` and a `textContent`
// write over element children are recorded per element, so B6's "the thread is never rebuilt" is a
// behaviour, not a grep.

if (typeof globalThis.CSS === "undefined") {
  globalThis.CSS = { escape: (s) => String(s).replace(/["\\\]]/g, "\\$&") };
}
if (typeof globalThis.requestAnimationFrame === "undefined") {
  globalThis.requestAnimationFrame = (fn) => {
    fn(0);
    return 0;
  };
}

const kebab = (key) => "data-" + key.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase());

function fakeDocument() {
  const doc = { created: [], replaced: [], wiped: [], violations: [], openWrites: [], nodeType: 9 };
  const make = (tag) => {
    const el = makeElement(doc, String(tag).toLowerCase());
    doc.created.push(el);
    return el;
  };
  Object.assign(doc, {
    createElement: make,
    createElementNS: (_ns, tag) => make(tag),
    createTextNode: (text) => makeText(doc, text),
    createDocumentFragment: () => makeElement(doc, "#fragment"),
    createComment: (text) => Object.assign(makeText(doc, ""), { nodeType: 8, data: String(text), get textContent() { return ""; } }),
    write() {
      doc.violations.push("document.write");
      throw new Error("model text is rendered with textContent only (document.write)");
    },
  });
  doc.documentElement = makeElement(doc, "html");
  doc.documentElement._isRoot = true;
  doc.head = makeElement(doc, "head");
  doc.body = makeElement(doc, "body");
  doc.documentElement.append(doc.head, doc.body);
  doc.querySelector = (s) => doc.documentElement.querySelector(s);
  doc.querySelectorAll = (s) => doc.documentElement.querySelectorAll(s);
  doc.getElementById = (id) => doc.documentElement.querySelectorAll("*").find((e) => e.id === id) || null;
  doc.addEventListener = () => {};
  doc.removeEventListener = () => {};
  return doc;
}

function makeText(doc, text) {
  const node = {
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
    get nodeValue() {
      return this.data;
    },
    get parentElement() {
      return this.parentNode;
    },
    get nextSibling() {
      return sibling(this, 1);
    },
    get previousSibling() {
      return sibling(this, -1);
    },
    get isConnected() {
      return connected(this);
    },
    remove() {
      if (this.parentNode) this.parentNode._detach(this);
    },
  };
  return node;
}

function sibling(node, step) {
  const p = node.parentNode;
  if (!p) return null;
  const i = p.childNodes.indexOf(node);
  return p.childNodes[i + step] || null;
}

function elementSibling(node, step) {
  const p = node.parentNode;
  if (!p) return null;
  const kids = p.childNodes.filter((c) => c.nodeType === 1);
  const i = kids.indexOf(node);
  return kids[i + step] || null;
}

function connected(node) {
  for (let n = node; n; n = n.parentNode) if (n._isRoot) return true;
  return false;
}

const MARKUP = ["innerHTML", "outerHTML"];

function makeElement(doc, tag) {
  const attrs = new Map();
  const childNodes = [];
  const listeners = [];
  const style = {};
  const target = {
    nodeType: 1,
    tagName: tag.toUpperCase(),
    localName: tag,
    nodeName: tag.toUpperCase(),
    ownerDocument: doc,
    parentNode: null,
    childNodes,
    style,
    scrollTop: 0,
    scrollLeft: 0,
    scrollHeight: 0,
    scrollWidth: 0,
    clientHeight: 0,
    clientWidth: 0,
    offsetHeight: 0,
    offsetWidth: 0,
    offsetTop: 0,
    surface: undefined,
    _isRoot: false,
    _attrs: attrs,
    _listeners: listeners,
    get id() {
      return attrs.get("id") || "";
    },
    set id(v) {
      attrs.set("id", String(v));
    },
    get className() {
      return attrs.get("class") || "";
    },
    set className(v) {
      attrs.set("class", String(v));
    },
    get title() {
      return attrs.get("title") || "";
    },
    set title(v) {
      attrs.set("title", String(v));
    },
    get hidden() {
      return attrs.has("hidden");
    },
    set hidden(v) {
      if (v) attrs.set("hidden", "");
      else attrs.delete("hidden");
    },
    get inert() {
      return attrs.has("inert");
    },
    set inert(v) {
      if (v) attrs.set("inert", "");
      else attrs.delete("inert");
    },
    // T-0180 C2: HTMLDetailsElement.open reflects the attribute; a property write is recorded.
    get open() {
      return attrs.has("open");
    },
    set open(v) {
      doc.openWrites.push({ how: "property", value: !!v, tag });
      if (v) attrs.set("open", "");
      else attrs.delete("open");
    },
    get children() {
      return childNodes.filter((c) => c.nodeType === 1);
    },
    get childElementCount() {
      return this.children.length;
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
    get lastElementChild() {
      const k = this.children;
      return k[k.length - 1] || null;
    },
    get nextSibling() {
      return sibling(proxy, 1);
    },
    get previousSibling() {
      return sibling(proxy, -1);
    },
    get nextElementSibling() {
      return elementSibling(proxy, 1);
    },
    get previousElementSibling() {
      return elementSibling(proxy, -1);
    },
    get parentElement() {
      return this.parentNode;
    },
    get isConnected() {
      return connected(proxy);
    },
    get textContent() {
      return childNodes.map((c) => (c.nodeType === 8 ? "" : c.textContent)).join("");
    },
    set textContent(v) {
      if (childNodes.some((c) => c.nodeType === 1)) doc.wiped.push(proxy);
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
    classList: {
      _list: () => (attrs.get("class") || "").split(/\s+/).filter(Boolean),
      _set(list) {
        attrs.set("class", list.join(" "));
      },
      add(...n) {
        const l = this._list();
        for (const c of n) if (!l.includes(c)) l.push(c);
        this._set(l);
      },
      remove(...n) {
        this._set(this._list().filter((c) => !n.includes(c)));
      },
      toggle(c, force) {
        const has = this._list().includes(c);
        const on = force === undefined ? !has : !!force;
        if (on) this.add(c);
        else this.remove(c);
        return on;
      },
      contains(c) {
        return this._list().includes(c);
      },
      replace(a, b) {
        if (!this.contains(a)) return false;
        this._set(this._list().map((c) => (c === a ? b : c)));
        return true;
      },
    },
    setAttribute(name, value) {
      attrs.set(String(name).toLowerCase(), String(value));
    },
    getAttribute(name) {
      const n = String(name).toLowerCase();
      return attrs.has(n) ? attrs.get(n) : null;
    },
    hasAttribute(name) {
      return attrs.has(String(name).toLowerCase());
    },
    removeAttribute(name) {
      attrs.delete(String(name).toLowerCase());
    },
    toggleAttribute(name, force) {
      const on = force === undefined ? !this.hasAttribute(name) : !!force;
      if (on) this.setAttribute(name, "");
      else this.removeAttribute(name);
      return on;
    },
    getAttributeNames() {
      return [...attrs.keys()];
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
      doc.replaced.push(proxy);
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
    before(...nodes) {
      const parent = proxy.parentNode;
      if (parent) for (const n of nodes) parent.insertBefore(n, proxy);
    },
    after(...nodes) {
      const parent = proxy.parentNode;
      if (!parent) return;
      let ref = proxy.nextSibling;
      for (const n of nodes) parent.insertBefore(n, ref);
    },
    insertAdjacentElement(where, el) {
      const w = String(where).toLowerCase();
      if (w === "beforeend") proxy.appendChild(el);
      else if (w === "afterbegin") proxy.prepend(el);
      else if (w === "beforebegin") proxy.before(el);
      else proxy.after(el);
      return el;
    },
    remove() {
      if (proxy.parentNode) proxy.parentNode._detach(proxy);
    },
    contains(node) {
      for (let n = node; n; n = n.parentNode) if (n === proxy) return true;
      return false;
    },
    getRootNode() {
      let n = proxy;
      while (n.parentNode) n = n.parentNode;
      return n._isRoot ? doc : n;
    },
    matches(selector) {
      return matchesList(proxy, selector, null);
    },
    closest(selector) {
      for (let n = proxy; n && n.nodeType === 1; n = n.parentNode) if (matchesList(n, selector, null)) return n;
      return null;
    },
    querySelectorAll(selector) {
      return descendants(proxy).filter((e) => matchesList(e, selector, proxy));
    },
    querySelector(selector) {
      return proxy.querySelectorAll(selector)[0] || null;
    },
    addEventListener(type, fn) {
      listeners.push([type, fn]);
    },
    // T-0164 B: fire this element's own listeners for `type` (no bubbling: the renderer's button
    // listens on the button). The event is the plain shape a click handler reads.
    _fire(type) {
      const ev = { type, target: proxy, currentTarget: proxy, button: 0, detail: 1, isTrusted: true, preventDefault() {}, stopPropagation() {} };
      for (const [t, fn] of [...listeners]) if (t === type) fn.call(proxy, ev);
    },
    removeEventListener() {},
    dispatchEvent() {
      return true;
    },
    scrollTo() {},
    scrollBy() {},
    scrollIntoView() {},
    focus() {},
    blur() {},
    getBoundingClientRect() {
      return { x: 0, y: 0, top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0 };
    },
    insertAdjacentHTML() {
      doc.violations.push("insertAdjacentHTML");
      throw new Error("model text is rendered with textContent only (insertAdjacentHTML)");
    },
  };
  const dataset = new Proxy(
    {},
    {
      get: (_o, key) => (typeof key === "string" && attrs.has(kebab(key)) ? attrs.get(kebab(key)) : undefined),
      set: (_o, key, value) => (attrs.set(kebab(String(key)), String(value)), true),
      deleteProperty: (_o, key) => (attrs.delete(kebab(String(key))), true),
      has: (_o, key) => attrs.has(kebab(String(key))),
      ownKeys: () =>
        [...attrs.keys()].filter((k) => k.startsWith("data-")).map((k) => k.slice(5).replace(/-([a-z])/g, (_m, c) => c.toUpperCase())),
      getOwnPropertyDescriptor: (_o, key) =>
        attrs.has(kebab(String(key))) ? { enumerable: true, configurable: true, value: attrs.get(kebab(String(key))) } : undefined,
    },
  );
  target.dataset = dataset;
  for (const sink of MARKUP) {
    Object.defineProperty(target, sink, {
      get: () => (sink === "innerHTML" ? "" : undefined),
      set: () => {
        doc.violations.push(sink);
        throw new Error(`model text is rendered with textContent only (${sink} was set)`);
      },
    });
  }
  const proxy = new Proxy(target, {
    get(obj, key, receiver) {
      if (typeof key === "symbol" || key in obj) return Reflect.get(obj, key, receiver);
      if (key === "then" || key === "toJSON") return undefined;
      throw new Error(
        `the DOM double in tests/hud/thread.test.mjs has no Element.${String(key)}; extend the ` +
          "double (this is a gap in the test, not a defect in the thread)",
      );
    },
    set(obj, key, value) {
      if (key === "open") return Reflect.set(obj, key, value, obj); // T-0180 C2: the recording setter
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

// A small selector engine: compound selectors (tag, *, #id, .class, [attr], [attr="v"], :scope)
// joined by descendant or child combinators, in comma lists. Anything else is a named failure.
function parseCompound(text) {
  const parts = [];
  let s = text;
  const re = /^(\*|[a-zA-Z][a-zA-Z0-9-]*|#[\w-]+|\.[\w-]+|\[\s*[\w-]+\s*(?:=\s*(?:"[^"]*"|'[^']*'|[^\]\s]+)\s*)?\]|:scope)/;
  while (s.length) {
    const m = re.exec(s);
    if (!m) throw new Error(`the DOM double's selector engine cannot parse ${JSON.stringify(text)}; extend it`);
    parts.push(m[1]);
    s = s.slice(m[1].length);
  }
  return parts;
}

function matchCompound(el, parts, scope) {
  for (const p of parts) {
    if (p === "*") continue;
    if (p === ":scope") {
      if (el !== scope) return false;
    } else if (p[0] === "#") {
      if (el.id !== p.slice(1)) return false;
    } else if (p[0] === ".") {
      if (!el.classList.contains(p.slice(1))) return false;
    } else if (p[0] === "[") {
      const m = /^\[\s*([\w-]+)\s*(?:=\s*(?:"([^"]*)"|'([^']*)'|([^\]\s]+))\s*)?\]$/.exec(p);
      const v = el.getAttribute(m[1]);
      const want = m[2] ?? m[3] ?? m[4];
      if (want === undefined ? v === null : v !== want) return false;
    } else if (el.localName !== p.toLowerCase()) return false;
  }
  return true;
}

function matchesComplex(el, selector, scope) {
  const tokens = selector.trim().replace(/\s*>\s*/g, " > ").split(/\s+/);
  // Left to right: each compound with the combinator that joins it to the compound on its left.
  const steps = [];
  let comb = null;
  for (const tok of tokens) {
    if (tok === ">") {
      comb = ">";
      continue;
    }
    steps.push({ parts: parseCompound(tok), comb: steps.length ? comb || " " : null });
    comb = null;
  }
  const walk = (node, i) => {
    if (!matchCompound(node, steps[i].parts, scope)) return false;
    if (i === 0) return true;
    if (steps[i].comb === ">") {
      const p = node.parentNode;
      return !!(p && p.nodeType === 1 && walk(p, i - 1));
    }
    for (let p = node.parentNode; p && p.nodeType === 1; p = p.parentNode) if (walk(p, i - 1)) return true;
    return false;
  };
  return walk(el, steps.length - 1);
}

function matchesList(el, selector, scope) {
  return String(selector)
    .split(",")
    .some((s) => s.trim() && matchesComplex(el, s, scope));
}

// --- fixtures and helpers ---------------------------------------------------------------------

const FIXTURE_LINES = readFileSync(FIXTURE, "utf8").split("\n").filter((l) => l.trim());
const FIXTURE_MSGS = FIXTURE_LINES.map((l) => JSON.parse(l));

async function decoder() {
  const { decode } = await load("protocol.js");
  return (raw) => {
    const d = decode(raw);
    assert.equal(d.kind, "astra", `premise: the fixture line decodes as astra.*: ${raw.slice(0, 120)}`);
    return d.msg;
  };
}

async function thread() {
  const { createThread } = await threadModule();
  const doc = fakeDocument();
  const root = doc.createElement("div");
  root.setAttribute("data-thread", "");
  doc.body.appendChild(root);
  const t = createThread(root);
  assert.ok(t && typeof t.handle === "function" && typeof t.cardSlot === "function", "createThread(root) returns { handle, cardSlot }");
  const mounted = { children: root.childNodes.length, created: doc.created.length };
  return { doc, root, t, mounted, handle: (m) => t.handle(m) };
}

const articles = (root) => root.querySelectorAll("article").filter((a) => a.hasAttribute("data-turn-id"));
const turnIds = (root) => articles(root).map((a) => a.getAttribute("data-turn-id"));
function articleOf(root, turn) {
  const found = articles(root).filter((a) => a.getAttribute("data-turn-id") === turn);
  assert.equal(found.length, 1, `B2: exactly one article[data-turn-id="${turn}"]; found ${found.length} (articles: ${JSON.stringify(turnIds(root))})`);
  return found[0];
}
const squeeze = (s) => String(s).replace(/\s+/g, " ").trim();
const utterances = (el) => el.querySelectorAll("[data-utterance-id]");
const toolCards = (el) => el.querySelectorAll("[data-tool-call-id]");
const bubbles = (el) => el.querySelectorAll("[data-transcript]");

function preorder(root) {
  return [root, ...descendants(root)];
}
function before(root, a, b) {
  const order = preorder(root);
  return order.indexOf(a) >= 0 && order.indexOf(a) < order.indexOf(b);
}

const CATALOG = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json";
const transcript = (turn_id, text, final = true, extra = {}) => ({ type: "astra.transcript", turn_id, text, final, replay: false, ...extra });
const spoken = (turn_id, utterance_id, text, extra = {}) => ({ type: "astra.spoken", turn_id, utterance_id, text, replay: false, ...extra });
const turnState = (turn_id, state, thinking_ms = null) => ({ type: "astra.turn.state", state, turn_id, thinking_ms, replay: false });
const played = (turn_id, entries) => ({ type: "astra.spoken.played", turn_id, played_ms: 1000, utterances: entries, replay: false });
const toolCall = (turn_id, call_id, over = {}) => ({
  type: "astra.tool.call",
  turn_id,
  call_id,
  name: "scroll",
  arguments: { direction: "down" },
  status: "proposed",
  source: "draft",
  action: null,
  reason: null,
  query_problem: null,
  replay: false,
  ...over,
});
const fence = (turn_id, over = {}) => ({ type: "astra.ui.fence", turn_id, canvases: 1, close: "written", continued: false, speech_refused: 0, ...over });
function surface(surface_id, canvas_idx = 0, { text = `card ${surface_id} c${canvas_idx}`, repaired = 0, final = false, replay = false } = {}) {
  return {
    type: "astra.ui.surface",
    surface_id,
    canvas_idx,
    final,
    repaired,
    replay,
    messages: [
      { version: "v0.9.1", createSurface: { surfaceId: surface_id, catalogId: CATALOG, sendDataModel: true } },
      { version: "v0.9.1", updateComponents: { surfaceId: surface_id, components: [{ id: "root", component: "Text", text }] } },
    ],
  };
}

/** T-0164 B4: a surface with one gate-made button (the UiGate RB shape: `__gate.clicked` with the
 *  model's label as context.label, the label lifted into `__gate_go_label`). No sendDataModel. */
const GATE_LABEL = "Show me more";
function gateCard(surface_id, canvas_idx = 0) {
  return {
    type: "astra.ui.surface",
    surface_id,
    canvas_idx,
    final: true,
    repaired: 1,
    replay: false,
    messages: [
      { version: "v0.9.1", createSurface: { surfaceId: surface_id, catalogId: CATALOG } },
      {
        version: "v0.9.1",
        updateComponents: {
          surfaceId: surface_id,
          components: [
            { id: "root", component: "Column", children: ["t", "go"] },
            { id: "t", component: "Text", text: "Pick one." },
            { id: "go", component: "Button", child: "__gate_go_label", action: { event: { name: "__gate.clicked", context: { label: GATE_LABEL } } } },
            { id: "__gate_go_label", component: "Text", text: GATE_LABEL },
          ],
        },
      },
    ],
  };
}
/** The one <button> inside `surfaceId`'s light-DOM card root (T-0164 B1: no shadow root). */
function gateButton(root, surfaceId) {
  const w = root.querySelectorAll("[data-ui-surface]").filter((e) => e.getAttribute("data-ui-surface") === surfaceId)[0];
  assert.ok(w, `T-0164 B4: a card wrapper for ${surfaceId}`);
  const cardRoots = w.children.filter((c) => c.hasAttribute("data-a2ui-root"));
  assert.equal(cardRoots.length, 1, `T-0164 B1: ${surfaceId}'s wrapper holds one [data-a2ui-root] child; it holds ${cardRoots.length}`);
  const buttons = cardRoots[0].querySelectorAll("button");
  assert.equal(buttons.length, 1, `T-0164 B4/A12: one <button> in ${surfaceId}'s card; found ${buttons.length}`);
  return buttons[0];
}

// --- the A2UI panel against the thread --------------------------------------------------------

async function uiIndex() {
  // T-0164 B1: the panel mounts the real engine (hud/src/ui/engine.js). The fake vendor seam that
  // stood in for the bundle before group B was removed with the bundle (T-0164 D, T-0167).
  return load("ui/index.js");
}

/** A thread with the real A2UI panel mounted on its card slots, delivered to in main.js's order. */
async function threadWithCards({ panelFirst = false } = {}) {
  const base = await thread();
  const index = await uiIndex();
  const handlers = [];
  const sent = [];
  const logged = [];
  const result = await index.mount({
    document: base.doc,
    cardSlot: (id) => base.t.cardSlot(id),
    subscribe: (h) => (handlers.push(h), () => {}),
    send: (m) => (sent.push(m), Promise.resolve()),
    log: (l) => logged.push(String(l)),
  });
  assert.ok(
    result && result.mounted,
    `B4: the ui panel mounts with a host that provides cardSlot (and no #shown-channel); it said ` +
      `${JSON.stringify(result)} and logged ${JSON.stringify(logged)}`,
  );
  // The engine is index.js's own, so no processor is exposed (the fake vendor's was, before T-0164 B).
  const processor = null;
  const deliver = (m) => {
    if (!panelFirst) base.handle(m);
    for (const h of handlers) h(m);
    if (panelFirst) base.handle(m);
  };
  return { ...base, deliver, sent, logged, processor };
}

/** The rendered card element: <a2ui-surface> (vendor) until T-0164 B, then [data-a2ui-root] (B2). */
const SURFACE_SEL = T0164_B ? "[data-a2ui-root]" : "a2ui-surface";
const SURFACE_NAME = T0164_B ? "[data-a2ui-root]" : "<a2ui-surface>";
const surfaceEls = (root) => root.querySelectorAll(SURFACE_SEL);
/** The surface id a card element shows: the vendor element's `.surface.id`; our root's wrapper id. */
function surfaceIdOf(el) {
  if (!el) return undefined;
  if (!T0164_B) return el.surface && el.surface.id;
  const w = el.parentNode;
  // T-0164 B1: every card is light DOM, `[data-ui-surface] > [data-a2ui-root]` (a direct child).
  return w && w.nodeType === 1 && w.hasAttribute("data-ui-surface") ? w.getAttribute("data-ui-surface") : undefined;
}
function wrapperOf(root, id) {
  const found = root.querySelectorAll("[data-ui-surface]").filter((w) => w.getAttribute("data-ui-surface") === id);
  assert.equal(found.length, 1, `B4: one [data-ui-surface="${id}"] wrapper; found ${found.length}`);
  return found[0];
}

// =================================================================================================
// B2 — one article per turn, created only by content
// =================================================================================================

pendingT0154B("T-0154 B2 (adversarial): the fixture's 238 astra.turn.state messages create 0 articles", async () => {
  const { doc, root, mounted, handle } = await thread();
  const decode = await decoder();
  const states = FIXTURE_LINES.filter((l) => JSON.parse(l).type === "astra.turn.state");
  assert.equal(states.length, 238, "premise: the fixture holds 238 turn.state messages (counted from the file)");
  for (const raw of states) assert.doesNotThrow(() => handle(decode(raw)));
  assert.deepEqual(turnIds(root), [], "B2: a turn state is not content — no article for any of t3–t11");
  assert.equal(root.querySelectorAll("article").length, 0, "…and no article of any kind");
  assert.equal(root.childNodes.length, mounted.children, "…the thread is exactly as it was mounted");
  assert.equal(doc.created.length, mounted.created, "…and not even an element was built for them");
});

pendingT0154B("T-0154 B2: replaying the whole fixture creates exactly 9 articles, t3–t11 in first-seen order", async () => {
  const { root, t, handle } = await thread();
  const decode = await decoder();
  for (const raw of FIXTURE_LINES) handle(decode(raw));
  assert.deepEqual(
    turnIds(root),
    ["t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"],
    "B2: one article per turn with content, in the order the turns' first content arrived",
  );
  // Read from the fixture by hand (task file: 11 astra.spoken, 11 distinct utterance ids; t11 has 3).
  assert.equal(utterances(root).length, 11, "B3: one element per utterance_id — 11 in the fixture");
  assert.deepEqual(
    utterances(articleOf(root, "t11")).map((u) => u.getAttribute("data-utterance-id")),
    ["t11-u1", "t11-u2", "t11-u3"],
    "B3: t11's three utterances, in arrival order, in t11's article",
  );
  for (const a of articles(root)) {
    const turn = a.getAttribute("data-turn-id");
    for (const u of utterances(a))
      assert.equal(u.getAttribute("data-turn-id"), turn, `B3: an utterance in ${turn}'s article carries data-turn-id=${turn}`);
    const said = FIXTURE_MSGS.filter((m) => m.type === "astra.transcript" && m.turn_id === turn).pop();
    const b = bubbles(a);
    assert.equal(b.length, 1, `B3: ${turn}'s article holds one user bubble ([data-transcript]); found ${b.length}`);
    assert.equal(squeeze(b[0].textContent), squeeze(said.text), `B3: ${turn}'s bubble is the transcript verbatim`);
  }
  // Action cards in their own turn's article (B7): t3-call and t10-call, nowhere else.
  assert.deepEqual(toolCards(articleOf(root, "t3")).map((c) => c.getAttribute("data-tool-call-id")), ["t3-call"]);
  assert.deepEqual(toolCards(articleOf(root, "t10")).map((c) => c.getAttribute("data-tool-call-id")), ["t10-call"]);
  assert.equal(toolCards(root).length, 2, "B7: two calls in the fixture, two cards in the thread");
  const t3card = toolCards(articleOf(root, "t3"))[0];
  assert.ok(squeeze(t3card.textContent).includes("Would search for: omelette recipe"), "B7: toolLine copy kept");
  assert.ok(squeeze(t3card.textContent).includes("dry run — nothing was done"), "B7/U2: the dry-run notice kept");
  // The card slot of each surface id is inside that turn's article (B4).
  for (const turn of ["t3", "t4", "t5", "t6", "t8", "t9", "t10"]) {
    const slot = t.cardSlot(`turn-${turn}`);
    assert.ok(slot, `B4: cardSlot("turn-${turn}") returns the slot`);
    assert.ok(articleOf(root, turn).contains(slot), `B4: turn-${turn}'s slot is inside ${turn}'s article`);
  }
  assert.equal(articles(root).length, 9, "B4: asking for the slots made no new article");
});

pendingT0154B("T-0154 B2: each kind of content opens its turn's article on its own", async () => {
  const { root, handle } = await thread();
  handle(transcript("a1", "only the user spoke"));
  handle(spoken("a2", "a2-u1", "only the assistant spoke"));
  handle(surface("turn-a3"));
  handle(toolCall("a4", "a4-c1"));
  // A fence notice the page has words for (T-0054 criterion 15: close "open" -> incomplete).
  handle(fence("a5", { close: "open" }));
  assert.deepEqual(turnIds(root), ["a1", "a2", "a3", "a4", "a5"], "B2: transcript, utterance, surface, action card and fence notice each open an article");
});

pendingT0154B("T-0154 B2 negative: turn states, a quiet fence, a stray played report and other types open no article", async () => {
  const { root, mounted, handle } = await thread();
  for (const s of ["IDLE", "LISTENING", "ENDPOINTING", "THINKING", "SPEAKING", "INTERRUPTED"]) handle(turnState("n1", s, 5));
  // T-0054 criterion 16: a whole fence (written, 1 canvas, nothing refused) says nothing at all.
  handle(fence("n2", { close: "written", canvases: 1 }));
  // T-0030 criterion 17: a played report for an utterance the page never saw invents nothing.
  handle(played("n3", [{ utterance_id: "n3-u1", played_text: "x", played_words: 1, complete: false }]));
  handle({ type: "astra.error", code: "x", message: "y" });
  handle({ type: "astra.assist.state", flags: {}, badge: "pixels" });
  handle({ type: "astra.canvas.commit", request_id: "r", canvas_idx: 0, text: "n4 says hi" });
  handle({ type: "astra.zzz", turn_id: "n5" });
  for (const junk of [null, undefined, "astra.spoken", 7, {}]) assert.doesNotThrow(() => handle(junk), `handle(${JSON.stringify(junk)}) threw`);
  assert.deepEqual(turnIds(root), [], "B2: none of these is content");
  assert.equal(root.childNodes.length, mounted.children, "…the thread is unchanged");
});

pendingT0154B("T-0154 B2: a late message for an older turn updates its article in place — never moves or duplicates it", async () => {
  const { root, handle } = await thread();
  handle(transcript("t1", "first question"));
  handle(spoken("t1", "t1-u1", "first answer"));
  handle(transcript("t2", "second question"));
  handle(spoken("t2", "t2-u1", "second answer"));
  const a1 = articleOf(root, "t1");
  const a2 = articleOf(root, "t2");
  handle(spoken("t1", "t1-u2", "a late second line for turn one"));
  handle(transcript("t1", "first question, corrected"));
  handle(spoken("t1", "t1-u1", "first answer, revised"));
  assert.deepEqual(turnIds(root), ["t1", "t2"], "B2: first-seen order holds — t1 stays first");
  assert.equal(articleOf(root, "t1"), a1, "B2: t1's article is the same element (updated in place)");
  assert.equal(articleOf(root, "t2"), a2, "…and so is t2's");
  assert.deepEqual(utterances(a1).map((u) => squeeze(u.textContent)), ["first answer, revised", "a late second line for turn one"]);
  assert.equal(bubbles(a1).length, 1, "B3: still one bubble for t1");
  assert.equal(squeeze(bubbles(a1)[0].textContent), "first question, corrected");
  assert.equal(utterances(a2).length, 1, "nothing of t1's leaked into t2's article");
  // A surface for a never-seen turn arrives after both: it goes at the end.
  handle(surface("turn-t0"));
  assert.deepEqual(turnIds(root), ["t1", "t2", "t0"], "B2: first-seen order of content, not of turn-id value");
});

// =================================================================================================
// B3 — turn layout, and the spoken honesty rules
// =================================================================================================

pendingT0154B("T-0154 B3: inside an article — bubble, then the assistant row: avatar, utterances, meta, card slot, action cards", async () => {
  const { root, t, handle } = await thread();
  handle(transcript("t5", "Show me a packing list"));
  handle(fence("t5", { close: "open", canvases: 1 })); // published before the surface (ruling A12 i)
  handle(surface("turn-t5", 0, { repaired: 2 }));
  handle(spoken("t5", "t5-u1", "Here is a packing list."));
  handle(spoken("t5", "t5-u2", "It has nine items."));
  handle(toolCall("t5", "t5-c1"));
  const a = articleOf(root, "t5");
  const [bubble] = bubbles(a);
  assert.ok(bubble, "B3: the user bubble is in the article");
  const rows = a.querySelectorAll("[data-assistant]");
  assert.equal(rows.length, 1, `B3: one assistant row ([data-assistant]) per article; found ${rows.length}`);
  const row = rows[0];
  assert.ok(before(root, bubble, row), "B3: the user bubble comes before the assistant row");
  const avatars = row.querySelectorAll("[data-avatar]");
  assert.equal(avatars.length, 1, "B3: the assistant row has one avatar ([data-avatar])");
  assert.equal(avatars[0].getAttribute("aria-hidden"), "true", "B3: the avatar is decorative (aria-hidden=\"true\")");
  assert.equal(squeeze(avatars[0].textContent), "", "…and carries no text");
  const us = utterances(row);
  assert.deepEqual(us.map((u) => u.getAttribute("data-utterance-id")), ["t5-u1", "t5-u2"], "B3: the utterances are in the assistant row");
  const metas = row.querySelectorAll("[data-turn-meta]");
  assert.equal(metas.length, 1, "B3: one meta line ([data-turn-meta]) in the assistant row");
  const slot = t.cardSlot("turn-t5");
  assert.ok(row.contains(slot), "B3/B4: the card slot is in the assistant row");
  const [card] = toolCards(row);
  assert.ok(card, "B3/B7: the action card is in the assistant row");
  assert.ok(before(root, avatars[0], us[0]), "B3 order: avatar before the utterances");
  assert.ok(before(root, us[1], metas[0]), "B3 order: utterances before the meta line");
  assert.ok(before(root, metas[0], slot), "B3 order: meta line before the card slot");
  assert.ok(before(root, slot, card), "B3 order: card slot before the action cards");
  assert.ok(!slot.contains(card) && !card.contains(slot), "the slot and the action list are separate");
});

pendingT0154B("T-0154 B3: the user bubble is keyed by turn — a partial is replaced, a final stays one line, a replay adds none", async () => {
  const { root, handle } = await thread();
  handle(transcript("t1", "wh", false));
  handle(transcript("t1", "what is on", false));
  assert.equal(bubbles(articleOf(root, "t1")).length, 1, "T-0008 criterion 8: the partial is replaced in place");
  assert.equal(squeeze(bubbles(articleOf(root, "t1"))[0].textContent), "what is on");
  handle(transcript("t1", "what is on my screen?", true));
  handle(transcript("t1", "what is on my screen?", true, { replay: true }));
  handle(transcript("t2", "next", false));
  assert.equal(bubbles(articleOf(root, "t1")).length, 1, "a final line and its replay are one bubble");
  assert.equal(squeeze(bubbles(articleOf(root, "t1"))[0].textContent), "what is on my screen?");
  assert.equal(squeeze(bubbles(articleOf(root, "t2"))[0].textContent), "next", "a new turn's partial starts its own bubble in its own article");
  assert.equal(bubbles(root).length, 2);
});

pendingT0154B("T-0154 B3: struck words come only from astra.spoken.played — never from turn state or time", async () => {
  const { root, handle } = await thread();
  handle(spoken("t1", "t1-u1", "one two three four five"));
  handle(turnState("t1", "SPEAKING"));
  handle(turnState("t1", "INTERRUPTED"));
  const a = articleOf(root, "t1");
  const [u] = utterances(a);
  assert.equal(u.querySelectorAll("[data-unplayed]").length, 0, "T-0030 criterion 22: INTERRUPTED alone strikes nothing");
  assert.ok(!u.hidden && !u.hasAttribute("data-dim"), "no utterance is hidden or dimmed by turn state");
  assert.equal(squeeze(u.textContent), "one two three four five", "the full text stays");
  assert.ok(a.hasAttribute("data-interrupted") || a.querySelector("[data-interrupted]"), "B3: the interrupted turn carries data-interrupted");
  handle(played("t1", [{ utterance_id: "t1-u1", played_text: "one two", played_words: 2, complete: false }]));
  const [u2] = utterances(articleOf(root, "t1"));
  const tails = u2.querySelectorAll("[data-unplayed]");
  assert.equal(tails.length, 1, "T-0030 criterion 19: one [data-unplayed] tail once the played report arrives");
  assert.equal(squeeze(tails[0].textContent), "three four five", "…holding the words from index played_words (2) on");
  assert.equal(squeeze(u2.textContent), "one two three four five", "…and the text is still whole (struck, never removed)");
});

pendingT0154B("T-0154 B3: the meta line says interrupted, the fence notice and repaired N — and nothing on a healthy turn", async () => {
  const { root, handle } = await thread();
  handle(spoken("t1", "t1-u1", "a healthy turn"));
  handle(fence("t1", { close: "written", canvases: 1 }));
  handle(surface("turn-t1", 0, { repaired: 0 }));
  const healthy = articleOf(root, "t1");
  assert.ok(!/repaired/i.test(healthy.textContent), "repaired 0 shows no repair count");
  assert.ok(!/interface|interrupted/i.test(healthy.textContent), "a whole fence and an uninterrupted turn say nothing");

  handle(spoken("t2", "t2-u1", "this one was cut off"));
  handle(fence("t2", { close: "open", canvases: 1 }));
  handle(surface("turn-t2", 0, { repaired: 4 }));
  handle(turnState("t2", "INTERRUPTED"));
  const meta = articleOf(root, "t2").querySelector("[data-turn-meta]");
  assert.ok(meta, "B3: t2's article has a meta line");
  const text = squeeze(meta.textContent);
  // T-0054 criterion 15's label for close "open", verbatim (hud/src/ui/fence.js status table).
  assert.ok(text.includes("the interface was cut off — showing what arrived"), `B3: the fence notice is in the meta line; it reads ${JSON.stringify(text)}`);
  assert.ok(text.includes("repaired 4"), `B3: "repaired N" from the surface's count; it reads ${JSON.stringify(text)}`);
  assert.ok(/interrupted/i.test(text), `B3: the interrupted mark; it reads ${JSON.stringify(text)}`);
});

pendingT0154B("T-0154 B3 (adversarial): model text is text — markup in a transcript or an utterance creates no element", async () => {
  const { doc, root, handle } = await thread();
  const evil = '<img src=x onerror=alert(1)><a href="http://example.com">x</a>';
  handle(transcript("t1", evil));
  handle(spoken("t1", "t1-u1", evil));
  assert.deepEqual(doc.violations, [], `no markup sink was touched: ${doc.violations}`);
  const tags = new Set(doc.created.map((e) => e.localName));
  for (const banned of ["img", "a", "script", "iframe"]) assert.ok(!tags.has(banned), `no <${banned}> was created`);
  assert.equal(squeeze(bubbles(articleOf(root, "t1"))[0].textContent), evil);
  assert.equal(squeeze(utterances(root)[0].textContent), evil);
});

// =================================================================================================
// B4 — cards inline, honestly
// =================================================================================================

pendingT0154B("T-0154 B4: cardSlot — the named turn's slot, the same element every time, an end article for an unattributable id", async () => {
  const { root, t, handle } = await thread();
  handle(transcript("t1", "q1"));
  handle(transcript("t2", "q2"));
  const s1 = t.cardSlot("turn-t1");
  assert.ok(s1 && s1.nodeType === 1, "B4: cardSlot returns an element");
  assert.ok(articleOf(root, "t1").contains(s1), "B4: in the article of the turn the id names (turn-<turn_id>)");
  assert.equal(t.cardSlot("turn-t1"), s1, "B4: the same slot for the same id");
  assert.deepEqual(turnIds(root), ["t1", "t2"], "asking for an existing turn's slot opens nothing");

  const s9 = t.cardSlot("turn-t9");
  assert.deepEqual(turnIds(root), ["t1", "t2", "t9"], "B4: a surface for a turn not yet seen opens that turn's article at the end");
  assert.ok(articleOf(root, "t9").contains(s9));

  const stray = t.cardSlot("weird-surface");
  assert.ok(stray && stray.nodeType === 1, "B4: an id that names no turn still gets a slot — it is never dropped");
  const all = root.querySelectorAll("article");
  const last = all[all.length - 1];
  assert.ok(last.contains(stray), "B4: …in its own article at the end of the thread");
  for (const turn of ["t1", "t2", "t9"]) assert.ok(!articleOf(root, turn).contains(stray), `…not inside ${turn}'s article`);
  assert.equal(t.cardSlot("weird-surface"), stray, "…and the same slot next time");
  const count = root.querySelectorAll("article").length;
  for (const junk of [null, undefined, "", 7, {}]) assert.equal(t.cardSlot(junk), null, `cardSlot(${JSON.stringify(junk)}) is null`);
  assert.equal(root.querySelectorAll("article").length, count, "…and creates nothing");
});

pendingT0154B("T-0154 B4: the panel renders each surface in its turn's slot — one <a2ui-surface> per id, canvas k morphs k−1", async () => {
  const { root, t, deliver, processor } = await threadWithCards();
  deliver(transcript("t3", "omelette card please"));
  deliver(surface("turn-t3", 0));
  const [el3] = surfaceEls(root);
  assert.ok(el3, `B4: a decoded astra.ui.surface renders an ${SURFACE_NAME}`);
  deliver(surface("turn-t3", 1));
  deliver(transcript("t4", "packing list please"));
  deliver(surface("turn-t4", 0));
  deliver(surface("turn-t3", 2));
  const els = surfaceEls(root);
  assert.equal(els.length, 2, `B4 (T-0024 criterion 7 amended): one ${SURFACE_NAME} per surface id — 2 ids, ${els.length} elements`);
  assert.ok(els.includes(el3), "B4: canvas 1 and 2 of turn-t3 morphed the element canvas 0 created");
  assert.ok(t.cardSlot("turn-t3").contains(el3), "B4: turn-t3's surface is in t3's slot");
  const el4 = els.find((e) => e !== el3);
  assert.ok(t.cardSlot("turn-t4").contains(el4), "B4: turn-t4's surface is in t4's slot");
  assert.equal(surfaceIdOf(el3), "turn-t3", "the element shows its own surface");
  assert.equal(surfaceIdOf(el4), "turn-t4");
  if (T0164_B) {
    // T-0164 B2/A4: canvas 2's own text is what turn-t3's root shows now (the morph applied it).
    assert.equal(squeeze(el3.textContent), "card turn-t3 c2", "B4 (T-0164 B2): turn-t3's root shows canvas 2's text");
    assert.equal(squeeze(el4.textContent), "card turn-t4 c0");
  }
  assert.equal(wrapperOf(root, "turn-t3").getAttribute("data-ui-canvas-idx"), "2", "B4: [data-ui-canvas-idx] is per surface: t3 at canvas 2");
  assert.equal(wrapperOf(root, "turn-t4").getAttribute("data-ui-canvas-idx"), "0", "…t4 at canvas 0");
  assert.ok(t.cardSlot("turn-t3").contains(wrapperOf(root, "turn-t3")) || t.cardSlot("turn-t3") === wrapperOf(root, "turn-t3"), "the wrapper is (in) the slot");
  if (T0164_B) assert.equal(root.querySelectorAll("[data-ui-surface]").length, 2, "premise: two card wrappers");
  else assert.equal(processor.surfaces.size, 2, "premise: the engine holds two surfaces");
});

pendingT0154B("T-0154 B4: an unattributable surface renders in its own end article; a surface before its turn's text opens the turn once", async () => {
  const { root, deliver } = await threadWithCards({ panelFirst: true });
  deliver(transcript("t1", "q1"));
  // The panel sees the surface first (it asks for the slot before the thread has handled it).
  deliver(surface("turn-t2", 0));
  deliver(transcript("t2", "q2"));
  deliver(spoken("t2", "t2-u1", "a2"));
  assert.deepEqual(turnIds(root), ["t1", "t2"], "B2/B4: one t2 article, whichever subscriber saw the surface first");
  deliver(surface("loose-card", 0));
  const all = root.querySelectorAll("article");
  const el = surfaceEls(root).find((e) => surfaceIdOf(e) === "loose-card");
  assert.ok(el, "B4: a surface id that names no turn is rendered, never dropped");
  assert.ok(all[all.length - 1].contains(el), "B4: …in its own article at the end of the thread");
  assert.ok(!articleOf(root, "t1").contains(el) && !articleOf(root, "t2").contains(el));
});

// O-B3 (3), reviewer 1's T3: "in its own article" is per surface id — two ids that name no turn
// are two articles, never one shared bin.
pendingT0154B("T-0154 B4 (O-B3 T3): two unattributable surface ids render in two separate end articles", async () => {
  const { root, t, deliver } = await threadWithCards();
  deliver(transcript("t1", "q1"));
  deliver(surface("loose-a", 0));
  deliver(transcript("t2", "q2"));
  deliver(surface("loose-b", 0));
  const elA = surfaceEls(root).find((e) => surfaceIdOf(e) === "loose-a");
  const elB = surfaceEls(root).find((e) => surfaceIdOf(e) === "loose-b");
  assert.ok(elA && elB, "B4: both unattributable surfaces render (neither is dropped)");
  const all = root.querySelectorAll("article");
  const holderA = all.filter((a) => a.contains(elA));
  const holderB = all.filter((a) => a.contains(elB));
  assert.equal(holderA.length, 1, "premise: loose-a sits in exactly one article");
  assert.equal(holderB.length, 1, "premise: loose-b sits in exactly one article");
  assert.notEqual(holderA[0], holderB[0], "B4/T3: two surface ids that name no turn are two articles, not one");
  assert.ok(!holderA[0].contains(elB) && !holderB[0].contains(elA), "…neither holds the other's card");
  assert.notEqual(t.cardSlot("loose-a"), t.cardSlot("loose-b"), "…and two card slots");
  // Hand-derived order: t1's text, loose-a, t2's text, loose-b — each article where its first
  // content arrived (B2), none inside a turn's article.
  assert.deepEqual(
    all.map((a) => a.getAttribute("data-turn-id") || (a.contains(elA) ? "loose-a" : a.contains(elB) ? "loose-b" : "?")),
    ["t1", "loose-a", "t2", "loose-b"],
    "B2/B4: articles in first-seen order; the loose ones are articles of their own",
  );
  assert.equal(surfaceEls(holderA[0]).length, 1, `one ${SURFACE_NAME} in loose-a's article`);
  assert.equal(surfaceEls(holderB[0]).length, 1, `one ${SURFACE_NAME} in loose-b's article`);
});

// O-B3 (3), reviewer 1's T4: a surface id names a turn only as `turn-<turn_id>` (B4). A bare id that
// happens to equal an existing turn id names no turn, so it must not land in that turn's article.
pendingT0154B("T-0154 B4 (O-B3 T4): a bare surface id equal to an existing turn id does not land in that turn", async () => {
  const { root, t, deliver } = await threadWithCards();
  deliver(transcript("t3", "omelette card please"));
  deliver(spoken("t3", "t3-u1", "Here is the card."));
  const a3 = articleOf(root, "t3");
  deliver(surface("t3", 0));
  const bare = surfaceEls(root).find((e) => surfaceIdOf(e) === "t3");
  assert.ok(bare, "B4: the bare-id surface renders (never dropped)");
  assert.ok(!a3.contains(bare), "B4/T4: surface id \"t3\" is not turn-t3 — it must not render in t3's article");
  assert.ok(!a3.contains(t.cardSlot("t3")), "…and its card slot is not t3's slot");
  assert.equal(surfaceEls(a3).length, 0, "t3's article holds no card at all");
  assert.deepEqual(turnIds(root), ["t3"], "no second t3 article and no new turn article");
  const all = root.querySelectorAll("article");
  assert.equal(all.length, 2, "B4: the bare id gets its own article");
  assert.ok(all[1].contains(bare), "…at the end of the thread");
  // The control: the proper id still lands in t3 (so the row above is not passing because nothing lands anywhere).
  deliver(surface("turn-t3", 0));
  const proper = surfaceEls(root).find((e) => surfaceIdOf(e) === "turn-t3");
  assert.ok(proper && a3.contains(proper), "B4: turn-t3 renders in t3's article");
  assert.ok(!a3.contains(bare), "…and the bare one still does not");
});

// =================================================================================================
// B6 — node identity survives updates (adversarial)
// =================================================================================================

pendingT0154B("T-0154 B6 (adversarial): 50 further messages leave the same <a2ui-surface>, article, slot and action-card nodes", async () => {
  const { doc, root, t, deliver } = await threadWithCards();
  deliver(transcript("t9", "Turn this recipe into a shopping list card."));
  deliver(spoken("t9", "t9-u1", "Here's your shopping list."));
  deliver(surface("turn-t9", 0));
  deliver(toolCall("t9", "t9-c1", { status: "proposed" }));
  const article = articleOf(root, "t9");
  const slot = t.cardSlot("turn-t9");
  const [el] = surfaceEls(root);
  const [card] = toolCards(article);
  assert.ok(el && card, "premise: the card and the action card rendered");
  const wrapper = el.parentNode;
  doc.replaced.length = 0;
  doc.wiped.length = 0;

  const STATES = ["SPEAKING", "IDLE", "LISTENING", "ENDPOINTING", "THINKING"];
  const fifty = [];
  for (let k = 0; k < 50; k++) {
    switch (k % 7) {
      case 0: fifty.push(spoken("t9", "t9-u1", `Here's your shopping list, revision ${k}.`)); break;
      case 1: fifty.push(transcript("t10", `next question ${k}`, k > 40)); break;
      case 2: fifty.push(turnState(k < 25 ? "t9" : "t10", STATES[k % STATES.length], k)); break;
      case 3: fifty.push(played("t9", [{ utterance_id: "t9-u1", played_text: "Here's", played_words: 1, complete: false }])); break;
      case 4: fifty.push(surface("turn-t10", k)); break; // another turn's surface, canvases growing
      case 5: fifty.push(toolCall("t9", "t9-c1", { status: k % 2 ? "dry_run" : "proposed" })); break;
      default: fifty.push(spoken("t10", "t10-u1", `t10 line ${k}`)); break;
    }
  }
  assert.equal(fifty.length, 50);
  for (const [k, m] of fifty.entries()) {
    deliver(m);
    const where = `after message ${k + 1}/50 (${m.type})`;
    assert.equal(articleOf(root, "t9"), article, `B6: t9's article is the same node ${where}`);
    assert.equal(t.cardSlot("turn-t9"), slot, `B6: t9's card slot is the same node ${where}`);
    assert.ok(slot.contains(el) && el.parentNode === wrapper, `B6: the same ${SURFACE_NAME} is still in t9's slot ${where}`);
    assert.ok(root.contains(el) && el.isConnected, `B6: …and in the document ${where}`);
    assert.equal(surfaceEls(root).filter((e) => surfaceIdOf(e) === "turn-t9").length, 1, `one t9 surface ${where}`);
    assert.equal(toolCards(article).find((c) => c.getAttribute("data-tool-call-id") === "t9-c1"), card, `B6: the same action-card node ${where}`);
  }
  const containers = new Set([root, ...root.querySelectorAll("article"), slot, t.cardSlot("turn-t10"), card.parentNode]);
  const rebuilt = [...doc.replaced, ...doc.wiped].filter((n) => containers.has(n));
  assert.deepEqual(
    rebuilt.map((n) => `${n.localName}${n.getAttribute("data-turn-id") ? `[data-turn-id=${n.getAttribute("data-turn-id")}]` : ""}`),
    [],
    "B6 / implementation note 2: the thread, an article, a card slot or the action list was replaced " +
      "(replaceChildren or a textContent write over element children)",
  );
  assert.deepEqual(turnIds(root), ["t9", "t10"]);
});

// =================================================================================================
// B7 — action cards per turn, kept as history
// =================================================================================================

pendingT0154B("T-0154 B7: each turn's action cards render in its own article; earlier turns keep theirs (T-0090 'replace' amended)", async () => {
  const { root, handle } = await thread();
  handle(transcript("t1", "search for omelettes"));
  handle(toolCall("t1", "t1-c1", { name: "google_search", arguments: { query: "omelette" } }));
  handle(toolCall("t1", "t1-c1", { name: "google_search", arguments: { query: "omelette" }, status: "dry_run", action: "would search for: omelette" }));
  handle(transcript("t2", "now scroll"));
  handle(toolCall("t2", "t2-c1"));
  const a1 = articleOf(root, "t1");
  const c1 = toolCards(a1)[0];
  assert.deepEqual(toolCards(a1).map((c) => c.getAttribute("data-tool-call-id")), ["t1-c1"], "B7/H-T128-3: t1 keeps its card as history");
  assert.deepEqual(toolCards(articleOf(root, "t2")).map((c) => c.getAttribute("data-tool-call-id")), ["t2-c1"], "B7: t2's card is in t2's article");
  assert.ok(squeeze(c1.textContent).includes("Would search for: omelette"));
  assert.ok(squeeze(c1.textContent).includes("dry run — nothing was done"), "U2: the dry-run notice stays on the history card");
  assert.equal(c1.querySelector("[data-tool-status]").getAttribute("data-tool-status"), "dry_run");
});

pendingT0154B("T-0154 B7 (U7/U10): a superseded turn's late tool call is ignored — no new card, no update, no reopened or moved turn", async () => {
  const { root, handle } = await thread();
  handle(toolCall("t1", "t1-c1"));
  handle(toolCall("t2", "t2-c1"));
  handle(transcript("t3", "third"));
  handle(toolCall("t3", "t3-c1"));
  const a1 = articleOf(root, "t1");
  const c1 = toolCards(a1)[0];
  handle(toolCall("t1", "t1-c1", { status: "cancelled", reason: "interrupted" }));
  handle(toolCall("t1", "t1-c9", { name: "press_key", arguments: { combo: "Ctrl+L" } }));
  handle(toolCall("t2", "t2-c1", { status: "dry_run" }));
  assert.deepEqual(turnIds(root), ["t1", "t2", "t3"], "U7: no turn reopened, moved or duplicated");
  assert.equal(articleOf(root, "t1"), a1);
  assert.deepEqual(toolCards(a1).map((c) => c.getAttribute("data-tool-call-id")), ["t1-c1"], "U10: no t1-c9 card");
  assert.equal(toolCards(a1)[0], c1, "the history card is the same node");
  assert.equal(c1.querySelector("[data-tool-status]").getAttribute("data-tool-status"), "proposed", "U10: the superseded turn's late status is ignored");
  assert.equal(toolCards(articleOf(root, "t2"))[0].querySelector("[data-tool-status]").getAttribute("data-tool-status"), "proposed", "…for t2 as well (superseded by t3)");
  assert.deepEqual(toolCards(root).map((c) => c.getAttribute("data-tool-call-id")), ["t1-c1", "t2-c1", "t3-c1"], "three turns, three history cards, in thread order");
});

// =================================================================================================
// B8 — only the newest surface speaks (the node half; the real click is in the Chrome module)
// =================================================================================================

pendingT0154B("T-0154 B8 (adversarial): older card wrappers are inert and an older surface's action sends nothing", async () => {
  const { root, deliver, sent, processor } = await threadWithCards();
  deliver(transcript("t3", "q3"));
  deliver(surface("turn-t3", 0));
  deliver(transcript("t4", "q4"));
  deliver(surface("turn-t4", 0));
  deliver(surface("turn-t3", 1)); // an older surface's later canvas does not make it the newest
  assert.equal(wrapperOf(root, "turn-t3").hasAttribute("inert"), true, "B8/H-T154-4: t3's card wrapper is inert");
  assert.equal(wrapperOf(root, "turn-t4").hasAttribute("inert"), false, "B8: the newest surface's wrapper is not inert");
  // Before T-0164 B: the vendor's onAction, called as the vendor would. After: the engine is
  // index.js's own, so the action is raised the only way a user can — a click on a rendered
  // Button (gateCard below adds one to each surface; T-0164 B4).
  const act = T0164_B
    ? (surfaceId) => gateButton(root, surfaceId)._fire("click")
    : (surfaceId) =>
        processor.onAction({ name: "choose", surfaceId, sourceComponentId: "button-1", timestamp: "2026-10-25T18:00:00Z", context: {} });
  if (T0164_B) {
    deliver(gateCard("turn-t3", 2));
    deliver(gateCard("turn-t4", 1));
  }
  act("turn-t3");
  assert.deepEqual(sent.filter((m) => m && m.type === "astra.ui.action"), [], "B8 (T-0024 criterion 9): an older card's action sends nothing");
  act("turn-t4");
  const actions = sent.filter((m) => m && m.type === "astra.ui.action");
  assert.equal(actions.length, 1, "B8: the newest surface's action is sent (the other direction)");
  assert.equal(actions[0].surface_id, "turn-t4");
  deliver(transcript("t5", "q5"));
  deliver(surface("turn-t5", 0));
  assert.equal(wrapperOf(root, "turn-t4").hasAttribute("inert"), true, "B8: t4 becomes inert once t5's surface is created");
  assert.equal(wrapperOf(root, "turn-t5").hasAttribute("inert"), false);
});

// =================================================================================================
// B10 — a reconnect replay is idempotent
// =================================================================================================

pendingT0154B("T-0154 B10: replayed transcript, utterances, tool calls and surfaces change no count", async () => {
  const { root, deliver } = await threadWithCards();
  const open = [
    transcript("t7", "what's the weather"),
    spoken("t7", "t7-u1", "Let me look."),
    spoken("t7", "t7-u2", "It is sunny."),
    toolCall("t7", "t7-c1", { name: "google_search", arguments: { query: "weather" } }),
    surface("turn-t7", 0),
    surface("turn-t7", 1),
  ];
  for (const m of open) deliver(m);
  const count = () => ({
    articles: root.querySelectorAll("article").length,
    utterances: utterances(root).length,
    bubbles: bubbles(root).length,
    cards: toolCards(root).length,
    surfaces: surfaceEls(root).length,
  });
  const before = count();
  assert.deepEqual(before, { articles: 1, utterances: 2, bubbles: 1, cards: 1, surfaces: 1 }, "premise");
  // T-0037: the hello of a reconnecting page replays the open turn's messages, marked replay.
  for (const m of open) deliver({ ...m, replay: true });
  for (const m of open) deliver({ ...m, replay: true });
  assert.deepEqual(count(), before, "B10: a reconnect replay changes neither the article count nor the utterance count");
});

// =================================================================================================
// B11 — degradation
// =================================================================================================

pendingT0154B("T-0154 B11: the ui panel with a host that has no card-slot provider logs one line, renders nothing, never throws", async () => {
  const index = await uiIndex();
  const doc = fakeDocument();
  const legacy = doc.createElement("section"); // a host still offering the retired #shown-channel
  doc.body.appendChild(legacy);
  const headBefore = doc.head.childNodes.length;
  const handlers = [];
  const logged = [];
  let result;
  await assert.doesNotReject(async () => {
    result = await index.mount({
      document: doc,
      shown: legacy,
      subscribe: (h) => (handlers.push(h), () => {}),
      send: () => Promise.resolve(),
      log: (l) => logged.push(String(l)),
    });
  }, "B11: mount never throws or rejects");
  for (const h of handlers) assert.doesNotThrow(() => h(surface("turn-t1", 0)));
  assert.equal(result && result.mounted, false, "B11: nothing is mounted without a card-slot provider");
  assert.equal(logged.length, 1, `B11: exactly one log line; got ${JSON.stringify(logged)}`);
  assert.match(logged[0], /slot/i, `B11: the line names the missing card-slot provider; got ${JSON.stringify(logged[0])}`);
  assert.equal(legacy.childNodes.length, 0, "B11: nothing rendered into the retired shown channel");
  assert.equal(doc.head.childNodes.length, headBefore, "…and no stylesheet was added");
  // T-0164 B2: re-keyed to our root; the vendor's element is still counted (neither may exist).
  assert.equal(doc.querySelectorAll("a2ui-surface, [data-a2ui-root]").length, 0, "…and no surface element exists");
});

pendingT0154B("T-0154 B11: with no ui panel at all, the thread still renders text and action cards", async () => {
  const { root, handle } = await thread();
  handle(transcript("t1", "q"));
  handle(surface("turn-t1", 0)); // no panel is subscribed: the slot stays empty, nothing throws
  handle(spoken("t1", "t1-u1", "a"));
  handle(toolCall("t1", "t1-c1"));
  const a = articleOf(root, "t1");
  assert.equal(squeeze(utterances(a)[0].textContent), "a");
  assert.equal(toolCards(a).length, 1);
  assert.equal(root.querySelectorAll("a2ui-surface, [data-a2ui-root]").length, 0, "the thread never renders A2UI itself (T-0024 criterion 4)");
});

// =================================================================================================
// B3 — the fence notice on the turn's meta line: T-0054 criteria 15-17 and rulings B8/B11, re-pinned
// =================================================================================================
//
// T-0054's panel rows in tests/hud/ui_fence.test.mjs mounted hud/src/ui/index.js into the retired
// #shown-channel and read [data-ui-fence], [data-ui-fence-canvases] and [data-ui-fence-refused] off
// the panel's one container. B1 retires that container and B3 puts "the fence notice" on the turn's
// meta line, so those five rows were moved here (G2, tied to B3/B4) and re-keyed to the article of
// the turn the fence names. What is kept, word for word: the three hooks and their values (the
// status, the server's canvas count, the server's refusal count), a healthy turn adding nothing to
// the DOM (criterion 16), a refused-only fence before any surface reaching the page (criterion 17),
// and a turn's own fence surviving that turn's own later canvases (ruling B11). What changes, by
// ruling B8's own reason ("the pane must never carry a verdict over a surface it is not about"): an
// abandoned turn's verdict now stays in *its own* article, under its own card, as history; it is
// never shown in, or over, the next turn's article. Labels are criterion 15's table (the strings of
// hud/src/ui/fence.js's labelFor, verbatim, as the B3 meta row above already reads them).

/** A stable serialisation of a subtree of the DOM double (criterion 16's before/after). */
function serialize(node) {
  if (!node) return "";
  if (node.nodeType === 3) return node.data;
  if (node.nodeType !== 1) return "";
  const attrs = [...node._attrs.entries()]
    .sort(([a], [b]) => (a < b ? -1 : 1))
    .map(([k, v]) => ` ${k}="${v}"`)
    .join("");
  return `<${node.localName}${attrs}>${node.childNodes.map(serialize).join("")}</${node.localName}>`;
}

/** The fence marks inside one turn's article: on the meta line, or on an element inside it. */
function fenceMarks(article) {
  const meta = article.querySelector("[data-turn-meta]");
  const marked = article.querySelectorAll("[data-ui-fence]");
  return {
    meta,
    marked,
    status: marked.length ? marked[0].getAttribute("data-ui-fence") : null,
    canvases: (article.querySelector("[data-ui-fence-canvases]") || { getAttribute: () => null }).getAttribute("data-ui-fence-canvases"),
    refused: (article.querySelector("[data-ui-fence-refused]") || { getAttribute: () => null }).getAttribute("data-ui-fence-refused"),
    onMeta: marked.length > 0 && !!meta && marked.every((m) => m === meta || meta.contains(m)),
    text: meta ? squeeze(meta.textContent) : "",
  };
}

pendingT0154B("T-0154 B3 (T-0054 criterion 16 re-pinned): a healthy turn's fence adds nothing to its article", async () => {
  const { root, deliver } = await threadWithCards();
  deliver(transcript("t6", "make me a card"));
  deliver(spoken("t6", "t6-u1", "Here it is."));
  deliver(surface("turn-t6", 0));
  const article = articleOf(root, "t6");
  const before = serialize(article);
  for (const healthy of [fence("t6", { canvases: 1, close: "written" }), fence("t6", { canvases: 1, close: "inferred" })]) {
    deliver(healthy);
    assert.equal(
      serialize(article),
      before,
      'T-0054 criterion 16: for status "whole" with speech_refused 0 the article is byte-identical — no ' +
        `notice, no [data-ui-fence] mark, the card untouched. The message was ${JSON.stringify(healthy)}`,
    );
  }
  assert.equal(fenceMarks(article).marked.length, 0, "no [data-ui-fence] anywhere in a healthy turn");
});

pendingT0154B("T-0154 B3 (T-0054 criterion 15 re-pinned): reassembled and refused reach the turn's meta line with the server's counts", async () => {
  const { root, deliver, sent } = await threadWithCards();
  deliver(transcript("t6", "make me a card"));
  deliver(surface("turn-t6", 0));
  deliver(fence("t6", { canvases: 2, close: "inferred", continued: true, speech_refused: 1 }));
  const article = articleOf(root, "t6");
  const marks = fenceMarks(article);
  assert.ok(marks.meta, "B3: t6's article has a meta line ([data-turn-meta])");
  assert.equal(marks.status, "reassembled", `T-0054 criterion 15: [data-ui-fence] carries the status, in t6's article; marks ${JSON.stringify({ ...marks, meta: !!marks.meta, marked: marks.marked.length })}`);
  assert.ok(marks.onMeta, "B3: the fence marks sit on the turn's meta line (or inside it), not elsewhere in the article");
  assert.equal(marks.canvases, "2", "the server's canvas count, echoed");
  assert.equal(marks.refused, "1", "the server's refusal count, echoed");
  const text = marks.text;
  assert.ok(text.includes("reassembled from 2 canvases"), `criterion 15's wording reaches the meta line; it reads ${JSON.stringify(text)}`);
  assert.ok(text.includes("1 piece of interface data was not spoken"), `…with the refusal clause; it reads ${JSON.stringify(text)}`);
  assert.equal(sent.length, 0, "T-0054 criterion 17: nothing is sent, ever");
});

pendingT0154B("T-0154 B3 (T-0054 criterion 17 re-pinned): a refused-only fence before any surface opens its turn and says so; another turn's fence stays there", async () => {
  const { root, deliver, sent, logged } = await threadWithCards();
  assert.doesNotThrow(() => deliver(fence("t6", { canvases: 0, close: null, speech_refused: 2 })));
  assert.deepEqual(turnIds(root), ["t6"], "B2: a fence notice is content — it opens t6's article");
  const t6 = fenceMarks(articleOf(root, "t6"));
  assert.equal(t6.status, "refused-only", "criterion 17: the turn where nothing was shown still gets its notice");
  assert.ok(t6.text.includes("nothing was shown"), `…in words: ${JSON.stringify(t6.text)}`);
  deliver(surface("turn-t6", 0));
  assert.doesNotThrow(() => deliver(fence("t99", { canvases: 9, close: "abandoned" })));
  assert.deepEqual(turnIds(root), ["t6", "t99"], "t99's notice opens t99's own article, at the end");
  assert.equal(fenceMarks(articleOf(root, "t99")).status, "abandoned");
  assert.equal(fenceMarks(articleOf(root, "t6")).status, "refused-only", "…and does not touch t6's");
  assert.equal(sent.length, 0, "criterion 17: nothing is sent");
  assert.ok(logged.length <= 3, `at most one log line per unusual message: ${JSON.stringify(logged)}`);
});

pendingT0154B("T-0154 B3 (T-0054 ruling B8 re-pinned): an abandoned turn's verdict stays in its own article and never reaches the next turn's", async () => {
  const { root, deliver } = await threadWithCards();
  deliver(transcript("t6", "first"));
  deliver(surface("turn-t6", 0));
  deliver(fence("t6", { canvases: 5, close: "abandoned", continued: true }));
  assert.equal(fenceMarks(articleOf(root, "t6")).status, "abandoned", "premise: t6's verdict is on its meta line");
  const t6Meta = fenceMarks(articleOf(root, "t6")).text;
  assert.ok(t6Meta.includes("the interface never finished"), `premise, in words: ${JSON.stringify(t6Meta)}`);
  // Turn 2: a healthy surface with its own id and — T-0054 criterion 10 — no fence message at all.
  deliver(transcript("t7", "second"));
  deliver(surface("turn-t7", 0));
  const t7 = articleOf(root, "t7");
  assert.ok(t7.querySelector("[data-ui-surface]"), "premise: t7's card rendered in t7's article");
  for (const mark of ["[data-ui-fence]", "[data-ui-fence-canvases]", "[data-ui-fence-refused]"]) {
    assert.equal(t7.querySelector(mark), null, `ruling B8: t7 carries nothing about t6's fence (${mark})`);
  }
  assert.ok(!/never finished|cut off|reassembled/.test(t7.textContent), `ruling B8: no verdict text in t7's article: ${JSON.stringify(squeeze(t7.textContent))}`);
  assert.equal(fenceMarks(articleOf(root, "t6")).status, "abandoned", "B2/B3: t6's verdict stays with t6's card, as history (it is about t6, and says so by where it is)");
});

pendingT0154B("T-0154 B3 (T-0054 ruling B11 re-pinned): a turn's own fence survives that turn's own later canvases", async () => {
  const { root, deliver } = await threadWithCards();
  // The wire order of a real Session (T-0054 ruling A12 i): the fence of canvas k precedes its surface.
  deliver(fence("t6", { canvases: 1, close: "open", continued: false }));
  deliver(surface("turn-t6", 1));
  assert.equal(fenceMarks(articleOf(root, "t6")).status, "incomplete", 'criterion 15: close "open" reads "incomplete" and t6\'s own surface does not erase it');
  assert.ok(fenceMarks(articleOf(root, "t6")).text.includes("the interface was cut off"));
  deliver(fence("t6", { canvases: 1, close: "written", continued: false, speech_refused: 1 }));
  deliver(surface("turn-t6", 2));
  const t6 = fenceMarks(articleOf(root, "t6"));
  assert.equal(t6.refused, "1", "criterion 15's last row: the refusal count stays over t6's next canvas");
  assert.ok(t6.text.includes("1 piece of interface data was not spoken"), `…in words: ${JSON.stringify(t6.text)}`);
  deliver(surface("turn-t7", 1));
  assert.equal(fenceMarks(articleOf(root, "t7")).marked.length, 0, "ruling B8, unchanged by B11: t7 carries none of t6's marks");
  assert.equal(fenceMarks(articleOf(root, "t6")).refused, "1", "…and t6 keeps its own");
});

// =================================================================================================
// T-0164 group B — the swap: the panel renders with our engine (node halves; Chrome halves in
// tests/integration/test_hud_ui_surface.py and test_hud_rehearsal.py)
// =================================================================================================
//
// These rows key on the new contract explicitly ([data-a2ui-root], implementation note 5), never on
// SURFACE_SEL, so before activation they fail against the vendor seam for the right reason: there is
// no light-DOM card root.

const ISO_UTC = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/; // Date.prototype.toISOString's shape

pendingT0164B("T-0164 B1: a decoded surface renders as a light-DOM [data-a2ui-root], the wrapper's only child; no <a2ui-surface> is ever created", async () => {
  const { doc, root, t, deliver } = await threadWithCards();
  deliver(transcript("t3", "omelette card please"));
  deliver(surface("turn-t3", 0, { text: "Omelette: 3 eggs" }));
  const wrapper = wrapperOf(root, "turn-t3");
  assert.ok(t.cardSlot("turn-t3").contains(wrapper), "premise (T-0154 B4): the wrapper is in t3's slot");
  const kids = wrapper.children;
  assert.equal(kids.length, 1, `B1: the wrapper holds exactly one element (the card root); it holds ${kids.map((k) => k.localName)}`);
  assert.ok(kids[0].hasAttribute("data-a2ui-root"), `B1: [data-ui-surface] > [data-a2ui-root]; the child is <${kids[0].localName}>`);
  assert.equal(squeeze(kids[0].textContent), "Omelette: 3 eggs", "B1/A4: the model's Text, verbatim, as the card's text");
  const vendorTags = doc.created.filter((e) => e.localName === "a2ui-surface").length;
  assert.equal(vendorTags, 0, `B1: no <a2ui-surface> element is created at all; ${vendorTags} were`);
  assert.equal(root.querySelectorAll("a2ui-surface").length, 0, "…and none is in the thread");
});

pendingT0164B("T-0164 B4: a click on the newest card's gate button sends exactly one astra.ui.action in today's wire shape", async () => {
  const { root, deliver, sent } = await threadWithCards();
  deliver(transcript("t3", "q3"));
  deliver(gateCard("turn-t3"));
  deliver(transcript("t4", "q4"));
  deliver(gateCard("turn-t4"));
  gateButton(root, "turn-t3")._fire("click"); // an older card (B8): nothing
  assert.deepEqual(sent.filter((m) => m && m.type === "astra.ui.action"), [], "B8 (kept): an older card's button sends nothing");
  gateButton(root, "turn-t4")._fire("click");
  const actions = sent.filter((m) => m && m.type === "astra.ui.action");
  assert.equal(actions.length, 1, `B4: one click, one astra.ui.action; sent ${JSON.stringify(actions)}`);
  const [a] = actions;
  // action.js's wire (unchanged, T-0024 criterion 9): {type, surface_id, name, context, data_model}.
  assert.deepEqual(Object.keys(a).sort(), ["context", "data_model", "name", "surface_id", "type"], `B4: the wire keys are today's; got ${JSON.stringify(a)}`);
  assert.equal(a.surface_id, "turn-t4", "B4: the gate's id, the one inside the messages");
  assert.equal(a.name, "__gate.clicked", "B4: the gate's reserved event name, verbatim");
  assert.deepEqual(Object.keys(a.context).sort(), ["a2ui", "label"], `B4: context.label and context.a2ui and no other key; got ${JSON.stringify(a.context)}`);
  assert.equal(a.context.label, GATE_LABEL, "B4: context.label is the model's label, byte-equal");
  assert.deepEqual(Object.keys(a.context.a2ui).sort(), ["sourceComponentId", "timestamp"], `B4: context.a2ui; got ${JSON.stringify(a.context.a2ui)}`);
  assert.equal(a.context.a2ui.sourceComponentId, "go", "B4: sourceComponentId is the Button's own id");
  assert.match(a.context.a2ui.timestamp, ISO_UTC, "B4: the timestamp is an ISO-8601 string from toISOString");
  assert.equal(a.data_model, null, "B4/A9: no sendDataModel on this surface, so data_model is null");
});

pendingT0164B("T-0164 B1/B4: dataModel() reads the engine — a sendDataModel surface's click carries the resolved path and the surface model", async () => {
  const { root, deliver, sent } = await threadWithCards();
  const id = "turn-t5";
  const model = { answers: { q1: ["b"], q2: [] }, score: "" };
  deliver(transcript("t5", "quiz me"));
  deliver({
    type: "astra.ui.surface",
    surface_id: id,
    canvas_idx: 0,
    final: true,
    repaired: 0,
    replay: false,
    messages: [
      { version: "v0.9.1", createSurface: { surfaceId: id, catalogId: CATALOG, sendDataModel: true } },
      {
        version: "v0.9.1",
        updateComponents: {
          surfaceId: id,
          components: [
            { id: "root", component: "Button", child: "lbl", action: { event: { name: "check_quiz", context: { quizId: "fractions_1", answers: { path: "/answers" } } } } },
            { id: "lbl", component: "Text", text: "Check answers" },
          ],
        },
      },
      { version: "v0.9.1", updateDataModel: { surfaceId: id, path: "/", value: model } },
    ],
  });
  gateButton(root, id)._fire("click");
  const actions = sent.filter((m) => m && m.type === "astra.ui.action");
  assert.equal(actions.length, 1, `one click, one action; sent ${JSON.stringify(actions)}`);
  const [a] = actions;
  assert.equal(a.name, "check_quiz");
  assert.deepEqual(a.context.answers, { q1: ["b"], q2: [] }, "B4 / A10: {path: '/answers'} resolved one level deep from the data model");
  assert.equal(a.context.quizId, "fractions_1", "a literal passes through verbatim");
  assert.deepEqual(a.data_model, model, "B1: data_model is engine.dataModelFor(the newest surface) — the surface's own model");
});

pendingT0164B("T-0164 B2 (A7 through the panel): the same canvas re-sent at 3 later canvas_idx creates no element and keeps the root", async () => {
  const { doc, root, deliver } = await threadWithCards();
  deliver(transcript("t1", "q1"));
  deliver(gateCard("turn-t1", 0));
  const [rootEl] = wrapperOf(root, "turn-t1").children;
  assert.ok(rootEl && rootEl.hasAttribute("data-a2ui-root"), "premise (B1): t1's card root is light DOM");
  const created = doc.created.length;
  const wiped = doc.wiped.length;
  for (const idx of [1, 2, 3]) deliver(gateCard("turn-t1", idx)); // identical messages, later canvases
  assert.equal(wrapperOf(root, "turn-t1").getAttribute("data-ui-canvas-idx"), "3", "premise: the bridge applied all three (canvas 3 is the last applied)");
  assert.equal(wrapperOf(root, "turn-t1").children[0], rootEl, "B2/A7: the card root is the same node");
  assert.equal(doc.created.length - created, 0, `B2/A7: an identical canvas creates no element; ${doc.created.length - created} were created`);
  assert.equal(doc.wiped.length - wiped, 0, "B2/A7: …and no element's children were wiped by a textContent write");
});

// =================================================================================================
// T-0162 group D — the page-wide clear: the thread, its cards and the Tools pane (D-2, D-4)
// =================================================================================================
//
// docs/tasks/T-0162.md D-2: "On astra.session.cleared (live or replayed), every panel clears once per
// new resets value: the thread: articles, card slots, A2UI surfaces and inline tool cards; the Tools
// pane; …  A repeated resets value clears nothing." D-4: "Before the reset, older cards stay inert
// (T-0154 B8). After it, the first new surface is the newest one and sends actions." Ruling
// (ordering with T-0164): the ui panel drops its surfaces through engine.drop(surfaceId).
//
// API this pins (names the task leaves open):
//   createThread(root) -> { handle, cardSlot, clear() }   clear() empties the thread (owned file
//                     note "hud/src/panels/thread.js: clear()"); handle(astra.session.cleared) clears
//                     once per new `resets`, as every other panel does on its own subscription.
//   ui/index.js and tools/index.js hear of the reset on their host.subscribe, the one seam a
//   mounted panel has (mountPanels returns no panel handles to main.js).
//
// Adversarial (R1): a card from before the clear never speaks after it, even if its button node is
// still held (a DOM-only clear that leaves the engine's surface and the panel's card map would let
// it); the Tools pane does not re-render the old turns when the next call arrives (a clear of the
// DOM alone, with the model kept, would). Guard both ways: the hello's replayed `resets` keeps the
// new session's thread; resets 2 clears it.
//
// Strict xfail until docs/tasks/T-0162.md's status records "group D: tests …, activated <sha>".

const T0162_TASK = fileURLToPath(new URL("docs/tasks/T-0162.md", REPO));
const T0162_STATUS = existsSync(T0162_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0162_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const T0162_D_PENDING =
  !/\bgroup D:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i.test(T0162_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0162D(name, fn) {
  test(name + (T0162_D_PENDING ? " [pending T-0162 D: strict xfail until group D is activated]" : ""), async (t) => {
    if (!T0162_D_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail("XPASS(strict): this T-0162 group D row passed while docs/tasks/T-0162.md's status does not record group D's activation.");
    }
    t.diagnostic("expected failure (T-0162 D pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

const sessionCleared = (resets, extra = {}) => ({ type: "astra.session.cleared", resets, ended_turn_id: null, memory_forgotten: 0, replay: false, ...extra });

/** Two pre-reset turns, each with a transcript, an utterance, an inline tool card and a gate card. */
async function twoTurnsWithCards() {
  const rig = await threadWithCards();
  const { root, deliver } = rig;
  for (const n of [3, 4]) {
    deliver(turnState(`t${n}`, "THINKING", 100));
    deliver(transcript(`t${n}`, `question ${n}`));
    deliver(spoken(`t${n}`, `u${n}`, `Answer ${n}.`));
    deliver(toolCall(`t${n}`, `c${n}`, { status: "executed", action: "scrolled down" }));
    deliver(gateCard(`turn-t${n}`));
    deliver(turnState(`t${n}`, "IDLE"));
  }
  // premises, from the messages above (T-0154 B2/B4/B7/B8)
  assert.deepEqual(turnIds(root), ["t3", "t4"], "premise: two articles");
  assert.equal(toolCards(root).length, 2, "premise: two inline tool cards");
  assert.equal(root.querySelectorAll("[data-ui-surface]").length, 2, "premise: two card wrappers");
  assert.equal(wrapperOf(root, "turn-t3").hasAttribute("inert"), true, "D-4 premise (T-0154 B8): before the reset the older card is inert");
  assert.equal(wrapperOf(root, "turn-t4").hasAttribute("inert"), false, "D-4 premise: the newest is not");
  return rig;
}

/** Nothing of a pre-reset turn is anywhere in the document (not only not in the thread). */
function assertEmptied(doc, root, why) {
  assert.deepEqual(turnIds(root), [], `${why}: no article survives; found ${JSON.stringify(turnIds(root))}`);
  assert.equal(root.querySelectorAll("[data-transcript]").length, 0, `${why}: no user bubble`);
  assert.equal(root.querySelectorAll("[data-utterance-id]").length, 0, `${why}: no utterance`);
  assert.equal(doc.documentElement.querySelectorAll("[data-tool-call-id]").length, 0, `${why}: no inline tool card`);
  assert.equal(doc.documentElement.querySelectorAll("[data-ui-surface]").length, 0, `${why}: no card wrapper (card slot)`);
  assert.equal(doc.documentElement.querySelectorAll("[data-a2ui-root]").length, 0, `${why}: no A2UI card root left in the page`);
  assert.equal(squeeze(root.textContent).includes("Answer"), false, `${why}: no pre-reset text in the thread`);
}

pendingT0162D("T-0162 D-2: astra.session.cleared empties the thread — articles, bubbles, utterances, card slots, cards and tool cards", async () => {
  const { doc, root, deliver } = await twoTurnsWithCards();
  deliver(sessionCleared(1, { ended_turn_id: null, memory_forgotten: 2 }));
  assertEmptied(doc, root, "D-2");
});

pendingT0162D("T-0162 D-2: createThread(root).clear() empties the thread's own DOM (the owned clear())", async () => {
  const { doc, root, t, handle } = await thread();
  assert.equal(typeof t.clear, "function", "D-2: createThread returns { handle, cardSlot, clear }");
  handle(transcript("t3", "q3"));
  handle(spoken("t3", "u3", "Answer 3."));
  handle(toolCall("t3", "c3"));
  const slot = t.cardSlot("turn-t3");
  assert.ok(slot, "premise: t3's card slot exists");
  t.clear();
  assert.deepEqual(turnIds(root), [], "clear(): no article");
  assert.equal(slot.isConnected, false, "clear(): the old card slot is gone from the page");
  assert.equal(doc.documentElement.querySelectorAll("[data-tool-call-id]").length, 0, "clear(): no tool card");
  // A slot asked for afterwards is a new one, in a new article (the old one is not re-attached).
  handle(transcript("t5", "q5"));
  const fresh = t.cardSlot("turn-t5");
  assert.ok(fresh && fresh !== slot && fresh.isConnected, "after clear() a new turn gets its own new slot");
  assert.deepEqual(turnIds(root), ["t5"]);
});

pendingT0162D("T-0162 D-4: after the clear the first new surface is the newest — not inert — and its button sends one action", async () => {
  const { root, deliver, sent } = await twoTurnsWithCards();
  deliver(sessionCleared(1));
  deliver(transcript("t5", "q5"));
  deliver(gateCard("turn-t5"));
  assert.deepEqual(turnIds(root), ["t5"], "the new session's thread holds its own turn only");
  assert.equal(wrapperOf(root, "turn-t5").hasAttribute("inert"), false, "D-4: the first new surface is not inert");
  gateButton(root, "turn-t5")._fire("click");
  const actions = sent.filter((m) => m && m.type === "astra.ui.action");
  assert.equal(actions.length, 1, `D-4: the first new surface sends actions; sent ${JSON.stringify(actions)}`);
  assert.equal(actions[0].surface_id, "turn-t5");
});

pendingT0162D("T-0162 D-2 / R1 (adversarial): a card from before the clear never sends an action after it, even with its button node held", async () => {
  const { root, deliver, sent } = await twoTurnsWithCards();
  const held = gateButton(root, "turn-t4"); // the newest card's button: it would speak before the clear
  deliver(sessionCleared(1));
  held._fire("click");
  assert.deepEqual(sent.filter((m) => m && m.type === "astra.ui.action"), [], "R1: the dropped surface (engine.drop) sends nothing");
  // …and still nothing once a new surface exists (the old one does not become "rendered" again)
  deliver(transcript("t5", "q5"));
  deliver(gateCard("turn-t5"));
  held._fire("click");
  const actions = sent.filter((m) => m && m.type === "astra.ui.action");
  assert.ok(actions.every((a) => a.surface_id !== "turn-t4"), `R1: no action ever names the pre-reset surface; sent ${JSON.stringify(actions)}`);
});

pendingT0162D("T-0162 D-2 (guard, too wide): the hello's replayed cleared (same resets) keeps the new session's thread, node for node", async () => {
  const { root, deliver } = await twoTurnsWithCards();
  deliver(sessionCleared(1));
  deliver(transcript("t5", "q5"));
  deliver(spoken("t5", "u5", "Answer 5."));
  deliver(toolCall("t5", "c5"));
  deliver(gateCard("turn-t5"));
  const article = articleOf(root, "t5");
  const card = wrapperOf(root, "turn-t5");
  deliver(sessionCleared(1, { replay: true })); // the reconnect's hello reply begins with it (A-5)
  deliver(sessionCleared(1));
  assert.equal(articleOf(root, "t5"), article, "D-2: a repeated resets value clears nothing — the same article node");
  assert.equal(wrapperOf(root, "turn-t5"), card, "…the same card");
  assert.equal(toolCards(article).length, 1, "…and its tool card");
  // the other side: a second Restart clears the second session
  deliver(sessionCleared(2));
  assert.deepEqual(turnIds(root), [], "D-2: resets 2 is new, so the thread clears again");
});

pendingT0162D("T-0162 D-2 (reviewer A T-5): an unattributable surface's own end article is cleared too", async () => {
  const { doc, root, deliver } = await twoTurnsWithCards();
  deliver(surface("loose-card-1", 0)); // an id that names no turn: its own article at the end (T-0154 B4)
  const all = () => root.querySelectorAll("article");
  assert.equal(all().length, 3, `premise: two turn articles and one end article; found ${all().length}`);
  deliver(sessionCleared(1));
  assert.equal(all().length, 0, `D-2: every article goes, the unattributable one included; ${all().length} remain`);
  assert.equal(doc.documentElement.querySelectorAll("[data-ui-surface]").length, 0, "…with its card");
});

pendingT0162D("T-0162 D-2 (reviewer A T-4): a CLI restarted behind the page counts from 1 again — 1 after 2 clears", async () => {
  const { root, deliver } = await twoTurnsWithCards();
  deliver(sessionCleared(1));
  deliver(sessionCleared(2));
  deliver(transcript("t5", "q5"));
  deliver(spoken("t5", "u5", "Answer 5."));
  assert.deepEqual(turnIds(root), ["t5"], "premise: the session after reset 2 has one turn");
  deliver(sessionCleared(1, { replay: true })); // a new CLI's first reset, replayed at the reconnect's hello
  assert.deepEqual(turnIds(root), [], "D-2: a different resets value (1 after 2) is a new reset and clears");
});

// --- the Tools pane (hud/src/tools/index.js) ------------------------------------------------------

async function toolsPane({ lastResets } = {}) {
  const { mount } = await load("tools/index.js");
  const doc = fakeDocument();
  const main = doc.createElement("main");
  doc.body.appendChild(main);
  const handlers = [];
  const host = { document: doc, subscribe: (h) => (handlers.push(h), () => {}), log: () => {} };
  if (lastResets !== undefined) host.lastResets = () => lastResets;
  mount(host);
  const deliver = (m) => handlers.forEach((h) => h(m));
  const calls = () => doc.documentElement.querySelectorAll("[data-tools-call]");
  const turnsShown = () => doc.documentElement.querySelectorAll("[data-tools-turn]").map((e) => e.getAttribute("data-tools-turn"));
  return { doc, deliver, calls, turnsShown };
}

pendingT0162D("T-0162 D-2: the Tools pane clears on astra.session.cleared, and the next call does not bring the old turns back", async () => {
  const p = await toolsPane();
  for (const n of [3, 4]) {
    p.deliver(turnState(`t${n}`, "THINKING", 100));
    p.deliver(toolCall(`t${n}`, `c${n}`, { status: "proposed" }));
    p.deliver(toolCall(`t${n}`, `c${n}`, { status: "executed", action: "scrolled down" }));
  }
  assert.equal(p.calls().length, 2, "premise: two calls in the pane (B-1)");
  p.deliver(sessionCleared(1));
  assert.equal(p.calls().length, 0, "D-2: no call line survives the reset");
  assert.deepEqual(p.turnsShown(), [], "D-2: no turn group survives the reset");
  assert.equal(p.doc.documentElement.querySelector("[data-tools-root]").getAttribute("data-tools-calls"), "0", "the pane's count is back to 0");
  // R1: a pane that only emptied its DOM re-renders the model's old turns at the next call
  p.deliver(turnState("t5", "THINKING", 100));
  p.deliver(toolCall("t5", "c9", { name: "press_key", arguments: { keys: "ctrl+l" } }));
  assert.deepEqual(p.turnsShown(), ["t5"], `R1: only the new session's turn is shown; shown ${JSON.stringify(p.turnsShown())}`);
  assert.equal(p.calls().length, 1);
  // guard, too wide: the hello's replay of the same reset keeps it
  p.deliver(sessionCleared(1, { replay: true }));
  assert.deepEqual(p.turnsShown(), ["t5"], "D-2: a repeated resets value clears nothing");
  p.deliver(sessionCleared(2));
  assert.equal(p.calls().length, 0, "D-2: resets 2 clears again");
});

pendingT0162D("T-0162 D-2 (reviewer A T-1, ruling (a)): a Tools pane mounted after the hello's replay seeds from host.lastResets and keeps the new session at the next replay", async () => {
  // The page heard reset 1 replayed at its first hello before this panel finished mounting; main.js's
  // tracker says so through host.lastResets(). A pane with its own fresh tracker would take the next
  // reconnect's replay of the same reset as new and wipe the session it belongs to.
  const p = await toolsPane({ lastResets: 1 });
  p.deliver(turnState("t5", "THINKING", 100));
  p.deliver(toolCall("t5", "c5", { status: "executed", action: "scrolled down" }));
  const line = p.calls()[0];
  assert.ok(line, "premise: the new session's call is shown");
  p.deliver(sessionCleared(1, { replay: true })); // the reconnect's hello
  assert.equal(p.calls().length, 1, "ruling (a): the seeded pane does not clear on the value the page already accepted");
  assert.equal(p.calls()[0], line, "…the same call node");
  p.deliver(sessionCleared(2));
  assert.equal(p.calls().length, 0, "the other side: a new value still clears the seeded pane");
});

// B7 (negative, fixed refs per .claude/rules/tests.md): the wire and the Conductor do not move in
// group B. BASE is the commit T-0164-b-swap branched from (main at 9b6ee6e0, the group A impl merge).
// TIP is null until group B merges; then it is pinned to that merge's M^2 and the row compares
// BASE..TIP. Until then it compares BASE with the working tree (uncommitted edits included).
const B7_BASE = "9b6ee6e028a33aea68d053135d0bf4400e4047a1";
const B7_TIP = "101c0c7e44dc70a926287a238d3b202b5756e280"; // M^2 of "Merge T-0164 group B impl" (set at the merge)
const B7_PATHS = ["local_astra/", "tests/fakes/", "hud/src/protocol.js", "hud/src/panels/thread.js", "services/"];

test("T-0164 B7: local_astra/, tests/fakes/, protocol.js, thread.js and services/ are unchanged since the group B base", (t) => {
  const repo = fileURLToPath(REPO);
  const git = (...args) => execFileSync("git", ["-C", repo, ...args], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
  try {
    git("cat-file", "-e", `${B7_BASE}^{commit}`);
  } catch {
    t.skip(`the base commit ${B7_BASE} is not in this clone`);
    return;
  }
  const groupC = /\bgroup C:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i.test(T0164_STATUS);
  assert.ok(!(groupC && B7_TIP === null), "B7: group C is activated but B7_TIP is still null — pin group B's merge tip (M^2) so the row compares BASE..TIP, not a moving tree");
  const range = B7_TIP ? [B7_BASE, B7_TIP] : [B7_BASE];
  const changed = git("diff", "--name-only", ...range, "--", ...B7_PATHS).split("\n").filter(Boolean);
  assert.deepEqual(changed, [], `B7: group B touches none of ${B7_PATHS.join(", ")}; changed since ${B7_BASE.slice(0, 8)}: ${changed.join(", ")}`);
});

// =================================================================================================
// T-0180 group C — the "Model thoughts" disclosure in the thread (docs/tasks/T-0180.md C1, C7, C9;
// *Tests to write*, Group C, thread.test.mjs). The disclosure's own rows (C1 summary, C2-C6) are in
// tests/hud/thoughts.test.mjs, which states the DOM hooks: details[data-model-thoughts] in the
// assistant body, [data-thought-attempt] blocks inside it.
//
// Gate: the rows below run through `pendingT0180C` (docs/tasks/T-0180.md's **Status:** line must record
// "group C: tests <a>..<b>, activated <sha>"; ASTRA_RUN_PENDING=1 runs them now). The C9 differential is
// a negative (green today by construction) and runs ungated: it must stay green through group C.
//
// Placement (James, 2026-10-28): "right under the turn's spoken/written answer … after the utterances and
// the turn's inline cards, as the last child of the assistant body. It stays last when later utterances or
// cards arrive for the same turn." The assistant body is read as the element that holds the utterances and
// the card slot (cardSlot()'s parent), so the rows do not depend on its class name.

const T0180_TASK = fileURLToPath(new URL("docs/tasks/T-0180.md", REPO));
const T0180_STATUS = existsSync(T0180_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0180_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const T0180_C_ACTIVATED = /\bgroup C:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const T0180_C_PENDING = !T0180_C_ACTIVATED.test(T0180_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0180C(name, fn) {
  test(name + (T0180_C_PENDING ? " [pending T-0180 C: strict xfail until group C is activated]" : ""), async (t) => {
    if (!T0180_C_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0180 group C row passed while docs/tasks/T-0180.md's status does " +
          "not record group C's activation (" + JSON.stringify(T0180_STATUS.slice(0, 80)) + ").",
      );
    }
    t.diagnostic("expected failure (T-0180 C pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0180 C gate (thread rows): only T-0180's own group C phrase switches them on", () => {
  assert.ok(T0180_STATUS.length > 0, "docs/tasks/T-0180.md must carry a **Status:** line");
  assert.equal(T0180_C_ACTIVATED.test("Group B tests activated 2026-10-28 at 8c6c8759 (publish 21)"), false);
  assert.equal(T0180_C_ACTIVATED.test("(Group A activated 2026-10-28 at 38e90ce8: main c0cb001e merged)"), false);
  assert.equal(T0180_C_ACTIVATED.test("in progress (group C: tests a1b2c3d..e4f5a6b, activated c3d4e5f)"), true);
});

/** One astra.thought (contracts/protocol.py AstraThought), every field written out. */
function thoughtMsg(turn_id, attempt_id, { step = 1, kind = "step", text = "", end = "open", omitted_chars = 0, replay = false } = {}) {
  return { type: "astra.thought", turn_id, request_id: `${attempt_id}-req`, attempt_id, step, kind, text, omitted_chars, end, replay };
}

async function assertDecodes(m) {
  const { decode } = await load("protocol.js");
  const d = decode(JSON.stringify(m));
  assert.equal(d.kind, "astra", `premise: the row's astra.thought decodes (A2): ${JSON.stringify(d).slice(0, 200)}`);
}

function disclosureIn(article) {
  const found = article.querySelectorAll("[data-model-thoughts]");
  assert.equal(found.length, 1, `C1: one [data-model-thoughts] in the turn's article; found ${found.length}`);
  assert.equal(found[0].localName, "details", "C1: a native <details>");
  return found[0];
}

/** The C1 checks: in the assistant body (cardSlot's parent, the utterances' parent), its last element child,
 *  after every utterance, the card slot's cards and every inline tool card in document order. */
function assertLast(t, root, turn, why) {
  const article = articleOf(root, turn);
  const d = disclosureIn(article);
  const body = t.cardSlot(`turn-${turn}`).parentNode;
  assert.equal(d.parentNode, body, `C1 (${why}): the disclosure sits in the assistant body, beside the card slot`);
  assert.ok(d.closest("[data-assistant]"), `C1 (${why}): inside the assistant row`);
  for (const u of utterances(article)) assert.equal(u.parentNode, body, `premise (${why}): utterances live in the same body`);
  assert.equal(body.lastElementChild, d, `C1 (${why}): the disclosure is the assistant body's last child; the last is <${body.lastElementChild && body.lastElementChild.localName} ${body.lastElementChild && body.lastElementChild.getAttributeNames().join(" ")}>`);
  for (const el of [...utterances(article), ...toolCards(article), ...article.querySelectorAll("[data-ui-surface]")])
    assert.ok(before(root, el, d), `C1 (${why}): ${el.getAttributeNames().join(" ")} comes before the disclosure`);
  return d;
}

pendingT0180C("T-0180 C1: the disclosure sits under the answer and the turn's cards, the assistant body's last child, and stays last", async () => {
  const rig = await threadWithCards();
  const { root, deliver, t } = rig;
  const first = thoughtMsg("t1", "t1-a1", { text: "Plan the card.", end: "done" });
  await assertDecodes(first);
  deliver(turnState("t1", "THINKING", 100));
  deliver(transcript("t1", "Show me a recipe"));
  deliver(spoken("t1", "t1-u1", "Here is one."));
  deliver(toolCall("t1", "c1", { status: "executed", action: "scrolled down" }));
  deliver(gateCard("turn-t1"));
  deliver(first);
  assert.equal(root.querySelectorAll("[data-ui-surface]").length, 1, "premise: the card rendered in the slot");
  const d = assertLast(t, root, "t1", "after answer, tool card and card");
  // later content of the same turn
  deliver(spoken("t1", "t1-u2", "And a second line."));
  assertLast(t, root, "t1", "after a later utterance");
  deliver(toolCall("t1", "c2", { status: "executed", action: "scrolled down" }));
  assertLast(t, root, "t1", "after a later tool card");
  deliver(surface("turn-t1", 1, { text: "a later canvas" }));
  assertLast(t, root, "t1", "after a later card canvas");
  deliver(fence("t1", { close: "open", canvases: 2 }));
  deliver(thoughtMsg("t1", "t1-a2", { step: 2, text: "Second step.", end: "open" }));
  assert.equal(assertLast(t, root, "t1", "after a fence notice and a second attempt"), d, "C1: never rebuilt: the same <details>");
});

pendingT0180C("T-0180 C1: a thought that arrives before the answer still ends up under it", async () => {
  const { root, t, handle } = await thread();
  handle(transcript("t2", "Where is the button?"));
  handle(thoughtMsg("t2", "t2-a1", { text: "Look first.", end: "open" }));
  assertLast(t, root, "t2", "thought first");
  handle(spoken("t2", "t2-u1", "At the top right."));
  handle(thoughtMsg("t2", "t2-a1", { text: "Look first. Found it.", end: "done" }));
  handle(toolCall("t2", "c9", { status: "executed", action: "clicked" }));
  const d = assertLast(t, root, "t2", "answer and tool card after the thought");
  assert.ok(before(root, utterances(articleOf(root, "t2"))[0], d), "C1: the answer is above the disclosure");
});

pendingT0180C("T-0180 C1: a thought-only turn creates its article (no transcript, no answer)", async () => {
  const { root, t, handle } = await thread();
  handle(turnState("t7", "THINKING", 10)); // a turn state alone creates nothing (T-0154 B2)
  assert.deepEqual(turnIds(root), [], "premise: no article from a turn state");
  handle(thoughtMsg("t7", "t7-a1", { text: "Only thinking so far.", end: "open" }));
  assert.deepEqual(turnIds(root), ["t7"], "C1: the first astra.thought creates the turn's article");
  const article = articleOf(root, "t7");
  assert.equal(bubbles(article).length, 0, "C1: no user bubble is invented");
  assertLast(t, root, "t7", "thought-only turn");
  handle(thoughtMsg("t7", "t7-a1", { text: "Only thinking so far, then done.", end: "done" }));
  assert.deepEqual(turnIds(root), ["t7"], "C1: an update creates no second article");
});

pendingT0180C("T-0180 C7: an update to a closed disclosure does not pull a reader who scrolled away back to the bottom", async () => {
  const { root, handle } = await thread();
  for (let i = 0; i < 6; i++) handle(spoken(`t${i}`, `t${i}-u1`, `Answer ${i}.`));
  handle(thoughtMsg("t5", "t5-a1", { text: "first", end: "open" }));
  // the reader scrolls up: 2000 px of thread, a 500 px window, 300 px from the top (1200 px from the bottom)
  root.scrollHeight = 2000;
  root.clientHeight = 500;
  root.scrollTop = 300;
  root._fire("scroll");
  handle(thoughtMsg("t5", "t5-a1", { text: "first second", end: "open" }));
  handle(thoughtMsg("t5", "t5-a2", { step: 2, text: "another attempt", end: "done" }));
  const d = disclosureIn(articleOf(root, "t5"));
  assert.equal(d.hasAttribute("open"), false, "premise: the disclosure is closed");
  assert.equal(d.querySelectorAll("[data-thought-attempt]").length, 2, "premise: the updates rendered");
  assert.equal(root.scrollTop, 300, `C7 (T-0154 B9): the reader stays where they scrolled; scrollTop is ${root.scrollTop}`);
});

// C9 (negative, ungated). BASE is main at 74ae8b25 (groups A and B merged, the page untouched), where this
// branch started. The fixture session (no astra.thought in it), a reset and a partial replay are fed to
// BASE's thread.js (read with `git archive`) and to the working tree's; the two thread DOMs must be
// byte-identical. An oracle independent of the implementation: BASE's own code.
const C9_BASE = "74ae8b25229c65494530c18a26e00e503a932c19";

test("T-0180 C9: with no astra.thought in the stream, the thread DOM is identical to BASE's", async (t) => {
  const repo = fileURLToPath(REPO);
  const git = (...args) => execFileSync("git", ["-C", repo, ...args], { stdio: ["ignore", "pipe", "pipe"], maxBuffer: 1 << 28 });
  try {
    git("cat-file", "-e", `${C9_BASE}^{commit}`);
  } catch {
    t.skip(`the base commit ${C9_BASE} is not in this clone`);
    return;
  }
  const dir = mkdtempSync(join(tmpdir(), "astra-hud-t0180-base-"));
  writeFileSync(join(dir, "src.tar"), git("archive", "--format=tar", C9_BASE, "hud/src"));
  execFileSync("tar", ["-xf", join(dir, "src.tar"), "-C", dir]);
  writeFileSync(join(dir, "package.json"), '{"type":"module","private":true}');
  const baseUrl = pathToFileURL(join(dir, "hud/src/panels/thread.js")).href;
  console.log(`# T-0180 C9 origin: base=${baseUrl} tip=${srcRoot().href}`);
  const base = await import(baseUrl);
  const tip = await threadModule();
  const { decode } = await load("protocol.js");
  const decoded = (raw) => decode(raw).msg;
  const stream = [
    ...FIXTURE_LINES.map(decoded),
    sessionCleared(1),
    ...FIXTURE_LINES.slice(0, 120).map(decoded),
    toolCall("t99", "c99", { status: "executed", action: "scrolled down" }),
    fence("t99", { close: "open", canvases: 2 }),
  ];
  assert.ok(!stream.some((m) => m && m.type === "astra.thought"), "premise: the stream carries no astra.thought");
  const render = (createThread) => {
    const doc = fakeDocument();
    const root = doc.createElement("div");
    root.setAttribute("data-thread", "");
    doc.body.appendChild(root);
    const th = createThread(root);
    for (const m of stream) th.handle(m);
    return { html: serialize(root), articles: articles(root).length };
  };
  const want = render(base.createThread);
  const got = render(tip.createThread);
  assert.ok(want.articles >= 2 && want.html.length > 500, `premise: BASE rendered a real thread (${want.articles} articles)`);
  assert.equal(got.html, want.html, "C9: a stream without astra.thought renders the same thread DOM as BASE");
});
