// T-0180 group C — the "Model thoughts" disclosure of each turn in the HUD thread (docs/tasks/T-0180.md
// C1-C6, rulings H-T180-2 / H-T180-5 and James 2026-10-27/28: collapsed by default, the page never opens
// it), the node half. The thread rows of C1 (placement), C7 and C9 are in tests/hud/thread.test.mjs; the
// Chrome half (click, keyboard, reload, contrast, overflow, the denoising view) is
// tests/integration/test_t0180_hud_thoughts.py.
//
// Every expected value below is written out by hand from the criteria (the labels, the status lines,
// the omitted line), never read from an implementation: hud/src/panels/thoughts.js does not exist
// when this file is written.
//
// Gate. Every row runs through `pendingT0180C`, the Status-phrase strict-xfail gate of
// tests/hud/vision.test.mjs, reading docs/tasks/T-0180.md. While its **Status:** line does not record
// "group C: tests <a>..<b>, activated <sha>", each row must FAIL (an XPASS fails the row: a stub that
// satisfies it is vacuous). ASTRA_RUN_PENDING=1 runs them for real now. Group A's or B's activation
// wording ("Group A activated … at 38e90ce8", "Group B tests activated … at 8c6c8759") does not switch
// these rows on.
//
// API these rows pin (the task names the files and leaves the JS names and hooks to the test-author;
// these are them):
//
//   hud/src/panels/thoughts.js — importable in node, no DOM at module top level, no Math.random.
//     export const THOUGHT_TYPE = "astra.thought"   (the convention of TOOL_CALL_TYPE, SURFACE_TYPE)
//     It exports the helper thread.js calls per article (implementation note C); its other names are
//     the implementer's. The rows drive it through the thread, the path main.js uses:
//
//   hud/src/panels/thread.js — createThread(root).handle(m) for every decoded message, as today.
//
//   DOM hooks (inside the turn's article[data-turn-id]):
//     details[data-model-thoughts]           one per turn that has had an astra.thought (C1); a native
//                                            <details>, never given `open` by the page (C2)
//       summary (its first element child)    "Model thoughts", plus " · N steps" with N > 1 blocks (C1)
//       [data-thought-attempt="<id>"]        one block per attempt_id, in first-seen order (C3)
//         [data-thought-label]               "Step 2 · re-ask" … (C3)
//         [data-thought-text]                the committed text, as text (textContent === msg.text)
//         [data-thought-status]              the status line; absent or empty for `done`
//         [data-thought-omitted]             the omitted line; absent or empty when omitted_chars == 0
//
// Readings the task leaves open (stated, not guessed silently):
//   - `done` shows no status line: C3 lists exactly four status lines (open, stopped_repeating,
//     interrupted, error) and none for a normal end.
//   - " · N steps": N is the number of blocks ("when there is more than one block"; recorded in the
//     task's "Group C test-author readings"). Most rows pin N where blocks and distinct step numbers
//     agree (steps 1, 2, 3); the shared-request_id row pins it where they do not (a step-1 attempt and
//     its step-1 retry: two blocks, "· 2 steps").
//   - The summary text is compared after removing any ▸▾▶▼ character and squeezing whitespace: the
//     arrow is the native marker (::marker), and a page that draws its own arrow glyph is not wrong.
//   - The omitted line's number may carry a thousands separator ("4000" or "4,000").
//   - "Once a block is final, a later `open` message is ignored": the later open's text is ignored too
//     (the block keeps the final text and status). A later *final* message after a final is unpinned.
//   - The viewer's own toggle is simulated in node by writing the `open` attribute directly, bypassing the
//     double's record of page writes; the real click and keyboard are the Chrome rows.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);

// --- the T-0180 group C strict-xfail gate --------------------------------------------------------

const TASK = fileURLToPath(new URL("docs/tasks/T-0180.md", REPO));
const STATUS = existsSync(TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(TASK, "utf8")) || [])[1] || "").trim()
  : "";
/** "group C: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6" — the phrase the orchestrator writes. */
const C_ACTIVATED = /\bgroup C:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const C_PENDING = !C_ACTIVATED.test(STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0180C(name, fn) {
  test(name + (C_PENDING ? " [pending T-0180 C: strict xfail until group C is activated]" : ""), async (t) => {
    if (!C_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0180 group C row passed while docs/tasks/T-0180.md's status does " +
          "not record group C's activation (" + JSON.stringify(STATUS.slice(0, 80)) + "). Either " +
          "group C has landed (record 'group C: tests …, activated <sha>' in the status) or the " +
          "row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0180 C pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0180 C gate: the status line is readable and only group C's own phrase activates the rows", () => {
  assert.ok(STATUS.length > 0, "docs/tasks/T-0180.md must carry a **Status:** line");
  // Hand-written samples, two of them today's wording for groups A and B.
  assert.equal(C_ACTIVATED.test("in progress. Group C next (the page; the DOM halves of B9). Groups B and C planned."), false);
  assert.equal(C_ACTIVATED.test("(Group A activated 2026-10-28 at 38e90ce8: main c0cb001e merged, T-0177 A in)"), false);
  assert.equal(C_ACTIVATED.test("Group B tests activated 2026-10-28 at 8c6c8759 (publish 21, differential 5)"), false);
  assert.equal(C_ACTIVATED.test("in progress (group B: tests 1a2b3c4, activated 9a8b7c6; group C: tests planned)"), false);
  assert.equal(C_ACTIVATED.test("in progress (group C: tests a1b2c3d..e4f5a6b, activated c3d4e5f)"), true);
  assert.equal(C_ACTIVATED.test("in progress. group C: tests a1b2c3d, activated c3d4e5f; nothing else"), true);
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

// Origin proof (.claude/rules/tests.md): where the modules under test were imported from.
console.log(`# thoughts origin: repo=${fileURLToPath(REPO)} modules=${srcRoot().href}`);

async function load(name) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch (err) {
    assert.fail(
      `T-0180 C: hud/src/${name} must be importable in node as an ES module with no DOM at module ` +
        `top level. node could not import it: ${err.message}`,
    );
  }
}

// --- a DOM double ----------------------------------------------------------------------------------
//
// tests/hud/thread.test.mjs's double (closed by delegation: a member it does not implement is a named
// failure; markup sinks throw), plus `open`: the HTMLDetailsElement property reflects the attribute, and
// every write the page makes — the property, setAttribute("open"), toggleAttribute("open") — is recorded
// in doc.openWrites, so C2's "the page never sets open" is a behaviour, not a grep.

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
    set nodeValue(v) {
      this.data = String(v);
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
    // HTMLDetailsElement.open reflects the attribute; a page write is recorded (C2).
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
      const n = String(name).toLowerCase();
      if (n === "open") doc.openWrites.push({ how: "setAttribute", value: true, tag });
      attrs.set(n, String(value));
    },
    getAttribute(name) {
      const n = String(name).toLowerCase();
      return attrs.has(n) ? attrs.get(n) : null;
    },
    hasAttribute(name) {
      return attrs.has(String(name).toLowerCase());
    },
    removeAttribute(name) {
      const n = String(name).toLowerCase();
      if (n === "open") doc.openWrites.push({ how: "removeAttribute", value: false, tag });
      attrs.delete(n);
    },
    toggleAttribute(name, force) {
      const n = String(name).toLowerCase();
      const on = force === undefined ? !attrs.has(n) : !!force;
      if (n === "open") doc.openWrites.push({ how: "toggleAttribute", value: on, tag });
      if (on) attrs.set(n, "");
      else attrs.delete(n);
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
        `the DOM double in tests/hud/thoughts.test.mjs has no Element.${String(key)}; extend the ` +
          "double (this is a gap in the test, not a defect in the page)",
      );
    },
    set(obj, key, value) {
      if (key === "open") {
        // route through the recording setter above
        return Reflect.set(obj, key, value, obj);
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

// --- the rig ----------------------------------------------------------------------------------------

async function rig() {
  const { createThread } = await load("panels/thread.js");
  assert.equal(typeof createThread, "function", "premise: thread.js exports createThread (T-0154 B)");
  const { decode } = await load("protocol.js");
  const doc = fakeDocument();
  const root = doc.createElement("div");
  root.setAttribute("data-thread", "");
  doc.body.appendChild(root);
  const t = createThread(root);
  /** Every message goes through the real decoder first (A2), so a row cannot feed a shape the wire refuses. */
  const deliver = (m) => {
    const d = decode(JSON.stringify(m));
    assert.equal(d.kind, "astra", `premise: the row's message decodes as astra.*: ${JSON.stringify(m).slice(0, 160)} -> ${JSON.stringify(d).slice(0, 200)}`);
    t.handle(d.msg);
  };
  return { doc, root, t, deliver };
}

/** One astra.thought (contracts/protocol.py AstraThought, A1), every field written out. */
function thought(turn_id, attempt_id, { step = 1, kind = "step", text = "", end = "open", omitted_chars = 0, replay = false, request_id = `${attempt_id}-req` } = {}) {
  return { type: "astra.thought", turn_id, request_id, attempt_id, step, kind, text, omitted_chars, end, replay };
}
const spoken = (turn_id, utterance_id, text) => ({ type: "astra.spoken", turn_id, utterance_id, text, replay: false });
const transcript = (turn_id, text) => ({ type: "astra.transcript", turn_id, text, final: true, replay: false });
const sessionCleared = (resets) => ({ type: "astra.session.cleared", resets, ended_turn_id: null, memory_forgotten: 0, replay: false });

const squeeze = (s) => String(s).replace(/\s+/g, " ").trim();
const ARROWS = /[▸▾▶▼►]/g; // ▸ ▾ ▶ ▼ ►
const summaryText = (d) => squeeze(String(d.firstElementChild ? d.firstElementChild.textContent : "").replace(ARROWS, ""));

function articleOf(root, turn) {
  const found = root.querySelectorAll("article").filter((a) => a.getAttribute("data-turn-id") === turn);
  assert.equal(found.length, 1, `exactly one article[data-turn-id="${turn}"]; found ${found.length}`);
  return found[0];
}

/** The turn's one disclosure (C1). */
function disclosureOf(root, turn) {
  const article = articleOf(root, turn);
  const found = article.querySelectorAll("[data-model-thoughts]");
  assert.equal(found.length, 1, `C1: one [data-model-thoughts] in turn ${turn}'s article; found ${found.length}`);
  const d = found[0];
  assert.equal(d.localName, "details", `C1: the disclosure is a native <details>; it is <${d.localName}>`);
  return d;
}

const blocksOf = (d) => d.querySelectorAll("[data-thought-attempt]");
function blockOf(d, attempt) {
  const found = blocksOf(d).filter((b) => b.getAttribute("data-thought-attempt") === attempt);
  assert.equal(found.length, 1, `C3: one [data-thought-attempt="${attempt}"] block; found ${found.length} (blocks: ${JSON.stringify(blocksOf(d).map((b) => b.getAttribute("data-thought-attempt")))})`);
  return found[0];
}
const part = (block, hook) => {
  const el = block.querySelector(`[${hook}]`);
  return el ? el.textContent : "";
};
const labelOf = (b) => squeeze(part(b, "data-thought-label"));
const textOf = (b) => {
  const el = b.querySelector("[data-thought-text]");
  assert.ok(el, "C3: the block's body is a [data-thought-text] element");
  return el.textContent;
};
const statusOf = (b) => squeeze(part(b, "data-thought-status"));
const omittedOf = (b) => squeeze(part(b, "data-thought-omitted"));
const pageOpened = (doc) => doc.openWrites.filter((w) => w.value);

/** A canonical string of a subtree (tag, sorted attributes, text) for before/after comparisons. */
function serialize(node) {
  if (node.nodeType === 3) return JSON.stringify(node.data);
  if (node.nodeType !== 1) return "";
  const attrs = node.getAttributeNames().sort().map((k) => `${k}=${JSON.stringify(node.getAttribute(k))}`).join(" ");
  return `<${node.localName}${attrs ? " " + attrs : ""}>${node.childNodes.map(serialize).join("")}</${node.localName}>`;
}

/** The viewer's own toggle (the real click and keys are the Chrome rows): the attribute, unrecorded. */
function viewerToggle(details, on) {
  if (on) details._attrs.set("open", "");
  else details._attrs.delete("open");
}

// --- the module itself (C6, implementation notes) -----------------------------------------------------

/** The module's code with comments and string-free prose removed (a source guard must not match prose). */
function codeOf(path) {
  const src = readFileSync(fileURLToPath(new URL(path, REPO)), "utf8");
  return src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:\\])\/\/.*$/gm, "$1");
}

pendingT0180C("T-0180 C6: thoughts.js imports in node with no DOM and names its one type, astra.thought", async () => {
  assert.equal(typeof globalThis.document, "undefined", "premise: node has no document");
  const mod = await load("panels/thoughts.js");
  assert.equal(mod.THOUGHT_TYPE, "astra.thought", "C6: hud/src/panels/thoughts.js exports THOUGHT_TYPE = \"astra.thought\"");
});

pendingT0180C("T-0180 C6: thoughts.js reads no canvas or denoise state and draws no random number (code, comments stripped)", async () => {
  const path = "hud/src/panels/thoughts.js";
  assert.ok(existsSync(fileURLToPath(new URL(path, REPO))), `C: ${path} exists (Owned files: new)`);
  const code = codeOf(path);
  const imports = [...code.matchAll(/\bfrom\s+["']([^"']+)["']|\bimport\s*\(\s*["']([^"']+)["']/g)].map((m) => m[1] || m[2]);
  const bad = imports.filter((s) => /denoise|canvas/i.test(s));
  assert.deepEqual(bad, [], `C6/T-0159 B-4: thoughts.js imports nothing of the canvas or the denoising view; it imports ${JSON.stringify(imports)}`);
  assert.ok(!/astra\.canvas/.test(code), "C6: thoughts.js names no astra.canvas.* type (it reads only astra.thought). (Prose can trip this guard: comments are stripped, strings are not.)");
  assert.ok(!/Math\.random/.test(code), "implementation note C: no Math.random in thoughts.js");
});

// --- C1 / C2: one disclosure, collapsed, never opened by the page -------------------------------------

pendingT0180C("T-0180 C1/C2: the first astra.thought makes one closed <details data-model-thoughts> with a summary 'Model thoughts'", async () => {
  const { doc, root, deliver } = await rig();
  deliver(transcript("t1", "What is on the page?"));
  deliver(spoken("t1", "t1-u1", "A search box."));
  deliver(thought("t1", "t1-a1", { text: "The page shows a search box.", end: "open" }));
  const d = disclosureOf(root, "t1");
  assert.equal(d.hasAttribute("open"), false, "C2: collapsed by default (no open attribute)");
  assert.equal(d.firstElementChild && d.firstElementChild.localName, "summary", "C1: the <details>' first element child is its <summary>");
  assert.equal(summaryText(d), "Model thoughts", "C1: one block: the summary reads exactly 'Model thoughts'");
  assert.deepEqual(pageOpened(doc), [], `C2: the page never sets open; it wrote ${JSON.stringify(doc.openWrites)}`);
});

pendingT0180C("T-0180 C2 adversarial: no end, replay or reset makes the page set open, on any element", async () => {
  const { doc, root, deliver } = await rig();
  const ends = ["open", "done", "stopped_repeating", "interrupted", "error"];
  ends.forEach((end, i) => deliver(thought("t1", `t1-a${i + 1}`, { step: i + 1, text: `thought ${i}`, end: "open" })));
  ends.forEach((end, i) => deliver(thought("t1", `t1-a${i + 1}`, { step: i + 1, text: `thought ${i} more`, end })));
  ends.forEach((end, i) => deliver(thought("t1", `t1-a${i + 1}`, { step: i + 1, text: `thought ${i} more`, end, replay: true })));
  const d = disclosureOf(root, "t1");
  assert.equal(blocksOf(d).length, 5, "premise: five blocks rendered");
  assert.equal(d.hasAttribute("open"), false, "C2: still closed after errors, trips, interrupts and a replay");
  deliver(sessionCleared(1));
  deliver(thought("t2", "t2-a1", { text: "after the reset", end: "error" }));
  assert.equal(disclosureOf(root, "t2").hasAttribute("open"), false, "C2: closed after a reset too");
  assert.deepEqual(pageOpened(doc), [], `C2: the page never set open; it wrote ${JSON.stringify(doc.openWrites)}`);
});

pendingT0180C("T-0180 C1: the summary reads 'Model thoughts · N steps' once there is more than one block", async () => {
  const { root, deliver } = await rig();
  deliver(thought("t1", "t1-a1", { step: 1, text: "one", end: "done" }));
  assert.equal(summaryText(disclosureOf(root, "t1")), "Model thoughts");
  deliver(thought("t1", "t1-a2", { step: 2, text: "two", end: "done" }));
  assert.equal(summaryText(disclosureOf(root, "t1")), "Model thoughts · 2 steps", "C1: two blocks");
  deliver(thought("t1", "t1-a3", { step: 3, text: "three", end: "open" }));
  assert.equal(summaryText(disclosureOf(root, "t1")), "Model thoughts · 3 steps", "C1: three blocks");
  // an update of an existing attempt is not a new block
  deliver(thought("t1", "t1-a3", { step: 3, text: "three, finished", end: "done" }));
  assert.equal(summaryText(disclosureOf(root, "t1")), "Model thoughts · 3 steps", "C1/C4: an update is not a block");
});

pendingT0180C("T-0180 C1: the disclosure is never rebuilt: the same <details>, <summary> and blocks across 40 updates", async () => {
  const { doc, root, deliver } = await rig();
  deliver(spoken("t1", "t1-u1", "Answer."));
  deliver(thought("t1", "t1-a1", { text: "a", end: "open" }));
  const d = disclosureOf(root, "t1");
  const summary = d.firstElementChild;
  const first = blockOf(d, "t1-a1");
  let text = "a";
  for (let i = 0; i < 20; i++) {
    text += ` w${i}`;
    deliver(thought("t1", "t1-a1", { text, end: "open" }));
  }
  deliver(thought("t1", "t1-a1", { text, end: "done" }));
  for (let i = 0; i < 19; i++) deliver(thought("t1", "t1-a2", { step: 2, text: `b${i}`, end: "open" }));
  assert.equal(disclosureOf(root, "t1"), d, "C1: the same <details> node");
  assert.equal(d.firstElementChild, summary, "C1: the same <summary> node");
  assert.equal(blockOf(d, "t1-a1"), first, "C1/C4: the first attempt's block keeps its node");
  assert.equal(doc.replaced.filter((n) => n === d || d.contains(n) || n === articleOf(root, "t1")).length, 0, "C1: no replaceChildren on the disclosure or its article");
  assert.equal(textOf(first), text, "premise: the updates were applied");
});

// --- C3: the labels, the body, the status and omitted lines --------------------------------------------

pendingT0180C("T-0180 C3: one labelled block per attempt, in first-seen order, with the six labels written out", async () => {
  const { root, deliver } = await rig();
  const plan = [
    ["t1-a", 1, "step", "Step 1"],
    ["t1-b", 2, "step", "Step 2"],
    ["t1-c", 2, "reask", "Step 2 · re-ask"],
    ["t1-d", 1, "retry", "Step 1 · retry"],
    ["t1-e", 3, "repair", "Step 3 · repair"],
    ["t1-f", 2, "reground", "Step 2 · ReGround"],
  ];
  for (const [id, step, kind] of plan) deliver(thought("t1", id, { step, kind, text: `text of ${id}`, end: "open" }));
  // a later message of the first attempt does not move it
  deliver(thought("t1", "t1-a", { step: 1, kind: "step", text: "text of t1-a, done", end: "done" }));
  const d = disclosureOf(root, "t1");
  assert.deepEqual(blocksOf(d).map((b) => b.getAttribute("data-thought-attempt")), plan.map((p) => p[0]), "C3: first-seen order, keyed by attempt_id");
  for (const [id, , , label] of plan) assert.equal(labelOf(blockOf(d, id)), label, `C3: the label of ${id}`);
  assert.equal(textOf(blockOf(d, "t1-c")), "text of t1-c", "C3: each block its own text");
});

pendingT0180C("T-0180 C3: two attempts of the same step and kind are two blocks (keyed by attempt_id, not by label)", async () => {
  const { root, deliver } = await rig();
  deliver(thought("t1", "t1-x", { step: 2, kind: "reask", text: "first re-ask", end: "done" }));
  deliver(thought("t1", "t1-y", { step: 2, kind: "reask", text: "second re-ask", end: "done" }));
  const d = disclosureOf(root, "t1");
  assert.equal(blocksOf(d).length, 2, "C3: one block per attempt_id");
  assert.equal(textOf(blockOf(d, "t1-x")), "first re-ask");
  assert.equal(textOf(blockOf(d, "t1-y")), "second re-ask");
});

pendingT0180C("T-0180 C3 adversarial: attempts that share one request_id are still one block each (keyed by attempt_id, B-R4)", async () => {
  // The wire's shape (group B, ruling B-R4; seen on a real session's wire): request_id is the turn's
  // request ("t1-gen"), attempt_id the engine's id with the session prefix. A retry of the same request
  // (T-0171 / T-0179, kind "retry") keeps the request_id and gets a new attempt_id.
  const { root, deliver } = await rig();
  deliver(thought("t1", "3512ab-t1-gen-1", { request_id: "t1-gen", step: 1, text: "FIRST-ATTEMPT", end: "stopped_repeating" }));
  deliver(thought("t1", "3512ab-t1-gen-2", { request_id: "t1-gen", step: 1, kind: "retry", text: "SECOND-ATTEMPT", end: "open" }));
  const d = disclosureOf(root, "t1");
  assert.deepEqual(blocksOf(d).map((b) => b.getAttribute("data-thought-attempt")), ["3512ab-t1-gen-1", "3512ab-t1-gen-2"], "C3: one block per attempt_id, not per request_id");
  assert.equal(textOf(blockOf(d, "3512ab-t1-gen-1")), "FIRST-ATTEMPT", "C3: the tripped attempt keeps its own text");
  assert.equal(statusOf(blockOf(d, "3512ab-t1-gen-1")), "stopped: repeating");
  assert.equal(labelOf(blockOf(d, "3512ab-t1-gen-2")), "Step 1 · retry");
  assert.equal(textOf(blockOf(d, "3512ab-t1-gen-2")), "SECOND-ATTEMPT");
  assert.equal(summaryText(d), "Model thoughts · 2 steps", "C1: two blocks");
});

pendingT0180C("T-0180 C3: the status lines: thinking…, stopped: repeating, interrupted, stopped: error, and none when done", async () => {
  const { root, deliver } = await rig();
  const want = [
    ["open", "thinking…"],
    ["stopped_repeating", "stopped: repeating"],
    ["interrupted", "interrupted"],
    ["error", "stopped: error"],
    ["done", ""],
  ];
  want.forEach(([end], i) => deliver(thought("t1", `t1-s${i}`, { step: i + 1, text: `body ${i}`, end })));
  const d = disclosureOf(root, "t1");
  want.forEach(([end, line], i) => {
    const b = blockOf(d, `t1-s${i}`);
    assert.equal(statusOf(b), line, `C3: end ${end} shows ${JSON.stringify(line)}`);
    assert.equal(textOf(b), `body ${i}`, `C3: the body of the ${end} block is its text alone (the status line is not in [data-thought-text])`);
  });
});

pendingT0180C("T-0180 C3: omitted_chars > 0 adds '… N more characters not shown (full text in the session log)'; 0 adds nothing", async () => {
  const { root, deliver } = await rig();
  const capped = "ab".repeat(8000); // 16000 code points: the cap (A6)
  deliver(thought("t1", "t1-a1", { text: capped, omitted_chars: 4000, end: "done" }));
  deliver(thought("t1", "t1-a2", { step: 2, text: "short", end: "done" }));
  const d = disclosureOf(root, "t1");
  const line = omittedOf(blockOf(d, "t1-a1"));
  assert.match(line, /^… 4,?000 more characters not shown \(full text in the session log\)$/, `C3: the omitted line reads ${JSON.stringify(line)}`);
  assert.equal(textOf(blockOf(d, "t1-a1")), capped, "C3: the carried text is shown whole; the line is outside [data-thought-text]");
  assert.equal(omittedOf(blockOf(d, "t1-a2")), "", "C3: no omitted line when omitted_chars is 0");
});

pendingT0180C("T-0180 C3 boundary: omitted_chars 1 (the smallest positive count) still shows the omitted line", async () => {
  // Kills an off-by-one in the "> 0" test (e.g. `<= 1` treated as nothing omitted): one character
  // past the cap is still text the viewer is not shown (H-T180-5).
  const { root, deliver } = await rig();
  deliver(thought("t1", "t1-a1", { text: "ab".repeat(8000), omitted_chars: 1, end: "done" }));
  const line = omittedOf(blockOf(disclosureOf(root, "t1"), "t1-a1"));
  assert.match(line, /^… 1 more characters? not shown \(full text in the session log\)$/, `C3: omitted_chars 1 shows the line; it reads ${JSON.stringify(line)}`);
});

pendingT0180C("T-0180 C3 adversarial: '</script><b>x' and an <img onerror> are text, newlines kept, no element made from them", async () => {
  const { doc, root, deliver } = await rig();
  const evil = "</script><b>x</b>\n<img src=x onerror=alert(1)>\n  two spaces kept\t<|channel>thought";
  const before = doc.created.length;
  deliver(thought("t1", "t1-a1", { text: evil, end: "done" }));
  const d = disclosureOf(root, "t1");
  assert.equal(textOf(blockOf(d, "t1-a1")), evil, "C3: textContent equals the committed text, byte for byte");
  const made = doc.created.slice(before).map((e) => e.localName);
  assert.ok(!made.includes("b") && !made.includes("img") && !made.includes("script"), `C3/privacy: no element from the text; the page made ${JSON.stringify(made)}`);
  assert.equal(d.querySelectorAll("b, img, script, a").length, 0, "C3/privacy: no <b>, <img>, <script> or link inside the disclosure");
  assert.deepEqual(doc.violations, [], `C3: no markup sink touched: ${JSON.stringify(doc.violations)}`);
});

// --- C4: the latest message wins; a final block ignores a later open; replay is idempotent ------------

pendingT0180C("T-0180 C4: the latest message per attempt wins (cumulative text, its end)", async () => {
  const { root, deliver } = await rig();
  deliver(thought("t1", "t1-a1", { text: "I see", end: "open" }));
  deliver(thought("t1", "t1-a1", { text: "I see a box", end: "open" }));
  const d = disclosureOf(root, "t1");
  assert.equal(textOf(blockOf(d, "t1-a1")), "I see a box", "C4: the second open replaces the first (text is cumulative, never appended)");
  assert.equal(statusOf(blockOf(d, "t1-a1")), "thinking…");
  deliver(thought("t1", "t1-a1", { text: "I see a box and a button", end: "interrupted" }));
  assert.equal(textOf(blockOf(d, "t1-a1")), "I see a box and a button");
  assert.equal(statusOf(blockOf(d, "t1-a1")), "interrupted", "C4: the final end replaces thinking…");
  assert.equal(blocksOf(d).length, 1, "C4: still one block");
});

pendingT0180C("T-0180 C4 adversarial: once a block is final, a later open (a stale or replayed message) is ignored", async () => {
  const { root, deliver } = await rig();
  deliver(thought("t1", "t1-a1", { text: "final text", end: "done" }));
  deliver(thought("t1", "t1-a1", { text: "fin", end: "open" }));
  deliver(thought("t1", "t1-a1", { text: "fin", end: "open", replay: true }));
  const d = disclosureOf(root, "t1");
  assert.equal(textOf(blockOf(d, "t1-a1")), "final text", "C4: the final text stays");
  assert.equal(statusOf(blockOf(d, "t1-a1")), "", "C4: no thinking… comes back on a final block");
  deliver(thought("t1", "t1-a2", { step: 2, text: "tripped", end: "stopped_repeating" }));
  deliver(thought("t1", "t1-a2", { step: 2, text: "trip", end: "open" }));
  assert.equal(statusOf(blockOf(d, "t1-a2")), "stopped: repeating", "C4: any final end is final");
  assert.equal(textOf(blockOf(d, "t1-a2")), "tripped");
});

pendingT0180C("T-0180 C4: replay is idempotent — the hello's replay of the same thoughts changes nothing and keeps every node", async () => {
  const { root, deliver } = await rig();
  const live = [
    thought("t1", "t1-a1", { step: 1, text: "Look at the page.", end: "open" }),
    thought("t1", "t1-a1", { step: 1, text: "Look at the page.\nClick search.", end: "done" }),
    thought("t1", "t1-a2", { step: 2, kind: "reask", text: "Retry the click.", end: "interrupted" }),
  ];
  deliver(spoken("t1", "t1-u1", "Searching."));
  for (const m of live) deliver(m);
  const d = disclosureOf(root, "t1");
  const nodes = [d, d.firstElementChild, ...blocksOf(d)];
  const before = serialize(articleOf(root, "t1"));
  // the journal replays the latest per attempt, replay=true (A4); twice, as two reconnects would
  for (let k = 0; k < 2; k++) {
    deliver({ ...live[1], replay: true });
    deliver({ ...live[2], replay: true });
  }
  assert.equal(serialize(articleOf(root, "t1")), before, "C4: the replay leaves the article byte-identical");
  assert.deepEqual([d, d.firstElementChild, ...blocksOf(d)], nodes, "C4: and every node of the disclosure kept");
});

// --- C1: per turn, never mixed --------------------------------------------------------------------------

pendingT0180C("T-0180 C1 adversarial: thoughts go to their own turn's article; a late thought of t1 after t2 opened stays in t1", async () => {
  const { root, deliver } = await rig();
  deliver(thought("t1", "t1-a1", { text: "THOUGHT-OF-T1", end: "open" }));
  deliver(spoken("t2", "t2-u1", "Second answer."));
  deliver(thought("t2", "t2-a1", { text: "THOUGHT-OF-T2", end: "open" }));
  deliver(thought("t1", "t1-a1", { text: "THOUGHT-OF-T1 late", end: "interrupted" }));
  const order = root.querySelectorAll("article").map((a) => a.getAttribute("data-turn-id"));
  assert.deepEqual(order, ["t1", "t2"], "C1: the late thought does not move or duplicate t1's article");
  const d1 = disclosureOf(root, "t1");
  const d2 = disclosureOf(root, "t2");
  assert.equal(textOf(blockOf(d1, "t1-a1")), "THOUGHT-OF-T1 late");
  assert.ok(!d2.textContent.includes("THOUGHT-OF-T1"), "C1: t1's thought is not in t2's disclosure");
  assert.ok(!d1.textContent.includes("THOUGHT-OF-T2"), "C1: t2's thought is not in t1's disclosure");
});

// --- C2: the viewer's open state survives updates ----------------------------------------------------------

pendingT0180C("T-0180 C2: updates keep the viewer's state — opened stays open, closed again stays closed", async () => {
  const { doc, root, deliver } = await rig();
  deliver(thought("t1", "t1-a1", { text: "one", end: "open" }));
  const d = disclosureOf(root, "t1");
  viewerToggle(d, true);
  deliver(thought("t1", "t1-a1", { text: "one two", end: "open" }));
  deliver(thought("t1", "t1-a1", { text: "one two three", end: "done" }));
  deliver(thought("t1", "t1-a2", { step: 2, text: "next", end: "error" }));
  deliver({ ...thought("t1", "t1-a2", { step: 2, text: "next", end: "error" }), replay: true });
  assert.equal(d.hasAttribute("open"), true, "C2: an update never closes what the viewer opened");
  viewerToggle(d, false);
  deliver(thought("t1", "t1-a3", { step: 3, text: "third", end: "open" }));
  deliver(thought("t1", "t1-a3", { step: 3, text: "third done", end: "done" }));
  assert.equal(d.hasAttribute("open"), false, "C2: and never reopens what the viewer closed");
  assert.deepEqual(doc.openWrites, [], `C2: the page wrote no open state at all: ${JSON.stringify(doc.openWrites)}`);
});

// --- C5: reset ---------------------------------------------------------------------------------------------

pendingT0180C("T-0180 C5: astra.session.cleared removes the disclosures with their articles; a new session starts empty", async () => {
  const { doc, root, deliver } = await rig();
  deliver(spoken("t1", "t1-u1", "One."));
  deliver(thought("t1", "t1-a1", { text: "PRE-RESET-T1", end: "done" }));
  deliver(thought("t2", "t2-a1", { text: "PRE-RESET-T2", end: "open" }));
  assert.equal(root.querySelectorAll("[data-model-thoughts]").length, 2, "premise: two disclosures");
  deliver(sessionCleared(1));
  assert.equal(doc.documentElement.querySelectorAll("[data-model-thoughts]").length, 0, "C5: no disclosure survives the reset");
  assert.equal(root.querySelectorAll("article").length, 0, "C5: their articles go with them");
  assert.ok(!doc.documentElement.textContent.includes("PRE-RESET"), "C5: no pre-reset thought text anywhere in the document");
  deliver(thought("t3", "t3-a1", { text: "new session", end: "open" }));
  const d = disclosureOf(root, "t3");
  assert.equal(blocksOf(d).length, 1, "C5: the next session's first thought starts a fresh disclosure");
});

// --- C6: canvas messages never reach the disclosure ---------------------------------------------------------

const DRAFT = "DRAFTSENTINEL";
const CANVAS_COMMIT = "CANVASCOMMITSENTINEL";
function canvasMessages() {
  return [
    {
      type: "astra.canvas.step",
      request_id: "t1-a1-req",
      canvas_idx: 0,
      step: 3,
      max_steps: 8,
      canvas_length: 4,
      canvas: [1, 2, 3, 4],
      argmax: [5, 6, 7, 8],
      stable: "0f",
      phase: "denoise",
      segment: "thought",
      text_preview: `${DRAFT} preview`,
      argmax_pieces: [DRAFT, " a", " b", " c"],
    },
    { type: "astra.canvas.step", request_id: "t1-a1-req", canvas_idx: 0, step: 8, max_steps: 8, canvas_length: 4, phase: "commit", committed_tokens: [5, 6], committed_pieces: [DRAFT, " a"], segment: "thought", text_preview: DRAFT },
    { type: "astra.canvas.commit", request_id: "t1-a1-req", canvas_idx: 0, segment: "thought", text: `${CANVAS_COMMIT} text`, steps_used: 8 },
    { type: "astra.canvas.synthetic", request_id: "t1-a1-req", reason: "test" },
  ];
}

pendingT0180C("T-0180 C6 adversarial: canvas frames and canvas commits (thought segment, pieces, previews) leave the thread untouched", async () => {
  const { doc, root, deliver } = await rig();
  for (const m of canvasMessages()) deliver(m); // before any thought: they create nothing
  assert.equal(doc.documentElement.querySelectorAll("[data-model-thoughts]").length, 0, "C6: a canvas message makes no disclosure");
  deliver(thought("t1", "t1-a1", { text: "COMMITTEDSENTINEL", end: "open" }));
  const d = disclosureOf(root, "t1");
  const before = serialize(root);
  for (const m of canvasMessages()) deliver(m); // between the thoughts
  assert.equal(serialize(root), before, "C6: canvas messages change nothing in the thread");
  deliver(thought("t1", "t1-a1", { text: "COMMITTEDSENTINEL done", end: "done" }));
  assert.equal(textOf(blockOf(d, "t1-a1")), "COMMITTEDSENTINEL done", "premise: the committed thought is shown");
  for (const s of [DRAFT, CANVAS_COMMIT]) assert.ok(!root.textContent.includes(s), `C6/B2: ${s} (a canvas draft or relay commit) is nowhere in the thread`);
});
