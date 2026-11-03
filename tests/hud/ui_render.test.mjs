// T-0164 group A: the renderer core (hud/src/ui/render.js + engine.js), node rows over the census fixture
// with a trapping fake DOM. docs/tasks/T-0164.md A2-A13 (A1/A9/A10's engine-level rows are in
// tests/hud/ui_engine.test.mjs).
//
// Every expected text comes from tests/fixtures/a2ui/genui_census_20261027.jsonl, whose `expected_texts`
// the test-author computed in Python from the gate's own reachability helper
// (local_astra.ui.a2ui_schema.component_children), never from the JS under test (MANIFEST.md).
//
// The contract these rows pin (the task leaves the constructor open; this is the one shape they use, kept
// in `harness()` below so it is one place to read):
//
//   import { createEngine } from "hud/src/ui/engine.js";
//   const engine = createEngine({ document, onAction, log });
//   engine.processor.processMessages([message])      // what bridge.js drives, unchanged (A1)
//   engine.processor.onSurfaceCreated(handler)       // handler({ id, element }): `element` is the card
//                                                    // root, [data-a2ui-root]; the host attaches it
//   engine.dataModelFor(surfaceId) -> object | null  // A1/A9
//   engine.drop(surfaceId)                           // A9
//
//   DOM hooks (T-0164 implementation note 5): [data-a2ui-root] on the card root; data-a2-id and
//   data-a2-kind on every component's outer element; data-a2-empty on a rootless card.
//   Native inputs: TextField <input>/<textarea>, CheckBox <input type=checkbox>, Slider <input type=range>,
//   DateTimeInput <input type=date|time|datetime-local>; buttons and chips <button type="button"> (A12).
//
// Pending convention (node tests cannot carry pytest's `pending`; the T-0162/T-0159 mechanism): every
// row below runs as a strict xfail until docs/tasks/T-0164.md's Status records
// "group A: tests …, activated <sha>". A row that passes while pending fails (XPASS strict).
// ASTRA_RUN_PENDING=1 runs them for real.

import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";
import { join, relative } from "node:path";

const REPO = fileURLToPath(new URL("../../", import.meta.url));
const FIXTURE = join(REPO, "tests/fixtures/a2ui/genui_census_20261027.jsonl");
/** The gate's catalog id (local_astra/ui: the createSurface.catalogId UiGate emits; a2ui.js GATE_CATALOG_ID). */
const GATE_CATALOG_ID = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json";

// node 18 treats a bare .js file as CommonJS unless a package.json in scope says {"type":"module"}, and
// group D deletes hud/package.json: the sources are copied once into a temp directory that carries the
// flag (the shim of tests/hud/ui_bridge.test.mjs and panels.test.mjs).
let ESM_ROOT = null;
function srcRoot() {
  if (ESM_ROOT) return ESM_ROOT;
  const dir = mkdtempSync(join(tmpdir(), "astra-hud-t0164a-"));
  writeFileSync(join(dir, "package.json"), '{"type":"module","private":true}');
  cpSync(join(REPO, "hud/src"), join(dir, "src"), { recursive: true });
  ESM_ROOT = pathToFileURL(join(dir, "src") + "/");
  return ESM_ROOT;
}

// --- the pending gate (T-0164 group A) ------------------------------------------------------------------

const T0164_TASK = join(REPO, "docs/tasks/T-0164.md");
const T0164_STATUS = existsSync(T0164_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0164_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const T0164_A_PENDING =
  !/\bgroup A:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i.test(T0164_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function rowA(name, optsOrFn, maybeFn) {
  const fn = maybeFn || optsOrFn;
  const opts = maybeFn ? optsOrFn : {};
  test(name + (T0164_A_PENDING ? " [pending T-0164 A: strict xfail until group A is activated]" : ""), opts, async (t) => {
    if (!T0164_A_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail("XPASS(strict): this T-0164 group A row passed while docs/tasks/T-0164.md does not record group A's activation.");
    }
    t.diagnostic("expected failure (T-0164 A pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

async function load(name) {
  const path = join(REPO, "hud/src/ui", name);
  if (!existsSync(path)) assert.fail(`hud/src/ui/${name} does not exist yet (T-0164 A owned file; the implementer adds it)`);
  try {
    return await import(new URL(`ui/${name}`, srcRoot()).href);
  } catch (err) {
    assert.fail(`hud/src/ui/${name} must import in node with no document (A1, T-0024 criterion 2): ${err && err.stack}`);
  }
}

// <fake-dom> ---------------------------------------------------------------------------------------------
// tests/hud/ui_engine.test.mjs imports exactly this region (between the two markers) as a data: module, so
// both modules use one trapping double. Keep it free of imports.
//
// A DOM double that (1) counts every mutation (A7: creates, inserts, removes, text/attribute/value/style
// writes, counted even when the written value is unchanged, as a MutationObserver would), (2) throws a
// named ForbiddenSink on every markup sink and fetch path A11 names and records it (so a renderer that
// swallows the throw is still caught), and (3) closes by delegation: reading a name it does not model is
// recorded in `rec.gaps` (and reported by its own row) instead of silently reading `undefined` from a real
// DOM that is not there.
export const FAKE_DOM_SOURCE_MARKER = "fake-dom";
export function makeDom() {
  const HTML_NS = "http://www.w3.org/1999/xhtml";
  const SVG_NS = "http://www.w3.org/2000/svg";
  const FORBIDDEN_TAGS = new Set(["img", "picture", "source", "video", "audio", "iframe", "object", "embed", "link", "script", "style", "a", "use", "image", "foreignobject"]);
  const FORBIDDEN_ATTRS = new Set(["src", "srcset", "href", "xlink:href", "action", "formaction", "poster", "srcdoc", "data", "background"]);
  const SINK_PROPS = new Set(["innerHTML", "outerHTML", "insertAdjacentHTML", "setHTMLUnsafe", "attachShadow", "src", "srcset", "href", "action", "formAction", "poster", "srcdoc", "shadowRoot", "createContextualFragment"]);
  const rec = { mutations: [], creates: [], violations: [], gaps: new Set(), counting: true };
  const mutate = (op, node, detail) => {
    if (rec.counting) rec.mutations.push({ op, node, detail });
  };
  const forbid = (what) => {
    rec.violations.push(what);
    const err = new Error(`ForbiddenSink: the renderer reached ${what} (T-0164 A11: no markup sink, no fetch path)`);
    err.name = "ForbiddenSink";
    throw err;
  };
  let doc = null;

  const wrap = (target) =>
    new Proxy(target, {
      get(t, prop, receiver) {
        if (typeof prop === "symbol" || prop in t) return Reflect.get(t, prop, receiver);
        if (SINK_PROPS.has(prop)) forbid(`${t.nodeName}.${prop}`);
        if (prop !== "then" && prop !== "toJSON") rec.gaps.add(`${t.nodeName || "node"}.${prop}`);
        return undefined;
      },
      set(t, prop, value, receiver) {
        if (typeof prop === "string") {
          if (SINK_PROPS.has(prop)) forbid(`${t.nodeName}.${prop} = …`);
          if (/^on[a-z]+$/.test(prop)) rec.gaps.add(`${t.nodeName}.${prop} (an on* property handler; the fake models addEventListener)`);
        }
        return Reflect.set(t, prop, value, receiver);
      },
    });

  class FakeEvent {
    constructor(type, init = {}) {
      this.type = type;
      this.bubbles = init.bubbles !== false;
      this.key = init.key;
      this.shiftKey = Boolean(init.shiftKey);
      this.target = null;
      this.currentTarget = null;
      this.defaultPrevented = false;
      this._stop = false;
      this.isTrusted = true;
    }
    preventDefault() {
      this.defaultPrevented = true;
    }
    stopPropagation() {
      this._stop = true;
    }
    stopImmediatePropagation() {
      this._stop = true;
    }
    composedPath() {
      const path = [];
      for (let n = this.target; n; n = n.parentNode) path.push(n);
      return path;
    }
  }

  class Base {
    constructor() {
      this._parent = null;
      this._children = [];
      this._listeners = new Map();
    }
    get parentNode() {
      return this._parent;
    }
    get parentElement() {
      return this._parent && this._parent.nodeType === 1 ? this._parent : null;
    }
    get ownerDocument() {
      return doc;
    }
    get childNodes() {
      return [...this._children];
    }
    get children() {
      return this._children.filter((c) => c.nodeType === 1);
    }
    get childElementCount() {
      return this.children.length;
    }
    get firstChild() {
      return this._children[0] || null;
    }
    get lastChild() {
      return this._children[this._children.length - 1] || null;
    }
    get firstElementChild() {
      return this.children[0] || null;
    }
    get lastElementChild() {
      const c = this.children;
      return c[c.length - 1] || null;
    }
    _sib(d, el) {
      if (!this._parent) return null;
      const list = el ? this._parent.children : this._parent._children;
      const i = list.indexOf(this);
      return list[i + d] || null;
    }
    get nextSibling() {
      return this._sib(1, false);
    }
    get previousSibling() {
      return this._sib(-1, false);
    }
    get nextElementSibling() {
      return this._sib(1, true);
    }
    get previousElementSibling() {
      return this._sib(-1, true);
    }
    get isConnected() {
      let n = this;
      while (n._parent) n = n._parent;
      return n === doc;
    }
    hasChildNodes() {
      return this._children.length > 0;
    }
    contains(other) {
      for (let n = other; n; n = n._parent) if (n === this) return true;
      return false;
    }
    getRootNode() {
      let n = this;
      while (n._parent) n = n._parent;
      return n;
    }
    _detach(count = true) {
      const p = this._parent;
      if (!p) return;
      p._children.splice(p._children.indexOf(this), 1);
      this._parent = null;
      if (count) mutate("remove", this, p);
    }
    _insert(node, ref) {
      if (node.nodeType === 11) {
        for (const c of [...node._children]) this._insert(c, ref);
        return node;
      }
      if (node.contains(this)) throw new Error("HierarchyRequestError: a node cannot contain its ancestor");
      node._detach(false);
      const i = ref ? this._children.indexOf(ref) : -1;
      if (ref && i < 0) throw new Error("NotFoundError: the reference node is not a child");
      if (i < 0) this._children.push(node);
      else this._children.splice(i, 0, node);
      node._parent = this;
      mutate("insert", node, this);
      return node;
    }
    _nodes(items) {
      return items.map((x) => (typeof x === "string" ? doc.createTextNode(x) : x));
    }
    appendChild(node) {
      return this._insert(node, null);
    }
    insertBefore(node, ref) {
      return this._insert(node, ref || null);
    }
    removeChild(node) {
      if (node._parent !== this) throw new Error("NotFoundError: not a child");
      node._detach();
      return node;
    }
    replaceChild(node, old) {
      this._insert(node, old);
      old._detach();
      return old;
    }
    append(...items) {
      for (const n of this._nodes(items)) this._insert(n, null);
    }
    prepend(...items) {
      const first = this.firstChild;
      for (const n of this._nodes(items)) this._insert(n, first);
    }
    replaceChildren(...items) {
      for (const c of [...this._children]) c._detach();
      this.append(...items);
    }
    remove() {
      this._detach();
    }
    before(...items) {
      if (this._parent) for (const n of this._nodes(items)) this._parent._insert(n, this);
    }
    after(...items) {
      if (!this._parent) return;
      const next = this.nextSibling;
      for (const n of this._nodes(items)) this._parent._insert(n, next);
    }
    replaceWith(...items) {
      if (!this._parent) return;
      const p = this._parent;
      const next = this.nextSibling;
      this._detach();
      for (const n of this._nodes(items)) p._insert(n, next);
    }
    addEventListener(type, fn, opts) {
      if (!fn) return;
      if (!this._listeners.has(type)) this._listeners.set(type, []);
      const list = this._listeners.get(type);
      if (list.some((l) => l.fn === fn)) return;
      list.push({ fn, once: Boolean(opts && typeof opts === "object" && opts.once) });
    }
    removeEventListener(type, fn) {
      const list = this._listeners.get(type);
      if (list) this._listeners.set(type, list.filter((l) => l.fn !== fn));
    }
    dispatchEvent(event) {
      event.target = this;
      const path = event.composedPath();
      for (const node of event.bubbles ? path : [this]) {
        event.currentTarget = node;
        for (const l of [...(node._listeners.get(event.type) || [])]) {
          if (l.once) node.removeEventListener(event.type, l.fn);
          if (typeof l.fn === "function") l.fn.call(node, event);
          else if (l.fn && typeof l.fn.handleEvent === "function") l.fn.handleEvent(event);
        }
        if (event._stop) break;
      }
      event.currentTarget = null;
      return !event.defaultPrevented;
    }
  }

  class Text extends Base {
    constructor(data) {
      super();
      this._data = String(data);
    }
    get nodeType() {
      return 3;
    }
    get nodeName() {
      return "#text";
    }
    get data() {
      return this._data;
    }
    set data(v) {
      this._data = String(v);
      mutate("text", this, v);
    }
    get nodeValue() {
      return this._data;
    }
    set nodeValue(v) {
      this.data = v;
    }
    get textContent() {
      return this._data;
    }
    set textContent(v) {
      this.data = v == null ? "" : v;
    }
    get wholeText() {
      return this._data;
    }
  }

  class Fragment extends Base {
    get nodeType() {
      return 11;
    }
    get nodeName() {
      return "#document-fragment";
    }
    get textContent() {
      return this._children.map((c) => c.textContent).join("");
    }
    querySelector(sel) {
      return queryAll(this, sel)[0] || null;
    }
    querySelectorAll(sel) {
      return queryAll(this, sel);
    }
  }

  const camelToKebab = (s) => s.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase());
  const REFLECT = { id: "id", className: "class", title: "title", lang: "lang", dir: "dir", role: "role", name: "name", placeholder: "placeholder", min: "min", max: "max", step: "step", htmlFor: "for", autocomplete: "autocomplete", inputMode: "inputmode", pattern: "pattern", ariaLabel: "aria-label", ariaChecked: "aria-checked", ariaPressed: "aria-pressed", ariaSelected: "aria-selected", ariaExpanded: "aria-expanded", ariaControls: "aria-controls", ariaLabelledByText: "aria-labelledby", ariaHidden: "aria-hidden", ariaOrientation: "aria-orientation" };
  const BOOL = { hidden: "hidden", inert: "inert", disabled: "disabled", required: "required", readOnly: "readonly", multiple: "multiple" };

  class Element extends Base {
    constructor(tag, ns) {
      super();
      this._ns = ns;
      this._local = ns === HTML_NS ? tag.toLowerCase() : tag;
      this._attrs = new Map();
      this._value = "";
      this._valueDirty = false;
      this._checked = false;
      this._sel = [0, 0];
      const el = this;
      const style = { _props: new Map() };
      this.style = new Proxy(style, {
        get(t, p) {
          if (p === "setProperty")
            return (k, v) => {
              if (String(v).includes("url(")) forbid(`style ${k}: url(…)`);
              t._props.set(k, String(v));
              mutate("style", el, k);
            };
          if (p === "removeProperty")
            return (k) => {
              t._props.delete(k);
              mutate("style", el, k);
            };
          if (p === "getPropertyValue") return (k) => t._props.get(k) || "";
          if (p === "cssText") return [...t._props].map(([k, v]) => `${k}: ${v};`).join(" ");
          if (typeof p !== "string") return undefined;
          return t._props.get(camelToKebab(p)) || "";
        },
        set(t, p, v) {
          if (String(v).includes("url(")) forbid(`style ${String(p)}: url(…)`);
          if (p === "cssText") {
            t._props.clear();
            for (const part of String(v).split(";")) {
              const [k, ...rest] = part.split(":");
              if (k && k.trim()) t._props.set(k.trim(), rest.join(":").trim());
            }
          } else t._props.set(camelToKebab(String(p)), String(v));
          mutate("style", el, p);
          return true;
        },
      });
      this.dataset = new Proxy(
        {},
        {
          get: (_, p) => (typeof p === "string" ? (el.getAttribute("data-" + camelToKebab(p)) ?? undefined) : undefined),
          set: (_, p, v) => {
            el.setAttribute("data-" + camelToKebab(String(p)), String(v));
            return true;
          },
          deleteProperty: (_, p) => {
            el.removeAttribute("data-" + camelToKebab(String(p)));
            return true;
          },
          has: (_, p) => el.hasAttribute("data-" + camelToKebab(String(p))),
        },
      );
    }
    get nodeType() {
      return 1;
    }
    get localName() {
      return this._local;
    }
    get tagName() {
      return this._ns === HTML_NS ? this._local.toUpperCase() : this._local;
    }
    get nodeName() {
      return this.tagName;
    }
    get namespaceURI() {
      return this._ns;
    }
    getAttribute(name) {
      const v = this._attrs.get(String(name).toLowerCase());
      return v === undefined ? null : v;
    }
    hasAttribute(name) {
      return this._attrs.has(String(name).toLowerCase());
    }
    getAttributeNames() {
      return [...this._attrs.keys()];
    }
    setAttribute(name, value) {
      const n = String(name).toLowerCase();
      if (FORBIDDEN_ATTRS.has(n)) forbid(`setAttribute("${n}")`);
      if (/^on/.test(n)) forbid(`setAttribute("${n}") (an on* attribute)`);
      if (n === "style" && String(value).includes("url(")) forbid('setAttribute("style") with url(…)');
      this._attrs.set(n, String(value));
      mutate("attr", this, n);
    }
    setAttributeNS(_ns, name, value) {
      this.setAttribute(String(name).replace(/^.*:/, (m) => (m === "xlink:" ? m : "")), value);
    }
    removeAttribute(name) {
      const n = String(name).toLowerCase();
      this._attrs.delete(n);
      mutate("attr", this, n);
    }
    toggleAttribute(name, force) {
      const has = this.hasAttribute(name);
      const want = force === undefined ? !has : Boolean(force);
      if (want) this.setAttribute(name, "");
      else this.removeAttribute(name);
      return want;
    }
    get attributes() {
      return [...this._attrs].map(([name, value]) => ({ name, value }));
    }
    get classList() {
      const el = this;
      const list = () => (el.getAttribute("class") || "").split(/\s+/).filter(Boolean);
      const write = (xs) => el.setAttribute("class", xs.join(" "));
      return {
        add: (...c) => write([...new Set([...list(), ...c])]),
        remove: (...c) => write(list().filter((x) => !c.includes(x))),
        toggle: (c, force) => {
          const has = list().includes(c);
          const want = force === undefined ? !has : Boolean(force);
          write(want ? [...new Set([...list(), c])] : list().filter((x) => x !== c));
          return want;
        },
        replace: (a, b) => write(list().map((x) => (x === a ? b : x))),
        contains: (c) => list().includes(c),
        get length() {
          return list().length;
        },
        get value() {
          return list().join(" ");
        },
        [Symbol.iterator]: () => list()[Symbol.iterator](),
      };
    }
    get tabIndex() {
      const v = this.getAttribute("tabindex");
      return v === null ? (["button", "input", "textarea", "select"].includes(this._local) ? 0 : -1) : Number(v);
    }
    set tabIndex(v) {
      this.setAttribute("tabindex", String(v));
    }
    get type() {
      if (this._local === "button") return this.getAttribute("type") || "submit";
      return this.getAttribute("type") || "text";
    }
    set type(v) {
      this.setAttribute("type", v);
    }
    get value() {
      if (this._local === "input" || this._local === "textarea" || this._local === "select") {
        if (!this._valueDirty) return this.getAttribute("value") ?? (this.type === "range" ? "50" : this.type === "checkbox" ? "on" : "");
        return this._value;
      }
      return this.getAttribute("value") ?? "";
    }
    set value(v) {
      this._value = v == null ? "" : String(v);
      this._valueDirty = true;
      this._sel = [this._value.length, this._value.length];
      mutate("value", this, v);
    }
    get valueAsNumber() {
      return Number(this.value);
    }
    set valueAsNumber(v) {
      this.value = String(v);
    }
    get defaultValue() {
      return this.getAttribute("value") ?? "";
    }
    set defaultValue(v) {
      this.setAttribute("value", v);
    }
    get checked() {
      return this._checked;
    }
    set checked(v) {
      this._checked = Boolean(v);
      mutate("value", this, "checked");
    }
    get selectionStart() {
      return this._sel[0];
    }
    get selectionEnd() {
      return this._sel[1];
    }
    setSelectionRange(a, b) {
      this._sel = [a, b];
    }
    select() {
      this._sel = [0, this.value.length];
    }
    get textContent() {
      return this._children.map((c) => c.textContent).join("");
    }
    set textContent(v) {
      for (const c of [...this._children]) c._detach(false);
      mutate("text", this, v);
      const s = v == null ? "" : String(v);
      if (s) {
        const t = new Text(s);
        this._children.push(t);
        t._parent = this;
      }
    }
    get innerText() {
      return this.textContent;
    }
    set innerText(v) {
      this.textContent = v;
    }
    insertAdjacentElement(pos, el) {
      const map = { beforebegin: () => this.before(el), afterbegin: () => this.prepend(el), beforeend: () => this.append(el), afterend: () => this.after(el) };
      map[String(pos).toLowerCase()]();
      return el;
    }
    insertAdjacentText(pos, text) {
      this.insertAdjacentElement(pos, doc.createTextNode(text));
    }
    matches(sel) {
      return matchesList(this, sel);
    }
    closest(sel) {
      for (let n = this; n && n.nodeType === 1; n = n._parent) if (matchesList(n, sel)) return n;
      return null;
    }
    querySelector(sel) {
      return queryAll(this, sel)[0] || null;
    }
    querySelectorAll(sel) {
      return queryAll(this, sel);
    }
    getElementsByTagName(tag) {
      return queryAll(this, tag);
    }
    focus() {
      const prev = doc._active;
      if (prev === this) return;
      doc._active = this;
      if (prev && prev !== doc.body) {
        prev.dispatchEvent(new FakeEvent("blur", { bubbles: false }));
        prev.dispatchEvent(new FakeEvent("focusout"));
      }
      this.dispatchEvent(new FakeEvent("focus", { bubbles: false }));
      this.dispatchEvent(new FakeEvent("focusin"));
    }
    blur() {
      if (doc._active !== this) return;
      doc._active = doc.body;
      this.dispatchEvent(new FakeEvent("blur", { bubbles: false }));
      this.dispatchEvent(new FakeEvent("focusout"));
    }
    click() {
      if (this.hasAttribute("disabled")) return;
      const toggles = this._local === "input" && (this.type === "checkbox" || this.type === "radio");
      const before = this._checked;
      if (toggles) this._checked = this.type === "radio" ? true : !before;
      const ok = this.dispatchEvent(new FakeEvent("click"));
      if (toggles && !ok) this._checked = before;
      if (toggles && ok && before !== this._checked) {
        this.dispatchEvent(new FakeEvent("input"));
        this.dispatchEvent(new FakeEvent("change"));
      }
    }
    scrollIntoView() {}
  }
  for (const [prop, attr] of Object.entries(REFLECT)) {
    Object.defineProperty(Element.prototype, prop, {
      get() {
        return this.getAttribute(attr) ?? "";
      },
      set(v) {
        this.setAttribute(attr, v);
      },
    });
  }
  for (const [prop, attr] of Object.entries(BOOL)) {
    Object.defineProperty(Element.prototype, prop, {
      get() {
        return this.hasAttribute(attr);
      },
      set(v) {
        if (v) this.setAttribute(attr, "");
        else this.removeAttribute(attr);
      },
    });
  }

  // --- selectors: tag, #id, .class, [attr], [attr="v"], compound, descendant and child combinators, lists
  function parseCompound(s) {
    const out = { tag: null, id: null, classes: [], attrs: [] };
    const re = /([a-zA-Z][\w-]*)|#([\w-]+)|\.([\w-]+)|\[([\w:-]+)(?:([~^$*|]?=)\s*(?:"([^"]*)"|'([^']*)'|([^\]]*)))?\]|(\*)/g;
    let m;
    let consumed = 0;
    while ((m = re.exec(s))) {
      consumed += m[0].length;
      if (m[1]) out.tag = m[1].toLowerCase();
      else if (m[2]) out.id = m[2];
      else if (m[3]) out.classes.push(m[3]);
      else if (m[4]) out.attrs.push({ name: m[4].toLowerCase(), op: m[5] || null, value: m[6] ?? m[7] ?? m[8] ?? null });
    }
    if (consumed !== s.length) {
      rec.gaps.add(`selector "${s}" (the fake models tag, #id, .class, [attr], [attr=v], descendant/child combinators)`);
    }
    return out;
  }
  function matchCompound(el, c) {
    if (!el || el.nodeType !== 1) return false;
    if (c.tag && el._local.toLowerCase() !== c.tag) return false;
    if (c.id && el.getAttribute("id") !== c.id) return false;
    const cls = (el.getAttribute("class") || "").split(/\s+/);
    if (c.classes.some((k) => !cls.includes(k))) return false;
    for (const a of c.attrs) {
      const v = el.getAttribute(a.name);
      if (v === null) return false;
      if (a.op === "=" && v !== a.value) return false;
      if (a.op === "~=" && !v.split(/\s+/).includes(a.value)) return false;
      if (a.op === "^=" && !v.startsWith(a.value)) return false;
      if (a.op === "*=" && !v.includes(a.value)) return false;
    }
    return true;
  }
  function matchesOne(el, sel) {
    const tokens = sel.trim().replace(/\s*>\s*/g, " > ").split(/\s+/);
    const parts = [];
    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i] === ">") continue;
      parts.push({ c: parseCompound(tokens[i]), child: tokens[i - 1] === ">" });
    }
    const go = (node, k) => {
      if (!matchCompound(node, parts[k].c)) return false;
      if (k === 0) return true;
      if (parts[k].child) return go(node._parent, k - 1);
      for (let a = node._parent; a && a.nodeType === 1; a = a._parent) if (go(a, k - 1)) return true;
      return false;
    };
    return go(el, parts.length - 1);
  }
  function matchesList(el, sel) {
    return String(sel)
      .split(",")
      .some((s) => s.trim() && matchesOne(el, s));
  }
  function queryAll(root, sel) {
    const out = [];
    const walk = (n) => {
      for (const c of n._children) {
        if (c.nodeType === 1) {
          if (matchesList(c, sel)) out.push(c);
          walk(c);
        }
      }
    };
    walk(root);
    return out;
  }

  class Document extends Base {
    constructor() {
      super();
      this._active = null;
      this.defaultView = null;
    }
    get nodeType() {
      return 9;
    }
    get nodeName() {
      return "#document";
    }
    get documentElement() {
      return this._children[0] || null;
    }
    get head() {
      return this.documentElement && this.documentElement.children[0];
    }
    get body() {
      return this.documentElement && this.documentElement.children[1];
    }
    get activeElement() {
      return this._active && this._active.isConnected ? this._active : this.body;
    }
    createElement(tag) {
      if (FORBIDDEN_TAGS.has(String(tag).toLowerCase())) forbid(`createElement("${tag}")`);
      const el = wrap(new Element(String(tag), HTML_NS));
      rec.creates.push(el);
      mutate("create", el, tag);
      return el;
    }
    createElementNS(ns, tag) {
      const local = String(tag).replace(/^.*:/, "");
      if (FORBIDDEN_TAGS.has(local.toLowerCase())) forbid(`createElementNS(…, "${tag}")`);
      const el = wrap(new Element(local, ns === HTML_NS ? HTML_NS : ns));
      rec.creates.push(el);
      mutate("create", el, tag);
      return el;
    }
    createTextNode(data) {
      const t = wrap(new Text(data));
      rec.creates.push(t);
      mutate("create", t, "#text");
      return t;
    }
    createDocumentFragment() {
      const f = wrap(new Fragment());
      mutate("create", f, "#fragment");
      return f;
    }
    getElementById(id) {
      return queryAll(this, "*").find((e) => e.getAttribute("id") === id) || null;
    }
    querySelector(sel) {
      return queryAll(this, sel)[0] || null;
    }
    querySelectorAll(sel) {
      return queryAll(this, sel);
    }
    write() {
      forbid("document.write");
    }
    writeln() {
      forbid("document.writeln");
    }
  }

  doc = wrap(new Document());
  rec.counting = false;
  const html = doc.createElement("html");
  doc.appendChild(html);
  html.appendChild(doc.createElement("head"));
  html.appendChild(doc.createElement("body"));
  rec.creates.length = 0;
  rec.counting = true;
  const quiet = (fn) => {
    const was = rec.counting;
    rec.counting = false;
    try {
      return fn();
    } finally {
      rec.counting = was;
    }
  };
  return { document: doc, rec, FakeEvent, quiet, HTML_NS, SVG_NS };
}
// </fake-dom> --------------------------------------------------------------------------------------------

// --- fixture --------------------------------------------------------------------------------------------

const FIXTURE_LINES = readFileSync(FIXTURE, "utf8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l));
const HEADER = FIXTURE_LINES[0].header;
const SURFACES = FIXTURE_LINES.slice(1);
const byKey = (key) => {
  const s = SURFACES.find((x) => x.key === key);
  assert.ok(s, `fixture surface ${key} is missing from genui_census_20261027.jsonl`);
  return s;
};

// --- harness: the real bridge in front of the engine, as hud/src/ui/index.js will mount it (group B) ----

async function harness({ withBridge = true } = {}) {
  const { createEngine } = await load("engine.js");
  const { createSurfaceBridge } = await load("bridge.js");
  const dom = makeDom();
  const actions = [];
  const logs = [];
  const roots = new Map();
  const created = [];
  const engine = createEngine({ document: dom.document, onAction: (a) => actions.push(a), log: (line) => logs.push(String(line)) });
  assert.ok(engine && engine.processor && typeof engine.processor.processMessages === "function", "A1: createEngine(...).processor.processMessages");
  engine.processor.onSurfaceCreated((surface) => {
    created.push(surface);
    const element = surface && surface.element;
    assert.ok(element && element.nodeType === 1, `onSurfaceCreated hands { id, element }: element is the card root; got ${surface && Object.keys(surface)}`);
    roots.set(surface.id, element);
    dom.quiet(() => {
      const slot = dom.document.createElement("div");
      slot.setAttribute("data-ui-surface", surface.id);
      dom.document.body.appendChild(slot);
      slot.appendChild(element);
    });
  });
  const bridge = withBridge ? createSurfaceBridge({ processor: engine.processor, log: (l) => logs.push("[bridge] " + l) }) : null;
  const idx = new Map();
  const deliver = (surfaceId, messages) => {
    const next = idx.has(surfaceId) ? idx.get(surfaceId) + 1 : 0;
    idx.set(surfaceId, next);
    const ok = bridge.handle({ type: "astra.ui.surface", surface_id: surfaceId, canvas_idx: next, messages, final: false });
    assert.equal(ok, true, `the bridge applied canvas ${next} of ${surfaceId}`);
    assert.equal(bridge.stats().errors, 0, `the engine rejected a message: ${logs.filter((l) => l.startsWith("[bridge]")).join(" | ")}`);
  };
  const deliverFixture = (s, upTo = s.canvases.length) => {
    for (const cv of s.canvases.slice(0, upTo)) deliver(s.surface_id, cv);
    return root(s.surface_id);
  };
  const root = (id) => {
    const r = roots.get(id);
    assert.ok(r, `no card root was handed out for surface ${id}`);
    return r;
  };
  return { dom, engine, bridge, actions, logs, created, roots, deliver, deliverFixture, root };
}

const CREATE = (id, extra = {}) => ({ version: "v0.9.1", createSurface: { surfaceId: id, catalogId: GATE_CATALOG_ID, ...extra } });
const COMPONENTS = (id, components) => ({ version: "v0.9.1", updateComponents: { surfaceId: id, components } });
const DATA = (id, path, value) => ({ version: "v0.9.1", updateDataModel: value === undefined ? { surfaceId: id, path } : { surfaceId: id, path, value } });
const surface = (id, components, { send = false, model } = {}) => [
  CREATE(id, send ? { sendDataModel: true } : {}),
  COMPONENTS(id, components),
  ...(model === undefined ? [] : [DATA(id, "/", model)]),
];

// --- DOM reading helpers (independent of how the renderer builds things) ---------------------------------

function* walk(node) {
  for (const c of node.childNodes) {
    yield c;
    if (c.nodeType === 1) yield* walk(c);
  }
}
const elements = (root) => [...walk(root)].filter((n) => n.nodeType === 1);
const comp = (root, cid) => elements(root).find((e) => e.getAttribute("data-a2-id") === cid) || null;
const comps = (root, cid) => elements(root).filter((e) => e.getAttribute("data-a2-id") === cid);
const INPUT_TAGS = new Set(["input", "textarea"]);
const inputsOf = (root) => elements(root).filter((e) => INPUT_TAGS.has(e.localName));
const isHidden = (el, root) => {
  for (let n = el; n && n !== root.parentNode; n = n.parentNode) {
    if (n.nodeType !== 1) continue;
    if (n.hasAttribute("hidden") || n.style.display === "none") return true;
  }
  return false;
};

/** Document-order chunks a person can read: text nodes, and each input's value and placeholder. */
function readableChunks(root) {
  const out = [];
  for (const n of walk(root)) {
    if (n.nodeType === 3 && n.data.trim()) out.push(n.data);
    if (n.nodeType === 1 && INPUT_TAGS.has(n.localName) && !["checkbox", "range", "radio"].includes(n.type)) {
      if (n.value) out.push(n.value);
      const ph = n.getAttribute("placeholder");
      if (ph) out.push(ph);
    }
  }
  return out;
}
const isSubsequence = (needle, hay) => {
  let i = 0;
  for (const h of hay) if (i < needle.length && h === needle[i]) i++;
  return i === needle.length;
};
const charBag = (strings) => {
  const bag = new Map();
  for (const ch of [...strings.join("")].filter((c) => !/\s/.test(c))) bag.set(ch, (bag.get(ch) || 0) + 1);
  return [...bag].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
};

/** The accessible name, computed the way the page's markup can supply it (aria-labelledby, aria-label, label). */
function accessibleName(el, document) {
  const by = el.getAttribute("aria-labelledby");
  if (by) return by.split(/\s+/).map((id) => (document.getElementById(id) || { textContent: "" }).textContent).join(" ").trim();
  const aria = el.getAttribute("aria-label");
  if (aria) return aria.trim();
  const id = el.getAttribute("id");
  if (id) {
    const lab = elements(document.body).find((e) => e.localName === "label" && e.getAttribute("for") === id);
    if (lab) return lab.textContent.trim();
  }
  for (let n = el.parentNode; n && n.nodeType === 1; n = n.parentNode) if (n.localName === "label") return n.textContent.trim();
  if (el.localName === "button" || ["tab", "radio", "checkbox", "option"].includes(el.getAttribute("role"))) return el.textContent.trim();
  return "";
}

function typeInto(dom, input, text, caret = text.length) {
  dom.quiet(() => {
    input.focus();
    input.value = text;
    input.setSelectionRange(caret, caret);
    input.dispatchEvent(new dom.FakeEvent("input"));
    input.dispatchEvent(new dom.FakeEvent("change"));
  });
}
const userClick = (dom, el) => dom.quiet(() => el.click());
const chipsOf = (el) => elements(el).filter((e) => e.localName === "button");
const chipState = (chip) => chip.getAttribute("aria-checked") === "true" || chip.getAttribute("aria-pressed") === "true";

/** How many engine log lines a benign surface costs on its first canvas and on two identical re-sends. */
async function benignLines() {
  const h = await harness();
  const comps0 = [{ id: "root", component: "Column", children: ["t"] }, { id: "t", component: "Text", text: "benign" }];
  const own = () => h.logs.filter((l) => !l.startsWith("[bridge]")).length;
  const start = own();
  h.deliver("benign", surface("benign", comps0));
  const first = own() - start;
  h.deliver("benign", [COMPONENTS("benign", comps0)]);
  h.deliver("benign", [COMPONENTS("benign", comps0)]);
  return { first, resends: own() - start - first };
}

const mutationSummary = (rec) =>
  rec.mutations
    .slice(0, 8)
    .map((m) => `${m.op}:${m.node && m.node.nodeName}${m.detail && typeof m.detail === "string" ? "(" + m.detail + ")" : ""}`)
    .join(", ");

// =========================================================================================================
// A1 (module half): both modules import in node with no document
// =========================================================================================================

rowA("A1: render.js and engine.js import in node with no document, and touch no DOM at module top level", async () => {
  assert.equal(typeof globalThis.document, "undefined", "precondition: node has no document");
  await load("render.js");
  await load("engine.js");
  // A fresh evaluation of both modules (cache-busting query) with trapped DOM globals: reading any of them
  // at module top level is recorded.
  const touched = [];
  const names = ["document", "window", "customElements", "HTMLElement", "Node", "MutationObserver", "navigator"];
  for (const n of names) Object.defineProperty(globalThis, n, { configurable: true, get: () => (touched.push(n), undefined) });
  let render;
  let engine;
  try {
    render = await import(new URL("ui/render.js?nodom", srcRoot()).href);
    engine = await import(new URL("ui/engine.js?nodom", srcRoot()).href);
  } finally {
    for (const n of names) delete globalThis[n];
  }
  assert.deepEqual(touched, [], "no DOM global is read at module top level");
  assert.ok(Object.keys(render).length > 0, "render.js exports its walker");
  assert.equal(typeof engine.createEngine, "function", "engine.js exports createEngine");
  // and the document arrives as an argument, later
  const e = engine.createEngine({ document: makeDom().document, onAction: () => {}, log: () => {} });
  assert.equal(typeof e.processor.processMessages, "function");
  assert.equal(typeof e.processor.onSurfaceCreated, "function");
  assert.equal(typeof e.dataModelFor, "function");
  assert.equal(typeof e.drop, "function");
});

// =========================================================================================================
// A2: every advertised kind renders from catalog props
// =========================================================================================================

// Not pending: a fixture-integrity guard (the committed copy against the Python source), true today.
test("A2: the fixture header's kind list is ui_instructions.ADVERTISED_ORDER (the 15 kinds the prompt promises)", async () => {
  const src = readFileSync(join(REPO, "local_astra/conductor/ui_instructions.py"), "utf8");
  const block = /ADVERTISED_ORDER:\s*tuple\[str, \.\.\.\]\s*=\s*\(([^)]*)\)/.exec(src);
  assert.ok(block, "ADVERTISED_ORDER is a literal tuple in local_astra/conductor/ui_instructions.py");
  const kinds = [...block[1].replace(/#.*$/gm, "").matchAll(/"([A-Za-z]+)"/g)].map((m) => m[1]);
  assert.deepEqual(HEADER.advertised_order, kinds, "the fixture header's committed copy matches the Python tuple");
  assert.equal(kinds.length, 15);
});

const KIND_CASES = {
  Text: {
    components: [
      { id: "root", component: "Column", children: ["h1", "h2", "h3", "h4", "h5", "b", "c"] },
      ...["h1", "h2", "h3", "h4", "h5"].map((v) => ({ id: v, component: "Text", variant: v, text: `variant ${v}` })),
      { id: "b", component: "Text", variant: "body", text: "variant body" },
      { id: "c", component: "Text", variant: "caption", text: "variant caption" },
    ],
    check(root) {
      for (const v of ["h1", "h2", "h3", "h4", "h5", "b", "c"]) {
        const el = comp(root, v);
        assert.ok(el, `Text ${v} has an element`);
        assert.equal(el.textContent, `variant ${v === "b" ? "body" : v === "c" ? "caption" : v}`);
      }
    },
  },
  Column: {
    components: [
      { id: "root", component: "Column", children: ["x", "y"] },
      { id: "x", component: "Text", text: "first" },
      { id: "y", component: "Text", text: "second" },
    ],
    check(root) {
      const col = comp(root, "root") || root;
      assert.deepEqual(readableChunks(col), ["first", "second"]);
    },
  },
  Row: {
    components: [
      { id: "root", component: "Column", children: ["r"] },
      { id: "r", component: "Row", justify: "spaceBetween", align: "center", children: ["k", "v"] },
      { id: "k", component: "Text", text: "Key" },
      { id: "v", component: "Text", text: "Value" },
    ],
    check(root) {
      assert.deepEqual(readableChunks(comp(root, "r")), ["Key", "Value"]);
    },
  },
  List: {
    components: [
      { id: "root", component: "Column", children: ["l", "h"] },
      { id: "l", component: "List", direction: "vertical", children: ["i1", "i2", "i3"] },
      { id: "h", component: "List", direction: "horizontal", children: ["j1"] },
      { id: "i1", component: "Text", text: "one" },
      { id: "i2", component: "Text", text: "two" },
      { id: "i3", component: "Text", text: "three" },
      { id: "j1", component: "Text", text: "across" },
    ],
    check(root) {
      assert.deepEqual(readableChunks(comp(root, "l")), ["one", "two", "three"]);
      assert.deepEqual(readableChunks(comp(root, "h")), ["across"]);
    },
  },
  Card: {
    components: [
      { id: "root", component: "Column", children: ["intro", "tile"] },
      { id: "intro", component: "Text", text: "outside" },
      { id: "tile", component: "Card", child: "inner" },
      { id: "inner", component: "Text", text: "inside the tile" },
    ],
    check(root) {
      const tile = comp(root, "tile");
      assert.ok(tile, "a nested Card has its own element");
      assert.deepEqual(readableChunks(tile), ["inside the tile"]);
    },
  },
  Divider: {
    components: [
      { id: "root", component: "Column", children: ["a", "d", "b"] },
      { id: "a", component: "Text", text: "above" },
      { id: "d", component: "Divider", axis: "horizontal" },
      { id: "b", component: "Text", text: "below" },
    ],
    check(root) {
      const d = comp(root, "d");
      assert.ok(d, "the Divider has an element");
      assert.equal(d.textContent.trim(), "", "a Divider holds no words");
      const order = elements(root).filter((e) => ["a", "d", "b"].includes(e.getAttribute("data-a2-id"))).map((e) => e.getAttribute("data-a2-id"));
      assert.deepEqual(order, ["a", "d", "b"]);
    },
  },
  Icon: {
    components: [
      { id: "root", component: "Row", children: ["i1", "i2", "i3", "i4", "i5"] },
      ...["check", "close", "search", "info", "home"].map((name, k) => ({ id: `i${k + 1}`, component: "Icon", name })),
    ],
    check(root, dom) {
      const svgs = elements(root).filter((e) => e.namespaceURI === dom.SVG_NS && e.localName === "svg");
      assert.ok(svgs.length >= 1, "at least one of check/close/search/info/home renders an inline <svg> (implementation note 9)");
      assert.equal(root.textContent.replace(/\s/g, ""), "", "an Icon shows no words: not its name, not a ligature");
    },
  },
  Tabs: {
    components: [
      { id: "root", component: "Tabs", tabs: [{ title: "First tab", child: "p1" }, { title: "Second tab", child: "p2" }] },
      { id: "p1", component: "Text", text: "panel one" },
      { id: "p2", component: "Text", text: "panel two" },
    ],
    check(root) {
      const tabs = elements(root).filter((e) => e.getAttribute("role") === "tab");
      assert.deepEqual(tabs.map((t) => t.textContent.trim()), ["First tab", "Second tab"]);
      const p1 = elements(root).find((e) => e.getAttribute("data-a2-id") === "p1");
      assert.ok(p1 && !isHidden(p1, root), "the first tab's panel shows");
    },
  },
  Modal: {
    components: [
      { id: "root", component: "Column", children: ["m"] },
      { id: "m", component: "Modal", trigger: "open", content: "body" },
      { id: "open", component: "Text", text: "Show details" },
      { id: "body", component: "Text", text: "The details" },
    ],
    check(root, dom) {
      assert.ok(root.textContent.includes("Show details"), "the trigger renders");
      const shown = () => elements(root).some((e) => e.getAttribute("data-a2-id") === "body" && !isHidden(e, root) && e.textContent === "The details");
      assert.equal(shown(), false, "H-T164-5: the content is closed until the trigger is activated");
      const trigger = elements(root).find((e) => e.localName === "button" && e.textContent.includes("Show details"));
      assert.ok(trigger, "the trigger is a <button> (A12)");
      userClick(dom, trigger);
      assert.equal(shown(), true, "H-T164-5: activating the trigger shows the content inline");
      assert.equal(elements(root).filter((e) => e.localName === "dialog").length, 0, "no <dialog>");
    },
  },
  Button: {
    components: [
      { id: "root", component: "Column", children: ["b1", "b2", "b3"] },
      ...["default", "primary", "borderless"].map((variant, k) => ({ id: `b${k + 1}`, component: "Button", variant, child: `l${k + 1}`, action: { event: { name: "__gate.clicked", context: { label: `Go ${variant}` } } } })),
      ...["default", "primary", "borderless"].map((variant, k) => ({ id: `l${k + 1}`, component: "Text", text: `Go ${variant}` })),
    ],
    check(root) {
      for (const [k, v] of ["default", "primary", "borderless"].entries()) {
        const b = comp(root, `b${k + 1}`);
        const button = b && (b.localName === "button" ? b : elements(b).find((e) => e.localName === "button"));
        assert.ok(button, `Button b${k + 1} is a <button>`);
        assert.equal(button.textContent.trim(), `Go ${v}`);
      }
    },
  },
  CheckBox: {
    components: [
      { id: "root", component: "Column", children: ["c1", "c2"] },
      { id: "c1", component: "CheckBox", label: "Unticked", value: false },
      { id: "c2", component: "CheckBox", label: "Ticked", value: true },
    ],
    check(root) {
      const boxes = inputsOf(root).filter((e) => e.type === "checkbox");
      assert.equal(boxes.length, 2, "two <input type=checkbox>");
      assert.deepEqual(boxes.map((b) => b.checked), [false, true], "checked follows the literal value");
      assert.ok(root.textContent.includes("Unticked") && root.textContent.includes("Ticked"), "the labels show");
    },
  },
  ChoicePicker: {
    components: [
      { id: "root", component: "Column", children: ["p"] },
      { id: "p", component: "ChoicePicker", label: "Size", variant: "mutuallyExclusive", options: [{ label: "Small", value: "s" }, { label: "Medium", value: "m" }, { label: "Large", value: "l" }], value: ["m"] },
    ],
    check(root) {
      const chips = chipsOf(comp(root, "p"));
      assert.deepEqual(chips.map((c) => c.textContent.trim()), ["Small", "Medium", "Large"]);
      assert.deepEqual(chips.map(chipState), [false, true, false], "the literal value selects Medium");
    },
  },
  TextField: {
    components: [
      { id: "root", component: "Column", children: ["t1", "t2"] },
      { id: "t1", component: "TextField", label: "Name", value: "Ada" },
      { id: "t2", component: "TextField", label: "Notes", variant: "longText" },
    ],
    check(root) {
      const ins = inputsOf(root);
      assert.equal(ins.length, 2, "two text inputs");
      assert.equal(ins[0].value, "Ada", "the literal value shows in the input");
    },
  },
  Slider: {
    components: [
      { id: "root", component: "Column", children: ["s"] },
      { id: "s", component: "Slider", label: "Volume", min: 0, max: 10, value: 4 },
    ],
    check(root) {
      const range = inputsOf(root).find((e) => e.type === "range");
      assert.ok(range, "an <input type=range>");
      assert.equal(Number(range.value), 4);
      assert.equal(Number(range.getAttribute("min")), 0);
      assert.equal(Number(range.getAttribute("max")), 10);
    },
  },
  DateTimeInput: {
    components: [
      { id: "root", component: "Column", children: ["d"] },
      { id: "d", component: "DateTimeInput", label: "Departure", enableDate: true, value: "2026-11-02" },
    ],
    check(root) {
      const input = inputsOf(root).find((e) => ["date", "datetime-local", "time"].includes(e.type));
      assert.ok(input, "a native date/time input");
      assert.equal(input.value, "2026-11-02");
    },
  },
};

for (const kind of HEADER.advertised_order) {
  rowA(`A2: ${kind} renders from its catalog props`, async () => {
    const h = await harness();
    const c = KIND_CASES[kind];
    assert.ok(c, `a case for every advertised kind (${kind})`);
    h.deliver(`k-${kind}`, surface(`k-${kind}`, c.components));
    const root = h.root(`k-${kind}`);
    assert.ok(root.hasAttribute("data-a2ui-root"), "the card root carries data-a2ui-root");
    const own = c.components.find((x) => x.component === kind && x.id !== "root") || c.components[0];
    const el = comp(root, own.id) || (own.id === "root" ? root : null);
    assert.ok(el, `${kind} ${own.id} has an outer element with data-a2-id`);
    if (el !== root) assert.equal(el.getAttribute("data-a2-kind"), kind, "data-a2-kind names the kind");
    c.check(root, h.dom);
    assert.deepEqual(h.dom.rec.violations, []);
  });
}

// =========================================================================================================
// A3: the fallback for everything else is nothing, logged once per (surface, kind)
// =========================================================================================================

rowA("A3: Image, Video, AudioPlayer, an unknown kind, a FunctionCall value and a template render nothing, log once, siblings stay", async () => {
  const h = await harness();
  const comps3 = [
    { id: "root", component: "Column", children: ["before", "img", "vid", "aud", "gizmo", "fn", "tpl", "after"] },
    { id: "before", component: "Text", text: "before" },
    { id: "img", component: "Image", url: "https://example.com/x.png" },
    { id: "vid", component: "Video", url: "https://example.com/v.mp4" },
    { id: "aud", component: "AudioPlayer", url: "https://example.com/a.mp3" },
    { id: "gizmo", component: "Gizmo", text: "gizmo words" },
    { id: "fn", component: "Text", text: { call: "formatString", args: { value: "called words" }, returnType: "string" } },
    { id: "tpl", component: "List", children: { path: "/items", componentId: "row" } },
    { id: "row", component: "Text", text: "template row" },
    { id: "after", component: "Text", text: "after" },
  ];
  const before = h.logs.length;
  h.deliver("s3", surface("s3", comps3, { model: { items: [{ a: 1 }, { a: 2 }] } }));
  const root = h.root("s3");
  assert.deepEqual(readableChunks(root), ["before", "after"], "only the siblings' words show, in order");
  for (const cid of ["img", "vid", "aud", "gizmo"]) assert.equal(comp(root, cid), null, `${cid} renders no element`);
  for (const w of ["gizmo words", "called words", "template row", "formatString"]) assert.ok(!root.textContent.includes(w), `no "${w}"`);
  const first = h.logs.slice(before).filter((l) => !l.startsWith("[bridge]"));
  for (const kind of ["Image", "Video", "AudioPlayer", "Gizmo"]) {
    assert.equal(first.filter((l) => l.includes(kind)).length, 1, `one log line names ${kind}; got ${JSON.stringify(first)}`);
  }
  assert.ok(first.length >= 6, `one line each for the 4 kinds, the FunctionCall and the template; got ${JSON.stringify(first)}`);
  // the same canvas twice more: not one line per canvas
  const mid = h.logs.length;
  h.deliver("s3", [COMPONENTS("s3", comps3)]);
  h.deliver("s3", [COMPONENTS("s3", comps3)]);
  const { resends } = await benignLines();
  const again = h.logs.slice(mid).filter((l) => !l.startsWith("[bridge]"));
  assert.equal(again.length, resends, `later canvases add no fallback line for the same (surface, kind); got ${JSON.stringify(again)}`);
  // a second surface is a new (surface, kind)
  const other = h.logs.length;
  h.deliver("s3b", surface("s3b", [{ id: "root", component: "Column", children: ["img"] }, { id: "img", component: "Image", url: "data:image/png;base64,AAAA" }]));
  assert.equal(h.logs.slice(other).filter((l) => l.includes("Image")).length, 1, "a second surface logs its Image once");
  assert.deepEqual(h.dom.rec.violations, [], "and nothing reached a fetch path");
});

// =========================================================================================================
// A4: verbatim, through textContent only
// =========================================================================================================

rowA("A4: every fixture surface shows expected_texts verbatim and in tree order", async () => {
  assert.ok(SURFACES.length >= 11, "the 8 named payloads, the quiz and the census sample");
  const h = await harness();
  for (const s of SURFACES) {
    const root = h.deliverFixture(s);
    const chunks = readableChunks(root);
    for (const want of s.expected_texts) assert.ok(chunks.includes(want), `${s.key}: "${want}" is shown byte-equal; shown ${JSON.stringify(chunks)}`);
    assert.ok(isSubsequence(s.expected_texts, chunks), `${s.key}: the model's strings appear in tree order\nwant ${JSON.stringify(s.expected_texts)}\ngot  ${JSON.stringify(chunks)}`);
  }
  assert.deepEqual(h.dom.rec.violations, []);
});

const ADVERSARIAL = {
  markup: "<img src=x onerror=alert(1)>",
  closer: "</div><script>",
  entity: "&amp;",
  markdown: "**bold**",
  newline: "line one\nline two",
  rtl: "שלום مرحبا ‮evil",
  emoji: "\u{1F373} \u{1F1F5}\u{1F1F9} \u{1F468}‍\u{1F469}‍\u{1F467}",
  long: "x".repeat(10000),
};

rowA("A4: adversarial strings render as that exact string and create no element", async () => {
  const count = async (text) => {
    const h = await harness();
    const before = h.dom.rec.creates.length;
    h.deliver("adv", surface("adv", [{ id: "root", component: "Column", children: ["t", "b"] }, { id: "t", component: "Text", text }, { id: "b", component: "Button", child: "bl", action: { event: { name: "__gate.clicked", context: { label: text } } } }, { id: "bl", component: "Text", text }]));
    const made = h.dom.rec.creates.slice(before).filter((n) => n.nodeType === 1);
    return { h, made, root: h.root("adv") };
  };
  const plain = await count("plain");
  for (const [name, text] of Object.entries(ADVERSARIAL)) {
    const { h, made, root } = await count(text);
    assert.equal(comp(root, "t").textContent, text, `${name}: the Text's element holds the exact string`);
    assert.equal(comp(root, "bl").textContent, text, `${name}: the button label holds the exact string`);
    assert.equal(made.length, plain.made.length, `${name}: the string created no element (${made.length} vs ${plain.made.length} for "plain")`);
    assert.deepEqual(made.map((e) => e.localName), plain.made.map((e) => e.localName), `${name}: the same elements as for "plain"`);
    assert.deepEqual(h.dom.rec.violations, [], `${name}: no sink`);
  }
});

// =========================================================================================================
// A5: nothing synthesised
// =========================================================================================================

rowA("A5: every fixture card's text is exactly the model's strings (no other non-whitespace character)", async () => {
  const h = await harness();
  for (const s of SURFACES) {
    const root = h.deliverFixture(s);
    assert.deepEqual(charBag(readableChunks(root)), charBag(s.expected_texts), `${s.key}: the card shows exactly the model's characters\nwant ${JSON.stringify(s.expected_texts)}\ngot  ${JSON.stringify(readableChunks(root))}`);
  }
});

rowA("A5: the t9 card renders 10 strings (title + 9 rows), 0 checkboxes, 0 numbers the model did not write", async () => {
  const s = byKey("desk-t9");
  assert.equal(s.expected_texts.length, 10, "fixture: t9 is a title and 9 rows");
  const h = await harness();
  const root = h.deliverFixture(s);
  const chunks = readableChunks(root);
  assert.deepEqual(chunks, s.expected_texts, "exactly the 10 strings, in order");
  const boxes = elements(root).filter((e) => (e.localName === "input" && e.type === "checkbox") || e.getAttribute("role") === "checkbox");
  assert.equal(boxes.length, 0, "no checkbox the model did not write");
  const digits = (xs) => xs.join("").replace(/[^0-9]/g, "");
  assert.equal(digits(chunks), digits(s.expected_texts), "no step number, count or total");
});

rowA("A5: a surface with no root yet renders an empty card frame with data-a2-empty and no words", async () => {
  const h = await harness();
  h.deliver("noroot", [CREATE("noroot"), COMPONENTS("noroot", [{ id: "orphan", component: "Text", text: "orphan words" }])]);
  const root = h.root("noroot");
  assert.ok(root.hasAttribute("data-a2-empty"), "data-a2-empty on the rootless card");
  assert.equal(root.textContent.replace(/\s/g, ""), "", "no words: no 'Loading surface…', no orphan text");
  h.deliver("noroot", [COMPONENTS("noroot", [{ id: "root", component: "Column", children: ["orphan"] }, { id: "orphan", component: "Text", text: "orphan words" }])]);
  assert.equal(root.hasAttribute("data-a2-empty"), false, "the marker goes once root arrives");
  assert.deepEqual(readableChunks(root), ["orphan words"]);
});

// =========================================================================================================
// A6: dangling references and cycles
// =========================================================================================================

const noIdTwiceOnAPath = (root) => {
  for (const el of elements(root)) {
    const id = el.getAttribute("data-a2-id");
    if (!id) continue;
    for (let a = el.parentNode; a && a !== root.parentNode; a = a.parentNode) {
      if (a.nodeType === 1 && a.getAttribute("data-a2-id") === id) return id;
    }
  }
  return null;
};

rowA("A6: a dangling child renders nothing and its siblings render in order", async () => {
  const h = await harness();
  h.deliver("dang", surface("dang", [{ id: "root", component: "Column", children: ["a", "ghost", "b"] }, { id: "a", component: "Text", text: "alpha" }, { id: "b", component: "Text", text: "beta" }]));
  const root = h.root("dang");
  assert.deepEqual(readableChunks(root), ["alpha", "beta"]);
  assert.equal(comp(root, "ghost"), null, "no element for an id no component defines");
});

const CYCLES = {
  "a -> b -> a": [{ id: "root", component: "Column", children: ["a", "tail"] }, { id: "a", component: "Column", children: ["b", "atext"] }, { id: "b", component: "Column", children: ["a", "btext"] }, { id: "atext", component: "Text", text: "in a" }, { id: "btext", component: "Text", text: "in b" }, { id: "tail", component: "Text", text: "tail" }],
  "a self-reference": [{ id: "root", component: "Column", children: ["me", "tail"] }, { id: "me", component: "Column", children: ["me", "metext"] }, { id: "metext", component: "Text", text: "me" }, { id: "tail", component: "Text", text: "tail" }],
  "a Card whose child is its own ancestor": [{ id: "root", component: "Column", children: ["outer", "tail"] }, { id: "outer", component: "Column", children: ["card"] }, { id: "card", component: "Card", child: "outer" }, { id: "tail", component: "Text", text: "tail" }],
  "a 200-deep chain past the depth bound": [{ id: "root", component: "Column", children: ["d0", "tail"] }, ...Array.from({ length: 200 }, (_, k) => ({ id: `d${k}`, component: "Column", children: [k === 199 ? "leaf" : `d${k + 1}`] })), { id: "leaf", component: "Text", text: "deepest" }, { id: "tail", component: "Text", text: "tail" }],
};

for (const [name, components] of Object.entries(CYCLES)) {
  rowA(`A6: ${name} terminates, renders each component at most once on any root-to-leaf path, and logs one line`, async () => {
    const h = await harness();
    const before = h.logs.length;
    h.deliver("cyc", surface("cyc", components));
    const root = h.root("cyc");
    assert.equal(noIdTwiceOnAPath(root), null, "no component id appears twice on one root-to-leaf path");
    assert.ok(readableChunks(root).includes("tail"), "the siblings after the cycle still render");
    const lines = h.logs.slice(before).filter((l) => !l.startsWith("[bridge]"));
    const { first } = await benignLines();
    assert.equal(lines.length, first + 1, `one log line for the cycle or the depth bound (beyond the ${first} a benign surface logs); got ${JSON.stringify(lines)}`);
    if (name.includes("200")) {
      const chain = elements(root).filter((e) => /^d\d+$/.test(e.getAttribute("data-a2-id") || ""));
      assert.ok(chain.length >= 1 && chain.length < 200, `the depth bound stops the chain somewhere below 200 (rendered ${chain.length})`);
      assert.ok(!root.textContent.includes("deepest"), "the leaf past the bound is not rendered");
    }
  });
}

rowA("A6: a 10 000-child Row terminates and renders each child at most once", async () => {
  const h = await harness();
  const kids = Array.from({ length: 10000 }, (_, k) => ({ id: `c${k}`, component: "Text", text: `w${k}` }));
  h.deliver("wide", surface("wide", [{ id: "root", component: "Row", children: kids.map((k) => k.id) }, ...kids]));
  const root = h.root("wide");
  const seen = new Map();
  for (const e of elements(root)) {
    const id = e.getAttribute("data-a2-id");
    if (id) seen.set(id, (seen.get(id) || 0) + 1);
  }
  assert.ok([...seen.values()].every((n) => n === 1), "no child rendered twice (termination is the row finishing; no wall-time bar, rules/tests.md)");
  assert.ok(readableChunks(root).length >= 1, "the Row rendered");
});

// =========================================================================================================
// A7: morph in place, not re-creation
// =========================================================================================================

rowA("A7: an identical canvas re-applied causes zero DOM mutations (every fixture surface, 3 re-sends)", async () => {
  const h = await harness();
  for (const s of SURFACES) {
    h.deliverFixture(s);
    const last = s.canvases[s.canvases.length - 1];
    for (let k = 0; k < 3; k++) {
      h.dom.rec.mutations.length = 0;
      h.deliver(s.surface_id, last);
      assert.equal(h.dom.rec.mutations.length, 0, `${s.key}: re-send ${k + 1} of the last canvas mutated the DOM: ${mutationSummary(h.dom.rec)}`);
    }
  }
});

rowA("A7: the card root and every component node stay === across the t9 and flight canvases", async () => {
  for (const key of ["desk-t9", "trial-flight", "desk-t10"]) {
    const s = byKey(key);
    assert.ok(s.canvases.length >= 2, `${key} has more than one canvas`);
    const h = await harness();
    let previous = null;
    let rootNode = null;
    const sent = new Map(); // component id -> its JSON as last sent
    for (const cv of s.canvases) {
      const createsBefore = h.dom.rec.creates.length;
      // the components this canvas adds or changes (a changed one may legitimately build nodes)
      const touched = new Set();
      for (const m of cv) {
        for (const c of (m.updateComponents && m.updateComponents.components) || []) {
          if (sent.get(c.id) !== JSON.stringify(c)) touched.add(c.id);
          sent.set(c.id, JSON.stringify(c));
        }
      }
      h.deliver(s.surface_id, cv);
      const root = h.root(s.surface_id);
      if (rootNode) assert.equal(root, rootNode, `${key}: the card root is the same object`);
      rootNode = root;
      const now = new Map(elements(root).filter((e) => e.hasAttribute("data-a2-id")).map((e) => [e.getAttribute("data-a2-id"), e]));
      if (previous) {
        for (const [id, el] of previous) {
          if (now.has(id) && now.get(id).getAttribute("data-a2-kind") === el.getAttribute("data-a2-kind")) {
            assert.equal(now.get(id), el, `${key}: component ${id} keeps its node`);
          }
        }
        // nodes are created only for the components this canvas added or changed
        for (const made of h.dom.rec.creates.slice(createsBefore)) {
          if (!made.isConnected) continue;
          const id = ownerId(made.nodeType === 1 ? made : made.parentNode, root);
          assert.ok(id && touched.has(id), `${key}: a node was created inside ${id || "the frame"}, which this canvas neither added nor changed (touched: ${[...touched]})`);
        }
      }
      previous = now;
    }
  }
});

rowA("A7: a kind change replaces only that component's subtree; a dropped reference detaches its node", async () => {
  const h = await harness();
  const base = [{ id: "root", component: "Column", children: ["a", "x", "b", "gone"] }, { id: "a", component: "Text", text: "alpha" }, { id: "x", component: "Text", text: "was text" }, { id: "b", component: "Text", text: "beta" }, { id: "gone", component: "Text", text: "goodbye" }];
  h.deliver("kc", surface("kc", base));
  const root = h.root("kc");
  const [a, x, b, gone] = ["a", "x", "b", "gone"].map((id) => comp(root, id));
  h.deliver("kc", [COMPONENTS("kc", [{ id: "root", component: "Column", children: ["a", "x", "b"] }, { id: "a", component: "Text", text: "alpha" }, { id: "x", component: "Button", child: "xl", action: { event: { name: "__gate.clicked", context: { label: "now a button" } } } }, { id: "xl", component: "Text", text: "now a button" }, { id: "b", component: "Text", text: "beta" }])]);
  assert.equal(h.root("kc"), root, "same root");
  assert.equal(comp(root, "a"), a, "a sibling keeps its node");
  assert.equal(comp(root, "b"), b, "the other sibling keeps its node");
  assert.notEqual(comp(root, "x"), x, "the changed kind gets a new node");
  assert.equal(comp(root, "x").getAttribute("data-a2-kind"), "Button");
  assert.equal(x.isConnected, false, "the old Text node is detached");
  assert.equal(gone.isConnected, false, "a component the model no longer references is detached");
  assert.deepEqual(readableChunks(root), ["alpha", "now a button", "beta"]);
});

// =========================================================================================================
// A8: the user's state survives a morph
// =========================================================================================================

rowA("A8 (adversarial): typed text, caret and focus in the flight card's Origin city survive 5 later canvases", async () => {
  const s = byKey("trial-flight");
  const h = await harness();
  const root = h.deliverFixture(s, 1);
  const origin = inputsOf(root).find((e) => accessibleName(e, h.dom.document) === "Origin city" || e.getAttribute("placeholder") === "Origin city");
  assert.ok(origin, "the Origin city field is an input named or placeheld by the model's words");
  typeInto(h.dom, origin, "Lisbo", 3);
  const later = [s.canvases[1], ...[1, 2, 3, 4].map((k) => s.canvases[1].map((m) => (m.updateComponents ? COMPONENTS(s.surface_id, m.updateComponents.components.map((c) => (c.id === "title" ? { ...c, text: `Flight Booking ${k}` } : c))) : m)))];
  for (const cv of later) {
    h.deliver(s.surface_id, cv);
    const now = inputsOf(root).find((e) => e === origin);
    assert.ok(now, "the same input node is still in the card");
    assert.equal(origin.value, "Lisbo", "the typed text is kept");
    assert.deepEqual([origin.selectionStart, origin.selectionEnd], [3, 3], "the caret is kept");
    assert.equal(h.dom.document.activeElement, origin, "focus is kept");
  }
  assert.ok(root.textContent.includes("Flight Booking 4"), "the later canvases did apply (the title changed)");
});

rowA("A8: a chip, a tick, a slider, a date, typed text and the selected tab survive 5 later canvases; a new model value wins", async () => {
  const comps8 = (title, tfValue = "seed") => [
    { id: "root", component: "Column", children: ["title", "form", "tabs"] },
    { id: "title", component: "Text", text: title },
    { id: "tabs", component: "Tabs", tabs: [{ title: "First", child: "first" }, { title: "Other", child: "other" }] },
    { id: "first", component: "Text", text: "first panel" },
    { id: "form", component: "Column", children: ["tf", "cb", "sl", "dt", "cp"] },
    { id: "tf", component: "TextField", label: "Note", value: tfValue },
    { id: "cb", component: "CheckBox", label: "Agree", value: false },
    { id: "sl", component: "Slider", label: "Volume", min: 0, max: 10, value: 3 },
    { id: "dt", component: "DateTimeInput", label: "When", enableDate: true, value: "2026-11-02" },
    { id: "cp", component: "ChoicePicker", label: "Size", options: [{ label: "S", value: "s" }, { label: "M", value: "m" }, { label: "L", value: "l" }], value: [] },
    { id: "other", component: "Text", text: "other panel" },
  ];
  const h = await harness();
  h.deliver("st", surface("st", comps8("v0")));
  const root = h.root("st");
  const tf = inputsOf(comp(root, "tf"))[0] || comp(root, "tf");
  const cb = inputsOf(comp(root, "cb")).find((e) => e.type === "checkbox");
  const sl = inputsOf(comp(root, "sl")).find((e) => e.type === "range");
  const dt = inputsOf(comp(root, "dt")).find((e) => ["date", "datetime-local", "time"].includes(e.type));
  const chipM = chipsOf(comp(root, "cp")).find((c) => c.textContent.trim() === "M");
  assert.ok(tf && cb && sl && dt && chipM, "every control is there");
  typeInto(h.dom, tf, "seedling");
  userClick(h.dom, cb);
  h.dom.quiet(() => {
    sl.value = "7";
    sl.dispatchEvent(new h.dom.FakeEvent("input"));
    sl.dispatchEvent(new h.dom.FakeEvent("change"));
    dt.value = "2026-12-25";
    dt.dispatchEvent(new h.dom.FakeEvent("input"));
    dt.dispatchEvent(new h.dom.FakeEvent("change"));
  });
  userClick(h.dom, chipM);
  // the tabs sit beside the form (not around it), so every control stays connected and the rows below
  // read live nodes; "Other" is not the initial tab, so keeping it is the user's state
  const tabOther = elements(root).find((e) => e.getAttribute("role") === "tab" && e.textContent.trim() === "Other");
  userClick(h.dom, tabOther);
  assert.equal(tabOther.getAttribute("aria-selected"), "true", "precondition: the Other tab is selected");
  for (let k = 1; k <= 5; k++) {
    h.deliver("st", [COMPONENTS("st", comps8(`v${k}`))]);
    assert.ok(root.textContent.includes(`v${k}`), `canvas ${k} applied`);
    assert.ok([tf, cb, sl, dt, chipM].every((n) => n.isConnected && root.contains(n)), `canvas ${k}: the same control nodes are still in the card`);
    assert.equal(tf.value, "seedling", `canvas ${k}: typed text kept`);
    assert.equal(cb.checked, true, `canvas ${k}: tick kept`);
    assert.equal(Number(sl.value), 7, `canvas ${k}: slider kept`);
    assert.equal(dt.value, "2026-12-25", `canvas ${k}: date kept`);
    assert.equal(chipState(chipM), true, `canvas ${k}: chip kept`);
    const sel = elements(root).filter((e) => e.getAttribute("role") === "tab" && e.getAttribute("aria-selected") === "true").map((e) => e.textContent.trim());
    assert.deepEqual(sel, ["Other"], `canvas ${k}: selected tab kept`);
  }
  // the guard's other side (rules/tests.md): the model changing that component's own value prop wins
  h.deliver("st", [COMPONENTS("st", comps8("v6", "model rewrote this"))]);
  assert.equal(tf.value, "model rewrote this", "a new literal value from the model replaces the user's text (implementation note 4)");
});

// =========================================================================================================
// A9: data model and bindings (render half; the dataModelFor/drop engine rows are in ui_engine.test.mjs)
// =========================================================================================================

rowA("A9: updateDataModel at / and at a nested path updates every path-bound Text and input; a missing value deletes the key", async () => {
  const h = await harness();
  const id = "dm";
  h.deliver(id, surface(id, [
    { id: "root", component: "Column", children: ["greet", "name", "agree", "vol", "when"] },
    { id: "greet", component: "Text", text: { path: "/user/greeting" } },
    { id: "name", component: "TextField", label: "Name", value: { path: "/user/name" } },
    { id: "agree", component: "CheckBox", label: "Agree", value: { path: "/agree" } },
    { id: "vol", component: "Slider", label: "Volume", min: 0, max: 10, value: { path: "/vol" } },
    { id: "when", component: "DateTimeInput", label: "When", enableDate: true, value: { path: "/when" } },
  ], { send: true, model: { user: { greeting: "Hello", name: "Ada" }, agree: true, vol: 2, when: "2026-11-02" } }));
  const root = h.root(id);
  const name = inputsOf(comp(root, "name"))[0];
  const agree = inputsOf(comp(root, "agree")).find((e) => e.type === "checkbox");
  const vol = inputsOf(comp(root, "vol")).find((e) => e.type === "range");
  const when = inputsOf(comp(root, "when"))[0];
  assert.equal(comp(root, "greet").textContent, "Hello");
  assert.deepEqual([name.value, agree.checked, Number(vol.value), when.value], ["Ada", true, 2, "2026-11-02"]);
  h.deliver(id, [DATA(id, "/user/greeting", "Bonjour"), DATA(id, "/vol", 9)]);
  assert.equal(comp(root, "greet").textContent, "Bonjour", "a nested-path update reaches the bound Text");
  assert.equal(Number(vol.value), 9, "and the bound Slider");
  h.deliver(id, [DATA(id, "/", { user: { greeting: "Hi", name: "Grace" }, agree: false, vol: 5, when: "2027-01-01" })]);
  assert.deepEqual([comp(root, "greet").textContent, name.value, agree.checked, Number(vol.value), when.value], ["Hi", "Grace", false, 5, "2027-01-01"], "a / update reaches every binding");
  h.deliver(id, [DATA(id, "/user/greeting")]);
  assert.equal((comp(root, "greet") || { textContent: "" }).textContent, "", "a missing value deletes the key: the bound Text shows nothing");
  assert.equal("greeting" in h.engine.dataModelFor(id).user, false, "the key is gone from the model");
});

rowA("A9: a user edit to a path-bound input writes its value at the path with the right type; a literal input writes nothing", async () => {
  const h = await harness();
  const id = "edit";
  h.deliver(id, surface(id, [
    { id: "root", component: "Column", children: ["name", "agree", "vol", "when", "pick", "multi", "lit"] },
    { id: "name", component: "TextField", label: "Name", value: { path: "/name" } },
    { id: "agree", component: "CheckBox", label: "Agree", value: { path: "/agree" } },
    { id: "vol", component: "Slider", label: "Volume", min: 0, max: 10, value: { path: "/vol" } },
    { id: "when", component: "DateTimeInput", label: "When", enableDate: true, value: { path: "/when" } },
    { id: "pick", component: "ChoicePicker", label: "One", variant: "mutuallyExclusive", options: [{ label: "A", value: "a" }, { label: "B", value: "b" }], value: { path: "/pick" } },
    { id: "multi", component: "ChoicePicker", label: "Many", variant: "multipleSelection", options: [{ label: "X", value: "x" }, { label: "Y", value: "y" }, { label: "Z", value: "z" }], value: { path: "/multi" } },
    { id: "lit", component: "TextField", label: "Literal", value: "fixed" },
  ], { send: true, model: { name: "", agree: false, vol: 0, when: "", pick: [], multi: [] } }));
  const root = h.root(id);
  typeInto(h.dom, inputsOf(comp(root, "name"))[0], "Ada");
  userClick(h.dom, inputsOf(comp(root, "agree")).find((e) => e.type === "checkbox"));
  const vol = inputsOf(comp(root, "vol")).find((e) => e.type === "range");
  h.dom.quiet(() => {
    vol.value = "6";
    vol.dispatchEvent(new h.dom.FakeEvent("input"));
    vol.dispatchEvent(new h.dom.FakeEvent("change"));
  });
  typeInto(h.dom, inputsOf(comp(root, "when"))[0], "2026-11-02");
  const pick = chipsOf(comp(root, "pick"));
  userClick(h.dom, pick[0]);
  userClick(h.dom, pick[1]);
  const multi = chipsOf(comp(root, "multi"));
  userClick(h.dom, multi[2]);
  userClick(h.dom, multi[0]);
  const before = JSON.stringify(h.engine.dataModelFor(id));
  typeInto(h.dom, inputsOf(comp(root, "lit"))[0], "user text");
  assert.equal(JSON.stringify(h.engine.dataModelFor(id)), before, "a literal-valued input writes nothing to the model");
  const model = h.engine.dataModelFor(id);
  assert.equal(model.name, "Ada", "TextField: a string");
  assert.equal(model.agree, true, "CheckBox: a boolean");
  assert.equal(model.vol, 6, "Slider: a number, not the string '6'");
  assert.equal(model.when, "2026-11-02", "DateTimeInput: a string");
  assert.deepEqual(model.pick, ["b"], "mutuallyExclusive ChoicePicker: a list holding only the last pick");
  assert.deepEqual([...model.multi].sort(), ["x", "z"], "multipleSelection ChoicePicker: the list of picked values");
  assert.equal(h.actions.length, 0, "editing sends no action (A10)");
});

rowA("A9 (adversarial, quiz): b on q1 and a on q3, then Check answers -> resolved context.answers and dataModelFor holds the same", async () => {
  const s = byKey("quiz");
  assert.equal(s.send_data_model, true, "fixture: the quiz sets sendDataModel");
  const h = await harness();
  const root = h.deliverFixture(s);
  const optionB = chipsOf(comp(root, "q1_c")).find((c) => c.textContent.trim() === "3/4");
  const optionA = chipsOf(comp(root, "q3_c")).find((c) => c.textContent.trim() === "2/3");
  assert.ok(optionB && optionA, "q1 option b is labelled 3/4 and q3 option a is labelled 2/3 (fixture)");
  userClick(h.dom, optionB);
  userClick(h.dom, optionA);
  assert.equal(h.actions.length, 0, "picking sends nothing");
  const check = elements(comp(root, "check_btn")).concat([comp(root, "check_btn")]).find((e) => e && e.localName === "button");
  userClick(h.dom, check);
  assert.equal(h.actions.length, 1, "one action");
  // By hand from the fixture: updateDataModel "/" seeds answers {q1..q5: []}; the picks replace q1 and q3.
  const answers = { q1: ["b"], q2: [], q3: ["a"], q4: [], q5: [] };
  assert.deepEqual(h.actions[0].context, { quizId: "fractions_1", answers }, "context resolved one level deep: answers from the model, quizId literal");
  assert.deepEqual(h.engine.dataModelFor(s.surface_id).answers, answers, "dataModelFor holds the same answers");
});

// =========================================================================================================
// A10: actions (render half)
// =========================================================================================================

rowA("A10: a gate button click calls onAction once with {name, surfaceId, sourceComponentId, timestamp, context}; the label verbatim", async () => {
  const s = byKey("trial-flight");
  const h = await harness();
  const root = h.deliverFixture(s);
  const btn = [comp(root, "btn"), ...elements(comp(root, "btn"))].find((e) => e && e.localName === "button");
  assert.ok(btn, "the gate button is a <button>");
  userClick(h.dom, btn);
  assert.equal(h.actions.length, 1, "exactly one onAction call");
  const a = h.actions[0];
  assert.deepEqual(Object.keys(a).sort(), ["context", "name", "sourceComponentId", "surfaceId", "timestamp"]);
  assert.equal(a.name, "__gate.clicked");
  assert.equal(a.surfaceId, s.surface_id, "the gate's rewritten turn-… id, the one inside the messages");
  assert.equal(a.sourceComponentId, "btn");
  assert.deepEqual(a.context, { label: "Search Flights" }, "the gate's label arrives verbatim");
  assert.match(a.timestamp, /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
  assert.equal(new Date(a.timestamp).toISOString(), a.timestamp, "Date.prototype.toISOString's form");
});

rowA("A10: context resolves one level deep: a {path} becomes the model value, literals and nested objects pass verbatim", async () => {
  const h = await harness();
  h.deliver("ctx", surface("ctx", [
    { id: "root", component: "Column", children: ["b"] },
    { id: "b", component: "Button", child: "bl", action: { event: { name: "go", context: { bound: { path: "/x" }, lit: "plain", num: 3, nested: { inner: { path: "/x" } } } } } },
    { id: "bl", component: "Text", text: "Go" },
  ], { model: { x: { deep: [1, 2] } } }));
  userClick(h.dom, elements(h.root("ctx")).find((e) => e.localName === "button"));
  assert.equal(h.actions.length, 1);
  assert.deepEqual(h.actions[0].context, { bound: { deep: [1, 2] }, lit: "plain", num: 3, nested: { inner: { path: "/x" } } });
});

rowA("A10 (negative): no action from a Button without an event, typing, picking, ticking, sliding, tabs, a Modal, focus or scrolling", async () => {
  const h = await harness();
  h.deliver("neg", surface("neg", [
    { id: "root", component: "Column", children: ["none", "fn", "tf", "cp", "cb", "sl", "tabs", "m"] },
    { id: "none", component: "Button", child: "l1" },
    { id: "l1", component: "Text", text: "No action" },
    { id: "fn", component: "Button", child: "l2", action: { functionCall: { call: "openUrl", args: { url: "https://example.com" } } } },
    { id: "l2", component: "Text", text: "Function call" },
    { id: "tf", component: "TextField", label: "Type here" },
    { id: "cp", component: "ChoicePicker", label: "Pick", options: [{ label: "P", value: "p" }, { label: "Q", value: "q" }], value: [] },
    { id: "cb", component: "CheckBox", label: "Tick", value: false },
    { id: "sl", component: "Slider", label: "Slide", min: 0, max: 5, value: 1 },
    { id: "tabs", component: "Tabs", tabs: [{ title: "T1", child: "t1" }, { title: "T2", child: "t2" }] },
    { id: "t1", component: "Text", text: "tab one" },
    { id: "t2", component: "Text", text: "tab two" },
    { id: "m", component: "Modal", trigger: "mt", content: "mc" },
    { id: "mt", component: "Text", text: "Open" },
    { id: "mc", component: "Text", text: "Opened" },
  ]));
  const root = h.root("neg");
  const button = (cid) => [comp(root, cid), ...elements(comp(root, cid))].find((e) => e && e.localName === "button");
  userClick(h.dom, button("none"));
  userClick(h.dom, button("fn"));
  typeInto(h.dom, inputsOf(comp(root, "tf"))[0], "words");
  userClick(h.dom, chipsOf(comp(root, "cp"))[1]);
  userClick(h.dom, inputsOf(comp(root, "cb")).find((e) => e.type === "checkbox"));
  const sl = inputsOf(comp(root, "sl")).find((e) => e.type === "range");
  h.dom.quiet(() => {
    sl.value = "4";
    sl.dispatchEvent(new h.dom.FakeEvent("input"));
    sl.dispatchEvent(new h.dom.FakeEvent("change"));
  });
  userClick(h.dom, elements(root).find((e) => e.getAttribute("role") === "tab" && e.textContent.trim() === "T2"));
  userClick(h.dom, elements(root).find((e) => e.localName === "button" && e.textContent.includes("Open")));
  h.dom.quiet(() => {
    for (const el of elements(root).filter((e) => e.localName === "button" || e.localName === "input")) {
      el.focus();
      el.dispatchEvent(new h.dom.FakeEvent("keydown", { key: "Tab" }));
      el.blur();
    }
    root.dispatchEvent(new h.dom.FakeEvent("scroll"));
    root.dispatchEvent(new h.dom.FakeEvent("wheel"));
  });
  assert.deepEqual(h.actions, [], "none of these is an action");
  assert.deepEqual(h.dom.rec.violations, [], "and the functionCall reached no fetch path");
});

// =========================================================================================================
// A11: no network and no markup sinks
// =========================================================================================================

rowA("A11: no forbidden element, attribute, style url( or sink over every fixture surface and an Image/data:/URL surface", async () => {
  const h = await harness();
  for (const s of SURFACES) h.deliverFixture(s);
  h.deliver("net", surface("net", [
    { id: "root", component: "Column", children: ["i1", "i2", "t", "v", "a"] },
    { id: "i1", component: "Image", url: "https://example.com/x.png" },
    { id: "i2", component: "Image", url: "data:image/png;base64,iVBORw0KGgo=" },
    { id: "t", component: "Text", text: "https://example.com/see-this" },
    { id: "v", component: "Video", url: "https://example.com/v.mp4" },
    { id: "a", component: "AudioPlayer", url: "https://example.com/a.mp3" },
  ]));
  assert.deepEqual(h.dom.rec.violations, [], "the trapping DOM recorded a sink (the renderer may have caught the throw)");
  const BAD_TAGS = ["img", "picture", "source", "video", "audio", "iframe", "object", "embed", "link", "script", "style", "a"];
  for (const el of elements(h.dom.document.body)) {
    assert.ok(!BAD_TAGS.includes(el.localName), `no <${el.localName}>`);
    for (const name of el.getAttributeNames()) {
      assert.ok(!["src", "srcset", "href", "action", "formaction", "poster"].includes(name) && !/^on/.test(name), `no ${name} attribute on <${el.localName}>`);
    }
    const style = el.getAttribute("style") || "";
    assert.ok(!style.includes("url("), "no url( in a style");
  }
  assert.equal(comp(h.root("net"), "t").textContent, "https://example.com/see-this", "a URL in Text is just text");
});

const FORBIDDEN_SOURCE = ["innerHTML", "outerHTML", "insertAdjacentHTML", "setHTMLUnsafe", "document.write", "DOMParser", "createContextualFragment", "attachShadow", "eval", "Function(", "fetch(", "XMLHttpRequest", "WebSocket", "EventSource", "sendBeacon"];

/** JS source with string literals, template literals and comments blanked (newlines kept). */
function stripStringsAndComments(src) {
  let out = "";
  let i = 0;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    const d = src[i + 1];
    if (c === "/" && d === "/") {
      while (i < n && src[i] !== "\n") i++;
    } else if (c === "/" && d === "*") {
      i += 2;
      while (i < n && !(src[i] === "*" && src[i + 1] === "/")) out += src[i++] === "\n" ? "\n" : "";
      i += 2;
    } else if (c === '"' || c === "'" || c === "`") {
      i++;
      while (i < n && src[i] !== c) {
        if (src[i] === "\\") i++;
        else if (src[i] === "\n") out += "\n";
        i++;
      }
      i++;
      out += '""';
    } else {
      out += c;
      i++;
    }
  }
  return out;
}

const listJs = (dir) =>
  readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? listJs(p) : p.endsWith(".js") ? [p] : [];
  });

// Not pending: a negative guard over every file under hud/src/ui/, true of today's files and kept true as
// render.js and engine.js land (a forbidden name in either turns it red the moment it is added).
test("A11: a source scan of hud/src/ui/** (strings and comments stripped) finds no markup sink or transport", () => {
  const scan = (code) =>
    FORBIDDEN_SOURCE.filter((word) => {
      const re = word.endsWith("(") ? new RegExp(`\\b${word.slice(0, -1).replace(".", "\\.")}\\s*\\(`) : new RegExp(`\\b${word.replace(".", "\\.")}\\b`);
      return re.test(stripStringsAndComments(code));
    });
  // the scanner is not vacuous, and prose does not trip it
  assert.deepEqual(scan('el.innerHTML = x; const r = fetch (u); new WebSocket(u); el["outerHTML"];'), ["innerHTML", "fetch(", "WebSocket"]);
  assert.deepEqual(scan('// never innerHTML or fetch(\n/* attachShadow */ const s = "eval DOMParser";'), []);
  const hits = [];
  const files = listJs(join(REPO, "hud/src/ui"));
  assert.ok(files.some((f) => f.endsWith("bridge.js")), "the scan reads hud/src/ui/");
  for (const file of files) for (const word of scan(readFileSync(file, "utf8"))) hits.push(`${relative(REPO, file)}: ${word}`);
  assert.deepEqual(hits, [], "a forbidden name in code (strings and comments are stripped first, so prose cannot trip this; a regex literal could)");
});

// =========================================================================================================
// A12: accessibility semantics
// =========================================================================================================

rowA("A12: buttons and chips are <button type=button>; mutuallyExclusive chips are radios, multipleSelection chips are not", async () => {
  const h = await harness();
  for (const s of SURFACES) h.deliverFixture(s);
  h.deliver("aria", surface("aria", [
    { id: "root", component: "Column", children: ["one", "many"] },
    { id: "one", component: "ChoicePicker", label: "One", variant: "mutuallyExclusive", options: [{ label: "A", value: "a" }, { label: "B", value: "b" }], value: ["a"] },
    { id: "many", component: "ChoicePicker", label: "Many", variant: "multipleSelection", options: [{ label: "X", value: "x" }, { label: "Y", value: "y" }], value: ["y"] },
  ]));
  for (const b of elements(h.dom.document.body).filter((e) => e.localName === "button")) assert.equal(b.getAttribute("type"), "button", `<button> "${b.textContent}" has type="button"`);
  const root = h.root("aria");
  const one = chipsOf(comp(root, "one"));
  assert.deepEqual(one.map((c) => c.getAttribute("role")), ["radio", "radio"], "mutuallyExclusive: role=radio");
  assert.deepEqual(one.map((c) => c.getAttribute("aria-checked")), ["true", "false"]);
  assert.ok(elements(comp(root, "one")).concat([comp(root, "one")]).some((e) => e.getAttribute("role") === "radiogroup"), "inside a radiogroup");
  assert.ok(one.every((c) => !c.hasAttribute("aria-pressed")), "no aria-pressed inside a radiogroup (the mock's mistake)");
  const many = chipsOf(comp(root, "many"));
  assert.ok(many.every((c) => c.getAttribute("role") !== "radio"), "multipleSelection: not radios");
  const state = many.map((c) => c.getAttribute("aria-pressed") ?? (c.getAttribute("role") === "checkbox" ? c.getAttribute("aria-checked") : null));
  assert.deepEqual(state, ["false", "true"], "aria-pressed (or checkbox semantics) carries the selection");
  assert.ok(!elements(comp(root, "many")).concat([comp(root, "many")]).some((e) => e.getAttribute("role") === "radiogroup"), "not a radiogroup");
});

/** The final component map of a fixture surface. */
const finalComponents = (s) => {
  const out = new Map();
  for (const cv of s.canvases) for (const m of cv) for (const c of (m.updateComponents && m.updateComponents.components) || []) out.set(c.id, c);
  return out;
};
const ownerId = (el, root) => {
  for (let n = el; n && n !== root.parentNode; n = n.parentNode) if (n.nodeType === 1 && n.hasAttribute("data-a2-id")) return n.getAttribute("data-a2-id");
  return null;
};

rowA("A12: every input has an accessible name from the model's own words, and none is invented", async () => {
  const h = await harness();
  let named = 0;
  for (const s of SURFACES) {
    const root = h.deliverFixture(s);
    const words = new Set(s.expected_texts.map((t) => t.trim()));
    const tree = finalComponents(s);
    for (const input of inputsOf(root)) {
      const cid = ownerId(input, root);
      const c = tree.get(cid);
      assert.ok(c, `${s.key}: the input sits inside a component element (data-a2-id)`);
      // Where the model's words for this field are: its own label, or a Text sibling in the Row that holds it.
      const row = [...tree.values()].find((p) => p.component === "Row" && Array.isArray(p.children) && p.children.includes(cid));
      const rowLabel = row && row.children.map((k) => tree.get(k)).find((k) => k && k.component === "Text" && typeof k.text === "string");
      const nameable = typeof c.label === "string" || Boolean(rowLabel);
      const name = accessibleName(input, h.dom.document);
      if (nameable) {
        named += 1;
        assert.ok(name, `${s.key}: the ${c.component} ${cid} has an accessible name (its label or the Row's Text)`);
      }
      if (name) assert.ok(words.has(name), `${s.key}: ${cid}'s name "${name}" is the model's own words, not invented`);
    }
    for (const group of elements(root).filter((e) => ["radiogroup", "group"].includes(e.getAttribute("role")))) {
      const name = accessibleName(group, h.dom.document);
      if (name) assert.ok(words.has(name), `${s.key}: a choice group's name "${name}" is the model's words`);
    }
  }
  assert.ok(named >= 8, `the fixture's flight cards hold at least 8 nameable inputs (counted ${named})`);
});

rowA("A12: Tabs expose tablist/tab/aria-selected and arrow keys move between tabs", async () => {
  const h = await harness();
  h.deliver("tabs", surface("tabs", [{ id: "root", component: "Tabs", tabs: [{ title: "One", child: "p1" }, { title: "Two", child: "p2" }, { title: "Three", child: "p3" }] }, { id: "p1", component: "Text", text: "first" }, { id: "p2", component: "Text", text: "second" }, { id: "p3", component: "Text", text: "third" }]));
  const root = h.root("tabs");
  assert.ok(elements(root).some((e) => e.getAttribute("role") === "tablist"), "a tablist");
  const tabs = elements(root).filter((e) => e.getAttribute("role") === "tab");
  assert.deepEqual(tabs.map((t) => t.getAttribute("aria-selected")), ["true", "false", "false"]);
  h.dom.quiet(() => {
    tabs[0].focus();
    tabs[0].dispatchEvent(new h.dom.FakeEvent("keydown", { key: "ArrowRight" }));
  });
  const moved = h.dom.document.activeElement === tabs[1] || tabs[1].getAttribute("aria-selected") === "true";
  assert.ok(moved, "ArrowRight moves to the second tab (focus or selection)");
  h.dom.quiet(() => {
    const at = h.dom.document.activeElement === tabs[1] ? tabs[1] : tabs[1];
    at.focus();
    at.dispatchEvent(new h.dom.FakeEvent("keydown", { key: "ArrowLeft" }));
  });
  assert.ok(h.dom.document.activeElement === tabs[0] || tabs[0].getAttribute("aria-selected") === "true", "ArrowLeft moves back");
  assert.equal(h.actions.length, 0, "switching tabs sends nothing");
});

// =========================================================================================================
// Review rows (T-0164 A review, rulings 639d6037): F1 placement bound, T3 pins
// =========================================================================================================

/** A chain of `depth` Columns, each listing its successor twice (a DAG, not a cycle), ending in one Text. */
const sharedChain = (depth) => [
  { id: "root", component: "Column", children: ["c0"] },
  ...Array.from({ length: depth }, (_, k) => ({ id: `c${k}`, component: "Column", children: k === depth - 1 ? ["leaf", "leaf"] : [`c${k + 1}`, `c${k + 1}`] })),
  { id: "leaf", component: "Text", text: "leaf" },
];
const PLACEMENT_CAP = 2000; // ruling 639d6037: a render pass places at most 2,000 component views

rowA("F1 (adversarial): a 21-deep shared-children DAG (2^22 placements uncapped) stays within the placement bound, logs once, and does bounded work", async () => {
  // Built by hand: 21 Columns c0..c20, each children [next, next]; passes UiGate (every id defined, no
  // cycle, all reachable: checked with a fresh UiGate() at 639d6037 by the test-author, ok=True complete=True).
  const h = await harness();
  // A probe an uncapped renderer survives (2^14 placements), so its failure is a named assertion here
  // and not the heap abort the 21-deep surface causes in such a renderer (seen at 11fb3885).
  h.deliver("probe", surface("probe", sharedChain(13)));
  const probe = elements(h.root("probe")).length;
  assert.ok(probe <= PLACEMENT_CAP + 100, `a 13-deep shared DAG put ${probe} elements in the card; the placement bound is ${PLACEMENT_CAP} views`);
  const before = h.logs.length;
  const createsBefore = h.dom.rec.creates.length;
  const mutationsBefore = h.dom.rec.mutations.length;
  h.deliver("dag", surface("dag", sharedChain(21)));
  const root = h.root("dag");
  const els = elements(root);
  assert.ok(els.length <= PLACEMENT_CAP + 100, `the card holds ${els.length} elements; the placement bound is ${PLACEMENT_CAP} views (~2,100 elements)`);
  // work, counted in DOM operations, not wall time (rules/tests.md): linear in the bound, not in 2^22
  const created = h.dom.rec.creates.length - createsBefore;
  const mutated = h.dom.rec.mutations.length - mutationsBefore;
  assert.ok(created <= 3 * (PLACEMENT_CAP + 100), `created ${created} nodes for a capped pass`);
  assert.ok(mutated <= 8 * (PLACEMENT_CAP + 100), `${mutated} DOM mutations for a capped pass`);
  const lines = h.logs.slice(before).filter((l) => !l.startsWith("[bridge]"));
  const { first, resends } = await benignLines();
  assert.equal(lines.length, first + 1, `one line for the cap on this surface; got ${JSON.stringify(lines)}`);
  // re-sending the same canvas does not log the cap again (one line per surface) and stays bounded
  const mid = h.logs.length;
  const mid2 = h.dom.rec.creates.length;
  h.deliver("dag", [COMPONENTS("dag", sharedChain(21))]);
  h.deliver("dag", [COMPONENTS("dag", sharedChain(21))]);
  assert.equal(h.logs.slice(mid).filter((l) => !l.startsWith("[bridge]")).length, resends, "no second cap line for the same surface");
  assert.ok(h.dom.rec.creates.length - mid2 <= 3 * (PLACEMENT_CAP + 100), "re-sends stay bounded");
});

rowA("F1 (adversarial): a 38-deep shared-children DAG (2^39 placements uncapped) finishes at all", { timeout: 60000 }, async () => {
  // An uncapped walk never finishes (2^39 visits); a capped one is milliseconds. The row's own
  // completion is the measure; node's timeout turns a runaway walk into a named failure.
  const h = await harness();
  h.deliver("probe38", surface("probe38", sharedChain(13)));
  assert.ok(elements(h.root("probe38")).length <= PLACEMENT_CAP + 100, "precondition: the renderer caps a 2^14-placement DAG (else the 38-deep one cannot finish)");
  h.deliver("dag38", surface("dag38", sharedChain(38)));
  assert.ok(elements(h.root("dag38")).length <= PLACEMENT_CAP + 100);
});

rowA("F1 (the too-strict side): surfaces under the placement bound render fully", async () => {
  const h = await harness();
  // 1) a shared DAG of 1,023 placements: 9 Columns doubling, 512 leaf placements
  h.deliver("dag9", surface("dag9", sharedChain(9)));
  assert.equal(readableChunks(h.root("dag9")).filter((t) => t === "leaf").length, 512, "every leaf placement of a 1,023-placement DAG renders");
  // 2) a flat Column of 1,998 Texts: 1,999 placements with the root, under the cap of 2,000
  const kids = Array.from({ length: 1998 }, (_, k) => ({ id: `w${k}`, component: "Text", text: `word ${k}` }));
  h.deliver("flat", surface("flat", [{ id: "root", component: "Column", children: kids.map((k) => k.id) }, ...kids]));
  assert.deepEqual(readableChunks(h.root("flat")), kids.map((k) => k.text), "all 1,998 children render, in order");
  // 3) the largest fixture payload (most expected texts) renders every string
  const largest = [...SURFACES].sort((a, b) => b.expected_texts.length - a.expected_texts.length)[0];
  const chunks = readableChunks(h.deliverFixture(largest));
  assert.ok(isSubsequence(largest.expected_texts, chunks), `${largest.key} renders all ${largest.expected_texts.length} strings`);
  const lines = h.logs.filter((l) => !l.startsWith("[bridge]") && /2[, ]?000|cap|bound|placement/i.test(l));
  assert.deepEqual(lines, [], "no cap line for surfaces under the bound");
});

rowA("T3: a generated label id never equals a model component id (adversarial ids shaped like generated ones)", async () => {
  const h = await harness();
  // The model may name a component anything; these ids cover the a2-<n>-<m> shape and plain words.
  const decoys = [];
  for (let n = 0; n <= 400; n++) for (let m = 0; m <= 8; m++) decoys.push({ id: `a2-${n}-${m}`, component: "Text", text: "decoy" });
  for (const w of ["label", "lbl", "name", "tf-label", "f1-label", "a2-label", "a2ui-label"]) decoys.push({ id: w, component: "Text", text: "decoy" });
  const comps9 = [
    { id: "root", component: "Column", children: ["r1", "f2", "cb", "sl", "dt", "cp"] },
    { id: "r1", component: "Row", children: ["l1", "f1"] },
    { id: "l1", component: "Text", text: "From:" },
    { id: "f1", component: "TextField", label: "Origin city" },
    { id: "f2", component: "TextField", label: "Destination" },
    { id: "cb", component: "CheckBox", label: "Agree", value: false },
    { id: "sl", component: "Slider", label: "Volume", min: 0, max: 5, value: 1 },
    { id: "dt", component: "DateTimeInput", label: "When", enableDate: true, value: "" },
    { id: "cp", component: "ChoicePicker", label: "Size", options: [{ label: "S", value: "s" }], value: [] },
    ...decoys,
  ];
  const model = new Set(comps9.map((c) => c.id));
  for (let k = 0; k < 3; k++) h.deliver(`ids${k}`, surface(`ids${k}`, comps9));
  const domIds = elements(h.dom.document.body).map((e) => e.getAttribute("id")).filter(Boolean);
  assert.ok(domIds.length >= 1, "the labelled inputs carry generated ids (aria-labelledby / for targets)");
  const clash = domIds.filter((id) => model.has(id));
  assert.deepEqual(clash, [], "no DOM id equals a model component id");
  assert.equal(new Set(domIds).size, domIds.length, "and DOM ids are unique on the page");
});

rowA("T3: an unknown Icon name renders nothing (no svg, no path)", async () => {
  const h = await harness();
  h.deliver("ico", surface("ico", [
    { id: "root", component: "Row", children: ["a", "x", "p", "b"] },
    { id: "a", component: "Text", text: "before" },
    { id: "x", component: "Icon", name: "notARealIcon" },
    { id: "p", component: "Icon", name: { path: "/icon" } },
    { id: "b", component: "Text", text: "after" },
  ], { model: { icon: "alsoNotAnIcon" } }));
  const root = h.root("ico");
  const svgish = elements(root).filter((e) => e.namespaceURI === h.dom.SVG_NS);
  assert.deepEqual(svgish.map((e) => e.localName), [], "no svg and no path for an unknown name");
  for (const cid of ["x", "p"]) {
    const el = comp(root, cid);
    assert.ok(el === null || (elements(el).length === 0 && el.textContent === ""), `${cid}: nothing rendered (not even its name as text)`);
  }
  assert.deepEqual(readableChunks(root), ["before", "after"]);
});

rowA("T3: a new model literal wins for ChoicePicker (selection re-keyed) and CheckBox (value re-set)", async () => {
  const h = await harness();
  const comps3 = (pick, tick) => [
    { id: "root", component: "Column", children: ["cp", "cb"] },
    { id: "cp", component: "ChoicePicker", label: "Size", options: [{ label: "A", value: "a" }, { label: "B", value: "b" }, { label: "C", value: "c" }], value: pick },
    { id: "cb", component: "CheckBox", label: "Agree", value: tick },
  ];
  h.deliver("lit", surface("lit", comps3(["a"], false)));
  const root = h.root("lit");
  const chips = chipsOf(comp(root, "cp"));
  const cb = inputsOf(comp(root, "cb")).find((e) => e.type === "checkbox");
  userClick(h.dom, chips[1]); // the user picks B
  userClick(h.dom, cb); // ticks ...
  userClick(h.dom, cb); // ... and unticks: user state false, touched
  h.deliver("lit", [COMPONENTS("lit", comps3(["a"], false))]);
  assert.deepEqual(chips.map(chipState), [false, true, false], "unchanged literals keep the user's pick (A8)");
  h.deliver("lit", [COMPONENTS("lit", comps3(["c"], true))]);
  const now = chipsOf(comp(root, "cp"));
  assert.deepEqual(now.map(chipState), [false, false, true], "the model's new value [c] replaces the user's pick");
  const cbNow = inputsOf(comp(root, "cb")).find((e) => e.type === "checkbox");
  assert.equal(cbNow.checked, true, "the model's new value true re-sets the user's untick");
});

rowA("T3: a same-tag kind change (Column -> List) replaces the node", async () => {
  const h = await harness();
  const kids = [{ id: "a", component: "Text", text: "one" }, { id: "b", component: "Text", text: "two" }];
  h.deliver("tag", surface("tag", [{ id: "root", component: "Column", children: ["x", "tail"] }, { id: "x", component: "Column", children: ["a", "b"] }, { id: "tail", component: "Text", text: "tail" }, ...kids]));
  const root = h.root("tag");
  const x = comp(root, "x");
  const tail = comp(root, "tail");
  h.deliver("tag", [COMPONENTS("tag", [{ id: "root", component: "Column", children: ["x", "tail"] }, { id: "x", component: "List", children: ["a", "b"] }, { id: "tail", component: "Text", text: "tail" }, ...kids])]);
  assert.notEqual(comp(root, "x"), x, "a new node for the new kind, though both may be the same tag");
  assert.equal(comp(root, "x").getAttribute("data-a2-kind"), "List");
  assert.equal(x.isConnected, false, "the Column's node is detached");
  assert.equal(comp(root, "tail"), tail, "the sibling keeps its node");
  assert.deepEqual(readableChunks(root), ["one", "two", "tail"]);
});

rowA("T3: tab arrow keys move focus to the next tab, not only the selection", async () => {
  const h = await harness();
  h.deliver("tf", surface("tf", [{ id: "root", component: "Tabs", tabs: [{ title: "One", child: "p1" }, { title: "Two", child: "p2" }, { title: "Three", child: "p3" }] }, { id: "p1", component: "Text", text: "first" }, { id: "p2", component: "Text", text: "second" }, { id: "p3", component: "Text", text: "third" }]));
  const tabs = () => elements(h.root("tf")).filter((e) => e.getAttribute("role") === "tab");
  h.dom.quiet(() => tabs()[0].focus());
  h.dom.quiet(() => h.dom.document.activeElement.dispatchEvent(new h.dom.FakeEvent("keydown", { key: "ArrowRight" })));
  assert.equal(h.dom.document.activeElement, tabs()[1], "ArrowRight focuses the second tab");
  h.dom.quiet(() => h.dom.document.activeElement.dispatchEvent(new h.dom.FakeEvent("keydown", { key: "ArrowRight" })));
  assert.equal(h.dom.document.activeElement, tabs()[2], "and the third");
  h.dom.quiet(() => h.dom.document.activeElement.dispatchEvent(new h.dom.FakeEvent("keydown", { key: "ArrowLeft" })));
  assert.equal(h.dom.document.activeElement, tabs()[1], "ArrowLeft moves focus back");
});

// =========================================================================================================
// The double itself: no DOM name the renderer used went unmodelled
// =========================================================================================================

rowA("fake DOM: every DOM name the renderer read over the whole fixture is one the double models", async () => {
  const h = await harness();
  for (const s of SURFACES) h.deliverFixture(s);
  assert.deepEqual([...h.dom.rec.gaps], [], "the renderer read a DOM name the double does not model: extend the double in this module, do not change the renderer");
});

// =========================================================================================================
// A13: scope (negative) — nothing else moves in group A
// =========================================================================================================

/** The commit group A branched from (docs/tasks/T-0164.md: main at fc684b82 + the task file, 06a7acf7). */
const A13_BASE = "06a7acf7027de6242524c63ba3b58d1fd57dde93";
/** Group A's merged tip (M^2). null until the merge; whoever merges group A pins it here (rules/tests.md). */
const A13_TIP = "8c049dac813fa4633ebee9b876cfcbf5ced55c9f"; // M^2 of "Merge T-0164 group A impl" (set at the merge)
const A13_PATHS = ["hud/src/ui/bridge.js", "hud/src/ui/action.js", "hud/src/ui/fence.js", "hud/src/ui/a2ui.js", "hud/src/ui/index.js", "hud/src/ui/card.css", "hud/src/ui/style.css", "tests/hud/ui_bridge.test.mjs", "tests/hud/ui_action.test.mjs", "tests/hud/ui_fence.test.mjs"];
const git = (...args) => execFileSync("git", ["-C", REPO, ...args], { maxBuffer: 64 << 20 });

test("A13: bridge/action/fence/a2ui/index/card.css/style.css, hud/vendor/** and the three ui_* node modules are byte-identical to the base", (t) => {
  try {
    git("cat-file", "-e", `${A13_BASE}^{commit}`);
  } catch {
    t.skip(`the base commit ${A13_BASE} is not in this clone`);
    return;
  }
  const bActivated = /\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i.test(T0164_STATUS);
  if (A13_TIP === null && bActivated) {
    assert.fail("A13: group B is activated but A13_TIP is still null: pin group A's merge tip (M^2) in tests/hud/ui_render.test.mjs so this row compares BASE..TIP instead of a moving tree");
  }
  const vendor = git("ls-tree", "-r", "--name-only", A13_BASE, "--", "hud/vendor").toString().split("\n").filter(Boolean);
  assert.ok(vendor.length >= 1, "the base has hud/vendor/ files, so the vendor half is not vacuous");
  for (const path of [...A13_PATHS, ...vendor]) {
    const base = git("show", `${A13_BASE}:${path}`);
    const tip = A13_TIP ? git("show", `${A13_TIP}:${path}`) : existsSync(join(REPO, path)) ? readFileSync(join(REPO, path)) : null;
    assert.ok(tip !== null, `A13: ${path} still exists`);
    assert.ok(Buffer.compare(base, tip) === 0, `A13: ${path} differs from ${A13_BASE.slice(0, 8)} (group A adds files only)`);
  }
});
