// Our own A2UI v0.9.1 renderer: a light-DOM walker of the component tree `UiGate` validated and
// `hud/src/protocol.js` decoded (T-0164 group A; the look is the approved "option A" mock).
//
// One view per surface. `view.render()` walks the surface's components from `root` and *morphs*
// the card in place: every component id keeps its DOM node across canvases while its kind (and
// the shape its parent gives it) is unchanged, and every write is skipped when the value is
// already there, so re-applying an identical canvas mutates nothing (A7). The user's state —
// typed text, picked chips, ticks, sliders, dates, the selected tab — lives on the component's
// record and survives later canvases until the model changes that component's own value (A8).
//
// The honesty rules this module exists for:
//   * the model's words enter the DOM through `textContent`, `value` and `placeholder` only;
//     nothing here parses markup, and markdown stays literal (A4, H-T164-4);
//   * nothing is synthesised: no words, ticks, numbers or "Loading…" the model did not write (A5;
//     T-0184 H-T184-1 admits four derived numbers on the v2 kinds only, see ./cards_v2.js);
//   * no element or attribute that can fetch anything: Image, Video and AudioPlayer render
//     nothing (A3, A11, H-T164-3); icons are a small built-in inline-SVG set;
//   * dangling references, cycles and over-deep trees terminate and render nothing (A6).
//
// No DOM access at module top level: `node --test tests/hud` imports this file with no document
// (T-0024 criterion 2). The document arrives with `createSurfaceView({ document })`.
//
// T-0184 B: a view made with `v2: true` (its createSurface named the v2 catalog) also draws the
// seven v2 kinds of ./cards_v2.js; on a basic surface those kinds render nothing and log once
// each (B2). Every basic kind renders the same on both.

import { V2_KINDS, createV2Builders } from "./cards_v2.js";

/** The 15 kinds the prompt advertises (`ui_instructions.ADVERTISED_ORDER`), all drawn here. */
export const RENDERED_KINDS = Object.freeze([
  "Text",
  "Column",
  "Row",
  "List",
  "Card",
  "Divider",
  "Icon",
  "Tabs",
  "Modal",
  "Button",
  "CheckBox",
  "ChoicePicker",
  "TextField",
  "Slider",
  "DateTimeInput",
]);

/** Deeper than this from the card's body renders nothing (A6; the census never nests past ~8). */
export const MAX_DEPTH = 40;

/** At most this many component views per render pass; shared children count once per placement. */
export const MAX_PLACEMENTS = 2000;

const TEXT_VARIANTS = new Set(["h1", "h2", "h3", "h4", "h5", "caption", "body"]);
const FIELD_INPUTS = new Set(["TextField", "DateTimeInput", "Slider", "ChoicePicker"]);
const INPUT_KINDS = new Set(["TextField", "DateTimeInput", "ChoicePicker", "Slider", "CheckBox"]);
const FLEX = new Set(["start", "center", "end", "stretch", "spaceBetween", "spaceAround", "spaceEvenly"]);
const V2_KIND_SET = new Set(V2_KINDS);

/** The SVG namespace: an XML identifier, never fetched (the egress scan admits it here only). */
const SVG_NS = "http://www.w3.org/2000/svg";

/** A small inline-SVG set for catalog icon names (implementation note 9); others render nothing. */
const ICONS = {
  add: "M12 5v14M5 12h14",
  arrowBack: "M19 12H5M11 18l-6-6 6-6",
  arrowForward: "M5 12h14M13 6l6 6-6 6",
  calendarToday: "M4 7h16v13H4zM4 11h16M8 4v4M16 4v4",
  call: "M5 4h4l2 5-2.5 1.5a11 11 0 0 0 5 5L15 13l5 2v4a1 1 0 0 1-1 1A16 16 0 0 1 4 5a1 1 0 0 1 1-1z",
  check: "M5 12.5l4.5 4.5L19 7.5",
  close: "M6 6l12 12M18 6L6 18",
  delete: "M5 7h14M10 11v6M14 11v6M7 7l1 13h8l1-13M9 7V4h6v3",
  edit: "M4 20h4L19 9l-4-4L4 16zM13 7l4 4",
  error: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 7v6M12 16h.01",
  event: "M4 7h16v13H4zM4 11h16M8 4v4M16 4v4M12 15h4v3h-4z",
  favorite: "M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z",
  help: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6v.6M12 17h.01",
  home: "M4 11l8-7 8 7M6 10v10h12V10",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 11v5M12 8h.01",
  locationOn: "M12 21s-6-5.5-6-11a6 6 0 0 1 12 0c0 5.5-6 11-6 11zM12 12.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5z",
  lock: "M6 11h12v9H6zM8 11V8a4 4 0 0 1 8 0v3",
  mail: "M4 6h16v12H4zM4 7l8 6 8-6",
  menu: "M4 7h16M4 12h16M4 17h16",
  person: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM5 20a7 7 0 0 1 14 0",
  phone: "M5 4h4l2 5-2.5 1.5a11 11 0 0 0 5 5L15 13l5 2v4a1 1 0 0 1-1 1A16 16 0 0 1 4 5a1 1 0 0 1 1-1z",
  refresh: "M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7",
  search: "M10.5 17a6.5 6.5 0 1 0 0-13 6.5 6.5 0 0 0 0 13zM20 20l-4.8-4.8",
  send: "M4 12l16-8-6 16-2-7z",
  settings: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1",
  shoppingCart: "M3 4h2l2.5 11h10L20 8H6.5M9 20h.01M17 20h.01",
  star: "M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8-5.2-2.7-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z",
  warning: "M12 4l9 16H3zM12 10v4M12 17h.01",
};

// --- values ---------------------------------------------------------------------------------

const isObject = (v) => Boolean(v) && typeof v === "object" && !Array.isArray(v);
/** A DataBinding: `{ path }`. */
const isPath = (v) => isObject(v) && typeof v.path === "string" && !("call" in v);
/** A FunctionCall value: `{ call, args? }` — not evaluated in v0.1 (H-T164-6). */
const isCall = (v) => isObject(v) && typeof v.call === "string";
/** A template ChildList: `{ componentId, path }` — not expanded in v0.1 (H-T164-6). */
const isTemplate = (v) => isObject(v) && typeof v.componentId === "string";

function childIds(component) {
  const kids = component.children;
  return Array.isArray(kids) ? kids.filter((k) => typeof k === "string") : [];
}

function flexClass(prefix, value) {
  return typeof value === "string" && FLEX.has(value) ? ` ga-${prefix}-${value}` : "";
}

// --- diffed DOM writes (A7: never write a value that is already there) ------------------------

function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

function setAttr(el, name, value) {
  if (value === null || value === undefined) {
    if (el.hasAttribute(name)) el.removeAttribute(name);
    return;
  }
  const v = String(value);
  if (el.getAttribute(name) !== v) el.setAttribute(name, v);
}

function setClass(el, cls) {
  setAttr(el, "class", cls);
}

function setProp(el, name, value) {
  if (el[name] !== value) el[name] = value;
}

/** Make `parent`'s children exactly `nodes`, in order, touching only what differs. */
function syncChildren(parent, nodes) {
  let cur = parent.firstChild;
  for (const node of nodes) {
    if (cur === node) {
      cur = cur.nextSibling;
      continue;
    }
    parent.insertBefore(node, cur);
  }
  while (cur) {
    const next = cur.nextSibling;
    parent.removeChild(cur);
    cur = next;
  }
}

// --- the view ---------------------------------------------------------------------------------

/** Views made on this page: a prefix that keeps generated label ids unique across surfaces. */
let views = 0;

/**
 * createSurfaceView({ document, surfaceId, components, read, write, dispatch, log, v2 }) -> view
 *
 *   components()           the surface's components: Map(id -> component object)
 *   read(path)             a data-model value (JSON pointer), or undefined
 *   write(path, value)     a user edit of a path-bound input (the engine owns the model)
 *   dispatch(action, id)   a Button's `action` and its component id (the engine resolves it)
 *   v2                     the surface's catalog is the v2 one: the v2 kinds are drawn too
 *
 *   view.root              the card root `[data-a2ui-root]`, the same node for the view's life
 *   view.render()          morph the card to the current components and data model
 *   view.destroy()         detach the root and forget every record
 */
export function createSurfaceView({
  document: doc,
  surfaceId = "",
  components = () => new Map(),
  read = () => undefined,
  write = () => {},
  dispatch = () => {},
  log = () => {},
  v2 = false,
} = {}) {
  const viewSeq = ++views;
  const logged = new Set();
  const logOnce = (key, line) => {
    if (logged.has(key)) return;
    logged.add(key);
    try {
      log(`a2ui surface ${surfaceId}: ${line}`);
    } catch {
      // a logger that throws must not take the card down
    }
  };

  const records = new Map(); // key -> record
  let comps = new Map();
  let pass = null; // { used: Set, placed: Map }

  const root = doc.createElement("div");
  root.setAttribute("data-a2ui-root", "");
  root.setAttribute("class", "ga-card");
  const bodyCol = doc.createElement("div");
  bodyCol.setAttribute("class", "ga-col");

  const el = (tag) => doc.createElement(tag);

  /** A string-valued prop: literal, `{path}`, or (unevaluated) FunctionCall -> "". */
  function str(value) {
    if (typeof value === "string") return value;
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    if (isCall(value)) {
      logOnce("FunctionCall", "a FunctionCall value is not evaluated in v0.1 and renders nothing");
      return "";
    }
    if (isPath(value)) {
      const x = read(value.path);
      if (x === undefined || x === null) return "";
      return typeof x === "object" ? JSON.stringify(x) : String(x);
    }
    return "";
  }

  /** The raw value of a prop: literal, or the data-model value at `{path}`. */
  function val(value) {
    if (isPath(value)) return read(value.path);
    if (isCall(value)) {
      logOnce("FunctionCall", "a FunctionCall value is not evaluated in v0.1 and renders nothing");
      return undefined;
    }
    return value;
  }

  function rerender() {
    try {
      render();
    } catch (err) {
      logOnce("render-error", `render failed: ${(err && err.message) || err}`);
    }
  }

  /** Render component `id` under `ctx`; -> its record, or null when it renders nothing. */
  function renderId(id, ctx) {
    if (typeof id !== "string") return null;
    const c = comps.get(id);
    if (!c) {
      logOnce(`dangling:${id}`, `child "${id}" is not defined by any component and renders nothing`);
      return null;
    }
    if (ctx.ancestors.has(id)) {
      logOnce("cycle", `component "${id}" refers to its own ancestor; the cycle is cut there`);
      return null;
    }
    if (ctx.depth > MAX_DEPTH) {
      logOnce("depth", `the tree is deeper than ${MAX_DEPTH}; components below that render nothing`);
      return null;
    }
    const kind = c.component;
    if (V2_KIND_SET.has(kind) && !V2_BUILDERS) {
      logOnce(`v2kind:${kind}`, `component kind ${JSON.stringify(kind)} is a v2 card kind, which this basic-catalog surface does not draw; it renders nothing`);
      return null;
    }
    const builder = Object.prototype.hasOwnProperty.call(BUILDERS, kind)
      ? BUILDERS[kind]
      : V2_BUILDERS && Object.prototype.hasOwnProperty.call(V2_BUILDERS, kind)
        ? V2_BUILDERS[kind]
        : null;
    if (!builder) {
      logOnce(`kind:${kind}`, `component kind ${JSON.stringify(kind)} is not drawn by this page and renders nothing`);
      return null;
    }
    const skip = builder.skip ? builder.skip(c) : null;
    if (skip) {
      logOnce(skip[0], skip[1]);
      return null;
    }

    // A component shared by several parents is placed once per reference, so a chain of shared
    // children grows as 2^depth; the pass stops placing past the bound (A6, review F1).
    if (pass.count >= MAX_PLACEMENTS) {
      logOnce("placements", `the card places more than ${MAX_PLACEMENTS} components; the rest render nothing`);
      return null;
    }
    pass.count += 1;
    const n = pass.placed.get(id) || 0;
    pass.placed.set(id, n + 1);
    const key = n === 0 ? id : `${id}\u0000${n}`;
    const shape = builder.shape(c, ctx);
    let rec = records.get(key);
    if (!rec || rec.kind !== kind || rec.shape !== shape) {
      rec = { key, id, kind, shape, el: null, state: {}, comp: c, ctx };
      rec.el = builder.create(rec, c, ctx);
      rec.el.setAttribute("data-a2-id", id);
      rec.el.setAttribute("data-a2-kind", kind);
      records.set(key, rec);
    }
    rec.comp = c;
    rec.ctx = ctx;
    pass.used.add(key);
    ctx.ancestors.add(id);
    try {
      builder.update(rec, c, ctx);
    } finally {
      ctx.ancestors.delete(id);
    }
    return rec;
  }

  const child = (ctx, extra = {}) => ({
    depth: ctx.depth + 1,
    ancestors: ctx.ancestors,
    role: null,
    labelled: false,
    label: "",
    labelId: null,
    trigger: null,
    ...extra,
  });

  let domIds = 0;
  const nextDomId = () => {
    let id;
    do id = `a2-label-${viewSeq}-${++domIds}`;
    while (comps.has(id));
    return id;
  };

  /** Render `ids` as children of `parent` (in order; missing ones are skipped). */
  function renderInto(parent, ids, ctx) {
    const nodes = [];
    for (const id of ids) {
      const rec = renderId(id, child(ctx));
      if (rec) nodes.push(rec.el);
    }
    syncChildren(parent, nodes);
  }

  /**
   * Every input is one shell, `[label span, control]`, whether or not a Row labels it: the
   * shell must not change when the Row's mode does (A7). `tag` is `label` for native inputs.
   */
  function fieldParts(rec, tag, control) {
    const wrap = el(tag);
    const span = el("span");
    span.setAttribute("class", "ga-flabel");
    wrap.appendChild(span);
    wrap.appendChild(control);
    rec.state.labelEl = span;
    rec.state.control = control;
    return wrap;
  }

  /**
   * The input's accessible name and its own label, from the model's words only (A12):
   *   * labelled by its Row's Text: `aria-labelledby` that Text; its own label (if any) is shown
   *     as the placeholder (`asPlaceholder`, TextField) or in the shell's span;
   *   * otherwise: its own label in the span above the control, and `aria-label`.
   */
  function updateField(rec, ctx, label, asPlaceholder) {
    const { labelEl: span, control } = rec.state;
    setClass(rec.el, ctx.labelled ? "ga-control" : "ga-field");
    const shown = ctx.labelled && asPlaceholder ? "" : label;
    setText(span, shown);
    setAttr(span, "hidden", shown ? null : "");
    if (asPlaceholder) setAttr(control, "placeholder", ctx.labelled && label ? label : null);
    setAttr(control, "aria-labelledby", ctx.labelled ? ctx.labelId : null);
    setAttr(control, "aria-label", !ctx.labelled && label ? label : null);
  }

  // --- builders: shape() decides when a node must be replaced; update() is diffed ------------

  // A Text's element depends only on its variant (and on sitting in a Button), never on the
  // role its Row gives it: a Row's mode can change as its children stream in, and the Text node
  // must survive that (A7).
  const textTag = (c, ctx) => {
    if (ctx.role === "inline") return "span";
    const v = TEXT_VARIANTS.has(c.variant) ? c.variant : "body";
    return /^h[1-5]$/.test(v) ? v : "p";
  };

  const BUILDERS = {
    Text: {
      skip: (c) => (isCall(c.text) ? ["FunctionCall", "a FunctionCall value is not evaluated in v0.1 and renders nothing"] : null),
      shape: (c, ctx) => textTag(c, ctx),
      create: (rec, c, ctx) => el(textTag(c, ctx)),
      update(rec, c, ctx) {
        const v = TEXT_VARIANTS.has(c.variant) ? c.variant : "body";
        let cls = ctx.role === "flabel" ? "ga-t ga-flabel" : `ga-t ga-${v}`;
        if (ctx.role === "k") cls += " ga-k";
        if (ctx.role === "v") cls += " ga-v";
        setClass(rec.el, cls);
        // A field label is referenced by its input's aria-labelledby (a generated id, never the
        // component id, so two surfaces cannot collide).
        if (ctx.role === "flabel") {
          // never equal to a model id, even one a later canvas adds
          if (!rec.state.domId || comps.has(rec.state.domId)) rec.state.domId = nextDomId();
          setAttr(rec.el, "id", rec.state.domId);
        } else {
          setAttr(rec.el, "id", null);
        }
        setText(rec.el, str(c.text));
      },
    },

    Column: {
      skip: (c) => (isTemplate(c.children) ? ["template", "a template ChildList is not expanded in v0.1 and renders nothing"] : null),
      shape: () => "div",
      create: () => el("div"),
      update(rec, c, ctx) {
        setClass(rec.el, "ga-col" + flexClass("align", c.align) + flexClass("justify", c.justify));
        renderInto(rec.el, childIds(c), ctx);
      },
    },

    Row: {
      skip: (c) => (isTemplate(c.children) ? ["template", "a template ChildList is not expanded in v0.1 and renders nothing"] : null),
      // One element whatever the mode: the mode follows the children's kinds, which stream in.
      shape: () => "div",
      create: () => el("div"),
      update(rec, c, ctx) {
        const mode = rowMode(c);
        const ids = childIds(c);
        if (mode === "kv") {
          setClass(rec.el, "ga-kv");
          const k = renderId(ids[0], child(ctx, { role: "k" }));
          const v = renderId(ids[1], child(ctx, { role: "v" }));
          syncChildren(rec.el, [k, v].filter(Boolean).map((r) => r.el));
          return;
        }
        if (mode === "field") {
          setClass(rec.el, "ga-field");
          const labelComp = comps.get(ids[0]);
          const label = labelComp && !isCall(labelComp.text) ? str(labelComp.text) : "";
          const l = renderId(ids[0], child(ctx, { role: "flabel" }));
          const labelId = l ? l.state.domId : null;
          const input = renderId(ids[1], child(ctx, { labelled: Boolean(labelId), label, labelId }));
          syncChildren(rec.el, [l, input].filter(Boolean).map((r) => r.el));
          return;
        }
        if (mode === "pair") {
          setClass(rec.el, "ga-pair");
        } else {
          setClass(rec.el, "ga-row" + flexClass("align", c.align) + flexClass("justify", c.justify));
        }
        renderInto(rec.el, ids, ctx);
      },
    },

    List: {
      skip: (c) => (isTemplate(c.children) ? ["template", "a template ChildList is not expanded in v0.1 and renders nothing"] : null),
      shape: () => "div",
      create(rec) {
        const list = el("div");
        list.setAttribute("role", "list");
        rec.state.items = new Map(); // child key -> listitem wrapper
        return list;
      },
      update(rec, c, ctx) {
        setClass(rec.el, "ga-list" + (c.direction === "horizontal" ? " h" : "") + flexClass("align", c.align));
        const items = rec.state.items;
        const seen = new Set();
        const nodes = [];
        for (const id of childIds(c)) {
          const r = renderId(id, child(ctx));
          if (!r) continue;
          let li = items.get(r.key);
          if (!li) {
            li = el("div");
            li.setAttribute("class", "ga-li");
            li.setAttribute("role", "listitem");
            items.set(r.key, li);
          }
          seen.add(r.key);
          syncChildren(li, [r.el]);
          nodes.push(li);
        }
        for (const k of [...items.keys()]) if (!seen.has(k)) items.delete(k);
        syncChildren(rec.el, nodes);
      },
    },

    Card: {
      shape: () => "div",
      create(rec) {
        const tile = el("div");
        tile.setAttribute("class", "ga-tile");
        const col = el("div");
        col.setAttribute("class", "ga-col");
        rec.state.col = col;
        return tile;
      },
      update(rec, c, ctx) {
        const r = renderId(c.child, child(ctx));
        if (!r) return syncChildren(rec.el, []);
        if (r.kind === "Column") return syncChildren(rec.el, [r.el]);
        syncChildren(rec.state.col, [r.el]);
        syncChildren(rec.el, [rec.state.col]);
      },
    },

    Divider: {
      shape: () => "hr",
      create: () => el("hr"),
      update(rec, c) {
        setClass(rec.el, "ga-hr" + (c.axis === "vertical" ? " v" : ""));
      },
    },

    Icon: {
      skip: (c) => (iconPath(c) ? null : ["icon", "an Icon whose name this page has no picture for renders nothing"]),
      shape: () => "svg",
      // A decorative, wordless inline SVG built with DOM calls (implementation note 9).
      create(rec) {
        const svg = doc.createElementNS(SVG_NS, "svg");
        svg.setAttribute("viewBox", "0 0 24 24");
        svg.setAttribute("class", "ga-icon");
        svg.setAttribute("aria-hidden", "true");
        const path = doc.createElementNS(SVG_NS, "path");
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", "currentColor");
        path.setAttribute("stroke-width", "1.8");
        path.setAttribute("stroke-linecap", "round");
        path.setAttribute("stroke-linejoin", "round");
        svg.appendChild(path);
        rec.state.path = path;
        return svg;
      },
      update(rec, c) {
        setAttr(rec.state.path, "d", iconPath(c));
      },
    },

    Tabs: {
      shape: () => "div",
      create(rec) {
        const wrap = el("div");
        wrap.setAttribute("class", "ga-col ga-tabs");
        const bar = el("div");
        bar.setAttribute("class", "ga-tabs-bar");
        bar.setAttribute("role", "tablist");
        wrap.appendChild(bar);
        rec.state.bar = bar;
        rec.state.tabs = []; // [{ button, panel }]
        rec.state.selected = 0;
        bar.addEventListener("keydown", (event) => onTabKey(rec, event));
        return wrap;
      },
      update(rec, c, ctx) {
        const tabs = Array.isArray(c.tabs) ? c.tabs : [];
        const st = rec.state;
        while (st.tabs.length < tabs.length) {
          const i = st.tabs.length;
          const button = el("button");
          button.setAttribute("type", "button");
          button.setAttribute("class", "ga-tab");
          button.setAttribute("role", "tab");
          button.addEventListener("click", () => selectTab(rec, i, false));
          const panel = el("div");
          panel.setAttribute("class", "ga-tabpanel");
          panel.setAttribute("role", "tabpanel");
          st.tabs.push({ button, panel });
        }
        st.tabs.length = tabs.length;
        if (st.selected >= tabs.length) st.selected = 0;
        tabs.forEach((tab, i) => {
          const { button, panel } = st.tabs[i];
          const on = i === st.selected;
          setText(button, str(tab && tab.title));
          setAttr(button, "aria-selected", on ? "true" : "false");
          setAttr(button, "tabindex", on ? "0" : "-1");
          // Every panel is rendered, so a field in another tab keeps what the user did there; only
          // the selected one is in the card (what is not shown is not the card's text).
          const r = tab ? renderId(tab.child, child(ctx)) : null;
          syncChildren(panel, r ? [r.el] : []);
        });
        syncChildren(st.bar, st.tabs.map((t) => t.button));
        const shown = st.tabs[st.selected];
        syncChildren(rec.el, shown ? [st.bar, shown.panel] : [st.bar]);
      },
    },

    Modal: {
      shape: () => "div",
      create(rec) {
        const wrap = el("div");
        wrap.setAttribute("class", "ga-modal");
        const content = el("div");
        content.setAttribute("class", "ga-modal-content");
        rec.state.content = content;
        rec.state.open = false;
        rec.toggle = () => {
          rec.state.open = !rec.state.open;
          rerender();
        };
        const disclose = el("button");
        disclose.setAttribute("type", "button");
        disclose.setAttribute("class", "ga-disclose");
        disclose.addEventListener("click", () => rec.toggle());
        rec.state.disclose = disclose;
        return wrap;
      },
      update(rec, c, ctx) {
        const st = rec.state;
        const triggerComp = comps.get(c.trigger);
        const isButton = triggerComp && triggerComp.component === "Button";
        const t = renderId(c.trigger, child(ctx, { trigger: isButton ? rec : null }));
        let head = null;
        if (t && isButton) {
          head = t.el;
        } else if (t) {
          syncChildren(st.disclose, [t.el]);
          head = st.disclose;
        }
        if (head) setAttr(head, "aria-expanded", st.open ? "true" : "false");
        // The content is always rendered (its fields keep the user's state) but is in the card
        // only while open: closed content is not shown text.
        const body = renderId(c.content, child(ctx));
        syncChildren(st.content, body ? [body.el] : []);
        const parts = [head, st.open ? st.content : null].filter(Boolean);
        syncChildren(rec.el, parts);
      },
    },

    Button: {
      shape: () => "button",
      create(rec) {
        const b = el("button");
        b.setAttribute("type", "button");
        b.addEventListener("click", () => {
          if (rec.ctx && rec.ctx.trigger) {
            rec.ctx.trigger.toggle();
            return;
          }
          const action = rec.comp && rec.comp.action;
          if (isObject(action) && isObject(action.event) && typeof action.event.name === "string") {
            dispatch(action, rec.id);
          } else if (isObject(action) && "functionCall" in action) {
            logOnce("functionCall-action", "a functionCall action is not evaluated in v0.1; the click sends nothing");
          }
        });
        return b;
      },
      update(rec, c, ctx) {
        const variant = c.variant === "borderless" ? " borderless" : c.variant === "default" ? "" : " primary";
        setClass(rec.el, "ga-btn" + variant);
        const r = renderId(c.child, child(ctx, { role: "inline" }));
        syncChildren(rec.el, r ? [r.el] : []);
      },
    },

    TextField: {
      shape: (c) => (c.variant === "longText" ? "textarea" : "input"),
      create(rec, c) {
        const long = c.variant === "longText";
        const input = el(long ? "textarea" : "input");
        input.setAttribute("class", "ga-input");
        const onEdit = () => {
          if (isPath(rec.comp.value)) {
            write(rec.comp.value.path, input.value);
            rerender();
          }
        };
        input.addEventListener("input", onEdit);
        input.addEventListener("change", onEdit);
        rec.state.input = input;
        return fieldParts(rec, "label", input);
      },
      update(rec, c, ctx) {
        const input = rec.state.input;
        if (c.variant !== "longText") {
          setAttr(input, "type", c.variant === "number" ? "number" : c.variant === "obscured" ? "password" : "text");
        }
        updateField(rec, ctx, str(c.label), true);
        syncValue(rec, input, c.value, (x) => (x === undefined || x === null ? "" : String(x)));
      },
    },

    DateTimeInput: {
      shape: () => "input",
      create(rec) {
        const input = el("input");
        input.setAttribute("class", "ga-input");
        const onEdit = () => {
          if (isPath(rec.comp.value)) {
            write(rec.comp.value.path, input.value);
            rerender();
          }
        };
        input.addEventListener("input", onEdit);
        input.addEventListener("change", onEdit);
        rec.state.input = input;
        return fieldParts(rec, "label", input);
      },
      update(rec, c, ctx) {
        const input = rec.state.input;
        const date = c.enableDate === true;
        const time = c.enableTime === true;
        setAttr(input, "type", date && time ? "datetime-local" : time ? "time" : "date");
        setAttr(input, "min", typeof c.min === "string" ? c.min : null);
        setAttr(input, "max", typeof c.max === "string" ? c.max : null);
        updateField(rec, ctx, str(c.label), false);
        syncValue(rec, input, c.value, (x) => (x === undefined || x === null ? "" : String(x)));
      },
    },

    Slider: {
      shape: () => "input",
      create(rec) {
        const input = el("input");
        input.setAttribute("type", "range");
        input.setAttribute("class", "ga-range");
        const onEdit = () => {
          if (isPath(rec.comp.value)) {
            write(rec.comp.value.path, Number(input.value));
            rerender();
          }
        };
        input.addEventListener("input", onEdit);
        input.addEventListener("change", onEdit);
        rec.state.input = input;
        return fieldParts(rec, "label", input);
      },
      update(rec, c, ctx) {
        const input = rec.state.input;
        const min = typeof c.min === "number" ? c.min : 0;
        const max = typeof c.max === "number" ? c.max : 100;
        setAttr(input, "min", String(min));
        setAttr(input, "max", String(max));
        updateField(rec, ctx, str(c.label), false);
        syncValue(rec, input, c.value, (x) => (typeof x === "number" && Number.isFinite(x) ? String(x) : String(min)));
      },
    },

    CheckBox: {
      shape: () => "label",
      create(rec) {
        const wrap = el("label");
        wrap.setAttribute("class", "ga-check");
        const box = el("input");
        box.setAttribute("type", "checkbox");
        const span = el("span");
        wrap.appendChild(box);
        wrap.appendChild(span);
        const onEdit = () => {
          if (isPath(rec.comp.value)) {
            write(rec.comp.value.path, Boolean(box.checked));
            rerender();
          }
        };
        box.addEventListener("change", onEdit);
        rec.state.box = box;
        rec.state.span = span;
        return wrap;
      },
      update(rec, c) {
        setText(rec.state.span, str(c.label));
        const box = rec.state.box;
        if (isPath(c.value)) {
          setProp(box, "checked", Boolean(read(c.value.path)));
          return;
        }
        // A literal: the tick is the user's until the model writes a different value (A8).
        const literal = Boolean(val(c.value));
        if (rec.state.model !== literal) {
          rec.state.model = literal;
          setProp(box, "checked", literal);
        }
      },
    },

    ChoicePicker: {
      shape: (c) => (c.variant === "multipleSelection" ? "multi" : "one"),
      create(rec, c) {
        const box = el("div");
        box.setAttribute("class", "ga-chips");
        box.setAttribute("role", c.variant === "multipleSelection" ? "group" : "radiogroup");
        rec.state.box = box;
        rec.state.chips = [];
        rec.state.selected = [];
        rec.state.model = undefined;
        box.addEventListener("keydown", (event) => onChipKey(rec, event));
        return fieldParts(rec, "div", box);
      },
      update(rec, c, ctx) {
        const st = rec.state;
        const multi = c.variant === "multipleSelection";
        const options = Array.isArray(c.options) ? c.options : [];
        updateField(rec, ctx, str(c.label), false);
        const selected = chosen(rec, c);
        while (st.chips.length < options.length) {
          const i = st.chips.length;
          const b = el("button");
          b.setAttribute("type", "button");
          b.setAttribute("class", "ga-chip");
          b.addEventListener("click", () => pickChip(rec, i));
          st.chips.push(b);
        }
        st.chips.length = options.length;
        let focusable = options.findIndex((o) => o && selected.includes(String(o.value)));
        if (focusable < 0) focusable = 0;
        options.forEach((o, i) => {
          const b = st.chips[i];
          const on = Boolean(o) && selected.includes(String(o.value));
          setText(b, str(o && o.label));
          if (multi) {
            setAttr(b, "aria-pressed", on ? "true" : "false");
          } else {
            setAttr(b, "role", "radio");
            setAttr(b, "aria-checked", on ? "true" : "false");
            setAttr(b, "tabindex", i === focusable ? "0" : "-1");
          }
        });
        syncChildren(st.box, st.chips);
      },
    },
  };

  // The v2 kinds, only on a v2 surface (B2). They are leaves: no child id is rendered from them.
  const V2_BUILDERS = v2 ? createV2Builders({ document: doc, svg: (tag) => doc.createElementNS(SVG_NS, tag), setText, setAttr, setClass, syncChildren, dispatch }) : null;

  /** The path data for a catalog icon name this page draws, or null (an `svgPath` is not drawn in v0.1). */
  function iconPath(c) {
    const name = isPath(c.name) ? read(c.name.path) : c.name;
    return typeof name === "string" && Object.prototype.hasOwnProperty.call(ICONS, name) ? ICONS[name] : null;
  }

  function rowMode(c) {
    const ids = childIds(c);
    const kinds = ids.map((id) => {
      const k = comps.get(id);
      return k ? k.component : null;
    });
    const noJustify = c.justify === undefined;
    if (noJustify && ids.length === 2 && kinds[0] === "Text" && kinds[1] === "Text") return "kv";
    if (noJustify && ids.length === 2 && kinds[0] === "Text" && FIELD_INPUTS.has(kinds[1])) return "field";
    if (ids.length >= 2 && kinds.every((k) => INPUT_KINDS.has(k))) return "pair";
    return "row";
  }

  /**
   * An input's value: a `{path}` is the data model's (a later updateDataModel wins); a literal
   * belongs to the user once rendered, until the model writes a *different* literal (A8).
   */
  function syncValue(rec, input, value, toText) {
    if (isPath(value)) {
      setProp(input, "value", toText(read(value.path)));
      rec.state.model = undefined;
      return;
    }
    const literal = toText(val(value));
    if (rec.state.model !== literal) {
      rec.state.model = literal;
      setProp(input, "value", literal);
    }
  }

  /** The selected option values of a ChoicePicker, as strings. */
  function chosen(rec, c) {
    const toList = (x) => (Array.isArray(x) ? x : x === undefined || x === null ? [] : [x]).map(String);
    if (isPath(c.value)) return toList(read(c.value.path));
    const key = JSON.stringify(Array.isArray(c.value) ? c.value : null);
    if (rec.state.model !== key) {
      rec.state.model = key;
      rec.state.selected = toList(Array.isArray(c.value) ? c.value : []);
    }
    return rec.state.selected;
  }

  function pickChip(rec, i) {
    const c = rec.comp;
    const options = Array.isArray(c.options) ? c.options : [];
    const o = options[i];
    if (!o) return;
    const v = String(o.value);
    const current = chosen(rec, c);
    let next;
    if (c.variant === "multipleSelection") {
      const set = new Set(current);
      if (set.has(v)) set.delete(v);
      else set.add(v);
      next = options.map((x) => String(x && x.value)).filter((x) => set.has(x));
    } else {
      next = [v];
    }
    if (isPath(c.value)) write(c.value.path, next);
    else rec.state.selected = next;
    rerender();
  }

  function onChipKey(rec, event) {
    const c = rec.comp;
    if (c.variant === "multipleSelection") return;
    const chips = rec.state.chips;
    const at = chips.indexOf(event.target);
    if (at < 0 || chips.length === 0) return;
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    let to = null;
    if (step) to = (at + step + chips.length) % chips.length;
    else if (event.key === "Home") to = 0;
    else if (event.key === "End") to = chips.length - 1;
    if (to === null) return;
    if (typeof event.preventDefault === "function") event.preventDefault();
    pickChip(rec, to);
    if (typeof chips[to].focus === "function") chips[to].focus();
  }

  function selectTab(rec, i, focus) {
    const tabs = rec.state.tabs;
    if (i < 0 || i >= tabs.length) return;
    if (rec.state.selected !== i) {
      rec.state.selected = i;
      rerender();
    }
    if (focus && typeof tabs[i].button.focus === "function") tabs[i].button.focus();
  }

  function onTabKey(rec, event) {
    const tabs = rec.state.tabs;
    const at = tabs.findIndex((t) => t.button === event.target);
    if (at < 0 || tabs.length === 0) return;
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    let to = null;
    if (step) to = (at + step + tabs.length) % tabs.length;
    else if (event.key === "Home") to = 0;
    else if (event.key === "End") to = tabs.length - 1;
    if (to === null) return;
    if (typeof event.preventDefault === "function") event.preventDefault();
    selectTab(rec, to, true);
  }

  /**
   * Root unwrap (H-T164-8): a root Card, or a Column holding exactly one Card, renders into the
   * frame instead of drawing a card inside the card.
   */
  function bodyId() {
    let id = "root";
    const visited = new Set();
    while (!visited.has(id)) {
      visited.add(id);
      const c = comps.get(id);
      if (!c) break;
      if (c.component === "Card" && typeof c.child === "string") {
        id = c.child;
        continue;
      }
      const kids = childIds(c);
      if (c.component === "Column" && !isTemplate(c.children) && kids.length === 1) {
        const only = comps.get(kids[0]);
        if (only && only.component === "Card" && typeof only.child === "string") {
          id = kids[0];
          continue;
        }
      }
      break;
    }
    return id; // a cycle among the unwrapped Cards is cut by renderId's ancestor check
  }

  function render() {
    const next = components();
    comps = next instanceof Map ? next : new Map();
    pass = { used: new Set(), placed: new Map(), count: 0 };
    try {
      if (!comps.has("root")) {
        setAttr(root, "data-a2-empty", "");
        syncChildren(root, []);
      } else {
        setAttr(root, "data-a2-empty", null);
        const id = bodyId();
        const ancestors = new Set();
        const ctx = { depth: 0, ancestors, role: null, labelled: false, label: "", labelId: null, trigger: null };
        const rec = id === null ? null : renderId(id, ctx);
        if (!rec) {
          syncChildren(root, []);
        } else if (rec.kind === "Column") {
          syncChildren(root, [rec.el]);
        } else {
          syncChildren(bodyCol, [rec.el]);
          syncChildren(root, [bodyCol]);
        }
      }
      // An unreferenced node is already detached by its parent's sync. Its record is kept while
      // its component is still defined, because the gate can drop a reference in one message
      // and restore it in the next message of the same canvas (desk t10). Without that, the
      // node would be rebuilt and the user's state in it lost.
      for (const [key, rec] of [...records]) {
        if (pass.used.has(key)) continue;
        const c = comps.get(rec.id);
        if (key !== rec.id || !c || c.component !== rec.kind) records.delete(key);
      }
    } finally {
      pass = null;
    }
  }

  return {
    root,
    render,
    destroy() {
      records.clear();
      const parent = root.parentNode;
      if (parent) parent.removeChild(root);
    },
  };
}
