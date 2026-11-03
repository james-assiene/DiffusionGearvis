// The seven v2 card kinds (T-0184 group B): Stat, StatTiles, Comparison, Bars, Steps, Source and
// Actions, drawn on a surface whose createSurface named the v2 catalog (./a2ui.js). ./render.js
// owns the walk, the records and the morph; this module only builds and updates one leaf
// component's subtree in the approved look (the classes are styled by ./card.css).
//
// The honesty rules (docs/tasks/T-0184.md B4, H-T184-1):
//   * every model string is the whole text of one element (`textContent`), in the model's order;
//   * the only text this module derives is (i) String(value) beside a bar's unit, (ii) the Bars
//     tick numbers, (iii) "<percent>%" of a tile and (iv) the step ordinals 1..n, each with
//     String(n) (never toLocaleString), each a text node (never CSS generated content);
//   * no word is ever written here: no "Recommended", no "Source", no unit the model did not give;
//   * `accessibility` on a v2 kind is never read (group A review hand-off): its strings would be
//     text outside that closed list;
//   * glyphs are inline SVG made with createElementNS, painted by ./card.css classes; nothing here
//     can fetch anything.
//
// Nothing is interactive (James 2026-10-28) but Actions: one <button type="button"> per label,
// whose click sends exactly what a gate-made Button with that label sends (T-0164 B4).
//
// No DOM access at module top level: the document arrives through `createV2Builders(h)`.

/** The seven kinds, in the catalog's order. */
export const V2_KINDS = Object.freeze(["Stat", "StatTiles", "Comparison", "Bars", "Steps", "Source", "Actions"]);

/** The event a gate-made Button sends (component_repair.py's RB rule): Actions sends the same. */
const GATE_CLICKED = "__gate.clicked";

const isObject = (v) => Boolean(v) && typeof v === "object" && !Array.isArray(v);
/** A v2 string is the model's own literal; anything else shows nothing. */
const s = (v) => (typeof v === "string" ? v : "");
const has = (v) => typeof v === "string" && v.length > 0;
const finite = (v) => typeof v === "number" && Number.isFinite(v);
const list = (v) => (Array.isArray(v) ? v.map((x) => (isObject(x) ? x : {})) : []);
/** A number written for a style value: short, never NaN or an exponent. */
const num = (x) => (finite(x) ? String(Math.round(x * 1e6) / 1e6) : "0");

// --- glyphs ------------------------------------------------------------------------------------

const CLOUD_LOW = "M20 54h34a12 12 0 0 0 0-24 16 16 0 0 0-30.5 5A10 10 0 0 0 20 54z";
const CLOUD_HIGH = "M20 44h34a12 12 0 0 0 0-24 16 16 0 0 0-30.5 5A10 10 0 0 0 20 44z";

/** Each Stat icon (A2's nine names) as [tag, attributes, class] parts in a 72x72 box. */
const STAT_GLYPHS = {
  sun: [
    ["circle", { cx: "36", cy: "36", r: "13" }, "ga-g-sun"],
    ["path", { d: "M36 9v7M36 56v7M9 36h7M56 36h7M17 17l5 5M50 50l5 5M17 55l5-5M50 22l5-5" }, "ga-g-ray"],
  ],
  partlyCloudy: [
    ["circle", { cx: "28", cy: "28", r: "12" }, "ga-g-sun"],
    ["path", { d: "M22 50h30a10 10 0 0 0 0-20 14 14 0 0 0-26.5 4.5A8 8 0 0 0 22 50z" }, "ga-g-cloud"],
  ],
  cloud: [["path", { d: CLOUD_LOW }, "ga-g-cloud"]],
  rain: [
    ["path", { d: CLOUD_HIGH }, "ga-g-cloud"],
    ["path", { d: "M28 52l-3 9M39 52l-3 9M50 52l-3 9" }, "ga-g-drop"],
  ],
  storm: [
    ["path", { d: CLOUD_HIGH }, "ga-g-cloud"],
    ["path", { d: "M39 48l-7 11h8l-5 10" }, "ga-g-bolt"],
  ],
  snow: [
    ["path", { d: CLOUD_HIGH }, "ga-g-cloud"],
    ["circle", { cx: "27", cy: "56", r: "2.5" }, "ga-g-flake"],
    ["circle", { cx: "38", cy: "62", r: "2.5" }, "ga-g-flake"],
    ["circle", { cx: "49", cy: "56", r: "2.5" }, "ga-g-flake"],
  ],
  fog: [
    ["path", { d: CLOUD_HIGH }, "ga-g-cloud"],
    ["path", { d: "M14 54h44M20 63h32" }, "ga-g-mist"],
  ],
  wind: [["path", { d: "M10 28h34a8 8 0 1 0-8-8M10 40h46a8 8 0 1 1-8 8M10 52h22" }, "ga-g-mist"]],
  night: [["path", { d: "M44 12a24 24 0 1 0 16 38A20 20 0 0 1 44 12z" }, "ga-g-moon"]],
};

/** The Source chip's glyph (a page), 24x24. */
const SOURCE_GLYPH = [["path", { d: "M5 4h14v16H5zM8.5 9h7M8.5 12.5h7M8.5 16h4" }, "ga-g-line"]];

/** Bars ticks: nice steps (one significant digit 1, 2, 2.5 or 5 times a power of ten). */
const NICE = [1, 2, 2.5, 5];
const MAX_TICKS = 5;
/** The fewest tick labels a drawn scale has (0 and its end); fewer than that draws none (B7a). */
const MIN_TICKS = 2;
/** The least gap between two tick labels, in px, for them to count as side by side (B7a). */
const TICK_GAP = 6;
/** The Bars width (px) at or under which card.css's @container stacks each row (B9a). */
const NARROW_BARS = 340;

/**
 * The scale for a largest value `max` > 0: ticks 0, step, ..., n*step with n*step >= max, at most
 * `limit` (default five) of them, the smallest nice step that fits. -> { end, ticks: [number] },
 * or null when there is no scale (every bar 0).
 */
export function barScale(max, limit = MAX_TICKS) {
  if (!finite(max) || max <= 0) return null;
  const most = Math.max(MIN_TICKS, Math.min(MAX_TICKS, Math.floor(limit)));
  const clean = (x) => Number(x.toPrecision(12));
  for (let e = Math.floor(Math.log10(max)) - 2; e <= Math.floor(Math.log10(max)) + 1; e++) {
    for (const m of NICE) {
      const step = clean(m * 10 ** e);
      const n = Math.ceil(clean(max / step));
      if (n + 1 <= most) {
        const ticks = [];
        for (let k = 0; k <= n; k++) ticks.push(clean(k * step));
        return { end: ticks[ticks.length - 1], ticks };
      }
    }
  }
  return { end: max, ticks: [0, max] };
}

/**
 * createV2Builders(h) -> { [kind]: { shape, create, update } }
 *
 *   h.document        the page's document
 *   h.svg(tag)        an element in the SVG namespace (./render.js holds the namespace: the one
 *                     HUD file the egress scan admits it in)
 *   h.setText/setAttr/setClass/syncChildren   ./render.js's diffed writes
 *   h.dispatch(action, componentId)           a click's action, as a Button's (the engine resolves it)
 */
export function createV2Builders(h) {
  const { document: doc, svg: svgEl, setText, setAttr, setClass, syncChildren, dispatch } = h;

  const el = (tag, cls) => {
    const node = doc.createElement(tag);
    if (cls) node.setAttribute("class", cls);
    return node;
  };

  function glyph(parts, viewBox, cls) {
    const svg = svgEl("svg");
    svg.setAttribute("viewBox", viewBox);
    svg.setAttribute("class", cls);
    svg.setAttribute("aria-hidden", "true");
    for (const [tag, attrs, partCls] of parts) {
      const part = svgEl(tag);
      for (const [k, v] of Object.entries(attrs)) part.setAttribute(k, v);
      part.setAttribute("class", partCls);
      svg.appendChild(part);
    }
    return svg;
  }

  /**
   * The drawn width of a Bars scale in px (0 when the value's room takes the whole column), or -1
   * when it is not laid out (detached, hidden, or a page with no layout).
   */
  function scaleWidth(st) {
    const t = st.ticksEl;
    if (!t.isConnected || typeof t.getClientRects !== "function") return -1;
    if (!t.getClientRects().length) return -1;
    const w = t.getBoundingClientRect().width;
    return finite(w) && w > 0 ? w : 0;
  }

  /** Writes a Bars scale of at most `limit` tick labels (0: none) and the fills on it. */
  function drawScale(st, limit) {
    const values = st.values || [];
    let vmax = 0;
    for (const v of values) if (v !== null && v > vmax) vmax = v;
    const scale = vmax > 0 ? barScale(vmax, limit || MAX_TICKS) : null;
    const ticks = scale && limit > 0 ? scale.ticks : [];
    // With no labels drawn the scale's end is not shown: the longest bar fills the column.
    const end = ticks.length ? scale.end : vmax;
    st.rows.forEach((p, k) => {
      const v = values[k];
      const frac = end > 0 && v !== null ? v / end : 0;
      setAttr(p.fill, "style", `width: calc((100% - ${st.room}) * ${num(frac)})`);
    });
    while (st.ticks.length < ticks.length) st.ticks.push(el("span"));
    st.ticks.length = ticks.length;
    ticks.forEach((t, k) => {
      const span = st.ticks[k];
      const edge = k === 0 ? " first" : k === ticks.length - 1 ? " last" : "";
      setClass(span, "ga-tick" + edge);
      setAttr(span, "style", `left: ${num((t / scale.end) * 100)}%`);
      setText(span, String(t));
    });
    syncChildren(st.ticksEl, st.ticks);
  }

  /** True when the drawn tick labels sit side by side inside the scale (B7a). */
  function ticksFit(st) {
    const outer = st.ticksEl.getBoundingClientRect();
    const boxes = st.ticks.map((t) => t.getBoundingClientRect());
    for (let k = 0; k < boxes.length; k++) {
      if (boxes[k].left < outer.left - 0.5 || boxes[k].right > outer.right + 0.5) return false;
      if (k > 0 && boxes[k - 1].right + TICK_GAP > boxes[k].left) return false;
    }
    return true;
  }

  /**
   * Draws the most tick labels (of 5, 4, 3, 2) whose scale's labels do not touch at the drawn
   * width, or none when even 0 and the end cannot sit side by side (B7a, ruling 195866ca).
   */
  function fitTicks(st) {
    st.fitWidth = scaleWidth(st);
    let chosen = 0;
    let drawn = null;
    for (let limit = MAX_TICKS; limit >= MIN_TICKS; limit--) {
      const key = barScale(st.fitMax, limit).ticks.join(",");
      if (key === drawn) continue; // the same ticks as the larger limit: already measured
      drawn = key;
      drawScale(st, limit);
      if (ticksFit(st)) {
        chosen = limit;
        break;
      }
    }
    st.limit = chosen;
    if (!chosen) drawScale(st, 0);
  }

  return {
    Stat: {
      shape: () => "div",
      create(rec) {
        const st = rec.state;
        const root = el("div", "ga-stat");
        st.main = el("div", "ga-stat-main");
        st.label = el("span", "ga-stat-label");
        st.line = el("div", "ga-stat-line");
        st.value = el("span", "ga-stat-value");
        st.detail = el("span", "ga-stat-detail");
        st.icon = null;
        st.glyph = null;
        return root;
      },
      update(rec, c) {
        const st = rec.state;
        setText(st.label, s(c.label));
        setText(st.value, s(c.value));
        setText(st.detail, s(c.detail));
        syncChildren(st.line, has(c.detail) ? [st.value, st.detail] : [st.value]);
        syncChildren(st.main, [st.label, st.line]);
        const icon = typeof c.icon === "string" && Object.prototype.hasOwnProperty.call(STAT_GLYPHS, c.icon) ? c.icon : null;
        if (icon !== st.icon) {
          st.icon = icon;
          st.glyph = icon ? glyph(STAT_GLYPHS[icon], "0 0 72 72", "ga-stat-glyph") : null;
        }
        syncChildren(rec.el, st.glyph ? [st.main, st.glyph] : [st.main]);
      },
    },

    StatTiles: {
      shape: () => "div",
      create(rec) {
        rec.state.tiles = [];
        return el("div", "ga-tiles");
      },
      update(rec, c) {
        const tiles = list(c.tiles);
        const parts = rec.state.tiles;
        while (parts.length < tiles.length) {
          const p = {
            box: el("div"),
            title: el("span", "ga-tile-title"),
            line: el("div", "ga-tile-line"),
            value: el("span", "ga-tile-value"),
            secondary: el("span", "ga-tile-secondary"),
            pct: el("div", "ga-tile-pct"),
            track: el("div", "ga-track"),
            fill: el("div", "ga-track-fill"),
            plabel: el("span", "ga-tile-plabel"),
          };
          p.track.appendChild(p.fill);
          parts.push(p);
        }
        parts.length = tiles.length;
        tiles.forEach((t, k) => {
          const p = parts[k];
          setClass(p.box, "ga-stile" + (t.highlight === true ? " hl" : ""));
          setText(p.title, s(t.title));
          setText(p.value, s(t.value));
          setText(p.secondary, s(t.secondary));
          syncChildren(p.line, has(t.secondary) ? [p.value, p.secondary] : [p.value]);
          const shown = [p.title, p.line];
          if (finite(t.percent)) {
            const width = Math.max(0, Math.min(100, t.percent));
            setAttr(p.fill, "style", `width: ${num(width)}%`);
            setText(p.plabel, `${String(t.percent)}%`);
            syncChildren(p.pct, [p.track, p.plabel]);
            shown.push(p.pct);
          }
          syncChildren(p.box, shown);
        });
        syncChildren(rec.el, parts.map((p) => p.box));
      },
    },

    Comparison: {
      shape: () => "div",
      create(rec) {
        const st = rec.state;
        const root = el("div", "ga-cmp");
        // B5a: Chrome makes a scrolling element keyboard-focusable on its own; only Actions
        // buttons may take focus, so the scroller opts out of the Tab order.
        root.setAttribute("tabindex", "-1");
        st.grid = el("div", "ga-cmp-grid");
        st.corner = el("div", "ga-cmp-head ga-cmp-corner");
        st.heads = [];
        st.rows = [];
        root.appendChild(st.grid);
        return root;
      },
      update(rec, c) {
        const st = rec.state;
        const columns = list(c.columns);
        const rows = list(c.rows);
        const cellsOf = (r) => (Array.isArray(r.cells) ? r.cells : []);
        // Extra cells become columns with no header; missing cells render empty (B4).
        const width = Math.max(columns.length, 0, ...rows.map((r) => cellsOf(r).length));
        const hl = Number.isInteger(c.highlight) && c.highlight >= 0 && c.highlight < columns.length ? c.highlight : -1;
        setAttr(st.grid, "style", `grid-template-columns: minmax(88px, max-content) repeat(${width}, minmax(112px, 1fr))`);
        const nodes = [st.corner];

        while (st.heads.length < width) st.heads.push({ cell: el("div"), badge: el("span", "ga-badge"), title: el("span", "ga-cmp-title") });
        st.heads.length = width;
        st.heads.forEach((head, k) => {
          const col = k < columns.length ? columns[k] : null;
          setClass(head.cell, "ga-cmp-head ga-cmp-c" + (k === hl ? " hl" : ""));
          const kids = [];
          if (col && has(col.badge)) {
            setText(head.badge, col.badge);
            kids.push(head.badge);
          }
          if (col && has(col.title)) {
            setText(head.title, col.title);
            kids.push(head.title);
          }
          syncChildren(head.cell, kids);
          nodes.push(head.cell);
        });

        while (st.rows.length < rows.length) st.rows.push({ label: el("div", "ga-cmp-cell ga-cmp-label"), cells: [] });
        st.rows.length = rows.length;
        rows.forEach((r, j) => {
          const row = st.rows[j];
          setText(row.label, s(r.label));
          nodes.push(row.label);
          const cells = cellsOf(r);
          while (row.cells.length < width) row.cells.push({ cell: el("div"), text: el("span") });
          row.cells.length = width;
          row.cells.forEach((part, k) => {
            setClass(part.cell, "ga-cmp-cell ga-cmp-c" + (k === hl ? " hl" : ""));
            const v = k < cells.length ? cells[k] : undefined;
            const toned = isObject(v) && (v.tone === "ok" || v.tone === "warn") ? v.tone : null;
            const text = typeof v === "string" ? v : isObject(v) ? s(v.text) : "";
            setClass(part.text, toned ? `ga-pill ${toned}` : "ga-cmp-val");
            setText(part.text, text);
            syncChildren(part.cell, text ? [part.text] : []);
            nodes.push(part.cell);
          });
        });
        syncChildren(st.grid, nodes);
      },
    },

    Bars: {
      shape: () => "div",
      create(rec) {
        const st = rec.state;
        const root = el("div", "ga-bars");
        st.title = el("span", "ga-bars-title");
        st.rowsEl = el("div", "ga-bars-rows");
        st.rows = [];
        st.axis = el("div", "ga-bar-row ga-bars-axis");
        st.axisPad = el("div");
        st.axisCell = el("div", "ga-bars-axis-cell");
        st.ticksEl = el("div", "ga-ticks");
        st.ticks = [];
        st.axisCell.appendChild(st.ticksEl);
        st.axis.appendChild(st.axisPad);
        st.axis.appendChild(st.axisCell);
        // B7a: how many tick labels fit depends on the drawn width of the scale, so it is
        // measured, and measured again when that width changes (a resize, a new room).
        st.limit = MAX_TICKS; // the tick count the last fit chose (0: no labels fit)
        st.fitMax = null; // the largest value that fit was for
        st.fitWidth = -1; // the scale width it was measured at
        const win = doc.defaultView;
        const RO = win && win.ResizeObserver;
        if (typeof RO === "function") {
          // The refit runs in the next animation frame, never inside the observer's callback, so
          // it cannot change layout while notifications are being delivered (B7c).
          const refit = () => {
            st.frame = 0;
            const w = scaleWidth(st);
            if (w >= 0 && w !== st.fitWidth && st.fitMax !== null) fitTicks(st);
          };
          st.frame = 0;
          st.observer = new RO(() => {
            if (st.frame) return;
            if (typeof win.requestAnimationFrame === "function") st.frame = win.requestAnimationFrame(refit);
            else refit();
          });
          st.observer.observe(st.ticksEl);
        }
        return root;
      },
      update(rec, c) {
        const st = rec.state;
        const bars = list(c.bars);
        const unit = has(c.unit) ? c.unit : null;
        const values = bars.map((b) => (finite(b.value) && b.value >= 0 ? b.value : null));
        let top = -1;
        values.forEach((v, k) => {
          if (v !== null && (top < 0 || v > values[top])) top = k;
        });
        const vmax = top >= 0 && values[top] > 0 ? values[top] : null;
        // The value label (the maximum's String(value) and the unit, B6a: from the maximum only,
        // so a smaller value never moves the scale) sits on one line (B9a). Its estimated width
        // `pair` (in ch of the value's font, plus the gap) is kept as room after the longest bar
        // when it takes at most half the bar column. Otherwise, and in a Bars at most NARROW_BARS
        // wide (card.css stacks the row there), every value takes its own line under its bar and
        // no room is kept: the bars and the scale use the whole column. `100cqi` is the Bars' own
        // width (an inline-size container); 50% is of the bar column (the fill's and the value's
        // flex container). Every fill shares one scale.
        const digits = top >= 0 ? String(values[top]).length : 0;
        const chars = digits + (unit ? unit.length + 1 : 0);
        const pair = `${num(chars * 1.25)}ch + 16px`;
        const beside = `max(0px, (50% - (${pair})) * 1000 + (${pair}))`;
        const room = `min(${pair}, ${beside}, max(0px, (100cqi - ${NARROW_BARS}px) * 1000))`;
        const below = `max(${pair} - 50%, ${NARROW_BARS}px + 0.5px - 100cqi)`;
        const valStyle = `flex-basis: clamp(0%, ${below} * 1000, 100%)`;
        st.room = room;

        while (st.rows.length < bars.length) {
          const p = {
            row: el("div", "ga-bar-row"),
            label: el("span", "ga-bar-label"),
            cell: el("div", "ga-bar-cell"),
            fill: el("div"),
            val: el("span", "ga-bar-val"),
            num: el("span", "ga-bar-num"),
            unit: el("span", "ga-bar-unit"),
          };
          p.row.appendChild(p.label);
          p.row.appendChild(p.cell);
          p.cell.appendChild(p.fill);
          p.cell.appendChild(p.val);
          st.rows.push(p);
        }
        st.rows.length = bars.length;
        st.values = values;
        bars.forEach((b, k) => {
          const p = st.rows[k];
          const v = values[k];
          setClass(p.row, "ga-bar-row" + (k === top ? " max" : ""));
          setText(p.label, s(b.label));
          setClass(p.fill, "ga-bar-fill" + (k === top ? " max" : ""));
          const shown = [];
          if (v !== null) {
            setText(p.num, String(v));
            shown.push(p.num);
            if (unit) {
              setText(p.unit, unit);
              shown.push(p.unit);
            }
          }
          syncChildren(p.val, shown);
          setAttr(p.val, "style", valStyle);
        });
        syncChildren(st.rowsEl, st.rows.map((p) => p.row));
        setAttr(st.ticksEl, "style", `width: calc(100% - ${room})`);

        const kids = [];
        if (has(c.title)) {
          setText(st.title, c.title);
          kids.push(st.title);
        }
        kids.push(st.rowsEl);
        // The axis stays while there is a scale, even with no label drawn, so its width is
        // still measured when the card widens again.
        if (vmax !== null) kids.push(st.axis);
        syncChildren(rec.el, kids);

        if (vmax !== st.fitMax) {
          // A new largest value: start from the full scale, then fit it to the drawn width.
          st.fitMax = vmax;
          st.limit = MAX_TICKS;
          st.fitWidth = -1;
          drawScale(st, MAX_TICKS);
          if (vmax !== null && scaleWidth(st) >= 0) fitTicks(st);
        } else {
          drawScale(st, st.limit);
        }
      },
    },

    Steps: {
      shape: () => "ol",
      create(rec) {
        rec.state.steps = [];
        return el("ol", "ga-steps");
      },
      update(rec, c) {
        const steps = list(c.steps);
        const parts = rec.state.steps;
        while (parts.length < steps.length) {
          const p = {
            li: el("li", "ga-step"),
            ord: el("span", "ga-step-num"),
            body: el("div", "ga-step-body"),
            title: el("span", "ga-step-title"),
            detail: el("span", "ga-step-detail"),
          };
          p.li.appendChild(p.ord);
          p.li.appendChild(p.body);
          parts.push(p);
        }
        parts.length = steps.length;
        steps.forEach((step, k) => {
          const p = parts[k];
          setText(p.ord, String(k + 1));
          setText(p.title, s(step.title));
          setText(p.detail, s(step.detail));
          syncChildren(p.body, has(step.detail) ? [p.title, p.detail] : [p.title]);
        });
        syncChildren(rec.el, parts.map((p) => p.li));
      },
    },

    Source: {
      shape: () => "div",
      create(rec) {
        const root = el("div", "ga-source");
        rec.state.text = el("span", "ga-source-text");
        root.appendChild(glyph(SOURCE_GLYPH, "0 0 24 24", "ga-source-glyph"));
        root.appendChild(rec.state.text);
        return root;
      },
      update(rec, c) {
        setText(rec.state.text, s(c.text));
      },
    },

    Actions: {
      shape: () => "div",
      create(rec) {
        rec.state.buttons = [];
        return el("div", "ga-actions");
      },
      update(rec, c) {
        const labels = Array.isArray(c.labels) ? c.labels.map(s) : [];
        const buttons = rec.state.buttons;
        while (buttons.length < labels.length) {
          const i = buttons.length;
          const b = el("button");
          b.setAttribute("type", "button");
          // Exactly a gate-made Button's action: the event name and `{label}`, from this
          // component's id (the engine adds surfaceId, sourceComponentId and the timestamp).
          b.addEventListener("click", (event) => {
            // B5b: one click, one action, even inside a gate-admitted Button.
            if (event && typeof event.stopPropagation === "function") event.stopPropagation();
            const now = Array.isArray(rec.comp && rec.comp.labels) ? rec.comp.labels[i] : undefined;
            if (typeof now !== "string") return;
            dispatch({ event: { name: GATE_CLICKED, context: { label: now } } }, rec.id);
          });
          buttons.push(b);
        }
        buttons.length = labels.length;
        labels.forEach((label, k) => {
          setClass(buttons[k], "ga-act" + (k === 0 ? " primary" : ""));
          setText(buttons[k], label);
        });
        syncChildren(rec.el, buttons);
      },
    },
  };
}
