// The assist panel: four checkboxes, the server's badge, the counters (T-0025 criteria 1-4, 7, 8).
//
// This is the DOM half; everything that decides what may leave the page is in ./gate.js and is
// driven by node without a browser. `mount(host)` is the contract `hud/src/panels.js` calls, and
// it must never throw: a host that predates this seam (no document, no send, no subscribe)
// degrades to nothing rendered and nothing sent, leaving every other panel and the socket working
// (T-0008 criterion 10, T-0025 criterion 1).
//
// The panel inserts its own element and its own <style>; hud/index.html and hud/styles.css are
// T-0008's files and are not touched. T-0162 B-6: given `host.settingsSlot` the same
// #assist-panel goes there, headed "Context helpers" and explained in one sentence; without one
// it appends to <main> as before.

import { MAX_FPS } from "../frames.js";
import { ASSIST_KINDS, ASSIST_STATE_TYPE, createAssistGate } from "./gate.js";
import { registryFor } from "./extract.js";

/** How often an extraction is *offered*. The gate throttles to 1000 / MAX_FPS exactly as the
 *  frame pipeline does, so this only has to be at least that often (main.js uses 250 ms too). */
const OFFER_TICK_MS = 250;

/** T-0154 A5 (H-T154-2): the human label of each kind, and the one line an unavailable row
 *  shows (the full reason stays in data-assist-reason and in the row's title). */
const LABELS = {
  page_text: "Page text",
  pdf_text: "PDF text",
  cells: "Spreadsheet cells",
  at_spi: "Accessibility tree",
};
const UNAVAILABLE_LINE = "Needs the desktop helper";

/** T-0162 B-6 (from D12a): what the helpers are, under the Settings popover's heading. */
export const CONTEXT_HELPERS_HEADING = "Context helpers";
export const CONTEXT_HELPERS_SENTENCE =
  "Gearvis reads your screen from pixels only; these optional helpers also send it the page's " +
  "text, a PDF's text layer, spreadsheet cells or the accessibility tree. They are all off by " +
  "default, and every turn records which ones were on.";

const STYLE = `
#assist-panel .assist-badge { position: absolute; z-index: 2; top: var(--space-4); right: 18px; max-width: 60%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-xs); line-height: 1.5; color: var(--muted); font-weight: 500; }
#assist-panel .assist-explain { margin: 0 0 var(--space-3); font-size: var(--text-xs); line-height: 1.5; color: var(--muted); white-space: normal; }
#assist-panel .assist-rows { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; }
#assist-panel .assist-row { margin: 0; }
#assist-panel .assist-row label { display: flex; gap: var(--space-3); align-items: center; justify-content: space-between; }
#assist-panel .assist-text { display: flex; flex-direction: column; min-width: 0; }
#assist-panel [data-assist-label] { font-size: 14px; }
#assist-panel .assist-why { display: block; font-size: var(--text-xs); line-height: 1.4; margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--muted); }
#assist-panel .assist-why:empty { display: none; }
/* Checkboxes drawn as switches (A5): still input[type=checkbox]; off, on and disabled differ. */
#assist-panel input[type="checkbox"] { appearance: none; flex: none; position: relative; width: 34px; height: 20px; margin: 0; border: 0; border-radius: var(--radius-pill); background: var(--surface-3); cursor: pointer; }
#assist-panel input[type="checkbox"]::before { content: ""; position: absolute; top: 3px; left: 3px; width: 14px; height: 14px; border-radius: 50%; background: var(--faint); }
#assist-panel input[type="checkbox"]:checked { background: var(--accent); }
#assist-panel input[type="checkbox"]:checked::before { left: 17px; background: var(--knob-on); }
#assist-panel input[type="checkbox"]:disabled { opacity: 0.45; cursor: not-allowed; }
/* T-0056 ruling A7: "this kind is not available here" is said by the disabled checkbox and by a
   recessed *colour* on its name — never by an opacity fade on the row. The fade also dimmed
   .assist-why, and the reason the human needs in order to understand the greyed-out box computed
   3.40:1 dark / 2.48:1 light under it; a fade on the label alone still left the name at 3.98:1
   in the light scheme (measured). --faint is 6.72:1 dark / 5.54:1 light on a pane, and the three
   row states stay distinct: unavailable --faint, available --ink, on --accent. */
#assist-panel .assist-row[data-assist-available="false"] label { color: var(--faint, #8fa0aa); }
#assist-panel .assist-row[data-assist-on="true"] label { color: var(--accent, #6fd3a0); }
#assist-panel .assist-pending { font-size: 11px; color: var(--warn, #f0b357); }
#assist-panel .assist-counters { color: var(--dim, #8fa0aa); margin: 12px 0 0; font-size: var(--text-xs); }
`;

/** What `capture.js` will report as the live source for a given `?source=` (see main.js). */
function captureOf(search) {
  const requested = new URLSearchParams(search || "").get("source") || "screen";
  return {
    synthetic: requested === "canvas" || requested === "synthetic",
    source: requested === "camera" ? "camera" : "screen",
  };
}

/** An unavailable row says one short line; the full reason is the row's and the line's title. */
function describe(row, availability) {
  const reason = availability.available ? "" : availability.reason || "";
  row.why.textContent = availability.available ? "" : UNAVAILABLE_LINE;
  for (const el of [row.row, row.why]) {
    if (reason) el.setAttribute("title", reason);
    else el.removeAttribute("title");
  }
}

/** T-0162 B-6: the element the panel goes in — the host's settings slot (an element, or a function
 *  returning one) when it gives one, else <main> as before. */
function slotOf(host) {
  const slot = typeof host.settingsSlot === "function" ? host.settingsSlot() : host.settingsSlot;
  return slot && typeof slot.appendChild === "function" ? slot : null;
}

function buildPanel(doc, availability, slot) {
  const style = doc.createElement("style");
  style.textContent = STYLE;
  (doc.head || doc.body).appendChild(style);

  const panel = doc.createElement("section");
  panel.id = "assist-panel";
  panel.className = "pane";

  const heading = doc.createElement("h2");
  heading.textContent = slot ? CONTEXT_HELPERS_HEADING : "Context";
  panel.appendChild(heading);

  // The server's badge, verbatim, on the heading's row (T-0154 A5); empty, it shows nothing.
  const badge = doc.createElement("b");
  badge.className = "assist-badge";
  badge.setAttribute("data-assist-badge", "");
  panel.appendChild(badge);

  if (slot) {
    const explain = doc.createElement("p");
    explain.className = "assist-explain";
    explain.setAttribute("data-assist-explain", "");
    explain.textContent = CONTEXT_HELPERS_SENTENCE;
    panel.appendChild(explain);
  }

  const list = doc.createElement("ul");
  list.className = "assist-rows";
  const rows = {};
  for (const kind of ASSIST_KINDS) {
    const row = doc.createElement("li");
    row.className = "assist-row";
    row.setAttribute("data-assist-kind", kind);
    row.setAttribute("data-assist-on", "false");
    row.setAttribute("data-assist-available", String(availability[kind].available));
    row.setAttribute("data-assist-reason", availability[kind].reason || "");

    const label = doc.createElement("label");
    const box = doc.createElement("input");
    box.type = "checkbox";
    box.disabled = !availability[kind].available;
    const text = doc.createElement("span");
    text.className = "assist-text";
    const name = doc.createElement("span");
    name.setAttribute("data-assist-label", "");
    name.textContent = LABELS[kind] || kind;
    const pending = doc.createElement("span");
    pending.className = "assist-pending";
    const why = doc.createElement("small");
    why.className = "assist-why";
    text.append(name, pending, why);
    label.append(text, box);

    row.append(label);
    list.appendChild(row);
    rows[kind] = { row, box, pending, why };
    describe(rows[kind], availability[kind]);
  }
  panel.appendChild(list);

  const counters = doc.createElement("p");
  counters.className = "assist-counters";
  const sent = doc.createElement("b");
  sent.setAttribute("data-assist-sent", "0");
  sent.textContent = "0";
  const dropped = doc.createElement("b");
  dropped.setAttribute("data-assist-dropped", "0");
  dropped.textContent = "0";
  counters.append("text sent ", sent, " dropped ", dropped);
  panel.appendChild(counters);

  (slot || doc.querySelector("main") || doc.body).appendChild(panel);
  return { panel, badge, rows, sent, dropped };
}

/**
 * mount(host) — host is { shown, document, send, subscribe, log } (T-0025 criterion 13), plus
 * the optional `settingsSlot` (T-0162 B-6): the header's Settings popover the panel goes in.
 *
 * `send` is the raw `socket.send`; the panel validates its own egress against its own whitelist
 * in gate.js. `subscribe` is called with every decoded astra.* message. Neither is required:
 * without a document nothing is rendered, and without `send` nothing is ever offered.
 */
export function mount(host) {
  const h = host && typeof host === "object" ? host : {};
  const doc = h.document;
  const log = typeof h.log === "function" ? h.log : () => {};
  const send = typeof h.send === "function" ? h.send : null;
  const { source, synthetic } = captureOf(globalThis.location && globalThis.location.search);
  const { extractors, reasons } = registryFor({ source, synthetic, doc: doc || null });

  const gate = createAssistGate({
    send: send || (() => Promise.reject(new Error("this host has no socket"))),
    now: () => Date.now(),
    extractors,
    reasons,
    log,
  });

  // Declared before the subscription because an inbound message may arrive before — or without —
  // any rendering at all (criterion 1).
  let ui = null;

  // Criterion 1: a panel with no document must still survive an inbound message.
  if (typeof h.subscribe === "function") {
    h.subscribe((message) => {
      if (!message || message.type !== ASSIST_STATE_TYPE) return;
      gate.onState(message);
      render();
    });
  }

  if (!doc || typeof doc.createElement !== "function") {
    log("assist panel: this host has no document, so no rows are rendered and nothing is offered");
    return { gate };
  }

  ui = buildPanel(doc, gate.available(), slotOf(h));

  function renderCounters() {
    if (!ui) return;
    const stats = gate.stats();
    ui.sent.setAttribute("data-assist-sent", String(stats.sent));
    ui.sent.textContent = String(stats.sent);
    ui.dropped.setAttribute("data-assist-dropped", String(stats.dropped));
    ui.dropped.textContent = String(stats.dropped);
  }

  function render() {
    if (!ui) return;
    const flags = gate.flags();
    const pending = new Set(gate.pending());
    const availability = gate.available();
    for (const kind of ASSIST_KINDS) {
      const row = ui.rows[kind];
      // Criterion 3: on-ness comes from the server's echo, never from the click. A clicked box
      // stays as the user left it while its toggle is unechoed, and says so.
      row.row.setAttribute("data-assist-on", String(Boolean(flags[kind])));
      row.row.setAttribute("data-assist-available", String(availability[kind].available));
      row.row.setAttribute("data-assist-reason", availability[kind].reason || "");
      describe(row, availability[kind]);
      row.pending.textContent = pending.has(kind) ? "waiting for the server" : "";
      if (!pending.has(kind)) row.box.checked = Boolean(flags[kind]);
    }
    // Criterion 4: the badge is the server's string, character for character.
    ui.badge.textContent = gate.badge();
    ui.badge.setAttribute("data-assist-badge", gate.badge());
    renderCounters();
  }

  for (const kind of ASSIST_KINDS) {
    ui.rows[kind].box.addEventListener("change", () => {
      if (!gate.toggle(kind, ui.rows[kind].box.checked)) {
        ui.rows[kind].box.checked = Boolean(gate.flags()[kind]);
      }
      render();
    });
  }

  render();

  if (send) {
    globalThis.setInterval(() => {
      gate.offer(source).then(renderCounters, renderCounters);
      renderCounters();
    }, OFFER_TICK_MS);
  }

  log(
    `assist panel mounted: source=${source}${synthetic ? " (synthetic)" : ""}, ` +
      `extractors=${Object.keys(extractors).join(",") || "none"}, at most one text per kind per ` +
      `${1000 / MAX_FPS} ms`,
  );
  return { gate };
}
