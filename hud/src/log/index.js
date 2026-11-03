// The session-log panel: one checkbox, the server's path and the writer's counters (T-0043
// criterion 19).
//
// This is the DOM half; everything that decides what the box may claim is in ./gate.js and is
// driven by node without a browser. `mount(host)` is the contract `hud/src/panels.js` calls, and
// it must never throw: a host that predates this seam (no document, no send, no subscribe)
// degrades to nothing rendered and nothing sent, leaving every other panel and the socket working
// (T-0008 criterion 10, T-0025 criterion 1).
//
// The panel inserts its own element and its own <style>; hud/index.html, hud/styles.css,
// hud/src/main.js and hud/src/state.js are T-0008's files and are not touched.
//
// Three states are rendered, never two (criterion 19). Before any `astra.log.state` has arrived
// the panel says **unknown**: it does not tick the box, it names no file and it claims nothing,
// because a page that says a session is being recorded when it is not is the failure this panel
// exists to prevent.

import { LOG_STATE_TYPE, createLogGate } from "./gate.js";

const STYLE = `
#log-panel .log-row { display: flex; gap: 6px; align-items: baseline; }
#log-panel .log-pending { font-size: 11px; color: var(--warn, #f0b357); }
#log-panel .log-path { font-family: var(--mono, monospace); font-size: 11px; word-break: break-all; }
#log-panel .log-counters { color: var(--dim, #8fa0aa); margin: 0; font-size: 11px; }
#log-panel .log-error { display: block; font-size: 11px; color: var(--warn, #f0b357); }
#log-panel[data-log-state="unknown"] .log-status { color: var(--dim, #8fa0aa); }
#log-panel[data-log-state="on"] .log-status { color: var(--accent, #6fd3a0); }
`;

/** What the panel says about a state it has not been told. Never "on", never a ticked box. */
const STATUS = { unknown: "not yet known", on: "logging to", off: "not logging" };

function buildPanel(doc) {
  const style = doc.createElement("style");
  style.textContent = STYLE;
  (doc.head || doc.body).appendChild(style);

  const panel = doc.createElement("section");
  panel.id = "log-panel";
  panel.className = "pane";
  panel.setAttribute("data-log-panel", "");
  panel.setAttribute("data-log-state", "unknown");

  const heading = doc.createElement("h2");
  heading.textContent = "session log";
  panel.appendChild(heading);

  const row = doc.createElement("label");
  row.className = "log-row";
  const box = doc.createElement("input");
  box.type = "checkbox";
  box.setAttribute("data-log-toggle", "");
  const name = doc.createElement("span");
  name.textContent = "Log this session";
  const pending = doc.createElement("span");
  pending.className = "log-pending";
  pending.setAttribute("data-log-pending", "");
  row.append(box, name, pending);
  panel.appendChild(row);

  const where = doc.createElement("p");
  where.className = "log-counters";
  const status = doc.createElement("span");
  status.className = "log-status";
  status.setAttribute("data-log-status", "");
  const path = doc.createElement("code");
  path.className = "log-path";
  path.setAttribute("data-log-path", "");
  where.append(status, " ", path);
  panel.appendChild(where);

  const counters = doc.createElement("p");
  counters.className = "log-counters";
  const records = doc.createElement("b");
  records.setAttribute("data-log-records", "");
  const dropped = doc.createElement("b");
  dropped.setAttribute("data-log-dropped", "");
  counters.append("records ", records, " dropped ", dropped);
  panel.appendChild(counters);

  const error = doc.createElement("small");
  error.className = "log-error";
  error.setAttribute("data-log-error", "");
  panel.appendChild(error);

  (doc.querySelector("main") || doc.body).appendChild(panel);
  return { panel, box, pending, status, path, records, dropped, error };
}

/**
 * mount(host) — host is { shown, document, send, subscribe, log }, T-0025's seam.
 *
 * `send` is the raw `socket.send`; the panel validates its own egress against its own whitelist
 * in gate.js. `subscribe` is called with every decoded astra.* message. Neither is required:
 * without a document nothing is rendered, and without `send` nothing is ever emitted.
 */
export function mount(host) {
  const h = host && typeof host === "object" ? host : {};
  const doc = h.document;
  const log = typeof h.log === "function" ? h.log : () => {};
  const send = typeof h.send === "function" ? h.send : null;

  const gate = createLogGate({ send, log });

  // Declared before the subscription because an inbound state may arrive before — or without —
  // any rendering at all.
  let ui = null;

  if (typeof h.subscribe === "function") {
    h.subscribe((message) => {
      if (!message || message.type !== LOG_STATE_TYPE) return;
      gate.onState(message);
      render();
    });
  }

  if (!doc || typeof doc.createElement !== "function") {
    log("log panel: this host has no document, so the session-log checkbox is not rendered");
    return { gate };
  }

  ui = buildPanel(doc);

  function render() {
    if (!ui) return;
    const known = gate.known();
    const on = gate.on();
    const state = known ? (on ? "on" : "off") : "unknown";
    ui.panel.setAttribute("data-log-state", state);
    // Criterion 19: the box shows the server's last word and nothing else. Not the click (a box
    // that ticks itself claims a recording nobody has agreed to make), and not a guess at the
    // server's default while the state is unknown.
    ui.box.checked = on;
    ui.pending.textContent = gate.pending() ? "waiting for the server" : "";
    ui.status.textContent = STATUS[state];
    // Verbatim: this is the file the human is about to open, so a shortened or prettified one is
    // the wrong file. Nothing is shown until the server has named one.
    ui.path.textContent = gate.path() || "";
    const stats = gate.stats();
    ui.records.textContent = known ? String(stats.records) : "";
    ui.dropped.textContent = known ? String(stats.dropped) : "";
    // Criterion 4: the reason logging is off reaches the person looking at the page.
    ui.error.textContent = stats.error || "";
  }

  ui.box.addEventListener("change", () => {
    // The gate decides whether anything left the page; the render that follows puts the box back
    // to the server's truth either way.
    gate.toggle(ui.box.checked);
    render();
  });

  render();
  log("log panel mounted: the checkbox shows the server's astra.log.state, never the click");
  return { gate };
}
