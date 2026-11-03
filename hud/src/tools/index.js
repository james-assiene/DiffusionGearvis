// The Tools pane (T-0162 B-1 to B-5): every tool call of every turn, newest turn first, one line
// per call_id (the tool, its arguments in short form, the status), each a native <details> with
// the whole call. It replaces the rail's Context pane, which moved behind the header's Settings.
//
// Fed `astra.tool.call` and `astra.turn.state` only, so no thought, canvas or spoken text can
// reach it (B-5); astra.session.cleared only empties it (T-0162 D-2). Every model string is set
// with textContent. The pane builds its own <section> and appends it to <main> (T-0154: panes
// self-append); its look is hud/styles.css, tokens only.
//
// `mount(host)` is the contract hud/src/panels.js calls and never throws: a host without a
// document or a subscription renders nothing.

import {
  SOURCE_LABELS,
  argumentLines,
  createToolsModel,
  shortArgs,
  shortResult,
  shownLabel,
  statusLabel,
} from "./model.js";
import { createClearTracker, seedFrom } from "../session/gate.js";

const EMPTY_LINE = "no tool calls yet — the calls Gearvis makes appear here, newest turn first";

function el(doc, tag, className, text) {
  const node = doc.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Render into `root` (the pane's list). Returns { handle(message), clear() }. `lastResets` is the
 *  page's last accepted reset count at mount (T-0162 D-2), so a replay it missed stays old news. */
export function createToolsPane(root, { lastResets = null } = {}) {
  const doc = root.ownerDocument;
  let model = createToolsModel();
  const resets = createClearTracker(lastResets);
  const turnEls = new Map(); // turn_id -> { box, list }
  const callEls = new Map(); // turn_id + call_id -> parts

  function turnEl(turnId) {
    let parts = turnEls.get(turnId);
    if (!parts) {
      const box = el(doc, "section", "tools-turn");
      box.dataset.toolsTurn = turnId;
      const head = el(doc, "p", "tools-turn-id", `turn ${turnId}`);
      const list = el(doc, "ol", "tools-calls");
      box.append(head, list);
      parts = { box, list };
      turnEls.set(turnId, parts);
    }
    return parts;
  }

  function callEl(call, list) {
    const key = `${call.turnId}\u0000${call.callId}`;
    let parts = callEls.get(key);
    if (!parts) {
      const item = el(doc, "li", "tools-call");
      const details = el(doc, "details", "tools-details");
      details.dataset.toolsCall = call.callId;
      const summary = el(doc, "summary", "tools-summary");
      const line = el(doc, "span", "tools-line");
      line.dataset.toolsLine = "";
      const status = el(doc, "span", "tools-status");
      // C-3: the one-line result, in the <summary> so it shows while the call is folded
      const result = el(doc, "span", "tools-detail-line tools-result");
      result.dataset.toolsResult = "";
      result.hidden = true;
      summary.append(line, status, result);
      const body = el(doc, "div", "tools-detail");
      body.dataset.toolsDetail = "";
      details.append(summary, body);
      item.appendChild(details);
      list.appendChild(item);
      parts = { item, line, status, body, result };
      callEls.set(key, parts);
    }
    return parts;
  }

  /** One `key: value` line of the detail. */
  function row(text) {
    return el(doc, "p", "tools-detail-line", text);
  }

  function fill(parts, call) {
    parts.line.textContent = shortArgs(call.name, call.arguments);
    parts.line.title = parts.line.textContent;
    // C-3: the result's outcome when the final message carries one, else the wire status
    parts.status.dataset.toolsStatus = call.result ? call.result.outcome : call.status;
    parts.status.textContent = shownLabel(call);
    const result = shortResult(call) || "";
    parts.result.textContent = result;
    parts.result.hidden = !result;
    const rows = [row(`tool: ${call.name}`)];
    if (call.step !== null && call.step !== undefined) rows.push(row(`step: ${call.step}`));
    for (const line of argumentLines(call.arguments)) rows.push(row(line));
    rows.push(row(`source: ${SOURCE_LABELS[call.source] || call.source}`));
    if (call.action) rows.push(row(`action: ${call.action}`));
    if (call.reason) rows.push(row(`reason: ${call.reason}`));
    if (call.queryProblem) rows.push(row(`query_problem: ${call.queryProblem}`));
    const statuses = row(`statuses: ${call.statuses.map(statusLabel).join(" → ")}`);
    statuses.dataset.toolsStatuses = "";
    rows.push(statuses);
    parts.body.replaceChildren(...rows);
  }

  function render() {
    // appendChild moves an existing node, so an opened <details> stays open across a re-order.
    for (const turn of model.view()) {
      const { box, list } = turnEl(turn.turnId);
      for (const call of turn.calls) fill(callEl(call, list), call);
      root.appendChild(box);
    }
    root.dataset.toolsCalls = String(callEls.size);
  }

  /** T-0162 D-2: every turn and call goes, and the empty line shows again. */
  function clear() {
    for (const { box } of turnEls.values()) box.remove();
    turnEls.clear();
    callEls.clear();
    model = createToolsModel();
    root.dataset.toolsCalls = "0";
  }

  function handle(message) {
    if (resets.accept(message)) {
      clear();
      return;
    }
    if (model.onMessage(message)) render();
  }

  root.dataset.toolsCalls = "0";
  return { handle, clear };
}

function buildPane(doc) {
  const pane = el(doc, "section", "pane pane-tools");
  pane.id = "tools-panel";
  pane.setAttribute("aria-label", "Tools");
  pane.appendChild(el(doc, "h2", "", "Tools"));
  const empty = el(doc, "p", "empty-line tools-empty", EMPTY_LINE);
  empty.dataset.toolsEmpty = "";
  const root = el(doc, "div", "tools");
  root.dataset.toolsRoot = "";
  pane.append(empty, root);
  (doc.querySelector("main") || doc.body).appendChild(pane);
  return root;
}

/** mount(host) — host is { document, subscribe, log }. */
export function mount(host) {
  const h = host && typeof host === "object" ? host : {};
  const doc = h.document;
  const log = typeof h.log === "function" ? h.log : () => {};
  if (!doc || typeof doc.createElement !== "function") {
    log("tools pane: this host has no document, so nothing is rendered");
    return {};
  }
  const pane = createToolsPane(buildPane(doc), { lastResets: seedFrom(h) });
  if (typeof h.subscribe === "function") h.subscribe(pane.handle);
  else log("tools pane: this host has no subscription, so no call is shown");
  return pane;
}
