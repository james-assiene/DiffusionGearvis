// The Tools pane's DOM-free half (T-0162 B-1 to B-4): the short form of a call's arguments, and
// the per-turn, per-call record the pane draws from `astra.tool.call` (plus `astra.turn.state`
// for the turn order). Nothing here reads a thought, a canvas or a spoken line: the pane is fed
// tool calls only (B-5).
//
// No DOM at module top level: `node --test tests/hud` imports this module.

export const TOOL_CALL_TYPE = "astra.tool.call";
export const TURN_STATE_TYPE = "astra.turn.state";

/** B-3: the six wire statuses, shown verbatim. */
export const STATUS_LABELS = {
  proposed: "proposed",
  repairing: "repairing",
  dry_run: "dry run",
  executed: "executed",
  refused: "refused",
  cancelled: "cancelled",
};

/** C-3: the six outcomes of a final message's `result` (contracts/protocol.py ToolCallOutcome).
 *  `failed` and `timeout` come from that field, never from the reason's wording. */
export const OUTCOME_LABELS = {
  executed: "executed",
  dry_run: "dry run",
  refused: "refused",
  failed: "failed",
  cancelled: "cancelled",
  timeout: "timeout",
};

/** The tools that act in the browser, whose result names the hosts they reached (T-0160 B-6; `wait`, T-0202 A5). */
const BROWSER_TOOLS = new Set(["click_at", "type_at", "scroll", "press_key", "page_overview", "wait"]);

export const SOURCE_LABELS = { draft: "from the draft", repair: "repaired" };

export const TYPE_TEXT_MAX = 40; // code points of type_at's text
export const QUERY_MAX = 60; // code points of google_search's query
export const OTHER_MAX = 80; // code points of any other tool's compact JSON

const ELLIPSIS = "…";

/** `text` cut at `max` code points (never inside a surrogate pair), with "…" when cut. */
export function truncate(text, max) {
  const points = Array.from(String(text));
  return points.length <= max ? points.join("") : points.slice(0, max).join("") + ELLIPSIS;
}

const arg = (args, key) => {
  const v = args[key];
  return v === undefined || v === null ? "?" : String(v);
};

const at = (args) => `(${arg(args, "x")}, ${arg(args, "y")})`;

/** B-2: one short line per call. Pure; the pane sets it with textContent. */
export function shortArgs(name, args) {
  const a = args && typeof args === "object" && !Array.isArray(args) ? args : {};
  switch (name) {
    case "click_at":
      return `click_at ${at(a)}`;
    case "type_at":
      return (
        `type_at "${truncate(arg(a, "text"), TYPE_TEXT_MAX)}" at ${at(a)}` +
        (a.press_enter === true ? " + Enter" : "")
      );
    case "scroll":
      return `scroll ${arg(a, "direction")}`;
    case "press_key":
      return `press_key ${arg(a, "combo")}`;
    case "google_search":
      return `google_search "${truncate(arg(a, "query"), QUERY_MAX)}"`;
    case "page_overview":
      return "page_overview";
    case "wait":
      return "wait";
    default: {
      let json;
      try {
        json = JSON.stringify(a);
      } catch {
        json = "{…}";
      }
      return `${String(name)} ${truncate(json, OTHER_MAX)}`;
    }
  }
}

/** The label of a wire status, verbatim for one this page does not know. */
export function statusLabel(status) {
  return STATUS_LABELS[status] || String(status);
}

/** A final message's `result` (T-0162 C-1), or null; the decoder already checked its shape. */
function resultOf(m) {
  const r = m && m.result;
  return r && typeof r === "object" && typeof r.outcome === "string" ? r : null;
}

/** C-3: what the pane shows as the call's status: the result's outcome when there is one. */
export function shownLabel(call) {
  const r = resultOf(call);
  if (r) return OUTCOME_LABELS[r.outcome] || String(r.outcome);
  return statusLabel(call ? call.status : "");
}

/**
 * C-3: the one-line result of an astra.tool.call (or the pane's record of one, which keeps the
 * wire's `name`, `action` and `result`); null while it carries no `result`. Read from the fields
 * only, never from the action's prose:
 *   google_search, executed:  `N results (h1, h2, h3)` (`1 result (h)`) or `no results`
 *   a browser tool with hosts: `<action> · reached h1, h2` plus ` +n` beyond them
 *   anything else:             the action
 */
export function shortResult(message) {
  const r = resultOf(message);
  if (!r) return null;
  const action = typeof message.action === "string" ? message.action : "";
  const hosts = Array.isArray(r.hosts) ? r.hosts.map(String) : [];
  const more = Number.isInteger(r.hosts_more) && r.hosts_more > 0 ? r.hosts_more : 0;
  if (message.name === "google_search" && r.outcome === "executed") {
    const n = Number.isInteger(r.results) ? r.results : 0;
    if (n <= 0) return "no results";
    const count = `${n} result${n === 1 ? "" : "s"}`;
    return hosts.length ? `${count} (${hosts.join(", ")})` : count;
  }
  if (BROWSER_TOOLS.has(message.name) && hosts.length) {
    const reached = `reached ${hosts.join(", ")}${more > 0 ? ` +${more}` : ""}`;
    return action ? `${action} · ${reached}` : reached;
  }
  return action;
}

/** B-4: every argument as `key: value` (a non-string value as compact JSON). */
export function argumentLines(args) {
  const a = args && typeof args === "object" && !Array.isArray(args) ? args : {};
  return Object.keys(a).map((key) => {
    const v = a[key];
    let text;
    if (typeof v === "string") text = v;
    else {
      try {
        text = JSON.stringify(v);
      } catch {
        text = String(v);
      }
    }
    return `${key}: ${text}`;
  });
}

/**
 * The pane's record (B-1): turns newest first, calls in arrival order within a turn, one entry
 * per call_id. `onMessage` returns true when the record changed.
 *
 * A turn's place is when the page first heard of it, from either `astra.turn.state` or its first
 * `astra.tool.call`; a turn with no call is never listed.
 */
export function createToolsModel() {
  const order = []; // turn ids, first-seen order (oldest first)
  const turns = new Map(); // turn_id -> { turnId, calls: Map(call_id -> call) }

  function turnFor(turnId) {
    let turn = turns.get(turnId);
    if (!turn) {
      turn = { turnId, calls: new Map() };
      turns.set(turnId, turn);
      order.push(turnId);
    }
    return turn;
  }

  function onMessage(m) {
    if (!m || typeof m !== "object") return false;
    if (m.type === TURN_STATE_TYPE) {
      if (typeof m.turn_id !== "string" || turns.has(m.turn_id)) return false;
      turnFor(m.turn_id);
      return false; // nothing to show until the turn has a call
    }
    if (m.type !== TOOL_CALL_TYPE) return false;
    if (typeof m.turn_id !== "string" || typeof m.call_id !== "string") return false;
    const turn = turnFor(m.turn_id);
    let call = turn.calls.get(m.call_id);
    if (!call) {
      call = { callId: m.call_id, turnId: m.turn_id, statuses: [] };
      turn.calls.set(m.call_id, call);
    }
    call.name = String(m.name);
    call.arguments = m.arguments && typeof m.arguments === "object" ? m.arguments : {};
    call.status = String(m.status);
    call.source = String(m.source);
    call.action = typeof m.action === "string" ? m.action : null;
    call.reason = typeof m.reason === "string" ? m.reason : null;
    call.queryProblem = typeof m.query_problem === "string" ? m.query_problem : null;
    call.step = Number.isInteger(m.step) ? m.step : null;
    call.result = resultOf(m); // the wire's own object, read by shortResult/shownLabel
    if (call.statuses[call.statuses.length - 1] !== call.status) call.statuses.push(call.status);
    return true;
  }

  /** [{ turnId, calls: [call] }], newest turn first, turns without calls left out. */
  function view() {
    const out = [];
    for (let i = order.length - 1; i >= 0; i -= 1) {
      const turn = turns.get(order[i]);
      if (turn.calls.size === 0) continue;
      out.push({ turnId: turn.turnId, calls: [...turn.calls.values()] });
    }
    return out;
  }

  return { onMessage, view };
}
