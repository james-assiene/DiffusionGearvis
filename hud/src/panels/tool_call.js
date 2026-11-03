// The action card (T-0090 criteria 4 and 5): what the assistant would do, one card per call_id.
//
// In v1 nothing is executed (T-0080d H1: dry run only) and a search query is never fetched or
// linked (H2), so the card is display-only: it builds plain elements, writes model text with
// textContent and nothing else, registers no handler and sends nothing back. A later message for
// the same call_id updates its card in place, in wire order (proposed -> repairing -> dry_run |
// refused | cancelled); the line is rebuilt from the latest arguments, never cleared.
//
// No DOM at module top level: `node --test tests/hud` imports this module.

export const TOOL_CALL_TYPE = "astra.tool.call";

/** Ruling U10, mirroring hud_server.SUPERSEDED_TURNS: turns whose cards were replaced. */
const SUPERSEDED_TURNS = 8;

const DRY_RUN_NOTICE = "dry run — nothing was done";

const STATUS_LABELS = {
  proposed: "proposed",
  repairing: "repairing",
  dry_run: "dry run",
  executed: "executed",
  refused: "refused",
  cancelled: "cancelled",
};

const SOURCE_LABELS = { draft: "from the draft", repair: "repaired" };

/** A coordinate or a word from the arguments, shown as text; a missing one reads "?". */
const arg = (args, key) => {
  const v = args[key];
  return v === undefined || v === null ? "?" : String(v);
};

/** The one "Would …" line per tool (criterion 4; x and y are the 0–999 grid, ruling U1). */
export function toolLine(name, args) {
  const a = args && typeof args === "object" ? args : {};
  switch (name) {
    case "click_at":
      return `Would click at (${arg(a, "x")}, ${arg(a, "y")}) on screen`;
    case "type_at":
      return (
        `Would type "${arg(a, "text")}" at (${arg(a, "x")}, ${arg(a, "y")})` +
        (a.press_enter === true ? " and press Enter" : "")
      );
    case "scroll":
      return `Would scroll ${arg(a, "direction")}`;
    case "press_key":
      return `Would press ${arg(a, "combo")}`;
    case "google_search":
      return `Would search for: ${arg(a, "query")}`;
    default:
      return `Would call ${String(name)} ${JSON.stringify(a)}`;
  }
}

/**
 * Mount the card list into `root`. Returns { handle(message) }; only astra.tool.call is acted on,
 * every other message is ignored.
 *
 * `listFor(turnId)` (optional) names the element a turn's cards go in — the thread passes the
 * action list of that turn's article (T-0154 B7). It is asked only for a message that is accepted,
 * so a superseded turn's late message never opens anything. Without it every card goes in `root`.
 *
 * T-0154 B7 (amends T-0090's "replace on a new turn" by name): earlier turns keep their cards as
 * history. U7/U10 still hold: a message for a superseded turn is ignored — its card is neither
 * updated nor joined by a new one.
 */
export function createToolCallPanel(root, { listFor } = {}) {
  const cards = new Map(); // turn_id + call_id -> the card's parts
  let turnId = null;
  const superseded = []; // the last SUPERSEDED_TURNS turn ids before the current one
  const containerFor = (turn) => {
    const el = typeof listFor === "function" ? listFor(turn) : null;
    return el || root;
  };

  function part(doc, tag, className, hook) {
    const el = doc.createElement(tag);
    el.className = className;
    if (hook) el.dataset[hook] = "";
    return el;
  }

  function build(callId, list) {
    const doc = root.ownerDocument;
    const card = part(doc, "div", "tool-card");
    card.dataset.toolCallId = callId;
    const head = part(doc, "div", "tool-head");
    const status = part(doc, "span", "tool-status");
    const source = part(doc, "span", "tool-source");
    head.append(status, source);
    const line = part(doc, "p", "tool-line", "toolLine");
    const warning = part(doc, "p", "tool-warning", "toolWarning");
    const reason = part(doc, "p", "tool-reason", "toolReason");
    const notice = part(doc, "p", "tool-notice", "toolDryRun");
    card.append(head, line, warning, reason, notice);
    list.appendChild(card);
    return { card, status, source, line, warning, reason, notice };
  }

  /** Show `text` in `el`, or empty and hide it (a hidden element's text still counts as text). */
  function show(el, text) {
    el.textContent = text || "";
    el.hidden = !text;
  }

  function handle(m) {
    if (!m || typeof m !== "object" || m.type !== TOOL_CALL_TYPE) return;
    if (typeof m.call_id !== "string") return;
    if (m.turn_id !== turnId) {
      // Ruling U10 (first-seen order: `turnId` is the newest, `superseded` the 8 before it): a
      // message of a superseded turn is ignored; the first card of a never-seen turn becomes
      // current. The earlier turns' cards stay where they are, as history (T-0154 B7).
      if (superseded.includes(m.turn_id)) return;
      if (turnId !== null) {
        superseded.push(turnId);
        if (superseded.length > SUPERSEDED_TURNS) superseded.shift();
      }
      turnId = m.turn_id;
    }
    // Keyed by turn and call: a history card is never re-addressed by another turn's call.
    const key = `${m.turn_id}\u0000${m.call_id}`;
    let parts = cards.get(key);
    if (!parts) {
      parts = build(m.call_id, containerFor(m.turn_id));
      cards.set(key, parts);
    }
    parts.status.dataset.toolStatus = String(m.status);
    parts.status.textContent = STATUS_LABELS[m.status] || String(m.status);
    parts.source.dataset.toolSource = String(m.source);
    parts.source.textContent = SOURCE_LABELS[m.source] || String(m.source);
    parts.line.textContent = toolLine(m.name, m.arguments);
    show(parts.warning, m.query_problem ? `warning: ${m.query_problem}` : "");
    show(parts.reason, m.reason ? `reason: ${m.reason}` : "");
    show(parts.notice, m.status === "dry_run" ? DRY_RUN_NOTICE : ""); // ruling U2: iff dry_run
  }

  /** T-0162 D-2: every card goes. The current turn joins the superseded ones, so a late message of
   *  a pre-reset turn opens nothing (turn ids never repeat after a reset). */
  function clear() {
    for (const parts of cards.values()) parts.card.remove();
    cards.clear();
    if (turnId !== null) {
      superseded.push(turnId);
      if (superseded.length > SUPERSEDED_TURNS) superseded.shift();
    }
    turnId = null;
  }

  return { handle, clear };
}
