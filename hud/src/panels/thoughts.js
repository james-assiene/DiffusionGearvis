// The "Model thoughts" disclosure of one turn (T-0180 group C): a native <details>, collapsed by
// default, holding one labelled block per attempt of the turn's requests.
//
// What it renders, and from what: astra.thought only (C6). Each message carries the attempt's
// cumulative committed thought text (contracts/protocol.py AstraThought), so the latest message
// of an attempt replaces its block's text; nothing is appended and nothing is derived from canvas
// frames, so a draft is never shown as a thought. Once a block is final, a later `open` message
// (a stale or replayed one) is ignored (C4).
//
// The page never writes `open` (C2): only the viewer's click or keyboard on the native <summary>
// toggles it, and an update keeps whatever state the viewer left. The <details>, its <summary> and
// every block are built once and updated in place (C1). Model text is written with textContent
// only, never as markup.
//
// No DOM at module top level: `node --test tests/hud` imports this module.

export const THOUGHT_TYPE = "astra.thought";

const KIND_SUFFIX = {
  step: "",
  reask: " · re-ask",
  retry: " · retry",
  repair: " · repair",
  reground: " · ReGround",
};

const STATUS_LINES = {
  open: "thinking…",
  done: "",
  stopped_repeating: "stopped: repeating",
  interrupted: "interrupted",
  error: "stopped: error",
};

const SUMMARY = "Model thoughts";

/** "Step 2 · re-ask": the session's step number and the reason for the attempt (B6). */
export function thoughtLabel(step, kind) {
  const suffix = Object.prototype.hasOwnProperty.call(KIND_SUFFIX, kind) ? KIND_SUFFIX[kind] : "";
  return `Step ${step}${suffix}`;
}

/** The status line of an attempt's end; a normal end (`done`) has none. */
export function thoughtStatus(end) {
  return Object.prototype.hasOwnProperty.call(STATUS_LINES, end) ? STATUS_LINES[end] : "";
}

/** The honest line for text past the cap (H-T180-5); none when nothing was omitted. */
export function omittedLine(omittedChars) {
  if (!Number.isInteger(omittedChars) || omittedChars <= 0) return "";
  return `… ${omittedChars} more characters not shown (full text in the session log)`;
}

const isId = (v) => typeof v === "string" && v.length > 0;

/** Write `text` into `node` only when it differs, so a replay changes nothing. */
function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

/**
 * createThoughtDisclosure(doc) -> { element, handle(message) }
 *
 *   element     the turn's <details data-model-thoughts>; the caller places it once.
 *   handle(m)   one decoded astra.thought of this turn; every other type is ignored.
 */
export function createThoughtDisclosure(doc) {
  const details = doc.createElement("details");
  details.className = "model-thoughts";
  details.setAttribute("data-model-thoughts", "");
  const summary = doc.createElement("summary");
  summary.className = "thoughts-summary";
  summary.textContent = SUMMARY;
  details.appendChild(summary);

  const blocks = new Map(); // attempt_id -> the block's parts, in first-seen order

  function newBlock(attemptId) {
    const block = doc.createElement("div");
    block.className = "thought-block";
    block.setAttribute("data-thought-attempt", attemptId);
    const head = doc.createElement("div");
    head.className = "thought-head";
    const label = doc.createElement("span");
    label.className = "thought-label";
    label.setAttribute("data-thought-label", "");
    const status = doc.createElement("span");
    status.className = "thought-status";
    status.setAttribute("data-thought-status", "");
    head.append(label, status);
    const text = doc.createElement("div");
    text.className = "thought-text";
    text.setAttribute("data-thought-text", "");
    const omitted = doc.createElement("p");
    omitted.className = "thought-omitted";
    omitted.setAttribute("data-thought-omitted", "");
    block.append(head, text, omitted);
    details.appendChild(block);
    const parts = { label, status, text, omitted, final: false };
    blocks.set(attemptId, parts);
    const n = blocks.size;
    setText(summary, n > 1 ? `${SUMMARY} · ${n} steps` : SUMMARY);
    return parts;
  }

  return {
    element: details,
    handle(m) {
      if (!m || m.type !== THOUGHT_TYPE || !isId(m.attempt_id)) return;
      const parts = blocks.get(m.attempt_id) || newBlock(m.attempt_id);
      const end = typeof m.end === "string" ? m.end : "open";
      if (parts.final && end === "open") return; // C4: a final block ignores a later open
      parts.final = end !== "open";
      setText(parts.label, thoughtLabel(m.step, m.kind));
      setText(parts.text, typeof m.text === "string" ? m.text : "");
      const status = thoughtStatus(end);
      setText(parts.status, status);
      if (parts.status.getAttribute("data-thought-status") !== end) parts.status.setAttribute("data-thought-status", end);
      setText(parts.omitted, omittedLine(m.omitted_chars));
    },
  };
}
