// What happened to the interface this turn (T-0054 group B, criteria 14 and 15).
//
// One `astra.ui.fence` in, one `{ status, label, canvases, refused }` out. No DOM, no clock, no
// state: `node --test tests/hud` imports this module and the contract rows run it out of process,
// so everything here is a pure function of its argument. The DOM half is ./index.js.
//
// The rule this module is written around (T-0054 D5, the honesty rule): **the page echoes, it
// does not derive.** The server says how the fence of a turn ended; this module turns that into
// words and nothing else. It never accumulates across messages, never computes elapsed time and
// never infers a state from the *absence* of a message — the notice is deliberately not replayed
// (criterion 8), so a module that remembered would contradict the server exactly when a page
// reloads mid-turn.
//
// The status table is criterion 15's, verbatim. `status` is what tests and screenshots key on;
// `label` is English the human may still move (ambiguity 4). They are separate on purpose.
//
//   server state                                          status          the pane says
//   no message                                            none            nothing
//   close written|inferred, not continued, canvases 1     whole           nothing
//   close written|inferred, continued (or > 1 canvas)     reassembled     reassembled from N canvases
//   close open                                            incomplete      the interface was cut off …
//   close abandoned                                       abandoned       the interface never finished …
//   close null, speech_refused > 0                        refused-only    … was not spoken; nothing was shown
//   anything this page does not know                      unknown         a refusal to guess
//
// A refusal count above zero always reaches the label (criterion 15's last row: "with any close",
// and `null` is a close — ruling B7): `refused-only` names it inside its own sentence, every other
// status gets it as a second clause. `whole` with a refusal is therefore not silent — the fence
// was fine and something else was not.

/** The one server->client type this module reads (contracts/protocol.AstraUiFence). */
export const FENCE_TYPE = "astra.ui.fence";

/** contracts/llm.py FenceClose — the four values, and nothing else is a close. */
const WRITTEN = "written";
const INFERRED = "inferred";
const OPEN = "open";
const ABANDONED = "abandoned";
const CLOSES = [WRITTEN, INFERRED, OPEN, ABANDONED];

const isCount = (value) => typeof value === "number" && Number.isInteger(value) && value >= 0;

/** The second clause: what the voice guard refused, counted by the server (T-0054 D4). */
function refusalClause(refused) {
  const pieces = refused === 1 ? "1 piece" : `${refused} pieces`;
  const verb = refused === 1 ? "was" : "were";
  return `${pieces} of interface data ${verb} not spoken`;
}

function labelFor(status, canvases, refused) {
  switch (status) {
    case "reassembled":
      return `reassembled from ${canvases} canvases`;
    case "incomplete":
      return "the interface was cut off — showing what arrived";
    case "abandoned":
      return "the interface never finished — what you see is incomplete";
    case "refused-only":
      // Criterion 15's last row — "speech_refused > 0 **with any close**, plus the count" — and
      // `null` is a close (ruling B7). The sentence names the count itself rather than carrying
      // the clause a second time after it.
      return (
        `part of the reply was interface data and was not spoken (${refusalClause(refused)}); ` +
        "nothing was shown"
      );
    case "unknown":
      return "this page does not understand what happened to the interface";
    default: // "none" and "whole" say nothing at all about the fence.
      return "";
  }
}

/**
 * describeFence(message) -> { status, label, canvases, refused }
 *
 * Total: whatever it is handed — nothing, a number, another message, a field of the wrong type,
 * a `close` value from a future contract — it answers and never throws. A panel that throws out
 * of a socket handler takes the page's message loop with it (T-0008 criterion 10).
 *
 * A message it cannot read is `unknown`, never `whole` and never `reassembled`: a page that
 * renders a value it does not know as success is how the panel starts lying again.
 */
export function describeFence(message) {
  if (message === undefined || message === null) {
    // Not a state reached by remembering that nothing arrived: it is the answer to being handed
    // nothing. T-0008's placeholder stands.
    return { status: "none", label: "", canvases: 0, refused: 0 };
  }

  const unknown = (canvases, refused) => ({
    status: "unknown",
    label: labelFor("unknown"),
    canvases,
    refused,
  });

  if (typeof message !== "object" || Array.isArray(message)) return unknown(0, 0);
  if (message.type !== FENCE_TYPE) return unknown(0, 0);

  const canvases = isCount(message.canvases) ? message.canvases : null;
  const refused =
    message.speech_refused === undefined
      ? 0 // the contract's default
      : isCount(message.speech_refused)
        ? message.speech_refused
        : null;
  const close = message.close === undefined ? null : message.close;
  const continued = message.continued === undefined ? false : message.continued;
  const known =
    canvases !== null &&
    refused !== null &&
    typeof continued === "boolean" &&
    (close === null || CLOSES.includes(close));
  if (!known) return unknown(canvases === null ? 0 : canvases, refused === null ? 0 : refused);

  let status;
  if (close === OPEN) status = "incomplete";
  else if (close === ABANDONED) status = "abandoned";
  else if (close === WRITTEN || close === INFERRED)
    // Criterion 15 gives `whole` one canvas and no carry; more than one canvas produced what is
    // on screen, so the pane says so rather than calling it whole.
    status = continued || canvases > 1 ? "reassembled" : "whole";
  else if (refused > 0) status = "refused-only";
  else status = "none"; // no fence was opened and nothing was refused: nothing to report.

  let label = labelFor(status, canvases, refused);
  // `refused-only`'s own sentence already names the count (ruling B7), so it is not appended
  // twice; every other status gets the clause after its own line.
  if (refused > 0 && status !== "refused-only") {
    label = label ? `${label} — ${refusalClause(refused)}` : refusalClause(refused);
  }
  return { status, label, canvases, refused };
}
