// The typed-question panel, DOM-free half (T-0039 criteria 11-13).
//
// Everything that decides what may leave the page lives here, so `node --test tests/hud` drives
// it without a browser: the strip, the cap, the closed socket and the running turn. Each refusal
// is the page's own; the session's busy answer and the codec's malformed answer are backstops,
// not the normal path (docs/tasks/T-0039.md criterion 12).
//
// No DOM at module top level; ./index.js is the half that touches elements.

/** The panel's own client->server whitelist (criterion 11). `hud/src/protocol.js` CLIENT_TYPES
 *  stays T-0008's two entries — the per-directory scan in tests/hud/protocol.test.mjs and
 *  tests/unit/test_hud_constants_parity.py are the two halves of that oracle. */
export const ASK_CLIENT_TYPES = ["astra.text"];

/** Code points, mirroring contracts/protocol.AstraText (criterion 2). Not UTF-16 units and not
 *  bytes: 2000 astral characters are 4000 units and are still one legal question. */
export const MAX_TEXT_CHARS = 2000;

const TURN_STATE = "astra.turn.state";
const IDLE = "IDLE";

/** Code points, which is what the contract counts. */
const codePoints = (text) => [...text].length;

/**
 * createAskForm({ send, isOpen, log }) -> { submit, onMessage, enabled }
 *
 *   submit(text) -> boolean   true iff exactly one astra.text was handed to `send`
 *   onMessage(message)        every decoded astra.* message (the panel host's subscribe)
 *   enabled() -> boolean      false while the last astra.turn.state is not IDLE
 */
export function createAskForm({ send, isOpen = () => true, log = () => {} } = {}) {
  let idle = true; // the page starts IDLE and ready for a question

  function submit(text) {
    const question = typeof text === "string" ? text.trim() : "";
    if (!question) {
      log("nothing to ask: the box is empty");
      return false;
    }
    if (codePoints(question) > MAX_TEXT_CHARS) {
      log(
        `that question is ${codePoints(question)} characters; the limit is ${MAX_TEXT_CHARS} ` +
          "(contracts/protocol.AstraText)",
      );
      return false;
    }
    if (!idle) {
      log("a turn is already running: the question was not sent");
      return false;
    }
    if (!isOpen()) {
      log("the socket is not open: the question was not sent");
      return false;
    }
    if (typeof send !== "function") {
      log("this host has no socket: the question was not sent");
      return false;
    }
    // One send per submit, and the message contracts/protocol.AstraText describes: nothing else.
    const sent = send({ type: ASK_CLIENT_TYPES[0], text: question });
    if (sent && typeof sent.then === "function") {
      // ws.js rejects while the socket is closing; that is a line, never an exception thrown at
      // the submit handler (T-0008 criterion 10: a panel never takes the page down).
      sent.then(undefined, (err) =>
        log(`the question was not sent: ${(err && err.message) || err}`),
      );
    }
    return true;
  }

  function onMessage(message) {
    // Criterion 13: the state the control follows is astra.turn.state and nothing else — a busy
    // error is the error strip's business and changes nothing here.
    if (!message || message.type !== TURN_STATE) return;
    idle = message.state === IDLE;
  }

  return { submit, onMessage, enabled: () => idle };
}
