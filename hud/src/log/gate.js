// The session-log gate (T-0043 criterion 19): what the checkbox is allowed to believe.
//
// No DOM here — `node --test tests/hud` imports this module and drives it with an injected
// `send`, exactly as T-0025's assist gate is driven. The element, the style and the click
// handler live in ./index.js.
//
// The rule the whole module exists for: **the server's echo settles the box, the click never
// does.** The server's default is ON (criterion 15), which is why the page may not guess it — a
// page that pre-ticked the box would be right most of the time and wrong exactly when the
// session is *not* being recorded (`--no-log`, or a log that disabled itself on an OSError,
// criterion 4). Someone would read a ticked box, speak freely believing there is a record, and
// find afterwards there is none. So this gate has three states, not two:
//
//   unknown   no astra.log.state has arrived   known() === false, on() === false, path() === null
//   waiting   a toggle went out, unechoed      pending() === true
//   settled   the server said so               on() is the server's flag, verbatim

/** The only client->server type this panel may emit (criterion 19). `protocol.js` CLIENT_TYPES is
 *  T-0008's statement about T-0008's code and is deliberately *not* widened (criterion 20). */
export const LOG_CLIENT_TYPES = ["astra.log.toggle"];
const [TOGGLE_TYPE] = LOG_CLIENT_TYPES;

/** The one server->client type this panel listens to (contracts/protocol.py AstraLogState). */
export const LOG_STATE_TYPE = "astra.log.state";

/**
 * createLogGate({ send, log }) -> gate
 *
 *   onState(message)   feed a decoded astra.log.state; nothing else confirms anything
 *   known()            false until the first astra.log.state has arrived
 *   on()               the server's flag (false while unknown — see known())
 *   pending()          a toggle went out whose echo has not arrived
 *   path()             the server's path string, verbatim; null while unknown
 *   stats()            { records, dropped, bytes, error }
 *   toggle(on)         -> true iff exactly one astra.log.toggle was handed to send
 */
export function createLogGate({ send, log = () => {} } = {}) {
  const post = typeof send === "function" ? send : null;
  let settled = false;
  let confirmed = false;
  let waiting = false;
  let pathText = null;
  let counters = { records: 0, dropped: 0, bytes: 0, error: null };

  return {
    onState(message) {
      // Strict on the type: the page's own toggle echoed back at it, an assist state, a bare
      // `{ on: true }` — none of them is the server's statement about the log, and none of them
      // may tick the box.
      if (!message || typeof message !== "object") return;
      if (message.type !== LOG_STATE_TYPE) return;
      if (typeof message.on !== "boolean") return;
      settled = true;
      waiting = false;
      confirmed = message.on;
      pathText = typeof message.path === "string" ? message.path : null;
      counters = {
        records: Number.isFinite(message.records) ? message.records : 0,
        dropped: Number.isFinite(message.dropped) ? message.dropped : 0,
        bytes: Number.isFinite(message.bytes) ? message.bytes : 0,
        error: typeof message.error === "string" ? message.error : null,
      };
    },

    known() {
      return settled;
    },

    on() {
      return settled && confirmed;
    },

    pending() {
      return waiting;
    },

    path() {
      return settled ? pathText : null;
    },

    stats() {
      return { ...counters };
    },

    toggle(on) {
      if (!post) {
        log("log toggle not sent: this host has no socket");
        return false;
      }
      const message = { type: TOGGLE_TYPE, on: Boolean(on) };
      // The request goes out now; nothing here changes what the box shows. `ws.js` rejects while
      // the socket is closed, and that is a log line, never a throw into the user's click.
      try {
        Promise.resolve(post(message)).catch((err) => {
          log(`log toggle not sent: ${err && err.message}`);
        });
      } catch (err) {
        log(`log toggle not sent: ${err && err.message}`);
        return false;
      }
      waiting = true;
      return true;
    },
  };
}
