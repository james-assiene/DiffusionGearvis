// The Restart gate (T-0162 D-1): what the header's Restart button is allowed to send.
//
// No DOM here — `node --test tests/hud` imports this module and drives it with an injected `send`,
// as the log and assist gates are driven. The button, the two-step inline confirm and their
// handlers live in ./index.js; the confirm is the page's half of "never by accident" (ruling
// H-T162-3), this gate is the wire's half: one reset in flight at a time.
//
// `reset()` hands exactly one astra.session.reset to `send` and then refuses until the server's
// astra.session.cleared arrives (`onCleared`), so a double-click on the confirm wipes once. A send
// that throws or rejects (ws.js rejects while the socket is closed) ends the flight: a stuck gate
// would be a dead button.
//
// The clear tracker is the page's one rule for "a new reset": every panel clears once per new
// `resets` value, live or replayed, and a repeated value (the hello's replay on a reconnect, a
// duplicate) clears nothing (D-2).

/** The only client->server type this panel may emit. `protocol.js` CLIENT_TYPES is not widened. */
export const SESSION_CLIENT_TYPES = ["astra.session.reset"];
const [RESET_TYPE] = SESSION_CLIENT_TYPES;

/** The server->client echo of a completed reset (contracts/protocol.py AstraSessionCleared). */
export const SESSION_CLEARED_TYPE = "astra.session.cleared";

/** The confirm's sentence (H-T162-3, R6, orchestrator ruling 2026-10-27 (c)): it names the memory
 *  wipe and the agent browser's return to its start page, because neither can be undone. */
export const CONFIRM_TEXT =
  "Start a new session? This clears the conversation and everything Gearvis remembers, " +
  "stops screen sharing, and sends Gearvis's own browser, if it is open, back to its start " +
  "page. Session logs already written stay on disk.";

/** The `resets` count of a well-formed astra.session.cleared, else null. */
export function resetsOf(message) {
  if (!message || typeof message !== "object" || message.type !== SESSION_CLEARED_TYPE) return null;
  return Number.isInteger(message.resets) && message.resets >= 1 ? message.resets : null;
}

/** The seed a panel's tracker starts from: the page's last accepted `resets`, or null. */
export function seedFrom(host) {
  if (!host || typeof host.lastResets !== "function") return null;
  try {
    return host.lastResets();
  } catch {
    return null;
  }
}

/**
 * createClearTracker(seed?) -> { accept(message) -> boolean, last() -> number|null }
 *
 * accept() is true exactly once per new `resets` value of an astra.session.cleared, live or
 * replayed; false for any other message and for the value accepted last.
 */
export function createClearTracker(seed = null) {
  // Seeded from the page's own tracker (host.lastResets(), main.js) by a panel that mounts late:
  // a cleared(n) replayed before it mounted is not new to it when the next reconnect replays n.
  let lastResets = Number.isInteger(seed) && seed >= 1 ? seed : null;
  return {
    accept(message) {
      const resets = resetsOf(message);
      if (resets === null || resets === lastResets) return false;
      lastResets = resets;
      return true;
    },
    last() {
      return lastResets;
    },
  };
}

/**
 * createSessionGate({ send, log }) -> gate
 *
 *   reset()        hand exactly one { type: "astra.session.reset" } to send; true iff it did.
 *                  While one is in flight (sent, not yet echoed) it sends nothing and is false.
 *   onCleared(m)   feed a decoded server message; an astra.session.cleared ends the flight
 *   pending()      a reset went out whose astra.session.cleared has not arrived
 */
export function createSessionGate({ send, log = () => {} } = {}) {
  const post = typeof send === "function" ? send : null;
  let flight = 0; // 0: nothing in flight; otherwise the id of the reset in flight

  let next = 0;
  return {
    reset() {
      if (flight !== 0) return false;
      if (!post) {
        log("restart not sent: this host has no socket");
        return false;
      }
      next += 1;
      const mine = next;
      let result;
      try {
        result = post({ type: RESET_TYPE });
      } catch (err) {
        log(`restart not sent: ${err && err.message}`);
        return false;
      }
      flight = mine;
      Promise.resolve(result).catch((err) => {
        log(`restart not sent: ${err && err.message}`);
        if (flight === mine) flight = 0;
      });
      return true;
    },

    onCleared(message) {
      if (resetsOf(message) !== null) flight = 0;
    },

    pending() {
      return flight !== 0;
    },

    /** The socket closed: a reset still unechoed may never have arrived, so the flight ends (the
     *  next hello's replayed astra.session.cleared says whether it did). */
    abandon() {
      flight = 0;
    },
  };
}
