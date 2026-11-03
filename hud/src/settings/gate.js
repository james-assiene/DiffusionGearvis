// T-0213 C: the Settings panel's egress gate: what the panel may send, and with which revision.
//
// No DOM here (`node --test tests/hud` can import it with an injected `send`). The pattern is
// ../assists/gate.js: the panel holds the server's last astra.settings.state and a change is a
// request, never a state; nothing is shown as changed until the server's echo says so.
//
// The one message is astra.settings.set {rev, changes, stop_turn} (A2a-1). `rev` is always the
// revision of the last state the server sent (the revision in force; a pending set does not move
// it), so a set made from a stale view is refused by the server as "stale", never applied.

/** The only client->server types the Settings panel may emit (T-0213 C). */
export const SETTINGS_CLIENT_TYPES = ["astra.settings.set"];
const [SET_TYPE] = SETTINGS_CLIENT_TYPES;

/** The one server->client type the gate reads: the echo of every setting (A2r-1). */
export const SETTINGS_STATE_TYPE = "astra.settings.state";

/**
 * createSettingsGate({ send, log }) -> gate
 *
 *   onState(message)          feed a decoded astra.settings.state; its rev is the one sets carry
 *   rev()                     the last state's rev, or null before any state
 *   set(changes, {stopTurn})  -> did a set go out? One message, the whole change set; never
 *                             before a state arrived, never with an empty or non-object change set
 *   lastKeys()                the keys of the last set that went out (where a "*" refusal belongs)
 */
export function createSettingsGate({ send, log = () => {} } = {}) {
  let rev = null;
  let lastKeys = [];

  const post = (message) => {
    try {
      return Promise.resolve(send(message)).catch((err) => {
        log(`settings change not sent: ${err && err.message}`);
      });
    } catch (err) {
      log(`settings change not sent: ${err && err.message}`);
      return Promise.resolve();
    }
  };

  return {
    onState(message) {
      if (!message || typeof message !== "object") return;
      if (message.type !== undefined && message.type !== SETTINGS_STATE_TYPE) return;
      if (Number.isInteger(message.rev) && message.rev >= 0) rev = message.rev;
    },

    rev() {
      return rev;
    },

    set(changes, { stopTurn = false } = {}) {
      if (typeof send !== "function") {
        log("settings change not sent: this host has no socket");
        return false;
      }
      if (rev === null) {
        log("settings change not sent: the session has not reported its settings yet");
        return false;
      }
      if (!changes || typeof changes !== "object" || Array.isArray(changes)) return false;
      const keys = Object.keys(changes);
      if (keys.length === 0) return false;
      const body = {};
      for (const key of keys) body[key] = changes[key];
      lastKeys = keys;
      post({ type: SET_TYPE, rev, changes: body, stop_turn: Boolean(stopTurn) });
      return true;
    },

    lastKeys() {
      return lastKeys.slice();
    },
  };
}
