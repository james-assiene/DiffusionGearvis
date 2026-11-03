// The assist egress gate (T-0025): what may leave this page, and when.
//
// No DOM here — `node --test tests/hud` imports this module and drives it with an injected
// `send`, `now` and extractor set (docs/tasks/T-0025.md implementation notes: "keep the pure part
// DOM-free"). The rows, the badge element and the counters live in ./index.js.
//
// The rule the whole module exists for (criterion 5, CLAUDE.md "off unless the user turns them
// on"): assist text leaves the page only for a kind the *server* confirmed in an
// `astra.assist.state`. A tick in a checkbox is a request, not a state; an unechoed request sends
// nothing. Unticking is the asymmetric half (ruling A6): it takes effect here immediately, so text
// extracted while the box was on never reaches the socket afterwards.

import { MAX_FPS } from "../frames.js";

/** The only client->server types this panel may emit (criterion 10). `protocol.js` CLIENT_TYPES
 *  is T-0008's statement about T-0008's code and is deliberately *not* widened (criterion 12). */
export const ASSIST_CLIENT_TYPES = ["astra.assist.toggle", "astra.frame.text"];
const [TOGGLE_TYPE, TEXT_TYPE] = ASSIST_CLIENT_TYPES;

/** The one server->client type this panel listens to: the echo of the effective flags (docs/02 §5). */
export const ASSIST_STATE_TYPE = "astra.assist.state";

/** `tuple(AssistFlags.model_fields)` — declaration order, mirrored from
 *  local_astra/contracts/perception.py and pinned by tests/unit/test_hud_constants_parity.py.
 *  It is the order the rows are shown in and the order `AssistState.texts_for` returns. */
export const ASSIST_KINDS = ["page_text", "pdf_text", "cells", "at_spi"];

/** UTF-8 bytes, the truncation marker included (ruling A1). `AstraFrameText.text` is unbounded and
 *  the codec envelope is 8 MiB; this cap keeps one assist text from filling it. */
export const MAX_TEXT_BYTES = 32768;

/** In-band, because `AstraFrameText` has no `truncated` field and adding one is a contract task:
 *  a silent truncation would let the model claim it read a whole document (criterion 9). */
export const TRUNCATION_MARKER =
  "\n[truncated by the HUD assist panel: the rest of this text was cut at the byte cap]";

/** Why a kind cannot be turned on here, when the registry gives no better reason (criterion 8). */
const DEFAULT_REASONS = {
  page_text: "no page-text extractor for this capture source: a shared tab hands this page a video track, not a document",
  pdf_text: "no PDF text layer is reachable from a browser tab; this kind arrives with the native daemon (docs/01 D6, M2+)",
  cells: "no spreadsheet cells are reachable from a browser tab; this kind arrives with the native daemon (docs/01 D6, M2+)",
  at_spi: "no accessibility tree is reachable from a browser tab; this kind arrives with the native daemon (docs/01 D6, M2+)",
};

const ENCODER = new TextEncoder();

const utf8SizeOf = (codePoint) =>
  codePoint < 0x80 ? 1 : codePoint < 0x800 ? 2 : codePoint < 0x10000 ? 3 : 4;

/**
 * Cut `text` to `maxBytes` **UTF-8 bytes**, marker included, on a code-point boundary.
 *
 * `String.prototype.slice` counts UTF-16 units, which is neither bytes (a snowman is 3 bytes in
 * one unit) nor characters (an astral character is 4 bytes in *two* units, and half of one is a
 * lone surrogate the encoder replaces with U+FFFD) — T-0008 ruling R11.
 *
 * -> { text, truncated }
 */
export function truncateUtf8(text, maxBytes = MAX_TEXT_BYTES) {
  const input = typeof text === "string" ? text : String(text === undefined || text === null ? "" : text);
  if (ENCODER.encode(input).length <= maxBytes) return { text: input, truncated: false };
  // The marker is part of the budget, so the result is never over the cap (ruling A1). A cap
  // smaller than the marker itself leaves nothing to keep; the marker still goes out.
  const budget = Math.max(0, maxBytes - ENCODER.encode(TRUNCATION_MARKER).length);
  let bytes = 0;
  let kept = "";
  for (const character of input) {
    const size = utf8SizeOf(character.codePointAt(0));
    if (bytes + size > budget) break;
    bytes += size;
    kept += character;
  }
  return { text: kept + TRUNCATION_MARKER, truncated: true };
}

/**
 * createAssistGate({ send, now, extractors, reasons, maxFps, log }) -> gate
 *
 *   extractors  { [kind]: ({ source, ts_ms }) -> string | null | Promise<string|null> }
 *   reasons     { [kind]: string } — why a kind has no extractor here (criterion 8)
 *
 *   onState(message)     feed a decoded astra.assist.state; nothing else confirms a flag
 *   flags()              the last server-confirmed flag set (all four keys)
 *   badge()              the server's badge string, verbatim ("" before any state)
 *   pending()            kinds whose toggle has not been echoed yet
 *   available()          { [kind]: { available, reason } }
 *   toggle(kind, on)     -> did a toggle go out? (a whole flag set, never a per-kind merge)
 *   offer(source)        -> Promise<{ sent: [kind], dropped: [kind] }>
 *   stats()              { sent, dropped, truncated, inFlight }
 *
 * Invariants, in lockstep with hud/src/frames.js: at most one extraction in flight per kind and at
 * most one message per kind per 1000 / MAX_FPS ms. An offer that arrives while either is true is
 * *dropped and counted*, never queued — a queue would replay stale text after the window, and
 * (criterion 6) possibly after the box was unticked.
 */
export function createAssistGate({
  send,
  now = () => Date.now(),
  extractors = {},
  reasons = {},
  maxFps = MAX_FPS,
  log = () => {},
} = {}) {
  const minGapMs = 1000 / maxFps;
  const confirmed = {};
  const desired = {};
  for (const kind of ASSIST_KINDS) {
    confirmed[kind] = false;
    desired[kind] = false;
  }
  let badgeText = "";
  let sent = 0;
  let dropped = 0;
  let truncated = 0;

  const perKind = new Map();
  const stateOf = (kind) => {
    let state = perKind.get(kind);
    if (!state) {
      state = { busy: false, lastAt: -Infinity };
      perKind.set(kind, state);
    }
    return state;
  };

  const hasExtractor = (kind) => typeof extractors[kind] === "function";
  const reasonFor = (kind) => {
    const given = reasons[kind];
    return typeof given === "string" && given.trim() ? given : DEFAULT_REASONS[kind];
  };

  /** Hand a message to the socket *now* (a toggle must be on the wire before the click returns),
   *  and turn a refusal — `ws.js` rejects while the socket is closed — into a drop, never a throw
   *  into the caller's click handler or extraction loop. */
  const post = (message) => {
    try {
      return Promise.resolve(send(message)).catch((err) => {
        log(`assist message not sent: ${err && err.message}`);
        throw err;
      });
    } catch (err) {
      log(`assist message not sent: ${err && err.message}`);
      return Promise.reject(err);
    }
  };

  async function run(kind, source, at) {
    const state = stateOf(kind);
    state.busy = true;
    state.lastAt = at;
    try {
      const extracted = await extractors[kind]({ source, ts_ms: at });
      // Criterion 6: the flags may have changed while the extractor worked. Text extracted while
      // a kind was on must not reach send() after the untick, even though the send was scheduled.
      if (!confirmed[kind]) {
        log(`assist text for ${kind} dropped: the kind is no longer on`);
        return "dropped";
      }
      if (typeof extracted !== "string" || extracted === "") return "dropped";
      const capped = truncateUtf8(extracted);
      if (capped.truncated) truncated += 1;
      await post({ type: TEXT_TYPE, source, ts_ms: at, kind, text: capped.text });
      return "sent";
    } catch (err) {
      log(`assist extraction for ${kind} failed: ${err && err.message}`);
      return "dropped";
    } finally {
      state.busy = false;
    }
  }

  const gate = {
    onState(message) {
      if (!message || typeof message !== "object") return;
      if (message.type !== undefined && message.type !== ASSIST_STATE_TYPE) return;
      const flags = message.flags;
      if (!flags || typeof flags !== "object") return;
      for (const kind of ASSIST_KINDS) {
        confirmed[kind] = Boolean(flags[kind]);
        desired[kind] = confirmed[kind]; // the echo arrived: nothing is pending any more
      }
      // Criterion 4: the badge is the server's string, character for character. The page holds no
      // second derivation of it and reconciles nothing against the flags beside it.
      badgeText = typeof message.badge === "string" ? message.badge : "";
    },

    flags() {
      return { ...confirmed };
    },

    badge() {
      return badgeText;
    },

    pending() {
      return ASSIST_KINDS.filter((kind) => desired[kind] !== confirmed[kind]);
    },

    available() {
      const out = {};
      for (const kind of ASSIST_KINDS) {
        out[kind] = { available: hasExtractor(kind), reason: reasonFor(kind) };
      }
      return out;
    },

    toggle(kind, on) {
      if (!ASSIST_KINDS.includes(kind)) return false;
      const wanted = Boolean(on);
      // Criterion 8: no click path on a kind this client cannot honour raises its flag. Lowering
      // one is always allowed — turning an assist off can never need a capability.
      if (wanted && !hasExtractor(kind)) {
        log(`${kind} cannot be turned on here: ${reasonFor(kind)}`);
        return false;
      }
      desired[kind] = wanted;
      if (!wanted) confirmed[kind] = false; // ruling A6: the untick takes effect before the echo
      const flags = {};
      for (const other of ASSIST_KINDS) {
        flags[other] = other === kind ? wanted : Boolean(confirmed[other]);
      }
      post({ type: TOGGLE_TYPE, flags }).catch(() => {});
      return true;
    },

    offer(source) {
      const at = now();
      const outcome = { sent: [], dropped: [] };
      const running = [];
      for (const kind of ASSIST_KINDS) {
        if (!confirmed[kind]) continue; // criterion 5: the server's flags, never the checkbox
        if (!hasExtractor(kind)) continue; // criterion 8: nothing invented for a kind we cannot read
        const state = stateOf(kind);
        if (state.busy || at - state.lastAt < minGapMs) {
          dropped += 1;
          outcome.dropped.push(kind);
          continue;
        }
        running.push(
          run(kind, source, at).then((result) => {
            if (result === "sent") sent += 1;
            else dropped += 1;
            outcome[result].push(kind);
          }),
        );
      }
      return Promise.all(running).then(() => outcome);
    },

    stats() {
      let inFlight = 0;
      for (const state of perKind.values()) if (state.busy) inFlight += 1;
      return { sent, dropped, truncated, inFlight };
    },
  };
  return gate;
}
