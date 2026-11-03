// The reducer behind the two channels, the turn chip and the thinking meter (criteria 8, 9, 11).
// Pure: no DOM, no clock of its own — the page clock is an argument, exactly as the turn machine
// takes time as an event (docs/02 §4). reduce(state, message, nowMs) -> new state.
//
// The played/unplayed (struck-through) distinction comes from the wire and from nowhere else
// (T-0030 criteria 17, 18 and 22): astra.spoken.played carries, once per playback run, how much
// of each utterance the device actually rendered. It is never derived from timing — no elapsed
// page clock, no INTERRUPTED turn state and no utterance length dims a single word.
//
// A message marked `replay: true` (T-0037) is a copy of one the server already published for the
// open turn, sent again at the hello of a (re)connecting page. It is reduced exactly like the
// message it copies and nothing here branches on the marker: the reducer is idempotent instead —
// one entry per utterance_id (withUtterance) and one transcript line per turn_id (withTranscript)
// — so a page whose state survived the socket (ws.js's backoff reconnect) ends where it was
// rather than with every line doubled.

/** docs/01 D11: the ~2000-token thinking budget, and the interim-update point. */
export const THINKING_MARKS_MS = [3000, 4000];

export function initialState() {
  return {
    turnState: null,
    turnId: null,
    /** { turnId, baseMs, atMs } while the server says THINKING, else null */
    thinking: null,
    /** [{ turn_id, utterances: [{ utterance_id, text }] }] in arrival order; an utterance the
     *  server measured also carries { played_words, complete, played_text } (T-0030) */
    spoken: [],
    /** [{ text, final, turn_id }] — the user's transcript, one line per turn */
    transcript: [],
    /** turn ids the server marked INTERRUPTED */
    interrupted: [],
    /** every turn id seen, so a late message for a finished turn cannot rewind the chip */
    seenTurns: [],
    /** the error strip: astra.error code/message */
    errors: [],
    /** T-0162 D-2: the `resets` of the last astra.session.cleared reduced, null before any */
    resets: null,
  };
}

const MAX_ERRORS = 5;

export function reduce(state, message, nowMs) {
  if (!message || typeof message.type !== "string") return state;
  switch (message.type) {
    case "astra.turn.state":
      return reduceTurnState(state, message, nowMs);
    case "astra.spoken":
      return { ...state, spoken: withUtterance(state.spoken, message) };
    case "astra.spoken.played":
      return withPlayed(state, message);
    case "astra.transcript":
      return { ...state, transcript: withTranscript(state.transcript, message) };
    case "astra.session.cleared":
      return reduceCleared(state, message);
    case "astra.error":
      return {
        ...state,
        errors: [...state.errors, { code: message.code, message: message.message }].slice(
          -MAX_ERRORS,
        ),
      };
    default:
      return state; // decoded and ignored: the surface, the canvas and the assists are not ours
  }
}

/** T-0162 D-2: a session reset, live or replayed, clears the transcript, the spoken and played
 *  state and the thinking meter once per new `resets` value; a repeated value returns the state
 *  unchanged. Errors, the turn chip and the turns already seen are kept: turn ids never repeat
 *  after a reset, so a late message for a pre-reset turn still cannot rewind the chip. */
function reduceCleared(state, message) {
  const resets = Number.isInteger(message.resets) && message.resets >= 1 ? message.resets : null;
  if (resets === null || resets === state.resets) return state;
  return { ...state, resets, thinking: null, spoken: [], transcript: [], interrupted: [] };
}

function reduceTurnState(state, message, nowMs) {
  const turnId = message.turn_id;
  const stale = turnId !== state.turnId && state.seenTurns.includes(turnId);
  if (stale) return state; // a late message for a finished turn does not rewind the chip
  const next = {
    ...state,
    turnState: message.state,
    turnId,
    seenTurns: state.seenTurns.includes(turnId) ? state.seenTurns : [...state.seenTurns, turnId],
  };
  // EVERY message snaps the meter to the server value, including a smaller one: the page may
  // extrapolate between messages, but it never invents time the server did not report.
  next.thinking =
    message.state === "THINKING"
      ? { turnId, baseMs: Number(message.thinking_ms) || 0, atMs: nowMs }
      : null;
  if (message.state === "INTERRUPTED" && !state.interrupted.includes(turnId)) {
    next.interrupted = [...state.interrupted, turnId];
  }
  return next;
}

function withUtterance(groups, message) {
  const out = groups.map((g) => ({ turn_id: g.turn_id, utterances: g.utterances.slice() }));
  let group = out.find((g) => g.turn_id === message.turn_id);
  if (!group) {
    group = { turn_id: message.turn_id, utterances: [] };
    out.push(group);
  }
  const at = group.utterances.findIndex((u) => u.utterance_id === message.utterance_id);
  // A fresh object on purpose (T-0030 criterion 18): a later astra.spoken drops any played
  // state the utterance carried. The cut was measured against the text that was played; a new
  // text has not been measured, and the page never strikes a string nobody measured.
  const utterance = { utterance_id: message.utterance_id, text: message.text };
  if (at >= 0) group.utterances[at] = utterance;
  else group.utterances.push(utterance);
  return out;
}

/** T-0030 criteria 17 and 18: record the cut on the utterances this page has already seen.
 *
 *  Nothing is invented: an entry for a turn or an utterance that never arrived as astra.spoken
 *  creates no group, no line and no text — the state is returned as it was. played_text is kept
 *  for reference only; the pane goes on displaying AstraSpoken.text, which is a *different*
 *  string (the TTS-normalised one), and played_words is the cut index that is valid in both. */
function withPlayed(state, message) {
  const entries = Array.isArray(message.utterances) ? message.utterances : [];
  const group = state.spoken.find((g) => g.turn_id === message.turn_id);
  if (!group || !entries.length) return state;
  let changed = false;
  const utterances = group.utterances.map((u) => {
    const entry = entries.find((e) => e && e.utterance_id === u.utterance_id);
    if (!entry) return u;
    changed = true;
    return {
      ...u,
      played_words: entry.played_words,
      complete: entry.complete,
      played_text: entry.played_text,
    };
  });
  if (!changed) return state;
  return {
    ...state,
    spoken: state.spoken.map((g) => (g === group ? { ...g, utterances } : g)),
  };
}

/** One line per turn: a partial is replaced by the next partial, a final one by its own replay.
 *
 *  The line is keyed by turn_id (T-0037 criterion 11) rather than by "the last line, if it is not
 *  final yet". That is what makes a replayed transcript idempotent: after a reconnect the server
 *  replays the turn's last astra.transcript, and the page already has that turn's line — keyed by
 *  position it would append a second, duplicated final line under the first. A message for a turn
 *  the page has not seen still starts a new line, so a new turn is still a new line. */
function withTranscript(lines, message) {
  const out = lines.map((l) => ({ text: l.text, final: l.final, turn_id: l.turn_id }));
  const line = { text: message.text, final: Boolean(message.final), turn_id: message.turn_id };
  const at = out.findIndex((l) => l.turn_id === message.turn_id);
  if (at >= 0) out[at] = line;
  else out.push(line);
  return out;
}

/** ms to show on the meter, or null: no elapsed number is shown outside THINKING. */
export function thinkingMs(state, nowMs) {
  if (!state || state.turnState !== "THINKING" || !state.thinking) return null;
  return state.thinking.baseMs + Math.max(0, nowMs - state.thinking.atMs);
}

export function spokenGroups(state) {
  return state.spoken;
}

export function transcriptLines(state) {
  return state.transcript;
}
