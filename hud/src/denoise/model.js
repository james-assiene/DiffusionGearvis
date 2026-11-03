// The denoising view's cell model (T-0128c B2, B4, B9). Pure: no DOM, no clock, no randomness —
// every state comes from the decoded astra.canvas.step frame in hand (and, for "revised", the
// frame received before it). Nothing here invents a position, a step or a frame.

/** T-0128c B3's column count (the mockup's `.noise { grid-template-columns: repeat(32, 1fr) }`).
 *  Kept for its importers; since T-0159 B-6 the page's token grid takes as many columns as fit the
 *  pane (styles.css), so the panel no longer reads it. */
export const COLUMNS = 32;

/** Every state a cell can be in, in the order the aria-label counts them. */
export const STATES = ["stable", "kept", "noise", "revised", "committed", "empty"];

/** True when bit `p` of a hex mask (MSB first, whole bytes: AstraCanvasStep.stable) is set. */
function bit(mask, p) {
  const nibble = parseInt(mask.charAt(p >> 2), 16);
  if (Number.isNaN(nibble)) return false;
  return (nibble & (8 >> (p & 3))) !== 0;
}

const isDenoise = (f) => Boolean(f) && (f.phase === undefined || f.phase === "denoise");

/**
 * B2: one state per position of `frame`.
 *
 * Denoise: "stable" if the stable bit is set; else "kept" if canvas[p] == argmax[p]; else "noise".
 * canvas == argmax is the "kept" proxy: the plugin re-noises un-kept positions with random ids, and a
 * random refill that happens to equal the argmax has probability ~1/262K per position (the Gemma 4
 * vocabulary), so the proxy is not exact but never systematically wrong.
 * A position is shown "revised" whenever its argmax differs from `previous`'s where `previous` had it
 * stable, whatever its current stable bit (ruling O-B (4)); `previous` counts only when it is a denoise
 * frame of the same (request_id, canvas_idx).
 * Commit: positions below len(committed_tokens) are "committed", the rest "empty".
 */
export function cellStates(frame, previous) {
  const n = frame.canvas_length;
  const out = new Array(n);
  if (!isDenoise(frame)) {
    const done = (frame.committed_tokens || []).length;
    for (let p = 0; p < n; p += 1) out[p] = p < done ? "committed" : "empty";
    return out;
  }
  const prev =
    isDenoise(previous) &&
    previous.request_id === frame.request_id &&
    previous.canvas_idx === frame.canvas_idx &&
    previous.canvas_length === frame.canvas_length
      ? previous
      : null;
  const { canvas, argmax, stable } = frame;
  for (let p = 0; p < n; p += 1) {
    if (prev && bit(prev.stable, p) && prev.argmax[p] !== argmax[p]) out[p] = "revised";
    else if (bit(stable, p)) out[p] = "stable";
    else if (canvas[p] === argmax[p]) out[p] = "kept";
    else out[p] = "noise";
  }
  return out;
}

/**
 * T-0159 B-1/B-6: the text piece at each position of `frame`, read from the frame and never decoded
 * here (the conductor's pinned detokeniser fills the fields; the page has no id -> text table).
 * Denoise: `argmax_pieces[p]`. Commit: `committed_pieces[p]` for p < len(committed_tokens); the
 * tail (the "empty" cells) has none. A frame without pieces gives no text anywhere (B-2): every
 * entry is null. A field that is not an array of `canvas_length` (denoise) or
 * `len(committed_tokens)` (commit) strings is treated as absent; the decoder refuses such a frame
 * anyway (protocol.js, A6), so this only guards against drawing text that is not the frame's.
 */
export function cellPieces(frame) {
  const n = frame.canvas_length;
  const out = new Array(n).fill(null);
  const rows = isDenoise(frame) ? frame.argmax_pieces : frame.committed_pieces;
  const want = isDenoise(frame) ? n : (frame.committed_tokens || []).length;
  if (!Array.isArray(rows) || rows.length !== want || want > n) return out;
  if (!rows.every((s) => typeof s === "string")) return out;
  for (let p = 0; p < want; p += 1) out[p] = rows[p];
  return out;
}

/**
 * T-0159 B2-1: true when `frame` is a denoise frame whose argmax piece at every position is `<eos>`
 * (the trailing canvas the engine starts after the answer commits). A frame without (valid) pieces,
 * a commit frame, or an empty canvas is never end of sequence.
 */
export function isEndOfSequence(frame) {
  if (!isDenoise(frame) || !(frame.canvas_length > 0)) return false;
  return cellPieces(frame).every((piece) => piece === "<eos>");
}

/** An added or special token as written in tokenizer.json: `<eos>`, `<|channel>`, `<0xE2>`. */
const SPECIAL = /^<[^<>\s]+>$/;
const visible = (s) => s.replace(/ /g, "␣").replace(/\n/g, "↵").replace(/\t/g, "⇥");

/**
 * T-0159 B-6: the display form of one piece. Returns { text, title, special }:
 *   - special (`<…>` form, byte literals included): text = the piece verbatim, special = true
 *     (the page sets it in the monospace utility face, smaller);
 *   - otherwise leading and trailing spaces are trimmed, a piece of only spaces shows "␣", and
 *     every newline shows "↵" and every tab "⇥";
 *   - title: the full piece with every space, newline and tab made visible (␣ ↵ ⇥).
 * null (no piece for this position) gives { text: "", title: "", special: false }.
 * The result is plain text for `textContent`/`title`, never markup (B-3).
 */
export function displayPiece(piece) {
  if (typeof piece !== "string") return { text: "", title: "", special: false };
  if (SPECIAL.test(piece)) return { text: piece, title: piece, special: true };
  const trimmed = piece.replace(/^ +| +$/g, "");
  const text = piece.length > 0 && trimmed === "" ? "␣" : trimmed.replace(/\n/g, "↵").replace(/\t/g, "⇥");
  return { text, title: visible(piece), special: false };
}

/** B4: the honest label, verbatim from the frame (0-based canvas_idx, H-T128c-5). */
export function label(frame) {
  const at = `${frame.step}/${frame.max_steps}`;
  return isDenoise(frame)
    ? `canvas ${frame.canvas_idx} · step ${at}`
    : `canvas ${frame.canvas_idx} · committed at step ${at}`;
}

/** How many cells are in each state (B3's aria-label). */
export function counts(states) {
  const out = Object.fromEntries(STATES.map((s) => [s, 0]));
  for (const s of states) out[s] += 1;
  return out;
}

/**
 * B9: FNV-1a, 32-bit, as 8 lowercase hex digits, of the state names joined by "," (ASCII bytes).
 * The empty array hashes no byte and is the offset basis 811c9dc5.
 */
export function digest(states) {
  const text = states.join(",");
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i) & 0xff;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}
