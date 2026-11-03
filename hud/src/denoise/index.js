// The denoising view (T-0128c group B, T-0159 group B): the Thinking pane's grid of the real
// per-step canvases the conductor relays from the plugin tap (astra.canvas.step). Mounted by
// hud/src/panels.js with the host hud/src/main.js passes ({ document, subscribe, ... }).
//
// T-0159 (James's reference image, the DiffusionGemma page's denoising figure): each canvas is a
// grid of rounded token cells, one per position, each filled with its T-0128c state colour and
// showing the token at that position, as the conductor decoded it (argmax_pieces /
// committed_pieces). Each canvas of the shown request is its own native <details> section.
//
// Honesty (B5, B-2) is the design constraint, not a feature:
//   * nothing is drawn, and [data-denoise-root] has no child, until the first decoded frame;
//   * every paint writes received frames only (each canvas's latest), never an interpolation or a
//     synthetic one;
//   * a cell shows text only from the frame it draws: no pieces, no text; ids are never decoded here;
//   * text goes in through textContent and title only, never as markup (B-3);
//   * an animation frame is requested only while a received frame waits to be painted, and a timer
//     runs only under reduced motion while a frame waits for its half-second slot (B8);
//   * no randomness anywhere.
// Time, frames and media queries come from host.document.defaultView (window in a browser), so the
// node tests can drive the page on a virtual clock. Colours, sizes and the theme are CSS only
// (styles.css): a theme switch or a resize needs no repaint.

import {
  cellPieces,
  cellStates,
  counts,
  digest,
  displayPiece,
  isEndOfSequence,
  label,
} from "./model.js";
import { createClearTracker, seedFrom } from "../session/gate.js";

const REDUCED_MS = 500; // B8: at most two redraws per second under prefers-reduced-motion: reduce
const SEEN_LIMIT = 64; // request ids remembered for "first frame" (bounded)
const REDUCE_QUERY = "(prefers-reduced-motion: reduce)";

function ariaLabel(frame, states) {
  const c = counts(states);
  const parts =
    frame.phase === "commit"
      ? [`${c.committed} committed`, `${c.empty} empty`]
      : [`${c.stable} stable`, `${c.kept} kept`, `${c.noise} noise`, `${c.revised} revised`];
  return `${label(frame)}: ${parts.join(", ")}`;
}

export function mount(host) {
  const doc = host.document;
  const view = doc.defaultView || globalThis;
  const root = doc.querySelector("[data-denoise-root]");
  if (!root) return; // no Thinking pane in this page: nothing to draw into
  const labelEl = doc.querySelector("[data-denoise-label]");
  const emptyEl = doc.querySelector("[data-denoise-empty]");
  const clock = view.performance || globalThis.performance;
  const now = () => clock.now();
  const reduceQuery = typeof view.matchMedia === "function" ? view.matchMedia(REDUCE_QUERY) : null;
  const reduced = () => Boolean(reduceQuery && reduceQuery.matches);

  let received = 0;
  let painted = 0;
  // canvas_idx -> { frame, states }: each canvas's latest received frame of the shown request that
  // waits for the next draw (review F1); pendingNewest is the newest of them.
  let pending = new Map();
  let pendingNewest = null;
  let noticeKind = null; // what would make the line's notice false: "frames", "pieces" or none
  const emptyCopy = emptyEl ? emptyEl.textContent : "";
  let rafId = null;
  let timerId = null;
  let lastPaintAt = null;
  let shown = null; // request id in view (B6): the one whose first frame arrived most recently
  let drawnRequest = null; // the request whose sections are in the root
  let sections = new Map(); // canvas_idx -> section, insertion order = order of arrival (B-7)
  let newest = null; // the section most recently appended
  let reopened = null; // B2-3: the section opened when the newest became end of sequence
  const seen = new Map(); // request id -> true, insertion order = LRU
  // T-0162 D-2: once per new astra.session.cleared, counted from the page's own tracker at mount
  const resets = createClearTracker(seedFrom(host));
  const lastFrame = new Map(); // request id -> last received frame ("previous" for B2's revised)

  const remember = (map, key, value) => {
    map.delete(key);
    map.set(key, value);
    while (map.size > SEEN_LIMIT) map.delete(map.keys().next().value);
  };

  /** B-7: one native disclosure per canvas_idx; a new one is appended open and the previous newest
   *  closes (older sections keep whatever state the viewer left them in). */
  function sectionFor(frame) {
    const key = frame.canvas_idx;
    let section = sections.get(key);
    if (section) return section;
    const el = doc.createElement("details");
    el.setAttribute("class", "denoise-section");
    el.setAttribute("data-denoise-canvas", String(key));
    const summary = doc.createElement("summary");
    summary.setAttribute("class", "denoise-summary");
    const grid = doc.createElement("div");
    grid.setAttribute("class", "denoise-grid");
    grid.setAttribute("role", "img");
    el.appendChild(summary);
    el.appendChild(grid);
    if (newest) newest.el.removeAttribute("open");
    el.setAttribute("open", "");
    root.appendChild(el);
    section = { el, summary, grid, cells: [], shown: [], eos: false };
    sections.set(key, section);
    newest = section;
    reopened = null; // B2-4 applies to the transition of the section that was newest then
    return section;
  }

  /** B-7: a new request starts with an empty view; the previous request's sections go. */
  function clearSections() {
    for (const s of sections.values()) if (s.el.parentNode === root) root.removeChild(s.el);
    sections = new Map();
    newest = null;
    reopened = null;
  }

  /** T-0159 B2-2..B2-4: act only on a section's transition into or out of end of sequence, so the
   *  viewer's later open/close choice is kept. Into: the section closes and, if it is the newest,
   *  the newest non-end-of-sequence section opens (B2-3). Out of: it opens again as the newest and
   *  the section B2-3 opened closes (B2-4). The cells are drawn either way (B2-5). */
  function markEndOfSequence(section, eos) {
    if (section.eos === eos) return;
    section.eos = eos;
    if (eos) {
      section.el.removeAttribute("open");
      if (section !== newest) return;
      let answer = null;
      for (const s of sections.values()) if (!s.eos) answer = s;
      if (answer) {
        answer.el.setAttribute("open", "");
        reopened = answer;
      }
    } else if (section === newest) {
      section.el.setAttribute("open", "");
      if (reopened && reopened !== section) reopened.el.removeAttribute("open");
      reopened = null;
    }
  }

  /** B-6/B-8: draw `frame` into its section, touching only the cells whose state or text changed. */
  function drawSection(section, frame, states) {
    const pieces = cellPieces(frame);
    const n = states.length;
    const { grid, cells, shown: was } = section;
    while (cells.length < n) {
      const cell = doc.createElement("span");
      cell.setAttribute("class", "denoise-cell");
      cell.setAttribute("data-denoise-cell", "");
      grid.appendChild(cell);
      cells.push(cell);
      was.push({ state: null, special: null, text: "", title: "" });
    }
    while (cells.length > n) {
      grid.removeChild(cells.pop());
      was.pop();
    }
    for (let p = 0; p < n; p += 1) {
      const cell = cells[p];
      const old = was[p];
      const shownPiece = displayPiece(pieces[p]);
      if (old.state !== states[p]) {
        cell.setAttribute("data-state", states[p]);
        old.state = states[p];
      }
      if (old.special !== shownPiece.special) {
        if (shownPiece.special) cell.setAttribute("data-special", "");
        else cell.removeAttribute("data-special");
        old.special = shownPiece.special;
      }
      if (old.text !== shownPiece.text) {
        cell.textContent = shownPiece.text;
        old.text = shownPiece.text;
      }
      if (old.title !== shownPiece.title) {
        if (shownPiece.title) cell.setAttribute("title", shownPiece.title);
        else cell.removeAttribute("title");
        old.title = shownPiece.title;
      }
    }
    const eos = isEndOfSequence(frame);
    section.summary.textContent = eos ? `canvas ${frame.canvas_idx} · end of sequence` : label(frame);
    markEndOfSequence(section, eos);
    grid.setAttribute("aria-label", ariaLabel(frame, states));
  }

  /** One draw: every canvas's latest waiting frame is written (in the order the canvases first
   *  waited), so a canvas superseded by the next one inside a slot still gets its section and its
   *  last frame (B-7, review F1). The label and the digest are the newest received frame's. */
  function paint() {
    rafId = null;
    if (pending.size === 0) return;
    const entries = [...pending.values()];
    const newestEntry = pendingNewest;
    pending = new Map();
    pendingNewest = null;
    if (newestEntry.frame.request_id !== drawnRequest) {
      clearSections();
      drawnRequest = newestEntry.frame.request_id;
    }
    let withPieces = false;
    for (const { frame, states } of entries) {
      drawSection(sectionFor(frame), frame, states);
      if (cellPieces(frame).some((piece) => piece !== null)) withPieces = true;
    }
    lastPaintAt = now();
    painted += 1;
    root.setAttribute("data-denoise-draws", String(painted));
    root.setAttribute("data-denoise-digest", digest(newestEntry.states));
    if (labelEl) labelEl.textContent = label(newestEntry.frame);
    // Review F2: a "no frames" notice (denoise_no_tap, a synthetic refusal) is false once a frame is
    // drawn; "no pieces" is false once a drawn frame carries pieces. Both leave the line then.
    if (noticeKind === "frames" || (noticeKind === "pieces" && withPieces)) clearNotice();
  }

  function requestPaint() {
    if (rafId === null) rafId = view.requestAnimationFrame(paint);
  }

  /** Paint the pending frames at the next animation frame — or, under reduced motion, at the next
   *  half-second slot (latest wins per canvas: a canvas's waiting frame is replaced, never queued). */
  function schedule() {
    if (pending.size === 0 || rafId !== null) return;
    const wait = reduced() && lastPaintAt !== null ? REDUCED_MS - (now() - lastPaintAt) : 0;
    if (wait > 0) {
      if (timerId === null)
        timerId = view.setTimeout(() => {
          timerId = null;
          schedule();
        }, wait);
      return;
    }
    if (timerId !== null) {
      view.clearTimeout(timerId);
      timerId = null;
    }
    requestPaint();
  }

  function onFrame(frame) {
    received += 1;
    root.setAttribute("data-denoise-frames", String(received));
    const id = frame.request_id;
    const first = !seen.has(id);
    remember(seen, id, true);
    // B6 (ruling O-C5): the request whose first frame arrived most recently takes the view at once;
    // frames of any other request (an older one never takes the view back) are counted, not drawn.
    if (first) {
      shown = id;
      pending = new Map(); // the previous request's waiting frames are never drawn now
      pendingNewest = null;
    }
    if (id !== shown) {
      remember(lastFrame, id, frame);
      return;
    }
    const states = cellStates(frame, lastFrame.get(id) || null);
    remember(lastFrame, id, frame);
    pendingNewest = { frame, states };
    pending.set(frame.canvas_idx, pendingNewest);
    // B8 (ruling O-B (2)): under reduced motion the label follows every frame even when the grid
    // waits for its slot.
    if (reduced() && labelEl) labelEl.textContent = label(frame);
    schedule();
  }

  /** B5/B-10: a notice on the line under the grid, visible beside drawn cells while it is true.
   *  `kind` says what makes it false: "frames" (no tap, a refused synthetic canvas) clears at the
   *  next drawn frame, "pieces" (denoise_no_pieces) at the next drawn frame that carries pieces. */
  function notice(text, kind) {
    if (!emptyEl) return;
    emptyEl.textContent = text;
    emptyEl.setAttribute("data-denoise-notice", "");
    noticeKind = kind;
  }

  function clearNotice() {
    noticeKind = null;
    if (!emptyEl) return;
    emptyEl.removeAttribute("data-denoise-notice");
    emptyEl.textContent = emptyCopy;
  }

  /** T-0162 D-2: the sections, the waiting frames and the label go, and the line under the grid
   *  is its empty copy again. `seen` is kept on purpose: a late frame of a pre-reset request is not
   *  a "first" frame, so it never takes the view (nothing is shown until a new request's frame). */
  function clear() {
    if (rafId !== null && typeof view.cancelAnimationFrame === "function") view.cancelAnimationFrame(rafId);
    rafId = null;
    if (timerId !== null) view.clearTimeout(timerId);
    timerId = null;
    pending = new Map();
    pendingNewest = null;
    clearSections();
    drawnRequest = null;
    shown = null;
    lastPaintAt = null;
    root.removeAttribute("data-denoise-digest");
    if (labelEl) labelEl.textContent = "";
    clearNotice();
  }

  function onMessage(msg) {
    if (!msg || typeof msg !== "object") return;
    if (resets.accept(msg)) clear();
    else if (msg.type === "astra.canvas.step") onFrame(msg);
    else if (msg.type === "astra.canvas.synthetic") {
      // B5: a synthetic fallback is never drawn; the line says it was refused and why.
      notice(`synthetic canvas refused: ${msg.reason}`, "frames");
    } else if (msg.type === "astra.error" && msg.code === "denoise_no_tap") {
      notice(msg.message, "frames");
    } else if (msg.type === "astra.error" && msg.code === "denoise_no_pieces") {
      notice(msg.message, "pieces");
    }
  }

  host.subscribe(onMessage);
  return { clear };
}
