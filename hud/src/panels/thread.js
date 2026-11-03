// The conversation thread (T-0154 group B): one article per turn, in the order the turns' first
// content arrived, holding the user bubble, then the assistant row — a decorative avatar, the
// utterances, a meta line, the turn's card slot and its action cards.
//
// The thread is the one owner of conversation DOM (implementation note 2). It builds each
// container once and only ever adds to it, updates it in place or removes one part of it: the
// thread, an article, a card slot and an action list are never rebuilt (no replaceChildren, no
// textContent write over element children — B6), so the <a2ui-surface> the ui panel puts in a
// slot and the action cards keep their identity across every later message.
//
// What it renders, and from what (nothing else is read):
//   * the bubble        astra.transcript, one line per turn (state.js withTranscript: a partial is
//                       replaced, a final one stays one line, a replay adds none);
//   * the utterances    astra.spoken, one element per utterance_id (state.js withUtterance), the
//                       text cut at the measured word index from astra.spoken.played and from
//                       nothing else (spoken.js renderUtteranceText, reused unchanged; T-0030
//                       criteria 17-22). No turn state and no clock dims or hides a word;
//   * the meta line     "interrupted" (astra.turn.state INTERRUPTED, as state.js records it), the
//                       fence notice (astra.ui.fence through ui/fence.js describeFence: the page
//                       echoes, it does not derive) and "repaired N" (the surface's own count);
//   * the action cards  astra.tool.call through tool_call.js, in the turn's own article (B7);
//   * the thoughts      astra.thought through thoughts.js: one collapsed "Model thoughts"
//                       <details> per turn, the assistant body's last child, under the answer and
//                       the turn's cards (T-0180 C1). The page never opens it;
//   * the card slot     an empty element. Only hud/src/ui/index.js puts a card in it, and only from
//                       a decoded astra.ui.surface through the bridge (T-0024 criterion 4). This
//                       module never renders A2UI.
//
// An article is created only by content (B2): a transcript, an utterance, a surface, an action
// card, a model thought (T-0180 C1), or a fence notice that says something. A turn state, a quiet fence, a played report for
// an utterance the page never saw and every other type create nothing — not even an element.
//
// No DOM at module top level: `node --test tests/hud` imports this module.

import { initialState, reduce } from "../state.js";
import { SESSION_CLEARED_TYPE } from "../session/gate.js";
import { describeFence, FENCE_TYPE } from "../ui/fence.js";
import { SURFACE_TYPE } from "../ui/bridge.js";
import { renderUtteranceText } from "./spoken.js";
import { createThoughtDisclosure, THOUGHT_TYPE } from "./thoughts.js";
import { createToolCallPanel, TOOL_CALL_TYPE } from "./tool_call.js";
import { renderBubble } from "./transcript.js";

/** The gate's surface ids are `turn-<turn_id>` (ui_gate.py surface_id_prefix, T-0010 A7). */
const SURFACE_PREFIX = "turn-";

/** describeFence statuses that say nothing (T-0054 criterion 16). */
const QUIET = ["none", "whole"];

/** B9: within this many px of the bottom, new content keeps the thread at its bottom. */
export const STICK_PX = 48;

/** The meta line's parts, in display order. */
const META_PARTS = ["interrupted", "fence", "repaired"];

const FENCE_MARKS = ["data-ui-fence", "data-ui-fence-canvases", "data-ui-fence-refused", "data-ui-fence-turn"];

/** The turn a surface id names, or null when it names none. */
export function turnOfSurface(surfaceId) {
  if (typeof surfaceId !== "string" || !surfaceId.startsWith(SURFACE_PREFIX)) return null;
  const turn = surfaceId.slice(SURFACE_PREFIX.length);
  return turn ? turn : null;
}

const isId = (v) => typeof v === "string" && v.length > 0;

/**
 * createThread(root) -> { handle(message), cardSlot(surfaceId) }
 *
 *   root          the [data-thread] element; everything is built with root.ownerDocument.
 *   handle(m)     one decoded server message of any type (main.js's subscriber loop also catches).
 *   cardSlot(id)  the card slot of the article the surface id names (creating the article), an
 *                 end article of its own for an id that names no turn (B4), null for a non-string
 *                 or empty id. The same element for the same id, every time.
 */
export function createThread(root) {
  const doc = root.ownerDocument;
  let state = initialState();
  const turns = new Map(); // turn_id -> the article's parts
  const loose = new Map(); // a surface id that names no turn -> its own article's parts
  const lastCanvas = new Map(); // surface id -> the last canvas_idx whose repair count is shown

  const el = (tag, className, attrs) => {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    return node;
  };

  // --- building, once per article ------------------------------------------------------------

  function newArticle(attrs) {
    const article = el("article", "turn", attrs);
    root.appendChild(article);
    held = false; // T-0180 C-R2: a new turn sticks a reader who was pinned before their toggle
    observe(article);
    return { article, bubble: null, row: null, utterances: new Map() };
  }

  /** The assistant row: avatar, body (utterances, meta line, card slot, action list). */
  function rowOf(rec) {
    if (rec.row) return rec;
    const row = el("div", "assistant", { "data-assistant": "" });
    const avatar = el("span", "avatar", { "data-avatar": "", "aria-hidden": "true" });
    const body = el("div", "assistant-body");
    const meta = el("div", "turn-meta", { "data-turn-meta": "" });
    const slot = el("div", "card-slot", { "data-card-slot": "" });
    const tools = el("div", "tool-calls", { "data-tool-calls": "" });
    body.append(meta, slot, tools);
    row.append(avatar, body);
    rec.article.appendChild(row);
    Object.assign(rec, { row, body, meta, slot, tools, parts: {} });
    return rec;
  }

  function turnRecord(turnId) {
    let rec = turns.get(turnId);
    if (rec) return rec;
    rec = rowOf(newArticle({ "data-turn-id": turnId }));
    turns.set(turnId, rec);
    // A turn the server already called INTERRUPTED before it had content says so on arrival.
    if (state.interrupted.includes(turnId)) markInterrupted(rec);
    return rec;
  }

  function looseRecord(surfaceId) {
    let rec = loose.get(surfaceId);
    if (rec) return rec;
    rec = rowOf(newArticle({ "data-surface-article": surfaceId }));
    loose.set(surfaceId, rec);
    return rec;
  }

  // --- the meta line ----------------------------------------------------------------------------

  function metaPart(rec, name) {
    if (rec.parts[name]) return rec.parts[name];
    const part = el("span", `meta-${name}`);
    const later = META_PARTS.slice(META_PARTS.indexOf(name) + 1)
      .map((n) => rec.parts[n])
      .find(Boolean);
    rec.meta.insertBefore(part, later || null);
    rec.parts[name] = part;
    return part;
  }

  function dropPart(rec, name) {
    const part = rec.parts[name];
    if (!part) return;
    part.remove();
    delete rec.parts[name];
  }

  function markInterrupted(rec) {
    if (rec.article.hasAttribute("data-interrupted")) return;
    rec.article.setAttribute("data-interrupted", "true");
    metaPart(rec, "interrupted").textContent = "interrupted";
  }

  // --- per message ------------------------------------------------------------------------------

  function renderTranscript(turnId) {
    const line = state.transcript.find((l) => l.turn_id === turnId);
    if (!line) return;
    const rec = turnRecord(turnId);
    if (!rec.bubble) {
      rec.bubble = el("p", "bubble", { "data-transcript": "" });
      rec.article.insertBefore(rec.bubble, rec.article.firstChild);
    }
    if (rec.bubbleLine === line) return;
    rec.bubbleLine = line;
    renderBubble(rec.bubble, line);
  }

  function renderUtterances(turnId) {
    const group = state.spoken.find((g) => g.turn_id === turnId);
    if (!group) return;
    const rec = turnRecord(turnId);
    for (const utterance of group.utterances) {
      let entry = rec.utterances.get(utterance.utterance_id);
      if (!entry) {
        const line = el("p", "utterance", {
          "data-utterance-id": utterance.utterance_id,
          "data-turn-id": turnId,
        });
        rec.body.insertBefore(line, rec.meta);
        entry = { line, shown: null };
        rec.utterances.set(utterance.utterance_id, entry);
      }
      // state.js hands back a new object only for an utterance that changed.
      if (entry.shown === utterance) continue;
      entry.shown = utterance;
      renderUtteranceText(doc, entry.line, utterance);
    }
  }

  function onFence(m) {
    const turnId = m.turn_id;
    if (!isId(turnId)) return; // a verdict the page cannot attribute is shown nowhere (ruling B8)
    const described = describeFence(m);
    const speak = !QUIET.includes(described.status) || described.refused > 0;
    if (!speak) {
      const rec = turns.get(turnId);
      if (rec) dropPart(rec, "fence");
      return;
    }
    const part = metaPart(turnRecord(turnId), "fence");
    part.setAttribute(FENCE_MARKS[0], described.status);
    part.setAttribute(FENCE_MARKS[1], String(described.canvases));
    part.setAttribute(FENCE_MARKS[2], String(described.refused));
    part.setAttribute(FENCE_MARKS[3], turnId);
    part.textContent = described.label;
  }

  function onSurface(m) {
    const surfaceId = m.surface_id;
    if (!isId(surfaceId)) return;
    const rec = slotRecord(surfaceId);
    const idx = Number.isInteger(m.canvas_idx) ? m.canvas_idx : null;
    if (idx === null) return;
    // The count belongs to the canvas it came with; a stale canvas does not rewind it (B5).
    if (lastCanvas.has(surfaceId) && idx <= lastCanvas.get(surfaceId)) return;
    lastCanvas.set(surfaceId, idx);
    const repaired = Number.isInteger(m.repaired) ? m.repaired : 0;
    if (repaired > 0) metaPart(rec, "repaired").textContent = `repaired ${repaired}`;
    else dropPart(rec, "repaired");
  }

  /** T-0180 C1: the turn's one disclosure, appended once as the assistant body's last child. The
   *  utterances go before the meta line and the cards into the slot and the action list, all built
   *  before it, so it stays last without ever being moved or rebuilt. */
  function onThought(m) {
    const rec = turnRecord(m.turn_id);
    if (!rec.thoughts) {
      rec.thoughts = createThoughtDisclosure(doc);
      rec.body.appendChild(rec.thoughts.element);
    }
    rec.thoughts.handle(m);
  }

  function slotRecord(surfaceId) {
    const turnId = turnOfSurface(surfaceId);
    return turnId ? turnRecord(turnId) : looseRecord(surfaceId);
  }

  const tools = createToolCallPanel(root, { listFor: (turnId) => (isId(turnId) ? turnRecord(turnId).tools : null) });

  /** T-0162 D-2: every article goes — bubbles, utterances, meta lines, card slots (and the A2UI
   *  surfaces in them; ui/index.js drops those from its engine on the same message) and the inline
   *  tool cards. The thread element itself is kept, so the empty line shows again. */
  function clearArticles() {
    tools.clear();
    for (const rec of [...turns.values(), ...loose.values()]) {
      if (resizes) resizes.unobserve(rec.article);
      rec.article.remove();
    }
    turns.clear();
    loose.clear();
    lastCanvas.clear();
    pinned = true;
    ownTop = null;
    held = false;
    anchor = null;
  }

  function apply(m) {
    switch (m.type) {
      case "astra.transcript":
      case "astra.spoken":
      case "astra.spoken.played":
      case "astra.turn.state": {
        const next = reduce(state, m, 0);
        if (next === state) return;
        const before = state;
        state = next;
        if (!isId(m.turn_id)) return;
        if (m.type === "astra.transcript") renderTranscript(m.turn_id);
        else if (m.type === "astra.turn.state") {
          if (next.interrupted !== before.interrupted && next.interrupted.includes(m.turn_id)) {
            const rec = turns.get(m.turn_id); // a turn state is not content: no article for it
            if (rec) markInterrupted(rec);
          }
        } else if (m.type === "astra.spoken.played") {
          if (turns.has(m.turn_id)) renderUtterances(m.turn_id);
        } else renderUtterances(m.turn_id);
        return;
      }
      case FENCE_TYPE:
        onFence(m);
        return;
      case SURFACE_TYPE:
        onSurface(m);
        return;
      case TOOL_CALL_TYPE:
        if (isId(m.turn_id)) tools.handle(m);
        return;
      case THOUGHT_TYPE:
        if (isId(m.turn_id)) onThought(m);
        return;
      case SESSION_CLEARED_TYPE: {
        // T-0162 D-2: once per new `resets` value (state.js decides; a repeat returns the state).
        const next = reduce(state, m, 0);
        if (next === state) return;
        state = next;
        clearArticles();
        return;
      }
      default:
        return;
    }
  }

  // --- B9: the thread sticks to its bottom unless the reader scrolled away ------------------------

  let pinned = true;
  let ownTop = null; // the scrollTop this module last set, so its own scroll is not the reader's

  const fromBottom = () => (root.scrollHeight || 0) - (root.scrollTop || 0) - (root.clientHeight || 0);

  function stick() {
    if (!pinned) return;
    root.scrollTop = root.scrollHeight || 0;
    ownTop = root.scrollTop;
  }

  // T-0180 C-R3: only a scroll that follows the reader's own input releases the hold. The browser's
  // scroll anchoring (content above the viewport grew) also fires `scroll`; for a held reader it
  // neither releases the hold nor unpins. Input: wheel, touch, a scroll key, a pointer held down on
  // the thread (its scrollbar). A scroll counts as the reader's while a pointer or touch is down, or
  // within INTENT_MS of the last wheel, scroll key or swipe (smooth scrolling and momentum outlast
  // the input). A click or tap (the summary's own toggle) opens no window. An unheld reader's
  // scroll recomputes `pinned` as before (B9).
  const INTENT_MS = 600;
  const SCROLL_KEYS = ["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " ", "Spacebar"];
  let intentAt = -Infinity;
  let pressed = 0; // pointers and touches down on the thread
  const now = () => Date.now();
  const intent = () => {
    intentAt = now();
  };
  const readerScrolling = () => pressed > 0 || now() - intentAt <= INTENT_MS;
  root.addEventListener("wheel", intent, { passive: true });
  let swiped = false; // a touch that moved: its momentum scroll outlasts the touch
  root.addEventListener("touchstart", () => {
    pressed += 1;
    swiped = false;
  }, { passive: true });
  root.addEventListener("touchmove", () => {
    swiped = true;
    intent();
  }, { passive: true });
  for (const type of ["touchend", "touchcancel"]) {
    root.addEventListener(type, () => {
      pressed = Math.max(0, pressed - 1);
      if (swiped) intent();
    }, { passive: true });
  }
  // On the document, capturing: after a click on plain thread text the focus is <body>, so a key
  // never reaches `root`, yet the browser scrolls the thread.
  const keyTarget = globalThis.document && typeof globalThis.document.addEventListener === "function"
    ? globalThis.document
    : root;
  keyTarget.addEventListener("keydown", (ev) => {
    if (!ev || !SCROLL_KEYS.includes(ev.key)) return;
    const t = ev.target;
    const near = (sel) => Boolean(t && typeof t.closest === "function" && t.closest(sel));
    // Keys typed into a field are text, not scrolling.
    if (near("input, textarea, select, [contenteditable]")) return;
    // Space on a summary or a button activates it; it does not scroll. Other keys still scroll.
    if ((ev.key === " " || ev.key === "Spacebar") && near("summary, button")) return;
    intent();
  }, true);
  root.addEventListener("pointerdown", (ev) => {
    if (ev && ev.pointerType === "touch") return; // counted by touchstart
    pressed += 1;
  });
  if (typeof globalThis.addEventListener === "function") {
    for (const type of ["pointerup", "pointercancel"]) {
      globalThis.addEventListener(type, (ev) => {
        if (ev && ev.pointerType === "touch") return;
        if (pressed > 0) pressed -= 1; // a tap or click is not a scroll: no window after it
      });
    }
  }

  root.addEventListener("scroll", () => {
    if (ownTop !== null && Math.abs((root.scrollTop || 0) - ownTop) < 1) return;
    ownTop = null;
    if (held && !readerScrolling()) return; // C-R3: not the reader's scroll; the hold stands
    held = false; // C-R2: the reader's own scroll ends the hold
    pinned = fromBottom() <= STICK_PX;
  });

  // T-0180 C-R1: the viewer opening or closing a "Model thoughts" disclosure never moves them.
  // Click, Enter and Space on a <summary> all fire `click` before the disclosure toggles, so a
  // capturing listener records where the summary is on screen; the resize that the toggle causes
  // then restores that position instead of re-sticking.
  //
  // C-R2: a toggle does not unpin. A reader who was pinned when they toggled is *held*: they stay
  // pinned, but growth of the turns already on screen (the toggle itself, thought updates, cards)
  // does not move them. The next new article releases the hold and sticks them to the bottom; their
  // own scroll or a reset releases it too. A reader who was not pinned is never held.
  let anchor = null; // { summary, top } of the viewer's last toggle, until its resize is handled
  let held = false;

  root.addEventListener(
    "click",
    (ev) => {
      const target = ev && ev.target;
      const summary = target && typeof target.closest === "function" ? target.closest("summary") : null;
      if (!summary || !root.contains(summary)) return;
      const details = summary.parentElement;
      if (!details || !details.hasAttribute("data-model-thoughts")) return;
      anchor = { summary, top: summary.getBoundingClientRect().top };
      if (pinned) held = true;
    },
    true,
  );

  function restoreAnchor() {
    const { summary, top } = anchor;
    anchor = null;
    if (!summary.isConnected) return;
    const delta = summary.getBoundingClientRect().top - top;
    if (delta !== 0) {
      root.scrollTop = (root.scrollTop || 0) + delta;
      ownTop = root.scrollTop;
    }
    if (!held) pinned = fromBottom() <= STICK_PX;
  }

  function onResize() {
    if (anchor) restoreAnchor();
    else if (!held) stick();
  }

  // A card renders after its message (the renderer draws asynchronously), so growth of an article
  // re-sticks too, except the growth of the viewer's own toggle (C-R1) and growth while a pinned
  // reader is held (C-R2). No observer in node: there the messages alone decide.
  const resizes = typeof globalThis.ResizeObserver === "function" ? new globalThis.ResizeObserver(onResize) : null;
  function observe(article) {
    if (resizes) resizes.observe(article);
  }

  // Whether a message changed the DOM, read synchronously with takeRecords() (no observer in node:
  // there every content message counts as a change).
  const mutations =
    typeof globalThis.MutationObserver === "function" ? new globalThis.MutationObserver(() => {}) : null;
  if (mutations) {
    mutations.observe(root, { childList: true, subtree: true, characterData: true, attributes: true });
  }

  return {
    handle(m) {
      if (!m || typeof m !== "object" || typeof m.type !== "string") return;
      if (mutations) mutations.takeRecords(); // anything pending is not this message's doing
      apply(m);
      // B9 / rulings O-B3 (2), O-B4: only a message that changed the thread's DOM re-sticks, and a
      // turn state never does (THINKING ticks at 20 Hz would pull a reader back to the bottom).
      // Neither clears a pending C-R1 anchor: canvas frames and turn states stream during a toggle.
      if (m.type === "astra.turn.state") return;
      if (mutations && mutations.takeRecords().length === 0) return;
      anchor = null; // C-R1: content a message added follows B9, not the viewer's last toggle
      if (!held) stick(); // C-R2: a new article (newArticle) released the hold before this
    },

    cardSlot(surfaceId) {
      if (!isId(surfaceId)) return null;
      return slotRecord(surfaceId).slot;
    },

    /** T-0162 D-2: empty the thread now, whatever the last reset was (handle() calls it once per
     *  new astra.session.cleared). The transcript and spoken state go with the articles. */
    clear() {
      const { resets, seenTurns } = state;
      state = { ...initialState(), resets, seenTurns };
      clearArticles();
    },
  };
}
