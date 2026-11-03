// The shown channel: the A2UI cards, inline in the conversation thread (T-0024, the renderer half
// of T-0010; T-0154 B4 puts each card in the card slot of the turn that produced it; T-0164 B
// renders them with our own light-DOM renderer, ./engine.js and ./render.js).
//
// `mount(host)` is the contract `hud/src/panels.js` calls and `hud/src/main.js` supplies:
// `{ document, cardSlot(surfaceId) -> Element, subscribe(handler) -> unsubscribe,
//    send(message) -> Promise, log }`. `cardSlot` is the thread's (hud/src/panels/thread.js): the
// slot of the article the surface id names, or an end article of its own for an id that names no
// turn. `host.shown` (T-0008's #shown-channel) is retired with that element (T-0154 B1).
//
// It must never throw out of `mountPanels`: a host with no document (that is `node --test`), a
// host with no card-slot provider, a message the renderer rejects — each is one log line and a
// page that keeps its socket, its capture pipeline and its thread (T-0008 criterion 10, T-0154
// B11).
//
// What renders is *only* what `UiGate` validated and `protocol.js` decoded as `astra.ui.surface`
// (criterion 4) — see ./bridge.js. What goes back is *only* `astra.ui.action` (criterion 9), and
// only for the most recently created surface (T-0154 B8, H-T154-4): every older card's wrapper is
// `inert` — see ./action.js. The fence notice and the repair count are the thread's meta line now
// (T-0154 B3); this panel renders cards and nothing else.

import { createEngine } from "./engine.js";
import { CARDS_V2_CATALOG_ID, GATE_CATALOG_ID } from "./a2ui.js";
import { createSurfaceBridge, SURFACE_TYPE } from "./bridge.js";
import { createActionHandler } from "./action.js";
import { createClearTracker, seedFrom } from "../session/gate.js";

/** The page's one card sheet (T-0164 B5), same origin as this module. */
const CARD_HREF = new URL("./card.css", import.meta.url).href;

/**
 * Link the card sheet into the document, once per page (T-0164 B5). The cards are light DOM, so
 * one page-level sheet reaches every card; the browser loads it from the page's own loopback
 * origin, as it does every other file (no transport of this panel's own, T-0024 criterion 3).
 */
function linkCardSheet(doc, log) {
  try {
    const link = doc.createElement("link");
    link.rel = "stylesheet";
    link.href = CARD_HREF;
    (doc.head || doc.body).appendChild(link);
  } catch (err) {
    log(`a2ui panel: the card sheet was not linked, cards render unstyled: ${(err && err.message) || err}`);
  }
}

/**
 * mount(host) -> { mounted, bridge? } — never throws, never rejects.
 */
export async function mount(host) {
  const h = host && typeof host === "object" ? host : {};
  const log = typeof h.log === "function" ? h.log : (line) => console.warn("[hud/ui]", line);
  const doc = h.document;

  if (!doc || typeof doc.createElement !== "function") {
    // `node --test`, or a host that predates the seam: render nothing, send nothing, take nothing
    // down (T-0025 criterion 1, T-0024 criterion 2 negative).
    log("a2ui panel: this host has no document, so no surface is rendered");
    return { mounted: false };
  }
  if (typeof h.cardSlot !== "function") {
    // T-0154 B11: the cards live in the thread's card slots; without that provider there is
    // nowhere honest to put one, so nothing is rendered (not even a stylesheet).
    log("a2ui panel: this host has no cardSlot() card-slot provider, so no surface is rendered");
    return { mounted: false };
  }

  linkCardSheet(doc, log);

  /** surface id -> { wrapper, element }, in creation order (the last one is the newest). */
  const cards = new Map();
  /** The surface id of the astra.ui.surface being handed to the bridge right now. */
  let arriving = null;

  // T-0184 B1: the basic catalog and our v2 cards; any other id renders nothing (T-0164 B3).
  const engine = createEngine({
    document: doc,
    catalogIds: [GATE_CATALOG_ID, CARDS_V2_CATALOG_ID],
    log,
    onAction: (action) => handler(action),
  });

  // `let`: a session reset starts a fresh bridge, so no pre-reset surface stays "the newest" or
  // keeps a canvas index (T-0162 D-2, D-4).
  let bridge = createSurfaceBridge({ processor: engine.processor, log });
  const resets = createClearTracker(seedFrom(h)); // the page's count at mount (T-0162 D-2)

  /** T-0162 D-2: every surface leaves the engine (`engine.drop`, T-0164) and its wrapper leaves its
   *  card slot; the next surface created is the newest one and the only one that sends actions. */
  function clear() {
    for (const id of engine.surfaces()) engine.drop(id);
    for (const card of cards.values()) card.wrapper.remove();
    cards.clear();
    bridge = createSurfaceBridge({ processor: engine.processor, log });
  }

  const handler = createActionHandler({
    send: typeof h.send === "function" ? h.send : undefined,
    log,
    // Criterion 9 and B8: the page speaks only for the newest surface it rendered, addressed by
    // the id the gate put inside the messages.
    isRendered: (id) => bridge.surfaceId() === id && cards.has(id),
    dataModel: () => engine.dataModelFor(bridge.surfaceId()),
  });

  // One card root `[data-a2ui-root]` per surface id (T-0024 criterion 7 as amended by T-0154 B4,
  // T-0164 B2): the engine morphs canvas k-1 into canvas k in that same element. A new surface
  // makes every older card inert (B8).
  engine.processor.onSurfaceCreated((surface) => {
    const id = surface && typeof surface.id === "string" ? surface.id : arriving;
    if (typeof id !== "string" || !id || cards.has(id)) return;
    const element = (surface && surface.element) || engine.element(id);
    if (!element) {
      log(`a2ui panel: surface ${id} has no card root; it is not rendered`);
      return;
    }
    let slot = null;
    try {
      slot = h.cardSlot(id);
    } catch (err) {
      log(`a2ui panel: the card slot for ${id} could not be had: ${(err && err.message) || err}`);
    }
    if (!slot) {
      log(`a2ui panel: no card slot for surface ${id}; it is not rendered`);
      return;
    }
    for (const older of cards.values()) older.wrapper.inert = true;
    const wrapper = doc.createElement("div");
    wrapper.className = "a2ui-panel card-wrap";
    wrapper.setAttribute("data-ui-surface", id);
    wrapper.appendChild(element);
    slot.appendChild(wrapper);
    cards.set(id, { wrapper, element });
  });

  if (typeof h.subscribe === "function") {
    h.subscribe((message) => {
      if (resets.accept(message)) {
        clear();
        return;
      }
      if (!message || message.type !== SURFACE_TYPE) return;
      arriving = typeof message.surface_id === "string" ? message.surface_id : null;
      const applied = bridge.handle(message);
      arriving = null;
      if (!applied) return;
      const card = cards.get(message.surface_id);
      const idx = bridge.canvasIdx(message.surface_id);
      if (card && idx !== null) card.wrapper.setAttribute("data-ui-canvas-idx", String(idx));
    });
  } else {
    log(`a2ui panel: this host has no subscribe(), so no ${SURFACE_TYPE} can ever reach it`);
  }

  log("a2ui panel mounted against the thread's card slots");
  return {
    mounted: true,
    get bridge() {
      return bridge; // the current one: a reset replaces it
    },
    clear,
  };
}
