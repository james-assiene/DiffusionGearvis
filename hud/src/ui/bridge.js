// The A2UI bridge: decoded `astra.ui.surface` messages -> the renderer's message processor
// (T-0024 criteria 4, 5, 7).
//
// No DOM here — `node --test tests/hud` imports this module and drives it with an injected
// processor, exactly as T-0008 does for `state.js`/`protocol.js`. The DOM half is ./index.js, the
// engine is ./engine.js (rendering through ./render.js, T-0164) and the identifiers it checks are
// in ./a2ui.js.
//
// The rule the module exists for (criterion 4, the honesty rule): the *only* thing that can put
// an A2UI message in front of the renderer is a message the Conductor's `UiGate` validated and
// `hud/src/protocol.js` decoded as `astra.ui.surface`. There is no other entry point — not spoken
// text, not a canvas commit, not an `<a2ui>` sentinel, not a URL parameter.
//
// Two throw sites of the engine's processor shape everything below (first measured from the
// vendored `@a2ui/web_core@0.11.x` MessageProcessor, docs/tasks/T-0024.md; ./engine.js keeps
// both for parity, T-0164):
//
//   * `Catalog not found: <id>`   — processCreateSurfaceMessage, on an unregistered catalogId;
//   * `Surface <id> already exists` — on the second createSurface for one id, and the gate
//     re-sends createSurface on every canvas of a turn by design (T-0010 A4).
//
// Both are handled *per A2UI message*, never per canvas: `processMessages` applies its array in
// order and throws out of it, so catching at canvas granularity would lose every message after
// the throwing one — which, for the re-sent createSurface, is the whole surface.

/** The one server->client type this panel consumes (contracts/protocol.AstraUiSurface). */
export const SURFACE_TYPE = "astra.ui.surface";

/** A2UI message kinds, as they appear as the single key beside `version`. */
const CREATE = "createSurface";
const DELETE = "deleteSurface";

/** The surfaceId an A2UI message addresses, and which kind it is. */
function target(message) {
  if (!message || typeof message !== "object") return { kind: null, surfaceId: null };
  for (const kind of [CREATE, DELETE, "updateComponents", "updateDataModel"]) {
    const payload = message[kind];
    if (payload && typeof payload === "object") {
      const surfaceId = typeof payload.surfaceId === "string" ? payload.surfaceId : null;
      return { kind, surfaceId };
    }
  }
  return { kind: null, surfaceId: null };
}

/**
 * createSurfaceBridge({ processor, log }) -> bridge
 *
 *   processor  the A2UI engine: `{ processMessages(messages) }` (the real one is
 *              `createEngine(...).processor` from ./engine.js). The bridge
 *              decides *when* it is called and with what; it never decides what the messages say.
 *
 *   handle(message) -> boolean   one message already decoded and classified `kind === "astra"`;
 *                                true iff it was an astra.ui.surface this bridge forwarded.
 *   surfaceId()     -> string|null   the most recently *created* surface (T-0154 B8: the only one
 *                                    that may send astra.ui.action)
 *   canvasIdx(id?)  -> number|null   the last canvas_idx applied to surface `id` (default: the
 *                                    newest surface)
 *   stats()         -> { surfaces, canvases, ignored, errors }
 *
 * Canvas order is per surface (T-0154 B5): every surface keeps its own last applied canvas_idx,
 * and a canvas at or below it is skipped, whichever surface was created in between.
 *
 * Never throws: a panel that throws out of a socket handler takes the page's message loop with it
 * (T-0008 criterion 10, T-0024 criterion 2 negative).
 */
export function createSurfaceBridge({ processor, log = () => {} } = {}) {
  const created = new Set(); // surfaces the engine really holds
  let mounted = null; // the most recently created surface id
  const appliedIdx = new Map(); // surface id -> the last canvas_idx applied to it
  let surfaces = 0;
  let canvases = 0;
  let ignored = 0;
  let errors = 0;

  const note = (line) => {
    try {
      log(line);
    } catch {
      // a logger that throws must not take the panel down either
    }
  };

  /** Hand exactly one A2UI message to the engine. -> did it apply? */
  function apply(message) {
    try {
      processor.processMessages([message]);
      return true;
    } catch (err) {
      errors += 1;
      note(`a2ui message rejected by the renderer: ${(err && err.message) || err}`);
      return false;
    }
  }

  return {
    handle(message) {
      if (!message || typeof message !== "object" || message.type !== SURFACE_TYPE) {
        // Criterion 4: everything else the page receives is somebody else's message. Counted so
        // "the panel is wired to the seam" is observable, never rendered.
        ignored += 1;
        return false;
      }
      const surfaceId = typeof message.surface_id === "string" ? message.surface_id : null;
      const idx = Number.isInteger(message.canvas_idx) ? message.canvas_idx : null;
      const messages = Array.isArray(message.messages) ? message.messages : null;
      if (!surfaceId || idx === null || !messages) {
        errors += 1;
        note(`astra.ui.surface is malformed and was dropped: ${JSON.stringify(message).slice(0, 200)}`);
        return false;
      }

      // Criterion 7 negative: a canvas we have already passed never re-runs. The gate sends the
      // whole surface again on every canvas, so replaying canvas k-1 after canvas k would push
      // older component properties back over newer ones. Per surface (T-0154 B5): a newer
      // surface created in between does not make an older surface's stale canvas new again.
      const last = appliedIdx.has(surfaceId) ? appliedIdx.get(surfaceId) : null;
      if (last !== null && idx <= last) {
        note(`canvas ${idx} of ${surfaceId} arrived after canvas ${last} and was skipped`);
        return false;
      }

      canvases += 1;
      for (const a2ui of messages) {
        const { kind, surfaceId: addressed } = target(a2ui);
        if (kind === DELETE) {
          // v1 is one surface per turn (T-0010 A7) and the gate never emits this. Executing it
          // would blank a rendered surface, so it is dropped and said out loud.
          note(`deleteSurface for ${addressed} was dropped: the gate emits one surface per turn`);
          continue;
        }
        if (kind === CREATE) {
          if (addressed && created.has(addressed)) continue; // criterion 7: idempotent by design
          if (apply(a2ui) && addressed) {
            created.add(addressed);
            surfaces += 1;
            mounted = addressed;
          }
          continue;
        }
        // A message for a surface the engine does not hold (its createSurface failed, or the
        // canvas never carried one) would throw `Surface … not found` for every component.
        if (addressed && !created.has(addressed)) {
          note(`a2ui ${kind || "message"} for unknown surface ${addressed} was dropped`);
          continue;
        }
        apply(a2ui);
      }
      if (created.has(surfaceId)) appliedIdx.set(surfaceId, idx);
      return true;
    },

    surfaceId() {
      return mounted;
    },

    canvasIdx(id = mounted) {
      return appliedIdx.has(id) ? appliedIdx.get(id) : null;
    },

    stats() {
      return { surfaces, canvases, ignored, errors };
    },
  };
}
