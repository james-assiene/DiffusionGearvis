// The A2UI engine for our own renderer: the processor `./bridge.js` drives, unchanged (T-0164 A1).
//
// It keeps the processor contract the bridge and `./action.js` depend on (the MessageProcessor
// shape they were written against, T-0024; measured 2026-10-27):
//
//   * per-message throws: `Catalog not found: <id>` on a createSurface whose catalogId is not one
//     of `catalogIds` (the page passes the gate's and the v2 one, T-0184 B1),
//     `Surface <id> already exists.` on a second createSurface, and
//     `Surface not found for message: <id>` on an update for a surface it does not hold;
//   * `onSurfaceCreated(handler)` fires synchronously inside the createSurface call;
//   * `getClientDataModel()` holds only the surfaces whose createSurface said `sendDataModel`;
//   * a Button's action context is resolved one level deep, and the action handed to `onAction`
//     is `{ name, surfaceId, sourceComponentId, timestamp: new Date().toISOString(), context }`.
//
// It owns one view per surface (./render.js) and re-renders that view after every message that
// changes it. No DOM access at module top level (T-0024 criterion 2): a document is only touched
// through the one passed to `createEngine({ document })`, and without one the engine still keeps
// surfaces and data models (what `node --test` drives).

import { A2UI_SPEC_VERSION, CARDS_V2_CATALOG_ID, GATE_CATALOG_ID } from "./a2ui.js";
import { createSurfaceView } from "./render.js";

const KINDS = ["createSurface", "updateComponents", "updateDataModel", "deleteSurface"];
/** Path segments the data model refuses (prototype pollution). */
const FORBIDDEN = new Set(["__proto__", "constructor", "prototype"]);

const isObject = (v) => Boolean(v) && typeof v === "object" && !Array.isArray(v);
const isNumeric = (s) => /^\d+$/.test(s);

function parsePath(path) {
  return String(path)
    .split("/")
    .filter((s) => s.length > 0)
    .map((raw) => {
      const seg = raw.replace(/~([01])/g, (_, g) => (g === "1" ? "/" : "~"));
      if (FORBIDDEN.has(seg)) throw new Error(`Forbidden path segment '${seg}' in path '${path}'.`);
      return seg;
    });
}

/** Absolute JSON pointer for a binding path (a relative path resolves against `/`). */
const absolute = (path) => (String(path).startsWith("/") ? String(path) : `/${path}`);

function getPath(data, path) {
  if (path === "/" || path === "") return data;
  let cur = data;
  for (const seg of parsePath(path)) {
    if (cur === undefined || cur === null || typeof cur !== "object") return undefined;
    if (Array.isArray(cur)) {
      if (!isNumeric(seg)) return undefined;
      cur = cur[parseInt(seg, 10)];
    } else {
      if (!Object.prototype.hasOwnProperty.call(cur, seg)) return undefined;
      cur = cur[seg];
    }
  }
  return cur;
}

/** DataModel.set: -> the new root. A missing `value` deletes the key (v0.9.1). */
function setPath(data, path, value) {
  if (path === "/" || path === "") return value;
  const segs = parsePath(path);
  const last = segs.pop();
  let root = data;
  if (root === undefined || root === null) root = {};
  else if (typeof root !== "object") throw new Error(`Cannot set path '${path}': the data model root is a primitive value.`);
  let cur = root;
  for (let i = 0; i < segs.length; i++) {
    const seg = segs[i];
    const nextSeg = i < segs.length - 1 ? segs[i + 1] : last;
    if (Array.isArray(cur) && !isNumeric(seg)) {
      throw new Error(`Cannot use non-numeric segment '${seg}' on an array in path '${path}'.`);
    }
    const key = Array.isArray(cur) ? parseInt(seg, 10) : seg;
    const has = Array.isArray(cur) ? key < cur.length : Object.prototype.hasOwnProperty.call(cur, key);
    const v = has ? cur[key] : undefined;
    if (v !== undefined && v !== null && typeof v !== "object") {
      throw new Error(`Cannot set path '${path}': segment '${seg}' is a primitive value.`);
    }
    if (v === undefined || v === null) cur[key] = isNumeric(nextSeg) ? [] : {};
    cur = cur[key];
  }
  if (Array.isArray(cur) && !isNumeric(last)) {
    throw new Error(`Cannot use non-numeric segment '${last}' on an array in path '${path}'.`);
  }
  if (value === undefined) {
    if (Array.isArray(cur)) cur[parseInt(last, 10)] = undefined;
    else delete cur[last];
  } else {
    cur[Array.isArray(cur) ? parseInt(last, 10) : last] = value;
  }
  return root;
}

const copy = (v) => (v === undefined ? undefined : JSON.parse(JSON.stringify(v)));

/**
 * createEngine({ document, onAction, log, catalogIds }) -> engine
 *
 *   engine.processor          `{ processMessages, onSurfaceCreated, getClientDataModel }` for
 *                             `createSurfaceBridge({ processor })`
 *   engine.dataModelFor(id)   the surface's data model iff its createSurface said
 *                             `sendDataModel: true`, else null (what AstraUiAction.data_model carries)
 *   engine.element(id)        the surface's card root `[data-a2ui-root]`, or null
 *   engine.drop(id)           remove the surface's DOM and state; a later createSurface of the
 *                             same id starts fresh (the hook T-0162 D calls on clear)
 *   engine.surfaces()         the ids the engine holds, in creation order
 */
export function createEngine({ document: doc = null, onAction, log = () => {}, catalogIds = [GATE_CATALOG_ID] } = {}) {
  const note = (line) => {
    try {
      log(line);
    } catch {
      // never take the engine down over a log line
    }
  };
  const catalogs = new Set(catalogIds);
  const surfaces = new Map(); // id -> { id, sendDataModel, components: Map, data, view }
  const handlers = [];

  function emitAction(action) {
    if (typeof onAction !== "function") return;
    try {
      onAction(action);
    } catch (err) {
      note(`a2ui action handler failed: ${(err && err.message) || err}`);
    }
  }

  /** A Button's `{event:{name, context}}`, resolved one level deep, as `./action.js` expects it. */
  function dispatch(surface, action, sourceComponentId) {
    const event = action.event;
    const context = {};
    if (isObject(event.context)) {
      for (const [key, value] of Object.entries(event.context)) {
        if (isObject(value) && typeof value.path === "string" && !("call" in value)) {
          context[key] = copy(readData(surface, value.path));
        } else if (isObject(value) && typeof value.call === "string") {
          note(`a2ui surface ${surface.id}: a FunctionCall in an action context is not evaluated in v0.1`);
          context[key] = undefined;
        } else {
          context[key] = value;
        }
      }
    }
    emitAction({
      name: event.name,
      surfaceId: surface.id,
      sourceComponentId,
      timestamp: new Date().toISOString(),
      context,
    });
  }

  function readData(surface, path) {
    try {
      return getPath(surface.data, absolute(path));
    } catch {
      return undefined;
    }
  }

  function render(surface) {
    if (!surface.view) return;
    try {
      surface.view.render();
    } catch (err) {
      note(`a2ui surface ${surface.id} did not render: ${(err && err.message) || err}`);
    }
  }

  function createSurface(payload) {
    const { surfaceId, catalogId, sendDataModel } = payload;
    if (!catalogs.has(catalogId)) throw new Error(`Catalog not found: ${catalogId}`);
    if (surfaces.has(surfaceId)) throw new Error(`Surface ${surfaceId} already exists.`);
    const surface = {
      id: surfaceId,
      sendDataModel: sendDataModel === true,
      components: new Map(),
      data: {}, // a sendDataModel surface with no data yet reports {} (review F2)
      view: null,
    };
    if (doc && typeof doc.createElement === "function") {
      surface.view = createSurfaceView({
        document: doc,
        surfaceId,
        components: () => surface.components,
        read: (path) => readData(surface, path),
        write: (path, value) => {
          try {
            surface.data = setPath(surface.data, absolute(path), copy(value));
          } catch (err) {
            note(`a2ui surface ${surfaceId}: an edit at ${path} was not stored: ${(err && err.message) || err}`);
          }
        },
        dispatch: (action, componentId) => dispatch(surface, action, componentId),
        log: note,
        // T-0184 B2: the v2 kinds are drawn only on a surface created under the v2 catalog
        v2: catalogId === CARDS_V2_CATALOG_ID,
      });
      render(surface);
    }
    surfaces.set(surfaceId, surface);
    const created = { id: surfaceId, element: surface.view ? surface.view.root : null };
    for (const handler of handlers.slice()) {
      try {
        handler(created);
      } catch (err) {
        note(`a2ui onSurfaceCreated handler failed: ${(err && err.message) || err}`);
      }
    }
  }

  function surfaceFor(payload) {
    const surface = surfaces.get(payload.surfaceId);
    if (!surface) throw new Error(`Surface not found for message: ${payload.surfaceId}`);
    return surface;
  }

  function updateComponents(payload) {
    const surface = surfaceFor(payload);
    const list = Array.isArray(payload.components) ? payload.components : [];
    for (const comp of list) {
      if (!isObject(comp) || typeof comp.id !== "string" || !comp.id) {
        throw new Error(`Component '${comp && comp.component}' is missing an 'id'.`);
      }
      if (!comp.component && !surface.components.has(comp.id)) {
        throw new Error(`Cannot create component ${comp.id} without a type.`);
      }
    }
    for (const comp of list) {
      const existing = surface.components.get(comp.id);
      const component = comp.component || (existing && existing.component);
      surface.components.set(comp.id, { ...comp, component });
    }
    render(surface);
  }

  function updateDataModel(payload) {
    const surface = surfaceFor(payload);
    const path = payload.path || "/";
    surface.data = setPath(surface.data, path, copy(payload.value));
    render(surface);
  }

  function processMessage(message) {
    if (!isObject(message)) throw new Error("A2UI message is not an object.");
    const present = KINDS.filter((k) => k in message);
    if (present.length > 1) throw new Error(`Message contains multiple update types: ${present.join(", ")}.`);
    const kind = present[0];
    if (!kind) return;
    const payload = message[kind];
    if (!isObject(payload)) throw new Error(`${kind} has no payload.`);
    if (kind === "createSurface") return createSurface(payload);
    if (!payload.surfaceId) return;
    if (kind === "deleteSurface") return drop(payload.surfaceId);
    if (kind === "updateComponents") return updateComponents(payload);
    return updateDataModel(payload);
  }

  function drop(surfaceId) {
    const surface = surfaces.get(surfaceId);
    if (!surface) return false;
    surfaces.delete(surfaceId);
    if (surface.view) surface.view.destroy();
    return true;
  }

  const processor = {
    processMessages(messages) {
      const list = Array.isArray(messages) ? messages : messages && Array.isArray(messages.messages) ? messages.messages : [];
      for (const message of list) processMessage(message);
    },
    onSurfaceCreated(handler) {
      if (typeof handler !== "function") return { unsubscribe() {} };
      handlers.push(handler);
      return {
        unsubscribe() {
          const at = handlers.indexOf(handler);
          if (at >= 0) handlers.splice(at, 1);
        },
      };
    },
    getClientDataModel() {
      const out = {};
      for (const surface of surfaces.values()) {
        if (surface.sendDataModel) out[surface.id] = copy(surface.data);
      }
      return Object.keys(out).length ? { version: A2UI_SPEC_VERSION, surfaces: out } : undefined;
    },
  };

  return {
    processor,
    dataModelFor(surfaceId) {
      const surface = surfaceId ? surfaces.get(surfaceId) : null;
      if (!surface || !surface.sendDataModel) return null;
      return isObject(surface.data) ? copy(surface.data) : null;
    },
    element(surfaceId) {
      const surface = surfaces.get(surfaceId);
      return surface && surface.view ? surface.view.root : null;
    },
    drop,
    surfaces() {
      return [...surfaces.keys()];
    },
  };
}
