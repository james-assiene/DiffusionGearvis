// The action round-trip's pure half: an A2UI client action -> one `astra.ui.action` on the socket
// (T-0024 criterion 9).
//
// No DOM here either, so `node --test tests/hud` can pin the mapping without a browser. The page
// half (which socket, which surface is rendered) is ./index.js.
//
// The shape is `local_astra/contracts/protocol.AstraUiAction`, and `Contract` is `extra="forbid"`:
// {type, surface_id, name, context, data_model} and nothing else. The A2UI action also carries
// `sourceComponentId` and `timestamp`, for which the contract has **no field** — a named spec gap
// (T-0007/T-0002 add it, with the fake, in one PR). Until then they ride under a single reserved
// `context.a2ui` key; a top-level key would simply be rejected by the codec, which is worse than
// useless because the Conductor would drop the whole action.

export const ACTION_TYPE = "astra.ui.action";

/** The one reserved key this page adds to the model's own context. */
export const A2UI_CONTEXT_KEY = "a2ui";

const isObject = (value) => Boolean(value) && typeof value === "object" && !Array.isArray(value);

/**
 * actionMessage(action, { dataModel, log }) -> the astra.ui.action payload, or null.
 *
 * `action` is the renderer's `ActionListener` argument:
 * `{ name, surfaceId, sourceComponentId, timestamp, context }`.
 *
 * null — nothing may be sent — when the object is not an action at all (a scroll, a tab switch
 * and typing into a TextField raise no A2UI action; only a handler that sends on anything it is
 * handed would emit one), when it has no event name, or when it has no context: an action with no
 * context is not an action the Conductor can route, and dropping it is safer than guessing one
 * (ruling A5).
 */
export function actionMessage(action, { dataModel = null, log } = {}) {
  const note = typeof log === "function" ? log : () => {};
  if (!isObject(action)) return null;
  const name = typeof action.name === "string" ? action.name : "";
  const surfaceId = typeof action.surfaceId === "string" ? action.surfaceId : "";
  if (!name || !surfaceId) return null;
  if (!isObject(action.context)) return null;

  const context = { ...action.context };
  const a2ui = { sourceComponentId: action.sourceComponentId, timestamp: action.timestamp };
  if (A2UI_CONTEXT_KEY in context) {
    // The model's value wins: overwriting it would silently change what the model asked for, and
    // the collision is the model's to fix. One line, and nothing else (ruling A1: the panel's log).
    note(
      `the model's own action context already holds "${A2UI_CONTEXT_KEY}", so the A2UI ` +
        `sourceComponentId=${action.sourceComponentId} and timestamp were dropped rather than ` +
        `overwriting it`,
    );
  } else {
    context[A2UI_CONTEXT_KEY] = a2ui;
  }

  return {
    type: ACTION_TYPE,
    // Criterion 9: the id *inside* the message — the gate rewrote the model's `lesson_1` to
    // `turn-t1` (T-0010 A7) and the renderer keys on the rewritten id. Sending the model's own id
    // back would address a surface the Conductor never had.
    surface_id: surfaceId,
    name,
    context,
    data_model: isObject(dataModel) ? dataModel : null,
  };
}

/**
 * createActionHandler({ send, log, isRendered, dataModel }) -> (action) => boolean
 *
 *   send(message)        the host's socket send (T-0025's seam); its rejection is logged, never
 *                        thrown into the renderer's event dispatch
 *   isRendered(id)       the page only speaks for a surface it actually rendered
 *   dataModel()          the rendered surface's client data model, or null
 *
 * Returns true iff exactly one message was handed to `send` — one per accepted action, never two
 * (a double-send is the failure mode criterion 9 names).
 */
export function createActionHandler({ send, log = () => {}, isRendered = () => true, dataModel } = {}) {
  const note = (line) => {
    try {
      log(line);
    } catch {
      // never take the renderer down over a log line
    }
  };

  return function onAction(action) {
    let model = null;
    try {
      model = typeof dataModel === "function" ? dataModel() : null;
    } catch (err) {
      note(`the client data model could not be read: ${(err && err.message) || err}`);
    }
    const message = actionMessage(action, { dataModel: model, log: note });
    if (!message) return false;
    if (!isRendered(message.surface_id)) {
      note(
        `an action for surface ${message.surface_id} was dropped: this page has not rendered it, ` +
          `so the Conductor could not route it`,
      );
      return false;
    }
    if (typeof send !== "function") {
      note("this host has no socket, so the action was not sent");
      return false;
    }
    try {
      Promise.resolve(send(message)).catch((err) =>
        note(`astra.ui.action was not sent: ${(err && err.message) || err}`),
      );
    } catch (err) {
      note(`astra.ui.action was not sent: ${(err && err.message) || err}`);
      return false;
    }
    return true;
  };
}
