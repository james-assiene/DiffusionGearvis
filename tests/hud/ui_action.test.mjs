// T-0024 — the action round-trip's pure half: hud/src/ui/action.js.
//
// Criterion 9 of docs/tasks/T-0024.md: the A2UI client action the renderer raises becomes exactly
// one `astra.ui.action` on the socket. The click itself and the production-codec decode are in
// tests/integration/test_hud_ui_surface.py; here the mapping is pinned with no DOM and no bundle.
//
// This file *skips itself* while hud/src/ui/action.js is absent (same convention as
// tests/hud/assists.test.mjs; node tests cannot carry pytest's `pending` marker).
//
// API this pins:
//
//   hud/src/ui/action.js — no DOM at module top level
//     ACTION_TYPE     = "astra.ui.action"
//     A2UI_CONTEXT_KEY = "a2ui"   the single reserved key the A2UI-only fields ride under
//     actionMessage(action, { dataModel = null, log } = {}) -> object | null
//       action is the renderer's `ActionListener` argument
//       ({ name, surfaceId, sourceComponentId, timestamp, context }); the result is the
//       `astra.ui.action` payload, or null when nothing may be sent.
//     createActionHandler({ send, log, isRendered, dataModel }) -> (action) => boolean
//       isRendered(surfaceId) -> bool   — the page only speaks for a surface it rendered
//       dataModel() -> object | null    — processor.getClientDataModel(), because the fixture's
//                                         createSurface sets sendDataModel: true
//       returns true iff one message was handed to `send`.
//
// The expected payload shape is read off local_astra/contracts/protocol.AstraUiAction, which is
// `extra="forbid"` and has fields {type, surface_id, name, context, data_model} — and, as the
// task's criterion 9 says in as many words, **no** source_component_id. That is why the A2UI-only
// fields are nested under context.a2ui and why a top-level smuggle is a test failure here.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);
const ACTION_JS = fileURLToPath(new URL("hud/src/ui/action.js", REPO));
const INSTALLED = existsSync(ACTION_JS);
const opts = INSTALLED
  ? {}
  : {
      skip:
        "T-0024 is not implemented yet: hud/src/ui/action.js does not exist. This file is part " +
        "of the acceptance suite of docs/tasks/T-0024.md (criterion 9, the action round-trip); " +
        "it runs the moment that module lands.",
    };

let ESM_ROOT = null;
function srcRoot() {
  if (ESM_ROOT) return ESM_ROOT;
  const real = fileURLToPath(new URL("hud/src", REPO));
  const dir = mkdtempSync(join(tmpdir(), "astra-hud-esm-"));
  writeFileSync(join(dir, "package.json"), '{"type":"module","private":true}');
  if (existsSync(real)) cpSync(real, join(dir, "src"), { recursive: true });
  ESM_ROOT = pathToFileURL(join(dir, "src") + "/");
  return ESM_ROOT;
}

async function load(name) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch (err) {
    assert.fail(
      `T-0024 adds hud/src/${name}, importable in node as an ES module with no DOM at module ` +
        `top level. node said: ${err.message}`,
    );
  }
}

//: The fields of local_astra/contracts/protocol.AstraUiAction (Contract is extra="forbid", so a
//: payload with any other key is rejected by the Conductor's codec, not merely ignored).
const ALLOWED_KEYS = ["context", "data_model", "name", "surface_id", "type"];

//: What `@a2ui/web_core`'s ActionListener is handed when the quiz's check button is clicked,
//: with the *gate's* surface id (T-0010 A7 rewrites the model's `lesson_1` to `turn-t1`) and the
//: model's own context from tests/fixtures/a2ui/quiz_example.jsonl (`check_btn`'s action.event:
//: {"name":"check_quiz","context":{"quizId":"fractions_1","answers":{"path":"/answers"}}}), with
//: `answers` already resolved by the binder to the chosen values.
const CLICK = Object.freeze({
  name: "check_quiz",
  surfaceId: "turn-t1",
  sourceComponentId: "check_btn",
  timestamp: "2026-09-14T10:00:00.000Z",
  context: { quizId: "fractions_1", answers: { q1: ["b"], q2: ["a"], q3: ["c"], q4: [], q5: [] } },
});

const DATA_MODEL = Object.freeze({
  answers: { q1: ["b"], q2: ["a"], q3: ["c"], q4: [], q5: [] },
  feedback: { q1: "", q2: "", q3: "", q4: "", q5: "" },
  score: "",
});

function handlerWith(mod, { isRendered = () => true, dataModel = () => DATA_MODEL } = {}) {
  const sent = [];
  const logged = [];
  const handler = mod.createActionHandler({
    send: (message) => {
      sent.push(message);
      return Promise.resolve();
    },
    log: (line) => logged.push(String(line)),
    isRendered,
    dataModel,
  });
  return { handler, sent, logged };
}

// --- criterion 9: the payload -------------------------------------------------------------

test("criterion 9: the click becomes one astra.ui.action with exactly the contract's fields", opts, async () => {
  const mod = await load("ui/action.js");
  assert.equal(mod.ACTION_TYPE, "astra.ui.action", "contracts CLIENT_TO_SERVER names this type");
  const msg = mod.actionMessage(CLICK, { dataModel: DATA_MODEL });
  assert.ok(msg, "criterion 9: a click on check_btn sends a message");
  assert.deepEqual(
    Object.keys(msg).sort(),
    ALLOWED_KEYS,
    "criterion 9: AstraUiAction is extra='forbid' — {type, surface_id, name, context, " +
      "data_model} and nothing else; the component id has 'nowhere else to go' than context.a2ui " +
      "and must not be smuggled as a top-level key",
  );
  assert.equal(msg.type, "astra.ui.action");
  assert.equal(msg.name, "check_quiz", "the model's own event name");
  assert.deepEqual(
    msg.data_model,
    DATA_MODEL,
    "criterion 9: data_model is populated because the fixture's createSurface sets " +
      "sendDataModel: true",
  );
});

test("criterion 9: surface_id is the id inside the message (turn-t1), never the model's lesson_1", opts, async () => {
  const mod = await load("ui/action.js");
  // The model's own id appears in the data model and in the context, so an implementation that
  // fishes for it has somewhere to fish. The gate rewrote it (T-0010 A7) and the renderer keys
  // on the rewritten id; sending `lesson_1` back would address a surface the Conductor does not
  // know.
  const action = {
    ...CLICK,
    context: { ...CLICK.context, lessonId: "lesson_1" },
  };
  const msg = mod.actionMessage(action, { dataModel: { ...DATA_MODEL, surfaceId: "lesson_1" } });
  assert.equal(
    msg.surface_id,
    "turn-t1",
    "criterion 9: surface_id == the gate's rewritten id, taken from the action the renderer " +
      "raised, not from anything the model wrote",
  );
});

test("criterion 9: the model's context survives and the A2UI-only fields ride under context.a2ui", opts, async () => {
  const mod = await load("ui/action.js");
  assert.equal(mod.A2UI_CONTEXT_KEY, "a2ui");
  const msg = mod.actionMessage(CLICK, { dataModel: DATA_MODEL });
  assert.equal(msg.context.quizId, "fractions_1", "the model's own context key is untouched");
  assert.deepEqual(
    msg.context.answers,
    CLICK.context.answers,
    "criterion 9: `answers` is the binder's resolved value, passed through unchanged",
  );
  assert.deepEqual(
    Object.keys(msg.context).sort(),
    ["a2ui", "answers", "quizId"],
    "criterion 9: one reserved key, added to the model's context — nothing else is invented",
  );
  assert.equal(msg.context.a2ui.sourceComponentId, "check_btn");
  assert.equal(msg.context.a2ui.timestamp, CLICK.timestamp);
  assert.ok(
    !("source_component_id" in msg) && !("sourceComponentId" in msg) && !("timestamp" in msg),
    "criterion 9, the named spec gap: AstraUiAction has no source_component_id field and is " +
      "extra='forbid'; adding one is T-0007's contracts change, and smuggling it as a top-level " +
      "key would simply be rejected by the codec",
  );
});

test("criterion 9: if the model's own context holds `a2ui`, the model's value wins and one line is logged", opts, async () => {
  const mod = await load("ui/action.js");
  const modelValue = { mine: true, note: "the model wrote this" };
  const logged = [];
  const msg = mod.actionMessage(
    { ...CLICK, context: { quizId: "fractions_1", a2ui: modelValue } },
    { dataModel: DATA_MODEL, log: (line) => logged.push(String(line)) },
  );
  assert.deepEqual(
    msg.context.a2ui,
    modelValue,
    "criterion 9: 'the model's value wins and the page writes an error-strip line rather than " +
      "overwriting it silently'",
  );
  assert.equal(
    logged.length,
    1,
    `criterion 9: exactly one error line about the collision; got ${logged.join(" | ")}`,
  );
  assert.ok(
    logged[0].includes("a2ui"),
    `the line must name the colliding key; got ${logged.join(" | ")}`,
  );
});

test("criterion 9: no data model means data_model is null, not an invented object", opts, async () => {
  const mod = await load("ui/action.js");
  const msg = mod.actionMessage(CLICK, {});
  assert.equal(
    msg.data_model,
    null,
    "AstraUiAction.data_model is `dict | None`; a surface created without sendDataModel sends " +
      "null, never {} presented as the user's answers",
  );
});

// --- criterion 9, the send path and its negatives --------------------------------------------

test("criterion 9: the handler sends exactly one message per action it accepts", opts, async () => {
  const mod = await load("ui/action.js");
  const { handler, sent } = handlerWith(mod);
  assert.equal(handler(CLICK), true);
  assert.equal(
    sent.length,
    1,
    "criterion 9 negative: 'exactly one message per click — a double-send is the failure mode to " +
      "test'",
  );
  assert.equal(sent[0].type, "astra.ui.action");
});

test("adversarial: an action for a surface the page never rendered sends nothing", opts, async () => {
  const mod = await load("ui/action.js");
  const { handler, sent, logged } = handlerWith(mod, { isRendered: (id) => id === "turn-t1" });
  assert.equal(handler({ ...CLICK, surfaceId: "lesson_1" }), false);
  assert.deepEqual(sent, [], "no message is sent at all for a surface the page never rendered");
  assert.ok(logged.length >= 1, "and the page says so rather than failing silently");
});

test("adversarial: an action with no context, or no name, sends nothing and does not throw", opts, async () => {
  const mod = await load("ui/action.js");
  const { handler, sent } = handlerWith(mod);
  for (const bad of [
    { ...CLICK, context: undefined },
    { ...CLICK, name: "" },
    { ...CLICK, name: undefined },
    {},
    null,
  ]) {
    let result;
    assert.doesNotThrow(() => {
      result = handler(bad);
    }, `the handler must survive ${JSON.stringify(bad)}`);
    assert.equal(result, false, `no message for ${JSON.stringify(bad)}`);
  }
  assert.deepEqual(
    sent,
    [],
    "docs/tasks/T-0024.md, Tests to write: \"an action whose context is undefined or whose " +
      'surfaceId is a surface the page never rendered -> no message is sent at all"',
  );
});

test("adversarial: nothing the renderer raises other than an action reaches the socket", opts, async () => {
  const mod = await load("ui/action.js");
  const { handler, sent } = handlerWith(mod);
  // The negative of criterion 9: scroll, tab switch and typing into a TextField do not raise an
  // A2UI action at all, so the only way they could reach the socket is a handler that sends on
  // anything it is handed. A pure mapping of a non-action object must produce nothing.
  for (const notAnAction of [
    { type: "scroll", surfaceId: "turn-t1" },
    { surfaceId: "turn-t1", context: { path: "/answers/q1", value: ["b"] } },
    "check_quiz",
    42,
  ]) {
    assert.doesNotThrow(() => handler(notAnAction));
  }
  assert.deepEqual(sent, [], "criterion 9 negative: no action is sent for an event that is not one");
});
