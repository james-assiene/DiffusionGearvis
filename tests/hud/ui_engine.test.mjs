// T-0164 group A: the engine (hud/src/ui/engine.js), node rows. docs/tasks/T-0164.md A1, A9, A10.
//
// The renderer rows (A2-A13) are in tests/hud/ui_render.test.mjs, and so is the contract both modules pin:
//
//   const engine = createEngine({ document, onAction, log });
//   engine.processor.processMessages([message])   // bridge.js drives this, unchanged
//   engine.processor.onSurfaceCreated(handler)    // handler({ id, element }) synchronously inside createSurface
//   engine.dataModelFor(surfaceId) -> object|null // iff that surface's createSurface.sendDataModel === true
//   engine.drop(surfaceId)                        // removes the surface's DOM and state (the T-0162 D hook)
//
// The trapping fake DOM is the one in ui_render.test.mjs: this module imports exactly the region between its
// `// <fake-dom>` markers as a data: module, so one double (not two that drift) serves both. Expected values
// come from the fixture (tests/fixtures/a2ui/genui_census_20261027.jsonl, built in Python through UiGate) and
// from the vendor behaviour the task records (implementation note 1), never from the engine.
//
// Pending: strict xfail until docs/tasks/T-0164.md's Status records "group A: tests …, activated <sha>"
// (the T-0162/T-0159 node mechanism); ASTRA_RUN_PENDING=1 runs the rows for real.

import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";
import { join } from "node:path";

const REPO = fileURLToPath(new URL("../../", import.meta.url));
const GATE_CATALOG_ID = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json";
/** The id the vendored renderer's own catalog carried; the gate never emits it, so it is foreign now. */
const VENDOR_CATALOG_ID = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json";

const T0164_TASK = join(REPO, "docs/tasks/T-0164.md");
const T0164_STATUS = existsSync(T0164_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0164_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const T0164_A_PENDING =
  !/\bgroup A:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i.test(T0164_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function rowA(name, fn) {
  test(name + (T0164_A_PENDING ? " [pending T-0164 A: strict xfail until group A is activated]" : ""), async (t) => {
    if (!T0164_A_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail("XPASS(strict): this T-0164 group A row passed while docs/tasks/T-0164.md does not record group A's activation.");
    }
    t.diagnostic("expected failure (T-0164 A pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

let ESM_ROOT = null;
function srcRoot() {
  if (ESM_ROOT) return ESM_ROOT;
  const dir = mkdtempSync(join(tmpdir(), "astra-hud-t0164a-engine-"));
  writeFileSync(join(dir, "package.json"), '{"type":"module","private":true}');
  cpSync(join(REPO, "hud/src"), join(dir, "src"), { recursive: true });
  ESM_ROOT = pathToFileURL(join(dir, "src") + "/");
  return ESM_ROOT;
}
async function load(name) {
  if (!existsSync(join(REPO, "hud/src/ui", name))) assert.fail(`hud/src/ui/${name} does not exist yet (T-0164 A owned file; the implementer adds it)`);
  try {
    return await import(new URL(`ui/${name}`, srcRoot()).href);
  } catch (err) {
    assert.fail(`hud/src/ui/${name} must import in node with no document (A1): ${err && err.stack}`);
  }
}

let FAKE = null;
async function fakeDom() {
  if (!FAKE) {
    const src = readFileSync(join(REPO, "tests/hud/ui_render.test.mjs"), "utf8");
    const m = /\/\/ <fake-dom>[^\n]*\n([\s\S]*?)\n\/\/ <\/fake-dom>/.exec(src);
    assert.ok(m, "tests/hud/ui_render.test.mjs holds the fake DOM between // <fake-dom> markers");
    FAKE = await import("data:text/javascript," + encodeURIComponent(m[1]));
  }
  return FAKE.makeDom();
}

const SURFACES = readFileSync(join(REPO, "tests/fixtures/a2ui/genui_census_20261027.jsonl"), "utf8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l))
  .slice(1);
const byKey = (key) => SURFACES.find((s) => s.key === key);

const CREATE = (id, extra = {}) => ({ version: "v0.9.1", createSurface: { surfaceId: id, catalogId: GATE_CATALOG_ID, ...extra } });
const COMPONENTS = (id, components) => ({ version: "v0.9.1", updateComponents: { surfaceId: id, components } });
const DATA = (id, path, value) => ({ version: "v0.9.1", updateDataModel: { surfaceId: id, path, value } });
const TEXT_ROOT = (word) => [{ id: "root", component: "Column", children: ["t"] }, { id: "t", component: "Text", text: word }];

async function engineWith() {
  const { createEngine } = await load("engine.js");
  const dom = await fakeDom();
  const actions = [];
  const logs = [];
  const created = [];
  const engine = createEngine({ document: dom.document, onAction: (a) => actions.push(a), log: (l) => logs.push(String(l)) });
  let inside = false;
  const apply = (message) => {
    inside = true;
    try {
      engine.processor.processMessages([message]);
    } finally {
      inside = false;
    }
  };
  engine.processor.onSurfaceCreated((surface) => {
    created.push({ surface, inside });
    if (surface && surface.element) dom.quiet(() => dom.document.body.appendChild(surface.element));
  });
  return { engine, dom, actions, logs, created, apply };
}

/** Every message of a fixture surface, with the vendor's per-message granularity. */
const messagesOf = (s) => s.canvases.flat();

// =========================================================================================================
// A1: the processor contract
// =========================================================================================================

rowA("A1: processMessages throws exactly where the vendor did: a duplicate create, an unknown surface, a foreign catalog", async () => {
  const h = await engineWith();
  // the guard's other side first: a well-formed sequence does not throw
  h.apply(CREATE("ok"));
  h.apply(COMPONENTS("ok", TEXT_ROOT("fine")));
  h.apply(DATA("ok", "/", { a: 1 }));
  assert.throws(() => h.apply(CREATE("ok")), /already exists/i, "a second createSurface for an existing id throws (Surface … already exists)");
  assert.throws(() => h.apply(COMPONENTS("nope", TEXT_ROOT("x"))), Error, "updateComponents for an unknown surface throws");
  assert.throws(() => h.apply(DATA("nope", "/", {})), Error, "updateDataModel for an unknown surface throws");
  const createdBefore = h.created.length;
  for (const foreign of [VENDOR_CATALOG_ID, "https://example.com/catalog.json", ""]) {
    assert.throws(() => h.apply({ version: "v0.9.1", createSurface: { surfaceId: `f-${foreign.length}`, catalogId: foreign } }), /catalog/i, `a createSurface with catalogId ${JSON.stringify(foreign)} throws (Catalog not found)`);
  }
  assert.equal(h.created.length, createdBefore, "a foreign catalog renders nothing: no surface is handed out");
  assert.throws(() => h.apply(COMPONENTS(`f-${VENDOR_CATALOG_ID.length}`, TEXT_ROOT("x"))), Error, "and the refused surface does not exist afterwards");
  assert.ok(!h.dom.document.body.textContent.includes("x"), "nothing of it rendered");
});

// T-0184 B1 amends the row above by name: the page's engine holds two catalogs, the basic one and
// `gearvis:cards/v2` (a2ui.js CARDS_V2_CATALOG_ID); every other id still throws and renders nothing.
// Strict xfail until docs/tasks/T-0184.md's Status block records "group B: tests …, activated <sha>".
const T0184_TASK = join(REPO, "docs/tasks/T-0184.md");
const T0184_STATUS = existsSync(T0184_TASK)
  ? ((/^\*\*Status:\*\*([\s\S]*?)\n\n\*\*/m.exec(readFileSync(T0184_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const T0184_B_PENDING =
  !/\bgroup B:[^;)\n]*\bactivated\s+[0-9a-f]{7,40}\b/i.test(T0184_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function rowT0184B(name, fn) {
  test(name + (T0184_B_PENDING ? " [pending T-0184 B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!T0184_B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) assert.fail("XPASS(strict): this T-0184 group B row passed while docs/tasks/T-0184.md does not record group B's activation.");
    t.diagnostic("expected failure (T-0184 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

const V2_CATALOG_ID = "gearvis:cards/v2"; // H-T184-4, hand-copied

rowT0184B("A1 as amended by T-0184 B1: an engine with both ids takes the basic and the v2 catalog and throws on every other id", async () => {
  const a2ui = await load("a2ui.js");
  assert.equal(a2ui.CARDS_V2_CATALOG_ID, V2_CATALOG_ID, "a2ui.js names the v2 id");
  const { createEngine } = await load("engine.js");
  const dom = await fakeDom();
  const created = [];
  const engine = createEngine({ document: dom.document, onAction: () => {}, log: () => {}, catalogIds: [GATE_CATALOG_ID, a2ui.CARDS_V2_CATALOG_ID] });
  engine.processor.onSurfaceCreated((s) => created.push(s));
  const apply = (m) => engine.processor.processMessages([m]);
  apply({ version: "v0.9.1", createSurface: { surfaceId: "b", catalogId: GATE_CATALOG_ID } });
  apply({ version: "v0.9.1", createSurface: { surfaceId: "v", catalogId: V2_CATALOG_ID } });
  apply(COMPONENTS("v", [{ id: "root", component: "Stat", label: "V2", value: "rendered" }]));
  assert.ok(created.find((c) => c.id === "v").element.textContent.includes("rendered"), "the v2 surface's Stat renders");
  for (const foreign of [VENDOR_CATALOG_ID, "gearvis:cards/v1", "gearvis:cards/v2/", "GEARVIS:CARDS/V2", "https://example.com/catalog.json", ""]) {
    assert.throws(() => apply({ version: "v0.9.1", createSurface: { surfaceId: `f-${foreign}`, catalogId: foreign } }), /catalog/i, `catalogId ${JSON.stringify(foreign)} throws`);
  }
  assert.deepEqual(created.map((c) => c.id), ["b", "v"], "no foreign surface was handed out");
});

rowA("A1: onSurfaceCreated hands { id } once per id, synchronously inside the createSurface call", async () => {
  const h = await engineWith();
  h.apply(CREATE("s1"));
  h.apply(COMPONENTS("s1", TEXT_ROOT("one")));
  h.apply(CREATE("s2", { sendDataModel: true }));
  assert.throws(() => h.apply(CREATE("s1")));
  assert.deepEqual(h.created.map((c) => c.surface && c.surface.id), ["s1", "s2"], "once per id, in order");
  assert.ok(h.created.every((c) => c.inside), "the handler ran inside processMessages (synchronously), not later");
  assert.ok(h.created.every((c) => c.surface.element && c.surface.element.hasAttribute("data-a2ui-root")), "with the card root, [data-a2ui-root]");
});

rowA("A1: the real bridge.js drives the engine unchanged over every fixture surface's canvases with stats().errors == 0", async () => {
  const { createSurfaceBridge } = await load("bridge.js");
  const h = await engineWith();
  const bridgeLogs = [];
  const bridge = createSurfaceBridge({ processor: h.engine.processor, log: (l) => bridgeLogs.push(l) });
  let canvases = 0;
  for (const s of SURFACES) {
    s.canvases.forEach((messages, k) => {
      assert.equal(bridge.handle({ type: "astra.ui.surface", surface_id: s.surface_id, canvas_idx: s.canvas_idxs[k], messages, final: k === s.canvases.length - 1 }), true);
      canvases += 1;
    });
  }
  assert.deepEqual(bridge.stats(), { surfaces: SURFACES.length, canvases, ignored: 0, errors: 0 }, `bridge log: ${bridgeLogs.join(" | ")}`);
  assert.deepEqual(h.created.map((c) => c.surface.id), SURFACES.map((s) => s.surface_id));
  assert.equal(bridge.surfaceId(), SURFACES[SURFACES.length - 1].surface_id);
});

// =========================================================================================================
// A9: dataModelFor and drop
// =========================================================================================================

rowA("A9: dataModelFor is the surface's data model iff createSurface.sendDataModel is true (both directions)", async () => {
  const quiz = byKey("quiz");
  assert.equal(quiz.send_data_model, true);
  const seeded = messagesOf(quiz).find((m) => m.updateDataModel && m.updateDataModel.path === "/").updateDataModel.value;
  const h = await engineWith();
  for (const m of messagesOf(quiz)) h.apply(m);
  assert.deepEqual(h.engine.dataModelFor(quiz.surface_id), seeded, "sendDataModel: true -> the model the fixture's updateDataModel set");
  // the same messages on surfaces whose createSurface leaves sendDataModel out, or sets it false
  for (const [id, flag] of [["quiet-absent", undefined], ["quiet-false", false]]) {
    for (const m of messagesOf(quiz)) {
      const copy = JSON.parse(JSON.stringify(m).replaceAll(quiz.surface_id, id));
      if (copy.createSurface) {
        delete copy.createSurface.sendDataModel;
        if (flag !== undefined) copy.createSurface.sendDataModel = flag;
      }
      h.apply(copy);
    }
    assert.equal(h.engine.dataModelFor(id), null, `${id}: no sendDataModel -> null, though the surface holds a model`);
  }
  assert.equal(h.engine.dataModelFor("never-created"), null);
  assert.equal(h.engine.dataModelFor(null), null);
  h.apply(DATA(quiz.surface_id, "/score", "4 of 5"));
  assert.equal(h.engine.dataModelFor(quiz.surface_id).score, "4 of 5", "it is the live model, not a snapshot of the first canvas");
});

rowA("A9: the quiz edits on a surface without sendDataModel leave dataModelFor === null", async () => {
  const quiz = byKey("quiz");
  const h = await engineWith();
  const id = "quiz-unsent";
  for (const m of messagesOf(quiz)) {
    const copy = JSON.parse(JSON.stringify(m).replaceAll(quiz.surface_id, id));
    if (copy.createSurface) delete copy.createSurface.sendDataModel;
    h.apply(copy);
  }
  const root = h.created.find((c) => c.surface.id === id).surface.element;
  const all = (n) => [n, ...n.childNodes.flatMap((c) => (c.nodeType === 1 ? all(c) : []))];
  const chip = all(root).find((e) => e.localName === "button" && e.textContent.trim() === "3/4");
  assert.ok(chip, "the quiz rendered its chips");
  h.dom.quiet(() => chip.click());
  assert.equal(h.engine.dataModelFor(id), null);
});

rowA("F2: a sendDataModel surface with no data yet reports {} from dataModelFor and in getClientDataModel (vendor parity)", async () => {
  const h = await engineWith();
  h.apply(CREATE("empty", { sendDataModel: true }));
  h.apply(COMPONENTS("empty", TEXT_ROOT("no data yet")));
  h.apply(CREATE("unsent"));
  assert.deepEqual(h.engine.dataModelFor("empty"), {}, "ruling 639d6037: {} , not null, before any updateDataModel");
  assert.equal(h.engine.dataModelFor("unsent"), null, "and a surface without sendDataModel stays null");
  assert.equal(typeof h.engine.processor.getClientDataModel, "function", "the processor keeps the vendor's getClientDataModel");
  const client = h.engine.processor.getClientDataModel();
  assert.ok(client && client.surfaces && typeof client.surfaces === "object", "getClientDataModel() -> { surfaces: { id: model } }");
  assert.deepEqual(client.surfaces.empty, {}, "it carries {} for the empty sendDataModel surface");
  assert.equal("unsent" in client.surfaces, false, "and omits the surface without sendDataModel");
});

rowA("A9: drop(surfaceId) removes the surface's DOM and state; a later create of the same id is fresh", async () => {
  const h = await engineWith();
  h.apply(CREATE("d", { sendDataModel: true }));
  h.apply(COMPONENTS("d", TEXT_ROOT("old words")));
  h.apply(DATA("d", "/", { kept: "no" }));
  const first = h.created[0].surface.element;
  assert.ok(first.isConnected && first.textContent.includes("old words"), "precondition: rendered");
  h.engine.drop("d");
  assert.equal(first.isConnected, false, "the card root is detached");
  assert.equal(h.engine.dataModelFor("d"), null, "the data model is gone");
  assert.throws(() => h.apply(COMPONENTS("d", TEXT_ROOT("x"))), Error, "the surface is unknown again");
  assert.doesNotThrow(() => h.engine.drop("never-was"), "dropping an unknown id is a no-op");
  h.apply(CREATE("d", { sendDataModel: true }));
  assert.equal(h.created.length, 2, "onSurfaceCreated fires again for the re-created id");
  const second = h.created[1].surface.element;
  assert.notEqual(second, first, "a fresh card root");
  assert.ok(!second.textContent.includes("old words"), "no old component survives");
  h.apply(COMPONENTS("d", TEXT_ROOT("new words")));
  assert.ok(second.textContent.includes("new words"));
  assert.ok(!("kept" in (h.engine.dataModelFor("d") || {})), "no old data-model key survives");
});

// =========================================================================================================
// A10: actions, through action.js unchanged
// =========================================================================================================

rowA("A10: the action's timestamp comes from Date.prototype.toISOString, and action.js maps it to one astra.ui.action", async () => {
  const { createActionHandler } = await load("action.js");
  const flight = byKey("trial-flight");
  const h = await engineWith();
  for (const m of messagesOf(flight)) {
    if (m.createSurface && h.created.some((c) => c.surface.id === flight.surface_id)) continue; // the gate re-sends create per canvas; the bridge skips it
    h.apply(m);
  }
  const root = h.created[0].surface.element;
  const all = (n) => [n, ...n.childNodes.flatMap((c) => (c.nodeType === 1 ? all(c) : []))];
  const button = all(root).find((e) => e.localName === "button" && e.textContent.trim() === "Search Flights");
  assert.ok(button, "the gate button renders as a <button> with the model's label");
  const real = Date.prototype.toISOString;
  Date.prototype.toISOString = function () {
    return "2001-02-03T04:05:06.789Z";
  };
  try {
    h.dom.quiet(() => button.click());
  } finally {
    Date.prototype.toISOString = real;
  }
  assert.equal(h.actions.length, 1);
  assert.equal(h.actions[0].timestamp, "2001-02-03T04:05:06.789Z", "timestamp: new Date().toISOString()");
  const sent = [];
  const handler = createActionHandler({ send: (m) => sent.push(m), isRendered: () => true, dataModel: () => h.engine.dataModelFor(flight.surface_id) });
  assert.equal(handler(h.actions[0]), true);
  assert.deepEqual(sent, [
    {
      type: "astra.ui.action",
      surface_id: flight.surface_id,
      name: "__gate.clicked",
      context: { label: "Search Flights", a2ui: { sourceComponentId: "btn", timestamp: "2001-02-03T04:05:06.789Z" } },
      data_model: null,
    },
  ], "the wire message is what action.js makes of the vendor's action shape; the flight sets no sendDataModel");
});
