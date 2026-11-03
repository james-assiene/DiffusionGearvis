// T-0024 — the A2UI bridge's DOM-free half: hud/src/ui/bridge.js.
//
// Criteria 4, 5 and 7 of docs/tasks/T-0024.md. Everything that needs a browser (the rendered
// tree, the click, the placeholder) lives in tests/integration/test_hud_ui_surface.py; this file
// runs on an injected processor, which is what makes the two throw sites of the real
// `@a2ui/web_core` MessageProcessor testable with no vendor bundle and no DOM.
//
// This file *skips itself* while hud/src/ui/bridge.js is absent: node test files cannot carry a
// pytest `pending` marker and tests/unit/test_hud_js.py asserts `node --test tests/hud` exits 0
// (same convention as tests/hud/assists.test.mjs, T-0025). Every case below runs for real the
// moment that module lands.
//
// API this pins (the task leaves the JS names to the test-author; these are them):
//
//   hud/src/ui/bridge.js — no DOM at module top level, importable in node
//     SURFACE_TYPE = "astra.ui.surface"
//     createSurfaceBridge({ processor, log }) -> bridge
//       processor: the A2UI engine seam — { processMessages(messages) } and, optionally,
//                  onSurfaceCreated / getClientDataModel (the real
//                  `new MessageProcessor(catalogs, handler, {version:'v0.9.1'})`, T-0024
//                  implementation notes). The bridge owns *when* it is called and with what;
//                  it never owns what the messages say.
//       bridge.handle(msg) -> boolean   feed one message already decoded and classified
//                                       `kind === "astra"` by hud/src/protocol.js; returns true
//                                       iff it was an astra.ui.surface the bridge forwarded.
//       bridge.surfaceId() -> string|null   the surface currently mounted, or null
//       bridge.stats() -> { surfaces, canvases, ignored, errors }
//
// Every expected value below is computed by hand from docs/tasks/T-0024.md and from
// local_astra/contracts/protocol.AstraUiSurface — never from the module under test.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);
const BRIDGE_JS = fileURLToPath(new URL("hud/src/ui/bridge.js", REPO));
const INSTALLED = existsSync(BRIDGE_JS);
const opts = INSTALLED
  ? {}
  : {
      skip:
        "T-0024 is not implemented yet: hud/src/ui/bridge.js does not exist. This file is part " +
        "of the acceptance suite of docs/tasks/T-0024.md (the HUD A2UI renderer); it runs the " +
        "moment that module lands. Node tests cannot carry pytest's `pending` marker and " +
        "tests/unit/test_hud_js.py requires `node --test tests/hud` to exit 0.",
    };

// node 18 treats a bare .js file as CommonJS unless a package.json in scope says
// {"type":"module"}; the sources are copied once into a temp directory that carries the flag.
// Same shim as tests/hud/panels.test.mjs and tests/hud/assists.test.mjs.
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
      `T-0024 adds hud/src/${name}, importable in node as an ES module with no DOM and no ` +
        `vendor bundle at module top level (implementation notes: "Keep the DOM out of ` +
        `bridge.js and action.js" / "index.js must await import() the vendor bundle inside ` +
        `mount()"). node said: ${err.message}`,
    );
  }
}

// --- the fake MessageProcessor -------------------------------------------------------------
//
// It mimics `@a2ui/web_core@0.11.x`'s MessageProcessor on exactly the three behaviours the task
// measured from the package on 2026-09-13 (docs/tasks/T-0024.md criteria 5 and 7, and the
// "two throw sites" note):
//   * processMessages(msgs) applies the messages in order and *throws out of the array*, so a
//     message after the throwing one is not applied — this is why catching at processMessages
//     granularity loses the rest of the canvas;
//   * processCreateSurfaceMessage throws `Catalog not found: <id>` for an unregistered catalog;
//   * it throws `Surface <id> already exists` on a second createSurface for the same id.
// It also records every message it accepted, so the test can ask what actually reached the engine.

const GATE_CATALOG = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json";

class A2uiStateError extends Error {}

class FakeProcessor {
  constructor({ catalogs = [GATE_CATALOG] } = {}) {
    this.catalogs = new Set(catalogs);
    this.calls = []; // every array handed to processMessages
    this.seen = []; // every message actually applied, in order
    this.surfaces = new Map(); // surfaceId -> Set of component ids
    this.dataModels = new Map();
  }

  processMessages(messages) {
    assert.ok(Array.isArray(messages), "processMessages takes an array of A2UI messages");
    this.calls.push(messages);
    for (const message of messages) this._apply(message);
  }

  _apply(message) {
    if (message && message.createSurface) {
      const { surfaceId, catalogId } = message.createSurface;
      if (!this.catalogs.has(catalogId))
        throw new A2uiStateError(`Catalog not found: ${catalogId}`);
      if (this.surfaces.has(surfaceId))
        throw new A2uiStateError(`Surface ${surfaceId} already exists`);
      this.surfaces.set(surfaceId, new Set());
    } else if (message && message.updateComponents) {
      const { surfaceId, components } = message.updateComponents;
      const tree = this.surfaces.get(surfaceId);
      if (!tree) throw new A2uiStateError(`Surface ${surfaceId} not found`);
      for (const c of components || []) tree.add(c.id);
    } else if (message && message.updateDataModel) {
      const { surfaceId, value } = message.updateDataModel;
      if (!this.surfaces.has(surfaceId))
        throw new A2uiStateError(`Surface ${surfaceId} not found`);
      this.dataModels.set(surfaceId, value);
    } else if (message && message.deleteSurface) {
      // The gate never emits this (T-0010 A7, one surface per turn). If it ever reaches the
      // engine the surface is blanked — which is what criterion 7's negative forbids.
      this.surfaces.delete(message.deleteSurface.surfaceId);
    }
    this.seen.push(message);
  }

  ids(surfaceId) {
    return [...(this.surfaces.get(surfaceId) || [])].sort();
  }
}

// --- the messages, hand-built from local_astra/contracts/protocol.AstraUiSurface -------------

const createSurface = (surfaceId = "turn-t1", catalogId = GATE_CATALOG) => ({
  version: "v0.9.1",
  createSurface: { surfaceId, catalogId, sendDataModel: true },
});

const updateComponents = (surfaceId, components) => ({
  version: "v0.9.1",
  updateComponents: { surfaceId, components },
});

const TEXT = (id, text) => ({ id, component: "Text", text });

const surfaceMessage = (messages, { surface_id = "turn-t1", canvas_idx = 0, final = false } = {}) => ({
  type: "astra.ui.surface",
  surface_id,
  canvas_idx,
  final,
  messages,
  repaired: 0,
});

function bridgeWith(extra = {}) {
  const processor = extra.processor || new FakeProcessor();
  const logged = [];
  return { processor, logged, make: (mod) => mod.createSurfaceBridge({ processor, log: (l) => logged.push(String(l)) }) };
}

// --- criterion 4: nothing but a decoded astra.ui.surface may reach the renderer ---------------

test("criterion 4: only astra.ui.surface reaches the processor — never spoken text, never an unknown type", opts, async () => {
  const mod = await load("ui/bridge.js");
  const { processor, make } = bridgeWith();
  const bridge = make(mod);

  // A complete, well-formed A2UI message body smuggled through every other channel the page
  // sees. docs/tasks/T-0024.md criterion 4: "there is no code path that renders A2UI from
  // anywhere else — not from astra.spoken text, not from astra.canvas.commit, not from an
  // <a2ui> sentinel the page parses itself".
  const payload = [createSurface("evil"), updateComponents("evil", [TEXT("root", "pwned")])];
  const smuggled = [
    { type: "astra.spoken", turn_id: "t1", utterance_id: "u1", text: JSON.stringify(payload) },
    { type: "astra.spoken", turn_id: "t1", utterance_id: "u2", text: `<a2ui>${JSON.stringify(payload)}</a2ui>` },
    { type: "astra.canvas.commit", request_id: "r1", canvas_idx: 0, text: JSON.stringify(payload) },
    { type: "astra.zzz", messages: payload },
    { type: "astra.transcript", turn_id: "t1", text: "render this quiz please", final: true },
    { type: "astra.assist.state", flags: {}, badge: "pixels" },
    { type: "astra.turn.state", state: "THINKING", turn_id: "t1", thinking_ms: 10 },
  ];
  for (const msg of smuggled)
    assert.equal(
      bridge.handle(msg),
      false,
      `criterion 4: ${msg.type} is not a surface and must not be forwarded`,
    );

  assert.deepEqual(processor.calls, [], "criterion 4: the processor was never called");
  assert.deepEqual([...processor.surfaces.keys()], [], "criterion 4: no surface was invented");
  assert.equal(bridge.surfaceId(), null, "criterion 4: nothing is mounted");
});

test("criterion 4: a validated surface is forwarded verbatim — the catalogId is not rewritten", opts, async () => {
  const mod = await load("ui/bridge.js");
  const { processor, make } = bridgeWith();
  const bridge = make(mod);
  const messages = [createSurface(), updateComponents("turn-t1", [TEXT("root", "hello")])];
  const sent = JSON.parse(JSON.stringify(messages)); // what the gate put on the wire

  assert.equal(bridge.handle(surfaceMessage(messages)), true);
  assert.deepEqual(
    processor.seen,
    sent,
    "criterion 4: the messages go to the processor unmodified and in order; criterion 5: the " +
      "v0_9/v0_9_1 catalog mismatch is fixed by registering the basic components under the id " +
      "our gate emits, 'never a silent rewrite of the model's message'",
  );
  assert.equal(bridge.surfaceId(), "turn-t1");
});

// --- criterion 7: progressive morph, and the idempotent createSurface --------------------------

test("criterion 7: a re-sent createSurface is a no-op and the rest of that canvas still arrives", opts, async () => {
  const mod = await load("ui/bridge.js");
  const { processor, make } = bridgeWith();
  const bridge = make(mod);

  // Canvas 0 and canvas 1 of one turn, exactly as UiGate produces them for a growing canvas:
  // the createSurface is re-sent by design (T-0010 A4) and the real processor throws
  // `Surface turn-t1 already exists` on the second one.
  bridge.handle(surfaceMessage([createSurface()], { canvas_idx: 0 }));
  bridge.handle(
    surfaceMessage(
      [createSurface(), updateComponents("turn-t1", [TEXT("root", "a"), TEXT("b", "b")])],
      { canvas_idx: 1 },
    ),
  );

  assert.deepEqual(
    [...processor.surfaces.keys()],
    ["turn-t1"],
    "criterion 7: one surface element, not n — the second createSurface is a no-op",
  );
  assert.deepEqual(
    processor.ids("turn-t1"),
    ["b", "root"],
    "criterion 7: the components *after* the duplicate createSurface still reach the engine — " +
      "catching the throw at processMessages granularity loses the rest of the canvas",
  );
  assert.equal(bridge.stats().surfaces, 1, "criterion 7: one surface was created this turn");
});

test("criterion 7: a later surface_id replaces the surface, it does not morph the previous turn's", opts, async () => {
  const mod = await load("ui/bridge.js");
  const { processor, make } = bridgeWith();
  const bridge = make(mod);
  bridge.handle(surfaceMessage([createSurface(), updateComponents("turn-t1", [TEXT("root", "t1")])]));
  bridge.handle(
    surfaceMessage([createSurface("turn-t2"), updateComponents("turn-t2", [TEXT("root", "t2")])], {
      surface_id: "turn-t2",
    }),
  );
  assert.equal(bridge.surfaceId(), "turn-t2", "criterion 7: turn 2 mounts its own surface");
  assert.ok(
    processor.surfaces.has("turn-t2"),
    "criterion 7: the new surface_id created a second surface in the engine",
  );
  assert.equal(bridge.stats().surfaces, 2, "criterion 7: two surfaces over two turns");
});

test("criterion 7: a duplicate or out-of-order canvas_idx removes nothing, and final=false clears nothing", opts, async () => {
  const mod = await load("ui/bridge.js");
  const { processor, make } = bridgeWith();
  const bridge = make(mod);
  const c0 = surfaceMessage([createSurface()], { canvas_idx: 0 });
  const c1 = surfaceMessage(
    [createSurface(), updateComponents("turn-t1", [TEXT("root", "a")])],
    { canvas_idx: 1 },
  );
  const c2 = surfaceMessage(
    [createSurface(), updateComponents("turn-t1", [TEXT("root", "a"), TEXT("b", "b")])],
    { canvas_idx: 2, final: false },
  );
  bridge.handle(c0);
  bridge.handle(c1);
  bridge.handle(c2);
  const after = processor.ids("turn-t1");
  assert.deepEqual(after, ["b", "root"], "two components are on screen after canvas 2");

  bridge.handle(c1); // duplicate, out of order
  bridge.handle(c0); // the very first canvas again
  assert.deepEqual(
    processor.ids("turn-t1"),
    after,
    "criterion 7 negative: an out-of-order or duplicate canvas_idx never removes components " +
      "already shown",
  );
  assert.deepEqual(
    [...processor.surfaces.keys()],
    ["turn-t1"],
    "criterion 7 negative: and it never re-creates (blanks) the surface",
  );
  assert.ok(
    !processor.seen.some((m) => m && m.deleteSurface),
    "criterion 7 negative: the bridge never asks the engine to delete a surface",
  );
});

// --- adversarial ------------------------------------------------------------------------------

test("adversarial: an empty messages array neither throws nor blanks the rendered surface", opts, async () => {
  const mod = await load("ui/bridge.js");
  const { processor, make } = bridgeWith();
  const bridge = make(mod);
  bridge.handle(surfaceMessage([createSurface(), updateComponents("turn-t1", [TEXT("root", "a")])]));
  const before = processor.ids("turn-t1");

  assert.doesNotThrow(
    () => bridge.handle(surfaceMessage([], { canvas_idx: 1 })),
    "a surface message with no A2UI messages must not throw out of the panel",
  );
  assert.deepEqual(processor.ids("turn-t1"), before, "nothing already shown is removed");
  assert.equal(bridge.surfaceId(), "turn-t1", "the surface stays mounted");
});

test("adversarial: a deleteSurface the gate never emits is dropped, not executed", opts, async () => {
  const mod = await load("ui/bridge.js");
  const { processor, make } = bridgeWith();
  const bridge = make(mod);
  bridge.handle(surfaceMessage([createSurface(), updateComponents("turn-t1", [TEXT("root", "a")])]));

  assert.doesNotThrow(() =>
    bridge.handle(
      surfaceMessage(
        [
          { version: "v0.9.1", deleteSurface: { surfaceId: "turn-t1" } },
          updateComponents("turn-t1", [TEXT("c", "c")]),
        ],
        { canvas_idx: 1 },
      ),
    ),
  );
  assert.ok(
    processor.surfaces.has("turn-t1"),
    "criterion 7 negative: a deleteSurface must not blank an already-rendered surface — the " +
      "gate emits one surface per turn (T-0010 A7) and deleteSurface is out of scope for v1",
  );
  assert.ok(
    !processor.seen.some((m) => m && m.deleteSurface),
    "the deleteSurface message is never handed to the engine",
  );
});

test("criterion 5 negative: an unknown catalogId logs one line, invents no surface, and does not poison the next one", opts, async () => {
  const mod = await load("ui/bridge.js");
  const processor = new FakeProcessor();
  const logged = [];
  const bridge = mod.createSurfaceBridge({ processor, log: (l) => logged.push(String(l)) });
  const bogus = "https://a2ui.org/specification/v0_9/catalogs/nope/catalog.json";

  assert.doesNotThrow(() =>
    bridge.handle(
      surfaceMessage([
        createSurface("turn-t1", bogus),
        updateComponents("turn-t1", [TEXT("root", "a")]),
      ]),
    ),
  );
  assert.deepEqual(
    [...processor.surfaces.keys()],
    [],
    "criterion 5 negative: an unknown catalogId produces an error line, 'not a blank pane and " +
      "not an invented surface'",
  );
  assert.equal(
    logged.filter((l) => l.includes(bogus)).length,
    1,
    `criterion 5 negative: exactly one line naming the unknown catalogId; got ${logged.join(" | ")}`,
  );
  assert.equal(bridge.stats().errors, 1, "the failure is counted, not swallowed");

  // The panel is still usable afterwards: the next, good surface renders.
  bridge.handle(
    surfaceMessage([createSurface("turn-t2"), updateComponents("turn-t2", [TEXT("root", "ok")])], {
      surface_id: "turn-t2",
    }),
  );
  assert.deepEqual(processor.ids("turn-t2"), ["root"], "a later good surface still renders");
});

test("adversarial: a processor that throws on an arbitrary message does not take the panel down", opts, async () => {
  const mod = await load("ui/bridge.js");
  const processor = new FakeProcessor();
  const original = processor._apply.bind(processor);
  processor._apply = (message) => {
    if (message && message.updateDataModel) throw new A2uiStateError("boom in the data model");
    return original(message);
  };
  const logged = [];
  const bridge = mod.createSurfaceBridge({ processor, log: (l) => logged.push(String(l)) });
  assert.doesNotThrow(() =>
    bridge.handle(
      surfaceMessage([
        createSurface(),
        { version: "v0.9.1", updateDataModel: { surfaceId: "turn-t1", path: "/", value: {} } },
        updateComponents("turn-t1", [TEXT("root", "still here")]),
      ]),
    ),
    "criterion 2 negative (T-0024): mount and the handlers never throw out of the panel",
  );
  assert.deepEqual(
    processor.ids("turn-t1"),
    ["root"],
    "the message after the throwing one still reaches the engine",
  );
  assert.ok(logged.length >= 1, "the throw is logged, not swallowed silently");
});

// =================================================================================================
// T-0154 group B — B5 (per-surface canvas order) and B8's bridge half (surfaceId() is the newest)
// =================================================================================================
//
// docs/tasks/T-0154.md B5: "For every surface, a canvas whose canvas_idx is <= the last one applied
// to that surface is skipped, including when a newer surface was created in between." With cards
// inline every earlier card stays on screen (B4), so an older surface's stale canvas is no longer
// harmless: applied after a newer surface exists, it pushes older component properties back over
// newer ones in a card the human can still see.
//
// Gate (O-P): rows that need group B run through `pendingT0154B` — strict xfail until
// docs/tasks/T-0154.md's **Status:** records "group B: tests …, activated <sha>". Rows that hold
// today and must keep holding (the guard's other direction) are ordinary rows.
// `ASTRA_RUN_PENDING=1 node --test tests/hud/ui_bridge.test.mjs` runs the pending rows for real.

const T0154_TASK = fileURLToPath(new URL("docs/tasks/T-0154.md", REPO));
const T0154_STATUS = existsSync(T0154_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0154_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const B_PENDING =
  !/\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i.test(T0154_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0154B(name, fn) {
  test(name + (B_PENDING ? " [pending T-0154 B: strict xfail until group B is activated]" : ""), opts, async (t) => {
    if (!B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null)
      assert.fail(
        "XPASS(strict): this T-0154 group B row passed before group B's activation is recorded in " +
          "docs/tasks/T-0154.md's status. Either B landed (record the activation) or the row is vacuous.",
      );
    t.diagnostic("expected failure (T-0154 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

/** How many updateComponents for `surfaceId` reached the engine. */
const updatesFor = (processor, surfaceId) =>
  processor.seen.filter((m) => m && m.updateComponents && m.updateComponents.surfaceId === surfaceId).length;

/** One canvas of one surface, exactly as UiGate builds it: createSurface re-sent, then the tree. */
const canvas = (surfaceId, idx, { final = false } = {}) =>
  surfaceMessage([createSurface(surfaceId), updateComponents(surfaceId, [TEXT("root", `${surfaceId} c${idx}`)])], {
    surface_id: surfaceId,
    canvas_idx: idx,
    final,
  });

pendingT0154B("T-0154 B5 (adversarial): t9 canvas 1, then t10 canvas 0, then t9 canvas 0 — the stale t9 canvas is skipped", async () => {
  const mod = await load("ui/bridge.js");
  const { processor, make } = bridgeWith();
  const bridge = make(mod);
  bridge.handle(canvas("turn-t9", 0));
  bridge.handle(canvas("turn-t9", 1));
  assert.equal(updatesFor(processor, "turn-t9"), 2, "premise: two t9 canvases applied");
  bridge.handle(canvas("turn-t10", 0)); // a newer surface is created in between
  bridge.handle(canvas("turn-t9", 0)); // stale: 0 <= 1, the last applied to turn-t9
  assert.equal(
    updatesFor(processor, "turn-t9"),
    2,
    "B5: a canvas of turn-t9 with canvas_idx <= the last one applied to turn-t9 is skipped, " +
      "even though turn-t10 was created in between (it would push canvas 0's tree back over canvas 1's)",
  );
  bridge.handle(canvas("turn-t9", 1, { final: true })); // the fixture's final re-send at the same idx
  assert.equal(updatesFor(processor, "turn-t9"), 2, "B5: an equal canvas_idx is skipped too (<=, not <)");
  assert.equal(updatesFor(processor, "turn-t10"), 1, "the newer surface was untouched");
});

test("T-0154 B5 (the other direction): a newer canvas of an older surface still applies after a newer surface exists", opts, async () => {
  // A guard needs a row for being too wide (.claude/rules/tests.md): a bridge that refused every
  // canvas of a surface that is no longer the newest would freeze a card the model is still writing.
  const mod = await load("ui/bridge.js");
  const { processor, make } = bridgeWith();
  const bridge = make(mod);
  bridge.handle(canvas("turn-t9", 1));
  bridge.handle(canvas("turn-t10", 0));
  bridge.handle(canvas("turn-t9", 2));
  assert.equal(updatesFor(processor, "turn-t9"), 2, "B5: t9 canvas 2 > 1 is applied although turn-t10 is newer");
  assert.equal(updatesFor(processor, "turn-t10"), 1);
});

test("T-0154 B8 (bridge half): surfaceId() is the most recently *created* surface, not the last one updated", opts, async () => {
  // T-0024 criterion 9 / H-T154-4: astra.ui.action is sent only for the newest surface, and
  // hud/src/ui/index.js asks the bridge which one that is. A later canvas of an older card must not
  // hand the older card the voice back.
  const mod = await load("ui/bridge.js");
  const { make } = bridgeWith();
  const bridge = make(mod);
  bridge.handle(canvas("turn-t9", 0));
  bridge.handle(canvas("turn-t10", 0));
  assert.equal(bridge.surfaceId(), "turn-t10");
  bridge.handle(canvas("turn-t9", 1));
  assert.equal(bridge.surfaceId(), "turn-t10", "B8: t9's later canvas does not make t9 the newest surface");
  bridge.handle(canvas("turn-t11", 0));
  assert.equal(bridge.surfaceId(), "turn-t11");
});

pendingT0154B("T-0154 B5 (adversarial, brute force): 300 seeded interleavings of three surfaces — exactly the per-surface-monotone canvases apply", async () => {
  // Oracle, independent of the bridge: a canvas applies iff its idx is greater than every idx
  // already *applied* to the same surface. Sequences come from a fixed-seed LCG (seed 154), 300 of
  // them, 12 messages each over turn-a / turn-b / turn-c with idx in 0..4, so duplicates,
  // regressions and interleavings across surfaces are all frequent.
  const mod = await load("ui/bridge.js");
  let seed = 154;
  const next = (n) => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed % n;
  };
  const ids = ["turn-a", "turn-b", "turn-c"];
  const failures = [];
  for (let run = 0; run < 300; run++) {
    const { processor, make } = bridgeWith();
    const bridge = make(mod);
    const last = new Map();
    const want = { "turn-a": 0, "turn-b": 0, "turn-c": 0 };
    const seq = [];
    for (let k = 0; k < 12; k++) {
      const id = ids[next(3)];
      const idx = next(5);
      seq.push(`${id.slice(5)}${idx}`);
      if (!last.has(id) || idx > last.get(id)) {
        last.set(id, idx);
        want[id] += 1;
      }
      bridge.handle(canvas(id, idx));
    }
    const got = Object.fromEntries(ids.map((id) => [id, updatesFor(processor, id)]));
    if (JSON.stringify(got) !== JSON.stringify(want)) failures.push(`${seq.join(" ")}: applied ${JSON.stringify(got)}, oracle ${JSON.stringify(want)}`);
  }
  assert.deepEqual(failures.slice(0, 5), [], `B5: ${failures.length}/300 sequences applied a stale canvas or refused a newer one`);
});
